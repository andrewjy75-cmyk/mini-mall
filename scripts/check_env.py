"""环境自检：一次性把"能不能跑起来"的所有前提都验一遍。

为什么需要这个脚本？
因为 RAG 项目的依赖链很长（Python → 库 → 数据库 → 模型下载 → 大模型 API），
任何一环没配好，报出来的错误都可能指向一个无关的地方。典型场景：

    用户上传文档成功 → 提问 → 报 "api_key client option must be set"
    这行报错藏在几十行堆栈里，用户很可能以为是代码问题，去翻源码。

这个脚本按顺序把每一环都验一遍，并且给出**明确的下一步动作**。
演示前跑一次，比现场翻车强。

用法：
    .venv\\Scripts\\python.exe scripts\\check_env.py
    .venv\\Scripts\\python.exe scripts\\check_env.py --skip-model   # 跳过耗时的模型检查
"""

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 检查结果的收集器：最后统一汇总，避免中途失败就看不到全貌
RESULTS: list[tuple[str, str, str]] = []   # (状态, 项目, 说明)


def record(ok: bool | None, name: str, detail: str = "") -> None:
    """记录一条检查结果。ok=None 表示提示信息而非通过/失败。"""
    status = "OK" if ok is True else ("FAIL" if ok is False else "INFO")
    RESULTS.append((status, name, detail))

    icon = {"OK": "[✓]", "FAIL": "[✗]", "INFO": "[i]"}[status]
    line = f"{icon} {name}"
    if detail:
        line += f"\n      {detail}"
    print(line)


def check_python() -> None:
    print("\n--- 1. Python 环境 ---")
    version = sys.version_info
    if version >= (3, 10):
        record(True, f"Python {version.major}.{version.minor}.{version.micro}")
    else:
        record(False, f"Python {version.major}.{version.minor}",
               "版本过低，本项目需要 3.10+（用到 `X | None` 语法）")


def check_packages() -> None:
    print("\n--- 2. 依赖包 ---")
    # (模块名, pip 包名, 用途, 是否必需)
    packages = [
        ("fastapi", "fastapi", "Web 框架", True),
        ("sqlalchemy", "sqlalchemy", "ORM", True),
        ("aiomysql", "aiomysql", "异步 MySQL 驱动", True),
        ("pydantic_settings", "pydantic-settings", "配置读取", True),
        ("multipart", "python-multipart", "文件上传（缺了上传接口会报错）", True),
        ("pypdf", "pypdf", "PDF 解析", True),
        ("docx", "python-docx", "Word 解析", True),
        ("langchain_core", "langchain-core", "LangChain 核心", True),
        ("langchain_text_splitters", "langchain-text-splitters", "文本切分", True),
        ("langchain_openai", "langchain-openai", "大模型接入", True),
        ("langchain_chroma", "langchain-chroma", "向量库集成", True),
        ("chromadb", "chromadb", "向量库", True),
        ("sentence_transformers", "sentence-transformers", "本地向量/重排模型", True),
        ("torch", "torch", "模型运行框架", True),
        ("httpx", "httpx", "评估脚本调用接口", True),
        ("reportlab", "reportlab", "生成演示 PDF（可选）", False),
    ]

    missing_required: list[str] = []
    for module, pip_name, purpose, required in packages:
        try:
            __import__(module)
            record(True, f"{pip_name}", purpose)
        except ImportError:
            if required:
                missing_required.append(pip_name)
                record(False, f"{pip_name}", f"缺失！用途：{purpose}")
            else:
                record(None, f"{pip_name}", f"未安装（可选，用途：{purpose}）")

    if missing_required:
        print("\n      安装缺失的依赖：")
        print(f"      .venv\\Scripts\\python.exe -m pip install {' '.join(missing_required)}")


def check_config() -> None:
    print("\n--- 3. 配置项 ---")
    from core.config import settings

    # 数据库
    if settings.MYSQL_PASSWORD:
        record(True, "数据库配置已填写",
               f"{settings.MYSQL_USER}@{settings.MYSQL_HOST}:{settings.MYSQL_PORT}/{settings.MYSQL_DB}")
    else:
        record(False, "数据库密码为空", "请检查 .env 里的 MYSQL_PASSWORD")

    # 大模型
    if settings.llm_configured:
        masked = settings.LLM_API_KEY[:6] + "..." + settings.LLM_API_KEY[-4:]
        record(True, "大模型 API Key 已配置", f"{masked} @ {settings.LLM_BASE_URL}")
    else:
        record(False, "未配置 LLM_API_KEY",
               "上传文档可用，但【提问会失败】。请在 .env 里填 LLM_API_KEY=sk-...")

    # HuggingFace 镜像 —— 国内环境的头号问题
    if settings.HF_ENDPOINT:
        record(True, "模型下载源已配置", settings.HF_ENDPOINT)
    else:
        record(None, "未配置 HF_ENDPOINT",
               "国内网络下 huggingface.co 通常不可达，建议设为 https://hf-mirror.com")

    # 向量模型
    record(None, "向量模型", settings.EMBEDDING_MODEL)
    record(None, "重排模型",
           f"{settings.RERANK_MODEL}（重排开关：{settings.RERANK_ENABLED}）")


async def check_database() -> None:
    print("\n--- 4. 数据库连通性 ---")
    try:
        from sqlalchemy import text

        from core.database import engine

        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
            # 顺便查一下 RAG 相关的表建了没有
            result = await conn.execute(text("SHOW TABLES"))
            tables = {row[0] for row in result.fetchall()}

        record(True, "数据库连接正常")

        need = {"knowledge_docs", "knowledge_chunks", "chat_sessions", "chat_messages"}
        missing = need - tables
        if missing:
            record(False, "RAG 数据表缺失",
                   f"缺少：{', '.join(sorted(missing))}\n"
                   "      请执行：.venv\\Scripts\\python.exe init_db.py")
        else:
            record(True, "RAG 数据表齐全",
                   f"共 {len(tables)} 张表，含 {', '.join(sorted(need))}")

        await engine.dispose()
    except Exception as exc:  # noqa: BLE001
        record(False, "数据库连接失败", f"{exc}\n      请检查 .env 里的 MYSQL_* 配置和数据库服务")


def check_model() -> None:
    print("\n--- 5. 向量模型（首次会下载，可能需要几分钟）---")
    try:
        from rag.embedder import get_model

        model = get_model()
        dim = model.get_sentence_embedding_dimension()
        record(True, "向量模型加载成功", f"向量维度 {dim}")

        # 真跑一次编码，顺便验证检索方向和归一化是否正常
        from rag.embedder import get_embeddings

        emb = get_embeddings()
        vecs = emb.embed_documents(["退货政策", "退款到账时间"])
        if vecs and len(vecs[0]) == dim:
            record(True, "向量化自测通过", f"编码 2 条，每条 {len(vecs[0])} 维")

            # 自检：相似的句子应该比不相似的更接近。
            # 这一步能抓出"查询指令前缀没加对"这类不明显的问题
            q = emb.embed_query("退货要多久")
            sims = [sum(a * b for a, b in zip(q, v)) for v in vecs]
            if sims[0] > sims[1]:
                record(True, "语义相似度方向正确",
                       f"「退货要多久」vs「退货政策」={sims[0]:.3f} > vs「退款到账时间」={sims[1]:.3f}")
            else:
                record(None, "语义相似度未达预期",
                       f"{sims[0]:.3f} vs {sims[1]:.3f}。"
                       "可能是向量未归一化或查询前缀配置有误，建议检查 EMBEDDING_QUERY_INSTRUCTION")
        else:
            record(False, "向量化结果异常", "返回了空向量或维度不匹配")
    except Exception as exc:  # noqa: BLE001
        record(False, "向量模型加载失败",
               f"{exc}\n"
               "      常见原因：网络无法访问模型下载源。"
               "请在 .env 里配置 HF_ENDPOINT=https://hf-mirror.com")


def check_vectorstore() -> None:
    print("\n--- 6. 向量库 ---")
    try:
        from rag.vectorstore import count_documents, get_chroma_dir, get_vectorstore

        store = get_vectorstore()
        count = count_documents(store)
        record(True, "向量库可用",
               f"目录 {get_chroma_dir()}，现有 {count} 条向量")
    except Exception as exc:  # noqa: BLE001
        record(False, "向量库初始化失败", str(exc))


def check_llm() -> None:
    print("\n--- 7. 大模型连通性 ---")
    from core.config import settings

    if not settings.llm_configured:
        record(None, "跳过（未配置 API Key）", "填好 LLM_API_KEY 后重跑本项检查")
        return

    try:
        from rag.llm import check_llm_available

        ok, message = check_llm_available()
        record(ok, "大模型调用测试", message)
    except Exception as exc:  # noqa: BLE001
        record(False, "大模型调用异常", str(exc))


def print_summary() -> int:
    """打印汇总，返回退出码。"""
    ok = sum(1 for s, _, _ in RESULTS if s == "OK")
    fail = sum(1 for s, _, _ in RESULTS if s == "FAIL")
    info = sum(1 for s, _, _ in RESULTS if s == "INFO")

    print("\n" + "=" * 64)
    print(f"  汇总：{ok} 项通过，{fail} 项失败，{info} 项提示")
    print("=" * 64)

    if fail:
        print("\n必须先解决失败项，否则部分功能不可用：")
        for status, name, _ in RESULTS:
            if status == "FAIL":
                print(f"  ✗ {name}")
        print("\n提示：失败项通常只需要改 .env 或装一个包。")
    else:
        print("\n环境检查全部通过，可以启动服务了：")
        print("  .venv\\Scripts\\python.exe main.py")

    return 1 if fail else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Mini Mall RAG 环境自检")
    parser.add_argument("--skip-model", action="store_true",
                        help="跳过向量模型加载检查（模型很大时省时间）")
    parser.add_argument("--skip-llm", action="store_true",
                        help="跳过大模型连通性检查（省一次 API 调用）")
    args = parser.parse_args()

    print("=" * 64)
    print("  Mini Mall 智能客服 RAG —— 环境自检")
    print("=" * 64)

    check_python()
    check_packages()
    check_config()
    asyncio.run(check_database())

    if args.skip_model:
        print("\n--- 5/6. 向量模型与向量库：已跳过（--skip-model）---")
    else:
        check_model()
        check_vectorstore()

    if args.skip_llm:
        print("\n--- 7. 大模型：已跳过（--skip-llm）---")
    else:
        check_llm()

    sys.exit(print_summary())


if __name__ == "__main__":
    main()
