"""向量库：用 Chroma 存向量并做相似度检索。

为什么选 Chroma？三个理由：
1. 纯本地文件持久化，不用额外起服务——"可本地完整运行"这条硬指标的第一保障
2. 装完就能用，零配置，不像 Milvus 那样要 docker
3. LangChain 官方集成的，接口统一

它的短板也要说清楚（优化文档里会提到）：
数据量上到百万级、需要多副本或高性能过滤时，就该换 Milvus / Qdrant 了。
本项目的规模（几千个块）用 Chroma 绰绰有余，杀鸡不必用牛刀。

★ 这里有一个极其容易搞错的地方，必须写在最前面：

    Chroma 返回的 score 是【距离】，不是相似度！
    距离越小 = 越相关，方向和我们直觉相反。

    而且距离的定义取决于建库时的距离度量：
        余弦距离（cosine）  ：范围 0~2，0 表示完全相同
        L2 距离             ：范围 0~∞，0 表示完全相同

    代码里凡是拿到 score 的地方，都必须先想清楚"这是距离还是相似度"。
    搞反的后果是：检索出一堆最不相关的内容，而且不报任何错。
    本模块统一在出口处把距离转成【相似度】（越大越相关），
    让上层只面对一种语义，把混乱挡在这一层里面。
"""

import logging
import shutil
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from core.config import settings
from rag.embedder import get_embeddings

logger = logging.getLogger(__name__)

# 单例。Chroma 客户端持有文件句柄，反复创建实例在 Windows 上容易踩到文件锁
_store: Any = None


def get_chroma_dir() -> Path:
    """返回向量库目录的绝对路径。

    ★ 必须用绝对路径。因为相对路径是相对于"进程当前工作目录"的，
      而工作目录会变：直接跑 main.py 是项目根目录，
      但从别的目录启动、或者用 pytest、或者被 IDE 拉起来时就未必了。
      一旦写错，现象是"每次重启知识库都空了"——数据写到了另一个目录去。
    """
    path = Path(settings.CHROMA_DIR)
    if not path.is_absolute():
        # 以项目根目录为基准（本文件在 rag/ 下，往上一层就是根）
        path = Path(__file__).resolve().parent.parent / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_vectorstore(*, force_new: bool = False):
    """获取全局唯一的 Chroma 实例。

    参数 force_new 用于测试或重建索引时强制拿一个新实例。
    """
    global _store
    if _store is not None and not force_new:
        return _store

    try:
        from langchain_chroma import Chroma
    except ImportError as exc:
        raise RuntimeError(
            "向量库依赖未安装。请先执行：\n"
            "    .venv\\Scripts\\python.exe -m pip install langchain-chroma chromadb"
        ) from exc

    store = Chroma(
        collection_name=settings.CHROMA_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(get_chroma_dir()),
        # ★ collection_metadata 只在"集合首次被创建"时生效。
        #   如果目录里已经存在同名集合，这里再怎么写都不会改动它。
        #   这也是为什么换距离度量必须删库重建，改配置是没用的。
        collection_metadata={"hnsw:space": "cosine"},
    )

    if not force_new:
        _store = store

    logger.info(
        "向量库就绪：目录=%s 集合=%s 现有 %s 条",
        get_chroma_dir(),
        settings.CHROMA_COLLECTION,
        count_documents(store),
    )
    return store


def count_documents(store=None) -> int:
    """统计向量库里有多少条。用于健康检查和"索引是否和库表对得上"的排查。"""
    store = store or get_vectorstore()
    try:
        return store._collection.count()
    except Exception as exc:  # noqa: BLE001
        logger.warning("统计向量条数失败：%s", exc)
        return 0


def add_chunks(chunks: list[Document], ids: list[str]) -> int:
    """把切好的块写入向量库。

    参数:
        chunks: Document 列表，page_content 是正文，metadata 会一起存进去
        ids:    与 chunks 一一对应的 id（用 KnowledgeChunk.id）

    ★ 为什么 id 要由调用方传进来，而不是让 Chroma 自己生成？
      因为 MySQL 那边的切片记录是主数据，两边必须能用同一个 id 对上。
      让向量库自己生成 id 的话，就得再维护一张映射表，纯属自找麻烦。

    ★ metadata 里存什么也有讲究：
      Chroma 的 metadata 只支持 str/int/float/bool，【不支持 list 或 dict】，
      塞进去会直接报错。所以要提前把不支持的字段挑出来。
    """
    if not chunks:
        return 0
    if len(chunks) != len(ids):
        raise ValueError(f"chunks 与 ids 数量不一致：{len(chunks)} vs {len(ids)}")

    store = get_vectorstore()

    # 过滤掉 Chroma 不接受的 metadata 值类型，避免整批写入失败
    for chunk in chunks:
        chunk.metadata = {
            k: v
            for k, v in chunk.metadata.items()
            if isinstance(v, (str, int, float, bool)) or v is None
        }

    store.add_documents(documents=chunks, ids=ids)
    logger.info("向量库写入 %s 条", len(chunks))
    return len(chunks)


def delete_by_ids(ids: list[str]) -> None:
    """按 id 删除向量。删除文档时调用。"""
    if not ids:
        return
    store = get_vectorstore()
    store.delete(ids=ids)
    logger.info("向量库删除 %s 条", len(ids))


def delete_by_doc_id(doc_id: str) -> int:
    """按文档 id 删除它名下的所有向量。

    走底层 collection 的 where 过滤，一次删干净。
    为什么不用 langchain 封装的方法？因为它只开放了按 ids 删，
    按 metadata 过滤得下到 _collection 这一层。这是有意为之的取舍：
    先试底层，失败就返回 0，让上层的"按 chunk id 删"兜底。
    """
    store = get_vectorstore()
    try:
        result = store._collection.delete(where={"doc_id": doc_id})
        logger.info("向量库按 doc_id=%s 删除完成", doc_id)
        return result if isinstance(result, int) else 0
    except Exception as exc:  # noqa: BLE001
        logger.warning("按 doc_id 删除向量失败（将由上层按 id 兜底）：%s", exc)
        return 0


def similarity_search(
    query: str,
    k: int = 5,
    score_threshold: float | None = None,
) -> list[tuple[Document, float]]:
    """相似度检索，返回 [(文档, 相似度), ...]。

    ★ 出口处已经把 Chroma 的【距离】转换成了【相似度】：
        余弦距离 d ∈ [0, 2]  →  相似度 = 1 - d ∈ [-1, 1]
      返回的 score 越大越相关，上层可以直接和阈值比较。

    关于 score_threshold：
      LangChain 的 similarity_search_with_relevance_scores 也能做阈值过滤，
      但它内部对距离的换算在不同版本里行为不一致（有时还会给出
      令人困惑的负值）。与其依赖它的实现细节，不如自己算、自己过滤——
      逻辑透明，出问题一眼能看懂。
    """
    if not query or not query.strip():
        return []

    store = get_vectorstore()
    # k 稍微多取一些：阈值过滤会刷掉一部分，多留点余量免得过滤后不够数
    raw = store.similarity_search_with_score(query, k=k)

    results: list[tuple[Document, float]] = []
    for doc, distance in raw:
        similarity = 1.0 - float(distance)
        if score_threshold is not None and similarity < score_threshold:
            continue
        results.append((doc, similarity))

    results.sort(key=lambda pair: pair[1], reverse=True)
    logger.debug(
        "检索 %r：原始 %s 条 → 过滤后 %s 条（阈值 %s）",
        query[:30],
        len(raw),
        len(results),
        score_threshold,
    )
    return results


def reset_collection() -> None:
    """清空整个集合。

    ⚠️ 危险操作，只在"重建索引"和测试里用。
    实现方式是直接把持久化目录删掉重建——因为前面说过，
    collection_metadata 只在创建时生效，靠改配置是改不动距离度量的。
    """
    global _store
    directory = get_chroma_dir()
    _store = None

    from langchain_chroma import Chroma  # noqa: F401  确保依赖可用再动文件

    if directory.exists():
        shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True, exist_ok=True)
    logger.warning("向量库已清空并重建：%s", directory)
