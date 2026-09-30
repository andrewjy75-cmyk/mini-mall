"""向量化：把文字变成一串数字（向量），让"语义相近"能被计算。

检索的本质是"算两段文字的相似度"。计算机不认识字，所以要把文字映射到
一个高维空间里的点：意思相近的句子，点就靠得近。这个映射就是 embedding。

为什么用本地的 BGE 而不是调 API 做向量？
1. 免费且不限额——建索引时要把成百上千个块全部向量化，调 API 是一笔持续开销
2. 不依赖网络，离线也能用，符合"可本地完整运行"
3. 数据不出本机，企业知识库场景下这点很重要
代价是首次要下载约 100MB 模型，且 CPU 上编码比 GPU 慢。

★ 两个非常容易踩、又非常隐蔽的坑，重点说明：

坑 1｜查询和文档的处理方式不一样
    BGE 这类模型训练时，查询侧是加了指令前缀的：
        "为这个句子生成表示以用于检索相关文章：退货要几天"
    而文档侧【不加】任何前缀。两侧不对称，是模型训练时就定下的约定。
    如果两边都加、或都不加，检索不会报错，但召回率会安静地下降——
    这种"不报错只是变差"的 bug 最难发现，所以代码上必须把两个入口分开。
    （注：bge-*-zh-v1.5 官方说过指令带来的提升已不明显，
     所以这里做成可配置项 EMBEDDING_QUERY_INSTRUCTION，留空即关闭。）

坑 2｜向量必须归一化
    不做归一化时，向量的"长度"会影响相似度计算结果，长文档天然占便宜。
    归一化后长度统一为 1，内积就等于余弦相似度，比较才公平。
    → 所以 encode 时固定 normalize_embeddings=True，
      向量库那边也必须配成余弦/内积距离，两者要配套，不能各配各的。
"""

import logging
import threading
from typing import Any

from core.config import settings  # ← 这行必须先执行：它会设置 HF_ENDPOINT
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class EmbeddingModelError(Exception):
    """向量模型加载失败。message 是给用户看的人话。"""


# ---------------------------------------------------------------------------
# 模型单例
#
# 为什么必须是单例？SentenceTransformer 加载一次要占用几百 MB 内存、耗时数秒。
# 如果每个请求都 new 一个，服务会在并发下直接被内存打爆。
# 用模块级变量 + 双重检查锁，保证多线程下也只加载一次。
# ---------------------------------------------------------------------------
_model: Any = None
_model_lock = threading.Lock()


def _load_sentence_transformer(model_name: str, device: str):
    """真正去加载模型（只应被 get_model 调用一次）。

    这里把 sentence_transformers 的 import 放在函数【内部】，是刻意的：
    - 它一进来就会拉起 torch，import 本身就要好几秒；
      放在模块顶层的话，任何 import 本模块的代码（比如启动服务）
      都得先等它，启动体验很差。
    - 万一 torch 因为平台问题装不上，也不至于让整个服务起不来，
      只有真正要用向量的功能才会报错——这叫"故障隔离"。
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbeddingModelError(
            "向量模型依赖未安装。请先执行：\n"
            "    .venv\\Scripts\\python.exe -m pip install sentence-transformers"
        ) from exc

    try:
        logger.info("正在加载向量模型 %s（首次会自动下载，请稍候）...", model_name)
        model = SentenceTransformer(model_name, device=device)
        # ★ sentence-transformers 6.x 把这个方法改了名：
        #     get_sentence_embedding_dimension  →  get_embedding_dimension
        #   这里做兼容取值，否则升级后会刷一堆 FutureWarning，
        #   而如果哪天旧名字被彻底移除，就会直接 AttributeError。
        #   「新名字优先、旧名字兜底」是应对库改名最稳的写法
        get_dim = getattr(model, "get_embedding_dimension", None) or getattr(
            model, "get_sentence_embedding_dimension", None
        )
        logger.info(
            "向量模型加载完成：%s，向量维度 %s",
            model_name,
            get_dim() if get_dim else "未知",
        )
        return model
    except Exception as exc:  # noqa: BLE001
        # 下载失败是这里最常见的失败原因，错误提示必须直指要害，
        # 否则用户只会看到一坨 requests 的堆栈，完全不知道该干什么
        raise EmbeddingModelError(
            f"向量模型 {model_name} 加载失败：{exc}\n"
            f"当前使用的模型下载源是：{settings.HF_ENDPOINT or 'HuggingFace 官方源'}\n"
            "请检查：\n"
            "  1. 网络能否访问该下载源（国内建议在 .env 里配 HF_ENDPOINT=https://hf-mirror.com）\n"
            "  2. 磁盘剩余空间是否足够（模型约 100MB）\n"
            f"  3. 也可以手动下载后放到本地目录，再把 EMBEDDING_MODEL 改成本地路径"
        ) from exc


def get_model():
    """拿全局唯一的向量模型。第一次调用时才加载（懒加载）。"""
    global _model
    # 第一层检查：绝大多数调用会命中这里，直接返回，不用抢锁
    if _model is not None:
        return _model

    with _model_lock:
        # 第二层检查：可能在上面的等待期间，别的线程已经加载好了
        if _model is None:
            _model = _load_sentence_transformer(
                settings.EMBEDDING_MODEL, settings.EMBEDDING_DEVICE
            )
    return _model


class BGEEmbeddings(Embeddings):
    """BGE 向量模型，实现 LangChain 的 Embeddings 接口。

    为什么继承 LangChain 的接口而不是自己随便写两个方法？
    因为一旦符合这个接口，就能直接塞进 LangChain 的整个生态：
    Chroma 向量库、各种 Retriever、评估器都认它，不用写胶水代码。
    这就是"用生态"的收益。
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        query_instruction: str | None = None,
        batch_size: int = 16,
    ):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.device = device or settings.EMBEDDING_DEVICE
        # 显式传空字符串表示"我知道这个前缀，但这次不要用"
        self.query_instruction = (
            settings.EMBEDDING_QUERY_INSTRUCTION
            if query_instruction is None
            else query_instruction
        )
        self.batch_size = batch_size

    # ---- LangChain Embeddings 接口要求的两个方法 ----

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """把【文档】批量转向量。

        注意这里【不加】查询指令前缀——这是坑 1 的文档侧。
        batch_size 控制一次送多少条进模型，太大爆内存，太小跑不满 CPU。
        """
        if not texts:
            return []

        model = get_model()
        # show_progress_bar=True 在传大文档时很有用（几百个块要跑十几秒，
        # 没有进度条会让人以为卡死了），但接口调用时会显得很吵，所以关闭
        vectors = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,   # 坑 2：必须归一化
            show_progress_bar=len(texts) > 50,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """把【用户问题】转向量。

        这里加指令前缀——坑 1 的查询侧。两侧不对称是故意的，不是笔误。
        """
        if not text or not text.strip():
            # 空查询没必要走一遍模型，返回空向量让上层去处理
            # （正常流程不该出现，属于防御性代码）
            raise ValueError("查询内容不能为空")

        prefixed = f"{self.query_instruction}{text}" if self.query_instruction else text
        return self.embed_documents([prefixed])[0]


# ---------------------------------------------------------------------------
# 懒加载单例：整个应用共用一份 Embeddings 对象
# ---------------------------------------------------------------------------
_embeddings: BGEEmbeddings | None = None
_embeddings_lock = threading.Lock()


def get_embeddings() -> BGEEmbeddings:
    """获取全局 Embeddings 实例（LangChain 生态各处都用这个）。"""
    global _embeddings
    if _embeddings is None:
        with _embeddings_lock:
            if _embeddings is None:
                _embeddings = BGEEmbeddings()
    return _embeddings


def warmup() -> bool:
    """预热：在服务启动时主动把模型加载好。

    为什么需要？因为模型是懒加载的，如果不预热，第一个提问的用户
    要等好几秒（模型加载时间），体验上会以为服务挂了。
    预热把这份等待挪到启动阶段。

    返回 True 表示成功。失败【不抛异常】——预热失败不该阻止服务启动，
    只是第一个请求会慢一点或报错。启动阶段的容错要宽松。
    """
    try:
        model = get_model()
        # 真跑一次编码：光加载模型不代表能用，有些环境要到编码时才暴露问题
        model.encode(["预热"], normalize_embeddings=True)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("向量模型预热失败，检索功能将不可用：%s", exc)
        return False
