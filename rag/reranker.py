"""重排（Rerank）：RAG 里"性价比最高"的一个优化点。

先说清楚它解决什么问题。

向量检索是【双塔】结构：查询和文档分别编码成向量，再算向量距离。
优点是快——文档向量可以提前算好存起来，检索时只算查询那一次。
但缺点也很致命：查询和文档在编码时**互相看不到对方**。
"苹果手机壳"和"苹果多少钱一斤"，在向量空间里可能离得很近，
因为"苹果"这个词的权重太大，模型没机会知道你到底想说什么。

重排用的是【交叉编码器】（CrossEncoder）：把「查询 + 文档」拼成一整句
一起送进模型。这样模型能逐词比对两者，精度明显更高。
代价是慢——每对「查询+文档」都要完整跑一次模型，无法预先计算。
所以重排不能替代向量检索，只能作为它的**第二道精排**：

    第一步（粗排/召回）：向量检索快速捞出 20 条候选  ← 要的是"不漏"
    第二步（精排/重排）：交叉编码器逐条打分，取前 5 条  ← 要的是"准"

这个"先粗后精"的两段式是所有工业级检索系统的标配，不只是 RAG。

★ 工程上最重要的设计：这个模块必须能【优雅降级】。
  reranker 模型有 1GB 多，首次下载慢、也可能失败。
  如果加载失败就让整个问答接口 500，那是非常糟糕的工程决策——
  重排带来的只是"更准一点"，不是"能不能用"。
  所以加载失败时自动退回"不重排"，并在响应里如实反映，让功能可用性优先。
"""

import logging
import math
import threading
from typing import Any

from langchain_core.documents import Document

from core.config import settings

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """基于 BGE CrossEncoder 的重排器。"""

    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.model_name = model_name or settings.RERANK_MODEL
        self.device = device or settings.EMBEDDING_DEVICE
        self._model: Any = None
        self._load_failed = False   # 记下失败状态，避免每次请求都重试一遍加载
        self._lock = threading.Lock()

    # ---- 模型加载 ----

    @property
    def available(self) -> bool:
        """重排是否可用。没加载过不算不可用，要真试过才知道。"""
        return not self._load_failed

    def _ensure_model(self) -> Any:
        """确保模型已加载。返回 None 表示不可用。"""
        if self._model is not None:
            return self._model
        if self._load_failed:
            # 已经失败过了，直接放弃，不要每个请求都去重试下载（那会拖垮服务）
            return None

        with self._lock:
            if self._model is not None:
                return self._model
            if self._load_failed:
                return None

            try:
                from sentence_transformers import CrossEncoder

                logger.info("正在加载重排模型 %s（约 1GB，首次下载较慢）...", self.model_name)
                self._model = CrossEncoder(self.model_name, device=self.device)
                logger.info("重排模型加载完成：%s", self.model_name)
            except Exception as exc:  # noqa: BLE001
                self._load_failed = True
                logger.error(
                    "重排模型 %s 加载失败，已自动降级为「不重排」模式。"
                    "问答功能仍可正常使用，只是精度会下降。原因：%s",
                    self.model_name,
                    exc,
                )
                return None

        return self._model

    # ---- 打分 ----

    def score(self, query: str, passages: list[str]) -> list[float]:
        """给每个「查询+段落」打相关性分数，返回 0~1 之间的值。

        ★ 关于 sigmoid：
          CrossEncoder 原始输出是 logits（大致 -10 ~ 10 的实数），不是概率。
          直接当"分数"用也能排序，但有两个坏处：
          1. 数值范围难看，不方便设阈值
          2. 不好和向量相似度放在一起比较
          用 sigmoid 压到 (0, 1) 后，直观且可比。
          但要记住它仍然只是【相对分数】，不是"正确的概率"——
          别声称"模型认为有 92% 概率相关"，那是过度解读。
        """
        model = self._ensure_model()
        if model is None or not passages:
            return []

        pairs = [[query, passage] for passage in passages]
        # batch_size 别开太大：重排是整段文本过模型，很吃内存
        logits = model.predict(pairs, batch_size=8, show_progress_bar=False)
        return [1.0 / (1.0 + math.exp(-float(x))) for x in logits]

    # ---- 主入口 ----

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 5,
        score_key: str = "rerank_score",
    ) -> list[tuple[Document, float]]:
        """对候选文档重排，返回前 top_k 条 [(文档, 分数), ...]。

        降级行为：模型不可用时，原样返回前 top_k 条（保持向量检索的顺序），
        分数沿用它们已有的分数，保证上层拿到的东西结构一致，不用分支处理。
        """
        if not documents:
            return []

        scores = self.score(query, [d.page_content for d in documents])

        if not scores:
            # 降级路径：用文档上已有的分数字段，没有就用 0，
            # 保证返回的元组结构不变，上层逻辑不用改
            logger.debug("重排不可用，保持向量检索原有顺序")
            return [(doc, float(doc.metadata.get(score_key, 0.0))) for doc in documents[:top_k]]

        # 把重排分数写回 metadata，方便排查"这条为什么被排上来"
        scored: list[tuple[Document, float]] = []
        for doc, s in zip(documents, scores):
            doc.metadata[score_key] = round(float(s), 4)
            scored.append((doc, float(s)))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# 单例
# ---------------------------------------------------------------------------
_reranker: CrossEncoderReranker | None = None
_reranker_lock = threading.Lock()


def get_reranker() -> CrossEncoderReranker:
    """获取全局重排器。"""
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                _reranker = CrossEncoderReranker()
    return _reranker


def rerank_documents(
    query: str,
    documents: list[Document],
    top_k: int = 5,
    enabled: bool | None = None,
) -> tuple[list[tuple[Document, float]], bool]:
    """重排的便捷入口。

    返回 (结果列表, 是否真的执行了重排)。
    为什么要返回第二个布尔值？因为调用方（和前端、和评估脚本）
    需要知道"这次到底有没有重排"——否则降级悄悄发生了，
    数据解读就全错了（把"没重排"当成"重排没用"）。
    """
    if not documents:
        return [], False

    use_rerank = settings.RERANK_ENABLED if enabled is None else enabled
    if not use_rerank:
        return [(doc, float(doc.metadata.get("vector_score", 0.0))) for doc in documents[:top_k]], False

    reranker = get_reranker()
    if not reranker.available and reranker._model is None:
        # 还没试过加载，那就试一次（第一次调用时触发）
        pass

    results = reranker.rerank(query, documents, top_k=top_k)

    # 判断这次是否真的重排了：模型存在即为真
    actually_reranked = reranker._model is not None
    if not actually_reranked:
        logger.warning("本次请求未执行重排（模型不可用），已降级为向量检索顺序")

    return results, actually_reranked
