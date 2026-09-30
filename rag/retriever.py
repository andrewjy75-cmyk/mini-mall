"""检索编排：把「向量召回 → 重排精排 → 上下文压缩」串成一条流水线。

这是整个 RAG 的"检索半边天"，也是效果差异的主要来源。
流程刻意分成三段，每段职责单一、可单独开关、可单独度量：

    ┌─────────────┐   ┌──────────┐   ┌──────────┐
    │ 向量检索召回 │ → │  重排精排 │ → │ 上下文压缩 │ → 最终结果
    │ recall_k=20 │   │  → top_k │   │  只留相关句 │
    └─────────────┘   └──────────┘   └──────────┘
       求"不漏"           求"准"          求"省"

为什么要分三段而不是一个大函数？
因为做调优对比时必须能把每一段单独打开/关闭，才能算清楚
"重排到底贡献了多少、压缩到底损失了多少"。如果揉成一个函数，
就只能做"全开 vs 全关"的对比，得不出有用的结论。

★ 关于"召回数量"和"最终数量"为什么要分开（recall_k vs top_k）：
  这是重排能否发挥作用的前提。如果召回只取 5 条、最终也取 5 条，
  那重排只是把同样的 5 条换了个顺序，没有任何新信息进来，
  它救不回"向量检索压根没找到正确答案"这种情况。
  必须让粗排多召回（比如 20 条），重排才有从中"捞回正确答案"的空间。
  这是很多人做重排却没效果的根本原因。
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field

from langchain_core.documents import Document

from core.config import settings
from rag.compressor import compress_documents
from rag.reranker import rerank_documents
from rag.vectorstore import similarity_search

logger = logging.getLogger(__name__)


@dataclass
class RetrieveResult:
    """一次检索的完整结果，含中间过程。

    为什么要保留中间过程（vector_hits）？
    因为排查"为什么答错了"时，必须能分清是**检索阶段没找对**，
    还是**找对了但模型没用好**。这是两类完全不同的问题，
    修法也完全不同（一个调检索参数，一个调 Prompt）。
    只保留最终结果的话，就永远查不清责任在哪一环。
    """

    # 最终喂给模型的片段 [(文档, 分数)]
    items: list[tuple[Document, float]] = field(default_factory=list)
    # 向量检索的原始候选池（完整保留，不因阈值被裁掉），用于对照分析和调阈值
    vector_hits: list[tuple[Document, float]] = field(default_factory=list)
    # 这次到底有没有真的执行重排（降级时会是 False）
    rerank_enabled: bool = False
    # 压缩统计
    compression_stats: dict = field(default_factory=dict)
    elapsed_ms: int = 0

    # ★ 这两个字段是为了让上层能区分"知识库里没有"和"检索出错了"。
    #   两者的处理方式完全不同：前者应该老实回答"没查到"，
    #   后者应该报错或重试。如果只给一个空的 items，上层就分不清了
    rejected: bool = False
    """是否因为相关度不足而判定"知识库中没有相关内容"。"""

    best_similarity: float = 0.0
    """本次召回的最高相似度。它和阈值的关系决定了是否拒答。"""

    @property
    def is_empty(self) -> bool:
        return not self.items


def retrieve(
    query: str,
    recall_k: int | None = None,
    top_k: int | None = None,
    use_rerank: bool | None = None,
    use_compression: bool | None = None,
    score_threshold: float | None = None,
) -> RetrieveResult:
    """执行完整检索流程（同步版）。

    评估脚本批量跑对比实验时用这个版本——同步写起来简单，
    不需要到处 await。

    ★ 流水线的顺序是踩过坑之后重排的，这里把教训说清楚：

      最初的写法是把"相似度阈值过滤"放在向量检索那一步，也就是
      「先按阈值过滤，再重排」。结果重排几乎从不生效——
      阈值一过，候选常常只剩两三条，比我设置的 top_k 还少，
      于是每次都被"候选不够就别重排了"这条优化判断跳过。
      表面上参数都开了，实际上重排一次都没跑。

      根因是我把两件不同的事混在了一起：

          阈值该管的是「知识库里到底有没有相关内容」（决定要不要答）
          而不是「哪些片段有资格送去重排」（决定先答哪个）

      把这两件事拆开之后，流水线才顺畅：

          ① 向量召回：拿满 recall_k 条，此处不做阈值裁剪
          ② 拒答判断：最高分低于阈值 → 知识库确实没有 → 空手而归
          ③ 重排精排：候选池充足，重排才有发挥空间
          ④ 上下文压缩
          ⑤ 取 top_k

      一句话：**阈值管"要不要答"，重排管"先答哪个"。**
    """
    started = time.perf_counter()

    recall_k = recall_k or settings.RETRIEVE_RECALL_K
    top_k = top_k or settings.RETRIEVE_TOP_K
    threshold = (
        settings.RETRIEVE_SCORE_THRESHOLD if score_threshold is None else score_threshold
    )

    # ---- 第 1 段：向量召回（拿满候选，不在这里裁剪）----
    # 注意 score_threshold=0.0：这里刻意【不做阈值过滤】。
    # 过滤放到第 2 段去，那是两个不同的决策点
    vector_hits = similarity_search(query, k=recall_k, score_threshold=0.0)
    # 把向量分数写进 metadata。后面无论走不走重排，都能回溯"它当年是多少分进来的"
    for doc, score in vector_hits:
        doc.metadata["vector_score"] = round(score, 4)

    elapsed_empty = int((time.perf_counter() - started) * 1000)

    if not vector_hits:
        logger.info("检索 %r 时向量库为空（知识库还没有入库任何文档）", query[:30])
        return RetrieveResult(
            items=[],
            vector_hits=[],
            rerank_enabled=False,
            compression_stats={},
            elapsed_ms=elapsed_empty,
            rejected=True,
            best_similarity=0.0,
        )

    best_similarity = vector_hits[0][1]

    # ---- 第 2 段：拒答判断 ----
    # 这是阈值真正该发挥作用的地方：它回答的是
    # "知识库里到底有没有能回答这个问题的内容"，而不是"要保留几条候选"
    if best_similarity < threshold:
        logger.info(
            "检索 %r 判定为知识库无相关内容：最高相似度 %.4f 低于阈值 %.2f，"
            "不传任何资料给模型，让它如实回答「没查到」",
            query[:30], best_similarity, threshold,
        )
        return RetrieveResult(
            items=[],
            vector_hits=vector_hits,
            rerank_enabled=False,
            compression_stats={},
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            rejected=True,
            best_similarity=best_similarity,
        )

    # ---- 第 3 段：重排精排 ----
    # 到了这里，候选池通常有十几二十条，重排有了充分的发挥空间。
    # 只有候选数不超过 top_k 时才跳过——那种情况下全部候选都会被送进 Prompt，
    # 排序变成一件没有意义的事，过一遍模型纯属浪费
    need_rerank = settings.RERANK_ENABLED if use_rerank is None else use_rerank
    if len(vector_hits) <= top_k:
        need_rerank = False

    if need_rerank:
        ranked, actually_reranked = rerank_documents(
            query, [doc for doc, _ in vector_hits], top_k=top_k, enabled=True
        )
    else:
        ranked, actually_reranked = (
            [(doc, score) for doc, score in vector_hits[:top_k]],
            False,
        )

    # ---- 第 4 段：上下文压缩 ----
    result_docs = [doc for doc, _ in ranked]
    compressed_docs, comp_stats = compress_documents(
        query, result_docs, enabled=use_compression
    )

    # 压缩会丢弃部分文档。这里把保留下来的文档和它们的分数重新对齐，
    # 靠的是 metadata 里的 chunk_id ——这正是"metadata 一个字都不能丢"的原因
    score_map = {doc.metadata.get("chunk_id"): score for doc, score in ranked}
    items: list[tuple[Document, float]] = []
    for doc in compressed_docs:
        chunk_id = doc.metadata.get("chunk_id")
        score = score_map.get(chunk_id)
        if score is None:
            # 压缩不应当产生新文档。真出现了说明有 bug，
            # 与其静默塞个 0 分进去污染数据，不如明确记下来
            logger.warning("压缩后出现了未知 chunk_id=%s 的文档，已用兜底分数", chunk_id)
            score = float(doc.metadata.get("rerank_score", 0.0))
        items.append((doc, score))

    elapsed = int((time.perf_counter() - started) * 1000)
    logger.info(
        "检索完成 %r：召回 %s → 重排%s → 压缩后 %s 条，耗时 %sms",
        query[:30],
        len(vector_hits),
        "已启用" if actually_reranked else "未启用",
        len(items),
        elapsed,
    )

    return RetrieveResult(
        items=items,
        vector_hits=vector_hits,
        rerank_enabled=actually_reranked,
        compression_stats=comp_stats,
        elapsed_ms=elapsed,
        rejected=False,
        best_similarity=best_similarity,
    )


async def aretrieve(query: str, **kwargs) -> RetrieveResult:
    """检索的异步版本，供 FastAPI 接口调用。

    ★ 为什么必须包一层 to_thread？
      向量编码、重排、Chroma 检索全是【同步阻塞】的 CPU/IO 操作。
      FastAPI 的 async 接口跑在事件循环里，如果在里面直接调同步函数，
      整个服务会被这一个请求卡住——包括那些本来该秒回的 /hello 接口。
      放到线程池里执行，事件循环就能继续处理别的请求。

      这是 FastAPI 项目里最容易犯、又最难察觉的错误之一：
      功能测试全过，一上并发就集体变慢，而且看不出是谁拖累了谁。
    """
    return await asyncio.to_thread(retrieve, query, **kwargs)
