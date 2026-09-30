"""上下文压缩（Contextual Compression）：检索回来一堆内容，但只留真正有用的。

要解决的问题很具体：向量检索返回 top_k=5 个块，每块 600 字，
一共 3000 字塞进 Prompt。但用户的那个问题，答案可能只藏在其中
某一块的两句话里。剩下 2900 字全是噪音，它们会：

1. 白花钱、白花时间——token 是计价和延迟的直接来源
2. 稀释重点——相关内容淹没在大段文本里，模型注意力被分散
3. ★ 诱发幻觉——这是最要紧的：Prompt 里塞了不相关的条款，
   模型很容易把 A 条款的限定条件嫁接到 B 条款上，答出一个
   "语法通顺、引用有据、但事实错误"的答案。

压缩的两种做法：

    做法 A｜抽取式（本项目默认）
        把块拆成句子，用向量算每句和问题的相关度，只留高分的句子。
        快（一次批量编码）、便宜（不额外调 LLM）、可解释。
        缺点：句子被抽走后，前后的限定语可能丢掉。

    做法 B｜生成式（用 LLM 改写压缩）
        让 LLM 读一遍然后重写成精简版。更连贯，但要额外一次调用，
        成本翻倍、延迟翻倍，而且——很讽刺——引入 LLM 就有引入新幻觉的风险。
        用一个可能幻觉的步骤去防幻觉，这个账要算清楚。

本项目的取舍：默认走 A。B 作为对比实验的一个配置项存在（在优化文档里讨论）。

★ 这里有一个必须警惕的副作用，也是我在优化文档里如实记录的教训：
  抽取式压缩切断了句子之间的上下文，可能把限定条件切没：
      原文："会员商品支持 7 天无理由退货。**定制类商品除外。**"
      如果只保留第一句 → 回答"会员商品都能退"，把例外丢了。
  这种"压缩导致的断章取义"比不压缩更危险，因为它看起来有出处。
  缓解办法有两个，代码里都做了：
  (1) 保留"邻居句子"——命中一句时，把它的前后句也带上，保住上下文；
  (2) 压缩只在块数较多时启用，块数少时压缩收益小于风险，不如不压。
"""

import logging
import re

from langchain_core.documents import Document

from core.config import settings
from rag.embedder import get_embeddings

logger = logging.getLogger(__name__)

# 中文句子边界。这里刻意不切分"。"以外的短标点——
# 按逗号切会把一个完整意思拆碎，反而制造断章取义。
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？；!?;])")

# 太短的句子（比如"附则"）没有语义价值，直接跳过不参与打分
_MIN_SENTENCE_CHARS = 6


def split_sentences(text: str) -> list[str]:
    """把一段文字切成句子，保留标点。

    保留标点是必须的：标点在向量里也携带信息，
    而且拼回去的时候没有标点会读起来很别扭。
    """
    if not text:
        return []

    # 先按换行粗切，换行是比标点更强的边界
    parts: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # 表格行（Markdown 表格）整行保留，不要按标点切碎，
        # 否则行列对应关系会被破坏，模型会张冠李戴
        if line.startswith("|"):
            parts.append(line)
            continue
        parts.extend(p for p in _SENTENCE_SPLIT.split(line) if p.strip())

    return [p.strip() for p in parts if p.strip()]


class EmbeddingCompressor:
    """抽取式压缩器：按句子级向量相似度筛选。"""

    def __init__(self, keep_sentences: int | None = None):
        self.keep_sentences = keep_sentences or settings.COMPRESS_KEEP_SENTENCES

    def compress(
        self, query: str, documents: list[Document]
    ) -> tuple[list[Document], dict]:
        """压缩文档列表。

        返回 (压缩后的文档列表, 统计信息)。
        统计信息（省了多少字）要回流到日志和埋点里——
        优化效果必须可量化，不能只靠感觉说"变好了"。
        """
        if not documents:
            return [], {"before_chars": 0, "after_chars": 0, "dropped_sentences": 0}

        # 1) 把所有文档的句子摊平，同时记下它属于哪个文档、在原文档里的位置
        #    这样才能在挑选之后按原顺序拼回去
        all_sentences: list[str] = []
        index_map: list[tuple[int, int]] = []   # (文档下标, 句内下标)

        per_doc_sentences: list[list[str]] = []
        for doc_idx, doc in enumerate(documents):
            sentences = split_sentences(doc.page_content)
            # 表格行和过短句子不参与打分，但也不该被丢掉（它们通常是表头）
            sentences = [s for s in sentences if len(s) >= _MIN_SENTENCE_CHARS] or sentences
            per_doc_sentences.append(sentences)
            for sent_idx, sentence in enumerate(sentences):
                all_sentences.append(sentence)
                index_map.append((doc_idx, sent_idx))

        if not all_sentences:
            return documents, {
                "before_chars": sum(len(d.page_content) for d in documents),
                "after_chars": sum(len(d.page_content) for d in documents),
                "dropped_sentences": 0,
            }

        # 2) ★ 一次性批量编码所有句子，而不是逐句调用。
        #    这是本模块性能的关键：100 个句子分 100 次调用要好几秒，
        #    一次批量送进模型只要几百毫秒。差别是数量级的。
        embeddings = get_embeddings()
        query_vec = embeddings.embed_query(query)
        sentence_vecs = embeddings.embed_documents(all_sentences)

        # 3) 算余弦相似度。
        #    因为向量都已归一化（见 embedder.py 坑 2），内积就是余弦相似度，
        #    不需要再除以模长——这正是"归一化"这个约定带来的便利
        scored: list[tuple[int, float]] = []
        for i, vec in enumerate(sentence_vecs):
            dot = sum(a * b for a, b in zip(query_vec, vec))
            scored.append((i, dot))

        # 4) 每个文档各挑前 N 句。
        #    ★ 为什么是"每个文档各挑"而不是"全局挑前 M 句"？
        #      全局挑的话，很可能 5 个块里全被第 1 个块占满，
        #      其余 4 个块一句不剩。而多文档来源恰恰是交叉验证、
        #      发现资料冲突的前提。按文档分配配额，保住来源多样性。
        selected: dict[int, list[int]] = {}
        for doc_idx in range(len(documents)):
            doc_candidates = [
                (i, s) for i, s in scored if index_map[i][0] == doc_idx
            ]
            doc_candidates.sort(key=lambda pair: pair[1], reverse=True)
            keep = doc_candidates[: self.keep_sentences]
            # ★★ 必须把【全局下标】换算成【该文档内的局部下标】再存，别存 i。
            #
            #   踩过的坑：原先这里直接存了 keep 里的 i，而那个 i 是
            #   all_sentences 里的【全局】下标；到了第 5 步却拿它去索引
            #   单个文档的句子列表 per_doc_sentences[doc_idx]。
            #   两套下标空间并不一致——
            #       文档 0 的句子正好排在全局列表最前面，两套下标恰好对齐，
            #       所以单文档或"第一个文档"测试时完全正常；
            #       从文档 1 开始，全局下标必然超出该文档的句子数，直接
            #       IndexError 把整个问答打成 500。
            #   这类"只在多文档、且特定顺序下才炸"的错误最难排查，
            #   因为单测跑第一个文档永远是绿的。
            selected[doc_idx] = sorted(index_map[i][1] for i, _ in keep)

        # 5) 拼回文档，并且带上"邻居句子"缓解断章取义
        result: list[Document] = []
        dropped = 0

        for doc_idx, doc in enumerate(documents):
            keep_indices = set(selected.get(doc_idx, []))
            if not keep_indices:
                # 这个块一句都没留下，说明它和问题基本无关，整块丢弃
                dropped += len(per_doc_sentences[doc_idx])
                continue

            sentences = per_doc_sentences[doc_idx]
            total = len(sentences)

            # ★ 邻居扩展：命中句的前后各带一句，保住限定条件。
            #   原文"会员商品可退。定制类除外。"里，如果只有第一句得分高，
            #   第二句靠这个扩展被带回来，才不至于答出错误结论
            expanded = set(keep_indices)
            for i in keep_indices:
                if i - 1 >= 0:
                    expanded.add(i - 1)
                if i + 1 < total:
                    expanded.add(i + 1)

            # ★ 防御：正常情况下 expanded 里的下标一定落在 [0, total) 内
            #   （keep_indices 已经做过换算和边界检查）。但这里是
            #   "一个下标算错就让整个问答返回 500"的关键位置，
            #   所以宁可再挡一道：少拼一句话可以接受，把接口打崩不行。
            #   真越界了会打 warning，不至于静默吞掉问题
            valid = [i for i in sorted(expanded) if 0 <= i < total]
            if len(valid) != len(expanded):
                logger.warning(
                    "上下文压缩时出现越界句下标（文档序号 %s，共 %s 个越界），已跳过。"
                    "这不应该发生，请检查句子下标是否用错了空间",
                    doc_idx, len(expanded) - len(valid),
                )
            kept_text = "".join(sentences[i] for i in valid)
            dropped += total - len(valid)

            if not kept_text.strip():
                continue

            kept = Document(
                page_content=kept_text,
                # ★ metadata 必须原样带过去，一个字都不能丢：
                #   溯源信息（来自哪份文档、第几块）全在这里，
                #   丢了 metadata，压缩后的内容就成了无法引用的"孤儿"
                metadata=dict(doc.metadata),
            )
            result.append(kept)

        stats = {
            "before_chars": sum(len(d.page_content) for d in documents),
            "after_chars": sum(len(d.page_content) for d in result),
            "dropped_sentences": dropped,
        }
        logger.info(
            "上下文压缩：%s 字 → %s 字（丢弃 %s 句，保留策略=每块%s句+邻居）",
            stats["before_chars"],
            stats["after_chars"],
            stats["dropped_sentences"],
            self.keep_sentences,
        )
        return result, stats


def compress_documents(
    query: str,
    documents: list[Document],
    enabled: bool | None = None,
    min_docs_to_compress: int = 3,
) -> tuple[list[Document], dict]:
    """压缩入口。

    ★ min_docs_to_compress 这个参数体现了一个重要的工程判断：
      只有 1~2 个块的时候，压缩的收益（省几百字）远远小于它的风险
      （切断上下文、丢掉限定条件）。这时候最该做的是"别动它"。
      优化不是无脑开开关，要知道什么时候【不该】优化。
    """
    use_compression = settings.COMPRESS_ENABLED if enabled is None else enabled

    empty_stats = {
        "before_chars": sum(len(d.page_content) for d in documents),
        "after_chars": sum(len(d.page_content) for d in documents),
        "dropped_sentences": 0,
        "skipped": True,
    }

    if not use_compression or not documents:
        return documents, empty_stats

    if len(documents) < min_docs_to_compress:
        # 块数太少，压缩收益不抵风险，直接原样返回
        logger.debug("候选块仅 %s 个，跳过压缩", len(documents))
        return documents, empty_stats

    compressor = get_compressor()
    compressed, stats = compressor.compress(query, documents)
    stats["skipped"] = False
    return compressed, stats


_compressor: EmbeddingCompressor | None = None


def get_compressor() -> EmbeddingCompressor:
    """全局压缩器单例。"""
    global _compressor
    if _compressor is None:
        _compressor = EmbeddingCompressor()
    return _compressor
