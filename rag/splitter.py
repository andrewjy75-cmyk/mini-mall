"""文本切分：把长文档切成一块一块，是 RAG 里最"不起眼但最要命"的一步。

为什么非要切？三个原因：
1. 模型的上下文窗口塞不下整本手册
2. 就算塞得下，把整篇丢进去问一个问题，模型会被大量无关内容干扰，更容易答错
3. 向量检索的单位就是块。整篇文档的向量是"平均语义"，等于什么都没表达

切分参数怎么定？这是效果和成本的权衡，没有标准答案，但方向是明确的：

    块太大 → 一块里混好几个主题，向量被平均掉，检索到它也说不到点子上；
             而且喂给模型的 token 变多，更贵也更慢
    块太小 → 一句话被拦腰截断，语义不完整，检索和回答都会出错

中文场景的经验值：400~800 字一块，重叠 10%~20%。

★ 重叠（overlap）为什么必须要有？
因为切分点大概率落在句子中间。如果恰好把"退货期限是 7 天"切成
"...退货期限是" 和 " 7 天..."，那两块都不完整，检索时可能两块都匹配不好。
让相邻块重叠一段，保证任何一句话至少完整地出现在某一个块里。
代价是存储和向量计算量略微增加——这点代价换来召回率的提升非常划算。
"""

import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.loader import _normalize_text

logger = logging.getLogger(__name__)

# ★ 分隔符的优先级顺序，这是中文切分和英文最大的不同。
# RecursiveCharacterTextSplitter 会依次尝试用这些分隔符切：
# 先用 "\n\n" 切，如果某一块还是超长，就退回用 "\n" 切，以此类推。
#
# 顺序设计的原则是"从语义边界强到弱"：
#     段落 > 换行 > 句号 > 叹号/问号 > 分号 > 逗号 > 空格 > 单字
# 中文用「。！？」而不是英文的 ".!?"，因为中文句号是「。」。
# 末尾的 "" 是兜底：实在没有可切的地方，就硬切。
CHINESE_SEPARATORS = [
    "\n\n",   # 段落
    "\n",     # 换行
    "。",     # 中文句号
    "！",
    "？",
    "；",     # 分号：同层级条款的分隔
    "，",     # 逗号：最后的语义边界
    ".", "!", "?", ";",   # 英文标点，处理中英混排内容
    " ",
    "",       # 兜底硬切
]


def build_splitter(
    chunk_size: int = 600, chunk_overlap: int = 100
) -> RecursiveCharacterTextSplitter:
    """构造切分器。

    关于 length_function：默认是 len（按字符数算）。
    这里【故意不改】成 tiktoken 那种按 token 算的方式——
    按 token 算需要额外下载编码文件，而且中文一个字符往往就接近一个 token，
    按字符数估已经够准，还省一次网络依赖。
    """
    if chunk_overlap >= chunk_size:
        # 重叠不能大于等于块大小，否则切分会不收敛（一直原地打转）
        raise ValueError("chunk_overlap 必须小于 chunk_size")

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=CHINESE_SEPARATORS,
        # keep_separator="end"：让标点留在上一句末尾，读起来才自然。
        # 如果是 True，标点会被放到下一块开头，看起来像"…块以『，』开头"，很怪
        keep_separator="end",
        length_function=len,
        # 不做 strip 之外的额外处理，标点本身对检索是有信息的
        strip_whitespace=True,
    )


def _merge_tiny_chunks(
    chunks: list[Document], min_chars: int = 50, max_chars: int = 1200
) -> list[Document]:
    """合并过短的碎片块。

    为什么需要这一步？看这个真实例子：
        "...请联系客服。（完）"      ← 倒数第二块末尾
        "附录 A"                     ← 最后一块只有 3 个字
    这种 3 个字的块，向量几乎没有语义，但检索时它就是一条合法候选，
    还可能因为"碰巧"相似度高而挤掉真正有用的块。

    策略：太短的块尝试并入前一块（前提是合并后不超过 max_chars），
    并不动第一块（它是文档开头，往往包含标题和摘要，信息密度高）。
    """
    if len(chunks) <= 1:
        return chunks

    merged: list[Document] = [chunks[0]]
    for chunk in chunks[1:]:
        prev = merged[-1]
        combined_len = len(prev.page_content) + len(chunk.page_content)

        if len(chunk.page_content) < min_chars and combined_len <= max_chars:
            # 合并内容，注意补一个换行，别把两句话直接粘在一起
            prev.page_content = prev.page_content.rstrip() + "\n" + chunk.page_content.lstrip()
            continue

        merged.append(chunk)

    return merged


def split_documents(
    docs: list[Document],
    chunk_size: int = 600,
    chunk_overlap: int = 100,
    doc_id: str = "",
    doc_title: str = "",
    min_chunk_chars: int = 50,
) -> list[Document]:
    """把解析出来的 Document 切成检索用的块。

    参数:
        docs: loader 的产物
        chunk_size / chunk_overlap: 切分参数
        doc_id / doc_title: 会写进每块的 metadata，检索时用来溯源
        min_chunk_chars: 小于这个长度的块会被尝试合并进上一块

    返回:
        切好的 Document 列表，metadata 里带 chunk_index（从 0 开始）

    ★ 为什么每个块都要带 doc_title？
      块脱离原文后是"孤儿"。"第 7 天可退货"这句话，如果不知道出自
      《退换货政策》还是《生鲜商品特殊说明》，检索时无法区分，回答时也无法引用。
      标题本身就是极强的上下文信号，一起存进向量库能明显提升区分度。
    """
    if not docs:
        return []

    # 先把每一页/每段各自的文本内部再清洗一次（loader 已做过，这里防二次拼接产生的脏数据）
    for doc in docs:
        doc.page_content = _normalize_text(doc.page_content)

    splitter = build_splitter(chunk_size, chunk_overlap)

    # 逐份切再合并。不要把所有 docs 拼成一个列表一次切——
    # 那样会把"上一页的最后一段"和"下一页的第一段"错误地连成一句
    all_chunks: list[Document] = []
    for doc in docs:
        pieces = splitter.split_documents([doc])
        all_chunks.extend(pieces)

    all_chunks = _merge_tiny_chunks(all_chunks, min_chars=min_chunk_chars)

    # 补 metadata 并重新编号。
    # 注意：编号必须在这里统一做，因为 _merge_tiny_chunks 会改变块的数量，
    # 在切分器里拿到的序号已经不可靠了
    for index, chunk in enumerate(all_chunks):
        chunk.metadata["doc_id"] = doc_id
        chunk.metadata["doc_title"] = doc_title
        chunk.metadata["chunk_index"] = index
        chunk.metadata["char_count"] = len(chunk.page_content)

    logger.info(
        "切分完成 doc=%s：%s 段 → %s 块（size=%s overlap=%s）",
        doc_title or doc_id,
        len(docs),
        len(all_chunks),
        chunk_size,
        chunk_overlap,
    )
    return all_chunks
