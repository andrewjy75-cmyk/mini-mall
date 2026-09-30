"""知识库表：上传的文档 + 切分出来的文本块。

为什么明明用了向量库，原文还要在 MySQL 里再存一份？
三个理由，缺一个都会难受：

1. 管理需要：文档列表、删除某份文档、重建索引，都要能按"文档"这个维度操作。
   向量库的删除接口是按 id 删的，想删"整份文档的所有块"得自己记 id，很别扭。
2. 对账需要：向量库里的每条向量都必须能对回原文，否则排查"检索结果为什么是它"
   就成了无头案。
3. 溯源需要：回答里要显示"这句话出自哪份文档的第几段"，这类展示字段不适合塞进
   向量库的 metadata。

所以分工很明确，记住这条线：
    MySQL   = 唯一事实来源（source of truth），存结构化记录和原文
    向量库   = 只管"相似度检索"这一件事，存向量 + 少量过滤字段
两边靠同一个 id 对上：KnowledgeChunk.id 就是向量库里的 vector_id。
"""

from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from core.database import Base

# 文档处理状态。走的是"异步流水线"的思路：上传先落库，处理过程改状态。
# 这样前端能轮询到进度，而不是一个请求卡在那里等半天。
DOC_STATUS_PENDING = "pending"        # 已入库，排队等解析
DOC_STATUS_PROCESSING = "processing"  # 正在解析 / 切分 / 向量化
DOC_STATUS_READY = "ready"            # 处理完，可被检索
DOC_STATUS_FAILED = "failed"          # 处理失败，error_msg 里有原因


class KnowledgeDoc(Base):
    """知识库文档：对应"用户上传的一个文件"。

    电商客服场景下，这就是"退换货政策.pdf""商品保修说明.docx"这类资料。
    """

    __tablename__ = "knowledge_docs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    # 展示用标题。默认取文件名（去掉后缀），允许上传时指定
    title: Mapped[str] = mapped_column(String(200), index=True)
    # 用户上传时的原始文件名，出错时要能报给用户看
    filename: Mapped[str] = mapped_column(String(255))
    # pdf / docx / txt。存下来是为了前端能显示对应图标，也便于按类型统计
    file_type: Mapped[str] = mapped_column(String(16), index=True)
    file_size: Mapped[int] = mapped_column(Integer, default=0)

    # ★ 内容指纹（sha256）。作用：同一个文件传两次，第二次直接拦掉。
    #   没有它，用户手抖点两下上传，知识库里就有两份一模一样的内容，
    #   检索时会返回两条几乎相同的片段，白白挤占上下文窗口。
    file_hash: Mapped[str] = mapped_column(String(64), index=True, default="")

    # 原始文件落盘路径。为什么要留着原文件？因为切分策略、向量模型以后都可能换，
    # 到时候要能"用新策略重新处理一遍"，没原文件就得让用户重传。
    stored_path: Mapped[str] = mapped_column(String(500), default="")

    # pending / processing / ready / failed，见文件顶部常量
    status: Mapped[str] = mapped_column(
        String(16), default=DOC_STATUS_PENDING, index=True
    )
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    # 失败原因。只给管理员看，不直接甩给终端用户
    error_msg: Mapped[str] = mapped_column(Text, default="")

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    def __repr__(self) -> str:
        return f"<KnowledgeDoc {self.title} {self.status} {self.chunk_count}块>"


class KnowledgeChunk(Base):
    """文档切片：切分后的一小段文本，是检索的最小单位。

    切多大切多小是有讲究的，这里先记结论，细节在 rag/splitter.py 里讲：
    - 切太大：一段里混了好几个主题，向量被"平均"掉，检索不准
    - 切太小：一句话被拦腰截断，语义不完整
    """

    __tablename__ = "knowledge_chunks"

    # ★ 这个 id 直接当向量库的 vector_id 用，是两边对上的唯一纽带。
    #   千万别用自增数字——多机部署会撞号。
    __table_args__ = (
        UniqueConstraint("doc_id", "chunk_index", name="uq_chunk_doc_index"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    doc_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("knowledge_docs.id"), index=True
    )
    # 在原文档里的序号（0 开始）。展示"第 3 段"、按顺序还原上下文都要靠它
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    char_count: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    def __repr__(self) -> str:
        return f"<Chunk {self.doc_id[:8]}#{self.chunk_index} {self.char_count}字>"
