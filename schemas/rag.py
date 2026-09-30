"""智能客服（RAG）相关的接口契约。

延续 schemas/user.py 的命名约定：XxxIn 是入参，XxxOut 是出参。

这里特意把几个"技术开关"暴露在了入参里（use_rerank / use_compression / top_k）：
正常业务接口不该让前端随便调这些，但本项目的定位是"技术验证 + 调优对比"，
需要一个能跑对照实验的入口。演示时可以说清楚：
    "这几个参数只对管理员开放，线上会锁死配置，这里开放是为了做 A/B 实验。"
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from schemas.general import PageResponse


# ==================== 知识库：文档管理 ====================


class KnowledgeDocOut(BaseModel):
    """知识库文档出参。"""

    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    filename: str
    file_type: str
    file_size: int
    status: str = Field(description="pending/processing/ready/failed")
    chunk_count: int
    char_count: int
    error_msg: str
    created_at: datetime


class KnowledgeDocUpdateIn(BaseModel):
    """修改文档信息（目前只允许改标题）。"""

    title: str = Field(..., min_length=1, max_length=200, description="新的标题")


class KnowledgeDocPage(PageResponse[KnowledgeDocOut]):
    """文档分页响应。继承通用分页结构，避免每个模块重写一遍。"""


class ChunkOut(BaseModel):
    """切片出参。用于"查看这份文档被切成了什么样"。"""

    model_config = ConfigDict(from_attributes=True)

    id: str
    chunk_index: int
    content: str
    char_count: int


class UploadResultOut(BaseModel):
    """上传结果。

    ★ 注意 filename 和 duplicate 这两个字段的设计意图：
      上传接口是"幂等"的——同一个文件传两次不会产生两份数据，
      第二次会返回 duplicate=true 并指回已有的那份文档。
    """

    doc: KnowledgeDocOut
    duplicate: bool = Field(default=False, description="是否为重复文件（未重复入库）")


# ==================== 问答：会话与消息 ====================


class ChatSessionCreateIn(BaseModel):
    """新建会话。"""

    title: str = Field(default="", max_length=100, description="会话标题，留空自动取首句")


class ChatSessionOut(BaseModel):
    """会话出参。"""

    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    message_count: int
    created_at: datetime
    updated_at: datetime


class ChatMessageOut(BaseModel):
    """消息出参。"""

    model_config = ConfigDict(from_attributes=True)

    id: str
    role: str
    content: str
    sources: list | None = None
    prompt_version: str = ""
    latency_ms: int = 0
    created_at: datetime


class SourceOut(BaseModel):
    """回答引用的一条来源。

    为什么要把 snippet（片段原文）返回给前端？
    因为"让用户能一眼核对原文"是抑制幻觉最有效的手段——
    比任何 Prompt 技巧都管用。用户看到原话，就不用盲信模型。
    """

    chunk_id: str
    doc_id: str
    doc_title: str
    score: float = Field(description="重排后的相关性分数，越大越相关")
    snippet: str = Field(description="命中的原文片段")


class ChatAskIn(BaseModel):
    """提问入参（对话的核心接口）。"""

    question: str = Field(..., min_length=1, max_length=2000, description="用户的问题")
    session_id: str | None = Field(
        default=None, description="会话 id；不传则自动新建一个会话"
    )

    # ---- 以下为实验开关，线上应锁定为服务端配置 ----
    top_k: int | None = Field(
        default=None, ge=1, le=20, description="最终喂给模型的片段数，默认取服务端配置"
    )
    use_rerank: bool | None = Field(
        default=None, description="是否启用重排，默认取服务端配置"
    )
    use_compression: bool | None = Field(
        default=None, description="是否启用上下文压缩，默认取服务端配置"
    )
    prompt_version: str | None = Field(
        default=None,
        description="指定 Prompt 版本（v1/v2/v3/v4），用于 A/B 对比；不传取服务端配置",
    )
    score_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "相似度阈值，低于它的片段会被丢弃。这是幻觉治理里最有效的一个参数，"
            "也是唯一一个'不相关的内容根本不进 Prompt'的手段。留空取服务端配置"
        ),
    )


class ChatAnswerOut(BaseModel):
    """回答出参。"""

    session_id: str
    message_id: str
    question: str
    answer: str
    sources: list[SourceOut] = Field(default_factory=list)

    # 调优埋点，原样回传，方便前端把"这一版 Prompt 有多快"显示出来
    prompt_version: str = ""
    model_name: str = ""
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # 检索到的候选片段总数（重排前），用于观察"召回够不够"
    retrieved_count: int = 0


# ==================== 检索调试（调优对比的硬证据）====================


class RetrieveDebugIn(BaseModel):
    """检索调试入参。

    这个接口是为"讲清楚重排到底做了什么"而设计的：
    它把「向量检索的原始排序」和「重排后的排序」并排返回，
    一眼就能看出重排把哪几条从后排提到了前排。
    """

    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20, description="最终返回几条")
    recall_k: int = Field(
        default=20, ge=1, le=50, description="重排前先召回几条（粗排的候选池大小）"
    )


class RetrievedItem(BaseModel):
    """单条检索结果。"""

    chunk_id: str
    doc_id: str
    doc_title: str
    snippet: str
    score: float = Field(description="当前排序依据的分数")
    rank: int = Field(description="当前排名（从 1 开始）")
    vector_rank: int | None = Field(
        default=None, description="它在向量检索阶段的原始排名，用于对照"
    )


class RetrieveDebugOut(BaseModel):
    """检索调试图。两种排序并排返回。"""

    query: str
    vector_hits: list[RetrievedItem] = Field(
        default_factory=list, description="向量检索的原始结果（未重排）"
    )
    reranked: list[RetrievedItem] = Field(
        default_factory=list, description="重排后的结果（未启用重排时与上面相同）"
    )
    rerank_enabled: bool = False
    elapsed_ms: int = 0
