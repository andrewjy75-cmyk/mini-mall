"""会话表：多轮对话的"记忆"存在哪、怎么存。

"多轮会话记忆"听着玄，拆开就两件事：

1. 【短期记忆】最近几轮的原话，直接塞进 Prompt。
   为什么只放"最近几轮"？因为上下文窗口是有限的，而且越长越贵、越容易跑偏
   （模型会被远处的无关内容干扰，这叫 lost in the middle）。
2. 【长期记忆】更早的对话不能直接丢，丢了下句问"那它多少钱"就接不上了。
   做法是压缩成一段摘要存起来，见 ChatSession.summary。

所以记忆不是"全都记住"，而是"分层保留"。

★ 另外，ChatMessage 里塞了一堆"看起来和学习无关"的埋点字段
  （prompt_version / latency_ms / prompt_tokens / sources）。
  这些是故意的：季度考核要交"幻觉调优对比"，如果等做完了才想起来要埋点，
  数据就永远拿不回来了。埋点必须在写第一行业务代码时就设计好。
"""

from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from core.database import Base

# 消息角色。固定三个值，别自己造词
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"


class ChatSession(Base):
    """一次多轮对话（相当于微信里的一个聊天窗口）。"""

    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    # 允许为空：考核演示时想让访客不登录也能试两句
    user_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id"), index=True, nullable=True
    )
    # 默认取用户说的第一句话前 30 个字，用户也能自己改
    title: Mapped[str] = mapped_column(String(100), default="新对话")

    # ★ 长期记忆：把超出窗口的历史对话压缩成一段摘要。
    #   注意它是"滚动更新"的——每轮把「旧摘要 + 新对话」再压缩一次，
    #   所以它不会无限增长。细节在 services/chat_service.py 里。
    summary: Mapped[str] = mapped_column(Text, default="")

    # ★ 摘要游标：前多少条消息已经被"揉进" summary 里了。
    #   为什么需要这个数字？因为摘要必须【幂等】：
    #   不能每次都把全部历史重新摘要一遍（越摘越模糊，还白花 token），
    #   也不能摘错位置（重复摘要或漏掉）。
    #   有了游标，逻辑就明确了：
    #       摘要 = [0, summarized_count) 这一段
    #       原文 = [summarized_count, 末尾] 这一段
    #   记忆 = summary + 原文这一段。
    summarized_count: Mapped[int] = mapped_column(Integer, default=0)

    message_count: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), index=True
    )

    def __repr__(self) -> str:
        return f"<ChatSession {self.title} {self.message_count}条>"


class ChatMessage(Base):
    """一条消息（用户问的，或助手答的）。"""

    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("chat_sessions.id"), index=True
    )
    # user / assistant，见文件顶部常量
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)

    # ---------------- 以下为「回答溯源 + 调优」埋点 ----------------

    # 这次回答引用了哪些知识片段。
    # 结构：[{"chunk_id":..., "doc_id":..., "doc_title":..., "score":..., "snippet":...}]
    # 用途有两个：一是前端要显示引用卡片（让用户能自己核对，这是治幻觉最有效的一招）；
    #             二是出了幻觉能倒查"是不是检索环节就找错了"
    sources: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # ★ 这次回答用的是哪一版 Prompt。做 A/B 对比全靠它——
    #   没有这个字段，"优化后效果变好了"这句话就没有任何证据。
    prompt_version: Mapped[str] = mapped_column(String(32), default="", index=True)
    model_name: Mapped[str] = mapped_column(String(64), default="")

    # 首字延迟 / 总耗时（毫秒）。优化往往拿效果换速度，
    # 只报"效果提升"不报"慢了 700ms"是不老实的
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)

    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    def __repr__(self) -> str:
        return f"<Msg {self.role} {self.content[:20]!r}>"
