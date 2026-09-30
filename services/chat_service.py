"""会话管理：多轮对话的"记忆"怎么存、怎么取、怎么压缩。

这个模块回答一个问题：用户第二句问"那运费谁出？"的时候，
系统怎么知道他在说什么？

答案是【分层记忆】，不是"把历史全部塞进去"：

    ┌──────────────────────────────────────────┐
    │  System Prompt（角色 + 铁律）              │  ← 每轮都带
    ├──────────────────────────────────────────┤
    │  长期记忆：summary（300 字以内的滚动摘要）  │  ← 每轮都带
    ├──────────────────────────────────────────┤
    │  短期记忆：最近 3 轮的原话                  │  ← 每轮都带
    ├──────────────────────────────────────────┤
    │  当前轮：检索到的资料 + 用户这次的问题       │  ← 每轮变化
    └──────────────────────────────────────────┘

为什么不干脆把全部历史都带上？
1. 上下文窗口有限，聊二十轮就爆了
2. 就算塞得下，太多历史会让模型"跑偏"——它会揪住很早之前的话题不放
3. 每轮都要为所有历史重新计费，成本随轮数线性上涨

所以真正的问题是【怎么优雅地忘掉旧信息】。本项目的答案是滚动摘要：
把早于最近 N 轮的对话，连同旧摘要一起，重新压成一份新摘要。

★ 摘要游标（ChatSession.summarized_count）是这里的关键设计，它保证了：
  - 幂等：已经摘要过的消息不会被重复摘要（否则会"摘要的摘要"，越摘越糊）
  - 可追溯：摘要覆盖到哪一条，是有明确边界记录的
  - 不丢原文：消息本身仍然完整存在数据库里，用户翻历史还能看到全部。
    摘要只是给【模型】看的压缩视图，不是对用户的数据做删减。
"""

import logging

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from core.exception import BizException
from models.chat import ROLE_ASSISTANT, ROLE_USER, ChatMessage, ChatSession
from rag.prompts import SUMMARY_PROMPT
from util.id_util import new_id

logger = logging.getLogger(__name__)


class ChatService:
    """会话与消息的业务逻辑。"""

    def __init__(self, db: AsyncSession):
        self.db = db

    # ==================== 会话 ====================

    async def create_session(self, user_id: str | None = None, title: str = "") -> ChatSession:
        """新建一个会话。"""
        session = ChatSession(
            id=new_id(),
            user_id=user_id,
            title=(title or "新对话")[:100],
        )
        self.db.add(session)
        await self.db.commit()
        await self.db.refresh(session)
        return session

    async def get_session(self, session_id: str) -> ChatSession:
        """按 id 取会话，取不到抛业务异常。"""
        session = await self.db.get(ChatSession, session_id)
        if session is None:
            raise BizException("会话不存在或已过期", code=3001)
        return session

    async def get_or_create_session(
        self, session_id: str | None, user_id: str | None = None
    ) -> ChatSession:
        """有就取，没有就新建。

        这是问答接口的入口逻辑：前端不传 session_id 时自动开一个新会话，
        省掉"先调建会话接口再提问"这一步。
        """
        if session_id:
            return await self.get_session(session_id)
        return await self.create_session(user_id=user_id)

    async def list_sessions(
        self, user_id: str | None = None, page: int = 1, page_size: int = 20
    ) -> tuple[list[ChatSession], int]:
        """我的会话列表，按最近活跃排序。"""
        stmt = select(ChatSession)
        count_stmt = select(func.count()).select_from(ChatSession)

        if user_id:
            stmt = stmt.where(ChatSession.user_id == user_id)
            count_stmt = count_stmt.where(ChatSession.user_id == user_id)

        total = await self.db.scalar(count_stmt) or 0
        stmt = stmt.order_by(ChatSession.updated_at.desc())
        stmt = stmt.offset((page - 1) * page_size).limit(page_size)
        sessions = list((await self.db.scalars(stmt)).all())
        return sessions, total

    async def delete_session(self, session_id: str) -> None:
        """删除会话，连同它的全部消息一起删。

        为什么要手动删消息？因为数据库里没有建外键级联。
        这是有意的：级联删除一旦配错，一个误操作能带走一大片数据。
        在应用层显式删，每一处删除都是看得见、可以加日志的。
        """
        from sqlalchemy import delete

        session = await self.get_session(session_id)
        await self.db.execute(
            delete(ChatMessage).where(ChatMessage.session_id == session.id)
        )
        await self.db.delete(session)
        await self.db.commit()
        logger.info("会话已删除：%s", session.id)

    # ==================== 消息 ====================

    async def list_messages(
        self, session_id: str, limit: int = 50, offset: int = 0
    ) -> list[ChatMessage]:
        """取会话的消息列表（升序，最早的在前面）。"""
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc())
            .offset(offset)
            .limit(limit)
        )
        return list((await self.db.scalars(stmt)).all())

    async def append_message(self, message: ChatMessage) -> ChatMessage:
        """追加一条消息并更新会话的轮次统计。"""
        self.db.add(message)

        session = await self.db.get(ChatSession, message.session_id)
        if session is not None:
            session.message_count += 1
            # ★ 用第一句用户提问当会话标题。
            #   为什么不建会话时就定标题？因为那时还不知道用户要问什么，
            #   全是"新对话"的列表根本没法定位。
            if (
                session.message_count == 1
                and message.role == ROLE_USER
                and session.title in ("新对话", "")
            ):
                session.title = message.content[:30]

        await self.db.commit()

        # 刷新会话（拿回数据库生成的 updated_at）和消息（拿回 created_at）。
        # 理由同 knowledge_service：这两个时间字段是 server_default 生成的，
        # 不刷新的话，接口层序列化时会触发异步 Session 里不允许的隐式惰性加载
        if session is not None:
            await self.db.refresh(session)
        await self.db.refresh(message)
        return message

    # ==================== 记忆 ====================

    async def load_memory(
        self, session: ChatSession
    ) -> tuple[str, list[BaseMessage]]:
        """装载记忆，返回 (长期摘要, 最近若干轮的消息对象)。

        ★ 只取 "summarized_count 之后" 的消息：
          再往前的已经被揉进 summary 了，重复带上等于给模型灌两遍同样的内容，
          既浪费 token，又容易让模型把同一件事当成两件事来说。
        """
        messages = await self.list_messages(
            session.id, limit=100, offset=session.summarized_count
        )

        # 只保留最近 N 轮（1 轮 = 一问一答，所以是 N*2 条）
        keep = settings.MEMORY_RECENT_TURNS * 2
        recent = messages[-keep:] if len(messages) > keep else messages

        history: list[BaseMessage] = []
        for msg in recent:
            if msg.role == ROLE_USER:
                history.append(HumanMessage(content=msg.content))
            elif msg.role == ROLE_ASSISTANT:
                history.append(AIMessage(content=msg.content))

        return session.summary or "", history

    async def maybe_summarize(self, session: ChatSession) -> bool:
        """检查是否需要触发摘要压缩。返回是否真的压缩了。

        触发条件：未被摘要的消息条数，超过了「最近轮数 + 触发阈值」。

        为什么阈值要比"最近保留轮数"大？
        因为如果一超过就立刻摘要，会出现"刚聊两轮就摘一次"的情况——
        每摘一次都要调一次 LLM（花钱、变慢），而且摘得太碎，
        每次只揉进去一两句，摘出来的东西信息量很低。
        留出一段缓冲（配置里的 MEMORY_SUMMARY_TRIGGER_TURNS），
        让摘要"攒够了再压一次"，性价比更高。
        """
        keep_recent = settings.MEMORY_RECENT_TURNS * 2
        trigger_at = settings.MEMORY_SUMMARY_TRIGGER_TURNS * 2

        unsummarized = session.message_count - session.summarized_count
        if unsummarized <= trigger_at:
            return False

        # 要摘要的是"最早的那一批"，最近 keep_recent 条留着当原文
        to_summarize = unsummarized - keep_recent
        messages = await self.list_messages(
            session.id, limit=to_summarize, offset=session.summarized_count
        )
        if not messages:
            return False

        conversation = "\n".join(
            f"{'用户' if m.role == ROLE_USER else '客服'}：{m.content}" for m in messages
        )
        previous = (
            f"已有摘要（请把新内容合并进去）：\n{session.summary}"
            if session.summary
            else "（当前还没有摘要）"
        )

        prompt = SUMMARY_PROMPT.format(
            previous_summary=previous, new_conversation=conversation
        )

        try:
            from rag.llm import get_llm

            llm = get_llm(temperature=0.2)
            result = await llm.ainvoke(prompt)
            summary = (result.content if hasattr(result, "content") else str(result)).strip()
        except Exception as exc:  # noqa: BLE001
            # ★ 摘要失败绝不能影响对话本身。
            #   记忆压缩是"锦上添花"的能力，它挂了顶多是下一轮忘掉些旧信息，
            #   而如果让它把整个问答请求带崩，那就本末倒置了。
            logger.warning("会话摘要生成失败，本轮跳过压缩（不影响对话）：%s", exc)
            return False

        if not summary:
            return False

        session.summary = summary[:2000]
        session.summarized_count += len(messages)
        await self.db.commit()

        logger.info(
            "会话 %s 已压缩 %s 条历史为摘要（游标 → %s）",
            session.id[:8],
            len(messages),
            session.summarized_count,
        )
        return True
