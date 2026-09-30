"""对话接口：多轮问答、会话管理。

两种问答方式都提供，各有用途：
    POST /chat/ask         非流式，一次性返回。适合脚本、自动化评估
    POST /chat/ask/stream  流式（SSE），逐字返回。适合真人使用

★ 为什么流式接口用的是 POST 而不是 GET？
  因为浏览器的 EventSource API 只支持 GET。要用 POST 就得用
  fetch + ReadableStream 手动解析。这多写了几行前端代码，
  但换来的是：问题可以带完整的请求体（长问题、多个开关参数），
  不用把东西全塞进 URL 查询串。这个取舍是划算的。
"""

import logging

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.deps import get_current_user_optional
from models.user import User
from schemas.general import PageResponse, Result
from schemas.rag import (
    ChatAnswerOut,
    ChatAskIn,
    ChatMessageOut,
    ChatSessionCreateIn,
    ChatSessionOut,
)
from services.chat_service import ChatService
from services.rag_service import RagService

logger = logging.getLogger(__name__)

chat_router = APIRouter(prefix="/chat", tags=["智能客服·对话"])


def get_chat_service(db: AsyncSession = Depends(get_db)) -> ChatService:
    return ChatService(db)


def get_rag_service(db: AsyncSession = Depends(get_db)) -> RagService:
    return RagService(db)


# ==================== 会话管理 ====================


@chat_router.post(
    "/sessions",
    response_model=Result[ChatSessionOut],
    summary="新建会话",
)
async def create_session(
    data: ChatSessionCreateIn,
    current_user: User | None = Depends(get_current_user_optional),
    service: ChatService = Depends(get_chat_service),
):
    """新建一个对话会话。不登录也能建（匿名会话）。"""
    session = await service.create_session(
        user_id=current_user.id if current_user else None,
        title=data.title,
    )
    return Result.success(data=ChatSessionOut.model_validate(session), message="会话已创建")


@chat_router.get(
    "/sessions",
    response_model=Result[PageResponse[ChatSessionOut]],
    summary="我的会话列表",
)
async def list_sessions(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    current_user: User | None = Depends(get_current_user_optional),
    service: ChatService = Depends(get_chat_service),
):
    """按最近活跃时间倒序返回会话。"""
    sessions, total = await service.list_sessions(
        user_id=current_user.id if current_user else None,
        page=page,
        page_size=page_size,
    )
    return Result.success(
        data=PageResponse[ChatSessionOut](
            items=[ChatSessionOut.model_validate(s) for s in sessions],
            total=total,
            page=page,
            page_size=page_size,
        )
    )


@chat_router.get(
    "/sessions/{session_id}/messages",
    response_model=Result[list[ChatMessageOut]],
    summary="会话的历史消息",
)
async def list_messages(
    session_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    service: ChatService = Depends(get_chat_service),
):
    """返回完整历史。

    ★ 注意这里返回的是【原始消息】，不是给模型看的摘要。
      摘要只是模型的压缩视图，用户翻历史时必须看到自己说过的原话。
      把两者混为一谈是常见的设计错误——用户会发现自己的话"被改写了"。
    """
    await service.get_session(session_id)
    messages = await service.list_messages(session_id, limit=limit, offset=offset)
    return Result.success(data=[ChatMessageOut.model_validate(m) for m in messages])


@chat_router.delete(
    "/sessions/{session_id}",
    response_model=Result[None],
    summary="删除会话及其全部消息",
)
async def delete_session(
    session_id: str,
    service: ChatService = Depends(get_chat_service),
):
    await service.delete_session(session_id)
    return Result.success(message="会话已删除")


# ==================== 问答 ====================


@chat_router.post(
    "/ask",
    response_model=Result[ChatAnswerOut],
    summary="提问（非流式，一次性返回）",
)
async def ask(
    data: ChatAskIn,
    current_user: User | None = Depends(get_current_user_optional),
    service: RagService = Depends(get_rag_service),
):
    """完整走一遍 RAG 链路并返回答案与引用来源。

    不传 session_id 会自动新建会话，返回体里带着 session_id，
    后续追问带上它就能延续上下文。
    """
    answer = await service.ask(
        question=data.question,
        session_id=data.session_id,
        user_id=current_user.id if current_user else None,
        top_k=data.top_k,
        use_rerank=data.use_rerank,
        use_compression=data.use_compression,
        prompt_version=data.prompt_version,
        score_threshold=data.score_threshold,
    )
    return Result.success(data=answer)


@chat_router.post(
    "/ask/stream",
    summary="提问（流式 SSE）",
    response_class=StreamingResponse,
)
async def ask_stream(
    data: ChatAskIn,
    current_user: User | None = Depends(get_current_user_optional),
    service: RagService = Depends(get_rag_service),
):
    """流式问答，返回 text/event-stream。

    事件顺序：
        event: sources  引用来源（先推，让用户立刻有反馈）
        event: token    正文增量（多次）
        event: done     结束，带耗时与 token 统计
        event: error    出错（终结事件）

    ★ 关于 X-Accel-Buffering: no 这个响应头：
      它看起来无关紧要，但没有它，SSE 在 Nginx 反向代理后面会被
      攒着一起发——流式就变成了"等全部生成完再一次性出现"，
      流式的意义完全消失。
      本地直连 FastAPI 时不需要它，但真实部署几乎一定在前面挂 Nginx，
      所以现在就加上，避免上线时踩这个坑。
    """
    generator = service.ask_stream(
        question=data.question,
        session_id=data.session_id,
        user_id=current_user.id if current_user else None,
        top_k=data.top_k,
        use_rerank=data.use_rerank,
        use_compression=data.use_compression,
        prompt_version=data.prompt_version,
        score_threshold=data.score_threshold,
    )

    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",   # 禁用 Nginx 缓冲，见上方说明
        },
    )
