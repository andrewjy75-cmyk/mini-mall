"""问答主流程：把记忆、检索、Prompt、大模型串成一次完整的回答。

这是整条 RAG 链路的"装配车间"。它的顺序是刻意设计的，每一步都有理由：

    ① 取/建会话             → 后续所有步骤都挂在这个会话上
    ② 先存用户提问           → ★ 顺序很关键，见下方说明
    ③ 装载记忆（摘要+近期）   → 决定"它记不记得上一句"
    ④ 检索知识库             → 决定"它有没有据可依"
    ⑤ 组装 Prompt           → 把角色、记忆、资料、问题拼在一起
    ⑥ 调用大模型             → 这一步最慢，也最容易失败
    ⑦ 存回答 + 埋点          → 调优数据就是这么攒出来的
    ⑧ 视情况压缩历史          → 让对话能无限聊下去

★ 为什么②要在④之前，也就是"先落库再检索"？
  因为检索、调模型都可能失败（网络抖动、余额不足）。如果等全部成功再存用户提问，
  一旦失败，用户的问题就凭空消失了——用户刷新页面会发现自己的话不见了，
  还得重打一遍。先存下来，失败时至少对话记录是完整的，重试也有依据。

★ 关于埋点，再强调一次设计意图：
  每条回答都会记下 prompt_version / latency_ms / token 用量 / 命中的片段。
  单看一条没什么用，但累计几十条之后，就能回答这些问题：
      "v2 比 v1 的幻觉率低多少？"     → 按 prompt_version 分组统计
      "重排值不值它带来的 300ms？"     → 对比开关重排时的效果和延迟
      "哪些问题检索根本命中不了？"     → 看 retrieved_count=0 的比例
  没有这些数据，所谓的"优化"就只能靠嘴说。
"""

import json
import logging
import re
import time
from collections.abc import AsyncGenerator

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from models.chat import ROLE_ASSISTANT, ROLE_USER, ChatMessage, ChatSession
from rag import prompts
from rag.llm import LLMError, LLMNotConfiguredError, friendly_error, get_llm
from rag.retriever import RetrieveResult, aretrieve
from schemas.rag import ChatAnswerOut, SourceOut
from services.chat_service import ChatService
from util.id_util import new_id

logger = logging.getLogger(__name__)

# 匹配模型输出的引用编号，形如 "依据：[1][3]" 里的 1 和 3
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")

# 片段截断长度：返回给前端的引用预览不用太长，能看明白是什么就行
_SNIPPET_CHARS = 200


def parse_citations(answer: str) -> list[int]:
    """从回答里解析出模型声称引用的资料编号。

    ★ 这个小函数是"幻觉检测"的一个低成本探针，别小看它：
      我们给模型的资料编号只到 N。如果它写出了 [7]，而这次只给了 3 条资料，
      那这个 [7] 就是凭空编的——它根本没在资料里找到依据，
      只是学着 v3 的格式凑了个编号出来。
      这类"格式正确、内容虚构"的回答是最危险的，因为它看起来最像真的。

      校验逻辑在调用方，这里只负责把编号抠出来。
    """
    if not answer:
        return []
    return [int(m) for m in _CITATION_PATTERN.findall(answer)]


def _resolve_citations(
    cited: list[int], sources: list[SourceOut]
) -> tuple[list[int], list[int]]:
    """把模型引用的编号转成 sources 的索引，并挑出越界的非法引用。

    返回 (合法的 0 基索引, 非法的原始编号)。
    非法引用数会记进日志，是衡量"模型是否在硬凑依据"的直接指标。
    """
    valid: list[int] = []
    invalid: list[int] = []
    for number in dict.fromkeys(cited):   # 去重但保持顺序
        if 1 <= number <= len(sources):
            valid.append(number - 1)
        else:
            invalid.append(number)
    return valid, invalid


def _build_sources(result: RetrieveResult) -> list[SourceOut]:
    """把检索结果转成对外的引用对象。"""
    sources: list[SourceOut] = []
    for doc, score in result.items:
        meta = doc.metadata or {}
        sources.append(
            SourceOut(
                chunk_id=str(meta.get("chunk_id", "")),
                doc_id=str(meta.get("doc_id", "")),
                # 标题缺失时给个兜底，避免前端显示空白
                doc_title=str(meta.get("doc_title") or meta.get("source") or "未知文档"),
                score=round(float(score), 4),
                snippet=doc.page_content[:_SNIPPET_CHARS],
            )
        )
    return sources


class RagService:
    """问答服务。"""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.chat_service = ChatService(db)

    # ==================== 内部工具 ====================

    async def _prepare(
        self,
        question: str,
        session_id: str | None,
        user_id: str | None,
        top_k: int | None,
        use_rerank: bool | None,
        use_compression: bool | None,
        prompt_version: str | None,
        score_threshold: float | None = None,
    ) -> tuple[str, prompts.PromptTemplate, list[BaseMessage], str, RetrieveResult]:
        """准备阶段：会话、记忆、检索。

        把①~⑤步单独抽出来，是为了让流式和非流式两条路径共用同一套前置逻辑。
        否则两份代码会逐渐漂移（改了一处忘了另一处），是 bug 的温床。
        """
        # ① 取或建会话
        session = await self.chat_service.get_or_create_session(session_id, user_id)

        # ② 先把用户提问落库（理由见模块 docstring）
        user_message = ChatMessage(
            id=new_id(),
            session_id=session.id,
            role=ROLE_USER,
            content=question,
        )
        await self.chat_service.append_message(user_message)

        # ③ 装载记忆
        summary, history = await self.chat_service.load_memory(session)

        # ④ 检索知识库（这一步是异步的，内部丢到线程池跑，不会阻塞事件循环）
        result = await aretrieve(
            question,
            top_k=top_k,
            use_rerank=use_rerank,
            use_compression=use_compression,
            score_threshold=score_threshold,
        )

        # ⑤ 取 Prompt 版本 + 拼资料块
        template = prompts.get_prompt(prompt_version or settings.PROMPT_VERSION)
        context_block = prompts.build_context_block(
            [
                (str(doc.metadata.get("doc_title", "")), doc.page_content)
                for doc, _ in result.items
            ]
        )

        # 把长期摘要塞进历史的最前面。
        # 用 SystemMessage 而不是 HumanMessage，是因为摘要描述的是"背景事实"，
        # 不是某一轮对话内容，放进 system 语义更准确，模型也更不容易把它当成"用户说的话"
        messages: list[BaseMessage] = []
        if summary:
            messages.append(
                SystemMessage(content=f"以下是本次对话早前内容的摘要，供你参考：\n{summary}")
            )
        messages.extend(history)

        return session.id, template, messages, context_block, result

    async def _finalize(
        self,
        session_id: str,
        question: str,
        answer: str,
        sources: list[SourceOut],
        template: prompts.PromptTemplate,
        latency_ms: int,
        prompt_tokens: int,
        completion_tokens: int,
        cited_indices: list[int],
    ) -> ChatMessage:
        """收尾：写回答消息 + 埋点、触发历史摘要。"""
        message = ChatMessage(
            id=new_id(),
            session_id=session_id,
            role=ROLE_ASSISTANT,
            content=answer,
            # 只把模型【真正引用到】的来源做标记，全量来源仍然返回给前端展示。
            # 为什么要区分？因为"提供了 5 条"和"用了 2 条"是两件事，
            # 后者才是模型实际依据的东西，溯源时要看的是它
            sources=[
                {
                    "chunk_id": s.chunk_id,
                    "doc_id": s.doc_id,
                    "doc_title": s.doc_title,
                    "score": s.score,
                    "cited": i in cited_indices,
                }
                for i, s in enumerate(sources)
            ],
            prompt_version=template.version,
            model_name=settings.LLM_MODEL,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        await self.chat_service.append_message(message)

        # ⑧ 触发历史压缩（内部会自己判断够不够格触发）
        session = await self.db.get(ChatSession, session_id)
        if session is not None:
            await self.chat_service.maybe_summarize(session)

        return message

    # ==================== 非流式问答 ====================

    async def ask(
        self,
        question: str,
        session_id: str | None = None,
        user_id: str | None = None,
        top_k: int | None = None,
        use_rerank: bool | None = None,
        use_compression: bool | None = None,
        prompt_version: str | None = None,
        score_threshold: float | None = None,
    ) -> ChatAnswerOut:
        """一次问答（非流式）。适合脚本调用和评估跑批。"""
        started = time.perf_counter()

        session_id, template, history, context_block, result = await self._prepare(
            question, session_id, user_id, top_k, use_rerank, use_compression,
            prompt_version, score_threshold,
        )
        sources = _build_sources(result)

        # ⑥ 调模型
        messages = [
            SystemMessage(content=template.system),
            *history,
            HumanMessage(content=template.human.format(context=context_block, question=question)),
        ]

        # ★ get_llm() 必须放在 try 里面。
        #
        #   它会先检查 API Key 配没配，没配就抛 LLMNotConfiguredError。
        #   原先这一句写在 try 外面，异常就一路冒到全局兜底处理器，
        #   用户看到的是"服务器开小差了，请稍后再试"——
        #   原因明明很明确（key 没填），却提示成"服务器坏了"，
        #   用户只会去翻代码，根本想不到是配置问题。
        #
        #   这类 bug 的典型特征：不是"没做异常处理"，而是"没盖住"。
        #   写 try 的时候一定要问一句：可能出错的那一行，在 try 里面吗？
        try:
            llm = get_llm(temperature=template.temperature)
            response = await llm.ainvoke(messages)
        except LLMError:
            # 我们自己抛的异常，message 已经是给人看的话了，原样交给全局处理器
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("调用大模型失败")
            raise LLMError(friendly_error(exc)) from exc

        answer = response.content if isinstance(response.content, str) else str(response.content)

        # 解析引用，校验合法性
        cited = parse_citations(answer)
        cited_indices, invalid = _resolve_citations(cited, sources)
        if invalid:
            logger.warning(
                "模型引用了不存在的资料编号 %s（本次只提供了 %s 条），疑似硬凑依据",
                invalid,
                len(sources),
            )

        # token 用量：不同版本 SDK 的字段位置不一样，做个兼容取值
        usage = getattr(response, "usage_metadata", None) or {}
        prompt_tokens = int(usage.get("input_tokens", 0) or 0)
        completion_tokens = int(usage.get("output_tokens", 0) or 0)

        latency_ms = int((time.perf_counter() - started) * 1000)

        message = await self._finalize(
            session_id, question, answer, sources, template,
            latency_ms, prompt_tokens, completion_tokens, cited_indices,
        )

        return ChatAnswerOut(
            session_id=session_id,
            message_id=message.id,
            question=question,
            answer=answer,
            sources=sources,
            prompt_version=template.version,
            model_name=settings.LLM_MODEL,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            retrieved_count=len(result.vector_hits),
        )

    # ==================== 流式问答 ====================

    async def ask_stream(
        self,
        question: str,
        session_id: str | None = None,
        user_id: str | None = None,
        top_k: int | None = None,
        use_rerank: bool | None = None,
        use_compression: bool | None = None,
        prompt_version: str | None = None,
        score_threshold: float | None = None,
    ) -> AsyncGenerator[str, None]:
        """一次问答（流式），产出 SSE 格式的字符串。

        ★ 事件顺序是刻意设计的：
          先推 sources（引用来源），再推 token（正文）。
          为什么？因为检索在生成之前就完成了，来源是【已知且确定】的。
          先把它推给前端，用户能立刻看到"参考了这 3 篇资料"，
          在大模型吐出第一个字之前就有反馈——感知延迟直接降低一大截。

        SSE 格式说明：每条消息是 `event: 类型\\ndata: 内容\\n\\n`，
        两个换行才算一条消息结束，少一个换行前端就收不到。
        """
        started = time.perf_counter()

        # ---- 准备阶段 ----
        try:
            session_id, template, history, context_block, result = await self._prepare(
                question, session_id, user_id, top_k, use_rerank, use_compression,
                prompt_version, score_threshold,
            )
        except LLMNotConfiguredError as exc:
            yield _sse("error", {"message": str(exc)})
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception("问答准备阶段失败")
            yield _sse("error", {"message": friendly_error(exc)})
            return

        sources = _build_sources(result)
        # 第一件事就把来源推出去
        yield _sse(
            "sources",
            {
                "session_id": session_id,
                "sources": [s.model_dump() for s in sources],
                "retrieved_count": len(result.vector_hits),
                "prompt_version": template.version,
                "search_ms": result.elapsed_ms,
            },
        )

        # ---- 生成阶段 ----
        messages = [
            SystemMessage(content=template.system),
            *history,
            HumanMessage(content=template.human.format(context=context_block, question=question)),
        ]

        buffer: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0

        try:
            # get_llm 同样要放进 try。理由和 ask() 一样，
            # 只是流式更麻烦：此时响应头已经发出去了，
            # 全局异常处理器根本插不上手，只能在这里自己兜住
            llm = get_llm(temperature=template.temperature, streaming=True)
            async for chunk in llm.astream(messages):
                text = chunk.content
                if not text:
                    continue
                if not isinstance(text, str):
                    text = str(text)
                buffer.append(text)
                yield _sse("token", {"text": text})

                # 流式的 token 用量通常在最后一个 chunk 上才带回来
                usage = getattr(chunk, "usage_metadata", None) or {}
                if usage:
                    prompt_tokens = int(usage.get("input_tokens", 0) or 0)
                    completion_tokens = int(usage.get("output_tokens", 0) or 0)
        except Exception as exc:  # noqa: BLE001
            logger.exception("流式生成中断")
            partial = "".join(buffer)
            # ★ 已经吐出去的字收不回来了，所以要把半截回答也存下来，
            #   否则用户看到的和数据库里记的对不上，后续排查会非常困惑
            latency_ms = int((time.perf_counter() - started) * 1000)
            if partial:
                await self._finalize(
                    session_id, question, partial + "\n\n[回答中断]",
                    sources, template, latency_ms, prompt_tokens, completion_tokens, [],
                )
            yield _sse("error", {"message": friendly_error(exc)})
            return

        answer = "".join(buffer)
        latency_ms = int((time.perf_counter() - started) * 1000)

        cited = parse_citations(answer)
        cited_indices, invalid = _resolve_citations(cited, sources)
        if invalid:
            logger.warning("流式回答引用了不存在的编号 %s", invalid)

        message = await self._finalize(
            session_id, question, answer, sources, template,
            latency_ms, prompt_tokens, completion_tokens, cited_indices,
        )

        yield _sse(
            "done",
            {
                "message_id": message.id,
                "session_id": session_id,
                "latency_ms": latency_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "prompt_version": template.version,
                "model_name": settings.LLM_MODEL,
                "cited": cited_indices,
                "invalid_citations": invalid,
            },
        )


def _sse(event: str, data: dict) -> str:
    """拼一条 SSE 消息。

    ensure_ascii=False 是必须的：默认会转义成 \\uXXXX，
    虽然功能上没错，但中文在抓包和日志里全变成转义序列，排查问题极其痛苦。
    """
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"
