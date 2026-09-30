"""大模型接入：用 LangChain 的 ChatOpenAI 连 DeepSeek。

★ 一个关键的认知：DeepSeek 提供的是**OpenAI 兼容接口**。
  意思是它的请求格式和 OpenAI 一模一样，只是域名和模型名不同。
  所以这里不需要为 DeepSeek 写专门的适配代码，直接用 LangChain 的
  ChatOpenAI，把 base_url 指过去就行：

      ChatOpenAI(base_url="https://api.deepseek.com/v1", model="deepseek-chat")

  这个事实的价值很大：以后要换成通义、Kimi、智谱、本地 vLLM，
  绝大多数也提供 OpenAI 兼容接口，**改一行配置即可切换，代码不用动**。
  这是"面向接口编程"在 LLM 领域的直接体现。

关于流式输出（streaming）：
客服场景下流式几乎是必须的。用户等 3 秒看到一个完整答案，
和答案一个字一个字往外蹦，体感差距巨大——虽然后者总耗时可能更长。
本模块通过参数控制，同一个模型工厂既支持流式也支持非流式。
"""

import logging
from typing import Any

from core.config import settings

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """大模型调用失败。message 是给人看的人话。"""


class LLMNotConfiguredError(LLMError):
    """没配 API Key。单独一个类型，方便启动时给出针对性提示。"""


# 模型工厂缓存：同样的参数只造一个实例，避免重复建连接池
_llm_cache: dict[tuple, Any] = {}


def get_llm(
    temperature: float | None = None,
    streaming: bool = False,
    model: str | None = None,
) -> Any:
    """获取 ChatOpenAI 实例（指向 DeepSeek）。

    参数:
        temperature: 采样温度。不传则用配置里的默认值（0.1）
        streaming:   是否流式输出
        model:       模型名，不传用配置默认（deepseek-chat）
    """
    if not settings.llm_configured:
        raise LLMNotConfiguredError(
            "还没有配置大模型 API Key，问答功能无法使用。\n"
            "请在项目根目录的 .env 文件里加上这一行：\n"
            "    LLM_API_KEY=sk-你的密钥\n"
            "密钥在 https://platform.deepseek.com/api_keys 申请。"
        )

    temperature = settings.LLM_TEMPERATURE if temperature is None else temperature
    model = model or settings.LLM_MODEL

    # 缓存键必须包含全部会改变实例行为的参数，否则会串配置
    cache_key = (model, temperature, streaming)
    if cache_key in _llm_cache:
        return _llm_cache[cache_key]

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise LLMError(
            "大模型依赖未安装。请先执行：\n"
            "    .venv\\Scripts\\python.exe -m pip install langchain-openai"
        ) from exc

    llm = ChatOpenAI(
        model=model,
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_BASE_URL,
        temperature=temperature,
        timeout=settings.LLM_TIMEOUT,
        max_retries=settings.LLM_MAX_RETRIES,
        streaming=streaming,
    )

    _llm_cache[cache_key] = llm
    logger.info("已创建 LLM 实例：model=%s temperature=%s streaming=%s", model, temperature, streaming)
    return llm


def friendly_error(exc: Exception) -> str:
    """把 SDK 抛出的技术性异常翻译成用户能看懂的话。

    ★ 这个函数看着不起眼，但它决定了"演示时翻车"和"演示时体面"的差别。
      真实场景里模型调用失败的原因就那几种，但堆栈信息全是英文技术词，
      直接甩给前端很难看，而且用户也不知道该做什么。
      翻译成人话之后，每种错误都附带一个"下一步该干什么"。
    """
    # 我们自己抛的 LLMError（含 LLMNotConfiguredError），
    # message 本来就是写给用户看的中文，直接返回。
    # 少了这一步，它会被下面的兜底分支包成
    # "调用大模型失败（LLMNotConfiguredError）：还没有配置…"，
    # 白白把 clean 的提示弄脏
    if isinstance(exc, LLMError):
        return str(exc)

    name = type(exc).__name__
    text = str(exc)

    # 认证失败：Key 写错、过期、或者带了多余的空格/引号
    if "Authentication" in name or "401" in text or "invalid_api_key" in text:
        return (
            "大模型 API Key 无效或已过期。请检查 .env 里的 LLM_API_KEY，"
            "注意复制时不要带上引号或多余空格。"
        )

    # 余额不足。DeepSeek 用 402 表示欠费
    if "402" in text or "Insufficient Balance" in text or "insufficient_quota" in text:
        return "大模型账户余额不足，请前往服务商控制台充值后再试。"

    # 限流
    if "RateLimit" in name or "429" in text:
        return "请求太频繁，被大模型服务商限流了。请稍等几秒再试。"

    # 超时 / 网络不通
    if "Timeout" in name or "timeout" in text.lower() or "Connect" in name:
        return (
            f"连接大模型服务超时（当前地址：{settings.LLM_BASE_URL}）。"
            "请检查网络，或确认该地址在当前网络环境下可访问。"
        )

    # 模型名写错
    if "model_not_found" in text or "does not exist" in text or "404" in text:
        return (
            f"模型 {settings.LLM_MODEL} 不存在。\n"
            f"请检查 .env 里的 LLM_MODEL 是否写对，"
            f"以及该模型在 {settings.LLM_BASE_URL} 这个服务上是否有权限使用。"
        )

    # 兜底：原文保留一部分，方便排查，但不整段甩出去
    return f"调用大模型失败（{name}）：{text[:200]}"


def check_llm_available() -> tuple[bool, str]:
    """做一次最轻量的连通性检查，启动时调用。

    为什么用一条极短的提问而不是 SDK 的 list_models？
    因为 list_models 在某些兼容实现上没做，反而会误报不可用。
    直接发一句最短的对话，能同时验证"网络通、Key 对、模型名对"三件事。
    """
    if not settings.llm_configured:
        return False, "未配置 LLM_API_KEY"

    try:
        llm = get_llm(temperature=0.0)
        llm.invoke("hi")
        return True, "大模型连通正常"
    except Exception as exc:  # noqa: BLE001
        return False, friendly_error(exc)
