"""Mini Mall 商城 —— 启动入口。

运行方式：
    .venv\\Scripts\\python.exe main.py
然后打开 http://127.0.0.1:8600/docs 看接口文档，
或者 http://127.0.0.1:8600/ 直接用智能客服聊天页面。

★ 启动时会做几件"看起来多余"的事（预热模型、检查大模型连通性），
  但它们的价值在演示当天就会体现出来：把"第一个请求特别慢"
  和"提问时报一个看不懂的错"这两类问题，提前到启动阶段暴露。
  启动日志里就能看到哪里没配好，而不是等讲到一半才发现。
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from api.chat_api import chat_router
from api.hello_api import hello_router
from api.knowledge_api import knowledge_router
from api.user_api import user_router
from core.config import settings
from core.database import engine
from core.exception import register_exception_handlers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# 前端（Vue）开发服务器的地址。跨域只放行这几个，不要用 "*"
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

# 静态文件目录（内置的极简聊天页）
STATIC_DIR = Path(__file__).resolve().parent / "static"


async def _startup_checks() -> None:
    """启动自检：把常见的配置问题在启动时就暴露出来。

    每一项都是"真的踩过坑"才加进来的：
    - 数据库连不上：服务看起来起来了，一登录就 500
    - 没配 LLM_API_KEY：上传文档一切正常，一提问就报错
    - 向量模型没预热：第一次检索要等好几秒，像卡死了
    - 重排模型不可用：功能正常但效果悄悄变差，没有提示就永远发现不了
    """
    logger.info("=" * 62)
    logger.info("  %s 正在启动（环境：%s）", settings.PROJECT_NAME, settings.APP_ENV)
    logger.info("=" * 62)

    # 1) 数据库连通性
    try:
        from sqlalchemy import text

        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("[✓] 数据库连接正常：%s:%s/%s",
                    settings.MYSQL_HOST, settings.MYSQL_PORT, settings.MYSQL_DB)
    except Exception as exc:  # noqa: BLE001
        logger.error("[✗] 数据库连接失败：%s", exc)
        logger.error("    请检查 .env 里的 MYSQL_* 配置，以及数据库服务是否已启动")

    # 2) 大模型配置
    if settings.llm_configured:
        logger.info("[✓] 大模型已配置：%s @ %s", settings.LLM_MODEL, settings.LLM_BASE_URL)
    else:
        logger.warning("[!] 未配置 LLM_API_KEY —— 知识库功能可用，但【提问会失败】")
        logger.warning("    请在 .env 里添加：LLM_API_KEY=sk-你的密钥")

    # 3) 预热向量模型
    #    放到线程里跑：模型加载是同步阻塞的，直接 await 会把事件循环卡住，
    #    连启动流程都走不完
    try:
        from rag.embedder import warmup

        logger.info("正在预热向量模型（首次运行会自动下载，请稍候）...")
        ok = await asyncio.to_thread(warmup)
        if ok:
            logger.info("[✓] 向量模型就绪：%s", settings.EMBEDDING_MODEL)
        else:
            logger.warning("[!] 向量模型预热失败，知识库检索将不可用（详见上方日志）")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[!] 向量模型预热异常：%s", exc)

    # 4) 重排模型提示
    #    重排模型有 1GB 多，启动时就下载会让人觉得"服务起不来"，
    #    所以这里【只提示不加载】，等第一次真正用到时再懒加载
    if settings.RERANK_ENABLED:
        logger.info(
            "[i] 重排已启用（%s）。该模型约 1GB，将在首次检索时自动下载；"
            "若下载失败会自动降级为不重排，功能不受影响。",
            settings.RERANK_MODEL,
        )

    logger.info("接口文档：http://127.0.0.1:%s/docs", settings.SERVER_PORT)
    logger.info("聊天页面：http://127.0.0.1:%s/", settings.SERVER_PORT)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时自检，关闭时释放连接池。

    用 lifespan 而不是已废弃的 @app.on_event，是 FastAPI 现在的推荐写法。
    """
    await _startup_checks()
    yield
    # 关闭：把数据库连接池和向量库句柄放掉。
    # 不释放的话，热重载时旧连接会一直挂着，MySQL 那边能看到一堆空闲连接
    await engine.dispose()
    logger.info("服务已关闭，连接池已释放")


def create_app() -> FastAPI:
    """创建应用。写成函数（工厂模式），以后写测试时可以造多个互不干扰的实例。"""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        description=(
            "一个从零搭建的练手商城：用户 / 商品 / 购物车 / 订单\n\n"
            "并在此基础上扩展了【智能客服 RAG】模块：\n"
            "- 知识库：PDF / Word 解析入库、向量检索\n"
            "- 对话：多轮记忆、检索重排、上下文压缩、引用溯源\n"
            "- 调优：多版本 Prompt 切换、检索调试接口"
        ),
        version="0.2.0",
        lifespan=lifespan,
    )

    # 跨域中间件：前端 5173 → 后端 8600，没有它浏览器会拦截请求
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册全局异常处理：所有异常统一变成 Result 信封
    register_exception_handlers(app)

    # RAG / 大模型相关的异常，单独处理。
    #
    # ★ 为什么要单独拎出来，而不是丢给 core/exception.py 的通用处理器？
    #
    #   1) 消息内容不同。这类异常的 message 本来就是写给用户看的中文，
    #      比如"还没有配置大模型 API Key，请在 .env 里加上 LLM_API_KEY=…"。
    #      直接透传，用户照着做就能解决。
    #      如果掉进兜底分支，只会看到"服务器开小差了，请稍后再试"——
    #      明明原因清清楚楚，却提示成"服务器坏了"，这是最糟糕的一种提示。
    #   2) 依赖方向。core/exception.py 属于最底层，不能反过来 import rag，
    #      否则分层就乱了。而 main.py 在最外层，core 和 rag 都可以依赖，
    #      放这里不会破坏任何一层的约束。
    from rag.llm import LLMError

    @app.exception_handler(LLMError)
    async def llm_error_handler(request: Request, exc: LLMError):
        """大模型调用相关失败：把"人话版"原因直接告诉用户。

        HTTP 状态码仍返回 200、用 code 表达错误，和项目其它接口保持一致，
        这样前端只需要看一套逻辑（先看 code，再看 message）。
        """
        logger.warning(
            "大模型错误: %s %s -> %s", request.method, request.url.path, exc
        )
        return JSONResponse(
            status_code=200,
            content={"code": 503, "message": str(exc), "data": None},
        )

    # 挂载路由：一个模块一个 router，统一加 /api 前缀
    app.include_router(hello_router, prefix="/api")
    app.include_router(user_router, prefix="/api")
    app.include_router(knowledge_router, prefix="/api")
    app.include_router(chat_router, prefix="/api")

    # 挂载静态目录（内置聊天页）
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def index():
        """根路径直接跳到聊天页，演示时少敲一截地址。"""
        if (STATIC_DIR / "chat.html").exists():
            return RedirectResponse(url="/static/chat.html")
        return RedirectResponse(url="/docs")

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.SERVER_HOST,
        port=settings.SERVER_PORT,
        reload=False,
    )
