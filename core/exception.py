"""业务异常 + 全局异常处理。

为什么要有自定义异常？

- service 层只管"发现业务问题就抛异常"（账号已存在、密码错误…），
  它不需要知道 HTTP 状态码、响应体长什么样。
- 统一在 main.py 注册处理器，由它决定最终怎么变成给前端的响应。

这样分层才干净：业务代码里看不到 HTTPException。
"""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class BizException(Exception):
    """业务异常：程序没出错，只是"规则不允许"。

    例：账号已被注册、库存不足、订单状态不允许取消。
    """

    def __init__(self, message: str, code: int = 1):
        self.message = message
        self.code = code
        super().__init__(message)


def register_exception_handlers(app: FastAPI) -> None:
    """把各类异常统一转换成 Result 信封。"""

    @app.exception_handler(BizException)
    async def biz_exception_handler(request: Request, exc: BizException):
        """业务异常 → code 用自定义值，HTTP 仍返回 200（前端看 code 判断）。"""
        logger.info("业务异常: %s", exc.message)
        return JSONResponse(
            status_code=200,
            content={"code": exc.code, "message": exc.message, "data": None},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """参数校验失败 → 把 Pydantic 的报错整理成人话。"""
        first = exc.errors()[0]
        # loc 形如 ('body', 'password')，取掉第一段（body/query）更像人话
        field = ".".join(str(x) for x in first["loc"][1:]) or "参数"
        message = f"参数有误：{field} —— {first['msg']}"
        logger.info("参数校验失败: %s", message)
        return JSONResponse(
            status_code=200,
            content={"code": 422, "message": message, "data": None},
        )

    @app.exception_handler(Exception)
    async def unknown_exception_handler(request: Request, exc: Exception):
        """兜底：任何没被处理的异常。

        ⚠️ 注意：返给用户的永远是"模糊提示"，真实错误只写日志。
           把堆栈信息暴露给前端 = 给攻击者送情报。
        """
        logger.exception("未预期的异常: %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=200,
            content={"code": 500, "message": "服务器开小差了，请稍后再试", "data": None},
        )
