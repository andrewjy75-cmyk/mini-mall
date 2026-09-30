"""第一个接口：确认服务真的跑起来了。

规范要点：
1. 一个模块一个 router，最后在 main.py 里挂上去
2. router 只负责"收参数、调服务、返回结果"，业务逻辑不写在这里
3. 每个接口都要有 tags，Swagger 文档里才会分类显示
"""

from fastapi import APIRouter

hello_router = APIRouter(tags=["练习"])


@hello_router.get("/hello", summary="打个招呼")
async def hello():
    """最简单的接口：返回一句问候，用来验证服务是否正常。"""
    return {"code": 0, "message": "商城后端跑起来了 🎉", "data": None}
