"""用户接口：注册、登录。

这一层只做三件事：
1. 收参数（靠 schemas 校验）
2. 调 service 干活（靠 Depends 拿 service）
3. 拼响应（统一用 Result 信封）
业务规则一行都不写在这里。
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from schemas.general import Result
from schemas.user import TokenOut, UserLoginIn, UserOut, UserRegisterIn
from services.user_service import UserService

user_router = APIRouter(prefix="/user", tags=["用户"])


def get_user_service(db: AsyncSession = Depends(get_db)) -> UserService:
    """把 service 也做成一个依赖。

    好处：接口函数里不用自己 new，而且以后要给 UserService 加别的依赖，
    只改这一个地方就够了。
    """
    return UserService(db)


@user_router.post("/register", response_model=Result[TokenOut], summary="注册")
async def register(
    data: UserRegisterIn,
    service: UserService = Depends(get_user_service),
):
    """注册新用户，成功后直接返回令牌（注册即登录）。"""
    user = await service.register(data)
    token = service.issue_token(user)
    return Result.success(
        data=TokenOut(token=token, user=UserOut.model_validate(user)),
        message="注册成功",
    )


@user_router.post("/login", response_model=Result[TokenOut], summary="登录")
async def login(
    data: UserLoginIn,
    service: UserService = Depends(get_user_service),
):
    """账号密码登录，返回令牌。"""
    user = await service.login(data)
    token = service.issue_token(user)
    return Result.success(
        data=TokenOut(token=token, user=UserOut.model_validate(user)),
        message="登录成功",
    )
