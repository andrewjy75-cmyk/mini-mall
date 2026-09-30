"""用户相关的接口契约（相当于 Java 的 DTO / VO）。

分三类，看名字就懂：
- XxxIn   ：前端传进来的（入参）
- XxxOut  ：返回给前端的（出参）
- 两者字段可以完全不同 —— 比如密码只进不出
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

# 账号规则：字母、数字、下划线，3~50 位
ACCOUNT_PATTERN = r"^[A-Za-z0-9_]{3,50}$"


class UserRegisterIn(BaseModel):
    """注册入参。"""

    account_name: str = Field(
        ...,
        pattern=ACCOUNT_PATTERN,
        description="登录账号（3~50 位字母/数字/下划线）",
        examples=["jerry"],
    )
    password: str = Field(
        ..., min_length=6, max_length=72, description="密码（6~72 位）"
    )
    nickname: str = Field(default="", max_length=50, description="昵称，可不填")


class UserLoginIn(BaseModel):
    """登录入参。"""

    account_name: str = Field(..., description="登录账号")
    password: str = Field(..., description="密码")


class UserOut(BaseModel):
    """用户信息出参。

    ★ 注意：这里【没有】password_hash。
      接口返回给前端的数据里绝不能出现密码哈希 —— 这是底线。
    """

    # from_attributes=True：允许直接从 ORM 对象转换
    # 也就是 UserOut.model_validate(user_orm_obj) 能直接用
    model_config = ConfigDict(from_attributes=True)

    id: str
    account_name: str
    nickname: str
    role: str
    created_at: datetime


class TokenOut(BaseModel):
    """登录 / 注册成功后的返回。"""

    token: str = Field(..., description="访问令牌，后续请求放进 Authorization 头")
    token_type: str = Field(default="Bearer", description="令牌类型，固定 Bearer")
    user: UserOut
