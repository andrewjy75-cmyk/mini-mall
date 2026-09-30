"""通用响应格式。

全项目所有接口都返回同一个"信封"，前端处理起来才统一：

    {"code": 0, "message": "成功", "data": {...真正的业务数据...}}

好处：前端只要写一次判断逻辑——
    if (res.code === 0) { 用 res.data } else { 弹 res.message }
"""

from typing import Generic, TypeVar

from pydantic import BaseModel, Field

# T 是个"占位类型"，用到时才确定。
# 写 Result[User] 就表示"data 里装的是 User"。
T = TypeVar("T")


class Result(BaseModel, Generic[T]):
    """统一响应信封。"""

    code: int = Field(default=0, description="0=成功，非 0=失败")
    message: str = Field(default="成功", description="给用户看的提示文字")
    data: T | None = Field(default=None, description="真正的业务数据")

    @classmethod
    def success(cls, data: T | None = None, message: str = "成功") -> "Result[T]":
        """成功响应。"""
        return cls(code=0, message=message, data=data)

    @classmethod
    def error(cls, message: str, code: int = 1) -> "Result[T]":
        """失败响应。"""
        return cls(code=code, message=message, data=None)


class PageResponse(BaseModel, Generic[T]):
    """分页响应。商品列表、订单列表都用它。"""

    items: list[T] = Field(default_factory=list, description="当前页数据")
    total: int = Field(default=0, description="总条数")
    page: int = Field(default=1, description="第几页")
    page_size: int = Field(default=10, description="每页多少条")

    @property
    def total_pages(self) -> int:
        """总页数（向上取整）。比如 21 条、每页 10 条 → 3 页。"""
        if self.page_size <= 0:
            return 0
        return (self.total + self.page_size - 1) // self.page_size
