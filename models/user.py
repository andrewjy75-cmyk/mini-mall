"""用户表。"""

from datetime import datetime

from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column

from core.database import Base


class User(Base):
    """用户。"""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    # 登录账号，唯一 + 索引（登录时按它查，必须快）
    account_name: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    # 只存哈希，绝不存明文密码
    password_hash: Mapped[str] = mapped_column(String(255))
    nickname: Mapped[str] = mapped_column(String(50), default="")
    # user=普通用户, admin=管理员
    role: Mapped[str] = mapped_column(String(16), default="user")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    def __repr__(self) -> str:
        return f"<User {self.account_name}>"
