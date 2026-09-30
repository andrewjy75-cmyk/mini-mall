"""用户业务逻辑：注册、登录。

这一层只写"规则"，不碰 HTTP：
- 出问题就 raise BizException，具体怎么变成响应由 core/exception.py 统一管
- 不直接操作数据库表，而是通过 Session + model 操作
"""

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.exception import BizException
from core.security import create_access_token, hash_password, verify_password
from models.user import User
from schemas.user import UserLoginIn, UserRegisterIn
from util.id_util import new_id

logger = logging.getLogger(__name__)


class UserService:
    """用户相关的业务逻辑。"""

    def __init__(self, db: AsyncSession):
        # 依赖注入：Session 由外部传进来（api 层通过 get_db 拿到）
        self.db = db

    async def register(self, data: UserRegisterIn) -> User:
        """注册：查重 → 哈希密码 → 入库。"""
        # ① 账号查重
        existing = await self.db.scalar(
            select(User).where(User.account_name == data.account_name)
        )
        if existing:
            raise BizException("该账号已被注册", code=1001)

        # ② 建对象（注意：存进去的是哈希，不是明文）
        user = User(
            id=new_id(),
            account_name=data.account_name,
            password_hash=hash_password(data.password),
            nickname=data.nickname or data.account_name,   # 没填昵称就用账号名
            role="user",
        )

        # ③ 落库
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)     # 刷新一下，把数据库生成的 created_at 取回来
        logger.info("新用户注册成功: %s", user.account_name)
        return user

    async def login(self, data: UserLoginIn) -> User:
        """登录：按账号查人 → 验密码。"""
        user = await self.db.scalar(
            select(User).where(User.account_name == data.account_name)
        )

        # ⚠️ 安全细节：账号不存在 和 密码错误，都返回【同一句提示】。
        #    否则攻击者可以靠提示差异，批量试探出"哪些账号真实存在"。
        if user is None or not verify_password(data.password, user.password_hash):
            raise BizException("账号或密码错误", code=1002)

        logger.info("用户登录成功: %s", user.account_name)
        return user

    def issue_token(self, user: User) -> str:
        """给用户签发访问令牌。"""
        return create_access_token(user.id, user.role)
