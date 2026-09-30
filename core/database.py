"""数据库基础设施：连接引擎、会话工厂、模型基类。

三个概念先记住：
- engine（引擎）    ：管"连接池"的，整个应用只需要一个
- Session（会话）   ：一次数据库对话，每个请求一个
- Base（基类）      ：所有表的父类，继承它 SQLAlchemy 才认这是张表
"""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from core.config import settings


class Base(DeclarativeBase):
    """所有数据模型的基类。"""


engine = create_async_engine( #管“连接池”的
    settings.database_url,
    echo=settings.DB_LOG,      # True 会把执行的 SQL 打印出来，调试时非常有用
    pool_pre_ping=True,        # 取连接前先探活，避免 MySQL 断开后一直报错
    pool_recycle=3600,         # 连接最多活 1 小时就回收重建
)


AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,    # commit 之后对象不过期，还能继续读它的属性
)


async def get_db():
    """FastAPI 依赖：每个请求发一个会话，请求结束自动关闭。

    用法：在接口函数参数里写 db: AsyncSession = Depends(get_db)
    """
    async with AsyncSessionLocal() as session:
        yield session
