"""建表脚本：把 models 里定义的表真正建到数据库里。

用法：
    .venv\\Scripts\\python.exe init_db.py

说明：
- 这是"首次建表"用的工具，重复运行是安全的（已存在的表会跳过）
- 以后要改表结构（加字段、改类型），应该引入 Alembic 做版本化迁移，
  而不是依赖这个脚本
"""

import asyncio

import models  # noqa: F401  ← 必须导入！不导入 SQLAlchemy 就看不到模型
from core.database import Base, engine


async def main() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    print("建表完成，数据库里现在有这些表：")
    for name in sorted(Base.metadata.tables):
        print(f"  - {name}")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
