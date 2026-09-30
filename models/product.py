"""商品分类表 + 商品表。"""

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from core.database import Base


class Category(Base):
    """商品分类，例如"手机数码""图书"。"""

    __tablename__ = "categories"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(50), unique=True)
    sort: Mapped[int] = mapped_column(Integer, default=0)  # 越小越靠前
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Product(Base):
    """商品。"""

    __tablename__ = "products"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(100), index=True)
    description: Mapped[str] = mapped_column(Text, default="")

    # ★ 价格单位是【分】，必须用整数。
    #   浮点数存钱会出错：0.1 + 0.2 = 0.30000000000000004
    price: Mapped[int] = mapped_column(Integer, default=0)

    stock: Mapped[int] = mapped_column(Integer, default=0)
    category_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("categories.id"), index=True
    )
    image_url: Mapped[str] = mapped_column(String(500), default="")
    # 1=上架（能买）  0=下架（不显示）
    status: Mapped[int] = mapped_column(Integer, default=1, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    def __repr__(self) -> str:
        return f"<Product {self.name} ￥{self.price / 100:.2f}>"
