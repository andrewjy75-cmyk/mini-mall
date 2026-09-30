"""购物车表。"""

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from core.database import Base


class CartItem(Base):
    """购物车条目：某人的某个商品，打算买几件。"""

    __tablename__ = "cart_items"

    # ★ 联合唯一约束：同一个人 + 同一个商品，只能有一条记录。
    #   没有它，用户狂点"加入购物车"就会造出好几条重复记录。
    __table_args__ = (
        UniqueConstraint("user_id", "product_id", name="uq_cart_user_product"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    product_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("products.id"), index=True
    )
    quantity: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
