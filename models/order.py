"""订单主表 + 订单项表。"""

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from core.database import Base


class Order(Base):
    """订单主表（一张订单的"头"）。"""

    __tablename__ = "orders"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    # 给用户看的订单号，唯一 + 索引
    order_no: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    # 订单总额，单位：分
    total_amount: Mapped[int] = mapped_column(Integer, default=0)
    # pending=待付款  paid=已付款  cancelled=已取消
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    paid_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    def __repr__(self) -> str:
        return f"<Order {self.order_no} {self.status}>"


class OrderItem(Base):
    """订单项（订单的"明细行"）。"""

    __tablename__ = "order_items"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    order_id: Mapped[str] = mapped_column(String(36), ForeignKey("orders.id"), index=True)
    product_id: Mapped[str] = mapped_column(String(36))

    # ★★ 下面两个字段是【快照】：
    #    下单那一刻把商品名和价格抄一份存下来。
    #    以后商品改价、下架、甚至被删除，历史订单显示的仍是"当时买的是什么价"。
    product_name: Mapped[str] = mapped_column(String(100))
    price: Mapped[int] = mapped_column(Integer)

    quantity: Mapped[int] = mapped_column(Integer)
    subtotal: Mapped[int] = mapped_column(Integer)  # 小计 = price * quantity
