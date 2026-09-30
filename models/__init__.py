"""models 包：所有数据模型（表结构的 Python 描述）。

重要：这里必须【真正 import】全部模型。

原因：Python 的 `import models` 只会执行 __init__.py 本身，
不会自动把 cart.py / user.py 这些子模块导进来。而 SQLAlchemy 是
"模型类被定义时才登记到 Base.metadata"的——不 import，它就不知道
有这些表，create_all 会一张表都不建。
"""

from models.cart import CartItem
from models.chat import ChatMessage, ChatSession
from models.knowledge import KnowledgeChunk, KnowledgeDoc
from models.order import Order, OrderItem
from models.product import Category, Product
from models.user import User

__all__ = [
    # 商城业务
    "User",
    "Category",
    "Product",
    "CartItem",
    "Order",
    "OrderItem",
    # 智能客服（RAG）
    "KnowledgeDoc",
    "KnowledgeChunk",
    "ChatSession",
    "ChatMessage",
]
