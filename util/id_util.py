"""ID 生成工具。"""

import uuid
from datetime import datetime


def new_id() -> str:
    """生成一个 36 位的 UUID 字符串，用作主键。

    为什么不用数据库自增 ID？
    1. 不暴露业务量（自增 ID 能被对手算出你一天卖了多少单）
    2. 多台机器同时生成也不会撞号
    """
    return str(uuid.uuid4())


def new_order_no() -> str:
    """生成订单号：年月日时分秒 + 6 位随机大写字母数字。

    时间开头方便肉眼排序；加随机后缀避免同一秒内重复。
    例：20250120153042A3F9C1
    """
    return datetime.now().strftime("%Y%m%d%H%M%S") + uuid.uuid4().hex[:6].upper()
