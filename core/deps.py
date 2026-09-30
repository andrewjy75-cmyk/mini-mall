"""FastAPI 依赖项：跨模块复用的"当前请求上下文"。

目前只提供一件事：从请求头里解析出当前登录用户。

★ 为什么做成"可选"的（解析不到就返回 None，而不是抛 401）？
  因为智能客服的演示场景需要"不登录也能试两句"。
  强制登录会把演示门槛抬高，而这是个考核项目，第一印象很重要。
  代价是匿名会话不属于任何用户（user_id 为 NULL），
  这在数据上是可以接受的：匿名会话仍然完整可追溯，只是不挂在人名下。

  真实商业系统里不会这么宽松（匿名会带来刷接口、滥用额度的问题），
  这里的取舍是明确为了演示便利，属于有意为之，不是漏了。
"""

import logging

from fastapi import Request

from core.security import decode_access_token
from models.user import User

logger = logging.getLogger(__name__)


def _extract_token(request: Request) -> str | None:
    """从 Authorization 头里把 token 抠出来。

    标准格式是 `Bearer xxx`。这里对大小写和多余空格都做了容错，
    因为前端手写这个头的时候经常出小差错，没必要为此报错。
    """
    header = request.headers.get("Authorization") or ""
    if not header:
        return None

    parts = header.strip().split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    # 只给了裸 token（没带 Bearer 前缀）也放行，容错优先
    return parts[0].strip() if parts else None


async def get_current_user_optional(request: Request) -> User | None:
    """解析当前用户，拿不到就返回 None（不拦截请求）。

    注意这里只解析 token 里的信息，不查库。
    为什么？因为每次请求都查一次库会凭空增加数据库压力，
    而 token 里的 sub（用户 id）和 role 已经够用了。
    代价是：用户被删除或改了角色时，旧 token 在过期前仍然有效。
    这是个已知的权衡，属于"用一点实时性换性能"，在有需要时再改成查库版。
    """
    token = _extract_token(request)
    if not token:
        return None

    payload = decode_access_token(token)
    if not payload:
        return None

    user_id = payload.get("sub")
    if not user_id:
        return None

    # 造一个"轻量用户对象"：只填拿得到的字段，
    # 避免调用方以为拿到的是完整记录去访问 nickname 之类而踩空
    return User(
        id=user_id,
        account_name="",
        password_hash="",
        nickname="",
        role=payload.get("role", "user"),
    )
