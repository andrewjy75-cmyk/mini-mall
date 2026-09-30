"""安全工具：密码哈希 + JWT 令牌。

两条铁律，先记牢：
1. 密码【永远】不存明文。而且要用"慢哈希"（bcrypt），
   让暴力尝试变慢——攻击者拿到库也没法快速试出密码。
2. JWT 靠密钥签名，服务器不用存 session。
   代价是：密钥一旦泄露，任何人都能伪造登录。所以密钥必须放 .env。
"""

from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from jwt import InvalidTokenError

from core.config import settings

# JWT 签名算法。HS256 = 对称算法（同一把密钥既签名又验证），够用且简单
ALGORITHM = "HS256"

# bcrypt 只处理前 72 字节，超长密码会被静默截断，这里主动限制长度
MAX_PASSWORD_BYTES = 72


def hash_password(plain_password: str) -> str:
    """把明文密码变成哈希串。

    bcrypt 会自带一段随机"盐"（salt）混进去，所以：
    - 同一个密码，两次哈希得到的字符串【不一样】
    - 攻击者拿现成的"彩虹表"查不出来
    """
    raw = plain_password.encode("utf-8")[:MAX_PASSWORD_BYTES]
    return bcrypt.hashpw(raw, bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, password_hash: str) -> bool:
    """校验密码是否正确。任何时候都不要自己写比较逻辑，交给 bcrypt。"""
    raw = plain_password.encode("utf-8")[:MAX_PASSWORD_BYTES]
    try:
        return bcrypt.checkpw(raw, password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        # 哈希串格式损坏（比如被手动改过），一律当校验失败
        return False


def create_access_token(user_id: str, role: str = "user") -> str:
    """签发一张通行证（JWT）。

    JWT 是"三段式"字符串，用点分隔：

        header . payload . signature
        ↑        ↑          ↑
      算法信息  数据内容    签名

    ⚠️ 前两段只是 Base64 编码——任何人拿到都能解开看内容！
       所以【绝对不要】把密码之类的敏感信息放进 payload。
       签名段的作用是：证明"内容确实是我签发的，且没被篡改"。
    """
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,   # subject：这张令牌属于谁
        "role": role,     # 顺带把角色带上，省一次查库
        "iat": now,       # issued at：签发时间
        "exp": now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, settings.TOKEN_SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict | None:
    """验证并解析令牌。过期 / 被篡改 / 格式错，一律返回 None。

    jwt.decode 会自动帮我们校验 exp（过期时间）和签名。
    """
    try:
        return jwt.decode(token, settings.TOKEN_SECRET_KEY, algorithms=[ALGORITHM])
    except InvalidTokenError:
        return None
