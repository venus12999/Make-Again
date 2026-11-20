"""鉴权:密码哈希(Argon2id 优先,兼容旧 PBKDF2)+ 会话 token + 当前用户依赖。

- 新密码用 argon2-cffi 的 Argon2id(OWASP 首选,内存硬,抗 GPU/ASIC),
  PHC 字符串自描述算法与参数,便于未来调参。
- verify_password 识别 $argon2 前缀走 argon2 校验;旧的 PBKDF2 hex 走兼容分支,
  保证存量账号(若有)仍可登录。
- 会话 token 仍为 256-bit 随机 opaque token(存库,可即时吊销)。
"""
from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta

from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session

from memory import store
from memory.db import get_session
from memory.models import utcnow

_PBKDF2_ITERATIONS = 200_000  # 旧格式兼容用,不再用于新密码
_SESSION_DAYS = 30

try:
    from argon2 import PasswordHasher
    _ph = PasswordHasher(time_cost=2, memory_cost=19456, parallelism=1)  # OWASP 最小配置:19MiB / 2 轮 / 1 并发
except Exception:  # argon2 不可用时仍可用 PBKDF2 链路
    _ph = None


def hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    """返回 (hash, salt_hex)。新密码用 Argon2id(PHC 字符串),salt 参数仅为兼容旧签名。

    注意:Argon2id 的 salt 与参数已内嵌在 PHC 字符串里,故 salt_hex 返回空串占位
    (DB 的 password_salt 列非空即可,无需存储额外 salt)。
    """
    if _ph is not None:
        return _ph.hash(password), ""
    # fallback:argon2 不可用时退回 PBKDF2(保持旧格式)
    salt = salt if salt is not None else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return digest.hex(), salt.hex()


def verify_password(password: str, salt_hex: str, hash_hex: str) -> bool:
    if not salt_hex and hash_hex.startswith("$argon2"):
        if _ph is None:
            return False
        try:
            return bool(_ph.verify(hash_hex, password))
        except Exception:  # 校验失败/哈希损坏/异常参数一律视为密码错误
            return False
    # 旧 PBKDF2 格式兼容:按存储的 salt 重算
    try:
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                                     bytes.fromhex(salt_hex), _PBKDF2_ITERATIONS)
    except (ValueError, TypeError):
        return False
    return secrets.compare_digest(digest.hex(), hash_hex)


def needs_rehash(hash_hex: str) -> bool:
    """是否需要重新哈希(参数过时或仍是旧格式);登录时可据此透明升级。"""
    if not hash_hex.startswith("$argon2"):
        return True
    if _ph is None:
        return False
    try:
        return bool(_ph.check_needs_rehash(hash_hex))
    except Exception:
        return False


def new_token() -> str:
    return secrets.token_hex(32)


def new_expiry() -> datetime:
    return utcnow() + timedelta(days=_SESSION_DAYS)


def get_current_user(authorization: str | None = Header(None), db: Session = Depends(get_session)) -> str:
    """解析 `Authorization: Bearer <token>` → 当前 user_id;未登录/过期抛 401。"""
    from api.security import authenticate
    return authenticate(db, authorization).id
