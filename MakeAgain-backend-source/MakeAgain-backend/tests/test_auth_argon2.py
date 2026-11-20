"""Argon2id 密码哈希 + 旧 PBKDF2 兼容测试。"""
import hashlib

import pytest

import auth


def test_new_password_uses_argon2():
    h, salt = auth.hash_password("secure-password")
    assert h.startswith("$argon2")
    assert salt == ""  # PHC 自描述,无需额外 salt 列
    assert auth.verify_password("secure-password", salt, h) is True
    assert auth.verify_password("wrong-password", salt, h) is False


def test_argon2_hashes_are_salted():
    h1, _ = auth.hash_password("same-password")
    h2, _ = auth.hash_password("same-password")
    assert h1 != h2  # 随机 salt,相同密码哈希不同


def test_legacy_pbkdf2_still_verifies():
    """旧格式:PBKDF2 hex + 独立 salt_hex,必须仍能登录。"""
    password = "legacy-password"
    salt = bytes.fromhex("a1b2c3d4a1b2c3d4a1b2c3d4a1b2c3d4")
    old_hash = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000).hex()
    assert auth.verify_password(password, salt.hex(), old_hash) is True
    assert auth.verify_password("wrong", salt.hex(), old_hash) is False


def test_needs_rehash():
    h, salt = auth.hash_password("pw")
    assert auth.needs_rehash(h) is False
    # 旧格式需要重哈希
    old = hashlib.pbkdf2_hmac("sha256", b"pw", b"x" * 16, 200_000).hex()
    assert auth.needs_rehash(old) is True


def test_corrupt_hash_returns_false():
    assert auth.verify_password("pw", "", "$argon2$garbage") is False
    assert auth.verify_password("pw", "nothex", "nothex") is False
