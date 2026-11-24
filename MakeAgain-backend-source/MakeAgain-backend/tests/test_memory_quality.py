"""记忆质量模块离线测试（包 F：Fact 校验 + 中文词形归一 + checkpoint）。

口径：全部为 synthetic-handwritten 构造数据 + 本地 SQLite，零 LLM / 零真 key / 零网络。
被测模块未接主链路，本文件只证明「模块自身行为」，不构成产品效果验证（未接线=未验证）。
"""
from __future__ import annotations

import json

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from memory.checkpoint import (
    CheckpointTableMissing,
    ensure_checkpoint_table,
    load_checkpoint,
    save_checkpoint,
    delete_checkpoint,
)
from memory.fact_check import check_facts, group_by_kind
from memory.tokenize_zh import normalize, tokenize, signature, overlap

MSG_DDL = """
CREATE TABLE chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT, session_id TEXT DEFAULT '', role TEXT,
    content TEXT, ts TEXT
)
"""
ENTRY_DDL = """
CREATE TABLE memory_stream (
    id TEXT PRIMARY KEY, user_id TEXT, companion_id TEXT,
    facts TEXT, content TEXT, summary TEXT, ts TEXT
)
"""


def _new_db(*, tables: bool = True):
    engine = create_engine("sqlite://")
    if tables:
        with engine.begin() as conn:
            conn.execute(text(MSG_DDL))
            conn.execute(text(ENTRY_DDL))
    return engine, sessionmaker(bind=engine)


def _msg(engine, uid: str, mid: int, content: str, ts: str, role: str = "user") -> None:
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO chat_messages (id,user_id,role,content,ts) VALUES (:m,:u,:r,:c,:t)"),
            {"m": mid, "u": uid, "r": role, "c": content, "t": ts},
        )


def _entry(engine, uid: str, eid: str, *, facts=None, content: str = "", ts: str = "") -> None:
    payload = json.dumps(facts, ensure_ascii=False) if facts is not None else ""
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO memory_stream (id,user_id,facts,content,ts) VALUES (:i,:u,:f,:c,:t)"),
            {"i": eid, "u": uid, "f": payload, "c": content, "t": ts},
        )


# ---------------------------------------------------------------- tokenize_zh

def test_normalize_fullwidth_traditional_case_zero_width():
    raw = "\u200bＨＩ，這是離別療愈　ＡＢＣ"
    assert normalize(raw) == "hi,这是离别疗愈 abc"
    assert normalize(None) == ""
    assert normalize("  a  b\n") == "a b"


@pytest.mark.parametrize("use_jieba", [True, False])
def test_tokenize_drops_punctuation_and_stopwords(use_jieba):
    tokens = tokenize("我今天真的很难过，但是会慢慢好起来。", use_jieba=use_jieba)
    assert tokens
    assert not any(any(ch in t for ch in "，。！？、") for t in tokens)
    assert "我" not in tokens and "真的" not in tokens
    assert any("难过" in t for t in tokens)


@pytest.mark.parametrize("use_jieba", [True, False])
def test_tokenize_ascii_words_and_numbers(use_jieba):
    tokens = tokenize("BGM Volume 50%", use_jieba=use_jieba)
    # jieba 会把 "50%" 保留成一个 token，兜底切分则是 "50"；两种口径都接受。
    assert {"bgm", "volume"}.issubset(set(tokens))
    assert any("50" in t for t in tokens)


def test_tokenize_empty_and_jieba_missing_fallback():
    assert tokenize("") == []
    assert tokenize("这是测试", use_jieba=False)  # 强制兜底也必须有 token


def test_signature_is_normalization_invariant_and_reproducible():
    assert signature("難過") == signature("难过") == frozenset({"难过"})
    assert signature("ＡＢＣ") == signature("abc") == frozenset({"abc"})
    assert signature("的了吗") == frozenset()  # 纯虚词不进签名
    assert "橘子" in overlap("用户的猫叫橘子", "它叫橘子")


# ---------------------------------------------------------------- fact_check

def test_check_facts_detects_correction_conflict():
    engine, Session = _new_db()
    _entry(engine, "u1", "e1", facts=[{"kind": "object", "fact": "用户的猫叫橘子", "confidence": 0.9}],
           ts="2026-11-01T00:00:00")
    _msg(engine, "u1", 1, "它叫橘子，陪了我八年", "2026-11-01T01:00:00")
    _msg(engine, "u1", 2, "它不叫橘子了，我改叫它团子", "2026-11-02T01:00:00")
    with Session() as db:
        issues = check_facts(db, "u1")
    assert issues, "应报出一条被后续对话推翻的 Fact"
    hit = [i for i in issues if i.kind == "conflict" and i.rule == "correction"]
    assert hit
    assert hit[0].evidence_message_id == 2
    assert "橘子" in hit[0].shared_signatures
    assert hit[0].fact_text == "用户的猫叫橘子"
    assert group_by_kind(issues)["conflict"]


def test_check_facts_detects_change_cue_stale():
    engine, Session = _new_db()
    _entry(engine, "u1", "e1", facts=[{"fact": "最近总是失眠"}], ts="2026-11-01T00:00:00")
    _msg(engine, "u1", 1, "我现在睡眠好多了，不再失眠", "2026-11-05T00:00:00")
    with Session() as db:
        issues = check_facts(db, "u1")
    stale = [i for i in issues if i.kind == "stale" and i.rule == "change_cue"]
    assert stale
    assert "失眠" in stale[0].shared_signatures


def test_check_facts_detects_temporal_stale():
    engine, Session = _new_db()
    _entry(engine, "u1", "e1", facts=[{"fact": "目前仍在吃药"}], ts="2026-09-01T00:00:00")
    _msg(engine, "u1", 1, "今天去复查了，医生说还可以", "2026-11-25T00:00:00")
    with Session() as db:
        issues = check_facts(db, "u1")
    assert any(i.kind == "stale" and i.rule == "temporal" for i in issues)


def test_check_facts_no_false_positive_on_unrelated_chat():
    engine, Session = _new_db()
    _entry(engine, "u1", "e1", facts=[{"fact": "用户喜欢喝咖啡"}], ts="2026-11-01T00:00:00")
    _msg(engine, "u1", 1, "今天天气不错", "2026-11-02T00:00:00")
    _msg(engine, "u1", 2, "我们聊聊吧", "2026-11-03T00:00:00")
    with Session() as db:
        assert check_facts(db, "u1") == []
        assert check_facts(db, "") == []


def test_check_facts_isolated_per_user_and_read_only():
    engine, Session = _new_db()
    _entry(engine, "u1", "e1", facts=[{"fact": "用户的猫叫橘子"}], ts="2026-11-01T00:00:00")
    _msg(engine, "u1", 1, "今天天气不错", "2026-11-02T00:00:00")
    _entry(engine, "u2", "e2", facts=[{"fact": "用户的狗叫旺财"}], ts="2026-11-01T00:00:00")
    _msg(engine, "u2", 2, "它不叫旺财了，改叫小黑", "2026-11-02T00:00:00")

    def snapshot():
        with engine.connect() as conn:
            return (
                conn.execute(text("SELECT * FROM chat_messages ORDER BY id")).all(),
                conn.execute(text("SELECT * FROM memory_stream ORDER BY id")).all(),
            )

    before = snapshot()
    with Session() as db:
        u1 = check_facts(db, "u1")
        u2 = check_facts(db, "u2")
        assert len(db.new) == 0 and len(db.dirty) == 0  # IdentitySet 用 len 判空
    assert u1 == []
    assert any(i.kind == "conflict" for i in u2)
    assert snapshot() == before, "check_facts 必须是只读的"


def test_check_facts_no_messages_or_content_fallback():
    engine, Session = _new_db()
    _entry(engine, "u1", "e1", facts=[{"fact": "用户的猫叫橘子"}], ts="2026-11-01T00:00:00")
    with Session() as db:
        assert check_facts(db, "u1") == []  # 没有 chat_messages 无从校验
    _entry(engine, "u1", "e2", content="用户的猫叫橘子", ts="2026-11-01T00:00:00")
    _msg(engine, "u1", 1, "它不叫橘子了", "2026-11-02T00:00:00")
    with Session() as db:
        issues = check_facts(db, "u1")
    assert any(i.source == "content" and i.kind == "conflict" for i in issues)


# ---------------------------------------------------------------- checkpoint

def test_checkpoint_roundtrip_and_stable_id():
    engine, Session = _new_db()
    with Session() as db:
        ensure_checkpoint_table(db)
        payload = {"step": 3, "anchors": ["林夏", "橘子"], "nested": {"done": True}}
        cid = save_checkpoint(db, "u1", "s1", payload, summary="第3轮已埋事实")
        db.commit()
        loaded = load_checkpoint(db, "u1", "s1")
    assert loaded is not None
    assert loaded["payload"] == payload
    assert loaded["summary"] == "第3轮已埋事实"
    assert loaded["id"] == cid
    assert loaded["created_at"]


def test_checkpoint_upsert_is_one_row_per_user_session():
    engine, Session = _new_db()
    with Session() as db:
        ensure_checkpoint_table(db)
        save_checkpoint(db, "u1", "s1", {"step": 1})
        db.commit()
        save_checkpoint(db, "u1", "s1", {"step": 9}, summary="更新")
        db.commit()
        loaded = load_checkpoint(db, "u1", "s1")
        rows = db.execute(text("SELECT COUNT(*) FROM session_checkpoints")).scalar()
    assert loaded["payload"] == {"step": 9}
    assert loaded["summary"] == "更新"
    assert rows == 1


def test_checkpoint_user_and_session_isolation():
    engine, Session = _new_db()
    with Session() as db:
        ensure_checkpoint_table(db)
        save_checkpoint(db, "u1", "s1", {"who": "u1"})
        db.commit()
        assert load_checkpoint(db, "u2", "s1") is None
        assert load_checkpoint(db, "u1", "s2") is None


def test_checkpoint_missing_table_has_actionable_error():
    engine, Session = _new_db(tables=False)
    with Session() as db:
        # SQLAlchemy 2.0：每次 DBAPI 异常后事务作废，必须先 rollback 再发下一条语句。
        for call in (
            lambda: load_checkpoint(db, "u1", "s1"),
            lambda: save_checkpoint(db, "u1", "s1", {"step": 1}),
            lambda: delete_checkpoint(db, "u1", "s1"),
        ):
            with pytest.raises(CheckpointTableMissing):
                call()
            db.rollback()


def test_checkpoint_does_not_commit_and_delete_works():
    engine, Session = _new_db()
    with Session() as db:
        ensure_checkpoint_table(db)
        db.commit()
        save_checkpoint(db, "u1", "s1", {"step": 1})
        db.rollback()  # 模块未自行 commit -> 回滚后应查不到
        assert load_checkpoint(db, "u1", "s1") is None
        save_checkpoint(db, "u1", "s1", {"step": 2})
        db.commit()
        assert delete_checkpoint(db, "u1", "s1") is True
        db.commit()
        assert delete_checkpoint(db, "u1", "s1") is False
        assert load_checkpoint(db, "u1", "s1") is None
