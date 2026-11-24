"""U-004 · 危机事件持久化 + 路由验收（纯离线，MOCK_LLM 由 conftest 提供，零真实 key）。

口径：本测试只验证存储层与鉴权路由的**写边界**（表/列存在、读写回环、用户隔离、
未鉴权拒绝），不验证真后端 8000 端口、不跑危机真实模型评测、不改 crisis_flow 内存逻辑。
"""
import pytest
from sqlalchemy import inspect

from conftest import account
from memory.db import Base, SessionLocal, engine
from memory import crisis_store
from memory.models import CrisisEvent
from emotion import crisis as crisis_gate

T0 = 1_700_000_000.0  # 固定 epoch，确定性控制 ts


@pytest.fixture
def db():
    with SessionLocal() as session:
        yield session


def _event(user_id, *, eid="ev-1", severity="level3", keywords=None, text="我想死",
           companion_id=None, level=3, state="pending_confirm", ts=T0):
    """模拟 crisis_flow.handle_crisis 返回的 event dict（synthetic，非真人语料）。"""
    return {
        "id": eid,
        "user_id": user_id,
        "companion_id": companion_id,
        "session_id": "sess-1",
        "ts": ts,
        "text": text,
        "level": level,
        "severity": severity,
        "hit": keywords if keywords is not None else ["想死"],
        "state": state,
        "review_required": level >= 3,
    }


# ---- 1. 表与列：create_all 建表 / migrations_b07 遍历全部模型列 ---------------

def test_crisis_event_table_and_columns_registered():
    insp = inspect(engine)
    assert "crisis_events" in insp.get_table_names()
    db_cols = {c["name"] for c in insp.get_columns("crisis_events")}
    model_cols = {c.name for c in CrisisEvent.__table__.columns}
    assert model_cols <= db_cols
    # migrations_b07.migrate 遍历 Base.metadata.sorted_tables 的每个 table.columns，
    # 这张表进了 metadata 就等于自动纳入补列/索引机制，不需要单独迁移脚本。
    assert CrisisEvent.__table__ in Base.metadata.sorted_tables


# ---- 2. store 读写回环 + 用户隔离 --------------------------------------------

def test_save_and_list_roundtrip_with_user_isolation(db):
    crisis_store.save_crisis_event(_event("u-A", eid="ev-A", companion_id="c-A"))
    crisis_store.save_crisis_event(_event("u-B", eid="ev-B", keywords=["解脱"],
                                          severity="level2", level=2, text="想解脱"))

    rows_a = crisis_store.list_crisis_events(db, "u-A")
    rows_b = crisis_store.list_crisis_events(db, "u-B")
    assert [r["id"] for r in rows_a] == ["ev-A"]
    assert [r["id"] for r in rows_b] == ["ev-B"]

    a = rows_a[0]
    assert a["user_id"] == "u-A" and a["companion_id"] == "c-A"
    assert a["severity"] == "level3" and a["level"] == 3
    assert a["keywords"] == ["想死"]          # hit → keywords 映射
    assert a["state"] == "pending_confirm"
    assert a["review_required"] is True
    assert a["session_id"] == "sess-1"
    assert a["ts"].startswith("2023-11-14")   # epoch → naive UTC ISO（row._mapping 读列）
    assert crisis_store.count_crisis_events(db, "u-A") == 1
    assert crisis_store.count_crisis_events(db, "u-B") == 1


def test_order_newest_first_and_limit(db):
    for i, ts in enumerate((T0, T0 + 10, T0 + 20)):
        crisis_store.save_crisis_event(_event("u-order", eid=f"ev-{i}", ts=ts))
    rows = crisis_store.list_crisis_events(db, "u-order", limit=2)
    assert [r["id"] for r in rows] == ["ev-2", "ev-1"]
    assert len(crisis_store.list_crisis_events(db, "u-order", limit=99)) == 3


# ---- 3. 路由：带鉴权、只回本人数据 -------------------------------------------

def test_api_lists_only_authenticated_user_events(client):
    u1, h1, _ = account(client, "crisis-a")
    u2, h2, _ = account(client, "crisis-b")
    crisis_store.save_crisis_event(_event(u1, eid="ev-u1", text="只想看看 U1"))
    crisis_store.save_crisis_event(_event(u2, eid="ev-u2", text="不该出现在 U1 的响应里"))

    r1 = client.get("/api/crisis/events", headers=h1)
    assert r1.status_code == 200, r1.text
    body = r1.json()
    assert body["ok"] is True and body["count"] == 1
    assert [e["id"] for e in body["events"]] == ["ev-u1"]
    assert "不该出现" not in r1.text

    r2 = client.get("/api/crisis/events", headers=h2)
    assert r2.status_code == 200
    assert [e["id"] for e in r2.json()["events"]] == ["ev-u2"]


def test_api_requires_auth(client):
    r = client.get("/api/crisis/events")
    assert r.status_code == 401


# ---- 4. 硬门槛零回归：本包不碰 crisis.py -------------------------------------

def test_hard_gate_detection_unchanged():
    assert crisis_gate.detect("我想死") == {"level": 3, "hit": ["想死"], "is_crisis": True}
    assert crisis_gate.detect("我不想死")["level"] == 0
