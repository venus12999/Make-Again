"""UC-4 跨 Journey 隔离行为验收（U4-V1 ~ U4-V4 + 越权守卫）。

全绿不是标准：每条 V 打印真实命令可复现的原始证据（pytest -s），并断言拒绝行为。
被测对象只有 memory/isolation.py；不修改任何业务主链路 / 队友文件。
"""
from __future__ import annotations

import json

import pytest

from memory import isolation as iso
from memory import recall as recall_mod
from memory import store as mstore
from memory import vector_index as vi_mod
from memory.db import SessionLocal
from memory.models import Portrait, utcnow


def _seed_user(db, uid: str, *, consent: bool = True):
    mstore.get_or_create_user(db, uid)
    state = mstore.get_or_create_user_state(db, uid)
    if consent:
        state.consent_accepted_at = utcnow()
        db.commit()
    return state


def _portrait(db, uid: str, cid: str, name: str) -> Portrait:
    row = Portrait(user_id=uid, companion_id=cid, kind="object",
                   content={"称呼": name}, status="confirmed", version=1)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def _ids(hits) -> list[str]:
    return [str(h["entry"].id) for h in hits]


# ---------------------------------------------------------------------------
# U4-V1 · 用户 A 查不到用户 B 记忆（含双向越权守卫）
# ---------------------------------------------------------------------------
def test_u4_v1_user_a_cannot_see_user_b():
    db = SessionLocal()
    try:
        _seed_user(db, "u-A")
        _seed_user(db, "u-B")
        ca = mstore.create_companion(db, "u-A", relationship_type="pet")
        cb = mstore.create_companion(db, "u-B", relationship_type="pet")
        ma = iso.scoped_add_memory(db, "u-A", ca.id, content="A 的猫叫豆豆", importance=9.0)
        mb = iso.scoped_add_memory(db, "u-B", cb.id, content="B 的隐私：对桂花糕过敏", importance=9.0)

        scope_a = iso.build_scope(db, "u-A", ca.id)
        hits = iso.scoped_recall(db, "u-A", ca.id, "桂花糕 过敏", top_k=10)

        print("\n[U4-V1] A scope:", json.dumps(scope_a.as_dict(), ensure_ascii=False))
        print("[U4-V1] B memory id:", mb.id, "in A allowed_ids:", str(mb.id) in scope_a.allowed_ids)
        print("[U4-V1] A recall ids:", _ids(hits), "A memory:", ma.id)

        assert str(ma.id) in scope_a.allowed_ids
        assert str(mb.id) not in scope_a.allowed_ids
        assert str(mb.id) not in _ids(hits)
        assert any(str(ma.id) == mid for mid in _ids(hits))

        # 越权必含：A 读 B 的记忆、B 读 A 的记忆，双向拒绝
        with pytest.raises(iso.ScopeDenied) as e1:
            iso.assert_scope(db, "u-A", ca.id, mb.id)
        with pytest.raises(iso.ScopeDenied) as e2:
            iso.assert_scope(db, "u-B", cb.id, ma.id)
        print("[U4-V1] A->B denied:", e1.value.code, "| B->A denied:", e2.value.code)
        assert e1.value.code == "SCOPE_DENIED" and e2.value.code == "SCOPE_DENIED"
    finally:
        db.close()


# ---------------------------------------------------------------------------
# U4-V2 · 归档 Journey 的记忆默认不进新 Journey 检索（画像也不继承）
# ---------------------------------------------------------------------------
def test_u4_v2_archived_journey_not_inherited():
    db = SessionLocal()
    try:
        _seed_user(db, "u-arch")
        old = mstore.create_companion(db, "u-arch", relationship_type="pet")
        old_mem = iso.scoped_add_memory(db, "u-arch", old.id,
                                        content="豆豆是一只小猫，我爱豆豆", importance=10.0)
        old_portrait = _portrait(db, "u-arch", old.id, "豆豆")
        mstore.archive_companion(db, old.id, ritual="dissolved")

        new = mstore.create_companion(db, "u-arch", relationship_type="breakup")
        new_mem = iso.scoped_add_memory(db, "u-arch", new.id,
                                        content="新旅程的狗叫糯米", importance=10.0)
        new_portrait = _portrait(db, "u-arch", new.id, "糯米")

        scope = iso.build_scope(db, "u-arch", new.id)
        hits = iso.scoped_recall(db, "u-arch", new.id, "豆豆 小猫 糯米", top_k=10)
        portraits = iso.scoped_portraits(db, "u-arch", new.id)

        print("\n[U4-V2] archived id:", old.id, "new id:", new.id)
        print("[U4-V2] new scope:", json.dumps(scope.as_dict(), ensure_ascii=False))
        print("[U4-V2] recall ids:", _ids(hits))
        print("[U4-V2] portrait rows:", [(p.id, p.companion_id, p.content) for p in portraits])

        assert old.id not in scope.allowed_companion_ids
        assert str(old_mem.id) not in scope.allowed_ids
        assert str(old_mem.id) not in _ids(hits)
        assert str(new_mem.id) in _ids(hits)
        assert all(p.companion_id == new.id for p in portraits)
        assert old_portrait.id not in {p.id for p in portraits}
        assert new_portrait.id in {p.id for p in portraits}

        # 归档 Journey 自身只读：读得到、写不进
        assert iso.scoped_get_memory(db, "u-arch", old.id, old_mem.id).id == old_mem.id
        with pytest.raises(iso.ScopeDenied) as ew:
            iso.assert_scope(db, "u-arch", old.id, old_mem.id, operation="write")
        with pytest.raises(iso.ScopeDenied) as ea:
            iso.scoped_add_memory(db, "u-arch", old.id, content="归档后续写")
        print("[U4-V2] archived write denied:", ew.value.code, "| add denied:", ea.value.code)
        assert ew.value.code == "ARCHIVED_READ_ONLY" and ea.value.code == "ARCHIVED_READ_ONLY"
    finally:
        db.close()


# ---------------------------------------------------------------------------
# U4-V3 · 显式授权后可跨 Journey 召回（含伪造授权越权测试）
# ---------------------------------------------------------------------------
def test_u4_v3_explicit_cross_journey_grant():
    db = SessionLocal()
    try:
        _seed_user(db, "u-grant")
        _seed_user(db, "u-attacker")
        old = mstore.create_companion(db, "u-grant", relationship_type="pet")
        old_mem = iso.scoped_add_memory(db, "u-grant", old.id, content="豆豆的旧回忆", importance=10.0)
        mstore.archive_companion(db, old.id, ritual="dissolved")
        new = mstore.create_companion(db, "u-grant", relationship_type="breakup")
        new_mem = iso.scoped_add_memory(db, "u-grant", new.id, content="糯米的新回忆", importance=10.0)

        before = iso.scoped_recall(db, "u-grant", new.id, "豆豆", top_k=10)
        grant = iso.grant_cross_journey(db, "u-grant", old.id, new.id, reason="用户显式授权回看旧旅程")
        scope_after = iso.build_scope(db, "u-grant", new.id, grants=[grant])
        after = iso.scoped_recall(db, "u-grant", new.id, "豆豆", grants=[grant], top_k=10)

        print("\n[U4-V3] grant:", json.dumps(grant.as_dict(), ensure_ascii=False))
        print("[U4-V3] before ids:", _ids(before), "after ids:", _ids(after))
        print("[U4-V3] granted scope:", json.dumps(scope_after.as_dict(), ensure_ascii=False))

        assert str(old_mem.id) not in _ids(before)
        assert old.id in scope_after.allowed_companion_ids
        assert str(old_mem.id) in scope_after.allowed_ids
        assert str(old_mem.id) in _ids(after)
        assert str(new_mem.id) in _ids(after)
        assert iso.scoped_get_memory(db, "u-grant", new.id, old_mem.id, grants=[grant]).id == old_mem.id

        # 授权只放宽读，不允许借授权改写旧 Journey 的记忆
        with pytest.raises(iso.ScopeDenied) as ew:
            iso.assert_scope(db, "u-grant", new.id, old_mem.id, grants=[grant], operation="write")
        print("[U4-V3] cross-journey write denied:", ew.value.code)
        assert ew.value.code == "SCOPE_DENIED"

        # 越权必含：攻击者拿别人的 companion_id 伪造授权
        attacker_c = mstore.create_companion(db, "u-attacker", relationship_type="pet")
        with pytest.raises(iso.ScopeDenied) as eg:
            iso.grant_cross_journey(db, "u-attacker", old.id, attacker_c.id, reason="伪造")
        forged = iso.CrossJourneyGrant("u-attacker", old.id, attacker_c.id, "伪造")
        with pytest.raises(iso.ScopeDenied) as ef:
            iso.build_scope(db, "u-grant", new.id, grants=[forged])
        print("[U4-V3] forged grant denied:", eg.value.code, "| mismatched grant denied:", ef.value.code)
        assert eg.value.code == "GRANT_SCOPE_DENIED" and ef.value.code == "GRANT_SCOPE_DENIED"
    finally:
        db.close()


# ---------------------------------------------------------------------------
# U4-V4 · scope 过滤发生在召回前：mock 证明未过滤库不被加载
# ---------------------------------------------------------------------------
def test_u4_v4_filter_before_recall_mock(monkeypatch):
    db = SessionLocal()
    try:
        _seed_user(db, "u-mock")
        _seed_user(db, "u-other")
        mine_comp = mstore.create_companion(db, "u-mock", relationship_type="pet")
        other_comp = mstore.create_companion(db, "u-other", relationship_type="pet")
        mine = iso.scoped_add_memory(db, "u-mock", mine_comp.id, content="豆豆陪我度过冬天", importance=10.0)
        other = iso.scoped_add_memory(db, "u-other", other_comp.id, content="豆豆是别人的隐私记忆", importance=10.0)
        old = mstore.create_companion(db, "u-mock", relationship_type="pet", make_current=False)
        old_mem = iso.scoped_add_memory(db, "u-mock", old.id, content="旧旅程的豆豆", importance=10.0)

        calls: list[dict] = []

        def boom(*args, **kwargs):
            raise AssertionError("未过滤库被加载：先召回后过滤，违反 U4-V4")

        def spy_score(entries, query, **kwargs):
            calls.append({"n": len(entries), "ids": [str(e.id) for e in entries]})
            return original_score(entries, query, **kwargs)

        original_score = iso.score_scoped_entries
        # 三条未过滤加载路径全部设成炸弹：全量 list_memories、基线 recall、vec0 索引加载
        monkeypatch.setattr(mstore, "list_memories", boom)
        monkeypatch.setattr(recall_mod, "recall", boom)
        monkeypatch.setattr(vi_mod, "get_index_for", boom)
        monkeypatch.setattr(iso, "score_scoped_entries", spy_score)

        hits = iso.scoped_recall(db, "u-mock", mine_comp.id, "豆豆", top_k=10)
        ids = _ids(hits)

        print("\n[U4-V4] scorer saw only these ids:", calls)
        print("[U4-V4] recall ids:", ids)
        print("[U4-V4] other-user memory:", other.id, "archived memory:", old_mem.id,
              "mine:", mine.id)

        assert len(calls) == 1, "打分层只能拿到一次已过滤批次"
        assert set(calls[0]["ids"]) == {str(mine.id)}, "打分层不得看到未过滤库"
        assert str(other.id) not in ids and str(old_mem.id) not in ids
        assert str(mine.id) in ids
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 附加安全边界：未 consent 默认拒绝（宁严勿松）
# ---------------------------------------------------------------------------
def test_u4_extra_no_consent_denies_everything():
    db = SessionLocal()
    try:
        _seed_user(db, "u-noconsent", consent=False)
        c = mstore.create_companion(db, "u-noconsent", relationship_type="pet")
        scope = iso.build_scope(db, "u-noconsent", c.id)
        print("\n[U4-EXTRA] no-consent scope:", json.dumps(scope.as_dict(), ensure_ascii=False))
        print("[U4-EXTRA] recall:", iso.scoped_recall(db, "u-noconsent", c.id, "任何"))

        assert scope.allowed_ids == frozenset()
        assert iso.scoped_recall(db, "u-noconsent", c.id, "任何") == []
        with pytest.raises(iso.ScopeDenied) as e:
            iso.scoped_add_memory(db, "u-noconsent", c.id, content="未授权写入")
        print("[U4-EXTRA] write denied:", e.value.code)
        assert e.value.code == "CONSENT_REQUIRED"
    finally:
        db.close()
