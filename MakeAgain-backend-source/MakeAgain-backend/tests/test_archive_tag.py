"""F1 (MEMORY_FINAL_CLOSURE_PLAN) 标签切换归档——行为验收，不是"全绿"。

F1-V1 归档后 current 查不到旧 Journey 记忆、as_of 能看到
F1-V2 其它 Journey 不受影响
F1-V3 幂等（重复归档不重复处理）
F1-V4 全量 pytest（见命令输出，不在此文件）

每个断言都 print 原始证据；用 `pytest -s tests/test_archive_tag.py` 采集。
"""
from __future__ import annotations

import datetime as dt

import pytest

from memory.five_layer import FiveLayerAdapter

START = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
AS_OF_BEFORE = "2026-01-05T00:00:00Z"


def _adapter(tmp_path):
    """隔离五层第二存储 + 冻结时钟，保证双时态断言确定性（沿用 test_five_layer 口径）。"""
    from agent_memory.core.clock import FrozenClock
    from agent_memory.core.config import Config

    clock = FrozenClock(START)
    config = Config()
    config.manage.git_commit = False
    return FiveLayerAdapter(tmp_path / "five-layer", clock=clock, config=config), clock


def _names(rows):
    return sorted(row["name"] for row in rows)


# ---------------------------------------------------------------- F1-V1
def test_f1_v1_archive_current_empty_as_of_visible(tmp_path):
    adapter, clock = _adapter(tmp_path)
    adapter.add("ja-cat", "养猫", user_id="u1", companion_id="J-A",
                provenance=["用户原话：我养了一只猫"], valid_from=START)
    adapter.add("ja-dog", "养狗", user_id="u1", companion_id="J-A",
                provenance=["用户原话：我还养了一条狗"], valid_from=START)
    before = adapter.current("养")

    clock.advance(days=10)
    archived = adapter.archive_journey("J-A")

    after = adapter.current("养")
    historic = adapter.current("养", as_of=AS_OF_BEFORE)
    print("\n[F1-V1] before archive, current(养):",
          [(r["name"], r["status"]) for r in before])
    print("[F1-V1] archive_journey('J-A') returned:", archived)
    print("[F1-V1] after archive, current(养):", after)
    print("[F1-V1] after archive, current(养, as_of=" + AS_OF_BEFORE + "):",
          [(r["name"], r["status"], r["invalid_at"]) for r in historic])
    print("[F1-V1] present_in_current now / as_of:",
          {n: (adapter.present_in_current(n), adapter.present_in_current(n, as_of=AS_OF_BEFORE))
           for n in ("ja-cat", "ja-dog")})

    assert archived == 2
    assert _names(before) == ["ja-cat", "ja-dog"]
    assert after == [], "归档后 current 不得再返回旧 Journey 的任何记忆"
    assert _names(historic) == ["ja-cat", "ja-dog"], "as_of 历史时点仍能看到被归档的旧记忆"
    assert all(r["status"] == "invalid" and r["invalid_at"] for r in historic)
    assert all(adapter.present_in_current(n) is False for n in ("ja-cat", "ja-dog"))
    assert all(adapter.present_in_current(n, as_of=AS_OF_BEFORE) is True
               for n in ("ja-cat", "ja-dog"))


# ---------------------------------------------------------------- F1-V2
def test_f1_v2_other_journey_untouched(tmp_path):
    adapter, clock = _adapter(tmp_path)
    adapter.add("ja-cat", "养猫", user_id="u1", companion_id="J-A",
                provenance=["用户原话：我养了一只猫"], valid_from=START)
    adapter.add("jb-flower", "喜欢茉莉", user_id="u1", companion_id="J-B",
                provenance=["用户原话：我最喜欢的花是茉莉"], valid_from=START)
    adapter.add("untagged-quiet", "独处时光", user_id="u1",
                provenance=["用户原话：我喜欢独处"], valid_from=START)  # 无 Journey 归属

    clock.advance(days=10)
    archived = adapter.archive_journey("J-A")

    rec_a = adapter.record("ja-cat")
    rec_b = adapter.record("jb-flower")
    cur_b = adapter.current("茉莉")
    cur_untagged = adapter.current("独处")
    cur_a = adapter.current("养")
    print("\n[F1-V2] archive_journey('J-A') returned:", archived)
    print("[F1-V2] J-A record after archive:", rec_a)
    print("[F1-V2] J-B record after archive:", rec_b)
    print("[F1-V2] current(茉莉) [other Journey]:", [(r["name"], r["status"]) for r in cur_b])
    print("[F1-V2] current(独处) [untagged]:", [(r["name"], r["status"]) for r in cur_untagged])
    print("[F1-V2] current(养) [archived Journey]:", cur_a)

    assert archived == 1
    assert rec_a["invalid_at"], "J-A 记录应被归档"
    assert rec_b["invalid_at"] is None, "J-B 记录不得被触碰"
    assert _names(cur_b) == ["jb-flower"] and adapter.present_in_current("jb-flower") is True
    assert _names(cur_untagged) == ["untagged-quiet"], "无归属记录不应被任何 Journey 归档"
    assert cur_a == []


# ---------------------------------------------------------------- F1-V3
def test_f1_v3_idempotent(tmp_path):
    adapter, clock = _adapter(tmp_path)
    adapter.add("ja-cat", "养猫", user_id="u1", companion_id="J-A",
                provenance=["用户原话：我养了一只猫"], valid_from=START)
    adapter.add("ja-dog", "养狗", user_id="u1", companion_id="J-A",
                provenance=["用户原话：我还养了一条狗"], valid_from=START)
    clock.advance(days=10)

    first = adapter.archive_journey("J-A")
    rec_first = adapter.record("ja-cat")
    total_after_first = len(adapter._store.records(include_invalid=True))

    second = adapter.archive_journey("J-A")
    third = adapter.archive_journey("J-A")
    rec_last = adapter.record("ja-cat")
    total_after_repeat = len(adapter._store.records(include_invalid=True))

    print("\n[F1-V3] returns: first / second / third =", first, second, third)
    print("[F1-V3] ja-cat invalid_at after first:", rec_first["invalid_at"])
    print("[F1-V3] ja-cat invalid_at after repeats:", rec_last["invalid_at"])
    print("[F1-V3] total records (all) after first / after repeats:",
          total_after_first, total_after_repeat)
    print("[F1-V3] current(养) after repeats:", adapter.current("养"))

    assert first == 2
    assert second == 0 and third == 0, "重复归档应返回 0，不重复处理"
    assert rec_last["invalid_at"] == rec_first["invalid_at"], "重复归档不得改动已有 invalid_at"
    assert total_after_repeat == total_after_first, "重复归档不得产生重复/新记录"
    assert adapter.current("养") == []
