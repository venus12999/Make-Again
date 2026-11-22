"""E6 完整旅程跨段链路评测的行为验收（全离线、不烧 key、不联网）。

对应任务 E6-V1~V5：
- V1 --mock 跑通 3 场景 × 5 检查点；批次 persona 全标 synthetic-handwritten。
- V2 段间衔接点有真实数字：真实 hy4 走查/复跑实物（outputs/walkthrough、
  outputs/e2）过 CP1；产品函数在无 LLM 下验证「报告落库→对话注入」与
  「记忆写入→召回→触发时机」两个衔接的真实行为与数字。
- V3 真 hy4 1 场景（报告+对话+回写成功）由 CLI 手工慢跑，证据落 outputs/e6/；
  本文件在证据存在时校验其结构，不存在则明确 skip（不伪称跑过）。
- V4 四种故障注入（注入点缺失/回写失败/召回失败/触发失败）→ exit 1 + 对应 CP FAIL，
  证明检查点不是摆设（"全绿不是标准"）。
- V5 分层说明写进 docs/E6_FULL_JOURNEY_RESULT.md；本文件随全量 pytest 无回归。

口径：persona/mock trace 均为 synthetic-handwritten 构造，非真人数据；
真实报告数字来自 D-042/D-045 的 hy4 走查实物，key 不进仓库。
"""
from __future__ import annotations

import json
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND / "eval"))

import e6_full_journey as e6  # noqa: E402
import e2_interview_chain as e2  # noqa: E402
from agent import interview, companion, nudge  # noqa: E402
from memory import store  # noqa: E402
from memory.db import _RawSessionLocal  # noqa: E402

BATCH = BACKEND / "eval" / "batches" / "E6_完整旅程.json"
RUNNER = BACKEND / "eval" / "e6_full_journey.py"
WALKTHROUGH = REPO / "outputs" / "walkthrough" / "journey_walkthrough.json"
E2_REAL = REPO / "outputs" / "e2" / "e2_real_breakup_rescored.json"
E6_REAL = REPO / "outputs" / "e6" / "e6_real_breakup.json"
LAYER_DOC = REPO / "docs" / "E6_FULL_JOURNEY_RESULT.md"


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(RUNNER), *args],
        cwd=str(BACKEND), capture_output=True, text=True, encoding="utf-8", timeout=180,
    )


# ------------------------------------------------------------------ V1: 结构与口径
def test_v1_mock_runs_three_personas_with_five_junction_checkpoints(tmp_path):
    out = tmp_path / "mock.json"
    proc = _run_cli("--mock", "--out", str(out))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["mode"] == "mock" and data["summary"]["users"] == 3
    assert all(r["source_id"].startswith("e6-synthetic-") for r in data["results"])
    assert {r["loss_type"] for r in data["results"]} == {"breakup", "loved_one", "pet"}
    assert data["summary"]["checkpoint_pass_counts"] == {name: 3 for name in e6.CHECKPOINT_NAMES}
    assert data["summary"]["overall_pass"] is True
    for r in data["results"]:
        assert set(r["checkpoints"]) == set(e6.CHECKPOINT_NAMES)
        ev = r["evidence"]
        for key in ("report_memory_count", "portrait_key_count", "context_anchor_hits",
                    "memories_before", "memories_after", "days_shift", "recall_hits",
                    "negative_recall_hits", "guide_anchor_hits"):
            assert isinstance(ev[key], int), key
    print("\n[V1] 3 persona × 5 衔接检查点全 PASS；口径=" + data["口径"])


def test_batch_is_synthetic_traceable_and_registered():
    obj = json.loads(BATCH.read_text(encoding="utf-8"))
    assert obj["batch"] == BATCH.stem and obj["n"] == len(obj["items"]) == 3
    idx = json.loads((BATCH.parent / "_index.json").read_text(encoding="utf-8"))
    assert BATCH.stem in idx["batches"]  # ci_runner 的 batch_files_match_index 依赖登记
    for it in obj["items"]:
        assert it["id"] == it["source_id"]
        assert it["source"]["dataset"] == "synthetic-handwritten"
        assert it["persona"]["loss_type"] in ("breakup", "loved_one", "pet")
        assert len(it["persona"]["answers"]) == 5 and len(it["persona"]["anchors"]) >= 2
        assert all(it["journey"].get(k) for k in (
            "dialogue_query", "recall_query", "recall_negative_query",
            "trigger_local_now", "trigger_daytime_now"))
    print("\n[数据口径] 3 条 synthetic-handwritten + source_id 可追溯 + 已登记 _index")


# ------------------------------------------------------------------ V2: 真实数字
def test_v2_real_hy4_artifacts_ground_cp1_numbers():
    """CP1 用真实 hy4 走查/复跑实物做数值锚点，不用 mock 数字冒充产品证据。"""
    walk = json.loads(WALKTHROUGH.read_text(encoding="utf-8"))["results"][0]
    e2run = json.loads(E2_REAL.read_text(encoding="utf-8"))["results"][0]
    assert (walk["n_turns"], walk["gen_filler_turns"]) == (20, 15)
    assert (e2run["trace"]["n_turns"], e2run["trace"]["gen_filler_turns"]) == (13, 8)
    assert e2run["trace"]["elapsed_s"] == 415.3
    pers = e2run["evidence"]["personalization"]
    assert pers["matched_anchors"] == ["小雅", "跨年", "江边", "到老", "到家了吗"]

    item = e2.load_batch()[0]
    scored = e6.evaluate_journey(item, e2run["trace"] | {"report": e2run["report"]})
    assert scored["checkpoints"]["CP1_report_ready"] is True
    assert scored["evidence"]["turns"]["n_turns"] == 13
    print("\n[V2真实数字] 走查20轮/空转15；真hy4复跑13轮/空转8/415.3s/锚点5/5 → CP1 PASS")


def test_v2_report_to_dialogue_injection_uses_real_product_code(monkeypatch):
    """报告落库→对话注入：无 LLM，压真实 _persist_report + _context_block。"""
    report = json.loads(WALKTHROUGH.read_text(encoding="utf-8"))["results"][0]["report"]
    uid = "e6-inject-" + uuid.uuid4().hex[:8]
    with _RawSessionLocal() as db:
        store.get_or_create_user(db, uid, loss_type="breakup")
        interview._persist_report(db, uid, "breakup", report)
        report_mems = [m for m in store.list_memories(db, uid) if m.type == "report"]
        assert len(report_mems) == 2, "报告 summary + 关系复盘各一条（interview._persist_report）"
        portraits = {"user": store.get_portrait(db, uid, "user"),
                     "object": store.get_portrait(db, uid, "object")}
        assert portraits["user"] and portraits["object"]
        # _context_block 消费 recall() 的 {"entry": MemoryEntry} 结构（companion.chat 内部同款）
        ctx = companion._context_block(db, uid, "小雅和跨年那句话", portraits,
                                       [{"entry": m} for m in report_mems])
        assert "[用户画像]" in ctx and "[思念对象画像]" in ctx
        anchors = ["小雅", "跨年", "江边", "到老", "到家了吗"]
        hits = [a for a in anchors if a in ctx]
        assert hits, f"报告内容没有注入对话上下文: {ctx[:300]}"
        print(f"\n[V2段2] 报告记忆={len(report_mems)} 画像字段="
              f"{len(portraits['user']) + len(portraits['object'])} 注入锚点={hits}")


def test_v2_writeback_recall_and_trigger_numbers_are_real(monkeypatch):
    """回写→召回→触发：无 LLM，压真实 store.add_memory / recall / nudge 规则。"""
    monkeypatch.setenv("VECTOR_INDEX_DB_PATH", ":memory:")
    monkeypatch.setenv("ENTITY_GRAPH_DB_PATH", ":memory:")
    from memory import recall as recall_mod
    if hasattr(recall_mod, "reset_index_cache"):
        recall_mod.reset_index_cache()

    uid = "e6-recall-" + uuid.uuid4().hex[:8]
    with _RawSessionLocal() as db:
        store.get_or_create_user(db, uid, loss_type="breakup")
        before = len(store.list_memories(db, uid))
        e = store.add_memory(db, uid, type="chat", content="她叫小雅，跨年在江边说要到老，现在手机再也不响",
                             importance=9.0)
        after = len(store.list_memories(db, uid))
        assert after == before + 1
        hits = recall_mod.recall(db, uid, "小雅 跨年 江边 到老", top_k=5, recent_text="")
        neg = recall_mod.recall(db, uid, "今天晚饭吃什么", top_k=5, recent_text="")
        assert len(hits) >= 1 and any("小雅" in (h["entry"].summary or h["entry"].content) for h in hits)
        assert len(neg) == 0, "U-013 query-intent 门：无关 query 不得注入旧记忆"

        late = __import__("datetime").datetime(2026, 11, 23, 23, 30)
        day = __import__("datetime").datetime(2026, 11, 23, 14, 30)
        late_rules = [c["rule_key"] for c in nudge._build_candidates(db, uid, late, breakup=True)]
        day_rules = [c["rule_key"] for c in nudge._build_candidates(db, uid, day, breakup=True)]
        assert "late_night" in late_rules, late_rules
        assert "late_night" not in day_rules, day_rules
        print(f"\n[V2段3-5] 记忆{before}→{after} recall={len(hits)} top_score="
              f"{hits[0]['score']:.4f} 负例={len(neg)} 深夜触发={late_rules} 白天={day_rules}")
        assert str(e.id)


# ------------------------------------------------------------------ V3: 真 hy4 慢评证据
def test_v3_real_hy4_one_scenario_evidence_if_present():
    if not E6_REAL.exists():
        pytest.skip("真 hy4 慢评需手工跑 CLI 生成 outputs/e6/e6_real_breakup.json（不进 pytest）")
    data = json.loads(E6_REAL.read_text(encoding="utf-8"))
    assert data["mode"] == "real" and data["summary"]["users"] == 1
    r = data["results"][0]
    assert r["loss_type"] == "breakup"
    assert r["e2_checkpoints"]["CP2_report_generated"] is True
    ev = r["evidence"]
    assert ev["report_memory_count"] >= 1 and ev["memories_after"] > ev["memories_before"]
    assert ev["writeback_anchor_hits"] >= 1
    assert ev["recall_hits"] >= 1 and ev["negative_recall_hits"] == 0
    assert all(r["checkpoints"][n] for n in e6.CHECKPOINT_NAMES[1:]), \
        "V3 要求报告→对话→回写→召回→触发五段衔接全部成立"
    # 不粉饰：本次真跑 CP1 失败（报告字段缺 + 19轮>16），E6 如实暴露；
    # 若失败必须能定位到 E2 的证据，不允许"整体 FAIL 但说不清哪里错"。
    if not r["overall_pass"]:
        assert r["checkpoints"]["CP1_report_ready"] is False
        assert ev["turns"]["n_turns"] > ev["turns"]["max_allowed_turns"] or ev["missing_fields"]
    print(f"\n[V3真实慢评] 报告生成={r['e2_checkpoints']['CP2_report_generated']} "
          f"记忆{ev['memories_before']}→{ev['memories_after']} "
          f"回写锚点={ev['writeback_anchor_hits']} job={ev['job_status']} "
          f"召回={ev['recall_hits']}(top={ev['recall_top_score']:.4f}) 触发={ev['late_nudge_rule']} "
          f"CP1={'PASS' if r['checkpoints']['CP1_report_ready'] else 'FAIL(已披露)'} "
          f"轮数={ev['turns']['n_turns']}")


# ------------------------------------------------------------------ V4: 守卫有效性
@pytest.mark.parametrize("fault,cp", [
    ("missing_report_injection", "CP2_report_to_dialogue"),
    ("missing_writeback", "CP3_dialogue_to_writeback"),
    ("recall_miss", "CP4_memory_to_crossday_recall"),
    ("trigger_miss", "CP5_recall_to_trigger"),
])
def test_v4_fault_injection_fails_only_the_targeted_junction(tmp_path, fault, cp):
    out = tmp_path / f"fault-{fault}.json"
    proc = _run_cli("--mock", "--fault", fault, "--out", str(out))
    assert proc.returncode == 1
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["summary"]["overall_pass"] is False
    assert all(r["checkpoints"][cp] is False for r in data["results"])
    others = [n for n in e6.CHECKPOINT_NAMES if n != cp]
    assert all(all(r["checkpoints"][n] for n in others) for r in data["results"])
    print(f"\n[V4守卫] 注入 {fault} → exit=1，仅 {cp} FAIL（衔接检查有区分度）")


# ------------------------------------------------------------------ V5: 分层文档
def test_v5_layering_documented():
    if not LAYER_DOC.exists():
        pytest.skip("docs/E6_FULL_JOURNEY_RESULT.md 待授权后补")
    text = LAYER_DOC.read_text(encoding="utf-8")
    assert "快评" in text and "深评" in text and "--mock" in text
