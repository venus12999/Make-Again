"""E1 表达段中文原生批重造 · 行为验收（E1-V1 ~ E1-V4 + 不变量守卫）。

本文件只做离线确定性检查：读重造后的 01 批 JSON、从基线 commit 取旧批做逐字对比、
调用 build_e1_batch 的纯函数校验确定性。不联网、不读 key、不调用 LLM。
真实模型 judge 数字属于单点快评（run_benchmarks），不在本文件范围。

口径：
- E1-V1 n>=30 且三场景（分手失恋/亲人离世/宠物离世）各>=6；
- E1-V2 全中文原生 language=zh + 每条 source_id/provenance/scene/user_input/reference 可追溯；
- E1-V3 宠物离世条目标 synthetic-handwritten 自造口径；
- E1-V4 HEART 5 维 rubric + power_analysis 元信息保留、统计功效不退化。
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND / "eval"))

BATCH = BACKEND / "eval" / "batches" / "01_中文共情_CPsyCoun.json"
BASELINE = "aad3e951"
REL_BATCH = "MakeAgain-backend/eval/batches/01_中文共情_CPsyCoun.json"
HEART_KEYS = (
    "human_alignment", "empathic_responsiveness", "attunement",
    "resonance", "task_following",
)
REQUIRED_ITEM_FIELDS = ("id", "source_id", "scene", "language", "user_input", "reference", "source", "provenance")
PRODUCT_SCENES = ("分手失恋", "亲人离世", "宠物离世")
LEGACY_TAXONOMY = ("分手失恋", "亲人离世", "情绪困扰", "人际关系")


def _load() -> dict:
    return json.loads(BATCH.read_text(encoding="utf-8"))


def _baseline_items() -> list[dict]:
    out = subprocess.run(
        ["git", "show", f"{BASELINE}:{REL_BATCH}"],
        cwd=str(REPO), capture_output=True, text=True, encoding="utf-8",
    )
    assert out.returncode == 0, out.stderr[:200]
    return json.loads(out.stdout)["items"]


# ------------------------------------------------------------------ E1-V1
def test_v1_n_and_three_product_scenario_balance():
    d = _load()
    items = d["items"]
    assert d["n"] == len(items) >= 30
    counts = Counter(i["scene"] for i in items)
    for scene in PRODUCT_SCENES:
        assert counts[scene] >= 6, f"{scene} 仅 {counts[scene]} 条，未达产品场景配比"
    # ci_runner.EXPECTED_SCENES 的四个旧场景名必须全在（只增不减）
    for scene in LEGACY_TAXONOMY:
        assert counts[scene] >= 1, f"ci_runner taxonomy 缺 {scene}"
    assert counts == Counter(d["e1_rebuild"]["scene_distribution"])
    assert len({i["id"] for i in items}) == len(items)
    assert len({i["source_id"] for i in items}) == len(items)


# ------------------------------------------------------------------ E1-V2
def test_v2_every_item_chinese_native_and_traceable():
    d = _load()
    for it in d["items"]:
        for key in REQUIRED_ITEM_FIELDS:
            assert key in it and it[key], f"{it.get('id')} 缺 {key}"
        assert it["language"] == "zh", f"{it['id']} 非中文批"
        # 旧 34 条逐字保留，其中个别 reference 仅 25 字（如 cp-734），只要求非空且不空话
        assert len(it["user_input"]) >= 30 and len(it["reference"]) >= 10, it["id"]
        # 中文原生：CJK 占比足够高；无翻译批次遗留字段
        cjk = len(re.findall(r"[\u4e00-\u9fff]", it["user_input"]))
        assert cjk / len(it["user_input"]) >= 0.25, f"{it['id']} 不像中文原生语料"
        for translated_marker in ("needs_translation", "user_input_zh", "translated_from"):
            assert translated_marker not in it, f"{it['id']} 疑似翻译集残留 {translated_marker}"
        prov = it["provenance"]
        assert prov["kind"] in ("authoritative", "synthetic-handwritten"), it["id"]
        assert isinstance(prov.get("synthetic"), bool), it["id"]
        src = it["source"]
        assert isinstance(src, dict) and src.get("dataset"), it["id"]
        # source_id 能与数据集对上（ci_runner 同口径：canonical 前缀含 dataset 名）
        assert src["dataset"] in it["source_id"] or it["source_id"] == it["id"], it["id"]


# ------------------------------------------------------------------ E1-V3
def test_v3_pet_loss_explicitly_marked_synthetic_handwritten():
    d = _load()
    pets = [i for i in d["items"] if i["scene"] == "宠物离世"]
    assert len(pets) >= 6
    for it in pets:
        prov = it["provenance"]
        assert prov["kind"] == "synthetic-handwritten", it["id"]
        assert prov["synthetic"] is True, it["id"]
        assert it["source_id"].startswith("synthetic-handwritten:pet-loss:"), it["id"]
        assert "自造" in prov.get("note", ""), it["id"]
        assert "宠物" in prov.get("scene", ""), it["id"]
    not_found = d["e1_rebuild"]["pet_loss_not_found"]
    assert not_found["status"] == "NOT_FOUND"
    assert not_found["english_only_evidence"], "须留英文 pet-loss 集存在但语言错配的证据 URL"
    assert all(u.startswith("https://") for u in not_found["english_only_evidence"])


# ------------------------------------------------------------------ E1-V4
def test_v4_heart_rubric_and_power_analysis_retained():
    d = _load()
    assert "HEART" in d["eval"]
    for k in HEART_KEYS:
        assert k in d["eval"], f"eval 缺维度 {k}"
    assert d["rubric"]["keys"] == list(HEART_KEYS)
    assert d["rubric"]["spec"] == "eval/heart_rubric.py::build_judge_rubric"
    assert d["eval_control"]["text"], "旧自编维度对照字段须保留"
    power = d["power_analysis"]
    assert power["target_n"] == 30
    assert power["actual_n"] == len(d["items"]) >= 30
    assert power["alpha"] == 0.05 and power["power"] == 0.8 and power["repetitions"] >= 3
    rb = d["e1_rebuild"]["research_basis"]
    assert rb["cpsycoun"]["decay_weight"] == 0.4 and rb["cpsycoun"]["task_weight"] == 0.7
    assert rb["psyqa"]["decay_weight"] == 0.1 and rb["psyqa"]["task_weight"] == 0.4
    assert "2405.16433" in rb["cpsycoun"]["urls"][0]
    assert "2106.01702" in rb["psyqa"]["urls"][0]
    for key in ("deep_emotional_support_zh", "auradial", "matcha", "kardiabench"):
        assert rb[key]["published"] and rb[key]["url"] and rb[key]["age_days"] >= 0, key
    import build_e1_batch as beb
    for cfg in rb.values():
        if "decay_weight" in cfg and "age_days" in cfg:
            assert cfg["decay_weight"] == beb.decay_weight(cfg["age_days"])


# ------------------------------------------------------------------ 不变量
def test_legacy_34_items_are_byte_for_byte_preserved_on_core_fields():
    """旧批逐字保留：与 H1 baseline 的 per-case 配对仍成立，HEART 守卫也不破。"""
    old_by_id = {i["id"]: i for i in _baseline_items()}
    new_by_id = {i["id"]: i for i in _load()["items"]}
    assert len(old_by_id) == 34
    assert set(old_by_id).issubset(new_by_id), "旧 34 条 id 必须全在"
    for oid, old in old_by_id.items():
        new = new_by_id[oid]
        for key in ("user_input", "reference", "scene", "source_id", "provenance", "source"):
            assert new.get(key) == old.get(key), f"{oid} 的 {key} 被改动"


def test_pet_items_are_deterministic_from_build_script():
    """宠物条自造内容是脚本常量，重跑不漂移（可复现）。"""
    import build_e1_batch as beb
    generated = {i["id"]: i for i in beb.build_pet_items()}
    in_batch = {i["id"]: i for i in _load()["items"] if i["scene"] == "宠物离世"}
    assert set(generated) == set(in_batch)
    for pid, item in generated.items():
        assert in_batch[pid]["user_input"] == item["user_input"]
        assert in_batch[pid]["reference"] == item["reference"]
        assert in_batch[pid]["source_id"] == item["source_id"]


def test_no_key_material_in_batch_metadata():
    """key 明文不入库：批元信息里不能出现 sk-/Bearer 等凭据样式。"""
    raw = BATCH.read_text(encoding="utf-8")
    assert "sk-" not in raw and "Bearer " not in raw
