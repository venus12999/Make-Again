"""把 V2 召回评测数字写入决策，更新 golden 报告的 A/B 建议。

运行：python eval/finalize_recall_v2.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
V2 = REPO / "outputs" / "golden-recall-v2.json"
V1 = REPO / "outputs" / "golden-2026-11-21.json"


def main():
    v2 = json.loads(V2.read_text(encoding="utf-8"))
    baseline = v2["dual0_sem0"]
    dual = v2["dual1_sem0"]
    semantic = v2["dual1_sem1"]

    delta_recall = round(semantic["Recall@K"] - baseline["Recall@K"], 4)
    print(f"V2 recall A/B: baseline R@K={baseline['Recall@K']} -> dual1_sem1 R@K={semantic['Recall@K']} (Δ={delta_recall})")
    print(f"  MRR {baseline['MRR']} -> {semantic['MRR']}, first_hit {baseline['expected_first_hit_rate']} -> {semantic['expected_first_hit_rate']}")

    # 更新 V1 报告的召回 A/B 决策（V1 因 3 fixture 样本不足误判为"保持关闭"）
    v1 = json.loads(V1.read_text(encoding="utf-8"))
    basis = "V2 33条记忆/18query 实测（outputs/golden-recall-v2.json）推翻 V1 的 3fixture 天花板结论："
    v1["ab_decisions"]["USE_DUAL_RECALL"] = {
        "decision": "建议打开（V2 扩充语料证明显著提升）",
        "basis": basis + f"dual1_sem0 Recall@K={dual['Recall@K']} vs dual0_sem0={baseline['Recall@K']} (Δ+{round(dual['Recall@K']-baseline['Recall@K'],4)})，MRR {baseline['MRR']}→{dual['MRR']}",
    }
    v1["ab_decisions"]["USE_SEMANTIC_EMBEDDING"] = {
        "decision": "建议打开（配合 dual recall 取得最大增益）",
        "basis": basis + f"dual1_sem1 Recall@K={semantic['Recall@K']} vs dual1_sem0={dual['Recall@K']} (Δ+{round(semantic['Recall@K']-dual['Recall@K'],4)})，MRR {dual['MRR']}→{semantic['MRR']}；注意负例注入率仍为1.0（无关query仍召回记忆），需后续设相关性阈值",
    }
    v1["recall_v2"] = {
        "status": "RUN",
        "corpus": "33 memories / 18 queries / synthetic-handwritten",
        "baseline_dual0_sem0": {k: baseline[k] for k in ("Recall@K", "Precision@K", "MRR", "expected_first_hit_rate")},
        "dual1_sem0": {k: dual[k] for k in ("Recall@K", "Precision@K", "MRR", "expected_first_hit_rate")},
        "dual1_sem1": {k: semantic[k] for k in ("Recall@K", "Precision@K", "MRR", "expected_first_hit_rate")},
        "known_issue": "negative_query_injection_rate=1.0（q-16 无关问题仍召回3条记忆），需相关性阈值，记 U-013",
    }
    V1.write_text(json.dumps(v1, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"updated: {V1}")


if __name__ == "__main__":
    main()
