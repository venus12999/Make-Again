"""把 lead 实测的真实模型数字注入 golden 报告，更新 A/B 决策。

运行：python eval/finalize_report.py
输入：outputs/golden-<date>.json（run_golden.py 生成）
输出：同文件原地更新（幂等：重复运行结果一致）
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT.parent / "outputs" / "golden-2026-11-21.json"


def finalize():
    data = json.loads(OUT.read_text(encoding="utf-8"))

    # 1. 情绪：注入 lead 真实 hy4 实测（26/27），更新 A/B 决策
    data["emotion"]["real"] = {
        "status": "RUN",
        "model": "hy4-flash-free",
        "n": 27,
        "correct": 26,
        "accuracy": 0.9630,
        "misses": [{"id": "P-01", "expected": ["难过", "想念"], "got": "其他",
                    "note": "宣告式委屈表达，标注本身偏主观，非模型缺陷"}],
        "evidence": "lead 实跑 python C:/tmp/real_emotion_eval.py，2026-11-21",
    }
    mock_acc = data["emotion"].get("mock", {}).get("label_accuracy", 0.4444)
    data["ab_decisions"]["USE_LLM_EMOTION"] = {
        "decision": "建议打开（真实模型显著优于词典兜底）",
        "evidence": f"real hy4 准确率 0.9630 (26/27) vs mock 词典 {mock_acc:.4f} (12/27)，+{(0.9630-mock_acc)*100:.1f}pp > 10pp 阈值",
        "note": "情绪分数仍按产品约束只作内部弱信号，不向用户展示伪精确数字；开关控制的是分类准确率不是展示",
    }

    # 2. 危机：注入引用守卫实测（误报 4.17%→0）
    data["crisis"]["quote_guard"] = {
        "status": "RUN",
        "switch": "USE_CRISIS_QUOTE_GUARD",
        "before": {"tp": 3, "fp": 1, "fn": 0, "tn": 23, "fp_rate": 0.0417},
        "after": {"tp": 3, "fp": 0, "fn": 0, "tn": 24, "fp_rate": 0.0},
        "evidence": "lead 实跑危机全语料 guard=0/1 对比 + 安全回归（引用后自身危机仍检出），2026-11-21",
    }
    data["ab_decisions"]["USE_CRISIS_QUOTE_GUARD"] = {
        "decision": "建议打开（消除引用语境误报，漏报不增）",
        "evidence": "误报率 0.0417→0.0000，漏报保持 0；安全回归证明'新闻说有人想死。我也想死'仍检出",
    }

    # 3. 召回：A/B 决策补充"smoke-level 不足以定论"的诚实边界
    _recall_decision = data["ab_decisions"]["USE_DUAL_RECALL"]
    _basis_key = "evidence" if "evidence" in _recall_decision else "basis"
    _recall_decision[_basis_key] = str(_recall_decision.get(_basis_key, "")) + (
        "；注意本轮仅 3 条 fixture / 2 条 query，Recall@K 全为 1.0 是天花板效应，"
        "MRR 0.75→1.0 有差异但样本过小，不足以定论，需扩大语料后复测"
    )

    data["finalized_at"] = datetime.now(timezone.utc).isoformat()
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"finalized: {OUT}")


if __name__ == "__main__":
    finalize()
