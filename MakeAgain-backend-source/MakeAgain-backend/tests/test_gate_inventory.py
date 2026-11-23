"""门禁清单静态检查（CI 离线跑，防止"套壳"复发）。

验收标准：核心产品风险必须都有门禁定义；ACTIVE 门禁必须有可执行 command。
不检查指标数值（那需要真 key，由本地 gate_all.py 跑），只检查"门禁体系没被掏空"。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "eval"))


def test_core_risks_have_gate_definitions():
    import gate_all
    # 三个核心产品风险必须在清单里（可以被标 TODO，但不能消失）
    for risk in ["fact_retention", "crisis_recall", "chain_complete"]:
        assert risk in gate_all.GATES, f"核心风险 {risk} 没有门禁定义——评测体系被掏空"
        g = gate_all.GATES[risk]
        assert g.get("risk") and g.get("metric") and g.get("status"), f"{risk} 门禁定义不完整"


def test_active_gates_have_runnable_command():
    import gate_all
    actives = {n: g for n, g in gate_all.GATES.items() if g["status"] == "ACTIVE"}
    assert actives, "没有任何活跃门禁——评测体系形同虚设"
    for name, g in actives.items():
        assert g.get("command"), f"活跃门禁 {name} 没有可执行 command（套壳：声称有门禁却跑不了）"
        assert isinstance(g["threshold"], (int, float)), f"活跃门禁 {name} 阈值不可判定"


def test_fact_retention_gate_is_active_with_real_evidence():
    import gate_all
    g = gate_all.GATES["fact_retention"]
    assert g["status"] == "ACTIVE", "长对话丢事实门禁被降级——D-065的真实bug会重新漏过去"
    assert "0.66" in str(g.get("command")) or g["threshold"] >= 0.6
