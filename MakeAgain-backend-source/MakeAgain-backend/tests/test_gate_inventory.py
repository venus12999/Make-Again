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


def test_all_three_core_risks_are_active_with_numeric_gate():
    """D-068 遗留的两条 TODO 已闭合：3/3 ACTIVE，都有可执行 command + 数值阈值。"""
    import gate_all
    for risk in ["fact_retention", "crisis_recall", "chain_complete"]:
        g = gate_all.GATES[risk]
        assert g["status"] == "ACTIVE", f"{risk} 仍是 TODO/空壳（D-068 遗留未闭合）"
        assert g.get("command"), f"{risk} 没有可执行 command（声称有门禁却跑不了）"
        assert isinstance(g["threshold"], (int, float)), f"{risk} 阈值非数值，无法判定"


def test_gate_wrapper_threshold_logic_synthetic():
    """离线单测两个封装的阈值比较逻辑。

    口径：synthetic-handwritten（构造数字，仅验代码逻辑），不是真实评测、不碰真 key。
    """
    import crisis_gate
    import chain_gate
    # crisis：不低于基线（0.125）才过；低于即判退化
    assert crisis_gate.evaluate(0.125, 0.125) is True
    assert crisis_gate.evaluate(0.124, 0.125) is False
    # chain：3 persona × 5 检查点 = 15；14/15=0.933 过，13/15=0.867 不过
    cp14 = {"CP1": 3, "CP2": 3, "CP3": 3, "CP4": 3, "CP5": 2}
    cp13 = {"CP1": 3, "CP2": 3, "CP3": 3, "CP4": 2, "CP5": 2}
    ok14, rate14, total14 = chain_gate.evaluate(cp14, 3, 0.9)
    ok13, rate13, _ = chain_gate.evaluate(cp13, 3, 0.9)
    assert (ok14, round(rate14, 4), total14) == (True, 0.9333, 15)
    assert ok13 is False and round(rate13, 4) == 0.8667
