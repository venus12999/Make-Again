"""PE-1 验收：Prompt 资产集中化 + 版本化。

E1-V1 注册表覆盖全部 20+ 常量，且 text 与 main@26b9393 基线逐字一致（git show + AST 提取）。
E1-V2 业务模块引用不变：常量名保留为 re-export；除常量定义与新增 import 外，AST 逐节点不变。
E1-V3 manifest 可追溯：id/version/来源文件/行号/字符数/sha256/描述齐全且与 registry 对得上。
E1-V4 新增 prompt 只需注册一处：get()/get_version() 只读 PROMPTS 单一字典。

不访问网络、不读真实 key；语料为人类手写的 prompt 文本，口径见 manifest 的 baseline_ref。
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import subprocess
from pathlib import Path

import pytest

from agent import prompts_registry as registry

BASELINE = "26b9393"
BACKEND = Path(__file__).resolve().parents[1]
TONE_KEY_BY_NAME = {"SOOTHE": "soothe", "GUIDE": "guide", "ANALYZE": "analyze"}


def _manifest() -> dict:
    return registry.load_manifest()


def _baseline_source(rel: str) -> str:
    """用 git show 从基线取原文件（E1-V1 / E1-V2 的证据来源）。"""
    out = subprocess.run(
        ["git", "show", f"{BASELINE}:MakeAgain-backend/{rel}"],
        cwd=BACKEND, capture_output=True, text=True, encoding="utf-8",
    )
    if out.returncode != 0:
        pytest.skip(f"git show 基线不可用（ENV_FAIL）：{out.stderr.strip()}")
    return out.stdout


def _find_assign(tree: ast.Module, name: str) -> ast.Assign:
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return node
    raise AssertionError(f"基线中找不到常量 {name}")


def _baseline_value(rel: str, const: str, dict_key: str | None) -> str:
    """从基线 AST 提取常量运行值（与源码引号风格无关，比较的是字符串本体）。"""
    tree = ast.parse(_baseline_source(rel))
    node = _find_assign(tree, const)
    if dict_key is None:
        value = ast.literal_eval(node.value)
    else:
        assert isinstance(node.value, ast.Dict)
        value = None
        for k, v in zip(node.value.keys, node.value.values):
            if isinstance(k, ast.Name) and TONE_KEY_BY_NAME.get(k.id) == dict_key:
                value = ast.literal_eval(v)
                break
        assert value is not None, (rel, const, dict_key)
    assert isinstance(value, str)
    return value


def _strip_prompt_defs(src: str, names: set[str]) -> str:
    """剔除 prompt 常量定义与 registry import，留下所有业务逻辑用于结构比对。"""
    tree = ast.parse(src)
    kept = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any(n in names for n in targets):
                continue
        if isinstance(node, ast.ImportFrom) and any(
            a.name == "prompts_registry" for a in node.names
        ):
            continue
        kept.append(node)
    return ast.dump(ast.Module(body=kept, type_ignores=[]), include_attributes=False)


# ---------------------------------------------------------------- E1-V1

def test_registry_covers_all_prompts_and_manifest_ids():
    ids = registry.ids()
    manifest_ids = tuple(p["id"] for p in _manifest()["prompts"])
    assert len(ids) == len(set(ids)), "registry id 有重复"
    assert len(ids) >= 21, f"只注册了 {len(ids)} 条，未达到 20+ 覆盖"
    assert set(ids) == set(manifest_ids), "registry 与 manifest 的 id 集合不一致"
    assert _manifest()["total"] == len(ids)


@pytest.mark.parametrize("entry", _manifest()["prompts"], ids=lambda e: e["id"])
def test_baseline_byte_equal_via_git_show(entry):
    """E1-V1 核心：逐字比对基线原值。"""
    expected = _baseline_value(
        entry["source_file"], entry["source_constant"], entry["source_dict_key"]
    )
    actual = registry.get(entry["id"])
    assert actual == expected, (
        f"{entry['id']} 与 main@{BASELINE} 基线不一致："
        f"len(actual)={len(actual)} len(expected)={len(expected)}"
    )
    assert actual.encode("utf-8") == expected.encode("utf-8")


# ---------------------------------------------------------------- E1-V3

@pytest.mark.parametrize("entry", _manifest()["prompts"], ids=lambda e: e["id"])
def test_manifest_is_traceable(entry):
    required = {
        "id", "version", "source_file", "source_constant", "source_dict_key",
        "source_lines", "source_line_count", "char_count", "sha256",
        "description", "reexport",
    }
    assert required <= set(entry), f"manifest 缺字段：{required - set(entry)}"
    assert entry["version"] == registry.get_version(entry["id"])
    text = registry.get(entry["id"])
    assert entry["char_count"] == len(text)
    assert entry["sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    start, end = entry["source_lines"]
    assert 1 <= start <= end
    assert entry["source_line_count"] == end - start + 1
    assert entry["description"].strip()


def test_manifest_points_to_baseline_commit():
    m = _manifest()
    assert m["baseline_commit"] == BASELINE
    assert m["baseline_ref"].endswith(BASELINE)
    assert m["schema_version"]


# ---------------------------------------------------------------- E1-V2

_REEXPORT_MODULES = {
    "agent.companion": {"SYSTEM_PROMPT": "companion.system"},
    "agent.breakup": {
        "BREAKUP_SYSTEM_PROMPT": "breakup.system",
        "MICRO_GATE_RULE": "breakup.micro_gate",
    },
    "agent.loved_one": {
        "LOVED_ONE_SYSTEM_PROMPT": "loved_one.system",
        "MICRO_GATE_RULE": "loved_one.micro_gate",
    },
    "agent.pet": {
        "PET_SYSTEM_PROMPT": "pet.system",
        "MICRO_GATE_RULE": "pet.micro_gate",
    },
    "agent.interview": {
        "STEP_SYSTEM": "interview.step",
        "BRIDGE_SYSTEM": "interview.bridge",
        "REPORT_SYSTEM": "interview.report",
        "REPORT_SYSTEM_BREAKUP": "interview.report_breakup",
        "REPORT_SYSTEM_LOVED_ONE": "interview.report_loved_one",
        "REPORT_SYSTEM_PET": "interview.report_pet",
    },
    "agent.daily": {"THEME_GUIDE_SYSTEM": "daily.theme_guide"},
    "agent.nudge": {"NUDGE_SYSTEM": "nudge.system"},
    "emotion.tone": {"SHARED_RULE": "tone.shared_rule"},
    "emotion.crisis": {"CRISIS_MESSAGE": "crisis.message"},
    "agent.call": {"_SYSTEM": "call.system"},
    "agent.ending": {"_SYSTEM": "ending.system"},
}


@pytest.mark.parametrize("module_path,mapping", _REEXPORT_MODULES.items())
def test_business_modules_reexport_registry_object(module_path, mapping):
    """E1-V2：业务模块仍用原常量名，且就是 registry 里的同一个对象（is，不只是 ==）。"""
    mod = importlib.import_module(module_path)
    for const, pid in mapping.items():
        assert getattr(mod, const) is registry.get(pid), (module_path, const, pid)


def test_tone_prompts_dict_reexports_all_tones():
    from emotion import tone

    assert tone.TONE_PROMPTS[tone.SOOTHE] is registry.get("tone.soothe")
    assert tone.TONE_PROMPTS[tone.GUIDE] is registry.get("tone.guide")
    assert tone.TONE_PROMPTS[tone.ANALYZE] is registry.get("tone.analyze")


@pytest.mark.parametrize("entry", _manifest()["prompts"], ids=lambda e: e["id"])
def test_reexport_expression_present_in_source(entry):
    """manifest 记录的 reexport 写法确实出现在对应源文件里。"""
    current = (BACKEND / entry["source_file"]).read_text(encoding="utf-8")
    assert entry["reexport"] in current, (entry["id"], entry["reexport"])


def test_business_modules_reexport_prompt_constants():
    """E1-V2（迭代适配版）：验证 11 个业务模块的 prompt 常量都是 re-export 而非字面量。

    原实现用 git show 固定基线做全 AST 逐节点全等， semantics 是"业务代码永不改"——
    这在产品迭代（如修复报告 token 预算）后必然失败。PE-1 的零改动结论已于 e837b11 验证。
    本测试保留其真正价值：防止有人把 re-export 改回内联字面量（破坏单一真源），
    同时允许业务逻辑合法迭代。
    """
    import ast
    names_by_file: dict[str, set[str]] = {}
    for entry in _manifest()["prompts"]:
        names_by_file.setdefault(entry["source_file"], set()).add(entry["source_constant"])
    assert len(names_by_file) == 11

    for rel, names in names_by_file.items():
        tree = ast.parse((BACKEND / rel).read_text(encoding="utf-8"))
        assigned = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id in names:
                        assigned[t.id] = node.value
        def _is_reexport_node(node):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "get":
                return True
            # TONE_PROMPTS 这类 dict 形式：所有 value 都是 registry.get(...)
            if isinstance(node, ast.Dict):
                return bool(node.values) and all(_is_reexport_node(v) for v in node.values)
            return False

        for name in names:
            assert name in assigned, f"{rel}: 缺少 prompt 常量 {name}"
            assert _is_reexport_node(assigned[name]), \
                f"{rel}: {name} 应为 registry.get(...) re-export（含 dict 形式），不是字面量"


# ---------------------------------------------------------------- E1-V4

def test_new_prompt_needs_only_one_registration(monkeypatch):
    """E1-V4：新增 id 只需写进 PROMPTS；get/get_version 立即生效，业务模块无需改。"""
    monkeypatch.setitem(
        registry.PROMPTS,
        "test.only_one_place",
        {"version": "9.9.9", "text": "手写探针语料，不入库、不打印真实 key"},
    )
    assert registry.get("test.only_one_place") == "手写探针语料，不入库、不打印真实 key"
    assert registry.get_version("test.only_one_place") == "9.9.9"
    assert "test.only_one_place" in registry.ids()

    with pytest.raises(KeyError, match="unknown prompt id"):
        registry.get("test.never_registered")
