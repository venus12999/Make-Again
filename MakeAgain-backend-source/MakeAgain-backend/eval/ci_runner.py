#!/usr/bin/env python3
"""H2 · CI 离线评测门禁（eval/batches 结构校验 + 确定性检查）。

只做能被代码确定的事，不调用任何 LLM、不读任何 key 文件、不联网。
输出单一结构化 JSON 到 stdout；退出码 0=PASS / 1=FAIL，直接做 CI gate。
强制 `MOCK_LLM=true`（无条件覆盖外部取值），保证 CI 里绝不碰真 key。

调研依据（2026-11-23 实查，URL + 日期）：
1. GitHub Actions 官方 workflow 语法 —— push/pull_request paths 过滤、on/jobs/steps 结构：
   https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax （访问 2026-11-23）
   官方 Python 构建测试教程（actions/checkout@v4 + actions/setup-python@v5 的标准组合）：
   https://docs.github.com/en/actions/tutorials/build-and-test-code/python （访问 2026-11-23）
   → 本仓 workflow 照此用 checkout@v4 / setup-python@v5，push 每次跑、PR 在 prompt/eval 变更时必跑。
2. promptfoo 官方 CI/CD 与 GitHub Actions 指南（Last updated on Nov 21, 2026）：
   https://www.promptfoo.dev/docs/integrations/ci-cd/
   https://www.promptfoo.dev/docs/integrations/github-action/
   → 借鉴"PR 路径过滤 + 结构化结果 + 阈值不达标 exit 1 做 quality gate"，
     不借鉴其"把 OPENAI_API_KEY 注入 CI"的做法（本仓红线：CI 不碰真 key）。
3. DeepEval 官方《Regression Testing LLM Systems in CI/CD》（文档快照 2026-07-09）：
   https://github.com/confident-ai/deepeval/blob/main/docs/content/guides/guides-regression-testing-in-cicd.mdx
   → 借鉴"把评测集条目当 unit test、push/PR 都触发回归"；其 deepeval test run 需 API key，
     本仓改为纯本地确定性检查。
4. Anthropic《Demystifying evals for AI agents》（2026-01-08）：
   https://www.anthropic.com/research/demystifying-evals-for-ai-agents
   原文要点：自动化 eval 可 "run on every commit"；regression eval 应保持接近 100% 通过率；
   code-based graders 具备 fast / cheap / objective / reproducible 特性，优先于 model-based。
   Anthropic Claude Academy《Continuous evals in CI》（© 2026）：
   https://academy.claude.com/courses/ai-native-sdlc-playbook/continuous-evals-in-ci
   → 本门禁定位 = 每次提交跑的确定性回归层；模型判定/judge 校准留给本地与人工校准环节。

用法（在 MakeAgain-backend 目录下）：
    python eval/ci_runner.py                       # stdout 打印 JSON，exit 0/1
    python eval/ci_runner.py --out gate.json       # 同时落盘（CI 上传产物）
    python eval/ci_runner.py --batches-dir <dir>   # 指向临时目录做构造性验证

检查项（全部离线、确定性）：
    mock_llm_forced / batches_dir_exists / index_valid / batch_files_match_index /
    json_parse / batch_schema / item_schema / source_id_traceability / taxonomy_coverage
source_id 兼容两种可追溯写法：本地 id（cp-786）或 canonical（CPsyCoun:train:786，需
source.id/provenance 锚回本条）。重复 id/source_id 作为 warning 如实暴露，不计入 exit code。

安全：不打印、不入库任何 key；结果 JSON 不含时间戳/临时路径，同输入字节一致。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DEFAULT_BATCHES = HERE / "batches"
SCHEMA_VERSION = "1.0"

# taxonomy 期望来自 eval/build_benchmarks.py 的分组定义（每类至少 1 条即算覆盖）。
# H1 扩样本只增不减，不应删类；删类/改名会让 CI 失败，这是刻意的门禁语义。
EXPECTED_SCENES: dict[str, tuple[str, ...]] = {
    "01_中文共情_CPsyCoun": ("分手失恋", "亲人离世", "情绪困扰", "人际关系"),
    "02_支持策略_ESConv": ("分手", "丧亲/失去", "抑郁情绪"),
    "03_情绪识别_GoEmotions": (
        "情绪-sadness", "情绪-anger", "情绪-love",
        "情绪-grief", "情绪-annoyance", "情绪-nervousness",
    ),
    "04_危机安全_CueSupport": ("高危-支持性", "高危-有害回复", "中危-支持性", "中危-忽视"),
}
REQUIRED_BATCH_KEYS = ("batch", "source", "paper", "eval", "n", "items")
MAX_DETAILS = 50


def force_mock_llm() -> str:
    """无条件强制 mock，返回实际 env 值（供检查项记录）。"""
    os.environ["MOCK_LLM"] = "true"
    return os.environ["MOCK_LLM"]


def _check(name: str, ok: bool, message: str,
           details: list[str] | None = None, metrics: dict[str, Any] | None = None,
           warnings: list[str] | None = None) -> dict:
    result = {
        "name": name,
        "status": "pass" if ok else "fail",
        "message": message,
        "details": (details or [])[:MAX_DETAILS],
        "metrics": metrics or {},
    }
    if warnings:
        # 非阻断告警：如实暴露基线数据瑕疵，但不把 H1 尚未修完的既有数据
        # 变成 H2 门禁红灯；缺 source_id / 缺字段 / n 不符等仍是硬失败。
        result["warnings"] = warnings[:MAX_DETAILS]
    return result


def _guard(name: str, fn: Callable[[], dict]) -> dict:
    """单个检查项抛异常时记为 FAIL，绝不吞掉变成 PASS，也不崩掉整个 JSON 输出。"""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - gate 需要把任何异常变成可诊断的 FAIL
        return _check(name, False, f"EXCEPTION: {type(exc).__name__}: {exc}")


def _short(detail: str) -> str:
    """细节文本限长，避免坏数据把 JSON 撑爆（同时避免打印任何敏感内容）。"""
    return detail if len(detail) <= 240 else detail[:240] + "..."


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _effective_source_id(item: dict) -> Any:
    """兼容两种写法：显式 item.source_id（可为 canonical 形式），或批次原有的 item.source.id。

    两者都缺 = 不可追溯 = FAIL（构造性验收 H2-V3 即删掉两者）。
    """
    if not isinstance(item, dict):
        return None
    explicit = item.get("source_id")
    if isinstance(explicit, str) and explicit.strip():
        return explicit
    source = item.get("source")
    if isinstance(source, dict):
        sid = source.get("id")
        if isinstance(sid, str) and sid.strip():
            return sid
    return None


def _traceable_pair(item: dict, sid: str, local_id: Any) -> bool:
    """source_id 与 item.id 的对应关系是否可追溯。

    两种都接受：
      1) 旧格式 source_id == item.id（如 cp-786），并且 source.dataset 非空；
      2) canonical 格式（如 CPsyCoun:train:786）：含 ':' 且带数据集名，
         同时用 source.id == item.id 或 provenance(dataset,row_index) 锚回本条。
    单有任意字符串不算可追溯；缺 source_id 由调用方判 FAIL。
    """
    if sid == local_id:
        return True
    if ":" not in sid:
        return False
    source = item.get("source") if isinstance(item.get("source"), dict) else {}
    prov = item.get("provenance") if isinstance(item.get("provenance"), dict) else {}
    datasets = [x for x in (source.get("dataset"), prov.get("dataset")) if isinstance(x, str) and x]
    if not datasets or not any(d in sid for d in datasets):
        return False
    return (source.get("id") == local_id
            or (bool(prov.get("dataset")) and prov.get("row_index") is not None))


def run(batches_dir: Path) -> dict:
    """跑全部检查，返回结构化结果（不含时间戳，确定性）。"""
    checks: list[dict] = []

    mock_value = force_mock_llm()
    checks.append(_check(
        "mock_llm_forced", mock_value == "true",
        "MOCK_LLM=true 强制覆盖；本进程不加载 LLM 客户端/key 文件/网络",
        metrics={"mock_llm": mock_value},
    ))

    if not batches_dir.is_dir():
        checks.append(_check(
            "batches_dir_exists", False, f"batches 目录不存在: {batches_dir.name}",
            details=[f"path={_short(str(batches_dir))}"],
        ))
        return _finalize(checks, 0, 0)

    checks.append(_check("batches_dir_exists", True, f"batches 目录存在: {batches_dir.name}"))

    # ---- index 与文件集合 ----
    index_path = batches_dir / "_index.json"

    def check_index() -> dict:
        if not index_path.is_file():
            return _check("index_valid", False, "缺 _index.json")
        idx = _load_json(index_path)
        names = idx.get("batches") if isinstance(idx, dict) else None
        ok = (isinstance(idx, dict) and bool(idx.get("schema_version"))
              and isinstance(names, list) and bool(names)
              and all(isinstance(n, str) and n for n in names)
              and len(set(names)) == len(names))
        return _check("index_valid", bool(ok), "index schema/batches 列表校验",
                      [] if ok else ["schema_version 或 batches 列表缺失/重复/非字符串"],
                      {"batches_in_index": len(names) if isinstance(names, list) else 0})

    checks.append(_guard("index_valid", check_index))

    actual_files = sorted(p for p in batches_dir.glob("*.json") if p.name != "_index.json")
    actual_names = [p.stem for p in actual_files]

    def check_files_match() -> dict:
        idx_names: list[str] = []
        if index_path.is_file():
            idx = _load_json(index_path)
            if isinstance(idx, dict) and isinstance(idx.get("batches"), list):
                idx_names = [n for n in idx["batches"] if isinstance(n, str)]
        missing = sorted(set(idx_names) - set(actual_names))
        extra = sorted(set(actual_names) - set(idx_names))
        ok = bool(idx_names) and not missing and not extra
        details = []
        if missing:
            details.append("index 列出但文件缺失: " + ", ".join(_short(x) for x in missing))
        if extra:
            details.append("文件存在但 index 未列: " + ", ".join(_short(x) for x in extra))
        return _check("batch_files_match_index", ok,
                      "_index.json 与 batches/*.json 一一对应", details,
                      {"files": len(actual_names), "indexed": len(idx_names)})

    checks.append(_guard("batch_files_match_index", check_files_match))

    # ---- JSON 可解析 ----
    parsed: dict[str, Any] = {}

    def check_parse() -> dict:
        bad = []
        for p in actual_files:
            try:
                obj = _load_json(p)
                if not isinstance(obj, dict):
                    bad.append(f"{p.name}: top-level 不是 object")
                    continue
                parsed[p.stem] = obj
            except Exception as exc:  # noqa: BLE001
                bad.append(f"{p.name}: {type(exc).__name__}: {_short(str(exc))}")
        return _check("json_parse", not bad, "每个批次是合法 JSON object",
                      [_short(x) for x in bad], {"files": len(actual_files)})

    checks.append(_guard("json_parse", check_parse))

    # ---- 批次结构 + n 一致性 ----
    def check_batch_schema() -> dict:
        bad = []
        for name, obj in parsed.items():
            for key in REQUIRED_BATCH_KEYS:
                if key not in obj:
                    bad.append(f"{name}: 缺顶层字段 {key}")
            if obj.get("batch") != name:
                bad.append(f"{name}: batch 字段={obj.get('batch')!r} 与文件名不符")
            items = obj.get("items")
            if not isinstance(items, list) or not items:
                bad.append(f"{name}: items 不是非空 list")
            elif obj.get("n") != len(items):
                bad.append(f"{name}: n={obj.get('n')!r} 但 items 实际 {len(items)} 条")
        return _check("batch_schema", not bad, "批次顶层字段齐全且 n==len(items)",
                      [_short(x) for x in bad], {"batches": len(parsed)})

    checks.append(_guard("batch_schema", check_batch_schema))

    all_items: list[tuple[str, int, dict]] = []
    for name, obj in parsed.items():
        items = obj.get("items")
        if isinstance(items, list):
            all_items.extend((name, i, it) for i, it in enumerate(items) if isinstance(it, dict))

    # ---- item 结构 ----
    def check_item_schema() -> dict:
        bad = []
        warnings = []
        seen: set[tuple[str, str]] = set()
        for name, i, item in all_items:
            for key in ("id", "scene", "language", "user_input"):
                v = item.get(key)
                if not isinstance(v, str) or not v.strip():
                    bad.append(f"{name}#{i}: {key} 缺失/空/非字符串")
            ident = item.get("id")
            if isinstance(ident, str):
                key = (name, ident)
                if key in seen:
                    warnings.append(f"{name}#{i}: 重复 id={_short(ident)}（既有基线数据瑕疵，建议 H1 去重的抽样修复）")
                seen.add(key)
        return _check("item_schema", not bad, "item 必填字段齐全（重复 id 记 warning）",
                      [_short(x) for x in bad], {"items": len(all_items)},
                      [_short(x) for x in warnings])

    checks.append(_guard("item_schema", check_item_schema))

    # ---- source_id 可追溯 ----
    def check_source_id() -> dict:
        bad = []
        warnings = []
        seen: set[tuple[str, str]] = set()
        traced = 0
        for name, i, item in all_items:
            sid = _effective_source_id(item)
            ident = item.get("id")
            if not isinstance(sid, str) or not sid.strip():
                bad.append(f"{name}#{i}: 缺 source_id（item.source_id 与 item.source.id 都为空）")
                continue
            if not _traceable_pair(item, sid, ident):
                bad.append(f"{name}#{i}: source_id={_short(sid)} 无法与 item.id={_short(str(ident))}/dataset/provenance 对上")
                continue
            source = item.get("source")
            dataset = source.get("dataset") if isinstance(source, dict) else None
            if not isinstance(dataset, str) or not dataset.strip():
                bad.append(f"{name}#{i}: source.dataset 缺失/空（无法追到原始数据集）")
                continue
            key = (dataset, sid)
            if key in seen:
                warnings.append(f"{name}#{i}: source_id 重复 dataset={_short(dataset)} id={_short(sid)}（既有基线抽样重复，记 warning）")
            else:
                seen.add(key)
            traced += 1
        return _check("source_id_traceability", not bad,
                      "每条 item 的 source_id 非空且可追溯（本地 id 或 canonical dataset:split:row，重复记 warning）",
                      [_short(x) for x in bad],
                      {"items": len(all_items), "traceable": traced},
                      [_short(x) for x in warnings])

    checks.append(_guard("source_id_traceability", check_source_id))

    # ---- taxonomy 覆盖 ----
    def check_taxonomy() -> dict:
        bad = []
        covered = 0
        expected_total = 0
        for name, expected in EXPECTED_SCENES.items():
            if name not in parsed:
                bad.append(f"{name}: 批次缺失，无法核对 taxonomy")
                continue
            scenes = {it.get("scene") for it in parsed[name].get("items", []) if isinstance(it, dict)}
            expected_total += len(expected)
            for scene in expected:
                if scene in scenes:
                    covered += 1
                else:
                    bad.append(f"{name}: taxonomy 缺场景 {scene}")
        return _check("taxonomy_coverage", not bad,
                      "4 批次权威 taxonomy 场景全覆盖（每类至少 1 条）",
                      [_short(x) for x in bad],
                      {"expected": expected_total, "covered": covered})

    checks.append(_guard("taxonomy_coverage", check_taxonomy))

    return _finalize(checks, len(parsed), len(all_items))


def _finalize(checks: list[dict], batches: int, items: int) -> dict:
    warnings = []
    for c in checks:
        detail = c.pop("warnings", None)
        if detail:
            warnings.append({
                "check": c["name"],
                "message": "非阻断告警：已如实暴露，不计入 exit code",
                "details": detail,
            })
    failed = sum(1 for c in checks if c["status"] != "pass")
    ok = failed == 0
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": "ci_runner",
        "mock_llm_forced": True,
        "status": "PASS" if ok else "FAIL",
        "exit_code": 0 if ok else 1,
        "summary": {
            "checks_total": len(checks),
            "checks_passed": len(checks) - failed,
            "checks_failed": failed,
            "batches_checked": batches,
            "items_checked": items,
            "warnings_total": sum(len(w["details"]) for w in warnings),
        },
        "checks": checks,
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="离线评测门禁（JSON + exit code）")
    parser.add_argument("--batches-dir", type=Path, default=DEFAULT_BATCHES,
                        help="batches 目录（默认 eval/batches）")
    parser.add_argument("--out", type=Path, default=None, help="可选：同时把 JSON 写到文件")
    parser.add_argument("--quiet", action="store_true", help="不向 stdout 打印 JSON")
    args = parser.parse_args(argv)

    result = run(args.batches_dir)
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    if not args.quiet:
        print(text)
    return int(result["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
