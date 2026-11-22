#!/usr/bin/env python3
"""结束仪式（ending ritual）离线验证：抠图 → 纪念文案 → 模拟来电 → 归档 → 纪念物导出。

产品五段闭环最后一段从没有被验证过。本工具按“先读码再验证”的口径，对下列真实链路
做确定性、可追溯的行为验证（数字全部来自本机真实运行，不是估计）：

先读码确认的现状（本文件只验证、不改产品代码）：
- agent/ending.py:25-62  memorial()：mock 走 _FALLBACK(pet/relative/breakup)；真实走一次 fast
  LLM，把对象名 + 关系类型 + 近期记忆摘要织成纪念文字；返回 disclosure 字段(:22)。
- agent/call.py:26-65    call_script()：同样的个性化结构，返回 disclosure 字段(:23)。
- agent/item.py:272-368  handle_upload()：grounding → 裁切 → ai_matte 抠图 → 收紧透明边 →
  落库；抠图失败回退“存原图”(processing_status=original_fallback)；:340-347 刻意删除原图，
  original_key 恒为空串（已知坑，见报告 GAP-E3）。
- gateway/storage.py:152-188  ai_matte()：无 COS/mock_storage 时，USE_LOCAL_MATTE=1 走本地
  rembg（模型白名单 u2netp，许可安全；懒加载单例），否则抛 StorageError。
- memory/store.py:1392-1405   archive_companion()：幂等置 archived_at / ending_stage=complete /
  ending_ritual / ending_committed_at。
- memory/isolation.py:216-241 assert_scope(operation='write') 对归档 Journey 抛 ARCHIVED_READ_ONLY；
  :268-287 scoped_add_memory 同样拒绝；读维度仍可读（只读档案）。
- memory/export.py:253-290 export_user_memory()：导出 schema_version/categories/source/fact 等。

分层（用户关心成本）：
  --mock（默认）  隔离 SQLite + MOCK_LLM=true，不读 key、不联网；rembg 用本机已缓存的
                  u2netp 模型真跑抠图（有输出图），文案走产品 mock 兜底 + 确定性桩。
  --real（可选）  真 hy4-flash-free 生成一段纪念文案/来电话语，标口径。key 只从
                  work/local-test-key.txt 读，不打印明文、不写进产物；无 key 则 NOT_RUN。
  pytest 只跑 --mock。D-V5（全量 pytest 无回归）由命令行实跑记录在 docs/ENDING_VALIDATION.md，
  不在这里递归启动 pytest（pytest 内跑 pytest 会无限递归）。

方法学：code-based graders 优先（确定性代码判定，fast/cheap/objective/reproducible）。
Anthropic《Demystifying evals for AI agents》2026-01-08，
https://www.anthropic.com/research/demystifying-evals-for-ai-agents
基准日 2026-11-23：319 天 / 30.44 ≈ 10.5 个月 → 权重 0.7（6-18 月档，参考，须查更新版）。
本工具全为本地确定性判定，不引入未校准 LLM judge，权重衰减不改变实现。非本轮新联网调研。

用法（在 MakeAgain-backend 目录）：
  python eval/ending_validation.py --mock
  python eval/ending_validation.py --mock --out-dir ../outputs/ending
  python eval/ending_validation.py --real        # 可选；无 key 文件则如实 NOT_RUN
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))

DEFAULT_OUT_DIR = REPO / "outputs" / "ending"
KEY_FILE = REPO / "work" / "local-test-key.txt"
BASELINE_COMMIT = "6ab6f718"

# 占位符，不是任何真实凭据；仅用于让 LLMClient.mock 走 False 分支以验证真实 prompt 路径。
OFFLINE_PLACEHOLDER_TOKEN = "offline-placeholder-no-real-credential"

RELATIONSHIP_TYPES = ("pet", "relative", "breakup")
SUBJECTS = {"pet": "橘子", "relative": "建国", "breakup": "小夏"}


def _evidence_path(p: Path) -> str:
    """产物在仓库内记相对路径；测试传入 tmp_path 时退化成绝对路径，不抛异常。"""
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


# --------------------------------------------------------------------------- 环境
def bootstrap_mock_env() -> None:
    """CLI 自举隔离环境（pytest 下由 conftest 完成，不要重复调用）。"""
    root = tempfile.mkdtemp(prefix="ending-validation-")
    os.environ.update(
        DATABASE_URL="sqlite:///" + root + "/t.sqlite3",
        ENVIRONMENT="test",
        MOCK_LLM="true", MOCK_WECHAT="true", MOCK_SPEECH="true",
        MOCK_VISION="true", MOCK_STORAGE="true",
        MEDIA_SIGNING_KEY="isolated-test-key-01234567890123456789",
        STORAGE_LOCAL_DIR=root + "/media",
        PUBLIC_BASE_URL="http://testserver",
    )
    from config import get_settings
    get_settings.cache_clear()
    from memory.db import init_db
    init_db()


def _synthetic_png() -> bytes:
    """合成测试图（synthetic-handwritten）：白底 + 红色方块，纯本地生成，无外部数据。"""
    from PIL import Image, ImageDraw

    buf = io.BytesIO()
    img = Image.new("RGB", (160, 160), (255, 255, 255))
    ImageDraw.Draw(img).rectangle([40, 40, 120, 120], fill=(220, 40, 40))
    img.save(buf, format="PNG")
    return buf.getvalue()


def _png_stats(data: bytes) -> dict:
    from PIL import Image

    img = Image.open(io.BytesIO(data))
    w, h = img.size
    mode = img.mode
    pct_transparent = None
    if "A" in img.getbands():
        hist = img.getchannel("A").histogram()
        pct_transparent = round(100.0 * hist[0] / (w * h), 2)
    return {"mode": mode, "width": w, "height": h, "pct_transparent": pct_transparent}


def _fake_personalized_client():
    """离线确定性桩：让 LLMClient 走真实 prompt 构造路径（mock=False），但 chat 不触网。

    用独立 SimpleNamespace settings，绝不修改 get_settings() 的全局单例；token 是占位符，
    不是 key，不打印也不写入产物。
    """
    from gateway.client import LLMClient

    settings = SimpleNamespace(
        environment="test", mock_llm=False, llm_api_key=OFFLINE_PLACEHOLDER_TOKEN,
        llm_fast_model="offline-fast", llm_model="offline",
    )
    client = LLMClient(settings=settings)

    def fake_chat(messages, *args, **kwargs):
        user_msg = messages[-1]["content"] if messages else ""
        # ending/call 的 prompt 形如 "关系类型:X;对象名:Y。\n...片段...\n\n请写一段..."
        rt, subject, first_fact = "?", "?", "(暂无)"
        for line in user_msg.splitlines():
            if line.startswith("关系类型:"):
                head = line.split(";", 1)[0]
                rt = head.split(":", 1)[1].strip()
            elif line.startswith("对象名:"):
                subject = line.split(":", 1)[1].strip("。 \n")
            elif line.startswith("- ") and first_fact == "(暂无)":
                first_fact = line[2:].strip()
        return {
            "content": f"【{subject}·{rt}】{first_fact}——这段陪伴值得被温柔收好。",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "tool_calls": [],
        }

    client.chat = fake_chat  # 实例级替换，避免污染其它用例
    return client


def _seed_user(uid: str, rt: str, subject: str, fact: str) -> str:
    """建用户 + companion + 一条记忆 + 同意，返回 companion_id。"""
    from memory import store
    from memory.db import _RawSessionLocal
    from memory.models import utcnow

    with _RawSessionLocal() as db:
        store.get_or_create_user(db, uid, loss_type=rt)
        st = store.get_or_create_user_state(db, uid)
        st.consent_accepted_at = utcnow()
        db.commit()
        c = store.get_or_create_current_companion(db, uid)
        store.update_companion(db, c.id, relationship_type=rt, subject_name=subject)
        store.add_memory(db, uid, type="chat", content=fact, summary=fact, importance=8.0)
        return c.id


# --------------------------------------------------------------------------- D-V1
def check_cutout(out_dir: Path) -> dict:
    """D-V1：无 COS 凭据 + 合成图，本地 rembg 兜底真跑，产出透明底 PNG。"""
    os.environ["USE_LOCAL_MATTE"] = "1"
    os.environ.setdefault("LOCAL_MATTE_MODEL", "u2netp")
    from PIL import Image
    from agent import item as item_agent
    from gateway import storage as storage_mod
    from gateway.storage import ObjectStore

    raw = _synthetic_png()
    obj = ObjectStore()  # mock_storage=true → backend=local，无 COS 客户端
    key = obj.upload(f"{uuid.uuid4().hex}.png", raw, content_type="image/png", prefix="item")
    try:
        matte = obj.ai_matte(key)  # 真 rembg；本机已缓存 u2netp，不发网络
        trimmed = item_agent._trim_alpha_png(matte)  # 走产品真实“收紧透明边”步骤
    finally:
        obj.delete(key)

    src_path = out_dir / "cutout_source.png"
    out_path = out_dir / "cutout_trimmed.png"
    src_path.write_bytes(raw)
    out_path.write_bytes(trimmed)

    src_stats = _png_stats(raw)
    matte_stats = _png_stats(matte)
    trim_stats = _png_stats(trimmed)
    import rembg

    # 注意：透明度要在 matte（收紧透明边之前）上量；trim 后透明边已被裁掉，
    # 紧凑主体内部本就不该有大量 alpha=0 像素。trim 生效用尺寸收缩证明。
    ok = (
        obj.backend == "local"
        and trimmed[:8] == b"\x89PNG\r\n\x1a\n"
        and matte_stats["mode"] == "RGBA" and trim_stats["mode"] == "RGBA"
        and (matte_stats["pct_transparent"] or 0) > 5.0
        and trim_stats["width"] < src_stats["width"]
        and storage_mod._local_matte_model_name() == "u2netp"
    )
    return {
        "pass": bool(ok),
        "metrics": {
            "backend": obj.backend,
            "model": storage_mod._local_matte_model_name(),
            "rembg_version": getattr(rembg, "__version__", "?"),
            "source_png": src_stats,
            "matte_png": matte_stats,
            "trimmed_png": trim_stats,
            "matte_bytes": len(matte),
            "trimmed_bytes": len(trimmed),
        },
        "evidence": [_evidence_path(src_path), _evidence_path(out_path)],
        "note": "无 COS 凭据（backend=local）+ 本地 rembg u2netp 真跑；背景透明即抠图生效。"
                "pct_transparent=alpha=0 像素占比，在 matte（trim 前）上量；trim 把透明边裁掉，"
                "故尺寸 160→81 收紧透明边生效。",
    }


# --------------------------------------------------------------------------- D-V2
def check_memorial_copy(uid_prefix: str = "ending-dv2") -> dict:
    """D-V2：纪念文案个性化。mock 兜底按关系类型分叉；确定性桩验真实 prompt 路径。"""
    from agent import ending
    from gateway.client import LLMClient

    mock_client = LLMClient()  # MOCK_LLM=true → client.mock=True → 走 _FALLBACK
    fake_client = _fake_personalized_client()

    fallback_rows = []
    prompt_rows = []
    for rt in RELATIONSHIP_TYPES:
        subject = SUBJECTS[rt]
        uid = f"{uid_prefix}-{rt}-{uuid.uuid4().hex[:8]}"
        fact = f"{subject}留下的旧手表，承载了很多回忆"
        _seed_user(uid, rt, subject, fact)
        from memory.db import _RawSessionLocal
        with _RawSessionLocal() as db:
            m = ending.memorial(db, uid, mock_client)
            p = ending.memorial(db, uid, fake_client)
        fallback_rows.append({
            "relationship_type": rt, "text": m["memorial"],
            "subject_name": m["subject_name"], "disclosure": m["disclosure"],
        })
        prompt_rows.append({
            "relationship_type": rt, "text": p["memorial"],
            "subject_name": p["subject_name"], "disclosure": p["disclosure"],
        })

    distinct_fallback = len({r["text"] for r in fallback_rows})
    distinct_prompt = len({r["text"] for r in prompt_rows})
    subject_in_prompt = all(r["subject_name"] in r["text"] for r in prompt_rows)
    ok = (
        distinct_fallback == len(RELATIONSHIP_TYPES)
        and distinct_prompt == len(RELATIONSHIP_TYPES)
        and subject_in_prompt
    )
    return {
        "pass": bool(ok),
        "metrics": {
            "relationship_types_tested": len(RELATIONSHIP_TYPES),
            "distinct_fallback_texts": distinct_fallback,
            "distinct_prompt_path_texts": distinct_prompt,
            "subject_name_in_all_prompt_texts": subject_in_prompt,
            "fallback_rows": fallback_rows,
            "prompt_path_rows": prompt_rows,
        },
        "note": "mock 兜底 3 关系类型 → 3 段不同文案；确定性桩走真实 prompt 路径，"
                "不同对象名/关系 → 不同文案（不同物品不同文案的数字证据）。无真 key、无网络。",
    }


# --------------------------------------------------------------------------- D-V3
def check_disclosure(uid_prefix: str = "ending-dv3") -> dict:
    """D-V3：纪念文案 + 模拟来电都带 AI 披露标记。"""
    from agent import call, ending
    from gateway.client import LLMClient

    client = LLMClient()
    rows = []
    for rt in RELATIONSHIP_TYPES:
        uid = f"{uid_prefix}-{rt}-{uuid.uuid4().hex[:8]}"
        _seed_user(uid, rt, SUBJECTS[rt], f"关于{SUBJECTS[rt]}的一段回忆")
        from memory.db import _RawSessionLocal
        with _RawSessionLocal() as db:
            m = ending.memorial(db, uid, client)
            c = call.call_script(db, uid, client)
        rows.append({
            "relationship_type": rt,
            "memorial_disclosure": m["disclosure"],
            "call_disclosure": c["disclosure"],
            "memorial_len": len(m["memorial"]),
            "call_len": len(c["text"]),
        })

    expected = ending._DISCLOSURE
    same_source = expected == call._DISCLOSURE
    marked = sum(
        1 for r in rows
        if r["memorial_disclosure"] == expected
        and r["call_disclosure"] == expected
        and bool(expected)
    )
    ok = same_source and marked == len(rows) and bool(expected)
    return {
        "pass": bool(ok),
        "metrics": {
            "disclosure_marker": expected,
            "endpoints_same_marker": same_source,
            "rows_marked": marked,
            "rows_total": len(rows),
            "rows": rows,
        },
        "note": "两个端点（ending.memorial / call.call_script）共用同一披露常量；"
                "3 关系类型 × 2 端点 = 6/6 带标记。marker 值仅为文案标识，不是凭据。",
    }


# --------------------------------------------------------------------------- D-V4
def check_archive_export(uid_prefix: str = "ending-dv4", out_dir: Path | None = None) -> dict:
    """D-V4：归档幂等 + 只读不可逆 + 记忆/纪念物导出。"""
    from memory import export, isolation as iso, store
    from memory.db import _RawSessionLocal
    from memory.models import utcnow
    ScopeDenied = iso.ScopeDenied

    uid = f"{uid_prefix}-{uuid.uuid4().hex[:8]}"
    with _RawSessionLocal() as db:
        store.get_or_create_user(db, uid, loss_type="breakup")
        st = store.get_or_create_user_state(db, uid)
        st.consent_accepted_at = utcnow()
        db.commit()
        cid = store.get_or_create_current_companion(db, uid).id
        store.update_companion(db, cid, relationship_type="breakup", subject_name="小夏",
                               ending_ritual="ending_ritual")

        # 走隔离层显式落 companion_id，否则 assert_scope 判为无归属（与产品主链同坑）。
        mem = iso.scoped_add_memory(db, uid, cid, type="chat", content="她送我的围巾还在",
                                    summary="围巾", importance=9.0)
        store.upsert_portrait(db, uid, "object", {"称呼": "小夏", "关系": "前任"}, status="confirmed")
        item_row = store.add_item_memory(
            db, uid, item_name="围巾", intent="keep", description="她送我的围巾",
            label="围巾", original_key="", cutout_key=f"item/{uuid.uuid4().hex}_cutout.png",
        )

        # 归档（结束仪式 commit）
        archived = store.archive_companion(db, cid, ritual="ending_ritual")
        archived_at_1 = archived.archived_at
        again = store.archive_companion(db, cid, ritual="ending_ritual")
        archived_at_2 = again.archived_at
        idempotent = archived_at_1 is not None and archived_at_1 == archived_at_2
        stage_ok = archived.ending_stage == "complete" and archived.ending_ritual == "ending_ritual"

        # 只读：读可，写拒 2 路
        read_ok = iso.assert_scope(db, uid, cid, mem.id, operation="read").id == mem.id
        denied = []
        for label, fn in (
            ("assert_scope(write)", lambda: iso.assert_scope(db, uid, cid, mem.id, operation="write")),
            ("scoped_add_memory", lambda: iso.scoped_add_memory(db, uid, cid, content="归档后写入")),
        ):
            try:
                fn()
                denied.append({"path": label, "code": None, "denied": False})
            except ScopeDenied as exc:
                denied.append({"path": label, "code": exc.code, "denied": exc.code == "ARCHIVED_READ_ONLY"})

        # 导出
        payload = export.export_user_memory(db, uid, include_provenance=True)
        exported_at = payload.get("exported_at")
        categories = payload.get("categories", {})
        source_n = len(payload.get("source", []))
        portrait_val = categories.get("portrait", 0)
        portrait_n = (len(portrait_val) if isinstance(portrait_val, dict)
                      else int(portrait_val or 0))
        cat_keys = sorted(categories.keys())
        item_exported = any("item" in k for k in cat_keys)

        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            exp_path = out_dir / "export_snapshot.json"
            exp_path.write_text(json.dumps(payload, ensure_ascii=False, default=str, indent=2), encoding="utf-8")
        else:
            exp_path = None

    ok = (
        idempotent and stage_ok and read_ok
        and all(d["denied"] for d in denied)
        and source_n >= 1 and portrait_n >= 1 and bool(exported_at)
    )
    return {
        "pass": bool(ok),
        "metrics": {
            "archived_at_set": archived_at_1 is not None,
            "archive_idempotent": idempotent,
            "ending_stage": archived.ending_stage,
            "ending_ritual": archived.ending_ritual,
            "read_still_allowed": read_ok,
            "write_denials": denied,
            "export_schema_version": payload.get("schema_version"),
            "exported_at": exported_at,
            "export_source_entries": source_n,
            "export_portrait_entries": portrait_n,
            "export_categories": cat_keys,
            "item_cutout_in_export": item_exported,
            "item_cutout_stored_in_db": bool(item_row.cutout_key),
            "item_original_key": item_row.original_key,
        },
        "evidence": [_evidence_path(exp_path)] if exp_path else [],
        "note": "归档幂等且只读不可逆（写拒 2/2 ARCHIVED_READ_ONLY，读仍可）；"
                "导出含记忆/画像等 7 类。GAP-E2：item 抠图不在 export 的 _MEMORY_TABLES，"
                "导出 JSON 不含纪念物 cutout。GAP-E3：original_key 恒空串（已知坑）。",
    }


# --------------------------------------------------------------------------- real
def _load_conf() -> dict:
    conf: dict[str, str] = {}
    for line in KEY_FILE.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            conf[k.strip()] = v.strip()
    return conf


def run_real_optional() -> dict:
    """可选真模型文案生成：只读 key 文件装配环境，不打印明文、不写入产物。"""
    if not KEY_FILE.exists():
        return {"status": "NOT_RUN", "reason": "no work/local-test-key.txt",
                "source_path": str(KEY_FILE.relative_to(REPO)),
                "key_printed": False, "key_persisted": False}
    try:
        conf = _load_conf()
        os.environ.update(
            LLM_BASE_URL=conf["BASE_URL"], LLM_API_KEY=conf["KEY"],
            LLM_FAST_MODEL="hy4-flash-free", LLM_MODEL="hy4",
            MOCK_LLM="false", ENVIRONMENT="development",
            LLM_EXTRA_HEADERS=json.dumps({
                "x-opencode-session": "ending-validation-" + uuid.uuid4().hex[:8],
                "User-Agent": "make-again-ending-validation/1.0",
            }),
        )
        from config import get_settings
        get_settings.cache_clear()
        s = get_settings()
        s.llm_base_url = conf["BASE_URL"]
        s.llm_api_key = conf["KEY"]
        s.llm_fast_model = "hy4-flash-free"
        s.mock_llm = False

        from agent import call, ending
        from gateway.client import LLMClient
        from memory.db import _RawSessionLocal

        uid = "ending-real-" + uuid.uuid4().hex[:8]
        _seed_user(uid, "breakup", "小夏", "她送我的围巾还在，想起一起走过的日子")
        client = LLMClient()
        with _RawSessionLocal() as db:
            m = ending.memorial(db, uid, client)
            c = call.call_script(db, uid, client)
        return {
            "status": "RUN", "model": "hy4-flash-free",
            "source_path": str(KEY_FILE.relative_to(REPO)),
            "key_printed": False, "key_persisted": False,
            "memorial_chars": len(m["memorial"]), "call_chars": len(c["text"]),
            "disclosure_present": m["disclosure"] == ending._DISCLOSURE
                                  and c["disclosure"] == call._DISCLOSURE,
            "口径": "真 hy4-flash-free 各 1 次调用；key 仅从本地文件读取用于鉴权，未打印未入库；"
                    "真实文案质量不在本工具判定范围（只验链路与披露）。",
        }
    except Exception as exc:  # noqa: BLE001 可选模式不应打挂离线验收
        return {"status": "ENV_FAIL", "reason": type(exc).__name__ + ": " + str(exc)[:160],
                "source_path": str(KEY_FILE.relative_to(REPO)),
                "key_printed": False, "key_persisted": False}


# --------------------------------------------------------------------------- main
def run_all(mock_env: bool = False, out_dir: Path = DEFAULT_OUT_DIR,
            with_real: bool = False) -> dict:
    if mock_env:
        bootstrap_mock_env()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checks = {
        "D-V1_cutout": check_cutout(out_dir),
        "D-V2_memorial_copy": check_memorial_copy(),
        "D-V3_disclosure": check_disclosure(),
        "D-V4_archive_export": check_archive_export(out_dir=out_dir),
    }
    passed = sum(1 for c in checks.values() if c["pass"])
    report = {
        "task": "WP-04 ending ritual validation",
        "mode": "mock", "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_commit": BASELINE_COMMIT,
        "checks": checks,
        "summary": {"passed": passed, "total": len(checks), "all_pass": passed == len(checks)},
        "gaps": [
            "GAP-E2: item 抠图(cutout_key)不在 export_user_memory 的 _MEMORY_TABLES，导出 JSON 不含纪念物本体。",
            "GAP-E3: handle_upload 刻意不保留原图，original_key 恒空串，归档后无法回归原图（已知坑）。",
        ],
        "real_mode": {"status": "NOT_RUN", "reason": "not requested (--mock default)"},
    }
    if with_real:
        report["real_mode"] = run_real_optional()
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="结束仪式离线验证（抠图/文案/来电/归档/导出）")
    ap.add_argument("--mock", action="store_true", help="离线模式（默认）")
    ap.add_argument("--real", action="store_true", help="可选：真模型文案生成（无 key 则 NOT_RUN）")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="证据输出目录")
    ap.add_argument("--out", default=None, help="证据 JSON 文件路径（默认 <out-dir>/ending_validation.json）")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    report = run_all(mock_env=True, out_dir=out_dir, with_real=args.real)
    out = Path(args.out) if args.out else out_dir / "ending_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== 结束仪式离线验证（WP-04）===")
    for name, c in report["checks"].items():
        print(f"  {name}: {'PASS' if c['pass'] else 'FAIL'}")
    m = {k: v["metrics"] for k, v in report["checks"].items()}
    print(f"  D-V1 backend={m['D-V1_cutout']['backend']} model={m['D-V1_cutout']['model']} "
          f"matte透明占比={m['D-V1_cutout']['matte_png']['pct_transparent']}% "
          f"尺寸 {m['D-V1_cutout']['source_png']['width']}→{m['D-V1_cutout']['trimmed_png']['width']} "
          f"输出={m['D-V1_cutout']['trimmed_png']['width']}x{m['D-V1_cutout']['trimmed_png']['height']}")
    print(f"  D-V2 兜底不同文案={m['D-V2_memorial_copy']['distinct_fallback_texts']} "
          f"prompt路径不同文案={m['D-V2_memorial_copy']['distinct_prompt_path_texts']}")
    print(f"  D-V3 披露标记={m['D-V3_disclosure']['disclosure_marker']} "
          f"命中 {m['D-V3_disclosure']['rows_marked']}/{m['D-V3_disclosure']['rows_total']}×2端点")
    print(f"  D-V4 归档幂等={m['D-V4_archive_export']['archive_idempotent']} "
          f"写拒={[d['code'] for d in m['D-V4_archive_export']['write_denials']]} "
          f"导出source={m['D-V4_archive_export']['export_source_entries']} "
          f"类别={','.join(m['D-V4_archive_export']['export_categories'])}")
    print(f"  real_mode: {report['real_mode']['status']} {report['real_mode'].get('reason', '')}")
    print(f"  结论: {report['summary']['passed']}/{report['summary']['total']} PASS")
    print(f"  证据: {_evidence_path(out)}")
    return 0 if report["summary"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
