#!/usr/bin/env python3
"""体验走查 WP-01：3 个不同用户走完 访谈→双画像→报告，看报告是否真个性化。

用同步报告（interview.answer async_report=False），报告当场生成可拿到真实内容。
真实 hy4，隔离 DB。key 不打印不入库。

用法：python eval/walkthrough_journey.py [--limit 1]  # --limit 只跑前N个用户(调试用)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
sys.path.insert(0, str(BACKEND))
OUT = BACKEND.parent / "outputs" / "walkthrough"
OUT.mkdir(parents=True, exist_ok=True)


def load_conf():
    conf = {}
    for line in (BACKEND.parent / "work" / "local-test-key.txt").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            conf[k.strip()] = v.strip()
    return conf


# 3 个背景不同的模拟用户（synthetic-persona 手工构造，非真实用户）
PERSONAS = [
    {"tag": "分手-被出轨", "loss_type": "breakup", "answers": [
        "她叫小雅,我们在一起三年,是大学同学。她出轨了,是被我发现的。",
        "最常想起的是去年跨年,我们在江边倒计时,她说要到老。现在那句话像刀子。",
        "走神最多是下班回到空房间,以前她会发消息问'到家了吗'。现在手机再也不响。",
        "放不下的是'为什么是我'。我把能给的都给了,还是被丢下。",
        "我想带着这段记忆往前走,但不想再信任何人。",
    ]},
    {"tag": "丧亲-父亲突然离世", "loss_type": "loved_one", "answers": [
        "我爸,他叫建国,退休前是中学数学老师。去年心梗走的,很突然。",
        "反复想起他教我骑自行车,在小区那条路上扶着后座,说'别怕,我在'。",
        "最难受是过年回家,进门喊'爸我回来了',没人应。",
        "想不通的是他走前一天我们还通电话,他说明天包饺子,结果没等到明天。",
        "我想带着他对我的好继续过,把他教我的耐心传下去。",
    ]},
    {"tag": "宠物离世-陪伴15年的猫", "loss_type": "pet", "answers": [
        "是只橘猫,叫橘子,陪伴我十五年,从大学毕业到现在。上周走了,肾衰竭。",
        "最想起的是每个加班深夜回家,它蹲在门口等我,尾巴甩啊甩。",
        "最难的是早上醒来,习惯性要去倒猫粮,走到碗前才想起它不在了。",
        "放不下的是最后那几天,我出差没在身边,它是自己走的。我一直自责。",
        "我想记住它,不想快点忘记。它是我这十五年最稳的陪伴。",
    ]},
]


def walk_one(p, db, client):
    """走完一个 persona 的旅程，返回记录。"""
    from agent import interview
    from memory import store

    rec = {"tag": p["tag"], "loss_type": p["loss_type"]}
    uid = "walk-" + uuid.uuid4().hex[:8]
    store.get_or_create_user(db, uid)
    t0 = time.time()
    st = interview.start(db, uid, p["loss_type"])
    sid = st["session_id"]
    rec["session_id"] = sid
    print(f"  start: {sid[:12]}")

    turns = []
    answers = list(p["answers"])
    gen_filler = 0
    for turn in range(22):  # 超过 MAX_TURNS=20，让强制结束生效
        sess = store.get_interview(db, sid)
        state = sess.state or {}
        if state.get("done"):
            break
        qs = [h.get("content", "") for h in state.get("history", []) if h.get("role") == "assistant"]
        question = qs[-1] if qs else ""
        if answers:
            ans = answers.pop(0)
        else:
            gen_filler += 1
            ans = "嗯,我补充一点:这件事我其实一直没跟别人细说,现在跟你讲了,心里稍微松一点。我还在慢慢消化,也不知道接下来会怎样。"
        resp = interview.answer(db, sid, ans, client, async_report=False)
        turns.append({"turn": turn + 1, "question": question[:120], "answer": ans[:80],
                      "action": resp.get("action"), "done": resp.get("done")})
        print(f"  turn{turn+1}: action={resp.get('action')} done={resp.get('done')} q={question[:40]}")
        if resp.get("action") == "complete":
            rec["report"] = resp.get("report")
            break

    sess = store.get_interview(db, sid)
    state = sess.state or {}
    rec["n_turns"] = len(turns)
    rec["gen_filler_turns"] = gen_filler
    rec["done"] = bool(state.get("done"))
    rec["report_ready"] = bool(state.get("report_ready"))
    rec["elapsed_s"] = round(time.time() - t0, 1)
    rec["turns"] = turns
    if not rec.get("report"):
        rec["report"] = state.get("report")
    print(f"  结果: done={rec['done']} report_ready={rec['report_ready']} turns={len(turns)} ({rec['elapsed_s']}s)")
    if rec.get("report"):
        print(f"  报告预览: {json.dumps(rec['report'], ensure_ascii=False)[:400]}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前N个用户(调试)")
    args = ap.parse_args()

    conf = load_conf()
    os.environ.update(
        LLM_BASE_URL=conf["BASE_URL"], LLM_API_KEY=conf["KEY"],
        LLM_FAST_MODEL="hy4-flash-free", LLM_MODEL="hy4",
        MOCK_LLM="false", ENVIRONMENT="development",
        LLM_EXTRA_HEADERS=json.dumps({
            "x-opencode-session": "walkthrough-" + uuid.uuid4().hex[:8],
            "User-Agent": "make-again-walkthrough/1.0",
        }),
        DATABASE_URL="sqlite:///" + tempfile.mkdtemp(prefix="walkthrough-") + "/t.sqlite3",
        MEDIA_SIGNING_KEY="x" * 40,
        STORAGE_LOCAL_DIR=tempfile.mkdtemp(prefix="walkthrough-media-"),
        PUBLIC_BASE_URL="http://testserver",
    )
    from config import get_settings
    get_settings.cache_clear()  # .env 里的 deepseek-chat 已进 lru_cache，清掉让 env 生效
    s = get_settings()
    # 显式覆盖（.env 优先级问题：直接改 settings 字段最稳）
    s.llm_base_url = conf["BASE_URL"]
    s.llm_api_key = conf["KEY"]
    s.llm_fast_model = "hy4-flash-free"
    s.llm_model = "hy4"
    s.llm_report_model = ""  # 空=用 llm_fast_model
    s.mock_llm = False
    # extra headers 由 gateway.client 从 LLM_EXTRA_HEADERS env 读取（前面已 setenv）

    from memory.db import init_db, _RawSessionLocal
    from memory.scope import current_db
    init_db()
    from gateway.client import LLMClient
    client = LLMClient()
    print(f"[config] model={client.settings.llm_fast_model} base={client.settings.llm_base_url[:40]}")

    personas = PERSONAS[:args.limit] if args.limit else PERSONAS
    results = []
    db = _RawSessionLocal()
    db.info.update(managed=True, borrowed=True)
    token = current_db.set(db)
    try:
        for p in personas:
            print(f"\n=== 走查用户: {p['tag']} ===")
            try:
                results.append(walk_one(p, db, client))
            except Exception as e:
                print(f"  错误: {str(e)[:150]}")
                results.append({"tag": p["tag"], "error": str(e)[:200]})
    finally:
        current_db.reset(token)
        db.close()

    out = OUT / "journey_walkthrough.json"
    out.write_text(json.dumps({"note": "synthetic-persona 手工构造非真实用户；报告为真实hy4同步生成", "results": results},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[walkthrough] done -> {out}")


if __name__ == "__main__":
    main()
