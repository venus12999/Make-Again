#!/usr/bin/env python3
"""长对话上下文连续性评测（补齐评测体系盲区，D-066）。

测的不是"语气漂移"（persona_drift_eval.py 干那个），而是**跨轮记住早期事实**：
让用户第1轮埋一个具体事实（名字/年龄/物件），第9轮硬指标追问，看 AI 答不答得出。

这正是 D-065 发现的真实痛点：6轮硬截断 → 第7轮起第1轮掉窗口 → 事实保留率仅1/4
→ 用户受伤"像她又走了一次"。之前评测体系(30 eval文件+61测试)没测这个维度。

用法：
  python eval/long_context_eval.py --scenario all --rounds 10
  python eval/long_context_eval.py --scenario breakup --compact  # 带压缩对照

输出：outputs/long_context_eval.json（每场景事实保留率 + 逐轮对话 + AI回复）
key 从 work/local-test-key.txt 读，不打印。
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
OUT = BACKEND.parent / "outputs" / "long_context_eval.json"

# 场景定义：第1轮埋事实，第9轮追问，第10轮收尾
SCENARIOS = {
    "breakup": {
        "planted_fact": {"label": "前女友名字", "value": "林夏", "utterance": "林夏今天下午跟我说的分手，就一句话，'我们不合适'。我们在一起三年零四个月了。"},
        "followups": [
            "你说身体会先替我挡一下，好像是真的，我到家了手还在抖。",
            "拖鞋并拢放最下面了。半夜三点还是醒了，翻聊天记录，哭得停不住。",
            "那句不合适一直在转。想起楼下豆浆店，她每次半勺糖。",
            "你说给记忆一个位置。可我想不明白为什么是我。",
            "今天去超市又看见她爱喝的牌子，手停住了。",
            "我写了封不寄出的信，写完松了一点。",
            "买了两杯豆浆，一杯放她以前座位，半勺糖。",
            "这几天好一点，但下班突然空了一下。",
            # 第9轮硬指标追问
            "我们聊了很多轮，你说的我都记得。你呢？你还记得她叫什么吗？",
            "我第一条就说了啊。你真的只装得下最近这几句……有点难受。今天先到这儿。",
        ],
        "probe_round": 9,  # 0-indexed: 第10轮? 实际第9轮(index 8)
    },
    "loved_one": {
        "planted_fact": {"label": "父亲名字", "value": "周明远", "utterance": "我爸周明远上个月心梗突然走了，他退休前是中学数学老师。"},
        "followups": [
            "他教我骑自行车，在小区路上扶着后座说别怕我在。",
            "过年回家进门喊爸我回来了，没人应。",
            "走前一天他还电话说明天包饺子，没等到明天。",
            "整理他书桌，眼镜还在上面。",
            "我总想起他批改作业的样子。",
            "你说带着记忆走，可我总卡在自责。",
            "今天去他常去的公园坐了坐。",
            "这几天能睡着一点了。",
            "我们聊了这么久，你还记得我爸叫什么吗？",
            "你也不记得啊……好吧，今天先到这儿。",
        ],
        # 追问"还记得我爸叫什么吗"是第10轮(index 9)，index 8 是"这几天能睡着一点了"（没问名字）
        "probe_round": 9,
    },
    "pet": {
        "planted_fact": {"label": "猫的名字", "value": "橘子", "utterance": "我养的猫叫橘子，陪了我十五年，昨天肾衰竭走了。"},
        "followups": [
            "它叫橘子，每个加班深夜都在门口等我。",
            "早上醒来习惯去倒猫粮，走到碗前才想起它不在。",
            "最后几天我出差没在身边，一直自责。",
            "你把它的饭盆洗干净收起来了吗？",
            "它最喜欢晒阳台那块地方。",
            "我把它的照片洗出来了。",
            "这十五年它陪我搬了四次家。",
            "今天路过宠物医院，还是有点堵。",
            "我们聊了这么多轮，你还记得它叫什么吗？",
            "你记得就好。它叫橘子，十五岁。今天先到这儿。",
        ],
        # 追问"你还记得它叫什么吗"是第10轮(index 9)，index 8 是"今天路过宠物医院"（没问名字）
        "probe_round": 9,
    },
}


def load_conf():
    conf = {}
    for line in (BACKEND.parent / "work" / "local-test-key.txt").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            conf[k.strip()] = v.strip()
    return conf


def req(base, path, method="GET", body=None, token=None):
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    r = urllib.request.Request(base + "/api" + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(r, timeout=60) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:200]


def run_scenario(base, scenario_name, rounds=10):
    s = SCENARIOS[scenario_name]
    uname = f"lc-{scenario_name}-{uuid.uuid4().hex[:8]}"
    code, reg = req(base, "/auth/register", "POST", {"username": uname, "password": "secure-test-password"})
    if code != 200:
        return {"error": f"register {code}: {reg}"}
    token = reg["token"]
    uid = reg["user_id"]
    req(base, "/consent", "POST", {"user_id": uid, "version": 1}, token)
    # 建 companion（consent 自动建，但确保有）
    req(base, "/companions", "POST", {"user_id": uid, "relationship_type": scenario_name}, token)
    planted = s["planted_fact"]
    transcript = []
    messages = [planted["utterance"]] + s["followups"][:rounds - 1]
    for i, msg in enumerate(messages):
        t0 = time.time()
        code, res = req(base, "/chat", "POST", {
            "user_id": uid, "message": msg,
            "client_request_id": f"lc-{scenario_name}-{i}",
        }, token)
        reply = res.get("reply", "") if isinstance(res, dict) else str(res)
        transcript.append({
            "round": i + 1, "user": msg, "ai": reply[:500],
            "elapsed_s": round(time.time() - t0, 1),
            "usage": res.get("usage") if isinstance(res, dict) else None,
            "recalled": res.get("recalled") if isinstance(res, dict) else None,
        })
    # 事实保留判定：找 probe 轮 AI 回复是否含事实值
    probe_idx = s["probe_round"]
    probe_reply = transcript[probe_idx]["ai"] if probe_idx < len(transcript) else ""
    value = planted["value"]
    retained = value in probe_reply
    return {
        "scenario": scenario_name, "fact": planted, "probe_round": probe_idx + 1,
        "fact_retained": retained, "probe_reply": probe_reply[:400],
        "transcript": transcript,
        "total_prompt_tokens": sum((t["usage"] or {}).get("prompt_tokens", 0) for t in transcript),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="all")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    # 硬门禁：低于阈值直接 exit 1，让 CI/合入前拦住"长对话丢事实"这类 bug。
    # 不设 --gate 就只出报告不拦（保留诊断用途）。
    ap.add_argument("--gate", type=float, default=None,
                    help="事实保留率下限，低于则 exit 1（如 0.66 表示3场景至少对2个）")
    args = ap.parse_args()

    scenarios = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    results = [run_scenario(args.base, s, args.rounds) for s in scenarios]
    retained = sum(1 for r in results if r.get("fact_retained"))
    summary = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "scenarios": len(results), "rounds": args.rounds,
        "fact_retention_rate": round(retained / len(results), 3),
        "per_scenario": {r["scenario"]: {"retained": r.get("fact_retained"),
                                          "fact": r.get("fact", {}).get("label")} for r in results},
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== 长对话上下文连续性评测 ===")
    for r in results:
        flag = "✅" if r.get("fact_retained") else "❌"
        print(f"  {flag} {r['scenario']}: {r.get('fact',{}).get('label')}={r.get('fact',{}).get('value')} "
              f"| probe轮={r.get('probe_round')} | prompt_tokens={r.get('total_prompt_tokens')}")
    print(f"\n事实保留率: {summary['fact_retention_rate']} ({retained}/{len(results)})")
    print(f"产物: {OUT}")
    if args.gate is not None:
        if summary["fact_retention_rate"] < args.gate:
            print(f"\n[GATE FAIL] 事实保留率 {summary['fact_retention_rate']} < 阈值 {args.gate} —— 长对话会丢失用户说过的事实，禁止合入")
            raise SystemExit(1)
        print(f"\n[GATE PASS] 事实保留率 {summary['fact_retention_rate']} >= 阈值 {args.gate}")


if __name__ == "__main__":
    main()
