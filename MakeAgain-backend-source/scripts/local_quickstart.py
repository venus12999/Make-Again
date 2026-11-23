#!/usr/bin/env python3
"""本地一键实测：队友在自己电脑跑这个脚本，验证后端核心链路可用。

做 5 件事（全用 mock + 独立临时 SQLite，不碰任何现有数据/凭据/服务器）：
1. preflight 离线检查
2. migrate 建库
3. 启动 uvicorn（子进程）
4. curl 健康检查 + 注册→consent→chat 核心流程
5. 关闭服务器，输出报告

用法：
  python scripts/local_quickstart.py
  python scripts/local_quickstart.py --port 8123
无需任何真实 key、无需微信/COS、无需服务器。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.error
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))


def http(method: str, url: str, body=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=0, help="0=自动选端口")
    args = ap.parse_args()

    import socket
    # 自动选空闲端口（--port 可指定）
    if args.port == 0:
        s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    else:
        port = args.port

    root = tempfile.mkdtemp(prefix="makeagain-local-")
    env = dict(os.environ)
    env.update(
        ENVIRONMENT="development",
        DATABASE_URL=f"sqlite:///{root}/local.sqlite3",
        MOCK_LLM="true", MOCK_WECHAT="true", MOCK_SPEECH="true",
        MOCK_VISION="true", MOCK_STORAGE="true",
        MEDIA_SIGNING_KEY="local-quickstart-key-0123456789012",
        STORAGE_LOCAL_DIR=f"{root}/media",
        PUBLIC_BASE_URL=f"http://127.0.0.1:{port}",
    )
    result = {"port": port, "db": f"{root}/local.sqlite3", "steps": [], "verdict": "INCOMPLETE"}

    def step(name, ok, detail=""):
        result["steps"].append({"name": name, "ok": ok, "detail": str(detail)[:200]})
        print(f"  {'✅' if ok else '❌'} {name}: {str(detail)[:160]}")

    print("=== Make Again 后端本地实测（全 mock，不碰真实凭据）===\n")
    # Step 1: preflight
    p = subprocess.run([sys.executable, "-c", "import preflight,json;print(json.dumps(preflight.check()))"],
                       cwd=str(BACKEND), env=env, capture_output=True, text=True, timeout=60)
    try:
        pf = json.loads(p.stdout.strip().splitlines()[-1])
        step("preflight 离线检查", pf.get("ok", False), pf.get("errors") or pf.get("checks"))
    except Exception as e:
        step("preflight 离线检查", False, p.stdout[-100:] + p.stderr[-100:] + str(e))

    # Step 2: migrate
    p = subprocess.run([sys.executable, "admin_cli.py", "migrate"], cwd=str(BACKEND), env=env,
                       capture_output=True, text=True, timeout=60)
    step("migrate 建库", p.returncode == 0, p.stdout.strip().splitlines()[-1] if p.stdout else p.stderr[-100:])

    # Step 3: 启动 API
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(BACKEND), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{port}"
    health_ok = False
    last_err = ""
    import select as _select
    try:
        for _ in range(60):  # 最多 30 秒（lifespan 迁移可能慢）
            if server.poll() is not None:
                # 进程已退出，读全部输出
                last_err = server.stdout.read().decode(errors="replace")[-600:]
                break
            try:
                code, body = http("GET", f"{base}/api/health", )
                if code == 200 and isinstance(body, dict) and body.get("status") == "ok":
                    health_ok = True
                    break
            except Exception as he:
                last_err = str(he)[:200]
            time.sleep(0.5)
    except Exception as e:
        last_err = str(e)[:200]
    step("API 启动 + health", health_ok,
         f"{base}/api/health" if health_ok else f"启动失败(poll={server.poll()}): {last_err}")
    if not health_ok:
        server.terminate()
        result["verdict"] = "FAILED_AT_STARTUP"
        return _write_out(result)

    try:
        # Step 4: 未授权访问应 401
        code, _ = http("GET", f"{base}/api/state")
        step("未授权 state 返回 401", code == 401, f"HTTP {code}")

        # Step 5: 注册→consent→chat
        code, reg = http("POST", f"{base}/api/auth/register",
                         {"username": "local-demo", "password": "secure-test-password"})
        step("注册", code == 200 and bool(reg.get("token")), f"user_id={reg.get('user_id','?')}")
        if code != 200:
            result["verdict"] = "FAILED"; return _write_out(result)
        h = {"Authorization": f"Bearer {reg['token']}"}

        code, consent = http("POST", f"{base}/api/consent",
                             {"user_id": reg["user_id"], "version": 1}, h)
        step("consent 授权", code == 200, consent)

        code, chat = http("POST", f"{base}/api/chat",
                          {"user_id": reg["user_id"], "message": "我养的猫昨天走了"}, h)
        reply = chat.get("reply", "") if isinstance(chat, dict) else ""
        step("chat 核心链路", code == 200 and bool(reply), f"reply={reply[:50] if reply else chat}")

        code, st = http("GET", f"{base}/api/state", headers=h)
        step("state 可读(带 token)", code == 200, f"HTTP {code}")

        all_ok = all(s["ok"] for s in result["steps"])
        result["verdict"] = "LOCAL_OK" if all_ok else "FAILED"
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except Exception:
            server.kill()

    _write_out(result)


def _write_out(result):
    out = BACKEND.parent / "outputs" / "local_quickstart.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n总判定: {result['verdict']}\n报告: {out}")
    return result


if __name__ == "__main__":
    main()
