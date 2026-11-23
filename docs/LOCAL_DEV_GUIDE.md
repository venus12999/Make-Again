# 本地开发 / 实测指南（队友可直接用）

> 目的：队友在自己电脑本地实测后端，不需要服务器、不需要真实 key、不需要微信/COS。
> 前端目前 100% 走 `local-service.js`（本地假数据），还没接真后端——见 `docs/FRONTEND_FIT_AUDIT.md`。

## 1. 一键验证后端可用（30 秒）

```bash
cd MakeAgain-backend
pip install -r requirements.txt        # 首次
python scripts/local_quickstart.py
```

脚本自动做：preflight 检查 → migrate 建独立临时库 → 启动 uvicorn → 验证
`health` / 未授权 401 / 注册 / consent / chat / state，全 mock，**不碰现有数据和凭据**。
输出 `outputs/local_quickstart.json`，总判定 `LOCAL_OK` 即全部可用。
实测输出（2026-11-23）：7 步全绿。

## 2. 手动启动后端（要自己调 API 时）

```bash
cd MakeAgain-backend
# 复制 .env.example 为 .env，保持 MOCK_*=true、ENVIRONMENT=development
# DATABASE_URL 用独立 sqlite 路径（别指向别人或生产库）
python preflight.py
python admin_cli.py migrate
uvicorn main:app --host 127.0.0.1 --port 8000
# 另开终端（后台记忆/周报 job 需要消费者）：
python admin_cli.py worker --forever --poll-seconds 3
```

**前提**：Python 3.12 + FFmpeg 在 PATH（MP3 解码需要）。

## 3. 前端契合度现状

- `utils/service.js` 当前只有 `module.exports = require('./local-service')`——前端完全走本地假后端。
- 后端 74 个 API 路由齐全，前端 service 契约 11 个方法基本都能映射（详见 `docs/FRONTEND_FIT_AUDIT.md`）。
- 前端要接真后端需要写 cloud adapter（TODO，需要队友配合，因为前端是其范围）。

## 4. 接手背景（本周工作定位）

- 11-07 接手时就是完整后端 125 文件（不是从零写）；我们基于它改造 61 文件 +10851/-781 行。
- 队友仓库 11-14 初始化（后端+小程序前端）、11-17 改 UI，之后零提交；11-18 至今全是我们的 PR #25-35。
- 本地优先，不碰服务器部署。
