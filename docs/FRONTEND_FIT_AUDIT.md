# 前后端契合度审计（2026-11-23）

> 起因：用户要"本地可用，队友能在自己电脑实测"，并查契合度。不碰服务器部署。
> 方法：读队友仓库前端 service 契约 + 主仓后端 API 路由，逐项对照。

## 1. 核心结论

**前端 100% 走本地假后端，从没接过真后端。**
- `utils/service.js` 只有 2 行：`module.exports = require('./local-service')`
- `utils/local-service.js`（146行）是纯本地 storage 模拟，无 fetch、无 /api/
- 前端 11-17 后再无更新（队友仅 2 个提交：11-14 初始化、11-17 UI）

**所以"契合度"不是接口对不上，是根本没连接。** 后端 74 个 API 路由齐全，前端 11 个 service 方法都能映射。

## 2. 前端 service 契约（11 方法）→ 后端映射

| 前端方法 | 后端 API | 契合 |
|---|---|---|
| acceptConsent | POST /api/consent | ✅ |
| getOverview | GET /api/state + /api/home/{uid} + /api/initial-report | ✅ |
| getChatState | GET /api/state + /api/chat/history | ✅ |
| startChat | POST /api/interview/start 或 /api/companions | ⚠️ 语义需对齐 |
| sendChatMessage | POST /api/chat | ✅ |
| listMessages | GET /api/chat/history/{uid}/days | ✅ |
| saveMessage | （后端由 /api/chat 自动落库，无需单独方法） | ⚠️ 契约差异 |
| getProfile | GET /api/state + /api/mood | ⚠️ 字段映射 |
| saveProfile | PATCH /api/state | ✅ |
| listCollection | GET /api/item + /api/photo + /api/diary | ✅ |
| markLetterRead | POST /api/treehole/letter 相关 | ⚠️ 需确认 |

## 3. 让队友本地实测的方案（不碰服务器）

1. 新建 `utils/cloud-service.js`：实现同样 11 个方法，内部 fetch 真后端 `http://127.0.0.1:8000/api/...`
2. `service.js` 改为按配置切换：`LOCAL_SERVICE_MODE=cloud` 或读一个 `utils/config.js` 开关，默认仍 local（不破坏现有）
3. 后端本地启动：`cd MakeAgain-backend && uvicorn main:app --port 8000`（有 .env.example）
4. 队友在自己电脑跑后端 + 小程序指向本地后端，即可实测

## 4. 诚实边界
- 小程序是微信生态（wx.request），本地实测需要开发者工具或真机，不是纯浏览器——"本地实测"对前端有环境要求
- cloud adapter 的鉴权：后端有 auth（register/login/Bearer token），前端 local 版是 guest 无鉴权——adapter 要处理登录/token
- 这是"契出适配层"，不是"改产品核心"；本地可做

## 5. 接手背景（回答用户问题）
- 主仓 11-07 接手时就是完整后端 125 文件（不是从零写）
- 队友仓库 11-14 初始化时是同样后端 + 小程序前端
- 队友仅 2 个提交（11-14 初始化、11-17 UI），之后零提交；11-18 至今全是我们的 PR #25-35
- 我们基于接手后端改造 61 文件 +10851/-781 行（记忆重写、评测体系等）
