# 前端同学交接说明（可直接转给他）

## 一句话现状

**前端现在 100% 用本地假数据（`local-service.js`），从没连过真后端。** 后端已经写好了、能本地跑、全部验证过，就差前端这边接上。

## 你要做的（4 步）

1. **看文档**：`docs/LOCAL_DEV_GUIDE.md`（后端怎么本地跑）+ `docs/FRONTEND_FIT_AUDIT.md`（接口对应情况，注意：里面标了哪些是查证的、哪些需要你逐个对，别直接信"都能映射"）。
2. **跑后端**：在你电脑 `cd MakeAgain-backend && python scripts/local_quickstart.py`，30 秒出 `LOCAL_OK` 就是后端可用（全 mock，不需要真实 key/服务器）。要自己调 API 就按 LOCAL_DEV_GUIDE 手动起 uvicorn + worker。
3. **写 cloud-service.js**：照抄 `utils/local-service.js` 的 11 个方法，把每个方法内部改成发网络请求到本机后端 `http://127.0.0.1:8000/api/...`（用 `wx.request`）。11 个方法：`acceptConsent / getOverview / markLetterRead / listMessages / saveMessage / getProfile / saveProfile / listCollection / startChat / getChatState / sendChatMessage`。
4. **改 service.js 开关**：现在是写死的 `module.exports = require('./local-service')`，改成可切换 local/cloud。微信开发者工具勾"不校验合法域名"才能连本机 IP。

## 你要交付的

- `utils/cloud-service.js`（11 个方法连真后端）
- `service.js` 开关改造
- 微信开发者工具里连本机后端的实测截图/结果

## 两个提醒（避免踩坑）

1. **登录**：真后端要 token（register/login 发 Bearer token），假后端不用——cloud adapter 里要先注册/登录拿 token，之后每个请求带上。
2. **接口逐个对，别猜**：后端有 74 个路由，但前端 11 方法不是一一同名。`FRONTEND_FIT_AUDIT.md` 里只有 startChat 是查证过的，其余要你拿方法签名和后端路由逐个对（比如 sendChatMessage→`/api/chat`、acceptConsent→`/api/consent`）。对不上就告诉我，后端可以加/改接口。

## 边界

- 云端部署、支付、微信真机订阅消息——先都不管，只做本地连本机后端。
- 你改前端代码，我不碰（前端是你的范围）。对接口有问题随时找我。
