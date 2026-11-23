# 前后端接线最终状态（2026-11-23）

> 用户指出单条 mock 验证不严谨 → 按真实用户路径重做 → 定位并修复 8 个根因 → **真 hy4 后端 14/14 全绿**。

## 1. 最终结果

`tests/user-journey-e2e.js` 按真实用户操作路径验收（打开→同意→建档4题→自由聊天→重开→读数据）：

- **mock 后端：14/14（13 通过 + 1 跳过：mock 模式不验模型质量）**
- **真 hy4 后端：14/14 全通过**

真 hy4 真实例子：用户问"我出差没陪在它身边"→ AI 真回复"出差不是你不爱橘子…自责很容易把最后几天放大成全部，可橘子拥有的，是你实实在在陪它的八年"。重开后后端历史 days=1、state/home 200——记忆跨会话持久。

## 2. 修复的 8 个根因（都是真问题，逐个看请求/响应定位的）

1. **guest.id ≠ 后端 user_id** → 403 OWNER_MISMATCH；修：注册后存 `BACKEND_UID`
2. **前端全同步调用**，async 版 cloud-service 让页面拿到 Promise 崩；修：全同步签名 + 后台队列
3. **建档最后一题触发 /chat** → 紧接自由聊天 CONVERSATION_BUSY；修：用发送前状态判断
4. **/api/chat 并发保护** → 串行队列 `chatQueue`
5. **POST /companions 多余**（consent 会自动建 companion）→ 409 ACTIVE_COMPANION_EXISTS 让 ready 链误判；修：去掉该调用
6. **getOverview 在 guest 未建时缓存了 ensureAccount=false** → 之后永远拿不到 token；修：只在 TOKEN 存在时算 ready，失败不缓存
7. **AI 消息 ownerId 用了后端 uid** → local.listMessages 按本地 guest.id 过滤，AI 消息被过滤掉；修：用本地 guest.id
8. **快照在 companion 就绪前拉 home**（409 COMPANION_REQUIRED）→ 缓存空快照；修：等 companionReady 后才拉，只缓存成功结果

## 3. 接线架构（最终）

- `utils/cloud-service.js`：11 方法同步签名，页面零改动；建档阶段本地 UX，自由聊天走真后端 /api/chat（真 AI + 记忆回写 + 每日心情），读取走 state/home/history
- `utils/config.js`：CLOUD 开关 + API_BASE
- `utils/service.js`：按开关切 local/cloud
- `pages/chat/chat.js`：`_setRefresh` 回调（AI 回复异步到达自动刷新，仅 2 行）
- `tests/user-journey-e2e.js`：完整用户路径验收脚本

## 4. 诚实边界

- Node 模拟 `wx.request`（http 转发），不是真微信开发者工具——**UI 渲染/wxml 仍需队友在开发者工具验证**
- 验证的是"用户操作路径的数据流"，后端契约/安全模型（幂等/串行/companion）已完全理顺
- 云端部署/支付/真机订阅消息——按约定不管

## 5. 队友实测步骤

1. 后端带真 key 起（见 `docs/LOCAL_DEV_GUIDE.md`）
2. `utils/config.js` 设 `CLOUD: true`
3. 微信开发者工具关域名校验，打开小程序——从使用者角度体验，AI 回复是真的

冻结点 `v0.1-before-frontend-integration` 保留（接线前状态可对照）。
