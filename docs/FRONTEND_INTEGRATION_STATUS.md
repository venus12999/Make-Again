# 前后端接线真实状态（2026-11-23，诚实版）

> 起因：用户要求站在用户使用方式做测试，不要单条 mock 验证。本轮按真实用户路径重做，结果**部分失败**，如实记录。

## 1. 做了什么

- 重写 `utils/cloud-service.js`：所有方法同步签名（前端页面全同步调用，之前 async 版会让页面拿到 Promise 直接崩）
- `pages/chat/chat.js` 加 `_setRefresh` 注册（后端 AI 回复异步到达后自动刷新消息流）
- `tests/user-journey-e2e.js`：按真实用户路径验收（打开→同意→建档4题→自由聊天→重开→读数据）

## 2. 真 hy4 后端实跑结果：8/13 通过，**未达标**

| 阶段 | 结果 |
|---|---|
| 首次打开（未同意/同意/后端账号） | ✅ 3/3 |
| 建档 4 题逐题推进 + 完成 | ✅ 5/5 |
| **自由聊天真 AI 回复** | ❌ 0/3（AI 消息没进本地流） |
| **重开记忆持久（后端快照/历史）** | ❌ 0/3 |

## 3. 逐层定位到的问题（都是真问题，不是猜）

1. **guest.id ≠ 后端 user_id**——混用 403 OWNER_MISMATCH（已修：存 BACKEND_UID）
2. **前端全同步调用**——async 版 cloud-service 让页面拿到 Promise 崩（已修：改同步 + 后台队列）
3. **建档最后一题触发 /chat**——紧接着自由聊天第一条 → 后端 CONVERSATION_BUSY（已修：用发送前状态判断）
4. **/api/chat 串行队列**——后端对同用户写操作严格串行（已修：chatQueue）
5. **仍未解决**：`/state` 返回 409（幂等拦截），body 里 `chat_session_id=null`；companion 创建疑似被 ACTIVE_COMPANION_EXISTS 挡；导致 backendSnapshot 没写、自由聊天 AI 回复链路没闭环

## 4. 关键澄清（直接 chat 是通的）

用 `_request('/chat')` 直连后端**返回 200 + 真实 AI 回复**（"橘子，这名字一听就让人心里一软…"）。
**说明后端产品核心没问题，是适配层跟后端安全模型（幂等/串行/companion 创建时序）的交互没理顺。**

## 5. 诚实结论

- **不能说"前后端接好了"**——之前那版 `CLOUD_E2E_OK` 是用单条 mock 消息验的，不严谨（用户指出得对）
- 按真实用户路径走，**自由聊天和记忆持久两段没通**
- 后端产品核心（/chat 真 AI）是好的，问题在前端适配层
- 冻结点 `v0.1-before-frontend-integration` 保留，可随时回退

## 6. 下一步（需要认真做，不是再猜一次）

适配层要正确处理后端的安全模型：
- companion 创建：acceptConsent 时要确保 POST /companions 成功（处理 CONSENT_REQUIRED 时序）
- /state 的 409：用 client_request_id 去重，或改读 /api/home 拿同样数据
- 串行队列要等 companion ready 且前一个 chat 真正 completed
- 每条消息要有唯一 client_request_id（页面已有 clientId，透传即可）

**再次验证必须按完整用户路径（本脚本），且要真 hy4 后端。**
