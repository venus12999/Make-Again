# 前端实测交接（adapter 已写好，直接体验）

## 一句话

**前后端已经接好了（我写的 `utils/cloud-service.js`），你不用再写适配层——打开开关就能从产品使用者角度真实体验。**

冻结点：`v0.1-before-frontend-integration`（接线前的状态，可随时对照）。

## 你要做的最小步骤（3 步）

### 1. 启动后端
```bash
cd MakeAgain-backend-source
python scripts/local_quickstart.py     # 一键验证后端可用（30秒，LOCAL_OK）
# 要真实 AI 体验的话，用真 key 手动起（见 docs/LOCAL_DEV_GUIDE.md）
```

### 2. 打开前端云开关
编辑 `utils/config.js`：
```js
module.exports = { CLOUD: true, API_BASE: 'http://127.0.0.1:8000' }
```
- 微信开发者工具里跑 → `127.0.0.1:8000`
- 真机跑 → `http://<你电脑局域网IP>:8000`
- 微信开发者工具 → 详情 → 本地设置 → 勾「不校验合法域名」

### 3. 实测
在微信开发者工具里打开小程序 → 同意 → 建档 → 聊天。AI 回复是**真后端真实生成的**，不再是本地假话。

## 我做了什么（你不用做）

- `utils/cloud-service.js`：实现了全部 11 个 service 方法，关键节点（consent/chat）调真后端；保留前端数据模型，页面零改动
- `utils/config.js`：云开关 + API 地址
- `utils/service.js`：改成按开关切 local/cloud
- `tests/cloud-service-e2e.js`：Node 模拟 wx 的端到端验证脚本

**已验证**：mock 后端和真 hy4 后端都跑通 `CLOUD_E2E_OK`——前端发消息，真 AI 回复显示在消息流里。

## 接线踩过的坑（已修，记给你知道）

1. **前端 guest.id ≠ 后端 user_id**——混用会 403 OWNER_MISMATCH，adapter 里注册后保存后端真实 uid 用
2. 微信开发者工具必须关域名校验才能连本机 IP
3. 连真后端要 token——adapter 自动注册/登录拿 token，你不用管

## 边界

- 云端部署/支付/微信真机订阅消息——先不管，只做本地连本机后端
- 真 AI 体验需要真 key（wire 在后端），mock 模式下回复是假的
- 前端代码我动了你不用重复做；有问题找我
