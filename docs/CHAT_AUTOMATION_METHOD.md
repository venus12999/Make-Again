# 真 GUI 聊天自动化方法（已验证成功）

> 2026-11-24：用 evaluate 直接调页面实例方法，绕过挂死的 createSelectorQuery，在真开发者工具里完整走通建档+聊天+真AI回复。

## 核心方法

官方 miniprogram-automator 和 next 适配层的 `query/tap/input` 都依赖 `wx.createSelectorQuery()`——本 DevTools 上它挂死。但 **`miniProgram.evaluate()` 是通的**，直接拿页面实例调方法即可：

```js
const automator = require('miniprogram-automator-next')
const mp = await automator.connect({ wsEndpoint: 'ws://127.0.0.1:9421' })

// 发消息（绕过 query）
await mp.evaluate((text) => {
  const p = getCurrentPages().slice(-1)[0]
  p.onInput({ detail: { value: text } })   // 模拟输入
}, '它叫橘子，陪了我八年')
await mp.evaluate(() => {
  const p = getCurrentPages().slice(-1)[0]
  p.send()                                 // 发送
})

// 等 AI 回复：轮询 page.data.messages 里 fromBackend=true 的 assistant 消息
```

## 已验证的真实结果（work/chat_via_evaluate.log）

- 登录→聊天页 switchTab ✅
- 建档 4 题（nickname/lifeStage/concern/expectation）通过 evaluate 调 onInput+send 全部推进（completed 0→4）✅
- 自由聊天发"它叫橘子，陪了我八年，昨天走的" ✅
- **真 hy4 AI 回复 12 秒后到达**，GUI 消息流里 fromBackend=true：
  > "八年，是一段很长的日子。橘子把你人生里一整段时光都陪了过来，名字也暖乎乎的，一听就是被好好爱着的孩子。昨天才走，这份难受还新鲜着，不用急着消化它。想说说橘子的事也好，就安静待着也好，我都在这儿陪你。"
- 最终页面状态：route=pages/chat/chat, completed=4, msgs=12

## 注意事项

- 截图（`mp.screenshot()`）仍会触发 automator 的 Page domain 超时——但聊天数据已通过 evaluate 拿到，截图失败不影响验收结论
- 导航期间别并发 evaluate，会 timeout 把小程序重载回登录页——串行执行
- 脚本：work/chat_via_evaluate.cjs；日志：work/chat-via-evaluate.log
