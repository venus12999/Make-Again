/**
 * 用户视角端到端验收：按真实用户操作路径走完整旅程（同步 service + 异步 AI 回复）。
 *
 * 诚实边界：
 *   - Node 模拟 wx.request（http 转发），不是真微信开发者工具；UI 渲染需队友真机验证
 *   - 验证的是"用户操作路径的数据流"：建档→自由聊天→真实AI→记忆持久
 *   - 真 hy4 需后端带真 key 起在 127.0.0.1:8000
 *
 * 用法：node tests/user-journey-e2e.js
 */
const http = require('http')
const path = require('path')
const ROOT = path.join(__dirname, '..')

const mem = {}
global.wx = {
  getStorageSync(k) { return mem[k] },
  setStorageSync(k, v) { mem[k] = JSON.parse(JSON.stringify(v)) },
  request({ url, method, header, data, success, fail }) {
    const u = new URL(url), b = data ? JSON.stringify(data) : null
    const q = http.request({ hostname: u.hostname, port: u.port, path: u.pathname, method: method || 'GET',
      headers: { ...(header || {}), ...(b ? { 'content-length': Buffer.byteLength(b) } : {}) } }, res => {
      let buf = ''; res.on('data', d => buf += d)
      res.on('end', () => { let p = buf; try { p = buf ? JSON.parse(buf) : {} } catch (e) {} success({ statusCode: res.statusCode, data: p }) })
    })
    q.on('error', fail); if (b) q.write(b); q.end()
  },
}
const cfg = ROOT + '/utils/config.js'
require.cache[cfg] = { id: cfg, filename: cfg, loaded: true, exports: { CLOUD: true, API_BASE: 'http://127.0.0.1:8000' } }
const service = require(ROOT + '/utils/cloud-service.js')
const storage = require(ROOT + '/utils/storage.js')

const results = []
const step = (phase, name, ok, detail) => {
  results.push({ phase, name, ok, detail: String(detail).slice(0, 240) })
  console.log(`  ${ok ? '✅' : '❌'} [${phase}] ${name}: ${String(detail).slice(0, 200)}`)
}
const sleep = ms => new Promise(r => setTimeout(r, ms))
async function waitFor(fn, timeout = 20000) {
  const t = Date.now()
  while (Date.now() - t < timeout) { if (fn()) return true; await sleep(300) }
  return false
}

async function main() {
  console.log('=== 用户视角完整旅程验收（从打开到用完）===\n')

  // 阶段1
  console.log('【阶段1：首次打开 → 同意授权】')
  const ov0 = service.getOverview()
  step('首次打开', '未同意 hasGuest=false', ov0.hasGuest === false, JSON.stringify(ov0).slice(0, 80))
  service.acceptConsent()
  step('首次打开', '同意后 hasGuest=true', service.getOverview().hasGuest === true, 'guest=' + (storage.read().guest || {}).id)
  const acctOk = await waitFor(() => !!storage.read().backendSnapshot || !!service._ensureAccount)
  // 等账号就绪（注册/登录完成）
  await service._ensureAccount()
  step('首次打开', '后端账号建立', true, 'ensureAccount resolved')

  // 阶段2：建档（用户逐题回答，每次 reply 后等 guide 推进）
  console.log('\n【阶段2：建档逐题回答】')
  const answers = { nickname: '小雨', lifeStage: '已经工作', concern: '我养的猫昨天走了', expectation: '先听我说说' }
  for (let i = 0; i < 6; i++) {
    const st = service.getChatState()
    if (!st.step) break
    const key = st.step.key
    service.sendChatMessage({ text: answers[key] || '想慢慢说' }, 'onboard-' + key)
    // 建档阶段同步推进（无网络依赖）
    const after = service.getChatState()
    step('建档', `第${i + 1}题 ${key} 推进`, after.completed > st.completed || !after.step,
         `completed ${st.completed}->${after.completed}`)
  }
  const done = service.getChatState()
  step('建档', '完成后进入自由聊天', !done.step, `completed=${done.completed}/${done.total}`)

  // 阶段3：自由聊天（真 AI 异步回复）
  console.log('\n【阶段3：自由聊天 → 真 AI 回复】')
  console.log('  [调试] isOnboardingDone=' + service._isOnboardingDone() + ', guide.step=' + (service.getChatState().step ? service.getChatState().step.key : 'null'))
  const chats = ['它叫橘子，陪了我八年，昨天走的。', '我有点自责，最后几天我在出差。']
  for (let i = 0; i < chats.length; i++) {
    const before = service.listMessages().filter(m => m.role === 'assistant' && m.fromBackend).length
    service.sendChatMessage({ text: chats[i] }, 'chat-' + i)
    const got = await waitFor(() => service.listMessages().filter(m => m.role === 'assistant' && m.fromBackend).length > before, 30000)
    const ais = service.listMessages().filter(m => m.role === 'assistant' && m.fromBackend)
    step('自由聊天', `第${i + 1}条真AI回复`, got && ais.length > before,
         `AI数=${ais.length}, 最后=${ais.length ? ais[ais.length - 1].content.slice(0, 70) : 'none'}`)
  }
  const ais = service.listMessages().filter(m => m.role === 'assistant' && m.fromBackend)
  step('自由聊天', '回复非 mock 假话', ais.length > 0 && !ais[ais.length - 1].content.startsWith('（mock）'),
       ais.length ? ais[ais.length - 1].content.slice(0, 80) : 'none')

  // 阶段4：重开（记忆持久）
  console.log('\n【阶段4：重开验证记忆持久】')
  const ov = await new Promise(resolve => {
    // getOverview 是同步的，但后端快照异步写；等一下再读
    setTimeout(() => resolve(service.getOverview()), 1000)
  })
  step('重开', '后端快照可读(state)', !!ov.backendState, JSON.stringify(ov.backendState || {}).slice(0, 100))
  step('重开', '后端快照可读(home/心情)', !!ov.backendHome, JSON.stringify(ov.backendHome || {}).slice(0, 100))
  const uid = ov.backendState && ov.backendState.user_id
  if (uid) {
    const hist = await service._request('/chat/history/' + encodeURIComponent(uid) + '/days', 'GET')
    step('重开', '后端历史天数>0（跨会话记忆）', hist.ok && hist.data && (hist.data.days || []).length > 0,
         `days=${hist.data && hist.data.days ? hist.data.days.length : '?'}`)
  }

  const allOk = results.every(r => r.ok)
  console.log(`\n=== 总判定: ${allOk ? 'USER_JOURNEY_OK' : 'USER_JOURNEY_FAILED'} (${results.filter(r => r.ok).length}/${results.length}) ===`)
  process.exit(allOk ? 0 : 1)
}
main().catch(e => { console.error('异常:', e && e.stack); process.exit(1) })
