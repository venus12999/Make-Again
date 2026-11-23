/**
 * 验证 cloud-service 能连真后端（Node 环境模拟 wx）。
 * 用 http 模块把 wx.request 转成真实 HTTP 请求，跑完整流程。
 *
 * 用法：
 *   1) 后端先跑起来（python scripts/local_quickstart.py 或手动 uvicorn，端口 8000）
 *   2) node tests/cloud-service-e2e.js
 */
const http = require('http')
const path = require('path')
const Module = require('module')

// ---- 模拟微信小程序环境 ----
const memStore = {}
global.wx = {
  getStorageSync(k) { return memStore[k] },
  setStorageSync(k, v) { memStore[k] = v },
  request({ url, method, header, data, success, fail }) {
    const u = new URL(url)
    const body = data ? JSON.stringify(data) : null
    const req = http.request({
      hostname: u.hostname, port: u.port, path: u.pathname,
      method: method || 'GET',
      headers: { ...(header || {}), ...(body ? { 'content-length': Buffer.byteLength(body) } : {}) },
    }, res => {
      let buf = ''
      res.on('data', d => buf += d)
      res.on('end', () => {
        try { success({ statusCode: res.statusCode, data: buf ? JSON.parse(buf) : {} }) }
        catch (e) { fail(e) }
      })
    })
    req.on('error', fail)
    if (body) req.write(body)
    req.end()
  },
}

// 让 require('./config') 返回 CLOUD=true + 指定端口
const configPath = path.join(__dirname, '..', 'utils', 'config.js')
require.cache[configPath] = {
  id: configPath, filename: configPath, loaded: true,
  exports: { CLOUD: true, API_BASE: 'http://127.0.0.1:' + (process.env.PORT || 8000) },
}

const service = require(path.join(__dirname, '..', 'utils', 'cloud-service.js'))

async function main() {
  const results = []
  const step = (name, ok, detail) => { results.push({ name, ok, detail }); console.log(`  ${ok ? '✅' : '❌'} ${name}: ${detail}`) }

  console.log('=== cloud-service 连真后端端到端验证（Node 模拟 wx）===\n')
  try {
    const ov = await service.acceptConsent()
    step('acceptConsent + 后端注册/登录/consent', ov && ov.hasGuest === true, JSON.stringify(ov).slice(0, 120))

    service.startChat()
    const cs = service.getChatState()
    step('getChatState（前端访谈引导）', !!cs && Array.isArray(cs.messages), 'messages=' + (cs.messages || []).length)

    // 发一条消息，应该从后端拿到 AI 回复
    const before = service.listMessages().length
    const r = await service.sendChatMessage({ text: '我养的猫昨天走了，我很想它' }, 'e2e-1')
    const after = service.listMessages()
    const aiMsgs = after.filter(m => m.role === 'assistant' && String(m.id).startsWith('ai_'))
    step('sendChatMessage 拿到真实 AI 回复', r && r.role === 'user' && aiMsgs.length >= 1,
         'userMsg=' + (r ? r.id : '?') + ' aiMsgs=' + aiMsgs.length + ' 最后AI=' + (aiMsgs.length ? aiMsgs[aiMsgs.length-1].content.slice(0,50) : 'none'))
    step('消息写入本地流', after.length > before, before + ' -> ' + after.length)

    const allOk = results.every(x => x.ok)
    console.log(`\n总判定: ${allOk ? 'CLOUD_E2E_OK' : 'CLOUD_E2E_FAILED'}`)
    process.exit(allOk ? 0 : 1)
  } catch (e) {
    console.error('\n❌ 异常:', e && e.stack)
    process.exit(1)
  }
}

main()
