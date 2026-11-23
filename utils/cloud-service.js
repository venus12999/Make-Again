/**
 * cloud-service —— 后端当数据源的适配层（同步签名，页面零改动）。
 *
 * 关键约束：前端页面全部同步调用 service 方法（无 await），所以本模块所有方法
 * 必须同步返回。网络请求在后台异步进行，结果写回 storage；页面 refresh() 重拉可见。
 *
 * 分层：
 *   - 建档阶段（chat-guide 有 step）：前端 UX 本地建档（6字段，后端无对应概念）
 *   - 自由聊天：sendChatMessage 同步落本地 + 后台调 /api/chat，真实 AI 回复异步写 storage
 *   - 读取：getChatState/listMessages 从 storage 读（含异步补进来的后端消息）
 *   - getOverview/saveProfile/listCollection：本地为准 + 后台同步后端
 *
 * 配置：utils/config.js（CLOUD / API_BASE）。微信开发者工具需关域名校验。
 */
const local = require('./local-service')
const storage = require('./storage')
const { LOSS_TYPES } = require('./constants')
const { getGuide } = require('./chat-guide')
const config = require('./config')

const API_BASE = (config && config.API_BASE) || 'http://127.0.0.1:8000'
let TOKEN = null
let BACKEND_UID = null
let accountReady = null   // Promise：确保只注册/登录一次
let companionReady = null // Promise：consent + companion 建好，自由聊天前必须等它
let chatQueue = Promise.resolve()  // /api/chat 串行队列（后端有 CONVERSATION_BUSY 并发保护）

function request(path, method, body) {
  return new Promise((resolve) => {
    const header = { 'content-type': 'application/json' }
    if (TOKEN) header.Authorization = 'Bearer ' + TOKEN
    wx.request({
      url: API_BASE + '/api' + path, method: method || 'GET', header, data: body,
      success(res) {
        if (res.statusCode >= 200 && res.statusCode < 300) resolve({ ok: true, data: res.data, status: res.statusCode })
        else resolve({ ok: false, data: res.data, status: res.statusCode })
      },
      fail(err) { resolve({ ok: false, data: err, status: 0 }) },
    })
  })
}

function ensureAccount() {
  if (!accountReady) {
    accountReady = (async () => {
      const state = storage.read()
      if (!state.guest) return false
      const username = 'ma_' + state.guest.id
      const password = 'ma-local-password'
      let r = await request('/auth/register', 'POST', { username, password })
      if (!r.ok) r = await request('/auth/login', 'POST', { username, password })
      if (!r.ok || !r.data || !r.data.token) {
        console.warn('[cloud] 后端登录失败，回退本地模式', r.status)
        return false
      }
      TOKEN = r.data.token
      BACKEND_UID = r.data.user_id
      return true
    })()
  }
  return accountReady
}

function toBackendLossType(v) { return { relative: 'loved_one', friend: 'breakup', '': 'breakup' }[v] || v }
function isOnboardingDone() { try { return !getGuide(local.getProfile()).step } catch (e) { return false } }

// ---- 同步方法（页面直接调）----

function acceptConsent() {
  const overview = local.acceptConsent()
  companionReady = ensureAccount().then(ok => {
    if (!ok) return false
    const state = storage.read()
    const j = state.journeys.find(x => x.id === state.currentJourneyId)
    const lt = toBackendLossType(j && j.profile && j.profile.lossType ? j.profile.lossType.value : '')
    return request('/consent', 'POST', { user_id: BACKEND_UID, version: 1 })
      .then(() => request('/companions', 'POST', { user_id: BACKEND_UID, relationship_type: lt }))
      .then(r => {
        // 409 ACTIVE_COMPANION_EXISTS 也算就绪（companion 已在）
        if (!r.ok && r.status !== 409) console.warn('[cloud] companion 创建失败', r.status, r.data)
        return true
      })
  }).catch(e => { console.warn('[cloud] consent 同步失败', e); return false })
  return overview
}

function getOverview() {
  const ov = local.getOverview()
  ensureAccount().then(ok => {
    if (!ok || !ov.hasGuest) return
    Promise.all([
      request('/state', 'GET'),
      request('/home/' + encodeURIComponent(BACKEND_UID), 'GET'),
    ]).then(([st, home]) => {
      ov.backendState = st.ok ? st.data : null
      ov.backendHome = home.ok ? home.data : null
      storage.update(s => { s.backendSnapshot = { state: ov.backendState, home: ov.backendHome, at: Date.now() } })
    }).catch(() => {})
  })
  // 若之前已缓存快照，直接带上（同步可见）
  const snap = storage.read().backendSnapshot
  if (snap) { ov.backendState = snap.state; ov.backendHome = snap.home }
  return ov
}

function markLetterRead() { return local.markLetterRead() }

function listMessages() { return local.listMessages() }  // 含异步补进来的后端消息

function saveMessage(text, clientId) { return local.saveMessage(text, clientId) }
function getProfile() { return local.getProfile() }

function saveProfile(fields) {
  const result = local.saveProfile(fields)
  const lt = toBackendLossType((fields.lossType || {}).value || '')
  ensureAccount().then(ok => {
    if (ok) request('/companions', 'POST', { user_id: BACKEND_UID, relationship_type: lt }).catch(() => {})
  })
  return result
}

function listCollection(kind) {
  const items = local.listCollection(kind)
  if (kind === 'letters') return items
  ensureAccount().then(ok => {
    if (!ok) return
    const map = { items: '/item/', photos: '/photo/' }
    if (!map[kind]) return
    request(map[kind] + encodeURIComponent(BACKEND_UID), 'GET').then(r => {
      if (r.ok && r.data) {
        const arr = r.data.items || r.data.photos || r.data.list || []
        if (Array.isArray(arr)) storage.update(s => { s.backendCollection = s.backendCollection || {}; s.backendCollection[kind] = arr })
      }
    }).catch(() => {})
  })
  const cached = storage.read().backendCollection
  return (cached && cached[kind] && cached[kind].length) ? cached[kind] : items
}

function startChat() { local.startChat() }

/**
 * 发消息：同步返回本地消息 + 后台调真后端；真实 AI 回复异步写 storage。
 * 建档阶段纯本地 UX，不调后端。
 */
function sendChatMessage(payload, clientId) {
  // 关键：用发送前的建档状态判断。最后一题回答时发送前还没建档完成 → 纯本地，不调后端。
  // 否则最后一题会触发一个 /chat，紧接着自由聊天第一条又发 /chat → CONVERSATION_BUSY。
  const wasOnboarding = !isOnboardingDone()
  const saved = local.sendChatMessage(payload, clientId)
  if (wasOnboarding) return saved
  const text = (payload.text || '').trim()
  if (!text && !payload.skip) return saved

  // 自由聊天前必须等 companion 建好，且 /api/chat 串行发送（后端有 CONVERSATION_BUSY 并发保护）
  const ready = companionReady || ensureAccount()
  chatQueue = chatQueue.then(() => ready).then(() => ensureAccount()).then(ok => {
    if (!ok) return null
    return request('/chat', 'POST', {
      user_id: BACKEND_UID,
      message: payload.skip ? '这个问题先跳过' : text,
      client_request_id: clientId,
    })
  }).then(r => {
    if (!r || !r.ok || !r.data || !r.data.reply) {
      if (r) console.warn('[cloud] /api/chat 失败', r.status, r.data)
      return
    }
    const now = Date.now()
    const aiId = r.data.assistant_message_id ? 'be_' + r.data.assistant_message_id : 'ai_' + now.toString(36)
    storage.update(s => {
      if (s.messages.some(m => m.id === aiId)) return
      s.messages.push({
        id: aiId, clientId: 'ai_' + (clientId || ''),
        journeyId: s.currentJourneyId, ownerId: BACKEND_UID,
        role: 'assistant', type: 'text', content: r.data.reply,
        createdAt: now,
        timeLabel: new Date(now).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' }),
        fromBackend: true,
        backend: { emotion: r.data.emotion, crisis: r.data.crisis, tool: r.data.tool },
      })
    })
    if (typeof global._makeAgainRefresh === 'function') global._makeAgainRefresh()
  }).catch(e => console.warn('[cloud] 后端回复异常', e))
  return saved
}

function getChatState() {
  const base = local.getChatState()
  // 把缓存的后端快照带上（供页面显示 mood 等）
  const snap = storage.read().backendSnapshot
  if (snap) base.backendHome = snap.home
  return base
}

module.exports = {
  acceptConsent, getOverview, markLetterRead, listMessages, saveMessage,
  getProfile, saveProfile, listCollection, startChat, getChatState, sendChatMessage,
  _request: request, _ensureAccount: ensureAccount, _toBackendLossType: toBackendLossType,
  _isOnboardingDone: isOnboardingDone,
  _setRefresh(fn) { global._makeAgainRefresh = fn },
}
