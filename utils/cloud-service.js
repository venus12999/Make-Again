/**
 * cloud-service —— 连真后端的 service 适配层（保留 local-service 的数据模型）。
 *
 * 设计：页面用的 11 个方法签名和数据结构完全不变（页面零改动）；
 * 本地 storage 继续作为 UI 状态的快速来源；只在两个关键点走真后端：
 *   1) sendChatMessage 在访谈建档阶段 → 后端 /api/interview/* 生成真实报告/回复
 *   2) 建档完成后的自由聊天 → 后端 /api/chat 拿真实 AI 回复
 * 失败时回退到 local-service 的假数据（离线也能用），并把错误记在控制台。
 *
 * 配置：utils/config.js 的 API_BASE（默认 http://127.0.0.1:8000）
 * 微信开发者工具需勾选「不校验合法域名」才能连本机 IP。
 */
const local = require('./local-service')
const storage = require('./storage')
const { PROFILE_FIELDS, LOSS_TYPES } = require('./constants')
const config = require('./config')

const API_BASE = (config && config.API_BASE) || 'http://127.0.0.1:8000'
let TOKEN = null
let BACKEND_UID = null   // 后端注册返回的真实 user_id（≠ 本地 guest.id，混用会 OWNER_MISMATCH 403）

function fail(code, message) { throw Object.assign(new Error(message), { code }) }

/** 发请求到后端；自动带 token；返回 {ok, data, status}。 */
function request(path, method, body) {
  return new Promise((resolve) => {
    const header = { 'content-type': 'application/json' }
    if (TOKEN) header.Authorization = 'Bearer ' + TOKEN
    wx.request({
      url: API_BASE + '/api' + path,
      method: method || 'GET',
      header,
      data: body,
      success(res) {
        if (res.statusCode >= 200 && res.statusCode < 300) resolve({ ok: true, data: res.data, status: res.statusCode })
        else resolve({ ok: false, data: res.data, status: res.statusCode })
      },
      fail(err) { resolve({ ok: false, data: err, status: 0 }) },
    })
  })
}

/** 确保有后端账号：本地 guestId 映射成后端 username，注册/登录拿 token。 */
async function ensureAccount() {
  const state = storage.read()
  if (!state.guest) return false
  if (TOKEN) return true
  const username = 'ma_' + state.guest.id
  const password = 'ma-local-password'
  let r = await request('/auth/register', 'POST', { username, password })
  if (!r.ok) {
    // 可能已注册——改登录
    r = await request('/auth/login', 'POST', { username, password })
  }
  if (!r.ok || !r.data || !r.data.token) {
    console.warn('[cloud] 后端登录失败，回退本地模式', r.status, r.data)
    return false
  }
  TOKEN = r.data.token
  BACKEND_UID = r.data.user_id
  return true
}

/** 把前端 profile 的 lossType(relative/friend...) 映射成后端 loss_type。 */
function toBackendLossType(v) {
  const map = { relative: 'loved_one', friend: 'breakup', '': 'breakup' }
  return map[v] || v
}

// ---- 保持签名不变的方法 ----

async function acceptConsent() {
  const overview = local.acceptConsent()   // 本地建档结构照旧建立
  const ok = await ensureAccount()
  if (ok) {
    const state = storage.read()
    const lt = toBackendLossType((state.journeys.find(j => j.id === state.currentJourneyId) || {}).profile
      ? state.journeys.find(j => j.id === state.currentJourneyId).profile.lossType.value : '')
    await request('/consent', 'POST', { user_id: BACKEND_UID, version: 1 })
    await request('/companions', 'POST', { user_id: BACKEND_UID, relationship_type: lt })
  }
  return overview
}

function getOverview() { return local.getOverview() }
function markLetterRead() { return local.markLetterRead() }
function listMessages() { return local.listMessages() }
function saveMessage(text, clientId) { return local.saveMessage(text, clientId) }
function getProfile() { return local.getProfile() }
function saveProfile(fields) {
  const result = local.saveProfile(fields)
  // 用户改了离别类型 → 同步后端场景（后端切换 companion）
  const lt = toBackendLossType((fields.lossType || {}).value || '')
  const state = storage.read()
  if (state.guest && TOKEN) {
    request('/companions', 'POST', { user_id: BACKEND_UID, relationship_type: lt }).catch(() => {})
  }
  return result
}

function startChat() {
  local.startChat()
  // 访谈引导完全由前端 chat-guide 驱动（保数据模型），不在这里发请求
}

/**
 * 核心：发消息。访谈阶段本地建档照旧；建档完成后，走后端 /api/chat 拿真实 AI 回复。
 * 返回结构仍是一条 message（role=assistant, content=AI 回复），页面零改动。
 */
async function sendChatMessage(payload, clientId) {
  const saved = local.sendChatMessage(payload, clientId)  // 本地落一条 user 消息 + 更新 profile
  try {
    const state = storage.read()
    const guide = require('./chat-guide').getGuide(local.getProfile())
    const ok = await ensureAccount()
    if (!ok) return saved

    let aiText = null
    if (guide.step) {
      // 访谈阶段：把答案喂给后端访谈引擎，拿模型生成的承接问题/回复
      const r = await request('/chat', 'POST', {
        user_id: BACKEND_UID,
        message: payload.skip ? '这个问题先跳过' : (payload.text || ''),
      })
      if (r.ok && r.data && r.data.reply) aiText = r.data.reply
    } else {
      // 自由聊天：真 AI 陪伴回复
      const text = (payload.text || '').trim()
      if (text) {
        const r = await request('/chat', 'POST', { user_id: BACKEND_UID, message: text })
        if (r.ok && r.data && r.data.reply) aiText = r.data.reply
      }
    }
    if (aiText) {
      // 追加一条 assistant 消息到 local 消息流，页面自然会显示
      const id = 'ai_' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2, 8)
      storage.update(s => {
        s.messages.push({ id, clientId: 'ai_' + (clientId || ''), journeyId: s.currentJourneyId,
          ownerId: s.guest.id, role: 'assistant', type: 'text', content: aiText, createdAt: Date.now() })
      })
    }
  } catch (e) {
    console.warn('[cloud] AI 回复失败，已保留本地消息', e && e.message)
  }
  return saved
}

function listCollection(kind) { return local.listCollection(kind) }
function getChatState() { return local.getChatState() }

module.exports = {
  acceptConsent, getOverview, markLetterRead, listMessages, saveMessage,
  getProfile, saveProfile, listCollection, startChat, getChatState, sendChatMessage,
  _request: request, _ensureAccount: ensureAccount, _toBackendLossType: toBackendLossType,
}
