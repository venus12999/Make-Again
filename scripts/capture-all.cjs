// Connect to the automated devtools session and screenshot every page.
// Each page gets a fresh connection so one flaky call cannot kill the run.
const fs = require('node:fs')
const path = require('node:path')
const automator = require('miniprogram-automator')
const wait = ms => new Promise(r => setTimeout(r, ms))
const out = path.resolve(__dirname, '../artifacts/ui/current')
const routes = [
  ['home', '/pages/home/home', 'switchTab'],
  ['chat', '/pages/chat/chat', 'switchTab'],
  ['collection', '/pages/collection/collection', 'switchTab'],
  ['profile', '/pages/profile/profile', 'switchTab'],
  ['letter', '/pages/letter/letter', 'navigateTo'],
]
async function capture(name, route, method) {
  const mini = await automator.connect({ wsEndpoint: 'ws://127.0.0.1:9420' })
  try {
    await mini.reLaunch('/pages/login/login')
    await wait(800)
    await mini[method](route)
    await wait(1800)
    await mini.pageScrollTo(0).catch(() => {})
    await wait(500)
    const current = await mini.currentPage()
    const file = path.join(out, name + '.png')
    await mini.screenshot({ path: file })
    console.log(name, '->', current.path, 'OK')
  } finally { mini.disconnect() }
}
;(async () => {
  fs.mkdirSync(out, { recursive: true })
  for (const [name, route, method] of routes) {
    try { await capture(name, route, method) }
    catch (e) { console.log(name, 'FAILED:', e.message.split('\n')[0]) }
    await wait(1500)
  }
})().catch(e => { console.error(e); process.exitCode = 1 })
