// Capture one frame of the native login animation: node scripts/capture-login-frame.cjs <delayMs> <name>
// A fresh process per frame — the automator socket can drop mid-session, so never reuse it.
const path = require('node:path')
const automator = require('miniprogram-automator')
const wait = ms => new Promise(r => setTimeout(r, ms))
const [delayArg, name] = process.argv.slice(2)
const delay = Number(delayArg || 0)
const file = path.resolve(__dirname, '../artifacts/ui/current/' + (name || 'login-anim') + '.png')
;(async () => {
  let mini, lastError
  for (let attempt = 0; attempt < 5 && !mini; attempt++) {
    try {
      await wait(attempt * 2500)
      mini = await automator.connect({ wsEndpoint: 'ws://127.0.0.1:9420' })
      await mini.reLaunch('/pages/login/login')
    } catch (error) { lastError = error; try { mini.disconnect() } catch {}; mini = null }
  }
  if (!mini) throw lastError
  try {
    await wait(delay)
    await mini.screenshot({ path: file })
    console.log('saved', file, 'at +' + delay + 'ms')
  } finally { mini.disconnect() }
})().catch(e => { console.error(e.message || e); process.exitCode = 1 })
