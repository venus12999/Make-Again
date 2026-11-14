// Capture frames of the letter opening animation via the real login flow
// (direct reLaunch into the letter page leaves component internals unqueryable).
// node scripts/capture-letter-open.cjs <delayMs> <name>
const path = require('node:path')
const automator = require('miniprogram-automator')
const wait = ms => new Promise(r => setTimeout(r, ms))
const [delayArg, name] = process.argv.slice(2)
const delay = Number(delayArg || 4100)
const file = path.resolve(__dirname, '../artifacts/ui/current/' + (name || 'letter-open') + '.png')
;(async () => {
  let mini, lastErr
  for (let i = 0; i < 5; i++) {
    try {
      await wait(i * 3000)
      mini = await automator.connect({ wsEndpoint: 'ws://127.0.0.1:9420' })
      await mini.reLaunch('/pages/login/login')
      await wait(1800)
      const loginPage = await mini.currentPage()
      await mini.mockWxMethod('login', { code: 'letter-capture' })
      try { await (await loginPage.$('.wechat-login')).tap() } finally { await mini.restoreWxMethod('login') }
      await wait(2200)
      let letterPage = await mini.currentPage()
      for (let n = 0; n < 20 && letterPage.path !== 'pages/letter/letter'; n++) {
        await wait(300)
        letterPage = await mini.currentPage()
      }
      await (await letterPage.$('.ml-envelope')).tap()
      await wait(delay)
      await mini.screenshot({ path: file })
      console.log(name, 'saved at +' + delay + 'ms')
      break
    } catch (e) { lastErr = e; try { mini.disconnect() } catch {}; mini = null }
  }
  if (!mini) throw lastErr
})().catch(e => { console.error(String(e && e.message || e).split('\n')[0]); process.exitCode = 1 })
