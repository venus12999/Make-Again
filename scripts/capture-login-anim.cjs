// Restart the brand animation in-session with setData and capture mid-flight frames.
// Retries the whole run until every frame lands or attempts run out.
const path = require('node:path')
const automator = require('miniprogram-automator')
const wait = ms => new Promise(r => setTimeout(r, ms))
const out = path.resolve(__dirname, '../artifacts/ui/current')
const frames = [200, 500, 900, 1300, 1700, 2300]
;(async () => {
  const done = new Set()
  for (let run = 0; run < 6 && done.size < frames.length; run++) {
    let mini
    try {
      mini = await automator.connect({ wsEndpoint: 'ws://127.0.0.1:9420' })
      await mini.reLaunch('/pages/login/login')
      await wait(2000)
      const page = await mini.currentPage()
      await page.setData({ animate: false })
      await wait(300)
      await mini.evaluate(() => { const p = getCurrentPages().slice(-1)[0]; p._drawBrand(0); p._playBrand(); return { ok: true } })
      const start = Date.now()
      for (let i = 0; i < frames.length; i++) {
        if (done.has(i)) continue
        const at = frames[i] - (Date.now() - start)
        if (at > 0) await wait(at)
        const file = path.join(out, 'anim-' + String(i).padStart(2, '0') + '.png')
        await mini.screenshot({ path: file })
        done.add(i)
        console.log('frame', i, 'saved (+' + frames[i] + 'ms)')
      }
    } catch (error) {
      console.log('run', run, 'dropped:', String((error && error.message) || error).split('\n')[0])
    } finally { try { mini.disconnect() } catch {} }
    await wait(1200)
  }
  if (done.size < frames.length) { console.error('missing frames:', frames.filter((_, i) => !done.has(i))); process.exitCode = 1 }
})()
