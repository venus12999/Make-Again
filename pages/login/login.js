const strokes = require('./brand-strokes.js')

Page({
  data: {
    animate: false,
    isLoggingIn: false,
    loginError: '',
    brandFallback: false,
    brandCanvasStyle: '',
    mottoDelay: Math.max(800, strokes.total - 200)
  },
  onReady() { this._initBrandCanvas(0) },
  onShow() {
    this._active = true
    this._loginAttempt = (this._loginAttempt || 0) + 1
    this.setData({ animate: false, isLoggingIn: false, loginError: '' })
    wx.nextTick(() => {
      if (!this._active) return
      this.setData({ animate: true })
      this._playBrand()
    })
  },
  onHide() { this._active = false; this._loginAttempt++ },
  onUnload() { this._active = false; this._loginAttempt++ },

  // Draw the Caveat brand word on a 2d canvas: each glyph outline is traced
  // stroke by stroke, then its fill fades in. Outline data and the shared
  // timeline come from brand-strokes.js.
  _initBrandCanvas(attempt) {
    wx.createSelectorQuery().in(this)
      .select('#brandCanvas')
      .fields({ node: true, size: true })
      .exec(res => {
        const canvas = res && res[0] && res[0].node
        if (!canvas) {
          if (attempt < 8) setTimeout(() => { if (this._active !== false) this._initBrandCanvas(attempt + 1) }, 300)
          else this.setData({ brandFallback: true })
          return
        }
        const info = wx.getWindowInfo()
        const dpr = info.pixelRatio || 1
        const emPx = 96 / 750 * info.windowWidth
        const scale = emPx / strokes.upem
        const heightUnits = strokes.ascent - strokes.descent
        canvas.width = Math.round(strokes.advance * scale * dpr)
        canvas.height = Math.round(heightUnits * scale * dpr)
        const cssWidth = Math.round(strokes.advance * scale)
        const cssHeight = Math.round(heightUnits * scale)
        if (canvas.style) {
          canvas.style.width = cssWidth + 'px'
          canvas.style.height = cssHeight + 'px'
        }
        this.setData({ brandCanvasStyle: `width:${cssWidth}px;height:${cssHeight}px` })
        this._brand = {
          canvas,
          ctx: canvas.getContext('2d'),
          scale: scale * dpr,
          lineWidth: Math.max(1.4, emPx * dpr * 0.022)
        }
        if (this.data.animate) this._playBrand()
      })
  },
  _playBrand() {
    const brand = this._brand
    if (!brand) return
    const start = Date.now()
    const tick = () => {
      if (!this._active) return
      const t = Date.now() - start
      this._drawBrand(t)
      if (t <= strokes.total + 120) brand.raf = brand.canvas.requestAnimationFrame(tick)
    }
    if (brand.raf) brand.canvas.cancelAnimationFrame(brand.raf)
    tick()
  },
  _drawBrand(t) {
    const brand = this._brand
    if (!brand) return
    const { ctx, scale: s, lineWidth, canvas } = brand
    const toY = y => (strokes.ascent - y) * s
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    ctx.lineWidth = lineWidth
    ctx.strokeStyle = '#4C4555'
    for (const letter of strokes.letters) {
      if (t >= letter.fillAt && letter.contours.length) {
        const alpha = Math.min(1, (t - letter.fillAt) / strokes.fillFade)
        if (alpha > 0) {
          ctx.globalAlpha = alpha
          ctx.beginPath()
          for (const contour of letter.contours) {
            const p = contour.points
            ctx.moveTo((letter.x + p[0]) * s, toY(p[1]))
            for (let i = 2; i < p.length; i += 2) ctx.lineTo((letter.x + p[i]) * s, toY(p[i + 1]))
            ctx.closePath()
          }
          ctx.fillStyle = '#4C4555'
          ctx.fill('evenodd')
          ctx.globalAlpha = 1
        }
      }
      for (const contour of letter.contours) {
        const local = t - contour.start
        if (local <= 0) continue
        const p = contour.points
        const n = p.length / 2
        const upto = Math.min(1, local / contour.dur) * (n - 1)
        const full = Math.floor(upto)
        ctx.beginPath()
        ctx.moveTo((letter.x + p[0]) * s, toY(p[1]))
        for (let i = 1; i <= full; i++) ctx.lineTo((letter.x + p[i * 2]) * s, toY(p[i * 2 + 1]))
        if (full < n - 1) {
          const frac = upto - full
          const x1 = p[full * 2], y1 = p[full * 2 + 1], x2 = p[full * 2 + 2], y2 = p[full * 2 + 3]
          ctx.lineTo((letter.x + x1 + (x2 - x1) * frac) * s, toY(y1 + (y2 - y1) * frac))
        }
        ctx.stroke()
      }
    }
  },

  loginWithWechat() {
    if (!this._active || this.data.isLoggingIn) return
    const attempt = ++this._loginAttempt
    this.setData({ isLoggingIn: true, loginError: '' })
    const current = () => this._active && attempt === this._loginAttempt
    const fail = message => {
      if (current()) this.setData({ isLoggingIn: false, loginError: message })
    }
    wx.login({
      timeout: 10000,
      success: result => {
        if (!current()) return
        if (!result.code) { fail('微信登录暂时未能完成，请重试。'); return }
        // The current app is local-only. A code is not a verified account session;
        // account binding still requires a server-side code2Session integration.
        // Always show the envelope after login, even when the letter was read on this device before.
        wx.redirectTo({
          url: '/pages/letter/letter?first=1',
          fail: () => wx.switchTab({
            url: '/pages/home/home',
            fail: () => fail('暂时无法进入，请重试。')
          })
        })
      },
      fail: () => fail('微信登录暂时未能完成，请重试。')
    })
  }
})
