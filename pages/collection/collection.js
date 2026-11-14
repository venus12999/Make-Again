const service = require('../../utils/service')
const { COLLECTION_TABS } = require('../../utils/constants')
const { showError } = require('../../utils/ui')
Page({
  data: { tabs: COLLECTION_TABS, active: 'board', items: [], error: '', emptyTitle: '', emptyDescription: '', showChatGuide: false, openingChat: false },
  onShow() {
    this._active = true
    const globalData = getApp().globalData
    if (globalData.pendingLetterArrival) {
      globalData.pendingLetterArrival = false
      globalData.pendingChatGuide = true
      this.setData({ active: 'letters' })
      wx.pageScrollTo({ scrollTop: 0, duration: 0 })
    } else if (!globalData.pendingChatGuide && this.data.active !== 'board') {
      this.setData({ active: 'board' })
    }
    this.setData({ showChatGuide: !!globalData.pendingChatGuide })
    this.syncGuideTabBar()
    this.refresh()
  },
  onHide() { this._active = false; this.syncGuideTabBar() },
  onUnload() { this._active = false; this.syncGuideTabBar() },
  blockGuideTouch() {},
  syncGuideTabBar() {
    const hidden = !!(this._active && this.data.showChatGuide && this.data.active === 'letters')
    if (hidden === !!this._guideTabBarHidden) return
    this._guideTabBarHidden = hidden
    // The spotlight supplies its own navigation while the native tab bar is hidden.
    const method = hidden ? 'hideTabBar' : 'showTabBar'
    if (typeof wx[method] === 'function') wx[method]({ animation: false })
  },
  selectTab(event) { this.setData({ active: event.detail.value || event.currentTarget.dataset.key }); this.syncGuideTabBar(); this.refresh() },
  openBoard(event) { this.setData({ active: event.currentTarget.dataset.key }); this.syncGuideTabBar(); this.refresh() },
  backToBoard() { this.setData({ active: 'board', error: '' }); this.syncGuideTabBar() },
  refresh() {
    const content = {
      items: ['一些物品，装着一段故事', '物品上传与保存将在后续阶段接入。'],
      photos: ['把想留住的瞬间，慢慢收藏', '照片上传与关联故事将在后续阶段接入。'],
      memories: ['不必急着把记忆整理好', 'AI 记忆提取与编辑将在后续阶段接入。'],
      letters: ['等待下一封信', '旅程报告与仪式信件将在后续阶段接入。']
    }
    if (this.data.active === 'board') return
    try {
      const empty = content[this.data.active] || content.letters
      this.setData({ items: service.listCollection(this.data.active), emptyTitle: empty[0], emptyDescription: empty[1], error: '' })
    } catch (error) { this.setData({ items: [], error: error.message }) }
  },
  readLetter() { wx.navigateTo({ url: '/pages/letter/letter' }) },
  startChat() {
    if (this.data.openingChat) return
    this.setData({ openingChat: true })
    wx.switchTab({
      url: '/pages/chat/chat',
      success: () => { getApp().globalData.pendingChatGuide = false; this.setData({ showChatGuide: false }); this.syncGuideTabBar() },
      fail: () => showError(new Error('暂时无法打开聊天，请再试一次。')),
      complete: () => this.setData({ openingChat: false })
    })
  }
})
