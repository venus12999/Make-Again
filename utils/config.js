/**
 * 前端连接配置。
 *
 * 本地实测：
 *   1) 后端跑起来（见 docs/LOCAL_DEV_GUIDE.md）
 *   2) API_BASE 填你跑后端的机器地址：本机微信开发者工具用 http://127.0.0.1:8000，
 *      真机用 http://<你电脑局域网IP>:8000
 *   3) 微信开发者工具 → 详情 → 本地设置 → 勾选「不校验合法域名、web-view、TLS 版本以及 HTTPS 证书」
 *   4) CLOUD 设 true 即连真后端；false 则继续用本地假数据
 */
module.exports = {
  CLOUD: false,
  API_BASE: 'http://127.0.0.1:8000',
}
