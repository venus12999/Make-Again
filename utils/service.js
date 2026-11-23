// Pages depend on this boundary.
// CLOUD=true → 连真后端（cloud-service）；false → 本地假数据（local-service）。
// 配置见 utils/config.js
const config = require('./config')
module.exports = (config && config.CLOUD) ? require('./cloud-service') : require('./local-service')
