# Make Again 产品 ↔ 工程映射总档

> 本文件是后端总档。用途：把产品功能映射到工程实现，记录当前状态，明确优先级与不做项。
> 基线：`main @ 737cd8a`（2026-11-20，官方 SDK 网关已合入）。
> 技术栈：Python 3.12 + FastAPI + SQLAlchemy + SQLite（当前实现）；不切 Hono/PG17。
> 代码规模：约 11,600 行核心代码（agent/memory/emotion/gateway/api），74 个 API 端点。

## 0. 当前优先级（最高原则）

**本地优先，云端后置。**

先把产品在本地完整做出来、能演示、能验证，再谈云端部署。云端部署本身不难（Docker + Nginx + 域名，已有配置），是顺水推舟的事，不是当前重点。

- 不折腾部署、不追求上线、不为了云端改造架构
- 所有工作以"本地能跑、产品能用"为验收标准
- 前端不在本档范围内（由前端同学负责，本档只讲后端与产品映射）

## 1. 不做项（明确排除，本轮不管）

| 产品项 | 状态 | 原因 |
|---|---|---|
| 亲人声音克隆 | 不做 | 无能力，用户已确认 |
| 宠物 AI 视频生成 | 不做 | 无能力，用户已确认 |
| NFC 实体卡 | 不做 | 商业化后置 |
| 消散动效 | 不做（后端） | 纯前端效果 |
| 会员付费商品配置 | 暂不管 | 支付网关代码就绪，但无默认可售套餐（需运营配置） |
| 微信真机 / 支付 / 订阅消息真实验证 | 暂不管 | 需真实凭据与部署环境 |

## 2. 产品 → 工程完整映射

### A. 账号与引导

| 产品功能 | 工程模块 | 关键文件 | 状态 |
|---|---|---|---|
| 微信/密码登录、会话恢复 | 认证 | `api/routes.py` (`/auth/*`), `auth.py`, `gateway/wechat.py` | ✅ |
| 主创欢迎信（可编辑） | 欢迎信 | `memory/models.py`, `memory/seed.py`, `api/routes.py` (`/welcome-letter/*`) | ✅ |
| 首次同意 Consent | Consent | `api/routes.py` (`/consent`), `api/security.py` | ✅ |
| 引导阶段 | onboarding | `agent/onboarding.py`, `/onboarding/*` | ✅ |

### B. 访谈与画像

| 产品功能 | 工程模块 | 关键文件 | 状态 |
|---|---|---|---|
| Question List 逐题访谈 | 访谈引擎 | `agent/interview.py`（1001 行）, `/interview/*` | ✅ |
| 用户补充叙述重生成 | 同上 | `/interview/revise` | ✅ |
| 个人画像 + 对象画像 | 画像 | `memory/portrait.py`, `memory/facts.py` | ✅ |
| 关系类型推断 | 关系 | `agent/relationship.py`, `/relationship/*` | ✅（含 friend） |
| 初始/跟踪报告 | 报告 | `agent/report.py`, `agent/async_report.py`, `/reports` | ✅ |

### C. 主界面（核心）

| 产品功能 | 工程模块 | 关键文件 | 状态 |
|---|---|---|---|
| AI 聊天（通用/分手/亲友/宠物四引擎） | 陪伴 | `agent/companion.py`, `agent/breakup.py`, `agent/loved_one.py`, `agent/pet.py`, `/chat` | ✅ |
| 时间/情绪触发建议 | Nudge | `agent/nudge.py`, `/nudge/*` | ✅ |
| 每日心情（AI 判断） | 情绪 | `emotion/classifier.py`, `/mood/*`, `memory/models.py`(DailyMood) | ✅ |
| 今日主题 | 主题 | `agent/daily.py` | ✅ |
| 情绪日历 | 日历 | `memory/calendar.py`, `/home` | ✅ |
| 危机干预 | 危机 | `emotion/crisis.py`（前置硬门槛） | ✅ |
| 语气路由 | 语气 | `emotion/tone.py` | ✅ |
| 首页聚合 / 状态 | 聚合 | `agent/home.py`, `/state`, `/home` | ✅ |

### D. 聊天工具（tool calling）

| 产品功能 | 工程模块 | 关键文件 | 状态 |
|---|---|---|---|
| 物品纪念/寄存卡片 | 物品 | `agent/item.py`（384 行，含抠图链路） | ✅ |
| 拍立得照片卡片 | 照片 | `agent/photo.py`, `/photo/*` | ✅ |
| 待办清单卡片 | 待办 | `agent/todo.py`, `/todo/*` | ✅ |
| 写给自己卡片 | 便签 | `agent/note.py`, `/diary` | ✅ |
| 回声壁 | Echo | `agent/echo.py`, `/echo/popup/*` | ✅ |

### E. 记忆系统

| 产品功能 | 工程模块 | 关键文件 | 状态 |
|---|---|---|---|
| 记忆抽取（后台异步） | 抽取 | `memory/extract.py`, `memory/async_write.py` | ✅ |
| 记忆召回（打分公式） | 召回 | `memory/recall.py` | ✅ |
| 事实/情绪节点/时序 | 存储 | `memory/facts.py`, `memory/reflect.py`, `memory/temporal.py`, `agent/history.py` | ✅ |
| 会话上下文（6 轮） | 上下文 | `memory/session_context.py` | ✅ |
| 记忆库 | ORM | `memory/store.py`（1351 行）, `memory/models.py`（521 行） | ✅ |

### F. 时间看板

| 产品功能 | 工程模块 | 关键文件 | 状态 |
|---|---|---|---|
| 周报 | 周报 | `agent/weekly.py`, `/weekly-report/*` | ✅ |
| 漂流瓶/树洞 | 社区 | `agent/treehole.py`（383 行）, `api/community.py`, `agent/moderation.py`（PII 扫描） | ✅ |
| 分页 | 分页 | `api/pagination.py` | ✅ |

### G. 结束旅程

| 产品功能 | 工程模块 | 关键文件 | 状态 |
|---|---|---|---|
| 纪念文案 | 纪念 | `agent/ending.py`, `/ending/content` | ✅ |
| 模拟来电 | 来电 | `agent/call.py`, `/call/content` | ✅ |
| 物件抠图（COS AIPicMatting） | 视觉 | `gateway/vision.py`, `gateway/storage.py` | ✅ |
| 归档/revision | 归档 | `/ending/commit`, `memory/models.py` | ✅ |

### H. 后端基础设施

| 功能 | 工程模块 | 关键文件 | 状态 |
|---|---|---|---|
| 数据库/模型/迁移 | 存储 | `memory/db.py`, `memory/migrations_b07.py`, `memory/b07_models.py` | ✅ |
| 幂等/回执/审计/鉴权 | 中间件 | `api/security.py`（246 行）, `api/b07.py`（221 行） | ✅ |
| 后台 worker | 任务 | `memory/jobs.py`, `admin_cli.py` | ✅ |
| 媒体签名/限流 | 中间件 | `api/media.py`, `api/limits.py` | ✅ |
| 模型网关（官方 openai SDK） | LLM | `gateway/client.py` | ✅ |
| 视觉识别（官方腾讯云 SDK） | 视觉 | `gateway/vision.py` | ✅ |
| 语音 TTS/ASR | 语音 | `gateway/speech.py` | ✅ |
| 订阅消息 / 支付 / COS | 集成 | `gateway/messages.py`, `gateway/payments.py`, `gateway/storage.py` | ✅（代码就绪） |
| 部署 | 运维 | `Dockerfile`, `docker-compose.yml`, `deploy/` | ✅（配置就绪，未部署） |

## 3. 状态总览（诚实版）

| 层面 | 状态 |
|---|---|
| 后端代码 | ✅ 产品主线功能全覆盖（除第 1 节不做项） |
| 后端测试 | ✅ 55 passed（mock 模式，无真实模型） |
| 真实模型效果 | ⚠️ 未验证（无 key） |
| 云端部署 | ⚠️ 配置就绪，未部署（按"本地优先"原则，暂不做） |
| 用户实际使用 | ⚠️ 队友仓库前端仍为 local-service，云端能力未接入（前端范围，本档不管） |

## 4. 关键架构说明（聊天链路怎么做的）

一次聊天：危机检测（硬门槛）→ 召回记忆（新鲜度 0.35 + 重要度 0.35 + 相关度 0.3，30 天半衰期）→ 组装上下文（画像 + 8 条记忆 + 最近 6 轮 + 今日主题 + 语气）→ 调快速模型 + 工具卡片 → 规则兜底 → 同步存聊天历史 + 后台异步抽取记忆（依赖 worker 常驻）。

**注意**：worker 不跑，聊天能回复但记忆不会写入。本地演示时 API + worker 都要启动。

## 5. 验收标准（本地优先）

- 本地启动：API + worker 双进程
- `pytest tests -q` 全绿
- 关键链路可演示：登录 → Consent → 聊天 → 记忆 → 报告 → 树洞 → 结束仪式
- 不追求：云端部署、真机、真实模型

## 6. 相关文档

- 产品 PRD：`docs/Make_Again_Team_PRD.md`
- 技术实现参考：`docs/Make_Again_Team_Technical_Implementation.md`
- 阶段开发记录：`docs/backend-development-plan.md`
- 后端运行说明：`MakeAgain-backend/README.md`
