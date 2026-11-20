# Make Again · 合规基线（WP7）

> 范围：AI 生成内容标识、PIPL 最小必要、数据删除/导出。
> 不设未成年人门槛：本产品是失恋/丧亲/宠物离世后的疗愈陪伴，不是虚拟亲密关系/AI 伴侣；
> 未成年人同样是目标用户。若将来被监管归为"虚拟亲密关系"，由产品方决策，不在代码里写死年龄拦截。
> 实现载体：`api/compliance.py`（独立路由 + 事件契约）。开关 `USE_COMPLIANCE_GATE`
> 用 `os.getenv` 读取，默认关闭；**关闭时全部判定退化为不拦截/不落事件/不写审计**，
> 便于回滚与 A/B。本文把每一条措施标为【技术已实现】或【需产品/法务决定】。

---

## 0. 现状：AI 披露字段完整性核对（任务 3 的缺口记录）

读码核对结果（2026-11-20，基线 210162c）：

| 位置 | 现有披露 | 缺口 |
|---|---|---|
| `agent/ending.py` `memorial()` | 单字符串 `disclosure="以下文字由 AI 根据你留下的故事整理"` | 缺 `is_ai_generated` / `generated_by` / `content_type` / `explicit_label` / `implicit_marker` |
| `agent/call.py` `call_script()` | 同上单字符串 `disclosure` | 同上；模拟来电文案（含后续 TTS 音频）无独立 AI 标识字段 |
| `api/routes.py` `/api/chat` | **无** disclosure 字段（`grep disclosure routes.py` 无匹配） | 对话回复完全没有 AI 披露，前端无字段可展示 |
| 音频 `/speech/tts` | 无 | 合成语音无隐式水印/元数据标识 |

**结论**：现有披露是"一个纯文案字符串"，不满足结构化显式+隐式标识要求。

- 【技术已实现】`api/compliance.py` 提供 `DISCLOSURE_TEMPLATE` 结构化字段模板，
  含 `is_ai_generated`、`ai_disclosure`、`generated_by`、`content_type`、
  `explicit_label`、`implicit_marker`、`legacy_disclosure`（兼容旧字符串）。
  端点 `GET /api/compliance/disclosure`、`GET /api/compliance/disclosure/audit`
  返回模板与缺口清单。
- 【需产品/法务决定】把结构化字段真正写进 `/api/chat`、`/ending/content`、
  `/call/content` 响应体，以及 TTS 音频隐式水印——`agent/*` 与 `api/routes.py`
  不属于本工作包所有权，须由 W0 集成时接入；具体展示文案与位置由产品/法务确认。

---

## 1. AI 生成披露

依据《人工智能生成合成内容标识办法》（显式标识 + 隐式标识双轨）落地：

- 【技术已实现】
  - 结构化模板（见第 0 节），`ai_disclosure="本内容由人工智能生成，非真人实时对话或真人创作"`。
  - `explicit_label="AI 生成 / 虚拟陪伴"`：前端应在模拟来电、纪念文案、对话气泡显著位置展示。
  - `implicit_marker="ai_generated_content_v1"`：供内容元数据/水印层（W0 后续）写入。
  - `content_type` 区分 `simulated_call` / `memorial_text` / `chat_reply` / `tts_audio`。
  - `legacy_disclosure` 保留旧文案字段，前端灰度期可同时兼容。
- 【需产品/法务决定】显式标识的展示文案、字号/时长（语音需朗读或伴随提示）、
  隐式标识的技术载体（元数据字段 vs 音频水印算法）及标准符合性需法务确认。

## 3. PIPL 最小必要

- 【技术已实现】合规模块内存表**只存契约字段**：
  - 删除请求事件仅存 `user_id / scope / status / ts / reason`，不存聊天正文、照片、记忆内容。
  - 敏感审计仅存 `user_id / source / categories / level / crisis_level / ts`，
    `mark_sensitive()` 返回体不含原始 `text`（原始内容不被复制到审计日志）。
  - 严格访问日志按 `user_id` 隔离（`get_sensitive_audit(user_id)` / `get_deletion_requests(user_id)`）。
- 【需产品/法务决定】全站字段级数据地图与留存期限（PIPL 第 19 条）、
  敏感个人信息单独同意、跨境传输评估——需法务出制度文本，代码侧只提供标记/审计能力。

## 4. 数据删除 / 导出

- 【技术已实现】`POST /api/compliance/data-delete`：
  - 只**标记**删除请求（`status=pending_erasure`、`physical_delete=False`），
    落事件表 `GET /api/compliance/data-delete/requests`，不调用任何删除逻辑。
  - 支持 `scope`（`memories/photos/items/letters/chat/portrait/all`）与 `request_id`，
    校验非法 scope 返回 `invalid_scope`。
  - 开关关闭时返回 `status=not_enforced`、不落事件，保证可回滚。
- 【需产品/法务决定 / W0 集成】
  - **真实物理删除未在本次实现**：`store.py` / `models.py` / 对象存储不属于本包所有权，
    且物理删除涉及外键、对象存储文件、异步任务队列的级联清理，须由 W0 接入。
    行为验收表"用户删除账号后从存储中真实清除"因此记为**未达成 + 原因**（见报告）。
  - "15 个工作日内响应"、删除前二次确认、删除后不可恢复提示、导出格式为 JSON/可视化，
    均需产品/法务定义；本模块不提供导出端点（任务仅要求删除契约）。

## 5. 拟人化互动办法要点（技术映射）

| 办法要点 | 本代码侧措施 | 归属 |
|---|---|---|
| AI 身份显著披露、防止误认真人 | `DISCLOSURE_TEMPLATE` 显式+隐式标识 | 模板【已实现】，接入响应体【W0】 |
| 防止情感依赖 / 过度依恋 | 复用危机硬门槛 `emotion.crisis.detect`，命中自伤即敏感标记 + 严格审计 | 【已实现】标记，话术干预见 WP4 |
| 敏感内容单独标记与更严格访问控制 | `mark_sensitive()`：`self_harm`/`grief` 分类、`access_policy=restricted`、`strict_audit=True`、写 `SENSITIVE_AUDIT` | 【已实现】 |
| 用户自主退出 / 数据清除 | `data-delete` 请求事件契约 | 契约【已实现】，物理删除【W0/待定】 |
| 日志留存与审计 | 敏感审计事件带 `iso` 时间戳、按 user 隔离 | 内存版【已实现】，持久化【W0】 |

> 拟人化互动的具体法规条文与适用版本（征求意见稿/正式稿）由法务最终确认；
> 本表只做"办法要点 → 代码能力"的工程映射，不构成法律意见。

---

## 6. 开关与回滚

| 项 | 值 |
|---|---|
| 开关名 | `USE_COMPLIANCE_GATE` |
| 读取方式 | `os.getenv("USE_COMPLIANCE_GATE","0")`，支持 `1/true/yes/on` |
| 默认 | 关闭（`0`） |
| 关闭行为 | 披露模板仍可读（`enforced=False`）；删除请求不落事件；敏感标记不写审计 |
| 集成动作 | W0 在 `main.py` 加 `app.include_router(compliance.router)` 并按需打开开关 |

---

## 7. 验收与证据

真实命令与原始输出见 `work/wp7_evidence.py` 与 `tests/test_compliance.py`，逐项对应
MASTER_PLAN_PARALLEL.md 第 5 节 WP7 表三行（AI 标识 / 数据删除 / 敏感数据）。
