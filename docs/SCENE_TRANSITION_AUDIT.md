# V3 · 场景切换现状审计（SCENE TRANSITION AUDIT）

> 日期：2026-11-23。分支：`scenario-validate/V3`（从 main `af217fdf` 切出，不合并 main）。
> 任务来源：`docs/THREE_SCENARIO_VALIDATION_PLAN.md` V3。
> 口径：**先验证现状、不改产品代码**。全部离线（MOCK_LLM=true + 显式 monkeypatch `LLMClient.chat`），
> 不读 `work/local-test-key.txt`、不发网络、不跑真 hy4。
> 测试文件：`MakeAgain-backend/tests/test_scene_transition.py`（本分支新增）。
> 测试口径说明：`test_*_current` = 刻画今天真实行为（应 PASS，print 原始数字作证据）；
> `test_*_expected` = 断言应有产品行为，标 `@pytest.mark.xfail(strict=True)`，**XFAIL = 产品缺口**，
> 若产品修好会 XPASS(strict) 变红，提示缺口已闭合。

---

## 0. 结论摘要（不粉饰）

| 验收项 | 结果（真实数字） | 判定 |
|---|---|---|
| V3-V1 场景切换测试 | 本文件 9 项：**7 passed / 2 xfailed / 0 failed**；其中切换子集 4 项 = 3 passed + 1 xfailed | 部分验证（缺口 GAP-V1） |
| V3-V2 归档只读 | 2 项 **全 passed**；旧 companion `archived_at` 非空、`ending_stage=complete`、`ending_ritual=scene_transition`；写拒绝 2/2 均为 `ARCHIVED_READ_ONLY`；旧记忆进新 scope = `False`，召回只命中新记忆 1 条 | 已验证（限库层 + 显式切换路径） |
| V3-V3 新 Journey 隔离 | 2 项：**1 passed（隔离层 scoped_portraits：旧画像 0 条）+ 1 xfailed（主链读取层仍返回旧对象 `{'称呼':'小夏'}`）** | 部分验证（缺口 GAP-V3） |
| V3-V4 audit 缺口+行号 | 本文件 §2，三条缺口均带源码行号 + 测试数字 | 已验证 |
| V3-V5 全量 pytest | `519 passed, 4 skipped, 2 xfailed, 0 failed, exit 0`（详见 §3）；基线 509 collected / 506 passed | 已验证（无回归） |

**一句话结论**：场景切换状态机**定义了但未接线**（api/agent 零调用，`detect_scene_transition` 根本不存在）；
显式调用时归档只读与隔离层画像隔离都成立，但**主链画像读取层仍继承旧对象画像**，且默认开关关闭使切换整体 no-op。

---

## 1. 代码现状：场景切换现在到底怎么实现

1. **真源**：`companions.relationship_type`（`memory/models.py:394-425`，`archived_at` 置位=只读档案态，
   见 `:422`）。`store.effective_loss_type()`（`memory/store.py:52-68`）优先读当前 companion 的
   relationship_type，回落遗留 `user_states.relationship_type` / `users.loss_type`（后者已在
   `memory/models.py:337` 标注弃用）。
2. **状态机**：`memory/state.py:264` `SceneStateMachine`，`:274-290` `decide()` 产出 `SceneDecision`，
   `:292-336` `apply()`：目标==当前 → NOOP；否则 `:321-325` 归档旧 companion
   （`store.archive_companion(..., ritual="scene_transition")`）+ `:325-327` 新建并置当前。
   对外入口 `:342 transition_scene` / `:348 on_relationship_change` / `:359 scene_decision_for`。
   **任务点名的 `detect_scene_transition` 在仓库中不存在**（测试实测 `hasattr(state, ...)==False`）。
3. **开关**：`memory/state.py:237-239` `USE_PERSONA` 默认 `"0"`，`:295-300` `apply()` 在开关关闭时
   直接返回 `changed=False` 且不写库。
4. **路由**：`agent/companion.py:181-209` `chat()` 只在 `:199` **读** `effective_loss_type` 选场景引擎
   （breakup/loved_one/pet/通用），**没有任何"识别用户表达了新关系"的检测或写入**。
   `agent/interview.py:374-400` `start()` 在 `:375` 通过 `get_or_create_user(loss_type=...)` 写的是
   遗留 `users.loss_type`，不改 `companions.relationship_type`，因此访谈返回后真源不变
   （既有 `tests/test_persona.py:58-59` 已记录该现象）。
5. **手动新旅程 API**：`api/routes.py:1090-1108` `POST /companions` 能新建当前 companion，但 `:1100`
   要求旧 companion **先已归档**（否则 409 `ACTIVE_COMPANION_EXISTS`）；归档走 `:1046-1060` `/ending/commit`。
   这是一条**用户手动两步流程**，不是"表达进入新关系"自动触发。
6. **归档只读与隔离**：`memory/isolation.py:107-109` `is_archived`、`:216-241` `assert_scope`（写操作
   对归档 Journey 抛 `ARCHIVED_READ_ONLY`）、`:268-287` `scoped_add_memory`、`:290-301`
   `scoped_portraits`（按 companion_id 过滤画像）。这套守卫有效，但需调用方主动使用。

---

## 2. 产品缺口清单（真实缺口 + 证据行号）

### GAP-V1 · 场景切换状态机"定义了但未接线"，自然语言表达不触发切换 【缺口】
- **证据（测试数字）**：`test_v1_scene_machine_current_has_no_production_caller` 打印
  `detect_scene_transition 是否存在: False`、`api/ + agent/ 中对切换入口的调用: []`；
  源码扫描排除 `memory/state.py` 与全部 test 文件后，`api/`、`agent/` 中对
  `transition_scene / on_relationship_change / scene_decision_for / SceneStateMachine` 的调用数为 **0**。
- **证据（行为）**：`test_v1_chat_message_current_does_NOT_switch` 在 `USE_PERSONA=1` 下对已完成 breakup
  旅程的用户说"我开始新恋情了"，结果 `切换前 loss_type: breakup | 切换后 loss_type: breakup | companion 段数: 1`。
- **期望测试**：`test_v1_expected_chat_message_starts_new_journey`（XFAIL）实测
  `companion 段数: 1, archived_at: ['None']`，期望为 2。
- **源码行号**：定义 `memory/state.py:264,342,348,359`；零调用方 `agent/companion.py:181-209`、
  `api/routes.py:1090-1108`；无检测函数（全仓 grep 无命中）；`agent/interview.py:375` 只写遗留字段。
- **影响**：产品叙事"陪你带着关系继续生活"里"进入新关系→开启新旅程"这一段，只能靠用户手动点两次
  （先结束仪式再新建），对话里说出来不会发生任何事。**这是主链路缺口，本分支按纪律未改代码。**

### GAP-V1b · 默认开关关闭使上述切换整体 no-op 【配置缺口】
- `memory/state.py:237-239` `USE_PERSONA` 默认 0，`:295-300` 关闭时 `apply()` 不写库。
- 本审计为刻画"状态机本身是否可用"，所有切换测试显式 `monkeypatch.setenv("USE_PERSONA","1")`。
  若产品期望开箱即用，默认值需 lead 决策；本分支不改默认值。

### GAP-V2 · 归档只读成立，但只在隔离层/显式切换路径被验证 【边界，非硬缺口】
- **已验证数字**：显式 `transition_scene` 后旧 companion
  `archived_at=2026-11-22 22:30:59.263246`、`ending_stage=complete`、`ending_ritual=scene_transition`；
  归档 Journey 自身记忆**读得到**（id 一致），**写拒绝 2/2**：
  `assert_scope(operation='write')` 与 `scoped_add_memory` 均返回 `ARCHIVED_READ_ONLY`；
  新 Journey `allowed_companion_ids` 只含新 id，旧记忆 `in allowed_ids: False`，召回命中只含新记忆 1 条。
- **边界**：该只读性依赖调用方走 `memory/isolation.py`（`:216-241,:268-287`）或 API 层 `api/security.py:169`
  的归档校验；本审计未验证"自然语言触发切换"路径（该路径本身不存在，见 GAP-V1），
  也未做 API 层 409 端到端断言（非本包必测项）。**不写成"全链路已验证"**。

### GAP-V3 · 新 Journey 继承旧对象画像（主链读取层与隔离层口径不一致）【缺口】
- **隔离层 PASS**：`test_v3_isolation_layer_portraits_not_inherited_current` 实测
  `新 Journey scoped_portraits 条数: 0`，旧 Journey 带归属画像 id 出现在新 Journey: `False`。
- **主链 FAIL（XFAIL）**：`test_v3_main_chain_object_portrait_current_still_inherited` 实测切换后
  `loss_type: pet`，但 `store.get_portrait(object) = {'称呼': '小夏', '关系': '前任'}`、
  `store.get_portrait(user) = {'昵称': '林薇'}`、`mention_count 取到的对象称呼: 小夏`。
  期望测试 `test_v3_expected_main_chain_object_portrait_not_inherited`（XFAIL）实测仍返回
  `{'称呼': '小夏', '关系': '前任'}`。
- **源码行号**：
  - `memory/store.py:164-171` `get_portrait()` 只按 `user_id + kind` 取最新一行，**无 companion_id 过滤**；
  - `memory/store.py:174-190` `upsert_portrait()` 同样只按 `user_id + kind` 定位，新行 `companion_id` 留空；
  - `agent/interview.py:744-745` 报告落库正是用这两个函数写 `user` / `object` 画像 → 主链画像天然无 Journey 归属；
  - `memory/models.py:37-38,41-51` `Portrait` 虽有 `CompanionScoped.companion_id` 列，但上述读写没用它；
  - 对比 `memory/isolation.py:290-301` `scoped_portraits()` 明确按 `companion_id` 过滤 —— **两套口径打架**；
  - 下游受影响点：`memory/state.py:181-189` `mention_count()` 用 `get_portrait(...,"object")` 取称呼，
    切换后仍按旧对象名计数。
- **影响**：新旅程的画像注入/称呼引用/报告仍带着上一段的对象，"不继承"只在隔离层成立、主链不成立。
  另：可变 persona 画像（`agent/persona.py:53-58`）是 user 级设计，**跨 Journey 共享属预期行为**，
  本次不把它判为缺口，仅记录口径差异。

---

## 3. 验收明细（真实命令 + 输出）

### V3-V1/V2/V3 · 单文件离线测试
```
cd MakeAgain-backend
python -m pytest tests/test_scene_transition.py -q -rsx
结果: 7 passed, 2 xfailed, 1 warning in 4.80s ; PYTEST_EXIT=0
XFAIL 1: test_v1_expected_chat_message_starts_new_journey  (GAP-V1)
XFAIL 2: test_v3_expected_main_chain_object_portrait_not_inherited (GAP-V3)
```
带 `-s` 的原始数字（节选，完整日志 `work/v3_scene_transition_evidence.log`）：
```
[V1-CURRENT] detect_scene_transition 是否存在: False
[V1-CURRENT] api/ + agent/ 中对切换入口的调用: []
[V1-CURRENT] 显式切换前 loss_type: breakup → 后 loss_type: pet ; action=archive_and_create
[V1-CURRENT] 用户话语: 我开始新恋情了 → 切换前 loss_type: breakup | 后 loss_type: breakup | companion 段数: 1
[V1-EXPECTED] companion 段数: 1 archived_at: ['None']            → XFAIL
[V2-CURRENT] 旧 archived_at 非空 / ending_stage=complete / ending_ritual=scene_transition
[V2-CURRENT] 归档写拒绝: ARCHIVED_READ_ONLY | 归档新增拒绝: ARCHIVED_READ_ONLY
[V2-CURRENT] 旧记忆 in allowed_ids: False ; 召回命中只含新记忆 1 条
[V3-CURRENT] 新 Journey scoped_portraits 条数: 0 ; 旧画像出现在新 Journey: False
[V3-CURRENT] store.get_portrait(object) = {'称呼': '小夏', '关系': '前任'}  → XFAIL(期望不含小夏)
```
证据日志（未入库）：`work/v3_scene_transition_test.log`、`work/v3_scene_transition_evidence.log`。

### V3-V5 · 全量无回归
```
cd MakeAgain-backend
python -m pytest -q
519 passed, 4 skipped, 2 xfailed, 2 warnings in 150.96s (0:02:30) ; PYTEST_EXIT=0
```
基线核对：从 `af217fdf` 建临时 worktree 执行 `--collect-only -q` 得 **509 tests collected**
（任务给定 506 passed + skipped），当前工作区 525 collected = 509（基线）+ 本分支新增
`tests/test_scene_transition.py` 9 项 + 队友未跟踪文件 `tests/test_scenario_run.py` 7 项。
**队友的 `tests/test_scenario_run.py` 是共享工作区未跟踪文件，未纳入本分支提交。**
临时 worktree 已 `git worktree remove` 清理。

---

## 4. 边界 / 未验证 / NOT_RUN

- **未跑真 hy4**：本包按任务只用离线 monkeypatch LLM；三场景真模型验证属 V1/V2 工作包，不在本分支范围。
- **自然语言检测是"缺口"而非"测试写错"**：`test_*_expected` 用 `xfail(strict=True)` 固化期望，
  当前 XFAIL 是真实的"产品没做"，不是把失败藏起来；修复后会自动 XPASS 报红提醒闭合。
- **未改产品代码**：`agent/`、`memory/`、`api/` 无任何改动；本分支只新增 1 个测试文件 + 本 audit 文档
  （+ 台账 `docs/PROGRESS.md` 追加一行）。
- **未验证**：API 层对归档 companion 的 409 端到端断言、`/companions` 手动两步流程的界面体验、
  persona_profile 跨 Journey 共享是否为产品预期（倾向预期，待 lead 决策）。
- **建议（不实施）**：若 lead 决定开新工作包修复，优先级建议 GAP-V1（接线+检测）> GAP-V3（画像按
  companion_id 过滤并在报告落库时写入归属）> GAP-V1b（开关默认值决策）；本分支不擅自开工。

## 相关
- 计划：`docs/THREE_SCENARIO_VALIDATION_PLAN.md`；评测约定：项目根 `AGENTS.md`
- 既有场景状态机测试：`MakeAgain-backend/tests/test_persona.py`
- 既有 Journey 隔离测试：`MakeAgain-backend/tests/test_journey_isolation.py`
