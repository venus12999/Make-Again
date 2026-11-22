# WP-04 · 结束仪式验证（ENDING VALIDATION）

> 日期：2026-11-23。分支：`eval/ending-validation`（从 main `6ab6f718` 切出，独立 worktree，不合并 main）。
> 任务来源：产品五段闭环最后一段「结束仪式」（抠图→纪念文案→模拟来电→归档）此前从未验证。
> 口径：**先读码确认现状，再离线验证；不改产品代码**。默认 `--mock`：不读 key、不发网络、不跑真 hy4；
> rembg 用本机已缓存的 u2netp 模型真跑抠图（不是假抠图）。`--real` 为可选真模型文案生成，标口径。
> 产物：`MakeAgain-backend/eval/ending_validation.py`（验证器）、
> `MakeAgain-backend/tests/test_ending_validation.py`（行为验收）、本文件（报告）。
> 新缺口编号：**GAP-E2 / GAP-E3**（避免与 U-018 的场景切换 GAP-V* 混淆）。

---

## 0. 结论摘要（不粉饰）

| 验收项 | 结果（真实数字） | 判定 |
|---|---|---|
| D-V1 抠图兜底真跑通 | `backend=local`（无 COS 凭据）、`model=u2netp`、rembg 2.0.87；合成图 160×160 RGB → matte RGBA **背景透明占比 71.84%** → trim 后 **81×81** RGBA；有输出图 `outputs/ending/cutout_source.png`、`cutout_trimmed.png` | 已验证（含证据） |
| D-V2 文案个性化有数字 | 兜底路径 3 关系类型 → **3 段不同文案**；确定性桩走真实 prompt 路径，不同对象名/关系 → **3 段不同文案**，`subject_name in text` = 3/3 | 已验证（离线桩口径） |
| D-V3 AI 披露标记 | marker=`以下文字由 AI 根据你留下的故事整理`；两端点共用同一常量；**3/3 场景 × 2 端点 = 6/6** 带标记 | 已验证 |
| D-V4 归档只读 + 导出 | 归档幂等（两次 `archived_at` 相同）、`ending_stage=complete`；读仍可；写拒 **2/2 `ARCHIVED_READ_ONLY`**；导出 schema 1.0、source 1 条、portrait 1 条、10 类 categories；`item` 抠图**不在**导出范围、原图 key 恒空 | 已验证（含已知缺口） |
| D-V5 全量 pytest 无回归 | `523 passed, 6 skipped, 2 xfailed, 0 failed, exit 0`（312.64s）；基线 517 passed + 本包新增 6 项 = 523，无失败无回归 | 已验证（无回归） |

**一句话结论**：结束仪式五段在离线口径下**全部跑通且有数字**；但matic「纪念物本体不进导出」「原图不保留」是真实产品缺口，
以及真模型文案质量（`--real`）本机无 key 文件故 **NOT_RUN**，不伪称验证过。

---

## 1. 真实链路（读码确认，非照抄报告行号）

1. **抠图兜底**：`gateway/storage.py:171-188` `ai_matte()` 无 COS / mock_storage → 若 `USE_LOCAL_MATTE=1`
   走 `_ai_matte_local()`(:152-169) 调 `rembg.remove`，模型白名单只放行 `u2netp`/`isnet-general-use`(:25-41)
   （rembg 2.x 默认 `bria-rmbg` 是商用付费许可，明确禁止）；否则抛 `StorageError` 由
   `agent/item.py:317-322` 回退「存原图」`processing_status=original_fallback`。
   `agent/item.py:209-269` `_trim_alpha_png` 用纯 Python 收紧透明边（阈值 200，避开 AIPicMatting 噪点）。
2. **纪念文案个性化**：`agent/ending.py:25-62` `memorial()` 取当前 companion 的 `relationship_type` /
   `subject_name`(:28-37) → mock 返回 `_FALLBACK[rt]`(:16-20,40-42)，真实把对象名+关系+近 12 条记忆摘要
   写进 prompt(:44-57) 调一次 fast 模型；返回 `disclosure`(:22,42/62)。
3. **模拟来电**：`agent/call.py:26-65` `call_script()` 结构同上（近 8 条记忆、40~80 字、>120 字截断），
   自带同一披露常量 `disclosure`(:23,43/65)；音频不落服务端，由前端 `/speech/tts` 生成。
4. **归档不可逆**：`memory/store.py:1392-1405` `archive_companion()` 幂等置 `archived_at` /
   `ending_stage=complete` / `ending_ritual` / `ending_committed_at`；
   `memory/isolation.py:216-241` `assert_scope(write)` 与 `:268-287` `scoped_add_memory` 对归档 Journey
   一律 `ScopeDenied(ARCHIVED_READ_ONLY)`，读维度仍可读（只读档案）。
5. **纪念物导出**：`memory/export.py:253-290` `export_user_memory()` 导出 schema_version / categories /
   source / candidate / fact / provenance_graph，覆盖 7 张记忆表 + 画像(:39-47)。
   **物品抠图 `item_memories` 不在 `_MEMORY_TABLES`**，故导出 JSON 不含纪念物本体（见 GAP-E2）。

---

## 2. 实现口径 / 方法学

- **离线不烧 key**：`--mock` 下隔离 SQLite + MOCK_LLM/MOCK_STORAGE=true；`LLMClient()` 走产品 mock 兜底；
  另一路用独立 `SimpleNamespace` settings + 实例级 `chat` 桩走**真实 prompt 构造路径**，`offline-placeholder-...`
  是占位符不是 key，不打印、不写入产物。测试断言输出 JSON/stdout 不含任何 `llm_api_key` 或占位符。
- **合成图**：`eval/ending_validation.py:_synthetic_png()`（白底 160×160 + 红方块），标 synthetic-handwritten，
  无外部数据。rembg 透明占比在 **matte（trim 前）** 上量；trim 后透明边已裁掉，紧凑主体内部 alpha=0 很少，
  所以「trim 生效」改用尺寸收缩 160→81 证明，量错位置会误判 FAIL（本轮实际踩到并修正）。
- **`--real`（可选）**：只从 `work/local-test-key.txt` 读 key 装配环境（key 明文不打印、不入库），
  用 hy4-flash-free 各生成一段纪念文案/来电话语，只验链路与披露、不判文案质量。本机该 key 文件不存在 →
  如实 `NOT_RUN`。pytest 不跑此模式。
- **方法学**：code-based graders 优先（fast/cheap/objective/reproducible）。
  Anthropic《Demystifying evals for AI agents》2026-01-08，
  https://www.anthropic.com/research/demystifying-evals-for-ai-agents
  基准日 2026-11-23：319 天 / 30.44 ≈ 10.5 月 → 权重 **0.7**（6-18 月档，参考，须查更新版）。
  本工具全为本地确定性判定，不引入未校准 LLM judge，衰减不改变实现；非本轮新联网调研，沿用 E2/E5 同来源复核。

---

## 3. 验收明细（真实命令 + 输出）

### 3.1 D-V1..D-V4 · 验证器 CLI（离线，真 rembg）

```
cd MakeAgain-backend
python eval/ending_validation.py --mock
=== 结束仪式离线验证（WP-04）===
  D-V1_cutout: PASS
  D-V2_memorial_copy: PASS
  D-V3_disclosure: PASS
  D-V4_archive_export: PASS
  D-V1 backend=local model=u2netp matte透明占比=71.84% 尺寸 160→81 输出=81x81
  D-V2 兜底不同文案=3 prompt路径不同文案=3
  D-V3 披露标记=以下文字由 AI 根据你留下的故事整理 命中 3/3×2端点
  D-V4 归档幂等=True 写拒=['ARCHIVED_READ_ONLY', 'ARCHIVED_READ_ONLY'] 导出source=1 类别=candidate,chat_message,daily_chat_summary,diary,echo,emotion,fact,memory,mood,portrait
  real_mode: NOT_RUN not requested (--mock default)
  结论: 4/4 PASS
  证据: outputs/ending/ending_validation.json
PYTEST_EXIT=0
```
证据文件（本分支新增，未入库产物）：`outputs/ending/ending_validation.json`、
`outputs/ending/cutout_source.png`、`outputs/ending/cutout_trimmed.png`、`outputs/ending/export_snapshot.json`。

### 3.2 新行为验收文件

```
cd MakeAgain-backend
python -m pytest tests/test_ending_validation.py -q -rsx
......                                                                   [100%]
6 passed, 1 warning in 26.60s ; PYTEST_EXIT=0
```
覆盖 V1~V5（V5 的 CLI 子进程复跑 `--mock` 并校验证据结构、缺口与无 key 泄漏）。

### 3.3 D-V5 · 全量 pytest 无回归

```
cd MakeAgain-backend
python -m pytest -q
........................................................................ [ 13%]
................................................................s....... [ 27%]
...(略)...                                                                [100%]
523 passed, 6 skipped, 2 xfailed, 2 warnings in 312.64s (0:05:12) ; PYTEST_EXIT=0
```
基线核对：main `6ab6f718` = 517 passed；本分支新增 `tests/test_ending_validation.py` 6 项 →
517 + 6 = 523 passed，0 failed。2 xfailed 是既有的 U-018 场景切换缺口（strict xfail，非本包引入）。

### 3.4 `--real` 可选模式（无 key，如实 NOT_RUN）

```
cd MakeAgain-backend
python eval/ending_validation.py --real
NO KEY FILE (expected NOT_RUN)
  real_mode: NOT_RUN no work/local-test-key.txt
  结论: 4/4 PASS ; EXIT=0
```
key 文件不存在，未读、未打印、未入库；真模型文案质量不在本轮验证范围。

---

## 4. 产品缺口 / 边界 / 未验证

### GAP-E2 · 纪念物本体不进导出 【产品缺口】
- **证据（数字）**：`export_user_memory()` 的 `_MEMORY_TABLES`(:39-47) 只覆盖 7 张记忆表 + 画像；
  实测导出 categories = `candidate,chat_message,daily_chat_summary,diary,echo,emotion,fact,memory,mood,portrait`，
  **无 `item`**；验证器同步落了一条 `item_memories`（`cutout_key` 非空）但 `item_cutout_in_export=False`。
- **影响**：用户做完结束仪式后，导出的「全部记忆」JSON 不含抠图纪念物本体/URL，纪念物只活在 DB + 对象存储。
  若产品叙事承诺「纪念物可带走」，这是缺口。本分支按纪律**未改产品代码**。

### GAP-E3 · 原图不保留（已知坑，非新发现）
- **证据**：`agent/item.py:340-347` 显式删除 `original_key` 与 `crop_key`，落库 `original_key=""`；
  实测 `item_original_key == ""`。归档后只有 `cutout_key`，无法回溯原图。
- **状态**：AGENTS.md「本项目的坑」已记录，这里作为结束仪式验证的固化断言，不重复开工作包。

### 边界 / NOT_RUN（如实标注）
- **未跑真 hy4**：本机 `work/local-test-key.txt` 不存在，`--real` = `NOT_RUN`；真实文案质量未验证。
  D-V2 的“不同物品不同文案”数字来自**离线确定性桩走真实 prompt 路径**口径，不是真人/真模型判分。
- **抠图口径**：rembg `u2netp` 在合成图上 alpha=0 背景 71.84%，主体边界有半透明过渡（非像素级完美分割）；
  本验证只证明「无 COS 凭据也能抠出透明底 PNG 且有输出图」，不证明任意真实照片的抠图质量。
- **未验证**：API 层 `/ending/commit` 的端到端 HTTP 409 行为、TTS 音频生成与播放、真 COS 数据万象抠图链路。
- **不改产品代码**：`eval/`、`tests/`、`docs/` 之外的文件零改动；`store.py` / `companion.py` 边界未碰
  （与画像隔离包 B、场景切换包 A 并行，守边界）。

---

## 相关
- 台账：`docs/PROGRESS.md`（U-018 场景切换缺口；本包为 WP-04 结束仪式，台账追加由 lead 统一落，避免并发冲突）
- 场景切换审计：`docs/SCENE_TRANSITION_AUDIT.md`
- 验证器 / 测试：`MakeAgain-backend/eval/ending_validation.py`、`MakeAgain-backend/tests/test_ending_validation.py`
