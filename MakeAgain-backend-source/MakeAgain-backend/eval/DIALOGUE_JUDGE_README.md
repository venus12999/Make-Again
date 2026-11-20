# 对话质量评测框架（V2-B · dialogue_judge.py）

> 分支 `eval/golden-dialogue`，从 main `61a7305` 建，不合并 main。
> 所有权：仅 `MakeAgain-backend/eval/dialogue_judge.py`、`eval/dialogue_cases.json`、本 README。
> 指标口径全部来自 `docs/OLD_REPO_QUALITY_METRICS.md` §1/§2（旧仓库 V2 文档），**不自创指标**。

## 1. 它做什么

用**真实 hy4-flash-free**（opencode-go 网关，免费额度）跑两阶段评测，全程 `temperature=0`：

1. **生成**：把每条 case 的用户输入 + 场景背景喂给带产品硬边界的 system prompt，让 hy4-flash-free 生成回复。
2. **判定**：把「用户原话 + 上下文 + 期望行为 + 负例清单 + 待评回复」交给同一个 hy4-flash-free 做的 judge，输出结构化 JSON：
   `repetition.detected` / `unsupported_assertion.detected` / `empathy.score(0-4)` /
   `task_completion.score(0-4)` / `negative_example_hit[]`，每条都带证据与理由。
3. **对照实验（B-V1）**：内置两条**人工构造**的回复（好/坏），坏回复故意逐字重复用户整句两次并塞入无依据断言。
   同一个 judge 必须给出明显不同的分数，否则判 FAIL。
4. **复现性（B-V3）**：对照 + 前 N 条生成 case 二次 judge，逐字段比对决策字段；同时全程标注 `temperature=0`。

输出：人读报告 + `===JSON===` 结构化结果都打印到 stdout；`--out` 可另存 JSON 文件。
脚本不修改业务代码、语料、requirements、PROGRESS.md 或队友仓库。

## 2. 指标口径（来自 OLD_REPO_QUALITY_METRICS.md）

| 指标 | 来源 | 本脚本的计算方式 |
|---|---|---|
| 重复复述率 | §1 对话行 / §2 Agent 行「重复复述」 | `judge 判 repetition.detected=true 的 case 数 / 成功判定的 case 数` |
| 无依据断言率 | §1 / §2「无依据断言」 | `judge 判 unsupported_assertion.detected=true 的 case 数 / 成功判定的 case 数` |
| 任务完成率 | §1 对话行「任务完成率」 | `judge 的 task_completion.score 均值 / 4`（逐条 0-4，越界扣分） |
| 用户有用性反馈 | §1 对话行（旧仓库要求「有评测」） | 无真实用户反馈，故用 judge 的 `empathy.score / 4` 与 `negative_example_hit` 近似，**明确标注为近似，非真人评分** |

**共情恰当性（0-4）**是复合辅助分，不是产品 KPI——旧仓库 §0 明确「情绪分数或疗愈阶段不作产品成功指标，除非有专业依据」。
负例命中既有 judge 的语义判定，也有一条确定性的字面命中 (`literal_negative_hit`) 做交叉验证。

judge 判据写在脚本 `JUDGE_SYSTEM`（严格按字面，不脑补）：
- 重复复述：逐字重复用户整句 ≥1 次，或重复多处措辞且几乎无新增信息 → true；简短引用关键词并给新回应不算。
- 无依据断言：断言了 user_input/context 没有的事实、心理状态、因果或承诺 → true；开放式提问与陪伴不算。
- 越界（冒充逝者/前任/宠物口吻、诊断、医疗建议、替用户做决定、承诺「一定会好」）在任务完成度里扣分。

## 3. 语料口径（诚实标注）

`dialogue_cases.json` 共 **12 条 case**：breakup 4 / loved_one 4 / pet 4，覆盖三类离别。
每条含 `user_input` + `scene` + `context` + `expected_behavior`（共情焦点/不复述/无断言/边界/任务）
+ `negative_examples`（不该出现的表述，2–4 条）。

- `source = synthetic-handwritten`：**全部为手写模拟语料，非真实用户数据**；场景取自 PRD，不是采集到的用户原话。
- B-V1 的好/坏对照回复同样是**人工构造、非模型输出、非真实用户数据**。
- 语料 SHA256 写入结果 JSON，可追到具体版本。

## 4. 怎么跑

key 只放主仓库 `work/local-test-key.txt`（gitignore），明文不打印不入库。**worktree 里没有该文件**，需显式传路径：

```powershell
cd MakeAgain-backend
python eval/dialogue_judge.py --key-file 'D:\12412\dsh工作区\projects\make-again-mvp\work\local-test-key.txt'
```

可选项：

| 参数 | 作用 |
|---|---|
| `--cases <路径>` | 指定 case JSON（默认 `eval/dialogue_cases.json`） |
| `--key-file <路径>` | 指定 key 文件（默认 `<repo>/work/local-test-key.txt`；也认 env `DIALOGUE_KEY_FILE`） |
| `--model <名>` | 生成模型（默认 `hy4-flash-free`；可用 env `LLM_FAST_MODEL` 覆盖） |
| `--judge-model <名>` | judge 模型（默认与生成模型相同） |
| `--repro-samples N` | B-V3 二次 judge 的 case 条数（默认 2；0 则只做对照复现） |
| `--controls-only` | 只跑 B-V1 对照，省钱省时（**不足以验收 B-V2**） |
| `--out <路径>` | 可选：把 JSON 另存到文件（默认只 stdout） |

脚本从 key 文件解析 `KEY=` / `BASE_URL=` 后设置（明文只在进程内，绝不打印）：
`LLM_BASE_URL=<读到的 BASE_URL>`、`LLM_API_KEY=<读到的 KEY>`、`MOCK_LLM=false`、
`LLM_EXTRA_HEADERS={"x-opencode-session": "dialogue-judge-<随机>", "User-Agent": "make-again-judge/1.0"}`、
`HF_HOME=D:\12412\huggingface`、`HF_HUB_OFFLINE=1`、`ENVIRONMENT=test`、`USE_INSTRUCTOR=0`。
随机 session 只写进结果 JSON 的 `run.extra_headers`（不是密钥）；依赖 `openai` / `pydantic-settings` / `json_repair`（已在 requirements）。

**token 预算**：`hy4-flash-free` 非推理模型，实测 1000/2000 max_tokens 均 2~5s 出正文；脚本用 `2000 → 4000 → 6000` 递增重试兜住偶发空正文。B-V1 对照实测 judge 2000 预算够用（good completion=1057 / bad completion=1310）。若改回推理型 `hy4`，需把预算上调到 4000+。

## 5. key 安全（B-V4）

- 明文 key 只从文件读入进程内存；**不 print、不写 JSON、不写日志、不入库**。
- 所有异常文本过 `_redact()`：替换完整 key 与 `Bearer xxx` 形式后再输出。
- 输出 JSON 的 `key` 段只记「来源路径 + 用途 + in_memory_only/printed/committed=false」，无明文。
- 退出前扫描结果 JSON 是否含 key 明文，含则 B-V4 判 FAIL（本次实测：未发现泄漏）。
- 无 key / key 无效 / 网关不可达 / 探针空回复 → **状态 `NOT_RUN`**，退出码 0，不编造任何数字。

验证命令（应输出 `状态：NOT_RUN`）：

```powershell
python eval/dialogue_judge.py --key-file 'D:\不存在\no-key.txt'
```

## 6. 验收映射（B-V1~V4）

| 验收 | 判定位置 | 判据 |
|---|---|---|
| B-V1 | `controls.b_v1.checks` 六项 | 坏回复必须复述=true、断言=true；好回复两项=false；共情与任务分好>坏 |
| B-V2 | `acceptance["B-V2"]` | 12 条真实生成+判定跑完，`metrics` 输出真实复述率/断言率/共情/任务完成度 |
| B-V3 | `reproducibility` | `all_identical`（含负例列表）与 `core_scores_identical`（两个 bool + 两个 score）二次 judge 比对；另全程 `temperature=0` |
| B-V4 | `acceptance["B-V4"]` | 结果 JSON 无 key 明文；无 key 优雅 NOT_RUN |

如果 judge 解析失败，该条记 `judge_error`，**不计入分母也不编造分数**；`n_failed` 如实报告。

## 7. 本次实跑摘要（2026-11-21，真实 hy4-flash-free）

复现命令（在 `MakeAgain-backend` 目录）：

```powershell
python eval/dialogue_judge.py --key-file 'D:\12412\dsh工作区\projects\make-again-mvp\work\local-test-key.txt' --out '...\work\dialogue-judge-final.json'
python eval/dialogue_judge.py --controls-only --key-file 'D:\12412\dsh工作区\projects\make-again-mvp\work\local-test-key.txt'
python eval/dialogue_judge.py --key-file 'D:\不存在\no-key.txt'   # 应输出 状态：NOT_RUN
```

语料 SHA256=`0105125e1fdf215f591fc8ace931d4f1e314f29533d07cd2ee1f650d1cb76d6f`，12 条（breakup/loved_one/pet 各 4）。

### 7.1 B-V1 对照（人工构造好/坏回复，同一 judge）

| 回复 | 重复复述 | 无依据断言 | 共情 | 任务完成 | 负例命中 | 证据 |
|---|---|---|---|---|---|---|
| 好 | false | false | **4/4** | **4/4** | 0 | 「完成共情和温和的开放式邀请，未命令做决定、未诊断、未做无依据承诺」 |
| 坏 | **true** | **true** | **0/4** | **0/4** | **3** | 「逐字重复用户整句两次」「断言‘他根本不爱你…’这些事实和心理状态均未被用户输入或上下文支持」「命令用户放下、别看、向前看」 |

六项判定全 true → **B-V1 PASS**。坏回复 judge `max_tokens_used=4000`（completion 1655），好回复 2000（completion 924）。

### 7.2 B-V2 全量（12 条 case × 真实生成 + judge，成功 12 条，失败 0）

| 指标 | 真实数字 |
|---|---|
| 重复复述率 | **0.0000**（0/12） |
| 无依据断言率 | **0.0000**（0/12） |
| 共情恰当性均分 | **4.0000/4** |
| 任务完成度均分 | **3.8333/4** |
| 负例表述命中率 / 字面命中率 | **0.0000 / 0.0000** |
| 分场景任务完成均分 | breakup 3.7500 / loved_one 3.7500 / pet 4.0000（各 n=4，复述率与断言率均 0） |

→ **B-V2 PASS**。

### 7.3 B-V3 复现性

`temperature=0`；最终全量运行 4 个二次 judge 样本（D-01、D-02、CTRL-GOOD、CTRL-BAD）全部决策字段一致 → PASS。
**诚实边界**：先前一次独立全量运行中，CTRL-BAD 的 `negative_example_hit` 语义列表出现 4 条 vs 3 条的不一致（核心评分字段仍全一致）；说明 opencode-go 即使 temperature=0 也不保证 free-form 列表逐字复现。脚本把该字段单列在 `reproducibility.samples` 并以 `core_scores_identical`/`all_identical` 两个字段区分，不一致时不隐藏。

### 7.4 B-V4 key 安全

`git check-ignore -v work/local-test-key.txt` 命中 `.gitignore`；结果 JSON 明文扫描未发现 key；最终 run 的随机会话头为 `dialogue-judge-fd865e784f61`（非密钥，可写）；无 key 命令实测输出 `状态：NOT_RUN` 且退出码 0。

### 7.5 边界

- judge 是**模型近似评分，不是真人评分**；12 条手写 case 不是真实用户数据（`synthetic-handwritten`），数字不能外推为产品整体质量。
- 共情分仅作内部弱信号，不作产品 KPI（OLD_REPO_QUALITY_METRICS.md §0）。
- 旧仓库建议「模拟来电默认不做、树洞暂缓」等比当前 PRD 保守；产品取舍以当前 PRD 与用户决策为准，本评测只给质量数字。
