# PE-2 Prompt 效果 Golden Set（eval/prompt_golden.py）

> 分支 `prompt/golden`，从 main `26b9393` 建，不合并 main。
> 所有权：仅 `MakeAgain-backend/eval/prompt_golden.py`、`eval/prompt_cases.json`、本 README。
> 目标：把 `agent/breakup.py` L29-L69 的 69 行细则转成**可测行为维度**，用真实 hy4-flash-free 跑出逐维度数字，供 PE-3（A/B）和 PE-4（微调评估）引用。

## 1. 它做什么

全程 `temperature=0`，真实 opencode-go 网关 `hy4-flash-free`（免费额度）：

1. **生成**：用 AST 从 `agent/breakup.py` 抽出真实 `BREAKUP_SYSTEM_PROMPT`（不是重写一个近似 prompt），填入 `SHARED_RULE` / `MICRO_GATE_RULE` / `TONE_PROMPTS[SOOTHE]`，把每条 case 的 `context_block` 填进 `{context}` 占位符；对 24 条手写语料生成回复。
2. **判定**：同一模型充当 judge，按 8 个维度逐维度打 0-5 分 + 一句理由，输出结构化 JSON。每条 case 只对 `applicable_dimensions` 打分，其余维度 score=null / reason=「不适用」，不编造。
3. **好/坏对照（E2-V3）**：8 个维度各一条**人工构造**的好回复 + 坏回复；判据 `good≥4 且 bad≤2 且差值≥2`。
4. **双次复现（E2-V4）**：整套流水线独立跑两遍；分两层核验——(a) 对同一批冻结回复二次判定是否逐字段确定；(b) 两遍 8 维结论（均分≥4.0 判「达标」）是否稳定、均分漂移是否 ≤1.0。

脚本不改业务代码、语料、requirements、PROGRESS.md 或队友仓库。

## 2. 8 个行为维度（从 breakup.py 细则映射）

| ID | 维度 | 规则出处 | 适用面 | 0-5 分核心判据 |
|---|---|---|---|---|
| D1_empathy_first | 先共情后给答案 | L31/L36/L45/L46 | universal | 第一句用 TA 的话/具体事承接，不用「我理解你/抱抱你/这很正常/我懂」垫场；承接后须给观察 |
| D2_no_repetition | 不复述用户原话 | L37/L49/L50 | universal | 0=逐字重复整句；3=引用偏多；5=完全不复述或极短关键词带过 |
| D3_no_filler | 不空话垫场 | L36/L46/L48-L52 | universal | 0=通篇「加油/会好的/时间会治愈/想开点」；5=每句都有具体内容 |
| D4_non_question_ending | 结尾不全是问句 | L31/L36/L44/L47/L53-L57 | universal | 0=连环问/二选一复合问句；4-5=陈述句收尾、零问号 |
| D5_cognitive_reframe | 认知重构 | L41 | conditional | 自动思维/认知扭曲时先共情一句再温柔给「另一面」；不反驳不说教不下诊断不停在反问 |
| D6_victim_protection | 受害者保护 | L41 + L117-L121 | conditional | 被伤害方零归咎，转共情 + 识别不健康关系 + 重建自我 |
| D7_quote_restraint | 引用克制 | L37/L38 | conditional | 不主动翻旧事；不把泛指具体化（上课→烘焙课）；不把旧状态当当下；不写「我看到你…」 |
| D8_length_control | 长度控制 | L31/L48-L52 | universal | 以 payload 的 `char_count` 为准：5=≤120 字，4=120-150，0=>300 |

- `universal` 维度对全部 24 条 case 打分；`conditional` 维度只对以它为主题的 3 条 case 打分并求均分。
- 每条维度的定义、6 档锚点、坏例短语都写在 `prompt_cases.json` 的 `dimensions` 里，judge prompt 由它动态拼出——**单一事实源**，改维度不用改脚本。

## 3. 语料口径（诚实标注）

- `prompt_cases.json`：**24 条 case，8 维 × 3 条**；`source = synthetic-handwritten`。
- **全部为手写模拟语料，非真实用户数据**；场景取自 breakup.py / PRD 的分手陪伴设定，用于评测回复质量，不是采集到的用户原话。
- D5/D6/D7 各有 3 条条件语料：D5 含自动思维/认知扭曲，D6 含被出轨/家暴/PUA 情境，D7 的 `context_block` 故意塞入旧记忆（烘焙课 / 一个月前失眠 / 前任爱喝咖啡+猫豆豆）以触发引用克制测试。
- E2-V3 的好/坏对照回复同样是**人工构造、非模型输出、非真实用户数据**。
- 语料 SHA256 = `22c4ba0ecf3cf854c3c542e90c0f53586dffc5b3b91db5ee50714ffc258bb68f`（写入结果 JSON，可追版本）。
- 生成用的 prompt 模板 SHA256 = `d1ec3322b110c9348cadc7852e94ebb88ccc7af5664ae26acf694d9139dfd104`（2963 字，AST 抽取后填入）。

## 4. 怎么跑

key 只放主仓库 `work/local-test-key.txt`（gitignore），明文不打印不入库。**worktree 里没有该文件**，需显式传路径：

```powershell
cd MakeAgain-backend
python eval/prompt_golden.py --runs 2 --session-id 'prompt-golden-stable-001' `
  --key-file 'D:\12412\dsh工作区\projects\make-again-mvp\work\local-test-key.txt' `
  --out 'D:\12412\dsh工作区\projects\make-again-mvp\work\prompt-golden-final.json'
```

| 参数 | 作用 |
|---|---|
| `--key-file <路径>` | key 文件（默认 `<repo>/work/local-test-key.txt`；worktree 必须显式传主仓库路径） |
| `--runs N` | 流水线遍数（默认 2；2 遍才能验收 E2-V4） |
| `--session-id <id>` | `x-opencode-session`；留空随机。key 文件 REQUIRED_HEADERS 建议稳定 session，实测给固定值 |
| `--model / --judge-model` | 生成 / judge 模型（默认都 `hy4-flash-free`） |
| `--controls-only` | 只跑 8 维好/坏对照（16 次 judge，省钱；**不足以验收 E2-V2**） |
| `--out <路径>` | 结构化 JSON 另存（默认只 stdout；stdout 有 `===JSON===` 分隔） |

脚本设置（明文只在进程内）：`LLM_BASE_URL` / `LLM_API_KEY` 从 key 文件读入，`MOCK_LLM=false`、`USE_INSTRUCTOR=0`、`LLM_EXTRA_HEADERS={"x-opencode-session": "...", "User-Agent": "make-again-prompt-golden/1.0"}`、`HF_HOME=D:\12412\huggingface`。
生成预算 `1000→2000→4000` tokens 递增重试；judge 预算 `3200→6400→8000`（**首次全量跑到 1600 首档时有 6 条 judge JSON 被截断解析失败，已上调首档后重跑，最终两遍均 24/24 成功**）。

## 5. key 安全

- 明文 key 只读入进程内存；**不 print、不写 JSON、不写日志、不入库**。
- 所有异常文本过 `_redact()`（替换完整 key 与 `Bearer xxx`）。
- 输出 JSON 的 `key` 段只记「来源路径 + in_memory_only/printed/committed=false」；退出前扫描结果 JSON 是否含 key 明文，含则判 FAIL。
- 无 key / key 无效 / 网关不可达 / 探针空回复 → **状态 `NOT_RUN`**，退出码 0，不编造数字。
- 实测命令 `python eval/prompt_golden.py --key-file 'D:\不存在\no-key.txt'` → 输出 `状态：NOT_RUN`。

## 6. 验收 E2-V1~V4（真实命令与结果）

复现命令见 §4；证据文件 `work/prompt-golden-final.json` + `work/prompt-golden-final.log`（work/ 已 gitignore，不入本分支）。
运行时间：2026-11-22 03:30 +08:00；模型 hy4-flash-free；temperature=0；两遍独立生成+判定；session `prompt-golden-stable-001`。

### E2-V1 覆盖≥8 维度 — **PASS**

维度表 8 个（D1–D8），每维 3 条 case，judge 输出 8 个维度分数键。命令输出：
`[E2-V1] 维度覆盖：8 个（要求≥8）`。

### E2-V2 真实 hy4 每维度分数（非 NOT_RUN）— **PASS**

两遍均 `生成+判定 24 条，成功 24，失败 0`。逐维度均分（0-5）：

| 维度 | run1 | run2 | 结论 |
|---|---|---|---|
| D1 先共情后给答案 | 4.8333 | 4.8333 | 达标 |
| D2 不复述用户原话 | 3.3333 | 3.1667 | **待改进**（主要短板：模型常把用户原句前半照搬进来，虽然后半有新观察） |
| D3 不空话垫场 | 4.9583 | 4.9167 | 达标 |
| D4 结尾不全是问句 | 5.0000 | 4.8750 | 达标 |
| D5 认知重构（3 条） | 5.0000 | 5.0000 | 达标 |
| D6 受害者保护（3 条） | 5.0000 | 5.0000 | 达标 |
| D7 引用克制（3 条） | 3.6667 | 3.6667 | **待改进**（D7-02 主动翻出「凌晨三点还醒着」旧记忆，被判 1 分） |
| D8 长度控制 | 5.0000 | 5.0000 | 达标（24/24 均 ≤120 字） |

真实例子（judge 理由，run1）：D5「精准点出个人化自责，先承接被困感，再给出不是一人之错和体谅边界的另一面」；D6「不归咎用户，明确出轨是对方选择与背叛，并帮用户卸下不属于自己的责任」。

### E2-V3 judge 区分好/坏 — **PASS（8/8）**

run1 的 8 维好/坏分（good, bad）：D1 (5,1)、D2 (4,0)、D3 (5,0)、D4 (5,0)、D5 (5,0)、D6 (4,0)、D7 (5,0)、D8 (5,0)。
全部满足 good≥4 且 bad≤2 且差≥2。坏回复被判 0 分的典型理由：「逐字重复整句两遍」「通篇空话套话」「归咎用户」「把泛指上课具体化成烘焙课」「char_count 323>300」。
run2 同表仅 D1 为 (4,1)，仍 8/8 分离 → 区分能力本身稳定。

### E2-V4 两次运行可复现 — **部分达成（诚实标注，非全绿）**

`temperature=0`，但实测 opencode-go **生成不逐字复现**：同一输入、同一 client/session 连续 3 次生成得到 3 个不同版本（探针证据 `work/_pg_probe_gen.py`）。因此按两层口径核验，结果：

- **8/8 维结论稳定**：两遍每维「达标/待改进」判定完全一致（上表）。
- **均分最大漂移 0.1666**（0-5 量表，3.3%），远小于设定上限 1.0。
- **judge 对同一冻结回复二次判定**：run1 3/6 逐字段一致，run2 4/6；不一致项全部是主观维度 D3/D4 的 1 分之差（如 `D2-01 D3_no_filler 5→4`、`D2-02 D4_non_question_ending 5→4`），没有维度翻转。
- 逐 case / 逐维度分数**并非逐字一致**（14 条 case 分数有 ±1 微调），差异清单全部写在 JSON 的 `reproducibility.differing_*`，不隐藏。

**判定**：`部分达成/未达成（judge确定=False 结论稳定=True 漂移=0.1666，差异已如实列出）`。
**结论**：评分**结论**可复现，评分**逐字数值**在 temperature=0 的 opencode-go 上不可达；不硬凑 PASS。

## 7. 边界

- judge 是**模型近似评分，不是真人评分**；24 条手写 case 不是真实用户数据（`synthetic-handwritten`），数字不能外推为产品整体质量。
- 生成用 `temperature=0` 为可复现；产品线上 `breakup_chat` 用 `temperature=0.7`，口径不同，已在 `notes` 标注。
- 维度 D7 的低分说明「引用克制」是 prompt 的真实短板（旧记忆注入后模型会主动翻），可直接作为 PE-3 A/B 与 PE-4 微调评估的输入；D2 的低分说明「不复述」同样偏弱。
- conditional 维度样本只有 3 条，均分对单条敏感；不外推到全部对话。
- 本包不修改 prompt 本身，只建立测量口径；改 prompt 后应重跑本脚本对比维度分数（PE-3 的工作）。
