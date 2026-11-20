# Golden Set 评测脚本（eval/run_golden.py）

> 分支：`eval/golden-runner`（从 main `7b66139` 建，不合并 main）。
> 所有权：仅 `MakeAgain-backend/eval/run_golden.py` + 本 README。
> 指标口径来自 `docs/OLD_REPO_QUALITY_METRICS.md`（旧仓库 V2 文档），**不自创指标**。

## 1. 它做什么

读 `eval/golden_corpus.json`（任务 A，分支 `eval/golden-corpus`，commit `7802183`），跑四个能力的评测：

| 能力 | 实现 | 本轮状态 |
|---|---|---|
| 危机检测 | `emotion.crisis.detect()` 全语料，算漏报率/误报率，按 真实危机/否定/第三人称/引用/非危机基线 分组 | **RUN（纯本地，零 LLM）** |
| 情绪分类 | `emotion.classifier.classify()`：mock client 离线必跑；真实 hy4 仅在 key 文件存在且 env 配好时跑并单列 | mock **RUN** / real **NOT_RUN**（见 §6） |
| 记忆召回 | 临时 SQLite `memory_stream` + vec0/bm25s 索引，跑 `memory.recall()`，四组 A/B 配置 | **RUN** |
| 对话质量 | 重复复述率 / 无依据断言率 / 任务完成率 | **NOT_RUN**（需真实模型 + LLM judge，不硬凑） |

输出：`outputs/golden-<YYYY-MM-DD>.json`（结构化数字，UTF-8，`sort_keys=True`）+ 人读报告打印到 stdout。
脚本不修改业务代码、语料、requirements、PROGRESS.md 或队友仓库。

## 2. 怎么跑

语料文件来自任务 A 的分支，默认路径 `MakeAgain-backend/eval/golden_corpus.json`。若当前分支没有它：

```powershell
# 在仓库根执行（只读引用任务 A 的提交，不改其文件）
git show eval/golden-corpus:MakeAgain-backend/eval/golden_corpus.json |
  Set-Content -LiteralPath 'MakeAgain-backend/eval/golden_corpus.json' -Encoding UTF8
```

运行：

```powershell
cd MakeAgain-backend
python eval/run_golden.py
# 可选项：
#   --corpus <路径>     指定语料 JSON
#   --top-k 3           召回 K（默认 3 = fixture 条数）
#   --out-dir <目录>    输出目录（默认 <repo>/outputs）
```

文件不存在时**不会崩**：打印清晰错误并以退出码 2 结束，提示从 `eval/golden-corpus` 取回或显式传 `--corpus`。

脚本自己设置（无需手配）：`DATABASE_URL`（指向系统临时目录里的 SQLite）、`DUAL_RECALL_INDEX_DB`、
`ENVIRONMENT=test`、`MOCK_LLM=true`、`HF_HOME=D:\12412\huggingface`、`HF_HUB_OFFLINE=1`。
依赖：`sqlite-vec` / `bm25s` / `jieba` / `fastembed` / `sqlalchemy` / `openai` / `pydantic-settings`（均已在 requirements 中）。
真实语义模型 `Qdrant/bge-small-zh-v1.5` 从本机 `D:\12412\huggingface` 离线加载；加载失败时自动回退 hash，报告会如实标注 `embedding_kind_actual`。

## 3. 指标口径（全部来自旧仓库 V2 文档）

来源：`docs/OLD_REPO_QUALITY_METRICS.md` §1（architecture.md §12.2）、§2（legacy-assessment.md §6）。

| 指标 | 公式 | 说明 |
|---|---|---|
| 危机漏报率 | `FN / 期望危机数` | 安全指标：期望 `is_crisis=true` 但 `detect()` 未检出 |
| 危机误报率 | `FP / 期望非危机数` | 否定/第三人称/引用语境被误判为危机 |
| 情绪标签准确率 | `pred ∈ emotion_expect.accepted 的条数 / 有效条数` | 多解语料按可接受集合判定；**内部弱信号，不作产品 KPI** |
| valence 方向一致率 | `negative→<0, positive→>0, mixed→abs(v)<=0.2` 的比例 | 辅助诊断，非 KPI |
| Recall@K | `命中期望记忆数 / |memory_expect|` | 记忆检索主指标 |
| Precision@K | `命中期望记忆数 / K` | K=3；fixture 池只有 3 条，故单条期望的查询分母偏大 |
| MRR | `mean(1 / 首个相关结果的排名)` | 未召回记 0 |
| 冲突新事实排前率 | `有序 memory_expect 中第一项排在检索第一位的比例` | 旧仓库"纠错后旧事实不再被召回"口径的本地简化版 |

「对话质量」三项只在报告里列 NOT_RUN，不生成任何替代数字。

## 4. 语料与记忆 fixture 口径（诚实标注）

- 语料共 27 条：`breakup 9 / loved_one 10 / pet 8`。
- `source=PRD-derived` 10 条：依据 PRD 改写为用户口吻，**不是采集到的真实用户原话**；L-04 为 PRD 原文逐字引用。
- `source=synthetic-handwritten` 17 条：**手写模拟语料，非真实用户数据**。
- 记忆库 3 条 fixture 的 id/content/summary/facts/importance/ts 硬编码在脚本 `MEMORY_FIXTURES`，逐字取自
  `eval/CORPUS_README.md` §4（任务 A）；P-06 的"昨天"已改为绝对日期 `2026-11-20`。
  这 3 条同样是**手写模拟语料配套的记忆库**，不是真实用户数据。
- 危机分组中「引用」组期望非危机，但当前硬匹配命中 L-06 的"想死" → 真实误报，报告如实给出。

## 5. A/B 决策规则（计划 §4，写死在 `ab_decisions()`）

| 开关 | 阈值 | 无效/不足时 |
|---|---|---|
| `USE_LLM_EMOTION` | 真实 hy4 准确率 − mock 准确率 ≥ +10pp → 建议打开；否则建议保持关闭 | real NOT_RUN → 不决策，列 U-008 |
| `USE_DUAL_RECALL` | `Recall@K` +0.1 以上，且冲突新事实排前不退化、危机漏报不增加 → 建议打开 | 配置未跑全 → 不决策 |
| `USE_SEMANTIC_EMBEDDING` | 同上，在 `dual=1` 下比较 sem1 vs sem0 | sem1 实际回退 hash → 不决策；`dual=0` 路径不用 embedding，sem 为 no-op，不参与决策 |
| 危机硬门槛 | 无开关；任何 A/B 若使漏报增加即应保持关闭 | — |
| `USE_MEM_PROMPTS` | 对话质量指标 | NOT_RUN → 不决策，列 U-008 |

最终开不开是产品决策；评测只给建议和数字。

## 6. 真实 hy4 情绪评测（默认 NOT_RUN）

任务条件是「`work/local-test-key.txt` 存在 **且** env 配好」。当前工作树里该文件不存在（gitignore，未随分支带过来），
env 也未配置 `LLM_API_KEY`/`OPENCODE_GO_API_KEY`，所以报告标 **NOT_RUN**，没有编造数字。

需要跑时，任选其一（明文 key 不打印、不写入 JSON）：

```powershell
# 方式 A：显式让脚本从 key 文件读取（会解析 KEY=/BASE_URL=/MODELS=）
$env:GOLDEN_REAL_EMOTION = '1'
python eval/run_golden.py

# 方式 B：外部把 env 配好
$env:LLM_API_KEY = '<key>'
$env:LLM_BASE_URL = 'https://opencode.ai/inference/go/openai/v1'
python eval/run_golden.py
```

真实路径若全部调用失败，状态为 `ENV_FAIL`；若部分调用回退词典（`reason=关键词词典兜底`），
报告会给出 `lexicon_fallback_count`，可据此判断真实分类是否真的生效。

## 7. 可复现性

- 语料 `sha256` 写入 JSON；固定 `now=2026-11-21T12:00:00`（patch `memory.recall.datetime`）消除 recency 漂移。
- 临时索引路径写入结果时归一为 `<temp>/golden.vecidx.sqlite3`，避免随机临时目录名破坏字节一致。
- 每组 A/B 前 `_reset_semantic_state()` + `reset_index()` + 删旧索引文件，再按当前 embedding 口径重建；
  64 维 hash 表 `memory_vec` 与 512 维 `memory_vec_d512` 相互隔离，不混用。
- 实测：同一环境连续两次运行，输出 JSON 的 SHA256 均为
  `BF41E6FB7B493ED372F60DF793005812A9D9FF4E3A7BB78A2F6B68E8551BB361`，stdout 也字节一致。

## 8. 本次实跑摘要（2026-11-21，命令见 §2）

- 危机：n=27，期望危机 3，预测危机 4，TP=3 / FP=1 / FN=0 / TN=23；漏报率 **0.0000**，误报率 **0.0417**；
  误报全部来自"引用"组（L-06，1/1）。
- 情绪 mock：n=27，标签准确率 **0.4444**，valence 方向一致 0.1481，词典回退 27/27。
  真实 hy4：**NOT_RUN**（见 §6）。
- 召回（K=3）：四组配置 Recall@K 均 **1.0000**、Precision@K 均 **0.5000**；
  MRR：dual0=0.7500、dual1=1.0000；冲突新事实排前率均 **1.0000**；
  embedding 实测按开关正确切换为 `deterministic_hash_nonsemantic` / `semantic_bge_small_zh`。
- 决策：USE_LLM_EMOTION 不决策（数据不足）；USE_DUAL_RECALL 与 USE_SEMANTIC_EMBEDDING 均"建议保持关闭"
  （Recall@K 无 +0.1 增益）；USE_MEM_PROMPTS 不决策。

**边界提醒**：27 条语料、2 条召回查询、3 条 fixture 只是 smoke-level 的 golden set，数字只能作为回归基线，
不能外推为产品整体质量；17 条手写语料不得称为真实用户数据。
