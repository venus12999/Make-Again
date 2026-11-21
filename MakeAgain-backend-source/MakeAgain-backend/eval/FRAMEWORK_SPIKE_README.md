# U-009 记忆框架 3 候选 spike：graphiti / hindsight / MemOS vs 现有 stack

> 分支：`memory-finish/framework-spike`（从 main `29cda50` 建，受管 worktree `framework-spike`）。
> 日期：2026-11-21。工作包：WP-4。基线：main `29cda50`（269 passed）。
> 兑现 D-015 承诺：「选 graphiti / hindsight / MemOS 做真实语料 spike 后定去留」。
> 产物：`eval/framework_spike.py`（本文件同目录）、`outputs/framework-spike-2026-11-21.json`。

---

## 0. 结论（先给，不模棱两可）

**三个候选都不换。保持现有 agent-memory-core + 自研拼装，继续零 LLM、SQLite 本地链路；只按零成本方式借鉴各自最强点。**

| 候选 | 值得换吗 | 一句话理由 | 什么条件下重看 |
|---|---|---|---|
| graphiti | **否** | 最强项是双时间窗失效 + episodes 溯源；前者我们用 recency+supersede 已覆盖主要行为，后者有 provenance_store；其 RRF/MMR/时序规则我们已逐行吸收。换库换来 Neo4j + LLM 税。 | 出现「按任意历史时点查询当时真值」且超出单库能力 |
| hindsight | **否** | LongMemEval SOTA 是真的，但价值来自 LLM 学习闭环（retain→consolidate→reflect），与「先确认后成事实、逐条可溯源纠错」红线冲突；还需 Postgres+pgvector+LLM key。 | 产品明确接受黑盒学习信念且愿意付运维 |
| memos | **否** | 显式 edit/delete + 自然语言反馈最贴用户控制权，但我们已自建 confirm/correct/export；它的 100% 本地 SQLite 是 **npm 插件**（DSH/Hermes/OpenClaw），不是可嵌 FastAPI 后端的库，核心方案要 MySQL 或 Neo4j+Qdrant + LLM。 | 记忆层整体外包给带 GUI 的 agent 运行时 |

**不是"都有道理"的折中**：`final_decision.switch_any = false` 是机器可读的明确判定；三者的 `worth_switching` 全为 `false`。

**更重要的发现**：当前最好配置 `dual1_sem1` 的 Recall@K 只有 **0.5529**，负例注入率 **1.0**（`outputs/golden-recall-v2.json` 既有产物）。真缺口是我们自己的 query-intent / 召回权重问题（对应 WP-3），换框架不会自动修好，反而把 LLM 成本乘以消息数。

---

## 1. 方法与证据等级（不硬凑数字的纪律）

| 等级 | 含义 | 本 spike 是否使用 |
|---|---|---|
| `DOC_LEVEL_PREDICTION` | 由上游 README/源码**逐字声明**推导的机制支持判断；未运行上游系统 | ✅ 18 个 query 全用此级 |
| `MEASURED_INSTALL_ONLY` | 只做 pip 安装 / import，证明可安装性，不证明行为 | ✅ graphiti |
| `NOT_RUN` | 缺服务或密钥，**如实不跑**，不用替身数字冒充实测 | ✅ 三候选端到端全部 |

明确不做的事：
- 不装 Neo4j / Postgres 服务 / MySQL / Milvus 整包；
- 不写 fake LLM / fake embedder 替身来凑召回分数（mem0 那次已说明替身不能裁语义质量）；
- 不改业务代码、requirements、PROGRESS.md、队友仓库、前端。

语料口径（诚实标注）：`eval/recall_corpus.json` 的 **33 记忆 / 18 query** 是 `synthetic-handwritten` 生成者手写模拟语料，标注者=生成者，不是真实用户统计。本文所有「对照」只针对该语料的边界设计。

---

## 2. V1 三候选的真实 gh api 源码证据

所有证据在 `framework_spike.py::EVIDENCE` 中固化为 `id / url / quote / fetch / conclusion`，`python eval/framework_spike.py --verify` 会用 `gh api -H 'Accept: application/vnd.github.raw+json' <fetch>` 重新拉取并逐字校验 quote。实测 **19/19 OK，fail=0**（输出见 §6 V1）。

### 2.1 graphiti（getzep/graphiti，Apache-2.0，32976★，push 2026-11-20）

| id | URL | 关键结论 |
|---|---|---|
| GF-01 | [README#L81](https://github.com/getzep/graphiti/blob/main/README.md#L81) | 定位是 framework 不是成品；多租户/鉴权/入库/运维自建 |
| GF-02 | [README#L120-L136](https://github.com/getzep/graphiti/blob/main/README.md#L120-L136) | 事实有有效窗口，旧事实失效不删除——正对养猫→猫去世 |
| GF-03 | [README#L125-L128](https://github.com/getzep/graphiti/blob/main/README.md#L125-L128) | Episodes 溯源到原始真源 |
| GF-04 | [extract_edges.py#L60-L69](https://github.com/getzep/graphiti/blob/main/graphiti_core/prompts/extract_edges.py#L60-L69) | `EdgeTimestamps.valid_at/invalid_at` 显式字段 |
| GF-05 | [edge_operations.py#L438-L460](https://github.com/getzep/graphiti/blob/main/graphiti_core/utils/maintenance/edge_operations.py#L438-L460) | 真实 invalidate/contradiction 消解，不只是相似度排序 |
| GF-06 | [search_utils.py#L1551](https://github.com/getzep/graphiti/blob/main/graphiti_core/search/search_utils.py#L1551) | RRF/MMR 上游实现——我们已在 `memory/recall.py` 逐行复制 |
| GF-07 | [pyproject.toml#L13-L16](https://github.com/getzep/graphiti/blob/main/pyproject.toml#L13-L16) | 硬依赖 `neo4j>=5.26.0` + OpenAI key（README Installation） |

### 2.2 hindsight（vectorize-io/hindsight，MIT，62372★，push 2026-11-21）

| id | URL | 关键结论 |
|---|---|---|
| HS-01 | [README#L46-L52](https://github.com/vectorize-io/hindsight/blob/main/README.md#L46-L52) | LongMemEval SOTA，README 声明第三方复现（Virginia Tech / Washington Post） |
| HS-02 | [README#L320-L328](https://github.com/vectorize-io/hindsight/blob/main/README.md#L320-L328) | 语义+BM25+图+时间四路并行，RRF + cross-encoder 重排 |
| HS-03 | [README#L350-L352](https://github.com/vectorize-io/hindsight/blob/main/README.md#L350-L352) | observation 带证据引用+证明计数，新证据 refine 旧信念而非覆盖 |
| HS-04 | [README#L385](https://github.com/vectorize-io/hindsight/blob/main/README.md#L385) | bank 强隔离：`Isolation is strict: no cross-bank leakage.` |
| HS-05 | [README#L86](https://github.com/vectorize-io/hindsight/blob/main/README.md#L86) | `Docker (external PostgreSQL)` |
| HS-06 | [compose#L18-L20](https://github.com/vectorize-io/hindsight/blob/main/docker/docker-compose/external-pg/docker-compose.yaml#L18-L20) | `pgvector/pgvector:pg` + `HINDSIGHT_API_LLM_API_KEY` 必填 |

### 2.3 MemOS（MemTensor/MemOS，Apache-2.0，12470★，push 2026-11-20）

| id | URL | 关键结论 |
|---|---|---|
| MS-01 | [README#L51](https://github.com/MemTensor/MemOS/blob/main/README.md#L51) | 统一 API：add / retrieve / edit / delete |
| MS-02 | [README#L55](https://github.com/MemTensor/MemOS/blob/main/README.md#L55) | Memory Feedback & Correction：自然语言反馈纠错 |
| MS-03 | [README#L110-L115](https://github.com/MemTensor/MemOS/blob/main/README.md#L110-L115) | 自托管 `Neo4j + Qdrant` |
| MS-04 | [README#L110-L115](https://github.com/MemTensor/MemOS/blob/main/README.md#L110-L115) | Local Plugin `None (local SQLite)`，但面向 DSH/Hermes/OpenClaw 的 npm 插件 |
| MS-05 | [pyproject.toml#L38-L45](https://github.com/MemTensor/MemOS/blob/main/pyproject.toml#L38-L45) | 核心依赖 `pymysql`——「要 MySQL」不是误解 |
| MS-06 | [pyproject.toml#L102](https://github.com/MemTensor/MemOS/blob/main/pyproject.toml#L102) | 可选 `pymilvus` / neo4j / qdrant，至少再起 1~2 服务 |

> 前置描述的「MemOS 要 MySQL/Milvus」经查**两种说法都对**：核心 pyproject 含 pymysql + 可选 pymilvus（MS-05/06）；README 自托管矩阵写 Neo4j+Qdrant（MS-03）。已按源码分别记录，不挑一个当结论。

---

## 3. V2 用我们的冲突/召回语料逐项对照

脚本读取真实 `eval/recall_corpus.json`，为 **18/18 条 query** 各生成一行 `case_matrix`，每行含三候选的 `support_level`（EXPLICIT_DOC / PARTIAL_DOC / NO_DOC_CLAIM）、机制说明、证据 id，且**每行 `execution_status = NOT_RUN`、`evidence_grade = DOC_LEVEL_PREDICTION`**。

汇总（脚本实测输出）：

```
[support_summary] {'graphiti': {'PARTIAL_DOC': 6, 'EXPLICIT_DOC': 5, 'NO_DOC_CLAIM': 7},
                   'hindsight': {'PARTIAL_DOC': 15, 'EXPLICIT_DOC': 2, 'NO_DOC_CLAIM': 1},
                   'memos': {'PARTIAL_DOC': 8, 'NO_DOC_CLAIM': 10}}
```

### 3.1 养猫→猫去世冲突用例（重点）

| query | 我们的期望 | graphiti（文档层） | hindsight（文档层） | memOS（文档层） |
|---|---|---|---|---|
| q-02 `现在养猫吗？` | 只召回 `mem-cat-deceased`，旧 `mem-cat-owned` 不召回 | **EXPLICIT**：`invalid_at` 自动失效旧边，当前真值只剩新事实（GF-02/05） | PARTIAL：observation refine 旧信念，但「是否只返新事实」无承诺（HS-03） | PARTIAL：有 edit/delete，但要调用方自己决定何时改（MS-01） |
| q-03 `我养的猫还在吗？` | 新事实排旧事实之前 | **EXPLICIT**：bi-temporal 区分 valid time（GF-02/04） | PARTIAL：temporal 路 + 重排有利，无排序承诺（HS-02） | NO_DOC_CLAIM |
| q-17 `橘子埋在哪里？` | 新事实含埋葬地点为主答 | EXPLICIT（同上 + 溯源到 episode） | PARTIAL | PARTIAL |
| q-15 `我现在有对象吗？` | 现伴侣命中、前任不召回 | **EXPLICIT**：同一关系可建不同 valid_at 边（GF-02/05） | PARTIAL：mental model 重写不可逐事实审计（HS-03） | PARTIAL：edit/feedback 需人工触发（MS-02） |

**判定**：文档层 graphiti 对冲突类最强；但「文档层支持」≠「我们语料上实测更好」。我们零 LLM 的 supersede + recency 已在 q-02/q-03 给出正确排序（既有语料标注与 baseline 口径），换库收益未量化。

### 3.2 其它边界类（逐类结论）

| 边界 | query | 三候选文档层表现 | 对换库决策的影响 |
|---|---|---|---|
| 相似主题干扰（6 条：q-05/08/09/11/14/18） | 爷爷手表 vs 外公眼镜、用户猫毛过敏 vs 前任狗毛过敏等 | graphiti NO_DOC_CLAIM；hindsight PARTIAL（cross-encoder 重排）；memOS NO_DOC_CLAIM | **没有一个承诺主体消歧**；这是语义模型能力问题，不是框架选型问题 |
| 时间敏感（q-06 去年 / q-13 周年） | 固定 now=2026-11-21 | graphiti EXPLICIT（双时间窗）；hindsight EXPLICIT（temporal 路）；memOS NO_DOC_CLAIM | graphiti/hindsight 在此维度真有内建能力，但仍未跑 |
| 无关记忆负例（q-16） | 晚饭吃什么，期望不召回 | **三家全部 NO_DOC_CLAIM**（RAG 系统普遍返回 top-k） | 我方实测负例注入率 **1.0**，这是自己的 query-intent 缺口（WP-3），换哪家都挡不住 |

> 完整 18 行逐项表在 JSON 的 `case_matrix`；重点四组在 `cross_case_focus`。

---

## 4. 差距分析 + 迁移代价（我们现有实现 vs 候选）

现有 stack：`agent-memory-core + 自研拼装` —— SQLite 五层库（Source/Candidate/Fact/Event/Summary/Index）、确定性 hash embedding、recency/importance/relevance 打分、双路召回（vec0 + bm25s）、RRF/MMR、confirm/correct/export/isolation、`provenance_store`，**写入 0 LLM / 检索 0 LLM**（见 `docs/MEM0_EVAL.md` 实测口径）。

| 维度 | graphiti | hindsight | memOS |
|---|---|---|---|
| 要新起的服务 | Neo4j/FalkorDB + LLM + embedder | Postgres+pgvector + LLM（+可选重排模型） | MySQL 或 Neo4j+Qdrant + LLM |
| 写入 LLM 税 | 每 episode/边抽取要 LLM | retain 抽取 + consolidate 要 LLM | add 要 LLM |
| 我们已吸收的部分 | RRF/MMR/normalize_l2、valid_at/invalid_at prompt 规则（已在 recall.py/temporal.py） | 无（可借鉴 observation 证据计数） | 无（可借鉴显式 edit/feedback 语义） |
| 要重做的链路 | store/recall/confirm/provenance + 部署 + 数据迁移 | add/recall 全入口、五层库压平、确认流/纠错流/删除级联全废 | FastAPI 无法直接用 npm 插件；控制权 3 路由迁 edit/feedback；L1/L2/L3 与五层映射 |
| 不可逆风险 | 图库 schema 与 LLM 抽取结果难回滚 | 「模型学习出的信念」不可逐条审计，撞产品红线 | 本地插件与自托管核心两套数据模型，选错入口代价高 |
| 共同代价 | 三家都要：重做写入/召回/确认/纠错/删除链路、数据双写迁移、**重跑我们的 18 query 评测**（LongMemEval 分数不能迁移） | 同左 | 同左 |

---

## 5. NOT_RUN 清单与原因（不硬凑）

| 项 | 状态 | 原因 |
|---|---|---|
| 三候选端到端行为 spike（在 33/18 语料上真实 add+recall） | **NOT_RUN** | 写入链路全要 LLM 抽取 + 外部数据库；本机无任何 LLM key（OPENAI/ANTHROPIC/DEEPSEEK/HINDSIGHT 全 ABSENT）、无 Neo4j(7687 CLOSED)/MySQL(3306 CLOSED)。5432 虽有端口但不是本项目 pgvector 实例且无 key。 |
| hindsight pip 安装 + 本地跑 | **NOT_RUN** | 需 Postgres+pgvector 服务 + `HINDSIGHT_API_LLM_API_KEY`（HS-05/06）；不装整包。 |
| MemOS 自托管跑 | **NOT_RUN** | 核心要 MySQL 或 Neo4j+Qdrant + LLM（MS-03/05/06）；本地 SQLite 版是 npm 插件不适用于本后端。 |
| 语义相关性公平裁定 | **NOT_RUN** | 无真实 LLM/embedding；不写替身分数。 |
| graphiti 行为实测 | **NOT_RUN**（只做 `MEASURED_INSTALL_ONLY`） | pip 可装（0.31.2），首次 `import graphiti_core` 报缺 `httpx`，手动装后 import OK；但这只证明可安装，不证明行为。上游依赖声明缺传递依赖是实测事实。 |

前置检查只记录存在性、不记任何 key 明文。

---

## 6. 验收逐条：真实命令 + 输出

工作目录 `D:/12412/dsh工作区/projects/make-again-mvp`，受管 worktree `D:\12412\DshWorktrees\make-again-mvp\framework-spike`（分支 `memory-finish/framework-spike`）。PowerShell，Python 3.12.9。

### V1 三个候选都有真实 gh api 源码证据（URL + 关键结论）

命令：
```powershell
cd 'D:\12412\DshWorktrees\make-again-mvp\framework-spike\MakeAgain-backend'
python eval/framework_spike.py --verify
Write-Output "VERIFY_EXIT=$LASTEXITCODE"
```
真实输出（19 行逐条 OK，末尾汇总）：
```
EVIDENCE GF-01 OK          https://github.com/getzep/graphiti/blob/main/README.md#L81
...
EVIDENCE GF-06 OK          https://github.com/getzep/graphiti/blob/main/graphiti_core/search/search_utils.py#L1551
...
EVIDENCE HS-02 OK          https://github.com/vectorize-io/hindsight/blob/main/README.md#L320-L328
...
EVIDENCE MS-04 OK          https://github.com/MemTensor/MemOS/blob/main/README.md#L110-L115
...
VERIFY_RESULT ok=19 fail=0
VERIFY_EXIT=0
```
（完整 19 行见 §2 三张证据表和 JSON 的 `evidence` 字段。）

### V2 用我们的冲突/召回用例逐项对照

命令：`python eval/framework_spike.py --check`
```
[corpus] memories=33 queries=18
[case_matrix] rows=18 all_execution_status=NOT_RUN
[support_summary] {'graphiti': {'PARTIAL_DOC': 6, 'EXPLICIT_DOC': 5, 'NO_DOC_CLAIM': 7}, 'hindsight': {'PARTIAL_DOC': 15, 'EXPLICIT_DOC': 2, 'NO_DOC_CLAIM': 1}, 'memos': {'PARTIAL_DOC': 8, 'NO_DOC_CLAIM': 10}}
[self_check] errors=[]
CHECK_EXIT=0
```
自检强制 `case_matrix` 的 query id 序列与 corpus 完全一致（18/18），并强制每行的执行状态为 `NOT_RUN`——有硬凑数字即报错退出。

### V3 明确去留结论（不是「都有道理」）

同一命令输出：
```
[decision] KEEP_CURRENT_STACK_SELECTIVE_BORROW switch_any=False
```
JSON 校验命令：
```powershell
python -c "import json,pathlib; d=json.loads(pathlib.Path('../outputs/framework-spike-2026-11-21.json').read_text(encoding='utf-8')); print('verdict=',d['final_decision']['verdict']); print('switch_any=',d['final_decision']['switch_any']); print('per_candidate=',{k:v['worth_switching'] for k,v in d['final_decision']['per_candidate'].items()})"
```
输出：
```
verdict= KEEP_CURRENT_STACK_SELECTIVE_BORROW
switch_any= False
per_candidate= {'graphiti': False, 'hindsight': False, 'memos': False}
```
`self_check` 还强制三者 verdict 一致为 `DO_NOT_SWITCH`；若有人改成「各有道理」的含糊结论，脚本 exit 1。

### V4 不硬凑数字，装不了标 NOT_RUN

生成命令：`python eval/framework_spike.py`
```
[NOT_RUN] end_to_end=NOT_RUN
[self_check] errors=[]
[write] outputs\framework-spike-2026-11-21.json bytes=45943
GEN_EXIT=0
```
JSON 中 `runtime_prerequisites.end_to_end_behavior = "NOT_RUN"`，`lightweight_checks` 三候选分别标 `MEASURED_INSTALL_ONLY` / `NOT_RUN` / `NOT_RUN`，18 行 case 全部 `execution_status=NOT_RUN`。

### V5 越权 / 边界测试（必含）

命令与真实输出：
```powershell
# 1) 脚本只读 corpus+baseline，唯一写目标已断言
[scope] read=['MakeAgain-backend\\eval\\recall_corpus.json', 'outputs\\golden-recall-v2.json']
[scope] write=outputs\framework-spike-2026-11-21.json

# 2) --check 不得改写产物：SHA256 前后比对
OUTPUT_SHA256_BEFORE=0248B493902F13E7F62145C86AF51525D8DCB4BD5A20BEFCE3246ED12EB1F062
OUTPUT_SHA256_AFTER=0248B493902F13E7F62145C86AF51525D8DCB4BD5A20BEFCE3246ED12EB1F062
UNCHANGED=True

# 3) worktree 只出现分配文件，git diff 无已跟踪文件改动
git status --short
  ?? MakeAgain-backend/eval/FRAMEWORK_SPIKE_README.md
  ?? MakeAgain-backend/eval/framework_spike.py
  ?? outputs/framework-spike-2026-11-21.json
git diff --stat
  (空)
```
补充边界：`self_check` 校验输出路径必须是 `outputs/framework-spike-YYYY-MM-DD.json`，否则 exit 1（防止写到队友文件/台账/requirements）。脚本不 import 任何 `memory.*` 业务模块。

---

## 7. 复现命令

```powershell
$wt = 'D:\12412\DshWorktrees\make-again-mvp\framework-spike'
cd "$wt\MakeAgain-backend"

python eval/framework_spike.py --verify   # V1：gh api 复验 19 条证据
python eval/framework_spike.py --check    # V2/V3/V5：自检，不写文件
python eval/framework_spike.py            # V4：生成 outputs/framework-spike-2026-11-21.json
cd $wt; git status --short                # V5：只应看到 3 个分配文件
```

依赖：Python 3.12.9 标准库 + `gh` CLI（已登录，token 只用于读公开仓库；命令中不出现任何 key 明文）。无第三方包要求。

---

## 8. 本次改动文件（严格所有权）

- `MakeAgain-backend/eval/framework_spike.py`（新建）
- `MakeAgain-backend/eval/FRAMEWORK_SPIKE_README.md`（新建，本文件）
- `outputs/framework-spike-2026-11-21.json`（新建，由脚本生成）

未改业务代码、未改 `requirements`、未改 `PROGRESS.md`、未碰队友仓库与前端；未提交密钥。提交信息：
`eval: framework spike - graphiti/hindsight/MemOS vs current stack`。
