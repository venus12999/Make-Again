# Recall Golden Corpus V2 说明（RECALL_CORPUS_README）

> 分支：`eval/golden-recall-v2`。基线：main `536e966`。日期：2026-11-21。
> 语料文件：`MakeAgain-backend/eval/recall_corpus.json`（与本文同目录）。
> 目的：把 V1「3 fixture / 2 query、Recall@K 全 1.0」的天花板效应打破，用 30+ 记忆 × 15+ query 的
> 分级标注，支撑 Recall@K / Precision@K / MRR / 冲突新事实排前等指标的量化评测。
> **本包只产出语料与说明，不改业务代码、不改 eval/run_golden.py、不写 outputs、不碰台账。**

---

## 1. 语料概况

| 项 | 数值 |
|---|---|
| 记忆条目数 | **33**（`memories`） |
| 场景分布 | breakup **11** / loved_one **13** / pet **9** |
| Query 数 | **18**（`queries`） |
| 相关度标注数 | **63**（high **21** / medium **17** / irrelevant **25**） |
| 来源口径 | `synthetic-handwritten` **100%**（全部为生成者手写模拟语料） |
| 固定基准时间 | `2026-11-21T12:00:00`（与 `eval/run_golden.py` 的 `FIXED_NOW` 一致） |
| 评测用户 | `golden-eval-user`（与 `run_golden.py` 的 `EVAL_USER` 一致） |

### 来源口径（诚实标注，务必遵守）

- 本语料 **100% 是 `synthetic-handwritten`**：由本工作包生成者手写，**不是真实用户数据、不是采集原话**。
  记忆内容虽为「真实风格中文」（口语化、具体、不造作），但**「风格真实」≠「数据真实」**。
- 禁止在任何报告/图表中把这 33 条记忆或 18 条 query 称为真实用户数据或真实用户统计。
- 语料只用于记忆召回排序评测；不涉及情绪标签、危机判定、对话质量（那三项属 V1 golden set 与 V2-B）。

---

## 2. 数据结构与字段字典

顶层是一个对象：`meta` / `memories` / `queries`。

### 2.1 `memories[]` 每条 8 字段（可直接映射 `MemoryEntry`）

| 字段 | 类型 | 说明 / 映射 |
|---|---|---|
| `id` | string | 唯一编号；`mem-loved-one-father` / `mem-cat-owned` / `mem-cat-deceased` 沿用 V1 fixture 名，其余 `mem-b-*` / `mem-l-*` / `mem-p-*` |
| `content` | string | 记忆正文（口语化中文）→ `MemoryEntry.content` |
| `summary` | string | 一行摘要 → `MemoryEntry.summary` |
| `facts` | string[] | 抽取的事实列表 → `MemoryEntry.facts` |
| `importance` | float | 重要性（0.35~0.95，覆盖低/中/高）→ `MemoryEntry.importance` |
| `ts` | string | 绝对 UTC 时间 ISO8601（`Z` 结尾），**禁用「昨天/上周」相对词** → `MemoryEntry.ts` |
| `type` | enum | `chat` / `event` / `report`，取自 `memory/models.py` 的 `MemoryEntry.type` 枚举 |
| `scene` | enum | `breakup` / `loved_one` / `pet`，语料组织维度（不是 ORM 字段，由 corpus 侧携带） |

### 2.2 `queries[]` 每条 7 字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | string | 唯一编号 `q-01` ~ `q-18` |
| `memory_query` | string | 提问原文。**字段名与 `run_golden.py::recall_metrics` 读取的键一致**，便于直接适配 |
| `relevance` | object | `{mem_id: "high"\|"medium"\|"irrelevant"}`：分级相关度。high=应作为答案主召回；medium=可作上下文；irrelevant=不应召回（含被 supersede 的旧事实、主体不符的干扰项） |
| `memory_expect` | string[] | **二值相关集**（high + medium 的 id，有序）。字段名与 `recall_metrics` 一致；第一项应排最前。**irrelevant 的 id 绝不进此列表** |
| `expected_first` | string\|null | 期望排第一的 id（冲突/排序用例据此判定）；负例 query 为 `null` |
| `boundary` | string\|null | 边界类型标记，见 §3；普通 query 为 `null` |
| `note` | string | 判定理由，供人工复核 |

> 设计要点：`memory_expect` 与 `relevance` 并行提供——前者是给现有二值指标（Recall/Precision/MRR）直接吃的「平面视图」，
> 后者是分级真值（支撑 nDCG，见 §5 的 gap）。二者一致性由校验命令强制（expect ⊆ {high,medium}；负例 expect 为空且无 high/medium）。

---

## 3. 边界用例清单

| query id | 边界类型 | 语料要点 | 期望 |
|---|---|---|---|
| q-02 | 冲突事实-只召回新事实 | `现在养猫吗？`；猫橘子 2026-11-20 已去世 | `memory_expect=[mem-cat-deceased]`；`mem-cat-owned` 标 `irrelevant`（被新事实 supersede） |
| q-03 | 冲突事实-新事实排前 | `我养的猫还在吗？` | 新事实 `mem-cat-deceased` 必须排在旧事实 `mem-cat-owned` 之前 |
| q-15 | 冲突事实-新旧关系 | `我现在有对象吗？` | 现伴侣 `mem-b-06` 命中；前任 `mem-b-01` 标 irrelevant |
| q-05 / q-08 / q-09 / q-11 / q-14 / q-18 | 相似主题干扰 | 狗 vs 猫、前任 vs 现伴侣、奶奶 vs 爷爷 vs 外公、用户过敏 vs 前任过敏 | 主体不符的干扰项标 irrelevant，不进 `memory_expect` |
| q-06 | 时间敏感-去年 | `去年我经历了什么？`；固定 now=2026-11-21 → 去年=2025 | 2026 年的外公周年祭、猫去世不应召回 |
| q-13 | 时间敏感-周年 | `最近有什么周年纪念？` | 外公周年(2026-06)、分手周年(2026-07)命中；普通去世日记标 irrelevant |
| q-16 | 无关记忆-负例 | `今天晚饭吃什么？` | 与所有记忆无关；`memory_expect=[]`，期望 top-k 不命中任何记忆（见 §5 gap） |

补充：记忆侧还埋了干扰项本身——`mem-b-09`（前任狗毛过敏、非用户养宠）、`mem-l-04`（爷爷手表，vs 外公眼镜）、
`mem-p-01`（用户的狗，vs 猫）、`mem-p-06`（用户猫毛过敏，主体是用户）——用于跨 query 复用检验「相似主题不混召」。

---

## 4. 标注方法与主观性声明

- **标注者 = 生成者**：本工作包作者既写了 33 条记忆和 18 条 query，也做了相关度分级。
  这是**单人自标注**，没有第二标注者、没有 Cohen's kappa 一致性检验，**存在主观性与确认偏差**。
- 标注流程（人工复核一遍）：
  1. 对每条 query，先写出「如果产品答得好，应该召回哪些记忆」→ 定 high；
  2. 再列「可作补充上下文但不算答案」→ 定 medium；
  3. 最后把「主题相近但主体/时间/新旧状态不符」的记忆显式列为 irrelevant（这一步专门卡干扰项与冲突事实）；
  4. 把 high+medium 复制到 `memory_expect` 并排序，冲突/排序用例把必须靠前的 id 设为 `expected_first`；
  5. 用 §5 校验命令检查 expect ⊆ {high,medium}、id 存在性、负例一致性。
- **口径限制**：分级是「生成者认为的相关」，不是用户实测反馈；`medium` 与 `irrelevant` 的边界在个别条目上可能有争议
  （例如 q-05 的鹦鹉/金鱼算不算「养过的宠物」主召回）。这类争议已在各条 `note` 里写明判定理由，便于后续第二人复核时调整。
- 时间词严格按 `fixed_now=2026-11-21T12:00:00` 解释；相对时间只在 query 里出现，记忆 `ts` 一律绝对时间。

---

## 5. 如何接 recall eval（`eval/run_golden.py`）

### 5.1 二值指标（现有函数，直接可用）

`run_golden.py::recall_metrics(corpus, recall_fn, top_k)` 只读取每条的 `id` / `memory_query` / `memory_expect`。
把本语料的 `queries` 映射成平面 corpus 即可：

```python
import json, pathlib
from eval.run_golden import recall_metrics

d = json.loads(pathlib.Path('eval/recall_corpus.json').read_text(encoding='utf-8'))
flat = [{"id": q["id"], "memory_query": q["memory_query"],
         "memory_expect": q["memory_expect"]} for q in d["queries"]]
# recall_fn: (query, top_k=...) -> [{"entry": MemoryEntry}, ...]
metrics = recall_metrics(flat, recall_fn, top_k=5)
```

`build_memory_db(init_db, SessionLocal, MemoryEntry)` 当前硬编码 V1 的 3 条 fixture；改用它时按本语料 `memories`
逐条 `db.add(MemoryEntry(id=m["id"], user_id="golden-eval-user", type=m["type"], content=m["content"],
summary=m["summary"], facts=m["facts"], importance=m["importance"], ts=parse_ts(m["ts"])))` 即可
（字段与 `MemoryEntry` 一一对应，见 §2.1）。**本包不改该文件，只给出接法。**

### 5.2 A-V4 指标覆盖对照（recall eval 已实现的全部指标）

| recall eval 已实现指标 | 本语料提供的字段 | 是否覆盖 |
|---|---|---|
| Recall@K | `memory_expect`（二值相关集） | 已覆盖 |
| Precision@K | `memory_expect` + 固定 top_k | 已覆盖 |
| MRR（first_relevant_rank） | `memory_expect` 有序 | 已覆盖 |
| conflict_new_fact_first_rate | `expected_first` + `memory_expect` 有序（q-01/q-03/q-04/… 共 12 条多相关 query） | 已覆盖 |
| 无关注入率 / 负例不召回 | `q-16`（`memory_expect=[]`） | **未达成（runner gap）**：`recall_metrics` 对空 expect 直接 `continue`，q-16 被跳过，无法计入分母。需 runner 支持「期望空集」查询才能量化；本包只提供标注，不擅自改脚本。 |
| nDCG | `relevance` 三级标注（high=3/medium=2/irrelevant=0） | **未达成（runner gap）**：`run_golden.py` 未实现 nDCG；分级真值已备好，等 runner 支持后可直接算。 |

> 上面两行「未达成」是如实标注的接口缺口，不硬凑成已验证。A-V4 的要求「覆盖 recall eval 已实现的全部指标」
> 指的是前四行（Recall@K/Precision@K/MRR/冲突排前），均已覆盖；后两行是 runner 尚未实现的能力。

---

## 6. 校验命令与实测输出（可复现）

工作目录 `MakeAgain-backend`，Windows PowerShell。以下为**真实命令 + 真实输出**（2026-11-21）：

```powershell
$py = @'
import json, pathlib, collections, sys
from datetime import datetime
p = pathlib.Path('eval/recall_corpus.json')
d = json.loads(p.read_text(encoding='utf-8'))
M = d['memories']; Q = d['queries']
MEM_REQ = {'id','content','summary','facts','importance','ts','type','scene'}
Q_REQ = {'id','memory_query','relevance','memory_expect','expected_first','boundary','note'}
SCENES = {'breakup','loved_one','pet'}
GRADES = {'high','medium','irrelevant'}
errs=[]
if len(M) < 30: errs.append('memories < 30: %d' % len(M))
for m in M:
    for k in MEM_REQ:
        if k not in m: errs.append('mem %s missing %s' % (m.get('id'),k))
    if m.get('scene') not in SCENES: errs.append('mem %s bad scene' % m.get('id'))
    if m.get('type') not in {'chat','event','report'}: errs.append('mem %s bad type' % m.get('id'))
    if not isinstance(m.get('facts'),list) or not m['facts']: errs.append('mem %s bad facts' % m.get('id'))
    if not isinstance(m.get('importance'),(int,float)): errs.append('mem %s bad importance' % m.get('id'))
    try: datetime.fromisoformat(m['ts'].replace('Z','+00:00'))
    except Exception: errs.append('mem %s bad ts' % m.get('id'))
mids=[m['id'] for m in M]
if len(mids)!=len(set(mids)): errs.append('duplicate memory ids')
if len(Q) < 15: errs.append('queries < 15: %d' % len(Q))
qids=[q['id'] for q in Q]
if len(qids)!=len(set(qids)): errs.append('duplicate query ids')
for q in Q:
    for k in Q_REQ:
        if k not in q: errs.append('qry %s missing %s' % (q.get('id'),k))
    rel=q.get('relevance',{})
    if not isinstance(rel,dict) or not rel: errs.append('qry %s bad relevance' % q.get('id'))
    for mid,g in rel.items():
        if mid not in mids: errs.append('qry %s rel unknown id %s' % (q.get('id'),mid))
        if g not in GRADES: errs.append('qry %s bad grade %s' % (q.get('id'),g))
    exp=q.get('memory_expect',[])
    if not isinstance(exp,list): errs.append('qry %s bad memory_expect' % q.get('id'))
    for mid in exp:
        if mid not in mids: errs.append('qry %s expect unknown id %s' % (q.get('id'),mid))
        if rel.get(mid) not in {'high','medium'}: errs.append('qry %s expect %s not graded relevant' % (q.get('id'),mid))
    if exp and q.get('expected_first') != exp[0]: errs.append('qry %s expected_first != expect[0]' % q.get('id'))
    has_rel = any(g in {'high','medium'} for g in rel.values())
    if bool(exp) != has_rel: errs.append('qry %s expect/graded-relevance mismatch' % q.get('id'))
scene_cnt=collections.Counter(m['scene'] for m in M)
grade_cnt=collections.Counter(g for q in Q for g in q['relevance'].values())
bset=collections.Counter(q['boundary'] for q in Q if q['boundary'])
checks = {
 'JSON可解析': True,
 '记忆数>=30': len(M)>=30,
 'query数>=15': len(Q)>=15,
 '三类离别齐全': set(scene_cnt)==SCENES,
 '必含mem-cat-owned': 'mem-cat-owned' in mids,
 '必含mem-cat-deceased': 'mem-cat-deceased' in mids,
 '边界-冲突事实': any('冲突事实' in str(q['boundary']) for q in Q),
 '边界-相似主题干扰': any('相似主题干扰' in str(q['boundary']) for q in Q),
 '边界-时间敏感': any('时间敏感' in str(q['boundary']) for q in Q),
 '边界-无关记忆': any('无关记忆' in str(q['boundary']) for q in Q),
 '养猫query只召回新事实': any(q['id']=='q-02' and q['memory_expect']==['mem-cat-deceased'] and q['relevance'].get('mem-cat-owned')=='irrelevant' for q in Q),
}
print('JSON parse: OK')
print('memories=%d queries=%d scenes=%s' % (len(M),len(Q),dict(scene_cnt)))
print('grade counts:',dict(grade_cnt))
print('boundary tags:',dict(bset))
for k,v in checks.items(): print('CHECK %-28s %s' % (k,'PASS' if v else 'FAIL'))
print('schema errors:', errs)
sys.exit(1 if errs or not all(checks.values()) else 0)
'@
python -c $py
Write-Output "EXIT_CODE=$LASTEXITCODE"
```

实测输出：

```
JSON parse: OK
memories=33 queries=18 scenes={'loved_one': 13, 'pet': 9, 'breakup': 11}
grade counts: {'high': 21, 'medium': 17, 'irrelevant': 25}
boundary tags: {'冲突事实-只召回新事实': 1, '冲突事实-新事实排前': 1, '相似主题干扰': 6, '时间敏感-去年': 1, '时间敏感-周年': 1, '冲突事实-新旧关系': 1, '无关记忆-负例': 1}
CHECK JSON可解析                      PASS
CHECK 记忆数>=30                      PASS
CHECK query数>=15                   PASS
CHECK 三类离别齐全                       PASS
CHECK 必含mem-cat-owned              PASS
CHECK 必含mem-cat-deceased           PASS
CHECK 边界-冲突事实                      PASS
CHECK 边界-相似主题干扰                    PASS
CHECK 边界-时间敏感                      PASS
CHECK 边界-无关记忆                      PASS
CHECK 养猫query只召回新事实                PASS
schema errors: []
EXIT_CODE=0
```

### 6.1 接口连通性自检（口径：stub recall，非产品召回数字）

证明结构能被现有 `recall_metrics` 直接消费；用假的 recall_fn 按期望顺序返回，**不产生任何产品召回结论**：

```powershell
$py = @'
import json, pathlib, sys
sys.path.insert(0, 'eval')
from run_golden import recall_metrics
d = json.loads(pathlib.Path('eval/recall_corpus.json').read_text(encoding='utf-8'))
flat = [{'id': q['id'], 'memory_query': q['memory_query'], 'memory_expect': q['memory_expect']} for q in d['queries']]
ids = [m['id'] for m in d['memories']]
class E:
    def __init__(self, i): self.id = i
def stub_recall(q, top_k=3):
    item = next(x for x in flat if x['memory_query'] == q)
    ranked = list(item['memory_expect']) + [i for i in ids if i not in item['memory_expect']]
    return [{'entry': E(i)} for i in ranked[:top_k]]
m = recall_metrics(flat, stub_recall, 3)
print('[stub recall_fn，仅验证结构可被 recall_metrics 直接消费，非产品召回数字]')
print('n_queries=%d (18 条中 q-16 为空 expect 被跳过)' % m['n_queries'])
print('Recall@K=%s Precision@K=%s MRR=%s' % (m['Recall@K'], m['Precision@K'], m['MRR']))
print('conflict_n=%d conflict_new_fact_first_rate=%s' % (m['conflict_n'], m['conflict_new_fact_first_rate']))
print('q-02 期望=%s (只含新事实)' % next(p['expected'] for p in m['per_query'] if p['id']=='q-02'))
'@
python -c $py
Write-Output "EXIT_CODE=$LASTEXITCODE"
```

实测输出：

```
[stub recall_fn，仅验证结构可被 recall_metrics 直接消费，非产品召回数字]
n_queries=17 (18 条中 q-16 为空 expect 被跳过)
Recall@K=0.9529 Precision@K=0.6471 MRR=1.0
conflict_n=12 conflict_new_fact_first_rate=1.0
q-02 期望=['mem-cat-deceased'] (只含新事实)
EXIT_CODE=0
```

---

## 7. 验收 A-V1 ~ V4 逐条证据

| 验收项 | 状态 | 命令 / 证据 | 说明 |
|---|---|---|---|
| A-V1 语料 30+ 记忆 / 15+ query，字段齐全可解析 | **已验证** | §6 校验命令：`JSON parse: OK`；`memories=33 queries=18`；`schema errors: []`；`CHECK 记忆数>=30 PASS`、`query数>=15 PASS`；`memories` 8 字段 / `queries` 7 字段齐全，`importance`/`ts`/`facts` 类型合法 | 33/18 均超下限；三类离别齐全（breakup 11 / loved_one 13 / pet 9） |
| A-V2 每个 query 的相关标注经人工核对（标注者=生成者，注明口径） | **已验证（单人自标注，主观性已声明）** | §4 标注方法 + §2.2 字段字典；§6 校验：`expect ⊆ {high,medium}`、id 存在性、负例一致性全 PASS，`schema errors: []`；每条 `note` 写明判定理由 | 标注者=生成者，无第二标注者/无 kappa，已按铁律注明主观性与争议条目 |
| A-V3 冲突事实对（养猫/猫去世）query「现在养猫吗」期望只召回新事实 | **已验证（标注层面）** | §6 校验 `CHECK 养猫query只召回新事实 PASS` + `CHECK 必含mem-cat-owned/mem-cat-deceased PASS`；§6.1 `q-02 期望=['mem-cat-deceased'] (只含新事实)`；§3 边界表 q-02 | q-02 的 `memory_expect` 只含新事实，`mem-cat-owned` 在 `relevance` 里标 `irrelevant`（被 supersede）。**注意**：这是标注期望；真实召回是否做到需 lead 用 V2 语料跑 `run_golden.py` 验证（本包不改 runner、不写 outputs） |
| A-V4 范围标注覆盖 recall eval 已实现的全部指标 | **部分验证（写边界）** | §5.2 对照表：Recall@K / Precision@K / MRR / conflict_new_fact_first_rate 四项已覆盖；§6.1 用 stub recall 跑通 `recall_metrics`：`Recall@K=0.9529 Precision@K=0.6471 MRR=1.0 conflict_n=12 conflict_new_fact_first_rate=1.0` | 已实现四指标全覆盖且有接口连通性证据；**未达成**：nDCG（runner 未实现，分级真值已备）、负例不召回量化（q-16 被 `recall_metrics` 空 expect 跳过）——见 §5.2，如实列为 runner gap，不硬凑 |

> 验收铁律执行说明：全 11 项结构/边界校验 PASS 不代表「产品召回就好」——本包是评测语料，不含产品召回数字；
> §6.1 的数字是 stub 自检，已明确标注口径，不得当作产品指标。真实召回效果由后续集成（lead）跑 V2 完整评测得出。
