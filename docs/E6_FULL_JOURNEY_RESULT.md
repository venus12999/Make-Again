# E6 完整旅程跨段链路评测 · 交付说明

> 日期：2026-11-23。基线 main `807a7a15`（476 passed）。分支：`eval-rebuild/E6`（不合并 main）。
> 所有权：`MakeAgain-backend/eval/e6_full_journey.py`（新）、
> `MakeAgain-backend/eval/batches/E6_完整旅程.json`（新）、
> `MakeAgain-backend/tests/test_e6_journey.py`（新）、本文件与台账/_index 登记（待 lead 授权后显式提交）。

## 1. 产品本质与为什么做 E6

产品是情感陪伴，价值在五段闭环 **表达 → 理解 → 沉淀 → 调用 → 行动** 串起来，
判据是"被准确理解 / 记忆没弄错 / 下一步有帮助"。E1-E5 都是单点或单段，
只有 E6 把 `walkthrough_journey.py` 与 `e2_interview_chain.py` 扩成"完整旅程"：
访谈 → 报告 → 多轮对话 → 记忆回写 → 跨天召回 → 触发，逐段设衔接检查点。
链路 bug 只能靠链路评测抓（AGENTS.md 评测工作约定）。

## 2. 分层（用户关心成本）

| 层 | 跑什么 | 频率 | 成本/口径 |
|---|---|---|---|
| **日常快评** | E1 单点（改 prompt 日常跑）、`E6 --mock`、`E2 --mock` | 每次改动 | 秒级、纯确定性、不烧 key；构造 trace 验链路结构，**不是产品行为证据** |
| **每周/大改深评** | `E2` 真 hy4 1 人 + `E6` 真 hy4 **1 场景**（`--limit 1`） | 每周或大改 | 约 10 分钟/场景，key 从 `work/local-test-key.txt` 读，不打印不入库 |
| **全量深评（非常规）** | E6 三场景全部真 hy4 | 大版本/发布前单独排期 | 本轮明确只跑 1 场景（标口径，避免卡住） |

慢评测不进日常 CI；CI 门禁仍只跑 `ci_runner.py` 的离线确定性检查与全量 pytest。

## 3. 五段衔接检查点（代码判定，不新增 LLM judge）

| CP | 衔接点 | 判据 | 真实数字来源 |
|---|---|---|---|
| CP1 | 访谈→报告 | 复用 E2 的 5 检查点（走到 complete/报告生成/字段齐全/个性化/追问有界） | `outputs/walkthrough/journey_walkthrough.json`（20 轮/空转 15）、`outputs/e2/e2_real_breakup_rescored.json`（13 轮/空转 8/415.3s/锚点 5/5） |
| CP2 | 报告→对话记忆注入 | `interview._persist_report` 落画像 + type=report 记忆；`companion._context_block` 注入画像/相关记忆；锚点命中 ≥1 | 报告记忆条数、画像字段数、上下文锚点命中数 |
| CP3 | 对话→记忆回写 | chat 的 durable memory job 完成后记忆数增加且新记忆含 persona 锚点 | job 终态、记忆数 before→after、回写锚点命中数 |
| CP4 | 记忆→跨天召回 | ts 回拨 3 天 + 新 session_id 模拟重开 App；正向 query 召回 ≥1 且 top 命中锚点；"今天晚饭吃什么"负例 0 条 | recall_hits、top_score、negative_recall_hits |
| CP5 | 召回→触发时机 | 本地 23:30 出 nudge 候选、14:30 不出 late_night；`daily.generate_theme_guide` 从真实记忆生成个性化引导（≠主题 desc） | rule_key、guide_personalized、guide_anchor_hits |

阈值口径（自造标 synthetic-handwritten，非真人标注）：跨天=3 天、注入锚点≥1、召回≥1、负例=0。
Context/Recall 的底层分数由产品代码实算，不是估算。

## 4. 数据口径与来源

- 3 条 persona 全部 `synthetic-handwritten`，由 `eval/batches/E2_理解段链路.json` 扩展而来，
  上游是 `eval/walkthrough_journey.py` 的 PERSONAS；每条有 `source_id` 与
  `source.derived_from`，可追到具体文件/行。
- 中文原生权威宠物离世语料：**NOT_FOUND**（沿用 D-031 结论），故宠物条自造并显式标注。
- mock trace 是构造值，**只验评测结构**；真 hy4 输出才作为产品行为证据。
- 跨天与 consent/companion 是 **eval 侧夹具**（隔离 DB 内改 ts / 授权 / 建陪伴对象），
  不改产品主链路 `agent/`、`memory/`、`api/` 业务代码。

## 5. 方法学（日期 + 按天精确衰减）

Anthropic《Demystifying evals for AI agents》2026-01-08，
https://www.anthropic.com/research/demystifying-evals-for-ai-agents
基准日 2026-11-23：`(2026-11-23 − 2026-01-08).days = 319`，`319 / 30.44 ≈ 10.5 个月`
→ **权重 0.7**（6-18 月档：参考，须查更新版）。采用其"code-based graders 优先"与
"真实失败做回归用例"两条；本评测 5 个检查点全部是确定性代码判定，不引入新 LLM judge，
故该来源的衰减不影响实现。本轮无其他新增外部调研；本地一手代码/走查实物按权重 1.0 采信。

## 6. 验收证据（命令 + 输出摘要）

待真 hy4 慢评完成后补 V3 数字；V1/V2/V4/V5 以命令输出为准，逐条写入台账 PROGRESS.md。

- V1：`python eval/e6_full_journey.py --mock` → 3 场景 × 5 CP 全 PASS，exit 0。
- V2：段间衔接点均输出int 数字（报告记忆 2、注入锚点 3、记忆 2→3、召回 2、负例 0）；
  产品函数无 LLM 验证报告落库→上下文注入、写入→召回→触发的输出见 pytest 打印。
- V3：`python eval/e6_full_journey.py --limit 1 --out ../outputs/e6/e6_real_breakup.json`
  → 真 hy4 分手 persona 跑完五段；**exit=1，不粉饰**：CP2-CP5 全 PASS，
  CP1（复用 E2 的报告质量/轮数检查）FAIL。真实数字：访谈 **19 轮 / 空转 14**（阈值 16，
  已知 U-016 残留，E2 上次真跑是 13 轮/8 filler）；本次报告字段退化——
  `quote/user_portrait/goal/heal_plan/relationship_analysis` 缺失、个性化仅命中 1/5 锚点
  （上一次 E2 真跑 5/5），但报告对象非空、对话注入锚点 **5/5**；
  报告记忆 1 条、画像字段 1 个；durable memory job `completed`，记忆 **20→21**、回写命中 2 锚点；
  回拨 3 天后正向召回 **5 条**、top_score **0.7020**、负例 **0**；
  23:30 触发 `late_night`、14:30 不触发、主题引导个性化 True。
  结论：**报告生成、对话、回写、跨天召回、触发全部真实成功**（V3 指定的三段成立），
  E6 同时抓到 E2 报告质量退化这一真实问题，未用"全绿"掩盖。
- V4：4 种故障注入各自 exit 1 且只打挂目标 CP（报告注入缺失/回写失败/召回失败/触发失败）。
- V5：索引稳定后 `python -m pytest tests -q` → **501 passed, 3 skipped, 0 failed, exit 0**
  （`work/e6_full_pytest_final.log`；含共享工作区里队友未提交的 E3/E4/E5 测试，
  本分支自身新增 11 条）。首次跑曾 498 passed/2 failed，失败源于并发改 `_index.json`
  的瞬时不一致，不是本分支代码；索引稳定后 `tests/test_ci_runner.py` 7 passed、
  `eval/ci_runner.py` exit=0。

## 7. 边界与诚实披露

- 真 hy4 只跑分手 1 场景；丧亲/宠物只做 mock 结构验收，不冒充真实产品证据。
- 跨天是夹具模拟（ts 回拨），dual 索引 meta 未回拨，本检查只验"跨天后仍能召回"，
  不验时间衰减曲线本身。
- E2 真实复跑残留的"轻度追问空转"（U-016：13 轮/8 filler）不是 E6 修复对象；
  E6 只验证衔接是否成立，不粉饰该残留。本次 E6 真跑更差（19 轮/14 filler）且报告字段退化，
  已作为 CP1 FAIL 与证据数字如实留在 `outputs/e6/e6_real_breakup.json`，
  供 lead 决定是否作为 U-016 的补充证据或单独开产品 issue。
- 本报告只交付评测与证据，不改 `agent/`、`memory/`、`api/` 业务代码；产品侧修复不在本分支范围。
