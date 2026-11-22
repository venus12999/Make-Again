# H3 真人金标标注指南（HUMAN_GOLDEN_GUIDE）

> 配套工具：`eval/human_golden.py`（`init` / `export` / `import` / `consistency` 四个命令）。
> 数据：`eval/human_golden.json`（**构造金标占位，非真人判定**）。

---

## 0. 当前状态（先说清楚，绝不假装有真人）

**本仓库现在没有任何真人标注。** `human_golden.json` 的 30 条全部从
`judge_golden.json` 的**构造金标**导入：

- `user_input` 抽自四批权威集（CPsyCoun / ESConv / GoEmotions / CueSupport-MH），`source_id` 可回溯原文；
- `reply_to_label` / `anchor_reply` 是评测执行代理按产品标准**人工构造**的好/坏对照，**不是真实用户对话、不是模型生成、不是真人判定**；
- 每条 `verdict="constructed"`、`constructed_verdict` = 构造标签、`human_label=null`、`status="pending"`。

因此：

1. 现在跑 `consistency` 得到的 κ/α **不是真人一致性数字**。若要跑通流程，必须先用
   `--annotator "PIPELINE_SELFTEST_CONSTRUCTED(构造标签,非真人)"` 把**构造标签**回填，
   工具会把口径标成 `CONSTRUCTED_DRY_RUN_NON_HUMAN`，仅证明流水线可运行。
2. 当前 30 条（含 14 条失败样本）**低于主流建议的 50–100 条**，定位是占位 + 流程验证；
   真人金标需由**产品方**补全新样本并逐条判定，不能把构造标签当真人金标交差。
3. 没有 key、不发网络、不改 agent / memory / api 主链路。

---

## 1. 主流依据（先调研后落地，URL + 发布日期）

| 出处 | 日期 | 与本工作流对应的做法 |
|---|---|---|
| [Airbnb《Eval-Driven Development: Lessons from Evaluating GenAI at Scale》](https://medium.com/airbnb-engineering/eval-driven-development-lessons-from-evaluating-genai-at-scale-e817e5ae5788) | 2026-07-28 | judge 必须先对**真人金标**校准才可信；金标 **50–100 条且必须含 bad examples**；一致性用 **Cohen's kappa / Krippendorff's alpha**，目标 high 80s–90s；**专家对标签分歧就 hard stop**，先解决再继续；随失败模式演化**周期性重校准**；指定一名最终人类决策者。[摘要](https://agentnativeengineering.com/field-notes/2026-07-28-calibrate-llm-judge-before-it-grades-releases/)（2026-07-28） |
| [Netflix《The Lifecycle of LLM-as-a-Judge: Building, Aligning, and Monitoring at Scale》](https://netflixtechblog.com/the-lifecycle-of-llm-as-a-judge-building-aligning-and-monitoring-at-scale-c95bd8283508) | 2026-10 | 四阶段：人类专家定义"好"并手工造 hard examples → judge 对齐人类标签**和书面理由** → 上线 → 人类持续查漂移；每条 **≥3 名标注者多数票**定真值；每周抽样约 300 条（含 judge 拦截的）复核；越 drift 阈值触发重训与新基准；除标签分歧外还查 "right answer, wrong reason"。[摘要](https://pulse.adyog.com/insights/netflix-llm-judge-reasons-not-just-verdicts)（2026-10-15） |
| [Netflix / Data Science in the Wild《Grading the Graders: Can an AI Judge Your Synopsis?》](https://rss.com/podcasts/data-science-in-the-wild/3151092/) | 2026-09-15 | judge 与创意写手一致 85%+、写手之间约 80%；one judge per criterion；多次采样取共识平滑误判。 |

调研结论如何落进代码：`RESEARCH_BASIS` 常量写进 `human_golden.py` 模块注释与
`human_golden.json.research_basis`；`export` 带 rubric + 空标签 + 含失败样本；
`consistency` 同时算 κ 与 α 并输出具体分歧 case 供"先解决分歧"；逐维度一致率写入
`consistency_curve.jsonl` 供 criteria drift 持续观察。

---

## 2. 谁标 / 标什么 / 怎么标

### 谁标
- 产品方指定 **2–3 名真人标注者**（Netflix 口径：每条≥3 人、多数票定真值）。
- 同时指定 **1 名最终决策者**（`annotators.named_final_decision_maker`），专管分歧裁决与 rubric 变更审批。
- 标注者姓名/标识只写 `annotator` 字段，不写任何密钥或个人信息。

### 标什么
- 每条给 `reply_to_label`（待评回复，即当前 judge 流程里被评的候选回复）打二元标签：
  **pass（该维度达标） / fail（不达标）**，并写一句 `human_rationale`（书面理由）。
- 判定依据是同条内联的 `rubric`（四维之一）：
  `empathy_specificity` / `strategy_appropriateness` / `emotion_accuracy` / `crisis_safety`。
- `user_input` 与 `rubric` 必读；`anchor_reply` 是对照回复，仅作语境参考，不要据此改变对候选回复的判断。
- **失败样本必须保留**：当前占位集 30 条中 14 条 `constructed_verdict=fail`；真人补标时也要刻意覆盖
  好/坏两类，否则一致率会被"全是 pass"虚高（Airbnb: golden MUST include bad examples）。

### 盲标与 judge 对照
- `export --blind` 会隐藏 `judge_verdict` / `judge_reason`，避免真人被 judge 结论锚定（推荐正式标注时用）。
- 默认导出**会带上** judge 的 verdict/reason，这是为了能对比 "right answer, wrong reason"
  （Netflix RART）：结论一致但理由不一致也要登记，不算真正对齐。

---

## 3. 分歧怎么处理（hard stop）

1. `consistency` 输出的 `disagreements[]` 是**逐条具体 case**：含 `id`、`source_id`、
   `user_input`、`reply_to_label`、`judge_verdict` + `judge_reason`、
   `human_label` + `human_rationale`，`resolution="PENDING_HUMAN_ADJUDICATION"`。
2. 专家对某条标签有分歧时 **hard stop**（Airbnb）：标注者讨论并书面记录理由；
   若分歧来自 rubric 含糊，先改 `rubric` 再重标该条，**不要靠调阈值把分歧抹平**。
3. 除"标签不一致"外，还要看 judge 与真人**理由是否一致**；结论对但理由错的 case 同样进待裁决清单。
4. 1 名最终决策者对仍无法一致的 case 拍板，并把裁决写进该条 `human_rationale`。

---

## 4. criteria drift 持续校准

- 失败模式会演化，rubric 和金标都要**周期性重校准**（Airbnb）。
- 每次 `consistency` 追加一行到 `outputs/human_golden/consistency_curve.jsonl`，含总体
  κ/α/一致率与**逐维度**一致率；按维度观察是否有维度漂移（某维度一致率持续下滑）。
- Netflix 口径：每周按维度抽样复核（含 judge 拦截的样本），越过 drift 阈值触发补标与新基准；
  **阈值由产品方按业务定**，本工具只提供观察值，不替产品方下结论。

---

## 5. 命令流程（在 MakeAgain-backend 目录）

```bash
# 1) 导出待标注集（正式标注推荐 --blind 防锚定；默认 JSONL + CSV 都出）
python eval/human_golden.py export --blind

# 2) 人工在 JSONL/CSV 里逐条填写 human_label(pass/fail)、human_rationale、annotator
#    只填有把握的；留空的行会被 import 跳过（不算标注）

# 3) 回填（默认写 outputs/human_golden/human_golden_labeled.json 副本，不改源金标）
python eval/human_golden.py import outputs/human_golden/annotation_export.jsonl

# 4) 算 judge-人一致率 + 分歧清单（judge 标签默认取真实 judge_consistency.json）
python eval/human_golden.py consistency \
    --golden outputs/human_golden/human_golden_labeled.json

# 只跑通流程的构造标签干跑（非真人！产物口径会自动标 CONSTRUCTED_DRY_RUN_NON_HUMAN）
# 用 constructed_verdict 当 human_label 回填，annotator 写含 PIPELINE_SELFTEST_CONSTRUCTED 的标识
```

导出文件字段说明：`id/source_id/dimension/rubric/user_input/reply_to_label` 是标注输入；
`verdict=constructed`、`constructed_verdict` 仅供对照，**不是真人标签**；
`human_label/human_rationale/annotator` 是待回填空字段。

---

## 6. 诚实边界

- `consistency` 在**无人工标签**时返回 `kappa=None` + `note=NO_HUMAN_LABELS`，不编造数字。
- 构造标签干跑得到的 κ/α 会明确标 `CONSTRUCTED_DRY_RUN_NON_HUMAN`，**不能**写进对外材料当真人一致性。
- `import` 对非法 id / 非法标签**整批拒绝**，不部分写入、不猜标签。
- 真人身份本工具不独立验证：`HUMAN_ANNOTATION_CLAIMED_NOT_INDEPENDENTLY_VERIFIED` 只是"声称来自真人"，
  需要产品方自己的标注流程保证。
