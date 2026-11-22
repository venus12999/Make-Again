# 评测统计门禁方法升级（2026-09 新方法，分支 eval/method-upgrade）

> 日期：2026-11-23。分支 `eval/method-upgrade`，从 main `6ab6f718` 切出，不合并 main。
> 基准日（时间衰减）：**2026-11-23**。所有权：`MakeAgain-backend/eval/iterate.py`、
> `MakeAgain-backend/tests/test_eval_statistics.py`、本文件；不碰 `store.py` / `companion.py`。
> 口径：URL 与发布日期为 2026-11-23 当天联网核实；月数按 `(基准日 - 发布日期).days / 30.44`
> 精算，不用年份感觉（AGENTS.md「调研时间衰减」）。构造数据均为手写模拟分数，非真实用户数据。

---

## 1. 调研结果：URL + 发布日期 + 按天权重（C-V1）

| 方法 | 一手 URL | 发布日期 | 距基准日天数 | 月数 | 权重档 |
|---|---|---|---:|---:|---|
| Anthropic《A Statistical Approach to Model Evaluations》（CLT/聚类标准误/配对差分/功效分析；配套论文 arXiv:2411.00640） | <https://www.anthropic.com/research/statistical-approach-to-model-evals>；<https://arxiv.org/abs/2411.00640> | 2024-11-19（博客页标 "published 2024-11-19"，当天读到的页面原文） | 734 天 | 24.11 月 | **0.4**（18–36 月，只作方法论基础，确认未被推翻） |
| **evalstats**《How to Run Statistics over LLM Judges and Trust the Results: Calibrated Inference for Small-Sample AI Evaluation with evalstats》 | <https://arxiv.org/abs/2609.35815>（HTML <https://arxiv.org/html/2609.35815v1>；代码 <https://github.com/ianarawjo/evalstats>） | **2026-09-20**（arXiv abs 页 "Submitted on 20 Sep 2026"，v1 提交 20 Sep 2026 16:49:41 UTC） | 64 天 | 2.10 月 | **1.0**（≤6 月，全信可直接用） |
| **Anthropic playbook**《Automating eval design and hillclimbing with Claude》（Lance Martin） | <https://claude.dev/blog/automating-eval-design-and-hillclimbing/>（技能仓库 <https://github.com/anthropics/skills/tree/main/skills/claude-api>） | **2026-09-28**（页面 "PUBLISHED Sep 28, 2026"；二手索引页独立佐证同日） | 56 天 | 1.84 月 | **1.0**（≤6 月，全信可直接用） |

权重计算实测（`pwsh` 当天执行）：

```
2024-11-19 -> days=734 months=24.11 weight=0.4
2026-09-20 -> days=64  months=2.1   weight=1.0
2026-09-28 -> days=56  months=1.84  weight=1.0
```

**替代即清零检查**：未发现两篇 2026-09 新方法有更新版或关停公告；两者均与
`docs/REUSE_REAUDIT.md` §4 的「2026 下半年真正 1.0 的新东西」口径一致。Anthropic 2024-11
旧博客未被明确撤回，但按 0.4 档只作基础，不单独作为新开关依据。

## 2. 一手方法要点（只取原文可核实的表述）

**evalstats（arXiv:2609.35815，1.0）**
- 摘要原文：对 raw LLM judge 分数跑统计会导致虚高假阳性；反直觉地，对很多评分者一致性指标，
  假阳性风险在「接近完美」的人机一致处最高。
- 摘要原文：通过 prediction-powered inference（PPI）实现 9 个假设检验，含首批 4 个秩检验
  （Wilcoxon signed-rank、Mann-Whitney U 及 omnibus 变体）的 PPI 校正。
- 摘要原文：用 Monte Carlo 模拟给出小样本 AI 评测（N<100）的 CI / p 值 / FWER 方法建议，
  并**警告不要用 bootstrap CI**；引入 bootstrap-adaptive power tuning 稳定小校准集下的 PPI++。
- 工具 `evalstats` 自动选择校准方法；示例包含"judge 在 substantial agreement 下仍会得出假显著结论"。

**Anthropic playbook（claude.dev，1.0）**
- grader 自洽：同一输出让 grader 打两次，报告判定是否变化；上线前读一批打分过的 transcript。
- 开跑前检查评测噪声（纯偶然能造成的分差）是否小于「最小可行动改进」；否则建议加重复次数或样本。
- hillclimb：随机拆 train/test；每轮只改一处根因；train 涨而 test 平 -> 判过拟合并回退；
  有退化也回退；**若增益在噪声内，明确写明并建议不合入**。
- 过拟合来源：把失败 transcript 粘进 prompt、针对评测样例加 harness 特性；建议只让 hillclimber 读 train。
- 示例：44 条工单 30 train / 14 held-out，最终 held-out 78.6%→90.5%。

**Anthropic 2024-11（0.4，方法论基础）**
- 五条建议：CLT 报 SEM/95%CI；相关问题聚类标准误；题内重采样降方差；**共享题目用配对差分**；功效分析算所需样本量。
- 我们已有的 bootstrap×1000 + 精确配对符号检验 + MDE/n<30 功效标注即源于此（S1，commit 6e170c2）。

## 3. 落地映射：本分支做了什么 / 没做什么（C-V2/C-V3/C-V4）

| 新方法（权重） | 本分支动作 | 代码位置 |
|---|---|---|
| bootstrap CI 小样本不可靠（evalstats 1.0） | **已做**：N<100 在报告里标"仅作对照，判定以符号检验为主"；bootstrap 保留作对照不删 | `iterate.py` `SMALL_SAMPLE_N=100`、`BOOTSTRAP_CAVEAT`、`format_metric_report` |
| 符号检验为主、CI 仅参考（evalstats 1.0） | **已做**：稳健模式以精确配对符号检验方向为 evidence；CI 方向只写进 reason 作参考，不再能否决 | `compare_metric(robust_gate=True)` 默认开启 |
| 噪声门槛 / 噪声<最小改进才开跑 / 增益在噪声内不合入（playbook 1.0） | **已做**：`noise_gate()` 用配对差值 sd/√n×1.96 得噪声带；增益≤噪声带 -> `持平（增益在噪声内，建议不合入）`；噪声带≥该指标最小可行动改进 -> 提示加重复/样本 | `noise_gate()`、`compare_metric` 判定分支、报告"噪声门槛"行 |
| train/test 拆分防过拟合（playbook 1.0） | **部分（只标注，不做物理拆分）**：同基线同批已 compare≥2 次时打印 `[overfit] 该批已被用于 N 次调参对比，存在过拟合风险…`，并写入快照；提示"train 涨 test 平应回退"。**这不是 held-out 验证，结论仅参考** | `overfit_warning()`、`prior_compare_count()`、`cmd_compare_statistical` |
| grader 自洽双跑（playbook 1.0） | **不重复造轮子**：既有 `eval/judge_calibration.py:judge_consistency_gate()`（S2）已做 Kappa/一致率前置门，compare 仍在调用 | `cmd_compare_statistical` 末尾 S2 门（未改） |
| PPI 校正（evalstats 1.0） | **未做（诚实边界）**：PPI 需要人标注校准集（prediction-powered inference 的 labeled subset），本项目金标仍是构造占位（见 D-043/H3），没有可信人标校准集；硬套会制造"看起来校正过"的假数字。等真人金标回填后再评估 | 无 |
| CLT/配对/功效（Anthropic 0.4） | **保留**：既有 bootstrap+配对符号检验+MDE 全部保留，作为对照基线 | S1 既有实现未删 |

**回退开关（C-V4）**：
- `compare --no-robust`：关闭 2026 稳健判定（噪声门槛/符号优先/过拟合标注），CI+符号检验一致判定逐字恢复升级前逻辑；报告中不再出现新标注。
- `compare --no-repetitions`：仍走 `cmd_compare_legacy`，输出与最初旧代码逐字节一致（既有 V7 golden 测试继续通过）。
- 两个开关互不干扰：前者只关新门禁，后者关整个统计门禁。

## 4. 验收（真实命令 + 输出，C-V2~V5）

### C-V2 新判定逻辑真实数字
```
cd MakeAgain-backend
python -c "import sys,json; sys.path.insert(0,'eval'); import iterate as it; \
 pc=lambda vals:{f'c{i+1:02d}':[float(v)]*3 for i,v in enumerate(vals)}; \
 a=it.compare_metric('judge_avg',pc([7.0]*20),pc([8.0]*19+[0.0])); \
 b=it.compare_metric('judge_avg',pc([7.0]*20),pc([8.0]*19+[0.0]),robust_gate=False); \
 print('ROBUST',json.dumps({k:a[k] for k in ('n','delta_mean','sign','noise','verdict')},ensure_ascii=False)); \
 print('OLD_GATE',b['verdict'])"
```
实际输出（节选，完整见 §4 测试）：
```
ROBUST= n=20, b_mean=7.0, a_mean=7.6, delta_mean=0.6,
        sign={pos:19, neg:1, p_value:4e-05, significant:true},
        noise={noise_se:0.4, noise_band:0.784, min_improve:0.1, gain_above_noise:false},
        verdict=持平（增益在噪声内，建议不合入）
OLD_GATE= 改善
OVERALL= ⚠ 增益在噪声内，建议不合入（先加重复次数/样本量再开跑）
```
报告行（真实 stdout）：
```
    噪声门槛: 噪声带=0.7840 SE=0.4000 最小可行动改进=0.1000 增益高于噪声=False
    ⚠ 噪声≥最小可行动改进，建议加重复次数或样本量再开跑（playbook）
```

### C-V3/C-V4 构造数据 + 回退开关
```
python -m pytest tests/test_eval_statistics.py -q -rs
19 passed, 1 warning in 4.30s
```
其中 `test_c_noise_gate_blocks_gain_within_noise` 断言：稳健=「持平（增益在噪声内，建议不合入）」、
关闭开关=「改善」；`test_c_no_robust_switch_restores_old_statistical_output` 断言新标注全部消失；
`test_v7_no_repetitions_byte_identical` 继续保证旧路径逐字节一致。

### C-V5 全量无回归
见 §5 实测结果（本文件提交前填入，命令=`python -m pytest -q`）。

## 5. 全量验收结果

```
cd D:/12412/DshWorktrees/make-again-mvp/eval-method-upgrade/MakeAgain-backend
python -m pytest -q
524 passed, 6 skipped, 2 xfailed, 2 warnings in 321.58s (0:05:21)
PYTEST_EXIT=0
```
基线核对：任务给定 main `6ab6f718` = 517 passed；本分支新增 7 项构造测试
（12 → 19 in `tests/test_eval_statistics.py`），517 + 7 = **524 passed**，
skipped/xfailed 数量不变，**0 failed，无回归**。耗时较长因多代理并发跑同一 CPU。

## 6. 边界 / 未验证 / 不做

- **PPI 未落地**：无人标校准集，不伪造校正数字；新判定是「符号检验为主 + 噪声门槛」，不等同 evalstats 的 PPI 校正。
- **train/test 未物理拆分**：只做同批反复 compare 的风险标注；真正 held-out 需先按 playbook 批次拆分并改造 snapshot/compare 接口，超出本包范围。
- **噪声带是正态近似**（1.96×SE），对极端非正态/小 n 只是量级参考；与 MDE/功效一起看，不单看一个数字。
- **危机红线不受开关影响**：`crisis_detect_rate` 下降仍一律拦（安全红线优先于任何稳健开关）。
- 构造数据口径：手写模拟分数（非真人、非真实模型输出），只验证判定逻辑，不证明产品效果。
- 本分支只提交 3 个分配文件；台账 `docs/PROGRESS.md` 由 lead 集成时追加，避免并发冲突。

## 相关

- 时间衰减规则：项目根 `AGENTS.md`；复审依据 `docs/REUSE_REAUDIT.md`
- 既有门禁：`docs/EVAL_ITERATION_LOOP.md`、`docs/P0_EVAL_HARDENING_RESULT.md`、`eval/judge_calibration.py`
- 评测纪律：`docs/EVAL_DISCIPLINE.md`、`docs/EVAL_LOOP_AUTHORITATIVE_RESEARCH.md`
