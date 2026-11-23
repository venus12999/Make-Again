# 交接文档（新窗口必读，2026-11-24）

## 0. 一句话现状

产品后端+小程序前端都已存在、能跑。**产品本身在真人手里是正常的**（用户在微信开发者工具里点着正常、AI 回复是真的）。
但**一个真实 bug 没修**：长对话丢早期事实（事实保留率 1/3，用户"像她又走了一次"）。评测体系只有 1/3 门禁活跃，没做到"漏斗定位"。

---

## 1. 开工前必读（按顺序，不读不许动手）

1. **本仓 `AGENTS.md`** —— 项目铁律：台账强制、时间衰减权重、评测约定、**禁止 git add -A**（有 38762 文件误提交前科）、本项目的坑
2. **工作区根 `D:\12412\dsh工作区\AGENTS.md`** —— 通用规则
3. **`docs/PROGRESS.md`** —— 唯一台账。**先读 §0.1 活动决策/NO-GO + 最近 10 条明细**，尤其是 D-059~D-068（最近全部纠错/推翻/未验证项）。状态词只用：已验证（含证据）/部分验证（写边界）/未验证/NOT_RUN/ENV_FAIL/已撤回
4. **`docs/FRONTEND_INTEGRATION_STATUS.md`** —— 前端接线的真实状态（两轮 GUI 验收 + 误判澄清）
5. **`docs/CONTEXT_REUSE_PLAN.md` + `docs/CONTEXT_ENGINEERING_RESEARCH.md`** —— 长对话上下文的竞品调研和复用改造方案
6. **`docs/LONG_CHAT_AUDIT.md`（在队友仓 `make-again-post-merge/MakeAgain-backend/work/`）** —— 多场景10轮长对话的原始实测

---

## 2. 用户是谁、红线是什么（最重要，我反复栽在这）

用户极度重结果、极度厌恶"套壳"和"把局部当整体"。**我这一轮被骂的核心原因，新窗口必须避免：**

| 我犯的错 | 正确做法 |
|---|---|
| 拿 Node 模拟 wx 的 14/14 说"验收通过" | 只认真 GUI/真实链路；模拟必须标口径，不能当结论 |
| automator 误判"登录不跳转/记忆没落库"就照单全收报 ❌不可用 | 用户真人操作 > 工具结论；工具报 bug 先想是不是工具误判 |
| 建了 30 个 eval 工具 + 9 份文档就说"体系有效" | 唯一验收标准：**能不能拦住真实 bug**，不是工具数量 |
| ci_runner 390 行全是 JSON 格式校验，叫"评测门禁" | 门禁必须测产品行为，否则就是套壳 |
| 写完代码/文档就说"做完了" | 说"部分验证"并写清边界；没验证就是没验证 |

**用户偏好**：说人话、不墨迹、不要英文黑话不解释、不要 premature 云端/部署（云端和本地只是承载形式）。**每次结论必须有真实命令/输出/截图/DB 数字**，不许靠推理下结论。

---

## 3. 项目结构（两个仓库）

- **主仓**：`D:\12412\dsh工作区\projects\make-again-mvp`（当前目录）
  - 后端：`MakeAgain-backend/`（Python 3.12 + FastAPI + SQLAlchemy + SQLite，本地优先）
  - 评测：`MakeAgain-backend/eval/`，产物 `outputs/`
  - 台账/文档：`docs/`
- **队友仓**：`D:\12412\dsh工作区\projects\make-again-post-merge`
  - 前端小程序（app.json/pages/）+ 后端副本 `MakeAgain-backend-source/MakeAgain-backend/`
  - 前端入口 `utils/service.js` → 按 `config.js` 的 CLOUD 开关切 local/cloud
  - 微信开发者工具已配置好，`.ide-status=On`，服务端口自动化走 ws://127.0.0.1:9421

**真 key**：`work/local-test-key.txt`（gitignore，绝不打印/入库）。后端 hy4 端点 `https://opencode.ai/inference/go/openai/v1` + `x-opencode-session`。

---

## 4. 当前真实状态（按验收标准标注）

### ✅ 能用的
- 后端产品核心：三场景对话、访谈→报告、记忆链路、触发、结束仪式，都有真实验证（见台账 D-050~D-058）
- 前端：cloud-service 适配层已写、真 GUI 聊天自动化成功（D-062，用 `miniProgram.evaluate()` 调 `onInput()/send()` 绕过挂死的 createSelectorQuery，方法见队友仓 `docs/CHAT_AUTOMATION_METHOD.md`）
- 全量测试：571 passed（开关关时零回归）

### ❌ 真实 bug / 未完成
1. **长对话丢早期事实**（D-065/D-066）：事实保留率 1/3。`eval/long_context_eval.py` 实测：breakup"林夏"❌、loved_one"周明远"❌、pet"橘子"✅。用户第10轮"像她又走了一次"——产品最不能接受的体验
2. **压缩代码写了但没起作用**（D-065）：`memory/context_compaction.py`（从 PI 源码移植）+ 接入 `agent/companion.py`，但**只接在 `_chat_generic`——三个场景走专属引擎，压根没接上**；且 `USE_CONTEXT_COMPACTION` 默认关。压缩有效性标"未验证"
3. **评测体系只有 1/3 门禁活跃**（D-068）：`eval/gate_all.py` 里 fact_retention 是 ACTIVE（1/3 < 0.66 会 exit 1，已验证能拦住 bug）；crisis_recall / chain_complete 是 TODO（空的，别假装在工作）。漏斗只有入口+CI防掏空，没有"分数异常自动路由到深挖探针"

### 唯一值得继续做的事
**把压缩接到三个场景引擎上，然后跑 `python eval/gate_all.py`（或 `long_context_eval.py --gate 0.66`），看事实保留率能不能从 1/3 提上去。** 提上去 = bug 真修了；提不上去 = 接着查根因。别的都是虚的。

---

## 5. 并发子代理工作方式（用户明确要求掌握）

本仓支持 Agent Teams / subagent。我这一轮用过的模式，新窗口照做：

- **派子代理的时机**：独立、可并长的任务（如：3 路竞品调研、多场景长对话扮演真实用户、真 GUI 自动化验收）→ 用 `workflow`（多 agent 编排）或 `subagent`。1-2 个 delegations → 直接 `subagent`/`subagent_fork`
- **给子代理的 prompt 必须自包含**：它不共享本窗口上下文，要写清工作目录、环境状态、key 路径、已知误判（别让它重复报）、铁律（禁 git add -A、key 不打印不入库）、时限（40分钟收口，卡住标 ENV_FAIL 不无限等）
- **并发写冲突**：子代理各自在共享目录写文件，**写作用域要分开**（如各自 `work/xxx_*.json`），提交用独立分支 + 显式 `git add <文件>`，原子提交（checkout→add→commit→push），绝不 `add -A`
- **收结果**：`wait_agent` 等完成，完成后**逐条 review 它的结论**——这一轮子代理就报了 2 个假 bug（登录不跳转、记忆没落库）和 1 个真 bug（SQLAlchemy text 化）。不要照单全收
- **同步到队友仓**：建分支→拷贝文件→显式 add→commit→push→`gh pr create`→`gh pr merge`

---

## 6. 常用命令速查

```bash
cd MakeAgain-backend
# 全量测试（约5分钟，需 FFmpeg 在 PATH，HF_HUB_OFFLINE=1 + HF_HOME）
python -m pytest tests -q
# 长对话事实保留率硬门禁（需真 key + 后端起在 8000）
python eval/gate_all.py              # 或 python eval/long_context_eval.py --gate 0.66
python eval/gate_all.py --list       # 看哪些门禁活跃（诚实标注 1/3）
# 后端一键本地实测（全 mock）
python scripts/local_quickstart.py
```

后端启动需：`ENVIRONMENT=development` + `DATABASE_URL` + `MEDIA_SIGNING_KEY`（≥32字符）+ mock 开关；真模型还要 `LLM_*` + key → 从 `work/local-test-key.txt` 读，**用脚本注入 env，不要打印、不要写进文件**。

---

## 7. 最后一句（最重要）

新窗口别再犯我的错：**"写了/建了" ≠ "做对了" ≠ "验证了"**。用户要的是能用、能拦住真实 bug 的产品。当前唯一真 bug 是长对话丢事实，唯一该做的验证是事实保留率门禁。先把这个解决，再说别的。
