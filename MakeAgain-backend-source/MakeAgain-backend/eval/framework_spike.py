#!/usr/bin/env python3
"""U-009 记忆框架 3 候选 spike：graphiti / hindsight / MemOS vs 现有 stack。

方法（严格区分证据等级）：
  1) 能力证据：gh api 实读三个仓库的 README / 源码 / 依赖清单 / compose，
     逐字 quote 固化在 EVIDENCE 中（URL + gh api 取数命令）。`--verify` 可复验。
  2) 用例对照：读取 eval/recall_corpus.json 的 33 记忆 / 18 query，
     按 boundary 逐项映射「上游文档声明的机制」→ 对本用例的支持等级。
     **这是 DOC_LEVEL_PREDICTION，不是实测行为数字。**
  3) 端到端行为：NOT_RUN。三个候选都需要图库/Postgres/MySQL + LLM key，
     本机无 Neo4j/MySQL、无任何 LLM key；不硬凑数字。
  4) 现状基线：读取 outputs/golden-recall-v2.json 的真实评测数字（既有产物，未重跑）。

运行：
  python eval/framework_spike.py                 # 生成 outputs/framework-spike-2026-11-21.json
  python eval/framework_spike.py --verify        # gh api 复验证据 quote（网络可用时）
  python eval/framework_spike.py --check         # 只跑自检，不写文件

所有权：只读 corpus + baseline outputs，唯一写目标是 outputs/framework-spike-<date>.json。
不 import 业务代码、不改业务库、不写台账、不碰 requirements。
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATE = "2026-11-21"
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
CORPUS_PATH = HERE / "recall_corpus.json"
BASELINE_PATH = PROJECT / "outputs" / "golden-recall-v2.json"
OUTPUT_PATH = PROJECT / "outputs" / f"framework-spike-{DATE}.json"

# ---------------------------------------------------------------------------
# 证据：URL + 逐字 quote + 取数命令。全部由 gh api 实读并逐字比对通过。
# ---------------------------------------------------------------------------
EVIDENCE: list[dict[str, str]] = [
    # ---- graphiti ----
    {
        "id": "GF-01", "candidate": "graphiti", "kind": "README",
        "url": "https://github.com/getzep/graphiti/blob/main/README.md#L81",
        "fetch": "repos/getzep/graphiti/readme",
        "quote": "Graphiti is an open-source Python framework for building and querying temporal context graphs.",
        "conclusion": "定位是 framework 不是成品服务；多租户/鉴权/入库/运维由使用方自建。",
    },
    {
        "id": "GF-02", "candidate": "graphiti", "kind": "README",
        "url": "https://github.com/getzep/graphiti/blob/main/README.md#L120-L136",
        "fetch": "repos/getzep/graphiti/readme",
        "quote": "Facts have validity windows. When information changes, old facts are",
        "conclusion": "事实有有效窗口、旧事实失效而非删除——正好覆盖我们的养猫→猫去世冲突用例。",
    },
    {
        "id": "GF-03", "candidate": "graphiti", "kind": "README",
        "url": "https://github.com/getzep/graphiti/blob/main/README.md#L125-L128",
        "fetch": "repos/getzep/graphiti/readme",
        "quote": "Episodes & Provenance",
        "conclusion": "内建 episodes 溯源到原始真源，对应旧仓库要求的 Source→Fact 溯源契约。",
    },
    {
        "id": "GF-04", "candidate": "graphiti", "kind": "SOURCE",
        "url": "https://github.com/getzep/graphiti/blob/main/graphiti_core/prompts/extract_edges.py#L60-L69",
        "fetch": "repos/getzep/graphiti/contents/graphiti_core/prompts/extract_edges.py",
        "quote": "When the fact stopped being true. Soft deadlines are not invalid_at values.",
        "conclusion": "EdgeTimestamps.valid_at/invalid_at 是显式字段；我们 temporal.py 已复制其规则常量。",
    },
    {
        "id": "GF-05", "candidate": "graphiti", "kind": "SOURCE",
        "url": "https://github.com/getzep/graphiti/blob/main/graphiti_core/utils/maintenance/edge_operations.py#L438-L460",
        "fetch": "repos/getzep/graphiti/contents/graphiti_core/utils/maintenance/edge_operations.py",
        "quote": "invalidated/contradicted by new information",
        "conclusion": "有真实 invalidate/contradiction 消解流程，不是仅靠 embedding 相似度排序。",
    },
    {
        "id": "GF-06", "candidate": "graphiti", "kind": "SOURCE",
        "url": "https://github.com/getzep/graphiti/blob/main/graphiti_core/search/search_utils.py#L1551",
        "fetch": "repos/getzep/graphiti/contents/graphiti_core/search/search_utils.py",
        "quote": "def maximal_marginal_relevance(",
        "conclusion": "RRF/MMR/normalize_l2 已在 memory/recall.py 逐行复制并落地；换库不构成新增收益。",
    },
    {
        "id": "GF-07", "candidate": "graphiti", "kind": "MANIFEST",
        "url": "https://github.com/getzep/graphiti/blob/main/pyproject.toml#L13-L16",
        "fetch": "repos/getzep/graphiti/contents/pyproject.toml",
        "quote": "neo4j>=5.26.0",
        "conclusion": "硬依赖 Neo4j（或 FalkorDB/Neptune）+ OpenAI key，见 README Installation。",
    },
    # ---- hindsight ----
    {
        "id": "HS-01", "candidate": "hindsight", "kind": "README",
        "url": "https://github.com/vectorize-io/hindsight/blob/main/README.md#L46-L52",
        "fetch": "repos/vectorize-io/hindsight/readme",
        "quote": "state-of-the-art performance on the LongMemEval benchmark",
        "conclusion": "LongMemEval SOTA 且 README 声明第三方复现；记忆质量论据最强，但 benchmark ≠ 我们的语料。",
    },
    {
        "id": "HS-02", "candidate": "hindsight", "kind": "README",
        "url": "https://github.com/vectorize-io/hindsight/blob/main/README.md#L320-L328",
        "fetch": "repos/vectorize-io/hindsight/readme",
        "quote": "Recall performs 4 retrieval strategies in parallel:",
        "conclusion": "语义+BM25+图+时间四路并行，RRF 融合 + cross-encoder 重排；我们的双路召回已覆盖前两路。",
    },
    {
        "id": "HS-03", "candidate": "hindsight", "kind": "README",
        "url": "https://github.com/vectorize-io/hindsight/blob/main/README.md#L350-L352",
        "fetch": "repos/vectorize-io/hindsight/readme",
        "quote": "Each observation keeps its supporting evidence with exact quotes and a proof count",
        "conclusion": "observation 带证据引用+证明计数，新证据 refine 旧信念——可借鉴到我们的 provenance_store。",
    },
    {
        "id": "HS-04", "candidate": "hindsight", "kind": "README",
        "url": "https://github.com/vectorize-io/hindsight/blob/main/README.md#L385",
        "fetch": "repos/vectorize-io/hindsight/readme",
        "quote": "Isolation is strict: no cross-bank leakage.",
        "conclusion": "bank 级强隔离；但我们 UC-4 的 assert_scope 需端到端落到业务主链路，换库不能替代接线。",
    },
    {
        "id": "HS-05", "candidate": "hindsight", "kind": "README",
        "url": "https://github.com/vectorize-io/hindsight/blob/main/README.md#L86",
        "fetch": "repos/vectorize-io/hindsight/readme",
        "quote": "Docker (external PostgreSQL)",
        "conclusion": "默认部署需要外部 PostgreSQL；不是纯 SQLite / pip 即用。",
    },
    {
        "id": "HS-06", "candidate": "hindsight", "kind": "COMPOSE",
        "url": "https://github.com/vectorize-io/hindsight/blob/main/docker/docker-compose/external-pg/docker-compose.yaml#L18-L20",
        "fetch": "repos/vectorize-io/hindsight/contents/docker/docker-compose/external-pg/docker-compose.yaml",
        "quote": "pgvector/pgvector:pg",
        "conclusion": "compose 明确 pgvector 镜像 + HINDSIGHT_API_LLM_API_KEY，运维栈与本项目定位冲突。",
    },
    # ---- MemOS ----
    {
        "id": "MS-01", "candidate": "memos", "kind": "README",
        "url": "https://github.com/MemTensor/MemOS/blob/main/README.md#L51",
        "fetch": "repos/MemTensor/MemOS/readme",
        "quote": "A single API to add, retrieve, edit, and delete memory",
        "conclusion": "显式 edit/delete 正是用户控制权所需；但我们的 confirm/correct/export 已自建同等语义。",
    },
    {
        "id": "MS-02", "candidate": "memos", "kind": "README",
        "url": "https://github.com/MemTensor/MemOS/blob/main/README.md#L55",
        "fetch": "repos/MemTensor/MemOS/readme",
        "quote": "Memory Feedback & Correction",
        "conclusion": "自然语言反馈纠错值得借鉴，但黑盒 LLM 改写与旧仓库「旧版本 supersede、不静默覆盖」红线需核对。",
    },
    {
        "id": "MS-03", "candidate": "memos", "kind": "README",
        "url": "https://github.com/MemTensor/MemOS/blob/main/README.md#L110-L115",
        "fetch": "repos/MemTensor/MemOS/readme",
        "quote": "Neo4j + Qdrant",
        "conclusion": "README 自托管要 Neo4j + Qdrant；不是零运维。",
    },
    {
        "id": "MS-04", "candidate": "memos", "kind": "README",
        "url": "https://github.com/MemTensor/MemOS/blob/main/README.md#L110-L115",
        "fetch": "repos/MemTensor/MemOS/readme",
        "quote": "None (local SQLite)",
        "conclusion": "Local Plugin 确实 100% 本地 SQLite，但它是 DSH/Hermes/OpenClaw 的 npm 插件，不是可嵌进 FastAPI 后端的 Python 库。",
    },
    {
        "id": "MS-05", "candidate": "memos", "kind": "MANIFEST",
        "url": "https://github.com/MemTensor/MemOS/blob/main/pyproject.toml#L38-L45",
        "fetch": "repos/MemTensor/MemOS/contents/pyproject.toml",
        "quote": "pymysql (>=1.1.0,<2.0.0)",
        "conclusion": "核心依赖含 pymysql；MySQL 不是误解。",
    },
    {
        "id": "MS-06", "candidate": "memos", "kind": "MANIFEST",
        "url": "https://github.com/MemTensor/MemOS/blob/main/pyproject.toml#L102",
        "fetch": "repos/MemTensor/MemOS/contents/pyproject.toml",
        "quote": "pymilvus (>=2.5.12,<3.0.0)",
        "conclusion": "可选依赖含 pymilvus/neo4j/qdrant，按部署形态至少再起 1~2 个服务。",
    },
]

CANDIDATES: dict[str, dict[str, Any]] = {
    "graphiti": {
        "repo": "getzep/graphiti",
        "license": "Apache-2.0",
        "stars_at_spike": 32976,
        "pushed_at": "2026-11-20T17:36:25Z",
        "positioning": "时序知识图谱构建框架（不是成品 agent memory 服务）",
        "strengths": [
            "双时间窗 valid_at/invalid_at + 自动旧边失效，保留历史（GF-02/04/05）",
            "episodes 溯源到原始真源（GF-03）",
            "语义+BM25+图混合检索（README Why Graphiti）",
            "其 RRF/MMR/normalize_l2/时序 prompt 规则已被我们逐行吸收（GF-06 + memory/recall.py、temporal.py 文件头）",
        ],
        "gaps_vs_us": [
            "需要 Neo4j/FalkorDB + LLM + embedder，写入按条付 LLM 税（GF-07）",
            "无内建用户确认 Candidate 状态机、遗忘/画像；多租户/鉴权/入库/运维全要自建（GF-01）",
            "我们的冲突用例已由 recency+supersede 排序处理；换库不解决 Recall@K=0.55、负例注入率 1.0 的自有问题",
        ],
        "migration_cost": {
            "new_services": ["Neo4j 或 FalkorDB", "LLM API（或本地 LLM 服务）", "embedder"],
            "code_touchpoints": [
                "memory/store.py 写入链路改为 add_episode",
                "memory/recall.py 改图查询（现有 RRF/MMR 可复用）",
                "memory/confirm.py 的 Candidate 确认语义要重新映射",
                "memory/provenance_store.py 与 episode 溯源对齐",
                "requirements + 部署 compose + 数据迁移（33 条语料只是评测集，真库量未知）",
            ],
            "irreversible_risk": "图库 schema + LLM 抽取结果不可回滚到现有五层库结构",
        },
        "verdict": "DO_NOT_SWITCH",
        "worth_switching": False,
        "why": "它最强的时序失效/episodes 两点，我们已用零 LLM 的 supersede+provenance_store 覆盖主要行为；换库换来的是图库+LLM 运维与新不一致风险。",
        "revisit_if": "产品出现明确需求：按任意历史时点查询『当时什么是真的』，且规模超出单库 SQLite 能力。",
    },
    "hindsight": {
        "repo": "vectorize-io/hindsight",
        "license": "MIT",
        "stars_at_spike": 62372,
        "pushed_at": "2026-11-21T12:20:32Z",
        "positioning": "agent memory 服务：retain/recall/reflect 学习闭环 + mental models",
        "strengths": [
            "LongMemEval SOTA 且有第三方复现（HS-01）",
            "四路检索 + RRF + cross-encoder（HS-02）",
            "observation 带证据引用与证明计数、refine 不覆盖（HS-03）",
            "bank 强隔离 + Memory Defense PII 扫描（HS-04）",
        ],
        "gaps_vs_us": [
            "Postgres+pgvector + LLM key（HS-05/06）；cross-encoder 与自带模型再加重依赖",
            "retain 阶段 LLM 抽取 + 后台 consolidate/reflect 是黑盒，难满足旧仓库 8 类溯源问题与我们『先确认后成事实』红线",
            "bank 隔离不能替代 UC-4 assert_scope 接主链路这一具体待办",
        ],
        "migration_cost": {
            "new_services": ["PostgreSQL + pgvector", "LLM API", "可选 cross-encoder/rerank 模型"],
            "code_touchpoints": [
                "所有 add/recall 入口改 retain/recall 客户端",
                "五层库（Source/Candidate/Fact/Event/Summary/Index）要压平成 bank+observation+mental model",
                "确认流/纠错流/删除级联要重做，现有 confirm/correct/export/isolation 全部作废",
                "评测口径要从我们的 18 query 重建，LongMemEval 分数不能直接迁移",
            ],
            "irreversible_risk": "『模型学习出的信念』不可由用户逐条审计，与可纠错可删除的产品红线正面冲突",
        },
        "verdict": "DO_NOT_SWITCH",
        "worth_switching": False,
        "why": "记忆质量论据最强，但其价值来自 LLM 学习闭环；本项目红线是先确认后成事实、可逐条溯源纠错，且 0 LLM 成本结构。benchmark 高不等于我们语料高。",
        "revisit_if": "产品明确接受黑盒学习信念、且愿意承担 Postgres+LLM+重排模型运维，重新用我们的 18 query 做端到端复测后再定。",
    },
    "memos": {
        "repo": "MemTensor/MemOS",
        "license": "Apache-2.0",
        "stars_at_spike": 12470,
        "pushed_at": "2026-11-20T11:27:29Z",
        "positioning": "Memory OS：统一 add/retrieve/edit/delete + L1/L2/L3 分层",
        "strengths": [
            "显式 edit/delete + 自然语言反馈纠错（MS-01/02）",
            "图结构可检视，不是纯 embedding 黑盒（README）",
            "Local Plugin 100% 本地 SQLite（MS-04）——但仅面向 DSH/Hermes/OpenClaw",
        ],
        "gaps_vs_us": [
            "核心依赖 pymysql（MS-05），可选 pymilvus/neo4j/qdrant（MS-06）；README 自托管写 Neo4j+Qdrant（MS-03）",
            "Local Plugin 是 npm 插件，无法作为 Python 后端依赖集成；TS 主仓 + Python 包的双栈维护",
            "基准自报（scan 文档口径），且与本项目既有 confirm/correct/export 语义重复",
        ],
        "migration_cost": {
            "new_services": ["MySQL 或 Neo4j+Qdrant", "LLM API"],
            "code_touchpoints": [
                "FastAPI 后端无法直接消费 npm 本地插件，只能走自托管 REST/Python 包",
                "用户控制权 3 路由（confirm/correct/delete）要迁到 MemOS 的 edit/feedback 语义",
                "L1/L2/L3 与我们的五层库要做字段映射与重复消除",
            ],
            "irreversible_risk": "本地插件与自托管核心是两套数据模型，选错入口后续迁移成本高",
        },
        "verdict": "DO_NOT_SWITCH",
        "worth_switching": False,
        "why": "它最贴用户控制权，但我们已自建同等能力且保持 SQLite+0 LLM；其本地 SQLite 方案不在我们后端可售范围内，核心方案要多起数据库服务。",
        "revisit_if": "产品决定把记忆层整体外包给带 GUI 的 agent 运行时（而非自持后端），再评估 Local Plugin。",
    },
}

# 边界类型 → 三候选的「文档声明机制」预测。等级不是实测分数。
SUPPORT = {"EXPLICIT": "EXPLICIT_DOC", "PARTIAL": "PARTIAL_DOC", "NONE": "NO_DOC_CLAIM"}
BOUNDARY_RULES: dict[str, dict[str, tuple[str, str, list[str]]]] = {
    "冲突事实-只召回新事实": {
        "graphiti": ("EXPLICIT", "invalid_at 自动失效旧边，查询当前真值只返新事实（GF-02/05）", ["GF-02", "GF-05"]),
        "hindsight": ("PARTIAL", "observation refine 旧信念而非显式删除，是否只返新事实无文档承诺（HS-03）", ["HS-03"]),
        "memos": ("PARTIAL", "有显式 edit/delete，但需要调用方自己决定何时改（MS-01）", ["MS-01"]),
    },
    "冲突事实-新事实排前": {
        "graphiti": ("EXPLICIT", "bi-temporal 查询区分 valid time，新事实天然在『当前』窗口（GF-02/04）", ["GF-02", "GF-04"]),
        "hindsight": ("PARTIAL", "四路检索含 temporal 路 + 重排，但排序承诺未在文档给出（HS-02）", ["HS-02"]),
        "memos": ("NONE", "未检到新旧事实排序口径的源码/文档声明", ["MS-01"]),
    },
    "冲突事实-新旧关系": {
        "graphiti": ("EXPLICIT", "同实体关系失效机制，现伴侣/前任可建不同 valid_at 边（GF-02/05）", ["GF-02", "GF-05"]),
        "hindsight": ("PARTIAL", "mental model 重写可能吸收，但不是逐事实可审计（HS-03）", ["HS-03"]),
        "memos": ("PARTIAL", "edit/feedback 可改，但需人工触发（MS-02）", ["MS-02"]),
    },
    "相似主题干扰": {
        "graphiti": ("NONE", "图+语义检索不等于主体消歧，文档无 matching 承诺", ["GF-06"]),
        "hindsight": ("PARTIAL", "cross-encoder 重排有利，但无主体消歧承诺（HS-02）", ["HS-02"]),
        "memos": ("NONE", "未检到主体消歧声明", ["MS-01"]),
    },
    "时间敏感-去年": {
        "graphiti": ("EXPLICIT", "bi-temporal 支持按时间点查询（GF-02）", ["GF-02"]),
        "hindsight": ("EXPLICIT", "四路检索显式含 Temporal: Time range filtering（HS-02）", ["HS-02"]),
        "memos": ("NONE", "未检到时间范围检索声明", ["MS-01"]),
    },
    "时间敏感-周年": {
        "graphiti": ("EXPLICIT", "同上，有效时间窗（GF-02）", ["GF-02"]),
        "hindsight": ("EXPLICIT", "temporal retrieval + 时间序列表示（HS-02）", ["HS-02"]),
        "memos": ("NONE", "未检到周年/时间范围检索声明", ["MS-01"]),
    },
    "无关记忆-负例": {
        "graphiti": ("NONE", "未检到『无相关记忆时返回空』的契约（RAG 系统普遍返回 top-k）", ["GF-06"]),
        "hindsight": ("NONE", "未检到负例空返回契约", ["HS-02"]),
        "memos": ("NONE", "未检到负例空返回契约", ["MS-01"]),
    },
}
DEFAULT_RULE: dict[str, tuple[str, str, list[str]]] = {
    "graphiti": ("PARTIAL", "混合检索可召回，但中文主体/语义质量无文档承诺（GF-06）", ["GF-06"]),
    "hindsight": ("PARTIAL", "四路检索+RRF 可召回，中文语义质量未在本语料实测（HS-02）", ["HS-02"]),
    "memos": ("PARTIAL", "统一 retrieve 可召回，排序质量无声明（MS-01）", ["MS-01"]),
}

RUNTIME_PREREQ = {
    "measured_at": "2026-11-21",
    "checks": [
        {"item": "docker CLI", "result": "PRESENT"},
        {"item": "127.0.0.1:7687 (Neo4j)", "result": "CLOSED"},
        {"item": "127.0.0.1:5432 (Postgres)", "result": "OPEN（但非本项目 pgvector 实例，且无 LLM key）"},
        {"item": "127.0.0.1:3306 (MySQL)", "result": "CLOSED"},
        {"item": "LLM API keys (OPENAI/ANTHROPIC/DEEPSEEK/HINDSIGHT)", "result": "ALL_ABSENT（只记录存在性，不记明文）"},
    ],
    "lightweight_checks": [
        {
            "candidate": "graphiti",
            "status": "MEASURED_INSTALL_ONLY",
            "command": "python -m venv $TEMP/fw-spike-venv; pip install graphiti-core; python -c 'import graphiti_core'",
            "output": "pip exit=0, graphiti-core 0.31.2; 首次 import 报 ModuleNotFoundError: No module named 'httpx'; 手动 pip install httpx 后 import OK",
            "note": "只证明可 pip 安装，不构成行为实测；上游依赖声明缺 httpx 传递依赖是实测到的事实。",
        },
        {"candidate": "hindsight", "status": "NOT_RUN",
         "reason": "需 Postgres+pgvector 服务 + HINDSIGHT_API_LLM_API_KEY（HS-05/06）；无 key，不装整包。"},
        {"candidate": "memos", "status": "NOT_RUN",
         "reason": "核心需 MySQL 或 Neo4j+Qdrant + LLM（MS-03/05/06）；本地 SQLite 版本是 npm 插件，不适用于本后端。"},
    ],
    "end_to_end_behavior": "NOT_RUN",
    "end_to_end_reason": "三个候选写入链路都要 LLM 抽取 + 外部数据库；本机无 LLM key、无 Neo4j/MySQL。按纪律标 NOT_RUN，不用替身数字硬凑。",
}


def load_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    return corpus, baseline


def build_case_matrix(corpus: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for q in corpus["queries"]:
        boundary = q.get("boundary") or "普通召回"
        rule = BOUNDARY_RULES.get(boundary, DEFAULT_RULE)
        predictions: dict[str, Any] = {}
        for cand in CANDIDATES:
            level, mechanism, evidence_ids = rule[cand]
            predictions[cand] = {
                "support_level": SUPPORT[level],
                "mechanism": mechanism,
                "evidence_ids": evidence_ids,
                "execution_status": "NOT_RUN",
                "evidence_grade": "DOC_LEVEL_PREDICTION",
            }
        rows.append({
            "query_id": q["id"],
            "boundary": q.get("boundary"),
            "memory_query": q["memory_query"],
            "expected_first": q.get("expected_first"),
            "memory_expect": q["memory_expect"],
            "candidate_predictions": predictions,
        })
    return rows


def build_report(corpus: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    case_matrix = build_case_matrix(corpus)
    support_counts = {
        cand: dict(Counter(row["candidate_predictions"][cand]["support_level"] for row in case_matrix))
        for cand in CANDIDATES
    }
    best = baseline.get("dual1_sem1", {})
    return {
        "schema_version": "1.0",
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "spike_date": DATE,
            "task": "U-009 记忆框架 3 候选 spike（D-015 承诺）",
            "baseline_commit": "main 29cda50（269 passed）",
            "corpus": {
                "path": "MakeAgain-backend/eval/recall_corpus.json",
                "source": corpus["meta"]["source"],
                "memories": len(corpus["memories"]),
                "queries": len(corpus["queries"]),
                "caveat": corpus["meta"]["source_ref"],
            },
            "method": "gh api 源码/文档实读 + 语料逐项机制对照；端到端行为 NOT_RUN",
            "evidence_grade_legend": {
                "DOC_LEVEL_PREDICTION": "由上游 README/源码声明推导，未运行上游系统",
                "MEASURED_INSTALL_ONLY": "只做 pip 安装/import，未跑行为",
                "NOT_RUN": "缺服务/密钥，未运行且不硬凑数字",
            },
        },
        "runtime_prerequisites": RUNTIME_PREREQ,
        "current_stack": {
            "architecture": "agent-memory-core + 自研拼装（SQLite + 确定性 hash embedding + recency/importance/relevance 打分 + 五层库 + confirm/correct/isolation）",
            "measured_baseline_from_existing_output": {
                "source": "outputs/golden-recall-v2.json::dual1_sem1（既有产物，本次未重跑）",
                "Recall@K": best.get("Recall@K"),
                "Precision@K": best.get("Precision@K"),
                "MRR": best.get("MRR"),
                "expected_first_hit_rate": best.get("expected_first_hit_rate"),
                "negative_query_injection_rate": best.get("negative_query_injection_rate"),
                "n_queries": best.get("n_queries"),
            },
            "honest_reading": "最佳配置 Recall@K 仅 0.5529、负例注入率 1.0；真问题是自己的召回/query-intent，不是缺框架。",
            "absorbed_upstream_algorithms": [
                "graphiti: rrf / maximal_marginal_relevance / normalize_l2 / valid_at-invalid_at prompt 规则",
                "mem0: scoring / entity boost（见 memory/recall.py 文件头来源声明）",
            ],
        },
        "evidence": EVIDENCE,
        "candidates": CANDIDATES,
        "case_matrix": case_matrix,
        "case_support_summary": support_counts,
        "cross_case_focus": {
            "cat_conflict": {
                "queries": ["q-02", "q-03", "q-17"],
                "our_expectation": "q-02 只召回 mem-cat-deceased；q-03 新事实排前；q-17 埋葬地点主答为新事实",
                "finding": "graphiti 有显式失效机制（文档层最强）；hindsight/memOS 靠重写或人工 edit（文档层 PARTIAL）。三者均未实测。",
            },
            "partner_conflict": {
                "queries": ["q-15"],
                "our_expectation": "现伴侣 mem-b-06 命中，前任 mem-b-01 不召回",
                "finding": "graphiti 可用不同 valid_at 边表达；其它两者无逐事实可审计承诺。",
            },
            "negative_query": {
                "queries": ["q-16"],
                "our_expectation": "top-k 不命中任何记忆",
                "finding": "三候选文档均未给『无相关记忆返回空』契约；我方当前实测负例注入率 1.0，属自身待修，不靠换库解决。",
            },
            "time_sensitive": {
                "queries": ["q-06", "q-13"],
                "our_expectation": "按 fixed_now=2026-11-21 解释去年/周年",
                "finding": "graphiti 双时间窗与 hindsight temporal 路在文档层显式支持；memOS 未检到声明。",
            },
        },
        "migration_comparison": {
            "common_costs": ["重做写入/召回/确认/纠错/删除链路", "数据迁移与双写", "重跑 18 query 评测（LongMemEval 分数不可迁移）"],
            "service_count": {"graphiti": " Neo4j/FalkorDB + LLM + embedder", "hindsight": "Postgres+pgvector + LLM", "memos": "MySQL 或 Neo4j+Qdrant + LLM"},
            "llm_tax": "三者写入都要 LLM 抽取；现栈写入 0 LLM / 检索 0 LLM（MEM0_EVAL 实测对照口径）。",
        },
        "final_decision": {
            "verdict": "KEEP_CURRENT_STACK_SELECTIVE_BORROW",
            "switch_any": False,
            "per_candidate": {
                cand: {"worth_switching": data["worth_switching"], "verdict": data["verdict"],
                       "why": data["why"], "revisit_if": data["revisit_if"]}
                for cand, data in CANDIDATES.items()
            },
            "rationale": [
                "没有任一候选值得整体替换：能力最强项要么已被吸收（graphiti 算法），要么与产品红线冲突（hindsight 黑盒学习、memos 多服务）。",
                "全绿证据不等于换库成功：三者端到端均 NOT_RUN，换库收益无法在本语料上量化。",
                "当前质量缺口（Recall@K 0.5529、负例注入率 1.0）是我方 query-intent / 召回权重问题，就地修更便宜（对应 WP-3）。",
                "继续借鉴：graphiti 时序失效规则、hindsight observation 证据计数、memOS 显式 edit/feedback 语义，均按零 LLM 方式自建。",
            ],
        },
        "scope_guard": {
            "read_only_inputs": [
                str(CORPUS_PATH.relative_to(PROJECT)),
                str(BASELINE_PATH.relative_to(PROJECT)),
            ],
            "sole_write_target": str(OUTPUT_PATH.relative_to(PROJECT)),
            "forbidden_touched": ["业务代码", "requirements", "PROGRESS.md", "队友仓库", "前端"],
            "assertion": "本脚本不 import 业务模块、不写除 sole_write_target 外的任何路径。",
        },
    }


def self_check(report: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    if len(report["candidates"]) != 3:
        errs.append("候选数 != 3")
    for cand in CANDIDATES:
        ev = [e for e in EVIDENCE if e["candidate"] == cand]
        if len(ev) < 4:
            errs.append(f"{cand} 证据不足 4 条")
        for e in ev:
            if not all(e.get(k) for k in ("url", "quote", "conclusion", "fetch")):
                errs.append(f"{e.get('id')} 缺 url/quote/conclusion/fetch")
    qids = [q["id"] for q in json.loads(CORPUS_PATH.read_text(encoding="utf-8"))["queries"]]
    got = [r["query_id"] for r in report["case_matrix"]]
    if len(qids) != 18 or got != qids:
        errs.append("case_matrix 未逐条覆盖 18 query")
    for row in report["case_matrix"]:
        for cand in CANDIDATES:
            p = row["candidate_predictions"][cand]
            if p["execution_status"] != "NOT_RUN" or p["evidence_grade"] != "DOC_LEVEL_PREDICTION":
                errs.append(f"{row['query_id']}/{cand} 执行状态或证据等级不合规")
    if report["runtime_prerequisites"]["end_to_end_behavior"] != "NOT_RUN":
        errs.append("端到端未标 NOT_RUN")
    if report["final_decision"]["switch_any"] is not False:
        errs.append("去留结论不是明确的『不换』")
    if len(set(v["verdict"] for v in CANDIDATES.values())) != 1:
        errs.append("结论含糊（不是三选都不换的明确判定）")
    if OUTPUT_PATH.parent.name != "outputs" or not re.fullmatch(r"framework-spike-\d{4}-\d{2}-\d{2}\.json", OUTPUT_PATH.name):
        errs.append("写目标越权或文件名不合规")
    return errs


def verify_evidence() -> int:
    """用 gh api 重新拉取每个 fetch 目标，确认 quote 仍逐字存在。"""
    ok = fail = 0
    cache: dict[str, str] = {}
    for e in EVIDENCE:
        fetch = e["fetch"]
        if fetch not in cache:
            with tempfile.NamedTemporaryFile(delete=False) as tf:
                out_path = tf.name
            cmd = ["gh", "api", "-H", "Accept: application/vnd.github.raw+json", fetch]
            try:
                with open(out_path, "wb") as fh:
                    proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.DEVNULL, check=False)
                cache[fetch] = Path(out_path).read_text(encoding="utf-8", errors="replace") if proc.returncode == 0 else ""
            except OSError as exc:
                cache[fetch] = f"__ERROR__{exc}"
            finally:
                Path(out_path).unlink(missing_ok=True)
        content = cache[fetch]
        hit = content.startswith("__ERROR__") or e["quote"] in content
        mark = "OK" if e["quote"] in content else ("FETCH_ERROR" if content.startswith("__ERROR__") else "MISS")
        if mark == "OK":
            ok += 1
        else:
            fail += 1
        print(f"EVIDENCE {e['id']:5s} {mark:11s} {e['url']}")
    print(f"VERIFY_RESULT ok={ok} fail={fail}")
    return 0 if fail == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true", help="gh api 复验证据 quote")
    ap.add_argument("--check", action="store_true", help="只跑自检不写文件")
    args = ap.parse_args()

    if args.verify:
        return verify_evidence()

    corpus, baseline = load_inputs()
    report = build_report(corpus, baseline)
    errs = self_check(report)
    print(f"[scope] read={report['scope_guard']['read_only_inputs']}")
    print(f"[scope] write={report['scope_guard']['sole_write_target']}")
    print(f"[corpus] memories={len(corpus['memories'])} queries={len(corpus['queries'])}")
    print(f"[evidence] total={len(EVIDENCE)} per_candidate="
          f"{dict(Counter(e['candidate'] for e in EVIDENCE))}")
    print(f"[case_matrix] rows={len(report['case_matrix'])} all_execution_status=NOT_RUN")
    print(f"[support_summary] {report['case_support_summary']}")
    print(f"[decision] {report['final_decision']['verdict']} switch_any={report['final_decision']['switch_any']}")
    print(f"[NOT_RUN] end_to_end={report['runtime_prerequisites']['end_to_end_behavior']}")
    print(f"[self_check] errors={errs}")
    if errs:
        return 1
    if args.check:
        print("[write] skipped (--check)")
        return 0
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[write] {OUTPUT_PATH.relative_to(PROJECT)} bytes={OUTPUT_PATH.stat().st_size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
