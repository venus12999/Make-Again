"""召回流程:recency + importance + relevance 混合打分。

R1 多路融合(开关 USE_RECALL_FUSION,默认关闭):
  关闭 / 新路径异常 → _recall_legacy():与 main@45f3351 的实现逐字节一致。
  开启 → 三路候选(recency / importance / relevance)→ rrf 融合
       → score_and_rank 精排 → MMR 去冗余,返回值仍是 list[dict],调用方零改动。
WP-A 双路召回(USE_DUAL_RECALL,默认关闭):关闭=上面 R1/legacy 路径逐字节不变;
开启=不调用 list_memories 全量读取,走 vec0 语义路 + bm25s 关键词路 → rrf 融合
→ score_and_rank 精排 → MMR 去冗余,返回仍是 list[dict]。
embedding 是确定性 hash(非语义模型),详见 memory/vector_index.py 文件头声明。
U-013 query-intent 门(开关 USE_QUERY_INTENT,默认关闭):
   关闭 → 本模块原有行为逐字节不变;
   开启 → 召回前用确定性规则判 query 是否与记忆相关(见 memory/query_intent.py),
          判为无关(如"今天晚饭吃什么")直接返回 [],不再注入旧记忆。

复制资产(全部经 gh api 实读源码、base64 解码,按行号复制):
  - rrf()                      graphiti `graphiti_core/search/search_utils.py:1551-1566`
  - maximal_marginal_relevance graphiti `graphiti_core/search/search_utils.py:1670-1708`
  - normalize_l2()             graphiti `graphiti_core/helpers.py:125-128`
  - get_bm25_params / normalize_bm25 / score_and_rank / ENTITY_BOOST_WEIGHT
                               mem0 `mem0/utils/scoring.py:16-139`
  - 热门实体平方衰减公式        mem0 `mem0/memory/main.py:1747-1827`(核心 1815-1822)
"""
from __future__ import annotations

import logging
import math
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from time import time
from typing import Any, Dict, List, Optional

from config import get_settings
from memory.facts import facts_text
from memory.store import list_memories
from memory.models import MemoryEntry
from memory.vector_index import embed_text, get_index_for

logger = logging.getLogger(__name__)
_TOKEN_RE = re.compile(r"[一-鿿]|[a-zA-Z0-9]+")
DEFAULT_MMR_LAMBDA = 0.5          # graphiti search_utils.py:66
ENTITY_BOOST_WEIGHT = 0.5         # mem0 utils/scoring.py:57


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text or ""))


def _relevance(query_tokens: set[str], entry_tokens: set[str]) -> float:
    """余弦式关键词重叠,无重叠为 0。"""
    if not query_tokens or not entry_tokens:
        return 0.0
    inter = query_tokens & entry_tokens
    if not inter:
        return 0.0
    return len(inter) / math.sqrt(len(query_tokens) * len(entry_tokens))


def _overlap_fraction(entry_tokens: set[str], recent_tokens: set[str]) -> float:
    """记忆自身词汇里已有多少比例出现在最近对话中(用于克制重复引用)。

    不对称:分母是记忆自身的词数。若记忆的大半关键词都已在最近对话里出现,
    说明这件事正在被聊,无需再作为「相关记忆」注入——否则刚说完又提,显得机械。
    """
    if not entry_tokens:
        return 0.0
    return len(entry_tokens & recent_tokens) / len(entry_tokens)


# ---------------------------------------------------------------------------
# 原路径:以下函数体逐字节保留 main@45f3351 的 recall.py:40-62(仅函数名改别名)。
# 开关关闭 / 新路径异常一律走这里,保证基线行为零变化。
# ---------------------------------------------------------------------------
def _recall_legacy(db, user_id: str, query: str, top_k: int | None = None, recent_text: str | None = None) -> list[dict]:
    s = get_settings()
    top_k = top_k or s.recall_top_k
    entries = list_memories(db, user_id)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    q_tokens = _tokens(query)
    recent_tokens = _tokens(recent_text) if recent_text else set()

    scored: list[dict] = []
    for e in entries:
        entry_tokens = _tokens(f"{facts_text(e.facts)} {e.summary or ''} {e.content}")
        # 克制:已出现在最近几轮对话里的记忆不再注入(它已在会话历史里,模型本就看得见)
        if recent_tokens and _overlap_fraction(entry_tokens, recent_tokens) >= s.recall_recent_overlap_threshold:
            continue
        days = max(0.0, (now - e.ts).total_seconds() / 86400.0)
        recency = math.exp(-days / 30.0)  # 30 天半衰期
        importance = (e.importance or 5.0) / 10.0
        relevance = _relevance(q_tokens, entry_tokens)
        score = s.recall_alpha * recency + s.recall_beta * importance + s.recall_gamma * relevance
        scored.append({"entry": e, "score": score, "recency": recency, "relevance": relevance})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def _use_recall_fusion_enabled() -> bool:
    """os.getenv 开关,默认关闭(与 USE_ALEMBIC 同风格)。"""
    return os.getenv("USE_RECALL_FUSION", "").strip().lower() in {"1", "true", "yes", "on"}


def _use_dual_recall_enabled() -> bool:
    """WP-A 双路召回开关,默认关闭。关闭时本模块行为与基线 cde75f8 完全一致。"""
    return os.getenv("USE_DUAL_RECALL", "").strip().lower() in {"1", "true", "yes", "on"}


def _use_query_intent_enabled() -> bool:
    """U-013 query-intent 门,默认关闭(需 golden 验证)。关闭时零调用、零行为变化。"""
    return os.getenv("USE_QUERY_INTENT", "").strip().lower() in {"1", "true", "yes", "on"}


def recall(db, user_id: str, query: str, top_k: int | None = None, recent_text: str | None = None) -> list[dict]:
    """对外入口:签名/返回形态不变。开关关闭或融合失败 → 原路径 fallback。"""
    # U-013:召回前判意图,与记忆无关直接短路。纯函数判定,不读 DB/user_id,不越权。
    # 开关关闭时这段代码只做一次 os.getenv,其余逐字节走原路径。
    if _use_query_intent_enabled():
        from memory.query_intent import is_memory_irrelevant
        if is_memory_irrelevant(query):
            logger.debug("query-intent gate: irrelevant query, skip recall")
            return []
    if _use_dual_recall_enabled():
        try:
            return _recall_dual(db, user_id, query, top_k=top_k, recent_text=recent_text)
        except Exception as exc:
            logger.warning("dual recall failed, fallback to R1/legacy path: %s", exc)
    if not _use_recall_fusion_enabled():
        return _recall_legacy(db, user_id, query, top_k=top_k, recent_text=recent_text)
    try:
        return _recall_fusion(db, user_id, query, top_k=top_k, recent_text=recent_text)
    except Exception as exc:
        logger.warning("recall fusion failed, fallback to legacy path: %s", exc)
        return _recall_legacy(db, user_id, query, top_k=top_k, recent_text=recent_text)


# ---------------------------------------------------------------------------
# 复制资产 1:graphiti rrf() —— search/search_utils.py:1551-1566(16 行,零依赖)
# ---------------------------------------------------------------------------
def rrf(
    results: list[list[str]], rank_const=1, min_score: float = 0
) -> tuple[list[str], list[float]]:
    scores: dict[str, float] = defaultdict(float)
    for result in results:
        for i, uuid in enumerate(result):
            scores[uuid] += 1 / (i + rank_const)

    scored_uuids = [term for term in scores.items()]
    scored_uuids.sort(reverse=True, key=lambda term: term[1])

    sorted_uuids = [term[0] for term in scored_uuids]

    return [uuid for uuid in sorted_uuids if scores[uuid] >= min_score], [
        scores[uuid] for uuid in sorted_uuids if scores[uuid] >= min_score
    ]


# ---------------------------------------------------------------------------
# 复制资产 2:mem0 utils/scoring.py:16-139
# ---------------------------------------------------------------------------
def get_bm25_params(query: str, *, lemmatized: Optional[str] = None) -> tuple:
    """Get BM25 sigmoid parameters based on query length.

    Longer queries tend to have higher raw BM25 scores, so we adjust
    the sigmoid midpoint and steepness accordingly.

    Returns:
        (midpoint, steepness) for sigmoid normalization.
    """
    if lemmatized is None:
        # [本地化] mem0 原实现在此 import mem0.utils.lemmatization(带 spaCy 依赖);
        # 本模块调用方始终显式传 lemmatized,此处仅作无依赖兜底。
        lemmatized = query or ""
    num_terms = len(lemmatized.split()) if lemmatized else 1

    if num_terms <= 3:
        return 5.0, 0.7
    elif num_terms <= 6:
        return 7.0, 0.6
    elif num_terms <= 9:
        return 9.0, 0.5
    elif num_terms <= 15:
        return 10.0, 0.5
    else:
        return 12.0, 0.5


def normalize_bm25(raw_score: float, midpoint: float, steepness: float) -> float:
    """Normalize BM25 score to [0, 1] using logistic sigmoid."""
    return 1.0 / (1.0 + math.exp(-steepness * (raw_score - midpoint)))


def score_and_rank(
    semantic_results: List[Dict[str, Any]],
    bm25_scores: Dict[str, float],
    entity_boosts: Dict[str, float],
    threshold: float,
    top_k: int,
    explain: bool = False,
) -> List[Dict[str, Any]]:
    """Score candidates additively and return top-k results.

    Threshold gates the semantic score BEFORE combining -- candidates
    below the threshold are excluded even if BM25/entity would boost them.
    The divisor adapts to which signals are active (mem0 scoring.py:77-101).
    """
    has_bm25 = bool(bm25_scores)
    has_entity = bool(entity_boosts)

    max_possible = 1.0
    if has_bm25:
        max_possible += 1.0
    if has_entity:
        max_possible += ENTITY_BOOST_WEIGHT

    scored: List[Dict[str, Any]] = []

    for result in semantic_results:
        mem_id = result.get("id")
        if mem_id is None:
            continue

        semantic_score = result.get("score") or 0.0
        if semantic_score < threshold:
            continue

        mem_id_str = str(mem_id)
        bm25_score = bm25_scores.get(mem_id_str, 0.0)
        entity_boost = entity_boosts.get(mem_id_str, 0.0)

        raw_combined = semantic_score + bm25_score + entity_boost
        combined = min(raw_combined / max_possible, 1.0)

        scored_result = {
            "id": mem_id_str,
            "score": combined,
            "payload": result.get("payload"),
        }
        if explain:
            scored_result["score_details"] = {
                "semantic_score": semantic_score,
                "bm25_score": bm25_score,
                "entity_boost": entity_boost,
                "raw_score": raw_combined,
                "max_possible_score": max_possible,
                "final_score": combined,
                "threshold": threshold,
            }
        scored.append(scored_result)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# 复制资产 3:graphiti MMR —— search/search_utils.py:1670-1708
# 归一化来自 graphiti_core/helpers.py:125-128(normalize_l2)
# numpy 延迟导入:开关关闭时不引入任何新依赖。
# ---------------------------------------------------------------------------
def normalize_l2(embedding: list[float]):
    import numpy as np

    embedding_array = np.array(embedding)
    norm = np.linalg.norm(embedding_array, 2, axis=0, keepdims=True)
    return np.where(norm == 0, embedding_array, embedding_array / norm)


def maximal_marginal_relevance(
    query_vector: List[float],
    candidates: Dict[str, List[float]],
    mmr_lambda: float = DEFAULT_MMR_LAMBDA,
    min_score: float = -2.0,
) -> tuple[List[str], List[float]]:
    import numpy as np

    start = time()
    query_array = np.array(query_vector)
    candidate_arrays: dict = {}
    for uuid, embedding in candidates.items():
        candidate_arrays[uuid] = normalize_l2(embedding)

    uuids: List[str] = list(candidate_arrays.keys())

    similarity_matrix = np.zeros((len(uuids), len(uuids)))

    for i, uuid_1 in enumerate(uuids):
        for j, uuid_2 in enumerate(uuids[:i]):
            u = candidate_arrays[uuid_1]
            v = candidate_arrays[uuid_2]
            similarity = np.dot(u, v)

            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity

    mmr_scores: dict[str, float] = {}
    for i, uuid in enumerate(uuids):
        max_sim = np.max(similarity_matrix[i, :])
        mmr = mmr_lambda * np.dot(query_array, candidate_arrays[uuid]) + (mmr_lambda - 1) * max_sim
        mmr_scores[uuid] = mmr

    uuids.sort(reverse=True, key=lambda c: mmr_scores[c])

    end = time()
    logger.debug(f'Completed MMR reranking in {(end - start) * 1000} ms')

    return [uuid for uuid in uuids if mmr_scores[uuid] >= min_score], [
        mmr_scores[uuid] for uuid in uuids if mmr_scores[uuid] >= min_score
    ]


def _mmr_diverse_top_k(query_vector, vectors, order, top_k, mmr_lambda=DEFAULT_MMR_LAMBDA) -> List[str]:
    """graphiti 单次 MMR 打分公式的贪心选型:max_sim 改为「对已选集合取最大」。

    graphiti 原函数一次算出每个候选对其余全部候选的最大相似度并排序;本地候选池
    小、词袋向量稀疏,单次打分会把重复项和独特项打平。这里沿用同一公式
    (lambda*sim(q,d) + (lambda-1)*max_sim(d,selected))做贪心挑选,才能真正去掉语义重复项。
    """
    import numpy as np

    q = np.array(query_vector)
    arrays = {k: normalize_l2(v) for k, v in vectors.items()}
    selected: List[str] = []
    remaining = [k for k in order if k in arrays]
    while remaining and len(selected) < top_k:
        best, best_score = None, float("-inf")
        for k in remaining:
            sim_q = float(np.dot(q, arrays[k]))
            max_sim = max((float(np.dot(arrays[k], arrays[s])) for s in selected), default=0.0)
            score = mmr_lambda * sim_q + (mmr_lambda - 1) * max_sim
            if score > best_score:
                best, best_score = k, score
        selected.append(best)
        remaining.remove(best)
    return selected


# ---------------------------------------------------------------------------
# 复制资产 4:mem0 memory/main.py:1747-1827 的「热门实体平方衰减」公式
# 本地无 embedding_model / entity_store / filters:用「命中词项的记忆条数」当
# num_linked(等价于 mem0 的 linked_memory_ids 长度),相似度命中即 1.0(等价于
# mem0 的 embedding cosine >= 0.5 门槛);1815-1822 的公式与 max 聚合原样保留。
# ---------------------------------------------------------------------------
def _compute_entity_boosts(query_entities, candidates) -> Dict[str, float]:
    seen = set()
    deduped = []
    for entity_text in list(query_entities)[:8]:
        key = (entity_text or "").strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(entity_text)

    if not deduped:
        return {}
    memory_boosts: Dict[str, float] = {}

    try:
        for entity_text in deduped:
            matched = [c["id"] for c in candidates if entity_text in c["tokens"]]
            num_linked = max(len(matched), 1)
            # mem0 memory/main.py:1815-1817(原样复制)
            memory_count_weight = 1.0 / (1.0 + 0.001 * ((num_linked - 1) ** 2))
            similarity = 1.0
            boost = similarity * ENTITY_BOOST_WEIGHT * memory_count_weight

            for memory_id in matched:  # mem0 memory/main.py:1819-1822
                if memory_id:
                    memory_key = str(memory_id)
                    memory_boosts[memory_key] = max(memory_boosts.get(memory_key, 0.0), boost)

    except Exception as e:
        logger.warning("Entity boost computation failed: %s", e)

    return memory_boosts


# ---------------------------------------------------------------------------
# 融合主路径:三路候选 → rrf → score_and_rank → MMR
# ---------------------------------------------------------------------------
def _recall_fusion(db, user_id: str, query: str, top_k: int | None = None,
                   recent_text: str | None = None) -> list[dict]:
    s = get_settings()
    top_k = top_k or s.recall_top_k
    entries = list_memories(db, user_id)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    q_tokens = _tokens(query)
    recent_tokens = _tokens(recent_text) if recent_text else set()

    cands: list[dict] = []
    for e in entries:
        entry_tokens = _tokens(f"{facts_text(e.facts)} {e.summary or ''} {e.content}")
        # 与基线一致的「正在聊」克制门槛(原样保留)
        if recent_tokens and _overlap_fraction(entry_tokens, recent_tokens) >= s.recall_recent_overlap_threshold:
            continue
        days = max(0.0, (now - e.ts).total_seconds() / 86400.0)
        cands.append({
            "id": str(e.id),
            "entry": e,
            "tokens": entry_tokens,
            "recency": math.exp(-days / 30.0),
            "importance": (e.importance or 5.0) / 10.0,
            "relevance": _relevance(q_tokens, entry_tokens),
        })
    if not cands:
        return []

    # 三路候选:现有 recency / importance / relevance 各算一路
    paths = [
        [c["id"] for c in sorted(cands, key=lambda c: c["recency"], reverse=True)],
        [c["id"] for c in sorted(cands, key=lambda c: c["importance"], reverse=True)],
        [c["id"] for c in sorted(cands, key=lambda c: c["relevance"], reverse=True)],
    ]
    fused_ids, fused_scores = rrf(paths)
    by_id = {c["id"]: c for c in cands}

    # BM25 近似:词项 idf 之和 → mem0 的 sigmoid 归一化(lemmatized 显式传参,无 spaCy)
    n_cands = len(cands)
    doc_freq: dict[str, int] = {}
    for c in cands:
        for t in q_tokens & c["tokens"]:
            doc_freq[t] = doc_freq.get(t, 0) + 1
    midpoint, steepness = get_bm25_params(query, lemmatized=" ".join(sorted(q_tokens)))
    bm25_scores: Dict[str, float] = {}
    for c in cands:
        raw = sum(
            math.log(1.0 + (n_cands - doc_freq.get(t, 0) + 0.5) / (doc_freq.get(t, 0) + 0.5))
            for t in q_tokens if t in c["tokens"]
        )
        bm25_scores[c["id"]] = normalize_bm25(raw, midpoint, steepness)

    entity_boosts = _compute_entity_boosts(q_tokens, cands)

    # RRF 分数归一化到 [0,1] 作为 score_and_rank 的 semantic_score
    # (rank_const=1 → 单路满分 1.0,三路满分 = len(paths))
    semantic = [
        {"id": mid, "score": min(sc / len(paths), 1.0), "payload": by_id[mid]}
        for mid, sc in zip(fused_ids, fused_scores)
    ]
    threshold = float(os.getenv("USE_RECALL_FUSION_THRESHOLD", "0") or 0)
    ranked = score_and_rank(semantic, bm25_scores, entity_boosts, threshold, len(semantic), explain=True)

    if not ranked:
        return []

    # 本地词袋 embedding 向量(零外部模型),归一化复用 graphiti normalize_l2
    ranked_ids = [r["id"] for r in ranked]
    vocab = sorted(set().union(*(by_id[mid]["tokens"] for mid in ranked_ids)) | q_tokens)
    if not vocab:
        return [{"entry": by_id[r["id"]]["entry"], "score": r["score"],
                 "recency": by_id[r["id"]]["recency"], "relevance": by_id[r["id"]]["relevance"],
                 "score_details": r.get("score_details")} for r in ranked[:top_k]]
    idx = {t: i for i, t in enumerate(vocab)}
    vectors = {}
    for mid in ranked_ids:
        v = [0.0] * len(vocab)
        for t in by_id[mid]["tokens"]:
            v[idx[t]] = 1.0
        vectors[mid] = v
    qvec = [1.0 if t in q_tokens else 0.0 for t in vocab]

    detail = {r["id"]: r for r in ranked}
    mmr_addrs, mmr_scores = maximal_marginal_relevance(qvec, vectors)
    mmr_by_id = dict(zip(mmr_addrs, mmr_scores))
    # 主序保持融合精排的相关性,graphiti 单次 MMR 分数只做同分时的去冗余偏好;
    # 真正的重复项剔除由下面的贪心 MMR 完成(避免无关项被单次公式抬到第一)。
    order = sorted(ranked_ids, key=lambda mid: (detail[mid]["score"], mmr_by_id.get(mid, 0.0)), reverse=True)
    final_ids = _mmr_diverse_top_k(qvec, vectors, order, top_k)

    out = [
        {"entry": by_id[mid]["entry"],
         "score": detail[mid]["score"],
         "recency": by_id[mid]["recency"],
         "relevance": by_id[mid]["relevance"],
         "score_details": detail[mid].get("score_details")}
        for mid in final_ids
    ]
    return out


# ---------------------------------------------------------------------------
# WP-A:向量 + 关键词双路召回（开关 USE_DUAL_RECALL）
#
# 与 R1 _recall_fusion 的本质区别：这里**完全不调用 list_memories**。
# 候选来自 sqlite-vec vec0 的 KNN 与 bm25s 的 BM25 检索；命中 id 后再用
# db.get(MemoryEntry, id) 做点查拿真实 ORM 对象（返回契约与基线一致）。
# 索引由 index_memory() / rebuild_index() 写入，由 purge_memory() 级联清理。
# ---------------------------------------------------------------------------
def _dual_entry_payload(entry) -> tuple:
    """从 ORM 对象或 dict 抽索引所需字段（与基线召回使用的文本口径一致）。"""
    if isinstance(entry, dict):
        user_id = str(entry.get("user_id") or "")
        text = f"{facts_text(entry.get('facts'))} {entry.get('summary') or ''} {entry.get('content') or ''}"
        mid = str(entry.get("id") or "")
        ts, importance = entry.get("ts"), entry.get("importance")
    else:
        user_id = str(getattr(entry, "user_id", "") or "")
        text = (f"{facts_text(getattr(entry, 'facts', None))} "
                f"{getattr(entry, 'summary', None) or ''} {getattr(entry, 'content', '') or ''}")
        mid = str(getattr(entry, "id", "") or "")
        ts, importance = getattr(entry, "ts", None), getattr(entry, "importance", 5.0)
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts)
        except ValueError:
            ts = None
    return mid, user_id, text, ts, (float(importance) if importance is not None else 5.0)


def index_memory(entry, db=None) -> bool:
    """供 store.add_memory 写后调用的 upsert（幂等）。返回是否成功；失败只告警不影响业务。"""
    try:
        mid, uid, text, ts, imp = _dual_entry_payload(entry)
        if not mid:
            return False
        get_index_for(db).upsert(mid, uid, text, ts=ts, importance=imp, embedding=embed_text(text))
        return True
    except Exception as exc:
        logger.warning("dual index upsert failed: %s", exc)
        return False


def purge_memory(memory_id: str, db=None) -> bool:
    """供 store 删除路径调用：级联清理 vec0 向量 + bm25s 元数据，返回是否删到行。"""
    try:
        return get_index_for(db).purge(memory_id)
    except Exception as exc:
        logger.warning("dual index purge failed: %s", exc)
        return False


def rebuild_index(db, user_id: str, entries: list) -> int:
    """bootstrap：从调用方传入的 entries 重建某用户索引；本函数不调用 list_memories。"""
    idx = get_index_for(db)
    idx.clear_user(user_id)
    return sum(1 for e in entries if index_memory(e, db=db))


def _recall_dual(db, user_id: str, query: str, top_k: int | None = None,
                 recent_text: str | None = None) -> list[dict]:
    """vec0 语义路 + bm25s 关键词路 → rrf → score_and_rank → MMR。"""
    s = get_settings()
    top_k = top_k or s.recall_top_k
    idx = get_index_for(db)
    qvec = embed_text(query)
    fetch_k = max(top_k * 4, 20)

    vec_hits = idx.vector_search(qvec, user_id, fetch_k)
    kw_hits = idx.keyword_search(query, user_id, fetch_k)
    if not vec_hits and not kw_hits:
        logger.debug("dual recall: empty index for user %s (call rebuild_index/index_memory)", user_id)
        return []

    cand_ids = list(dict.fromkeys([h["id"] for h in vec_hits] + [h["id"] for h in kw_hits]))
    meta = idx.get_meta(cand_ids)
    if not meta:
        return []

    q_tokens = _tokens(query)
    recent_tokens = _tokens(recent_text) if recent_text else set()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    vec_scores = {h["id"]: h["score"] for h in vec_hits}
    kw_scores = {h["id"]: h["score"] for h in kw_hits}

    # 复用 graphiti rrf()：两路有序 id 列表融合
    fused_ids, fused_scores = rrf([[h["id"] for h in vec_hits], [h["id"] for h in kw_hits]])
    fused_map = dict(zip(fused_ids, fused_scores))
    fused_ids = [mid for mid in fused_ids if mid in meta]
    if not fused_ids:
        return []

    cands: list[dict] = []
    for mid in fused_ids:
        m = meta[mid]
        entry_tokens = m["tokens"]
        if recent_tokens and _overlap_fraction(entry_tokens, recent_tokens) >= s.recall_recent_overlap_threshold:
            continue
        ts = m["ts"] or now
        days = max(0.0, (now - ts).total_seconds() / 86400.0)
        cands.append({
            "id": mid,
            "tokens": entry_tokens,
            "recency": math.exp(-days / 30.0),
            "relevance": _relevance(q_tokens, entry_tokens),
            "vec_score": vec_scores.get(mid, 0.0),
            "kw_score": kw_scores.get(mid, 0.0),
            "embedding": m["embedding"],
        })
    if not cands:
        return []
    by_id = {c["id"]: c for c in cands}

    # RRF 分数归一到 [0,1] 作为 semantic_score（复用 mem0 score_and_rank）
    max_fused = max(fused_map.values()) or 1.0
    semantic = [
        {"id": c["id"], "score": min(fused_map.get(c["id"], 0.0) / max_fused, 1.0), "payload": c}
        for c in cands
    ]
    # bm25s 原始分 → mem0 sigmoid 归一化
    midpoint, steepness = get_bm25_params(query, lemmatized=" ".join(sorted(q_tokens)))
    bm25_scores = {
        c["id"]: normalize_bm25(c["kw_score"], midpoint, steepness)
        for c in cands if c["kw_score"] > 0.0
    }
    entity_boosts = _compute_entity_boosts(q_tokens, cands)
    threshold = float(os.getenv("USE_DUAL_RECALL_THRESHOLD", "0") or 0)
    ranked = score_and_rank(semantic, bm25_scores, entity_boosts, threshold, len(semantic), explain=True)
    if not ranked:
        return []

    # MMR 去冗余（复用 graphiti 贪心选型）；向量来自索引，不再全量读取记忆
    detail = {r["id"]: r for r in ranked}
    order = sorted(
        [r["id"] for r in ranked],
        key=lambda mid: (detail[mid]["score"], by_id[mid]["recency"]),
        reverse=True,
    )
    vectors = {mid: by_id[mid]["embedding"] for mid in order
               if len(by_id[mid]["embedding"]) == len(qvec)}
    final_ids = _mmr_diverse_top_k(qvec, vectors, order, top_k) if vectors else order[:top_k]

    out: list[dict] = []
    for mid in final_ids:
        entry = db.get(MemoryEntry, mid)  # 点查，不是 list_memories 全量读取
        if entry is None:
            continue
        sd = dict(detail[mid].get("score_details") or {})
        # G2: 端到端结果反映真实 embedding 口径（hash 或 fastembed 语义），不再硬编码。
        try:
            from memory.vector_index import embedding_kind as _embedding_kind
            _kind = _embedding_kind()
        except Exception:
            _kind = "deterministic_hash_nonsemantic"
        sd.update({
            "vec_score": by_id[mid]["vec_score"],
            "bm25_raw": by_id[mid]["kw_score"],
            "embedding_kind": _kind,
        })
        out.append({
            "entry": entry,
            "score": detail[mid]["score"],
            "recency": by_id[mid]["recency"],
            "relevance": by_id[mid]["relevance"],
            "score_details": sd,
        })
    return out
