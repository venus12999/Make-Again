#!/usr/bin/env python3
"""E1 表达段 · 中文原生评测批重造（先读码/看实物，再设计）。

为什么重造（读码与误差分析结论，非拍脑袋）
------------------------------------------------
旧 `eval/batches/01_中文共情_CPsyCoun.json`（n=34）场景比为
分手失恋 1 / 亲人离世 9 / 情绪困扰 12 / 人际关系 12 / 宠物离世 0，
与产品「三类离别（分手/亲人离世/宠物离世）+ 五段闭环（表达→理解→沉淀→调用→行动）」
严重错配；D-041 误差分析逐条看 25 个失败 case 原文后裁定全部是「评测集错配」，
0 条真产品失败。主力 3 批又是英文翻译集（N-007），不能当中文陪伴产品的主力评测。
本脚本按 E1 铁律重造表达段中文原生批：**中文原生、不翻译、逐条可追溯**。

调研依据（URL + 发布日期 + 距今权重；今天 2026-11-23，按项目 AGENTS.md 按天精确算）
--------------------------------------------------------------------------------
权重曲线：≤6 月=1.0 / 6-18 月=0.7 / 18-36 月=0.4 / >36 月=0.1。
注意：「任务给定权重」是语料在本批中的重要度（用户任务书写 CPsyCoun 0.7 /
PsyQA 0.4），「时间衰减权重」是证据新鲜度，二者概念不同，均如实记录，
不把旧论文伪装成新鲜证据。月数=(今天-发布日期).days/30.44。

- CPsyCoun（主源，权威中文咨询多轮对话）
  arXiv https://arxiv.org/abs/2405.16433 （v1 提交 2024-05-26，911 天/29.9 月→0.4）
  ACL   https://aclanthology.org/2024.findings-acl.830/ （Findings ACL 2024，2024-08）
  HF    https://huggingface.co/datasets/CAS-SIAT-XinHai/CPsyCoun
  任务权重 0.7；时间衰减权重 0.4；旧 34 条即从此集 HF datasets-server rows API 实拉。
- PsyQA（失恋场景补充源，权威中文长文咨询问答，22K 问题/56K 回答）
  arXiv https://arxiv.org/abs/2106.01702 （v1 提交 2021-06-03，1999 天/65.7 月→0.1）
  ACL   https://aclanthology.org/2021.findings-acl.130/ （Findings ACL-IJCNLP 2021）
  HF    https://huggingface.co/datasets/lsy641/PsyQA （test.json 实拉 2026-11-23）
  任务权重 0.4；时间衰减权重 0.1（遗迹档，只作中文原生恋爱/失恋语料来源，
  其 meta 标注「恋爱,失恋」22+ 条；不作选型主依据）。
- deep-emotional-support-zh（2025-26 新中文集，交叉参考，本批未取条）
  https://huggingface.co/datasets/AngelWarmSmile123/deep-emotional-support-zh
  公告 2026-07-07，139 天/4.6 月→1.0；6253 条人工撰写中文情感支持对话；
  CC-BY-NC-SA-4.0（评测可用，商用需注意）。
- AuraDial（2025 新中文集，交叉参考，本批未取条）
  https://aclanthology.org/2025.findings-emnlp.155/ （Findings EMNLP 2025，
  2025-11，387 天/12.7 月→0.7）；30 万单轮 + 9 万多轮，真实用户提问改写。
- MATCHA（2026 中文多障碍集，交叉参考，本批未取条）
  https://arxiv.org/abs/2610.35512 （2026-10-25，29 天/1.0 月→1.0）；
  life-event taxonomy 含 Breakup/divorce，但为社媒标注非陪伴对话。
- KardiaBench（2025 情感对话集，交叉参考，本批未取条）
  https://arxiv.org/abs/2512.01282 / https://huggingface.co/datasets/Jhcircle/KadiaBench
  （2025-12-03，355 天/11.7 月→0.7）；22080 条多轮情感对话。

宠物离世：权威中文集 NOT_FOUND
--------------------------------------------------------------------------------
调研：CPsyCoun/PsyQA 均无「宠物离世」类目；英文侧存在 pet-loss 语料
（TEA-Dialog "Dealing with the loss of a pet" https://huggingface.co/datasets/XingYuSSS/TEA-Dialog ，
augesc "puppy dead" https://huggingface.co/datasets/heegyu/augesc ，
AttuneBench family/pets https://public.attunebench.com/ ），但**语言错配**，
翻译会重蹈 N-007「英文翻译集→假失败」覆辙。按铁律②：宠物离世 6 条自造，
逐条明确标 `provenance.kind="synthetic-handwritten"` / `synthetic=true` /
`source_id` 前缀 `synthetic-handwritten:pet-loss:`，绝不冒充权威集。

不变量（防止重建退化）
------------------------------------------------
1. 旧 34 条逐字保留（user_input/scene/provenance 不动）——与 H1 baseline 的
   per-case 配对仍成立，也满足 test_heart_rubric 的动态 git 基线守卫。
2. HEART 5 维 rubric / eval_control / power_analysis 元信息逐字保留，
   power_analysis.actual_n 同步到新 n（n≥30 统计功效不退化）。
3. ci_runner.EXPECTED_SCENES 要求的四个旧场景名全部保留，只**新增**宠物离世。

用法
------------------------------------------------
    python eval/build_e1_batch.py                      # 重造 01 批
    PSYQA_TEST_CACHE=<path> python eval/build_e1_batch.py   # 复用已下载 test.json
"""
from __future__ import annotations

import json
import os
import subprocess
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
BATCH_DIR = HERE / "batches"
BATCH_NAME = "01_中文共情_CPsyCoun"
OUT = BATCH_DIR / f"{BATCH_NAME}.json"
REPO = HERE.parent.parent
BASELINE_COMMIT = "aad3e951"
REL_BATCH = f"MakeAgain-backend/eval/batches/{BATCH_NAME}.json"

PSYQA_TEST_URL = "https://huggingface.co/datasets/lsy641/PsyQA/resolve/main/test.json"
PSYQA_FETCH_DATE = "2026-11-23"
# 从 PsyQA test.json 实拉的失恋场景 questionID（meta 关键词含「恋爱,失恋」）。
# 选择口径：真实失恋自述、描述≥60 字、排除自伤/危机条（危机由 04 批覆盖）。
PSYQA_BREAKUP_IDS = [9165, 16319, 8004, 9965, 2217, 20689, 1485, 18265]

RESEARCH_BASIS = {
    "as_of": "2026-11-23",
    "decay_curve": "AGENTS.md: <=6m=1.0 / 6-18m=0.7 / 18-36m=0.4 / >36m=0.1; months=(today-published).days/30.44",
    "cpsycoun": {
        "task_weight": 0.7, "decay_weight": 0.4,
        "published": "2024-05-26 (arXiv v1); 2024-08 (Findings ACL 2024)",
        "age_days": 911, "age_months": 29.9,
        "urls": [
            "https://arxiv.org/abs/2405.16433",
            "https://aclanthology.org/2024.findings-acl.830/",
            "https://huggingface.co/datasets/CAS-SIAT-XinHai/CPsyCoun",
        ],
    },
    "psyqa": {
        "task_weight": 0.4, "decay_weight": 0.1,
        "published": "2021-06-03 (arXiv v1); 2021-08 (Findings ACL-IJCNLP 2021)",
        "age_days": 1999, "age_months": 65.7,
        "urls": [
            "https://arxiv.org/abs/2106.01702",
            "https://aclanthology.org/2021.findings-acl.130/",
            "https://huggingface.co/datasets/lsy641/PsyQA",
        ],
    },
    "deep_emotional_support_zh": {
        "decay_weight": 1.0, "published": "2026-07-07 (HF announcement)",
        "age_days": 139, "age_months": 4.6, "used_in_batch": False,
        "url": "https://huggingface.co/datasets/AngelWarmSmile123/deep-emotional-support-zh",
    },
    "auradial": {
        "decay_weight": 0.7, "published": "2025-11 (Findings EMNLP 2025)",
        "age_days": 387, "age_months": 12.7, "used_in_batch": False,
        "url": "https://aclanthology.org/2025.findings-emnlp.155/",
    },
    "matcha": {
        "decay_weight": 1.0, "published": "2026-10-25 (arXiv)",
        "age_days": 29, "age_months": 1.0, "used_in_batch": False,
        "url": "https://arxiv.org/abs/2610.35512",
    },
    "kardiabench": {
        "decay_weight": 0.7, "published": "2025-12-03 (arXiv 2512.01282)",
        "age_days": 355, "age_months": 11.7, "used_in_batch": False,
        "url": "https://huggingface.co/datasets/Jhcircle/KadiaBench",
    },
}

PET_LOSS_NOT_FOUND = {
    "status": "NOT_FOUND",
    "scope": "中文原生宠物离世情感陪伴/心理咨询权威集",
    "note": "CPsyCoun/PsyQA 无该类目；英文 pet-loss 集存在但语言错配，"
            "按 N-007 教训不翻译充当主力。故 6 条自造并明确标 synthetic-handwritten。",
    "english_only_evidence": [
        "https://huggingface.co/datasets/XingYuSSS/TEA-Dialog",
        "https://huggingface.co/datasets/heegyu/augesc",
        "https://public.attunebench.com/",
    ],
}

# 宠物离世 6 条：中文原生手写，非翻译、非模型批量生成；每条带自造口径。
# reference 按产品本质写：陪伴带着关系继续生活，不空话、不催促、不比较。
PET_SYNTHETIC_ITEMS = [
    {
        "_id": "pet-syn-001",
        "user_input": "我家那只橘猫昨天凌晨走了，十七岁。早上起来它常趴的那块地板还是温的，我下意识伸手去摸，什么都没有。我在卫生间哭了很久，转头又觉得自己没资格这么难过——它只是一只猫而已。可我知道，我失去的是一个每天等我回家的人。",
        "reference": "十七年的清晨和傍晚，它把那块地板睡成了自己的位置，也把你的回家当成一天里最重要的事。你说「只是一只猫」，但这份难过不需要谁来批准。今天要是愿意，就跟我聊聊它吧——它最爱趴的姿势、它叫什么、它最后一次蹭你是什么时候。",
    },
    {
        "_id": "pet-syn-002",
        "user_input": "豆豆上周三走了，胰腺炎，抢救了一夜。我到现在没敢收拾它的窝，牵引绳还挂在门后。晚上遛弯我还是会绕去它最喜欢的那个花坛，走到一半才想起来它不在了。家里人说「再养一只就好了」，可我想念的是豆豆，不是一条狗。",
        "reference": "牵引绳还挂在门后，这个细节比任何话都更像它还在。绕到花坛那半路突然停住的那一刻，我听着都心疼。「再养一只就好了」——他们大概不知道，你想念的是那个只属于你的豆豆。它的窝什么时候收、要不要收，都你自己说了算，没人催你。",
    },
    {
        "_id": "pet-syn-003",
        "user_input": "我养了九年的边牧在小区门口被车撞了，就几秒钟，我连一句再见都没来得及说。现在一闭眼就是它最后看我的那个眼神。我反复想是不是那天我牵绳松了，如果我不松手它是不是还在。这种念头一天要转好几遍，停不下来。",
        "reference": "几秒钟的事，却要拿好多个夜晚去消化。你反复回放那一刻，不是真的在怪自己，是因为太想再见它一面、太想给它一个交代。那个眼神里有它对你九年的信任，这一点没有被那几秒钟夺走。你说的那句「没来得及说再见」，它其实听了一辈子。",
    },
    {
        "_id": "pet-syn-004",
        "user_input": "花花是十四岁的泰迪，走之前那几天已经站不起来了，我每天把它抱在怀里喂水。它走的那天我在宠物医院签了字，手一直在抖。之后我总想起它临走前舔我手的那一下，不知道那是告别还是只是习惯。我不敢跟朋友讲，怕他们说我矫情。",
        "reference": "十四岁，从蹦蹦跳跳到站不起来，你每天都抱着它喂水——那不是矫情，那是你把能给它的都给了。它最后舔你手那一下，是不是「告别」没人能替它定义，但它选择把最后一点力气留给你，这件事是真的。你愿意讲给谁听、什么时候讲，都由你定。",
    },
    {
        "_id": "pet-syn-005",
        "user_input": "我一个人住，猫是我每天回家唯一会迎接我的。它走了一个月，家里安静得让我害怕。我现在下班不愿意回去，坐在车里能待一个小时。朋友劝我赶紧再养一只，可我一想到要重新认识一只猫、重新建立那种默契，就觉得累。",
        "reference": "下班坐在车里那一个小时，我懂——不是不想回家，是怕推开家门没有那一声迎接。它是你在这个家里最稳定的那个「有人等你」，这份空缺不是再养一只就能立刻填上的。累了就先歇着，不用给自己定「什么时候该好起来」的期限。",
    },
    {
        "_id": "pet-syn-006",
        "user_input": "整理衣柜翻出一件旧卫衣，上面还有小雪的毛。它是我考研那年陪我的猫，天天趴在我腿上，我哭它也蹭我。它走了两年了，我以为自己早就好了，结果摸到这件衣服还是一下子绷不住。我是不是太没出息了？两年了还这样。",
        "reference": "两年了，衣服上还留着它的毛，像是它替你把那段最难的日子一起存了下来。绷不住不是没出息，是那段一起熬过来的日子太真了，真到身体比脑子记得还牢。你不用「早就好了」才配得上它——它陪你走过的那一年，本来就值得你偶尔再哭一次。",
    },
]


def decay_weight(published_days_ago: int) -> float:
    """按 AGENTS.md 衰弱曲线给时间权重（仅供自校验，不参与数据落地）。"""
    months = published_days_ago / 30.44
    if months <= 6:
        return 1.0
    if months <= 18:
        return 0.7
    if months <= 36:
        return 0.4
    return 0.1


def load_legacy_batch() -> dict:
    """从基线 commit 取旧批（不依赖工作区，重复运行结果一致）。"""
    out = subprocess.run(
        ["git", "show", f"{BASELINE_COMMIT}:{REL_BATCH}"],
        cwd=str(REPO), capture_output=True, text=True, encoding="utf-8",
    )
    if out.returncode != 0:
        raise RuntimeError(f"git show {BASELINE_COMMIT} 失败: {out.stderr[:200]}")
    return json.loads(out.stdout)


def fetch_psyqa() -> list[dict]:
    """实拉 PsyQA test.json（可用 PSYQA_TEST_CACHE 复用同一份真实快照）。"""
    cache = os.environ.get("PSYQA_TEST_CACHE")
    if cache and Path(cache).exists():
        rows = json.loads(Path(cache).read_text(encoding="utf-8"))
    else:
        with urllib.request.urlopen(PSYQA_TEST_URL, timeout=180) as resp:
            rows = json.loads(resp.read())
    return rows


def build_psyqa_items(rows: list[dict]) -> list[dict]:
    by_id = {r.get("questionID"): r for r in rows}
    items = []
    for qid in PSYQA_BREAKUP_IDS:
        row = by_id.get(qid)
        if row is None:
            raise RuntimeError(f"PsyQA questionID={qid} 不在实拉快照中")
        answers = row.get("answers") or []
        answer = (answers[0].get("answer_text") or "") if answers else ""
        question = (row.get("question") or "").strip()
        description = (row.get("description") or "").strip()
        user_input = (question + "\n" + description)[:600]
        if len(user_input) < 60 or not answer:
            raise RuntimeError(f"PsyQA questionID={qid} 内容不完整")
        sid_local = f"psyqa-{qid}"
        items.append({
            "id": sid_local,
            "source_id": f"PsyQA:test:{qid}",
            "scene": "分手失恋",
            "language": "zh",
            "user_input": user_input,
            "reference": answer[:400],
            "source": {
                "dataset": "PsyQA",
                "paper": "arXiv 2106.01702 Findings ACL-IJCNLP 2021",
                "id": sid_local,
                "question_id": qid,
                "keywords": row.get("keywords", ""),
            },
            "provenance": {
                "kind": "authoritative",
                "synthetic": False,
                "dataset": "PsyQA",
                "split": "test",
                "row_index": qid,
                "fetch": f"HF lsy641/PsyQA test.json ({PSYQA_FETCH_DATE} 实拉)",
                "url": PSYQA_TEST_URL,
            },
        })
    return items


def build_pet_items() -> list[dict]:
    items = []
    for n, raw in enumerate(PET_SYNTHETIC_ITEMS, 1):
        sid_local = raw["_id"]
        items.append({
            "id": sid_local,
            "source_id": f"synthetic-handwritten:pet-loss:{n:03d}",
            "scene": "宠物离世",
            "language": "zh",
            "user_input": raw["user_input"],
            "reference": raw["reference"],
            "source": {
                "dataset": "synthetic-handwritten",
                "paper": "自造（中文权威集无宠物离世类目，见 e1_rebuild.pet_loss_not_found）",
                "id": sid_local,
            },
            "provenance": {
                "kind": "synthetic-handwritten",
                "synthetic": True,
                "dataset": "synthetic-handwritten",
                "scene": "宠物离世",
                "author": "E1 eval rebuild agent (自造起草，非真人标注、非权威集)",
                "created": "2026-11-23",
                "fetch": "synthetic-handwritten",
                "note": "自造口径：中文原生起草，非权威集、非翻译、非真人金标；仅用于补产品三场景缺口，不可当真人语料统计",
            },
        })
    return items


def build() -> dict:
    for src, cfg in RESEARCH_BASIS.items():
        if "decay_weight" in cfg and "age_days" in cfg:
            assert cfg["decay_weight"] == decay_weight(cfg["age_days"]), src

    legacy = load_legacy_batch()
    old_items = legacy["items"]
    psyqa_items = build_psyqa_items(fetch_psyqa())
    pet_items = build_pet_items()
    items = old_items + psyqa_items + pet_items

    counts = Counter(it["scene"] for it in items)
    for scene in ("分手失恋", "亲人离世", "宠物离世"):
        assert counts[scene] >= 6, f"E1-V1 {scene} 只有 {counts[scene]} 条"
    for scene in ("情绪困扰", "人际关系"):
        assert counts[scene] >= 1, f"ci_runner taxonomy 缺 {scene}"

    # 逐字保留 rubric / eval_control / eval / power_analysis 的字段名与值，只同步 n。
    meta = {k: v for k, v in legacy.items() if k not in ("batch", "items", "n", "power_analysis")}
    meta["n"] = len(items)
    power = dict(legacy["power_analysis"])
    power["actual_n"] = len(items)
    power["note"] = power.get("note", "") + " E1 重造后 n 增至 %d（含 %d 条 PsyQA 权威 + %d 条自造宠物），统计功效不退化。" % (
        len(items), len(psyqa_items), len(pet_items))
    meta["power_analysis"] = power
    meta["e1_rebuild"] = {
        "date": "2026-11-23",
        "goal": "表达段中文原生重造：去英文翻译集、按产品三场景配比、逐条可追溯",
        "method": "旧34条逐字保留 + PsyQA 中文原生失恋 8 条 + 自造宠物离世 6 条（不翻译）",
        "research_basis": RESEARCH_BASIS,
        "pet_loss_not_found": PET_LOSS_NOT_FOUND,
        "scene_distribution": dict(counts),
        "invariants": [
            "旧 34 条 user_input/scene/provenance 逐字保留（H1 baseline 配对可比）",
            "HEART 5 维 rubric / eval_control / power_analysis 保留，actual_n 同步",
            "ci_runner.EXPECTED_SCENES 的四个旧场景名全保留，只新增宠物离世",
        ],
    }
    OUT.write_text(json.dumps({"batch": BATCH_NAME, **meta, "items": items},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[E1] {OUT.name}: {len(items)} items, scenes={dict(counts)}")
    return json.loads(OUT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    build()
