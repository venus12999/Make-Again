"""结构化输出:从模型文本里稳健提取 JSON。

底层用成熟库 json_repair 修复 LLM 常见的畸形 JSON(markdown 围栏、尾逗号、
前后杂散文字、缺括号引号、<think> 推理标签等);外层保留本项目的返回契约
``(parsed, raw)`` 与形态校验,并通过 USE_JSON_REPAIR 开关可秒级回滚到原来的
手写括号配平实现。
"""
from __future__ import annotations

import json
import os
import re
from typing import Any


def _use_json_repair() -> bool:
    return os.getenv("USE_JSON_REPAIR", "1").strip().lower() in ("1", "true", "yes", "on")


def parse_json(text: str) -> tuple[Any | None, str]:
    """返回 (解析出的对象或 None, 原始文本)。

    默认走 json_repair(成熟库);结果不是 dict/list 时回退到手写解析器,
    保证既有调用方拿到的形态与以前一致。
    """
    if not text:
        return None, text
    if _use_json_repair():
        try:
            from json_repair import repair_json
            repaired = repair_json(text, return_objects=True, ensure_ascii=False)
            # 形态校验:现有调用方全部要求 dict(结构化对象),只接受 dict;
            # 多 JSON 块(json_repair 返回 list)与数组根一律回退旧实现,严格保持契约。
            if isinstance(repaired, dict):
                return repaired, text
        except Exception:
            pass
    return _parse_json_legacy(text)


def _parse_json_legacy(text: str) -> tuple[Any | None, str]:
    """原始手写实现(开关关闭或 json_repair 结果形态不符时的回退路径)。"""
    cleaned = re.sub(r"```(?:json)?\s*", "", text).replace("```", "")
    for candidate in _balanced_json_blocks(cleaned):
        for cand in (candidate, _drop_trailing_commas(candidate)):
            try:
                return json.loads(cand), text
            except json.JSONDecodeError:
                continue
    return None, text


def _balanced_json_blocks(text: str) -> list[str]:
    """返回文本中所有「从 { 开始、括号配平」的片段,长的在前(顶层对象最长)。"""
    blocks: list[str] = []
    n = len(text)
    i = 0
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        in_str = False
        esc = False
        for j in range(i, n):
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    blocks.append(text[i:j + 1])
                    break
        i += 1
    blocks.sort(key=len, reverse=True)
    return blocks


def _drop_trailing_commas(text: str) -> str:
    """去掉对象/数组末尾的尾逗号(如 {...,} 或 [...,])。"""
    return re.sub(r",(\s*[}\]])", r"\1", text)
