#!/usr/bin/env python3
"""Quick error analysis of labeled_v1.jsonl.

对 LLM 标注结果做分布统计 + 规则对照：
- 标签分布（看是否塌缩）
- 与"段含引号 → character"的简单规则对比
- 与"段末标点 → pause_after"的简单规则对比
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
LABELS_PATH = HERE / "labels" / "labeled_v1.jsonl"
QUOTE_CHARS = ("「", "」", "“", "”", "『", "』")
PUNCT_PAUSE = {
    "，": "short", "、": "short", "；": "short",
    "。": "medium", "？": "medium", "！": "medium",
}


def rule_role(text: str) -> str:
    """Simple rule: if text contains a closing quote and starts after speech marker.

    最朴素的规则：含引号 → character，否则 narrator。
    """
    has_quote = any(q in text for q in QUOTE_CHARS)
    # If first char is quote → character ; if quote in middle → ambiguous ; none → narrator.
    # 引号在首 → character；夹杂 → ambiguous；无 → narrator
    if not has_quote:
        return "narrator"
    if text[0] in QUOTE_CHARS:
        return "character_X"
    return "ambiguous"


def rule_pause(text: str) -> str:
    """Pause from last punctuation.

    根据末字符标点给停顿强度。
    """
    last = text.rstrip()[-1] if text.strip() else ""
    return PUNCT_PAUSE.get(last, "short")


def main() -> int:
    """Run analysis and print report.

    跑分析并打印报告。
    """
    rows = [json.loads(l) for l in LABELS_PATH.read_text(encoding="utf-8").splitlines()]
    n = len(rows)
    print(f"[analyze] n={n} segments\n")

    # Distribution / 标签分布
    role_dist = Counter(r["label"].get("role") for r in rows)
    emo_dist = Counter(r["label"].get("emotion") for r in rows)
    pause_dist = Counter(r["label"].get("pause_after") for r in rows)
    print("── LLM label distribution ──────────────────────────────")
    print(f"role    : {dict(role_dist)}")
    print(f"emotion : {dict(emo_dist)}")
    print(f"pause   : {dict(pause_dist)}")

    # Diversity / 多样性
    print("\n── Collapse check ──────────────────────────────────────")
    for name, dist in [("role", role_dist), ("emotion", emo_dist), ("pause", pause_dist)]:
        top1, top1_n = dist.most_common(1)[0]
        print(f"{name:8s}: top-1 = {top1!r} @ {top1_n}/{n} ({100*top1_n/n:.0f}%)")

    # Rule vs LLM agreement / 规则对照
    print("\n── Rule baseline vs LLM ────────────────────────────────")
    role_agree = 0
    role_quoted_but_narrator = 0  # LLM missed dialogue / LLM 漏判对白
    pause_agree = 0
    for r in rows:
        text = r["text"]
        rule_r = rule_role(text)
        llm_r = r["label"].get("role", "")
        # Normalize character_A/B/C → character_X for compare.
        # 把 character_A/B/C 归一为 character_X 比较。
        llm_r_norm = "character_X" if llm_r.startswith("character_") else llm_r
        if rule_r == llm_r_norm:
            role_agree += 1
        if rule_r in ("character_X", "ambiguous") and llm_r_norm == "narrator":
            role_quoted_but_narrator += 1

        if rule_pause(text) == r["label"].get("pause_after"):
            pause_agree += 1

    print(f"role    rule==LLM : {role_agree}/{n} ({100*role_agree/n:.0f}%)")
    print(f"  LLM said narrator but text has quotes: "
          f"{role_quoted_but_narrator}/{n} "
          f"({100*role_quoted_but_narrator/n:.0f}%)")
    print(f"pause   rule==LLM : {pause_agree}/{n} ({100*pause_agree/n:.0f}%)")

    # Latency / 速度
    lat = [r["latency_s"] for r in rows]
    print(f"\n── Latency ─────────────────────────────────────────────")
    print(f"mean: {sum(lat)/len(lat):.1f}s  min: {min(lat):.1f}s  max: {max(lat):.1f}s")

    # Show worst cases / 看几个"看起来明显有情感但被标 neutral"的样本
    print("\n── Sample failures (segments containing 「 but LLM=narrator) ──")
    fails = [r for r in rows
             if any(q in r["text"] for q in QUOTE_CHARS)
             and r["label"].get("role") == "narrator"]
    for r in fails[:5]:
        print(f"\n[{r['chunk_id']}] LLM={r['label']}")
        print(f"text: {r['text'][:160]}")
    print(f"\n[total such failures: {len(fails)}/{n}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
