#!/usr/bin/env python3
"""Compare two labeling runs (e.g., 1.5B vs 7B) side by side.

并排对比两次标注结果，看分布与逐 chunk 差异。

Usage:
    conda run -n ai_study python compare.py labels/labeled_v1.jsonl labels/labeled_v2.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def load(path: Path) -> dict[str, dict]:
    """Load a labeled jsonl, indexed by chunk_id.

    读 jsonl，按 chunk_id 建索引。
    """
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        out[row["chunk_id"]] = row
    return out


def main(argv: list[str]) -> int:
    """Print side-by-side comparison + per-field agreement.

    打印并排对比 + 各字段一致率。
    """
    if len(argv) < 3:
        print("usage: compare.py <labeled_a.jsonl> <labeled_b.jsonl>")
        return 1
    a_path = Path(argv[1])
    b_path = Path(argv[2])
    a = load(a_path)
    b = load(b_path)
    common = sorted(set(a) & set(b))
    print(f"[compare] A={a_path.name} ({len(a)})  B={b_path.name} ({len(b)})  common={len(common)}\n")

    fields = ["role", "emotion", "pause_after"]
    for field in fields:
        a_dist = Counter(a[c]["label"].get(field) for c in common)
        b_dist = Counter(b[c]["label"].get(field) for c in common)
        agree = sum(1 for c in common if a[c]["label"].get(field) == b[c]["label"].get(field))
        print(f"── {field} ─────────────────────────────────")
        print(f"A: {dict(a_dist)}")
        print(f"B: {dict(b_dist)}")
        print(f"agree: {agree}/{len(common)} ({100*agree/len(common):.0f}%)\n")

    # Diversity / 多样性
    print("── Collapse comparison ──────────────────────────")
    for field in fields:
        a_dist = Counter(a[c]["label"].get(field) for c in common)
        b_dist = Counter(b[c]["label"].get(field) for c in common)
        a_top = a_dist.most_common(1)[0]
        b_top = b_dist.most_common(1)[0]
        print(f"{field:10s}  A top: {a_top[0]!r}@{100*a_top[1]/len(common):.0f}%  "
              f"B top: {b_top[0]!r}@{100*b_top[1]/len(common):.0f}%")

    # Latency / 速度
    print("\n── Latency ──────────────────────────────────────")
    a_lat = [a[c]["latency_s"] for c in common]
    b_lat = [b[c]["latency_s"] for c in common]
    print(f"A: mean {sum(a_lat)/len(a_lat):.1f}s   min {min(a_lat):.1f}s   max {max(a_lat):.1f}s")
    print(f"B: mean {sum(b_lat)/len(b_lat):.1f}s   min {min(b_lat):.1f}s   max {max(b_lat):.1f}s")

    # Show 5 segments where B has non-narrator/non-neutral / 看 5 个 B 跳出 collapse 的样本
    print("\n── Where B escaped collapse (role!=narrator OR emotion!=neutral) ──")
    escaped = [
        c for c in common
        if (not b[c]["label"].get("role", "").startswith("narrator")
            and b[c]["label"].get("role") != "narrator")
        or b[c]["label"].get("emotion") not in (None, "neutral")
    ]
    for c in escaped[:5]:
        print(f"\n[{c}]")
        print(f"A: {a[c]['label']}")
        print(f"B: {b[c]['label']}")
        print(f"text: {a[c]['text'][:140]}")
    print(f"\n[B escapes: {len(escaped)}/{len(common)}]")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
