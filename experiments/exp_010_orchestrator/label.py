#!/usr/bin/env python3
"""Label corpus segments with Qwen2.5-1.5B-Instruct (MLX 4bit).

用 MLX 后端跑 Qwen2.5-1.5B-Instruct-4bit，对 corpus/segments.jsonl 里的
片段做最小 schema 标注（role / emotion / pause_after）。

Usage:
    conda run -n ai_study python label.py --n 50
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

HERE = Path(__file__).resolve().parent
SEG_PATH = HERE / "corpus" / "segments.jsonl"
PROMPT_PATH = HERE / "prompts" / "min_schema_v1.txt"
LABELS_DIR = HERE / "labels"
LABELS_DIR.mkdir(exist_ok=True)

MODEL_ID = "mlx-community/Qwen2.5-3B-Instruct-4bit"

VALID_ROLE = {"narrator", "ambiguous"}  # plus character_X variants
VALID_EMOTION = {"neutral", "happy", "angry", "sad", "surprise"}
VALID_PAUSE = {"short", "medium", "long"}


def build_messages(prompt_template: str, text: str) -> list[dict]:
    """Build the chat messages list. We split the template at {TEXT}
    so the body becomes the user turn.

    构造 chat 消息：把模板按 {TEXT} 拆开，正文当 user turn。
    """
    body = prompt_template.replace("{TEXT}", text)
    return [
        {"role": "system", "content": "你是中文文学有声化标注助手。严格按要求输出 JSON。"},
        {"role": "user", "content": body},
    ]


def parse_label(raw: str) -> dict[str, Any] | None:
    """Try to extract a JSON object from the model output. Strict schema check.

    从模型输出中抽 JSON 对象，失败返回 None。
    """
    # Greedy first-brace .. last-brace. 取首花到末花，简单粗暴。
    m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj


def validate(label: dict[str, Any]) -> tuple[bool, list[str]]:
    """Check the label dict against the minimum schema.

    最小 schema 校验，返回 (ok, [问题列表])。
    """
    issues: list[str] = []
    role = label.get("role")
    if not isinstance(role, str) or (
        role not in VALID_ROLE and not role.startswith("character_")
    ):
        issues.append(f"role:{role!r}")
    if label.get("emotion") not in VALID_EMOTION:
        issues.append(f"emotion:{label.get('emotion')!r}")
    if label.get("pause_after") not in VALID_PAUSE:
        issues.append(f"pause_after:{label.get('pause_after')!r}")
    return (len(issues) == 0, issues)


def load_segments(limit: int, start: int = 0) -> list[dict]:
    """Read ``limit`` rows from segments.jsonl starting at index ``start``.

    从 segments.jsonl 第 start 条开始读 limit 条。
    """
    rows: list[dict] = []
    with SEG_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows[start: start + limit]


def main() -> int:
    """Driver: load model, loop segments, write labels jsonl.

    主流程：加载模型 → 逐段标注 → 写 jsonl。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="number of segments to label")
    ap.add_argument("--start", type=int, default=0, help="start index in segments.jsonl")
    ap.add_argument(
        "--out",
        default="labeled_v1.jsonl",
        help="output filename under labels/",
    )
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--temp", type=float, default=0.1)
    args = ap.parse_args()

    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    segments = load_segments(args.n, args.start)
    print(f"[load] {len(segments)} segments (start={args.start}) to label")

    print(f"[load] model {MODEL_ID}")
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    print(f"[load] model ready in {time.time() - t0:.1f}s")

    out_path = LABELS_DIR / args.out
    n_ok = 0
    n_bad = 0
    sampler = make_sampler(temp=args.temp)
    with out_path.open("w", encoding="utf-8") as fh:
        for i, seg in enumerate(segments):
            messages = build_messages(prompt_template, seg["text"])
            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            t_seg = time.time()
            raw = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=args.max_tokens,
                sampler=sampler,
                verbose=False,
            )
            dt = time.time() - t_seg

            label = parse_label(raw)
            if label is None:
                ok, issues = False, ["parse_failed"]
                label = {}
            else:
                ok, issues = validate(label)
            n_ok += int(ok)
            n_bad += int(not ok)

            row = {
                "chunk_id": seg["chunk_id"],
                "text": seg["text"],
                "n_chars": seg["n_chars"],
                "label": label,
                "raw": raw,
                "valid": ok,
                "issues": issues,
                "latency_s": round(dt, 2),
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()

            tag = "OK" if ok else "BAD"
            preview = seg["text"][:30].replace("\n", " ")
            print(
                f"[{i+1:>3}/{len(segments)}] {tag} {dt:.1f}s  {preview}…  → "
                f"{label.get('role')}/{label.get('emotion')}/{label.get('pause_after')}"
            )

    total = n_ok + n_bad
    print(f"\n[done] {n_ok}/{total} valid ({100*n_ok/total:.1f}%) → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
