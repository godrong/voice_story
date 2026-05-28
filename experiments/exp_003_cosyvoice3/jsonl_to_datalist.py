#!/usr/bin/env python3
"""Convert tier1/tier2 JSONL (or vanilla manifest.jsonl) to lora_train data.list.

Why this exists: ``lora_train.py``'s ``text_opener`` expects a tab-separated
``utt_id<TAB>wav_path<TAB>text`` file (CosyVoice's "data.list" convention),
but our pipeline writes JSONL. Tier 2 was shelved partly because nobody
wired this converter up. Single-purpose script; no project imports.

为什么需要这个：``lora_train.py`` 的 ``text_opener`` 吃的是
``utt_id<TAB>wav_path<TAB>text`` 三列 tab 分隔（CosyVoice 的 data.list 约定），
而我们整条管线写的是 JSONL。Tier 2 一直 shelved 一部分原因就是这条
转换工具没接上。本脚本单一用途，不依赖项目里其它模块。

Supports three input shapes (auto-detected by row keys):

支持三种输入形态（按行的 key 自动识别）：

1. **Pipeline manifest** (``datasets/<name>/manifest.jsonl`` from M1):
   columns ``chunk_id`` / ``audio_path`` / ``text``.
   主管线 manifest：``chunk_id`` / ``audio_path`` / ``text`` 三列。

2. **Tier 1/2 paired JSONL** (``datasets/two_tier/tier{1,2}_train.jsonl``):
   columns ``target_chunk_id`` / ``target_audio`` / ``text``. The "ref" side
   is dropped — for single-speaker LoRA we just train on the target.
   Tier 1/2 配对 JSONL：用 ``target_*`` 这一侧；"ref" 那一侧丢掉，
   单人 LoRA 只需要 target 的 (text, audio) 对。

3. **Arbitrary JSONL** with the three columns under any of the aliases
   below — controllable via flags.

Usage::

    python jsonl_to_datalist.py \\
        --input  datasets/two_tier/tier2_train.jsonl \\
        --output /root/autodl-tmp/datasets/tier2_train.data.list \\
        [--audio-root /root/autodl-tmp]   # rewrite relative paths
        [--esd-root  /root/autodl-tmp/ESD]  # expand $ESD_ROOT placeholders
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Field aliases tried in order — first hit wins.
# 字段别名优先级——按顺序取，第一个命中即用。
ID_KEYS = ("chunk_id", "target_chunk_id", "utt_id", "id")
WAV_KEYS = ("audio_path", "target_audio", "wav", "wav_path")
TEXT_KEYS = ("text", "transcript", "tts_text")


def _first(row: dict, keys: tuple[str, ...]) -> str | None:
    """Return the first non-empty value among the given key aliases.

    在给定的 key 别名中按顺序取第一个非空值。
    """
    for k in keys:
        v = row.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return None


def _rewrite_path(p: str, audio_root: Path | None, esd_root: Path | None) -> str:
    """Apply known prefix substitutions so paths point at real files on disk.

    把路径里常见的占位符 / 相对前缀替换成 GPU 机器上的真实路径：
    - ``$ESD_ROOT`` / ``${ESD_ROOT}`` → ``--esd-root``
    - 任何非绝对路径 → 拼到 ``--audio-root`` 下
    """
    if "${ESD_ROOT}" in p or "$ESD_ROOT" in p:
        if esd_root is None:
            raise ValueError(
                f"Row references {p!r} but --esd-root was not provided."
            )
        p = p.replace("${ESD_ROOT}", str(esd_root)).replace("$ESD_ROOT", str(esd_root))

    pp = Path(p)
    if not pp.is_absolute() and audio_root is not None:
        pp = audio_root / pp
    return str(pp)


def convert(input_path: Path, output_path: Path, *,
            audio_root: Path | None, esd_root: Path | None,
            require_exists: bool) -> tuple[int, int]:
    """Stream-convert JSONL → data.list. Returns (written, skipped).

    流式转换 JSONL → data.list。返回 (写入数, 跳过数)。

    Each emitted line is exactly:  ``<utt_id>\\t<wav_path>\\t<text>\\n``.
    Rows missing any of the three fields are skipped (logged to stderr).
    Rows whose audio file doesn't exist are also skipped when
    ``require_exists=True``.
    每行恰好是 ``utt_id<TAB>wav<TAB>text<LF>``。
    三个字段任一缺失的行被跳过（stderr 记录）；
    ``require_exists=True`` 时音频文件不存在的行也跳过。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    with input_path.open(encoding="utf-8") as src, \
         output_path.open("w", encoding="utf-8") as dst:
        for lineno, raw in enumerate(src, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"  skip line {lineno}: bad JSON ({e})", file=sys.stderr)
                skipped += 1
                continue

            utt = _first(row, ID_KEYS)
            wav = _first(row, WAV_KEYS)
            text = _first(row, TEXT_KEYS)
            if not (utt and wav and text):
                print(
                    f"  skip line {lineno}: missing id/wav/text "
                    f"(got id={bool(utt)} wav={bool(wav)} text={bool(text)})",
                    file=sys.stderr,
                )
                skipped += 1
                continue

            wav = _rewrite_path(wav, audio_root, esd_root)
            if require_exists and not os.path.exists(wav):
                print(f"  skip line {lineno}: wav missing on disk: {wav}",
                      file=sys.stderr)
                skipped += 1
                continue

            # Tab in text would break the 3-column contract — collapse to space.
            # 文本里的 tab 会破坏 3 列契约，统一转空格。
            text = text.replace("\t", " ").replace("\r", " ").replace("\n", " ")
            dst.write(f"{utt}\t{wav}\t{text}\n")
            written += 1

    return written, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--input", type=Path, required=True,
                    help="Input JSONL (manifest or tier1/tier2 paired).")
    ap.add_argument("--output", type=Path, required=True,
                    help="Output data.list (tab-separated, 3 columns).")
    ap.add_argument("--audio-root", type=Path, default=None,
                    help="Prefix for relative wav paths (e.g. project root on GPU box).")
    ap.add_argument("--esd-root", type=Path, default=None,
                    help="Expand ${ESD_ROOT} placeholder (Tier 1 ESD rows only).")
    ap.add_argument("--require-exists", action="store_true",
                    help="Skip rows whose wav file is missing on disk.")
    args = ap.parse_args()

    if not args.input.exists():
        print(f"input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    written, skipped = convert(
        args.input, args.output,
        audio_root=args.audio_root,
        esd_root=args.esd_root,
        require_exists=args.require_exists,
    )
    print(f"wrote {written} rows -> {args.output}  ({skipped} skipped)")
    if written == 0:
        print("WARNING: 0 rows written — check input schema and flags.",
              file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
