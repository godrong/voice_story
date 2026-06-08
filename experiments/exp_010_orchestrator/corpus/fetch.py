#!/usr/bin/env python3
"""Fetch public-domain Chinese fiction for orchestrator labeling pilot.

抓取公版中文小说（鲁迅短篇），切成 100-300 字片段供 LLM 标注。
Source: Project Gutenberg (鲁迅 1936 卒，公版无争议).

Outputs:
    corpus/raw/<work>.txt        原文
    corpus/segments.jsonl        切好的片段（chunk_id / source / text）
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

from opencc import OpenCC

T2S = OpenCC("t2s")  # Traditional → Simplified / 繁→简

QUOTE_OPEN = "「『“"  # 「 『 "
QUOTE_CLOSE = "」』”"  # 」 』 "

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "raw"
RAW_DIR.mkdir(exist_ok=True)
SEG_PATH = HERE.parent / "corpus" / "segments.jsonl"

# Gutenberg IDs verified to be 鲁迅 zh works.
# Gutenberg 验证可用的鲁迅作品 ID。
SOURCES = [
    {"id": "25332", "slug": "AQ", "title": "阿Q正传"},
]

GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"

# Segment target: 100-300 chars, prefer cut at 。？！
# 目标片段：100-300 字，优先在句末切
MIN_CHARS = 100
MAX_CHARS = 300
SOFT_CAP = 0.7  # rfind boundary must be >= SOFT_CAP * MAX_CHARS


def fetch_one(spec: dict) -> Path:
    """Download a single Gutenberg work. Skip if cached.

    下载单部作品，已缓存则跳过。
    """
    out = RAW_DIR / f"{spec['slug']}.txt"
    if out.exists() and out.stat().st_size > 1000:
        print(f"[cache] {spec['slug']} -> {out}")
        return out
    url = GUTENBERG_URL.format(id=spec["id"])
    print(f"[fetch] {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "voice-story-exp-010/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        text = r.read().decode("utf-8", errors="replace")
    out.write_text(text, encoding="utf-8")
    print(f"[saved] {len(text)} chars -> {out}")
    return out


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove Project Gutenberg header/footer markers.

    去掉 Gutenberg 头尾的免责声明、章节目录等。
    """
    start = re.search(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG", text)
    end = re.search(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG", text)
    if start:
        text = text[start.end():]
    if end:
        text = text[: end.start()]
    # Skip leading lines that contain no CJK chars (title/translator block).
    # 跳过开头无中文字符的元信息行。
    lines = text.splitlines()
    first_cjk = 0
    for idx, line in enumerate(lines):
        if re.search(r"[一-鿿]", line):
            first_cjk = idx
            break
    text = "\n".join(lines[first_cjk:])
    return text.strip()


def normalize_text(text: str) -> str:
    """Drop footnote markers, normalize half-width punctuation, strip ASCII residue,
    convert Traditional → Simplified Chinese.

    去脚注标记 〔n〕；半角古籍标点 ﹐﹕﹔﹗ 归一为 ，：；！；去掉 ASCII 残渣行；繁→简。
    """
    text = re.sub(r"〔[^〕]*〕", "", text)
    text = re.sub(r"\([0-9]+\)", "", text)
    # Half-width CJK punct → standard. 半角古籍标点归一。
    trans = str.maketrans("﹐﹕﹔﹗﹖﹒", "，：；！？。")
    text = text.translate(trans)
    # Drop lines that are mostly ASCII (Gutenberg meta lines).
    # 去掉 ASCII 占比高的行（Gutenberg 元信息行）。
    kept = []
    for line in text.splitlines():
        if not line.strip():
            kept.append("")
            continue
        cjk = sum(1 for c in line if "一" <= c <= "鿿")
        if cjk / max(len(line), 1) >= 0.5:
            kept.append(line)
    text = "\n".join(kept)
    # 繁→简，Qwen 简体训练数据更丰富。
    return T2S.convert(text)


def quote_depths(text: str) -> list[int]:
    """Return array where ``d[i]`` is quote depth after reading text[:i].

    返回 d[i] = 读完 text[:i] 之后的引号嵌套深度。用来判定"切在 i 是否在引号内部"。
    """
    d = 0
    depths = [0] * (len(text) + 1)
    for i, c in enumerate(text):
        if c in QUOTE_OPEN:
            d += 1
        elif c in QUOTE_CLOSE:
            d = max(0, d - 1)
        depths[i + 1] = d
    return depths


def segment_text(text: str) -> list[str]:
    """Cut into 100-300 char paragraphs at 。？！ boundaries that lie OUTSIDE quotes.

    在引号外的句末标点处切。窗口内若无合法切点，扩到 MAX_CHARS+50 再找；
    仍找不到则硬切（罕见）。
    """
    text = normalize_text(text)
    text = re.sub(r"\s+", "", text)  # 中文段，去全部空白
    depths = quote_depths(text)
    chunks: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if n - i <= MAX_CHARS:
            tail = text[i:]
            if len(tail) >= MIN_CHARS:
                chunks.append(tail)
            break
        # Try [MIN, MAX] then [MIN, MAX+50] for an out-of-quote sentence end.
        # 先在 [MIN, MAX] 找，找不到再扩 50 字。
        cut = -1
        for max_extend in (0, 50):
            hi = min(n, i + MAX_CHARS + max_extend)
            for off in range(hi - 1 - i, MIN_CHARS - 1, -1):
                ch = text[i + off]
                if ch in "。？！" and depths[i + off + 1] == 0:
                    cut = i + off + 1
                    break
            if cut >= 0:
                break
        if cut < 0:
            cut = i + MAX_CHARS  # hard cut / 硬切
        chunks.append(text[i:cut])
        i = cut
    return chunks


def main() -> int:
    """Fetch + segment all configured sources, write segments.jsonl.

    抓取并切片所有配置的作品，落到 segments.jsonl。
    """
    rows: list[dict] = []
    for spec in SOURCES:
        path = fetch_one(spec)
        raw = path.read_text(encoding="utf-8")
        body = strip_gutenberg_boilerplate(raw)
        segs = segment_text(body)
        for idx, seg in enumerate(segs):
            rows.append(
                {
                    "chunk_id": f"luxun_{spec['slug']}_{idx:04d}",
                    "source": spec["title"],
                    "text": seg,
                    "n_chars": len(seg),
                }
            )
        print(f"[seg] {spec['slug']}: {len(segs)} segments")

    with SEG_PATH.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[done] {len(rows)} segments -> {SEG_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
