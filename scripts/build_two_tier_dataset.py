#!/usr/bin/env python
"""Build the two-tier LoRA dataset (Tier 1 multi-speaker + Tier 2 avatar).

Phase 1 of the [RESEARCH_PLAN.md](../docs/RESEARCH_PLAN.md) execution. This
script ingests external datasets and emits manifest files in the M1
schema (matching ``datasets/trump_wef/manifest.jsonl``), so the downstream
LoRA training scripts can mix them freely.

Subcommands
-----------
- ``ingest-esd``     Walk an extracted ESD folder, compute quality
                     metrics, emit manifest.jsonl. Required for Tier 1.
- ``stats``          Print speaker × emotion distribution + the "≥30
                     chunks per (speaker, emotion)" verification gate.
- (future)           ``ingest-aishell3`` / ``ingest-libritts`` /
                     ``build-pairs`` / ``build-splits`` — Week 1+ work.

Why not run full M1 pipeline
----------------------------
ESD is **already** chunked + transcribed + cleaned by its original
authors. Running Demucs + VAD + ASR over it would duplicate that work
and risk introducing noise via re-segmentation. Instead, this script
acts as an "ESD adapter": preserve original chunks/text, populate only
the quality fields (SNR / DNSMOS / clipping) via ``core.eval`` so the
manifest schema matches M1.

构建双层 LoRA 训练所需的数据集（Tier 1 多说话人 + Tier 2 单说话人）。

是 [RESEARCH_PLAN.md](../docs/RESEARCH_PLAN.md) Phase 1 的执行入口。
把外部数据集吃进来、产出和 M1 一致 schema 的 manifest.jsonl，下游
LoRA 训练脚本可以直接混合使用。

之所以不在 ESD 上跑完整 M1 流水线：ESD 已经被原作者切片 + 转写 +
清洗，再跑 Demucs/VAD/ASR 是重复劳动且可能引入噪声。本脚本只做
"ESD → M1 schema 适配"：保留原 chunk 和文本，只通过 core.eval
补齐 SNR / DNSMOS / clipping 这几个质量字段。
"""

from __future__ import annotations

import json
import logging
import re
import sys
import os
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import typer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Import lazily inside commands to keep --help fast and avoid loading
# DNSMOS ONNX until needed.
# 命令内部再 import 重依赖，让 --help 快、避免没必要时加载 DNSMOS ONNX。

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, help="Build two-tier LoRA dataset (RESEARCH_PLAN Phase 1).")


# ---------------------------------------------------------------------------
# ESD constants
# ESD 的约定常量
# ---------------------------------------------------------------------------

# ESD folder layout (from HLTSingapore/Emotional-Speech-Data):
#   <root>/0001/Angry/0001_000351.wav
#                     0001_000352.wav
#                     ...
#   <root>/0001/Happy/...
#   <root>/0001/Neutral/0001/  (transcripts may live in a sibling dir)
#   <root>/0001/0001.txt       (one line per utterance: "<id>\t<text>\t<emotion>")
#
# Speaker id convention:
#   0001-0010 → Mandarin
#   0011-0020 → English
# Five emotion labels: Angry / Happy / Neutral / Sad / Surprise
#
# ESD 目录结构（HLTSingapore/Emotional-Speech-Data 的 release）：
#   0001-0010 是普通话；0011-0020 是英文
#   每个 speaker 下 5 个 emotion 文件夹，各 ~350 句
#   transcript 通常在 <speaker>/<speaker>.txt：每行 "id<TAB>text<TAB>emotion"

ESD_EMOTIONS = {"Angry", "Happy", "Neutral", "Sad", "Surprise"}
ESD_EMOTION_TO_TAG = {
    "Angry": "angry",
    "Happy": "happy",
    "Neutral": "neutral",
    "Sad": "sad",
    "Surprise": "surprise",
}
ESD_SPEAKER_RE = re.compile(r"^(\d{4})$")


def _esd_lang(speaker_id: str) -> str:
    """Return 'zh' for 0001-0010, 'en' for 0011-0020.

    根据 ESD 约定返回语种。
    """
    n = int(speaker_id)
    return "zh" if 1 <= n <= 10 else "en"


def _read_esd_transcript(speaker_dir: Path) -> dict[str, tuple[str, str]]:
    """Parse the per-speaker transcript file.

    解析每个 speaker 目录下的 transcript 文件，返回 {utterance_id: (text, emotion)}。

    The transcript file is usually ``<speaker_id>/<speaker_id>.txt`` with
    tab-separated columns: ``id<TAB>text<TAB>emotion``. Some ESD releases
    use a slightly different format; we try a couple of common variants
    and warn on any line we can't parse.

    通常是 ``<speaker_id>/<speaker_id>.txt``，制表符分隔三列：
    id、text、emotion。某些 ESD 版本格式略有差异，我们尝试几种常见变体，
    无法解析的行打 warning。
    """
    transcript = {}
    candidates = [
        speaker_dir / f"{speaker_dir.name}.txt",
        speaker_dir / f"{speaker_dir.name}.csv",
    ]
    tpath = next((c for c in candidates if c.exists()), None)
    if tpath is None:
        logger.warning("no transcript file found under %s", speaker_dir)
        return transcript

    for lineno, line in enumerate(tpath.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        # Variant 1: id<TAB>text<TAB>emotion
        # Variant 2: id<TAB>text  (emotion implicit in folder)
        parts = line.split("\t")
        if len(parts) >= 3:
            utt_id, text, emotion = parts[0].strip(), parts[1].strip(), parts[2].strip()
        elif len(parts) == 2:
            utt_id, text = parts[0].strip(), parts[1].strip()
            emotion = ""  # caller falls back to folder name
        else:
            logger.warning("could not parse line %d in %s: %r", lineno, tpath, line)
            continue
        transcript[utt_id] = (text, emotion)
    return transcript


# ---------------------------------------------------------------------------
# Manifest row
# manifest 行
# ---------------------------------------------------------------------------


@dataclass
class ManifestRow:
    """One row of the M1-compatible manifest.

    与 M1 manifest schema 兼容的一行。

    Fields are a subset of trump_wef/manifest.jsonl — the ones meaningful
    for ESD. Missing fields (e.g. pitch_*) are simply absent rather than
    null-filled, to keep the JSONL compact. Downstream training code
    should use ``.get(...)`` defensively.

    字段是 trump_wef/manifest.jsonl 的子集——只保留对 ESD 有意义的。
    缺失字段直接不写（不是 null），保持 JSONL 紧凑；下游训练代码用
    .get(...) 拿值。
    """

    manifest_version: str
    chunk_id: str
    audio_path: str
    source_file: str
    speaker_id: str
    text: str
    lang: str
    duration: float
    emotion_tag: str
    emotion_confidence: float
    snr_db: float
    mos_ovr: float
    mos_sig: float
    mos_bak: float
    clipped: bool
    source_dataset: str  # "esd" | "aishell3" | "libritts" | "trump_wef"


def _score_one_wav(wav_path: Path) -> dict:
    """Run quality metrics on a single wav, returning a dict of fields.

    跑单条 wav 的质量评分，返回字段 dict。

    Reuses ``core.eval.score_chunk`` (which loads wav + computes
    WADA-SNR + DNSMOS + clipping) plus a soundfile duration probe.
    复用 core.eval.score_chunk + soundfile 时长探测。
    """
    import soundfile as sf

    from core import eval as eval_mod

    info = sf.info(str(wav_path))
    duration = float(info.frames) / float(info.samplerate)
    snr, mos, clipped = eval_mod.score_chunk(wav_path)
    return {
        "duration": duration,
        "snr_db": snr,
        "mos_ovr": mos.ovr,
        "mos_sig": mos.sig,
        "mos_bak": mos.bak,
        "clipped": clipped,
    }


# ---------------------------------------------------------------------------
# Subcommand: ingest-esd
# 子命令：ingest-esd
# ---------------------------------------------------------------------------


@app.command("ingest-esd")
def ingest_esd(
    src: Path = typer.Option(..., help="Root of the extracted ESD folder (containing 0001/, 0002/, ...)."),
    out: Path = typer.Option(
        REPO_ROOT / "datasets" / "esd" / "manifest.jsonl",
        help="Output manifest.jsonl path.",
    ),
    speakers: list[str] = typer.Option(
        None, "--speaker", "-s",
        help="Only process specific speaker ids (e.g. -s 0011 -s 0012). Default: all.",
    ),
    emotions: list[str] = typer.Option(
        None, "--emotion", "-e",
        help="Only process specific emotions (e.g. -e Angry -e Happy). Default: all 5.",
    ),
    max_per_cell: int = typer.Option(
        -1, "--max-per-cell",
        help="Cap chunks per (speaker, emotion) cell; -1 = no cap. Useful for smoke testing.",
    ),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Walk ESD, compute quality metrics, emit M1-compatible manifest.

    遍历 ESD，跑质量指标，输出 M1 兼容 manifest。

    Output path layout (default):
        datasets/esd/manifest.jsonl

    Audio paths in the manifest are **absolute** so the file can live
    outside the repo if you keep raw ESD elsewhere (e.g. on AutoDL
    persistent disk).

    输出 manifest 里的 audio_path 是**绝对路径**——这样 ESD 原始数据
    放仓库外（比如 AutoDL 网盘）也能引用。
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    src = src.expanduser().resolve()
    out = out.expanduser().resolve()
    if not src.is_dir():
        typer.echo(f"ERROR: src does not exist or is not a directory: {src}", err=True)
        raise typer.Exit(1)

    want_speakers = set(speakers) if speakers else None
    want_emotions = set(emotions) if emotions else ESD_EMOTIONS
    if not want_emotions.issubset(ESD_EMOTIONS):
        bad = want_emotions - ESD_EMOTIONS
        typer.echo(f"ERROR: unknown emotion(s): {bad}; valid = {ESD_EMOTIONS}", err=True)
        raise typer.Exit(1)

    out.parent.mkdir(parents=True, exist_ok=True)
    speaker_dirs = sorted([d for d in src.iterdir() if d.is_dir() and ESD_SPEAKER_RE.match(d.name)])
    if not speaker_dirs:
        typer.echo(f"ERROR: no speaker folders (0001/, 0002/, ...) under {src}", err=True)
        raise typer.Exit(1)

    logger.info("ESD root: %s", src)
    logger.info("Found %d speaker dirs", len(speaker_dirs))
    logger.info("Speakers filter: %s", want_speakers or "ALL")
    logger.info("Emotions filter: %s", want_emotions)

    rows_written = 0
    skipped = 0
    cells_seen: Counter[tuple[str, str]] = Counter()

    with out.open("w", encoding="utf-8") as fp:
        for speaker_dir in speaker_dirs:
            spk = speaker_dir.name
            if want_speakers and spk not in want_speakers:
                continue
            lang = _esd_lang(spk)
            transcript = _read_esd_transcript(speaker_dir)
            logger.info("[%s/%s] %d transcript lines", spk, lang, len(transcript))

            for emotion_dir in sorted(speaker_dir.iterdir()):
                if not emotion_dir.is_dir() or emotion_dir.name not in want_emotions:
                    continue
                emotion = emotion_dir.name
                wavs = sorted(emotion_dir.glob("*.wav"))
                if max_per_cell > 0:
                    wavs = wavs[:max_per_cell]
                logger.info("  [%s/%s] %d wavs", spk, emotion, len(wavs))

                for wav_path in wavs:
                    utt_id = wav_path.stem  # e.g. "0011_000351"
                    text_emotion = transcript.get(utt_id)
                    if text_emotion is None:
                        # Missing transcript — skip rather than emit blank
                        # text, which would silently kill WER eval later.
                        # 没有 transcript 直接跳过：宁可丢条数据，
                        # 也不要写空 text 让下游 WER eval 静默挂掉。
                        logger.debug("no transcript for %s; skipping", wav_path.name)
                        skipped += 1
                        continue
                    text, _emotion_from_text = text_emotion

                    try:
                        scores = _score_one_wav(wav_path)
                    except Exception as e:  # noqa: BLE001 — keep ingestion robust
                        logger.warning("score failed for %s: %s; skipping", wav_path, e)
                        skipped += 1
                        continue

                    row = ManifestRow(
                        manifest_version="1.1",
                        chunk_id=f"esd_{spk}_{emotion}_{utt_id}",
                        audio_path=str(wav_path),
                        source_file=str(wav_path),  # ESD chunks ARE the source
                        speaker_id=f"esd_{spk}",
                        text=text,
                        lang=lang,
                        duration=scores["duration"],
                        emotion_tag=ESD_EMOTION_TO_TAG[emotion],
                        # Ground-truth label confidence = 1.0 (vs M1 which
                        # uses model-tagged emotion with confidence < 1).
                        # ground-truth 标注，置信度 1.0（M1 是模型标的所以 <1）。
                        emotion_confidence=1.0,
                        snr_db=scores["snr_db"],
                        mos_ovr=scores["mos_ovr"],
                        mos_sig=scores["mos_sig"],
                        mos_bak=scores["mos_bak"],
                        clipped=scores["clipped"],
                        source_dataset="esd",
                    )
                    fp.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
                    rows_written += 1
                    cells_seen[(spk, emotion)] += 1

                    if rows_written % 200 == 0:
                        logger.info("  ... %d rows written", rows_written)

    typer.echo(f"\n✓ wrote {rows_written} manifest rows to {out}")
    typer.echo(f"  skipped: {skipped}")
    typer.echo(f"  (speaker, emotion) cells: {len(cells_seen)}")
    if cells_seen:
        per_cell = sorted(cells_seen.values())
        typer.echo(f"  chunks/cell — min: {per_cell[0]}  median: {per_cell[len(per_cell)//2]}  max: {per_cell[-1]}")


# ---------------------------------------------------------------------------
# Subcommand: stats — verify the "≥30 per (speaker, emotion)" gate
# 子命令：stats —— 验证"每个 (speaker, emotion) ≥ 30 条"门槛
# ---------------------------------------------------------------------------


@app.command("stats")
def stats(
    manifest: Path = typer.Argument(..., help="Path to a manifest.jsonl produced by ingest-*."),
    min_per_cell: int = typer.Option(
        30, "--min-per-cell",
        help="Verification threshold: each (speaker, emotion) must have ≥ N chunks.",
    ),
) -> None:
    """Print speaker × emotion distribution + the verification gate.

    打印 (speaker, emotion) 分布 + 验收门槛是否通过。

    Verification (default): each (speaker, emotion) cell must have ≥ 30
    chunks. Below threshold = bad coverage = LoRA training will be
    speaker-biased.

    验收：每个 (speaker, emotion) 格子至少 30 条。低于阈值说明覆盖不均，
    LoRA 训练会偏 speaker。
    """
    manifest = manifest.expanduser().resolve()
    if not manifest.exists():
        typer.echo(f"ERROR: manifest not found: {manifest}", err=True)
        raise typer.Exit(1)

    by_speaker: dict[str, Counter[str]] = defaultdict(Counter)
    speakers_lang: dict[str, str] = {}
    total = 0
    by_emotion: Counter[str] = Counter()
    by_dataset: Counter[str] = Counter()

    with manifest.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # Older M1 manifests (trump_wef v0.1.x) don't have speaker_id —
            # fall back to source_file stem so the stats still print.
            # 老版本 M1 manifest 没 speaker_id；退化用 source_file stem。
            spk = row.get("speaker_id") or Path(row.get("source_file", "unknown")).stem
            emo = row.get("emotion_tag", "unknown")
            by_speaker[spk][emo] += 1
            speakers_lang.setdefault(spk, row.get("lang", "?"))
            by_emotion[emo] += 1
            by_dataset[row.get("source_dataset", "?")] += 1
            total += 1

    typer.echo(f"\n=== Manifest stats: {manifest.name} ===")
    typer.echo(f"Total chunks: {total}")
    typer.echo(f"Total speakers: {len(by_speaker)}")
    typer.echo(f"By dataset:")
    for ds, n in by_dataset.most_common():
        typer.echo(f"  {ds:12s} {n}")

    typer.echo(f"\nBy emotion (across all speakers):")
    for emo, n in by_emotion.most_common():
        typer.echo(f"  {emo:12s} {n}")

    typer.echo(f"\nBy speaker (showing (speaker, emotion) counts):")
    failed_cells: list[tuple[str, str, int]] = []
    for spk in sorted(by_speaker):
        emo_counts = by_speaker[spk]
        lang = speakers_lang.get(spk, "?")
        total_for_spk = sum(emo_counts.values())
        breakdown = ", ".join(f"{e}={n}" for e, n in sorted(emo_counts.items()))
        typer.echo(f"  {spk:14s} (lang={lang}) total={total_for_spk:5d}   [{breakdown}]")
        for emo, n in emo_counts.items():
            if n < min_per_cell:
                failed_cells.append((spk, emo, n))

    typer.echo(f"\n=== Verification: every (speaker, emotion) ≥ {min_per_cell} ? ===")
    if not failed_cells:
        typer.echo(f"  ✓ PASS — all {sum(len(c) for c in by_speaker.values())} cells have ≥ {min_per_cell} chunks")
    else:
        typer.echo(f"  ✗ FAIL — {len(failed_cells)} cell(s) below threshold:")
        for spk, emo, n in failed_cells[:20]:
            typer.echo(f"    {spk}  {emo}  {n}")
        if len(failed_cells) > 20:
            typer.echo(f"    ... and {len(failed_cells) - 20} more")


# ---------------------------------------------------------------------------
# Subcommand: rebase-paths — make audio_path portable across machines
# 子命令：rebase-paths —— 把 audio_path 改成跨机器可移植
# ---------------------------------------------------------------------------


@app.command("rebase-paths")
def rebase_paths(
    manifest: Path = typer.Argument(..., help="Path to a manifest.jsonl to rewrite."),
    strip_prefix: str = typer.Option(
        ..., "--strip-prefix",
        help="Absolute path prefix to remove (e.g. '/Users/attention/data/esd_raw/Emotion Speech Dataset/').",
    ),
    replace_with: str = typer.Option(
        "${ESD_ROOT}/", "--replace-with",
        help="Replacement prefix; defaults to '${ESD_ROOT}/'. Use env var syntax for portability.",
    ),
    fields: str = typer.Option(
        "audio_path,source_file", "--fields",
        help="Comma-separated manifest fields to rewrite.",
    ),
    in_place: bool = typer.Option(
        False, "--in-place",
        help="Modify manifest in place (writes a .bak backup first).",
    ),
    out: Path = typer.Option(
        None, "--out",
        help="Write rewritten manifest to this path. Required if --in-place is not set.",
    ),
) -> None:
    """Rewrite absolute paths to portable ``${VAR}/...`` form.

    把绝对路径改写成 ``${VAR}/...`` 形式，方便跨机器迁移。

    Why this exists
    ---------------
    The manifest produced by ``ingest-esd`` (and friends) stores absolute
    paths like ``/Users/me/data/esd_raw/.../foo.wav`` — when copied to a
    different machine (e.g. AutoDL `/root/autodl-fs/esd_raw/...`) those
    paths break. This command rewrites them to ``${ESD_ROOT}/foo.wav``
    so the training code can ``os.path.expandvars(path)`` against the
    machine-specific env var.

    为什么需要这个
    -------------
    ingest 出来的 manifest 用绝对路径，换机器（如本地 → AutoDL）就废了。
    本命令改写成 ``${ESD_ROOT}/foo.wav``，训练代码用 expandvars 按机器
    环境变量解析。

    Example
    -------
    ::

        # Local manifest -> AutoDL-portable
        python scripts/build_two_tier_dataset.py rebase-paths \\
            datasets/esd/manifest.jsonl \\
            --strip-prefix "/Users/attention/data/esd_raw/Emotion Speech Dataset/" \\
            --replace-with '${ESD_ROOT}/' \\
            --in-place

        # On AutoDL, before training:
        export ESD_ROOT=/root/autodl-fs/esd_raw/Emotion\\ Speech\\ Dataset
    """
    manifest = manifest.expanduser().resolve()
    if not manifest.exists():
        typer.echo(f"ERROR: manifest not found: {manifest}", err=True)
        raise typer.Exit(1)
    if not in_place and out is None:
        typer.echo("ERROR: must specify --in-place OR --out <path>", err=True)
        raise typer.Exit(1)

    target_fields = [f.strip() for f in fields.split(",") if f.strip()]
    if not target_fields:
        typer.echo("ERROR: --fields cannot be empty", err=True)
        raise typer.Exit(1)

    if in_place:
        backup = manifest.with_suffix(manifest.suffix + ".bak")
        backup.write_bytes(manifest.read_bytes())
        typer.echo(f"  backup: {backup}")
        out_path = manifest
    else:
        out_path = out.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # Read everything before opening the writer — otherwise an in-place
    # rewrite truncates the file before we read it. ~20 MB for ESD is
    # trivial to keep in memory.
    # 先把所有行读进内存——in-place 时如果先开 writer 会把文件清空。
    # ESD 才 20 MB，整个进内存毫无压力。
    lines = manifest.read_text(encoding="utf-8").splitlines()

    rewritten = 0
    unchanged = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row_changed = False
            for fld in target_fields:
                val = row.get(fld)
                if isinstance(val, str) and val.startswith(strip_prefix):
                    row[fld] = replace_with + val[len(strip_prefix):]
                    row_changed = True
            if row_changed:
                rewritten += 1
            else:
                unchanged += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    typer.echo(f"\n✓ wrote {out_path}")
    typer.echo(f"  rows rewritten: {rewritten}")
    typer.echo(f"  rows unchanged: {unchanged}")
    if unchanged > 0 and rewritten == 0:
        typer.echo(f"  ⚠️ no rows matched --strip-prefix={strip_prefix!r}; check the value", err=True)


# ---------------------------------------------------------------------------
# Subcommand: build-pairs — Tier 1 / Tier 2 training triples + eval splits
# 子命令：build-pairs —— Tier 1 / Tier 2 训练三元组 + eval split
# ---------------------------------------------------------------------------


@dataclass
class TrainingPair:
    """One training triple for LoRA: (text, ref_audio, target_audio) + meta.

    LoRA 训练用的一个三元组：(目标文本, 参考音频, 目标音频) + 元信息。

    The model learns to: given ``ref_audio`` as speaker conditioning and
    ``text`` as content, produce audio that sounds like ``target_audio``
    (same speaker as ref, but potentially different style/sentence). The
    style labels (``ref_style`` / ``target_style``) are NOT fed to the
    model directly — they're metadata for analysis and for style-balanced
    batch sampling at training time.

    训练时模型学：把 ``ref_audio`` 当 speaker 条件 + ``text`` 当内容，
    产出与 ``target_audio`` 听感一致的音频（与 ref 同 speaker、可能不同
    style/sentence）。``ref_style`` / ``target_style`` 不直接喂模型——
    只是元信息，用于分析和训练时按 style 做 batch balancing。
    """

    text: str
    ref_audio: str
    target_audio: str
    speaker_id: str
    target_style: str
    ref_style: str
    lang: str
    target_duration: float
    ref_duration: float
    target_mos_ovr: float
    ref_mos_ovr: float
    target_chunk_id: str
    ref_chunk_id: str
    source_dataset: str


@dataclass
class GoldClip:
    """A held-out high-quality clip used as SECS_vs_gold ground truth.

    held-out 的高质量片段，作为 SECS_vs_gold 评测的 ground truth。

    NEVER appears in any training pair (neither as ref nor target). Used
    only at eval time: ``SECS(synthesized_audio, gold_clip)`` measures
    true speaker identity transfer (avoids the ref-leakage假阳性).

    绝不进任何训练对（既不当 ref 也不当 target）。只在 eval 时用：
    ``SECS(合成音频, gold_clip)`` 测真正的说话人身份迁移，避免
    把 ref 直接用作 gold 的 ref-leakage 假阳性。
    """

    speaker_id: str
    chunk_id: str
    audio_path: str
    lang: str
    duration: float
    style: str
    mos_ovr: float
    source_dataset: str


def _select_gold_clips(
    chunks_for_speaker: list[dict],
    gold_count: int,
    preferred_style: str = "neutral",
) -> tuple[list[dict], set[str]]:
    """Pick highest-MOS clips of preferred_style as gold; return (golds, gold_ids).

    挑 mos_ovr 最高的 ``preferred_style`` 片段当 gold；返回 (gold 列表, gold chunk_id 集合)。

    Falls back to any style if preferred_style has < gold_count clips.
    若 preferred_style 数量不够则放宽到任意 style。
    """
    candidates = [c for c in chunks_for_speaker
                  if c.get("emotion_tag") == preferred_style]
    if len(candidates) < gold_count:
        # Fall back to all styles.
        candidates = list(chunks_for_speaker)
    candidates.sort(key=lambda c: c.get("mos_ovr", 0.0), reverse=True)
    golds = candidates[:gold_count]
    return golds, {c["chunk_id"] for c in golds}


def _build_tier1_pairs(
    chunks_for_speaker: list[dict],
    gold_ids: set[str],
    refs_per_target: int,
    min_ref_dur: float,
    max_target_dur: float,
    rng: random.Random,
) -> list[TrainingPair]:
    """Build (target, ref) pairs for a single speaker in Tier 1 logic.

    给单个 speaker 构造 Tier 1 训练对：同 speaker、不同 emotion 配对。

    For each non-gold target clip, sample N reference clips from the
    *same speaker* but with a *different emotion*, satisfying min/max
    duration constraints.

    对每个非 gold 的 target 片段，从同 speaker 但不同 emotion 的池子里
    随机抽 N 个 ref，并满足 min_ref_dur / max_target_dur 约束。
    """
    pool = [c for c in chunks_for_speaker if c["chunk_id"] not in gold_ids]
    # Group by emotion to enable cross-emotion sampling.
    # 按情绪分桶，方便做 cross-emotion 采样。
    by_emotion: dict[str, list[dict]] = defaultdict(list)
    for c in pool:
        by_emotion[c.get("emotion_tag", "unknown")].append(c)

    pairs: list[TrainingPair] = []
    for target in pool:
        if target.get("duration", 0) > max_target_dur:
            continue
        target_emotion = target.get("emotion_tag", "unknown")
        # Refs must be from a DIFFERENT emotion (cross-style is the whole point
        # of style-following training; same-emotion refs leak the answer).
        # ref 必须是**不同**情绪——cross-style 正是 style-following 训练的核心，
        # 同 emotion 的 ref 等于把答案泄露给模型。
        ref_pool = [c for emo, clips in by_emotion.items()
                    if emo != target_emotion
                    for c in clips
                    if c.get("duration", 0) >= min_ref_dur]
        if not ref_pool:
            continue
        # Sample without replacement up to refs_per_target.
        # 无放回采样最多 refs_per_target 个。
        n = min(refs_per_target, len(ref_pool))
        chosen_refs = rng.sample(ref_pool, n)
        for ref in chosen_refs:
            pairs.append(TrainingPair(
                text=target["text"],
                ref_audio=ref["audio_path"],
                target_audio=target["audio_path"],
                speaker_id=target["speaker_id"],
                target_style=target_emotion,
                ref_style=ref.get("emotion_tag", "unknown"),
                lang=target.get("lang", "unknown"),
                target_duration=float(target.get("duration", 0)),
                ref_duration=float(ref.get("duration", 0)),
                target_mos_ovr=float(target.get("mos_ovr", 0)),
                ref_mos_ovr=float(ref.get("mos_ovr", 0)),
                target_chunk_id=target["chunk_id"],
                ref_chunk_id=ref["chunk_id"],
                source_dataset=target.get("source_dataset", "unknown"),
            ))
    return pairs


def _build_tier2_pairs(
    chunks_for_speaker: list[dict],
    gold_ids: set[str],
    refs_per_target: int,
    min_ref_dur: float,
    max_target_dur: float,
    rng: random.Random,
) -> list[TrainingPair]:
    """Build (target, ref) pairs for Tier 2 (single-speaker avatar).

    构造 Tier 2 单说话人 avatar 训练对——不强制 cross-emotion，
    因为 Tier 2 数据通常不带 emotion 标签（如 Trump WEF），
    单纯 (text, ref, target) 即可。

    Tier 2 wants overfitting to a specific speaker's idiosyncrasies, so
    cross-emotion is nice-to-have, not required. The model gets all the
    speaker variation from a single deep speaker dataset.

    Tier 2 追求过拟合某个 speaker 的微观特征，cross-emotion 锦上添花
    不必强求。模型从单 speaker 的深度数据里学到全部说话人变化。
    """
    pool = [c for c in chunks_for_speaker if c["chunk_id"] not in gold_ids]
    pairs: list[TrainingPair] = []
    for target in pool:
        if target.get("duration", 0) > max_target_dur:
            continue
        ref_pool = [c for c in pool
                    if c["chunk_id"] != target["chunk_id"]
                    and c.get("duration", 0) >= min_ref_dur]
        if not ref_pool:
            continue
        n = min(refs_per_target, len(ref_pool))
        chosen_refs = rng.sample(ref_pool, n)
        for ref in chosen_refs:
            pairs.append(TrainingPair(
                text=target["text"],
                ref_audio=ref["audio_path"],
                target_audio=target["audio_path"],
                speaker_id=target["speaker_id"],
                target_style=target.get("emotion_tag", "unknown"),
                ref_style=ref.get("emotion_tag", "unknown"),
                lang=target.get("lang", "unknown"),
                target_duration=float(target.get("duration", 0)),
                ref_duration=float(ref.get("duration", 0)),
                target_mos_ovr=float(target.get("mos_ovr", 0)),
                ref_mos_ovr=float(ref.get("mos_ovr", 0)),
                target_chunk_id=target["chunk_id"],
                ref_chunk_id=ref["chunk_id"],
                source_dataset=target.get("source_dataset", "unknown"),
            ))
    return pairs


def _auto_pick_unseen(
    speakers_by_lang: dict[str, list[str]],
    per_lang: int,
) -> set[str]:
    """Deterministically pick N unseen speakers per language for eval.

    每种语言**确定性**地挑 N 个说话人作为 unseen eval 集。

    Deterministic = sort by id, take last N. This is reproducible without
    a seed and keeps the lowest-numbered speakers (often more cleaning
    effort, by convention) in training.
    确定性 = 按 id 排序取尾部 N 个。无需 seed 即可复现；编号小的（通常
    清洗更细致）留在训练里。
    """
    unseen: set[str] = set()
    for lang, spks in speakers_by_lang.items():
        spks_sorted = sorted(spks)
        unseen.update(spks_sorted[-per_lang:])
    return unseen


@app.command("build-pairs")
def build_pairs(
    manifest: Path = typer.Option(..., "--manifest", help="Input manifest.jsonl (rebase-paths preferred)."),
    out_dir: Path = typer.Option(
        REPO_ROOT / "datasets" / "two_tier",
        "--out-dir", help="Output directory for tier1_*.jsonl files.",
    ),
    tier: int = typer.Option(1, "--tier", help="1 = multi-speaker style LoRA; 2 = single-speaker avatar."),
    refs_per_target: int = typer.Option(1, "--refs-per-target", help="Sample N refs per target clip."),
    min_ref_dur: float = typer.Option(3.0, "--min-ref-dur", help="Min ref audio duration (s)."),
    max_target_dur: float = typer.Option(15.0, "--max-target-dur", help="Max target audio duration (s)."),
    min_mos_ovr: float = typer.Option(2.5, "--min-mos", help="Filter clips with mos_ovr below this."),
    unseen_per_lang: int = typer.Option(2, "--unseen-per-lang", help="Hold out N speakers per lang as unseen (tier 1 only)."),
    gold_per_speaker: int = typer.Option(2, "--gold-per-speaker", help="Pick top-N MOS clips per speaker as gold."),
    max_pairs_per_speaker: int = typer.Option(-1, "--max-pairs-per-speaker", help="Cap pairs per speaker (-1 = no cap)."),
    seed: int = typer.Option(42, "--seed"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Construct LoRA training pairs + eval splits from a manifest.

    从 manifest 构造 LoRA 训练对 + eval split。

    Outputs (under ``out_dir``):
      - ``tier{N}_train.jsonl``        — training pairs (excluding unseen speakers + gold clips)
      - ``tier{N}_eval_unseen.jsonl``  — pairs from held-out speakers (tier 1 only)
      - ``tier{N}_eval_gold.jsonl``    — per-speaker gold clips for SECS_vs_gold

    Pair construction rule
    ----------------------
    - **Tier 1**: ref must be from same speaker but **different emotion**
      than target (forces cross-style learning).
    - **Tier 2**: ref is any other clip from same speaker (no emotion
      constraint, since Tier 2 data often lacks emotion labels).

    Gold clip rule
    --------------
    Per speaker: pick top ``gold-per-speaker`` clips by ``mos_ovr``,
    prefer ``neutral`` emotion. These NEVER appear in training pairs.
    每 speaker：按 mos_ovr 取最高的 N 条（优先 neutral 情绪）作 gold；
    它们不进任何训练对，仅用于 SECS_vs_gold 评测。
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    manifest = manifest.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if tier not in (1, 2):
        typer.echo(f"ERROR: --tier must be 1 or 2, got {tier}", err=True)
        raise typer.Exit(1)

    # Load manifest, filter by quality gate.
    # 读 manifest，按质量阈值过滤。
    rng = random.Random(seed)
    chunks_by_speaker: dict[str, list[dict]] = defaultdict(list)
    speakers_by_lang: dict[str, set[str]] = defaultdict(set)
    total_in, total_kept = 0, 0
    with manifest.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total_in += 1
            if row.get("clipped"):
                continue
            if row.get("mos_ovr", 5.0) < min_mos_ovr:
                continue
            spk = row.get("speaker_id") or Path(row.get("source_file", "unknown")).stem
            row.setdefault("speaker_id", spk)
            chunks_by_speaker[spk].append(row)
            speakers_by_lang[row.get("lang", "unknown")].add(spk)
            total_kept += 1

    logger.info("loaded %d rows, kept %d after quality gate (mos_ovr ≥ %.1f, no clipping)",
                total_in, total_kept, min_mos_ovr)
    speakers_by_lang_list = {lang: list(spks) for lang, spks in speakers_by_lang.items()}
    logger.info("speakers by lang: %s", {lang: len(s) for lang, s in speakers_by_lang_list.items()})

    # Pick unseen speakers (tier 1 only).
    # 挑 unseen speakers（仅 tier 1）。
    unseen = _auto_pick_unseen(speakers_by_lang_list, unseen_per_lang) if tier == 1 else set()
    logger.info("unseen speakers (tier %d): %s", tier, sorted(unseen) or "(none, tier 2 doesn't hold-out speakers)")

    # Pick gold clips per speaker (all speakers, including unseen).
    # 每 speaker 挑 gold（包括 unseen，eval 时也要用）。
    gold_clips: list[GoldClip] = []
    gold_ids_by_speaker: dict[str, set[str]] = {}
    for spk, chunks in chunks_by_speaker.items():
        golds, ids = _select_gold_clips(chunks, gold_per_speaker)
        gold_ids_by_speaker[spk] = ids
        for g in golds:
            gold_clips.append(GoldClip(
                speaker_id=spk,
                chunk_id=g["chunk_id"],
                audio_path=g["audio_path"],
                lang=g.get("lang", "unknown"),
                duration=float(g.get("duration", 0)),
                style=g.get("emotion_tag", "unknown"),
                mos_ovr=float(g.get("mos_ovr", 0)),
                source_dataset=g.get("source_dataset", "unknown"),
            ))
    logger.info("gold clips: %d (across %d speakers)", len(gold_clips), len(chunks_by_speaker))

    # Build pairs per speaker.
    # 按 speaker 构造训练对。
    train_pairs: list[TrainingPair] = []
    eval_unseen_pairs: list[TrainingPair] = []

    pair_builder = _build_tier1_pairs if tier == 1 else _build_tier2_pairs
    for spk, chunks in chunks_by_speaker.items():
        pairs = pair_builder(
            chunks, gold_ids_by_speaker[spk],
            refs_per_target, min_ref_dur, max_target_dur, rng,
        )
        if max_pairs_per_speaker > 0 and len(pairs) > max_pairs_per_speaker:
            pairs = rng.sample(pairs, max_pairs_per_speaker)
        if spk in unseen:
            eval_unseen_pairs.extend(pairs)
            logger.info("  [unseen] %s: %d pairs", spk, len(pairs))
        else:
            train_pairs.extend(pairs)
            logger.info("  [train]  %s: %d pairs", spk, len(pairs))

    # Shuffle train pairs for good measure.
    # train pairs 打乱一次。
    rng.shuffle(train_pairs)

    # Write splits.
    # 写出三个 split。
    train_path = out_dir / f"tier{tier}_train.jsonl"
    unseen_path = out_dir / f"tier{tier}_eval_unseen.jsonl"
    gold_path = out_dir / f"tier{tier}_eval_gold.jsonl"

    with train_path.open("w", encoding="utf-8") as fp:
        for p in train_pairs:
            fp.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
    with unseen_path.open("w", encoding="utf-8") as fp:
        for p in eval_unseen_pairs:
            fp.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
    with gold_path.open("w", encoding="utf-8") as fp:
        for g in gold_clips:
            fp.write(json.dumps(asdict(g), ensure_ascii=False) + "\n")

    typer.echo(f"\n✓ Tier {tier} pairs built (seed={seed}):")
    typer.echo(f"  {train_path}            {len(train_pairs):6d} pairs")
    typer.echo(f"  {unseen_path}     {len(eval_unseen_pairs):6d} pairs")
    typer.echo(f"  {gold_path}        {len(gold_clips):6d} clips")

    # Sanity: style distribution in train.
    # 健全检查：train 集里的 style 分布。
    if train_pairs:
        style_pairs = Counter((p.ref_style, p.target_style) for p in train_pairs)
        typer.echo(f"\n  Top 10 (ref_style → target_style) pairs in train:")
        for (rs, ts), n in style_pairs.most_common(10):
            typer.echo(f"    {rs:10s} → {ts:10s}  {n}")


# ---------------------------------------------------------------------------
# Helper for training scripts (kept here so it's importable as
# ``from scripts.build_two_tier_dataset import resolve_audio_path``).
# 给训练脚本用的辅助函数。
# ---------------------------------------------------------------------------


def resolve_audio_path(path: str) -> Path:
    """Resolve a manifest audio_path with env var expansion.

    解析 manifest 里的 audio_path：支持 ``${VAR}/...`` 形式按环境变量替换。

    Training code should call this rather than ``Path(row["audio_path"])``
    directly, so the same manifest works on local + AutoDL without edits.

    训练代码应该用本函数而非直接 ``Path(row["audio_path"])``，
    这样同一份 manifest 在本地和 AutoDL 都能用，不用改 jsonl。
    """
    return Path(os.path.expandvars(path))


# ---------------------------------------------------------------------------
# Stubs for future dataset ingest (kept so --help lists them as TODO).
# 占位：未来要支持的数据集 ingest（保留在 --help 里当 TODO 提醒）。
# ---------------------------------------------------------------------------


@app.command("ingest-aishell3")
def ingest_aishell3(
    src: Path = typer.Option(..., help="Root of extracted AISHELL-3 (containing train/ test/)."),
    out: Path = typer.Option(
        REPO_ROOT / "datasets" / "aishell3" / "manifest.jsonl",
        help="Output manifest.jsonl path.",
    ),
    max_speakers: int = typer.Option(0, "--max-speakers", help="Cap on speakers (0 = all)."),
    train_only: bool = typer.Option(True, "--train-only / --include-test"),
    min_dur: float = typer.Option(1.0, "--min-dur"),
    max_dur: float = typer.Option(15.0, "--max-dur"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Ingest AISHELL-3 (218-spk Chinese studio corpus) → M1 manifest.

    把 AISHELL-3 转成 M1 manifest。AISHELL-3 是 218 说话人中文录音室
    数据集，没有 emotion 标签，全部标 `neutral`。

    Audio paths in the manifest are **absolute** from src.
    Use ``rebase-paths`` afterwards to make them portable.
    manifest 里的 audio_path 是**绝对路径**；之后用 rebase-paths 改写成
    ``${AISHELL3_ROOT}/...`` 形式。
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    src = src.expanduser().resolve()
    out = out.expanduser().resolve()
    if not src.is_dir():
        typer.echo(f"ERROR: src does not exist: {src}", err=True)
        raise typer.Exit(1)

    out.parent.mkdir(parents=True, exist_ok=True)

    splits = ["train"] if train_only else ["train", "test"]
    rows_written = 0
    speakers_seen: set[str] = set()
    skipped_no_text = 0
    skipped_dur = 0
    skipped_score = 0

    with out.open("w", encoding="utf-8") as fp:
        for split in splits:
            split_dir = src / split
            if not split_dir.is_dir():
                logger.warning("split dir not found: %s", split_dir)
                continue
            speaker_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())
            if max_speakers > 0:
                speaker_dirs = speaker_dirs[:max_speakers]
            logger.info("[%s] %d speakers", split, len(speaker_dirs))

            for spk_dir in speaker_dirs:
                spk_id = f"aishell3_{spk_dir.name}"
                speakers_seen.add(spk_id)
                wavs = sorted(spk_dir.glob("*.wav"))
                logger.info("  %s: %d wavs", spk_id, len(wavs))

                for wav_path in wavs:
                    text_path = wav_path.with_suffix(".txt")
                    if not text_path.exists():
                        skipped_no_text += 1
                        continue
                    text = text_path.read_text(encoding="utf-8").strip()
                    if not text:
                        skipped_no_text += 1
                        continue

                    try:
                        scores = _score_one_wav(wav_path)
                    except Exception as e:
                        logger.debug("score failed for %s: %s; skipping", wav_path, e)
                        skipped_score += 1
                        continue

                    dur = scores["duration"]
                    if dur < min_dur or dur > max_dur:
                        skipped_dur += 1
                        continue

                    row = ManifestRow(
                        manifest_version="1.1",
                        chunk_id=f"aishell3_{spk_dir.name}_{wav_path.stem}",
                        audio_path=str(wav_path),
                        source_file=str(wav_path),
                        speaker_id=spk_id,
                        text=text,
                        lang="zh",
                        duration=dur,
                        emotion_tag="neutral",
                        emotion_confidence=1.0,
                        snr_db=scores["snr_db"],
                        mos_ovr=scores["mos_ovr"],
                        mos_sig=scores["mos_sig"],
                        mos_bak=scores["mos_bak"],
                        clipped=scores["clipped"],
                        source_dataset="aishell3",
                    )
                    fp.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
                    rows_written += 1

                    if rows_written % 1000 == 0:
                        logger.info("  ... %d rows written (%d speakers so far)", rows_written, len(speakers_seen))

    typer.echo(f"\n✓ wrote {rows_written} manifest rows to {out}")
    typer.echo(f"  speakers: {len(speakers_seen)}")
    typer.echo(f"  skipped — no text: {skipped_no_text}  bad duration: {skipped_dur}  score fail: {skipped_score}")


@app.command("ingest-libritts")
def ingest_libritts() -> None:
    """(stub) Walk LibriTTS-R, emit manifest. Optional for English baseline.

    （占位）LibriTTS-R ingest；英文 baseline 可选。
    """
    typer.echo("Not yet implemented — see RESEARCH_PLAN §3.2 / Week 1.")
    raise typer.Exit(2)


if __name__ == "__main__":
    app()
