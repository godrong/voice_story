#!/usr/bin/env python
"""Run objective evaluation on every wav in experiments/exp_002_*/outputs/.

Reads `config.yaml` for the target text and reference audio definitions,
parses output filenames `{target}__{ref}__{instruct}.wav`, runs the four
objective metrics from `core.eval_tts` on each, and writes a markdown
report to `experiments/exp_002_ref_and_instruct/eval_objective.md`.

The `Transcriber` and WavLM model are loaded once and reused across all
wavs to avoid repeated cold-start cost.

对 experiments/exp_002_*/outputs/ 下的每条 wav 跑客观评测。

读取 config.yaml 拿到 target_text 与 reference 定义，按文件名
``{target}__{ref}__{instruct}.wav`` 解析出三段 id，跑 core.eval_tts
的四项指标，写 markdown 报告。

Transcriber 与 WavLM 模型在循环外一次性加载、复用，避免每条 wav 重新
冷启动。
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import yaml

# Make project root importable when run as a script.
# 作为脚本运行时把项目根加进 sys.path，使得能 import core / agents。
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.asr import Transcriber  # noqa: E402
from core.eval_tts import TTSEvalScores, evaluate_synthesis  # noqa: E402

logger = logging.getLogger(__name__)

EXP_DIR = REPO_ROOT / "experiments" / "exp_002_ref_and_instruct"
OUT_DIR = EXP_DIR / "outputs"
REPORT_PATH = EXP_DIR / "eval_objective.md"

FILENAME_RE = re.compile(r"^(?P<target>[^_]+(?:_[^_]+)*?)__(?P<ref>[^_]+(?:_[^_]+)*?)__(?P<instruct>[^_]+(?:_[^_]+)*?)\.wav$")


def _parse_filename(name: str) -> tuple[str, str, str] | None:
    """Split ``{target}__{ref}__{instruct}.wav`` into three ids.

    把文件名拆成 (target_id, ref_id, instruct_id)。

    The double-underscore separator is unambiguous because none of the
    components themselves contain ``__`` (verified against the runner that
    produced these files).
    双下划线分隔；组件本身不含 ``__``，所以分割无歧义。
    """
    parts = name.removesuffix(".wav").split("__")
    if len(parts) != 3:
        return None
    return parts[0], parts[1], parts[2]


def _load_config_maps(config_path: Path) -> tuple[dict[str, str], dict[str, dict]]:
    """Load `target_texts` and `references` from config.yaml.

    从 config.yaml 读 target_texts (id→text) 与 references (id→{audio, prompt_text})。

    Note: t1 and t2 are commented out in the current config (only t3 is
    active). For those we fall back to the literal text taken from the
    original commented block, kept here for historical wavs in outputs/.
    注：当前 config.yaml 把 t1/t2 注释掉了，但 outputs/ 里仍有这些遗留
    wav。这里硬编码 t1/t2 的文本以便仍能算 WER。
    """
    with config_path.open("r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)

    targets: dict[str, str] = {}
    for entry in cfg.get("target_texts", []) or []:
        if entry.get("id"):
            targets[entry["id"]] = entry["text"]

    # Hardcoded legacy texts (commented in current config) — keeping the
    # eval able to score historical wavs in outputs/.
    # 历史遗留文本（当前 config 已注释）；用于评估 outputs/ 里的旧 wav。
    if "t1_basic_test" not in targets:
        targets["t1_basic_test"] = (
            "Hello world. This is a test of voice cloning using CosyVoice 2 zero shot. "
            "The quick brown fox jumps over the lazy dog."
        )
    if "t2_trump_style" not in targets:
        targets["t2_trump_style"] = (
            "We are going to make voice cloning great again, believe me. Nobody does it better. "
            "It's tremendous, just tremendous."
        )

    refs: dict[str, dict] = {}
    for entry in cfg.get("references", []) or []:
        refs[entry["id"]] = {
            "audio": entry["audio"],
            "prompt_text": entry.get("prompt_text", ""),
            "chunk_id": entry.get("chunk_id", ""),
        }
    return targets, refs


def _scoreboard_row(
    target_id: str, ref_id: str, instruct_id: str, scores: TTSEvalScores,
) -> dict:
    """Compact dict for the report table.

    报告表格的紧凑 dict 行。
    """
    return {
        "target": target_id,
        "ref": ref_id,
        "instruct": instruct_id,
        "mos_nisqa": scores.mos_nisqa,
        "mos_p808": scores.mos_p808,
        "wer": scores.wer if scores.wer is not None else scores.cer,
        "wer_kind": "cer" if scores.cer is not None else "wer",
        "secs": scores.secs,
        "f0_rmse_hz": scores.f0_rmse_hz,
        "duration_s": scores.duration_s,
        "eval_time_s": scores.eval_time_s,
        "asr": scores.wer_transcript,
    }


def _fmt(val: float | None, ndigits: int = 3) -> str:
    """Format a float-or-None for the markdown table.

    格式化浮点 / None 给 markdown 表用。
    """
    if val is None:
        return "—"
    return f"{val:.{ndigits}f}"


def _build_report(rows: list[dict], targets: dict[str, str]) -> str:
    """Render the markdown report.

    渲染 markdown 报告：每个 target_id 一节，节内一张表 + 该组 Δ 对比。
    """
    lines: list[str] = []
    lines.append("# Experiment 002 — Objective Eval Baseline")
    lines.append("")
    lines.append(f"_Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")
    lines.append("")
    lines.append("**Metrics**:")
    lines.append("- **MOS-NISQA**: NISQA neural MOS predictor (primary naturalness axis), [1, 5], higher better.")
    lines.append("- **MOS-P808**: DNSMOS-P.808 (auxiliary; known to mis-rank TTS output, kept only for cross-check).")
    lines.append("- **WER/CER**: ASR cycle (faster-whisper / FunASR) vs `normalize_for_tts(target_text)`, [0, 1], lower better.")
    lines.append("- **SECS**: `microsoft/wavlm-base-plus-sv` x-vector cosine similarity vs ref audio, [-1, 1], higher better. > 0.7 ≈ same speaker.")
    lines.append("- **F0 RMSE**: librosa.pyin F0 RMSE over voiced overlap (Hz), lower better. None if voiced overlap < 5 frames.")
    lines.append("")

    by_target: dict[str, list[dict]] = {}
    for row in rows:
        by_target.setdefault(row["target"], []).append(row)

    for target_id in sorted(by_target):
        target_rows = by_target[target_id]
        target_text = targets.get(target_id, "(unknown)")
        lines.append(f"## {target_id}")
        lines.append("")
        snippet = target_text[:120].replace("\n", " ")
        if len(target_text) > 120:
            snippet += "..."
        lines.append(f"_target_text_: \"{snippet}\"")
        lines.append("")
        lines.append("| ref | instruct | MOS-NISQA | MOS-P808 | WER/CER | SECS | F0 RMSE | dur (s) | eval (s) |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for row in target_rows:
            wer_label = f"{_fmt(row['wer'], 3)} ({row['wer_kind']})" if row["wer"] is not None else "—"
            lines.append(
                f"| {row['ref']} | {row['instruct']} | "
                f"{_fmt(row['mos_nisqa'])} | {_fmt(row['mos_p808'])} | "
                f"{wer_label} | {_fmt(row['secs'])} | "
                f"{_fmt(row['f0_rmse_hz'], 2)} | "
                f"{_fmt(row['duration_s'], 1)} | {_fmt(row['eval_time_s'], 1)} |"
            )
        lines.append("")

        # Δ summary: split by instruct (none = zero_shot baseline) vs instruct mode.
        # Δ 小结：按 instruct=none vs 其它分组求均值。
        zero_shot = [r for r in target_rows if r["instruct"] == "none"]
        instruct_only = [r for r in target_rows if r["instruct"] != "none"]
        if zero_shot and instruct_only:
            lines.append(f"### Δ — instruct mode vs zero_shot baseline ({target_id})")
            lines.append("")
            lines.append("| Metric | zero_shot (mean) | instruct (mean) | Δ |")
            lines.append("|---|---|---|---|")
            for key, label, lower_better in [
                ("mos_nisqa", "MOS-NISQA", False),
                ("mos_p808", "MOS-P808", False),
                ("wer", "WER/CER", True),
                ("secs", "SECS", False),
                ("f0_rmse_hz", "F0 RMSE", True),
            ]:
                zs_vals = [r[key] for r in zero_shot if r.get(key) is not None]
                in_vals = [r[key] for r in instruct_only if r.get(key) is not None]
                if not zs_vals or not in_vals:
                    continue
                zs_mean = sum(zs_vals) / len(zs_vals)
                in_mean = sum(in_vals) / len(in_vals)
                delta = in_mean - zs_mean
                # Up arrow = instruct higher; flag direction based on
                # whether higher is better for this metric.
                # 上箭头表示 instruct 高；按指标方向标好坏。
                worse = (delta > 0) if lower_better else (delta < 0)
                marker = " ⚠️" if worse and abs(delta) > 0.05 else ""
                sign = "+" if delta >= 0 else ""
                lines.append(
                    f"| {label} | {zs_mean:.3f} | {in_mean:.3f} | "
                    f"**{sign}{delta:.3f}**{marker} |"
                )
            lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    """Entrypoint: scan outputs/, evaluate each, write report.

    入口：扫 outputs/ 评估每一条，写报告。
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    config_path = EXP_DIR / "config.yaml"
    targets, refs = _load_config_maps(config_path)
    logger.info("loaded %d target_texts, %d references", len(targets), len(refs))

    wavs = sorted(OUT_DIR.glob("*.wav"))
    logger.info("found %d wavs under %s", len(wavs), OUT_DIR)
    if not wavs:
        logger.error("no wavs found; aborting")
        return 1

    # Reusable models across the whole loop (saves ~10s per wav).
    # 跨循环复用模型（每条省 ~10s）。
    transcriber = Transcriber()

    rows: list[dict] = []
    for i, wav in enumerate(wavs, 1):
        parsed = _parse_filename(wav.name)
        if parsed is None:
            logger.warning("skip unparseable filename: %s", wav.name)
            continue
        target_id, ref_id, instruct_id = parsed
        target_text = targets.get(target_id)
        ref_meta = refs.get(ref_id)
        ref_wav = REPO_ROOT / ref_meta["audio"] if ref_meta else None

        logger.info("[%d/%d] %s | target=%s ref=%s instruct=%s",
                    i, len(wavs), wav.name, target_id, ref_id, instruct_id)

        scores = evaluate_synthesis(
            wav,
            ref_wav=ref_wav,
            target_text=target_text,
            lang="en",
            transcriber=transcriber,
        )
        logger.info(
            "  NISQA=%s P808=%s WER=%s SECS=%s F0=%sHz t=%.1fs",
            _fmt(scores.mos_nisqa), _fmt(scores.mos_p808),
            _fmt(scores.wer if scores.wer is not None else scores.cer),
            _fmt(scores.secs), _fmt(scores.f0_rmse_hz, 1), scores.eval_time_s,
        )
        rows.append(_scoreboard_row(target_id, ref_id, instruct_id, scores))

    report = _build_report(rows, targets)
    REPORT_PATH.write_text(report, encoding="utf-8")
    logger.info("wrote %s (%d rows)", REPORT_PATH, len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
