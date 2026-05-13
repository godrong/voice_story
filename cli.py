"""voice-story CLI: typer-based command entry.

Subcommands (M1):

    voice-story ingest --source kaggle --dataset-id <id> --name <speaker>
    voice-story ingest --source local  --name <speaker>
    voice-story dataset stats          --name <speaker>

Future subcommands (M2+):

    voice-story synthesize --speaker <name> --book <path> --out <m4b>

The CLI is a thin shell over `agents/root_agent.py::build_m1_pipeline`
so all real logic stays in the agents/core layer (and remains testable
without invoking typer).

voice-story 命令行入口（typer 实现）。

M1 阶段的子命令：

    voice-story ingest --source kaggle --dataset-id <id> --name <speaker>
    voice-story ingest --source local  --name <speaker>
    voice-story dataset stats          --name <speaker>

M2+ 后续：

    voice-story synthesize --speaker <name> --book <path> --out <m4b>

CLI 只是 agents/root_agent 的薄壳，所有真实逻辑放在 agents/core，
确保不依赖 typer 也能单元测试。
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler

from agents.dataset_agent import (
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_MOS_OVR,
    DEFAULT_MIN_SNR_DB,
    FilterThresholds,
    load_manifest,
)
from agents.root_agent import build_m1_pipeline, run_pipeline

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Voice Story — clone a streamer's voice and read books in it.",
)
dataset_app = typer.Typer(no_args_is_help=True, help="Inspect generated datasets.")
app.add_typer(dataset_app, name="dataset")

console = Console()


def _setup_logging(verbose: bool) -> None:
    """Configure rich-formatted logging once per CLI invocation.

    一次 CLI 调用配置一次 rich 日志格式化。
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )


@app.command()
def ingest(
    source: str = typer.Option(..., help="Source plugin: 'local' or 'kaggle'."),
    name: str = typer.Option(..., help="Speaker / dataset short name."),
    dataset_id: str = typer.Option(
        None, help="Kaggle dataset id (owner/slug). Required if --source kaggle.",
    ),
    inputs_root: Path = typer.Option(
        Path("inputs"), help="Inputs root for --source local.",
    ),
    out_root: Path = typer.Option(
        Path("datasets"), help="Datasets output root.",
    ),
    lang_hint: str = typer.Option(
        None, help="ISO 639-1 language hint ('en'/'zh'). Skips langid.",
    ),
    speaker_ref: Path = typer.Option(
        None, help="Reference WAV of the target speaker (multi-speaker sources only).",
    ),
    speaker_threshold: float = typer.Option(
        0.65, help="Cosine threshold for the speaker filter.",
    ),
    is_single_speaker: bool = typer.Option(
        False, help="Mark source as single-speaker (skips speaker filter).",
    ),
    needs_separation: bool = typer.Option(
        True, help="Run Demucs vocal separation. Default True per ADR-0008.",
    ),
    max_files: int = typer.Option(
        None, help="Cap on number of source files (dev iteration aid).",
    ),
    overwrite: bool = typer.Option(
        False, help="Re-run even when outputs already exist.",
    ),
    min_mos: float = typer.Option(DEFAULT_MIN_MOS_OVR, help="Manifest MOS-OVR floor."),
    min_snr: float = typer.Option(DEFAULT_MIN_SNR_DB, help="Manifest SNR floor (dB)."),
    min_conf: float = typer.Option(DEFAULT_MIN_CONFIDENCE, help="Manifest ASR confidence floor."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the M1 data pipeline for one source.

    跑 M1 数据管线：源采集 → 标准化 → 分离 → VAD → ASR → 过滤 → manifest。
    """
    _setup_logging(verbose)
    if source == "kaggle" and not dataset_id:
        raise typer.BadParameter("--dataset-id is required when --source kaggle")

    if source == "local":
        source_kwargs = {
            "name": name,
            "inputs_root": inputs_root,
            "lang_hint": lang_hint,
            "needs_separation": needs_separation,
            "is_single_speaker": is_single_speaker,
        }
    elif source == "kaggle":
        source_kwargs = {
            "name": name,
            "dataset_id": dataset_id,
            "lang_hint": lang_hint,
            "needs_separation": needs_separation,
            "is_single_speaker": is_single_speaker,
        }
    else:
        raise typer.BadParameter(f"Unknown --source {source!r} (use 'local' or 'kaggle')")

    thresholds = FilterThresholds(
        min_mos_ovr=min_mos, min_snr_db=min_snr, min_confidence=min_conf,
    )
    stages = build_m1_pipeline(
        source_type=source,
        source_kwargs=source_kwargs,
        speaker_reference=speaker_ref,
        speaker_threshold=speaker_threshold,
        thresholds=thresholds,
        max_files=max_files,
        overwrite=overwrite,
        lang_hint=lang_hint,
    )
    state = asyncio.run(run_pipeline(stages, out_root / name))
    console.print(f"[green]done[/green]: {state.manifest_path}")


@dataset_app.command("stats")
def dataset_stats(
    name: str = typer.Option(..., help="Speaker / dataset short name."),
    out_root: Path = typer.Option(
        Path("datasets"), help="Datasets root (where ingest wrote).",
    ),
) -> None:
    """Print summary stats for a dataset's manifest.jsonl + report.md.

    打印 dataset 的简要统计（行数 / 时长 / 平均 MOS）和 report.md 内容。
    """
    root = out_root / name
    manifest = root / "manifest.jsonl"
    report = root / "report.md"
    if not manifest.exists():
        console.print(f"[red]not found[/red]: {manifest}")
        raise typer.Exit(1)

    rows = list(load_manifest(manifest))
    total_dur = sum(r.get("duration", 0.0) for r in rows)
    avg_mos = (
        sum(r.get("mos_ovr", 0.0) for r in rows) / len(rows) if rows else 0.0
    )
    console.print(f"[bold]{name}[/bold]")
    console.print(f"  chunks:        {len(rows)}")
    console.print(f"  total duration: {total_dur:.1f}s ({total_dur / 60:.1f}min)")
    console.print(f"  avg MOS-OVR:   {avg_mos:.2f}")
    if report.exists():
        console.print("\n" + report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    app()
