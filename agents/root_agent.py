"""RootAgent: chains the M1 data-pipeline stages into a single runnable.

The orchestration is intentionally minimal — for M1 we use a plain
sequential `await stage.run(state)` loop. When the project actually
imports `google.adk` (M2+), we'll wrap each stage as a `BaseAgent`
subclass and assemble them with `SequentialAgent`. The lightweight
loop here keeps M1 testable without a full ADK install and matches
the same `state -> state` contract.

数据管线根 agent，串联 M1 各阶段。

编排刻意保持简单：M1 阶段就是一个 async 循环 `await stage.run(state)`。
等到 M2+ 真正引入 google.adk 时，会把每个 stage 包成 BaseAgent 子类
并用 SequentialAgent 装配。当前实现保持 ADK 不可用时也能跑通，
契约都是 `state -> state`，未来切换不需要改 stage 代码。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from core.sources import SourceMeta

from .dataset_agent import DatasetAgent, FilterThresholds
from .preprocess_agent import PreprocessAgent
from .source_agent import SourceAgent
from .state import PipelineState

logger = logging.getLogger(__name__)


class Stage(Protocol):
    """Minimal protocol every pipeline stage satisfies.

    每个 stage 都满足的最小 protocol：name + async run(state)。
    """

    name: str

    async def run(self, state: PipelineState) -> PipelineState: ...


def build_m1_pipeline(
    source_type: str,
    source_kwargs: dict[str, Any],
    *,
    speaker_reference: Path | str | None = None,
    speaker_threshold: float = 0.65,
    thresholds: FilterThresholds | None = None,
    max_files: int | None = None,
    overwrite: bool = False,
    lang_hint: str | None = None,
    enable_enhance: bool = True,
    enhance_mode: int = 0,
    vad_kwargs: dict | None = None,
) -> list[Stage]:
    """Construct the M1 data-pipeline stages in execution order.

    按执行顺序构造 M1 数据管线的所有 stage。

    Args:
        source_type: Source plugin name ("local" / "kaggle").
        source_kwargs: Forwarded to the Source constructor.
        speaker_reference: Optional target-speaker reference WAV path.
        speaker_threshold: Cosine cutoff for speaker filter.
        thresholds: FilterThresholds for the manifest gate.
        max_files: Cap on source file count (dev iteration aid).
        overwrite: Re-transcode even when raw/ already has output.
        lang_hint: Force a language for ASR (skips langid).
        enable_enhance: Run VoiceFixer between Demucs and VAD (ADR-0012).
            Default True; pass False for clean studio sources.
        enhance_mode: VoiceFixer mode (0=general/default, 1=stronger denoise,
            2=de-reverb).
        vad_kwargs: Forwarded to VAD constructor.

    Returns:
        List of Stage instances in pipeline order.
    """
    return [
        SourceAgent(
            source_type=source_type, source_kwargs=source_kwargs,
            max_files=max_files, overwrite=overwrite,
        ),
        PreprocessAgent(
            speaker_reference=speaker_reference,
            speaker_threshold=speaker_threshold,
            enable_enhance=enable_enhance,
            enhance_mode=enhance_mode,
            vad_kwargs=vad_kwargs,
        ),
        DatasetAgent(
            thresholds=thresholds,
            lang_hint=lang_hint,
        ),
    ]


async def run_pipeline(
    stages: Sequence[Stage], dataset_root: Path | str,
) -> PipelineState:
    """Run a list of stages sequentially against a fresh PipelineState.

    顺序执行 stage 列表；每个 stage 的输出是下一个 stage 的输入。

    Args:
        stages: Stage instances; typically from `build_m1_pipeline()`.
        dataset_root: Where the dataset gets written
            (e.g. `datasets/trump/`).

    Returns:
        Final PipelineState after all stages have run.
        所有 stage 执行完后的最终状态对象。
    """
    state = PipelineState(dataset_root=Path(dataset_root))
    state.ensure_dirs()
    for stage in stages:
        logger.info("=== %s ===", stage.name)
        state = await stage.run(state)
    return state
