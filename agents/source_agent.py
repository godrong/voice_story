"""SourceAgent: thin dispatcher that drives a `Source` plugin and standardizes outputs.

Responsibilities:
  1. Instantiate the requested `Source` (local / kaggle / future yt-dlp).
  2. Iterate `source.fetch()` to get raw media file paths.
  3. Standardize each file via `core.audio_io.to_standard_wav` so downstream
     stages work on uniform 24 kHz / 16-bit / mono WAV.
  4. Write outputs into `<dataset_root>/raw/`.

This agent does NOT do any audio analysis — it's a glue layer between the
plugin layer (`core/sources/*`) and the rest of the pipeline.

源采集 agent。

职责：
  1. 按请求实例化 Source 插件（local / kaggle / 未来的 URL 源）。
  2. 通过 source.fetch() 拿到原始媒体文件路径流。
  3. 用 core.audio_io.to_standard_wav 把每个文件统一转码到
     24 kHz / 16-bit / mono WAV。
  4. 输出落到 <dataset_root>/raw/。

本 agent 不做任何音频分析，只是 source 插件层和后续 pipeline 之间的胶水。
"""

from __future__ import annotations

import logging
from typing import Any

from core import audio_io
from core.sources import get_source

from .state import PipelineState

logger = logging.getLogger(__name__)


class SourceAgent:
    """Pipeline stage that fetches raw media and standardizes to WAV.

    数据管线第一个 stage：取原始媒体 + 标准化转码。

    Args:
        source_type: Short name registered in core.sources.REGISTRY.
                     源插件类型简称（"local" / "kaggle" / ...）。
        source_kwargs: kwargs forwarded to the Source constructor.
                       传给 Source 构造函数的参数字典。
        max_files: Optional cap on number of files (for fast iteration).
                   可选的最大文件数限制（开发阶段加快迭代）。
        overwrite: If True, re-transcode even if outputs already exist.
                   是否覆盖已有的标准化 WAV。
    """

    name = "source_agent"

    def __init__(
        self,
        source_type: str,
        source_kwargs: dict[str, Any],
        *,
        max_files: int | None = None,
        overwrite: bool = False,
    ) -> None:
        self.source_type = source_type
        self.source_kwargs = source_kwargs
        self.max_files = max_files
        self.overwrite = overwrite

    async def run(self, state: PipelineState) -> PipelineState:
        """Fetch from source plugin and standardize each file into raw/.

        从 source 插件取原始文件，逐个调 ffmpeg 转码到 raw/ 目录下。

        Args:
            state: Pipeline state. dataset_root must be set; this agent
                   writes source_meta and raw_files.
                   流水线状态对象。要求 dataset_root 已设置；
                   本 agent 写入 source_meta 与 raw_files。

        Returns:
            Same state, mutated in place.
            就地修改后的同一个 state 对象。

        Raises:
            FileNotFoundError: If source has no files to yield (local source
                               with empty inputs dir).
                               源没有产出任何文件（如本地源目录为空）。
        """
        source = get_source(self.source_type, **self.source_kwargs)
        state.source_meta = source.meta
        state.ensure_dirs()
        raw_dir = state.dataset_root / "raw"

        count = 0
        for src_path in source.fetch():
            if self.max_files is not None and count >= self.max_files:
                logger.info("SourceAgent: reached max_files=%d, stopping", self.max_files)
                break
            try:
                info = audio_io.probe(src_path)
            except Exception as e:
                logger.warning("SourceAgent: probe failed for %s: %s", src_path, e)
                continue
            logger.info(
                "SourceAgent: %s | %.1fs | %dHz | %dch | %s",
                src_path.name, info.duration, info.sample_rate, info.channels, info.codec,
            )
            out = audio_io.to_standard_wav(src_path, raw_dir, overwrite=self.overwrite)
            state.raw_files.append(out)
            count += 1

        if count == 0:
            raise FileNotFoundError(
                f"SourceAgent: no media files produced from "
                f"source_type={self.source_type!r} kwargs={self.source_kwargs!r}"
            )
        logger.info("SourceAgent: standardized %d file(s) -> %s", count, raw_dir)
        return state
