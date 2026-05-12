"""PreprocessAgent: chains separation -> VAD -> (optional) speaker filter.

Reads `state.raw_files` (standardized WAVs from SourceAgent), runs each
file through Demucs (per ADR-0008 always on), splits into chunks via
Silero VAD, and optionally filters by speaker similarity if the source
declares it's multi-speaker AND a reference clip is provided.

Outputs:
  - `<dataset_root>/vocals/<stem>.wav` per source
  - `<dataset_root>/chunks/<chunk_id>.wav` per chunk
  - `state.vocal_files` and `state.chunks` populated

预处理 agent：分离 → VAD → (可选) 说话人过滤。

读取 SourceAgent 写入的 raw_files，按以下顺序处理每个文件：
  1. Demucs 人声分离（ADR-0008 默认强制开）
  2. Silero VAD 切片成 3~15s
  3. （可选）说话人定位：仅当 source 声明多说话人且提供了参考片段时才跑

输出：
  - <dataset_root>/vocals/<stem>.wav 每个源一份
  - <dataset_root>/chunks/<chunk_id>.wav 每段一份
  - state.vocal_files / state.chunks 写满
"""

from __future__ import annotations

import logging
from pathlib import Path

from core.separation import Separator
from core.vad import VAD, chunk_id_for, write_chunks

from .state import ChunkInfo, PipelineState

logger = logging.getLogger(__name__)


class PreprocessAgent:
    """Pipeline stage that produces VAD-segmented vocal chunks.

    数据管线第二阶段：产出按说话人切片的人声 chunk 列表。

    Args:
        speaker_reference: Optional WAV path of the target speaker. If
            provided AND source is multi-speaker, runs SpeakerFilter.
            目标说话人参考 WAV（可选）。提供且 source 为多说话人时
            会启用 SpeakerFilter 过滤。
        speaker_threshold: Cosine cutoff for SpeakerFilter.
            说话人相似度阈值（默认 0.65）。
        separator_model: Demucs model name. Defaults to htdemucs.
            Demucs 模型名，默认 htdemucs。
        vad_kwargs: Forwarded to VAD constructor (min_sec / max_sec / ...).
            传给 VAD 构造函数的关键字参数。
    """

    name = "preprocess_agent"

    def __init__(
        self,
        *,
        speaker_reference: Path | str | None = None,
        speaker_threshold: float = 0.65,
        separator_model: str = "htdemucs",
        vad_kwargs: dict | None = None,
    ) -> None:
        self.speaker_reference = Path(speaker_reference) if speaker_reference else None
        self.speaker_threshold = speaker_threshold
        self.separator = Separator(model_name=separator_model)
        self.vad = VAD(**(vad_kwargs or {}))

    async def run(self, state: PipelineState) -> PipelineState:
        """Run separation + VAD over all raw files; populate vocal_files / chunks.

        对 state.raw_files 中的每个文件依次跑 Demucs + VAD，
        把产物写到 state.vocal_files 与 state.chunks。

        Args:
            state: Pipeline state. raw_files and source_meta must be set.
                   流水线状态对象。要求 raw_files 与 source_meta 已就位。

        Returns:
            Same state, mutated in place.
            就地修改后的 state。

        Raises:
            ValueError: If raw_files is empty.
                        raw_files 为空时抛出。
        """
        if not state.raw_files:
            raise ValueError("PreprocessAgent: state.raw_files is empty")
        state.ensure_dirs()
        vocals_dir = state.dataset_root / "vocals"
        chunks_dir = state.dataset_root / "chunks"

        for raw_path in state.raw_files:
            vocal_path = self.separator.separate(raw_path, vocals_dir)
            state.vocal_files.append(vocal_path)

            chunks = self.vad.chunk(vocal_path)
            stem = vocal_path.stem
            chunk_paths = write_chunks(chunks, chunks_dir, stem)
            for c, p in zip(chunks, chunk_paths):
                cid = chunk_id_for(stem, c)
                state.chunks.append(ChunkInfo(
                    chunk_id=cid,
                    path=p,
                    source_file=vocal_path,
                    start_sec=c.start_sec,
                    end_sec=c.end_sec,
                ))

        # Speaker filter is opt-in: needs both a multi-speaker source AND
        # a reference clip; otherwise it would just waste model load time.
        # 说话人过滤是 opt-in：需要多说话人源 + 参考片段两个条件都满足。
        if (
            state.source_meta is not None
            and not state.source_meta.is_single_speaker
            and self.speaker_reference is not None
        ):
            self._apply_speaker_filter(state)
        else:
            logger.info(
                "PreprocessAgent: skipping speaker filter "
                "(single_speaker=%s, ref=%s)",
                state.source_meta and state.source_meta.is_single_speaker,
                self.speaker_reference,
            )

        logger.info(
            "PreprocessAgent: %d vocal files -> %d chunks",
            len(state.vocal_files), len(state.chunks),
        )
        return state

    def _apply_speaker_filter(self, state: PipelineState) -> None:
        """Drop chunks whose embedding doesn't match the reference.

        过滤掉与参考说话人相似度低于阈值的 chunk。
        相似度记录保留在日志里，便于调阈值。
        """
        from core.speaker import SpeakerFilter
        sf = SpeakerFilter()
        sf.set_reference(self.speaker_reference)
        chunk_paths = [c.path for c in state.chunks]
        kept_paths, scores = sf.keep(chunk_paths, threshold=self.speaker_threshold)
        kept_set = set(kept_paths)
        before = len(state.chunks)
        state.chunks = [c for c in state.chunks if c.path in kept_set]
        logger.info(
            "PreprocessAgent: speaker filter %d -> %d (threshold=%.2f)",
            before, len(state.chunks), self.speaker_threshold,
        )
        # Lazy diagnostic dump for tuning the threshold later.
        # 把所有相似度 dump 到日志便于后续调阈值。
        for p, s in sorted(scores.items(), key=lambda x: x[1]):
            logger.debug("  %.3f  %s", s, p.name)
