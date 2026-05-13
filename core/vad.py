"""Voice Activity Detection + chunking using Silero VAD.

Splits a long vocal-only WAV into 3~15s utterance chunks suitable for
TTS training (CosyVoice 2 prefers this length range). Boundaries snap to
silence regions detected by Silero VAD so we don't cut mid-word.

Strategy:
  1. Run Silero VAD to get all speech regions.
  2. Greedily pack adjacent speech regions until either:
     a) cumulative duration would exceed `max_sec`, or
     b) the gap between two speech regions is "long enough" (>= 0.5s)
        AND we've already exceeded `min_sec`.
  3. Drop any chunk shorter than `min_sec`.

VAD 切片模块（基于 Silero VAD）。

把人声 WAV 切成 3~15s 短句，用于 TTS 训练。CosyVoice 2 偏好这个长度。
切片边界落在 Silero VAD 检测到的静音区，避免硬切单词。

策略：
  1. 跑 Silero VAD 拿到所有 speech segment。
  2. 贪心合并相邻 segment，直到：
     a) 累计时长会超过 max_sec，或
     b) 相邻 segment 之间静音 >=0.5s 且当前累计已超过 min_sec。
  3. 丢弃短于 min_sec 的片段。
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import audio_io

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Chunk:
    """A speech segment ready to be saved as a chunk WAV.

    一段已确定边界、可保存为切片 WAV 的语音区间。

    Attributes:
        start_sec: Start time in source audio (seconds).
                   在源音频中的起点（秒）。
        end_sec: End time in source audio (seconds).
                 在源音频中的终点（秒）。
        samples: 1-D float32 audio array for this chunk.
                 该切片的 float32 单声道波形数组。
    """

    start_sec: float
    end_sec: float
    samples: np.ndarray


class VAD:
    """Silero VAD wrapper that segments a WAV into TTS-ready chunks.

    Silero VAD 封装；负责把一整段人声 WAV 切成 3~15s 的可训练片段。

    The VAD model is loaded lazily on first use and reused across calls.

    模型首次调用时惰性加载，之后批量复用。

    Args:
        min_sec: Minimum chunk duration. Below this gets dropped.
                 最短切片时长，低于该值丢弃。
        max_sec: Maximum chunk duration before forced cut.
                 最长切片时长，超过则强制切。
        target_sec: Soft target around which to pack chunks.
                    切片软目标长度（贪心合并的"舒适区"）。
        min_silence_sec: Minimum silence gap to consider a boundary candidate.
                         视为可切边界的最小静音时长。
        threshold: Silero VAD speech-probability threshold in [0, 1].
                   Silero VAD 的 speech 概率阈值。
    """

    name = "silero-vad"

    def __init__(
        self,
        *,
        min_sec: float = 3.0,
        max_sec: float = 15.0,
        target_sec: float = 10.0,
        min_silence_sec: float = 0.5,
        threshold: float = 0.5,
    ) -> None:
        self.min_sec = min_sec
        self.max_sec = max_sec
        self.target_sec = target_sec
        self.min_silence_sec = min_silence_sec
        self.threshold = threshold
        self._vad = None  # lazy

    def _ensure_model(self):
        """Lazily load Silero VAD via the official package.

        惰性加载 Silero VAD 模型（首次调用时初始化）。
        """
        if self._vad is not None:
            return self._vad
        # Lazy import: silero_vad triggers a model download on first call.
        # 延迟导入：silero_vad 首次调用会触发模型下载。
        from silero_vad import load_silero_vad
        logger.info("VAD: loading Silero VAD")
        self._vad = load_silero_vad()
        return self._vad

    def chunk(self, wav_path: Path | str) -> list[Chunk]:
        """Read a WAV and return a list of bounded chunks.

        读取 WAV 并返回切好的 Chunk 列表（已合并到 3~15s 范围）。

        Args:
            wav_path: Standardized WAV path (24 kHz / mono).
                      已标准化的 WAV 路径。

        Returns:
            Chunks satisfying min_sec <= duration <= max_sec.
            满足时长约束的切片列表。
        """
        from silero_vad import get_speech_timestamps
        model = self._ensure_model()
        audio, sr = audio_io.load(wav_path)
        if sr != audio_io.TARGET_SR:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=audio_io.TARGET_SR)
            sr = audio_io.TARGET_SR

        # Silero VAD only accepts 8k or 16k (and multiples of 16k); our
        # standard 24k WAV must be resampled to 16k just for the VAD call.
        # We keep the original 24k audio for chunk slicing because chunk
        # files stay in the standard 24k pipeline format. Timestamps come
        # back in seconds, which is sample-rate independent.
        # Silero VAD 仅支持 8k/16k（及其倍数），24k 标准音频要在 VAD 调用前
        # 单独降采样到 16k；chunk 切片仍用原始 24k（pipeline 标准格式），
        # 时间戳以秒为单位与采样率无关。
        import torch
        import librosa
        VAD_SR = 16000
        if sr != VAD_SR:
            vad_audio = librosa.resample(audio, orig_sr=sr, target_sr=VAD_SR)
        else:
            vad_audio = audio
        speech = get_speech_timestamps(
            torch.from_numpy(vad_audio),
            model,
            sampling_rate=VAD_SR,
            threshold=self.threshold,
            return_seconds=True,
        )
        if not speech:
            logger.warning("VAD: no speech detected in %s", wav_path)
            return []

        chunks = list(self._pack(speech))
        out: list[Chunk] = []
        for start, end in chunks:
            if end - start < self.min_sec:
                continue
            s_idx = int(start * sr)
            e_idx = int(end * sr)
            out.append(Chunk(start, end, audio[s_idx:e_idx].copy()))
        logger.info("VAD: %s -> %d chunks", Path(wav_path).name, len(out))
        return out

    def _pack(
        self, speech: list[dict],
    ) -> Iterable[tuple[float, float]]:
        """Greedily pack adjacent speech regions into target-length chunks.

        贪心合并相邻 speech 区间，输出符合长度约束的切片边界。

        Args:
            speech: List of {"start": sec, "end": sec} from Silero VAD.
                    Silero VAD 返回的语音区间列表。

        Yields:
            (start_sec, end_sec) tuples within [min_sec, max_sec].
            (起点秒, 终点秒) 元组。
        """
        cur_start = speech[0]["start"]
        cur_end = speech[0]["end"]
        for nxt in speech[1:]:
            gap = nxt["start"] - cur_end
            new_end = nxt["end"]
            new_dur = new_end - cur_start

            # Force a cut if extending would blow the cap.
            # 超过 max_sec 强制切。
            if new_dur > self.max_sec:
                yield cur_start, cur_end
                cur_start, cur_end = nxt["start"], nxt["end"]
                continue

            # Honour a long-enough silence gap once we're past min_sec.
            # 静音足够长且当前已达到 min_sec，则在此切。
            cur_dur = cur_end - cur_start
            if gap >= self.min_silence_sec and cur_dur >= self.min_sec:
                yield cur_start, cur_end
                cur_start, cur_end = nxt["start"], nxt["end"]
                continue

            # Otherwise extend the current chunk.
            # 否则继续扩张当前 chunk。
            cur_end = new_end

            # Soft-target check: if we passed target_sec and have a small
            # gap-friendly spot, still close out.
            # 软目标检查：超过 target_sec 后遇到任何 gap 就收住。
            if (cur_end - cur_start) >= self.target_sec and gap > 0:
                yield cur_start, cur_end
                cur_start, cur_end = nxt["start"], nxt["end"]

        yield cur_start, cur_end


def write_chunks(chunks: list[Chunk], out_dir: Path | str, source_stem: str) -> list[Path]:
    """Save each Chunk as a WAV file with a stable id-based filename.

    把 Chunk 列表逐个落盘成 WAV，文件名采用 `<source_stem>__<startms>_<endms>.wav`
    的稳定格式，便于回溯。

    Args:
        chunks: Chunk list produced by `VAD.chunk()`.
                VAD.chunk() 产出的切片列表。
        out_dir: Directory to write into.
                 输出目录。
        source_stem: Stem of the parent file (used as filename prefix).
                     父文件的 stem，作为文件名前缀。

    Returns:
        List of written WAV paths in input order.
        写入的 WAV 路径列表（与输入顺序一致）。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for c in chunks:
        start_ms = int(c.start_sec * 1000)
        end_ms = int(c.end_sec * 1000)
        name = f"{source_stem}__{start_ms:08d}_{end_ms:08d}.wav"
        p = audio_io.save(out_dir / name, c.samples)
        paths.append(p)
    return paths


def chunk_id_for(source_stem: str, chunk: Chunk) -> str:
    """Build the canonical chunk_id used in PipelineState and manifests.

    生成 chunk 的标准 id（和 write_chunks 输出文件名一致，去掉后缀）。

    Args:
        source_stem: Parent file stem.
        chunk: The Chunk instance.

    Returns:
        Stable id string like "speech001__00012345_00018901".
        稳定的 id 字符串。
    """
    return f"{source_stem}__{int(chunk.start_sec * 1000):08d}_{int(chunk.end_sec * 1000):08d}"
