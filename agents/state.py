"""Shared mutable state object passed between pipeline stages.

Each stage reads what it needs from `PipelineState` and writes its outputs
back. This avoids ad-hoc keyword arguments between stages and keeps the
SequentialAgent contract simple: every stage has the same `run(state)`
signature. The state also doubles as a debugging snapshot — dumping it
after any stage shows exactly what's been computed so far.

Stage 间共享的可变状态对象。

每个 pipeline stage 从 PipelineState 读取所需输入、把产物写回去。
好处：所有 stage 的方法签名都是 `run(state) -> state`，
SequentialAgent 编排时不用关心字段细节；调试时打印 state 就能看到
每一步的产出。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from core.sources import SourceMeta


@dataclass
class ChunkInfo:
    """A single VAD-segmented audio chunk.

    一个由 VAD 切出来的音频片段。

    Attributes:
        chunk_id: Stable id, typically `<source_stem>__<start_ms>_<end_ms>`.
                  稳定 id，便于回溯到源文件与时间区间。
        path: Path to the chunk WAV on disk. 切片 WAV 路径。
        source_file: Path to the parent file (post-separation). 父文件路径。
        start_sec: Start offset in source. 在父文件中的起点（秒）。
        end_sec: End offset in source. 在父文件中的终点（秒）。
    """

    chunk_id: str
    path: Path
    source_file: Path
    start_sec: float
    end_sec: float


@dataclass
class TranscriptInfo:
    """ASR transcription result for a chunk.

    一个切片的 ASR 转写结果。

    Attributes:
        chunk_id: References ChunkInfo.chunk_id. 关联 ChunkInfo。
        text: Transcribed text with punctuation. 含标点的转写文本。
        lang: Detected ISO 639-1 language code. 检测到的语种。
        confidence: Average word-level confidence in [0, 1]. 平均置信度。
    """

    chunk_id: str
    text: str
    lang: str
    confidence: float


@dataclass
class QualityScore:
    """Quality metrics for a chunk (used by dataset filtering).

    切片的质量评分（数据集过滤阶段使用）。

    Attributes:
        chunk_id: References ChunkInfo.chunk_id. 关联 ChunkInfo。
        snr_db: WADA-SNR estimate in dB. WADA-SNR 信噪比估计。
        mos_ovr: DNSMOS overall score in [1, 5]. DNSMOS 总分。
        mos_sig: DNSMOS signal score. DNSMOS 信号分。
        mos_bak: DNSMOS background score. DNSMOS 背景分。
        clipped: Whether peak clipping was detected. 是否检测到削波。
    """

    chunk_id: str
    snr_db: float
    mos_ovr: float
    mos_sig: float
    mos_bak: float
    clipped: bool


@dataclass
class ProsodyScore:
    """Per-chunk prosody + emotion features (manifest v1.1, T2).

    单 chunk 的韵律 + 情绪特征束（manifest v1.1 的 T2 字段）。

    Style / reference selection (ADR-0010) reads these fields to match
    target sentences against dataset chunks. Distinct from QualityScore
    which gates training inclusion.

    风格控制 / reference 选择（ADR-0010）依据这些字段把目标句匹配到
    dataset chunk。与 QualityScore 不同 —— QualityScore 决定能不能
    进训练集，ProsodyScore 决定"听起来怎么样"。

    Attributes:
        chunk_id: References ChunkInfo.chunk_id. 关联 ChunkInfo。
        pitch_mean_hz: Mean F0 in Hz over voiced frames; NaN if unvoiced.
            voiced 帧的 F0 均值；全段 unvoiced 时为 NaN。
        pitch_std_hz: Std-dev of F0 in Hz. F0 标准差。
        energy_rms: Whole-chunk RMS in [0, 1]. 整段 RMS。
        loudness_lufs: Integrated loudness (ITU-R BS.1770).
            综合响度（LUFS）。
        speech_ratio: Fraction of frames above per-chunk RMS threshold.
            高于 chunk 内 RMS 阈值的帧占比。
        pace_units_per_sec: Characters/sec (ZH) or syllables/sec (EN).
            中文按字 / 英文按音节的语速。
        emotion_tag: Normalized top-1 emotion label from emotion2vec.
            emotion2vec 归一后的 top-1 情绪标签。
        emotion_confidence: Top-1 probability in [0, 1].
            top-1 概率。
    """

    chunk_id: str
    pitch_mean_hz: float
    pitch_std_hz: float
    energy_rms: float
    loudness_lufs: float
    speech_ratio: float
    pace_units_per_sec: float
    emotion_tag: str
    emotion_confidence: float


@dataclass
class PipelineState:
    """End-to-end state passed through every stage of the data pipeline.

    数据管线端到端共享的状态对象。每个 stage 读取自己需要的字段、
    把自己的产物追加进去，下一个 stage 就能直接消费。

    Attributes:
        dataset_root: Output root, typically `datasets/<name>/`.
                      输出根目录，约定为 datasets/<name>/。
        source_meta: Set by SourceAgent. 由 SourceAgent 写入。
        raw_files: Standardized WAV paths produced by SourceAgent.
                   SourceAgent 产出的标准化 WAV 列表。
        vocal_files: Vocal-only WAVs produced by separation step.
                     separation 步骤产出的人声分离结果。
        enhanced_files: VoiceFixer-enhanced WAVs (ADR-0012); empty when
                        --skip-enhance. enhance 步骤产出的增强语音；关闭
                        增强时为空。
        chunks: VAD-segmented chunks.
                VAD 切片结果。
        transcripts: ASR results keyed by chunk_id.
                     ASR 结果，按 chunk_id 索引。
        quality: Quality scores keyed by chunk_id.
                 质量评分，按 chunk_id 索引。
        prosody: Prosody / emotion features keyed by chunk_id (manifest v1.1).
                 韵律 / 情绪特征，按 chunk_id 索引（manifest v1.1）。
        manifest_path: Final manifest.jsonl produced by DatasetAgent.
                       DatasetAgent 产出的最终 manifest 路径。
    """

    dataset_root: Path
    source_meta: SourceMeta | None = None
    raw_files: list[Path] = field(default_factory=list)
    vocal_files: list[Path] = field(default_factory=list)
    enhanced_files: list[Path] = field(default_factory=list)
    chunks: list[ChunkInfo] = field(default_factory=list)
    transcripts: dict[str, TranscriptInfo] = field(default_factory=dict)
    quality: dict[str, QualityScore] = field(default_factory=dict)
    prosody: dict[str, ProsodyScore] = field(default_factory=dict)
    manifest_path: Path | None = None

    def ensure_dirs(self) -> None:
        """Create the standard subdirectories under dataset_root.

        在 dataset_root 下创建标准子目录（raw/vocals/enhanced/chunks），
        每个 stage 都假设这些目录存在。enhanced/ 仅在启用 VoiceFixer 时
        实际写入，目录预先创建无副作用。
        """
        for sub in ("raw", "vocals", "enhanced", "chunks"):
            (self.dataset_root / sub).mkdir(parents=True, exist_ok=True)
