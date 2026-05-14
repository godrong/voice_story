"""Prosody and emotion features for per-chunk style annotation.

Used by DatasetAgent to populate the v1.1 manifest fields that downstream
synthesis (ADR-0010) needs to pick references and generate instruct
prompts. Distinct from `core.eval` which gates "is this chunk training-
worthy"; this module answers "how does this chunk *sound*".

Features computed:

  - pitch_mean_hz / pitch_std_hz : F0 via librosa.pyin. Mean tells you
    overall register (high vs low voice); std is jitter and correlates
    with emotional arousal (calm narration vs animated speech).
  - energy_rms                   : full-chunk RMS energy in [0, 1].
  - loudness_lufs                : integrated loudness per ITU-R BS.1770
    (pyloudnorm). The right metric for postprocess normalization.
  - speech_ratio                 : fraction of 25 ms frames with RMS
    above the chunk's 30th percentile — cheap proxy for "how much of
    the chunk is actually voiced". No new model dependency.
  - pace_units_per_sec           : characters per second for ZH /
    syllables per second for EN. Backed by pypinyin / pronouncing,
    same packages dataset_agent already uses for phoneme coverage.
  - emotion_tag / emotion_confidence : top-1 label + probability from
    emotion2vec_plus_base (FunASR AutoModel). Labels are normalized to
    a fixed enum so downstream selectors can rely on the vocabulary.

All heavy models (emotion2vec) are lazy-loaded and cached process-wide.

逐 chunk 的韵律 + 情绪特征模块，供 DatasetAgent 填充 manifest v1.1 风格字段。
与 core.eval 的"能不能进训练集"不同，本模块回答"听起来什么样"。

特征：
  - pitch_mean_hz / pitch_std_hz：librosa.pyin 估计 F0；均值是音区，
    标准差是抖动（区分平稳叙述 vs 起伏激动）。
  - energy_rms：整段 RMS 能量，范围 [0, 1]。
  - loudness_lufs：按 ITU-R BS.1770 的综合响度（pyloudnorm），
    postprocess 响度归一应该用这个，而不是 RMS。
  - speech_ratio：25ms 帧 RMS 超过 30 分位的占比，廉价代理
    "chunk 里多少比例是真正在说话"。零外部依赖。
  - pace_units_per_sec：中文按字 / 英文按音节，每秒多少。
    复用 dataset_agent 已经依赖的 pypinyin / pronouncing。
  - emotion_tag / emotion_confidence：emotion2vec_plus_base
    （FunASR AutoModel）top-1 label + 概率。标签归一到固定枚举，
    下游 reference selector 可以稳定依赖。

emotion2vec 等重模型惰性加载，进程级缓存。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ---- Emotion label normalization ----------------------------------------
#
# emotion2vec_plus_base returns 9 raw labels in Chinese/English mix; we
# fold them to a 7-class enum that downstream code can rely on. "unknown"
# is intentional: when confidence is very low (<0.3), we don't pretend.
#
# emotion2vec_plus_base 输出 9 类原始标签（中英混排）；折叠成 7 类固定枚举
# 便于下游稳定使用。confidence<0.3 时保留 unknown，不强行判定。

EMOTION_LABELS = (
    "neutral", "happy", "sad", "angry",
    "fearful", "disgust", "surprised", "unknown",
)

_EMOTION_ALIAS = {
    "中性": "neutral", "neutral": "neutral",
    "高兴": "happy", "happy": "happy",
    "悲伤": "sad", "sad": "sad",
    "愤怒": "angry", "angry": "angry",
    "恐惧": "fearful", "fearful": "fearful", "fear": "fearful",
    "厌恶": "disgust", "disgust": "disgust",
    "惊讶": "surprised", "surprised": "surprised", "surprise": "surprised",
    # emotion2vec also emits "其他" / "<unk>" buckets.
    # emotion2vec 也会输出 "其他" / "<unk>" 桶。
    "其他": "unknown", "<unk>": "unknown", "unknown": "unknown",
}

_LOW_EMOTION_CONFIDENCE = 0.30


@dataclass(frozen=True)
class ProsodyFeatures:
    """Per-chunk prosody + emotion bundle for manifest v1.1.

    单 chunk 的韵律 + 情绪特征束，对应 manifest v1.1 的 T2 字段。

    Attributes:
        pitch_mean_hz: Mean F0 in Hz over voiced frames. NaN if unvoiced.
            voiced 帧的 F0 均值，全部 unvoiced 时为 NaN。
        pitch_std_hz: Std-dev of F0. 0 when only one voiced frame.
            F0 标准差；只有 1 个 voiced 帧时为 0。
        energy_rms: Whole-chunk RMS in [0, 1]. 整段 RMS。
        loudness_lufs: Integrated loudness per ITU-R BS.1770.
            综合响度（LUFS）。
        speech_ratio: Fraction of 25 ms frames above per-chunk RMS
            threshold. 25ms 帧中超过阈值的占比。
        pace_units_per_sec: Characters/sec for ZH, syllables/sec for EN.
            中文按字 / 英文按音节的语速。
        emotion_tag: Normalized top-1 emotion label.
            归一后的 top-1 情绪标签。
        emotion_confidence: Top-1 probability in [0, 1].
            top-1 概率。
    """

    pitch_mean_hz: float
    pitch_std_hz: float
    energy_rms: float
    loudness_lufs: float
    speech_ratio: float
    pace_units_per_sec: float
    emotion_tag: str
    emotion_confidence: float


# ---- F0 (pitch) ---------------------------------------------------------


def pitch_stats(audio: np.ndarray, sr: int) -> tuple[float, float]:
    """Compute mean and std of F0 via librosa.pyin.

    用 librosa.pyin 估计 F0，返回均值与标准差（Hz）。

    pyin returns NaN for unvoiced frames; we ignore those before stats.
    Returns (NaN, 0.0) if the entire chunk is unvoiced.

    pyin 在 unvoiced 帧上返回 NaN；统计前先剔除。全段 unvoiced 时返回 (NaN, 0)。

    Args:
        audio: 1-D float32 audio array.
        sr: Sample rate.

    Returns:
        (mean_hz, std_hz). 频率单位 Hz。
    """
    import librosa
    # pyin defaults: fmin=C2 (~65Hz), fmax=C7 (~2093Hz). For speech we
    # tighten fmax to 500Hz to stop bleed from harmonics.
    # pyin 默认 fmin=C2 (~65Hz), fmax=C7 (~2093Hz)；语音场景把 fmax 收到 500Hz，
    # 避免谐波串音。
    try:
        f0, _voiced_flag, _voiced_prob = librosa.pyin(
            audio.astype(np.float32),
            fmin=65.0, fmax=500.0, sr=sr,
        )
    except Exception as e:
        logger.warning("pitch_stats: pyin failed (%s); returning NaN", e)
        return float("nan"), 0.0
    voiced = f0[~np.isnan(f0)]
    if voiced.size == 0:
        return float("nan"), 0.0
    return float(voiced.mean()), float(voiced.std())


# ---- Loudness -----------------------------------------------------------


def loudness_lufs(audio: np.ndarray, sr: int) -> float:
    """Integrated loudness in LUFS (ITU-R BS.1770) via pyloudnorm.

    用 pyloudnorm 计算综合响度（LUFS, ITU-R BS.1770）。

    Short chunks (<0.4s) can fail BS.1770 gating; we catch and return
    -inf so the manifest can still be written.

    短于 0.4s 的 chunk 可能过不了 BS.1770 的门控；catch 后返回 -inf
    保证 manifest 仍能写入。

    Args:
        audio: 1-D float32 audio in [-1, 1].
        sr: Sample rate.

    Returns:
        Integrated loudness in LUFS, or -inf on failure.
        综合响度（LUFS），失败时返回 -inf。
    """
    import pyloudnorm as pyln
    try:
        meter = pyln.Meter(sr)
        return float(meter.integrated_loudness(audio.astype(np.float32)))
    except Exception as e:
        logger.warning("loudness_lufs: pyloudnorm failed (%s); returning -inf", e)
        return float("-inf")


# ---- Speech ratio (cheap, no external model) ----------------------------


def speech_ratio(audio: np.ndarray, sr: int, *, frame_ms: float = 25.0) -> float:
    """Estimate the fraction of frames that are "speaking" within a chunk.

    粗估 chunk 内"在说话"帧数的占比，不引入新模型依赖。

    Approach: split into 25 ms hop frames, compute per-frame RMS, then
    count frames whose RMS >= max(eps, 30th percentile). This captures
    internal pauses without needing VAD. It's relative-per-chunk, so
    cross-chunk comparisons are valid only within similar recording
    conditions.

    做法：25ms 帧切片 → 每帧 RMS → 数 RMS≥max(eps, 30 分位) 的帧占比。
    捕捉 chunk 内部静音 / 停顿，无需 VAD。相对值（每 chunk 各自门控），
    跨 chunk 比较仅在相似录音条件下才有意义。

    Args:
        audio: 1-D float32 audio.
        sr: Sample rate.
        frame_ms: Frame length in milliseconds.

    Returns:
        Ratio in [0, 1]. 静音很多 → 接近 0；满帧讲话 → 接近 1。
    """
    if audio.size == 0:
        return 0.0
    frame_len = max(1, int(sr * frame_ms / 1000.0))
    n_frames = audio.size // frame_len
    if n_frames == 0:
        return 1.0  # short chunk; treat as fully voiced
    trimmed = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt((trimmed.astype(np.float32) ** 2).mean(axis=1) + 1e-12)
    threshold = max(1e-4, float(np.percentile(rms, 30)))
    return float((rms >= threshold).mean())


# ---- Pace ---------------------------------------------------------------


def pace_units_per_sec(text: str, duration_sec: float, lang: str) -> float:
    """Speech pace in characters/sec (ZH) or syllables/sec (EN).

    中文按字 / 英文按音节计算每秒语速。

    Chinese: count CJK ideographs in `text`. English: sum the syllable
    count of each word via pronouncing; fall back to a rough heuristic
    (max(1, len(word)//3)) if pronouncing has no entry for the word.

    中文：数 `text` 中的 CJK 表意文字。
    英文：用 pronouncing 拿每个单词的音节数；查不到时退化成 max(1, len(word)//3)。

    Args:
        text: ASR transcript.
        duration_sec: Chunk duration in seconds (must be > 0).
        lang: ISO 639-1 code; non-zh treated as English.

    Returns:
        Units per second. 0 if text empty or duration<=0.
    """
    if not text or duration_sec <= 0:
        return 0.0
    if lang == "zh":
        # CJK Unified Ideographs basic block; covers ~99% of modern usage.
        # CJK 基础区，覆盖现代用法的绝大多数。
        units = sum(1 for ch in text if "一" <= ch <= "鿿")
        return units / duration_sec
    try:
        import pronouncing
        units = 0
        for word in re.findall(r"[A-Za-z']+", text.lower()):
            phones = pronouncing.phones_for_word(word)
            if phones:
                # Stressed vowels carry the digit suffix; count them.
                # 带重音标记的元音音素就是音节标记。
                units += sum(1 for ph in phones[0].split() if any(c.isdigit() for c in ph))
            else:
                units += max(1, len(word) // 3)
        return units / duration_sec
    except ImportError:
        # Fall back to word count if pronouncing is unavailable.
        # pronouncing 不可用时退化到词数 / 秒。
        return len(text.split()) / duration_sec


# ---- Emotion (emotion2vec via FunASR) -----------------------------------


@lru_cache(maxsize=1)
def _emotion_model():
    """Lazy-load emotion2vec_plus_base via FunASR's AutoModel.

    惰性加载 emotion2vec_plus_base（FunASR AutoModel），进程级缓存。

    First call triggers a ~300MB ModelScope download.
    首次调用会触发 ~300MB 的 ModelScope 下载。
    """
    from funasr import AutoModel
    logger.info("emotion: loading iic/emotion2vec_plus_base via FunASR")
    return AutoModel(model="iic/emotion2vec_plus_base", model_revision=None, disable_update=True)


def _normalize_emotion_label(raw: str) -> str:
    """Map a raw emotion2vec label to our normalized enum.

    把 emotion2vec 的原始标签映射到归一枚举（EMOTION_LABELS 之一）。
    """
    if not raw:
        return "unknown"
    key = raw.strip().lower()
    # emotion2vec_plus_base sometimes returns "中文/English" composite labels
    # like "高兴/happy"; try both halves.
    # 复合标签如 "高兴/happy"，两半都试一遍。
    for part in re.split(r"[/|,，]", key):
        part = part.strip()
        if part in _EMOTION_ALIAS:
            return _EMOTION_ALIAS[part]
    return "unknown"


def emotion(audio: np.ndarray, sr: int) -> tuple[str, float]:
    """Top-1 emotion tag + probability for the given chunk audio.

    返回 chunk 音频的 top-1 情绪标签 + 概率。

    Args:
        audio: 1-D float32 audio.
        sr: Sample rate. Required by FunASR; resampling is handled inside.

    Returns:
        (tag, confidence). Tag is one of EMOTION_LABELS. When confidence
        is below 0.30 the tag is forced to "unknown" so downstream
        selectors don't trust a coin flip.
        (标签, 置信度)。confidence<0.30 时强制 "unknown"，避免下游误用。
    """
    model = _emotion_model()
    # FunASR's emotion2vec wants a 16k float32 array; resample if needed.
    # FunASR 的 emotion2vec 需要 16k 单声道 float32；必要时降采样。
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
    try:
        res = model.generate(audio.astype(np.float32), granularity="utterance", extract_embedding=False)
    except Exception as e:
        logger.warning("emotion: inference failed (%s); returning unknown", e)
        return "unknown", 0.0
    # FunASR returns [{"labels": [...], "scores": [...]}].
    # FunASR 返回 [{"labels": [...], "scores": [...]}].
    if not res or "labels" not in res[0]:
        return "unknown", 0.0
    labels: list[str] = res[0]["labels"]
    scores: list[float] = list(res[0]["scores"])
    if not labels or not scores:
        return "unknown", 0.0
    top_idx = int(np.argmax(scores))
    tag = _normalize_emotion_label(labels[top_idx])
    conf = float(scores[top_idx])
    if conf < _LOW_EMOTION_CONFIDENCE:
        tag = "unknown"
    return tag, conf


# ---- One-shot façade ----------------------------------------------------


def score_prosody(
    wav_path: Path | str, *, text: str, lang: str,
) -> ProsodyFeatures:
    """Compute all v1.1 prosody features for one chunk in one call.

    一站式计算 manifest v1.1 的全部韵律 + 情绪特征。

    Args:
        wav_path: Standardized chunk WAV (24 kHz mono).
        text: ASR transcript (for pace).
        lang: ISO 639-1 language (for pace word/syllable rule).

    Returns:
        ProsodyFeatures bundle. 全部 T2 字段。
    """
    from . import audio_io
    audio, sr = audio_io.load(wav_path)
    pmean, pstd = pitch_stats(audio, sr)
    rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)) + 1e-10)
    loud = loudness_lufs(audio, sr)
    sratio = speech_ratio(audio, sr)
    duration = audio.size / sr
    pace = pace_units_per_sec(text, duration, lang)
    tag, conf = emotion(audio, sr)
    return ProsodyFeatures(
        pitch_mean_hz=pmean,
        pitch_std_hz=pstd,
        energy_rms=rms,
        loudness_lufs=loud,
        speech_ratio=sratio,
        pace_units_per_sec=pace,
        emotion_tag=tag,
        emotion_confidence=conf,
    )
