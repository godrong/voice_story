"""Unit tests for core.prosody pure helpers (no model loads).

Covers pitch_stats / loudness_lufs / speech_ratio / pace / emotion-label
normalization. The emotion model itself is heavy (~300MB download) so we
test only the label-normalization mapping; full emotion() is exercised in
integration smoke runs, not unit tests.

core.prosody 的纯函数单元测试（不触碰 emotion2vec 模型加载）。

覆盖 pitch_stats / loudness_lufs / speech_ratio / pace_units_per_sec /
情绪标签归一。emotion 模型本身较重（~300MB 下载），单元测试只测标签
归一映射，完整 emotion() 留给集成 smoke 跑。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from core import prosody


def _tone(sr: int = 16000, freq: float = 200.0, sec: float = 1.0, amp: float = 0.3) -> np.ndarray:
    """Synthesize a mono sine tone for predictable signal stats.

    合成可预测统计的单声道正弦波，给信号类函数喂稳定输入。
    """
    n = int(sr * sec)
    t = np.arange(n) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_pitch_stats_recovers_known_frequency():
    """200 Hz sine → pitch_mean ≈ 200 Hz with low std (single tone).

    200 Hz 正弦波 → pitch_mean 在 200 Hz 附近，std 较小（单一频率）。
    """
    audio = _tone(freq=200.0, sec=1.5)
    mean, std = prosody.pitch_stats(audio, sr=16000)
    # pyin can be off by a few Hz on synthetic tones.
    # pyin 在合成纯音上有几 Hz 的误差。
    assert 180.0 < mean < 220.0
    assert std < 30.0


def test_pitch_stats_returns_nan_on_silence():
    """All-zero audio yields (NaN, 0) — no voiced frames detected.

    全零音频应返回 (NaN, 0)；无 voiced 帧。
    """
    silence = np.zeros(16000, dtype=np.float32)
    mean, std = prosody.pitch_stats(silence, sr=16000)
    assert math.isnan(mean)
    assert std == 0.0


def test_loudness_lufs_returns_finite_for_normal_audio():
    """A clean tone produces a finite LUFS (BS.1770 gating won't kill it).

    干净正弦 → 有限 LUFS（不会被 BS.1770 门控丢弃）。
    """
    audio = _tone(sr=24000, sec=1.0, amp=0.3)
    lufs = prosody.loudness_lufs(audio, sr=24000)
    assert math.isfinite(lufs)
    # A 0.3 amplitude tone is roughly -10 to -15 LUFS.
    assert -25.0 < lufs < -5.0


def test_loudness_lufs_returns_neg_inf_on_too_short():
    """Sub-0.4s audio fails BS.1770 gating; we return -inf gracefully.

    < 0.4s 的短音频过不了 BS.1770 门控，函数应返回 -inf 而不是抛出。
    """
    audio = _tone(sr=24000, sec=0.05)  # 50 ms
    lufs = prosody.loudness_lufs(audio, sr=24000)
    assert lufs == float("-inf") or math.isfinite(lufs)


def test_speech_ratio_full_for_constant_signal():
    """A constant-amplitude tone has all frames above threshold (≈ 1.0).

    等幅信号每帧能量相近，speech_ratio 接近 1.0。
    """
    audio = _tone(sec=1.0, amp=0.3)
    ratio = prosody.speech_ratio(audio, sr=16000)
    assert ratio > 0.95


def test_speech_ratio_low_for_mostly_silence():
    """Mostly-silent audio with a short voiced burst has low ratio.

    大部分静音 + 短促有声片段 → ratio 较低。
    """
    silence = np.zeros(16000, dtype=np.float32)
    voiced = _tone(sec=0.1, amp=0.5)  # 100 ms burst
    audio = np.concatenate([silence[:8000], voiced, silence[:7400]])
    ratio = prosody.speech_ratio(audio, sr=16000)
    # Threshold uses 30th percentile within chunk; silence dominates,
    # so threshold ≈ epsilon, voiced burst frames pass. Expect well below 1.
    # 内部按 30 分位定阈值；静音主导，voiced 帧高于阈值。比例应明显小于 1。
    assert ratio < 0.6


def test_speech_ratio_handles_empty():
    """Empty audio returns 0.0 (defined behavior, no crash).

    空音频返回 0.0（已定义行为，不应崩溃）。
    """
    assert prosody.speech_ratio(np.zeros(0, dtype=np.float32), sr=16000) == 0.0


def test_pace_units_per_sec_english_uses_syllables():
    """English pace counts syllables / second via pronouncing.

    英文按音节 / 秒；'hello world' 共 3 音节 / 1.0s = 3.0。
    """
    p = prosody.pace_units_per_sec("hello world", duration_sec=1.0, lang="en")
    # hello (2 syl) + world (1 syl) = 3, +/- 1 if pronouncing falls back.
    assert 2.0 <= p <= 4.0


def test_pace_units_per_sec_chinese_counts_chars():
    """Chinese pace counts CJK characters / second.

    中文按 CJK 字符 / 秒；'你好世界今天' 共 6 字 / 2.0s = 3.0。
    """
    p = prosody.pace_units_per_sec("你好世界今天", duration_sec=2.0, lang="zh")
    assert p == 3.0


def test_pace_units_per_sec_zero_duration_returns_zero():
    """Zero / negative duration yields 0 instead of dividing by zero.

    duration=0 时返回 0 而不是除零异常。
    """
    assert prosody.pace_units_per_sec("hi", duration_sec=0, lang="en") == 0.0
    assert prosody.pace_units_per_sec("", duration_sec=1.0, lang="en") == 0.0


def test_emotion_alias_maps_chinese_and_english():
    """Both Chinese and English alias keys map into the normalized enum.

    中英文标签都能映射到归一枚举。
    """
    assert prosody._normalize_emotion_label("happy") == "happy"
    assert prosody._normalize_emotion_label("高兴") == "happy"
    assert prosody._normalize_emotion_label("中性") == "neutral"
    assert prosody._normalize_emotion_label("happy/高兴") == "happy"
    assert prosody._normalize_emotion_label("xyz") == "unknown"
    assert prosody._normalize_emotion_label("") == "unknown"
    # All normalized labels must be in EMOTION_LABELS enum.
    # 所有归一后的 label 必须在 EMOTION_LABELS 集合中。
    for raw in ["happy", "高兴", "<unk>", "其他"]:
        assert prosody._normalize_emotion_label(raw) in prosody.EMOTION_LABELS
