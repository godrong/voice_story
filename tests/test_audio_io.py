"""Unit tests for core.audio_io.

Verifies probing, standardization (sample-rate / channel changes), and
load/save round-trips. Every test only relies on system ffmpeg + a
locally-synthesized tone WAV — no network, no model downloads.

core.audio_io 单元测试。

只依赖系统 ffmpeg + 本地合成的正弦波 WAV，无需联网或模型下载。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core import audio_io


def test_is_supported_known_extensions():
    """Common extensions are recognised. 常见后缀均被识别。"""
    assert audio_io.is_supported("a.wav")
    assert audio_io.is_supported("a.mp3")
    assert audio_io.is_supported("a.mp4")
    assert audio_io.is_supported("a.M4A")
    assert not audio_io.is_supported("a.txt")


def test_probe_returns_metadata(tone_wav):
    """probe() returns sample rate / channels / duration. probe 返回元信息。"""
    p = tone_wav("probe.wav", sr=16000, seconds=0.3, channels=2)
    info = audio_io.probe(p)
    assert info.sample_rate == 16000
    assert info.channels == 2
    assert 0.25 < info.duration < 0.35


def test_probe_missing_file(tmp_path):
    """probe() raises FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        audio_io.probe(tmp_path / "nope.wav")


def test_to_standard_wav_resamples_and_downmixes(tone_wav, tmp_path):
    """Standardization yields 24kHz / mono regardless of input.

    无论输入采样率/声道数，标准化后都是 24 kHz 单声道。
    """
    src = tone_wav("stereo48k.wav", sr=48000, channels=2, seconds=0.3)
    out_dir = tmp_path / "raw"
    out = audio_io.to_standard_wav(src, out_dir)
    info = audio_io.probe(out)
    assert info.sample_rate == audio_io.TARGET_SR
    assert info.channels == 1


def test_to_standard_wav_idempotent(tone_wav, tmp_path):
    """Second call returns the existing file when overwrite=False.

    第二次调用不重新转码（幂等），返回已有路径。
    """
    src = tone_wav("idem.wav", sr=44100, seconds=0.3)
    out_dir = tmp_path / "raw"
    out1 = audio_io.to_standard_wav(src, out_dir)
    mtime1 = out1.stat().st_mtime_ns
    out2 = audio_io.to_standard_wav(src, out_dir)
    assert out1 == out2
    assert out2.stat().st_mtime_ns == mtime1


def test_load_returns_mono_float32(tone_wav):
    """load() yields a 1-D float32 array even for stereo input.

    load 永远返回 1-D float32 数组（多声道自动平均成单声道）。
    """
    p = tone_wav("stereo.wav", channels=2, seconds=0.2)
    audio, sr = audio_io.load(p)
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert sr == audio_io.TARGET_SR


def test_save_round_trip(tmp_path):
    """save() then load() recovers samples within PCM-16 quantization error.

    save→load 往返；样本一致到 PCM-16 量化误差范围内。
    """
    sr = audio_io.TARGET_SR
    audio = np.sin(2 * np.pi * 440 * np.arange(sr // 4) / sr).astype(np.float32) * 0.5
    out = audio_io.save(tmp_path / "rt.wav", audio, sr=sr)
    loaded, loaded_sr = audio_io.load(out)
    assert loaded_sr == sr
    assert loaded.shape == audio.shape
    assert np.allclose(loaded, audio, atol=1e-3)
