"""Shared pytest fixtures for the voice-story test suite.

Mostly utilities for synthesizing tiny WAV files on demand so we can
unit-test the audio pipeline without bundling binary fixtures in git.

共享 pytest fixture：按需合成微型 WAV 文件，避免在 git 里塞二进制 fixture。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from core import audio_io


@pytest.fixture
def tone_wav(tmp_path: Path):
    """Factory: write a sine-tone WAV at the requested params and return its path.

    Factory fixture：按参数合成正弦波 WAV 并返回路径。
    用闭包让单个测试在同一个 tmp_path 下生成多个文件。
    """

    def _make(
        name: str = "tone.wav",
        *,
        freq: float = 440.0,
        sr: int = audio_io.TARGET_SR,
        seconds: float = 0.5,
        channels: int = 1,
        amplitude: float = 0.5,
    ) -> Path:
        n = int(sr * seconds)
        t = np.arange(n) / sr
        signal = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        if channels > 1:
            signal = np.stack([signal] * channels, axis=1)
        path = tmp_path / name
        sf.write(str(path), signal, sr, subtype="PCM_16")
        return path

    return _make
