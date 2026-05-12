"""Audio I/O: probe arbitrary inputs and standardize to 24 kHz / 16-bit / mono WAV.

This module is the single entrypoint for all downstream stages to read or
write audio. By forcing every file through `to_standard_wav` first, the rest
of the pipeline (Demucs / VAD / ASR / TTS) can assume a fixed sample rate
and channel layout, simplifying every interface in the pipeline.

通用音频 I/O 模块（封装 ffmpeg 与 soundfile）。

数据管线的入口：所有下游模块（Demucs / VAD / ASR / TTS）都从这里读音频。
统一规则——任何输入先经 `to_standard_wav` 转成 24 kHz / 16-bit / mono WAV，
之后所有阶段就可以假设固定采样率与单声道，省掉每个模块自己写重采样逻辑。
24 kHz 的选择是为了对齐 CosyVoice 2 的训练采样率。
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SR = 24000
TARGET_CHANNELS = 1
TARGET_SUBTYPE = "PCM_16"

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".opus"}
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".flv", ".ts"}
SUPPORTED_EXTS = AUDIO_EXTS | VIDEO_EXTS


@dataclass(frozen=True)
class AudioInfo:
    """Metadata returned by `probe()`. 由 probe() 返回的音频元信息。

    Attributes:
        path: Source file path. 原始文件路径。
        duration: Length in seconds. 时长（秒）。
        sample_rate: Original sample rate in Hz. 原始采样率（Hz）。
        channels: Channel count (1=mono, 2=stereo). 声道数。
        codec: ffprobe codec name (e.g. "aac", "pcm_s16le"). 编解码器名。
        format_name: Container format (e.g. "mp4", "wav"). 容器格式名。
    """

    path: Path
    duration: float
    sample_rate: int
    channels: int
    codec: str
    format_name: str


class FFmpegMissingError(RuntimeError):
    """Raised when system ffmpeg/ffprobe is not on PATH.

    系统未安装 ffmpeg/ffprobe 时抛出。
    """


def _require_ffmpeg() -> None:
    """Fail fast if ffmpeg/ffprobe are missing.

    检查 ffmpeg/ffprobe 是否在 PATH 中，缺失立即报错（fail-fast）。
    """
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise FFmpegMissingError(
            "ffmpeg/ffprobe not found on PATH. Install via `brew install ffmpeg` (macOS)."
        )


def probe(path: Path | str) -> AudioInfo:
    """Return basic stream info for any audio/video file via ffprobe.

    用 ffprobe 探测任意音视频文件的元信息。视频文件取第一条音轨。

    Args:
        path: Path to an audio or video file.
              输入文件路径（音频或视频均可）。

    Returns:
        AudioInfo with duration / sample_rate / channels / codec / format.
        AudioInfo 实例，包含时长、采样率、声道数、编解码器、容器格式。

    Raises:
        FileNotFoundError: If `path` does not exist. 文件不存在。
        ValueError: If the file has no audio stream. 文件中没有音轨。
        FFmpegMissingError: If ffprobe is not installed. 未安装 ffprobe。
    """
    _require_ffmpeg()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    proc = subprocess.run(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", "-select_streams", "a:0",
            str(p),
        ],
        check=True, capture_output=True, text=True,
    )
    data = json.loads(proc.stdout)
    streams = data.get("streams") or []
    if not streams:
        raise ValueError(f"No audio stream found in {p}")
    s = streams[0]
    fmt = data.get("format", {})
    return AudioInfo(
        path=p,
        duration=float(fmt.get("duration", s.get("duration", 0.0))),
        sample_rate=int(s.get("sample_rate", 0)),
        channels=int(s.get("channels", 0)),
        codec=str(s.get("codec_name", "")),
        format_name=str(fmt.get("format_name", "")),
    )


def to_standard_wav(
    input_path: Path | str,
    out_dir: Path | str,
    *,
    out_name: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Transcode any input to 24 kHz / 16-bit / mono WAV via ffmpeg.

    把任意输入文件（mp3/m4a/flac/wav/视频等）转码为标准 WAV：
    24 kHz 采样率 / 16-bit PCM / 单声道。视频会自动剥离音轨。
    幂等：若输出已存在且 overwrite=False，直接复用，避免重复转码。

    Args:
        input_path: Source audio/video file. 任意 ffmpeg 能读的输入。
        out_dir: Directory to write the standardized WAV.
                 输出目录（不存在自动创建）。
        out_name: Output filename. Defaults to `<input_stem>.wav`.
                  输出文件名，默认是输入文件 stem + ".wav"。
        overwrite: If True, re-transcode even when the output exists.
                   是否覆盖已有输出。

    Returns:
        Path to the standardized WAV.
        输出文件路径。

    Raises:
        FFmpegMissingError: If ffmpeg is not installed.
        RuntimeError: If ffmpeg returns a non-zero exit code.
                      ffmpeg 转码失败时抛出，包含 stderr 内容。
    """
    _require_ffmpeg()
    src = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = out_name or (src.stem + ".wav")
    out_path = out_dir / name
    if out_path.exists() and not overwrite:
        return out_path

    cmd = [
        "ffmpeg", "-y" if overwrite else "-n", "-loglevel", "error",
        "-i", str(src),
        "-vn",
        "-ac", str(TARGET_CHANNELS),
        "-ar", str(TARGET_SR),
        "-c:a", "pcm_s16le",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {src}:\n{result.stderr}")
    return out_path


def load(path: Path | str) -> tuple[np.ndarray, int]:
    """Load a WAV (or anything soundfile supports) as float32 mono.

    用 soundfile 读音频，返回 float32 单声道波形和采样率。
    多声道音频自动平均成单声道（与下游一致性）。

    Args:
        path: Audio file path. 音频路径。

    Returns:
        Tuple of (samples, sample_rate).
        (samples, sample_rate) 二元组。samples 是 1-D float32 numpy 数组。
    """
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio, sr


def save(path: Path | str, audio: np.ndarray, sr: int = TARGET_SR) -> Path:
    """Save audio array as 16-bit PCM WAV.

    把 numpy 波形数组保存为 16-bit PCM WAV。
    父目录不存在会自动创建。

    Args:
        path: Output file path. 输出路径。
        audio: 1-D float32 array in [-1, 1]. 单声道 float32 波形。
        sr: Sample rate. Defaults to TARGET_SR (24 kHz).
            采样率，默认 24 kHz。

    Returns:
        The output path.
        实际写入的路径。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(p), audio, sr, subtype=TARGET_SUBTYPE)
    return p


def is_supported(path: Path | str) -> bool:
    """Check whether the file extension is in our supported set.

    判断文件后缀是否在支持的格式集合内（音频或视频均可）。

    Args:
        path: File path. 文件路径。

    Returns:
        True if the extension is supported. 是否支持。
    """
    return Path(path).suffix.lower() in SUPPORTED_EXTS
