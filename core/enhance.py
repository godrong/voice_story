"""Speech enhancement: clean residual noise / restore bandwidth on a vocal WAV.

Stage 2.5 of the M1 front-end (per ADR-0012). Runs *after* Demucs has stripped
musical BGM and *before* VAD chunks the file, so downstream DNSMOS scoring,
ASR transcription, and ref-audio selection all see a clean signal.

Why this exists: Demucs (`htdemucs`) is trained on song + instrument mixtures
and is excellent at removing musical accompaniment, but leaves behind broadband
noise, mic hiss, room reverb, and minor SFX residue on streamer-style content
(e.g. gaming live audio). On the biggvoice probe, post-Demucs DNSMOS-OVR sat
at 0.99~1.36 — far below the 3.0 manifest gate — and the bak (background) sub-
score was 1.52, indicating audible residual noise. VoiceFixer mode-0 lifts the
same chunk to OVR=2.66 / bak=3.06, restoring downstream scoring usefulness.

Model: `voicefixer` (44.1 kHz output) → resampled to the project TARGET_SR
(24 kHz) before being saved, so the rest of the pipeline sees one sample rate.

语音增强模块（管线 2.5 阶段，配 ADR-0012）。

为什么需要：Demucs 擅长去音乐性 BGM，但对游戏音效 / 麦克风噪声 / 房间混响
等"非音乐性脏东西"无能为力。biggvoice 探针显示 Demucs 输出的 DNSMOS-OVR
仍只有 1.0 左右，导致下游评分失真、ASR 偶发错字、ref 仍需手洗。本模块在
Demucs 之后加一层 VoiceFixer，把残余噪声压下去，让下游链路看到真正干净
的人声。

模型：`voicefixer` (mode 0 = general restoration)，输出固定 44.1 kHz，
保存前重采样回 24 kHz 与项目 TARGET_SR 对齐。
"""

from __future__ import annotations

import logging
from pathlib import Path

from . import audio_io

logger = logging.getLogger(__name__)

DEFAULT_MODE = 0  # 0=general, 1=stronger denoise, 2=de-reverb


class VoiceEnhancer:
    """Wraps VoiceFixer for cascade after Demucs.

    封装 VoiceFixer：Demucs 之后的二级清洗。

    The model is loaded lazily on first call so importing this module is
    cheap. After the first call it stays resident for batch reuse.

    模型在首次调用时惰性加载（保证 import 成本低），加载后常驻供批量复用。

    Args:
        mode: VoiceFixer restoration mode. 0=general (recommended),
              1=stronger noise suppression, 2=mainly de-reverb.
              VoiceFixer 修复模式：0=通用（默认推荐），1=强降噪，2=去混响。
        cuda: Whether to run on CUDA. Defaults to False (CPU); VoiceFixer's
              checkpointing path is brittle on MPS so we keep CPU for stability.
              是否用 CUDA。默认 False（CPU）。VoiceFixer 在 MPS 上的 checkpoint
              路径不稳定，CPU 更可靠；Linux+GPU 可以打开。
    """

    def __init__(self, mode: int = DEFAULT_MODE, cuda: bool = False) -> None:
        self.mode = mode
        self.cuda = cuda
        self._model = None  # lazy

    def _ensure_model(self):
        """Lazily import and instantiate VoiceFixer.

        惰性加载 VoiceFixer；首次调用会下载权重（~500 MB）。
        """
        if self._model is not None:
            return self._model
        # Lazy import: voicefixer drags in torchlibrosa / a custom progressbar
        # and adds ~2s of import time. We defer until actually used.
        # 延迟导入：voicefixer 副作用多，推到使用时再加载。
        from voicefixer import VoiceFixer

        logger.info(
            "VoiceEnhancer: loading VoiceFixer (mode=%d, cuda=%s)",
            self.mode, self.cuda,
        )
        self._model = VoiceFixer()
        return self._model

    def enhance(self, input_wav: Path | str, out_dir: Path | str) -> Path:
        """Run VoiceFixer and write a TARGET_SR mono WAV.

        对输入 WAV 跑 VoiceFixer，输出统一 24 kHz / mono / 16-bit。
        幂等：若输出已存在直接返回，VoiceFixer 是慢操作，重复运行浪费。

        Args:
            input_wav: Demucs-separated vocal WAV (TARGET_SR mono).
                       Demucs 分离后的人声 WAV（24 kHz mono）。
            out_dir: Output directory; file is written as
                     `<input_stem>.wav` inside.
                     输出目录，文件名为 <input_stem>.wav。

        Returns:
            Path of the enhanced WAV (24 kHz / 16-bit / mono).
            增强后的 WAV 路径（与项目 TARGET_SR 一致）。
        """
        src = Path(input_wav)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / src.name
        if out_path.exists():
            logger.debug("VoiceEnhancer: %s already exists, skipping", out_path.name)
            return out_path

        model = self._ensure_model()

        # VoiceFixer reads the file directly and writes to a temp WAV at 44.1
        # kHz. We then load that temp, resample to TARGET_SR, and save via
        # audio_io.save so the file matches the rest of the pipeline.
        # VoiceFixer 直接读文件、自己写一个 44.1 kHz 临时 WAV；我们读回来
        # 重采样到 TARGET_SR，再用 audio_io.save 落盘以保持管线一致性。
        tmp_path = out_path.with_suffix(".vf_raw.wav")
        try:
            model.restore(
                input=str(src), output=str(tmp_path),
                cuda=self.cuda, mode=self.mode,
            )
            audio, sr = audio_io.load(tmp_path)
            if sr != audio_io.TARGET_SR:
                import librosa
                audio = librosa.resample(
                    audio, orig_sr=sr, target_sr=audio_io.TARGET_SR,
                )
            audio_io.save(out_path, audio, sr=audio_io.TARGET_SR)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        logger.info("VoiceEnhancer: %s -> %s (mode=%d)", src.name, out_path, self.mode)
        return out_path
