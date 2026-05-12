"""Vocal separation: strip BGM / SFX from a standardized WAV using Demucs v4.

Why this is mandatory (per ADR-0008): main-streamer source audio almost
always carries music or sound effects. Training a voice cloner on raw
mixed audio "stains" the timbre with the BGM. Demucs is run on every file
even when the source declares clean audio (e.g. Trump speeches), so the
downstream pipeline behaves identically across sources.

Model choice: `htdemucs` (single non-bag model). The bag variant
`htdemucs_ft` gives ~0.3 SDR more but is ~4× slower; we keep speed for
iteration. Mac uses MPS, otherwise CPU. CUDA is auto-picked on Linux.

人声分离模块（Demucs v4 封装）。

为什么强制开（ADR-0008）：主播原始音频几乎都带 BGM / 音效，
直接拿来训练会"染色"音色。即使是干净音频（如演讲），也跑一遍
Demucs，保证 pipeline 行为一致。

模型选择：默认用 `htdemucs`（单模型）而非 `htdemucs_ft`（4-bag）。
后者 SDR 高 ~0.3 但速度慢 ~4 倍，对开发迭代不划算。
设备策略：Mac 走 MPS、Linux 自动选 CUDA、否则 CPU。
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from . import audio_io

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "htdemucs"


def _pick_device() -> str:
    """Choose the fastest available torch device.

    选择当前可用的最快设备：CUDA > MPS > CPU。
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Separator:
    """Wraps demucs to extract the vocal stem from any standardized WAV.

    封装 Demucs：从标准化 WAV 中提取人声 stem。

    The model is loaded lazily on first call so importing this module is
    cheap. After the first call the model stays resident for batch reuse.

    模型在首次调用时惰性加载（保证 import 成本低），加载后常驻内存
    供后续批量复用。

    Args:
        model_name: Demucs pretrained name. Defaults to "htdemucs".
                    Demucs 预训练模型名，默认 htdemucs。
        device: torch device ("cuda" / "mps" / "cpu"). Auto-pick if None.
                设备名，None 则自动挑最快的。
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device or _pick_device()
        self._model = None  # lazy

    def _ensure_model(self):
        """Lazily download and load the Demucs model.

        惰性加载 Demucs 模型；首次调用会自动下载权重（~80MB）。
        """
        if self._model is not None:
            return self._model
        # Lazy import: demucs pulls in torch / hydra / openunmix and adds
        # ~3s import time, so we defer it until actually separating.
        # 延迟导入：demucs 会引入大量副作用，这里推到使用时才 import。
        from demucs.pretrained import get_model
        logger.info("Separator: loading %s on %s", self.model_name, self.device)
        self._model = get_model(self.model_name)
        self._model.to(self.device)
        self._model.eval()
        return self._model

    def separate(self, input_wav: Path | str, out_dir: Path | str) -> Path:
        """Run Demucs and write the vocal stem WAV.

        对输入 WAV 跑 Demucs 分离，把 vocal stem 落到 out_dir。
        幂等：若输出已存在直接返回，避免重复计算（Demucs 是慢操作）。

        Args:
            input_wav: Standardized 24 kHz mono WAV path.
                       已标准化的 24 kHz 单声道 WAV 路径。
            out_dir: Output directory; the file is written as
                     `<input_stem>.wav` inside.
                     输出目录，文件名为 <input_stem>.wav。

        Returns:
            Path of the vocal stem WAV (24 kHz / 16-bit / mono).
            人声 stem 的文件路径（仍是 24 kHz / 单声道 / 16-bit）。
        """
        src = Path(input_wav)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / src.name
        if out_path.exists():
            logger.debug("Separator: %s already exists, skipping", out_path.name)
            return out_path

        from demucs.apply import apply_model
        model = self._ensure_model()

        wav, sr = audio_io.load(src)
        # Demucs expects (channels, samples) and its training sample rate.
        # We feed mono by duplicating to the channel count it expects.
        # Demucs 需要 (channels, samples) 格式且固定采样率，这里把单声道
        # 复制成 Demucs 期望的通道数后再喂入。
        if sr != model.samplerate:
            import torchaudio.functional as F
            wav_t = torch.from_numpy(wav).unsqueeze(0)
            wav_t = F.resample(wav_t, sr, model.samplerate)
        else:
            wav_t = torch.from_numpy(wav).unsqueeze(0)
        wav_t = wav_t.repeat(model.audio_channels, 1).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            sources = apply_model(model, wav_t, split=True, overlap=0.25)
        # `sources` shape: (batch=1, stems, channels, samples). We want vocals.
        # sources 形状：(batch=1, stems, channels, samples)，vocals 在指定 stem 索引。
        stem_idx = model.sources.index("vocals")
        vocals = sources[0, stem_idx].mean(dim=0).cpu().numpy()

        if model.samplerate != audio_io.TARGET_SR:
            import librosa
            vocals = librosa.resample(vocals, orig_sr=model.samplerate, target_sr=audio_io.TARGET_SR)

        audio_io.save(out_path, vocals, sr=audio_io.TARGET_SR)
        logger.info("Separator: %s -> %s", src.name, out_path)
        return out_path
