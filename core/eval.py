"""Audio-quality scoring used by the data pipeline (M1) and later by training.

Two metrics for the data-pipeline stage:

  - WADA-SNR: a classic blind SNR estimator that doesn't need a reference
    clean signal. Cheap, stable, decent for filtering obvious noise.
  - DNSMOS: Microsoft's neural MOS predictor (ONNX, ~30MB). Returns three
    scores: overall (OVR), signal quality (SIG), background noise (BAK).
    Used as the primary "is this chunk good enough to train on" gate.

Training-stage metrics (speaker_sim / WER / mos_pred) live in this module
too but are added in M3.

数据管线 (M1) 与后续训练阶段共用的音频质量评分模块。

M1 阶段两种指标：
  - WADA-SNR：经典盲 SNR 估计器，不需要干净参考信号。
    便宜稳定，适合过滤明显噪音。
  - DNSMOS：微软出品的神经 MOS 预测器（ONNX，~30MB）。
    返回三个分数：OVR（总分）/ SIG（信号质量）/ BAK（背景噪音）。
    M1 阶段作为"这个 chunk 能不能进训练集"的主闸门。

训练阶段指标（speaker_sim / WER / mos_pred）在 M3 时加入本模块。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

from . import audio_io

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DnsmosScores:
    """DNSMOS predicted MOS scores.

    DNSMOS 模型预测的三类 MOS 分数。

    Attributes:
        ovr: Overall MOS in [1, 5]. 总体 MOS。
        sig: Signal quality MOS. 信号质量 MOS。
        bak: Background quality MOS. 背景质量 MOS。
    """

    ovr: float
    sig: float
    bak: float


def wada_snr(audio: np.ndarray) -> float:
    """Estimate SNR (dB) from a single-channel signal via the WADA method.

    用 WADA 方法盲估单声道信号的 SNR（dB）。
    参考实现：Kim & Stern, 2008. 不需要干净参考信号。

    Args:
        audio: 1-D float32 audio array, ideally normalized to [-1, 1].
               单声道 float32 波形数组。

    Returns:
        Estimated SNR in dB. Clamped to a sane [-20, 100] range.
        估计的 SNR（dB），裁剪到 [-20, 100] 区间。
    """
    if audio.size == 0:
        return -20.0
    abs_audio = np.abs(audio).astype(np.float64) + 1e-10
    g_alpha = abs_audio.mean() / np.sqrt((abs_audio ** 2).mean())
    # WADA's lookup table approximation; we use the simple closed-form
    # derived from the gamma assumption (good enough for filtering).
    # WADA 的查表近似；这里用 gamma 假设下的闭式近似，适合做过滤。
    if g_alpha >= 0.999:
        return 100.0
    snr = 20.0 * np.log10((g_alpha / (1.0 - g_alpha)) + 1e-10)
    return float(np.clip(snr, -20.0, 100.0))


def detect_clipping(audio: np.ndarray, threshold: float = 0.99) -> bool:
    """Return True if the signal contains samples above `threshold` magnitude.

    判断信号是否削波：检查是否存在幅值超过 threshold 的采样。

    Args:
        audio: 1-D float32 audio in [-1, 1].
        threshold: Magnitude threshold; defaults to 0.99 (3 samples == clipped).

    Returns:
        True if any sample magnitude exceeds threshold.
    """
    return bool(np.any(np.abs(audio) >= threshold))


# ----- DNSMOS -------------------------------------------------------------
#
# We bundle the official Microsoft DNSMOS P.808 ONNX model lazily — it is
# small (~30MB) and freely downloadable. Cached under `models/dnsmos/`.
#
# DNSMOS 部分：使用微软官方的 P.808 ONNX 模型，体积小（~30MB），
# 首次使用时自动下载到 models/dnsmos/。

_DNSMOS_URL = "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
_DNSMOS_FILENAME = "sig_bak_ovr.onnx"
_DNSMOS_INPUT_LENGTH = 9.01  # seconds; model expects ~9s windows


def _dnsmos_model_path() -> Path:
    """Return the local cache path for the DNSMOS ONNX model.

    返回 DNSMOS ONNX 模型的本地缓存路径。
    """
    cache_root = Path(os.environ.get("VOICE_STORY_MODEL_DIR", "models"))
    return cache_root / "dnsmos" / _DNSMOS_FILENAME


def _ensure_dnsmos() -> Path:
    """Download the DNSMOS ONNX weights on first use.

    首次使用时下载 DNSMOS 权重；返回本地路径。
    """
    p = _dnsmos_model_path()
    if p.exists():
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    logger.info("DNSMOS: downloading %s", _DNSMOS_URL)
    urlretrieve(_DNSMOS_URL, p)
    return p


@lru_cache(maxsize=1)
def _dnsmos_session():
    """Lazy ONNX session for DNSMOS, cached process-wide.

    进程级缓存的 DNSMOS ONNX 推理 session。
    """
    import onnxruntime as ort
    model_path = _ensure_dnsmos()
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model_path), sess_options=sess_options)


def _audio_melspec(audio: np.ndarray, sr: int) -> np.ndarray:
    """Mel-spectrogram preprocessing matching DNSMOS's training pipeline.

    DNSMOS 训练管线对应的 mel-spectrogram 预处理步骤。
    """
    import librosa
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    target_len = int(_DNSMOS_INPUT_LENGTH * sr)
    if audio.shape[0] < target_len:
        audio = np.pad(audio, (0, target_len - audio.shape[0]))
    else:
        audio = audio[:target_len]
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=320, hop_length=160, n_mels=120,
    )
    mel = (librosa.power_to_db(mel, ref=np.max) + 40) / 40
    return mel.astype(np.float32)


def dnsmos(audio: np.ndarray, sr: int = audio_io.TARGET_SR) -> DnsmosScores:
    """Predict MOS-OVR / SIG / BAK with the bundled DNSMOS ONNX model.

    用 DNSMOS ONNX 模型预测 OVR / SIG / BAK 三个 MOS 分数。

    Args:
        audio: 1-D float32 audio array.
        sr: Sample rate (any; DNSMOS resamples to 16k internally).

    Returns:
        DnsmosScores with ovr / sig / bak in [1, 5].
        DnsmosScores 三个字段都在 [1, 5] 范围。
    """
    sess = _dnsmos_session()
    feat = _audio_melspec(audio, sr)
    feat = feat[np.newaxis, np.newaxis, :, :]  # (1, 1, mel, frames)
    inputs = {sess.get_inputs()[0].name: feat}
    outputs = sess.run(None, inputs)[0][0]
    sig, bak, ovr = float(outputs[0]), float(outputs[1]), float(outputs[2])
    return DnsmosScores(ovr=ovr, sig=sig, bak=bak)


def score_chunk(wav_path: Path | str) -> tuple[float, DnsmosScores, bool]:
    """One-shot quality scoring used by DatasetAgent.

    DatasetAgent 调用的一站式打分接口。

    Args:
        wav_path: Standardized chunk WAV.

    Returns:
        (snr_db, dnsmos_scores, clipped).
        (信噪比, DNSMOS 三分, 是否削波) 三元组。
    """
    audio, sr = audio_io.load(wav_path)
    snr = wada_snr(audio)
    mos = dnsmos(audio, sr)
    clipped = detect_clipping(audio)
    return snr, mos, clipped
