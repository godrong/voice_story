"""Objective evaluation metrics for TTS synthesis output.

Standard-practice four-axis eval:

  - MOS-NISQA: primary naturalness predictor in [1, 5]. NISQA_DIM also
    returns 4 sub-dimensions (noi/dis/col/loud) — we store them all.
    DNSMOS-P808 was the v1 stand-in but proved to be misleading for TTS
    output (it gave higher scores to instruct-mode synthesis that user
    listening tests rated as worse). DNSMOS scores are still recorded for
    reference but are not the primary MOS axis.
    UTMOS22 (VoiceMOS-Challenge 2022 winner) would be the gold standard,
    but its only pip install path requires fairseq, which doesn't build
    cleanly on modern Python.
  - WER / CER (faster-whisper / FunASR via core.asr): intelligibility,
    [0, 1], lower = better.
  - SECS (microsoft/wavlm-base-plus-sv): speaker similarity vs the ref
    audio in [-1, 1], higher = more like the same speaker.
  - F0 RMSE (librosa.pyin): prosody fidelity in Hz over voiced frames,
    lower = better.

This module is intentionally separate from `core/eval.py`. The latter
scores raw dataset chunks (M1 filtering gate); this one scores synthesis
output. Conflating them would mean having two different "DNSMOS" calls
with different semantics in the same file.

TTS 合成输出的客观评测指标。

四轴标准评测：
  - MOS-NISQA：主自然度预测，[1, 5]，越高越好。NISQA_DIM 还附带
    4 个子维度（noi/dis/col/loud），一并存。DNSMOS-P808 是 v1 的替身，
    但在 TTS 输出上被证实有问题（给主观分更差的 instruct 模式打了更
    高的 P808 分）。DNSMOS 仍记录为辅助分，不作为主自然度轴。
    最理想是 UTMOS22（VoiceMOS Challenge 2022 冠军），但 pip 路径
    依赖 fairseq，在现代 Python 上无法构建。
  - WER / CER（faster-whisper / FunASR，复用 core.asr）：可懂度，
    [0, 1]，越小越好。
  - SECS（microsoft/wavlm-base-plus-sv）：与 ref 音频的说话人相似度，
    [-1, 1]，越大越像同一人。
  - F0 RMSE（librosa.pyin）：voiced frames 上的基频 RMSE（Hz），
    越小越贴合参考韵律。

本模块刻意与 `core/eval.py` 分开。后者给原始 chunk 打分（M1 闸门），
本模块给合成输出打分。混在一个文件会导致同名 "DNSMOS" 调用在不同语义
下出现，难以维护。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np

from . import audio_io

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Score container
# 分数容器
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TTSEvalScores:
    """Objective scores for one synthesised utterance.

    一条合成音频的客观评分集合。

    Attributes:
        mos_nisqa: NISQA primary MOS prediction in [1, 5]. **Main
            naturalness axis.** Direction matches subjective listening
            tests on Trump-style synthesis (zero_shot > instruct).
            NISQA 主 MOS 预测（主自然度轴）；方向与主观听感一致。
        mos_noi / mos_dis / mos_col / mos_loud: NISQA 4 sub-dimensions
            (noisiness / discontinuity / coloration / loudness).
            NISQA 4 个子维度：噪音 / 不连续 / 色彩感 / 响度。
        mos_p808 / mos_ovr / mos_sig / mos_bak: DNSMOS scores (auxiliary,
            kept for cross-check; **don't rely on these for TTS quality**).
            DNSMOS 分数（辅助，TTS 质量评判不要靠这些）。
        wer / cer / wer_transcript: ASR-cycle word/character error rate.
            ASR cycle WER/CER + 转写文本。
        secs: WavLM speaker-verification cosine vs ref, [-1, 1]. > 0.7
            ≈ same speaker.
            WavLM SV 与参考音频的余弦相似度。
        f0_rmse_hz: F0 RMSE in Hz over jointly voiced frames; None if
            voiced overlap < 5 frames.
            voiced frame 上的 F0 RMSE（Hz）。
        duration_s / eval_time_s: timing.
            时长与评测耗时。
    """

    mos_nisqa: float | None = None
    mos_noi: float | None = None
    mos_dis: float | None = None
    mos_col: float | None = None
    mos_loud: float | None = None
    mos_p808: float | None = None
    mos_ovr: float | None = None
    mos_sig: float | None = None
    mos_bak: float | None = None
    wer: float | None = None
    cer: float | None = None
    wer_transcript: str | None = None
    secs: float | None = None
    f0_rmse_hz: float | None = None
    duration_s: float = 0.0
    eval_time_s: float = 0.0


# ---------------------------------------------------------------------------
# MOS via speechmos DNSMOS-P808
# MOS：通过 speechmos 的 DNSMOS-P808
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _dnsmos_module():
    """Lazy import of speechmos.dnsmos; cached process-wide.

    懒加载 speechmos.dnsmos；进程级缓存。
    """
    from speechmos import dnsmos
    return dnsmos


def _mos_scores(audio: np.ndarray, sr: int) -> dict[str, float]:
    """Run DNSMOS on a single waveform.

    对单条波形跑 DNSMOS，返回 {p808_mos, ovrl_mos, sig_mos, bak_mos}。

    Args:
        audio: 1-D float32 audio array.
        sr: Source sample rate; resampled to 16k internally.

    Returns:
        Dict with the four MOS keys.
    """
    import librosa
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    dnsmos = _dnsmos_module()
    return dnsmos.run(audio.astype(np.float32), sr=16000)


# ---------------------------------------------------------------------------
# MOS via NISQA (primary naturalness predictor)
# MOS：用 NISQA（主自然度预测）
# ---------------------------------------------------------------------------


_NISQA_WEIGHTS_URL = "https://github.com/gabrielmittag/NISQA/raw/master/weights/nisqa.tar"
REPO_ROOT = Path(__file__).resolve().parent.parent


def _nisqa_weights_path() -> Path:
    """Local cache path for the NISQA pretrained checkpoint.

    NISQA 预训练权重的本地缓存路径。
    """
    return REPO_ROOT / "models" / "nisqa" / "nisqa.tar"


def _ensure_nisqa_weights() -> Path:
    """Download NISQA checkpoint (~1 MB) on first use.

    首次使用时下载 NISQA 权重（~1MB），缓存到 models/nisqa/。
    """
    p = _nisqa_weights_path()
    if p.exists():
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    logger.info("downloading NISQA weights from %s ...", _NISQA_WEIGHTS_URL)
    from urllib.request import urlretrieve
    urlretrieve(_NISQA_WEIGHTS_URL, p)
    return p


def _nisqa_scores(wav_path: Path) -> dict[str, float]:
    """Run NISQA_DIM on a single wav, return all 5 sub-scores.

    对单条 wav 跑 NISQA_DIM，返回 5 维评分。

    NISQA's nisqaModel is constructed per call. The checkpoint is small
    (~1 MB), so reload cost is ~700-900 ms total — acceptable when the
    caller is a one-off /api/synthesize hook or a batch loop (16 wavs =
    ~15 s overhead). The chatty stdout from NISQA's own print() calls is
    silenced to keep server / runner logs readable.

    每次调用都重建 nisqaModel。checkpoint 只有 1MB，重载耗时 ~700-900ms，
    对单次 /api/synthesize hook 或 16 条批处理（~15s 开销）都可接受。
    NISQA 自带的 print 调用被静默以保持日志可读。

    Args:
        wav_path: Audio file path.

    Returns:
        Dict with keys {mos_pred, noi_pred, dis_pred, col_pred, loud_pred}.
        All values in [1, 5], higher = better.
    """
    import io
    from contextlib import redirect_stderr, redirect_stdout

    from nisqa.NISQA_model import nisqaModel

    args = {
        "mode": "predict_file",
        "pretrained_model": str(_ensure_nisqa_weights()),
        "deg": str(wav_path),
        "output_dir": None,
        "tr_bs_val": 1,
        "tr_num_workers": 0,
        "tr_parallel": False,
        # ms_channel = None means "mono / use single channel"; absent
        # from the v2.0 checkpoint so we set it explicitly.
        # ms_channel=None 表示单声道；v2.0 checkpoint 里没有这个字段，
        # 需要显式补上。
        "ms_channel": None,
    }
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        model = nisqaModel(args)
        df = model.predict()
    row = df.iloc[0]
    return {
        "mos_pred": float(row["mos_pred"]),
        "noi_pred": float(row["noi_pred"]),
        "dis_pred": float(row["dis_pred"]),
        "col_pred": float(row["col_pred"]),
        "loud_pred": float(row["loud_pred"]),
    }


# ---------------------------------------------------------------------------
# SECS via WavLM speaker verification
# SECS：用 WavLM speaker verification
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _wavlm_sv():
    """Lazy load microsoft/wavlm-base-plus-sv; cached process-wide.

    懒加载 microsoft/wavlm-base-plus-sv 模型（首次 ~400MB 下载）。
    """
    from transformers import AutoFeatureExtractor, WavLMForXVector
    logger.info("loading microsoft/wavlm-base-plus-sv ...")
    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    # use_safetensors=True forces the .safetensors checkpoint, which
    # avoids torch.load entirely and works under torch 2.2.x where
    # weights_only=True is blocked by CVE-2025-32434.
    # use_safetensors=True 强制走 .safetensors 权重，绕开 torch.load；
    # 在 torch 2.2.x 下 weights_only=True 会被 CVE-2025-32434 拒绝加载。
    model = WavLMForXVector.from_pretrained(
        "microsoft/wavlm-base-plus-sv", use_safetensors=True,
    )
    model.eval()
    return fe, model


def _secs_wavlm(
    syn_audio: np.ndarray, syn_sr: int,
    ref_audio: np.ndarray, ref_sr: int,
) -> float:
    """Cosine similarity between WavLM-SV x-vectors of syn and ref.

    用 WavLM-SV 提两段音频的 x-vector，算余弦相似度。

    Args:
        syn_audio / syn_sr: synthesised waveform + sr.
        ref_audio / ref_sr: reference waveform + sr.

    Returns:
        Cosine similarity in [-1, 1]; > 0.7 ≈ same speaker.
    """
    import librosa
    import torch

    if syn_sr != 16000:
        syn_audio = librosa.resample(syn_audio, orig_sr=syn_sr, target_sr=16000)
    if ref_sr != 16000:
        ref_audio = librosa.resample(ref_audio, orig_sr=ref_sr, target_sr=16000)

    fe, model = _wavlm_sv()
    with torch.no_grad():
        syn_inp = fe(
            [syn_audio.astype(np.float32)],
            sampling_rate=16000, return_tensors="pt", padding=True,
        )
        ref_inp = fe(
            [ref_audio.astype(np.float32)],
            sampling_rate=16000, return_tensors="pt", padding=True,
        )
        syn_emb = model(**syn_inp).embeddings
        ref_emb = model(**ref_inp).embeddings
    syn_emb = torch.nn.functional.normalize(syn_emb, dim=-1)
    ref_emb = torch.nn.functional.normalize(ref_emb, dim=-1)
    return float((syn_emb @ ref_emb.T).squeeze().item())


# ---------------------------------------------------------------------------
# WER / CER via ASR cycle + jiwer
# WER / CER：合成 → ASR → jiwer
# ---------------------------------------------------------------------------


def _wer_or_cer(
    syn_wav: Path, target_text: str, lang: str, transcriber,
) -> tuple[float, str, str]:
    """ASR-cycle WER (en) or CER (zh).

    合成→ASR→对比目标文本，英文算 WER，中文算 CER。

    Args:
        syn_wav: synthesised wav file path.
        target_text: ground-truth text (the prompt fed to TTS).
        lang: "en" or "zh".
        transcriber: a `core.asr.Transcriber` instance (reusable).

    Returns:
        (score, asr_text, metric_kind) — metric_kind is "wer" or "cer".
    """
    from jiwer import cer as jcer
    from jiwer import wer as jwer

    from . import text_norm

    result = transcriber.transcribe(syn_wav)
    asr_text = result.text.strip()
    # The TTS pipeline applies `text_norm.normalize_for_tts` to the input
    # before synthesis (see core/tts.py:269). For a fair WER, normalise
    # the *target* the same way so we compare against what was actually
    # spoken, not the raw user input. The ASR output is already
    # post-synthesis speech, so it doesn't need extra normalisation
    # beyond lowercasing.
    # TTS 合成前会对输入跑 normalize_for_tts；为公平比较，目标文本也走
    # 同一个规范化（不然 "It's" vs "It is" 这类全成 WER 错误）。
    target_norm = text_norm.normalize_for_tts(target_text, lang=lang)
    if lang == "zh":
        # CER: Chinese has no word boundaries; strip surrounding whitespace
        # on both sides.
        # CER：中文无词边界，去掉两端空白即可。
        score = float(jcer(target_norm.strip(), asr_text))
        return score, asr_text, "cer"
    score = float(jwer(target_norm.lower(), asr_text.lower()))
    return score, asr_text, "wer"


# ---------------------------------------------------------------------------
# F0 RMSE via pyin
# F0 RMSE：用 librosa pyin
# ---------------------------------------------------------------------------


def _f0_rmse(
    syn_audio: np.ndarray, syn_sr: int,
    ref_audio: np.ndarray, ref_sr: int,
) -> float:
    """F0 RMSE in Hz over frames voiced in both signals.

    在 syn 与 ref 双方都 voiced 的 frame 上算基频 RMSE（Hz）。

    Args:
        syn_audio / syn_sr: synthesised waveform.
        ref_audio / ref_sr: reference waveform.

    Returns:
        RMSE in Hz; NaN if voiced overlap < 5 frames (caller treats as None).
    """
    import librosa

    if syn_sr != 16000:
        syn_audio = librosa.resample(syn_audio, orig_sr=syn_sr, target_sr=16000)
    if ref_sr != 16000:
        ref_audio = librosa.resample(ref_audio, orig_sr=ref_sr, target_sr=16000)

    # fmin/fmax cover both male (down to ~80) and female (up to ~350) ranges.
    # fmin/fmax 覆盖男声（最低 ~80）到女声（最高 ~350）。50/400 留余量。
    f0_syn, voiced_syn, _ = librosa.pyin(syn_audio, fmin=50, fmax=400, sr=16000)
    f0_ref, voiced_ref, _ = librosa.pyin(ref_audio, fmin=50, fmax=400, sr=16000)

    # No DTW in v1: trim to min length and align linearly. Good enough for
    # similar-length pairs (5-15s each); DTW deferred to a second pass.
    # v1 不上 DTW：截到共同长度直接逐帧对齐。对相近时长的两段（各 5-15s）
    # 足够；DTW 留给第二轮升级。
    n = min(len(f0_syn), len(f0_ref))
    f0_syn = f0_syn[:n]
    f0_ref = f0_ref[:n]
    voiced_syn = voiced_syn[:n]
    voiced_ref = voiced_ref[:n]

    both_voiced = voiced_syn & voiced_ref & ~np.isnan(f0_syn) & ~np.isnan(f0_ref)
    if int(both_voiced.sum()) < 5:
        return float("nan")
    diff = f0_syn[both_voiced] - f0_ref[both_voiced]
    return float(np.sqrt(np.mean(diff ** 2)))


# ---------------------------------------------------------------------------
# Main entry
# 主入口
# ---------------------------------------------------------------------------


def evaluate_synthesis(
    syn_wav: Path,
    *,
    ref_wav: Path | None = None,
    target_text: str | None = None,
    lang: Literal["en", "zh"] = "en",
    transcriber=None,
    skip: tuple[str, ...] = (),
) -> TTSEvalScores:
    """Run all four objective metrics on a synthesised wav.

    对一条合成 wav 跑全部四项客观指标。

    Args:
        syn_wav: Path to the synthesised audio file.
            合成音频文件路径。
        ref_wav: Optional reference audio path. SECS and F0_RMSE only run
            when this is provided and the file exists.
            参考音频路径；提供且存在时才算 SECS 和 F0 RMSE。
        target_text: Ground-truth text. WER/CER only run when provided.
            目标文本；提供时才算 WER/CER。
        lang: "en" (WER) or "zh" (CER).
            语言；决定算 WER 还是 CER。
        transcriber: Pre-constructed `core.asr.Transcriber`. Built lazily
            if None — pass one in batch contexts to avoid repeated model
            loads.
            可复用的 Transcriber；批量评测时务必从外部传入避免重复加载。
        skip: Skip given metrics. Valid keys: "mos", "wer", "secs", "f0_rmse".
            跳过的指标名集合。

    Returns:
        TTSEvalScores with whatever could be computed (others left None).
        TTSEvalScores；未能计算的指标字段保持 None。
    """
    t0 = time.monotonic()
    syn_audio, syn_sr = audio_io.load(syn_wav)
    duration = float(len(syn_audio)) / float(syn_sr)

    # NISQA = primary naturalness MOS. Reads wav from disk directly (its
    # own loader normalises sr/channels), so we pass the path.
    # NISQA：主自然度 MOS；自带 loader 处理采样率/通道，直接传路径。
    nisqa_scores: dict[str, float] = {}
    if "nisqa" not in skip and "mos" not in skip:
        try:
            nisqa_scores = _nisqa_scores(syn_wav)
        except Exception as e:  # noqa: BLE001
            logger.warning("NISQA failed for %s: %s", syn_wav, e)

    # DNSMOS = auxiliary, kept for cross-check. Known to give misleading
    # ranks on TTS output; don't trust as the primary axis.
    # DNSMOS：辅助，TTS 排序常出错，不当主轴。Cast numpy float32 to
    # plain float so json.dumps doesn't choke downstream.
    # numpy float32 显式转 float，避免 JSON 序列化报错。
    mos_scores: dict[str, float] = {}
    if "dnsmos" not in skip and "mos" not in skip:
        try:
            raw = _mos_scores(syn_audio, syn_sr)
            mos_scores = {k: float(v) for k, v in raw.items()}
        except Exception as e:  # noqa: BLE001 — keep eval robust to single-metric failures
            logger.warning("DNSMOS failed for %s: %s", syn_wav, e)

    wer_val: float | None = None
    cer_val: float | None = None
    asr_text: str | None = None
    if target_text and "wer" not in skip:
        if transcriber is None:
            from .asr import Transcriber
            transcriber = Transcriber()
        try:
            score, asr_text, metric_kind = _wer_or_cer(syn_wav, target_text, lang, transcriber)
            if metric_kind == "cer":
                cer_val = score
            else:
                wer_val = score
        except Exception as e:  # noqa: BLE001
            logger.warning("WER/CER failed for %s: %s", syn_wav, e)

    secs_val: float | None = None
    f0_val: float | None = None
    if ref_wav is not None and Path(ref_wav).exists():
        ref_audio, ref_sr = audio_io.load(ref_wav)
        if "secs" not in skip:
            try:
                secs_val = _secs_wavlm(syn_audio, syn_sr, ref_audio, ref_sr)
            except Exception as e:  # noqa: BLE001
                logger.warning("SECS failed for %s: %s", syn_wav, e)
        if "f0_rmse" not in skip:
            try:
                f0_raw = _f0_rmse(syn_audio, syn_sr, ref_audio, ref_sr)
                f0_val = None if np.isnan(f0_raw) else f0_raw
            except Exception as e:  # noqa: BLE001
                logger.warning("F0 RMSE failed for %s: %s", syn_wav, e)

    return TTSEvalScores(
        mos_nisqa=nisqa_scores.get("mos_pred"),
        mos_noi=nisqa_scores.get("noi_pred"),
        mos_dis=nisqa_scores.get("dis_pred"),
        mos_col=nisqa_scores.get("col_pred"),
        mos_loud=nisqa_scores.get("loud_pred"),
        mos_p808=mos_scores.get("p808_mos"),
        mos_ovr=mos_scores.get("ovrl_mos"),
        mos_sig=mos_scores.get("sig_mos"),
        mos_bak=mos_scores.get("bak_mos"),
        wer=wer_val,
        cer=cer_val,
        wer_transcript=asr_text,
        secs=secs_val,
        f0_rmse_hz=f0_val,
        duration_s=duration,
        eval_time_s=time.monotonic() - t0,
    )
