"""Bilingual ASR: Whisper for English, FunASR Paraformer-zh for Chinese.

Language is detected once per file by faster-whisper's langid (which is
much cheaper than full transcription). Routing matrix:

    detected lang     | backend
    ------------------+----------------------
    "en" (or anything | faster-whisper large-v3
     non-zh)          | (with auto-fallback to medium on OOM)
    "zh"              | FunASR Paraformer-zh + ct-punc

Why two backends instead of one Whisper-multilingual: FunASR's
Paraformer-zh is consistently 2~4 percentage points lower WER than
Whisper on Mandarin while also being faster. The cost is one extra
dependency, which is worth it for the project's primary language.

See ADR-0007 for the language-routing rationale.

双语 ASR 模块。

工作流：先用 faster-whisper 的 langid 做一次廉价的语言检测，再按语种
路由到合适的后端：
    检测语言        | 后端
    ----------------+----------------------------
    "en" 或非中文   | faster-whisper large-v3
                    | （OOM 时自动 fallback medium）
    "zh"            | FunASR Paraformer-zh + ct-punc 标点恢复

为什么不直接用 Whisper 多语言：FunASR Paraformer-zh 在中文上 WER
比 Whisper 低 2~4 个百分点而且更快。代价是多一个依赖，对中文为主的项目
完全值得。详见 ADR-0007。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranscriptResult:
    """Output of `Transcriber.transcribe()`.

    单次转写的产出。

    Attributes:
        text: Transcribed text with punctuation restored.
              带标点恢复后的转写文本。
        lang: Detected ISO 639-1 code ("en", "zh", ...).
              ISO 639-1 语种码。
        confidence: Average word-level confidence in [0, 1]. None when
            the backend doesn't expose word probabilities.
            平均词级置信度（0~1），后端不支持时为 None。
        word_timestamps: Optional list of {"word", "start", "end"} dicts.
            可选的词级时间戳列表。
    """

    text: str
    lang: str
    confidence: float | None
    word_timestamps: list[dict] | None = None


def _pick_device_and_compute() -> tuple[str, str]:
    """Pick (device, compute_type) for faster-whisper.

    为 faster-whisper 挑选设备与精度：CUDA + float16 / MPS-as-CPU + int8 /
    CPU + int8。MPS 在 ctranslate2 里不直接支持，所以 Mac 走 CPU+int8。
    """
    if torch.cuda.is_available():
        return "cuda", "float16"
    return "cpu", "int8"


class WhisperBackend:
    """faster-whisper backend with auto-fallback from large-v3 to medium.

    faster-whisper 后端；large-v3 OOM 时自动降级到 medium。

    Args:
        model_size: Initial model size to try ("large-v3", "medium", ...).
            初始尝试加载的模型尺寸。
        download_root: Optional cache dir for model weights.
            模型权重缓存目录（默认 ~/.cache/huggingface/）。
    """

    def __init__(self, model_size: str = "large-v3", download_root: Path | str | None = None) -> None:
        self.model_size = model_size
        self.download_root = str(download_root) if download_root else None
        self._model = None
        self._loaded_size: str | None = None

    def _ensure_model(self):
        """Lazy load with fallback on OOM / RuntimeError.

        惰性加载；large-v3 加载失败（OOM 等）时降级到 medium 并记日志。
        """
        if self._model is not None:
            return self._model
        from faster_whisper import WhisperModel
        device, compute = _pick_device_and_compute()
        try:
            logger.info("WhisperBackend: loading %s on %s/%s", self.model_size, device, compute)
            self._model = WhisperModel(
                self.model_size, device=device, compute_type=compute,
                download_root=self.download_root,
            )
            self._loaded_size = self.model_size
        except (RuntimeError, MemoryError) as e:
            if self.model_size != "medium":
                logger.warning("WhisperBackend: %s failed (%s); falling back to medium", self.model_size, e)
                self._model = WhisperModel(
                    "medium", device=device, compute_type=compute,
                    download_root=self.download_root,
                )
                self._loaded_size = "medium"
            else:
                raise
        return self._model

    def detect_language(self, wav_path: Path | str) -> str:
        """Run only Whisper's language ID, much cheaper than full transcribe.

        只跑 Whisper 的语种识别，不做完整转写——比 transcribe() 快很多，
        用于决定路由到 Whisper 还是 FunASR。

        Args:
            wav_path: Audio file path.

        Returns:
            ISO 639-1 language code (e.g. "en", "zh").
        """
        model = self._ensure_model()
        # faster-whisper exposes detect_language via the segmenter; cheapest
        # is to call transcribe() with `vad_filter=True` and beam=1, then
        # only consume the first segment metadata. We skip beam search
        # entirely with beam_size=1 for langid use.
        # faster-whisper 没有独立的 detect_language API；这里走最低开销的
        # transcribe（beam=1 + vad_filter）拿到 info.language。
        _, info = model.transcribe(str(wav_path), beam_size=1, vad_filter=True, language=None)
        return info.language

    def transcribe(self, wav_path: Path | str, *, language: str | None = None) -> TranscriptResult:
        """Full transcription with optional fixed language.

        完整转写；提供 language 时跳过语种识别，否则自动检测。

        Args:
            wav_path: Audio path.
            language: Optional ISO 639-1 code; None = auto-detect.

        Returns:
            TranscriptResult with text + lang + confidence + word_timestamps.
        """
        model = self._ensure_model()
        segments, info = model.transcribe(
            str(wav_path),
            beam_size=5,
            vad_filter=False,  # we already VAD'd upstream
            language=language,
            word_timestamps=True,
        )
        words: list[dict] = []
        text_parts: list[str] = []
        confidences: list[float] = []
        for seg in segments:
            text_parts.append(seg.text.strip())
            if seg.words:
                for w in seg.words:
                    words.append({"word": w.word, "start": w.start, "end": w.end})
                    if w.probability is not None:
                        confidences.append(float(w.probability))
        text = " ".join(text_parts).strip()
        avg_conf = sum(confidences) / len(confidences) if confidences else None
        return TranscriptResult(
            text=text,
            lang=info.language,
            confidence=avg_conf,
            word_timestamps=words or None,
        )


class FunASRBackend:
    """FunASR Paraformer-zh + ct-punc backend for Mandarin transcription.

    FunASR 中文转写后端：Paraformer-zh 主模型 + ct-punc 标点恢复。

    Args:
        model_dir: Optional cache directory for ModelScope downloads.
            ModelScope 模型缓存目录。
    """

    def __init__(self, model_dir: Path | str | None = None) -> None:
        self.model_dir = str(model_dir) if model_dir else None
        self._asr = None
        self._punc = None

    def _ensure_models(self):
        """Lazily download + load Paraformer-zh and ct-punc.

        惰性下载并加载 Paraformer-zh 与 ct-punc（首次约 1.5GB 模型下载）。
        """
        if self._asr is not None and self._punc is not None:
            return
        # Lazy import: FunASR pulls in ModelScope which is ~heavy.
        # 延迟导入：FunASR 会引入 ModelScope，体积不小。
        from funasr import AutoModel
        logger.info("FunASRBackend: loading Paraformer-zh + ct-punc")
        kw = {"model_revision": None}
        if self.model_dir:
            kw["cache_dir"] = self.model_dir
        self._asr = AutoModel(model="paraformer-zh", **kw)
        self._punc = AutoModel(model="ct-punc", **kw)

    def transcribe(self, wav_path: Path | str) -> TranscriptResult:
        """Transcribe Mandarin audio with punctuation restoration.

        中文音频转写并恢复标点。

        Args:
            wav_path: Audio path.

        Returns:
            TranscriptResult; confidence is None (FunASR doesn't expose
            word-level probabilities by default).
            confidence 为 None（Paraformer 默认不输出词级概率）。
        """
        self._ensure_models()
        raw = self._asr.generate(input=str(wav_path))
        text = raw[0]["text"] if raw else ""
        if text:
            puncted = self._punc.generate(input=text)
            text = puncted[0]["text"] if puncted else text
        return TranscriptResult(text=text, lang="zh", confidence=None, word_timestamps=None)


class Transcriber:
    """Routing transcriber: picks Whisper or FunASR based on detected language.

    路由 transcriber：先 langid 再分流到合适的后端。

    Caching: both backends are kept warm after first use. To reduce memory
    you can construct a fresh Transcriber per run, but typical usage is
    one instance per pipeline run.

    缓存：两个后端在首次使用后常驻内存供批量复用。
    显存紧张时可每次新建实例。

    Args:
        lang_hint: Skip langid and force a specific language ("en"/"zh").
            语言提示；提供时跳过 langid 直接路由。
    """

    def __init__(self, lang_hint: str | None = None) -> None:
        self.lang_hint = lang_hint
        self.whisper = WhisperBackend()
        self.funasr: FunASRBackend | None = None  # lazy-init in zh path

    def transcribe(self, wav_path: Path | str) -> TranscriptResult:
        """Transcribe one audio file with auto language routing.

        转写一个音频文件，按检测到的语种自动路由到 Whisper 或 FunASR。

        Args:
            wav_path: Audio path.

        Returns:
            TranscriptResult.
        """
        lang = self.lang_hint or self.whisper.detect_language(wav_path)
        if lang == "zh":
            if self.funasr is None:
                self.funasr = FunASRBackend()
            return self.funasr.transcribe(wav_path)
        return self.whisper.transcribe(wav_path, language=lang)
