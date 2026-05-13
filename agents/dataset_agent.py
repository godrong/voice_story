"""DatasetAgent: ASR -> quality filter -> diversity sampling -> manifest.

Reads `state.chunks` produced by PreprocessAgent, runs each through the
bilingual ASR (core.asr), scores audio quality (core.eval), filters out
chunks that fail any threshold (MOS / SNR / confidence / clipping), then
emits a final `manifest.jsonl` plus a human-readable `report.md` with
diversity stats.

Diversity dimensions:
  * Phoneme coverage (pinyin声韵母 for ZH, CMU phoneme for EN)
  * Duration distribution (short / medium / long buckets)
  * Energy distribution (RMS quantile buckets)
  * Prosody (sentence-final punctuation: question / declarative /
    exclamation)

The agent does NOT itself sub-sample the manifest down to a target size;
it keeps every passing chunk and reports diversity as a measurement.
M5 / synthesis stages can read the same manifest and pick references
based on diversity tags later.

数据集构建 agent：ASR → 质量过滤 → 多样性统计 → 输出 manifest。

读取 PreprocessAgent 写入的 chunks，对每段：
  1. 调 core.asr 转写得到文本与置信度
  2. 调 core.eval 打质量分（SNR / DNSMOS / 削波）
  3. 应用门槛过滤
最后输出 manifest.jsonl + 人类可读的 report.md（含多样性统计）。

多样性维度：
  * 音素覆盖（中文声韵母 / 英文 CMU phoneme）
  * 时长分布（短/中/长 三档）
  * 能量分布（RMS 分位数）
  * 韵律（句末标点：问/陈/感）

本 agent 不做下采样，所有过门槛的 chunk 都进 manifest；多样性以指标形式
报告给后续 stage（M5/合成时按 tag 选 reference）。
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core import asr as asr_mod
from core import audio_io, eval as eval_mod

from .state import PipelineState, QualityScore, TranscriptInfo

logger = logging.getLogger(__name__)


# ---- Filtering thresholds ------------------------------------------------
#
# These are conservative defaults from the v0.1.0 plan. Tuning happens
# via flags in CLI, not by editing constants in code.
# 默认过滤阈值；通过 CLI 标志调整，避免在代码里硬编辑常量。

DEFAULT_MIN_MOS_OVR = 3.5
DEFAULT_MIN_SNR_DB = 15.0
DEFAULT_MIN_CONFIDENCE = 0.85


@dataclass
class FilterThresholds:
    """Bundle of pass/fail thresholds for manifest inclusion.

    一组 manifest 准入门槛；过滤判定用。

    Attributes:
        min_mos_ovr: Minimum DNSMOS OVR score. 最小 OVR。
        min_snr_db: Minimum WADA-SNR in dB. 最小 SNR。
        min_confidence: Minimum ASR avg confidence (skipped when None).
            ASR 最小平均置信度（confidence 为 None 时跳过该项）。
        require_no_clipping: Drop chunks with detected clipping.
            是否必须无削波。
    """

    min_mos_ovr: float = DEFAULT_MIN_MOS_OVR
    min_snr_db: float = DEFAULT_MIN_SNR_DB
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    require_no_clipping: bool = True


def _bucket_duration(sec: float) -> str:
    """Categorize chunk duration into short/medium/long.

    把切片时长分到 short/medium/long 三档（与 VAD 的 3~15s 对齐）。
    """
    if sec < 5.0:
        return "short"
    if sec < 10.0:
        return "medium"
    return "long"


def _bucket_energy(audio: np.ndarray) -> str:
    """Categorize chunk energy by RMS percentile against a fixed grid.

    基于 RMS 能量分到 quiet/normal/loud 三档，固定分位数避免按数据集漂移。
    """
    rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)) + 1e-10)
    if rms < 0.05:
        return "quiet"
    if rms < 0.15:
        return "normal"
    return "loud"


def _prosody_label(text: str) -> str:
    """Classify a sentence by its terminal punctuation.

    按句末标点判定句式：问句/感叹/陈述。
    支持中英文标点。
    """
    stripped = text.strip()
    if not stripped:
        return "declarative"
    last = stripped[-1]
    if last in {"?", "？"}:
        return "question"
    if last in {"!", "！"}:
        return "exclamation"
    return "declarative"


def _phonemes(text: str, lang: str) -> set[str]:
    """Return the set of phoneme units in `text` for the given language.

    抽取文本中包含的音素单元集合。
    中文用 pypinyin 拿声韵母；英文用 CMU pronouncing 拿 ARPAbet phoneme。

    Args:
        text: Source text.
        lang: ISO 639-1 language code.

    Returns:
        Set of phoneme strings. Empty if backend is unavailable.
    """
    if lang == "zh":
        try:
            from pypinyin import lazy_pinyin, Style
            initials = lazy_pinyin(text, style=Style.INITIALS, strict=False, errors="ignore")
            finals = lazy_pinyin(text, style=Style.FINALS, strict=False, errors="ignore")
            return {p for p in (initials + finals) if p}
        except ImportError:
            logger.warning("pypinyin not installed; phoneme coverage will be empty for zh")
            return set()
    # English / fallback
    try:
        import pronouncing
        out: set[str] = set()
        # Strip non-alpha to get word tokens. 简单按空格切，保留字母词。
        for word in re.findall(r"[A-Za-z']+", text.lower()):
            for p in pronouncing.phones_for_word(word):
                # Each phone string is space-separated ARPAbet symbols with stress.
                for ph in p.split():
                    # Drop trailing stress digits to get base phoneme (e.g. "AH0" -> "AH").
                    out.add(re.sub(r"\d", "", ph))
        return out
    except ImportError:
        logger.warning("pronouncing not installed; phoneme coverage will be empty for en")
        return set()


def _phoneme_universe(lang: str) -> set[str]:
    """Return the universe of phonemes considered for coverage computation.

    用于覆盖率分母的"总音素集合"。
    中文：pypinyin 的全部声母 + 韵母（约 60）。英文：CMU 39 个 ARPAbet。
    """
    if lang == "zh":
        # pypinyin's complete initials + finals reference list.
        initials = {
            "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
            "j", "q", "x", "zh", "ch", "sh", "r", "z", "c", "s",
            "y", "w", "",
        }
        finals = {
            "a", "o", "e", "i", "u", "v",
            "ai", "ei", "ui", "ao", "ou", "iu", "ie", "ve", "er",
            "an", "en", "in", "un", "vn",
            "ang", "eng", "ing", "ong",
            "ia", "ua", "uo", "iao", "iou", "uai", "uei",
            "ian", "uan", "van", "uen", "iang", "uang", "iong",
        }
        return initials | finals
    # CMU ARPAbet (without stress markers).
    return {
        "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH",
        "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K",
        "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH",
        "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH",
    }


class DatasetAgent:
    """Pipeline stage that produces the final training-ready manifest.

    数据管线最后一阶段：产出训练可用的 manifest 与质量报告。

    Args:
        thresholds: FilterThresholds for chunk admission.
            chunk 准入门槛配置。
        lang_hint: Forward to ASR (skips langid when set).
            语言提示，提供时 ASR 跳过 langid。
    """

    name = "dataset_agent"

    def __init__(
        self,
        *,
        thresholds: FilterThresholds | None = None,
        lang_hint: str | None = None,
    ) -> None:
        self.thresholds = thresholds or FilterThresholds()
        self.lang_hint = lang_hint
        self._transcriber: asr_mod.Transcriber | None = None

    def _ensure_transcriber(self) -> asr_mod.Transcriber:
        """Lazy-init the bilingual transcriber.

        惰性创建 Transcriber（含 Whisper / FunASR 两个后端）。
        """
        if self._transcriber is None:
            hint = self.lang_hint
            self._transcriber = asr_mod.Transcriber(lang_hint=hint)
        return self._transcriber

    async def run(self, state: PipelineState) -> PipelineState:
        """Score, filter, and emit manifest.jsonl + report.md.

        逐 chunk 评分 + 转写 + 过滤，最后写出 manifest.jsonl 和 report.md。

        Args:
            state: Pipeline state. chunks must be populated.
                   流水线状态对象，要求 chunks 已就位。

        Returns:
            Same state with `manifest_path`, `transcripts`, and `quality`
            populated.
            就地修改后的 state；写入 manifest_path / transcripts / quality。

        Raises:
            ValueError: If state.chunks is empty.
                        chunks 为空时抛出。
        """
        if not state.chunks:
            raise ValueError("DatasetAgent: state.chunks is empty")

        lang_hint = self.lang_hint or (state.source_meta and state.source_meta.lang_hint)
        if lang_hint and self._transcriber is None:
            self._transcriber = asr_mod.Transcriber(lang_hint=lang_hint)
        transcriber = self._ensure_transcriber()

        rows: list[dict] = []
        kept = 0
        dropped: Counter[str] = Counter()
        seen_phonemes: set[str] = set()
        bucket_dur: Counter[str] = Counter()
        bucket_energy: Counter[str] = Counter()
        bucket_prosody: Counter[str] = Counter()
        lang_tally: Counter[str] = Counter()

        for chunk in state.chunks:
            # ASR
            tr = transcriber.transcribe(chunk.path)
            state.transcripts[chunk.chunk_id] = TranscriptInfo(
                chunk_id=chunk.chunk_id, text=tr.text, lang=tr.lang,
                confidence=tr.confidence if tr.confidence is not None else 1.0,
            )
            # Quality
            snr, mos, clipped = eval_mod.score_chunk(chunk.path)
            state.quality[chunk.chunk_id] = QualityScore(
                chunk_id=chunk.chunk_id, snr_db=snr,
                mos_ovr=mos.ovr, mos_sig=mos.sig, mos_bak=mos.bak,
                clipped=clipped,
            )
            # Filter
            reason = self._filter_reason(tr, snr, mos, clipped)
            if reason is not None:
                dropped[reason] += 1
                logger.debug("DatasetAgent: drop %s (%s)", chunk.chunk_id, reason)
                continue

            # Diversity bookkeeping
            audio, _ = audio_io.load(chunk.path)
            dur = chunk.end_sec - chunk.start_sec
            bucket_dur[_bucket_duration(dur)] += 1
            bucket_energy[_bucket_energy(audio)] += 1
            bucket_prosody[_prosody_label(tr.text)] += 1
            seen_phonemes |= _phonemes(tr.text, tr.lang)
            lang_tally[tr.lang] += 1

            rows.append({
                "chunk_id": chunk.chunk_id,
                "audio_path": str(chunk.path),
                "source_file": str(chunk.source_file),
                "text": tr.text,
                "lang": tr.lang,
                "confidence": tr.confidence,
                "duration": dur,
                "snr_db": snr,
                "mos_ovr": mos.ovr,
                "mos_sig": mos.sig,
                "mos_bak": mos.bak,
            })
            kept += 1

        # Write manifest + report
        manifest_path = state.dataset_root / "manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8") as fp:
            for row in rows:
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")
        state.manifest_path = manifest_path

        report_path = state.dataset_root / "report.md"
        report = self._render_report(
            kept=kept, total=len(state.chunks),
            dropped=dropped, lang_tally=lang_tally,
            bucket_dur=bucket_dur, bucket_energy=bucket_energy,
            bucket_prosody=bucket_prosody, seen_phonemes=seen_phonemes,
        )
        report_path.write_text(report, encoding="utf-8")

        logger.info(
            "DatasetAgent: kept %d/%d (dropped %s) -> %s",
            kept, len(state.chunks), dict(dropped), manifest_path,
        )
        return state

    def _filter_reason(
        self, tr: asr_mod.TranscriptResult,
        snr: float, mos: eval_mod.DnsmosScores, clipped: bool,
    ) -> str | None:
        """Return the drop reason for a chunk, or None if it passes.

        判断单 chunk 是否被过滤；返回原因字符串，None 表示通过。
        """
        if self.thresholds.require_no_clipping and clipped:
            return "clipping"
        if snr < self.thresholds.min_snr_db:
            return "low_snr"
        if mos.ovr < self.thresholds.min_mos_ovr:
            return "low_mos"
        if (
            tr.confidence is not None
            and tr.confidence < self.thresholds.min_confidence
        ):
            return "low_confidence"
        if not tr.text.strip():
            return "empty_transcript"
        return None

    def _render_report(
        self, *, kept: int, total: int,
        dropped: Counter[str], lang_tally: Counter[str],
        bucket_dur: Counter[str], bucket_energy: Counter[str],
        bucket_prosody: Counter[str], seen_phonemes: set[str],
    ) -> str:
        """Render the human-readable report.md content.

        渲染 report.md 内容（含质量统计 + 多样性 + 音素覆盖率）。
        """
        # Compute coverage per language seen.
        # 按检测到的语种分别算音素覆盖率。
        coverage_lines = []
        for lang in lang_tally:
            uni = _phoneme_universe(lang)
            covered = seen_phonemes & uni
            ratio = len(covered) / max(1, len(uni))
            coverage_lines.append(
                f"  - {lang}: {len(covered)}/{len(uni)} = {ratio * 100:.1f}%"
            )

        return (
            f"# Dataset report\n\n"
            f"- Kept chunks: **{kept}** / {total}\n"
            f"- Drop reasons: {dict(dropped) or '(none)'}\n"
            f"- Languages: {dict(lang_tally) or '(none)'}\n\n"
            f"## Phoneme coverage\n\n"
            + "\n".join(coverage_lines)
            + f"\n\n## Duration buckets\n\n"
            f"  - short  (<5s): {bucket_dur['short']}\n"
            f"  - medium (5-10s): {bucket_dur['medium']}\n"
            f"  - long   (>=10s): {bucket_dur['long']}\n\n"
            f"## Energy buckets\n\n"
            f"  - quiet:  {bucket_energy['quiet']}\n"
            f"  - normal: {bucket_energy['normal']}\n"
            f"  - loud:   {bucket_energy['loud']}\n\n"
            f"## Prosody (terminal punctuation)\n\n"
            f"  - declarative: {bucket_prosody['declarative']}\n"
            f"  - question:    {bucket_prosody['question']}\n"
            f"  - exclamation: {bucket_prosody['exclamation']}\n"
        )


def load_manifest(manifest_path: Path | str) -> Iterable[dict]:
    """Stream JSONL rows from a manifest file.

    流式读取 manifest.jsonl 的每行（dict）。

    Args:
        manifest_path: Path to manifest.jsonl.

    Yields:
        Dict per line.
    """
    with Path(manifest_path).open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                yield json.loads(line)
