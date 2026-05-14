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

import hashlib
import json
import logging
import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core import asr as asr_mod
from core import eval as eval_mod
from core import prosody as prosody_mod

from .state import PipelineState, ProsodyScore, QualityScore, TranscriptInfo

logger = logging.getLogger(__name__)


# ---- Filtering thresholds ------------------------------------------------
#
# Defaults calibrated against Demucs-cleaned audio (see ADR-0009).
# WADA-SNR is recorded in the manifest as a diagnostic but no longer
# gates inclusion — it gives garbage on Demucs vocal stems. DNSMOS-OVR
# default lowered from 3.5 to 3.0 because Demucs leaves a slight
# artifact that costs ~0.4 OVR.
#
# 默认阈值已按 Demucs 处理后的音频校准（详见 ADR-0009）。
# WADA-SNR 不再用于过滤（在 vocal stem 上失真），只作为诊断数据写入 manifest。
# DNSMOS-OVR 默认从 3.5 降到 3.0，对应 Demucs artifact 的 ~0.4 分扣分。

DEFAULT_MIN_MOS_OVR = 3.0
DEFAULT_MIN_CONFIDENCE = 0.85

# Manifest schema version. Bump on any field add/rename/remove so that
# downstream consumers can reject unknown shapes.
# manifest 字段结构版本号；每次新增 / 改名 / 删字段都要 bump，
# 下游消费方据此识别。
MANIFEST_VERSION = "1.1"

# Default speaker tag for single-speaker sources. Multi-speaker future
# work overrides this in state-level metadata.
# 单说话人源的默认 speaker_id；多说话人扩展后由 source/state 覆写。
DEFAULT_SPEAKER_ID = "main"


@dataclass
class FilterThresholds:
    """Bundle of pass/fail thresholds for manifest inclusion.

    一组 manifest 准入门槛；过滤判定用。

    Attributes:
        min_mos_ovr: Minimum DNSMOS OVR score. 最小 OVR。
        min_confidence: Minimum ASR avg confidence (skipped when None).
            ASR 最小平均置信度（confidence 为 None 时跳过该项）。
        require_no_clipping: Drop chunks with detected clipping.
            是否必须无削波。
    """

    min_mos_ovr: float = DEFAULT_MIN_MOS_OVR
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


def _bucket_energy_adaptive(rms: float, p33: float, p66: float) -> str:
    """Categorize chunk energy by adaptive (per-ingest) RMS terciles.

    按本次 ingest 的 RMS 33/66 分位把 chunk 分到 quiet/normal/loud 三档。

    Why adaptive: post-Demucs RMS 范围因 source / 麦克风 / 演讲风格 而异，
    固定阈值会在某些 corpus 上把所有 chunk 都塞进同一档（ADR-0011）。
    Raw `energy_rms` field 仍写入 manifest，下游需要绝对值时直接用。
    Why adaptive: post-Demucs RMS scale varies by source/mic/style, so a
    fixed threshold collapses entire corpora into one bucket (ADR-0011).
    The raw energy_rms value is still in the manifest for any downstream
    that needs the absolute number.

    Args:
        rms: Per-chunk RMS in [0, 1].
        p33: 33rd-percentile RMS across the current ingest.
        p66: 66th-percentile RMS across the current ingest.

    Returns:
        One of "quiet" / "normal" / "loud".
    """
    if rms < p33:
        return "quiet"
    if rms < p66:
        return "normal"
    return "loud"


def _rms(audio: np.ndarray) -> float:
    """Compute the RMS of a 1-D float audio array.

    计算单声道 float 音频的 RMS（带极小 epsilon 防零）。
    """
    return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)) + 1e-10)


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


def _text_hash(text: str) -> str:
    """Stable short hash of normalized text for near-duplicate detection.

    返回标准化文本的稳定短哈希，用于近似重复检测。

    Normalization: lowercase + collapse whitespace + strip surrounding
    punctuation. Hash is first 16 hex chars of SHA-1 (16^16 ≈ 1.8e19
    keyspace, ample for dataset-scale dedup).

    标准化：小写 + 折叠空白 + 去首尾标点。哈希取 SHA-1 前 16 hex 字符
    （键空间约 1.8e19，dataset 量级去重足够）。
    """
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    normalized = normalized.strip(".,!?；。，！？\"'“”‘’")
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def _safe_float(value: float) -> float | None:
    """Coerce NaN / +-inf to None so the manifest JSON stays valid.

    把 NaN / ±inf 转成 None，保证 manifest JSON 合法（json 不支持这些值）。
    """
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return float(value)


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

        # Precompute neighbor chunk ids within each source_file so we
        # can link sentences for cross-sentence prosody. Built once up
        # front so the per-chunk loop can do O(1) lookups.
        # 预先按 source_file 排序 chunk，记录每个 chunk 的前后邻居 id；
        # 主循环只做 O(1) 查表。同源邻居用于跨句韵律 / 上下文 prompt。
        neighbors = self._build_neighbor_index(state.chunks)

        rows: list[dict] = []
        kept = 0
        dropped: Counter[str] = Counter()
        seen_phonemes: set[str] = set()
        bucket_dur: Counter[str] = Counter()
        bucket_prosody: Counter[str] = Counter()
        lang_tally: Counter[str] = Counter()
        emotion_tally: Counter[str] = Counter()

        # Pass 1: ASR / quality / prosody for every chunk; bucket labels
        # that depend on corpus-wide percentiles (energy_bucket) are
        # filled in pass 2 once we know p33/p66.
        # 第一遍：转写、质量、韵律。依赖 corpus 分位的 energy_bucket
        # 留到第二遍补上。
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
            reason = self._filter_reason(tr, mos, clipped)
            if reason is not None:
                dropped[reason] += 1
                logger.debug("DatasetAgent: drop %s (%s)", chunk.chunk_id, reason)
                continue

            # T1 derived: duration / prosody (cheap, single-chunk)
            # T1 派生：时长 / 韵律标签（单 chunk 即可，零跨样本依赖）
            dur = chunk.end_sec - chunk.start_sec
            dur_bucket = _bucket_duration(dur)
            prosody_label = _prosody_label(tr.text)

            bucket_dur[dur_bucket] += 1
            bucket_prosody[prosody_label] += 1
            seen_phonemes |= _phonemes(tr.text, tr.lang)
            lang_tally[tr.lang] += 1

            # T2: prosody + emotion. emotion2vec lazy-loads its model on
            # the first call; subsequent chunks reuse the cached session.
            # T2：韵律 + 情绪。emotion2vec 首次调用惰性加载，后续 chunk 复用缓存。
            pf = prosody_mod.score_prosody(chunk.path, text=tr.text, lang=tr.lang)
            state.prosody[chunk.chunk_id] = ProsodyScore(
                chunk_id=chunk.chunk_id,
                pitch_mean_hz=pf.pitch_mean_hz,
                pitch_std_hz=pf.pitch_std_hz,
                energy_rms=pf.energy_rms,
                loudness_lufs=pf.loudness_lufs,
                speech_ratio=pf.speech_ratio,
                pace_units_per_sec=pf.pace_units_per_sec,
                emotion_tag=pf.emotion_tag,
                emotion_confidence=pf.emotion_confidence,
            )
            emotion_tally[pf.emotion_tag] += 1

            prev_id, next_id = neighbors.get(chunk.chunk_id, (None, None))

            rows.append({
                # Schema / housekeeping
                "manifest_version": MANIFEST_VERSION,
                # Identity
                "chunk_id": chunk.chunk_id,
                "audio_path": str(chunk.path),
                "source_file": str(chunk.source_file),
                "speaker_id": DEFAULT_SPEAKER_ID,
                "prev_chunk_id": prev_id,
                "next_chunk_id": next_id,
                # Text
                "text": tr.text,
                "text_hash": _text_hash(tr.text),
                "lang": tr.lang,
                "confidence": tr.confidence,
                # Timing
                "start_sec": chunk.start_sec,
                "end_sec": chunk.end_sec,
                "duration": dur,
                "duration_bucket": dur_bucket,
                # Quality (existing)
                "snr_db": snr,
                "mos_ovr": mos.ovr,
                "mos_sig": mos.sig,
                "mos_bak": mos.bak,
                "clipped": clipped,
                # T1 prosody buckets (energy_bucket filled in pass 2 below)
                "energy_bucket": None,
                "prosody_label": prosody_label,
                # T2 prosody / emotion
                "energy_rms": _safe_float(pf.energy_rms),
                "loudness_lufs": _safe_float(pf.loudness_lufs),
                "speech_ratio": _safe_float(pf.speech_ratio),
                "pitch_mean_hz": _safe_float(pf.pitch_mean_hz),
                "pitch_std_hz": _safe_float(pf.pitch_std_hz),
                "pace_units_per_sec": _safe_float(pf.pace_units_per_sec),
                "emotion_tag": pf.emotion_tag,
                "emotion_confidence": _safe_float(pf.emotion_confidence),
            })
            kept += 1

        # Pass 2: derive corpus-wide RMS terciles from this ingest's
        # surviving chunks, then assign energy_bucket per row. Adaptive
        # rather than fixed thresholds, see ADR-0011.
        # 第二遍：基于本次 ingest 的存活 chunk RMS 算 33/66 分位，
        # 给每行补 energy_bucket。自适应而非硬阈值，见 ADR-0011。
        rms_values = [r["energy_rms"] for r in rows if r["energy_rms"] is not None]
        if rms_values:
            arr = np.asarray(rms_values, dtype=np.float64)
            p33 = float(np.percentile(arr, 33))
            p66 = float(np.percentile(arr, 66))
        else:
            p33 = p66 = 0.0
        bucket_energy: Counter[str] = Counter()
        for row in rows:
            rms = row["energy_rms"]
            label = _bucket_energy_adaptive(rms, p33, p66) if rms is not None else "unknown"
            row["energy_bucket"] = label
            bucket_energy[label] += 1

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
            emotion_tally=emotion_tally,
            energy_p33=p33, energy_p66=p66,
        )
        report_path.write_text(report, encoding="utf-8")

        logger.info(
            "DatasetAgent: kept %d/%d (dropped %s) -> %s",
            kept, len(state.chunks), dict(dropped), manifest_path,
        )
        return state

    @staticmethod
    def _build_neighbor_index(
        chunks: list,
    ) -> dict[str, tuple[str | None, str | None]]:
        """Map chunk_id → (prev_chunk_id, next_chunk_id) within source_file.

        在同一 source_file 内按起点排序，给每个 chunk 计算前后邻居 id。
        跨 source_file 不连接（不同录音的"邻居"对韵律没意义）。

        Args:
            chunks: List[ChunkInfo].

        Returns:
            dict mapping chunk_id to (prev_id, next_id); endpoints get
            None on the missing side.
            chunk_id → (前邻 id, 后邻 id) 的字典；端点一侧用 None。
        """
        by_source: dict[Path, list] = {}
        for c in chunks:
            by_source.setdefault(c.source_file, []).append(c)
        index: dict[str, tuple[str | None, str | None]] = {}
        for group in by_source.values():
            group.sort(key=lambda c: c.start_sec)
            for i, c in enumerate(group):
                prev_id = group[i - 1].chunk_id if i > 0 else None
                next_id = group[i + 1].chunk_id if i < len(group) - 1 else None
                index[c.chunk_id] = (prev_id, next_id)
        return index

    def _filter_reason(
        self, tr: asr_mod.TranscriptResult,
        mos: eval_mod.DnsmosScores, clipped: bool,
    ) -> str | None:
        """Return the drop reason for a chunk, or None if it passes.

        判断单 chunk 是否被过滤；返回原因字符串，None 表示通过。

        SNR 不参与过滤（见 ADR-0009），仅记录到 manifest 用于诊断，
        所以不在本函数签名里出现。
        SNR is recorded for diagnostics but never gates here (ADR-0009),
        so it's not in this function signature.
        """
        if self.thresholds.require_no_clipping and clipped:
            return "clipping"
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
        emotion_tally: Counter[str],
        energy_p33: float = 0.0, energy_p66: float = 0.0,
    ) -> str:
        """Render the human-readable report.md content.

        渲染 report.md 内容（含质量统计 + 多样性 + 音素覆盖率 + 情绪分布）。
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

        # Emotion lines: report all labels seen, sorted by frequency.
        # 情绪行：列出所有出现过的标签，按频率降序。
        emotion_lines = [
            f"  - {tag}: {n}"
            for tag, n in emotion_tally.most_common()
        ] or ["  - (none)"]

        return (
            f"# Dataset report\n\n"
            f"- Manifest version: **{MANIFEST_VERSION}**\n"
            f"- Kept chunks: **{kept}** / {total}\n"
            f"- Drop reasons: {dict(dropped) or '(none)'}\n"
            f"- Languages: {dict(lang_tally) or '(none)'}\n\n"
            f"## Phoneme coverage\n\n"
            + "\n".join(coverage_lines)
            + f"\n\n## Duration buckets\n\n"
            f"  - short  (<5s): {bucket_dur['short']}\n"
            f"  - medium (5-10s): {bucket_dur['medium']}\n"
            f"  - long   (>=10s): {bucket_dur['long']}\n\n"
            f"## Energy buckets (adaptive, ADR-0011)\n\n"
            f"  Thresholds this ingest: p33={energy_p33:.4f} / p66={energy_p66:.4f}\n"
            f"  - quiet:  {bucket_energy['quiet']}\n"
            f"  - normal: {bucket_energy['normal']}\n"
            f"  - loud:   {bucket_energy['loud']}\n\n"
            f"## Prosody (terminal punctuation)\n\n"
            f"  - declarative: {bucket_prosody['declarative']}\n"
            f"  - question:    {bucket_prosody['question']}\n"
            f"  - exclamation: {bucket_prosody['exclamation']}\n\n"
            f"## Emotion distribution (emotion2vec)\n\n"
            + "\n".join(emotion_lines)
            + "\n"
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
