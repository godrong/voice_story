"""Text normalization for TTS reading — fix what makes CosyVoice 2 stumble.

Scope (intentionally narrow per the project decision to do "B layer" only):

  1. Unicode artifacts → ASCII equivalents:
     curly quotes ('"') → straight ('"'), em / en dashes (—–) → '-',
     ellipsis (…) → '...', non-breaking + zero-width spaces stripped.
     CosyVoice 2's English text frontend handles ASCII much better than
     Unicode punctuation; smart quotes especially confuse the G2P.

  2. English contraction expansion:
     "he's" → "he is", "couldn't" → "could not", etc. Done because the
     model's prosody around contractions is awkward — it knows the right
     phonemes but produces a stilted rhythm. Expanding adds one syllable
     and gives the model fuller context to compose natural prosody.

  3. Whitespace collapse: any run of whitespace → single space, trim ends.

What this module does NOT do (out of scope by design):

  - Spell correction (misspellings get read literally; user fixes manually
    or runs a dedicated spell checker upstream)
  - Number / date / abbreviation expansion (CosyVoice 2 frontend handles
    common cases; we revisit if it falls down)
  - Style transfer / persona rewrite (changes meaning — handled at
    a higher layer, see ADR-0010 style_agent)
  - Chinese segmentation / pinyin annotation (CosyVoice 2 ZH frontend
    already strong; only do unicode pass)

TTS 朗读用的文本规范化模块（"B 层"，只做让 CosyVoice 不卡的部分）。

范围：unicode → ASCII（弯引号 / 破折号 / 省略号），英文缩写展开
（he's → he is，缩写本身发音对但节奏卡），空白压缩。

不做：拼写纠错 / 数字日期展开 / 风格改写 / 中文分词——这些另起 ADR。
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


# Smart-quote / dash / ellipsis / zero-width replacements.
# CosyVoice 2 handles ASCII punctuation reliably; Unicode punct trips its G2P.
# CosyVoice 2 处理 ASCII 标点稳；Unicode 标点会让 G2P 出错。
_UNICODE_FIXES: dict[str, str] = {
    "‘": "'", "’": "'",        # ' ' curly single quotes
    "“": '"', "”": '"',        # " " curly double quotes
    "–": "-", "—": "-",        # – — en/em dashes
    "…": "...",                     # … ellipsis
    " ": " ",                       # non-breaking space
    "​": "", "‌": "", "‍": "",  # zero-width chars
    "﻿": "",                        # BOM
}


def _strip_unicode_artifacts(text: str) -> str:
    """Replace fancy Unicode punctuation with ASCII equivalents + collapse whitespace.

    把弯引号 / 破折号 / 省略号 / 各种零宽字符替换成 ASCII，
    同时把多余空白压成单空格。修改前提：原文本身用了 unicode 标点。

    Args:
        text: Source text, possibly with smart quotes / em-dashes / etc.

    Returns:
        Cleaned text. Empty input returns empty.
    """
    if not text:
        return ""
    for needle, replacement in _UNICODE_FIXES.items():
        text = text.replace(needle, replacement)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _expand_contractions_en(text: str) -> str:
    """Expand English contractions (he's → he is, couldn't → could not).

    用 `contractions` 库展开英文缩写。库不在时退回到内置 ~30 条常用表，
    覆盖常见 case；任何场景都不会抛错。

    Why expand: CosyVoice 2 has correct G2P for "he's" but produces an
    awkward / robotic prosody around contractions in long-form English
    output. Expanding gives the model an extra syllable to compose
    natural rhythm. Cost: slight loss of casual register.

    为什么展开：CosyVoice 2 在英文长篇里对缩写的节奏处理偏生硬
    （音素对但韵律卡）。展开让模型多一个音节做韵律布置。
    代价：损失一点口语感。

    Args:
        text: English text. ASCII apostrophes only (run unicode pass first).

    Returns:
        Text with contractions expanded.
    """
    try:
        import contractions  # type: ignore[import-not-found]
        return contractions.fix(text)
    except ImportError:
        logger.warning("`contractions` not installed; using fallback table")
        return _expand_contractions_fallback(text)


# Fallback table covers ~30 most common English contractions.
# Used only when `contractions` lib isn't installed; case-insensitive match.
# 内置兜底表覆盖最常见 ~30 条英文缩写；contractions 包不在时启用。
_FALLBACK_PATTERNS: list[tuple[str, str]] = [
    (r"\bI'm\b", "I am"), (r"\byou're\b", "you are"),
    (r"\bhe's\b", "he is"), (r"\bshe's\b", "she is"),
    (r"\bit's\b", "it is"), (r"\bwe're\b", "we are"),
    (r"\bthey're\b", "they are"), (r"\bthat's\b", "that is"),
    (r"\bthere's\b", "there is"), (r"\bwhat's\b", "what is"),
    (r"\bwho's\b", "who is"), (r"\bhow's\b", "how is"),
    (r"\bcouldn't\b", "could not"), (r"\bwouldn't\b", "would not"),
    (r"\bshouldn't\b", "should not"), (r"\bdidn't\b", "did not"),
    (r"\bdoesn't\b", "does not"), (r"\bisn't\b", "is not"),
    (r"\baren't\b", "are not"), (r"\bwasn't\b", "was not"),
    (r"\bweren't\b", "were not"), (r"\bcan't\b", "cannot"),
    (r"\bwon't\b", "will not"), (r"\bdon't\b", "do not"),
    (r"\bI've\b", "I have"), (r"\bwe've\b", "we have"),
    (r"\byou've\b", "you have"), (r"\bthey've\b", "they have"),
    (r"\bI'll\b", "I will"), (r"\bwe'll\b", "we will"),
    (r"\byou'll\b", "you will"), (r"\bhe'll\b", "he will"),
    (r"\bshe'll\b", "she will"), (r"\bthey'll\b", "they will"),
    (r"\bI'd\b", "I would"), (r"\bwe'd\b", "we would"),
]


def _expand_contractions_fallback(text: str) -> str:
    """Internal fallback when the `contractions` library is unavailable.

    内置兜底实现；按 case-insensitive 匹配 _FALLBACK_PATTERNS 表。
    """
    for pattern, expansion in _FALLBACK_PATTERNS:
        text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
    return text


def normalize_for_tts(text: str, *, lang: str = "en") -> str:
    """One-shot text cleanup before sending to a TTS backend.

    给 TTS 后端发文本前的一站式清洗。英文走 unicode + 缩写展开两步，
    中文只走 unicode 步（中文无英文式缩写）。语义保持不变，只改朗读形式。

    Args:
        text: Source text (possibly with smart quotes, contractions, etc).
        lang: ISO 639-1 hint. "en" → unicode + contraction expansion.
              "zh" or anything else → unicode pass only.

    Returns:
        TTS-ready text. Returns "" for empty input.
    """
    if not text:
        return ""
    text = _strip_unicode_artifacts(text)
    if lang.lower().startswith("en"):
        text = _expand_contractions_en(text)
    return text
