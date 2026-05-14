"""Unit tests for core.text_norm — TTS-friendly text normalization.

Covers the unicode pass (smart quotes / dashes / ellipsis / zero-width),
English contraction expansion (both with the `contractions` lib and the
internal fallback table), and the language-aware top-level entrypoint.

core.text_norm 单元测试：unicode 清洗 / 英文缩写展开 / 语种路由。
"""

from __future__ import annotations

from core import text_norm


# ---- Unicode artifact stripping ----------------------------------------


def test_strip_smart_single_quotes():
    """Curly single quotes become straight ASCII apostrophes.

    弯单引号 ' ' → ASCII '。
    """
    out = text_norm._strip_unicode_artifacts("It’s great, isn‘t it")
    assert out == "It's great, isn't it"


def test_strip_smart_double_quotes():
    """Curly double quotes become straight ASCII double quotes.

    弯双引号 " " → ASCII "。
    """
    out = text_norm._strip_unicode_artifacts("He said “hello”.")
    assert out == 'He said "hello".'


def test_strip_em_and_en_dashes():
    """Em-dash (—) and en-dash (–) collapse to ASCII hyphen.

    em / en 破折号 → ASCII 连字符。
    """
    out = text_norm._strip_unicode_artifacts("a—b–c")
    assert out == "a-b-c"


def test_strip_ellipsis_to_three_dots():
    """Single-char ellipsis (…) becomes three ASCII dots.

    省略号 … → ASCII 三个点。
    """
    out = text_norm._strip_unicode_artifacts("wait… what?")
    assert out == "wait... what?"


def test_strip_zero_width_chars():
    """Zero-width chars (200B/200C/200D) and BOM are removed silently.

    零宽字符 + BOM 静默删除。
    """
    out = text_norm._strip_unicode_artifacts("hello​world﻿")
    assert out == "helloworld"


def test_collapse_whitespace():
    """Any run of whitespace collapses to a single space; ends trimmed.

    多空白合并成一个空格；首尾去空白。
    """
    out = text_norm._strip_unicode_artifacts("  hello   \t\n  world  ")
    assert out == "hello world"


def test_strip_empty_returns_empty():
    """Empty input is returned as empty string (not None / not error).

    空输入返回空字符串，不抛错。
    """
    assert text_norm._strip_unicode_artifacts("") == ""


# ---- English contraction expansion -------------------------------------


def test_expand_common_contractions():
    """Common contractions expand to their full forms via the library.

    常见缩写经库展开成完整形式。
    """
    out = text_norm._expand_contractions_en("I'm here, he's there, we're nowhere.")
    assert "I am" in out
    assert "he is" in out
    assert "we are" in out


def test_expand_negative_contractions():
    """Negative contractions (couldn't / wasn't / etc.) expand correctly.

    否定缩写展开。
    """
    out = text_norm._expand_contractions_en("She couldn't, wouldn't, didn't go.")
    assert "could not" in out
    assert "would not" in out
    assert "did not" in out


def test_expand_contractions_preserves_quotes():
    """Quoted strings without apostrophes pass through cleanly.

    无撇号的引号字符串原样穿过，不被误处理。
    """
    text = 'He said "hello world" and left.'
    out = text_norm._expand_contractions_en(text)
    assert out == text


def test_fallback_table_handles_basics():
    """The internal fallback table works without `contractions` installed.

    内置兜底表（contractions 包不在时启用）覆盖最常见缩写。
    """
    # Test the fallback path directly to keep CI predictable.
    # 直接测兜底实现，不依赖 contractions 包是否装。
    out = text_norm._expand_contractions_fallback(
        "I'm a developer. He's smart. They're learning. We can't stop."
    )
    assert "I am" in out
    assert "he is" in out.lower()
    assert "they are" in out.lower()
    assert "cannot" in out.lower()


# ---- normalize_for_tts (top-level entrypoint) --------------------------


def test_normalize_en_does_both_passes():
    """`normalize_for_tts(lang='en')` runs both unicode and contraction passes.

    EN 路径执行 unicode 清洗 + 缩写展开两步。
    """
    out = text_norm.normalize_for_tts(
        "It’s great… he’s smart!", lang="en",
    )
    # Unicode normalized: curly → ASCII; ellipsis → "..."
    # 缩写展开: It's → It is; he's → he is
    assert "..." in out
    assert "It is" in out or "it is" in out
    assert "he is" in out


def test_normalize_zh_only_unicode_pass():
    """For Chinese, only the unicode pass runs (no English-style contractions).

    中文 lang 只跑 unicode 步，不动文本本身。
    """
    out = text_norm.normalize_for_tts("“你好”…世界", lang="zh")
    # Smart quotes → ASCII; ellipsis → "..."
    # 弯引号 → ASCII；省略号 → 三个点。
    assert out == '"你好"...世界'


def test_normalize_empty_returns_empty():
    """Empty / falsy input returns empty without error.

    空输入返回空。
    """
    assert text_norm.normalize_for_tts("", lang="en") == ""
    assert text_norm.normalize_for_tts("", lang="zh") == ""


def test_normalize_no_op_on_clean_ascii():
    """ASCII text without contractions passes through unchanged.

    干净 ASCII 文本（无缩写、无 unicode）应无变化。
    """
    text = "The quick brown fox jumps over the lazy dog."
    assert text_norm.normalize_for_tts(text, lang="en") == text


def test_normalize_lang_inference_default_en():
    """Default lang is 'en' (most common case for current pipeline).

    默认 lang='en'；当前 pipeline 主要场景。
    """
    out = text_norm.normalize_for_tts("I'm fine.")
    assert "I am" in out
