"""Unit tests for agents.dataset_agent's pure helpers.

These cover the bucketing / phoneme / report logic that does NOT require
any model. The full async DatasetAgent.run() is exercised in higher-level
integration tests (not in M1 — added in M3 with eval framework).

agents.dataset_agent 中"纯函数"部分的单元测试。

不需要模型的辅助函数（分桶 / 音素抽取 / report 渲染）都在这里覆盖；
完整 async run() 留给 M3 的 eval 集成测试。
"""

from __future__ import annotations

from collections import Counter

from agents import dataset_agent as da


def test_bucket_duration_short_medium_long():
    """Duration boundaries: <5s short, [5,10) medium, >=10 long.

    时长分桶边界：<5 短，[5,10) 中，>=10 长。
    """
    assert da._bucket_duration(2.5) == "short"
    assert da._bucket_duration(7.5) == "medium"
    assert da._bucket_duration(12.0) == "long"
    assert da._bucket_duration(5.0) == "medium"
    assert da._bucket_duration(10.0) == "long"


def test_bucket_energy_adaptive_terciles():
    """Adaptive bucket uses caller-supplied p33/p66 to label any RMS.

    自适应分桶按调用方给的 p33/p66 阈值标记任意 RMS 值（ADR-0011）。
    边界点遵循 [<p33 quiet | <p66 normal | >=p66 loud) 的半开区间约定。
    """
    p33, p66 = 0.020, 0.025
    assert da._bucket_energy_adaptive(0.010, p33, p66) == "quiet"
    assert da._bucket_energy_adaptive(0.022, p33, p66) == "normal"
    assert da._bucket_energy_adaptive(0.030, p33, p66) == "loud"
    # Boundary cases. 边界值。
    assert da._bucket_energy_adaptive(p33, p33, p66) == "normal"
    assert da._bucket_energy_adaptive(p66, p33, p66) == "loud"


def test_rms_helper_matches_numpy():
    """`_rms` matches sqrt(mean(x^2)) for a tone.

    `_rms` 与 sqrt(mean(x^2)) 公式一致；正弦波 0.5 amp 的 RMS ≈ 0.354。
    """
    import numpy as np
    sr = 16000
    t = np.arange(sr) / sr
    audio = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    expected = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    assert abs(da._rms(audio) - expected) < 1e-3


def test_prosody_label_supports_chinese_and_english_punct():
    """Question / exclamation detected for both ZH and EN punctuation.

    问号与感叹号在中英文标点下都能识别。
    """
    assert da._prosody_label("Hello?") == "question"
    assert da._prosody_label("你好？") == "question"
    assert da._prosody_label("Wow!") == "exclamation"
    assert da._prosody_label("好棒！") == "exclamation"
    assert da._prosody_label("This is fine.") == "declarative"
    assert da._prosody_label("") == "declarative"


def test_phoneme_universe_sizes():
    """Phoneme universes are non-empty and CN > EN by count.

    音素总集非空，且中文集合大于英文集合（声韵母多于 ARPAbet）。
    """
    en = da._phoneme_universe("en")
    zh = da._phoneme_universe("zh")
    assert len(en) > 30
    assert len(zh) > len(en)


def test_text_hash_is_normalization_invariant():
    """Whitespace / case / surrounding punctuation don't change the hash.

    text_hash 对空白 / 大小写 / 首尾标点鲁棒，便于近似重复检测。
    """
    h1 = da._text_hash("Hello, world.")
    h2 = da._text_hash("  hello,    world.   ")
    h3 = da._text_hash("HELLO, WORLD")
    assert h1 == h2 == h3
    # Different content gives different hash.
    # 内容不同应得到不同 hash。
    assert da._text_hash("Goodbye, world.") != h1
    # 16-char hex.
    assert len(h1) == 16 and all(c in "0123456789abcdef" for c in h1)


def test_safe_float_handles_nan_and_inf():
    """NaN / inf become None; finite floats pass through.

    NaN / ±inf 被转成 None；有限浮点原样返回，None 也保留为 None。
    """
    import math
    assert da._safe_float(math.nan) is None
    assert da._safe_float(math.inf) is None
    assert da._safe_float(-math.inf) is None
    assert da._safe_float(None) is None
    assert da._safe_float(3.14) == 3.14
    assert da._safe_float(0.0) == 0.0


def test_build_neighbor_index_chains_within_source():
    """Neighbors link by start_sec within source_file; cross-source isolated.

    邻居在 source_file 内部按起点连接，跨 source_file 不连通；端点为 None。
    """
    from pathlib import Path
    from agents.state import ChunkInfo

    chunks = [
        ChunkInfo("a1", Path("/p/a1.wav"), Path("/srcA"), 0.0, 5.0),
        ChunkInfo("a2", Path("/p/a2.wav"), Path("/srcA"), 5.0, 10.0),
        ChunkInfo("a3", Path("/p/a3.wav"), Path("/srcA"), 10.0, 15.0),
        ChunkInfo("b1", Path("/p/b1.wav"), Path("/srcB"), 0.0, 5.0),
    ]
    idx = da.DatasetAgent._build_neighbor_index(chunks)
    assert idx["a1"] == (None, "a2")
    assert idx["a2"] == ("a1", "a3")
    assert idx["a3"] == ("a2", None)
    # Cross-source isolated; b1 is a singleton in its group.
    # 跨 source 隔离；b1 在其组内是单例。
    assert idx["b1"] == (None, None)


def test_filter_thresholds_defaults():
    """FilterThresholds defaults reflect ADR-0009 (post-Demucs calibration).

    默认门槛值反映 ADR-0009：去掉 SNR、MOS 降到 3.0。
    """
    t = da.FilterThresholds()
    assert t.min_mos_ovr == 3.0
    assert t.min_confidence == 0.85
    assert t.require_no_clipping is True
    assert not hasattr(t, "min_snr_db"), "min_snr_db should be removed (ADR-0009)"


def test_render_report_includes_key_sections():
    """report.md contains the headline counts + section headers.

    report 渲染包含关键统计块和章节标题（含 manifest v1.1 的 emotion 段）。
    """
    agent = da.DatasetAgent()
    txt = agent._render_report(  # noqa: SLF001
        kept=42, total=50,
        dropped=Counter({"low_mos": 5, "low_snr": 3}),
        lang_tally=Counter({"en": 42}),
        bucket_dur=Counter({"short": 10, "medium": 22, "long": 10}),
        bucket_energy=Counter({"quiet": 5, "normal": 30, "loud": 7}),
        bucket_prosody=Counter({"declarative": 30, "question": 7, "exclamation": 5}),
        seen_phonemes={"AH", "EH", "P", "T"},
        emotion_tally=Counter({"neutral": 30, "happy": 8, "unknown": 4}),
    )
    assert f"Manifest version: **{da.MANIFEST_VERSION}**" in txt
    assert "**42** / 50" in txt
    assert "## Phoneme coverage" in txt
    assert "## Duration buckets" in txt
    assert "## Energy buckets" in txt
    assert "## Prosody (terminal punctuation)" in txt
    assert "## Emotion distribution" in txt
    assert "neutral: 30" in txt
    assert "low_mos" in txt
