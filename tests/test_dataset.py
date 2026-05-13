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


def test_bucket_energy_quiet_normal_loud():
    """Energy bucket maps RMS to {quiet, normal, loud}.

    能量分桶把 RMS 映射到 {quiet, normal, loud}。
    """
    import numpy as np
    quiet = np.full(1000, 0.01, dtype=np.float32)
    normal = np.full(1000, 0.10, dtype=np.float32)
    loud = np.full(1000, 0.50, dtype=np.float32)
    assert da._bucket_energy(quiet) == "quiet"
    assert da._bucket_energy(normal) == "normal"
    assert da._bucket_energy(loud) == "loud"


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


def test_filter_thresholds_defaults():
    """FilterThresholds defaults match the v0.1 plan numbers.

    默认门槛值与 v0.1 规划一致。
    """
    t = da.FilterThresholds()
    assert t.min_mos_ovr == 3.5
    assert t.min_snr_db == 15.0
    assert t.min_confidence == 0.85
    assert t.require_no_clipping is True


def test_render_report_includes_key_sections():
    """report.md contains the headline counts + section headers.

    report 渲染包含关键统计块和章节标题。
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
    )
    assert "**42** / 50" in txt
    assert "## Phoneme coverage" in txt
    assert "## Duration buckets" in txt
    assert "## Energy buckets" in txt
    assert "## Prosody (terminal punctuation)" in txt
    assert "low_mos" in txt
