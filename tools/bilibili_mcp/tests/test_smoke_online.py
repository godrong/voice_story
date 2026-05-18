"""Online smoke test — real Bilibili download. Skipped by default.

Run manually with:

    conda run -n ai_study pytest tools/bilibili_mcp/tests/test_smoke_online.py \
        -v -m online

This will hit the live B 站 CDN. If it fails with 412/429 or "format not
found", update yt-dlp first: `pip install -U yt-dlp`.

在线烟测 —— 真下载 B 站音频，默认 skip。手动跑：

    conda run -n ai_study pytest tools/bilibili_mcp/tests/test_smoke_online.py \
        -v -m online

会打 B 站 CDN 真请求。报 412/429/format-not-found 时先 `pip install -U yt-dlp`。
"""

from __future__ import annotations

from pathlib import Path

import pytest


# A short, public Bilibili video. If this gets deleted, replace with another
# short clip (< 30s) before running. We intentionally don't pin a specific
# BV here because B 站 删稿/转私是常态。
#
# 一个公开的短视频。被删了就换一个 30s 内的短片再跑。我们故意不在代码里硬编
# 一个 BV 号 —— B 站删稿/转私是常态。
SMOKE_URL = "https://www.bilibili.com/video/BV1GJ411x7h7"  # placeholder; update before use


@pytest.mark.online
def test_probe_against_live_url():
    pytest.importorskip("yt_dlp")
    from tools.bilibili_mcp import extractor
    info = extractor.probe(SMOKE_URL)
    assert info.bvid, "probe should populate bvid"
    assert info.duration > 0, "probe should populate duration"
    assert info.parts, "parts list should have at least one entry"


@pytest.mark.online
def test_extract_short_segment(tmp_path: Path):
    pytest.importorskip("yt_dlp")
    from tools.bilibili_mcp import extractor
    r = extractor.extract_audio(
        SMOKE_URL,
        out_dir=tmp_path,
        time_range=(0.0, 5.0),
        audio_format="wav",
    )
    assert r.audio_path.exists()
    assert r.audio_path.stat().st_size > 1000  # at least some PCM bytes
    assert r.time_range == (0.0, 5.0)
