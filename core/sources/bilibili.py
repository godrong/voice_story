"""Bilibili source: download audio via the isolated `tools.bilibili_mcp` extractor.

This is the Source-protocol adapter for the standalone extractor. The actual
yt-dlp logic lives in `tools/bilibili_mcp/extractor.py` so it can be reused by
the MCP server. This file just adapts that core into the `Source` protocol
SourceAgent already knows how to drive.

Default output layout: `<inputs_root>/<name>/raw/<bvid>[_pN][_T...].wav`.
That matches what LocalSource would scan, so SourceAgent transparently
standardizes to 24 kHz mono WAV via `core.audio_io.to_standard_wav`.

B 站源插件 —— 通过隔离的 `tools.bilibili_mcp` extractor 下载音频。

真正的 yt-dlp 逻辑在 `tools/bilibili_mcp/extractor.py` 里（同时被 MCP server
复用）。这里只是把那个核心适配成 SourceAgent 已认得的 Source Protocol。

默认输出路径：`<inputs_root>/<name>/raw/<bvid>[_pN][_T...].wav`，
和 LocalSource 扫描的目录一致，下游 SourceAgent 会无感地通过
`core.audio_io.to_standard_wav` 标准化成 24 kHz 单声道 WAV。
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from . import SourceMeta


class BilibiliSource:
    """Yield audio files downloaded from a Bilibili URL.

    把 B 站 URL 当成"源"产出本地音频文件路径。

    Args:
        name: Dataset short name; output goes to `inputs/<name>/raw/`.
              数据集短名；产物落到 inputs/<name>/raw/。
        url: B 站 URL 或 BV 号；可以是单视频、多 P、合集 URL。
        inputs_root: 输入根目录，默认 ./inputs（与 LocalSource 对齐）。
        time_range: (start_sec, end_sec) 浮点秒数；None=全片。
                    多 P/合集时会应用到选中的每一 P（一般只对单视频有用）。
        parts: 多 P/合集时的 1-based 子项索引列表；None=全下。
        fetch_subtitles: 是否同时拉官方字幕（落到 raw/ 同目录）。
        sub_langs: 字幕语言优先级。
        lang_hint: 透传给 ASR 的 ISO 639-1 提示，与其它 Source 一致。
        needs_separation: 是否跑 Demucs，默认 True（B 站视频经常有 BGM）。
        is_single_speaker: 是否单说话人，默认 False（演讲/访谈一般 True 由用户传）。
        license: 授权说明字符串（仅记录用）。
        overwrite: 已存在同名输出时是否重新下载。
    """

    def __init__(
        self,
        name: str,
        *,
        url: str,
        inputs_root: Path | str = "inputs",
        time_range: tuple[float, float] | None = None,
        parts: list[int] | None = None,
        fetch_subtitles: bool = False,
        sub_langs: tuple[str, ...] = ("zh-CN", "zh-Hans", "en"),
        lang_hint: str | None = None,
        needs_separation: bool = True,
        is_single_speaker: bool = False,
        license: str = "Bilibili content; respect uploader terms and B 站 ToS",
        overwrite: bool = False,
    ) -> None:
        self.url = url
        self.time_range = time_range
        self.parts = parts
        self.fetch_subtitles = fetch_subtitles
        self.sub_langs = sub_langs
        self.overwrite = overwrite
        # We download straight into raw/ so SourceAgent's to_standard_wav call
        # in `<dataset_root>/raw/` doesn't double-copy. SourceAgent reads from
        # the file paths we yield and writes the standardized wav next to it.
        # 直接下到 raw/ 目录；SourceAgent 拿到我们 yield 的 Path 后会原地
        # 走 to_standard_wav，不会再多一次拷贝。
        self.out_dir = Path(inputs_root) / name / "raw"
        self.meta = SourceMeta(
            source_name=name,
            lang_hint=lang_hint,
            needs_separation=needs_separation,
            is_single_speaker=is_single_speaker,
            license=license,
            extra={
                "url": url,
                "time_range": list(time_range) if time_range else None,
                "parts": list(parts) if parts else None,
                "out_dir": str(self.out_dir),
            },
        )

    def fetch(self) -> Iterable[Path]:
        """Download audio via the extractor and yield resulting file paths.

        调用 extractor 下载音频，逐个 yield 落盘的文件路径。

        Yields:
            每个下载产物的本地 Path（音频文件）。
            单 P 视频产 1 个；多 P/合集按 `parts` 选择数产 N 个。

        Raises:
            BilibiliExtractError: 下载或转码失败时由 extractor 抛出。
        """
        # Lazy import to keep `from core.sources import REGISTRY` cheap
        # even when yt-dlp / mcp aren't installed in the env.
        # 延迟导入：让 REGISTRY 在未装 yt-dlp/mcp 时也能正常 import。
        from tools.bilibili_mcp import extractor

        # Probe once to decide single vs multi-P. Single-P with no `parts`
        # filter goes the simpler single-call path.
        # 先 probe 一次判断单/多 P；单 P 且没指定 parts 走更简单的单调用路径。
        info = extractor.probe(self.url)
        is_multi_p = len(info.parts) > 1 or self.parts is not None

        if is_multi_p:
            results = extractor.extract_playlist(
                self.url,
                self.out_dir,
                parts=self.parts,
                time_range=self.time_range,
                audio_format="wav",
                fetch_subtitles=self.fetch_subtitles,
                sub_langs=self.sub_langs,
                overwrite=self.overwrite,
            )
        else:
            results = [
                extractor.extract_audio(
                    self.url,
                    self.out_dir,
                    time_range=self.time_range,
                    audio_format="wav",
                    fetch_subtitles=self.fetch_subtitles,
                    sub_langs=self.sub_langs,
                    overwrite=self.overwrite,
                )
            ]

        for r in results:
            yield r.audio_path
