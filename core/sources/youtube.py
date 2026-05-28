"""YouTube source: download audio via the same yt-dlp extractor as Bilibili.

yt-dlp natively supports YouTube, so we reuse `tools.bilibili_mcp.extractor`
unchanged — only the Source-protocol adapter differs (different default
license string, different default lang hint behaviour, and we don't expose
the `parts` knob since YouTube videos aren't multi-P).

YouTube 源插件 —— 复用与 B 站相同的 yt-dlp extractor。

yt-dlp 原生支持 YouTube，所以 `tools.bilibili_mcp.extractor` 原封不动复用；
本文件只是 Source Protocol 适配器，与 BilibiliSource 的差别仅在默认 license
和不暴露 `parts`（YouTube 视频没有 B 站那种分 P 概念）。
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from . import SourceMeta


class YouTubeSource:
    """Yield audio files downloaded from a YouTube URL.

    把 YouTube URL 当成"源"产出本地音频文件路径。

    Args:
        name: dataset 短名；产物落到 inputs/<name>/raw/。
        url: YouTube URL (youtu.be / youtube.com 均可)。
        inputs_root: 输入根目录，默认 ./inputs（与其它 Source 对齐）。
        time_range: (start_sec, end_sec) 浮点秒数；None=全片。
        fetch_subtitles: 是否同时拉官方/自动字幕（YouTube 的 auto-subs 也算）。
        sub_langs: 字幕语言优先级。
        lang_hint: ISO 639-1 提示，透传给 ASR。
        needs_separation: 是否跑 Demucs，默认 True（多数 YouTube 视频有 BGM）。
        is_single_speaker: 是否单说话人，默认 False。
        license: 授权说明（仅记录用）。
        overwrite: 已存在同名输出时是否重新下载。
    """

    def __init__(
        self,
        name: str,
        *,
        url: str,
        inputs_root: Path | str = "inputs",
        time_range: tuple[float, float] | None = None,
        fetch_subtitles: bool = False,
        sub_langs: tuple[str, ...] = ("en", "zh-Hans", "zh-CN"),
        lang_hint: str | None = None,
        needs_separation: bool = True,
        is_single_speaker: bool = False,
        license: str = "YouTube content; respect uploader terms and YouTube ToS",
        overwrite: bool = False,
    ) -> None:
        self.url = url
        self.time_range = time_range
        self.fetch_subtitles = fetch_subtitles
        self.sub_langs = sub_langs
        self.overwrite = overwrite
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
                "out_dir": str(self.out_dir),
            },
        )

    def fetch(self) -> Iterable[Path]:
        """Download audio via the shared extractor and yield resulting paths.

        通过共享的 extractor 下载音频，逐个 yield 落盘的文件路径。

        Yields:
            每个下载产物的本地 Path（音频文件）。

        Raises:
            BilibiliExtractError: 下载或转码失败时由 extractor 抛出
                （类名虽叫 Bilibili，实际是 yt-dlp 通用错误）。
        """
        # Lazy import so importing the source registry doesn't require yt-dlp.
        # 延迟导入：仅在真正 fetch 时才依赖 yt-dlp。
        from tools.bilibili_mcp import extractor

        result = extractor.extract_audio(
            self.url,
            self.out_dir,
            time_range=self.time_range,
            audio_format="wav",
            fetch_subtitles=self.fetch_subtitles,
            sub_langs=self.sub_langs,
            overwrite=self.overwrite,
        )
        yield result.audio_path
