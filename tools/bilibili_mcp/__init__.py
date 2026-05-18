"""Bilibili audio-extraction agent: yt-dlp + ffmpeg wrapped behind MCP & Source.

This package is intentionally decoupled from the main voice_story pipeline.
It exposes:

- `extractor`: pure-Python functions (probe / extract_audio / extract_playlist /
  fetch_subtitles_only) that wrap yt-dlp's Python API.
- `server`: a stdio MCP server registering those functions as Claude-callable
  tools. Started by `.mcp.json` at the project root.

The same functions are reused by `core/sources/bilibili.py` so the main TTS
pipeline can consume B 站 URLs as a Source plugin alongside `local` / `kaggle`.

B 站音频提取 agent —— yt-dlp + ffmpeg 的双形态封装。

本包刻意独立于 voice_story 主管线（见 ADR-0003 把下载器隔离的决定）：

- `extractor`：纯 Python 函数（probe / extract_audio / extract_playlist /
  fetch_subtitles_only），是 yt-dlp Python API 的包装层。
- `server`：stdio 协议的 MCP server，把上面的函数注册成 Claude Code 可调用
  的工具。由项目根的 `.mcp.json` 启动。

同一组函数被 `core/sources/bilibili.py` 复用，让主管线可以像消费
`local` / `kaggle` 一样消费 B 站 URL。
"""

from .extractor import (
    BilibiliExtractError,
    ExtractResult,
    PartInfo,
    VideoInfo,
    extract_audio,
    extract_playlist,
    fetch_subtitles_only,
    probe,
)

__all__ = [
    "BilibiliExtractError",
    "ExtractResult",
    "PartInfo",
    "VideoInfo",
    "extract_audio",
    "extract_playlist",
    "fetch_subtitles_only",
    "probe",
]
