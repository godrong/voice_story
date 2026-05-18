"""MCP stdio server: exposes `extractor` functions as Claude-callable tools.

Uses FastMCP (from the official `mcp` Python SDK) which auto-generates JSON
schemas from Python type hints. Each tool returns a uniform envelope:

    {"ok": True,  "data": {...}}      # success
    {"ok": False, "error": "<msg>"}   # failure (BilibiliExtractError text)

The envelope is friendlier to LLM tool-loops than raising — the model sees a
structured failure and can react (e.g. ask the user about the failure reason)
without the tool call being marked "errored".

Run standalone for testing:

    conda run -n ai_study python -m tools.bilibili_mcp.server

Claude Code wires this up via the project's `.mcp.json`.

MCP stdio server —— 把 extractor 暴露成 Claude 可调用的工具。

用官方 `mcp` Python SDK 的 FastMCP：从函数类型注解自动生成 JSON schema，
LLM 那边能直接看到参数结构。每个工具返回统一信封：

    {"ok": True,  "data": {...}}      # 成功
    {"ok": False, "error": "<msg>"}   # 失败（BilibiliExtractError 文本）

用信封而不是抛异常对 LLM 工具循环更友好 —— 模型能看到结构化的失败信息，
顺势询问用户（比如要不要换代理），而不是被标成 "tool errored" 中断对话。

独立运行做调试：

    conda run -n ai_study python -m tools.bilibili_mcp.server

Claude Code 通过项目根的 `.mcp.json` 启动这个 server。
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import extractor

logger = logging.getLogger(__name__)


def _ok(data: Any) -> dict:
    """Wrap a successful payload. 包成功信封。"""
    return {"ok": True, "data": data}


def _err(msg: str) -> dict:
    """Wrap an error message. 包失败信封。"""
    return {"ok": False, "error": msg}


def _result_to_dict(r: extractor.ExtractResult) -> dict:
    """Serialize ExtractResult to a JSON-safe dict (Path -> str).

    ExtractResult 序列化成 JSON 友好的 dict（Path 转 str）。
    """
    d = asdict(r)
    d["audio_path"] = str(r.audio_path)
    d["subtitle_paths"] = [str(p) for p in r.subtitle_paths]
    return d


def _build_app():
    """Construct the FastMCP app with all 4 tools registered.

    构造 FastMCP 应用并注册 4 个工具。延迟构造方便 import 期不踩 mcp 依赖。
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:
        raise RuntimeError(
            "mcp SDK not installed. Run "
            "`pip install 'voice-story[bilibili]'` or `pip install mcp`."
        ) from e

    app = FastMCP("bilibili-extract")

    @app.tool()
    def bilibili_probe(url: str) -> dict:
        """Fetch metadata for a Bilibili video (no download).

        探测 B 站视频元数据 —— 标题、UP 主、时长、分 P 列表、可用字幕。
        不下载任何媒体字节，适合 LLM 在 extract 前先确认范围。

        Args:
            url: Full B 站 URL or bare BV id (e.g. "BV1xx411x7xx").
                 完整 URL 或裸 BV 号。

        Returns:
            Envelope: {"ok": True, "data": VideoInfo as dict} on success,
                      {"ok": False, "error": str} on failure.
        """
        try:
            info = extractor.probe(url)
            return _ok(asdict(info))
        except extractor.BilibiliExtractError as e:
            return _err(str(e))

    @app.tool()
    def bilibili_extract_audio(
        url: str,
        out_dir: str,
        start_sec: float | None = None,
        end_sec: float | None = None,
        audio_format: str = "wav",
        fetch_subtitles: bool = False,
        sub_langs: list[str] | None = None,
        overwrite: bool = False,
    ) -> dict:
        """Download audio from a single Bilibili video (optional segment + subs).

        下载单个 B 站视频的音频；可选时间区间切片 + 字幕。

        Args:
            url: 完整 URL 或 BV 号。
            out_dir: 输出目录（绝对路径或相对当前工作目录）。
            start_sec: 时间区间起点（秒）；与 end_sec 同时给才生效。
            end_sec: 时间区间终点（秒）。
            audio_format: 输出音频格式（"wav" / "m4a" / "opus"）。默认 wav。
            fetch_subtitles: 是否同时拉官方字幕。
            sub_langs: 优先字幕语言（["zh-CN","zh-Hans","en"]）。None=用默认。
            overwrite: 已存在同名输出时是否覆盖。

        Returns:
            Envelope; on success data is the ExtractResult dict
            with `audio_path`, `info`, `time_range`, `subtitle_paths`.
        """
        try:
            tr: tuple[float, float] | None = None
            if start_sec is not None and end_sec is not None:
                tr = (float(start_sec), float(end_sec))
            elif (start_sec is None) ^ (end_sec is None):
                return _err("start_sec and end_sec must be given together.")
            langs = tuple(sub_langs) if sub_langs else ("zh-CN", "zh-Hans", "en")
            r = extractor.extract_audio(
                url,
                Path(out_dir),
                time_range=tr,
                audio_format=audio_format,
                fetch_subtitles=fetch_subtitles,
                sub_langs=langs,
                overwrite=overwrite,
            )
            return _ok(_result_to_dict(r))
        except extractor.BilibiliExtractError as e:
            return _err(str(e))

    @app.tool()
    def bilibili_extract_playlist(
        url: str,
        out_dir: str,
        parts: list[int] | None = None,
        start_sec: float | None = None,
        end_sec: float | None = None,
        audio_format: str = "wav",
        fetch_subtitles: bool = False,
        sub_langs: list[str] | None = None,
        overwrite: bool = False,
    ) -> dict:
        """Download multiple parts of a multi-P video or collection.

        批量下载多 P 视频 / 合集中的多个子项。

        Args:
            url: 多 P 视频 / 合集 / 收藏夹 URL。
            out_dir: 输出目录。
            parts: 1-based 子项索引列表；None=全下。
            start_sec / end_sec: 同 extract_audio（会应用到每一 P）。
            其余同 extract_audio。

        Returns:
            Envelope; data is a list of ExtractResult dicts.
        """
        try:
            tr: tuple[float, float] | None = None
            if start_sec is not None and end_sec is not None:
                tr = (float(start_sec), float(end_sec))
            elif (start_sec is None) ^ (end_sec is None):
                return _err("start_sec and end_sec must be given together.")
            langs = tuple(sub_langs) if sub_langs else ("zh-CN", "zh-Hans", "en")
            results = extractor.extract_playlist(
                url,
                Path(out_dir),
                parts=parts,
                time_range=tr,
                audio_format=audio_format,
                fetch_subtitles=fetch_subtitles,
                sub_langs=langs,
                overwrite=overwrite,
            )
            return _ok([_result_to_dict(r) for r in results])
        except extractor.BilibiliExtractError as e:
            return _err(str(e))

    @app.tool()
    def bilibili_fetch_subtitles(
        url: str,
        out_dir: str,
        sub_langs: list[str] | None = None,
    ) -> dict:
        """Download only official subtitles (skip audio).

        只拉官方字幕，不下载音频。

        Args:
            url: 完整 URL 或 BV 号。
            out_dir: 字幕输出目录。
            sub_langs: 优先字幕语言列表。

        Returns:
            Envelope; data is a list of subtitle file path strings.
        """
        try:
            langs = tuple(sub_langs) if sub_langs else ("zh-CN", "zh-Hans", "en")
            paths = extractor.fetch_subtitles_only(url, Path(out_dir), sub_langs=langs)
            return _ok([str(p) for p in paths])
        except extractor.BilibiliExtractError as e:
            return _err(str(e))

    return app


def main() -> None:
    """Entry point for `python -m tools.bilibili_mcp.server`.

    `python -m tools.bilibili_mcp.server` 的入口。
    用 stdio transport（MCP 本地工具的标准做法）。
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    app = _build_app()
    app.run()  # FastMCP defaults to stdio transport.


if __name__ == "__main__":
    main()
