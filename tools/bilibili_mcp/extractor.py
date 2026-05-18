"""Pure-function core: yt-dlp + ffmpeg wrapped behind a small typed API.

Design contract (kept deliberately narrow so the MCP server and the Source
plugin can share it without surprises):

- All functions are synchronous. yt-dlp itself is synchronous; wrapping in
  asyncio gains nothing and adds confusion.
- Network/parsing errors from yt-dlp are caught and re-raised as
  `BilibiliExtractError` with an actionable hint (mirrors the fail-fast style
  used in `core/sources/kaggle.py::_check_auth`).
- No call to `core.audio_io.to_standard_wav` is made here. Whether to
  re-transcode to 24 kHz mono is the caller's choice (the SourceAgent will
  do it; raw MCP callers may want the original m4a/opus).
- Output filename convention: `<bvid>[_pN][_T<start>-<end>].<ext>`.
  Same names land next to LocalSource's expected layout.

下载器的纯函数核心 —— yt-dlp + ffmpeg 的薄包装。

设计契约（特意收窄，方便 MCP server 和 Source 插件共用）：

- 函数全部同步。yt-dlp 本身同步，套 asyncio 没收益且会混淆错误栈。
- yt-dlp 的网络/解析错误统一捕成 `BilibiliExtractError`，错误信息里附
  可操作指引（仿 `core/sources/kaggle.py::_check_auth` 的 fail-fast 风格）。
- 这一层 **不** 调 `core.audio_io.to_standard_wav`。是否要转 24 kHz mono
  由调用方决定（SourceAgent 会做；纯 MCP 调用方可能想要原始 m4a/opus）。
- 输出命名约定：`<bvid>[_pN][_T<start>-<end>].<ext>`，
  与 LocalSource 扫描目录的位置一致。
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# B 站 BV 号正则（容错 av 号）；用于从 yt-dlp info dict 缺字段时的兜底解析
_BVID_RE = re.compile(r"BV[0-9A-Za-z]{10}")


class BilibiliExtractError(RuntimeError):
    """Wraps any yt-dlp/network/parse failure with an actionable hint.

    把 yt-dlp/网络/解析失败统一包成一个异常，错误消息里带可操作指引。
    """


@dataclass(frozen=True)
class PartInfo:
    """One entry in a multi-P video or season playlist.

    多 P 视频或合集中的一条子项。

    Attributes:
        index: 1-based position within the playlist. 1 起的索引。
        title: Sub-title of this part. 子标题。
        duration: Length in seconds. 时长（秒）。
        url: Direct URL to this part (used for selective download).
             这一 P 的直接 URL，selective 下载会用到。
    """

    index: int
    title: str
    duration: float
    url: str


@dataclass(frozen=True)
class VideoInfo:
    """Result of `probe()` — metadata only, nothing downloaded yet.

    `probe()` 的返回值 —— 仅元数据，未真正下载任何字节。

    Attributes:
        url: Canonical URL that was queried. 被探测的规范 URL。
        bvid: BV identifier ("BV1xx411..."), empty if unavailable.
              BV 号；拿不到时为空字符串。
        title: Video title. 视频标题。
        uploader: UP 主 display name. UP 主昵称。
        duration: Total seconds (top-level; for multi-P, sum of parts).
                  总时长（多 P 时是各 P 之和）。
        parts: Sub-parts list. Single-P videos return `[<one PartInfo>]`.
               分 P 列表；单 P 视频返回一项。
        available_subtitles: lang_code -> first subtitle source URL.
                             语言代码到字幕直链的映射，方便后续决策。
    """

    url: str
    bvid: str
    title: str
    uploader: str
    duration: float
    parts: list[PartInfo]
    available_subtitles: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractResult:
    """On-disk asset produced by `extract_audio` / `extract_playlist`.

    `extract_audio` / `extract_playlist` 落盘后的产物。

    Attributes:
        audio_path: Path to the downloaded audio file.
                    下载后的音频文件路径。
        info: The originating VideoInfo (carried for downstream metadata).
              来源视频的 VideoInfo（带在身上方便下游回溯）。
        part_index: 1-based part index for multi-P, else None.
                    多 P 时填 1 起索引；单 P 为 None。
        time_range: (start_sec, end_sec) if a segment was cut, else None.
                    若做了时间区间剪辑则填 (起, 止) 秒数，否则 None。
        subtitle_paths: Subtitle files written alongside the audio.
                        与音频一起落盘的字幕文件列表。
    """

    audio_path: Path
    info: VideoInfo
    part_index: int | None = None
    time_range: tuple[float, float] | None = None
    subtitle_paths: list[Path] = field(default_factory=list)


def _hint() -> str:
    """Actionable troubleshooting hint appended to every error.

    所有错误后面都会附上这段排错指引。"""
    return (
        "\nTroubleshooting / 排错:\n"
        "  1) Update yt-dlp first: `pip install -U yt-dlp` "
        "(B 站常更新风控，yt-dlp 修复跟得很紧).\n"
        "  2) Check if the URL needs login (member-only / UP-restricted) — "
        "this module does not pass cookies in v1.\n"
        "  3) Try with a proxy if you see 412/429 from B 站 CDN.\n"
        "  4) Verify ffmpeg is on PATH (`ffmpeg -version`)."
    )


def _import_ytdl():
    """Lazy import yt_dlp; turn missing dep into an actionable error.

    延迟导入 yt_dlp；缺包时给出明确安装指引（不要硬依赖在 import 期）。
    """
    try:
        import yt_dlp  # noqa: WPS433 (intentional lazy import)
        return yt_dlp
    except ImportError as e:
        raise BilibiliExtractError(
            "yt-dlp is not installed. Install with "
            "`pip install 'voice-story[bilibili]'` or `pip install yt-dlp`."
            + _hint()
        ) from e


def _normalize_url(url: str) -> str:
    """Accept BV id, full URL, or copy-pasted-twice URL; return canonical URL.

    宽松输入：传 BV 号、完整 URL、甚至误粘两遍的 URL 都行，
    都归一化成 https://www.bilibili.com/video/<BV>。

    Defensive against common paste mistakes (the share-link copy on B 站 web
    sometimes ends up duplicated when users hit paste twice). If we can find
    a BV id anywhere in the string we use that as the source of truth — yt-dlp
    accepts both the bare BV id and the canonical URL.
    防御常见的"粘贴两次"误操作：只要能在字符串里抓到一个 BV 号，
    就用它重建规范 URL；yt-dlp 既吃裸 BV 号也吃完整 URL。
    """
    s = url.strip()
    m = _BVID_RE.search(s)
    if m:
        return f"https://www.bilibili.com/video/{m.group(0)}"
    return s


def _extract_bvid(info: dict, fallback_url: str) -> str:
    """Pull BV id from yt-dlp info dict, with regex fallback on the URL.

    从 yt-dlp 的 info dict 拿 BV 号；拿不到时从 URL 用正则兜底。
    """
    for key in ("id", "display_id", "webpage_url"):
        v = info.get(key)
        if isinstance(v, str):
            m = _BVID_RE.search(v)
            if m:
                return m.group(0)
    m = _BVID_RE.search(fallback_url)
    return m.group(0) if m else ""


def _info_to_video_info(info: dict, url: str) -> VideoInfo:
    """Convert yt-dlp's raw info dict to our typed VideoInfo.

    把 yt-dlp 返回的原始 info dict 转成我们的强类型 VideoInfo。
    单 P 与多 P 都统一成 `parts` 至少 1 项的结构，调用方无需分支。
    """
    bvid = _extract_bvid(info, url)
    entries = info.get("entries")
    if isinstance(entries, list) and entries:
        parts: list[PartInfo] = []
        for i, e in enumerate(entries, start=1):
            parts.append(
                PartInfo(
                    index=i,
                    title=str(e.get("title") or f"part_{i}"),
                    duration=float(e.get("duration") or 0.0),
                    url=str(e.get("webpage_url") or e.get("url") or url),
                )
            )
        total_dur = sum(p.duration for p in parts)
    else:
        parts = [
            PartInfo(
                index=1,
                title=str(info.get("title") or "untitled"),
                duration=float(info.get("duration") or 0.0),
                url=str(info.get("webpage_url") or url),
            )
        ]
        total_dur = parts[0].duration

    subs_raw = info.get("subtitles") or {}
    available: dict[str, str] = {}
    for lang, tracks in subs_raw.items():
        if isinstance(tracks, list) and tracks:
            available[str(lang)] = str(tracks[0].get("url") or "")

    return VideoInfo(
        url=url,
        bvid=bvid,
        title=str(info.get("title") or "untitled"),
        uploader=str(info.get("uploader") or info.get("channel") or ""),
        duration=float(total_dur),
        parts=parts,
        available_subtitles=available,
    )


def probe(url: str) -> VideoInfo:
    """Fetch video metadata without downloading any media bytes.

    只拉元数据，不下载任何媒体字节（yt-dlp 的 extract_info(download=False)）。

    Args:
        url: Full B 站 URL or a bare BV id.
             完整的 B 站 URL 或裸 BV 号。

    Returns:
        VideoInfo with title / parts / available subtitles.
        包含标题、分 P 列表、可用字幕语言的 VideoInfo。

    Raises:
        BilibiliExtractError: On any network or parse failure.
                              网络或解析失败时抛出（带排错指引）。
    """
    yt_dlp = _import_ytdl()
    real_url = _normalize_url(url)
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": False,  # We want full part metadata
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(real_url, download=False)
    except yt_dlp.utils.DownloadError as e:
        raise BilibiliExtractError(f"probe({url!r}) failed: {e}" + _hint()) from e
    if not isinstance(info, dict):
        raise BilibiliExtractError(
            f"probe({url!r}) returned non-dict info: {type(info)!r}" + _hint()
        )
    return _info_to_video_info(info, real_url)


def _outtmpl(out_dir: Path, name_prefix: str, ext_placeholder: bool = True) -> str:
    """Build yt-dlp's `outtmpl` so output filename is fully under our control.

    构造 yt-dlp 的 outtmpl，让输出文件名完全可控（不留 yt-dlp 默认的乱码标题）。
    """
    tail = ".%(ext)s" if ext_placeholder else ""
    return str(out_dir / f"{name_prefix}{tail}")


def _ranges_callback(start: float, end: float):
    """Build yt-dlp `download_ranges` callback for a single [start, end] segment.

    yt-dlp 的 download_ranges 接收一个回调，返回 list[dict]，
    每个 dict 描述要下载的区间。我们只用单段 [start, end]。
    """
    def _cb(info_dict, ydl):  # noqa: ANN001 - yt-dlp callback signature
        return [{"start_time": float(start), "end_time": float(end)}]
    return _cb


def _expected_audio_path(out_dir: Path, name_prefix: str, audio_format: str) -> Path:
    """Compute the post-postprocessor file path; yt-dlp doesn't return it directly.

    yt-dlp 的 postprocessor 处理后不会直接返回最终文件路径，按约定推算：
    `<out_dir>/<name_prefix>.<audio_format>`。
    """
    return out_dir / f"{name_prefix}.{audio_format}"


def extract_audio(
    url: str,
    out_dir: Path | str,
    *,
    time_range: tuple[float, float] | None = None,
    audio_format: str = "wav",
    fetch_subtitles: bool = False,
    sub_langs: tuple[str, ...] = ("zh-CN", "zh-Hans", "en"),
    overwrite: bool = False,
    progress_hooks: list | None = None,
) -> ExtractResult:
    """Download a single video's audio (optionally a time slice + subtitles).

    下载单个视频的音频，可选时间区间切片 + 同时拉字幕。

    Args:
        url: 完整 URL 或 BV 号。
        out_dir: 输出目录，不存在自动创建。
        time_range: (start_sec, end_sec) 浮点秒数；None=全片。
                    用 yt-dlp 的 download_ranges 在下载阶段就剪掉，
                    避免拉完整片再用 ffmpeg 二次处理。
        audio_format: 目标音频格式（wav/m4a/opus 等），通过 ffmpeg postprocessor 转。
                      默认 wav 以省一道转码（下游 audio_io 也是 wav）。
        fetch_subtitles: 是否同时拉官方字幕到 out_dir。
        sub_langs: 字幕语言优先级列表，按 B 站常见命名传 zh-CN / zh-Hans / en。
        overwrite: 已存在同名输出时是否覆盖。
        progress_hooks: yt-dlp 进度回调列表；每个 callable 收一个 dict（含
                        status / downloaded_bytes / total_bytes 等）。
                        webui PipelineCard 用它把下载进度写进 stage.detail。

    Returns:
        ExtractResult，含本地音频路径、来源 VideoInfo、time_range、字幕路径列表。

    Raises:
        BilibiliExtractError: yt-dlp 报错或 postprocessor 失败时抛出。
    """
    yt_dlp = _import_ytdl()
    real_url = _normalize_url(url)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # First probe so we can name files with the BV id deterministically.
    # 先 probe 一次，用 BV 号做确定性命名（不依赖 yt-dlp 的标题）。
    info = probe(real_url)
    bvid = info.bvid or "unknown"

    suffix = ""
    if time_range is not None:
        start, end = float(time_range[0]), float(time_range[1])
        if end <= start:
            raise BilibiliExtractError(
                f"time_range end ({end}) must be > start ({start})." + _hint()
            )
        suffix = f"_T{start:.1f}-{end:.1f}"
    name_prefix = f"{bvid}{suffix}"
    expected = _expected_audio_path(out_dir, name_prefix, audio_format)

    if expected.exists() and not overwrite:
        logger.info("extract_audio: %s already exists, reusing", expected)
        return ExtractResult(
            audio_path=expected,
            info=info,
            part_index=None,
            time_range=time_range,
            subtitle_paths=_existing_subtitles(out_dir, name_prefix),
        )

    opts: dict = {
        "format": "bestaudio/best",
        "outtmpl": _outtmpl(out_dir, name_prefix),
        "quiet": True,
        "no_warnings": True,
        "overwrites": overwrite,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": audio_format,
                # No preferredquality so ffmpeg picks a sensible default for wav.
                # 不指定 preferredquality；wav 用默认即可。
            }
        ],
    }
    if time_range is not None:
        opts["download_ranges"] = _ranges_callback(*time_range)
        # force_keyframes_at_cuts removes the small leading gap from container alignment.
        # 在切点强制关键帧，去掉容器对齐造成的开头小空白。
        opts["force_keyframes_at_cuts"] = True
    if fetch_subtitles:
        opts.update(
            writesubtitles=True,
            subtitleslangs=list(sub_langs),
            subtitlesformat="srt/best",
        )
    if progress_hooks:
        opts["progress_hooks"] = list(progress_hooks)

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([real_url])
    except yt_dlp.utils.DownloadError as e:
        raise BilibiliExtractError(
            f"extract_audio({url!r}) failed: {e}" + _hint()
        ) from e

    if not expected.exists():
        raise BilibiliExtractError(
            f"extract_audio: expected {expected} after download, but it doesn't exist. "
            f"yt-dlp may have changed its postprocessor naming." + _hint()
        )

    return ExtractResult(
        audio_path=expected,
        info=info,
        part_index=None,
        time_range=time_range,
        subtitle_paths=_existing_subtitles(out_dir, name_prefix),
    )


def _existing_subtitles(out_dir: Path, name_prefix: str) -> list[Path]:
    """Glob `<name_prefix>.*.srt|vtt` next to the audio file.

    在 out_dir 下用 glob 找与音频同名前缀的字幕文件，返回排序后的列表。
    """
    out: list[Path] = []
    for ext in (".srt", ".vtt"):
        out.extend(sorted(out_dir.glob(f"{name_prefix}*{ext}")))
    return out


def extract_playlist(
    url: str,
    out_dir: Path | str,
    *,
    parts: list[int] | None = None,
    time_range: tuple[float, float] | None = None,
    audio_format: str = "wav",
    fetch_subtitles: bool = False,
    sub_langs: tuple[str, ...] = ("zh-CN", "zh-Hans", "en"),
    overwrite: bool = False,
) -> list[ExtractResult]:
    """Extract multiple parts from a multi-P video or a season/collection URL.

    从多 P 视频或合集 URL 批量提取。

    Args:
        url: 多 P 视频 URL（带 ?p=N 或不带）/ 合集 URL / 收藏夹 URL。
        out_dir: 输出目录。
        parts: 要下载的子项 1-based 索引列表；None=全部。
        time_range: 与单视频相同；会被应用到每一 P（一般只对单视频有用）。
        其余参数同 extract_audio。

    Returns:
        ExtractResult 列表，每一 P 一个。

    Raises:
        BilibiliExtractError: 单 P 失败会立即抛出，已下载的 P 不回滚（幂等设计）。
    """
    info = probe(url)
    selected = info.parts if parts is None else [p for p in info.parts if p.index in set(parts)]
    if not selected:
        raise BilibiliExtractError(
            f"extract_playlist: no parts matched parts={parts!r}. "
            f"Available indices: {[p.index for p in info.parts]}" + _hint()
        )

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    results: list[ExtractResult] = []
    for part in selected:
        # We name with the original BV + part index so a single dir can host all parts.
        # 用 BV + 分 P 索引命名，让一个目录能同时容纳所有 P。
        suffix = f"_p{part.index}"
        if time_range is not None:
            start, end = float(time_range[0]), float(time_range[1])
            suffix += f"_T{start:.1f}-{end:.1f}"
        name_prefix = f"{info.bvid or 'unknown'}{suffix}"
        expected = _expected_audio_path(out_dir_p, name_prefix, audio_format)

        if expected.exists() and not overwrite:
            logger.info("extract_playlist: %s exists, reusing", expected)
            results.append(
                ExtractResult(
                    audio_path=expected,
                    info=info,
                    part_index=part.index,
                    time_range=time_range,
                    subtitle_paths=_existing_subtitles(out_dir_p, name_prefix),
                )
            )
            continue

        yt_dlp = _import_ytdl()
        opts: dict = {
            "format": "bestaudio/best",
            "outtmpl": _outtmpl(out_dir_p, name_prefix),
            "quiet": True,
            "no_warnings": True,
            "overwrites": overwrite,
            "postprocessors": [
                {"key": "FFmpegExtractAudio", "preferredcodec": audio_format}
            ],
        }
        if time_range is not None:
            opts["download_ranges"] = _ranges_callback(*time_range)
            opts["force_keyframes_at_cuts"] = True
        if fetch_subtitles:
            opts.update(
                writesubtitles=True,
                subtitleslangs=list(sub_langs),
                subtitlesformat="srt/best",
            )

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([part.url])
        except yt_dlp.utils.DownloadError as e:
            raise BilibiliExtractError(
                f"extract_playlist: part {part.index} ({part.url!r}) failed: {e}"
                + _hint()
            ) from e

        if not expected.exists():
            raise BilibiliExtractError(
                f"extract_playlist: expected {expected}, but it doesn't exist."
                + _hint()
            )

        results.append(
            ExtractResult(
                audio_path=expected,
                info=info,
                part_index=part.index,
                time_range=time_range,
                subtitle_paths=_existing_subtitles(out_dir_p, name_prefix),
            )
        )

    return results


def fetch_subtitles_only(
    url: str,
    out_dir: Path | str,
    *,
    sub_langs: tuple[str, ...] = ("zh-CN", "zh-Hans", "en"),
) -> list[Path]:
    """Download official subtitles for a video without downloading the audio.

    只拉官方字幕，不下载音频。常用于"已经有音频、但想省掉一道 ASR"的场景。

    Args:
        url: 完整 URL 或 BV 号。
        out_dir: 字幕输出目录。
        sub_langs: 优先尝试的字幕语言列表。

    Returns:
        实际落盘的字幕文件路径列表（可能为空，表示该视频没有任何官方字幕）。

    Raises:
        BilibiliExtractError: yt-dlp 报错时抛出（带排错指引）。
    """
    yt_dlp = _import_ytdl()
    real_url = _normalize_url(url)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    info = probe(real_url)
    name_prefix = info.bvid or "unknown"

    opts: dict = {
        "skip_download": True,
        "writesubtitles": True,
        "subtitleslangs": list(sub_langs),
        "subtitlesformat": "srt/best",
        "outtmpl": _outtmpl(out_dir_p, name_prefix, ext_placeholder=False) + ".%(ext)s",
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([real_url])
    except yt_dlp.utils.DownloadError as e:
        raise BilibiliExtractError(
            f"fetch_subtitles_only({url!r}) failed: {e}" + _hint()
        ) from e

    return _existing_subtitles(out_dir_p, name_prefix)


def iter_extract_results(results: Iterable[ExtractResult]) -> Iterable[Path]:
    """Convenience adapter: flatten ExtractResults into their audio Paths.

    把一串 ExtractResult 摊平成它们各自的音频 Path —— 给 BilibiliSource.fetch
    这种"只关心文件路径"的下游用，避免重复写解包。
    """
    for r in results:
        yield r.audio_path
