"""Kaggle source: download a Kaggle dataset via kagglehub and yield audio files.

Uses the official `kagglehub` package (no scraping). Auth is via
`~/.kaggle/kaggle.json` or the env vars `KAGGLE_USERNAME` / `KAGGLE_KEY`.
We fail-fast if creds are missing, with an actionable error message.

This is the first non-local source we ship; see ADR-0006 for why we treat
this as an extension of ADR-0003 (local-only) rather than a supersede.

Kaggle 数据集源插件。

通过官方 `kagglehub` 库下载（不是爬虫）。鉴权方式两种：
1. ~/.kaggle/kaggle.json 文件
2. 环境变量 KAGGLE_USERNAME / KAGGLE_KEY

未配置鉴权时立即报错并给出修复指引（fail-fast）。

这是 v0.1 第一个非本地源；ADR-0006 解释了为何把它视作 ADR-0003 的扩展
而非取代——kagglehub 是受控官方 API，没有反爬/限流的痛点。
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

from . import SourceMeta
from .. import audio_io


class KaggleSource:
    """Download a Kaggle dataset and yield supported media files inside it.

    下载 Kaggle 数据集并枚举其中的音视频文件。

    Args:
        name: Speaker / dataset short name. Used as `datasets/<name>/` dir.
              说话人或数据集的短名，决定输出 datasets/<name>/ 路径。
        dataset_id: Kaggle dataset id, e.g. "etaifour/trump-speeches-...".
                    Kaggle 数据集 ID，格式 "owner/slug"。
        lang_hint: Optional ISO 639-1 hint. None = ASR auto-detect.
                   语言提示，None 时下游自动检测。
        needs_separation: Whether to run Demucs. Default True (ADR-0008).
                          是否跑 Demucs，默认 True。
        is_single_speaker: Skip speaker diarization if True.
                           单说话人时跳过说话人定位。
        license: License string for record-keeping.
                 授权说明（仅记录用，不做强制校验）。
    """

    def __init__(
        self,
        name: str,
        *,
        dataset_id: str,
        lang_hint: str | None = None,
        needs_separation: bool = True,
        is_single_speaker: bool = False,
        license: str = "Kaggle dataset terms apply; see dataset page",
    ) -> None:
        self.dataset_id = dataset_id
        self.meta = SourceMeta(
            source_name=name,
            lang_hint=lang_hint,
            needs_separation=needs_separation,
            is_single_speaker=is_single_speaker,
            license=license,
            extra={"dataset_id": dataset_id},
        )
        self._cached_root: Path | None = None

    def _check_auth(self) -> None:
        """Fail-fast if Kaggle credentials are missing.

        鉴权前置检查：~/.kaggle/kaggle.json 和环境变量任一存在即可，
        都没有则立即报错并给出修复指引。
        """
        if Path("~/.kaggle/kaggle.json").expanduser().exists():
            return
        if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
            return
        raise RuntimeError(
            "Kaggle credentials not found. Either:\n"
            "  1) Place kaggle.json at ~/.kaggle/kaggle.json (chmod 600), or\n"
            "  2) Export KAGGLE_USERNAME and KAGGLE_KEY env vars.\n"
            "Get credentials at https://www.kaggle.com/settings/account -> Create New Token."
        )

    def _download(self) -> Path:
        """Download (cached) and return the local dataset root directory.

        调用 kagglehub.dataset_download；kagglehub 自带本地缓存，
        重复调用会复用已下载的副本。

        Returns:
            Local directory containing the unpacked dataset.
            数据集解压后的本地根目录。
        """
        if self._cached_root is not None:
            return self._cached_root
        self._check_auth()
        # Lazy import: keep import-time cheap and avoid hard dep at startup.
        # 延迟导入，避免启动时硬依赖 kagglehub。
        import kagglehub
        path = kagglehub.dataset_download(self.dataset_id)
        self._cached_root = Path(path)
        return self._cached_root

    def fetch(self) -> Iterable[Path]:
        """Download (if needed) and yield supported media files within the dataset.

        触发下载（已缓存则跳过），然后递归遍历产出所有支持后缀的文件。

        Yields:
            Path of each media file in sorted order.
            数据集中的音视频文件路径（按文件名排序）。
        """
        root = self._download()
        for p in sorted(root.rglob("*")):
            if p.is_file() and audio_io.is_supported(p):
                yield p
