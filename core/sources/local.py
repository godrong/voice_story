"""Local-directory source: scan `inputs/<name>/` for supported media files.

This is the simplest source plugin and the default for the MVP per ADR-0003.
The user drops files (mp3, wav, mp4, ...) into `inputs/<name>/` and this
source enumerates them. No network, no auth, no surprises.

本地目录源插件——MVP 默认源（见 ADR-0003）。

最简实现：用户把任意音视频文件丢到 `inputs/<name>/` 目录里，
本插件就把它们枚举出来。无网络、无鉴权、无平台依赖，
是开发与调试时最稳定的输入路径。
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from . import SourceMeta
from .. import audio_io


class LocalSource:
    """Yield supported media files under `inputs/<name>/`.

    扫描 `inputs/<name>/` 目录下所有受支持的音视频文件。

    Args:
        name: Speaker / dataset short name. Forms the subdir to scan.
              说话人或数据集的短名，对应 inputs/<name>/ 子目录。
        inputs_root: Root inputs directory. Defaults to `./inputs`.
                     输入根目录，默认项目根下的 inputs/。
        lang_hint: Optional ISO 639-1 hint forwarded to ASR.
                   语言提示，会传给下游 ASR。
        needs_separation: Whether Demucs should run. Default True.
                          是否跑 Demucs，默认 True。
        is_single_speaker: Skip speaker diarization if True.
                           是否单说话人；True 则跳过说话人定位。
        license: License string for record-keeping.
                 授权说明（仅记录用）。
    """

    def __init__(
        self,
        name: str,
        *,
        inputs_root: Path | str = "inputs",
        lang_hint: str | None = None,
        needs_separation: bool = True,
        is_single_speaker: bool = False,
        license: str = "",
    ) -> None:
        self.root = Path(inputs_root) / name
        self.meta = SourceMeta(
            source_name=name,
            lang_hint=lang_hint,
            needs_separation=needs_separation,
            is_single_speaker=is_single_speaker,
            license=license,
            extra={"root": str(self.root)},
        )

    def fetch(self) -> Iterable[Path]:
        """Walk the inputs subdir and yield supported files.

        递归遍历 inputs/<name>/，按字母序产出所有支持后缀的文件。

        Yields:
            Path of each media file, in sorted order for reproducibility.
            音视频文件路径，按文件名排序保证可复现。

        Raises:
            FileNotFoundError: If the inputs subdir doesn't exist.
                               目录不存在时抛出。
        """
        if not self.root.exists():
            raise FileNotFoundError(
                f"LocalSource: directory {self.root} does not exist. "
                f"Create it and drop audio/video files inside."
            )
        for p in sorted(self.root.rglob("*")):
            if p.is_file() and audio_io.is_supported(p):
                yield p