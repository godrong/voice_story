"""Source plugins: pluggable backends that yield raw audio file paths.

A `Source` knows how to obtain (download / locate / extract) raw media files
on disk and report metadata that downstream stages need (language hint,
whether vocal separation is required, single/multi-speaker, license).

Adding a new source = implementing the `Source` protocol and registering it
in `core/sources/__init__.py::REGISTRY`. Avoid putting source-specific logic
anywhere else in the pipeline.

源采集插件层。

每种"源"（本地目录 / Kaggle 数据集 / 未来的 yt-dlp URL 等）都实现统一的
`Source` Protocol：调用 `fetch()` 拿到磁盘上的原始媒体文件路径，附带一个
`SourceMeta`（语言提示、是否需要人声分离、单/多说话人、版权等）让下游模块
做差异化处理。

加新源的流程：实现 Source Protocol + 在本文件 REGISTRY 注册即可，
不要把源特有的逻辑泄漏到 pipeline 其它地方。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class SourceMeta:
    """Metadata attached to every fetched file.

    每个被 fetch 出来的文件都附带一份元信息，driver 下游模块的差异化处理。

    Attributes:
        source_name: Human-readable label (e.g. "trump", "my-podcast").
                     人类可读的标签，作为 dataset 子目录名。
        lang_hint: Optional ISO 639-1 code ("en", "zh"). None = auto-detect.
                   语言提示，None 时由 ASR 自己 langid。
        needs_separation: If True, run Demucs even though caller may know
                          the audio is clean. Defaults True for safety.
                          是否需要 Demucs。默认为 True 保证一致性，
                          见 ADR-0008。
        is_single_speaker: If True, skip the speaker-diarization filter.
                           是否单说话人。True 时跳过说话人定位步骤。
        license: Free-text license / attribution string for the dataset.
                 数据集授权说明（自由文本，记录用）。
        extra: Any source-specific metadata to forward downstream.
               源特有的元信息（dict），下游可选地读取。
    """

    source_name: str
    lang_hint: str | None = None
    needs_separation: bool = True
    is_single_speaker: bool = False
    license: str = ""
    extra: dict = field(default_factory=dict)


class Source(Protocol):
    """Protocol every source plugin must implement.

    所有源插件需实现的 Protocol。

    Implementations should be lazy: `fetch()` does the actual download /
    discovery only when consumed. Callers iterate to get file paths.

    实现要尽量惰性：`fetch()` 在被消费时才真正下载/扫描，
    调用方通过迭代取出文件路径。
    """

    meta: SourceMeta

    def fetch(self) -> Iterable[Path]:
        """Yield raw audio/video file paths on local disk.

        产出本地磁盘上的原始音视频文件路径流。

        Returns:
            Iterable of Paths. May be a generator (lazy is preferred).
            Path 的可迭代对象，建议用 generator 实现。
        """
        ...


def get_source(source_type: str, **kwargs) -> Source:
    """Factory: instantiate a Source by short name.

    工厂函数：按简称取一个 Source 实例（CLI 通过 --source 指定时调用）。

    Args:
        source_type: One of REGISTRY keys ("local", "kaggle", ...).
                     源类型简称，REGISTRY 里注册的 key。
        **kwargs: Forwarded to the Source class constructor.
                  转发给具体 Source 类的 __init__ 参数。

    Returns:
        A Source instance ready to `fetch()`.
        可直接 fetch() 的 Source 实例。

    Raises:
        ValueError: If `source_type` is not registered.
                    未注册的源类型。
    """
    if source_type not in REGISTRY:
        known = ", ".join(sorted(REGISTRY))
        raise ValueError(f"Unknown source_type {source_type!r}. Known: {known}")
    return REGISTRY[source_type](**kwargs)


# Delayed imports keep heavy deps (kagglehub) out of the import path
# until actually requested. 延迟导入：保证仅在用到 kaggle 时才加载 kagglehub。
def _local(**kw):
    from .local import LocalSource
    return LocalSource(**kw)


def _kaggle(**kw):
    from .kaggle import KaggleSource
    return KaggleSource(**kw)


REGISTRY = {
    "local": _local,
    "kaggle": _kaggle,
}