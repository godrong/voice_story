"""Unit tests for core.sources (LocalSource + factory).

KaggleSource is exercised only at the auth-check level (we don't hit the
network in tests). The Source factory is verified for both happy-path
and unknown-type errors.

core.sources 的单元测试。

KaggleSource 只测鉴权检查路径（不联网）。Factory 覆盖正常与异常分支。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from core import sources


def test_factory_unknown_type_raises():
    """get_source() with an unknown type raises ValueError. 未知类型抛 ValueError。"""
    with pytest.raises(ValueError, match="Unknown source_type"):
        sources.get_source("nonexistent", name="x")


def test_factory_local(tmp_path):
    """get_source('local') instantiates a LocalSource. local 工厂正常工作。"""
    inputs = tmp_path / "inputs"
    (inputs / "demo").mkdir(parents=True)
    src = sources.get_source("local", name="demo", inputs_root=inputs)
    assert src.meta.source_name == "demo"


def test_local_source_yields_audio_files(tmp_path, tone_wav):
    """LocalSource yields supported files in sorted order, ignoring the rest.

    LocalSource 按文件名排序产出受支持文件，跳过不支持的。
    """
    inputs = tmp_path / "inputs"
    speaker_dir = inputs / "demo"
    speaker_dir.mkdir(parents=True)
    # Move synthesized files into the speaker dir.
    a = tone_wav("z_audio.wav")
    b = tone_wav("a_audio.wav")
    (speaker_dir / a.name).write_bytes(a.read_bytes())
    (speaker_dir / b.name).write_bytes(b.read_bytes())
    (speaker_dir / "ignore.txt").write_text("not audio")

    src = sources.get_source("local", name="demo", inputs_root=inputs)
    found = [p.name for p in src.fetch()]
    assert found == ["a_audio.wav", "z_audio.wav"]


def test_local_source_missing_dir_raises(tmp_path):
    """LocalSource.fetch() raises FileNotFoundError if dir doesn't exist.

    LocalSource 找不到目录时抛 FileNotFoundError。
    """
    src = sources.get_source("local", name="ghost", inputs_root=tmp_path)
    with pytest.raises(FileNotFoundError):
        list(src.fetch())


def test_kaggle_source_auth_check_fails_without_creds(monkeypatch, tmp_path):
    """KaggleSource raises a clear error when no Kaggle creds exist.

    KaggleSource 在没有鉴权配置时给出可操作的报错。
    """
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setattr(Path, "expanduser", lambda self: tmp_path / str(self))

    src = sources.get_source(
        "kaggle", name="x", dataset_id="owner/slug",
    )
    with pytest.raises(RuntimeError, match="Kaggle credentials not found"):
        # _check_auth runs lazily inside fetch(); force it.
        src._check_auth()  # noqa: SLF001


def test_source_meta_defaults():
    """SourceMeta defaults to needs_separation=True and is_single_speaker=False.

    SourceMeta 的默认值符合 ADR-0008（默认开 Demucs）。
    """
    meta = sources.SourceMeta(source_name="x")
    assert meta.needs_separation is True
    assert meta.is_single_speaker is False
    assert meta.lang_hint is None
