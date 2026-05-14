"""Tests for core.tts: TTSBackend Protocol shape + LocalSubprocessTTS handshake / quit / failure paths.

We don't run real CosyVoice in unit tests (heavy + cross-env). Instead we
swap the worker script for tiny stub scripts that emulate the JSON-line
protocol — handshake "ready" / echo a synthesis result / fail loudly /
crash before handshake. This covers the client logic (timeout, error
mapping, quit handshake, crash detection) without ever loading a model.

core.tts 的单元测试：覆盖 TTSBackend Protocol + LocalSubprocessTTS 的
握手 / quit / 失败路径。

不在单测中跑真 CosyVoice（太重 + 跨 env）。改用极小的 stub 脚本模拟
JSON-line 协议的各种场景：握手 / 任务回响应 / 报错 / 启动崩溃。
覆盖客户端逻辑（超时、错误映射、quit 握手、崩溃检测）而无须加载任何模型。
"""

from __future__ import annotations

import os
import shutil
import sys
import textwrap
from pathlib import Path

import pytest

from core import tts


def _write_stub_worker(tmp_path: Path, body: str) -> Path:
    """Write a self-contained Python stub that acts like a TTS worker.

    把一段 Python 源码写成可执行的 stub worker，模拟 worker 的 stdin/stdout
    协议行为。stub 用系统 python 跑，不依赖 cosyvoice env。
    """
    stub = tmp_path / "stub_worker.py"
    stub.write_text(textwrap.dedent(body), encoding="utf-8")
    return stub


def _system_python() -> str:
    """Path to a Python interpreter we know exists (the test runner's own).

    返回当前测试用的 python 解释器路径，作为 stub worker 的解释器。
    """
    return sys.executable


def test_tts_backend_protocol_satisfied_by_local_subprocess(tmp_path):
    """LocalSubprocessTTS structurally satisfies the TTSBackend Protocol.

    LocalSubprocessTTS 在结构上满足 TTSBackend Protocol（Protocol 是结构化的
    duck typing，不需要继承）。
    """
    # We don't instantiate (would spawn a subprocess); just check method
    # signatures exist with correct names.
    # 不实例化（会 spawn 子进程），只检查方法名存在。
    assert callable(getattr(tts.LocalSubprocessTTS, "synthesize"))
    assert callable(getattr(tts.LocalSubprocessTTS, "close"))
    # Annotation check: synthesize takes the documented kwargs.
    import inspect
    sig = inspect.signature(tts.LocalSubprocessTTS.synthesize)
    for name in ("text", "ref_audio", "out_path", "prompt_text", "instruct", "mode"):
        assert name in sig.parameters, f"missing kw: {name}"


def test_factory_unknown_kind_raises():
    """get_tts_backend rejects unknown kinds with a clear ValueError.

    工厂收到未知 kind 报 ValueError，消息含未知值便于诊断。
    """
    with pytest.raises(ValueError, match="Unknown TTS backend kind"):
        tts.get_tts_backend("nonsense")


def test_handshake_ready_then_synthesize_echo(tmp_path):
    """End-to-end: handshake → one task → response → quit.

    端到端走通最小路径：握手 → 一个任务 → 响应 → quit。
    Stub 把 task 的 out 路径直接 echo 回去，写一个空 wav 文件做存在证据。
    """
    stub = _write_stub_worker(tmp_path, """
        import json, sys, pathlib
        sys.stdout.write(json.dumps({"event": "ready", "sample_rate": 24000}) + "\\n")
        sys.stdout.flush()
        for line in sys.stdin:
            line = line.strip()
            if not line: continue
            task = json.loads(line)
            if task.get("action") == "quit": break
            out = pathlib.Path(task["out"])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"")  # marker file
            sys.stdout.write(json.dumps({
                "id": task.get("id"), "ok": True, "out": str(out),
                "duration": 1.0, "sample_rate": 24000,
            }) + "\\n")
            sys.stdout.flush()
    """)
    tts_inst = tts.LocalSubprocessTTS(
        env_python=_system_python(),
        worker_script=stub,
        startup_timeout=10,
        task_timeout=10,
    )
    try:
        assert tts_inst.sample_rate == 24000
        ref = tmp_path / "ref.wav"
        ref.write_bytes(b"")  # exists; stub doesn't read it
        out = tts_inst.synthesize("hello", ref, out_path=tmp_path / "synth.wav")
        assert out.exists()
    finally:
        tts_inst.close()


def test_task_failure_raises_ttserror(tmp_path):
    """Worker reporting ok=false maps to TTSError with the worker's message.

    worker 报 ok=false 时主进程抛 TTSError，消息含 worker 的原始 err。
    """
    stub = _write_stub_worker(tmp_path, """
        import json, sys
        sys.stdout.write(json.dumps({"event": "ready", "sample_rate": 24000}) + "\\n")
        sys.stdout.flush()
        for line in sys.stdin:
            line = line.strip()
            if not line: continue
            task = json.loads(line)
            if task.get("action") == "quit": break
            sys.stdout.write(json.dumps({
                "id": task.get("id"), "ok": False, "err": "stub: forced failure",
            }) + "\\n")
            sys.stdout.flush()
    """)
    tts_inst = tts.LocalSubprocessTTS(
        env_python=_system_python(),
        worker_script=stub,
        startup_timeout=10,
        task_timeout=10,
    )
    try:
        with pytest.raises(tts.TTSError, match="stub: forced failure"):
            tts_inst.synthesize("hi", tmp_path / "ref.wav", out_path=tmp_path / "out.wav")
    finally:
        tts_inst.close()


def test_worker_crashes_before_handshake_raises(tmp_path):
    """If the worker exits without printing "ready", we raise quickly.

    worker 启动失败（不输出 ready）时主进程快速报错，不无限阻塞。
    """
    stub = _write_stub_worker(tmp_path, """
        import sys
        sys.exit(1)
    """)
    with pytest.raises((tts.TTSWorkerCrashed, tts.TTSError, TimeoutError)):
        tts.LocalSubprocessTTS(
            env_python=_system_python(),
            worker_script=stub,
            startup_timeout=5,
        )


def test_handshake_timeout(tmp_path):
    """A worker that never prints "ready" is killed after startup_timeout.

    worker 不发握手时，超过 startup_timeout 主进程 kill 并抛 TimeoutError。
    """
    stub = _write_stub_worker(tmp_path, """
        import time, sys
        time.sleep(60)  # never print ready
    """)
    with pytest.raises(TimeoutError):
        tts.LocalSubprocessTTS(
            env_python=_system_python(),
            worker_script=stub,
            startup_timeout=1.5,
        )


def test_missing_env_python_raises_clearly(tmp_path):
    """Non-existent env python path fails fast with FileNotFoundError.

    env_python 路径不存在时立即抛 FileNotFoundError，不静默启动失败。
    """
    with pytest.raises(FileNotFoundError, match="cosyvoice env python not found"):
        tts.LocalSubprocessTTS(env_python="/nonexistent/python")


def test_missing_worker_script_raises_clearly(tmp_path):
    """Non-existent worker script fails fast with FileNotFoundError.

    worker_script 不存在时立即抛 FileNotFoundError。
    """
    fake_python = tmp_path / "py"
    fake_python.write_text("")
    fake_python.chmod(0o755)
    with pytest.raises(FileNotFoundError, match="worker script missing"):
        tts.LocalSubprocessTTS(
            env_python=str(fake_python),
            worker_script=tmp_path / "no-such-script.py",
        )
