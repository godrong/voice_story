"""TTS backend interface + LocalSubprocessTTS implementation for M2~M4.

Defines a minimal `TTSBackend` Protocol and the current default
implementation `LocalSubprocessTTS`. The Protocol is the stable API the
rest of the project depends on (synthesis_agent, evaluation, etc.); the
implementation is swappable. M5+ will add a `CloudTTS` against the same
Protocol once CosyVoice training/inference moves off the laptop.

Why subprocess: CosyVoice 2 pins torch==2.3.1, which conflicts with the
ai_study env's torch 2.9.1 (M1 pipeline). Rather than downgrade and
break M1, we run CosyVoice in its own conda env `cosyvoice` and talk
to it via JSON-line over pipes. See `core/tts_worker.py` for the worker
side and discussion of the pattern in the project chat history.

TTS 后端接口 + 本地 subprocess 实现（M2~M4 默认）。

定义最小的 `TTSBackend` Protocol，与当前默认实现 `LocalSubprocessTTS`。
Protocol 是稳定的对外 API（synthesis_agent / evaluation 等都依赖它），
实现可换。M5+ 当 CosyVoice 训练/推理上云后，加 `CloudTTS` 走同一个 Protocol，
主流程一行不改。

为什么 subprocess：CosyVoice 2 钉 torch==2.3.1，与 ai_study env 的 torch
2.9.1（M1 pipeline 必需）冲突。不降级 ai_study，而是把 CosyVoice 放在
独立 conda env `cosyvoice` 里，主进程通过 stdin/stdout JSON-line 协议
驱动它。worker 侧见 `core/tts_worker.py`，模式讨论见项目聊天记录。
"""

from __future__ import annotations

import json
import logging
import select
import subprocess
import sys
import threading
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COSYVOICE_PYTHON = "/opt/homebrew/Caskroom/miniforge/base/envs/cosyvoice/bin/python"
DEFAULT_WORKER_SCRIPT = REPO_ROOT / "core" / "tts_worker.py"


class TTSError(RuntimeError):
    """Raised when the TTS backend reports a synthesis failure.

    TTS 后端报告合成失败时抛出；任务级错误，主流程可以捕获后继续。
    """


class TTSWorkerCrashed(TTSError):
    """Raised when the worker process has crashed (stdout closed unexpectedly).

    worker 子进程崩溃时抛出（stdout 提前关闭），需要重 spawn 才能恢复。
    """


class TTSBackend(Protocol):
    """Stable interface every TTS backend implements.

    所有 TTS 后端都遵循的稳定接口。

    The contract is intentionally tiny:
      synthesize(text, ref_audio, out_path, prompt_text, instruct, mode) -> Path
      close()
    Anything backend-specific (model dir, env path, endpoint URL) belongs
    in the implementation's __init__, not on this method.

    契约故意做得很薄：synthesize + close。后端特有的配置（模型路径、env、
    endpoint）放在实现的 __init__ 里，不污染统一接口。
    """

    def synthesize(
        self, text: str, ref_audio: Path, *,
        out_path: Path,
        prompt_text: str = "",
        instruct: str | None = None,
        mode: str = "zero_shot",
    ) -> Path: ...

    def close(self) -> None: ...


class LocalSubprocessTTS:
    """CosyVoice 2 in a sibling conda env, talked to via JSON-line pipes.

    本地 subprocess 实现：在 cosyvoice env 跑 tts_worker.py 长驻进程，
    通过 stdin / stdout JSON-line 协议同步请求-响应。

    Lifecycle:
      __init__       : spawn worker; block until "ready" handshake
      synthesize(...): write task line; readline response; raise on failure
      close()        : send {"action":"quit"}; wait; kill on timeout

    生命周期：__init__ spawn worker 并等握手 → synthesize 同步发任务收响应
    → close 发 quit 并等待退出（超时则 kill）。

    Args:
        env_python: Path to the cosyvoice conda env's python interpreter.
            cosyvoice env 的 python 解释器路径。
        worker_script: Path to tts_worker.py. Defaults to repo's core/.
            tts_worker.py 路径，默认在本仓 core/ 下。
        startup_timeout: Seconds to wait for the worker's ready handshake.
            等待握手的超时时间（秒）。
        task_timeout: Per-task synthesis timeout in seconds.
            单任务合成超时（秒）。
    """

    def __init__(
        self,
        env_python: str = DEFAULT_COSYVOICE_PYTHON,
        worker_script: Path | None = None,
        startup_timeout: float = 120.0,
        task_timeout: float = 180.0,
    ) -> None:
        self._env_python = env_python
        self._worker_script = Path(worker_script) if worker_script else DEFAULT_WORKER_SCRIPT
        self._task_timeout = task_timeout
        self._task_counter = 0
        self.sample_rate: int | None = None  # set after handshake

        if not Path(env_python).exists():
            raise FileNotFoundError(
                f"cosyvoice env python not found at {env_python}. "
                f"Create the env with `conda create -n cosyvoice python=3.10` "
                f"and install CosyVoice's requirements.txt."
            )
        if not self._worker_script.exists():
            raise FileNotFoundError(f"worker script missing: {self._worker_script}")

        logger.info("LocalSubprocessTTS: spawning worker via %s", env_python)
        # `-u` forces unbuffered stdout (defense in depth on top of the
        # explicit flush=True in the worker). Without unbuffered I/O the
        # pipe stays half-empty and our readline() blocks forever.
        # `-u` 强制 unbuffered（worker 内已 flush=True，这里多一层保险）。
        # 不强 unbuffered 时 pipe 半满，主进程 readline 永远阻塞。
        self._proc = subprocess.Popen(
            [env_python, "-u", str(self._worker_script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        # Drain stderr in a background thread so worker logs can't fill
        # the stderr pipe and deadlock the worker on its next stderr write.
        # 后台线程持续 drain stderr：避免 worker 大量 stderr 输出填满管道
        # 反过来阻塞 worker 自己的 stderr 写入。
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, name="tts_worker_stderr", daemon=True,
        )
        self._stderr_thread.start()

        self._wait_for_ready(startup_timeout)

    def _drain_stderr(self) -> None:
        """Background loop: copy worker stderr lines to our logger.

        后台 drain：把 worker 的 stderr 转写到本进程 logger，避免管道塞满。
        """
        for line in self._proc.stderr:
            line = line.rstrip()
            if line:
                logger.debug("worker.stderr: %s", line)

    def _wait_for_ready(self, timeout: float) -> None:
        """Block until the worker emits {"event":"ready", ...}, tolerating noise.

        阻塞读 stdout 直到拿到 ready 握手；忽略前面的非 JSON 杂讯
        （CosyVoice / modelscope / wetext 等三方包可能在 import 时
        把进度 / warning 打到 stdout，即使 worker 已重定向 stdout 也
        可能漏一两行——这里宽容跳过）；超时则 kill worker 并抛错。
        """
        import time
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._kill()
                raise TimeoutError(
                    f"worker did not signal ready within {timeout}s"
                )
            try:
                event = self._read_one_event(timeout=remaining)
            except TTSError as e:
                # Non-JSON line (third-party stdout pollution); log + retry.
                # 非 JSON 行（三方 stdout 污染）：记录后继续等下一行。
                logger.debug("ignoring pre-handshake noise: %s", e)
                continue
            if event.get("event") == "ready":
                self.sample_rate = int(event.get("sample_rate", 0)) or None
                logger.info(
                    "LocalSubprocessTTS: worker ready (sample_rate=%s)",
                    self.sample_rate,
                )
                return
            if event.get("event") == "fatal":
                self._kill()
                raise TTSError(f"worker startup failed: {event.get('err')}")
            # Any other event before ready is unexpected but harmless; log + wait.
            # 其它握手前事件：记录后继续等。
            logger.warning("unexpected pre-handshake event: %s", event)

    def _read_one_event(self, *, timeout: float) -> dict:
        """Read one JSON line from worker stdout with a select() timeout.

        从 worker stdout 读一行 JSON；用 select 实现超时，避免无限阻塞。
        """
        # Note: select() on a Python text-mode pipe works on POSIX because
        # the underlying fd is the same; on Windows we'd need a different
        # approach (we don't target Windows in this project).
        # 注：text-mode pipe 上 select 在 POSIX 有效（底层 fd 不变）。
        # Windows 需要别的方案，本项目不支持 Windows。
        r, _, _ = select.select([self._proc.stdout], [], [], timeout)
        if not r:
            self._kill()
            raise TimeoutError(
                f"worker did not respond within {timeout}s (likely hanging or slow load)"
            )
        line = self._proc.stdout.readline()
        if not line:
            raise TTSWorkerCrashed("worker closed stdout (likely crashed during startup)")
        try:
            return json.loads(line)
        except json.JSONDecodeError as e:
            raise TTSError(f"non-JSON response from worker: {line!r} ({e})") from e

    def synthesize(
        self, text: str, ref_audio: Path, *,
        out_path: Path,
        prompt_text: str = "",
        instruct: str | None = None,
        mode: str = "zero_shot",
    ) -> Path:
        """Send one synthesis task and block until the worker responds.

        发一条合成任务，阻塞等响应；返回输出 WAV 路径。

        Args:
            text: 要合成的目标文本。
            ref_audio: 参考音频路径（CosyVoice 内部按需 resample 到 16k）。
            out_path: 输出 WAV 路径（父目录会自动创建）。
            prompt_text: zero_shot 模式下参考音频对应的文本（必填）。
            instruct: instruct 模式下的风格指令（可加 <|endofprompt|> 后缀）。
            mode: "zero_shot" / "cross_lingual" / "instruct"。

        Returns:
            输出 WAV 的 Path。

        Raises:
            TTSError: 任务级失败（worker 报 ok=False）。
            TTSWorkerCrashed: worker 进程崩溃（stdout EOF）。
            TimeoutError: 单任务超过 task_timeout。
        """
        if self._proc.poll() is not None:
            raise TTSWorkerCrashed(
                f"worker already exited with code {self._proc.returncode}"
            )

        self._task_counter += 1
        task = {
            "id": self._task_counter,
            "text": text,
            "ref": str(Path(ref_audio).resolve()),
            "out": str(Path(out_path).resolve()),
            "prompt_text": prompt_text,
            "mode": mode,
        }
        if instruct is not None:
            task["instruct"] = instruct

        try:
            self._proc.stdin.write(json.dumps(task, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise TTSWorkerCrashed(f"failed to write task to worker: {e}") from e

        resp = self._read_one_event(timeout=self._task_timeout)
        if not resp.get("ok"):
            raise TTSError(
                f"task {self._task_counter} failed: {resp.get('err', 'unknown')}"
            )
        return Path(resp["out"])

    def close(self) -> None:
        """Send quit then wait; kill on timeout. Idempotent.

        发 quit、等 worker 干净退出；超时则 kill。可重复调用。
        """
        if self._proc.poll() is not None:
            return
        try:
            self._proc.stdin.write(json.dumps({"action": "quit"}) + "\n")
            self._proc.stdin.flush()
            self._proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._kill()
        logger.info("LocalSubprocessTTS: worker exited")

    def _kill(self) -> None:
        """Hard-kill the worker process.

        强杀 worker 进程；在超时或启动失败时调用。
        """
        try:
            self._proc.kill()
            self._proc.wait(timeout=5)
        except Exception:  # noqa: BLE001 — best-effort cleanup
            pass

    def __enter__(self) -> LocalSubprocessTTS:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def get_tts_backend(kind: str = "local", **kwargs) -> TTSBackend:
    """Factory: return a TTSBackend implementation by short name.

    工厂：按简称取一个 TTSBackend 实例。
    M2~M4 只有 'local'；M5+ 加 'cloud'；测试用 'mock'。

    Args:
        kind: "local"。M5+ 加 "cloud" / "mock"。
        **kwargs: 转发给具体实现的构造函数。

    Returns:
        TTSBackend 实例。
    """
    if kind == "local":
        return LocalSubprocessTTS(**kwargs)
    raise ValueError(f"Unknown TTS backend kind: {kind!r}")


if __name__ == "__main__":
    # Quick standalone smoke runner (only used during development).
    # 仅开发用的独立 smoke 入口：python -m core.tts <text> <ref> <out>
    import argparse
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("ref")
    ap.add_argument("out")
    ap.add_argument("--prompt-text", default="")
    ap.add_argument("--instruct", default=None)
    ap.add_argument("--mode", default="zero_shot")
    args = ap.parse_args()
    with get_tts_backend("local") as tts:
        out = tts.synthesize(
            args.text, Path(args.ref),
            out_path=Path(args.out),
            prompt_text=args.prompt_text,
            instruct=args.instruct,
            mode=args.mode,
        )
        print(out)
