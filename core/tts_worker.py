"""Long-running CosyVoice 2 worker, talked to via JSON-line over stdin/stdout.

This script lives in the voice-story repo but runs inside the sibling
`cosyvoice` conda env (because CosyVoice pins torch==2.3.1 which clashes
with ai_study's torch 2.9.1; ADR-0010 / cross-env design discussion).

Protocol (one JSON object per line in either direction):

    main → worker (request):
        {"id": <int>, "text": "...", "ref": "/abs/path/ref.wav",
         "out": "/abs/path/out.wav", "prompt_text": "...",
         "instruct": null|"风格指令<|endofprompt|>",
         "mode": "zero_shot"|"instruct"|"cross_lingual"}

    main → worker (control):
        {"action": "quit"}

    worker → main (handshake, on startup once model loaded):
        {"event": "ready", "sample_rate": 24000}

    worker → main (success):
        {"id": <int>, "ok": true, "out": "...", "duration": 4.2}

    worker → main (failure):
        {"id": <int>, "ok": false, "err": "..."}

stderr is reserved for human-readable logs — anything printed there is NOT
part of the protocol. Importantly all stdout writes use flush=True (Python
block-buffers pipes by default; without flush the main process readline
deadlocks).

CosyVoice 2 长驻 worker：通过 stdin/stdout 走 JSON-line 协议。

本脚本属于 voice-story 仓，但运行在兄弟 conda env `cosyvoice`（CosyVoice
钉 torch==2.3.1 与 ai_study 的 torch 2.9.1 冲突，详见 ADR-0010 / 跨 env
设计讨论）。

协议：每行一条 JSON
  - 请求：{"id", "text", "ref", "out", "prompt_text", "instruct", "mode"}
  - 控制：{"action": "quit"}
  - 启动握手：{"event": "ready", "sample_rate": 24000}
  - 成功：{"id", "ok": true, "out", "duration"}
  - 失败：{"id", "ok": false, "err"}

stderr 保留给人类日志，不在协议内。所有 stdout 写入必须 flush=True
（Python 默认 block-buffer pipe，不 flush 主进程 readline 会死锁）。
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path


def _log(*args, **kwargs) -> None:
    """Emit a human-readable log line to stderr (NOT stdout — that's protocol).

    所有日志走 stderr；stdout 只走 JSON 协议，不许污染。
    """
    print("[tts_worker]", *args, **kwargs, file=sys.stderr, flush=True)


# Cache the real stdout at module load before anyone redirects it.
# This is what _emit() always uses, regardless of any sys.stdout swap.
# 在任何重定向之前缓存真实 stdout；_emit 始终用它，
# 不受 sys.stdout 替换影响（解决 CosyVoice/wetext 等三方包写 stdout 的污染）。
_PROTOCOL_STDOUT = sys.stdout


def _emit(obj: dict) -> None:
    """Write a JSON line to the protocol stdout (immune to sys.stdout swaps).

    协议输出：JSON-line 写到模块加载时缓存的真 stdout，立即 flush。
    用 _PROTOCOL_STDOUT 而非 sys.stdout，避免被 CosyVoice 内部的
    print 调用污染（CosyVoice 加载时 modelscope 会向 stdout 打下载进度）。
    """
    _PROTOCOL_STDOUT.write(json.dumps(obj, ensure_ascii=False) + "\n")
    _PROTOCOL_STDOUT.flush()


def _load_model() -> object:
    """Load CosyVoice2 once at startup; return the loaded model object.

    启动时加载一次 CosyVoice2 模型；之后所有 task 复用。

    Path resolution:
      - voice_story/core/tts_worker.py → ../../CosyVoice 是兄弟仓
      - third_party/Matcha-TTS 必须先进 sys.path（CosyVoice 内部依赖）
    """
    repo_root = Path(__file__).resolve().parent.parent.parent / "CosyVoice"
    matcha = repo_root / "third_party" / "Matcha-TTS"
    sys.path.insert(0, str(matcha))
    sys.path.insert(0, str(repo_root))

    model_dir = repo_root / "pretrained_models" / "CosyVoice2-0.5B"
    if not model_dir.exists():
        raise FileNotFoundError(
            f"CosyVoice2-0.5B not found at {model_dir}. "
            f"Run modelscope snapshot_download('iic/CosyVoice2-0.5B', "
            f"local_dir='{model_dir}') first."
        )

    # Redirect sys.stdout to stderr during model load so any third-party
    # print calls (modelscope download progress, transformers warnings,
    # CosyVoice's own debug prints) cannot pollute our protocol stdout.
    # _emit() uses _PROTOCOL_STDOUT, which is captured before this swap.
    # 在模型加载期间把 sys.stdout 改向 stderr：modelscope 下载进度 /
    # transformers 警告 / CosyVoice 自带 debug print 都不会污染协议 stdout。
    # _emit 用的是早早缓存的 _PROTOCOL_STDOUT，不受影响。
    saved_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        # Lazy import: any import error inside CosyVoice surfaces here, not
        # at module-load time, so the main process can read the failure JSON.
        # The pyright ignore is intentional — CosyVoice lives in the sibling
        # repo and is wired into sys.path above, not pip-installed.
        # 延迟导入：CosyVoice 内部 import 错误在这里触发，主进程能拿到错误 JSON。
        # type-ignore 是有意：CosyVoice 在 sibling 仓里通过 sys.path 注入，
        # 不是 pip 包，静态检查器无法解析。
        from cosyvoice.cli.cosyvoice import AutoModel  # type: ignore[import-not-found]
        _log(f"loading CosyVoice2 from {model_dir}")
        t0 = time.monotonic()
        model = AutoModel(model_dir=str(model_dir))
        _log(f"model ready in {time.monotonic() - t0:.1f}s; sample_rate={model.sample_rate}")
        return model
    finally:
        sys.stdout = saved_stdout


def _synthesize(model, task: dict) -> dict:
    """Run one synthesis task; write output WAV; return response dict.

    单次合成：按 mode 调相应 inference API，把音频写入 task["out"]，返回响应。

    Modes:
      - "zero_shot"     : inference_zero_shot(text, prompt_text, ref_path)
      - "cross_lingual" : inference_cross_lingual(text, ref_path)
      - "instruct"      : inference_instruct2(text, instruct_text, ref_path)
        instruct text 必须带 "<|endofprompt|>" 后缀（CosyVoice 2 约定）。

    Args:
        model: 已加载的 AutoModel 实例。
        task: 主进程发来的 JSON 任务。

    Returns:
        响应 dict：{"id", "ok": True, "out", "duration", "sample_rate"}。
    """
    import torch
    import torchaudio

    text: str = task["text"]
    ref_path: str = task["ref"]
    out_path = Path(task["out"])
    mode: str = task.get("mode", "zero_shot")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chunks = []
    if mode == "zero_shot":
        prompt_text = task.get("prompt_text", "")
        for j in model.inference_zero_shot(text, prompt_text, ref_path, stream=False):
            chunks.append(j["tts_speech"])
    elif mode == "cross_lingual":
        for j in model.inference_cross_lingual(text, ref_path, stream=False):
            chunks.append(j["tts_speech"])
    elif mode == "instruct":
        instruct = task.get("instruct") or ""
        if "<|endofprompt|>" not in instruct:
            instruct = instruct + "<|endofprompt|>"
        for j in model.inference_instruct2(text, instruct, ref_path, stream=False):
            chunks.append(j["tts_speech"])
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    if not chunks:
        raise RuntimeError("CosyVoice produced no output chunks")

    audio = torch.cat(chunks, dim=1)
    torchaudio.save(str(out_path), audio, model.sample_rate)
    return {
        "id": task.get("id"),
        "ok": True,
        "out": str(out_path),
        "duration": float(audio.shape[1] / model.sample_rate),
        "sample_rate": int(model.sample_rate),
    }


def main() -> None:
    """Stdin loop: read JSON tasks, write JSON responses; quit on EOF or {"action":"quit"}.

    主循环：从 stdin 逐行读 JSON 任务、写响应；遇到 EOF 或 quit 干净退出。
    """
    try:
        model = _load_model()
    except Exception as e:  # noqa: BLE001 — fatal: surface to stderr + exit 1
        _log(f"FATAL model load failed: {e}")
        traceback.print_exc(file=sys.stderr)
        # Best-effort emit of failure on stdout so main isn't left hanging.
        # 尽力在 stdout 发一条失败信号，避免主进程死等。
        try:
            _emit({"event": "fatal", "ok": False, "err": str(e)})
        except Exception:
            pass
        sys.exit(1)

    _emit({"event": "ready", "sample_rate": int(model.sample_rate)})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            task = json.loads(line)
        except json.JSONDecodeError as e:
            _emit({"ok": False, "err": f"JSON decode failed: {e}", "id": None})
            continue

        if task.get("action") == "quit":
            _log("quit requested; exiting")
            break

        try:
            resp = _synthesize(model, task)
            _emit(resp)
        except Exception as e:  # noqa: BLE001 — per-task failure shouldn't kill worker
            _log(f"task {task.get('id')} failed: {e}")
            traceback.print_exc(file=sys.stderr)
            _emit({"id": task.get("id"), "ok": False, "err": str(e)})


if __name__ == "__main__":
    main()
