"""Long-running CosyVoice 3 worker — same JSON-line protocol as tts_worker.py.

This script is a near-clone of ``core/tts_worker.py``; only the model
loading path changes. CosyVoice 3's `AutoModel` auto-detects the version
from the presence of `cosyvoice3.yaml` in the model directory, and
`CosyVoice3` inherits `CosyVoice2`, so the inference API
(``inference_zero_shot`` / ``inference_instruct2``) is identical.

See ``core/tts_worker.py`` for the full protocol documentation.

CosyVoice 3 长驻 worker——与 core/tts_worker.py 共享完全相同的 JSON-line
协议，仅模型路径不同。CosyVoice 3 的 AutoModel 通过模型目录中
cosyvoice3.yaml 自动识别 v3；CosyVoice3 继承 CosyVoice2，推理 API
完全不变。
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path


def _log(*args, **kwargs) -> None:
    print("[tts_worker_cv3]", *args, **kwargs, file=sys.stderr, flush=True)


_PROTOCOL_STDOUT = sys.stdout


def _emit(obj: dict) -> None:
    _PROTOCOL_STDOUT.write(json.dumps(obj, ensure_ascii=False) + "\n")
    _PROTOCOL_STDOUT.flush()


def _model_dir() -> Path:
    """Resolve the CosyVoice3 model directory.

    解析 CosyVoice3 模型目录。
    """
    import os
    env = os.environ.get("COSYVOICE3_MODEL_DIR", "")
    if env:
        return Path(env)
    # Default: sibling CosyVoice repo's pretrained_models
    # 默认：兄弟仓 CosyVoice 下的 pretrained_models
    repo = Path(__file__).resolve().parent.parent.parent / "CosyVoice"
    return repo / "pretrained_models" / "CosyVoice3-0.5B"


def _load_model() -> object:
    repo_root = Path(__file__).resolve().parent.parent.parent / "CosyVoice"
    matcha = repo_root / "third_party" / "Matcha-TTS"
    sys.path.insert(0, str(matcha))
    sys.path.insert(0, str(repo_root))

    model_dir = _model_dir()
    if not model_dir.exists():
        raise FileNotFoundError(
            f"CosyVoice3 model not found at {model_dir}. "
            f"Run: python -c \"from modelscope import snapshot_download; "
            f"snapshot_download('iic/CosyVoice3-0.5B', local_dir='{model_dir}')\""
        )

    saved_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        from cosyvoice.cli.cosyvoice import AutoModel  # type: ignore[import-not-found]
        _log(f"loading CosyVoice3 from {model_dir}")
        t0 = time.monotonic()
        model = AutoModel(model_dir=str(model_dir))
        _log(f"model ready in {time.monotonic() - t0:.1f}s; sample_rate={model.sample_rate}")
        return model
    finally:
        sys.stdout = saved_stdout


def _synthesize(model, task: dict) -> dict:
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
        # CosyVoice 3 LLM hardcodes assert 151646 in text — requires
        # <|endofprompt|> in prompt_text even for zero_shot.
        # CV3 LLM 硬编码了 assert 151646 in text —— zero_shot 也必须在
        # prompt_text 里带 <|endofprompt|>。
        if "<|endofprompt|>" not in prompt_text:
            prompt_text = prompt_text + "<|endofprompt|>"
        for j in model.inference_zero_shot(text, prompt_text, ref_path, stream=False):
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
        raise RuntimeError("CosyVoice3 produced no output chunks")

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
    try:
        model = _load_model()
    except Exception as e:
        _log(f"FATAL model load failed: {e}")
        traceback.print_exc(file=sys.stderr)
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
        except Exception as e:
            _log(f"task {task.get('id')} failed: {e}")
            traceback.print_exc(file=sys.stderr)
            _emit({"id": task.get("id"), "ok": False, "err": str(e)})


if __name__ == "__main__":
    main()
