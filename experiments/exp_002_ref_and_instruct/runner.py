"""Experiment 002 runner: read config.yaml, drive one TTS worker, dump wavs.

Reads the matrix in config.yaml and synthesizes every (target × ref ×
instruct) combination. Spawns ONE long-lived TTS worker (CosyVoice 2 model
loaded once, ~140s on Mac) and reuses it across all matrix entries, so the
per-task amortized cost is just the inference time (~10-30s/sentence).

Outputs land in `outputs/{target_id}__{ref_id}__{instruct_id}.wav` and a
`results.md` summarizes timings + which strategies actually ran.

实验 002 runner：读 config.yaml，spawn 一次 TTS worker 跑完整 matrix。

模型加载只发生一次（~140s on Mac），之后每个 matrix 任务只付推理时间。
输出 wav 命名按 (target × ref × instruct) 组合，方便对比听感。
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.tts import get_tts_backend  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EXP_DIR / "config.yaml"
OUTPUT_DIR = EXP_DIR / "outputs"
RESULTS_PATH = EXP_DIR / "results.md"

logging.basicConfig(
    level=logging.INFO, stream=sys.stderr,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("exp_002")


def _load_config() -> dict:
    """Parse config.yaml and index references / instruct_prompts / targets by id.

    解析 config.yaml，把 references / instruct_prompts / target_texts
    按 id 建索引方便查表，避免每次 matrix 解析重新扫线性表。
    """
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return {
        "targets": {t["id"]: t for t in raw["target_texts"]},
        "refs": {r["id"]: r for r in raw["references"]},
        "instructs": {p["id"]: p for p in raw["instruct_prompts"]},
        "matrix": raw["matrix"],
    }


def _normalize_instruct(text: str | None) -> str | None:
    """Append the CosyVoice 2 instruct end-token if missing.

    CosyVoice 2 instruct 模式约定指令末尾必须有 <|endofprompt|>；
    用户写没写都行，runner 自动补。None 直接返回（zero_shot 模式）。
    """
    if text is None:
        return None
    if "<|endofprompt|>" in text:
        return text
    return text + "<|endofprompt|>"


def main() -> None:
    """Run all matrix entries through one shared TTS worker.

    一个 worker 跑完整 matrix；每条任务记录耗时 + 状态。
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _load_config()

    # Pre-validate matrix entries — fail-fast on typos before loading the
    # 7.6GB model. 在加载 7.6GB 模型前先做一遍 matrix 校验。
    rows = []
    for entry in cfg["matrix"]:
        target_id, ref_id, instruct_id = entry
        if target_id not in cfg["targets"]:
            raise KeyError(f"unknown target_id {target_id!r}")
        if ref_id not in cfg["refs"]:
            raise KeyError(f"unknown ref_id {ref_id!r}")
        if instruct_id not in cfg["instructs"]:
            raise KeyError(f"unknown instruct_id {instruct_id!r}")
        rows.append((target_id, ref_id, instruct_id))

    logger.info("matrix: %d entries", len(rows))

    log_lines: list[str] = []
    log_lines.append(f"# Experiment 002 results\n\nrun at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_lines.append(f"\n| # | target | ref | instruct | mode | wall (s) | output |\n|---|---|---|---|---|---|---|\n")

    t_start = time.monotonic()
    with get_tts_backend("local", startup_timeout=300, task_timeout=300) as tts:
        load_sec = time.monotonic() - t_start
        logger.info("worker ready (sample_rate=%s) in %.1fs", tts.sample_rate, load_sec)
        log_lines.append(f"\n_worker ready in {load_sec:.1f}s; sample_rate={tts.sample_rate}_\n")

        for i, (target_id, ref_id, instruct_id) in enumerate(rows, start=1):
            target = cfg["targets"][target_id]
            ref = cfg["refs"][ref_id]
            instruct_cfg = cfg["instructs"][instruct_id]

            text = target["text"]
            ref_path = REPO_ROOT / ref["audio"]
            prompt_text = ref["prompt_text"]
            instruct_text = _normalize_instruct(instruct_cfg.get("text"))
            mode = "instruct" if instruct_text else "zero_shot"

            out_name = f"{target_id}__{ref_id}__{instruct_id}.wav"
            out_path = OUTPUT_DIR / out_name

            logger.info("[%d/%d] %s | mode=%s", i, len(rows), out_name, mode)
            t0 = time.monotonic()
            try:
                tts.synthesize(
                    text, ref_path,
                    out_path=out_path,
                    prompt_text=prompt_text,
                    instruct=instruct_text,
                    mode=mode,
                )
                wall = time.monotonic() - t0
                status = f"{wall:.1f}"
                rel_out = out_path.relative_to(EXP_DIR)
                log_lines.append(f"| {i} | {target_id} | {ref_id} | {instruct_id} | {mode} | {status} | `{rel_out}` |\n")
                logger.info("  → %s (%.1fs)", out_path.name, wall)
            except Exception as e:  # noqa: BLE001 — record per-task failure
                wall = time.monotonic() - t0
                log_lines.append(f"| {i} | {target_id} | {ref_id} | {instruct_id} | {mode} | FAIL ({wall:.1f}) | {e!r} |\n")
                logger.warning("  → FAILED: %s", e)

    total = time.monotonic() - t_start
    log_lines.append(f"\n_total wall: {total:.1f}s ({total / 60:.1f}min)_\n")
    RESULTS_PATH.write_text("".join(log_lines), encoding="utf-8")
    logger.info("wrote %s", RESULTS_PATH)


if __name__ == "__main__":
    main()
