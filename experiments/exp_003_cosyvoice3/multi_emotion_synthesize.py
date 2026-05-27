#!/usr/bin/env python3
"""Multi-emotion zero-shot + LoRA synthesis for exp_003 interactive report.

Run on H800/4090 with GPU:
  python multi_emotion_synthesize.py \
    --esd_manifest /root/autodl-fs/voice_story/datasets/esd/manifest.jsonl \
    --output_dir outputs \
    --lora_ckpt /root/autodl-fs/voice_story/experiments/exp_003_cosyvoice3/ckpt_r8/checkpoint-200

Generates (text × emotion × condition) wavs + eval metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchaudio

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
COSYVOICE_ROOT = Path(os.environ.get("COSYVOICE_ROOT", "/root/autodl-fs/CosyVoice"))
sys.path.insert(0, str(COSYVOICE_ROOT))
sys.path.insert(0, str(COSYVOICE_ROOT / "third_party" / "Matcha-TTS"))
sys.path.insert(0, str(REPO_ROOT))

from cosyvoice.cli.cosyvoice import AutoModel


# ── Test texts (same 4 categories as exp_003 baseline) ────────────

TEST_TEXTS = {
    "zh_poem":   "八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。",
    "zh_news":   "随着人工智能技术的飞速发展，语音合成系统已经能够以惊人的准确度模仿人类的声音特征。",
    "zh_prose":  "春天来了，桃花开了，满山遍野都是粉红色的花朵，微风吹过，花瓣纷纷飘落。",
    "zh_ancient":"自三峡七百里中，两岸连山，略无阙处。重岩叠嶂，隐天蔽日。",
}

TEST_EMOTIONS = ["neutral", "angry", "happy", "sad", "surprise"]


# ── Helpers ───────────────────────────────────────────────────────

def _pick_emotion_refs(manifest_path: Path, n_per_emotion: int = 1) -> dict[str, list[dict]]:
    """Pick best-MOS ESD Chinese refs per emotion."""
    emo_refs = defaultdict(list)
    with manifest_path.open() as f:
        for line in f:
            r = json.loads(line.strip())
            if r.get("lang") != "zh": continue
            emo = r.get("emotion_tag", "unknown")
            dur = r.get("duration", 0)
            mos = r.get("mos_ovr", 0)
            if 3 <= dur <= 10 and mos >= 3.0:
                emo_refs[emo].append(r)
    # Pick top-MOS per emotion
    picked = {}
    for emo, refs in emo_refs.items():
        refs.sort(key=lambda x: x.get("mos_ovr", 0), reverse=True)
        picked[emo] = refs[:n_per_emotion]
    return picked


def _load_lora_llm(model, ckpt_path: str, rank: int = 8) -> object:
    """Load LoRA weights into CV3 LLM backbone."""
    from peft import PeftModel
    llm = model.model.llm  # CosyVoice3Model has .llm submodule
    peft = PeftModel.from_pretrained(llm, ckpt_path)
    model.model.llm = peft.merge_and_unload()  # merge for faster inference
    print(f"  LoRA r={rank} merged into LLM backbone")
    return model


def _synth(model, text: str, prompt_text: str, prompt_wav: str, mode: str = "zero_shot") -> torch.Tensor:
    """Synthesize, return waveform (1, T)."""
    if "<|endofprompt|>" not in prompt_text:
        prompt_text = prompt_text + "<|endofprompt|>"

    chunks = []
    if mode == "zero_shot":
        gen = model.inference_zero_shot(text, prompt_text, prompt_wav, stream=False)
    elif mode == "instruct":
        gen = model.inference_instruct2(text, prompt_text, prompt_wav, stream=False)
    else:
        raise ValueError(f"unknown mode: {mode}")

    for j in gen:
        chunks.append(j["tts_speech"])
    return torch.cat(chunks, dim=1)


def _eval_one(syn_wav: Path, ref_wav: Path, target_text: str):
    """Run 4-axis eval on one synthetic wav. Returns dict."""
    from core.eval_tts import evaluate_synthesis
    scores = evaluate_synthesis(syn_wav, ref_wav=ref_wav, target_text=target_text, lang="zh")
    return {
        "mos_nisqa": scores.mos_nisqa,
        "secs": scores.secs,
        "cer": scores.cer if scores.cer is not None else None,
        "f0_rmse_hz": scores.f0_rmse_hz,
    }


# ── Main ──────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esd_manifest", required=True, help="ESD manifest.jsonl path")
    ap.add_argument("--model_dir", default="/root/autodl-fs/models/CosyVoice3-0.5B")
    ap.add_argument("--output_dir", default="outputs")
    ap.add_argument("--lora_ckpt", default=None, help="LoRA checkpoint dir (peft format)")
    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--n_refs_per_emotion", type=int, default=1)
    ap.add_argument("--skip_eval", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Pick ESD emotion refs
    emo_refs = _pick_emotion_refs(Path(args.esd_manifest), args.n_refs_per_emotion)
    print(f"Emotion refs: { {k: len(v) for k, v in emo_refs.items()} }")
    available_emos = sorted(set(TEST_EMOTIONS) & set(emo_refs.keys()))
    if not available_emos:
        print("ERROR: no emotion refs matched. Check ESD manifest emo distribution.")
        sys.exit(1)
    print(f"Available emotions for synthesis: {available_emos}")

    # 2. Load base model
    print(f"\nLoading CV3 from {args.model_dir} ...")
    t0 = time.monotonic()
    model = AutoModel(model_dir=args.model_dir)
    print(f"  ready in {time.monotonic()-t0:.1f}s, sr={model.sample_rate}")

    # 3. Load LoRA checkpoint
    lora_model = None
    if args.lora_ckpt:
        print(f"\nLoading LoRA ckpt from {args.lora_ckpt} ...")
        lora_model = _load_lora_llm(model, args.lora_ckpt, args.lora_rank)

    # 4. Synthesize
    all_eval = []
    for text_id, text_content in TEST_TEXTS.items():
        for emo in available_emos:
            ref = emo_refs[emo][0]
            ref_wav_path = os.path.expandvars(ref["audio_path"])
            ref_text = ref["text"]

            # Zero-shot baseline
            print(f"\n[{text_id}/{emo}] zero_shot ...")
            t1 = time.monotonic()
            wav = _synth(model, text_content, ref_text, ref_wav_path, "zero_shot")
            zs_path = out / "multi_emotion" / f"baseline_{text_id}_{emo}.wav"
            zs_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(str(zs_path), wav, model.sample_rate)
            print(f"  wrote {zs_path} ({wav.shape[1]/model.sample_rate:.1f}s) in {time.monotonic()-t1:.1f}s")

            # LoRA (if ckpt given)
            if lora_model is not None:
                print(f"[{text_id}/{emo}] LoRA r={args.lora_rank} ...")
                t1 = time.monotonic()
                wav = _synth(lora_model, text_content, ref_text, ref_wav_path, "zero_shot")
                lora_path = out / "multi_emotion" / f"lora_r{args.lora_rank}_{text_id}_{emo}.wav"
                torchaudio.save(str(lora_path), wav, model.sample_rate)
                print(f"  wrote {lora_path} ({wav.shape[1]/model.sample_rate:.1f}s) in {time.monotonic()-t1:.1f}s")

            # Eval
            if not args.skip_eval:
                import subprocess
                ref_wav = Path(ref_wav_path)
                if ref_wav.exists():
                    try:
                        scores = _eval_one(zs_path, ref_wav, text_content)
                        scores["id"] = f"baseline_{text_id}_{emo}"
                        scores["text"] = text_content
                        scores["emotion"] = emo
                        scores["cond"] = "baseline"
                        all_eval.append(scores)
                        print(f"  eval: MOS={scores['mos_nisqa']:.2f} SECS={scores['secs']:.3f} CER={scores['cer']}")
                    except Exception as e:
                        print(f"  eval failed: {e}")

    # 5. Save eval JSON
    if all_eval:
        eval_path = out / "multi_emotion_eval.json"
        json.dump(all_eval, open(eval_path, "w"), ensure_ascii=False, indent=2)
        print(f"\nEval saved to {eval_path} ({len(all_eval)} entries)")

        # Print JS array snippet for copy-paste into report.html
        print("\n// --- Copy into report.html RAW[] ---")
        for e in all_eval:
            print(f"  {{text:\"{e['id'].split('_',2)[-1] if len(e['id'].split('_'))>2 else e['id']}\","+
                  f" label:\"\", emotion:\"{e['emotion']}\", cond:\"{e['cond']}\","+
                  f" mos_nisqa:{e['mos_nisqa']:.3f}, secs:{e['secs']:.3f}, cer:{e['cer']:.3f}, f0_rmse_hz:{e['f0_rmse_hz']:.1f}, asr:\"\"}},")

    print("\nDone.")


if __name__ == "__main__":
    main()
