#!/usr/bin/env python3
"""Multi-run SLR experiment — quantify semantic leakage with error bars.

Runs each condition 3 times with different torch seeds to measure
inference-time variance in CosyVoice 3's stochastic decoding.

7 conditions × 3 seeds = 21 syntheses on a single (ref, target) pair
known to trigger leakage (Sad emotion ref).

Outputs per run:
  - WAV file
  - ASR transcript
  - SLR (character overlap with ref text)
  - Duration

Aggregate: mean ± std per condition.
"""

import sys, os, json, time, logging
from pathlib import Path

import numpy as np
import torch
import torchaudio

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, "/root/CosyVoice")
sys.path.insert(0, "/root/CosyVoice/third_party/Matcha-TTS")
from cosyvoice.cli.cosyvoice import AutoModel

logging.basicConfig(level=logging.WARNING, format="%(message)s")
log = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────
MD = "/root/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"
OUT = Path("/root/autodl-tmp/slr_multirun")
OUT.mkdir(parents=True, exist_ok=True)

REF_WAV = "/root/autodl-tmp/esd_cn/wavs/0008_001229.wav"
REF_TEXT = "她和维克分手了，所以她申请转调。"

TARGET_TEXT = "春天来了，桃花开了，柳树绿了，小鸟在枝头唱歌。"

SEEDS = [42, 123, 456]
N_RUNS = len(SEEDS)

# ── SLR metric ──────────────────────────────────────────────────────────

def compute_slr(asr_text: str, ref_text: str, target_text: str) -> dict:
    """Compute semantic leakage rate with common-char correction.

    SLR = |ASR_chars ∩ ref_only_chars| / |ASR_chars|

    where ref_only_chars = ref characters NOT in target.
    This avoids inflating SLR from common function words (e.g. "了", "的")
    that appear in both ref and target text.
    """
    def char_set(s):
        return set(s.replace(" ", "").replace("，", "").replace("。", "").replace("、", ""))

    asr_chars = asr_text.replace(" ", "")
    ref_set = char_set(ref_text)
    target_set = char_set(target_text)
    ref_only = ref_set - target_set  # chars unique to ref

    if not asr_chars:
        return {
            "asr_len": 0,
            "ref_only_chars": "".join(sorted(ref_only)),
            "n_leaked": 0,
            "SLR": 0.0,
        }

    leaked = [c for c in asr_chars if c in ref_only]
    slr = len(leaked) / len(asr_chars)

    return {
        "asr_len": len(asr_chars),
        "ref_only_chars": "".join(sorted(ref_only)),
        "n_leaked": len(leaked),
        "leaked_chars": "".join(leaked),
        "SLR": round(slr, 4),
    }


# ── Helpers ─────────────────────────────────────────────────────────────

@torch.no_grad()
def wavlm_embedding(wav_path: str, model_wavlm):
    """Extract 192-dim WavLM speaker embedding (mean-pooled)."""
    import librosa
    y, _ = librosa.load(str(wav_path), sr=16000)
    inp = torch.from_numpy(y.astype(np.float32)).unsqueeze(0).cuda()
    out = model_wavlm(inp, output_hidden_states=True)
    e = out.last_hidden_state.mean(dim=1)[:, :192]
    return e


def synthesize(condition: str, modify_fn, seed: int, run_idx: int, log=log):
    """Run one synthesis and return result dict."""
    torch.manual_seed(seed)
    torch.cuda.empty_cache()

    prompt_text = REF_TEXT + "<|endofprompt|>"
    sentences = frontend.text_normalize(TARGET_TEXT, split=False, text_frontend=True)
    mi = frontend.frontend_zero_shot(
        str(sentences), prompt_text, REF_WAV, model.sample_rate, ""
    )

    mi2 = modify_fn(mi)

    wav_out = OUT / f"{condition}_seed{seed}.wav"
    try:
        t0 = time.monotonic()
        gen = cv3m.tts(**mi2, stream=False)
        chunks = [j["tts_speech"] for j in gen]
        audio = torch.cat(chunks, dim=1)
        torchaudio.save(str(wav_out), audio, model.sample_rate)
        duration_s = audio.shape[1] / model.sample_rate
        elapsed = time.monotonic() - t0
    except Exception as e:
        return {"condition": condition, "seed": seed, "run": run_idx,
                "error": str(e)[:200]}

    # ASR
    r = asr_model.generate(input=str(wav_out))
    asr_text = r[0]["text"] if r else ""

    # SLR
    slr_info = compute_slr(asr_text, REF_TEXT, TARGET_TEXT)

    log.info("  [%d/%d] %-18s seed=%d  dur=%.1fs  SLR=%.3f  asr=%s",
             run_idx, N_RUNS, condition, seed, duration_s, slr_info["SLR"],
             asr_text[:80])

    return {
        "condition": condition,
        "seed": seed,
        "run": run_idx,
        "duration_s": round(duration_s, 2),
        "elapsed_s": round(elapsed, 1),
        "asr_text": asr_text,
        **slr_info,
    }


# ── Main ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("SLR Multi-Run Experiment")
    print(f"  Ref:  {REF_TEXT}")
    print(f"  Target: {TARGET_TEXT}")
    print(f"  Conditions: 7 × runs: {N_RUNS} = {7 * N_RUNS} syntheses")
    print("=" * 70)

    # ── Load CV3 once ──
    print("\n[1/4] Loading CosyVoice 3 ...")
    t0 = time.monotonic()
    model = AutoModel(model_dir=MD)
    print(f"  Loaded in {time.monotonic() - t0:.0f}s, sr={model.sample_rate}")
    frontend = model.frontend
    cv3m = model.model

    # ── Load WavLM for "wavlm" / "wavlm_zero" conditions ──
    print("\n[2/4] Loading WavLM ...")
    from transformers import WavLMConfig, WavLMModel
    from safetensors.torch import load_file
    WAVLM_SNAP = "/root/.cache/huggingface/hub/models--microsoft--wavlm-base-plus-sv/snapshots/1bfd64eca136543feb28c5ffaf05381c6af33121"
    cfg = WavLMConfig.from_pretrained(WAVLM_SNAP, local_files_only=True)
    wm = WavLMModel(cfg)
    sd = load_file(f"{WAVLM_SNAP}/model.safetensors")
    wm.load_state_dict(sd, strict=False)
    wm = wm.cuda().eval()
    print("  WavLM ready")

    # ── Load ASR ──
    print("\n[3/4] Loading FunASR (paraformer-zh) ...")
    from funasr import AutoModel as FM
    asr_model = FM(model="paraformer-zh")
    print("  ASR ready")

    # ── Build condition table ──
    # Pre-compute zero tensors and WavLM embedding ONCE
    print("\n[4/4] Pre-extracting condition inputs ...")
    prompt_text = REF_TEXT + "<|endofprompt|>"
    sentences = frontend.text_normalize(TARGET_TEXT, split=False, text_frontend=True)
    mi_template = frontend.frontend_zero_shot(
        str(sentences), prompt_text, REF_WAV, model.sample_rate, ""
    )
    zst = torch.zeros_like(mi_template["llm_prompt_speech_token"])
    zspk = torch.zeros_like(mi_template["llm_embedding"])
    wavlm_emb = wavlm_embedding(REF_WAV, wm)

    # Apply prefix mask (used in condition 6)
    from cosyvoice.cli.content_mask import mask_speech_tokens as mask_st
    masked_st_prefix, _ = mask_st(
        mi_template["llm_prompt_speech_token"],
        prompt_wav=REF_WAV, prompt_text=REF_TEXT, lang="zh", mode="prefix",
    )

    CONDITIONS = [
        ("normal",      lambda mi: mi),
        ("zero_st",     lambda mi: {**mi,
            "llm_prompt_speech_token": zst,
            "flow_prompt_speech_token": zst}),
        ("zero_spk",    lambda mi: {**mi,
            "llm_embedding": zspk,
            "flow_embedding": zspk}),
        ("zero_both",   lambda mi: {**mi,
            "llm_prompt_speech_token": zst,
            "flow_prompt_speech_token": zst,
            "llm_embedding": zspk,
            "flow_embedding": zspk}),
        ("wavlm",       lambda mi: {**mi,
            "llm_embedding": wavlm_emb,
            "flow_embedding": wavlm_emb}),
        ("wavlm_zero",  lambda mi: {**mi,
            "llm_embedding": wavlm_emb,
            "flow_embedding": wavlm_emb,
            "llm_prompt_speech_token": zst,
            "flow_prompt_speech_token": zst}),
        ("prefix_mask", lambda mi: {**mi,
            "llm_prompt_speech_token": masked_st_prefix,
            "flow_prompt_speech_token": masked_st_prefix}),
    ]

    print(f"  {len(CONDITIONS)} conditions ready\n")

    # ── Run all ──
    all_results = []
    total = len(CONDITIONS) * N_RUNS
    idx = 0

    for cond_name, modify_fn in CONDITIONS:
        for run_i, seed in enumerate(SEEDS):
            idx += 1
            print(f"[{idx}/{total}] {cond_name} seed={seed} ...", end=" ", flush=True)
            result = synthesize(cond_name, modify_fn, seed, run_i)
            all_results.append(result)

    # ── Aggregate ──
    print("\n" + "=" * 70)
    print("RESULTS: mean ± std (N={})".format(N_RUNS))
    print("=" * 70)
    print(f"{'Condition':<18s} {'SLR_mean':>8s} {'SLR_±':>8s} {'SLR_range':>16s} {'Dur_mean':>8s}")
    print("-" * 65)

    from collections import defaultdict
    groups = defaultdict(list)
    for r in all_results:
        groups[r["condition"]].append(r)

    summary = {}
    for cond_name, _ in CONDITIONS:
        items = groups[cond_name]
        slrs = [it["SLR"] for it in items if "error" not in it]
        durs = [it["duration_s"] for it in items if "error" not in it]
        errors = [it for it in items if "error" in it]

        if not slrs:
            print(f"{cond_name:<18s}  ALL {len(errors)} RUNS FAILED")
            summary[cond_name] = {"error": f"{len(errors)} failures"}
            continue

        mean_slr = np.mean(slrs)
        std_slr = np.std(slrs, ddof=1)
        slr_range = f"{min(slrs):.3f}–{max(slrs):.3f}"
        mean_dur = np.mean(durs)

        flag = " ⚠️" if errors else ""
        print(f"{cond_name:<18s} {mean_slr:8.3f} {std_slr:8.3f} {slr_range:>16s} {mean_dur:8.1f}s{flag}")

        summary[cond_name] = {
            "SLR_mean": round(mean_slr, 4),
            "SLR_std": round(std_slr, 4),
            "SLR_range": slr_range,
            "SLR_values": [round(s, 4) for s in slrs],
            "dur_mean_s": round(mean_dur, 1),
            "n_runs": len(slrs),
            "n_errors": len(errors),
            "asr_texts": [it.get("asr_text", "") for it in items if "error" not in it],
            "leaked_chars_per_run": [it.get("leaked_chars", "") for it in items if "error" not in it],
        }

    # ── Per-condition detail ──
    print("\n" + "=" * 70)
    print("PER-RUN DETAIL")
    print("=" * 70)
    for cond_name, _ in CONDITIONS:
        items = groups[cond_name]
        print(f"\n--- {cond_name} ---")
        for it in items:
            if "error" in it:
                print(f"  seed={it['seed']} ERROR: {it['error'][:100]}")
            else:
                print(f"  seed={it['seed']}  SLR={it['SLR']:.3f}  "
                      f"leaked=[{it.get('leaked_chars','')}]  "
                      f"ASR={it['asr_text'][:80]}")

    # ── Save ──
    result_path = OUT / "slr_multirun_results.json"
    json.dump({
        "config": {
            "ref_wav": REF_WAV,
            "ref_text": REF_TEXT,
            "target_text": TARGET_TEXT,
            "seeds": SEEDS,
            "n_runs": N_RUNS,
        },
        "summary": summary,
        "runs": all_results,
    }, open(result_path, "w"), ensure_ascii=False, indent=2)
    print(f"\nSaved to {result_path}")

    # ── Key insight: relative SLR increase ──
    print("\n" + "=" * 70)
    print("KEY COMPARISON (Δ from normal baseline)")
    print("=" * 70)
    base = summary.get("normal", {}).get("SLR_mean", 0)
    for cond_name, _ in CONDITIONS:
        m = summary.get(cond_name, {}).get("SLR_mean")
        if m is not None and base > 0:
            delta = m - base
            direction = "↑ MORE LEAK" if delta > 0.01 else ("↓ LESS" if delta < -0.01 else "≈ SAME")
            print(f"  {cond_name:<18s} Δ={delta:+.3f}  {direction}")

    print("\nDone.")
