#!/usr/bin/env python3
"""CV3 Zero-Shot Boundary Test — multi-dimensional emotion sweep.

Systematically probes CosyVoice 3 zero-shot across:
  - 5 emotions (Angry, Happy, Neutral, Sad, Surprise)
  - 3 target texts (prose, poem, news)
  - 3 seeds (42, 123, 456)
  - + Sad ablation probes (zero_st, zero_spk × 3 seeds)

Four-axis eval per synthesis:
  1. CER  (FunASR paraformer-zh)  — content accuracy
  2. SECS (WavLM-Base-Plus-SV)    — speaker similarity
  3. F0 RMSE (librosa.pyin)       — prosody transfer
  4. SLR  (char overlap)          — semantic leakage (OUR metric)

Output:
  boundary_test/audio/{emotion}/{text_id}/{condition}_seed{seed}.wav
  boundary_test/results.json
"""

import sys, os, json, time, logging
from pathlib import Path
from collections import defaultdict

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

# ═══════════════════════════════════════════════════════════════════════════
MD = "/root/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"
OUT = Path("/root/autodl-tmp/boundary_test")
AUDIO = OUT / "audio"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456]

# One ref per emotion from speaker 0008 (Chinese female).
# Each ref has a DIFFERENT text → if Sad leaks but Neutral doesn't,
# it's the emotion, not a particular text.
REFS = {
    "Sad":      ("/root/autodl-tmp/esd_cn/wavs/0008_001229.wav",
                 "她和维克分手了，所以她申请转调。"),
    "Neutral":  ("/root/autodl-tmp/esd_cn/wavs/0008_000064.wav",
                 "我告退了，晚上回去拍给你。"),
    "Happy":    ("/root/autodl-tmp/esd_cn/wavs/0008_000920.wav",
                 "所以你必须要坚持，时间越长越好。"),
    "Angry":    ("/root/autodl-tmp/esd_cn/wavs/0008_000497.wav",
                 "不知道。或许一双新鞋。"),
    "Surprise": ("/root/autodl-tmp/esd_cn/wavs/0008_001641.wav",
                 "请勿进入竹林。不让进。"),
}

TARGETS = {
    "prose": "春天来了，桃花开了，满山遍野都是粉红色的花朵，微风吹过，花瓣纷纷飘落。",
    "poem": "八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。",
    "news": "随着人工智能技术的飞速发展，语音合成系统已经能够以惊人的准确度模仿人类的声音特征。",
}

WAVLM_SNAP = "/root/.cache/huggingface/hub/models--microsoft--wavlm-base-plus-sv/snapshots/1bfd64eca136543feb28c5ffaf05381c6af33121"


# ═══════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════

def chars(s):
    return set(s.replace(" ", "").replace("，", "").replace("。", "")
               .replace("、", "").replace("！", "").replace("？", ""))

def compute_slr(asr_text, ref_text, target_text):
    """SLR = |ASR ∩ (ref_chars − target_chars)| / |ASR|"""
    asr_stripped = asr_text.replace(" ", "")
    ref_only = chars(ref_text) - chars(target_text)
    if not asr_stripped:
        return {"SLR": 0.0, "leaked": "", "ref_only": "".join(sorted(ref_only))}
    leaked = [c for c in asr_stripped if c in ref_only]
    return {
        "SLR": round(len(leaked) / len(asr_stripped), 4),
        "leaked": "".join(leaked),
        "ref_only": "".join(sorted(ref_only)),
    }

def edit_distance(a, b):
    m, n = len(a), len(b)
    d = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): d[i][0] = i
    for j in range(n+1): d[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1,
                          d[i-1][j-1]+(0 if a[i-1]==b[j-1] else 1))
    return d[m][n]

def compute_cer(asr_text, target_text):
    a = asr_text.replace(" ", "")
    b = target_text.replace(" ", "").replace("，", "").replace("。", "")
    return round(edit_distance(a, b) / max(1, len(b)), 4)

def f0_contour(path):
    import librosa
    y, sr = librosa.load(str(path), sr=16000)
    f0, voiced, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sr,
                                 frame_length=2048, hop_length=320)
    return np.nan_to_num(f0, nan=0.0), voiced.astype(bool)

def compute_f0_rmse(syn_wav, ref_wav):
    f0_syn, v_syn = f0_contour(syn_wav)
    f0_ref, v_ref = f0_contour(ref_wav)
    L = min(len(f0_syn), len(f0_ref))
    both = v_syn[:L] & v_ref[:L]
    if both.sum() < 5:
        return None
    return round(float(np.sqrt(np.mean((f0_syn[:L][both]-f0_ref[:L][both])**2))), 1)


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1: SYNTHESIS
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.monotonic()

    # ── Load CV3 ──
    print("Loading CV3 ...")
    model = AutoModel(model_dir=MD)
    frt = model.frontend
    cv3m = model.model
    print(f"  Ready, sr={model.sample_rate}\n")

    records = []
    total = 5 * 3 * len(SEEDS) + 2 * len(SEEDS)  # 45 + 6 = 51
    idx = 0

    for emotion, (ref_wav, ref_text) in REFS.items():
        if not Path(ref_wav).exists():
            print(f"SKIP {emotion}: not found")
            continue

        prompt = ref_text + "<|endofprompt|>"

        for tid, ttext in TARGETS.items():
            sentences = frt.text_normalize(ttext, split=False, text_frontend=True)
            mi_base = frt.frontend_zero_shot(
                str(sentences), prompt, ref_wav, model.sample_rate, "")

            # Pre-build ablation variants for Sad+prose
            if emotion == "Sad" and tid == "prose":
                zst = torch.zeros_like(mi_base["llm_prompt_speech_token"])
                zspk = torch.zeros_like(mi_base["llm_embedding"])
                ablations = {
                    "zero_st":  {**mi_base, "llm_prompt_speech_token": zst,
                                 "flow_prompt_speech_token": zst},
                    "zero_spk": {**mi_base, "llm_embedding": zspk,
                                 "flow_embedding": zspk},
                }
            else:
                ablations = {}

            # Normal condition + ablations
            all_conditions = [("", mi_base)] + list(ablations.items())

            for cond_name, mi in all_conditions:
                for seed in SEEDS:
                    idx += 1
                    torch.manual_seed(seed)
                    torch.cuda.empty_cache()

                    fname = f"{cond_name}_seed{seed}.wav" if cond_name else f"seed{seed}.wav"
                    out_wav = AUDIO / emotion / tid / fname
                    out_wav.parent.mkdir(parents=True, exist_ok=True)

                    label = f"[{idx:2d}/{total}] {emotion:10s} {tid:6s}"
                    if cond_name:
                        label += f" {cond_name}"
                    label += f" seed={seed}"

                    rec = {
                        "emotion": emotion, "text_id": tid, "seed": seed,
                        "condition": cond_name or "normal",
                        "ref_wav": ref_wav, "ref_text": ref_text,
                        "target_text": ttext, "wav_path": str(out_wav),
                    }

                    try:
                        ts = time.monotonic()
                        gen = cv3m.tts(**mi, stream=False)
                        chunks = [j["tts_speech"] for j in gen]
                        audio = torch.cat(chunks, dim=1)
                        torchaudio.save(str(out_wav), audio, model.sample_rate)
                        rec["duration_s"] = round(audio.shape[1] / model.sample_rate, 1)
                        rec["elapsed_s"] = round(time.monotonic() - ts, 1)
                        rec["status"] = "ok"
                        print(f"  {label}  dur={rec['duration_s']:.1f}s")
                    except Exception as e:
                        rec["status"] = "error"
                        rec["error"] = str(e)[:150]
                        print(f"  {label}  FAIL: {str(e)[:100]}")

                    records.append(rec)

    print(f"\nSynthesis done: {len(records)} records, "
          f"{sum(1 for r in records if r['status']=='ok')} ok\n")

    # Free CV3 memory
    del model, frt, cv3m
    torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 2a: ASR → CER + SLR
    # ═══════════════════════════════════════════════════════════════════════
    print("Loading FunASR ...")
    from funasr import AutoModel as FM
    asr = FM(model="paraformer-zh")
    print("  ASR ready\n")

    ok = [r for r in records if r["status"] == "ok"]
    for i, r in enumerate(ok):
        if (i+1) % 15 == 0:
            print(f"  ASR {i+1}/{len(ok)}")
        result = asr.generate(input=r["wav_path"])
        r["asr_text"] = result[0]["text"] if result else ""
        r["CER"] = compute_cer(r["asr_text"], r["target_text"])
        slr = compute_slr(r["asr_text"], r["ref_text"], r["target_text"])
        r["SLR"] = slr["SLR"]
        r["leaked_chars"] = slr["leaked"]
        r["ref_only_chars"] = slr["ref_only"]

    del asr
    torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 2b: WavLM → SECS
    # ═══════════════════════════════════════════════════════════════════════
    print("\nLoading WavLM ...")
    from transformers import WavLMConfig, WavLMModel
    from safetensors.torch import load_file
    cfg = WavLMConfig.from_pretrained(WAVLM_SNAP, local_files_only=True)
    wavlm = WavLMModel(cfg)
    wavlm.load_state_dict(load_file(f"{WAVLM_SNAP}/model.safetensors"), strict=False)
    wavlm = wavlm.cuda().eval()
    print("  WavLM ready\n")

    for i, r in enumerate(ok):
        if (i+1) % 15 == 0:
            print(f"  SECS {i+1}/{len(ok)}")
        try:
            import librosa
            def extract(path):
                y, _ = librosa.load(str(path), sr=16000)
                inp = torch.from_numpy(y.astype(np.float32)).unsqueeze(0).cuda()
                out = wavlm(inp, output_hidden_states=True)
                e = out.last_hidden_state.mean(dim=1)[:, :192]
                return torch.nn.functional.normalize(e, dim=-1)
            e_syn = extract(r["wav_path"])
            e_ref = extract(r["ref_wav"])
            r["SECS"] = round(max(0.0, (e_syn * e_ref).sum(dim=-1).item()), 4)
        except Exception as e:
            r["SECS"] = None

    del wavlm
    torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 2c: librosa.pyin → F0 RMSE
    # ═══════════════════════════════════════════════════════════════════════
    print("\nF0 RMSE ...")
    for i, r in enumerate(ok):
        if (i+1) % 15 == 0:
            print(f"  F0 {i+1}/{len(ok)}")
        try:
            r["F0_RMSE_Hz"] = compute_f0_rmse(r["wav_path"], r["ref_wav"])
        except Exception:
            r["F0_RMSE_Hz"] = None

    # ═══════════════════════════════════════════════════════════════════════
    # REPORT
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("EMOTION SWEEP — normal condition, prose text")
    print(f"{'='*80}")
    header = f"{'Emotion':<12s} {'SLR':>7s} {'±':>6s} {'CER':>7s} {'SECS':>7s} {'F0_RMSE':>8s} {'Dur':>6s}"
    print(header)
    print("-" * len(header))

    groups = defaultdict(list)
    for r in ok:
        groups[(r["emotion"], r["text_id"], r["condition"])].append(r)

    summary = {}
    for emo in ["Neutral", "Happy", "Angry", "Surprise", "Sad"]:
        for tid in ["prose", "poem", "news"]:
            key = (emo, tid, "normal")
            items = groups.get(key, [])
            if len(items) < 2:
                continue
            slrs = [it["SLR"] for it in items]
            cers = [it["CER"] for it in items]
            secs = [it["SECS"] for it in items if it["SECS"] is not None]
            f0s = [it["F0_RMSE_Hz"] for it in items if it["F0_RMSE_Hz"] is not None]
            durs = [it["duration_s"] for it in items]

            s = {
                "SLR_mean": round(np.mean(slrs), 3),
                "SLR_std": round(np.std(slrs, ddof=1), 3) if len(slrs) > 1 else 0,
                "CER_mean": round(np.mean(cers), 3),
                "SECS_mean": round(np.mean(secs), 3) if secs else None,
                "F0_RMSE_mean": round(np.mean(f0s), 1) if f0s else None,
                "Dur_mean": round(np.mean(durs), 1),
                "n": len(items),
            }
            summary[f"{emo}/{tid}/normal"] = s

            if tid == "prose":
                print(f"{emo:<12s} {s['SLR_mean']:7.3f} {s['SLR_std']:6.3f} "
                      f"{s['CER_mean']:7.3f} {s['SECS_mean']:7.3f} "
                      f"{s['F0_RMSE_mean']:8.1f} {s['Dur_mean']:6.1f}")

    # Ablation comparison
    print(f"\n{'='*80}")
    print("SAD ABLATION — Sad/prose")
    print(f"{'='*80}")
    for cond in ["normal", "zero_st", "zero_spk"]:
        items = groups.get(("Sad", "prose", cond), [])
        if len(items) < 2:
            continue
        slrs = [it["SLR"] for it in items]
        cers = [it["CER"] for it in items]
        secs = [it["SECS"] for it in items if it["SECS"] is not None]
        f0s = [it["F0_RMSE_Hz"] for it in items if it["F0_RMSE_Hz"] is not None]
        summary[f"Sad/prose/{cond}"] = {
            "SLR_mean": round(np.mean(slrs), 3),
            "SLR_std": round(np.std(slrs, ddof=1), 3) if len(slrs) > 1 else 0,
            "CER_mean": round(np.mean(cers), 3),
            "SECS_mean": round(np.mean(secs), 3) if secs else None,
            "F0_RMSE_mean": round(np.mean(f0s), 1) if f0s else None,
            "n": len(items),
        }
        s = summary[f"Sad/prose/{cond}"]
        print(f"  {cond:12s} SLR={s['SLR_mean']:.3f}±{s['SLR_std']:.3f}  "
              f"CER={s['CER_mean']:.3f}  SECS={s['SECS_mean']:.3f}  "
              f"F0_RMSE={s['F0_RMSE_mean']:.1f}")

    # Per-run ASR detail for key conditions
    print(f"\n{'='*80}")
    print("PER-RUN ASR DETAIL — spot-check leakage")
    print(f"{'='*80}")
    for emo in ["Sad", "Neutral"]:
        items = groups.get((emo, "prose", "normal"), [])
        for r in sorted(items, key=lambda x: x["seed"]):
            print(f"\n  [{emo}] seed={r['seed']}  SLR={r['SLR']:.3f}  CER={r['CER']:.3f}")
            print(f"    REF:  {r['ref_text']}")
            print(f"    TGT:  {r['target_text'][:60]}...")
            print(f"    ASR:  {r.get('asr_text', '(n/a)')[:100]}")

    # Save
    result_path = OUT / "results.json"
    json.dump({
        "config": {
            "refs": {emo: {"wav": w, "text": t} for emo, (w, t) in REFS.items()},
            "targets": TARGETS,
            "seeds": SEEDS,
        },
        "summary": summary,
        "runs": ok,
    }, open(result_path, "w"), ensure_ascii=False, indent=2)
    print(f"\nResults: {result_path}")

    # Print directory tree for user navigation
    print(f"\n{'='*80}")
    print("AUDIO DIRECTORY — navigate and listen")
    print(f"{'='*80}")
    import subprocess
    subprocess.run(["find", str(AUDIO), "-type", "f", "-name", "*.wav",
                    "-printf", "%P\n"], stdout=subprocess.PIPE)
    # Simplified: just print the structure
    for emo_dir in sorted(AUDIO.iterdir()):
        if emo_dir.is_dir():
            print(f"\n  {emo_dir.name}/")
            for text_dir in sorted(emo_dir.iterdir()):
                if text_dir.is_dir():
                    files = sorted(text_dir.glob("*.wav"))
                    print(f"    {text_dir.name}/  ({len(files)} wavs)")
                    for f in files:
                        size_kb = f.stat().st_size / 1024
                        print(f"      {f.name}  ({size_kb:.0f} KB)")

    elapsed = (time.monotonic() - t0) / 60
    print(f"\nTotal: {elapsed:.0f} min")
    print("Done.")


if __name__ == "__main__":
    main()
