#!/usr/bin/env python3
"""Eval-only: ASR→CER+SLR, WavLM→SECS, librosa→F0 RMSE on pre-synthesized WAVs.

Reconstructs metadata from directory structure (speaker/emotion/text_id/seed.wav)
and re-runs deterministic ref selection to recover ref_wav/ref_text.
"""

import sys, os, json, time, random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Unbuffered stdout for real-time progress
sys.stdout.reconfigure(line_buffering=True)

AUDIO = Path("/root/autodl-tmp/boundary_test_multispk/audio")
OUT = Path("/root/autodl-tmp/boundary_test_multispk")
EMOTIONS = ["Neutral", "Happy", "Angry", "Surprise", "Sad"]
SEEDS = [42, 123, 456]

TARGETS = {
    "prose": "春天来了，桃花开了，满山遍野都是粉红色的花朵，微风吹过，花瓣纷纷飘落。",
    "dialogue": "你今天去超市了吗？记得买牛奶和面包，冰箱里已经没什么吃的了。",
    "story": "老张每天早上六点起床，泡一杯浓茶，坐在阳台上看报纸，这个习惯已经坚持了三十年。",
    "news": "随着人工智能技术的飞速发展，语音合成系统已经能够以惊人的准确度模仿人类的声音特征。",
    "emotional": "那一刻我突然明白了，有些告别不是结束，而是另一种开始，泪水模糊了我的视线。",
    "poem": "八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。",
}

WAVLM_SNAP = "/root/.cache/huggingface/hub/models--microsoft--wavlm-base-plus-sv/snapshots/1bfd64eca136543feb28c5ffaf05381c6af33121"

# ═══════════════════════════════════════════════════════════════════════════
# REF SELECTION (same deterministic logic as synthesis script)
# ═══════════════════════════════════════════════════════════════════════════

def select_refs(data_list_path, speakers, emotions):
    pool = defaultdict(list)
    with open(data_list_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            key = parts[0]
            wav = parts[1]
            text = parts[2]
            fields = key.split("_")
            spk = fields[1]
            emo = fields[2]
            pool[(spk, emo)].append((wav, text))

    refs = {}
    random.seed(42)
    for spk in speakers:
        for emo in emotions:
            candidates = pool.get((spk, emo), [])
            if not candidates:
                continue
            wav, text = random.choice(candidates)
            refs[(spk, emo)] = (wav, text)
    return refs

# ═══════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════

def chars(s):
    return set(s.replace(" ", "").replace("，", "").replace("。", "")
               .replace("、", "").replace("！", "").replace("？", ""))

def compute_slr(asr_text, ref_text, target_text):
    asr_s = asr_text.replace(" ", "")
    ref_only = chars(ref_text) - chars(target_text)
    if not asr_s: return 0.0
    leaked = sum(1 for c in asr_s if c in ref_only)
    return round(leaked / len(asr_s), 4)

def compute_cer(asr_text, target_text):
    a = asr_text.replace(" ", "")
    b = target_text.replace(" ", "").replace("，", "").replace("。", "")
    if not a or not b: return 1.0
    m, n = len(a), len(b)
    d = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): d[i][0] = i
    for j in range(n+1): d[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1,
                          d[i-1][j-1]+(0 if a[i-1]==b[j-1] else 1))
    return round(d[m][n]/max(1, len(b)), 4)

def compute_f0_rmse(syn_wav, ref_wav):
    import librosa
    def f0_extract(path):
        y, sr = librosa.load(str(path), sr=16000)
        f0, voiced, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sr,
                                     frame_length=2048, hop_length=320)
        return np.nan_to_num(f0, nan=0.0), voiced.astype(bool)
    f0_s, v_s = f0_extract(syn_wav)
    f0_r, v_r = f0_extract(ref_wav)
    L = min(len(f0_s), len(f0_r))
    both = v_s[:L] & v_r[:L]
    if both.sum() < 5: return None
    return round(float(np.sqrt(np.mean((f0_s[:L][both]-f0_r[:L][both])**2))), 1)

def secs_extractor(wavlm_model):
    import librosa
    def fn(path):
        y, _ = librosa.load(str(path), sr=16000)
        inp = torch.from_numpy(y.astype(np.float32)).unsqueeze(0).cuda()
        out = wavlm_model(inp, output_hidden_states=True)
        e = out.last_hidden_state.mean(dim=1)[:, :192]
        return torch.nn.functional.normalize(e, dim=-1)
    return fn

# ═══════════════════════════════════════════════════════════════════════════
# RECONSTRUCT RECORDS
# ═══════════════════════════════════════════════════════════════════════════

def build_records(refs):
    records = []
    for spk_dir in sorted(AUDIO.iterdir()):
        if not spk_dir.is_dir(): continue
        spk = spk_dir.name
        for emo_dir in sorted(spk_dir.iterdir()):
            if not emo_dir.is_dir(): continue
            emo = emo_dir.name
            key = (spk, emo)
            if key not in refs:
                continue
            ref_wav, ref_text = refs[key]
            if not Path(ref_wav).exists():
                continue
            for tid_dir in sorted(emo_dir.iterdir()):
                if not tid_dir.is_dir(): continue
                tid = tid_dir.name
                target_text = TARGETS.get(tid, "")
                if not target_text:
                    continue
                for seed in SEEDS:
                    wav_path = tid_dir / f"seed{seed}.wav"
                    if not wav_path.exists():
                        continue
                    records.append({
                        "speaker": spk, "emotion": emo, "text_id": tid,
                        "seed": seed,
                        "ref_wav": ref_wav, "ref_text": ref_text,
                        "target_text": target_text, "wav_path": str(wav_path),
                        "status": "ok",
                    })
    return records

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: ASR + CER + SLR
# ═══════════════════════════════════════════════════════════════════════════

def eval_asr(records):
    print(f"Loading FunASR ({len(records)} files to transcribe) ...")
    from funasr import AutoModel as FM
    asr = FM(model="paraformer-zh")
    for i, r in enumerate(records):
        if (i+1) % 50 == 0:
            print(f"  ASR {i+1}/{len(records)}")
        try:
            result = asr.generate(input=r["wav_path"])
            r["asr_text"] = result[0]["text"] if result else ""
        except Exception:
            r["asr_text"] = ""
        r["CER"] = compute_cer(r["asr_text"], r["target_text"])
        r["SLR"] = compute_slr(r["asr_text"], r["ref_text"], r["target_text"])
    del asr
    torch.cuda.empty_cache()
    print(f"  ASR done ({len(records)} files)")
    return records

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3: SECS
# ═══════════════════════════════════════════════════════════════════════════

def eval_secs(records):
    from transformers import WavLMConfig, WavLMModel
    from safetensors.torch import load_file
    print("Loading WavLM ...")
    cfg = WavLMConfig.from_pretrained(WAVLM_SNAP, local_files_only=True)
    wavlm = WavLMModel(cfg)
    wavlm.load_state_dict(load_file(f"{WAVLM_SNAP}/model.safetensors"), strict=False)
    wavlm = wavlm.cuda().eval()
    extract = secs_extractor(wavlm)

    ref_cache = {}
    for i, r in enumerate(records):
        if (i+1) % 50 == 0:
            print(f"  SECS {i+1}/{len(records)}")
        try:
            rw = r["ref_wav"]
            if rw not in ref_cache:
                ref_cache[rw] = extract(rw)
            e_syn = extract(r["wav_path"])
            r["SECS"] = round(max(0.0, (e_syn * ref_cache[rw]).sum(dim=-1).item()), 4)
        except Exception:
            r["SECS"] = None

    del wavlm
    torch.cuda.empty_cache()
    print(f"  SECS done ({len(ref_cache)} ref embeddings cached)")

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 4: F0 RMSE
# ═══════════════════════════════════════════════════════════════════════════

def eval_f0(records):
    import librosa
    ref_f0_cache = {}
    for i, r in enumerate(records):
        if (i+1) % 50 == 0:
            print(f"  F0 {i+1}/{len(records)}")
        try:
            rw = r["ref_wav"]
            if rw not in ref_f0_cache:
                y, sr = librosa.load(str(rw), sr=16000)
                f0, v, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sr,
                                        frame_length=2048, hop_length=320)
                ref_f0_cache[rw] = (np.nan_to_num(f0, nan=0.0), v.astype(bool))
            r["F0_RMSE_Hz"] = compute_f0_rmse(r["wav_path"], r["ref_wav"])
        except Exception:
            r["F0_RMSE_Hz"] = None
    print("  F0 done")

# ═══════════════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════════════

def report(records):
    groups = defaultdict(list)
    for r in records:
        groups[(r["speaker"], r["emotion"], r["text_id"])].append(r)

    # By-emotion aggregate
    print(f"\n{'='*80}")
    print("BY EMOTION (aggregate across 8 speakers x 6 texts x 3 seeds)")
    print(f"{'='*80}")
    print(f"{'Emotion':<12s} {'SLR':>7s} {'±':>6s} {'CER':>7s} {'±':>6s} "
          f"{'SECS':>7s} {'F0_RMSE':>8s} {'N':>5s}")
    print("-" * 75)

    emo_agg = {}
    for emo in EMOTIONS:
        items = [r for (s, e, t), lst in groups.items()
                 for r in lst if e == emo]
        if len(items) < 5: continue
        slrs = [r["SLR"] for r in items]
        cers = [r["CER"] for r in items]
        secs = [r["SECS"] for r in items if r["SECS"] is not None]
        f0s  = [r["F0_RMSE_Hz"] for r in items if r["F0_RMSE_Hz"] is not None]
        durs = [r.get("duration_s", 0) for r in items]
        emo_agg[emo] = {
            "SLR_mean": round(np.mean(slrs), 3), "SLR_std": round(np.std(slrs, ddof=1), 3),
            "CER_mean": round(np.mean(cers), 3), "CER_std": round(np.std(cers, ddof=1), 3),
            "SECS_mean": round(np.mean(secs), 3) if secs else None,
            "F0_RMSE_mean": round(np.mean(f0s), 1) if f0s else None,
            "N": len(items),
        }
        a = emo_agg[emo]
        print(f"{emo:<12s} {a['SLR_mean']:7.3f} {a['SLR_std']:6.3f} "
              f"{a['CER_mean']:7.3f} {a['CER_std']:6.3f} "
              f"{a['SECS_mean']:7.3f} {a['F0_RMSE_mean']:8.1f} "
              f"{a['N']:5d}")

    # By speaker x emotion (prose only)
    print(f"\n{'='*80}")
    print("BY SPEAKER x EMOTION (prose text, mean of 3 seeds)")
    print(f"{'='*80}")
    header = f"{'Spk':<6s} {'Emo':<12s} {'SLR':>7s} {'CER':>7s} {'SECS':>7s} {'F0_RMSE':>8s}"
    print(header)
    print("-" * len(header))

    speakers_sorted = sorted(set(r["speaker"] for r in records))
    for spk in speakers_sorted:
        for emo in EMOTIONS:
            items = groups.get((spk, emo, "prose"), [])
            if len(items) < 2: continue
            slrs = [r["SLR"] for r in items]
            cers = [r["CER"] for r in items]
            secs = [r["SECS"] for r in items if r["SECS"] is not None]
            f0s  = [r["F0_RMSE_Hz"] for r in items if r["F0_RMSE_Hz"] is not None]
            print(f"{spk:<6s} {emo:<12s} {np.mean(slrs):7.3f} {np.mean(cers):7.3f} "
                  f"{np.mean(secs):7.3f} {np.mean(f0s):8.1f}")
        print()

    # By text type (all emotions aggregate)
    print(f"\n{'='*80}")
    print("BY TEXT TYPE (aggregate across 8 speakers x 5 emotions x 3 seeds)")
    print(f"{'='*80}")
    print(f"{'Text':<12s} {'SLR':>7s} {'CER':>7s} {'SECS':>7s} {'F0_RMSE':>8s} {'N':>5s}")
    print("-" * 70)
    for tid in TARGETS:
        items = [r for (s, e, t), lst in groups.items()
                 for r in lst if t == tid]
        if len(items) < 5: continue
        slrs = [r["SLR"] for r in items]
        cers = [r["CER"] for r in items]
        secs = [r["SECS"] for r in items if r["SECS"] is not None]
        f0s  = [r["F0_RMSE_Hz"] for r in items if r["F0_RMSE_Hz"] is not None]
        print(f"{tid:<12s} {np.mean(slrs):7.3f} {np.mean(cers):7.3f} "
              f"{np.mean(secs):7.3f} {np.mean(f0s):8.1f} {len(items):5d}")

    # Sad-specific: by speaker x text type
    print(f"\n{'='*80}")
    print("SAD: BY SPEAKER x TEXT TYPE (SLR, mean of 3 seeds)")
    print(f"{'='*80}")
    text_ids = list(TARGETS.keys())
    header = f"{'Spk':<6s} " + " ".join(f"{t:>8s}" for t in text_ids) + f" {'Avg':>8s}"
    print(header)
    print("-" * len(header))
    for spk in speakers_sorted:
        vals = []
        for tid in text_ids:
            items = groups.get((spk, "Sad", tid), [])
            if items:
                v = np.mean([r["SLR"] for r in items])
                vals.append(v)
            else:
                vals.append(0)
        avg = np.mean(vals) if vals else 0
        print(f"{spk:<6s} " + " ".join(f"{v:8.3f}" for v in vals) + f" {avg:8.3f}")

    # Save
    result_path = OUT / "results.json"
    json.dump({
        "config": {"speakers": speakers_sorted, "emotions": EMOTIONS,
                   "targets": TARGETS, "seeds": SEEDS},
        "emotion_aggregate": {k: v for k, v in emo_agg.items()},
        "runs": records,
    }, open(result_path, "w"), ensure_ascii=False, indent=2)
    print(f"\nResults saved: {result_path}")
    return emo_agg

# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.monotonic()

    # Reconstruct refs (deterministic, same as synthesis)
    dl = "/root/autodl-tmp/esd_cn/train.data.list"
    speakers = ["0001","0002","0003","0004","0005","0006","0007","0008"]
    refs = select_refs(dl, speakers, EMOTIONS)
    print(f"Refs: {len(refs)} (speaker, emotion) pairs\n")

    # Build records from directory
    records = build_records(refs)
    print(f"Found {len(records)} WAV files to evaluate\n")

    # Eval phases
    records = eval_asr(records)
    eval_secs(records)
    eval_f0(records)

    # Report
    report(records)

    print(f"\nTotal eval time: {(time.monotonic()-t0)/60:.0f} min")
    print("Done.")

if __name__ == "__main__":
    main()
