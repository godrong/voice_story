#!/usr/bin/env python3
"""Multi-Speaker Boundary Test — 8 speakers × 5 emotions × 6 texts × 3 seeds.

Extends the single-speaker experiment to test generalizability:
  - Is Sad leakage universal across speakers, or speaker-specific?
  - Does the Angry F0 vs Sad SLR trade-off hold for all speakers?
  - Does content type (dialogue vs narrative vs formal) affect leakage?
  - Are there speaker-specific failure modes?

6 diverse target texts: prose, dialogue, story, news, emotional, poem.
Ref selection: one random ref per (speaker, emotion) from ESD data list.
Metrics: CER, SECS (WavLM-SV), F0 RMSE (pyin), SLR (semantic leakage).
"""

import sys, os, json, time, logging, random
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
OUT = Path("/root/autodl-tmp/boundary_test_multispk")
AUDIO = OUT / "audio"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456]
EMOTIONS = ["Neutral", "Happy", "Angry", "Surprise", "Sad"]

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
# REF SELECTION
# ═══════════════════════════════════════════════════════════════════════════

def select_refs(data_list_path, speakers, emotions):
    """Pick one ref per (speaker, emotion) from the data list."""
    pool = defaultdict(list)
    with open(data_list_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            key = parts[0]  # e.g. esd_0008_Sad_0008_001372
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
                print(f"  WARN: no samples for {spk}/{emo}")
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
# PHASE 1: SYNTHESIS
# ═══════════════════════════════════════════════════════════════════════════

def synthesize_all(refs, speakers):
    print("Loading CV3 ...")
    model = AutoModel(model_dir=MD)
    frt = model.frontend
    cv3m = model.model
    print(f"  Ready, sr={model.sample_rate}\n")

    records = []
    total = len(speakers) * len(EMOTIONS) * len(TARGETS) * len(SEEDS)
    idx = 0

    for spk in speakers:
        for emo in EMOTIONS:
            key = (spk, emo)
            if key not in refs:
                continue
            ref_wav, ref_text = refs[key]
            if not Path(ref_wav).exists():
                continue

            prompt = ref_text + "<|endofprompt|>"

            for tid, ttext in TARGETS.items():
                sentences = frt.text_normalize(ttext, split=False, text_frontend=True)
                try:
                    mi_base = frt.frontend_zero_shot(
                        str(sentences), prompt, ref_wav, model.sample_rate, "")
                except Exception as e:
                    for seed in SEEDS:
                        idx += 1
                        records.append({
                            "speaker": spk, "emotion": emo, "text_id": tid,
                            "seed": seed, "status": "error",
                            "error": f"frontend: {str(e)[:100]}",
                            "ref_wav": ref_wav, "ref_text": ref_text,
                            "target_text": ttext, "wav_path": "",
                        })
                    continue

                for seed in SEEDS:
                    idx += 1
                    torch.manual_seed(seed)
                    torch.cuda.empty_cache()

                    out_wav = AUDIO / spk / emo / tid / f"seed{seed}.wav"
                    out_wav.parent.mkdir(parents=True, exist_ok=True)

                    rec = {
                        "speaker": spk, "emotion": emo, "text_id": tid,
                        "seed": seed, "condition": "normal",
                        "ref_wav": ref_wav, "ref_text": ref_text,
                        "target_text": ttext, "wav_path": str(out_wav),
                    }

                    try:
                        ts = time.monotonic()
                        gen = cv3m.tts(**mi_base, stream=False)
                        chunks = [j["tts_speech"] for j in gen]
                        audio = torch.cat(chunks, dim=1)
                        torchaudio.save(str(out_wav), audio, model.sample_rate)
                        rec["duration_s"] = round(audio.shape[1]/model.sample_rate, 1)
                        rec["elapsed_s"] = round(time.monotonic()-ts, 1)
                        rec["status"] = "ok"
                    except Exception as e:
                        rec["status"] = "error"
                        rec["error"] = str(e)[:150]

                    records.append(rec)

                if idx % 20 == 0:
                    ok_n = sum(1 for r in records if r["status"]=="ok")
                    print(f"  [{idx}/{total}] {spk}/{emo}/{tid}  ok={ok_n}")

    ok_n = sum(1 for r in records if r["status"]=="ok")
    print(f"\nSynthesis done: {ok_n}/{len(records)} ok\n")
    del model, frt, cv3m
    torch.cuda.empty_cache()
    return records

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: ASR + SLR + CER
# ═══════════════════════════════════════════════════════════════════════════

def eval_asr(records):
    ok = [r for r in records if r["status"]=="ok"]
    print(f"Loading FunASR ({len(ok)} files to transcribe) ...")
    from funasr import AutoModel as FM
    asr = FM(model="paraformer-zh")
    for i, r in enumerate(ok):
        if (i+1) % 30 == 0: print(f"  ASR {i+1}/{len(ok)}")
        result = asr.generate(input=r["wav_path"])
        r["asr_text"] = result[0]["text"] if result else ""
        r["CER"] = compute_cer(r["asr_text"], r["target_text"])
        r["SLR"] = compute_slr(r["asr_text"], r["ref_text"], r["target_text"])
    del asr; torch.cuda.empty_cache()
    return ok

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3: SECS
# ═══════════════════════════════════════════════════════════════════════════

def eval_secs(ok_records):
    from transformers import WavLMConfig, WavLMModel
    from safetensors.torch import load_file
    print("Loading WavLM ...")
    cfg = WavLMConfig.from_pretrained(WAVLM_SNAP, local_files_only=True)
    wavlm = WavLMModel(cfg)
    wavlm.load_state_dict(load_file(f"{WAVLM_SNAP}/model.safetensors"), strict=False)
    wavlm = wavlm.cuda().eval()
    extract = secs_extractor(wavlm)

    # Cache ref embeddings
    ref_cache = {}
    for i, r in enumerate(ok_records):
        if (i+1) % 30 == 0: print(f"  SECS {i+1}/{len(ok_records)}")
        try:
            rw = r["ref_wav"]
            if rw not in ref_cache:
                ref_cache[rw] = extract(rw)
            e_syn = extract(r["wav_path"])
            r["SECS"] = round(max(0.0, (e_syn * ref_cache[rw]).sum(dim=-1).item()), 4)
        except Exception:
            r["SECS"] = None

    del wavlm; torch.cuda.empty_cache()
    print(f"  SECS done ({len(ref_cache)} ref embeddings cached)")

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 4: F0 RMSE
# ═══════════════════════════════════════════════════════════════════════════

def eval_f0(ok_records):
    # Cache ref F0 contours
    ref_f0_cache = {}
    for i, r in enumerate(ok_records):
        if (i+1) % 30 == 0: print(f"  F0 {i+1}/{len(ok_records)}")
        try:
            import librosa
            rw = r["ref_wav"]
            if rw not in ref_f0_cache:
                y, sr = librosa.load(str(rw), sr=16000)
                f0, v, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sr,
                                        frame_length=2048, hop_length=320)
                ref_f0_cache[rw] = (np.nan_to_num(f0, nan=0.0), v.astype(bool))
            r["F0_RMSE_Hz"] = compute_f0_rmse(r["wav_path"], r["ref_wav"])
        except Exception:
            r["F0_RMSE_Hz"] = None
    print(f"  F0 done")

# ═══════════════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════════════

def report(ok_records):
    groups = defaultdict(list)
    for r in ok_records:
        groups[(r["speaker"], r["emotion"], r["text_id"])].append(r)

    # By-emotion aggregate (across all speakers and texts)
    print(f"\n{'='*80}")
    print("BY EMOTION (aggregate across 8 speakers × 2 texts × 3 seeds)")
    print(f"{'='*80}")
    print(f"{'Emotion':<12s} {'SLR':>7s} {'±':>6s} {'CER':>7s} {'±':>6s} "
          f"{'SECS':>7s} {'F0_RMSE':>8s} {'Dur':>6s} {'N':>5s}")
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
        durs = [r["duration_s"] for r in items]
        emo_agg[emo] = {
            "SLR_mean": round(np.mean(slrs), 3), "SLR_std": round(np.std(slrs, ddof=1), 3),
            "CER_mean": round(np.mean(cers), 3), "CER_std": round(np.std(cers, ddof=1), 3),
            "SECS_mean": round(np.mean(secs), 3) if secs else None,
            "F0_RMSE_mean": round(np.mean(f0s), 1) if f0s else None,
            "Dur_mean": round(np.mean(durs), 1), "N": len(items),
        }
        a = emo_agg[emo]
        print(f"{emo:<12s} {a['SLR_mean']:7.3f} {a['SLR_std']:6.3f} "
              f"{a['CER_mean']:7.3f} {a['CER_std']:6.3f} "
              f"{a['SECS_mean']:7.3f} {a['F0_RMSE_mean']:8.1f} "
              f"{a['Dur_mean']:6.1f} {a['N']:5d}")

    # By speaker × emotion (prose only for clean comparison)
    print(f"\n{'='*80}")
    print("BY SPEAKER × EMOTION (prose text, mean of 3 seeds)")
    print(f"{'='*80}")
    header = f"{'Spk':<6s} {'Emo':<12s} {'SLR':>7s} {'CER':>7s} {'SECS':>7s} {'F0_RMSE':>8s} {'Dur':>6s}"
    print(header)
    print("-" * len(header))

    speakers_sorted = sorted(set(r["speaker"] for r in ok_records))
    for spk in speakers_sorted:
        for emo in EMOTIONS:
            items = groups.get((spk, emo, "prose"), [])
            if len(items) < 2: continue
            slrs = [r["SLR"] for r in items]
            cers = [r["CER"] for r in items]
            secs = [r["SECS"] for r in items if r["SECS"] is not None]
            f0s  = [r["F0_RMSE_Hz"] for r in items if r["F0_RMSE_Hz"] is not None]
            durs = [r["duration_s"] for r in items]
            print(f"{spk:<6s} {emo:<12s} {np.mean(slrs):7.3f} {np.mean(cers):7.3f} "
                  f"{np.mean(secs):7.3f} {np.mean(f0s):8.1f} {np.mean(durs):6.1f}")
        print()

    # Save
    result_path = OUT / "results.json"
    json.dump({
        "config": {"speakers": speakers_sorted, "emotions": EMOTIONS,
                   "targets": TARGETS, "seeds": SEEDS},
        "emotion_aggregate": {k: v for k, v in emo_agg.items()},
        "runs": ok_records,
    }, open(result_path, "w"), ensure_ascii=False, indent=2)
    print(f"Results: {result_path}")

    return emo_agg


# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.monotonic()

    # Select refs
    dl = "/root/autodl-tmp/esd_cn/train.data.list"
    speakers = ["0001","0002","0003","0004","0005","0006","0007","0008"]
    refs = select_refs(dl, speakers, EMOTIONS)
    print(f"Selected refs for {len(refs)} (speaker, emotion) pairs "
          f"({len(speakers)} spk × {len(EMOTIONS)} emo)\n")

    # Phase 1: Synthesize
    records = synthesize_all(refs, speakers)

    # Phase 2-4: Evaluate
    ok = eval_asr(records)
    eval_secs(ok)
    eval_f0(ok)

    # Report
    report(ok)

    print(f"\nTotal: {(time.monotonic()-t0)/60:.0f} min")
    print(f"Audio: {AUDIO}")
    print("Done.")

if __name__ == "__main__":
    main()
