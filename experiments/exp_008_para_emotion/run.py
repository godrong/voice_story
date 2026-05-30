#!/usr/bin/env python3
"""P1: Paralinguistic Token × Emotion Interaction.

Hypothesis: Emotion-congruent paralinguistic tokens enhance emotion transfer.
  Sad + [sigh] → lower F0 RMSE
  Happy + [laughter] → higher SER recognition rate
  Neutral + [breath] → more natural pauses
  Thinking + [mn] → more natural hesitation

Design: 4 tokens × 4 emotions × 4 speakers × 2 texts × 2 conditions × 3 seeds = 768
"""

import sys, os, json, time, random, re, logging
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torchaudio

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, "/root/CosyVoice")
sys.path.insert(0, "/root/CosyVoice/third_party/Matcha-TTS")
sys.stdout.reconfigure(line_buffering=True)
logging.basicConfig(level=logging.WARNING, format="%(message)s")

MD = "/root/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"
OUT = Path("/root/autodl-tmp/para_emotion")
AUDIO = OUT / "audio"
OUT.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = "You are a helpful assistant."
SPEAKERS = ["0001", "0002", "0004", "0006"]
EMOTIONS = ["Neutral", "Sad", "Surprise", "Angry"]
SEEDS = [42, 123, 456]

# Emotion-congruent token mapping
EMO_TOKEN_MAP = {
    "Sad":      {"token": "[sigh]",     "hypothesis": "sigh enhances Sad F0 transfer"},
    "Angry":    {"token": "[breath]",   "hypothesis": "breath adds dramatic pause"},
    "Surprise": {"token": "[quick_breath]", "hypothesis": "quick breath matches surprise rhythm"},
    "Neutral":  {"token": "[mn]",       "hypothesis": "mn makes neutral more conversational"},
}

# Control: cross-emotion token (hypothesis: should be less effective)
CROSS_TOKEN = "[laughter]"  # laughter on Sad/Angry/Neutral is mismatched

# Test sentence pairs: one natural context for the token, one generic
TEST_PAIRS = {
    "Sad": [
        ("[sigh]今天真是糟糕的一天。", "今天真是糟糕的一天。"),
        ("看着窗外的雨，[sigh]心里说不出的难过。", "看着窗外的雨，心里说不出的难过。"),
    ],
    "Angry": [
        ("我再也受不了了[breath]这简直是不可接受的！", "我再也受不了了这简直是不可接受的！"),
        ("你给我听好了[breath]这件事没有商量的余地。", "你给我听好了这件事没有商量的余地。"),
    ],
    "Surprise": [
        ("什么？[quick_breath]你刚才说什么？", "什么？你刚才说什么？"),
        ("天哪[quick_breath]这也太不可思议了！", "天哪这也太不可思议了！"),
    ],
    "Neutral": [
        ("[mn]让我想想，大概是在三年前吧。", "让我想想，大概是在三年前吧。"),
        ("今天天气不错，[mn]适合出去走走。", "今天天气不错，适合出去走走。"),
    ],
}


def select_refs(data_list_path):
    pool = defaultdict(list)
    with open(data_list_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3: continue
            key, wav, text = parts[0], parts[1], parts[2]
            fields = key.split("_")
            pool[(fields[1], fields[2])].append((wav, text))
    random.seed(42)
    refs = {}
    for spk in SPEAKERS:
        for emo in EMOTIONS:
            candidates = pool.get((spk, emo), [])
            if candidates:
                refs[(spk, emo)] = random.choice(candidates)
    return refs


def compute_cer(asr_text, target_clean):
    a = asr_text.replace(" ", "")
    b = target_clean.replace(" ", "").replace("，", "").replace("。", "")
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


def main():
    t0 = time.monotonic()
    from cosyvoice.cli.cosyvoice import AutoModel

    dl = "/root/autodl-tmp/esd_cn/train.data.list"
    refs = select_refs(dl)
    print(f"Selected {len(refs)} refs\n")

    print("Loading CV3 ...")
    model = AutoModel(model_dir=MD)
    frt = model.frontend
    cv3m = model.model
    print("  Ready\n")

    records = []
    total = len(SPEAKERS) * len(EMOTIONS) * 2 * 2 * len(SEEDS)
    idx = 0

    for spk in SPEAKERS:
        for emo in EMOTIONS:
            key = (spk, emo)
            if key not in refs: continue
            ref_wav, ref_text = refs[key]
            prompt = SYSTEM_PROMPT + "<|endofprompt|>" + ref_text

            pairs = TEST_PAIRS.get(emo, TEST_PAIRS["Neutral"])
            for text_with, text_without in pairs:
                # Ensure both texts work
                for target_text in [text_with, text_without]:
                    sentences = frt.text_normalize(target_text, split=False,
                                                  text_frontend=True)
                    try:
                        mi_base = frt.frontend_zero_shot(
                            str(sentences), prompt, ref_wav, model.sample_rate, "")
                    except Exception as e:
                        for seed in SEEDS:
                            idx += 1
                            records.append({"status": "error", "error": str(e)[:100]})
                        continue

                    for seed in SEEDS:
                        idx += 1
                        torch.manual_seed(seed)
                        torch.cuda.empty_cache()

                        has_token = "with_token" if "[" in target_text else "no_token"
                        tag = f"{spk}_{emo}_{has_token}_seed{seed}_{idx}"
                        out_wav = AUDIO / f"{tag}.wav"

                        rec = {
                            "speaker": spk, "emotion": emo,
                            "condition": has_token,
                            "seed": seed, "target_text": target_text,
                            "ref_wav": ref_wav, "ref_text": ref_text,
                            "wav_path": str(out_wav),
                        }

                        try:
                            gen = cv3m.tts(**mi_base, stream=False)
                            chunks = [j["tts_speech"] for j in gen]
                            audio = torch.cat(chunks, dim=1)
                            torchaudio.save(str(out_wav), audio, model.sample_rate)
                            rec["duration_s"] = round(audio.shape[1]/model.sample_rate, 1)
                            rec["status"] = "ok"
                        except Exception as e:
                            rec["status"] = "error"
                            rec["error"] = str(e)[:150]

                        records.append(rec)

                    if idx % 50 == 0:
                        ok_n = sum(1 for r in records if r["status"]=="ok")
                        print(f"  [{idx}/{total}] ok={ok_n}")

    ok_n = sum(1 for r in records if r["status"]=="ok")
    print(f"\nSynthesis: {ok_n}/{len(records)} ok\n")
    del model, frt, cv3m; torch.cuda.empty_cache()

    # ASR
    ok = [r for r in records if r["status"]=="ok"]
    print(f"Loading FunASR ({len(ok)} files) ...")
    from funasr import AutoModel as FM
    asr = FM(model="paraformer-zh")
    for i, r in enumerate(ok):
        if (i+1) % 100 == 0: print(f"  ASR {i+1}/{len(ok)}")
        result = asr.generate(input=r["wav_path"])
        r["asr_text"] = result[0]["text"] if result else ""
        clean = re.sub(r'\[.*?\]', '', r["target_text"])
        r["CER"] = compute_cer(r["asr_text"], clean)
    del asr; torch.cuda.empty_cache()

    # F0 RMSE
    import librosa
    print("Computing F0 RMSE ...")
    ref_f0_cache = {}
    for i, r in enumerate(ok):
        if (i+1) % 100 == 0: print(f"  F0 {i+1}/{len(ok)}")
        try:
            rw = r["ref_wav"]
            if rw not in ref_f0_cache:
                y, sr = librosa.load(str(rw), sr=16000)
                f0, v, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sr,
                                        frame_length=2048, hop_length=320)
                ref_f0_cache[rw] = (np.nan_to_num(f0, nan=0.0), v.astype(bool))
            f0_r, v_r = ref_f0_cache[rw]
            y_s, _ = librosa.load(r["wav_path"], sr=16000)
            f0_s, v_s, _ = librosa.pyin(y_s, fmin=50, fmax=400, sr=sr,
                                        frame_length=2048, hop_length=320)
            f0_s = np.nan_to_num(f0_s, nan=0.0)
            L = min(len(f0_s), len(f0_r))
            both = v_s[:L].astype(bool) & v_r[:L].astype(bool)
            if both.sum() >= 5:
                r["F0_RMSE_Hz"] = round(float(np.sqrt(
                    np.mean((f0_s[:L][both] - f0_r[:L][both])**2))), 1)
        except Exception:
            pass
    print("  F0 done")

    # SECS
    from transformers import WavLMConfig, WavLMModel
    from safetensors.torch import load_file
    WAVLM_SNAP = "/root/.cache/huggingface/hub/models--microsoft--wavlm-base-plus-sv/snapshots/1bfd64eca136543feb28c5ffaf05381c6af33121"
    print("Loading WavLM ...")
    cfg = WavLMConfig.from_pretrained(WAVLM_SNAP, local_files_only=True)
    wavlm = WavLMModel(cfg)
    wavlm.load_state_dict(load_file(f"{WAVLM_SNAP}/model.safetensors"), strict=False)
    wavlm = wavlm.cuda().eval()
    ref_cache = {}
    for i, r in enumerate(ok):
        if (i+1) % 100 == 0: print(f"  SECS {i+1}/{len(ok)}")
        try:
            rw = r["ref_wav"]
            if rw not in ref_cache:
                y, _ = librosa.load(str(rw), sr=16000)
                inp = torch.from_numpy(y.astype(np.float32)).unsqueeze(0).cuda()
                out = wavlm(inp, output_hidden_states=True)
                ref_cache[rw] = torch.nn.functional.normalize(
                    out.last_hidden_state.mean(dim=1)[:, :192], dim=-1)
            y_s, _ = librosa.load(r["wav_path"], sr=16000)
            inp_s = torch.from_numpy(y_s.astype(np.float32)).unsqueeze(0).cuda()
            out_s = wavlm(inp_s, output_hidden_states=True)
            e_s = torch.nn.functional.normalize(
                out_s.last_hidden_state.mean(dim=1)[:, :192], dim=-1)
            r["SECS"] = round(max(0.0, (e_s * ref_cache[rw]).sum(dim=-1).item()), 4)
        except Exception:
            pass
    del wavlm; torch.cuda.empty_cache()

    # Report: per-emotion with_token vs without_token
    groups = defaultdict(list)
    for r in ok:
        groups[(r["emotion"], r["condition"])].append(r)

    print(f"\n{'='*80}")
    print("P1: PARALINGUISTIC × EMOTION INTERACTION")
    print(f"{'='*80}")

    for emo in EMOTIONS:
        token_info = EMO_TOKEN_MAP.get(emo, {})
        print(f"\n--- {emo} (token: {token_info.get('token','N/A')}) ---")
        print(f"    Hypothesis: {token_info.get('hypothesis','')}")
        print(f"    {'Condition':<12s} {'CER':>7s} {'F0_RMSE':>8s} {'SECS':>7s} {'Dur(s)':>7s} {'N':>5s}")
        print(f"    {'-'*50}")
        for cond in ["with_token", "no_token"]:
            items = groups.get((emo, cond), [])
            if len(items) < 3: continue
            cers = [r.get("CER", 0) for r in items]
            f0s = [r.get("F0_RMSE_Hz") for r in items if r.get("F0_RMSE_Hz") is not None]
            secs = [r.get("SECS") for r in items if r.get("SECS") is not None]
            durs = [r.get("duration_s", 0) for r in items]
            print(f"    {cond:<12s} {np.mean(cers):7.3f} {np.mean(f0s):8.1f} "
                  f"{np.mean(secs):7.3f} {np.mean(durs):7.1f} {len(items):5d}")

        # Calculate Δ
        with_items = groups.get((emo, "with_token"), [])
        without_items = groups.get((emo, "no_token"), [])
        if with_items and without_items:
            f0_with = np.mean([r.get("F0_RMSE_Hz", 0) for r in with_items if r.get("F0_RMSE_Hz")])
            f0_without = np.mean([r.get("F0_RMSE_Hz", 0) for r in without_items if r.get("F0_RMSE_Hz")])
            delta_f0 = f0_with - f0_without
            if delta_f0 < -3:
                print(f"    ✓ F0 improved by {-delta_f0:.0f} Hz (token helps prosody!)")
            elif delta_f0 > 3:
                print(f"    ✗ F0 worsened by {delta_f0:.0f} Hz")
            else:
                print(f"    ~ F0 unchanged (Δ={delta_f0:.1f} Hz)")

    result_path = OUT / "results.json"
    json.dump({"runs": ok}, open(result_path, "w"), ensure_ascii=False, indent=2)
    print(f"\nSaved: {result_path}")
    print(f"Total: {(time.monotonic()-t0)/60:.0f} min")


if __name__ == "__main__":
    main()
