#!/usr/bin/env python3
"""System Prompt Ablation: Does the system prompt affect CV3 zero-shot quality?

CV3 official: "You are a helpful assistant.<|endofprompt|>" + ref_text

Hypotheses:
  1. Chinese prompt may improve Chinese target text synthesis
  2. Emotion-matching prompt may help high-arousal F0 transfer
  3. Empty/minimal prompt may work equally well (prompt is unnecessary)

Test: 4 emotions (Neutral, Sad, Surprise, Angry) × 2 speakers (best/worst F0)
      × 6 texts × 4 prompts × 3 seeds = 576 syntheses
"""

import sys, os, json, time, random, logging
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
OUT = Path("/root/autodl-tmp/prompt_ablation")
AUDIO = OUT / "audio"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456]

# 4 system prompts to test
PROMPTS = {
    "en_default": "You are a helpful assistant.",       # official default
    "zh_default": "你是一个乐于助人的助手。",              # Chinese equivalent
    "emotion_neutral": "Speak naturally and clearly.",    # emotion-agnostic
    "minimal": "",                                        # empty/minimal
}

TARGETS = {
    "prose": "春天来了，桃花开了，满山遍野都是粉红色的花朵，微风吹过，花瓣纷纷飘落。",
    "dialogue": "你今天去超市了吗？记得买牛奶和面包，冰箱里已经没什么吃的了。",
    "story": "老张每天早上六点起床，泡一杯浓茶，坐在阳台上看报纸，这个习惯已经坚持了三十年。",
    "news": "随着人工智能技术的飞速发展，语音合成系统已经能够以惊人的准确度模仿人类的声音特征。",
    "emotional": "那一刻我突然明白了，有些告别不是结束，而是另一种开始，泪水模糊了我的视线。",
    "poem": "八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。",
}

# Select speakers: best and worst F0 RMSE from exp_005
# Best: 0004 (F0 RMSE avg across emotions is lowest), 0006
# Worst: 0001, 0002 (high F0 RMSE)
SPEAKERS = ["0001", "0002", "0004", "0006"]  # cover best (0004) and worst (0002) F0 RMSE + diversity
EMOTIONS = ["Neutral", "Sad", "Surprise", "Angry"]


def select_refs(data_list_path):
    pool = defaultdict(list)
    with open(data_list_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3: continue
            key, wav, text = parts[0], parts[1], parts[2]
            fields = key.split("_")
            spk, emo = fields[1], fields[2]
            pool[(spk, emo)].append((wav, text))
    random.seed(42)
    refs = {}
    for spk in SPEAKERS:
        for emo in EMOTIONS:
            candidates = pool.get((spk, emo), [])
            if candidates:
                refs[(spk, emo)] = random.choice(candidates)
    return refs


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
    total = len(SPEAKERS) * len(EMOTIONS) * len(TARGETS) * len(PROMPTS) * len(SEEDS)
    idx = 0

    for spk in SPEAKERS:
        for emo in EMOTIONS:
            key = (spk, emo)
            if key not in refs: continue
            ref_wav, ref_text = refs[key]
            if not Path(ref_wav).exists(): continue

            for prompt_name, system_prompt in PROMPTS.items():
                if system_prompt:
                    prompt = system_prompt + "<|endofprompt|>" + ref_text
                else:
                    prompt = "<|endofprompt|>" + ref_text

                for tid, ttext in TARGETS.items():
                    sentences = frt.text_normalize(ttext, split=False, text_frontend=True)
                    try:
                        mi_base = frt.frontend_zero_shot(
                            str(sentences), prompt, ref_wav, model.sample_rate, "")
                    except Exception as e:
                        for seed in SEEDS:
                            idx += 1
                            records.append({"speaker": spk, "emotion": emo,
                                "prompt": prompt_name, "text_id": tid, "seed": seed,
                                "status": "error", "error": str(e)[:100]})
                        continue

                    for seed in SEEDS:
                        idx += 1
                        torch.manual_seed(seed)
                        torch.cuda.empty_cache()

                        tag = f"{spk}_{emo}_{prompt_name}_{tid}_seed{seed}"
                        out_wav = AUDIO / f"{tag}.wav"
                        out_wav.parent.mkdir(parents=True, exist_ok=True)

                        rec = {"speaker": spk, "emotion": emo,
                               "prompt": prompt_name, "text_id": tid, "seed": seed,
                               "ref_wav": ref_wav, "ref_text": ref_text,
                               "target_text": ttext, "wav_path": str(out_wav)}

                        try:
                            gen = cv3m.tts(**mi_base, stream=False)
                            chunks = [j["tts_speech"] for j in gen]
                            audio = torch.cat(chunks, dim=1)
                            torchaudio.save(str(out_wav), audio, model.sample_rate)
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
        r["CER"] = compute_cer(r["asr_text"], r["target_text"])
    del asr; torch.cuda.empty_cache()

    # SECS
    from transformers import WavLMConfig, WavLMModel
    from safetensors.torch import load_file
    WAVLM_SNAP = "/root/.cache/huggingface/hub/models--microsoft--wavlm-base-plus-sv/snapshots/1bfd64eca136543feb28c5ffaf05381c6af33121"
    print("Loading WavLM ...")
    cfg = WavLMConfig.from_pretrained(WAVLM_SNAP, local_files_only=True)
    wavlm = WavLMModel(cfg)
    wavlm.load_state_dict(load_file(f"{WAVLM_SNAP}/model.safetensors"), strict=False)
    wavlm = wavlm.cuda().eval()
    import librosa
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
            r["SECS"] = None
    del wavlm; torch.cuda.empty_cache()

    # F0 RMSE
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
            v_s = v_s.astype(bool)
            L = min(len(f0_s), len(f0_r))
            both = v_s[:L] & v_r[:L]
            if both.sum() >= 5:
                r["F0_RMSE_Hz"] = round(float(np.sqrt(
                    np.mean((f0_s[:L][both] - f0_r[:L][both])**2))), 1)
        except Exception:
            r["F0_RMSE_Hz"] = None

    # Report
    groups = defaultdict(list)
    for r in ok:
        groups[(r["emotion"], r["prompt"])].append(r)

    print(f"\n{'='*80}")
    print("PROMPT ABLATION: CER / SECS / F0 RMSE by Emotion × Prompt")
    print(f"{'='*80}")

    for emo in EMOTIONS:
        print(f"\n--- {emo} ---")
        print(f"{'Prompt':<20s} {'CER':>7s} {'SECS':>7s} {'F0_RMSE':>8s} {'N':>5s}")
        print("-" * 55)
        for prompt_name in PROMPTS:
            items = groups.get((emo, prompt_name), [])
            if len(items) < 3: continue
            cers = [r["CER"] for r in items]
            secs = [r["SECS"] for r in items if r["SECS"] is not None]
            f0s = [r["F0_RMSE_Hz"] for r in items if r["F0_RMSE_Hz"] is not None]
            print(f"{prompt_name:<20s} {np.mean(cers):7.3f} {np.mean(secs):7.3f} "
                  f"{np.mean(f0s):8.1f} {len(items):5d}")

    # Best prompt per emotion for F0 RMSE
    print(f"\n{'='*80}")
    print("BEST PROMPT PER EMOTION (by F0 RMSE)")
    print(f"{'='*80}")
    for emo in EMOTIONS:
        best = None
        best_f0 = 999
        for prompt_name in PROMPTS:
            items = groups.get((emo, prompt_name), [])
            f0s = [r["F0_RMSE_Hz"] for r in items if r.get("F0_RMSE_Hz") is not None]
            if f0s and np.mean(f0s) < best_f0:
                best_f0 = np.mean(f0s)
                best = (prompt_name, np.mean(f0s), np.mean([r["CER"] for r in items]),
                        np.mean([r["SECS"] for r in items if r["SECS"] is not None]))
        if best:
            print(f"  {emo:<12s} → {best[0]:<20s}  F0={best[1]:.0f} Hz  CER={best[2]:.3f}  SECS={best[3]:.3f}")

    result_path = OUT / "results.json"
    json.dump({"runs": ok}, open(result_path, "w"), ensure_ascii=False, indent=2)
    print(f"\nSaved: {result_path}")
    print(f"Total: {(time.monotonic()-t0)/60:.0f} min")


if __name__ == "__main__":
    main()
