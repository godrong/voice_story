#!/usr/bin/env python3
"""Clean multi-emotion zero-shot evaluation with CORRECT prompt format.

Prompt format: "You are a helpful assistant.<|endofprompt|>" + ref_text
8 speakers × 5 emotions × 6 texts × 3 seeds = 720 syntheses
Metrics: CER (FunASR), SECS (WavLM), F0 RMSE (librosa.pyin)
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
OUT = Path("/root/autodl-tmp/clean_multiemotion")
AUDIO = OUT / "audio"
OUT.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = "You are a helpful assistant."
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


def select_refs(data_list_path, speakers, emotions):
    pool = defaultdict(list)
    with open(data_list_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3: continue
            key, wav, text = parts[0], parts[1], parts[2]
            fields = key.split("_")
            spk, emo = fields[1], fields[2]
            pool[(spk, emo)].append((wav, text))
    refs = {}
    random.seed(42)
    for spk in speakers:
        for emo in emotions:
            candidates = pool.get((spk, emo), [])
            if candidates:
                wav, text = random.choice(candidates)
                refs[(spk, emo)] = (wav, text)
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


def synthesize_all(refs, speakers):
    from cosyvoice.cli.cosyvoice import AutoModel
    print("Loading CV3 ...")
    model = AutoModel(model_dir=MD)
    frt = model.frontend
    cv3m = model.model
    print(f"  Ready\n")

    records = []
    total = len(speakers) * len(EMOTIONS) * len(TARGETS) * len(SEEDS)
    idx = 0

    for spk in speakers:
        for emo in EMOTIONS:
            key = (spk, emo)
            if key not in refs: continue
            ref_wav, ref_text = refs[key]
            if not Path(ref_wav).exists(): continue

            # CORRECT format
            prompt = SYSTEM_PROMPT + "<|endofprompt|>" + ref_text

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
                        "seed": seed,
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

                if idx % 30 == 0:
                    ok_n = sum(1 for r in records if r["status"]=="ok")
                    print(f"  [{idx}/{total}] {spk}/{emo}/{tid}  ok={ok_n}")

    ok_n = sum(1 for r in records if r["status"]=="ok")
    print(f"\nSynthesis done: {ok_n}/{len(records)} ok\n")
    del model, frt, cv3m
    torch.cuda.empty_cache()
    return records


def eval_asr(records):
    ok = [r for r in records if r["status"]=="ok"]
    print(f"Loading FunASR ({len(ok)} files) ...")
    from funasr import AutoModel as FM
    asr = FM(model="paraformer-zh")
    for i, r in enumerate(ok):
        if (i+1) % 60 == 0: print(f"  ASR {i+1}/{len(ok)}")
        result = asr.generate(input=r["wav_path"])
        r["asr_text"] = result[0]["text"] if result else ""
        r["CER"] = compute_cer(r["asr_text"], r["target_text"])
    del asr; torch.cuda.empty_cache()
    return ok


def eval_secs(ok_records):
    from transformers import WavLMConfig, WavLMModel
    from safetensors.torch import load_file
    print("Loading WavLM ...")
    cfg = WavLMConfig.from_pretrained(WAVLM_SNAP, local_files_only=True)
    wavlm = WavLMModel(cfg)
    wavlm.load_state_dict(load_file(f"{WAVLM_SNAP}/model.safetensors"), strict=False)
    wavlm = wavlm.cuda().eval()
    extract = secs_extractor(wavlm)

    ref_cache = {}
    for i, r in enumerate(ok_records):
        if (i+1) % 60 == 0: print(f"  SECS {i+1}/{len(ok_records)}")
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


def eval_f0(ok_records):
    import librosa
    ref_f0_cache = {}
    for i, r in enumerate(ok_records):
        if (i+1) % 60 == 0: print(f"  F0 {i+1}/{len(ok_records)}")
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


def report(ok_records):
    groups = defaultdict(list)
    for r in ok_records:
        groups[(r["speaker"], r["emotion"], r["text_id"])].append(r)

    # By-emotion aggregate
    print(f"\n{'='*80}")
    print("BY EMOTION (8 speakers × 6 texts × 3 seeds)")
    print(f"{'='*80}")
    print(f"{'Emotion':<12s} {'CER':>7s} {'±':>6s} {'SECS':>7s} {'F0_RMSE':>8s} {'N':>5s}")
    print("-" * 65)

    emo_agg = {}
    for emo in EMOTIONS:
        items = [r for (s, e, t), lst in groups.items()
                 for r in lst if e == emo]
        if len(items) < 5: continue
        cers = [r["CER"] for r in items]
        secs = [r["SECS"] for r in items if r["SECS"] is not None]
        f0s  = [r["F0_RMSE_Hz"] for r in items if r["F0_RMSE_Hz"] is not None]
        emo_agg[emo] = {
            "CER_mean": round(np.mean(cers), 3), "CER_std": round(np.std(cers, ddof=1), 3),
            "SECS_mean": round(np.mean(secs), 3) if secs else None,
            "F0_RMSE_mean": round(np.mean(f0s), 1) if f0s else None,
            "N": len(items),
        }
        a = emo_agg[emo]
        print(f"{emo:<12s} {a['CER_mean']:7.3f} {a['CER_std']:6.3f} "
              f"{a['SECS_mean']:7.3f} {a['F0_RMSE_mean']:8.1f} {a['N']:5d}")

    # By speaker × emotion
    print(f"\n{'='*80}")
    print("BY SPEAKER × EMOTION (prose text, mean of 3 seeds)")
    print(f"{'='*80}")
    print(f"{'Spk':<6s} {'Emo':<12s} {'CER':>7s} {'SECS':>7s} {'F0_RMSE':>8s}")
    print("-" * 50)

    speakers_sorted = sorted(set(r["speaker"] for r in ok_records))
    for spk in speakers_sorted:
        for emo in EMOTIONS:
            items = groups.get((spk, emo, "prose"), [])
            if len(items) < 2: continue
            cers = [r["CER"] for r in items]
            secs = [r["SECS"] for r in items if r["SECS"] is not None]
            f0s  = [r["F0_RMSE_Hz"] for r in items if r["F0_RMSE_Hz"] is not None]
            print(f"{spk:<6s} {emo:<12s} {np.mean(cers):7.3f} {np.mean(secs):7.3f} {np.mean(f0s):8.1f}")
        print()

    # By text type
    print(f"\n{'='*80}")
    print("BY TEXT TYPE (8 speakers × 5 emotions × 3 seeds)")
    print(f"{'='*80}")
    print(f"{'Text':<12s} {'CER':>7s} {'SECS':>7s} {'F0_RMSE':>8s} {'N':>5s}")
    print("-" * 50)
    for tid in TARGETS:
        items = [r for (s, e, t), lst in groups.items()
                 for r in lst if t == tid]
        if len(items) < 5: continue
        cers = [r["CER"] for r in items]
        secs = [r["SECS"] for r in items if r["SECS"] is not None]
        f0s  = [r["F0_RMSE_Hz"] for r in items if r["F0_RMSE_Hz"] is not None]
        print(f"{tid:<12s} {np.mean(cers):7.3f} {np.mean(secs):7.3f} {np.mean(f0s):8.1f} {len(items):5d}")

    # Content-prosody trade-off analysis
    print(f"\n{'='*80}")
    print("CONTENT vs PROSODY TRADE-OFF (by emotion)")
    print(f"{'='*80}")
    for emo in EMOTIONS:
        a = emo_agg[emo]
        print(f"  {emo:<12s} CER={a['CER_mean']:.3f}  F0_RMSE={a.get('F0_RMSE_mean', 0):.0f} Hz  "
              f"SECS={a.get('SECS_mean', 0):.3f}")

    result_path = OUT / "results.json"
    json.dump({
        "config": {"speakers": speakers_sorted, "emotions": EMOTIONS,
                   "targets": TARGETS, "seeds": SEEDS,
                   "system_prompt": SYSTEM_PROMPT},
        "emotion_aggregate": {k: v for k, v in emo_agg.items()},
        "runs": ok_records,
    }, open(result_path, "w"), ensure_ascii=False, indent=2)
    print(f"\nSaved: {result_path}")
    return emo_agg


def main():
    t0 = time.monotonic()
    dl = "/root/autodl-tmp/esd_cn/train.data.list"
    speakers = ["0001","0002","0003","0004","0005","0006","0007","0008"]
    refs = select_refs(dl, speakers, EMOTIONS)
    print(f"Selected {len(refs)} refs ({len(speakers)} spk × {len(EMOTIONS)} emo)\n")

    records = synthesize_all(refs, speakers)
    ok = eval_asr(records)
    eval_secs(ok)
    eval_f0(ok)
    report(ok)

    print(f"\nTotal: {(time.monotonic()-t0)/60:.0f} min")
    print(f"Audio: {AUDIO}")


if __name__ == "__main__":
    main()
