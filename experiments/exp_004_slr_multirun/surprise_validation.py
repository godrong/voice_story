#!/usr/bin/env python3
"""Validation: Is semantic leakage caused by emotion label or by acoustic features?

Hypothesis: Slow speech rate + long duration → leakage, regardless of emotion label.
Test:    0001/Surprise (2.2 ch/s, 4.5s) — slowest non-Sad ref, same rate as leaky 0004/Sad
Control: 0005/Angry (5.3 ch/s, 2.4s) — fastest ref, should show zero leakage
         0002/Sad (2.3 ch/s, 6.5s) — known leakiest, positive control
"""

import sys, os, json, time, random
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

MD = "/root/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"
OUT = Path("/root/autodl-tmp/surprise_validation")
AUDIO = OUT / "audio"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456]

TARGETS = {
    "prose": "春天来了，桃花开了，满山遍野都是粉红色的花朵，微风吹过，花瓣纷纷飘落。",
    "dialogue": "你今天去超市了吗？记得买牛奶和面包，冰箱里已经没什么吃的了。",
    "story": "老张每天早上六点起床，泡一杯浓茶，坐在阳台上看报纸，这个习惯已经坚持了三十年。",
    "news": "随着人工智能技术的飞速发展，语音合成系统已经能够以惊人的准确度模仿人类的声音特征。",
    "emotional": "那一刻我突然明白了，有些告别不是结束，而是另一种开始，泪水模糊了我的视线。",
    "poem": "八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。",
}

# Three test conditions
TEST_CASES = [
    ("0001", "Surprise", "slowest non-Sad (2.2 ch/s, 4.5s) — HYPOTHESIS: should leak"),
    ("0005", "Angry", "fastest overall (5.3 ch/s, 2.4s) — should NOT leak"),
    ("0002", "Sad", "known leakiest (2.3 ch/s, 6.5s) — POSITIVE CONTROL"),
]


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
    for spk, emo in [("0001","Surprise"), ("0005","Angry"), ("0002","Sad")]:
        candidates = pool.get((spk, emo), [])
        if candidates:
            refs[(spk, emo)] = random.choice(candidates)
    return refs


def main():
    import librosa
    t0 = time.monotonic()
    from cosyvoice.cli.cosyvoice import AutoModel

    dl = "/root/autodl-tmp/esd_cn/train.data.list"
    refs = select_refs(dl)

    print("=" * 80)
    print("VALIDATION: Acoustic Features vs Emotion Labels")
    print("=" * 80)

    for (spk, emo), (wav_path, text) in refs.items():
        y, sr = librosa.load(wav_path, sr=16000)
        dur = len(y) / sr
        n_chars = len(text.replace(" ", "").replace("，", "").replace("。", ""))
        f0, voiced, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sr,
                                     frame_length=2048, hop_length=320)
        f0_clean = f0[voiced] if voiced.sum() > 0 else np.array([0])
        rate = n_chars / dur
        print(f"  {spk}/{emo}: \"{text}\"")
        print(f"    rate={rate:.1f} ch/s  dur={dur:.1f}s  chars={n_chars}  "
              f"F0={np.mean(f0_clean):.0f}±{np.std(f0_clean):.0f}Hz  "
              f"voicing={voiced.mean():.1%}")
    print()

    print("Loading CV3 ...")
    model = AutoModel(model_dir=MD)
    frt = model.frontend
    cv3m = model.model
    print("  Ready\n")

    records = []
    total = len(TEST_CASES) * len(TARGETS) * len(SEEDS)
    idx = 0

    for spk, emo, desc in TEST_CASES:
        if (spk, emo) not in refs: continue
        ref_wav, ref_text = refs[(spk, emo)]
        prompt = ref_text + "<|endofprompt|>"

        for tid, target_text in TARGETS.items():
            sentences = frt.text_normalize(target_text, split=False, text_frontend=True)
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
                        "target_text": target_text, "wav_path": "",
                    })
                continue

            for seed in SEEDS:
                idx += 1
                torch.manual_seed(seed)
                torch.cuda.empty_cache()

                tag = f"{spk}_{emo}_{tid}_seed{seed}"
                out_wav = AUDIO / f"{tag}.wav"
                out_wav.parent.mkdir(parents=True, exist_ok=True)

                rec = {
                    "speaker": spk, "emotion": emo, "text_id": tid,
                    "seed": seed, "ref_wav": ref_wav, "ref_text": ref_text,
                    "target_text": target_text, "wav_path": str(out_wav),
                    "description": desc,
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

                if idx % 10 == 0:
                    ok_n = sum(1 for r in records if r["status"]=="ok")
                    print(f"  [{idx}/{total}] {tag}  ok={ok_n}")

    ok_n = sum(1 for r in records if r["status"]=="ok")
    print(f"\nSynthesis done: {ok_n}/{len(records)} ok\n")

    del model, frt, cv3m
    torch.cuda.empty_cache()

    # ── ASR ──
    ok = [r for r in records if r["status"]=="ok"]
    print(f"Loading FunASR ({len(ok)} files) ...")
    from funasr import AutoModel as FM
    asr = FM(model="paraformer-zh")
    for i, r in enumerate(ok):
        if (i+1) % 10 == 0: print(f"  ASR {i+1}/{len(ok)}")
        result = asr.generate(input=r["wav_path"])
        r["asr_text"] = result[0]["text"] if result else ""
        r["CER"] = compute_cer(r["asr_text"], r["target_text"])
        r["SLR"] = compute_slr(r["asr_text"], r["ref_text"], r["target_text"])
    del asr; torch.cuda.empty_cache()

    # ── Report ──
    groups = defaultdict(list)
    for r in ok:
        groups[(r["speaker"], r["emotion"])].append(r)

    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"{'Condition':<30s} {'SLR':>7s} {'±':>6s} {'CER':>7s} {'±':>6s} {'N':>5s}")
    print("-" * 65)

    for spk, emo, desc in TEST_CASES:
        items = groups.get((spk, emo), [])
        if len(items) < 3: continue
        slrs = [r["SLR"] for r in items]
        cers = [r["CER"] for r in items]
        print(f"{spk}/{emo} ({desc[:30]:<30s}) "
              f"{np.mean(slrs):7.3f} {np.std(slrs, ddof=1):6.3f} "
              f"{np.mean(cers):7.3f} {np.std(cers, ddof=1):6.3f} "
              f"{len(items):5d}")

    # Per-text breakdown
    print(f"\n{'='*80}")
    print("PER-TEXT BREAKDOWN")
    print(f"{'='*80}")
    text_groups = defaultdict(list)
    for r in ok:
        text_groups[(r["speaker"], r["emotion"], r["text_id"])].append(r)

    for spk, emo, desc in TEST_CASES:
        print(f"\n{spk}/{emo} ({desc[:50]}):")
        for tid in TARGETS:
            items = text_groups.get((spk, emo, tid), [])
            if not items: continue
            slrs = [r["SLR"] for r in items]
            asr_samples = [r["asr_text"] for r in items if r["asr_text"]]
            print(f"  {tid:<12s} SLR={np.mean(slrs):.3f}±{np.std(slrs, ddof=1):.3f}  "
                  f"CER={np.mean([r['CER'] for r in items]):.3f}")
            if asr_samples and np.mean(slrs) > 0.05:
                print(f"           ASR: \"{asr_samples[0]}\"")

    # Save
    result_path = OUT / "results.json"
    json.dump({
        "test_cases": [{"speaker": s, "emotion": e, "description": d}
                       for s, e, d in TEST_CASES],
        "runs": ok,
    }, open(result_path, "w"), ensure_ascii=False, indent=2)
    print(f"\nSaved: {result_path}")
    print(f"Total: {(time.monotonic()-t0)/60:.0f} min")


if __name__ == "__main__":
    main()
