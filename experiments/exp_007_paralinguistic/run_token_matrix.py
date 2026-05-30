#!/usr/bin/env python3
"""P0: CV3 Paralinguistic Token Effectiveness Matrix.

12 tokens discovered in CV3 tokenizer (NOT documented in paper).
Systematically test which ones produce audible effects.
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
OUT = Path("/root/autodl-tmp/paralinguistic_matrix")
AUDIO = OUT / "audio"
OUT.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = "You are a helpful assistant."
SPEAKERS = ["0002", "0006"]
EMOTIONS = ["Neutral", "Sad"]
SEED = 42

# 12 tokens to test, each with context-appropriate test sentences
TOKEN_TESTS = {
    "[breath]": [
        "今天天气真好，[breath]我想出去走走。",
        "这个故事很长，[breath]让我慢慢讲给你听。",
    ],
    "[quick_breath]": [
        "然后他说[quick_breath]等一下我马上回来。",
        "我跑得很快[quick_breath]快喘不过气了。",
    ],
    "[laughter]": [
        "他讲了个笑话[laughter]大家都笑翻了。",
        "想起来那件事[laughter]真是太搞笑了。",
    ],
    "[sigh]": [
        "[sigh]今天真是累坏了。",
        "看着窗外的雨，[sigh]心里有些难过。",
    ],
    "[cough]": [
        "他[cough]清了清嗓子，开始演讲。",
        "不好意思[cough]我今天有点感冒。",
    ],
    "[mn]": [
        "[mn]让我想想这个问题怎么回答。",
        "[mn]大概是在三年前吧，我记得。",
    ],
    "[lipsmack]": [
        "[lipsmack]嗯，这个菜味道不错。",
        "他咂了咂嘴[lipsmack]开始说话。",
    ],
    "[clucking]": [
        "她[clucking]摇了摇头表示不满。",
        "听到这个消息[clucking]他有些失望。",
    ],
    "[noise]": [
        "外面很吵[noise]听不清楚你说什么。",
        "电话里有杂音[noise]信号不太好。",
    ],
    "[hissing]": [
        "收音机发出嘶嘶声[hissing]信号断了。",
        "老旧的音响[hissing]播放着模糊的音乐。",
    ],
    "[vocalized-noise]": [
        "他[vocalized-noise]含糊地应了一声。",
        "没听清他说什么[vocalized-noise]像是在自言自语。",
    ],
    "[accent]": [
        "他用浓重的口音说[accent]这姑娘长得可真俏。",
        "我老家那边[accent]管这个叫锅贴儿。",
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
    total = len(TOKEN_TESTS) * len(SPEAKERS) * len(EMOTIONS) * 2 * 2  # token × spk × emo × 2texts × (with+without)
    idx = 0

    for token, texts in TOKEN_TESTS.items():
        for spk in SPEAKERS:
            for emo in EMOTIONS:
                key = (spk, emo)
                if key not in refs: continue
                ref_wav, ref_text = refs[key]
                prompt = SYSTEM_PROMPT + "<|endofprompt|>" + ref_text

                for text_idx, text_with_token in enumerate(texts):
                    # Text WITHOUT token (control) — strip all [...] tokens
                    import re
                    text_clean = re.sub(r'\[.*?\]', '', text_with_token)
                    # Also strip <laughter> tags
                    text_clean = re.sub(r'</?laughter>', '', text_clean)

                    for condition, target_text in [("with_token", text_with_token),
                                                    ("no_token", text_clean)]:
                        idx += 1
                        torch.manual_seed(SEED)
                        torch.cuda.empty_cache()

                        tag = f"{token[1:-1]}_{spk}_{emo}_text{text_idx}_{condition}"
                        out_wav = AUDIO / f"{tag}.wav"
                        out_wav.parent.mkdir(parents=True, exist_ok=True)

                        rec = {
                            "token": token, "speaker": spk, "emotion": emo,
                            "text_idx": text_idx, "condition": condition,
                            "target_text": target_text, "text_clean": text_clean,
                            "ref_wav": ref_wav, "ref_text": ref_text,
                            "wav_path": str(out_wav),
                        }

                        try:
                            sentences = frt.text_normalize(target_text, split=False,
                                                          text_frontend=True)
                            mi_base = frt.frontend_zero_shot(
                                str(sentences), prompt, ref_wav, model.sample_rate, "")
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

                    if idx % 20 == 0:
                        ok_n = sum(1 for r in records if r["status"]=="ok")
                        print(f"  [{idx}/{total}] {tag}  ok={ok_n}")

    ok_n = sum(1 for r in records if r["status"]=="ok")
    print(f"\nSynthesis: {ok_n}/{len(records)} ok\n")
    del model, frt, cv3m; torch.cuda.empty_cache()

    # ASR + Analysis
    ok = [r for r in records if r["status"]=="ok"]
    print(f"Loading FunASR ({len(ok)} files) ...")
    from funasr import AutoModel as FM
    asr = FM(model="paraformer-zh")
    for i, r in enumerate(ok):
        if (i+1) % 30 == 0: print(f"  ASR {i+1}/{len(ok)}")
        result = asr.generate(input=r["wav_path"])
        r["asr_text"] = result[0]["text"] if result else ""
        r["CER"] = compute_cer(r["asr_text"], r["text_clean"])
        # Check if token word appears in ASR (tokens should NOT be spoken)
        token_word = r["token"].replace("[", "").replace("]", "")
        r["token_in_asr"] = token_word in r["asr_text"]
    del asr; torch.cuda.empty_cache()

    # Duration analysis (tokens may add natural pauses)
    print(f"\n{'='*80}")
    print("PARALINGUISTIC TOKEN EFFECTIVENESS MATRIX")
    print(f"{'='*80}")
    print(f"{'Token':<22s} {'ΔDur(s)':>8s} {'ΔCER':>8s} {'TokenInASR':>10s} {'Verdict':>12s}")
    print("-" * 65)

    token_results = {}
    for token in TOKEN_TESTS:
        with_items = [r for r in ok if r["token"]==token and r["condition"]=="with_token"]
        without_items = [r for r in ok if r["token"]==token and r["condition"]=="no_token"]
        if not with_items or not without_items: continue

        avg_dur_with = np.mean([r["duration_s"] for r in with_items])
        avg_dur_without = np.mean([r["duration_s"] for r in without_items])
        delta_dur = avg_dur_with - avg_dur_without

        avg_cer_with = np.mean([r["CER"] for r in with_items])
        avg_cer_without = np.mean([r["CER"] for r in without_items])
        delta_cer = avg_cer_with - avg_cer_without

        token_asr_rate = np.mean([r["token_in_asr"] for r in with_items])

        # Verdict logic
        if delta_dur > 0.1:
            verdict = "ADDS PAUSE"
        elif abs(delta_dur) < 0.05 and abs(delta_cer) < 0.02:
            verdict = "NO EFFECT"
        elif delta_cer > 0.1:
            verdict = "HARMS CER"
        else:
            verdict = "CHECK AUDIO"

        token_results[token] = {
            "delta_dur_s": round(delta_dur, 2),
            "delta_cer": round(delta_cer, 3),
            "token_in_asr_rate": round(token_asr_rate, 2),
            "verdict": verdict,
        }

        print(f"{token:<22s} {delta_dur:8.2f} {delta_cer:8.3f} "
              f"{token_asr_rate:10.0%} {verdict:>12s}")

    # Save
    result_path = OUT / "results.json"
    json.dump({
        "token_results": {k: v for k, v in token_results.items()},
        "runs": ok,
    }, open(result_path, "w"), ensure_ascii=False, indent=2)
    print(f"\nSaved: {result_path}")
    print(f"Total: {(time.monotonic()-t0)/60:.0f} min")


if __name__ == "__main__":
    main()
