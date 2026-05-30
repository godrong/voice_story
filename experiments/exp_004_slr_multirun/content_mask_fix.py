#!/usr/bin/env python3
"""Approach B: Content-Masked Speech Tokens — fix semantic leakage at inference time.

Root cause: longer/slower ref → more speech tokens per character → LLM has more
surface area to extract content → semantic leakage.

Fix: proportionally mask speech tokens that correspond to reference text content,
reducing token density in content-bearing regions while preserving prosody tokens
in silence/pause regions.

Test: top-3 leakiest (speaker, emotion, text) cases, 3 seeds each, with and without mask.
"""

import sys, os, json, time, random
from pathlib import Path
import numpy as np
import torch
import torchaudio

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, "/root/CosyVoice")
sys.path.insert(0, "/root/CosyVoice/third_party/Matcha-TTS")

sys.stdout.reconfigure(line_buffering=True)

MD = "/root/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"
OUT = Path("/root/autodl-tmp/content_mask_fix")
AUDIO = OUT / "audio"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456]

# Top leakiest (speaker, emotion, text) combos from multi-speaker results
# Selected based on Sad SLR by speaker × text type
TEST_CASES = [
    {"speaker": "0002", "emotion": "Sad", "text_id": "poem",    "target": "八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。"},
    {"speaker": "0002", "emotion": "Sad", "text_id": "dialogue","target": "你今天去超市了吗？记得买牛奶和面包，冰箱里已经没什么吃的了。"},
    {"speaker": "0004", "emotion": "Sad", "text_id": "dialogue","target": "你今天去超市了吗？记得买牛奶和面包，冰箱里已经没什么吃的了。"},
    {"speaker": "0002", "emotion": "Sad", "text_id": "prose",   "target": "春天来了，桃花开了，满山遍野都是粉红色的花朵，微风吹过，花瓣纷纷飘落。"},
    # Control: non-leaky Sad
    {"speaker": "0006", "emotion": "Sad", "text_id": "poem",    "target": "八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。"},
    # Control: Angry (should have zero leakage regardless)
    {"speaker": "0002", "emotion": "Angry","text_id": "poem",    "target": "八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。"},
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


def select_refs(data_list_path, speakers, emotions):
    """Deterministic ref selection (same seed=42 as before)."""
    from collections import defaultdict
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


def mask_speech_tokens_proportional(speech_token, ref_text, mask_ratio=1.0):
    """Mask speech tokens proportionally based on ref_text character positions.

    speech_token: tensor (1, T) — VQ-VAE discrete tokens at 25Hz
    ref_text: str — the reference audio transcript
    mask_ratio: float — what fraction of content tokens to mask (1.0 = all)

    Returns masked token tensor (same shape).
    """
    T = speech_token.shape[1]
    clean_text = ref_text.replace("<|endofprompt|>", "")
    n_chars = len(clean_text.replace(" ", "").replace("，", "").replace("。", ""))

    if n_chars == 0 or T == 0:
        return speech_token

    tokens_per_char = T / n_chars

    # Get unique token values to find a good mask value
    unique_tokens = speech_token.unique()
    # Use 0 as mask (common padding token in VQ-VAE)
    mask_value = 0

    masked = speech_token.clone()

    for i in range(n_chars):
        start = int(i * tokens_per_char)
        end = int((i + mask_ratio) * tokens_per_char)
        if start < T:
            masked[0, start:min(end, T)] = mask_value

    n_masked = (masked == mask_value).sum().item()
    return masked, n_masked, T


def mask_speech_tokens_uniform_thin(speech_token, keep_every=2):
    """Uniformly thin speech tokens: keep every Nth token, mask others.

    This is the simplest density reduction — doesn't need alignment.
    """
    T = speech_token.shape[1]
    masked = speech_token.clone()
    for i in range(T):
        if i % keep_every != 0:
            masked[0, i] = 0
    n_masked = T - (T // keep_every)
    return masked, n_masked, T


def main():
    t0 = time.monotonic()
    print("=" * 80)
    print("APPROACH B: Content-Masked Speech Tokens — SLR Fix Validation")
    print("=" * 80)

    # Load refs
    from cosyvoice.cli.cosyvoice import AutoModel
    dl = "/root/autodl-tmp/esd_cn/train.data.list"
    all_speakers = ["0001","0002","0003","0004","0005","0006","0007","0008"]
    all_emotions = ["Neutral", "Happy", "Angry", "Surprise", "Sad"]
    refs = select_refs(dl, all_speakers, all_emotions)

    print("Loading CV3 ...")
    model = AutoModel(model_dir=MD)
    frt = model.frontend
    cv3m = model.model
    print(f"  Ready\n")

    # Identify unique test configurations
    test_configs = []
    for tc in TEST_CASES:
        spk, emo, tid = tc["speaker"], tc["emotion"], tc["text_id"]
        key = (spk, emo)
        if key not in refs: continue
        ref_wav, ref_text = refs[key]
        if not Path(ref_wav).exists(): continue
        test_configs.append({
            **tc,
            "ref_wav": ref_wav,
            "ref_text": ref_text,
        })

    # Conditions to test
    conditions = [
        ("normal", None),                    # baseline (no mask)
        ("prop_mask", "proportional"),       # proportional content mask
        ("thin_mask", "uniform_thin"),       # uniform thinning
    ]

    records = []
    total = len(test_configs) * len(SEEDS) * len(conditions)
    idx = 0

    for cfg in test_configs:
        spk, emo, tid = cfg["speaker"], cfg["emotion"], cfg["text_id"]
        ref_wav, ref_text, target_text = cfg["ref_wav"], cfg["ref_text"], cfg["target"]

        prompt = ref_text + "<|endofprompt|>"
        sentences = frt.text_normalize(target_text, split=False, text_frontend=True)

        try:
            mi_base = frt.frontend_zero_shot(
                str(sentences), prompt, ref_wav, model.sample_rate, "")
        except Exception as e:
            print(f"  ERROR frontend {spk}/{emo}/{tid}: {e}")
            continue

        original_speech_token = mi_base["llm_prompt_speech_token"].clone()
        original_token_len = mi_base.get("llm_prompt_speech_token_len",
                                         torch.tensor([original_speech_token.shape[1]]))

        for cond_name, mask_type in conditions:
            for seed in SEEDS:
                idx += 1
                torch.manual_seed(seed)
                torch.cuda.empty_cache()

                tag = f"{spk}_{emo}_{tid}_{cond_name}_seed{seed}"
                out_wav = AUDIO / f"{tag}.wav"
                out_wav.parent.mkdir(parents=True, exist_ok=True)

                rec = {
                    "speaker": spk, "emotion": emo, "text_id": tid,
                    "seed": seed, "condition": cond_name, "mask_type": mask_type,
                    "ref_wav": ref_wav, "ref_text": ref_text,
                    "target_text": target_text, "wav_path": str(out_wav),
                }

                # Apply mask
                if mask_type == "proportional":
                    masked_token, n_masked, T_total = mask_speech_tokens_proportional(
                        original_speech_token.clone(), ref_text, mask_ratio=1.0)
                    mi_base["llm_prompt_speech_token"] = masked_token
                    rec["n_tokens_total"] = T_total
                    rec["n_tokens_masked"] = n_masked
                    rec["mask_pct"] = round(100 * n_masked / max(1, T_total), 1)
                elif mask_type == "uniform_thin":
                    masked_token, n_masked, T_total = mask_speech_tokens_uniform_thin(
                        original_speech_token.clone(), keep_every=2)
                    mi_base["llm_prompt_speech_token"] = masked_token
                    rec["n_tokens_total"] = T_total
                    rec["n_tokens_masked"] = n_masked
                    rec["mask_pct"] = round(100 * n_masked / max(1, T_total), 1)
                else:
                    mi_base["llm_prompt_speech_token"] = original_speech_token.clone()
                    rec["n_tokens_total"] = original_speech_token.shape[1]
                    rec["n_tokens_masked"] = 0
                    rec["mask_pct"] = 0.0

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

    print(f"\nSynthesis done: {sum(1 for r in records if r['status']=='ok')}/{len(records)} ok")

    # Clean up model to free GPU
    del model, frt, cv3m
    torch.cuda.empty_cache()

    # ── ASR Evaluation ──
    ok = [r for r in records if r["status"]=="ok"]
    print(f"\nLoading FunASR for {len(ok)} files ...")
    from funasr import AutoModel as FM
    asr = FM(model="paraformer-zh")
    for i, r in enumerate(ok):
        if (i+1) % 10 == 0: print(f"  ASR {i+1}/{len(ok)}")
        result = asr.generate(input=r["wav_path"])
        r["asr_text"] = result[0]["text"] if result else ""
        r["CER"] = compute_cer(r["asr_text"], r["target_text"])
        r["SLR"] = compute_slr(r["asr_text"], r["ref_text"], r["target_text"])
    del asr; torch.cuda.empty_cache()
    print("  ASR done")

    # ── Report ──
    print(f"\n{'='*80}")
    print("RESULTS: SLR BEFORE vs AFTER CONTENT MASKING")
    print(f"{'='*80}")

    # Group by (speaker, emotion, text_id, condition)
    from collections import defaultdict
    groups = defaultdict(list)
    for r in ok:
        groups[(r["speaker"], r["emotion"], r["text_id"], r["condition"])].append(r)

    print(f"\n{'Speaker':<8s} {'Emo':<8s} {'Text':<10s} {'Condition':<14s} "
          f"{'SLR':>7s} {'±':>6s} {'CER':>7s} {'Mask%':>7s} {'Tokens':>7s}")
    print("-" * 85)

    summary = defaultdict(list)
    for (spk, emo, tid, cond), items in sorted(groups.items()):
        slrs = [r["SLR"] for r in items]
        cers = [r["CER"] for r in items]
        mask_pcts = [r.get("mask_pct", 0) for r in items]
        n_tokens = [r.get("n_tokens_total", 0) for r in items]
        print(f"{spk:<8s} {emo:<8s} {tid:<10s} {cond:<14s} "
              f"{np.mean(slrs):7.3f} {np.std(slrs, ddof=1):6.3f} "
              f"{np.mean(cers):7.3f} {np.mean(mask_pcts):6.1f}% {np.mean(n_tokens):6.0f}")
        summary[(spk, emo, tid, cond) if cond != "normal" else ("REF", spk, emo, tid)].append({
            "cond": cond, "SLR_mean": np.mean(slrs), "SLR_std": np.std(slrs, ddof=1),
            "CER_mean": np.mean(cers),
        })

    # ── Before/After Comparison ──
    print(f"\n{'='*80}")
    print("SLR REDUCTION: Normal → Proportional Mask")
    print(f"{'='*80}")
    print(f"{'Case':<25s} {'Normal SLR':>10s} {'Masked SLR':>10s} {'Reduction':>10s}")
    print("-" * 60)

    reductions = []
    for cfg in test_configs:
        spk, emo, tid = cfg["speaker"], cfg["emotion"], cfg["text_id"]
        normal_items = groups.get((spk, emo, tid, "normal"), [])
        mask_items = groups.get((spk, emo, tid, "prop_mask"), [])
        thin_items = groups.get((spk, emo, tid, "thin_mask"), [])

        if normal_items and mask_items:
            normal_slr = np.mean([r["SLR"] for r in normal_items])
            mask_slr = np.mean([r["SLR"] for r in mask_items])
            thin_slr = np.mean([r["SLR"] for r in thin_items]) if thin_items else 0
            reduction = normal_slr - mask_slr
            pct = 100 * reduction / normal_slr if normal_slr > 0 else 0
            reductions.append((f"{spk}/{emo}/{tid}", normal_slr, mask_slr, thin_slr, reduction, pct))
            print(f"{spk}/{emo}/{tid:<15s} {normal_slr:10.3f} {mask_slr:10.3f} {reduction:9.3f} ({pct:.0f}%)")

    # Average reduction
    if reductions:
        avg_pct = np.mean([r[5] for r in reductions])
        print(f"\n  Average SLR reduction: {avg_pct:.0f}%")

    # Save
    result_path = OUT / "fix_results.json"
    json.dump({
        "test_cases": TEST_CASES,
        "conditions": conditions,
        "seeds": SEEDS,
        "reductions": [{"case": c, "normal_slr": n, "mask_slr": m, "thin_slr": t,
                        "reduction": r, "pct": p}
                       for c, n, m, t, r, p in reductions],
        "runs": ok,
    }, open(result_path, "w"), ensure_ascii=False, indent=2)
    print(f"\nSaved: {result_path}")
    print(f"Total: {(time.monotonic()-t0)/60:.0f} min")
    print("Done.")


if __name__ == "__main__":
    main()
