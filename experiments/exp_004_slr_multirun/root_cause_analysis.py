#!/usr/bin/env python3
"""Root Cause Analysis: WHY does Sad leak? What differentiates leaky vs immune speakers?

Extracts acoustic features from all 40 reference audios (8 spk × 5 emo),
then correlates with per-speaker Sad SLR to identify the root cause.

Features: F0 stats, speaking rate, voicing ratio, spectral centroid/bandwidth,
energy, speech token entropy (via CV3 VQ-VAE if available).
"""

import sys, os, json
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

RESULTS_PATH = "/root/autodl-tmp/boundary_test_multispk/results.json"
ESD_LIST = "/root/autodl-tmp/esd_cn/train.data.list"
EMOTIONS = ["Neutral", "Happy", "Angry", "Surprise", "Sad"]

import librosa
import torch


def extract_ref_features(wav_path, ref_text):
    """Extract acoustic features from a reference audio file.

    Returns dict with F0 stats, duration, rate, voicing, spectral, energy.
    """
    y, sr = librosa.load(str(wav_path), sr=16000)
    dur = len(y) / sr
    n_chars = len(ref_text.replace(" ", "").replace("，", "").replace("。", ""))

    # F0
    f0, voiced, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sr,
                                 frame_length=2048, hop_length=320)
    f0_clean = f0[voiced] if voiced.sum() > 0 else np.array([0])
    voicing_ratio = voiced.mean()

    # Spectral
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=320))
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr).squeeze()
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr).squeeze()

    # Energy / RMS
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=320).squeeze()

    # Speech rate
    speech_rate = n_chars / dur if dur > 0 else 0

    # F0 variation (coefficient of variation)
    f0_cv = np.std(f0_clean) / np.mean(f0_clean) if np.mean(f0_clean) > 0 else 0

    # F0 slope / trajectory (linear fit to voiced F0)
    if len(f0_clean) > 2:
        f0_slope = np.polyfit(np.arange(len(f0_clean)), f0_clean, 1)[0]
    else:
        f0_slope = 0

    return {
        "duration_s": round(dur, 2),
        "n_chars": n_chars,
        "speech_rate_chars_per_s": round(speech_rate, 2),
        "f0_mean_hz": round(float(np.mean(f0_clean)), 1),
        "f0_std_hz": round(float(np.std(f0_clean)), 1),
        "f0_min_hz": round(float(np.min(f0_clean)), 1),
        "f0_max_hz": round(float(np.max(f0_clean)), 1),
        "f0_range_hz": round(float(np.max(f0_clean) - np.min(f0_clean)), 1),
        "f0_cv": round(float(f0_cv), 4),
        "f0_slope": round(float(f0_slope), 4),
        "voicing_ratio": round(float(voicing_ratio), 3),
        "spectral_centroid_hz": round(float(np.mean(centroid)), 1),
        "spectral_bandwidth_hz": round(float(np.mean(bandwidth)), 1),
        "energy_rms_mean": round(float(np.mean(rms)), 5),
        "energy_rms_std": round(float(np.std(rms)), 5),
        "energy_rms_cv": round(float(np.std(rms) / np.mean(rms)), 3) if np.mean(rms) > 0 else 0,
    }


def main():
    print("=" * 80)
    print("ROOT CAUSE ANALYSIS: Semantic Leakage in Sad Emotion")
    print("=" * 80)

    # Load results
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    records = data["runs"]
    config = data["config"]
    speakers = config["speakers"]
    print(f"\nLoaded {len(records)} records, {len(speakers)} speakers")

    # ── Step 1: Extract per-speaker Sad performance ──
    sad_by_spk = defaultdict(list)
    for r in records:
        if r["emotion"] == "Sad":
            sad_by_spk[r["speaker"]].append(r)

    print(f"\n{'='*80}")
    print("SAD PERFORMANCE BY SPEAKER")
    print(f"{'='*80}")
    print(f"{'Spk':<6s} {'SLR':>7s} {'±':>6s} {'CER':>7s} {'±':>6s} {'SECS':>7s} {'F0_RMSE':>8s} {'Leaky?':>8s}")
    print("-" * 75)

    sad_perf = {}
    for spk in sorted(sad_by_spk.keys()):
        items = sad_by_spk[spk]
        slrs = [r["SLR"] for r in items]
        cers = [r["CER"] for r in items]
        secs = [r["SECS"] for r in items if r["SECS"] is not None]
        f0s  = [r["F0_RMSE_Hz"] for r in items if r["F0_RMSE_Hz"] is not None]
        avg_slr = np.mean(slrs)
        is_leaky = "YES" if avg_slr > 0.05 else "no"
        sad_perf[spk] = {
            "SLR_mean": round(np.mean(slrs), 3),
            "SLR_std": round(np.std(slrs, ddof=1), 3),
            "CER_mean": round(np.mean(cers), 3),
            "SECS_mean": round(np.mean(secs), 3),
            "F0_RMSE_mean": round(np.mean(f0s), 1),
            "is_leaky": is_leaky,
            "n": len(items),
        }
        p = sad_perf[spk]
        print(f"{spk:<6s} {p['SLR_mean']:7.3f} {p['SLR_std']:6.3f} "
              f"{p['CER_mean']:7.3f} {round(np.std(cers, ddof=1), 3):6.3f} "
              f"{p['SECS_mean']:7.3f} {p['F0_RMSE_mean']:8.1f} {is_leaky:>8s}")

    # ── Step 2: Extract reference audio features ──
    print(f"\n{'='*80}")
    print("REFERENCE AUDIO FEATURE EXTRACTION (40 refs = 8 spk × 5 emo)")
    print(f"{'='*80}")

    # Reconstruct refs (same deterministic selection)
    pool = defaultdict(list)
    with open(ESD_LIST) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3: continue
            key, wav, text = parts[0], parts[1], parts[2]
            fields = key.split("_")
            spk, emo = fields[1], fields[2]
            pool[(spk, emo)].append((wav, text))

    import random
    random.seed(42)
    refs = {}
    for spk in speakers:
        for emo in EMOTIONS:
            candidates = pool.get((spk, emo), [])
            if candidates:
                refs[(spk, emo)] = random.choice(candidates)

    # Extract features
    ref_features = {}
    for (spk, emo), (wav, text) in refs.items():
        if not Path(wav).exists(): continue
        feats = extract_ref_features(wav, text)
        ref_features[(spk, emo)] = feats
        if emo == "Sad":
            print(f"  {spk}/{emo}: F0={feats['f0_mean_hz']:.0f}±{feats['f0_std_hz']:.0f}Hz, "
                  f"voicing={feats['voicing_ratio']:.2f}, "
                  f"rate={feats['speech_rate_chars_per_s']:.1f} ch/s, "
                  f"spectral_cent={feats['spectral_centroid_hz']:.0f}Hz")

    # ── Step 3: Compare leaky vs non-leaky Sad refs ──
    print(f"\n{'='*80}")
    print("LEAKY vs NON-LEAKY SAD REF COMPARISON")
    print(f"{'='*80}")

    leaky_spks = [spk for spk in speakers if sad_perf.get(spk, {}).get("is_leaky") == "YES"]
    nonleaky_spks = [spk for spk in speakers if sad_perf.get(spk, {}).get("is_leaky") == "no"]

    print(f"Leaky speakers:    {leaky_spks}")
    print(f"Non-leaky speakers: {nonleaky_spks}")

    feature_names = [
        "f0_mean_hz", "f0_std_hz", "f0_range_hz", "f0_cv", "f0_slope",
        "speech_rate_chars_per_s", "voicing_ratio",
        "spectral_centroid_hz", "spectral_bandwidth_hz",
        "energy_rms_mean", "energy_rms_std", "energy_rms_cv",
        "duration_s", "n_chars",
    ]

    print(f"\n{'Feature':<30s} {'Leaky mean':>12s} {'Non-leaky mean':>16s} {'Diff':>10s} {'Ratio':>8s}")
    print("-" * 80)

    significant = []
    for feat_name in feature_names:
        leaky_vals = [ref_features[(spk, "Sad")][feat_name] for spk in leaky_spks]
        non_vals = [ref_features[(spk, "Sad")][feat_name] for spk in nonleaky_spks]
        lm = np.mean(leaky_vals)
        nm = np.mean(non_vals)
        diff = lm - nm
        ratio = lm / nm if nm != 0 else float('inf')

        marker = ""
        if abs(ratio - 1) > 0.15:  # >15% difference
            marker = " ← SIGNIFICANT"
            significant.append((feat_name, lm, nm, diff, ratio))

        print(f"{feat_name:<30s} {lm:12.3f} {nm:16.3f} {diff:10.3f} {ratio:8.2f}{marker}")

    print(f"\nSignificant differentiators:")
    for feat_name, lm, nm, diff, ratio in significant:
        direction = "higher" if diff > 0 else "lower"
        print(f"  • {feat_name}: leaky={lm:.3f} vs non-leaky={nm:.3f} "
              f"({direction} by {abs(ratio-1)*100:.0f}%)")

    # ── Step 4: Cross-emotion feature comparison ──
    print(f"\n{'='*80}")
    print("CROSS-EMOTION ACOUSTIC FEATURES (all 8 speakers averaged)")
    print(f"{'='*80}")
    emo_feats = defaultdict(list)
    for (spk, emo), feats in ref_features.items():
        emo_feats[emo].append(feats)

    print(f"{'Feature':<25s}", end="")
    for emo in EMOTIONS:
        print(f"{emo:>10s}", end="")
    print()
    print("-" * 80)

    key_features = ["f0_mean_hz", "f0_std_hz", "f0_cv", "speech_rate_chars_per_s",
                    "voicing_ratio", "spectral_centroid_hz", "energy_rms_cv"]
    for feat_name in key_features:
        print(f"{feat_name:<25s}", end="")
        for emo in EMOTIONS:
            vals = [f[feat_name] for f in emo_feats[emo]]
            print(f"{np.mean(vals):10.2f}", end="")
        print()

    # Sad vs others
    print(f"\n--- Sad vs Neutral (the core comparison) ---")
    for feat_name in key_features:
        sad_vals = [f[feat_name] for f in emo_feats["Sad"]]
        neu_vals = [f[feat_name] for f in emo_feats["Neutral"]]
        ratio = np.mean(sad_vals) / np.mean(neu_vals) if np.mean(neu_vals) != 0 else float('inf')
        print(f"  {feat_name:<28s} Sad={np.mean(sad_vals):.2f}  Neutral={np.mean(neu_vals):.2f}  "
              f"ratio={ratio:.2f}")

    # ── Step 5: SLR correlation with ref features ──
    print(f"\n{'='*80}")
    print("PEARSON CORRELATION: Ref Features vs Sad SLR (per speaker)")
    print(f"{'='*80}")
    print(f"{'Feature':<30s} {'r':>8s} {'p (approx)':>12s} {'Interpretation':>20s}")
    print("-" * 80)

    correlations = []
    for feat_name in feature_names:
        feat_vals = [ref_features[(spk, "Sad")][feat_name] for spk in speakers]
        slr_vals = [sad_perf[spk]["SLR_mean"] for spk in speakers]

        if np.std(feat_vals) < 1e-6:
            continue
        r = np.corrcoef(feat_vals, slr_vals)[0, 1]

        # Simple significance estimate
        n = len(speakers)
        t_stat = r * np.sqrt((n - 2) / (1 - r**2)) if abs(r) < 1 else 999

        interpretation = ""
        if abs(r) > 0.5:
            interpretation = "STRONG predictor"
        elif abs(r) > 0.3:
            interpretation = "moderate predictor"
        else:
            interpretation = "weak/no predictor"

        correlations.append((feat_name, r, t_stat, interpretation))

    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    for feat_name, r, t_stat, interp in correlations:
        print(f"{feat_name:<30s} {r:8.3f} {t_stat:12.2f} {interp:>20s}")

    # ── Step 6: The key narrative ──
    print(f"\n{'='*80}")
    print("ROOT CAUSE NARRATIVE")
    print(f"{'='*80}")

    top_corr = correlations[:5]
    print("\nTop 5 SLR predictors in Sad ref audio:")
    for feat_name, r, _, interp in top_corr:
        direction = "positive" if r > 0 else "negative"
        print(f"  • {feat_name}: r={r:.3f} ({direction} correlation) — {interp}")

    # Save
    output = {
        "sad_performance_by_speaker": sad_perf,
        "ref_features": {f"{spk}_{emo}": feats for (spk, emo), feats in ref_features.items()},
        "leaky_speakers": leaky_spks,
        "nonleaky_speakers": nonleaky_spks,
        "significant_differentiators": [(f, lm, nm, diff, ratio) for f, lm, nm, diff, ratio in significant],
        "top_correlations": [(f, float(r), float(t), i) for f, r, t, i in correlations[:10]],
        "emotion_averages": {emo: {feat: np.mean([f[feat] for f in feats])
                                   for feat in key_features}
                             for emo, feats in emo_feats.items()},
    }
    out_path = "/root/autodl-tmp/boundary_test_multispk/root_cause_analysis.json"
    json.dump(output, open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
