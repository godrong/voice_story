# Experiment 002 — Objective Eval Baseline

_Run at: 2026-05-17 18:32:27_

**Metrics**:
- **MOS-NISQA**: NISQA neural MOS predictor (primary naturalness axis), [1, 5], higher better.
- **MOS-P808**: DNSMOS-P.808 (auxiliary; known to mis-rank TTS output, kept only for cross-check).
- **WER/CER**: ASR cycle (faster-whisper / FunASR) vs `normalize_for_tts(target_text)`, [0, 1], lower better.
- **SECS**: `microsoft/wavlm-base-plus-sv` x-vector cosine similarity vs ref audio, [-1, 1], higher better. > 0.7 ≈ same speaker.
- **F0 RMSE**: librosa.pyin F0 RMSE over voiced overlap (Hz), lower better. None if voiced overlap < 5 frames.

## t1_basic_test

_target_text_: "Hello world. This is a test of voice cloning using CosyVoice 2 zero shot. The quick brown fox jumps over the lazy dog."

| ref | instruct | MOS-NISQA | MOS-P808 | WER/CER | SECS | F0 RMSE | dur (s) | eval (s) |
|---|---|---|---|---|---|---|---|---|
| r0_baseline_calm | none | 4.252 | 4.149 | 0.261 (wer) | 0.963 | 22.90 | 9.9 | 32.5 |
| r1_emphatic_businessman | en_rising | 4.329 | 4.148 | 0.174 (wer) | 0.629 | 57.56 | 9.1 | 21.3 |
| r1_emphatic_businessman | none | 4.165 | 4.258 | 0.217 (wer) | 0.959 | 46.36 | 6.9 | 21.5 |
| r1_emphatic_businessman | zh_rising | 4.403 | 3.863 | 0.304 (wer) | 0.680 | 59.83 | 10.1 | 20.5 |
| r2_emphatic_country | none | 4.267 | 3.881 | 0.261 (wer) | 0.958 | 55.66 | 11.4 | 23.4 |

### Δ — instruct mode vs zero_shot baseline (t1_basic_test)

| Metric | zero_shot (mean) | instruct (mean) | Δ |
|---|---|---|---|
| MOS-NISQA | 4.228 | 4.366 | **+0.138** |
| MOS-P808 | 4.096 | 4.006 | **-0.091** ⚠️ |
| WER/CER | 0.246 | 0.239 | **-0.007** |
| SECS | 0.960 | 0.654 | **-0.305** ⚠️ |
| F0 RMSE | 41.641 | 58.694 | **+17.053** ⚠️ |

## t2_trump_style

_target_text_: "We are going to make voice cloning great again, believe me. Nobody does it better. It's tremendous, just tremendous."

| ref | instruct | MOS-NISQA | MOS-P808 | WER/CER | SECS | F0 RMSE | dur (s) | eval (s) |
|---|---|---|---|---|---|---|---|---|
| r0_baseline_calm | none | 4.614 | 4.161 | 0.200 (wer) | 0.988 | 19.10 | 7.8 | 21.7 |
| r1_emphatic_businessman | en_emphatic | 4.441 | 3.850 | 0.250 (wer) | 0.946 | 69.00 | 6.9 | 21.3 |
| r1_emphatic_businessman | en_rising | 4.333 | 3.997 | 0.100 (wer) | 0.909 | 67.05 | 8.2 | 20.9 |
| r1_emphatic_businessman | none | 4.211 | 4.040 | 0.200 (wer) | 0.972 | 58.90 | 8.1 | 21.4 |

### Δ — instruct mode vs zero_shot baseline (t2_trump_style)

| Metric | zero_shot (mean) | instruct (mean) | Δ |
|---|---|---|---|
| MOS-NISQA | 4.413 | 4.387 | **-0.026** |
| MOS-P808 | 4.101 | 3.924 | **-0.177** ⚠️ |
| WER/CER | 0.200 | 0.175 | **-0.025** |
| SECS | 0.980 | 0.928 | **-0.052** ⚠️ |
| F0 RMSE | 39.001 | 68.025 | **+29.025** ⚠️ |

## t3_wangrong

_target_text_: "It’s great to be back in Beijing, truly fantastic. But I have to say, I’m very disappointed that Wang Rong couldn’t join..."

| ref | instruct | MOS-NISQA | MOS-P808 | WER/CER | SECS | F0 RMSE | dur (s) | eval (s) |
|---|---|---|---|---|---|---|---|---|
| r0_baseline_calm | none | 4.528 | 4.187 | 0.075 (wer) | 0.983 | 18.82 | 27.5 | 31.5 |
| r1_emphatic_businessman | en_emphatic | 4.601 | 4.162 | 0.113 (wer) | 0.854 | 66.06 | 25.4 | 31.5 |
| r1_emphatic_businessman | en_rising | 4.660 | 4.188 | 0.050 (wer) | 0.847 | 55.39 | 29.7 | 38.4 |
| r1_emphatic_businessman | none | 4.643 | 4.159 | 0.113 (wer) | 0.974 | 46.23 | 23.2 | 35.5 |
| r1_emphatic_businessman | zh_rising | 4.486 | 4.114 | 0.075 (wer) | 0.820 | 58.30 | 25.8 | 39.3 |
| r2_emphatic_country | none | 4.339 | 3.992 | 0.087 (wer) | 0.956 | 51.41 | 35.1 | 48.5 |

### Δ — instruct mode vs zero_shot baseline (t3_wangrong)

| Metric | zero_shot (mean) | instruct (mean) | Δ |
|---|---|---|---|
| MOS-NISQA | 4.503 | 4.582 | **+0.079** |
| MOS-P808 | 4.113 | 4.155 | **+0.042** |
| WER/CER | 0.092 | 0.079 | **-0.013** |
| SECS | 0.971 | 0.840 | **-0.131** ⚠️ |
| F0 RMSE | 38.819 | 59.916 | **+21.097** ⚠️ |

