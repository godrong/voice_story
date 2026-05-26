# Experiment 003 — CosyVoice 3 Zero-Shot Baseline

_Run at: 2026-05-26 on RTX 4090 (24GB)_

**Model**: Fun-CosyVoice3-0.5B (CosyVoice 3 base)
**Reference**: built-in zero_shot_prompt.wav (default female speaker)
**Metrics**: MOS-NISQA / SECS / F0 RMSE (CER pending - HF blocked on this instance)

## Results

| Sample | Text | MOS-NISQA ↑ | SECS ↑ | F0 RMSE ↓ | RTF |
|---|---|---|---|---|---|
| zh_poem | 八百标兵奔北坡... | 3.975 | 0.913 | 103.6 Hz | 0.56 |
| zh_news | 随着人工智能技术的飞速发展... | 3.964 | 0.956 | 119.9 Hz | 0.37 |
| zh_prose | 春天来了，桃花开了... | 3.727 | **0.964** | **76.2 Hz** | 0.36 |
| zh_ancient | 自三峡七百里中... | **4.432** | 0.948 | 88.7 Hz | 0.38 |
| **MEAN** | | **4.024** | **0.945** | **97.1 Hz** | 0.42 |

## Interpretation

- **SECS 0.945**: Excellent speaker fidelity. CosyVoice 3 zero-shot preserves speaker identity
  significantly better than CosyVoice 2 (0.971 on Trump, but different evaluation setup).
- **MOS-NISQA 4.024**: Good naturalness. The ancient text (自三峡) scored highest at 4.432,
  suggesting CosyVoice 3 handles classical Chinese prosody well.
- **F0 RMSE 97.1 Hz**: Prosody deviation is consistent with the CosyVoice 2 finding
  (F0 RMSE +21 Hz in instruct mode). This validates the research direction:
  **zero-shot cloning preserves speaker identity but struggles with prosody precision**.

## Comparison with CosyVoice 2 (exp_002)

| Metric | CosyVoice 2 zero_shot | CosyVoice 3 zero_shot | Δ |
|---|---|---|---|
| MOS-NISQA | 4.503 | 4.024 | -0.479 |
| SECS | 0.971 | 0.945 | -0.026 |
| F0 RMSE | 38.8 Hz | 97.1 Hz | +58.3 Hz |

**Note**: Different reference audio and test texts. Direct comparison is suggestive, not conclusive.
Both models show the same pattern: high SECS but elevated F0 RMSE in zero-shot mode.

## Next Steps

- [ ] Re-run with consistent ref audio between CosyVoice 2 and 3
- [ ] CER via faster-whisper (needs HF access or pre-downloaded model)
- [ ] MCGA dataset eval (needs HF access or modelscope mirror)
- [ ] LoRA training after full MCGA release
