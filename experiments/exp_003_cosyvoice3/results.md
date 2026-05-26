# Experiment 003 — CosyVoice 3 Zero-Shot Baseline

_Run at: 2026-05-26 on RTX 4090 (24GB)_

**Model**: Fun-CosyVoice3-0.5B (CosyVoice 3 base)
**Reference**: built-in zero_shot_prompt.wav (default female speaker)
**Metrics**: All 4 axes — MOS-NISQA / SECS / CER / F0 RMSE

## Results

| Sample | Text | MOS-NISQA ↑ | SECS ↑ | CER ↓ | F0 RMSE ↓ | RTF |
|---|---|---|---|---|---|---|
| zh_poem | 八百标兵奔北坡... | 3.975 | 0.913 | 0.188 | 103.6 Hz | 0.56 |
| zh_news | 随着人工智能技术的飞速发展... | 3.964 | 0.956 | **0.049** | 119.9 Hz | 0.37 |
| zh_prose | 春天来了，桃花开了... | 3.727 | **0.964** | 0.114 | **76.2 Hz** | 0.36 |
| zh_ancient | 自三峡七百里中... | **4.432** | 0.948 | 0.286 | 88.7 Hz | 0.38 |
| **MEAN** | | **4.024** | **0.945** | **0.159** | **97.1 Hz** | 0.42 |

### ASR Transcription

| Sample | Reference | ASR (Whisper large-v3) |
|---|---|---|
| zh_poem | 八百标兵奔北坡，北坡炮兵并排跑... | 八百镖兵奔北坡，北坡炮兵并排跑... (homophone: 标→镖) |
| zh_news | 随着人工智能技术的飞速发展... | 随着人工智能技术的飞速发展... (perfect, only punctuation) |
| zh_prose | 春天来了，桃花开了... | 春天来了，桃花开了... (perfect, only punctuation) |
| zh_ancient | 自三峡七百里中，两岸连山，略无阙处。重岩叠嶂... | 连山略无确处，重言叠账 (classical vocab errors) |

## Interpretation

- **SECS 0.945**: Excellent speaker fidelity. CosyVoice 3 zero-shot preserves speaker identity well.
- **MOS-NISQA 4.024**: Good naturalness. Ancient text scored highest at 4.432, suggesting CosyVoice 3
  handles classical Chinese prosody well despite vocabulary errors.
- **CER 0.159**: Good overall intelligibility. Modern Chinese near-perfect (CER 0.049 for news).
  Classical Chinese drops to 0.286 — CosyVoice 3 has less exposure to classical vocabulary.
  **This is a research-relevant finding for MCGA dataset evaluation.**
- **F0 RMSE 97.1 Hz**: Prosody deviation. Consistent with CosyVoice 2 finding (F0 RMSE +21 Hz in
  instruct mode). Validates the research direction: **zero-shot preserves speaker identity but
  prosody precision needs training-side improvement**.

## Comparison with CosyVoice 2 (exp_002)

| Metric | CosyVoice 2 zero_shot | CosyVoice 3 zero_shot |
|---|---|---|
| MOS-NISQA | 4.503 | 4.024 |
| SECS | 0.971 | 0.945 |
| F0 RMSE | 38.8 Hz | 97.1 Hz |

**Note**: Different reference audio and test texts. Direct comparison is suggestive, not conclusive.
Both models show the same qualitative pattern: high SECS, elevated F0 RMSE.

## Next Steps

- [x] 4-axis baseline established
- [ ] Consistent ref audio comparison between CosyVoice 2 and 3
- [ ] MCGA dataset zero-shot eval (needs HF access or modelscope mirror for data download)
- [ ] LoRA training after full MCGA release (train split)
