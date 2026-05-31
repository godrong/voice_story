# 实验日志

> 2026-05-30 · CV3 零样本情感合成完整实验记录

---

## exp_005 — 正确基线

- **日期**：2026-05-30
- **脚本**：`experiments/exp_005_clean_multiemotion/run_clean.py`
- **设计**：8spk × 5emo × 6text × 3seed = 720 合成
- **格式**：`"You are a helpful assistant.<|endofprompt|>" + ref_text`（修正后）
- **结果**：`results/h800_final/exp005_baseline.json` (540KB)
- **WAV**：`/root/autodl-tmp/clean_multiemotion/audio/` (720 文件, 636MB)
- **发现**：CER~0, SECS>0.95, 高唤醒 F0 差 1.8×, CER-F0 无相关

---

## exp_006 — System Prompt Ablation

- **日期**：2026-05-30
- **脚本**：`experiments/exp_006_prompt_ablation/run_ablation.py`
- **设计**：4spk × 4emo × 6text × 4prompts × 3seed = 1152 合成
- **Prompt 变体**：en_default / zh_default / emotion_neutral / minimal
- **结果**：`results/h800_final/exp006_prompt.json` (853KB)
- **WAV**：`/root/autodl-tmp/prompt_ablation/audio/` (1152 文件, 1.1GB)
- **发现**：中文 prompt 有害(CER 10×), 空 prompt 可用, 最佳 prompt 因情感而异
- **备注**：首次运行在 4090 上 F0 阶段崩溃，H800 上完整重跑评估

---

## P0 (exp_007) — Paralinguistic Token Matrix

- **日期**：2026-05-30
- **脚本**：`experiments/exp_007_paralinguistic/run_token_matrix.py`
- **设计**：12 tokens × 2spk × 2emo × 2text × 2cond = 192 合成
- **结果**：`results/h800_final/p0_token_matrix.json` (130KB)
- **WAV**：`/root/autodl-tmp/paralinguistic_matrix/audio/` (192 文件, 70MB)
- **本地音频**：`experiments/exp_007_paralinguistic/audio_samples/` (12 个对比文件)
- **发现**：`[laughter]`/`[clucking]`/`[hissing]` 有效, `[vocalized-noise]` 有害
- **Token 链路**：text token(151939) → Qwen2 embedding(896d) → silent FSQ codes → 25Hz→mel→24kHz

---

## P1 (exp_008) — Para × Emotion Interaction

- **日期**：2026-05-30
- **脚本**：`experiments/exp_008_para_emotion/run.py`
- **设计**：4 tokens × 4 emo × 4 spk × 2 text × 2 cond × 3 seed = 192 合成
- **结果**：`results/h800_final/p1_para_emotion.json` (112KB)
- **WAV**：`/root/autodl-tmp/para_emotion/audio/` (192 文件, 72MB)
- **发现**：仅 Angry+`[breath]` 有效(F0↓7Hz), Sad+`[sigh]` 反效果(F0↑17Hz)
- **备注**：4090D 上因输出缓冲被误杀 2 次，H800 上用 nohup+文件日志成功

---

## P2 (exp_009) — Emotion Visualization

- **日期**：2026-05-30
- **脚本**：`experiments/exp_009_emotion_viz/run_p2.py`
- **设计**：360 WavLM embeddings (72/emotion) from exp_005 WAVs
- **结果**：`results/h800_final/p2_emotion_viz.json` + `p2_emotion_viz.png`
- **发现**：CV accuracy=65.8%, Happy 最可分, Sad/Neutral 最模糊, PCA 32%
- **备注**：初版有 `.detach()` bug, 修复后重跑成功

---

## P3 (exp_008_longtext) — Long-form Drift

- **日期**：2026-05-30 (先于 P1/P2 完成)
- **脚本**：`experiments/exp_008_longtext_drift/run_drift.py`
- **设计**：7 段递增长度 (50→3200 chars) × 1 ref
- **结果**：`results/h800_final/p3_longtext.json` + `p3_drift_report.txt`
- **WAV**：`/root/autodl-tmp/longtext_drift/audio/` (54MB)
- **发现**：200 字 CER=0.12, 800 字=0.92, 1600+ 完全崩溃
- **备注**：F0 RMSE 列全 0, 疑似计算 bug

---

## E1 — Ref Quality Attribution

- **日期**：2026-05-30
- **方法**：本地分析 exp_005 结果 JSON
- **发现**：F0 RMSE 跨 ref 差异 9× (14-129Hz), CER-F0 不相关 (r=0.095), Neutral 最安全
- **Top-5 通用好 Ref**：0006/Neutral(14Hz), 0008/Sad(21Hz), 0004/Neutral(19Hz)

---

## 已废弃实验 (exp_002-004)

- **exp_002**：CosyVoice 2 instruct mode — 确认音色损失 (SECS -0.13)
- **exp_003**：LoRA style adaptation — rank=8, F0↓23Hz
- **exp_004**：基于 prompt 格式错误的"语义泄漏"调查 — 全部无效

---

## 硬件使用记录

| 机器 | GPU | VRAM | 时间段 | 主要任务 |
|------|-----|------|--------|------|
| 4090 #1 | RTX 4090 | 24GB | 05/28-05/30 | exp_002-007 合成 |
| 4090D | RTX 4090D | 24GB | 05/30 下午 | P1 调试（失败, ONNX CPU） |
| H800 | H800 PCIe | 80GB | 05/30 晚间 | P1-P3 完成, exp_006 eval |
