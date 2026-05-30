# Research Plan — CV3 Zero-Shot Emotional TTS Baseline

> _Last updated: 2026-05-30 · exp_005 完成，prompt 格式修正_

---

## 0. TL;DR

exp_005 用正确 prompt 格式建立 CV3 零样本情感合成基线（8 spk × 5 emo × 6 text × 3 seed = 720 合成）：

| 情感 | CER | SECS | F0 RMSE |
|------|-----|------|---------|
| Neutral | 0.007 | 0.970 | 46.0 Hz |
| Happy | 0.007 | 0.965 | 68.1 Hz |
| Angry | 0.009 | 0.959 | 72.5 Hz |
| Surprise | 0.008 | 0.965 | 83.9 Hz |
| Sad | 0.009 | 0.976 | 57.7 Hz |

**核心发现**：
- 内容保真度在所有情感下接近完美（CER ~0）
- **高唤醒情感（Surprise/Angry/Happy）F0 RMSE 是低唤醒（Neutral/Sad）的 1.5-1.8 倍**
- CV3 零样本在正确调用下表现良好，不需要训练/微调
- 之前 exp_002-004 报告的"语义泄漏"是 prompt 格式错误，已修正

---

## 1. 已完成工作

### 1.1 评测框架

- 4 维客观指标：CER (FunASR) / SECS (WavLM-SV) / F0 RMSE (librosa.pyin) / MOS (NISQA)
- 发现 DNSMOS-P808 对 TTS 输出给出反向排序，替换为 NISQA

### 1.2 Instruct Mode 缺陷验证 (exp_002)

- 确认官方承认的"instruct mode 无法通过文本控制音色"
- SECS -0.13, F0 RMSE +21 Hz

### 1.3 LoRA Style Adaptation (exp_003)

- LoRA rank=8 在 ESD 上训练，F0 RMSE 降低 23 Hz，SECS 不变
- 证明 style-speaker 解耦可行

### 1.4 方法论教训 (exp_004)

- `<|endofprompt|>` token 放置错误导致虚假的"语义泄漏"发现
- 修正流程：先对比官方 example，再大规模实验
- Memory: `feedback_check_official_first`

### 1.5 正确基线 (exp_005)

- 正确格式下 720 合成，建立 CV3 情感合成完整基线

---

## 2. 后续研究方向

### 方向 1：高唤醒情绪 F0 传递优化

**问题**：Surprise F0 RMSE (83.9 Hz) 是 Neutral (46.0 Hz) 的 1.8 倍。这是真实的模型能力边界。

**实验**：
- System prompt 消融：不同 prompt 是否影响 F0 传递？
- 参考音频筛选：同一情感内，选择 F0 更稳定的 ref 是否改善？
- 多 ref 平均：合并多个 ref 的韵律信号

### 方向 2：系统提示词作为控制杆

**问题**：官方 prompt 是英文的 `"You are a helpful assistant."`。中文 prompt 或情感相关 prompt 是否改变行为？

**实验**：
- 中文 prompt vs 英文 prompt 对比
- 情感提示词（如 "Speak with a sad tone."）的效果
- Prompt 对高唤醒情绪 F0 传递的影响

### 方向 3：参考音频质量因素

**问题**：40 个 ref 的质量差异（时长、SNR、F0 稳定性）如何影响合成？

**实验**：用 exp_005 数据做 ref 特征与 CER/SECS/F0 RMSE 的相关性分析

### 方向 4：长文本合成稳定性

**问题**：CV3 单句表现好，但长文本（段落级）是否保持质量？

**实验**：递增文本长度的合成质量曲线，找到退化点
