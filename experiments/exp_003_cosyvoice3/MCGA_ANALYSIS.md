# MCGA 数据集分析 & CosyVoice 3 训练方法设计

## 1. MCGA 数据结构

### 1.1 HuggingFace 实际内容

通过 `datasets.load_dataset("yxdu/MCGA")` 加载后：

| 维度 | 数值 |
|---|---|
| 总样本数 | ~1,950 (仅 test split 已发布) |
| 全量预期 | 22,000 样本 / 119 小时 |
| 音频时长 | 5s ~ 30s / 条 |
| 采样率 | 16kHz (Parquet 内嵌) |
| 说话人 | 28 (13男/15女)，无显式 speaker_id |

### 1.2 核心列

```
id        — 样本ID (如 "2692-1")
audio     — 音频 array + sampling_rate
asr       — 中文转录文本 (ASR 任务的 ground truth)
s2tt      — 英文翻译
gender    — 说话人性别 (男/女)
sec_1     — 简短标籤 (如 "中性")
sec_2     — 说话人描述 (如 "这是一段青年男声朗诵。")
sec_3     — 详细情感分析
genre     — 文学体裁 (赋/诗/文/词/曲)
author    — 作者 (如 屈原、韩愈)
title     — 作品名
dynasty   — 朝代 (先秦/唐/宋/元/明/清)
```

### 1.3 关键发现

**问题 1：无显式 speaker_id**

28 个说话人无法通过一个字段区分。只能用 `sec_2` 描述文本做启发式分组，粒度粗糙。

**问题 2：仅 test split 已发布**

全量 22,000 条中目前只放出了 ~1,950 条 test 集。等全量放出后才能做有意义的训练。

**问题 3：古典文学领域**

文本是文言文/古诗词，与通用 TTS 的训练分布 (新闻/对话) 差异大。这意味着：
- 评测 CosyVoice 3 的 zero-shot 泛化到古文领域 → 有价值的研究问题
- 用来做训练 → 会让模型偏向古文朗诵风格

---

## 2. CosyVoice 3 不是 CLIP 对齐

### 2.1 你的问题："是做 CLIP 对齐还是什么"

**答案：不是 CLIP 对齐。CosyVoice 3 是自回归生成模型，不是对比学习模型。**

| 模型范式 | CLIP / CLAP | CosyVoice 3 |
|---|---|---|
| 训练目标 | 对比损失 (contrastive) | 交叉熵 (autoregressive CE) |
| 学什么 | 音频-文本联合嵌入空间 | 文本 → 语音 token 的映射 |
| 推理方式 | 算相似度 | 逐 token 生成 |
| 典型架构 | 双塔 encoder | LLM decoder + Flow Matching |

CLIP/CLAP 对齐在 TTS 里的角色是**别的组件**——比如 speaker encoder (ECAPA-TDNN / WavLM) 或 emotion encoder (emotion2vec)——不是训练主目标。

### 2.2 CosyVoice 3 训练的实际流程

```
阶段 1: LLM (自回归)
  输入:  [text_tokens] + [speaker_embedding]
  输出:  [speech_token_1, speech_token_2, ..., speech_token_n]
  Loss:  CrossEntropy(predicted_token, ground_truth_token)
  作用:  学会"这段文字应该对应哪些语音单元"

阶段 2: Flow Matching (ODE)
  输入:  [speech_tokens] + [speaker_embedding]
  输出:  [mel_spectrogram]
  Loss:  L1(predicted_mel, target_mel)
  作用:  学会"语音 token 到声学特征的映射"

阶段 3: HiFiGAN (GAN + 重建)
  输入:  [mel_spectrogram]
  输出:  [waveform]
  Loss:  GAN loss + Mel reconstruction loss + Feature matching loss
  作用:  学会"声学特征到波形的转换"
```

**每个阶段的"对齐"都不是对比学习，而是监督学习的回归/分类。**

---

## 3. MCGA 可以怎么用于训练

### 3.1 当前可以做的事 (仅 test split)

| 用法 | 可行性 | 说明 |
|---|---|---|
| Zero-shot 评测 | ✅ 现在就做 | MCGA 作 eval set，测 CosyVoice 3 在古文领域的基础克隆能力 |
| 训练 | ❌ 数据不够 | 1,950 条太小，等全量 22,000 条放出 |
| 说话人分组分析 | ⚠️ 受限 | 无 speaker_id，只能靠 sec_2 试探 |

### 3.2 全量放出后可以做的事

当 22,000 条全部放出后：

**方案 A：CosyVoice 3 后训练 (官方 SFT)**

```
数据格式: utt_id <TAB> wav_path <TAB> text
训练方式: torchrun + DeepSpeed Stage 2
训练阶段: 先 LLM → 再 Flow → 最后 HiFiGAN
适用: 有 speaker_id 时可用
```

用 MCGA 的 `asr` 列当 text，`audio` 当训练目标。但这需要 speaker 标注来做 speaker-conditioned 训练。

**方案 B：LoRA 微调 LLM backbone**

```
注入位置: Qwen2Encoder 的 attention 层 (q/k/v/o proj)
冻结: 原始权重 + Flow + HiFiGAN
训练: 仅更新 LoRA 参数
Loss: cross-entropy on speech tokens
数据需求: 每条 = (text, target_audio, ref_audio_of_same_speaker)
```

这是我们的 QLoRA 方案。核心问题：**需要同 speaker 的 ref→target pair**，而 MCGA 缺 speaker_id。

**方案 C：不考虑 speaker 的纯 style adaptation (推荐)**

如果要绕过 speaker_id 缺失的问题：

```
输入:  text (asr 列)
目标:  audio (同一条的音频)
条件:  不注入 speaker embedding，让模型通过 prompt audio 自行提取

这实际上训练的是: "古文文本 → 古文朗诵风格" 的映射
而不是 "特定人 → 特定音色"
```

这与我们的 Tier 1 Style LoRA 设计完全对齐——多说话人混合训练，不针对单个 speaker。

---

## 4. 推荐路线

考虑到 MCGA 仅放出 test split 且无 speaker_id：

| 阶段 | 做什么 | 数据 |
|---|---|---|
| **现在** | Zero-shot eval：测 CosyVoice 3 在古文上的基础能力 | MCGA test split (~1,950 条) |
| **现在** | 分析 genre × speaker 分布，确认 diversity | prep_mcga.py 输出 |
| **等全量** | 用 MCGA 全量 + ESD 混合做 Tier 1 Style LoRA | MCGA full + ESD |
| **等全量** | 如果有 speaker_id，做 Tier 2 Avatar LoRA | MCGA 单 speaker 子集 |

### 4.1 当前推理评测就够了

MCGA test split 做评测集的价值：
1. 验证 CosyVoice 3 在**非通用领域**（古文）的 zero-shot 泛化
2. 回答研究问题："模型能否克隆一个朗诵古文的说话人？"
3. 4 维客观指标 + 跨 genre 对比 (赋 vs 诗 vs 词)
4. 这本身就是一个可写进面试的故事点

### 4.2 训练暂缓

等 MCGA 放出 train split 且有 speaker 标注后再做训练。当前先跑 inference eval 拿 baseline。
