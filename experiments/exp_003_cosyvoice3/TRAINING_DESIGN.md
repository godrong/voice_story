# CosyVoice 3 Training Design

## 0. Training Approaches

CosyVoice 3 是 LLM + Flow Matching + HiFiGAN 三阶段架构。训练有两种路线：

| 方法 | 官方后训练 (SFT) | QLoRA |
|---|---|---|
| 原理 | 全参微调 LLM/Flow/HiFiGAN | 在 attention 层插入低秩矩阵，冻结原权重 |
| 显存 | ~60GB (需 DeepSpeed Stage 2) | ~25GB (单卡 H800 可跑) |
| 训练速度 | 慢 | 快 (仅更新 ~3% 参数) |
| 过拟合风险 | 高 (小数据集) | 低 (LoRA 正则化效应) |
| 权重大小 | 完整模型 (~2GB) | 适配器 (~50MB) |
| 官方推荐 | 是 (examples/libritts/cosyvoice3/run.sh) | 否 (需自行添加) |

## 1. 官方后训练方法

### 1.1 设计原理

CosyVoice 3 使用 `cosyvoice/bin/train.py`，分三阶段分别训练：
- **LLM** (Qwen2Encoder backbone): 自回归文本→语音 token
- **Flow** (CausalMaskedDiffWithDiT): 语音 token→mel spectrogram  
- **HiFiGAN**: mel spectrogram→波形

每阶段用 DeepSpeed Stage 2 + AMP 混合精度。

### 1.2 关键超参数

| 参数 | 值 | 来源 |
|---|---|---|
| lr | 1e-5 | conf/cosyvoice3.yaml |
| warmup_steps | 2500 | conf/cosyvoice3.yaml |
| scheduler | constantlr | conf/cosyvoice3.yaml |
| max_epoch | 200 | conf/cosyvoice3.yaml |
| grad_clip | 5 | conf/cosyvoice3.yaml |
| batch (dynamic) | max_frames=2000 | conf/cosyvoice3.yaml |
| grad_accum | 2 (LLM/Flow) | conf/cosyvoice3.yaml |
| AMP | yes | run.sh --use_amp |
| DeepSpeed | Stage 2 | conf/ds_stage2.json |
| optimizer | Adam | conf/cosyvoice3.yaml |

### 1.3 训练命令

```bash
torchrun --nnodes=1 --nproc_per_node=1 \
    cosyvoice/bin/train.py \
    --config conf/cosyvoice3.yaml \
    --train_data data/train.data.list \
    --cv_data data/dev.data.list \
    --model llm \
    --checkpoint pretrained_models/Fun-CosyVoice3-0.5B/llm.pt \
    --model_dir exp/cosyvoice3/llm \
    --use_amp \
    --deepspeed_config conf/ds_stage2.json
```

## 2. QLoRA 方法

### 2.1 设计原理

在 LLM backbone 的 attention 层 (Q/K/V/O) 注入低秩适配器 (rank=16~64)：
- 冻结原始权重，仅训练 LoRA 参数
- 可在保持泛化能力的同时适应新说话人/风格
- 支持多 LoRA 切换 (每个说话人一个适配器包)

### 2.2 为什么选 QLoRA 而非 LoRA

- H800 单卡 80GB，但全量加载 CosyVoice 3 三阶段仍需 ~40GB
- 使用 4-bit quantization (NF4) 加载 LLM backbone，节省显存
- LoRA 层保持 BF16/FP16 精度

### 2.3 LoRA 配置设计

| 参数 | Tier 1 (Style LoRA) | Tier 2 (Avatar LoRA) |
|---|---|---|
| rank | 16 | 32 |
| alpha | 32 | 64 |
| target_modules | q_proj,k_proj,v_proj,o_proj | q_proj,k_proj,v_proj,o_proj |
| dropout | 0.05 | 0.05 |
| bias | "none" | "none" |

### 2.4 实现

参见 `train_lora.py`，核心逻辑：
1. 加载预训练 CosyVoice 3 LLM
2. 用 peft 注入 LoRA
3. 冻结非 LoRA 参数
4. 训练 text→speech_token 的 cross-entropy loss
5. 用 SECS + WER 做早停监控

## 3. Loss 函数和早停机制

### 3.1 训练 Loss

- **LLM**: Cross-entropy loss (预测下一个 speech token)，`length_normalized_loss: True`
- **Flow**: L1 regression loss (CFM reg_loss_type='l1')
- **HiFiGAN**: GAN loss (generator + discriminator, LSGAN)

### 3.2 早停机制

不在 training loss 上早停（语音 token loss 与感知质量不完全对齐），而是：

1. 每 N 步 (如 500) 保存 checkpoint
2. 对 checkpoint 在 dev set 上跑推理
3. 计算 SECS_vs_gold (说话人保真) + WER (可懂度) 
4. 若连续 3 个 checkpoint SECS 不提升且 WER 不下降则停止
5. 选 val SECS 最高的 checkpoint 为最佳模型

### 3.3 评测指标

| 指标 | 工具 | 方向 |
|---|---|---|
| SECS | WavLM-Base-Plus-SV | ↑，>0.7 同说话人 |
| MOS-NISQA | NISQA_DIM | ↑，[1,5] |
| WER/CER | FunASR (zh) / Whisper (en) | ↓ |
| F0 RMSE | librosa.pyin | ↓ |

## 4. 数据集

### 4.1 CV3-Eval (官方评测集)

- test-zh: 中文 zero-shot 评测样本
- test-en: 英文 zero-shot 评测样本  
- test-hard: 困难样本 (噪声、口音等)

### 4.2 训练数据集 (可选扩展)

沿用 voice_story 已有的 ESD + AISHELL-3:
- ESD: 20 speakers × 5 emotions × ~350 chunks
- AISHELL-3: 218 speakers 中文多说话人

数据格式需转换为 CosyVoice 3 的 `data.list` 格式 (每行: `utt_id<TAB>wav_path<TAB>text`)

## 5. 参考

- CosyVoice 3 paper: https://arxiv.org/abs/2505.17589
- Official training: examples/libritts/cosyvoice3/run.sh
- LoRA paper: https://arxiv.org/abs/2106.09685
- QLoRA paper: https://arxiv.org/abs/2305.14314
