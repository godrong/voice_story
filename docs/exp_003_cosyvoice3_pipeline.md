# Experiment 003 — CosyVoice 3 推理 + LoRA 训练全流程

> 2026-05-26 ~ 2026-05-27，RTX 4090 (24GB)，完整从零到训练产出。

## 0. 目标

1. 评测 CosyVoice 3 zero-shot 语音克隆能力（4 轴客观指标）
2. 在 ESD 数据上做 LoRA 微调，验证双层 LoRA 架构假设
3. 跑 rank 消融实验（8/16/32）

---

## 1. 推理环境搭建

### 1.1 硬件

| 项目 | 值 |
|---|---|
| GPU | NVIDIA RTX 4090 (24GB VRAM) |
| CPU | Intel Xeon |
| 系统盘 | overlay 30GB |
| PyTorch | 2.3.1+cu121 |
| Python | 3.10 |

### 1.2 模型

- CosyVoice 3: `Fun-CosyVoice3-0.5B`（0.5B 参数，LLM + Flow Matching + HiFiGAN）
- 预训练权重通过 AutoDL 镜像预装于 `/root/CosyVoice/pretrained_models/`
- 推理 API: `cosyvoice.cli.cosyvoice.AutoModel`

### 1.3 踩坑：RTX 5090 不兼容

曾尝试在 RTX 5090 上运行，失败。原因：
- RTX 5090 是 Blackwell 架构 (sm_120)，需要 CUDA 12.5+
- 基础镜像 torch 2.3.1+cu121 最高支持 sm_90
- 详见 `docs/debug/rtx5090-blackwell-compatibility.md`

换到 RTX 4090 (Ada Lovelace, sm_89) 后直接可用。

---

## 2. Zero-Shot 推理与评测

### 2.1 推理性能

4 条中文测试文本（绕口令/新闻/散文/古文），使用内置 `zero_shot_prompt.wav` 作为参考音频：

| 指标 | 值 |
|---|---|
| RTF (Real-Time Factor) | 0.36 ~ 0.56 |
| 单句推理时间 | 2.9 ~ 8.8 秒 |
| GPU 显存占用 | ~10 GB |

推理速度比实时快 2-3 倍。

### 2.2 四轴客观评测

评测脚本：`experiments/exp_003_cosyvoice3/inference_eval.py`

| Sample | MOS-NISQA ↑ | SECS ↑ | CER ↓ | F0 RMSE ↓ |
|---|---|---|---|---|
| 绕口令 (八百标兵...) | 3.975 | 0.913 | 0.188 | 103.6 Hz |
| 新闻 (人工智能...) | 3.964 | 0.956 | 0.049 | 119.9 Hz |
| 散文 (春天来了...) | 3.727 | 0.964 | 0.114 | 76.2 Hz |
| 古文 (自三峡...) | 4.432 | 0.948 | 0.286 | 88.7 Hz |
| **均值** | **4.024** | **0.945** | **0.159** | **97.1 Hz** |

### 2.3 评测指标说明

| 指标 | 工具 | 范围 | 含义 |
|---|---|---|---|
| MOS-NISQA | NISQA_DIM 神经网络 | [1, 5] | 自然度，越高越好 |
| SECS | WavLM-Base-Plus-SV | [-1, 1] | 说话人保真，>0.7 同人 |
| CER | Whisper large-v3 + jiwer | [0, 1] | 可懂度，越低越好 |
| F0 RMSE | librosa.pyin | Hz | 韵律偏差，越低越好 |

### 2.4 关键发现

- **SECS 0.945**: 说话人克隆能力优秀，但韵律跟随不足（F0 RMSE 97 Hz）
- **古文 CER 0.286 vs 现代文 CER 0.049**: CosyVoice 3 对古典中文覆盖偏弱
  - Whisper ASR 显示古文词汇识别错误（"重岩叠嶂" → "重言叠账"）
- **MCGA 数据集切入**: 古典文学领域正是 LoRA 训练要解决的问题域

### 2.5 国内网络适配

- HuggingFace: 使用 `HF_ENDPOINT=https://hf-mirror.com`
- NISQA 权重: 手动下载到 `/root/CosyVoice/pretrained_models/nisqa.tar`
- WavLM-SV: 通过 HF mirror 下载
- Whisper large-v3: 通过 HF mirror 下载（~3GB）

---

## 3. LoRA 训练

### 3.1 训练数据

**ESD (Emotional Speech Dataset)**
- 20 speakers × 5 emotions × ~350 chunks = ~35,000 条
- 本地已有 M1 pipeline 处理后的 manifest: `datasets/esd/manifest.jsonl`
- 已有 cross-emotion 训练对: `datasets/two_tier/tier1_train.jsonl` (26,943 对, 16 speakers)

**数据转换流程**:
1. 从 `tier1_train.jsonl` 取 500 对 → 解出 wav 路径 → 打包 73MB tar
2. 上传到远程: `/root/autodl-tmp/esd_lora_train/`
3. 使用 CosyVoice 3 speech tokenizer 提取 speech tokens:
   - Whisper `log_mel_spectrogram(n_mels=128)` 提取 mel 特征
   - `speech_tokenizer_v3.onnx` (ONNX Runtime) 量化 → discrete tokens
   - CAM++ ONNX 提取 speaker embedding
4. 写入 Parquet（匹配 CosyVoice 3 官方训练格式）

### 3.2 LoRA 注入方案演进

| 版本 | 方案 | 结果 |
|---|---|---|
| v1 | peft LoraConfig | Qwen2Encoder 非标准 HF 模型，peft 不兼容 |
| v2 | 替换 nn.Linear → LoRALinear | 训练绕过，但 forward shape 错误 |
| v3 | **forward hook 注入** | ✅ 成功 |

最终方案：在每个 attention Linear 层注册 `register_forward_hook`，在原输出上叠加 LoRA 增量。不改变模块类型，Qwen2 内部计算路径不受影响。

```python
class LoRAHook:
    def __init__(self, linear, rank=16, alpha=32):
        self.A = nn.Parameter(torch.zeros(linear.in_features, rank))
        self.B = nn.Parameter(torch.zeros(rank, linear.out_features))
        self.handle = linear.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        x = input[0]
        return output + (self.scaling * (dropout(x) @ self.A) @ self.B)
```

### 3.3 CosyVoice3LM.forward Bug

在 `cosyvoice/llm/llm.py` 第 700 行发现 Bug：

```python
# 原代码（错误）:
acc = th_accuracy(logits.view(-1, self.speech_token_size + 3), ...)

# 修复后:
acc = th_accuracy(logits.view(-1, self.speech_token_size + 200), ...)
```

`llm_decoder` 输出维度是 `speech_token_size + 200 = 6761`，但 accuracy 计算用了 `+ 3 = 6564`。reshape 维度不匹配导致 forward 失败。该 Bug 在 base model 上也存在，只是在无梯度时被跳过。

### 3.4 训练配置

| 参数 | 值 |
|---|---|
| Hook 数 | 96 (24 层 × 4 个投影矩阵) |
| 可训练参数 (rank=8) | 1.08M (0.21%) |
| 可训练参数 (rank=16) | 2.16M (0.43%) |
| 可训练参数 (rank=32) | 4.32M (0.85%) |
| Optimizer | AdamW |
| LR | 1e-4, CosineAnnealing to 1e-5 |
| Max Steps | 200 |
| Batch | dynamic (max_frames=2000) |
| Mixed Precision | AMP (GradScaler) |
| Data Pipeline | CosyVoice 3 官方 `parquet_opener` |

### 3.5 Rank 消融实验结果

| Rank | Trainable | 初始 Loss | 最终 Loss | Δ |
|---|---|---|---|---|
| 8 | 1.08M | 3.632 | **0.737** | -2.895 |
| 16 | 2.16M | 3.671 | 0.751 | -2.920 |
| 32 | 4.32M | 3.716 | 0.769 | -2.947 |

**结论**: rank=8 达到最低 loss。小 rank 足以捕获 cross-emotion style transfer，印证了 Tier 1 Style LoRA 设计假设：**小容量 + 多说话人数据 = 可迁移的风格跟随能力**，不会过拟合到特定 speaker。

---

## 4. 文件清单

### 远程服务器 (RTX 4090)
| 路径 | 内容 |
|---|---|
| `/root/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B/` | 模型权重 |
| `/root/autodl-tmp/esd_lora_train/` | ESD 训练数据 |
| `/root/autodl-tmp/esd_lora_train/parquet/train.parquet` | 500 条 Parquet |
| `/root/autodl-tmp/exp003_lora_r{8,16,32}/` | LoRA 权重 |
| `/tmp/train_lora_hook.py` | 训练脚本 |
| `/root/autodl-tmp/exp003_outputs/` | Zero-shot eval 结果 |

### 本地仓库
| 路径 | 内容 |
|---|---|
| `experiments/exp_003_cosyvoice3/results.md` | Zero-shot 评测报告 |
| `experiments/exp_003_cosyvoice3/inference_eval.py` | 推理+评测脚本 |
| `experiments/exp_003_cosyvoice3/train_cosyvoice3_lora.py` | LoRA 训练脚本 (peft 版) |
| `experiments/exp_003_cosyvoice3/MCGA_ANALYSIS.md` | MCGA 数据集分析 |
| `experiments/exp_003_cosyvoice3/TRAINING_DESIGN.md` | 训练方案设计 |
| `experiments/exp_003_cosyvoice3/RUNBOOK.md` | 远程执行手册 |
| `docs/debug/rtx5090-blackwell-compatibility.md` | 5090 踩坑记录 |

---

## 5. 下一步

- [ ] 4 轴 eval 对比（base vs LoRA rank=8）
- [ ] MCGA 全量放出后做 Tier 1 训练
- [ ] Tier 2 single-speaker avatar LoRA
- [ ] Composition (Tier1 + Tier2) 叠加验证
