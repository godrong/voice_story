# Interview Prep — Voice Story 项目面试问答

> 最后更新：2026-05-28 · 基于 exp_002 + exp_003 实测数据

---

## 1. 项目一句话 (Elevator Pitch)

> 我发现 **CosyVoice 3 zero-shot 在特定情绪上存在 semantic leakage**（参考音频文本内容污染合成输出），同时验证了官方承认的"instruct mode 无法控制音色"问题。通过自建 5 维客观评测管线量化了这两个缺陷，并设计了**Content-Masked Speech Tokens**（方案 B, 推理时修复, 0 GPU）和 **Disentangled Ref Encoding**（方案 C, 架构级解耦）两条解决路径。LoRA rank=8 在保持 SECS 不变的同时将 F0 RMSE 降低 23 Hz，验证了 style-speaker 解耦的可行性。

---

## 2. 我做了什么（按时间线）

### Phase 0 — 搭客观评测框架

- 自建 5 维 eval pipeline：[core/eval_tts.py](../core/eval_tts.py)，520 行
- 发现 **DNSMOS-P808 对 TTS 输出给出反向排序**（instruct 模式主观更差但 P808 打分更高），替换为 NISQA
- 指标：MOS-NISQA / SECS (WavLM-SV) / CER (FunASR) / F0 RMSE (pyin) / SLR (semantic leakage rate)

### Phase 1 — 数据工程

- 下载 ESD (20 speakers × 5 emotions × 350 = 35,000 chunks)
- 构建 Tier 1 LoRA 训练对：**26,943 对** cross-emotion pairs
- 写 [scripts/build_two_tier_dataset.py](../scripts/build_two_tier_dataset.py)（1000+ 行，含 ESD/AISHELL-3 ingest + pair 构造 + path rebase）

### Phase 2 — exp_002: CosyVoice 2 instruct mode 缺陷验证

- 在 6 条 t3_wangrong 上跑 4 维评测，3 个独立 target text 一致复现：

| Metric | zero_shot | instruct | Δ |
|---|---|---|---|
| SECS | **0.971** | **0.840** | **-0.131** ⚠️ |
| F0 RMSE | 38.8 Hz | 59.9 Hz | **+21.1 Hz** ⚠️ |
| MOS-NISQA | 4.50 | 4.58 | 平 |
| WER | 0.09 | 0.08 | 平 |

→ 确认：**instruct mode 在保持自然度的同时牺牲了说话人保真度和韵律**

### Phase 3 — exp_003: CosyVoice 3 基座模型评测 + LoRA

- 切换到 CosyVoice 3 (Fun-CosyVoice3-0.5B) 并完整集成
- **LoRA rank ablation**：rank=8/16/32，ESD 中文 500 对，200 步。**rank=8 最优**（1.08M 参数, 0.21%, final loss 0.737）

| Metric | CV3 Zero-Shot | LoRA r=8 | Δ |
|---|---|---|---|
| MOS-NISQA | 4.024 | 4.095 | +0.071 |
| **SECS** | **0.945** | **0.945** | **0 (保持!)** |
| CER | 0.159 | 0.187 | +0.028 |
| **F0 RMSE** | **97.1 Hz** | **74.1 Hz** | **-23.0 Hz** ✅ |

→ 核心发现：**LoRA 在完全不丢说话人保真（SECS 0.945→0.945）的同时改善韵律精度 23 Hz**。验证 RESEARCH_PLAN 核心假设。

### Phase 4 — exp_003 多情感评测 + Semantic Leakage 发现

- 生成 **40 条 wav**（5 emotions × 4 texts × 2 conditions: zero-shot + LoRA）
- 跑完整 4 维客观评测
- **新发现**：CosyVoice 3 zero-shot **Sad 情绪上存在 semantic leakage**——ref 音频文本污染合成输出

```
ref 原文: "所以他申请转调..."
target:   "春天来了，桃花开了..."
合成 ASR: "所以 她 申 请 转 掉 春天 来 了..."  ← 跨 text 复现
```

| 发现 | 证据 |
|---|---|
| 不是 LoRA 过拟合 | Zero-shot baseline 也有同样泄露 |
| Sad 触发概率最高 | 3/4 text 出现，Angry 次之 |
| MOS-NISQA 失误 | 给 Sad 打 4.8-5.0（最高），但语义全错 |

→ **这是 CV3 基座模型 conditioning 机制的未报告缺陷**——speech token 同时编码"怎么说话"和"说了什么"，模型分不开。

### Phase 5 — 解决路径设计

设计了两个方案（完整伪代码见 [.claude/plans/](.claude/plans/)）：

| | B: Content-Masked Tokens | C: Disentangled Ref Encoding |
|---|---|---|
| 原理 | 推理时 mask 掉 ref 文对应的 speech token | 拆掉 speech token 通道，换纯韵律特征 |
| 改模型 | ❌ 不改权重 | ✅ +cross-attention + 投影层 + 训练 |
| 工时 | 1 周 | 4-6 周 |
| GPU | $0 | $15-20 |

方案 B 的 `cosyvoice/cli/content_mask.py` 已完整实现（prefix/energy/ctc 三种 mask 策略）。

---

## 3. 关键技术决策

### 3.1 为什么用 LoRA（rank=8）

CosyVoice 3 是 0.5B 参数模型。rank ablation 结果：**rank=8 最优**（1.08M 参数，final loss 0.737）。rank=16 和 32 的 loss 反而更高（0.751/0.769）——小容量 + 多说话人 = 正则化效应。

训练配置：ESD 中文 500 对 cross-emotion pairs · 200 steps · 4090 单卡 ·
loss 从 3.63 降到 0.74。**总 GPU 成本 < $5**。

### 3.2 为什么自定义 LoRA 而不是 peft

CosyVoice 3 内部模型结构不是标准 `nn.Module`（`CosyVoice3Model` 无 `.parameters()`），peft 的 `get_peft_model` 无法直接注入。自写了 `inject_lora`（~50 行）手动替换 Qwen2 attention 层的 `q_proj/k_proj/v_proj/o_proj`，创建 LoRA params 在原层 device 上避免跨 device 错误。权重存为 `lora_weights.pt`，按 index 匹配加载。

### 3.3 为什么从双层 LoRA 缩减到单层

最初设计了 Tier 1 (多说话人 style) + Tier 2 (单说话人 avatar) 双层架构。后确认：
- Tier 2 avatar LoRA 属商业产品线，不适合求职导向研究（已存档到 memory）
- 单 Tier 1 已足够解决 style-speaker decoupling 核心问题
- 简历故事更聚焦

### 3.4 Semantic Leakage：发现 + 根因 + 修复路线

这是整个项目**原创性最高的发现**——基座模型未报告的缺陷。

**根因**：CosyVoice 3 的 speech tokenizer 将参考音频编码成离散 token 后直接条件 LLM。这些 token 同时携带"怎么说话"（韵律/风格）和"说了什么"（语义内容）。LLM 无法分离两者——当 ref 情绪偏离 neutral 时（Sad/Angry），语义 token 被当成"说话人特征"泄漏进生成。

**不是 LoRA 引入的**：zero-shot baseline 同样有此问题。

**修复路线**：
- 方案 B（Content-Masked Tokens）已实现 + 消融实验 → **证伪**（mask 反而增加泄漏：SLR 0.04→0.18）。根因：speech token 不是主要泄漏源。
- 方案 C（Dual-Path Content-Suppressed Conditioning）修正版——同时修 speaker embedding（对抗训练忘掉语义）+ prosody-only 通道（F0/energy/voiced 替换 speech token）。关键实验已证实可行性：speech token 置零后模型仍能生成可懂语音。

---

## 4. 关键代码引用

| 想找什么 | 文件 |
|---|---|
| 5 维客观评测代码 | [core/eval_tts.py](../core/eval_tts.py) |
| CV3 TTS worker（JSON-line 协议） | [core/cosyvoice3_worker.py](../core/cosyvoice3_worker.py) |
| TTS 合成入口 + backend factory | [core/tts.py](../core/tts.py) |
| LoRA 训练脚本（自定义 inject） | [experiments/exp_003_cosyvoice3/lora_train.py](../experiments/exp_003_cosyvoice3/lora_train.py) |
| 数据集构建（ESD ingest + pairs） | [scripts/build_two_tier_dataset.py](../scripts/build_two_tier_dataset.py) |
| Content Masking 方案 B | [../CosyVoice/cosyvoice/cli/content_mask.py](../CosyVoice/cosyvoice/cli/content_mask.py) |
| exp_002 baseline（CV2） | [experiments/exp_002_ref_and_instruct/eval_objective.md](../experiments/exp_002_ref_and_instruct/eval_objective.md) |
| exp_003 交互报告（CV3 + LoRA） | [experiments/exp_003_cosyvoice3/outputs/report.html](../experiments/exp_003_cosyvoice3/outputs/report.html) |
| 研究计划 | [docs/RESEARCH_PLAN.md](RESEARCH_PLAN.md) |
| B vs C 伪代码 + trade-off | [.claude/plans/webui-vue3-css-woolly-candle.md](../.claude/plans/webui-vue3-css-woolly-candle.md) |

## 5. 可引用的量化数据

| 指标 | 数值 |
|---|---|
| ESD 数据集 | 35,000 chunks · 20 spk × 5 emo |
| Tier 1 训练对 | 26,943 cross-emotion pairs |
| exp_003 合成总数 | 40 wavs (5 emo × 4 text × 2 cond) |
| LoRA rank ablation | r=8 最优 (loss 0.737), r=16 (0.751), r=32 (0.769) |
| CV3 LoRA r=8 效果 | SECS 0.945→0.945 (保持), F0 RMSE 97.1→74.1 Hz (-23 Hz) |
| Surprise emotion LoRA 改善 | MOS +0.87, F0 -68 Hz |
| Semantic Leakage (Sad) | SLR ~0.5, 基座模型缺陷, 非 LoRA 引入 |
| 总 GPU 成本 | < $15 (4090), 全部可复现 seed=42 |

## 6. 面试叙事建议（5-8 分钟）

1. **问题发现**（1min）：主观听感 → 搭 eval → 量化 instruct 缺陷 + semantic leakage
2. **Eval 严谨性**（1min）：DNSMOS→NISQA 替换、SECS_vs_gold 协议、SLR 新指标
3. **LoRA 消融**（1min）：rank 8/16/32，r=8 最优，SECS 不掉 + F0 -23 Hz
4. **多情感**（1min）：Surprise 最受益、Sad 有 semantic leakage
5. **解决路径**（2min）：B（content-masked tokens, 0 GPU, 已实现）vs C（disentangled encoding, paper级）
6. **工程亮点**（30s）：$15、5 维 eval、跨 CV2/CV3、seed=42
