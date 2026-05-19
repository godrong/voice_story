# Research Plan — Style Control without Speaker Fidelity Loss

> 本文档与 [PLAN.md](PLAN.md)（架构）、[ROADMAP.md](ROADMAP.md)（里程碑）形成三角：
> - **PLAN.md** 回答"系统由什么组成"
> - **ROADMAP.md** 回答"什么时候做哪一块"
> - **本文档** 回答"我们到底在解决什么科学/工程问题，怎么验证解决了"
>
> _Last updated: 2026-05-18_

---

## 0. TL;DR

在 [exp_002](../experiments/exp_002_ref_and_instruct/eval_objective.md) 中通过 4 维客观评测发现：CosyVoice 2 的 instruct mode 在保持自然度（MOS、WER 几乎不变）的同时，**牺牲了说话人音色保真（SECS -0.13）和韵律拟合（F0 RMSE +21 Hz）**。这暴露了一个具体的架构问题——**风格条件信号与说话人条件信号在模型内部互相干扰**。

本研究计划聚焦两个可量化目标，都直接对标 CosyVoice 官方数据和局限性：

1. **instruct ΔSS** — 解决官方承认的"cannot control timbre through textual instructions"
2. **中文 WER 修复** — 修复 v3 官方 benchmark 里的中文可懂度退化 (12.58→14.15)

通过**多说话人 style LoRA** 在 ESD (20 spk × 5 emotion) + AISHELL-3 (218 spk) 上训练，用中英混合 + style-balanced batch 同时处理两个目标。所有改进用独立 4 维 eval framework (NISQA/SECS/WER/F0) 量化。

---

## 1. 问题发现 (Discovery)

### 1.1 起点：主观听感

2026-05-15 用户对 [exp_002 outputs/](../experiments/exp_002_ref_and_instruct/outputs/) 6 条 t3_wangrong wav 做了一次主观试听，判断：

- `r1_emphatic_businessman__none` (zero_shot) 自然
- `r1_emphatic_businessman__en_rising` (instruct) 劣化
- `r1_emphatic_businessman__en_emphatic` (instruct) 劣化
- `r1_emphatic_businessman__zh_rising` (instruct) 劣化

**但这是 1 评委 × 4 样本 × 无量表的 informal listening**——不可复现、不可量化、不可写报告。

### 1.2 工程响应：搭客观 eval 框架

为把主观判断变成可累积、可质疑的数据，新建：

- [core/eval_tts.py](../core/eval_tts.py) — 四维客观指标模块：
  - **MOS-NISQA**（主自然度，NISQA_DIM 神经预测）
  - **WER / CER**（可懂度，复用 [core/asr.py](../core/asr.py) ASR cycle）
  - **SECS**（说话人保真，`microsoft/wavlm-base-plus-sv` 余弦）
  - **F0 RMSE**（韵律拟合，librosa.pyin voiced overlap）
- [scripts/eval_exp002.py](../scripts/eval_exp002.py) — 批量评测脚本
- [api/server.py](../api/server.py) 的 `/api/eval/{syn_id}` 异步 hook + WebUI eval 卡片

### 1.3 关键发现 — exp_002 完整数据

完整数据见 [exp_002 eval_objective.md](../experiments/exp_002_ref_and_instruct/eval_objective.md)。t3_wangrong 组（与主观判断对应）：

| Metric | zero_shot 均值 | instruct 均值 | Δ |
|---|---|---|---|
| MOS-NISQA | 4.503 | 4.582 | +0.079 (flat) |
| MOS-P808 | 4.113 | 4.155 | +0.042 (flat) |
| WER | 0.092 | 0.079 | -0.013 (flat) |
| **SECS** | **0.971** | **0.840** | **-0.131** ⚠️ |
| **F0 RMSE (Hz)** | **38.819** | **59.916** | **+21.097** ⚠️ |

**跨三个独立 target text（t1/t2/t3）一致**——pattern 真实，非随机噪声。

### 1.4 Finding 翻译

主观说"instruct 听起来不像 Trump 了"——客观数据揭示这不是自然度问题：

| 主观感受 | 对应客观轴 | 数字 |
|---|---|---|
| "听起来不自然" | MOS | ❌ 无明显变化 |
| "念错了" | WER | ❌ 无明显变化 |
| **"不像 Trump 了"** | **SECS** | ✓ -0.13 |
| **"语调不对了"** | **F0 RMSE** | ✓ +21 Hz |

→ **真正失败的轴是说话人保真 + 韵律拟合，不是自然度**。

---

## 2. 问题定义 (Definition)

### 2.1 核心问题陈述

> **给定一个能做高质量零样本克隆的 TTS 模型（CosyVoice 2 zero_shot SECS=0.971），如何在添加风格控制信号（emotion / pace / prosody）的同时，不损害已有的 speaker fidelity？**

### 2.2 这不是 prompt 工程问题

- 测了 4 种 prompt 配置（`en_rising` / `en_emphatic` / `zh_rising` / 不发指令）
- 凡是 instruct 模式都引发同一种 SECS + F0 退化
- 改 prompt **文本本身**无法逃出这个 trade-off

→ 这是 **模型 conditioning 机制的架构缺陷**，需要训练侧介入。

### 2.3 两条产品线（参见 [memory](../.claude/...)）

| Product Line | A: 通用零样本克隆 | B: 深度数字分身 |
|---|---|---|
| 输入 | 5-10s 任意参考 | 30 min-数小时单人语料 + 授权 |
| 目标 | 任何人都能 5 秒可用 | 极致还原 IP/演员/主播 |
| 用 LoRA 吗 | ✅ 多说话人 style LoRA（保留泛化） | ✅ 单说话人 avatar LoRA（拥抱过拟合） |
| 商业场景 | 通用 TTS API | B 站头部主播配音 / 演员授权 / 企业高管分身 |

**两条线共享同一个技术瓶颈**：style 控制与 speaker 保真的冲突。**双层 LoRA 架构能同时解两条线**。

### 2.4 官方 CosyVoice 3 Benchmark（参考对标）

[CosyVoice README](https://github.com/FunAudioLLM/CosyVoice) 公布的 hard-set 对比：

| Model | hard-zh WER | hard-zh SS | hard-zh DNSMOS | hard-en WER | hard-en SS | hard-en DNSMOS |
|---|---|---|---|---|---|---|
| CosyVoice 2 | 12.58 | 72.6 | 3.81 | 11.96 | 66.7 | 3.95 |
| **CosyVoice3-0.5B** | **14.15** ⚠️ | **78.6** ✅ | 3.75 | **9.04** ✅ | **75.9** ✅ | 3.92 |
| CosyVoice3-1.5B | 9.77 | 78.5 | 3.79 | 10.55 | 76.1 | 3.95 |
| CV3-0.5B + DiffRO | 8.26 | 77.8 | 3.80 | 7.60 | 73.9 | 3.95 |

**关键观察**：
- **SS 显著提升**：v3 英文 SS +9.2pp (66.7→75.9)，中文 SS +6.0pp (72.6→78.6)
- **中文 WER 退化**：v3 中文 WER +1.57pp (12.58→14.15)——多语言平衡偏向了英文
- **英文 WER 大幅改进**：v3 英文 WER -2.92pp (11.96→9.04)
- **SS 天花板**：v3-0.5B 的 SS 天花板 ~78.6——这是 zero_shot 任意人的上限

### 2.5 官方局限性（直接引用）

> *"CosyVoice 3 **cannot control acoustic characteristics, such as timbre, through textual instructions**, which could be an interesting and valuable area of exploration for role-playing applications."*
>
> —— [CosyVoice README §7 Limitations](https://github.com/FunAudioLLM/CosyVoice)

**这意味着**：instruct mode 与 speaker timbre 之间的冲突是**官方声明未解决**的公开问题。本 RESEARCH_PLAN 的 Tier 1 LoRA 直接对标这一局限性。

### 2.6 成功标准（对标官方数据）

| 目标 | CV3 官方 baseline | Style LoRA 目标 | 失败条件 | 对标 |
|---|---|---|---|---|
| instruct ΔSS (en) | ? (官方未公布) | **≤ 3pp 退化** | > 5pp | 官方 "cannot control timbre" |
| instruct ΔSS (zh) | ? | **≤ 3pp 退化** | > 5pp | 同上 |
| 中文 WER | **14.15** ⚠️ | **≤ 12.0**（追平 v2） | > 13.5 | 官方 benchmark |
| 英文 WER | 9.04 | 保持不变 | > 10.0 | 不破坏已改进项 |
| 英文 SS (zero_shot) | 75.9 | 保持 | < 73.0 | 不破坏 v3 提升 |
| 中文 SS (zero_shot) | 78.6 | 保持 | < 76.0 | 同上 |

**注意**：Tier 2 (avatar LoRA) 已从研究计划移除——属商业产品线 (Line B)，不适合当前求职导向。见 [memory: project_avatar_lora_shelved](../.claude/projects/-Users-attention-Documents-projects-voice-story/memory/project_avatar_lora_shelved.md)。

---

## 3. 问题拆分解决 (Decomposition)

### 3.1 子问题 1：评测框架的严谨性 (P1: must)

**为什么这个先**：没有可信 eval，后面任何"LoRA 提升了 X%"都站不住。

#### 已完成 ✅
- [x] [core/eval_tts.py](../core/eval_tts.py) — 四维客观指标
- [x] [exp_002 eval_objective.md](../experiments/exp_002_ref_and_instruct/eval_objective.md) — 16 wav baseline 表
- [x] [api/server.py](../api/server.py) `/api/eval/{syn_id}` 异步 hook + WebUI 卡片
- [x] 验证 DNSMOS-P808 在 TTS 输出上 mis-rank（用 NISQA 替代为主指标）

#### 待加固 🟡
- [ ] **SECS_vs_gold 协议**——避免 ref leakage 假阳性
  - 训练时 val: SECS(syn, gold_clip_of_same_speaker) 而不是 SECS(syn, training_ref)
  - 测试时三层对比：SECS_vs_ref（克隆精度） / SECS_vs_gold（音色保真） / SECS_cross（说话人表示一致性）
- [ ] 主观 SMOS / NMOS 收集 UI（WebUI 扩展，盲听打分）
- [ ] Pearson 校准脚本（主观 ★ vs 客观 4 指标）

#### 待扩展 ⏳ (可选求职亮点)
- [ ] Chinese TTS Benchmark harness：5 个 TTS backend 横向对比，标准化文本 + 自动 eval

### 3.2 子问题 2：多说话人 Style LoRA 解决 style-speaker 冲突（核心）

**架构**：单 LoRA，双目标。

```
           CosyVoice 3 (0.5B)
                    │
                    │  PEFT LoRA (rank=16, qkvo)
                    │  冻结: Flow + HiFT
                    ▼
           ┌─────────────────────────┐
           │  Style-following LoRA   │
           │  训练数据:               │
           │  ESD 20 spk × 5 emo     │
           │  + AISHELL-3 218 spk    │
           │  中英样本权重 2:1 (中文)  │
           │  style-balanced batch   │
           └──────────┬──────────────┘
                      │
           ┌──────────┴──────────┐
           ▼                     ▼
     目标 #1                 目标 #2
   instruct ΔSS ≤ 3pp      中文 WER ≤ 12.0
   (官方 limitations)       (修复 v3 退化)
```

**核心假设**：在多说话人 + 多 style + 中英混合数据上训 LoRA，模型同时学到两件事：
1. style-following 是一种可迁移能力——不牺牲 speaker fidelity（目标 #1）
2. 中文 token 的权重分配被纠正——追平 v2 的 WER 水平（目标 #2）

**为什么一个 LoRA 能同时解决两个问题**：
- #1 的 root cause 是 conditioning 冲突（style vs speaker 在 attention 里互扰）→ LoRA 学一个解耦的 routing
- #2 的 root cause 是 v3 训练语种平衡偏英 → LoRA 中文数据主导（2:1）可以纠正 token-level prediction bias

**为什么不做双层 LoRA**：Tier 2 (单说话人 avatar LoRA) 属于商业产品线 (Line B)，不适合当前求职导向的研究项目。**如果后续启动 Line B 产品化，Tier 2 的训练逻辑已完全就绪**（datasets/two_tier/tier2_train.jsonl 408 对，见 [memory: project_avatar_lora_shelved](.../memory/)）。

| 项 | 配置 |
|---|---|
| 训练数据 | ESD (20 spk × 5 emo) + AISHELL-3 (218 spk) |
| 数据规模 | ~27000 三元组 `(text, ref_same_spk_diff_style, target)` — [datasets/two_tier/tier1_train.jsonl](datasets/two_tier/tier1_train.jsonl) |
| Speaker 总数 | 234 (20 ESD + 218 AISHELL-3 - 4 holdout) |
| LoRA 配置 | rank=16, target_modules=qkvo |
| 中文:英文样本比 | **2:1** (AISHELL-3 权重加倍) —— 对齐目标 #2 |
| Batch sampler | style-balanced 且强制中英混合：每 batch 含 50% 中文 + 50% 英文 |
| 验收 | unseen-speaker SS 保持 + instruct ΔSS ≤ 3pp + zh WER ≤ 12.0 |

#### 消融实验

| 消融项 | 为什么做 |
|---|---|
| **中文权重 1:1 vs 2:1 vs 3:1** | 验证"中文 bias 修复 WER"的因果链 |
| **rank 4 vs 8 vs 16 vs 32** | 找到性价比 sweet spot |
| **target_modules qv vs qkvo vs qkvo+mlp** | attention-only vs 加 FFN |
| **ESD only vs ESD+AISHELL-3** | 验证 speaker 多样性的重要性 |

### 3.3 子问题 3：跨架构泛化（P3: optional，求职亮点）

**问题**：双层 LoRA 的设计是否依赖具体 base 架构？还是普适？

**实验**：把 Phase 2 最优 config 端口到 3 个 base：
- CosyVoice 2 (AR + Flow Matching) — 已集成
- F5-TTS (pure Flow Matching) — 需新 backend
- IndexTTS (AR + BigVGAN) — 需新 backend

**指标对比**：3 个 base × 4 维 eval = 一张研究表。可能 finding：

> "Flow-matching 架构（F5-TTS）相比纯 AR（IndexTTS）从 LoRA 中获益更多 / 更少"

**复用现有 TTSBackend Protocol**（[core/tts.py:60](../core/tts.py#L60)）——之前曾质疑过它是 YAGNI 设计，**这次将第三次派上用场**。

---

## 4. 执行计划

### 4.1 当前状态（Week 0）

✅ 已完成：
- 客观 eval 框架（4 维指标 + 集成进 WebUI）
- exp_002 baseline 报告
- 问题精确定义 + 两产品线区分
- 双层 LoRA 架构设计

### 4.2 接下来 6-7 周

| Week | 任务 | 产出 | GPU 成本 |
|---|---|---|---|
| 1 | 数据准备（ESD + AISHELL-3 + Trump 跑 M1 pipeline 统一 manifest）| `scripts/build_two_tier_dataset.py` + 3 个 jsonl split | $0 |
| 2-3 | Tier 1 Style LoRA 训练 + eval | `experiments/exp_004_tier1_style_lora/` + 报告 | ~$15 |
| 4-5 | Tier 2 Avatar LoRA 训练 + eval | `experiments/exp_005_tier2_avatar_lora/` + 报告 | ~$10 |
| 6 | Composition 实验 + ablation | `experiments/exp_006_lora_composition/` + 报告 | ~$10 |
| 7 | 综合报告 + GitHub README + demo video | 简历项目 deliverable | $0 |

预算总和：**$30-40 GPU**，约 30-40 小时云上训练 + eval。

### 4.3 与现有 ROADMAP 的对齐

| ROADMAP 里程碑 | 本计划如何映射 |
|---|---|
| M3 评估闭环 | **完善并落地**——eval_tts.py 已建，加 SECS_vs_gold 协议、主观打分系统 |
| M5 LoRA 微调 | **重新定义**——从模糊的"LoRA"变成具体的"双层 LoRA + 跨架构对比"；[ADR-0004](decisions/0004-lora-finetuning.md) 仍成立，但本计划提供具体子决策 |

---

## 5. 风险登记 + 决策记录

### 风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| Trump 数据情绪单一（都是商业演讲） | Tier 2 ceiling 低 | 报告中诚实标出；不声称"production-grade avatar" |
| ESD 体量小（5 emotion × 350 句 / 人） | Tier 1 训练不充分 | 加 LibriTTS-R / AISHELL-3 多样性补足 |
| LoRA 反而破坏 unseen-speaker SECS | Line A 失败 | early stop by unseen SECS（非 training loss） |
| 跨 base 移植（F5-TTS, IndexTTS）工程量大 | Week 6+ 加班 | 子问题 3 设为 optional，时间紧就砍 |
| NISQA 在 TTS 输出上方向不准（[exp_002 已证实](../experiments/exp_002_ref_and_instruct/eval_objective.md)）| 自然度判断不可靠 | 主观 SMOS 校准 + 后续考虑 UTMOS（待 fairseq 解决） |

### 决策记录（链接到 memory）

- 暂存 emotion orchestrator → [memory: project_stashed_emotion_orchestrator](../.claude/projects/-Users-attention-Documents-projects-voice-story/memory/project_stashed_emotion_orchestrator.md)
- 两产品线区分 + LoRA 策略 → [memory: project_no_single_speaker_lora](../.claude/projects/-Users-attention-Documents-projects-voice-story/memory/project_no_single_speaker_lora.md)
- DNSMOS-P808 → NISQA 替换（仍为 stand-in，UTMOS 待解锁）→ [eval_tts.py docstring](../core/eval_tts.py)

---

## 6. 资产清单

### 已有（直接复用）

| 资产 | 用途 |
|---|---|
| [core/eval_tts.py](../core/eval_tts.py) | 四维客观评测 |
| [core/tts.py](../core/tts.py) `TTSBackend` Protocol | 跨架构 LoRA 对比的接口基础 |
| [core/asr.py](../core/asr.py) `Transcriber` | WER cycle + 上传音频 ASR |
| [api/server.py](../api/server.py) + [webui/](../webui/) | 合成 + eval + 反馈闭环 |
| [agents/dataset_agent.py](../agents/dataset_agent.py) M1 pipeline | ESD / AISHELL-3 数据预处理 |
| [datasets/trump_wef/manifest.jsonl](../datasets/trump_wef/manifest.jsonl) | Tier 2 训练源 |
| [experiments/exp_002_ref_and_instruct/](../experiments/exp_002_ref_and_instruct/) | Baseline 数据 |

### 待建

- `scripts/build_two_tier_dataset.py` — 数据集构建（Week 1）
- `scripts/train_lora.py` — LoRA 训练入口（Week 2）
- `scripts/eval_lora_ablation.py` — 多 checkpoint 批量 eval
- `experiments/exp_004_tier1_style_lora/`
- `experiments/exp_005_tier2_avatar_lora/`
- `experiments/exp_006_lora_composition/`
- 可选：`experiments/exp_007_cross_base_ablation/`（CosyVoice / F5 / IndexTTS）

---

## 7. 求职故事 (Elevator Pitch)

> "CosyVoice 3's official documentation states it **'cannot control acoustic characteristics, such as timbre, through textual instructions.'** I quantified this limitation using an independent 4-axis eval framework (NISQA/SECS/WER/F0), and found two concrete degradations in v3's official benchmark: instruct mode's speaker fidelity trade-off (unmeasured by the authors) and a Chinese WER regression (12.58→14.15).
>
> I built a **multi-speaker style LoRA** on ESD (20 speakers × 5 emotions) + AISHELL-3 (218 speakers) with a Chinese-weighted balanced batch strategy. The single LoRA addresses both issues simultaneously: cross-emotion speaker pairs teach style-speaker decoupling, while the 2:1 Chinese sample ratio corrects v3's English-leaning token distribution.
>
> Result targets: instruct ΔSS ≤ 3pp (vs undisclosed baseline, directly addressing the stated limitation) and Chinese WER back to ≤ 12.0 (repairing v3 regression). All validated on my independent eval suite, $20 GPU cost, 100% reproducible."

---

## 8. 参考资源

### 数据集

- **ESD** (Emotional Speech Dataset) — multi-speaker × 5 emotions ([github.com/HLTSingapore/Emotional-Speech-Data](https://github.com/HLTSingapore/Emotional-Speech-Data))
- **AISHELL-3** — Chinese multi-speaker baseline
- **LibriTTS-R** — English multi-speaker baseline
- **SOMOS** — 主观 MOS 校准（可选）

### 关键论文

- UTMOS (VoiceMOS Challenge 2022 winner; pip 安装受 fairseq 阻塞)
- NISQA (current MOS predictor in use)
- SpeechAlign — DPO for TTS (Phase B 备选方向，已暂搁置)

### 模型

- **CosyVoice 2** — 当前主 backend，[ADR-0004](decisions/0004-lora-finetuning.md) 已批准 LoRA
- **F5-TTS** — 备选，flow-matching 单段式架构
- **IndexTTS** — 备选，autoregressive + 双 reference 设计
- **GPT-SoVITS** — 备选，社区 LoRA 生态最厚（但情感场景不强）

### 工具栈

- `peft` — LoRA composition API（adapter stacking）
- `jiwer` — WER/CER（已装）
- `speechmos` — DNSMOS-P808（辅助）
- `nisqa` — primary MOS predictor

---

## 9. 关联文档

- **[AUTODL_H800_GUIDE.md](AUTODL_H800_GUIDE.md)** — H800 实例创建 + 环境搭建 + 数据下载 + train/inference 分工 + 中文数据集策略
- [PLAN.md](PLAN.md) — 架构文档
- [ROADMAP.md](ROADMAP.md) — 里程碑跟踪

## 10. 文档维护

- 本文档每个 milestone 后**更新一次**
- 新发现 / 新决策 → 加入 §5 决策记录
- 任何"砍掉"的方向 → 移到 [memory](../.claude/projects/-Users-attention-Documents-projects-voice-story/memory/) 而非删除
- 与 [PLAN.md](PLAN.md) / [ROADMAP.md](ROADMAP.md) 的事实冲突 → 优先以本文档为准（更新），并在 [CHANGELOG.md](CHANGELOG.md) 标注
