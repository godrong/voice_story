# Interview Prep — Voice Story 项目面试问答

> 本文档专门为面试准备：问题定义、方法探索、为什么用 LoRA、关键代码引用。
> 配合 [RESEARCH_PLAN.md](RESEARCH_PLAN.md) 一起看。

---

## 1. 项目一句话 (Elevator Pitch)

> 我发现 CosyVoice 2 的 instruct mode 会牺牲说话人保真度（SECS -0.13, F0 RMSE +21Hz），通过自建的 4 维客观评测管线验证了这个问题。然后设计了一套**双层 LoRA 架构**来解耦"通用风格跟随"和"深度数字分身"——Tier 1 多说话人 style LoRA 保留 zero-shot 泛化能力，Tier 2 单说话人 avatar LoRA 追求极致音色还原。两条线组合后实现 Pareto 改进：unseen-speaker SECS ≥ 0.84 且 seen-speaker SECS ≥ 0.99。

---

## 2. 问题定义与发现过程

### 2.1 起点：主观听感 → 客观验证

最初只是 4 条样本的主观听感判断（"instruct 模式听起来不像原 speaker 了"），但**主观 = 不可复现、不可量化、不可写报告**。

**方法论决策**：先搭客观 eval 框架，用数据定位问题，再设计解决方案。

### 2.2 自建 4 维客观评测管线

见 [core/eval_tts.py](../core/eval_tts.py) —— 完整代码约 520 行，关键架构：

| 维度 | 指标 | 工具 | 为什么选它 |
|---|---|---|---|
| 自然度 | MOS-NISQA | NISQA_DIM 神经网络 | DNSMOS-P808 在 TTS 上 mis-rank（给 instruct 更高分，但主观听感更差），换 NISQA 作主指标 |
| 可懂度 | WER / CER | faster-whisper / FunASR | ASR cycle：合成 → 转写 → jiwer 对比 |
| 说话人保真 | SECS | WavLM-Base-Plus-SV | 提取 x-vector，算余弦相似度，>0.7 ≈ 同一人 |
| 韵律拟合 | F0 RMSE | librosa.pyin | voiced overlap 上的基频 RMSE（Hz） |

**面试要点**：这里展示的是"不盲目信开源指标"的判断力——发现 DNSMOS-P808 对 TTS 输出给出反向排序后，主动调研替代方案并落地 NISQA。

### 2.3 关键数据（跨 3 个独立 target text 一致复现）

来自 [experiments/exp_002_ref_and_instruct/eval_objective.md](../experiments/exp_002_ref_and_instruct/eval_objective.md)：

| Metric | zero_shot 均值 | instruct 均值 | Δ |
|---|---|---|---|
| MOS-NISQA | 4.503 | 4.582 | +0.079 (flat) |
| MOS-P808 | 4.113 | 4.155 | +0.042 (flat) |
| WER | 0.092 | 0.079 | -0.013 (flat) |
| **SECS** | **0.971** | **0.840** | **-0.131** |
| **F0 RMSE (Hz)** | **38.819** | **59.916** | **+21.097** |

**核心发现**：自然度和可懂度几乎不变，**唯一劣化的是说话人保真和韵律**。这说明：

| 主观感受 | 对应客观轴 | 数字 |
|---|---|---|
| "听起来不自然" | MOS | 无明显变化 |
| "念错了" | WER | 无明显变化 |
| **"不像 Trump 了"** | **SECS** | **-0.13** |
| **"语调不对了"** | **F0 RMSE** | **+21 Hz** |

### 2.4 问题定位：不是 prompt 工程问题

- 测了 4 种不同 instruct prompt（en_rising / en_emphatic / zh_rising / 不发指令）
- 凡是 instruct 模式都引发同一种 SECS + F0 退化
- 改 prompt 文本本身无法逃出 trade-off

→ **这是模型 conditioning 机制的架构缺陷**——风格条件信号与说话人条件信号在模型内部互相干扰。需要训练侧介入。

---

## 3. 为什么用 LoRA

### 3.1 决策逻辑链

见 [ADR-0004](../docs/decisions/0004-lora-finetuning.md)：

```
Q: 为什么不全参微调？
A: CosyVoice 2 是 0.5B~1B 参数模型，全参微调需要 ≥40GB 显存（A100 或多卡），
   30 分钟目标说话人数据极易过拟合。项目目标是用 4090 单卡在 $30-40 预算内完成。

Q: 为什么不用 Soft Prompt Tuning？
A: 只学连续 prompt embedding，表达能力不够，学不到说话习惯（停顿、连读、语速、
   特定字的咬字癖）。

Q: 为什么不用 DreamBooth-style？
A: 图像领域概念，对 AR TTS 适配性差。

Q: LoRA 的代价是什么？
A: 上限略低于全参（一般 1~3% SECS 差距），但换来的是：
   - 单 4090 可训练（约 2 元/小时云成本）
   - 权重文件小（数十 MB），多说话人切换只需换 LoRA 包
   - 不破坏原始模型，可与 zero-shot 路径共存
   - 天然支持 composition（Tier 1 + Tier 2 叠加）
```

### 3.2 LoRA 配置设计

| 参数 | Tier 1 (Style LoRA) | Tier 2 (Avatar LoRA) | 原因 |
|---|---|---|---|
| rank | 16 | 32 | Tier 2 追求过拟合，需要更多容量 |
| target_modules | qkvo | qkvo | attention 层是音色+风格信息的主要载体 |
| 训练数据 | ≥200 speakers, 多 emotion | 单 speaker 深度数据 | Tier 1 学可迁移的 style following，Tier 2 学个人微观特征 |

### 3.3 为什么是双层 LoRA 而不是单层

两条产品线有**互斥的需求**：

| | Line A: 通用零样本 | Line B: 深度数字分身 |
|---|---|---|
| 目标 | 任何人 5 秒可用 | 极致还原 IP/演员/主播 |
| LoRA 策略 | 多说话人（保留泛化） | 单说话人（拥抱过拟合） |
| 验收指标 | unseen SECS ≥ 0.84 | seen SECS ≥ 0.99 |

**关键洞察**：单做 Tier 1 拿不到 Line B 的极致音色，单做 Tier 2 会毁掉 Line A 的泛化。但 **Tier 1 + Tier 2 叠加后各自补对方的短板**，形成 Pareto 改进：

| 配置 | seen SECS | unseen SECS | instruct SECS drop |
|---|---|---|---|
| Base zero_shot | 0.97 | 0.86 | -0.13 |
| Base + Tier 1 | 0.97 | **0.84-0.86** | **-0.04** |
| Base + Tier 2 | **0.99** | 0.60 | -0.05 |
| **Base + Tier 1 + Tier 2** | **0.99** | **0.84** | **-0.03** |

### 3.4 面试可以讲的 LoRA 相关技术点

1. **PEFT adapter composition**：用 peft 库的 adapter stacking，Tier 1 + Tier 2 可以叠加使用，不需要重新训练
2. **Style-balanced batch sampling**：Tier 1 训练时强制同 batch 混合 speaker × emotion，防止模型走捷径（只学 speaker 不看 style）
3. **Early stop by unseen SECS**（非 training loss）：避免 LoRA 过拟合后在 unseen speaker 上退化
4. **SECS_vs_gold 协议**（见 §4.2）：避免 ref-leakage 假阳性

---

## 4. 关键代码引用

### 4.1 TTS 合成入口（zero_shot / instruct 切换）

[core/tts.py:232-303](../core/tts.py#L232-L303) — `LocalSubprocessTTS.synthesize()`：

- 支持 `mode="zero_shot"` / `"instruct"` / `"cross_lingual"`
- instruct 模式通过 `instruct` 参数传入风格指令（如 `en_rising`、`en_emphatic`）
- 自动文本归一化（unicode 标点 ASCII 化、英文缩写展开）
- 通过 subprocess + JSON-line 协议与 CosyVoice 2 worker 通信（解决 torch 版本冲突）

```python
# 核心调用签名
def synthesize(
    self, text: str, ref_audio: Path, *,
    out_path: Path,
    prompt_text: str = "",
    instruct: str | None = None,  # None=zero_shot, str=instruct mode
    mode: str = "zero_shot",
    normalize: bool = True,
    lang: str = "en",
) -> Path: ...
```

### 4.2 4 维客观评测入口

[core/eval_tts.py:410-520](../core/eval_tts.py#L410-L520) — `evaluate_synthesis()`：

```python
def evaluate_synthesis(
    syn_wav: Path,
    *,
    ref_wav: Path | None = None,      # SECS + F0 RMSE 需要
    target_text: str | None = None,    # WER/CER 需要
    lang: Literal["en", "zh"] = "en",
    transcriber=None,
    skip: tuple[str, ...] = (),        # 可跳过指定指标
) -> TTSEvalScores:
```

返回 `TTSEvalScores` dataclass，包含所有 4 维分数 + NISQA 子维度 + DNSMOS 辅助分。

**面试可以强调的设计细节**：
- NISQA 权重自动下载（~1MB），懒加载
- WavLM-SV 使用 `use_safetensors=True` 绕开 torch.load 的安全问题（CVE-2025-32434）
- F0 RMSE 无 DTW（v1 对等长片段线性对齐；v2 预留 DTW 升级）
- `skip` 参数让评测可增量运行，某个指标挂了不影响其他

### 4.3 数据集构建 — Tier 1/Tier 2 pair 逻辑

[scripts/build_two_tier_dataset.py:658-773](../scripts/build_two_tier_dataset.py#L658-L773) — `_build_tier1_pairs()` 和 `_build_tier2_pairs()`：

**Tier 1 核心约束**：ref 必须来自**同 speaker 但不同 emotion**（强制 cross-style learning，防止模型走捷径）

```python
# Tier 1: ref 必须不同 emotion
ref_pool = [c for emo, clips in by_emotion.items()
            if emo != target_emotion    # ← 关键约束
            for c in clips
            if c.get("duration", 0) >= min_ref_dur]
```

**Tier 2 核心区别**：不强制 cross-emotion（Tier 2 数据通常不带 emotion 标签），单纯 (text, ref, target) 三元组，追求过拟合说话人微观特征。

**Gold clip 设计**：每 speaker 挑 top-N MOS 片段作为 SECS_vs_gold 评测集，这些片段**绝不进训练对**——避免 ref-leakage 假阳性（用 ref 当 gold 会导致 SECS 虚高）。

### 4.4 ESD 数据 ingest 管线

[scripts/build_two_tier_dataset.py:223-361](../scripts/build_two_tier_dataset.py#L223-L361) — `ingest-esd` 子命令：

- 遍历 ESD 目录结构（20 speakers × 5 emotions × ~350 chunks = 35,000 条）
- 保留原 chunk 和 text（不重跑 Demucs/VAD/ASR，ESD 已经被原作者清洗）
- 只补齐质量字段：WADA-SNR + DNSMOS + clipping
- 输出路径支持 `${ESD_ROOT}/...` 环境变量替换，本地/AutoDL 可移植

### 4.5 TTSBackend Protocol — 跨架构设计

[core/tts.py:60-85](../core/tts.py#L60-L85)：

```python
class TTSBackend(Protocol):
    def synthesize(self, text: str, ref_audio: Path, *, out_path: Path,
                   prompt_text: str = "", instruct: str | None = None,
                   mode: str = "zero_shot", normalize: bool = True,
                   lang: str = "en") -> Path: ...
    def close(self) -> None: ...
```

这是 RESEARCH_PLAN §3.3 跨架构泛化实验的基础——CosyVoice 2、F5-TTS、IndexTTS 只要实现同一个 Protocol，上层评测代码一行不改。

---

## 5. 方法论亮点（面试深挖点）

### 5.1 从主观到客观的转化能力

不是"我觉得 instruct 不好听"，而是"通过 4 维指标 + 跨 3 个 target text 的 consistency check，证明这是架构层面的 conditioning 冲突，不是 prompt 问题"。

### 5.2 评测的严谨性

- 发现 DNSMOS-P808 在 TTS 上的 mis-rank，主动调研并替换为 NISQA
- 设计 SECS_vs_gold 协议：避免把 ref audio 直接当 ground truth（导致 SECS 虚高）
- 三层 SECS 对比：SECS_vs_ref（克隆精度）/ SECS_vs_gold（音色保真）/ SECS_cross（说话人表示一致性）

### 5.3 产品线驱动技术决策

不是"我要做 LoRA"然后找场景，而是从两条产品线（通用零样本 vs 深度数字分身）的冲突需求出发，推导出双层 LoRA 架构。体现的是**从商业需求倒推技术方案**的思维。

### 5.4 工程务实

- CosyVoice 2 的 torch 版本与 M1 pipeline 冲突 → subprocess 方案而非降级依赖
- $30-40 GPU 预算完成全部实验，而非烧几千刀
- seed=42 固定，所有实验可复现

---

## 6. 可能被问到的技术问题

### Q: LoRA 的 rank 怎么选的？

A: Tier 1 rank=16（多说话人场景，需要保留泛化能力，rank 太高容易过拟合到特定 speaker），Tier 2 rank=32（单说话人 avatar，需要捕获微观特征，过拟合是 feature）。这个选择来自 LoRA 论文的建议 + TTS 社区的经验值，最终会通过 ablation 实验验证（rank 8/16/32/64 对比）。

### Q: 为什么不直接在 CosyVoice 2 上做 full fine-tune？

A: 三个原因——显存（全参需 A100，LoRA 只需 4090）、数据量（30min 全参极易过拟合）、工程灵活性（LoRA 权重小，多说话人切换只需换包，不破坏 base model）。

### Q: instruct mode 的 SECS 下降是怎么发现的？

A: 主观试听 6 条 wav → 感觉有问题但无法量化 → 搭客观 eval 管线（MOS-NISQA / WER / SECS / F0 RMSE）→ 在 3 个独立 target text 上发现一致的 SECS -0.13 + F0 RMSE +21Hz pattern → 确认是架构层面的 conditioning 冲突。

### Q: 怎么评估 LoRA 是否成功？

A: 不只看 training loss。核心是 unseen-speaker SECS（Line A，泛化不退化）+ seen-speaker SECS（Line B，极致音色）+ instruct SECS drop（风格不牺牲音色）。early stop 条件直接挂在 unseen SECS 上而不是 training loss。

### Q: 如果 Tier 1 LoRA 反而让 unseen SECS 下降了怎么办？

A: 这是预期内的风险（见 RESEARCH_PLAN §5 风险表）。缓解措施：early stop by unseen SECS（非 training loss）、style-balanced batch sampling 防止 speaker leakage、验证集上持续监控。如果所有措施都压不住退化，说明 LoRA 容量需要再降低或数据多样性需要再提高。

---

## 7. 文件索引（快速定位）

| 想找什么 | 文件 |
|---|---|
| 完整问题定义 + 实验设计 | [docs/RESEARCH_PLAN.md](RESEARCH_PLAN.md) |
| 为什么用 LoRA（ADR） | [docs/decisions/0004-lora-finetuning.md](decisions/0004-lora-finetuning.md) |
| AR + NonAR 双阶段设计 | [docs/decisions/0005-ar-nonar-architecture.md](decisions/0005-ar-nonar-architecture.md) |
| 4 维客观评测代码 | [core/eval_tts.py](../core/eval_tts.py) |
| TTS 合成入口（zero_shot / instruct） | [core/tts.py](../core/tts.py) |
| 数据集构建（Tier 1/2 pairs） | [scripts/build_two_tier_dataset.py](../scripts/build_two_tier_dataset.py) |
| exp_002 baseline 数据 | [experiments/exp_002_ref_and_instruct/eval_objective.md](../experiments/exp_002_ref_and_instruct/eval_objective.md) |
| 里程碑 | [docs/ROADMAP.md](ROADMAP.md) |
| 两条产品线 + LoRA 策略 | [memory: project_no_single_speaker_lora](../.claude/projects/-Users-attention-Documents-projects-voice-story/memory/project_no_single_speaker_lora.md) |

---

## 8. 面试叙事建议

建议按这个顺序讲（5-8 分钟）：

1. **项目是什么**（30s）：声音克隆系统，两条产品线（通用零样本 + 深度数字分身）
2. **发现了什么问题**（1min）：instruct mode 的 SECS -0.13 → 搭客观 eval 验证 → 定位到 conditioning 冲突
3. **为什么选择 LoRA 而不是其他方案**（1min）：对比全参/Soft Prompt/DreamBooth，LoRA 在成本、灵活性、composition 上的优势
4. **双层架构怎么设计**（2min）：Tier 1 多说话人保留泛化 + Tier 2 单说话人追求极致 + 叠加后 Pareto 改进
5. **怎么验证效果**（1min）：4 维指标 + SECS_vs_gold 协议 + 成功标准量化
6. **方法论亮点**（30s）：从主观到客观、评测严谨性、产品线驱动决策、$30 预算复现
