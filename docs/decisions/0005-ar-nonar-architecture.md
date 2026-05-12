# ADR-0005: 进阶架构采用 AR 主干 + NonAR refiner 双阶段

- **Date**: 2026-05-12
- **Status**: Accepted（待 v0.4.x 阶段实施）

## Context

豆包式产品的两个主要痛点：
1. 一句话录入 → 单一参考样本，音色与情绪覆盖不够（已由数据管线的多样性采样解决）
2. 长篇朗读时音色 / 情绪粒度不够细，难以做到逐句可控

仅靠 zero-shot 多参考 prompting + LoRA 微调，难以在"音色相似度 + 情绪粒度 + 长稳定性"上同时达标。需要从模型架构层面拆分关注点。

## Decision

采用 **AR 主干 + NonAR refiner 双阶段架构**：

```
text ──► [AR backbone]  ──► semantic / coarse acoustic tokens
                                    │
                  reference audio ──┤
                                    ▼
                          [NonAR refiner]  ──► fine acoustic / waveform
                                    ▲
              style / emotion prompt ┘
```

- **AR 主干**（CosyVoice 2 LLM / IndexTTS-2 等）负责内容生成 + 基础韵律。LoRA 微调适应目标说话人的说话习惯（停顿、连读、语速）
- **NonAR refiner**（Flow Matching decoder，CosyVoice 自带 or F5-TTS）负责音色精修。在这里注入：
  - Speaker embedding（ECAPA-TDNN / WavLM-XL）做后融合强化音色
  - Emotion embedding（emotion2vec）做逐句情绪控制

## Alternatives

- **纯 AR**：长文本生成易累积错误（重复 / 漏字），缺乏对音色 / 情绪的显式控制接口
- **纯 NonAR**（如 F5-TTS）：推理快、稳定，但中文内容理解 / 韵律控制弱于 AR
- **AR + AR**（两个 AR 串联）：训练复杂度爆炸，且 refiner 阶段不需要内容建模

## Consequences

### 正向
- 关注点分离：AR 学"说什么 + 怎么说"，NonAR 学"声音怎样"
- 音色 / 情绪有显式接口，可逐句注入，颗粒度细
- 工程上可单独迭代：先做 AR LoRA（[ADR-0004](0004-lora-finetuning.md)），再做 NonAR refiner
- 与 CosyVoice 2 原生架构对齐，迁移成本低

### 负向 / 代价
- 双阶段推理延迟比单阶段高（流式化需要更精细的调度）
- 训练 / 评估流程更复杂（需要分别衡量 AR / NonAR 各自贡献）
- 接口设计要前瞻：speaker embedding 与 emotion embedding 的注入位置 / 维度需要先定下来再训练

### 后续需要观察
- NonAR refiner 是否需要 LoRA 化（小数据场景下可能仍需轻量适配）
- Speaker / Emotion embedding 提取器与 refiner 的训练分布是否对齐
- 实际听感上"AR LoRA only" vs "AR LoRA + NonAR 后融合" 的边际收益

## References

- [docs/PLAN.md](../PLAN.md) §2.2
- CosyVoice 2 技术报告
- F5-TTS: https://github.com/SWivid/F5-TTS
