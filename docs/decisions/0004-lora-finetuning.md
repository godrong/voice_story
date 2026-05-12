# ADR-0004: 训练侧使用 LoRA 微调而非全参微调

- **Date**: 2026-05-12
- **Status**: Accepted（待 v0.3.x 阶段实施）

## Context

当 zero-shot 的 speaker similarity 达不到 0.85 目标时，需要做模型微调。CosyVoice 2 / IndexTTS-2 这类 0.5B~1B 的 AR + Flow Matching 模型，全参微调代价：
- 显存：≥40GB（A100 或多卡）
- 时间：数小时
- 30 分钟的目标说话人数据，全参微调容易过拟合

需选择微调策略。

## Decision

采用 **LoRA 微调**（Low-Rank Adaptation）：
- 仅在 AR 主干的 attention 层注入低秩矩阵（rank=16~32）
- 训练时冻结原始权重，只更新 LoRA 参数（约总参数的 1~3%）
- 30 分钟数据 + 单张 4090/A100 + 1~2 小时即可收敛

未来扩展点：层级 LoRA（不同层不同 rank） + adapter-style 风格 / 情绪控制头。

## Alternatives

- **全参微调**：过拟合风险高、训练慢、显存大；不在小数据场景下值得
- **Soft Prompt Tuning**：只学习连续 prompt embedding；表达能力不够，难以学到说话习惯
- **DreamBooth-style 微调**：图像领域的概念，对 AR TTS 适配性差
- **不微调，纯多参考 prompting**：已是 MVP 方案；天花板较低，难以同时优化相似度 + 韵律 + 情绪

## Consequences

### 正向
- 单 GPU 即可训练，云成本可控（4090 约 2 元 / 小时）
- 训练快，迭代周期短
- LoRA 权重小（数十 MB），便于多说话人切换：每个声纹一个 LoRA 包
- 不破坏原始模型，可与 zero-shot 路径共存

### 负向 / 代价
- LoRA 上限略低于全参（一般 1~3% 相似度差距）
- 需要管理 LoRA 权重的版本与 dataset hash 对应关系
- AR + Flow Matching 双阶段：Flow 部分是否一起 LoRA 化需要实验

### 后续需要观察
- LoRA rank 与相似度 / 过拟合的关系
- 是否需要在 NonAR refiner 上也加 LoRA（见 [ADR-0005](0005-ar-nonar-architecture.md)）
- 多说话人混合数据 + 多 LoRA 切换的工程实践

## References

- [docs/PLAN.md](../PLAN.md) §2.2 "训练优化项"
- LoRA paper: https://arxiv.org/abs/2106.09685
