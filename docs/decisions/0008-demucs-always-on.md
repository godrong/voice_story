# ADR-0008: Demucs 人声分离默认强制开启

- **Date**: 2026-05-13
- **Status**: Accepted

## Context

数据管线的 PreprocessAgent 第一步是用 Demucs 做人声分离（剥 BGM / 音效）。原计划允许 source 通过 metadata `needs_separation=False` 跳过，理由是某些干净源（如演讲录音 / 朗读 audiobook）不需要分离，跳过能省时间。

实测发现两个问题：

1. **"干净音频"的判断不可靠**：Trump 演讲表面干净，实际某些段有听众噪声、麦克风咔嗒；audiobook 有的版本带轻微背景音乐。让 source 自报"我干净"会产出不一致的训练数据。
2. **跳过 / 不跳过引入不确定性**：合成结果对训练数据的统计性质很敏感。如果一个 dataset 里部分 chunk 走过 Demucs、部分没走，后续模型行为差异难以归因。

Demucs 的代价：在 Mac (M2/M3) 上约 3× 实时（5 分钟音频跑 1.5 分钟），在 GPU 上几乎实时。对 M1 阶段的开发节奏可接受。

## Decision

**Demucs 默认强制开启**，所有 source（含 source metadata `needs_separation=False`）都走分离。`needs_separation` 字段保留在 SourceMeta 结构里（向前兼容），但 PreprocessAgent 当前实现忽略它。

实际行为：
- 标准 WAV → Demucs vocal stem → VAD → 后续 stage
- 即使源是干净演讲也跑（接受 5~10 分钟额外耗时）

## Alternatives

- **按 source metadata 跳过**：原计划方案。问题如上：判断不可靠 + 数据不一致
- **按 SNR 自动决定**：先跑 WADA-SNR，> 阈值才跳过。增加判断逻辑复杂度，且仍有边界 case
- **完全不跑 Demucs**：训练数据质量天花板被 BGM 限制，对核心痛点 P3（"源音频脏"）放弃应对

## Consequences

### 正向
- 数据管线行为完全一致，便于 debug / 复现 / 回归
- 即使源听起来干净也能去掉细微噪音，训练数据更稳
- ADR-0008 简化了配置面（少一个用户要操心的开关）

### 负向 / 代价
- Trump 这类干净源会多消耗 ~5 分钟 Demucs 时间（Mac 上）
- 极端干净的录音棚音源理论上 SDR 可能轻微下降（Demucs 处理过的 stem 比原始干净音频略差），但实测影响可忽略
- 用户无法在 CLI 上"明确告诉系统跳过"——这是有意为之

### 后续需要观察
- 实际跑批 dataset 时，Demucs 是不是 pipeline 的瓶颈。若是，先优化模型选型（已用 `htdemucs` 而非 `htdemucs_ft`），再考虑 GPU 抽样或加 `--skip-separation` flag（届时需要新 ADR supersede 本 ADR）
- 如果 Demucs 处理录音棚级原始音频时引入 artifact（理论上可能），考虑提供 `--separator none` 的 escape hatch（仍需新 ADR）

## References

- [docs/PLAN.md](../PLAN.md) §3.A.3
- [core/separation.py](../../core/separation.py)
- [agents/preprocess_agent.py](../../agents/preprocess_agent.py)
- https://github.com/adefossez/demucs
