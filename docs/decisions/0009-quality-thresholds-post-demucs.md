# ADR-0009: 质量阈值按 Demucs 后音频校准

- **Date**: 2026-05-13
- **Status**: Accepted
- **Related**: ADR-0008（Demucs 默认强制开启）

## Context

ADR-0008 决定 Demucs 默认强制开启，所有 chunk 在质量评分前都已经过人声分离。但 v0.1.0 的初始过滤阈值（`min_mos_ovr=3.5` / `min_snr_db=15`）是按"原始混音音质"的经验值设的，没考虑 Demucs 处理对指标分布的影响。

M1 端到端首次跑（Trump WEF 2018 前 30 秒）暴露了两个问题：

1. **WADA-SNR 在 Demucs 输出上失真**：实测 5 个 chunk 的 SNR 都在 4~6 dB（远低于 15 dB 阈值），但人耳实际听感很干净。WADA 算法依赖"信号 + 噪声混合时幅值统计有特定规律"的假设；Demucs 已经把噪声剥到几乎为零，统计假设不成立，算法给出无意义的低值。结果是 5/5 chunk 全被错误过滤。
2. **DNSMOS-OVR 阈值偏严**：实测 OVR 平均 3.13（阈值 3.5），SIG 平均 3.69，BAK 平均 3.53。Demucs 重建人声时引入轻微 artifact（高频毛刺 / 边缘不自然），DNSMOS 训练时见过的是"原始麦克风录音"，对处理过的音频系统性扣 ~0.3~0.5 OVR。

## Decision

按 Demucs 后音频重新校准过滤门槛：

1. **WADA-SNR 完全退出过滤逻辑**：保留 `wada_snr()` 函数与 manifest 字段（仍是有用的诊断信号），但不再参与"chunk 是否进 manifest"的判定。从 `FilterThresholds` 删除 `min_snr_db` 字段，从 `_filter_reason()` 删除 SNR 检查分支，从 `cli.py` 删除 `--min-snr` 标志。
2. **DEFAULT_MIN_MOS_OVR 从 3.5 降到 3.0**：对应 Demucs artifact 的系统扣分。3.0 大约对应"Zoom 通话级清晰度"，作为训练数据下限合理。
3. **DNSMOS SIG / BAK 暂时不进 gate**：只用 OVR 一个分数判定，避免阈值面太宽。SIG / BAK 写进 manifest 仅作诊断。

## Alternatives

- **在 Demucs 之前算 SNR**：技术上可行，但增加一次额外的 audio_io.load 调用 + 数据流复杂度。Demucs 之前的"源音频质量"对训练价值不大（反正都要 Demucs），不值得。
- **保留 SNR 阈值但调到 0 dB**：等于没用，不如直接删干净
- **训练一个"post-Demucs 专用 DNSMOS"**：远超 v0.1 范围
- **干脆只看 ASR confidence**：丢失"音频质量"维度，confidence 高 + 音频差的 chunk 会被放过

## Consequences

### 正向
- 默认参数即可让"听感正常"的 chunk 通过，不需要每次 CLI 加 `--min-snr 0`
- 过滤逻辑更简单（少一个维度），更容易解释为何某个 chunk 被丢
- WADA-SNR 仍写进 manifest，未来真要用还能拿到数据

### 负向 / 代价
- 失去"过滤掉源头就脏的源音频"这个能力（理论上，但 Demucs 本身已经处理了大部分）
- 如果后续切换到不带 artifact 的分离器（如未来更强的 NonAR refiner），DNSMOS 阈值可能需要再校准——届时新 ADR supersede 本 ADR
- 假阳性风险：偶发的"DNSMOS 给出 3.0+ 但实际听感差"的 chunk 会被放过；M3 加 speaker similarity 检查后能进一步兜底

### 后续需要观察
- 用全量 27.5 分钟 Trump 数据跑后，OVR 分布的中位数 / 标准差，决定 3.0 是否还需要再调
- 中文音频（FunASR 转写）下的 OVR 分布是否与英文一致（语种间可能有系统差）
- 替换分离器后是否要重新校准

## References

- [docs/PLAN.md](../PLAN.md) §3.A.7
- [ADR-0008](0008-demucs-always-on.md)
- [agents/dataset_agent.py](../../agents/dataset_agent.py)
- 实测数据（30s smoke run）：5 chunks，OVR 2.92~3.63（mean 3.13），SNR 4.0~5.9 dB
