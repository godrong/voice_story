# ADR-0011: Energy bucket 使用自适应分位数而非固定阈值

- **Date**: 2026-05-14
- **Status**: Accepted

## Context

manifest v1.1（ADR-0010）需要给每个 chunk 打 `energy_bucket ∈ {quiet, normal, loud}`，供风格控制层选 reference（如"激昂段落用 loud 桶"）。

v1.0 的实现走固定阈值（RMS < 0.05 → quiet，< 0.15 → normal，否则 loud），优点是跨数据集结果可比；缺点是 post-Demucs 的 RMS 范围因**源类型 / 麦克风 / 演讲风格**漂移巨大：

| 数据集 | 典型 RMS 范围 |
|---|---|
| 演讲（Trump WEF） | 0.08 – 0.18 |
| 播客（清晰人声） | 0.03 – 0.10 |
| 直播切片（带混响） | 0.10 – 0.25 |

固定阈值 0.05 / 0.15 会把演讲数据集全部塞进 `loud`，把播客全塞进 `quiet`。bucket 失去区分度 → reference selector 拿到的"风格"信息坍缩。

## Decision

`energy_bucket` 改为 **每次 ingest 内 RMS 33/66 分位** 自适应分桶：

- Pass 1：所有 chunk 算原始 `energy_rms`（写入 manifest）
- Pass 2：取所有过过滤的 chunk 的 RMS，计算 p33 / p66
- 用 `[0, p33) / [p33, p66) / [p66, ∞)` 切 quiet / normal / loud

落到 manifest 的字段：
- `energy_rms`（绝对值，浮点）—— **跨数据集可比**
- `energy_bucket`（相对标签）—— **本数据集内可比**

report.md 会打印当前 ingest 的 p33 / p66，方便人工校核。

## Alternatives

- **保持固定阈值 + 加 raw RMS**：raw 值已经记录，问题是 bucket 标签失效，下游 reference 选择代码若按 `energy_bucket == "loud"` 筛会拿到错误的子集。
- **全局校准（一次性扫所有历史数据集求阈值）**：现实里数据集会持续新增（M2 + M5 + M6），全局校准要求每加一个 corpus 就重写所有老 manifest，工程上不可持续。
- **z-score 而非分位**：受极端值影响大；新闻直播里偶有一段静默会拉低均值。分位数稳健。
- **K-Means 聚 3 类**：能自适应，但同样的 RMS 在不同数据集里可能落进不同 cluster，跨集合解释变更难。分位是确定性的，更可预测。

## Consequences

### 正向
- 不同录音条件下 bucket 标签都有区分度，reference selector 在 dataset 内总能拿到三档
- 工程实现简单：一次 `np.percentile`，加 11 行代码
- raw `energy_rms` 保留，需要跨集合比较的下游（如训练 LoRA 时检测能量漂移）直接用 raw
- 与未来加 `loudness_lufs`（ITU-R BS.1770）不冲突 —— LUFS 是绝对量纲，能量 bucket 是相对量纲，二者分工：bucket 给 selector，LUFS 给 postprocess 响度归一

### 负向 / 代价
- 跨数据集 `energy_bucket` 不直接可比，必须显式用 raw `energy_rms` 做跨集合查询
- 极小 corpus（<10 chunks）分位计算意义不大，但 dataset_agent 的过滤已经把这种规模筛掉了；不额外处理
- DatasetAgent 从一次扫变两次扫；ASR / DNSMOS / emotion2vec 仍是单遍，pass 2 只在内存里走 list comprehension，IO 不变

### 后续需要观察
- 是否需要给 `loudness_lufs` 也加自适应 bucket（目前只存 raw LUFS，等 ADR-0010 落地后看 reference selector 需求）
- M5 LoRA 训练时是否需要在 raw `energy_rms` 上做归一化 augment（防音色漂移随响度漂移）
- 33/66 分位是否需要可调（当前硬编码；若 corpus 偏极端可考虑 25/75）

## References

- [docs/PLAN.md §3.A.8](../PLAN.md#L219-L228) dataset agent
- [ADR-0010](0010-style-control-llm-annotator.md) 风格控制 / manifest v1.1
- [agents/dataset_agent.py](../../agents/dataset_agent.py) `_bucket_energy_adaptive`
