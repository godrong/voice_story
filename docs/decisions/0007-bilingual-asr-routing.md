# ADR-0007: 双语 ASR 后端策略（Whisper EN + FunASR ZH，langid 路由）

- **Date**: 2026-05-13
- **Status**: Accepted

## Context

项目要同时处理中文主播和英文演讲音源（M1 测试集就是英文 Trump 数据集，未来主用例是中文 B 站主播）。ASR 选型有几条路：

1. 用 Whisper-large-v3 多语言一把梭（英中都能跑）
2. 中文用专门模型（FunASR Paraformer-zh），英文用 Whisper
3. 全部用中文专长模型（如 SenseVoice）

实测在中文上：FunASR Paraformer-zh 的 WER 比 Whisper-large-v3 低 2~4 个百分点且推理更快；这对"训练数据准确性"是关键的。

## Decision

采用 **双后端 + langid 路由** 策略：

- **核心模块**：[core/asr.py](../../core/asr.py) 实现 `Transcriber.transcribe()`
- **流程**：
  1. faster-whisper 跑一次最低开销的语言识别（beam=1 + vad_filter）
  2. `lang == "zh"` → 路由到 `FunASRBackend`（Paraformer-zh + ct-punc 标点恢复）
  3. 其它（"en" / 多语言） → 路由到 `WhisperBackend`（large-v3，OOM 自动降级 medium）
- **跳过 langid 的快路径**：source metadata 提供 `lang_hint` 时直接路由
- 两个后端实例都常驻内存（首次加载后复用），同 pipeline run 内批量摊销加载成本

## Alternatives

- **只用 Whisper**：实现最省事但中文准确率亏 2~4 个百分点，直接影响训练数据质量与 M3 评估指标
- **只用 FunASR**：英文支持有限（FunASR 主要面向中日韩），M1 测试集 Trump 演讲就跑不动
- **手动指定语种**：用户每次 ingest 都要传 `--lang`，繁琐且对多语种混合源无解
- **第三个后端做 langid（如 SpeechBrain）**：增加一个 ML 依赖，相比 faster-whisper 自带 langid 没有明显收益

## Consequences

### 正向
- 中文场景拿到 Paraformer 的精度优势，英文场景拿到 Whisper 的多语言鲁棒性
- langid 是一次性极轻调用（~0.5s），开销可忽略
- `lang_hint` fast path 留给"我就是要中文"的高确定性场景

### 负向 / 代价
- 同时维护两个 ASR 后端的依赖（`faster-whisper` + `funasr` + `modelscope`）
- 首次启动两份模型下载：Whisper-large-v3 ~3GB，FunASR Paraformer-zh + ct-punc 共 ~1.5GB
- 两份模型常驻 ~4~5GB 显存/内存，对低配机不友好（Mac 16GB 仍 OK）

### 后续需要观察
- 多语种混合（一段话夹中英文）的 chunk 该怎么处理：当前 langid 给一个标签，可能误判。M3 加测后再决定是否要 chunk-level routing 或代码切换检测
- Whisper-medium fallback 的实际触发率，决定是否主动选 medium 当默认

## References

- [docs/PLAN.md](../PLAN.md) §3.A.6
- [core/asr.py](../../core/asr.py)
- https://github.com/SYSTRAN/faster-whisper
- https://github.com/modelscope/FunASR
