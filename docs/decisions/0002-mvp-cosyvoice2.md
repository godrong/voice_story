# ADR-0002: MVP 阶段使用 CosyVoice 2 zero-shot

- **Date**: 2026-05-12
- **Status**: Accepted

## Context

需要在 MVP 阶段快速跑通"音频源 → 训练数据 → 合成"全链路，验证产品价值。完整 LoRA 微调流程复杂、需要 GPU 资源，不适合作为第一里程碑。

候选 TTS 模型：CosyVoice 2、IndexTTS-2、F5-TTS、FishSpeech、Spark-TTS、商业 API（MiniMax / 火山语音复刻）。

## Decision

MVP 使用 **CosyVoice 2-0.5B** 的 zero-shot 推理路径：
- 从用户提供的音频 dataset 中按多样性采样选 3~5 段参考片段（不同情绪 / 韵律 / 时长）
- 推理时通过多参考 prompting 合成目标文本
- 用 LLM（Claude via LiteLLM）做 dynamic reference selector：根据目标文本的情绪 / 韵律标签匹配最佳参考

## Alternatives

- **IndexTTS-2 zero-shot**：B 站出品，质量与 CosyVoice 接近，但生态与文档不如 CosyVoice 完善
- **F5-TTS / FishSpeech**：非自回归推理更快，但中文相似度略弱
- **商业 API**：质量稳定但每个声纹付费、无法深度调优、无法做后续 LoRA 路径
- **直接做 LoRA 微调**：MVP 阶段成本太高，且 zero-shot 已能验证产品形态

## Consequences

### 正向
- Mac 本地可跑（量化版 + MPS），开发反馈周期短
- 中文 zero-shot 相似度 SOTA 之一（baseline 可达 ~0.75 余弦相似度）
- 后续无缝升级 LoRA 微调（同模型同 tokenizer，见 [ADR-0004](0004-lora-finetuning.md)）
- 多参考 prompting 本身就是对"豆包一句话录入"痛点的直接回应

### 负向 / 代价
- 模型权重较大（约 1.5GB），首次启动慢
- 中文长文本合成时偶有重复 token / 漏字（CosyVoice 2 已知问题），需要后处理 + 重合成机制
- 非英文 / 中文场景效果未知

### 后续需要观察
- 跑出 baseline speaker similarity / WER / MOS 数字，决定是否需要尽早进入 LoRA 微调
- M 系列 Mac 的推理速度是否满足"实时朗读"流式接口需求

## References

- [docs/PLAN.md](../PLAN.md) §2.1
- https://github.com/FunAudioLLM/CosyVoice
