# Interview Prep — Voice Story 项目面试问答

> 最后更新：2026-05-30 · 基于 exp_005 干净数据（prompt 格式修正后）

---

## 1. 项目一句话 (Elevator Pitch)

> 我搭建了一套 **5 维客观评测管线**评估 CosyVoice 3 零样本情感合成质量。在系统排查过程中，我发现了一个 **prompt 格式与官方 example 不一致的低级错误**——这个经历教会我"先对比官方代码再做大规模实验"的研究方法论。修正后，确认 CV3 零样本在正确调用下内容保真度优秀（CER 接近 0），真正的研究发现是**高唤醒情绪（Surprise/Angry）的韵律传递精度显著低于低唤醒情绪（Neutral/Sad）**——F0 RMSE 差距高达 1.8 倍。同时通过 LoRA rank=8 在保持 SECS 不变的同时将 F0 RMSE 降低 23 Hz，验证了 style-speaker 解耦的可行性。

---

## 2. 我做了什么（按时间线）

### Phase 0 — 搭客观评测框架

- 自建 eval pipeline：[core/eval_tts.py](../core/eval_tts.py)
- 发现 DNSMOS-P808 对 TTS 输出给出反向排序，替换为 NISQA
- 指标：MOS-NISQA / SECS (WavLM-SV) / CER (FunASR) / F0 RMSE (librosa.pyin)

### Phase 1 — 数据工程

- 下载 ESD (20 speakers × 5 emotions × 350 = 35,000 chunks)
- 构建 Tier 1 LoRA 训练对：**26,943 对** cross-emotion pairs
- 写 [scripts/build_two_tier_dataset.py](../scripts/build_two_tier_dataset.py)

### Phase 2 — exp_002: CosyVoice 2 instruct mode 缺陷验证

- 确认官方承认的"instruct mode 无法通过文本指令控制音色"
- 量化：SECS 下降 0.13，F0 RMSE 上升 21 Hz

### Phase 3 — exp_003: LoRA style adaptation

- LoRA rank=8 在 ESD 上训练，F0 RMSE 降低 23 Hz 同时 SECS 保持不变
- 证明 style-speaker 解耦可行

### Phase 4 — exp_004: 疑似"语义泄漏"调查（后被修正）

- 发现 Sad 情感下 ASR 输出包含 ref 文本词汇
- 进行多说话人/多文本/多种子系统性实验
- **关键转折**：对比官方 example.py 后发现 `<|endofprompt|>` token 位置放反
- 修正后重新验证——泄漏完全消失。**不是模型缺陷，是调用错误。**

### Phase 5 — exp_005: 正确格式下完整基线

- 8 说话人 × 5 情感 × 6 文本 × 3 种子 = 720 合成，正确 prompt 格式
- 结果：
  - CER 接近 0（所有情感内容可懂度优秀）
  - SECS 0.959-0.976（说话人保真度优秀）
  - **唯一真实局限：高唤醒情绪 F0 传递差 1.8 倍**

---

## 3. 高频问题预答

### Q: 你最大的 finding 是什么？

两个层面：

**技术层面**——CV3 零样本在正确调用下表现很好，但高唤醒情绪（Surprise/Angry/Happy）的韵律传递明显差于低唤醒情绪（Neutral/Sad），F0 RMSE 差距最高达 83.9 vs 46.0 Hz。

**方法论层面**——我犯了一个有价值的错误。在没有对比官方 example 的情况下，把一个 prompt 格式错误（`<|endofprompt|>` 放反了）误解为"语义泄漏"模型缺陷，花了两天跑实验。之后我建立了"先跑通官方 example，再对比差异"的排查流程，这个教训比任何一个技术发现都重要。

### Q: 你的评测体系为什么权威？

| 指标 | 工具 | 权威性 |
|------|------|--------|
| CER | FunASR paraformer-zh | 阿里自研 SOTA 中文 ASR，CV3 论文同款 |
| SECS | WavLM-Base-Plus-SV | VoxCeleb SOTA，YourTTS/CV3 论文通用 |
| F0 RMSE | librosa.pyin | 语音转换论文必报基频指标 |
| MOS | NISQA | 2021+ TTS 论文最常用无参考 MOS |

每个指标都有论文引用支撑，不是拍脑袋选的。

### Q: 有必要做训练/微调吗？

**基于当前数据，不需要。** CV3 零样本在正确调用下 CER 接近完美、SECS 优秀。高唤醒情绪 F0 传递差是 TTS 领域共性限制而非 CV3 特有 bug。LoRA 微调只有在特定产品需求（如深度数字分身）下才有价值。

### Q: 你从这里学到的最大教训？

永远先跑通官方 example，逐行对比自己的调用方式和官方的差异。花 30 分钟验证 pipeline 正确性，比花两天在错误基础上做研究划算得多。这个习惯现在已经固化进我的工作流程。
