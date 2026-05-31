# Interview Prep — Voice Story 项目面试问答

> 最后更新：2026-05-31 · 基于 exp_002-009 全部数据

---

## 1. 项目一句话 (Elevator Pitch)

> 我设计了一个 **LLM Agent 驱动的长文本情感 TTS 系统**：针对 CosyVoice 3 在 200+ 字场景下的崩溃问题，用 LLM 将长文本智能切分并在自然位置插入副语言 token（呼吸、停顿、语气词），再通过 LoRA 微调保持跨段落的说话人一致性。在过程中完成了 2,600+ 次合成的系统性评测，发现了 CV3 论文未记录的 12 个副语言 token，并将 prompt 格式错误的排查教训固化为工作流程。

---

## 2. 完整技术故事线

### Act 1: 发现问题（系统性评测）

自建 4 维客观评测管线（CER/SECS/F0 RMSE/NISQA），在 ESD 中文情感数据集上跑 720 次 CV3 零样本合成。

**发现**：
- CER 接近 0——内容保真度不是问题
- 高唤醒情感 F0 RMSE 是低唤醒的 1.8×——韵律传递有情绪差异
- 发现了 12 个 CV3 论文未记录的副语言 token（`[breath]`/`[sigh]`/`[mn]` 等）

期间因 prompt 格式错误浪费 2 天——将教训写入 memory，此后所有实验先对比官方 example。

### Act 2: 诊断根因（控制变量实验）

| 实验 | 回答的问题 |
|------|---------|
| exp_006 (1,152 合成) | System prompt 有影响吗？→ 中文有害(10×CER)，空 prompt 可用 |
| P0 (192 合成) | 12 个 token 哪些有效？→ 3 个有效，1 个有害 |
| P1 (192 合成) | Token 能增强情感吗？→ 仅 Angry+`[breath]` 有效(-7Hz) |
| P3 (7 段文本) | 长度上限是多少？→ **200 字后 CER 崩溃** |
| E1 (数据分析) | 怎么选 ref？→ 差异 9×，Neutral 最安全 |

### Act 3: 构建解决方案

**问题定义**：CV3 零样本有 200 字长度天花板，无法直接用于有声书/长内容。

**方案**：
1. **LLM Agent 做文本预处理**：将长文本按语义边界切分为 <200 字的段落，在自然位置插入副语言 token（句间 `[breath]`，疑问后 `[mn]`，情感句插入 `[sigh]`/`[laughter]`）
2. **LoRA 微调保持一致性**：rank=8 在 ESD cross-emotion pairs 上训练，F0 RMSE 降低 23 Hz 同时 SECS 不变，确保跨段落的说话人风格一致
3. **优化 Ref 选择**：根据 E1 分析，优先使用 Neutral + 说话人 0006/0004/0008

**为什么这是最好的方向**：
- CV3 的长文本限制是真问题，不是调用错误——必须有工程方案
- LLM 做 token 插入比规则系统更智能——能根据上下文选择何时 `[breath]`、何时 `[mn]`
- 所有发现都服务于这个方案：P0 证明 token 有效，P1 指导 token 选择策略，E1 指导 ref 选择
- LoRA 不是可有可无的——跨段落一致性需要它

---

## 3. 高频问题预答

### Q: 这个项目最核心的贡献是什么？

不是"发现了 CV3 的 bug"，而是**完整的问题发现→诊断→解决方案链条**：

1. 搭建评测体系，系统性地找到 CV3 的能力边界（200 字天花板、高唤醒 F0 差）
2. 通过控制变量实验定位根因（token budget competition、训练长度限制）
3. 设计了 LLM Agent + LoRA 的工程方案来解决长文本问题
4. 过程中发现了 12 个未记录 token，追踪了完整的 token→音频链路

### Q: 为什么需要 LoRA？CV3 零样本已经够好了？

CV3 零样本在单句上确实够好（CER~0）。但长文本场景下，每次切分后的独立合成会导致：
- 段落间的韵律不连贯（每段重新从 ref 提取 prosody）
- 说话人音色微小漂移累积

LoRA rank=8 的作用不是"提高质量"，而是**锁定风格一致性**——让跨段落的多个合成听上去是同一个人、同一种情绪。实验数据：F0 RMSE 降 23 Hz 同时 SECS 不变，证明 style-speaker 解耦可行。

### Q: LLM Agent 具体怎么做？

```
长文本输入
  ↓
LLM 分析：语义边界检测 + 情感识别 + 语气词位置预测
  ↓
输出：[segment1] [breath] [segment2] [mn] [segment3] [sigh] [segment4]
  ↓
每段 < 200 chars，CV3 零样本分别合成（用 LoRA 保持一致性）
  ↓
音频拼接 + crossfade
```

LLM 的 prompt 示例：
> 你是一个 TTS 预处理助手。将以下文本切分为适合语音合成的段落（每段不超过 150 字），并在自然停顿处插入副语言标记。可用标记：[breath] 呼吸、[mn] 犹豫、[sigh] 叹气、[laughter] 笑声。

### Q: 评测体系 vs 官方 CV3-Eval 有什么差异？

| 维度 | 官方 | 我们 | 意义 |
|------|:---:|:---:|------|
| CER | ✓ | ✓ | 持平 |
| 说话人 | ERes2Net | WavLM-SV | 同思路 |
| 音质 | DNSMOS | NISQA | 不同实现 |
| **韵律(F0)** | **无** | **有** | **我们独有** |
| **副语言 token** | **无** | **12 token 矩阵** | **我们独有** |
| 测试规模 | 多语种大 | ESD 8 人 | 他们大 |

差异化价值：F0 RMSE 填补了韵律评测空白；副语言 token 系统性评测是全新的评测维度。

### Q: Prompt 格式错误具体是什么？怎么发现的？

```python
# 错误（我的版本）
prompt = ref_text + "<|endofprompt|>"
# → "今天真凉快。<|endofprompt|>"

# 正确（官方 example.py）
prompt = "You are a helpful assistant.<|endofprompt|>" + ref_text
# → "You are a helpful assistant.<|endofprompt|>今天真凉快。"
```

`<|endofprompt|>` 是 system prompt 和 ref 描述之间的分隔符。放反了导致模型把 ref 文本当成 system prompt 来执行而不是当成"参考音频说了什么"来描述。2 天 936 次合成白跑。此后所有实验先对比官方 example。

### Q: 副语言 token 是怎么从文本变成音频的？

`[breath]` → CosyVoice3Tokenizer 识别为 special token (ID≈151939) → Qwen2 embedding 层映射为 896 维向量 → 经 24 层 attention 后提升 silent FSQ code 的 log-prob → 生成类似静默的 speech token `[2,28,55,...]` (25Hz, 每个 40ms) → Flow DiT 渲染为 50Hz mel 谱 → HiFi-GAN 上采样 480× 输出 24kHz 波形。

0.5 秒的呼吸声 = 约 12-13 个 speech token。这些 token 从 LLM 的固定 budget 中支出——这就是为什么插入副语言 token 会让总时长缩短（Token Budget Competition）。

### Q: 长文本崩溃的根本原因？

CV3 训练时 `token_max_length=200` chars，`max_length=10240` frames（~102 秒音频）。模型从未见过需要生成超长序列的场景。实验数据：CER 在 200 字处出现拐点（0.12→0.92），与训练限制完全一致。

### Q: 给后来人的实用建议？

- 选 ref：Neutral 情感 + 说话人 0006/0004/0008，F0 RMSE 可低至 14 Hz
- System prompt：用英文或空 prompt，别用中文
- 文本长度：单次合成控制在 150 字以内
- 副语言 token：Angry 加 `[breath]` 有帮助，Sad 别加 `[sigh]`
- 先跑通官方 example 再开始实验
