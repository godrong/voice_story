# Voice Story

> 个性化声音克隆 + 长文本朗读系统：把主播的声纹复刻到一本书上，给你讲睡前故事。

## 状态

- 版本：`0.0.2`（exp_005 完成，prompt 格式修正）
- 阶段：CV3 零样本情感合成基线已建立，确认模型在正确调用下表现良好

## 目录

| 路径 | 内容 |
|---|---|
| `docs/PLAN.md` | 完整技术规划 |
| `docs/ROADMAP.md` | 勾选式 milestone 看板 |
| `docs/CHANGELOG.md` | 版本变更日志（Keep a Changelog） |
| `docs/decisions/` | ADR：关键架构决策记录 |
| `agents/` | ADK agent 实现 |
| `core/` | 底层 wrapper（ffmpeg / Demucs / VAD / ASR / TTS） |
| `inputs/` | 用户放入的原始音频/视频（gitignored） |
| `datasets/` | 处理后的训练数据（gitignored） |
| `models/` | 模型权重缓存（gitignored） |
| `books/` | 输入书本（gitignored） |
| `outputs/` | 合成音频（gitignored） |

## 协作规范

- 每个**关键技术决策**写一份 ADR（`docs/decisions/`），编号、不可改，只能用新 ADR supersede 旧的
- commit message 末尾带 `Refs: ADR-XXXX` 或 `Refs: PLAN#section` 让 git blame 可追溯
- 每次发版更新 `pyproject.toml` 的 version + `docs/CHANGELOG.md` + `git tag`

## CV3 零样本情感合成基线（exp_005，2026-05-30）

使用**正确 prompt 格式**（`"You are a helpful assistant.<|endofprompt|>" + ref_text`），
8 说话人 × 5 情感 × 6 文本 × 3 种子 = 720 合成。评测：CER / SECS / F0 RMSE。

### 发现

| 情感 | CER | SECS | F0 RMSE |
|------|-----|------|---------|
| Neutral | 0.007 | 0.970 | 46.0 Hz |
| Happy | 0.007 | 0.965 | 68.1 Hz |
| Angry | 0.009 | 0.959 | 72.5 Hz |
| Surprise | 0.008 | 0.965 | 83.9 Hz |
| Sad | 0.009 | 0.976 | 57.7 Hz |

- **CER 接近 0**：所有情感内容保真度优秀，不存在"语义泄漏"
- **高唤醒度情感 F0 RMSE 更高**：Surprise (83.9 Hz) 是 Neutral (46.0 Hz) 的 1.8 倍。这是 TTS 领域已知现象——高 arousal 的 F0 变化剧烈，模型难以完全复现
- **Sad 表现最好**：最佳 SECS (0.976) + 第二好 F0 RMSE (57.7 Hz)
- **CV3 零样本在正确调用下表现良好，不需要训练/微调**

### 教训

之前 exp_002-004 报告的"Sad 语义泄漏"是 `<|endofprompt|>` token 放置位置错误导致的。
`<|endofprompt|>` 是 system prompt 和 ref 文本之间的分隔符，不是 ref 文本的后缀。
详见 memory `feedback_check_official_first`。

---

## CV3 零样本调用参数

```python
# prompt_text 格式（最容易出错的地方）
prompt_text = "You are a helpful assistant.<|endofprompt|>" + ref_text
#              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^              ^^^^^^^^
#              system prompt（可替换）                         ref 文本（必须放后面）
#                               ^^^^^^^^^^^^^^^^
#                               分隔符，在两者之间

# 调用
mi_base = frt.frontend_zero_shot(
    tts_text,         # 目标合成文本
    prompt_text,      # system_prompt + <|endofprompt|> + ref_text
    prompt_wav,       # 参考音频路径
    model.sample_rate,# 24000
    ""                # zero_shot_spk_id，空字符串 = 零样本模式
)

gen = cv3m.tts(**mi_base, stream=False)
# stream=True  → 流式输出（逐 chunk）
# stream=False → 一次性生成
```

## 评测指标

| 指标 | 工具 | 衡量什么 | 好值 |
|------|------|---------|------|
| **CER** | FunASR paraformer-zh | 字符错误率，合成内容是否正确 | < 0.05 |
| **SECS** | WavLM-Base-Plus-SV | 说话人音色余弦相似度 | > 0.95 |
| **F0 RMSE** | librosa.pyin | 基频均方根误差，语调/韵律是否准确传递 | < 50 Hz |

## 已知局限

- **高唤醒情绪（Surprise/Angry/Happy）F0 RMSE 是 Neutral 的 1.5-1.8 倍**——业界共性问题，非 CV3 特有
- instruct mode 牺牲音色保真度（SECS -0.13），官方已知
- CV3 零样本在中文情感数据上 CER 接近完美，**不需要训练/微调**
