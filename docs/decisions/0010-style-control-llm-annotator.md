# ADR-0010: 风格控制 — global profile + LLM 句级标注 + instruct prompt

- **Date**: 2026-05-14
- **Status**: Accepted

## Context

朗读链路上，用户需要表达三类需求：
1. **音色（timbre）**：克隆哪位主播
2. **语调 / 情绪（prosody）**：激昂解说、低语紧张、平稳叙述……
3. **环境氛围（atmosphere）**：游戏 BGM、雨夜环境音……

CosyVoice 2 的 zero-shot 推理接受三类显式输入：`text`、`reference_audio(s)`、`instruct_prompt`（自然语言指令，原生支持）；后者用来精细控制语调 / 情绪而无需训练。
环境氛围不属于 TTS，应在 [3.C.3 postprocess](../PLAN.md#L276-L279) 层 ffmpeg 混音解决，不与 TTS 控制路径耦合。

主要问题：用户不应被要求逐句标注情绪 / 节奏。需要一个**自然语言 → 结构化风格参数**的中间层，且能与已有的 reference selector ([ADR-0002](0002-mvp-cosyvoice2.md))、AR+NonAR 双阶段 ([ADR-0005](0005-ar-nonar-architecture.md)) 串起来。

## Decision

采用 **三层风格控制 + LLM 句级标注** 设计：

```
┌─ Layer 1: global voice_profile ──────────────────────────────┐
│  speaker / base_style / bgm  ← 用户自然语言输入一次           │
└──────────────────────────────────────────────────────────────┘
                 ↓
┌─ Layer 2: per-sentence StyleSpec（LLM 生成）─────────────────┐
│  emotion / intensity / pace / instruct_prompt / ref_filter   │
└──────────────────────────────────────────────────────────────┘
                 ↓
┌─ Layer 3: 合成 + 后处理 ─────────────────────────────────────┐
│  tts(text, ref_audios, instruct_prompt) → postprocess(+bgm)  │
└──────────────────────────────────────────────────────────────┘
```

### 数据结构

**全局 profile**（一本书一次）：

```yaml
voice_profile:
  speaker: "<dataset_id>"           # → 决定 reference 池
  base_style: "游戏实况解说，激昂带紧张感"   # 自然语言，LLM 解析
  bgm: "outputs/bgm/dungeon.mp3"    # 可选；postprocess 混入
```

**StyleSpec**（LLM 逐句产出）：

```json
{
  "emotion": "tense",
  "intensity": 0.7,
  "pace": "fast",
  "instruct_prompt": "用急促压低的声音，像在躲避追兵",
  "ref_filter": {"emotion": ["tense", "whisper"], "energy_min": 0.6}
}
```

### 局部覆盖

支持在书本文本中内联 `[[style: 紧张低语]]`，强制覆盖该段 LLM 输出，作为重度用户的逃生口。

### Manifest schema v1.1

风格控制依赖 manifest 在 chunk 行级暴露 **音色 / 情绪 / 韵律** 标量。当前 v1.0 字段缺位严重；分两批扩展，T1+T2 在 M2 完成，T3+ 推到 M5。

**T1 — 零成本补全**（dataset_agent 已算过但只进 report 没入行）：

```jsonc
{
  // 既有字段保持不变
  "chunk_id": "...", "audio_path": "...", "source_file": "...",
  "text": "...", "lang": "zh", "confidence": 0.96,
  "duration": 8.4,
  "snr_db": 5.4, "mos_ovr": 3.61, "mos_sig": 4.13, "mos_bak": 3.83,

  // 新增（T1）
  "manifest_version": "1.1",
  "start_sec": 12.6, "end_sec": 21.0,        // 在 source_file 内的偏移
  "clipped": false,                          // 不仅过滤，也持久化
  "duration_bucket": "medium",               // short / medium / long
  "energy_bucket": "normal",                 // quiet / normal / loud
  "energy_rms": 0.082,                       // 原始值；bucket 由它派生
  "prosody_label": "declarative",            // question / declarative / exclamation
  "prev_chunk_id": "...", "next_chunk_id": "...",  // 跨句韵律邻居
  "text_hash": "...",                        // simhash，近似重复检测
  "speaker_id": "main",                      // 多人源未来扩展用
  "langid_confidence": 0.99                  // 区别于 ASR confidence
}
```

**T2 — 风格控制核心**（emotion2vec + 信号统计，ADR-0010 主轴所需）：

```jsonc
{
  "emotion_tag": "neutral",                  // top-1 ∈ {neutral, happy, sad, angry, fearful, disgust, surprised}
  "emotion_confidence": 0.71,                // top-1 概率
  "pitch_mean_hz": 142.3,                    // 基频均值 → 高/低音区
  "pitch_std_hz": 28.7,                      // 基频抖动 → 平稳 vs 激动
  "pace_units_per_sec": 4.2,                 // ZH 按字 / EN 按音节，归一
  "speech_ratio": 0.91,                      // VAD 非静音占比，chunk 紧凑度
  "silence_pad_ms_head": 80,                 // chunk 前缘静音，影响"喘息感"
  "silence_pad_ms_tail": 120,
  "loudness_lufs": -23.4                     // postprocess 响度归一参考
}
```

提取工具：
- `pitch_*` — [librosa.pyin](https://librosa.org) 或 [CREPE](https://github.com/marl/crepe)
- `emotion_*` — [emotion2vec](https://github.com/ddlBoJack/emotion2vec)，emotion_emb 留到 T3 存伴生 parquet
- `silence_pad_*` — 复用既有 Silero VAD 输出
- `loudness_lufs` — [pyloudnorm](https://github.com/csteinmetz1/pyloudnorm)
- `text_hash` — datasketch / 直接 `hashlib` + shingles

### 存储策略

embedding 不写 jsonl，避免主清单膨胀、保 grep-friendly：

```
datasets/<name>/
  manifest.jsonl              # 标量 + 短字符串
  embeddings.parquet          # chunk_id | speaker_emb | emotion_emb（T3 起）
  timestamps/<chunk_id>.json  # 词级时间戳（T3）
  report.md
```

`manifest_version` 字段一旦加字段就 bump，便于回归与回滚。被过滤掉的 chunk 写到 `dropped.jsonl` 分库治理，不污染主清单。

### 新增 / 改动模块

| 位置 | 变化 |
|---|---|
| `agents/style_agent.py` | **新增**。`(global_profile, sentence, prev_context) → StyleSpec`，Claude via LiteLLM |
| `core/tts.py` | 接 CosyVoice 2 instruct API，透传 `instruct_prompt` |
| `core/training/reference.py` | 新增 filter 模式：按 emotion/energy 过滤 manifest 后 top-k 选择 |
| `agents/dataset_agent.py` | manifest 扩字段：`emotion_tag, energy, pitch_mean, pace`（emotion2vec + 信号统计） |
| `core/book.py` | 支持 `[[style: ...]]` 内联标签解析 |
| `agents/synthesis_agent.py` | 串起：book → style_agent → reference selector → tts |

## Alternatives

- **用户逐句手标情绪 / 韵律**：表达力强但完全反产品直觉，弃。
- **完全交给 LLM 直出 SSML / 控制 token**：CosyVoice 2 对 SSML 支持不完整，且不同模型方言不同，绑死后期迁移痛苦；`instruct_prompt` 是 CosyVoice 2 原生且自然语言，泛化更好。
- **把 BGM 混进 TTS reference**：会污染音色和 speaker similarity 评估，且无法逐句独立调节，强烈否决。
- **省掉 LLM，规则匹配关键词 → emotion tag**：覆盖率低、上下文丢失（同一句"完了" 可以是绝望也可以是松一口气），不够。

## Consequences

### 正向
- 用户**只用自然语言描述一次**就能驱动整本书的风格控制
- 三类需求（音色 / 情绪 / 氛围）路径正交，调试和评估独立
- 与 [ADR-0002](0002-mvp-cosyvoice2.md) 的 reference selector、[ADR-0005](0005-ar-nonar-architecture.md) 的 NonAR emotion embedding 自然衔接 —— `StyleSpec` 是统一接口，M6 起把 `emotion` 字段同时喂给 emotion2vec encoder
- LLM 输出结构化 JSON，可缓存、可回放、可人审

### 负向 / 代价
- 增加一次 LLM 调用 / 句 —— 用 轻量模型 如Claude Haiku + prompt 缓存 + 段落级批处理控成本
- manifest 需要补情绪 / 能量字段，M1 dataset 阶段要追加一次重跑（数据量不大，可接受）
- 局部覆盖语法 `[[style: ...]]` 与未来其他内联标签（如 [[chapter: ...]]）需要统一命名空间，先约定为 `[[<key>: <value>]]`

### 后续需要观察
- LLM 上下文窗口策略：前 N 句滚动 / 章节级摘要 / 仅当前句 —— 影响连贯性与成本
- `instruct_prompt` 的稳定性：同一指令多次合成是否一致；不一致则需固定 seed 或回退到 reference-only
- 与 [ADR-0005](0005-ar-nonar-architecture.md) 的 emotion embedding 是否冗余：M6 时如果 embedding 已足够细粒度，`instruct_prompt` 可能降级为冗余通道

## References

- [docs/PLAN.md §3.B.1](../PLAN.md#L232-L241) tts 模块
- [docs/PLAN.md §3.B.2](../PLAN.md#L243-L255) reference selector
- [docs/PLAN.md §3.C.2](../PLAN.md#L267-L274) synthesis agent
- [ADR-0002](0002-mvp-cosyvoice2.md) CosyVoice 2 zero-shot
- [ADR-0005](0005-ar-nonar-architecture.md) AR + NonAR 双阶段
- CosyVoice 2 instruct mode: https://github.com/FunAudioLLM/CosyVoice
