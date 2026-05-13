# Voice Story — 个性化声音克隆讲故事系统

> 本文件按 **需求 → 架构 → 模块 → 里程碑** 顺序组织。每个模块是一张自包含的卡片，可独立读、可被里程碑引用。开发不按周/天排期，按模块完成度推进。

---

## 1. 需求

### 1.1 用户场景

> "我想克隆某个主播的声音（来源：B 站视频 / 音频链接 / 本地文件），让 TA 用我喜欢的音色给我读一本书的睡前故事。"

### 1.2 痛点（对标豆包声纹复刻）

| 痛点 | 现象 |
|---|---|
| P1 — 录入太粗糙 | 只录一句话，音色 / 语调 / 情绪样本覆盖太少，相似度差 |
| P2 — 情绪粒度不够 | 单一参考无法支持长篇朗读的情绪起伏、句间张力 |
| P3 — 源音频脏 | 主播音频带 BGM / 音效 / 多人声，直接训练会"染色" |
| P4 — 长文本稳定性 | 长篇合成时累积漏字 / 重复 token / 音色漂移 |

### 1.3 目标 / 非目标

**目标**
- 给定任意音频源，自动产出**高质量、多样性、带文本标注**的训练数据集
- 用该数据集做 zero-shot → LoRA → AR+NonAR 三段式演进的声音克隆
- 把一本书朗读成可播放的音频文件（睡前故事场景）

**非目标（明确不做）**
- 不做实时低延迟语音转换（VC），只做 TTS
- 不做歌唱合成
- 不做声纹诈骗 / 仿冒等敏感场景的鉴别

### 1.4 验收门槛（v1.0.0 全部达标）

| 指标 | 阈值 | 测量方法 |
|---|---|---|
| Speaker Similarity | > 0.85 | ECAPA-TDNN embedding 余弦相似度 |
| WER | < 5% | 合成音频回灌 ASR |
| MOS-pred | > 4.0 | UTMOS / NISQA 自动评分 |
| 情绪一致性 | distance < threshold | emotion2vec 距离 |
| 首字延迟（流式） | < 500ms | 流式接口首 chunk 时间 |

---

## 2. 架构

### 2.1 需求 → 架构映射

| 需求 / 痛点 | 架构方案 |
|---|---|
| P1（录入粗糙） | 数据管线（分离 + 切片 + 多样性采样）+ 多参考 prompting |
| P2（情绪粒度） | AR 主干 + NonAR refiner 双阶段，NonAR 端注入 speaker / emotion embedding |
| P3（源音频脏） | Demucs 强制人声分离 + 说话人 embedding 过滤 |
| P4（长稳定性） | 句级切分合成 + 跨句上下文 prompting + 低相似度自动重合成 |
| 任意源（URL / 本地） | Source plugin 架构（local / kaggle / future yt-dlp） |
| 双语 | 双后端 ASR（Whisper EN / FunASR ZH）+ 语言检测路由 |

### 2.2 数据流

```
[ 源音频 ]                                    [ 书本 ]
   │                                            │
   ▼                                            ▼
audio_io ── standardize (24kHz/16-bit/mono)   book (split chapters/sentences + 文本归一化)
   │                                            │
   ▼                                            │
separation (Demucs vocal stem)                  │
   │                                            │
   ▼                                            │
vad (3~15s chunks)                              │
   │                                            │
   ▼                                            │
speaker (target-only filter, optional)          │
   │                                            │
   ▼                                            │
asr (Whisper / FunASR, language-routed)         │
   │                                            │
   ▼                                            │
eval.quality + dataset.diversity                │
   │                                            │
   ▼                                            ▼
[ training-ready dataset ]  ◄────────►  synthesis (tts: ref + text → audio)
   │                                            │
   ▼                                            ▼
training (zero-shot → LoRA → AR+NonAR)     postprocess (loudness, m4b, streaming)
                                                │
                                                ▼
                                        [ 朗读音频 ]
```

### 2.3 顶层 agent 编排

```
RootAgent (ADK SequentialAgent)
├── SourceAgent       ─ 源采集
├── PreprocessAgent   ─ 分离 → 切片 → 说话人过滤
├── DatasetAgent      ─ ASR → 质量过滤 → 多样性采样 → manifest
├── TrainingAgent     ─ zero-shot / LoRA / AR+NonAR
├── SynthesisAgent    ─ 书本切片 → 逐句合成
└── PostprocessAgent  ─ 装配 → 流式
```

### 2.4 框架选型：Google ADK

| 维度 | Google ADK | Claude Agent SDK |
|---|---|---|
| 工作流原语 | Sequential / Parallel / Loop 原生 | 偏 Claude 主控，无 workflow 原语 |
| 非 LLM 计算 | BaseAgent 自由编排 | 强依赖 Claude 调度 |
| LLM 选择 | LiteLLM 接 Claude / Gemini / OpenAI | 锁定 Claude |
| 调试 | Dev UI 可视化每步 | 主要靠日志 |
| 部署 | Cloud Run / Agent Engine | 偏本地 |
| 评估 | 内建 trace + eval framework | 无 |

结论：本项目以非 LLM 重计算为主，ADK 更合身。LLM 用 Claude（via LiteLLM）做"判断"环节。详见 [ADR-0001](../voice_story/docs/decisions/0001-framework-google-adk.md)。

### 2.5 TTS 双阶段架构（v0.4 起）

```
text ──► [AR backbone]  ──► semantic / coarse acoustic tokens
                                  │
              reference audio ──┤
                                  ▼
                        [NonAR refiner]  ──► fine acoustic / waveform
                                  ▲
            speaker / emotion ──┘
```

- **AR**：CosyVoice 2 LLM 主干，LoRA 微调适应说话习惯
- **NonAR**：Flow Matching decoder，注入 ECAPA / emotion2vec embedding 做音色 + 情绪细粒度控制

详见 [ADR-0005](../voice_story/docs/decisions/0005-ar-nonar-architecture.md)。

---

## 3. 模块设计

> 每个模块卡片格式：**职责 / 输入 / 输出 / 关键文件 / 复用 / 关键决策**。
> 模块按数据流分组（A 数据管线 → B 合成训练 → C 朗读链路 → D 编排入口）。

### 3.A 数据管线（参与 M1）

#### 3.A.1 `audio_io` — 通用音频转码

- **职责**：把任意输入格式（mp3/m4a/flac/wav/mp4/mkv/...）统一转成 24kHz / 16-bit / mono WAV
- **输入**：任意路径
- **输出**：标准化 WAV 路径 + 元信息（duration / orig_sr / orig_format）
- **关键文件**：`core/audio_io.py`
- **复用**：[ffmpeg-python](https://github.com/kkroening/ffmpeg-python) + 系统 ffmpeg
- **决策**：标准化为 24kHz 与 CosyVoice 2 训练采样率对齐

#### 3.A.2 `sources` — 源采集（插件化）

- **职责**：把"源"抽象成 Protocol（`fetch() -> Iterable[Path]`），支持多后端
- **输入**：source_type + params（dataset_id / 本地目录 / URL 等）
- **输出**：原始音频文件路径流 + source metadata（lang_hint / needs_separation / license）
- **关键文件**：
  - `core/sources/__init__.py`（Protocol）
  - `core/sources/local.py`（扫 `inputs/<name>/`）
  - `core/sources/kaggle.py`（包 `kagglehub.dataset_download`）
  - `agents/source_agent.py`（dispatcher）
- **复用**：[kagglehub](https://pypi.org/project/kagglehub/)
- **决策**：
  - [ADR-0003](../voice_story/docs/decisions/0003-source-local-file-only.md)（本地文件起步）
  - **待补 ADR-0006**：Kaggle 作为内置 source，扩展 ADR-0003（kagglehub 是受控 API，无反爬痛点）
  - **未来**：`yt-dlp` 子模块独立加，不放进 v1.0 之前的关键路径

#### 3.A.3 `separation` — 人声分离

- **职责**：剥离 BGM / 音效，输出干净 vocal stem
- **输入**：标准 WAV
- **输出**：vocal WAV（同采样率）
- **关键文件**：`core/separation.py`
- **复用**：[Demucs v4 htdemucs](https://github.com/adefossez/demucs)（默认非 _ft，速度优先）；MPS 加速 fallback CPU
- **决策**：**默认强制开启**（即便干净音频也走，保证 pipeline 一致性）→ **待补 ADR-0008**

#### 3.A.4 `vad` — 切片

- **职责**：按静音区把长音频切成 3~15s 短句（CosyVoice 偏好长度）
- **输入**：vocal WAV
- **输出**：chunks list + `chunks/index.jsonl`（chunk_id / source_file / start / end）
- **关键文件**：`core/vad.py`
- **复用**：[Silero VAD](https://github.com/snakers4/silero-vad)
- **决策**：边界尽量落在静音区，避免硬切单词

#### 3.A.5 `speaker` — 说话人定位（可选）

- **职责**：多人场景下保留目标说话人 chunk
- **输入**：chunks + 目标说话人参考片段（5~10s）
- **输出**：过滤后的 chunks
- **关键文件**：`core/speaker.py`
- **复用**：[WeSpeaker / 3D-Speaker](https://github.com/wenet-e2e/wespeaker) 提 embedding + 余弦相似度
- **决策**：单人源（如演讲）跳过此步，由 source metadata `is_single_speaker` 控制

#### 3.A.6 `asr` — 双语转写

- **职责**：chunk → text + word timestamps + confidence
- **输入**：chunk WAV
- **输出**：`{text, lang, confidence, word_timestamps}`
- **关键文件**：`core/asr.py`
- **复用**：
  - [faster-whisper](https://github.com/SYSTRAN/faster-whisper) large-v3（EN）+ langid 检测
  - [FunASR Paraformer-zh](https://github.com/modelscope/FunASR) + ct-punc 标点恢复（ZH）
- **决策**：langid 自动路由 → **待补 ADR-0007**

#### 3.A.7 `eval` — 质量评估（含训练阶段也用）

- **职责**：给 chunk / 合成音频打质量分（数据管线用 SNR/MOS，训练阶段用 speaker sim/WER/MOS-pred）
- **输入**：音频 + 可选参考音频
- **输出**：`{snr, dnsmos.{ovr,sig,bak}, speaker_sim, wer, mos_pred, emo_distance}`
- **关键文件**：`core/eval.py`（quality / similarity / mos / wer 子模块）
- **复用**：
  - WADA-SNR 算法
  - [DNSMOS](https://github.com/microsoft/DNS-Challenge)（onnxruntime）
  - ECAPA-TDNN（数据管线 + 训练共用）
  - [UTMOS](https://github.com/sarulab-speech/UTMOSv2) / NISQA（训练阶段）
- **决策**：所有质量指标进 ADK evaluation framework，保证可回归

#### 3.A.8 `dataset` — 多样性采样 + manifest

- **职责**：过滤低质量 chunk，做覆盖均衡，输出 training-ready manifest
- **输入**：chunks + (text, snr, mos) 元信息
- **输出**：`datasets/<name>/manifest.jsonl` + `report.md`
- **关键文件**：`agents/dataset_agent.py`
- **复用**：
  - [pypinyin](https://pypi.org/project/pypinyin/)（中文声韵母覆盖）
  - [pronouncing](https://pypi.org/project/pronouncing/)（英文 CMU phoneme 覆盖）
- **决策**：过滤门槛 MOS≥3.5 / SNR≥15dB / confidence≥0.85 / 无 clipping；多样性维度：phoneme + 时长 + 能量 + 韵律（问/陈/感）

### 3.B 合成与训练（参与 M2 / M5 / M6）

#### 3.B.1 `tts` — 合成引擎

- **职责**：text + reference → 合成音频
- **输入**：text, reference_audio(s), [speaker_emb, emotion_emb]
- **输出**：合成 WAV
- **关键文件**：`core/tts.py`
- **复用**：[CosyVoice 2](https://github.com/FunAudioLLM/CosyVoice)（M2/M5 用 0.5B，M6 上 AR+NonAR）；可选 [F5-TTS](https://github.com/SWivid/F5-TTS) 作 NonAR refiner
- **决策**：
  - [ADR-0002](../voice_story/docs/decisions/0002-mvp-cosyvoice2.md)（MVP zero-shot）
  - [ADR-0005](../voice_story/docs/decisions/0005-ar-nonar-architecture.md)（双阶段）

#### 3.B.2 `training` — 训练 agent

- **职责**：从 dataset 选 reference / 训 LoRA / 注入 embedding
- **输入**：manifest + 目标文本 / 训练配置
- **输出**：
  - M2：reference 选择结果（不训练）
  - M5：LoRA 权重包
  - M6：LoRA + speaker/emotion adapter 权重
- **关键文件**：`agents/training_agent.py` + `core/training/{reference,lora,adapter}.py`
- **复用**：[peft](https://github.com/huggingface/peft) for LoRA、[ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN) / [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm)、[emotion2vec](https://github.com/ddlBoJack/emotion2vec)
- **决策**：
  - [ADR-0004](../voice_story/docs/decisions/0004-lora-finetuning.md)（LoRA 而非全参）
  - Reference selector 用 Claude（via LiteLLM）匹配情绪 / 韵律

### 3.C 朗读链路（参与 M4）

#### 3.C.1 `book` — 书本预处理

- **职责**：任意格式书本 → 章 / 段 / 句的归一化文本流
- **输入**：TXT / EPUB / PDF / Markdown
- **输出**：句子流 + 章节标记
- **关键文件**：`core/book.py`
- **复用**：[ebooklib](https://github.com/aerkalov/ebooklib)、[pypdf](https://github.com/py-pdf/pypdf)、[WeTextProcessing](https://github.com/wenet-e2e/WeTextProcessing)（中文数字 / 英文 / 符号归一化）

#### 3.C.2 `synthesis` — 朗读合成

- **职责**：逐句调 tts + 跨句韵律连续 + 动态参考选择
- **输入**：句子流 + dataset reference 集
- **输出**：每句的合成 WAV + metadata
- **关键文件**：`agents/synthesis_agent.py`
- **复用**：tts 模块 + LLM 情绪 / 韵律标注（reference selector）
- **决策**：合成层封装成 async generator，MVP collect-all 跑通，流式接口在 M7 直接接 consumer

#### 3.C.3 `postprocess` — 后处理 / 装配

- **职责**：响度归一 / 静音插入 / 章节标记 / 输出 m4b / 低质量自动重合成
- **输入**：句子级合成 WAV 序列
- **输出**：完整 m4b 或 mp3
- **关键文件**：`agents/postprocess_agent.py`
- **复用**：[pyloudnorm](https://github.com/csteinmetz1/pyloudnorm)（EBU R128 -16 LUFS）、ffmpeg metadata（chapter markers）
- **决策**：逐句跑 speaker similarity，<阈值自动重合成（≤2 次）

### 3.D 编排与入口（跨 milestone）

#### 3.D.1 `root_agent` — ADK 编排

- **职责**：把 Source / Preprocess / Dataset / Training / Synthesis / Postprocess 串成 SequentialAgent
- **关键文件**：`agents/root_agent.py`、`agents/preprocess_agent.py`
- **复用**：[google-adk](https://google.github.io/adk-docs/)

#### 3.D.2 `cli` — typer 入口

- **职责**：命令行入口
- **关键文件**：`cli.py`
- **命令**：
  - `voice-story ingest --source {local|kaggle} [...] --name <speaker>`
  - `voice-story dataset stats --name <speaker>`
  - `voice-story synthesize --speaker <name> --book <path> --out <m4b>`

---

## 4. 里程碑

> 不按时间排，按模块完成度。每完成一个 milestone 发一个版本号、写 tag、勾 ROADMAP。

| M | 名称 | 涉及模块 | 验收门槛 | 版本 | 状态 |
|---|---|---|---|---|---|
| M0 | scaffold | docs / pyproject / git | git init + 初始 5 份 ADR | v0.0.1 | ✅ |
| M1 | 数据管线 | 3.A.1~3.A.8 + 3.D.1 + 3.D.2(ingest) | ≥200 chunks / DNSMOS-OVR>3.5 / WER<10% / phoneme>80% | v0.1.0 | 🚧 |
| M2 | zero-shot 合成 | 3.B.1 + 3.B.2(reference) | speaker sim > 0.75 baseline | v0.1.1 |  |
| M3 | 评估闭环 | 3.A.7 扩展（speaker / WER / MOS-pred）+ ADK eval framework | 四指标 CI 化、回归脚本固化 | v0.1.2 |  |
| M4 | 朗读端到端 | 3.C.1~3.C.3 + 3.D.2(synthesize) | 500 字短故事 → m4b，平均 sim > 0.75 | v0.2.0 |  |
| M5 | LoRA 微调 | 3.B.2 LoRA 路径 + 云训练 | speaker sim > 0.85 | v0.3.0 |  |
| M6 | AR+NonAR | 3.B.1 双阶段升级 | 四指标全达标 | v0.4.0 |  |
| M7 | 流式 + 生产化 | 3.C.3 流式 + `core/downloader` URL 源 + 部署 | 首字延迟 < 500ms | v1.0.0 |  |

### M1 验收的具体跑法（当前重点）

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[preprocess,asr,adk,dev]"

# Kaggle 鉴权：~/.kaggle/kaggle.json（或 KAGGLE_USERNAME/KAGGLE_KEY）
voice-story ingest \
  --source kaggle \
  --dataset-id etaifour/trump-speeches-audio-and-word-transcription \
  --name trump

voice-story dataset stats --name trump
```

验收：
- `datasets/trump/manifest.jsonl` ≥ 200 行
- 平均 DNSMOS-OVR > 3.5
- 我们的 ASR 输出 vs 数据集自带 word-level transcription，WER < 10%
- phoneme 覆盖 > 80%
- 随机抽 5 段听感复检通过

---

## 5. 文档与版本追踪

### 5.1 文档分布

| 位置 | 角色 |
|---|---|
| `~/.claude/plans/b-harmonic-yeti.md` | **草稿**：与 Claude 协作中演进 |
| `docs/PLAN.md` | **正式版**：每次规划稳定后从草稿 sync 过来，进 git |
| `docs/CHANGELOG.md` | Keep a Changelog 格式，每次代码 / 规划变动追加 |
| `docs/ROADMAP.md` | 勾选式 milestone 看板（与本文件第 4 节同步） |
| `docs/decisions/*.md` | ADR：单决策一份，编号、不可改，supersede 用新 ADR |

### 5.2 SemVer 规则

| 区间 | 含义 |
|---|---|
| 0.0.x | scaffold |
| 0.1.x | M1（数据管线）+ M2（zero-shot）+ M3（eval） |
| 0.2.x | M4（朗读端到端） |
| 0.3.x | M5（LoRA） |
| 0.4.x | M6（AR+NonAR） |
| 1.0.0 | M7（流式 + 生产化） |

### 5.3 ADR 现状与待补

| ID | 标题 | 状态 |
|---|---|---|
| 0001 | 选择 Google ADK | ✅ |
| 0002 | MVP CosyVoice 2 zero-shot | ✅ |
| 0003 | 源采集仅本地文件 | ✅ |
| 0004 | LoRA 而非全参微调 | ✅ |
| 0005 | AR+NonAR 双阶段架构 | ✅ |
| 0006 | Kaggle 作为内置 source（扩展 0003） | 待补（随 M1 落地） |
| 0007 | 双语 ASR 策略 | 待补（随 M1 落地） |
| 0008 | Demucs 默认强制开启 | 待补（随 M1 落地） |

### 5.4 Commit / PR 回链

- commit message 末尾带 `Refs: ADR-XXXX` 或 `Refs: PLAN#section`
- PR 模板含"是否需要新 ADR"勾选
- CHANGELOG 条目尽量引用 ADR 编号

---

## 6. 验证方法

| 阶段 | 检查项 |
|---|---|
| 模块单元 | 每个 `core/*` 模块带 `tests/test_<module>.py`，覆盖 happy path + 1 edge case |
| 数据管线（M1） | 见第 4 节 M1 验收 |
| 合成（M2） | 用 dataset reference 合成测试句 → speaker sim > 0.75 |
| 朗读端到端（M4） | 500 字短故事 → m4b → 人耳复检 + 平均 sim |
| 回归 | 固定 5 段参考 × 20 测试句，每次架构变更跑 eval，看四指标曲线 |
| 流式预演（M7 前） | mock async generator 验证 synthesis → postprocess 接口签名 |

---

## 7. 风险登记

| 风险 | 关联模块 | 缓解 |
|---|---|---|
| Kaggle API 鉴权 | sources.kaggle | README 加 setup 指引；启动前 fail-fast 检查 |
| Demucs 在 Mac 上慢（~3× 实时） | separation | 用 htdemucs 而非 htdemucs_ft；MPS 加速；可选抽样跑 |
| Whisper-large-v3 显存（~10GB） | asr | 自动 fallback medium；日志记录回退 |
| FunASR 首次模型下载（~1.5GB） | asr | 进度条 + 缓存到 `models/funasr/` |
| 数据集自带 transcription 格式未知 | dataset | 落地时先 inspect 目录，再写 parser |
| Demucs 总开启浪费算力 | separation | 接受；未来若卡瓶颈再加 `--skip-separation`，须新 ADR supersede 0008 |
| ADK 新框架坑（长任务 / trace 体积） | root_agent | 单元小步集成；trace 落本地 sqlite 备查 |
| CosyVoice 2 长文本重复 / 漏字 | tts / synthesis | 句级切分 + 重合成机制；记录失败率 |
| LoRA 过拟合 | training | rank 调参 + holdout 测试集 |

---

## 8. 开放问题（待用户后续确认）

- 多语言扩展边界（目前 EN + ZH，是否覆盖日 / 韩等？）
- 云训练平台选型（AutoDL / runpod / GCP）—— M5 启动前决定
- 书本格式优先级（EPUB / PDF / TXT）—— 影响 `book` 模块的 ingestion 适配顺序
- 部署形态（CLI only / Web UI / 移动端）—— M7 前决定
