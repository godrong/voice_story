# Voice Story — 个性化声音克隆讲故事系统

## Context

目标是构建一个**声纹克隆 + 长文本朗读**的系统：从一段音频源（B 站视频 / 音频链接）中提取目标主播的声纹，然后用这个声纹给一本书做"睡前故事"朗读。

对标产品是豆包的声纹复刻，但其有两个痛点要解决：
1. **声纹录入太粗糙**：只录一句话，覆盖音色/语调样本太少，导致相似度差。
2. **音色/情绪粒度不够**：单一参考样本无法支持长篇朗读时的情绪变化。

因此项目分两个核心优化方向（也对应用户的两个 agent 分工）：
- **源音频 → 可训练数据** 这条链路必须做精：去 BGM / 多人声分离 / 质量过滤 / 多样性采样。
- **训练 / 推理架构**：采用 **AR（自回归）主导内容生成 + NonAR（非自回归）做音色与情绪细粒度修饰**的双阶段架构，从 MVP（CosyVoice 2 zero-shot）逐步演进。

输出形态：MVP 短文本一次性合成本地音频，预留流式接口。

---

## 整体架构

### Agent 编排框架选型：**Google ADK**（推荐）

| 维度 | Google ADK | Claude Agent SDK |
|---|---|---|
| 工作流原语 | SequentialAgent / ParallelAgent / LoopAgent 原生匹配流水线 | 偏向 Claude 主控 + 工具调用，无 workflow 原语 |
| 非 LLM 计算 | BaseAgent 子类自由编排（Demucs/VAD/ASR/TTS 都是非 LLM 重计算） | 强依赖 Claude 调度，对纯批处理不够顺手 |
| LLM 选择 | 通过 LiteLLM 接 Claude / Gemini / OpenAI，仍可用 Claude 做"判断" | 锁定 Claude |
| 调试 | Dev UI 可视化每一步输入输出，对多阶段 pipeline 友好 | 主要靠日志 |
| 部署 | Cloud Run / Agent Engine 一键部署，便于后续上云训练 | 偏本地 |
| 评估 | 内建 trace + evaluation framework，便于跑 speaker-similarity 回归 | 无 |

**结论**：此项目大部分 stage 是非 LLM 的重计算（音频处理 / 模型训练 / 合成），LLM 仅在"质量判断 / 文本规整 / 异常诊断"上有用。ADK 的 workflow 原语 + 非 LLM agent 支持 + 评估框架是更顺手的选型。LLM 部分仍用 Claude（通过 LiteLLM）。

### 顶层 Pipeline

```
RootAgent (SequentialAgent)
├── SourceAgent           ─ 输入：URL/文件路径 → 输出：原始音频
├── PreprocessAgent       ─ 人声分离 / VAD / 说话人定位 / 增强
├── DatasetAgent          ─ ASR 转写 / 质量过滤 / 多样性采样
├── TrainingAgent         ─ MVP: zero-shot 参考构建；进阶: LoRA 微调
├── SynthesisAgent        ─ 书本切片 → 逐段合成
└── PostprocessAgent      ─ 响度归一 / 章节拼装 / 流式接口
```

---

## Stage 1：源音频 → 可训练数据（独立 Agent）

这是用户明确要做独立 agent 的部分。目标：**从任意源音频产出干净、多样、带文本标注的训练数据集**。

### 1.1 源采集（SourceAgent）

**MVP 简化方案**：只支持**本地音频文件**（mp3 / wav / m4a / flac / 视频文件）。

- 用户自己负责把 B 站视频 / 网络音频先下载下来（浏览器插件 / yt-dlp 命令行都行），扔到 `inputs/` 目录
- SourceAgent 用 `ffmpeg-python` 统一转码为 WAV 24kHz / 16-bit / mono
- 这一步逻辑极薄：探测格式 → ffmpeg 转码 → 输出标准化文件路径

**好处**：避开 B 站 API 鉴权 / 反爬 / 链接解析这些"产品级"复杂度，开发心智负担最小。

**URL 下载作为可选后置模块**（不在 MVP 中）：
- 后续可单独加一个 `core/downloader.py` 封装 `yt-dlp`（支持 B 站 / YouTube / 直链）
- 这样主 pipeline 不依赖网络可用性、不需要处理反爬，调试也更可控

输出：原始 WAV（24kHz / 16-bit / mono，统一规格化）。

### 1.2 预处理（PreprocessAgent）—— 关键优化点

按顺序处理：

1. **人声分离**：[Demucs v4 htdemucs_ft](https://github.com/adefossez/demucs) 提取干净 vocal stem，去掉 BGM / 音效。主播音频几乎都有背景音，这一步是相似度的关键瓶颈。
2. **VAD 切片**：[Silero VAD](https://github.com/snakers4/silero-vad) 切成 3~15 秒短句（CosyVoice 喜欢这个长度）。
3. **说话人定位**（多说话人场景）：
   - [pyannote.audio](https://github.com/pyannote/pyannote-audio) 做 diarization
   - 用户提供一段 5~10 秒目标说话人参考片段
   - [WeSpeaker / 3D-Speaker](https://github.com/wenet-e2e/wespeaker) 提取 embedding，按余弦相似度过滤目标说话人片段
4. **音频增强**（可选）：[DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) 去残留噪声；[Resemble Enhance](https://github.com/resemble-ai/resemble-enhance) 做 super-resolution
5. **质量过滤**：
   - SNR 估计（WADA-SNR）丢弃低信噪比片段
   - [DNSMOS](https://github.com/microsoft/DNS-Challenge) 给每段打 MOS 分，保留 ≥3.5
   - 峰值削波检测
6. **响度归一**：EBU R128 / pyloudnorm

### 1.3 数据集构建（DatasetAgent）

1. **ASR 转写**：[FunASR Paraformer-zh](https://github.com/modelscope/FunASR)（中文准确率优于 Whisper）+ 标点恢复
2. **置信度过滤**：丢弃 confidence < 0.85 的转写
3. **多样性采样**（解决"豆包只录一句话"的痛点）：
   - **拼音覆盖**：用 pypinyin 计算 dataset 的声韵母覆盖率，针对性补采
   - **韵律多样性**：依据 ASR 的标点分布筛选问句 / 陈述句 / 感叹句
   - **时长分布**：保证短句 / 中句 / 长句都有
   - **能量分布**：避免全是平静段或全是激昂段
4. **数据集元信息**：`{audio_path, text, duration, snr, mos, embedding}` 落入 manifest.jsonl

**输出**：一个训练 / 推理可用的 dataset 目录，含 metadata 和质量报告。

---

## Stage 2：训练 / 克隆 Agent —— 用户的核心优化目标

### 2.1 MVP（不训练，纯 zero-shot）

直接用 **CosyVoice 2-0.5B**（[FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice)）：
- 从 dataset 中按"多样性采样"选 3~5 段参考（不同情绪 / 韵律），合成时按文本语境匹配最佳参考
- 用 LLM（Claude）做 reference selector：输入"目标文本片段 + 参考片段的 ASR 文本和元信息"，输出最匹配的 reference

这一步在 Mac 上可直接跑（CosyVoice 2 有量化版，M2/M3 16GB 内存能跑）。

### 2.2 进阶（AR 主导 + NonAR 颗粒化）—— 用户提出的架构方向

```
text ──► [AR backbone]  ──► semantic tokens / coarse acoustic tokens
                                    │
                  reference audio ──┤
                                    ▼
                          [NonAR refiner]  ──► fine acoustic / waveform
                                    ▲
              style / emotion prompt ┘
```

- **AR 部分**：CosyVoice 2 的 LLM 主干（或 Spark-TTS / IndexTTS-2）负责内容 + 基础韵律。**做 LoRA 微调**适应目标说话人的说话习惯（停顿、连读、语速）。
- **NonAR 部分**：基于 Flow Matching 的 decoder（CosyVoice 2 自带，或换成 [F5-TTS](https://github.com/SWivid/F5-TTS) 的 NonAR refiner）做音色精修。在这里**单独引入 speaker embedding 控制 + emotion embedding 控制**，颗粒度细到逐句。

**训练优化项**：
1. **LoRA 微调**而非全参微调（30 分钟数据足够）。在云 GPU 上跑（A100 / 4090，约 1~2 小时）。
2. **多参考 prompting**：推理时同时喂 2~3 段不同情绪的参考，让模型自己融合（CosyVoice 2 支持）。
3. **Speaker embedding 后融合**：用 ECAPA-TDNN / WavLM-XL 提取目标 embedding，在 decoder 输入端 condition，强化音色相似度。
4. **情绪 embedding**：用 [emotion2vec](https://github.com/ddlBoJack/emotion2vec) 标注 dataset 的情绪向量，作为可控条件。

### 2.3 评估指标（CI 化）

每次训练 / 调参后跑：
- **Speaker similarity**：ECAPA-TDNN 提取 embedding，合成音频 vs 目标参考的余弦相似度（目标 > 0.85）
- **WER**：合成音频回灌 ASR 得到的字错率（目标 < 5%）
- **MOS-pred**：UTMOS / NISQA 自动 MOS 评分（目标 > 4.0）
- **情绪一致性**：emotion2vec 距离（针对带情感的句子）

这些跑在 ADK 的 evaluation framework 里，保证迭代可追踪。

---

## Stage 3：书本 → 朗读音频（SynthesisAgent + PostprocessAgent）

### 3.1 书本预处理

- 支持 TXT / EPUB（`ebooklib`）/ PDF（`pypdf`）/ Markdown
- 章节切分 → 段落切分 → 句子切分（保证每句 < 模型最大 token）
- 文本归一化：数字、英文、符号、日期（用 [WeTextProcessing](https://github.com/wenet-e2e/WeTextProcessing)）
- 章节标题特殊处理（停顿、加长）

### 3.2 合成

- **逐句合成** + **跨句韵律连续性**：CosyVoice 2 支持历史上下文 prompting
- **动态参考选择**：用 LLM 给当前句子打"情绪 / 韵律标签"，匹配 dataset 中最相似的参考片段
- **流式接口预留**：合成层封装成 async generator，MVP 走 collect-all，未来直接换 stream consumer

### 3.3 后处理

- 句间静音插入（300~500ms，章末更长）
- 响度归一（EBU R128 -16 LUFS for audiobook）
- 章节标记 → m4b 输出（chapters）
- 质量复检：逐句跑 speaker similarity，相似度低的自动重合成（最多 2 次）

---

## MVP 实施路径（建议顺序）

0. **Day 0 - 项目骨架**：`git init` + 创建 `docs/PLAN.md`（同步本文件） + `docs/CHANGELOG.md` + `docs/ROADMAP.md` + 写入初始 5 份 ADR（ADR-0001~0005） + pyproject.toml + .gitignore（屏蔽 datasets/models/inputs/outputs/books）。首个 commit `chore: project scaffold (v0.0.1)`。
1. **Week 1 - 数据管线**：实现 SourceAgent（本地文件版） + PreprocessAgent + DatasetAgent，目标是从一段本地音频文件跑出干净 dataset
2. **Week 2 - MVP 合成**：接 CosyVoice 2 zero-shot，用 dataset 中的参考合成一段短文本，跑 speaker similarity 拿到 baseline 数字
3. **Week 3 - 评估闭环**：搭 ADK evaluation，跑 baseline 指标
4. **Week 4 - 朗读完整链路**：接入 SynthesisAgent + PostprocessAgent，合成一章短故事
5. **后续迭代**：LoRA 微调 / AR+NonAR 架构升级 / 流式接口

---

## 关键文件结构（预期）

```
voice_story/
├── pyproject.toml                  # uv 管理依赖
├── README.md                       # 项目入口
├── docs/                           # ⭐ 规划/决策/版本文档（与代码一起 git 管理）
│   ├── PLAN.md                     # 本规划文件的项目内副本（每次大改同步）
│   ├── CHANGELOG.md                # 版本变更日志（语义化版本）
│   ├── ROADMAP.md                  # MVP → 进阶路径，勾选式 milestone
│   └── decisions/                  # ADR（Architecture Decision Record）目录
│       ├── 0001-framework-google-adk.md
│       ├── 0002-mvp-cosyvoice2.md
│       ├── 0003-source-local-file-only.md
│       └── ...                     # 每个关键技术决策单独一份，不可改只可补
├── agents/
│   ├── source_agent.py             # 源采集（本地文件版）
│   ├── preprocess_agent.py         # 人声分离 / VAD / 说话人
│   ├── dataset_agent.py            # ASR / 过滤 / 多样性
│   ├── training_agent.py           # MVP zero-shot + 进阶 LoRA
│   ├── synthesis_agent.py          # 书本朗读
│   ├── postprocess_agent.py        # 装配 / 流式
│   └── root_agent.py               # ADK SequentialAgent 编排
├── core/
│   ├── audio_io.py                 # ffmpeg 封装
│   ├── separation.py               # Demucs 封装
│   ├── vad.py                      # Silero VAD
│   ├── speaker.py                  # 说话人 embedding (WeSpeaker)
│   ├── asr.py                      # FunASR
│   ├── tts.py                      # CosyVoice 2 封装
│   └── eval.py                     # speaker sim / WER / MOS
├── inputs/                         # 用户放入的原始音频/视频（gitignore）
├── datasets/                       # 处理产出的 dataset（gitignore）
├── models/                         # 模型权重缓存（gitignore）
├── books/                          # 输入书本（gitignore）
└── outputs/                        # 合成音频（gitignore）
```

---

## 规划文档落地 + 版本追踪机制

目的：让每次代码迭代都能追溯到"为什么这么做"，避免规划脱离仓库。

### 文档与代码同仓共存

- `~/.claude/plans/b-harmonic-yeti.md`（本文件）= **草稿/工作空间**，与 Claude 协作时演进
- `voice_story/docs/PLAN.md` = **正式落地版**，每次规划稳定后从草稿同步过来，跟着代码走 git
- 两者保持一致由开发者手动同步（每次架构调整后 sync 一次即可），避免双向自动同步带来的冲突

### 版本号约定（SemVer）

- `0.1.x` MVP 阶段（数据管线 + CosyVoice 2 zero-shot）
- `0.2.x` 朗读完整链路打通
- `0.3.x` LoRA 微调能力上线
- `0.4.x` AR + NonAR 双阶段架构
- `1.0.0` 流式接口 + 评估指标全部达标

版本号写在 `pyproject.toml`，每次发布到 `git tag`。

### CHANGELOG.md 格式（Keep a Changelog）

每次有代码 / 规划 / 模型变更都追加条目：

```markdown
## [0.1.2] - 2026-05-15
### Added
- DatasetAgent 增加 pinyin 覆盖率统计
### Changed
- VAD 阈值从 0.5 调到 0.6（来自 ADR-0007）
### Fixed
- ffmpeg 转码时丢失 metadata
```

### ADR（架构决策记录）

每个**关键技术决策**单独一份 markdown，编号 + 不可改（只能用新 ADR supersede 旧的）。模板：

```markdown
# ADR-XXXX: <决策标题>
- Date: 2026-05-12
- Status: Accepted | Superseded by ADR-YYYY
- Context: 决策背景
- Decision: 决定做什么
- Alternatives: 考虑过的其他方案
- Consequences: 取舍代价
```

**初始 ADR 列表**（项目启动时就写好）：
- ADR-0001: 选择 Google ADK 作为编排框架（vs Claude Agent SDK）
- ADR-0002: MVP 使用 CosyVoice 2 zero-shot
- ADR-0003: 源采集仅支持本地文件，URL 下载延后
- ADR-0004: 训练侧使用 LoRA 而非全参微调
- ADR-0005: 进阶架构 AR 主干 + NonAR refiner

### 代码与决策的回链

- commit message 末尾带 `Refs: ADR-XXXX` 或 `Refs: PLAN#section`，使 git blame 能追到决策
- PR 模板包含 "影响的 ADR / 是否需要新 ADR" 勾选
- CHANGELOG 条目尽量引用 ADR 编号

### Roadmap（勾选式）

`docs/ROADMAP.md` 用 GitHub-flavored checkbox 列出 milestone，每完成一个勾掉，作为对外可读的进度看板：

```markdown
## MVP (v0.1)
- [x] SourceAgent 本地文件加载
- [ ] PreprocessAgent: Demucs 人声分离
- [ ] DatasetAgent: FunASR 转写
- [ ] zero-shot 合成 demo
...
```

---

## 验证方法（端到端）

1. **数据管线验证**：放一个测试音频文件到 `inputs/` → 检查 `datasets/<speaker>/manifest.jsonl` 包含 ≥20 段，平均 MOS > 3.5，pinyin 覆盖率 > 80%
2. **MVP 合成验证**：用 dataset 跑 CosyVoice 2 zero-shot 合成测试句 → speaker similarity > 0.75（zero-shot 基线）
3. **朗读完整链路**：取一段 500 字短故事 → 端到端产出 m4b → 人耳验证 + speaker similarity 平均值
4. **回归测试**：固定一组测试句 + 参考集，每次架构变更后跑 eval，看四个指标曲线
5. **流式接口预演**：用 mock generator 验证 SynthesisAgent 的 async 接口签名能正确驱动 PostprocessAgent

---

## 开放问题（待用户后续确认）

- 是否要支持英文 / 多语言（目前默认中文优先）
- 云训练用什么平台（AutoDL / runpod / GCP）—— 等到要做 LoRA 微调时再决定
- 书本格式优先级（EPUB / PDF / TXT）—— 影响 SynthesisAgent 的 ingestion 适配优先级
