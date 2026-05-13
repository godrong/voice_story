# Roadmap

进度看板，与 [PLAN.md](PLAN.md) §4 里程碑对齐。每完成一项勾掉 `[ ]` → `[x]`，并同步更新 [CHANGELOG.md](CHANGELOG.md)。

---

## M0 — Scaffold ✅ (v0.0.1)

- [x] 目录结构 + pyproject.toml + .gitignore + README
- [x] PLAN / CHANGELOG / ROADMAP / 初始 5 份 ADR
- [x] git init + 首个 commit + tag

## M1 — 数据管线 ✅ (v0.1.0)

**目标**：任意源音频 → 干净 + 多样 + 带文本标注的训练 dataset

### 模块（PLAN §3.A）

- [x] `core/audio_io.py`：ffmpeg 封装（probe / to_standard_wav / load / save）
- [x] `core/sources/`：Source Protocol + LocalSource + KaggleSource
- [x] `agents/source_agent.py`：dispatcher
- [x] `core/separation.py`：Demucs v4 htdemucs（默认强制开，ADR-0008）
- [x] `core/vad.py`：Silero VAD + 3~15s 贪心打包
- [x] `core/speaker.py`：WeSpeaker embedding + 余弦相似度过滤
- [x] `core/asr.py`：双语 Whisper EN + FunASR ZH，langid 路由（ADR-0007）
- [x] `core/eval.py`：WADA-SNR + DNSMOS（ONNX）+ 削波检测
- [x] `agents/preprocess_agent.py`：分离 → VAD → speaker filter
- [x] `agents/dataset_agent.py`：过滤 + 多样性 + manifest.jsonl + report.md
- [x] `agents/root_agent.py`：Stage Protocol + build_m1_pipeline + run_pipeline
- [x] `cli.py`：typer 入口（ingest / dataset stats）
- [x] `tests/`：19 个 unit test 全部通过

### 待完成（M1 → v0.1.0 验收）

- [ ] 用 Kaggle Trump 数据集端到端跑通
- [ ] `datasets/trump/manifest.jsonl` ≥ 200 行
- [ ] 平均 DNSMOS-OVR > 3.5
- [ ] WER vs 数据集自带 transcription < 10%
- [ ] phoneme 覆盖 > 80%
- [ ] 装重 ML deps（demucs / silero-vad / faster-whisper / funasr / wespeaker）

## M2 — zero-shot 合成 (v0.1.1)

- [ ] `core/tts.py`：CosyVoice 2 推理封装（多参考 prompting）
- [ ] `agents/training_agent.py`：reference 选择（按多样性 + 情绪 / 韵律匹配）
- [ ] LLM reference selector（Claude via LiteLLM）
- [ ] 验收：speaker similarity > 0.75 baseline

## M3 — 评估闭环 (v0.1.2)

- [ ] `core/eval.py` 扩展：speaker_sim / WER / MOS-pred / 情绪一致性
- [ ] 固定测试集：5 段参考 × 20 测试句
- [ ] ADK evaluation framework 接入
- [ ] 回归 CI 脚本

## M4 — 朗读端到端 (v0.2.0)

- [ ] `core/book.py`：TXT / EPUB / PDF / MD ingestion
- [ ] 文本归一化（WeTextProcessing）
- [ ] `agents/synthesis_agent.py`：逐句合成 + 跨句韵律连续
- [ ] `agents/postprocess_agent.py`：响度归一 + 章节标记 + m4b
- [ ] 端到端：500 字短故事 → m4b

## M5 — LoRA 微调 (v0.3.0)

- [ ] 云训练平台选型（AutoDL / runpod / GCP）
- [ ] 训练数据 packaging
- [ ] CosyVoice 2 LoRA 训练脚本
- [ ] 微调权重加载 + 推理
- [ ] 验收：speaker similarity > 0.85

## M6 — AR + NonAR 双阶段 (v0.4.0)

- [ ] AR 主干 LoRA（停顿 / 连读 / 语速）
- [ ] NonAR refiner（Flow Matching decoder）独立训练
- [ ] Speaker embedding 后融合（ECAPA-TDNN / WavLM-XL）
- [ ] Emotion embedding (emotion2vec) 条件控制
- [ ] 验收：四指标全达标

## M7 — 流式 + 生产化 (v1.0.0)

- [ ] 流式接口完整实现
- [ ] URL 下载子模块（yt-dlp）
- [ ] Web / CLI 双入口
- [ ] 部署到 Cloud Run / Agent Engine
- [ ] 文档完整
