# Roadmap

进度看板。每完成一项勾掉 `[ ]` → `[x]`，并同步更新 [CHANGELOG.md](CHANGELOG.md)。

---

## v0.0.x — Scaffold ✅

- [x] 目录结构 + pyproject.toml + .gitignore + README
- [x] PLAN / CHANGELOG / ROADMAP / 初始 ADR
- [x] git init + 首个 commit

## v0.1.x — 数据管线 + MVP zero-shot 合成

**目标**：本地音频文件 → 干净 dataset → CosyVoice 2 zero-shot 合成短句

### Week 1 — 数据管线

- [ ] `core/audio_io.py`：ffmpeg 封装，输入任意音视频 → 标准 WAV (24kHz/16-bit/mono)
- [ ] `agents/source_agent.py`：本地文件加载 + 格式校验
- [ ] `core/separation.py`：Demucs v4 (htdemucs_ft) 人声分离
- [ ] `core/vad.py`：Silero VAD 切片（3~15 秒）
- [ ] `core/speaker.py`：WeSpeaker embedding + 余弦相似度过滤
- [ ] `agents/preprocess_agent.py`：编排上述步骤
- [ ] `core/asr.py`：FunASR Paraformer-zh 转写 + 标点恢复
- [ ] `core/eval.py`（quality 部分）：SNR / DNSMOS 打分
- [ ] `agents/dataset_agent.py`：质量过滤 + 多样性采样（拼音覆盖、韵律、时长、能量）
- [ ] manifest.jsonl 输出与质量报告

### Week 2 — MVP 合成

- [ ] `core/tts.py`：CosyVoice 2 推理封装（zero-shot 多参考 prompting）
- [ ] `agents/training_agent.py`：从 dataset 选 3~5 段参考片段（按多样性）
- [ ] LLM-based reference selector（用 Claude 通过 LiteLLM）
- [ ] 端到端 demo：本地音频 → dataset → 合成一段测试句
- [ ] 验收：speaker similarity > 0.75（zero-shot baseline）

### Week 3 — 评估闭环

- [ ] `core/eval.py`（speaker / WER / MOS-pred）
- [ ] 测试集固定：5 段参考 × 20 句测试文本
- [ ] ADK evaluation framework 接入
- [ ] 回归测试 CI 脚本

## v0.2.x — 朗读完整链路

- [ ] 书本 ingestion：TXT / EPUB / PDF
- [ ] 文本归一化（WeTextProcessing）
- [ ] `agents/synthesis_agent.py`：逐句合成 + 跨句韵律连续
- [ ] `agents/postprocess_agent.py`：响度归一 + 章节标记 + m4b 输出
- [ ] 流式接口（async generator）
- [ ] 端到端：500 字短故事 → m4b

## v0.3.x — LoRA 微调

- [ ] 云训练环境选型（AutoDL / runpod / GCP）
- [ ] 训练数据 packaging
- [ ] CosyVoice 2 LoRA 训练脚本
- [ ] 微调权重加载 + 推理
- [ ] 验收：speaker similarity > 0.85

## v0.4.x — AR + NonAR 双阶段架构

- [ ] AR 主干 LoRA 微调（停顿 / 连读 / 语速）
- [ ] NonAR refiner（Flow Matching decoder）独立训练
- [ ] Speaker embedding 后融合（ECAPA-TDNN / WavLM-XL）
- [ ] Emotion embedding (emotion2vec) 条件控制
- [ ] 验收：四项指标全达标（speaker sim > 0.85 / WER < 5% / MOS > 4.0 / 情绪一致）

## v1.0.0 — 生产就绪

- [ ] 流式接口完整实现
- [ ] URL 下载子模块（yt-dlp 封装，支持 B 站）
- [ ] Web / CLI 双入口
- [ ] 部署到 Cloud Run / Agent Engine
- [ ] 文档完整
