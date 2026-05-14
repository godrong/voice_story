# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 与 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

每次代码 / 规划 / 模型变更应追加条目，并尽量引用 ADR 编号。

## [Unreleased]

### Planned
- M2 续: multi-reference selector (按 manifest v1.1 emotion / energy / pitch 选 top-k)
- M2 续: synthesis_agent (book sentence → TTS 串联)
- M3: ADK evaluation framework + speaker-sim / WER / MOS-pred 回归脚本

## [0.1.4] - 2026-05-14

### Added — TTS 文本规范化（B 层，朗读专用）
- **core/text_norm.py**: 朗读前的文本清洗模块
  - Unicode pass: 弯引号 ('"') → ASCII，em/en 破折号 → '-'，省略号 (…) → "..."，
    零宽字符 / NBSP / BOM 删除，多空白合并
  - 英文缩写展开: `he's → he is` / `couldn't → could not` 等约 30 条；
    优先用 `contractions` 库（已装 ai_study），缺失时退到内置 fallback 表
  - `normalize_for_tts(text, lang="en"|"zh")` 一站式入口
  - 中文路径只跑 unicode 步（中文无英文式缩写）

### Changed
- **core/tts.py**: `LocalSubprocessTTS.synthesize()` 与 `TTSBackend` Protocol
  新增 `normalize=True`（默认）+ `lang="en"`（默认）两个 kwarg；
  text 与 prompt_text 都自动过 `normalize_for_tts`。
  设 `normalize=False` 走原始文本（用于实验对比 / 测试模型原生处理力）

### Why
- exp 002 round 2 听感反馈：CosyVoice 2 英文缩写处理（`he's` / `couldn't`）
  G2P 对但韵律生硬。展开成完整形式让模型多一个音节做韵律布置，发音更自然
- Trump 的 wangrong 测试文本含弯引号与 em-dash，TTS 模型对 unicode 标点
  G2P 偶发出错，统一 ASCII 化避免

### 不做（明确限定）
- 拼写纠错（漏字 typo 仍然原样朗读，由用户上游修）
- 数字 / 日期 / 缩略语展开（CosyVoice 2 frontend 自带，未崩前不补）
- 风格改写（C 层，会改语义，由 ADR-0010 的 style_agent 处理）

### Tests
- **tests/test_text_norm.py**: 16 个单测覆盖 unicode 各类替换、缩写展开
  （库 + 兜底两条路径）、语种路由、空输入、纯 ASCII 无操作等场景
- 所有 24 测试通过（16 text_norm + 8 既有 tts，无回归）

### Demo（用户的 wangrong 文本）
```
BEFORE: It's great to be back...I'm very disappointed that Wang Rong couldn't join us
AFTER:  It is great to be back...I am very disappointed that Wang Rong could not join us
```
9 处缩写展开 + 弯引号 / em-dash → ASCII。

Refs: PLAN#3.B.1

## [0.1.3] - 2026-05-14

### Added — M2 TTS 基础设施（zero-shot 第一次合成跑通）
- **core/tts.py**: `TTSBackend` Protocol（稳定接口，M5+ 切云只换实现）+
  `LocalSubprocessTTS` 实现：在 sibling `cosyvoice` conda env 跑长驻 worker，
  通过 stdin/stdout JSON-line 同步请求-响应
  - 跨 env 通信 / subprocess 模式选型见 chat history（同步长驻 worker
    + JSON-line + 文件传 audio，M2~M4 服役期）
  - 错误层级：`TTSError`（任务级）/ `TTSWorkerCrashed`（worker 崩溃）/
    `TimeoutError`（超时）
  - 握手容忍：宽容跳过 worker import 期间三方包写入 stdout 的杂讯
- **core/tts_worker.py**: 长驻 worker 本体，跑在 cosyvoice env
  - 启动一次加载 CosyVoice2-0.5B（~140s on Mac，模型 7.6GB）
  - 支持 zero_shot / cross_lingual / instruct 三种模式
  - `_PROTOCOL_STDOUT` 模块级缓存真 stdout，模型加载期间 sys.stdout 重定向
    到 stderr，防 modelscope / wetext / transformers 等三方包写 stdout 污染协议
  - stderr 走人类日志，与协议 stdout 完全隔离
- **tests/test_tts.py**: 8 个单测，用 stub worker 模拟所有协议路径
  （握手 / 任务成功 / 任务失败 / worker 崩溃 / 超时 / 缺 env python /
  缺 worker script / Protocol 形状）。无需真 CosyVoice 即可跑。

### Infra
- 新建 conda env `cosyvoice`（python 3.10），独立于 ai_study 避免
  torch 2.3.1 vs 2.9.1 ABI 冲突
- CosyVoice 仓 clone 到 `~/Documents/projects/CosyVoice`（与本项目同级）
- CosyVoice2-0.5B 权重 7.6GB 落到 `CosyVoice/pretrained_models/CosyVoice2-0.5B/`
  - modelscope 下了 18/20 文件后 llm.pt 卡 104 kB/s 失败
  - HF mirror + hf_transfer 4 分钟补完缺的两个（llm.pt 1.9GB + model.safetensors 942MB）

### Smoke run（zero-shot 第一次合成）
- 参考：Trump WEF 2018 前 10 秒 chunk（manifest v1.1 row 0）
- 目标文本："Hello world. This is a test of voice cloning using CosyVoice 2 zero shot. The quick brown fox jumps over the lazy dog."
- 输出：`outputs/smoke/trump_synth_v1.wav`，9.88s / 24kHz mono / RMS 0.026 / 不静音 / 无削波
- 时延：worker 启动 + 模型加载 140s（一次性），首句推理 29s
- 后续 M3 加 speaker similarity / WER 量化评估

Refs: PLAN#3.B.1, ADR-0010

## [0.1.2] - 2026-05-14

### Added — Manifest schema v1.1（M2 风格控制基线）
- **core/prosody.py**: 新模块，按 chunk 算 F0（librosa.pyin）/ LUFS（pyloudnorm）/ RMS / speech_ratio / pace / emotion2vec_plus_base top-1 标签。emotion 模型惰性加载，约 500MB
- **agents/state.py**: 新增 `ProsodyScore` dataclass，挂在 `PipelineState.prosody` 上
- **agents/dataset_agent.py**: manifest 升到 v1.1
  - 新字段（T1）：`manifest_version` / `start_sec` / `end_sec` / `duration_bucket` / `energy_bucket`（自适应） / `prosody_label` / `clipped` / `prev_chunk_id` / `next_chunk_id` / `text_hash` / `speaker_id`
  - 新字段（T2）：`emotion_tag` / `emotion_confidence` / `pitch_mean_hz` / `pitch_std_hz` / `energy_rms` / `loudness_lufs` / `speech_ratio` / `pace_units_per_sec`
  - 流水线改两遍：pass 1 算 raw 特征，pass 2 用 RMS 33/66 分位给 `energy_bucket` 自适应分桶（见 ADR-0011）
  - NaN / ±inf 序列化前归一到 `null`，保持 manifest JSON 合法
  - `report.md` 加 emotion 分布 + 当前 ingest 的 p33 / p66 阈值
- **agents/dataset_agent.py**: `_build_neighbor_index` 在同一 `source_file` 内按起点排序，O(1) 查 prev/next chunk id

### Added — 文档
- **ADR-0010**: 风格控制 — global profile + LLM 句级标注 + instruct prompt
- **ADR-0011**: Energy bucket 自适应分位数（取代固定阈值）

### Smoke run（trump_wef 前 6 chunks）
- kept 4/6（dropped 2 low_mos），manifest schema v1.1 字段全部填充
- pitch ~119–129Hz（Trump 男声音区一致），LUFS ~-33（Demucs vocal stem 典型量级）
- pace 3.8–4.75 syllables/sec（英语演讲正常区间），speech_ratio 0.7
- emotion: 全 neutral@1.0（正式演讲风格预期）
- p33/p66=0.0226/0.0227（小样本贴近，全量 234 chunks 时会拉开）
- 与旧 v1.0 manifest 比对，既有字段（text/confidence/duration/snr/MOS）数值完全一致

Refs: ADR-0010, ADR-0011, PLAN#3.A.8

## [0.1.1] - 2026-05-13

### Fixed
- `core/vad.py`: Silero VAD 仅支持 8k/16k 采样率，pipeline 标准音频 24k 之前会报 `ValueError`。在 VAD 调用前 resample 到 16k；chunk 切片仍用 24k 原始数据
- `core/eval.py`: DNSMOS sig_bak_ovr.onnx 期望 raw waveform `(1, samples)` 输入，原代码喂的是 mel-spectrogram `(1, 1, mel, frames)` 直接报 `InvalidArgument: rank 4 vs 2`。改为 9.01s @ 16k zero-pad / 截断的 raw waveform

### Changed
- 质量过滤阈值按 Demucs 后音频校准（ADR-0009）：
  - **WADA-SNR 退出过滤**，只作为 manifest 诊断字段（在 vocal stem 上失真）
  - `DEFAULT_MIN_MOS_OVR` 从 `3.5` 降到 `3.0`（Demucs artifact 系统扣 ~0.4 分）
  - `FilterThresholds.min_snr_db` 字段删除；`cli.py` 删除 `--min-snr` 标志
- `tests/test_dataset.py::test_filter_thresholds_defaults` 同步更新

### Added
- ADR-0009: 质量阈值按 Demucs 后音频校准

### Smoke run（30s Trump WEF 2018）
- 5 chunks，3 通过 / 2 因 low_mos 丢（OVR 2.92 / 2.94 < 3.0）
- 通过 chunks 平均 OVR 3.25，phoneme 覆盖 92.3%，Whisper confidence 平均 0.97

Refs: ADR-0009

## [0.1.0] - 2026-05-13

### Added — M1 数据管线（Nodes A + B + C）
- **core/audio_io.py**: ffmpeg 封装；`probe()` / `to_standard_wav()` / `load()` / `save()`，强制 24 kHz / 16-bit / mono
- **core/sources/**: 插件化源采集层
  - `Source` Protocol + `SourceMeta`（lang_hint / needs_separation / is_single_speaker / license）
  - `LocalSource`：扫 `inputs/<name>/`
  - `KaggleSource`：包装 `kagglehub.dataset_download`，鉴权 fail-fast
- **core/separation.py**: Demucs v4 (htdemucs) 封装；MPS / CUDA 自动选；惰性加载、幂等
- **core/vad.py**: Silero VAD + 贪心打包成 3~15s chunk；边界落静音区
- **core/speaker.py**: WeSpeaker embedding + 余弦相似度过滤
- **core/asr.py**: 双语 ASR——`Transcriber` 路由 Whisper-large-v3（EN，OOM 自动 fallback medium）/ FunASR Paraformer-zh（ZH，含 ct-punc 标点恢复）
- **core/eval.py**: 质量评估——WADA-SNR + DNSMOS（ONNX，自动下载）+ 削波检测；统一接口 `score_chunk()`
- **agents/state.py**: `PipelineState` + `ChunkInfo` / `TranscriptInfo` / `QualityScore` 三个 dataclass
- **agents/source_agent.py**: dispatcher，接 Source 插件 → 标准化输出到 `datasets/<name>/raw/`
- **agents/preprocess_agent.py**: 串联 Demucs → VAD → 可选 speaker filter
- **agents/dataset_agent.py**: ASR → 质量过滤 → 多样性采样 → 输出 `manifest.jsonl` + `report.md`
- **agents/root_agent.py**: 轻量 Stage Protocol + `build_m1_pipeline()` + `run_pipeline()`
- **cli.py**: typer 入口；`voice-story ingest` + `voice-story dataset stats`
- **tests/**: 19 个 unit test（audio_io / sources / dataset 纯函数），全部通过

### Added — 文档
- ADR-0006: Kaggle 作为内置 source（扩展 ADR-0003）
- ADR-0007: 双语 ASR 后端策略（Whisper EN + FunASR ZH，langid 路由）
- ADR-0008: Demucs 默认强制开启
- docs/PLAN.html: 单文件 HTML 版项目规划（带导航、模块卡、状态 pill）

### Changed
- `pyproject.toml`: 加 `kagglehub` 进 base deps；extras `preprocess` 加 `pronouncing` + `onnxruntime`，`asr` 加 `faster-whisper`
- `pyproject.toml`: 注册 `voice-story` 命令入口
- 项目规划重构：去掉 week/day 排期，按"需求 → 架构 → 模块 → 里程碑"组织（M0~M7）

Refs: PLAN#3.A, ADR-0006, ADR-0007, ADR-0008

## [0.0.1] - 2026-05-12

### Added
- 项目骨架：`docs/`、`agents/`、`core/`、`inputs/`、`datasets/`、`models/`、`books/`、`outputs/` 目录结构
- `pyproject.toml` 定义依赖与可选 extras
- `.gitignore` 屏蔽数据与模型文件
- `README.md` 项目入口
- `docs/PLAN.md` 完整技术规划（同步自 `~/.claude/plans/b-harmonic-yeti.md`）
- `docs/ROADMAP.md` 勾选式 milestone 看板
- `docs/CHANGELOG.md` 本文件
- `docs/decisions/ADR-0001` ~ `ADR-0005` 五份初始架构决策记录

Refs: PLAN#mvp-实施路径
