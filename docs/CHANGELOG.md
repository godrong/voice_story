# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 与 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

每次代码 / 规划 / 模型变更应追加条目，并尽量引用 ADR 编号。

## [Unreleased]

### Planned
- M2: CosyVoice 2 zero-shot 合成 + multi-reference selector
- M3: ADK evaluation framework + speaker-sim / WER / MOS-pred 回归脚本

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
