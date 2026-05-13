# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 与 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

每次代码 / 规划 / 模型变更应追加条目，并尽量引用 ADR 编号。

## [Unreleased]

### Planned
- M2: CosyVoice 2 zero-shot 合成 + multi-reference selector
- M3: ADK evaluation framework + speaker-sim / WER / MOS-pred 回归脚本

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
