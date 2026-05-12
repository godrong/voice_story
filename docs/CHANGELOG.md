# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 与 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

每次代码 / 规划 / 模型变更应追加条目，并尽量引用 ADR 编号。

## [Unreleased]

### Planned
- SourceAgent (本地文件版)
- PreprocessAgent: Demucs 人声分离 + Silero VAD + 说话人定位
- DatasetAgent: FunASR 转写 + 质量过滤 + 多样性采样
- CosyVoice 2 zero-shot 合成 demo

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
