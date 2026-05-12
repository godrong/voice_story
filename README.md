# Voice Story

> 个性化声音克隆 + 长文本朗读系统：把主播的声纹复刻到一本书上，给你讲睡前故事。

## 状态

- 版本：`0.0.1`（scaffold）
- 阶段：Day 0，项目骨架已建立。
- 详细规划见 [docs/PLAN.md](docs/PLAN.md)。

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

## 快速开始

```bash
# 用 uv 管理（推荐）或 pip
uv venv && source .venv/bin/activate
uv pip install -e ".[preprocess,asr,adk,dev]"

# 放一段音频到 inputs/，运行 pipeline
# （命令行入口待实现 - 见 ROADMAP）
```

## 协作规范

- 每个**关键技术决策**写一份 ADR（`docs/decisions/`），编号、不可改，只能用新 ADR supersede 旧的
- commit message 末尾带 `Refs: ADR-XXXX` 或 `Refs: PLAN#section` 让 git blame 可追溯
- 每次发版更新 `pyproject.toml` 的 version + `docs/CHANGELOG.md` + `git tag`
