# ADR-0006: Kaggle 作为内置 source plugin（扩展 ADR-0003）

- **Date**: 2026-05-13
- **Status**: Accepted
- **Extends**: ADR-0003

## Context

ADR-0003 把"源采集"在 MVP 阶段限定为本地文件，主要为了避开 B 站 API 鉴权 / 反爬等"产品级"复杂度。但 Week 1 的测试输入选定为 Kaggle 数据集 `etaifour/trump-speeches-audio-and-word-transcription`，并且：

- **kagglehub 是受控官方 API**：鉴权一次（`~/.kaggle/kaggle.json` 或环境变量），后续调用稳定，无反爬 / 限流痛点
- 数据集自带本地缓存，重复调用幂等
- 一行调用就能拿到本地路径：`kagglehub.dataset_download(...)`

引入 Kaggle source 不会复活 ADR-0003 想避开的复杂度，反而让"测试 / 演示"路径更顺畅。

## Decision

在 `core/sources/` 中新增 `KaggleSource`，注册进 `core.sources.REGISTRY` 让 CLI 通过 `--source kaggle --dataset-id <id>` 调用。本 ADR **扩展**而非 supersede ADR-0003：本地源仍是默认推荐路径，Kaggle 是受控的第二个源。

CLI 形如：
```bash
voice-story ingest --source kaggle \
  --dataset-id etaifour/trump-speeches-audio-and-word-transcription \
  --name trump
```

## Alternatives

- **继续只支持本地，让用户手动 `kaggle datasets download`**：可行但增加用户步骤，且每次重新下载浪费带宽（kagglehub 自带缓存）
- **直接接 yt-dlp 的 URL 源**：B 站 / YouTube 鉴权复杂、反爬不可控，仍在 ADR-0003 排除范围内

## Consequences

### 正向
- 一行命令复现 Trump 测试数据集，便于演示与回归
- KaggleSource 实现 ~80 行，复杂度可控
- 鉴权 fail-fast，未配 token 时立刻给出可操作错误（指向 https://www.kaggle.com/settings/account）

### 负向 / 代价
- 多一个依赖 `kagglehub`（PyPI 官方包）
- 用户需要 Kaggle 账号 + API token（首次有上手成本）
- 数据集授权各异：当前 `KaggleSource` 只在 metadata 里记录授权说明，不强制校验，由用户自负责任

### 后续需要观察
- 如果用户真的有"从 B 站 / YouTube 下载"的高频诉求，再单独写 ADR-00XX 引入 yt-dlp 子模块，仍走"扩展 ADR-0003"模式

## References

- [docs/PLAN.md](../PLAN.md) §3.A.2
- [ADR-0003](0003-source-local-file-only.md)
- https://github.com/Kaggle/kagglehub
