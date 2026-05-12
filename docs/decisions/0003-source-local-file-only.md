# ADR-0003: 源采集 MVP 阶段仅支持本地文件，URL 下载延后

- **Date**: 2026-05-12
- **Status**: Accepted

## Context

原始需求希望支持 B 站视频链接 / 任意音频 URL 作为输入。但实测引入这条路径会带来：
- B 站 API 鉴权 / 反爬 / 限流（cookie / WBI 签名维护成本高）
- yt-dlp / bilibili-api 等库的版本飘移
- 网络不可用时整个 pipeline 卡住，调试体验差
- 边界 case 多（直播录像 / 番剧 / 分 P / 大会员限定）

主开发心智需要聚焦在"训练 / 合成"的核心价值上，而非"下载工具"。

## Decision

**MVP（v0.1.x）只支持本地音频 / 视频文件输入**：
- 用户自行下载所需源（浏览器插件、yt-dlp 命令行等任意工具）
- 把文件放入项目的 `inputs/` 目录
- SourceAgent 仅做：格式探测 + ffmpeg 转码到 WAV 24kHz/16-bit/mono

URL 下载作为**可选后置模块**，规划在 v1.0.0 前接入：单独的 `core/downloader.py` 封装 yt-dlp，主 pipeline 解耦。

## Alternatives

- **MVP 即支持 B 站 URL**：引入 bilibili-api-python，但开发心智负担过大，且容易被平台改动卡死
- **支持直链音频 URL 但不支持平台 URL**：节省一部分复杂度，但用户场景里"主播音频"主要源自平台，价值打折

## Consequences

### 正向
- 大幅降低 MVP 开发心智负担
- 主 pipeline 不依赖网络，调试 / 测试更可控
- 解耦后，未来 URL 下载子模块可独立迭代、独立测试

### 负向 / 代价
- 用户使用门槛略高（要自己下载）
- 流程从"丢链接就有故事"变成"丢链接 + 等下载 + 丢文件"

### 后续需要观察
- 用户是否真的有"丢链接"诉求；若高频则 URL 下载模块优先级提前
- B 站政策变化是否影响最终方案选择

## References

- [docs/PLAN.md](../PLAN.md) §1.1
- 用户讨论：2026-05-12 简化第一步
