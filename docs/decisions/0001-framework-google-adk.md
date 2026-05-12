# ADR-0001: 选择 Google ADK 作为 agent 编排框架

- **Date**: 2026-05-12
- **Status**: Accepted

## Context

Voice Story 是一个多阶段流水线：源采集 → 预处理 → 数据集构建 → 训练 → 合成 → 后处理。每个阶段以非 LLM 重计算为主（Demucs / VAD / FunASR / CosyVoice），LLM 仅在"质量判断 / 文本规整 / 异常诊断"等少数环节有用。

需要选择 agent 编排框架，候选：Google ADK、Claude Agent SDK、LangGraph、纯 Python 脚本。

## Decision

采用 **Google ADK** 作为顶层编排框架。LLM 部分通过 LiteLLM 接入 Claude，仍享受 Claude 在"判断类"任务上的优势。

## Alternatives

- **Claude Agent SDK**：偏 Claude 主控 + 工具调用模式，缺 workflow 原语，对纯批处理不顺手；锁定 Claude，不利于后续测试其他 LLM。
- **LangGraph**：成熟度更高，但 ML 流水线评估能力弱，部署路径不如 ADK 清晰。
- **纯 Python 脚本**：最简单，但失去 trace / eval / dev UI / 部署等基础设施收益，长期维护成本高。

## Consequences

### 正向
- SequentialAgent / ParallelAgent / LoopAgent 原生匹配本项目流水线结构
- BaseAgent 子类自由编排非 LLM 重计算
- Dev UI 可视化每一步输入输出，调试效率高
- 内建 trace + evaluation framework，便于跑 speaker similarity / WER / MOS 回归
- Cloud Run / Agent Engine 一键部署，未来上云训练顺畅

### 负向 / 代价
- ADK 是 2025 年才发布的框架，相对新，社区生态不如 LangGraph 成熟
- 部分概念学习成本（agent / runner / session）

### 后续需要观察
- ADK 对长任务（>1 小时的 LoRA 训练）的处理能力
- Trace 数据量在大批 dataset 处理时的存储成本

## References

- [docs/PLAN.md](../PLAN.md) §"整体架构 / Agent 编排框架选型"
- https://google.github.io/adk-docs/
