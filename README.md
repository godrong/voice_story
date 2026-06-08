# voice_story

> **B 站抓主播声音 → 输入你的故事文本 → 拿到那个主播给你讲故事的音频。**
> Grab a streamer's voice from Bilibili, feed it your story, get back audio of them reading it.

基于阿里 [CosyVoice 3](https://funaudiollm.github.io/cosyvoice3/) (`Fun-CosyVoice3-0.5B-2512_RL`) 零样本 TTS，单张 4090 即可推理。

---

## 背景 / Why this project

市面上音色克隆产品要么贵、要么数据要自己准备、要么效果像机器人。CV3 零样本 TTS 在 2025 年末把质量门槛拉到了"3-10 秒 ref 就能克隆音色"的水平——**剩下的工程问题就是怎么把声纹从用户实际能拿到的地方（B 站视频、播客、自录音频）摘出来**，并解决长文本下的稳定性。

这个项目的差异点：
1. **B 站 / 本地视频 → 声纹 → 你的文本** 整套流水线 UI 化，不用手剪人声
2. 内置 **CER / SECS / F0 三指标自动评测**，每次合成都打分（不是只听个响）
3. **5 因子系统消融**（情感 / Prompt / 副语言 token / 文本长度 / 说话人）已沉淀在 [docs/RESEARCH.md](docs/RESEARCH.md)——告诉你什么 ref / 什么文本长度真的能用
4. **单卡 4090 全栈跑通**——TTS + ASR + 评测同机不切换

---

## 一句话用法 / What it does

```
B 站视频 URL                           输入文本
       │                                  │
       ▼                                  ▼
  ┌──────────┐                       ┌──────────┐
  │ yt-dlp   │                       │ 你写的    │
  │ 抽人声    │                       │ 任意中文  │
  │ + ASR    │                       │ (≤200字)  │
  └────┬─────┘                       └────┬─────┘
       │                                  │
       │      ┌─────────────────┐         │
       └─────►│ CosyVoice 3 零样本│◄────────┘
              │ 4090 推理 ~2-5s  │
              └────┬─────────────┘
                   ▼
              生成音频 + CER/SECS/F0 评分
```

---

## 快速上手 / Quickstart

> ~10 分钟从零到 WebUI 跑起来。前置：Linux/macOS、`conda`、`git`、NVIDIA GPU 推荐（CPU 也行，慢约 60×）。

```bash
# 1. 克隆本仓 + 兄弟仓 CosyVoice
git clone https://github.com/<you>/voice_story.git
git clone https://github.com/FunAudioLLM/CosyVoice.git
# 目录结构（必须）：
#   ./voice_story/      ← 本仓
#   ./CosyVoice/        ← 官方仓（兄弟目录）

# 2. 下载 CosyVoice 3 预训练模型（~3-5GB）
cd CosyVoice
pip install modelscope
modelscope download --model iic/CosyVoice3-0.5B \
    --local_dir pretrained_models/CosyVoice3-0.5B
# worker 默认在兄弟仓 ./CosyVoice/pretrained_models/CosyVoice3-0.5B 找模型
cd ..

# 3. 创建两个 conda env（详见下方"为什么两个 env"）
cd voice_story
conda create -n voice_story python=3.11 -y
conda run -n voice_story pip install -e ".[preprocess,asr]"

conda create -n cosyvoice3 python=3.10 -y
conda run -n cosyvoice3 pip install -r ../CosyVoice/requirements.txt
conda run -n cosyvoice3 pip install -e ../CosyVoice

# 4. 告诉 server 去哪找 cosyvoice3 env 的 Python（关键！）
#    Linux/4090 机器的 conda 路径和 macOS 不同，必须显式指定。
export VOICE_STORY_CV3_PYTHON="$(conda run -n cosyvoice3 which python)"

# 5. 启动 WebUI
conda run -n voice_story uvicorn api.server:app --host 0.0.0.0 --port 8000

# 6. 浏览器打开 http://localhost:8000
```

> **env python 路径**：server 通过 `VOICE_STORY_CV3_PYTHON` 环境变量找 CV3 worker 的解释器。
> 不设则回退到 macOS miniforge 默认路径（`/opt/homebrew/.../envs/cosyvoice3/bin/python`），
> Linux / 其它 conda 安装必须 `export` 这个变量。

---

## WebUI 流程

### 路线 A：从 B 站抓声音 → 生成故事

1. **粘 B 站视频 URL** （`https://www.bilibili.com/video/BVxxxxxxxxx`）
2. 系统自动跑：`yt-dlp 下载 → Demucs 去 BGM → VAD 切片 → ASR 转写`
3. 在切片列表里挑一段干净的（3-10s、清晰说话、无音乐）作为 ref
4. 右侧输入你想合成的文本
5. **Synthesize** → 拿到主播给你念这段话的音频
6. 评分写入 `outputs/webui/feedback.jsonl`

### 路线 B：上传本地音频

1. 拖一个 `.wav/.mp3/.m4a/.mp4` 文件
2. 同样的 VAD + ASR + 选段流程
3. 之后步骤同 A

### 路线 C：用内置数据集

如果你只想测试系统，内置 8 位 ESD 说话人（5 种情感各 350 句），点选即可。

---

## 命令行用法（无 WebUI 也能跑）

```bash
# 从本地音频建数据集
conda run -n voice_story python cli.py ingest \
    --source local --path ./inputs/your_audio.wav --name myvoice

# 看数据集统计
conda run -n voice_story python cli.py dataset stats --name myvoice
```

合成（M2 开发中，暂用 Python API）：

```python
from core.tts import get_tts_backend

tts = get_tts_backend()
tts.synthesize(
    text="今天天气真不错。",
    ref_audio="datasets/myvoice/chunk_0001.wav",
    ref_text="...这段 ref 的转录文本...",
    out_path="output.wav",
)
```

---

## 系统架构

```
┌──────────────────────────────────────────────────────────────┐
│  voice_story env (torch 2.9+)                                │
│                                                               │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│   │ WebUI    │  │ FastAPI  │  │ ASR      │  │ Eval     │    │
│   │ (Vue3)   │←→│ server   │←→│ FunASR   │  │ WavLM/   │    │
│   │  + cyto- │  │          │  │ Whisper  │  │ librosa  │    │
│   │  scape   │  │          │  │          │  │          │    │
│   └──────────┘  └────┬─────┘  └──────────┘  └──────────┘    │
│                       │                                       │
│                       │ JSON-line subprocess (ADR-0010)       │
└───────────────────────┼──────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────────┐
│  cosyvoice env (torch 2.3.1 钉死)                            │
│   ┌────────────────────────────────────────────────────┐    │
│   │  core/tts_worker.py (长驻 worker, 模型只加载一次)    │    │
│   │     └─ Fun-CosyVoice3-0.5B-2512_RL (~5GB)          │    │
│   └────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

**为什么两个 conda env？** CosyVoice 官方钉了 `torch==2.3.1`，本仓主框架用 torch 2.9+。两个 env 隔离依赖冲突，通过 subprocess JSON-line 协议通信。TTS 进程冷启 ~19s 后常驻复用，单次合成只花 GPU 推理时间。详见 [docs/decisions/ADR-0010](docs/decisions/)。

---

## 硬件需求

| 项 | 最低 | 推荐 | 实测 |
|---|---|---|---|
| 显卡 | 无（CPU 慢约 60×）| 8GB+ NVIDIA | **单张 RTX 4090** 即可全栈推理 |
| 内存 | 16 GB | 32 GB | TTS 加载占 ~5GB VRAM + ~3GB RAM |
| 磁盘 | 15 GB | 20+ GB | 模型 5GB + ASR ~2GB + 缓存 |
| 系统 | Linux / macOS | Ubuntu 22.04 / macOS 14+ | |
| Python | 3.10-3.12 | 3.11 | |

Apple Silicon 不能跑 CUDA，TTS 走 CPU 慢 60× 不实用；可走远程 GPU（[H800 部署指南](docs/AUTODL_H800_GUIDE.md)）。RTX 4090 单卡能舒服跑：单段合成 2-5 秒、全链路（含 ASR + 评测）<10 秒。

---

## 路线图 / Roadmap

✅ **已完成**
- B 站 / 本地音频 / 内置数据集 三种 ref 来源
- CV3 零样本合成 + 自动 CER/SECS/F0 评测
- 5 因子系统消融（详见 RESEARCH.md）
- WebUI 实时管线可视化

🚧 **开发中**
- **语气词 / 副语言 token 自动插入**（`[breath]` `[sigh]` `[laughter]` 等）——已知 token 效果因情感而异（exp_007），下一步是让 LLM orchestrator 在合适位置自动加
- **长文本分段合成**——CV3 训练上限 200 字，超过则崩溃，正在做 chunk + ref carryover 策略消融
- **LLM orchestrator**：自动切分文本 + 标注角色 / 情感 / 停顿（exp_010 已对比 1.5B/3B/7B 能力曲线）

📋 **规划**
- 单人 LoRA 微调通道（深度数字分身路线）
- 多角色对白：自动识别说话人 → 不同 ref 渲染
- M4B / 有声书导出

---

## 研究亮点

本项目同时是一份**系统消融研究**。详见 [docs/RESEARCH.md](docs/RESEARCH.md)：

| 实验 | 关键发现 |
|---|---|
| **exp_005** 五因子基线 | 720 合成。CER 接近 0；高 arousal 情感 F0 RMSE 是 Neutral 的 1.5-1.8× |
| **exp_006** Prompt 消融 | 中文 prompt 使 CER 升高 5-10×；空 prompt 通常不差于官方 |
| **exp_007** 副语言 token | `[breath]` 对 Angry 有效（F0 −7 Hz）；`[sigh]` 对 Sad 反效果（+17 Hz） |
| **exp_008** 长文本崩溃 | 200 字后 CER 急剧退化，1600+ 字完全崩 |
| **exp_010** 小 LLM 标注边界 | 4bit 量化下，中文文学情感标注能力门槛在 3B↔7B 之间 |

**Punchline**：
> 在 5 因子矩阵上系统消融了 CV3 零样本能力边界，发现 CER 接近完美但 200 字为推理上限——这一发现驱动了"LLM 切分 + 分段合成 + 副语言 token"的工程路线。

---

## 已知限制

- **长文本崩溃**：CV3 训练 `token_max_length=200`，超过则 CER 急剧退化 → 必须 chunk（路线图中）
- **高 arousal 情感漂移**：Surprise / Angry 的 F0 RMSE 比 Neutral 高 1.5-1.8×（业界共性）
- **B 站抓取**：依赖 `yt-dlp` cookie 配置，部分私享视频不可用；BGM 重的视频 Demucs 分离质量下降
- **Apple Silicon 无 CUDA**：本地只能 CPU，建议远程 GPU 或换台 Linux + NVIDIA

---

## 目录

| 路径 | 内容 |
|---|---|
| `api/` | FastAPI 后端 |
| `webui/` | Vue3 + Cytoscape 前端 |
| `core/` | TTS / ASR / VAD / 评测的底层 wrapper |
| `agents/` | 数据 ingest 管线（local / Bilibili / kaggle） |
| `cli.py` | typer 命令行入口 |
| `experiments/` | 编号实验脚本 + 结果 JSON |
| `datasets/` | 已 ingest 的数据集 |
| `docs/RESEARCH.md` | 研究方向、发现、未解问题 |
| `docs/EXPERIMENT_LOG.md` | 实验日志 |
| `docs/decisions/` | ADR：架构决策记录 |

---

## 协作规范（贡献者）

- 关键技术决策写 ADR（`docs/decisions/`），不可改、只能用新 ADR supersede 旧的
- commit message 末尾带 `Refs: ADR-XXXX` 或 `Refs: PLAN#section`
- 每次发版更新 `pyproject.toml` 的 version + `docs/CHANGELOG.md` + `git tag`
- 中英双语 docstring

---

## License

MIT。但请遵守上游：
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) 自带许可
- 模型权重 `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` 见 [HuggingFace](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) / ModelScope
- 抓取的 B 站内容自行确保合规
- 实验语料若涉公开数据集（如 ESD），按各自许可使用
