# exp_010 — Orchestrator Labeling Pilot

> 2026-05-31 · 三模型对比（1.5B / 3B / 7B Qwen2.5-Instruct-4bit）
> 本地 MLX + Apple Silicon M4

---

## TL;DR

**4bit 量化下中文文学 role/emotion 标注能力门槛在 3B↔7B 之间。**
- 1.5B / 3B：完全塌缩，role 全 narrator、emotion 全 neutral
- 7B：跳出塌缩，但 M4 散热降频致 sustained 推理不可用
- pause_after 字段：3B 起即可靠（标点驱动，确定性任务）
- 工程含义：本地小 LLM 不能做"判断性标注"；用 API 或云 GPU

---

## 设计

| 维度 | 选择 |
|---|---|
| 语料 | 鲁迅《阿Q正传》（Gutenberg #25332，公版无争议） |
| 预处理 | OpenCC 繁→简 + 引号感知 100-300 字切分 |
| LLM | Qwen2.5-{1.5B, 3B, 7B}-Instruct-4bit（MLX 后端） |
| Schema | 最小版：`role` / `emotion` / `pause_after` |
| 样本 | 50 段（idx 0-49）；7B 仅 10 段对白密集子集（idx 10-19） |
| 设备 | M4 笔记本（Apple Silicon MPS） |

### Schema

```jsonc
{
  "chunk_id": "luxun_AQ_0042",
  "text": "他这时确乎有些胡涂了。",
  "role": "narrator | character_<N> | ambiguous",
  "emotion": "neutral | happy | angry | sad | surprise",
  "pause_after": "short | medium | long"
}
```

---

## 结果

### 标签分布

```
==============================================================================
MODEL                 role top-1            emotion top-1         pause top-1
==============================================================================
1.5B-4bit (50)        narrator @ 100%       neutral @ 100%        short  @ 82%
3B-4bit  (50)         narrator @ 100%       neutral @ 100%        medium @ 100%
7B-4bit  (10 dlg)     narrator @ 50%        neutral @ 50%         medium @ 100%
==============================================================================
```

### 输出多样性（unique values used）

| 模型 | role | emotion | pause |
|:---|:---:|:---:|:---:|
| 1.5B-4bit | 1 | 1 | 2 |
| 3B-4bit | **1** | **1** | **1** |
| 7B-4bit | **3** | **3** | 1 |

3B 比 1.5B "更塌"——pause 字段从 2 个值塌成 1 个（全 medium）。
7B 是唯一在 role/emotion 上释放多样性的尺寸。

### 推理速度（M4 笔记本）

| 模型 | mean | min | max | n | 备注 |
|:---|---:|---:|---:|:---:|:---|
| 1.5B-4bit | 5.7s | 1.0 | 17.5 | 50 | 速度波动大 |
| 3B-4bit | **2.1s** | 1.9 | 2.3 | 50 | 最稳定 |
| 7B-4bit | 663s | 31.5 | **2923** | 10 | **散热降频崩溃** |

7B 前 6 段 30-50s 正常，第 7 段后暴增到 25-49 分钟/段——M4 持续推理触发 thermal throttle。

---

## 真实样本：7B 跳出 collapse 的 case

```
[seg luxun_AQ_0012]  "你算是什么东西"呢...
  1.5B → {role: narrator,    emotion: neutral, pause: short}
  3B   → {role: narrator,    emotion: neutral, pause: medium}
  7B   → {role: character_A, emotion: angry,   pause: medium}    ✓

[seg luxun_AQ_0017]  阿Q最初是失望，后来却不平了：看不上眼的王胡...
  3B   → {role: narrator,    emotion: neutral, pause: medium}
  7B   → {role: ambiguous,   emotion: angry,   pause: medium}    ✓

[seg luxun_AQ_0013]  这是未庄赛神的晚上...（场景描写）
  3B   → {role: narrator,    emotion: neutral, pause: medium}
  7B   → {role: ambiguous,   emotion: sad,     pause: medium}    ✓ 隐式情绪
```

---

## 工程结论

### 1. 字段-能力映射

| 字段 | 任务类型 | 最小可用尺寸 | 也可被规则覆盖？ |
|---|---|---|---|
| `pause_after` | 确定性（标点驱动） | 3B（或纯规则） | **✅ 规则更直接** |
| `role` (narrator/character) | 浅判断（引号驱动） | 7B+ | ~70% 可规则（含引号 → character） |
| `emotion` | 深判断（语义） | 7B+ | ❌ 关键词只能覆盖显式情感 |

### 2. 部署路径决策

| 用例 | 推荐 |
|---|---|
| 批量标 500+ 段 | **API**（DeepSeek-V3 ~$0.10 / Claude Sonnet ~$2） |
| 本地 pause-only 任务 | 3B-4bit on M4，或纯规则（更省） |
| 本地 role/emotion 标注 | **不可行**——本地小模型不能做判断性任务 |
| 云端 7B 标注 | AutoDL H800（已配 voice_story 训练环境） |

### 3. "M4 跑 7B" 的实际限制

不是显存（7B-4bit 仅 ~5GB），是**散热**：
- 前 ~5 分钟 sustained 推理速度正常
- 之后 CPU/GPU 时钟降至 ~30%
- 单 inference 延迟从 30s 暴增到 ~50min
- 解决：分 batch + 间隔冷却，或上云

---

## 方法论价值

这次 pilot 用约 1 小时 + 0 元，得到了三个清楚结论：
1. 1.5B-4bit 中文文学标注完全不可用
2. 3B-4bit 仅在确定性字段（pause）上够用
3. 7B-4bit 是判断性字段的最小可用尺寸，但 M4 散热限制其本地批量使用

**这避免了在 500 段标注上盲投。** Pilot 的目的就是廉价证伪。

---

## 文件

```
exp_010_orchestrator/
├── README.md                              # 本文件（含结果）
├── corpus/
│   ├── fetch.py                           # 抓取 + 繁→简 + 引号感知切分
│   ├── raw/AQ.txt                         # 阿Q正传原文
│   └── segments.jsonl                     # 92 段（avg 270 字）
├── prompts/
│   └── min_schema_v1.txt                  # few-shot 标注 prompt
├── label.py                               # MLX 推理主脚本
├── analyze.py                             # 单模型分布 + 规则对照
├── compare.py                             # 跨模型对比
└── labels/
    ├── labeled_v1.jsonl                   # 1.5B × 50 段
    ├── labeled_v3_3b.jsonl                # 3B   × 50 段
    └── labeled_v2_dialogue10.jsonl        # 7B   × 10 段对白
```

## 跑法

```bash
conda run -n ai_study python corpus/fetch.py
# 编辑 label.py 里 MODEL_ID 选 1.5B/3B/7B
conda run -n ai_study python label.py --n 50 --out labeled_v?.jsonl
conda run -n ai_study python analyze.py
```
