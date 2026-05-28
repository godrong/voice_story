# Tier 2 LoRA Runbook — Single-Speaker Deep Voice Cloning

**目的**: 从一个目标说话人的音频（B站/YouTube/本地文件）→ 数据集 manifest → 单人深度 LoRA → 用 SECS 验证语音相似度提升。

**Baseline 数字**（zero-shot, 没动）：
- SECS ≈ **0.945**（CosyVoice 3 zero-shot, 4 个中文测试文本平均）
- 验收：相对 baseline **SECS 提升量**（绝对 0.99 不强求，0.97 是现实主目标）

---

## Phase A：本机准备（你已经能跑）

### A1. 准备说话人数据

**路径 1 — 用现成的 Trump 408 条（最快，5 分钟验证训练代码不炸）**

```bash
# 1. 把 JSONL 转 lora_train.py 吃的 tab-separated 格式
conda run -n ai_study python experiments/exp_003_cosyvoice3/jsonl_to_datalist.py \
    --input  datasets/two_tier/tier2_train.jsonl \
    --output /tmp/tier2_trump.data.list \
    --audio-root .
# 期望输出：wrote 408 rows -> /tmp/tier2_trump.data.list  (0 skipped)
```

**路径 2 — 从 B 站抓张雪峰合集（端到端验证 ingest 链路）**

```bash
# 1. 通过 webui（推荐，能看到 PipelineCard 实时进度）
# 打开 http://127.0.0.1:8765 → "Build training dataset (M1 pipeline)" 卡片
# 填：source=Bilibili URL / name=zhang_xuefeng / url=<张雪峰合集URL> / single speaker ✓

# 1. 或者命令行（同样的结果）
conda run -n ai_study python -m cli ingest \
    --source bilibili \
    --url "https://www.bilibili.com/video/BVxxx" \
    --name zhang_xuefeng \
    --is-single-speaker \
    --lang-hint zh

# 2. 转 lora_train.py 格式
conda run -n ai_study python experiments/exp_003_cosyvoice3/jsonl_to_datalist.py \
    --input  datasets/zhang_xuefeng/manifest.jsonl \
    --output /tmp/tier2_zxf.data.list \
    --audio-root .
```

### A2. （可选）拉 30-60 分钟的训练数据

单人 LoRA 一般 >20 分钟才看得到 SECS 提升。建议 30-60 分钟（清洁的演讲/直播）。多个 URL 可以分别 ingest 到同一个 `--name`，manifest 会累积合并（v1.1 的 schema 是 append-friendly）。

---

## Phase B：上 GPU 训练（你需要做）

### B1. 把 data.list + 数据集传到 GPU

```bash
# 假设 GPU 机器走 ssh，user@host:/root
scp /tmp/tier2_trump.data.list   root@<gpu-host>:/root/autodl-tmp/datasets/
rsync -avz datasets/trump_wef/   root@<gpu-host>:/root/voice_story/datasets/trump_wef/
# 注意 data.list 里的 audio_path 是 'datasets/trump_wef/chunks/xxx.wav' 的相对路径，
# 需要在 GPU 机器上 cd 到 voice_story 项目根 才能让相对路径解析正确。
# 或者在转换时用 --audio-root /root/voice_story 改写成绝对路径。
```

### B2. 跑 Tier 2 LoRA

按 [TRAINING_DESIGN.md](TRAINING_DESIGN.md) 的 Tier 2 推荐配置：

| 参数 | Tier 1 (现有) | **Tier 2 (本次)** |
|---|---|---|
| rank | 16 → 8 已优化 | **32**（更高容量适合单人深度） |
| alpha | 32 | **64** |
| dropout | 0.05 | 0.05 |
| lr | 1e-4 | **5e-5**（数据少要更稳） |
| max_steps | 200 | **500**（数据少，多过几轮） |
| epochs | 20 | 50 |

```bash
ssh root@<gpu-host>
cd /root/voice_story

python /root/voice_story/experiments/exp_003_cosyvoice3/lora_train.py \
    --train_data /root/autodl-tmp/datasets/tier2_trump.data.list \
    --output_dir /root/autodl-tmp/exp003_tier2_trump \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --lr 5e-5 \
    --max_steps 500 \
    --epochs 50 \
    --save_interval 100
```

期望 console 输出：
```
INFO LoRA rank=32: 24 layers, trainable 4.3M / 500M (0.86%)
INFO step  100/500 | loss=0.812 | lr=...
INFO step  200/500 | loss=0.735 | ...
...
INFO Done! {"final_loss": 0.69, ...}
```

### B3. 评估 SECS

```bash
# 用 exp_003 现有的 inference_eval.py 跑评估
python /root/voice_story/experiments/exp_003_cosyvoice3/inference_eval.py \
    --lora_dir /root/autodl-tmp/exp003_tier2_trump/final \
    --eval_set tier2_holdout \
    --output_dir /root/autodl-tmp/exp003_tier2_trump/eval

# 关键指标：SECS（看 eval 输出 JSON 里 mos_secs / secs_mean 字段）
```

### B4. 对比 baseline 写回 results.md

```bash
# 把 final/metrics.json + eval/*.json 拷回本机
scp -r root@<gpu-host>:/root/autodl-tmp/exp003_tier2_trump/ \
    experiments/exp_003_cosyvoice3/runs/tier2_trump_$(date +%Y%m%d)/

# 然后在 experiments/exp_003_cosyvoice3/results.md 里追加一段：
# - SECS baseline (zero-shot): 0.945
# - SECS Tier 2 (rank=32, 500 steps, 408 pairs): 0.XX  (+0.XX vs baseline)
```

---

## Phase C：如果 SECS 提升 < 0.02，怎么调

| 症状 | 可能原因 | 试试 |
|---|---|---|
| loss 一直 0.8+ 不降 | LR 太小 | 提到 1e-4 |
| loss 降到 0.5 但 SECS 不升 | overfit | 减 max_steps 到 300 |
| SECS 升了 0.01-0.02 卡住 | rank 不够 | rank=64, alpha=128 |
| 听感变机械/有 artifact | overfit / LR 太大 | 降 LR 到 2e-5；减 step |
| 数据太少（<10 分钟） | 没办法 | 加更多录音/视频 |

**0.97 SECS 还不够 → 真到 0.99？**：在这个 LoRA 架构下基本到顶。要继续推：
- 换全参 fine-tune（不冻 LLM 权重，至少 80GB VRAM）
- 或者训 speaker embedding adapter 而不是 LoRA on attention（架构变更）

---

## 故障排除

- **`cosyvoice` 模块找不到**：检查 `COSYVOICE_ROOT` 环境变量，默认 `/root/CosyVoice`
- **CUDA OOM**：降 batch_size（lora_train.py 现在是 dynamic batch，可能要在 cosyvoice3.yaml 里调）
- **`text_opener` 跳过所有行**：检查 data.list 是不是真的 tab 分隔（不是空格）；`cat -A /tmp/tier2_trump.data.list | head -1` 看到 `^I` 才是 tab
- **manifest.jsonl 路径不对**：B 站 ingest 的产物在 `datasets/<name>/chunks/`；data.list 里的路径要能从 GPU 机器的 cwd 解析（或者用 `--audio-root` 改成绝对路径）

---

## 一句话流程

```
URL/上传 → webui "Build training dataset" → manifest.jsonl
       → jsonl_to_datalist.py → data.list
       → scp 到 GPU → lora_train.py → final/lora_weights.pt
       → inference_eval.py → SECS 数字
       → 写回 results.md
```
