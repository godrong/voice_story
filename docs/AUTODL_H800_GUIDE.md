# AutoDL H800 — 部署与开发指南

> _适用于 voice_story RESEARCH_PLAN Phase 2: Style LoRA 训练_
> _2026-05-19_

---

## 0. H800 卡选原因

H800 = NVIDIA Hopper 架构 (sm_90), 80GB VRAM, CUDA ≥ 11.8。
torch 2.3.1 **原生支持 Hopper**，CosyVoice 3 的依赖无需改动。

| 卡型 | VRAM | 架构 | torch 2.3.1 | LoRA 训练 | 价格 |
|---|---|---|---|---|---|
| RTX 4090 | 24GB | Ada | ✅ | 够用 (14-16GB) | ¥1.6/h |
| RTX 5090 | 32GB | Blackwell | ❌ 需 PT2.7+ | — | ¥3/h |
| **H800** | **80GB** | Hopper | ✅ 完美 | 绰绰有余 | ~¥4-6/h |

**80GB 的额外好处**：
- LoRA rank 可以上到 32/64，不用纠结 8/16
- batch size 可以从 4 扩到 16+，训练快 2-3×
- 后期可以尝试全参微调 ablation（对比 LoRA vs Full）
- 训练时同步跑 CV3 推理做 val，不 OOM

---

## 1. AutoDL 实例创建

### 1.1 配置

| 配置项 | 选 |
|---|---|
| 计费方式 | **按量计费** |
| 地区 | 有 H800 的区（通常华北/西北） |
| GPU | **NVIDIA H800 80GB 单卡** |
| 主机 | 默认（通常 32-64 核 / 128-256GB RAM） |
| 镜像 | **PyTorch 2.3.1 → Python 3.10 → CUDA 12.1** |
| 数据盘 | **autodl-fs 200GB**（AISHELL-3 20GB + ESD 3GB + CV3 模型 8GB + repo + checkpoints） |

**不要选 autodl-tmp**：
- autodl-tmp 关机释放 → checkpoint 全丢
- autodl-fs 持久化 → 训练到一半关机，ckpt 还在

### 1.2 成本预估

```
训练:       20h × ¥5/h   = ¥100
存储:      200GB × 2月    = ¥40
关机不释放:  80h × ¥0.5/h = ¥40
────────────────────────────────
合计:                     ≈ ¥180 (≈ $25)
```

H800 的 80GB 意味着训练快——20h 应该能跑完所有消融实验（9 cells × ~1h + eval 开销）。

---

## 2. 连接

### 2.1 获取连接信息

AutoDL 控制台 → 实例详情 → 复制 SSH 信息：

```
ssh -p XXXXX root@connect.xxx.seetacloud.com
密码: xxxxxxxx
```

### 2.2 连接方式

**终端（最快）**：
```bash
ssh -p XXXXX root@connect.xxx.seetacloud.com
# 粘贴密码

# 连上后
nvidia-smi
# 应该显示: NVIDIA H800, 80GB, CUDA 12.1
```

**VS Code Remote（推荐开发用）**：
1. `Cmd+Shift+P` → `Remote-SSH: Add New SSH Host`
2. 粘贴 `ssh -p XXXXX root@connect.xxx.seetacloud.com`
3. 保存到 `~/.ssh/config`
4. `Connect to Host` → 选刚才建的 → 输密码

---

## 3. 环境搭建

### 3.1 确认 GPU + torch

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
# 期望: 2.3.1 NVIDIA H800
```

### 3.2 Clone 两个 repo

```bash
cd /root/autodl-fs

# 生成 SSH key 给 GitHub
ssh-keygen -t ed25519 -C "wangrongview@163.com" -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub
# 复制整行 → https://github.com/settings/keys → New SSH key

# Clone
git clone git@github.com:godrong/voice_story.git
git clone https://github.com/FunAudioLLM/CosyVoice.git
```

### 3.3 装依赖

```bash
# CosyVoice（H800 + torch 2.3.1，不需要改动）
cd /root/autodl-fs/CosyVoice
pip install -r requirements.txt 2>&1 | tail -5

# voice_story
cd /root/autodl-fs/voice_story
pip install -e ".[eval,preprocess,asr]"

# 验证
python -c "
import jiwer, speechmos
import torch
print(f'jiwer={jiwer.__version__}  torch={torch.__version__}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.0f}GB')
"
# 期望: VRAM: 80GB
```

---

## 4. 模型 & 数据下载

### 4.1 CosyVoice 3 模型 (~8GB)

```bash
mkdir -p /root/autodl-fs/models

python -c "
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice3-0.5B',
                   local_dir='/root/autodl-fs/models/CosyVoice3-0.5B')
"
# 如果 ModelScope ID 失败，换 HF:
# huggingface-cli download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
#   --local-dir /root/autodl-fs/models/CosyVoice3-0.5B

# 验证
ls /root/autodl-fs/models/CosyVoice3-0.5B/cosyvoice3.yaml
# 应该存在
```

### 4.2 ESD wav (~3.2GB)

```bash
mkdir -p /root/autodl-fs/data

cd /root/autodl-fs/data
huggingface-cli download duanyu027/ESD --repo-type dataset --local-dir ESD_download

unzip -q ESD_download/ESD.zip -x "__MACOSX/*" "*.DS_Store"
mv "Emotion Speech Dataset" esd_wavs

echo 'export ESD_ROOT=/root/autodl-fs/data/esd_wavs' >> ~/.bashrc
source ~/.bashrc

# 验证
ls "$ESD_ROOT" | wc -l   # 应该 20
```

### 4.3 AISHELL-3 (~20GB, 中文核心数据集)

**这是 RESEARCH_PLAN 目标 #2（中文 WER 修复）的关键数据来源。**

```bash
cd /root/autodl-fs/data
wget https://www.openslr.org/resources/93/data_aishell3.tgz
# ~20GB, 国内下载快, 5-15 min

tar xzf data_aishell3.tgz
# 解压为 data_aishell3/

echo 'export AISHELL3_ROOT=/root/autodl-fs/data/data_aishell3' >> ~/.bashrc
source ~/.bashrc

# 看下结构
ls "$AISHELL3_ROOT"
# 应该有 train/ test/ 等目录
```

---

## 5. 环境验证

```bash
cd /root/autodl-fs/voice_story
source ~/.bashrc

# 验证 1: ESD manifest 可用
python scripts/build_two_tier_dataset.py stats datasets/esd/manifest.jsonl --min-per-cell 30
# 期望: ✓ PASS — all 100 cells

# 验证 2: CV3 能加载
python -c "
import sys; sys.path.insert(0,'/root/autodl-fs/CosyVoice')
sys.path.insert(0,'/root/autodl-fs/CosyVoice/third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel
m = AutoModel(model_dir='/root/autodl-fs/models/CosyVoice3-0.5B')
print(f'OK: {type(m).__name__}, sr={m.sample_rate}')
"
# 期望: OK: CosyVoice3, sr=24000

# 验证 3: training pair path 可解析
head -1 datasets/two_tier/tier1_train.jsonl | python -c "
import json, sys
r = json.loads(sys.stdin)
from scripts.build_two_tier_dataset import resolve_audio_path
p = resolve_audio_path(r['ref_audio'])
print(f'speaker={r[\"speaker_id\"]}  {r[\"ref_style\"]}→{r[\"target_style\"]}')
print(f'file: {p}  exists={p.exists()}')
"
# 期望: exists=True

# 验证 4: AISHELL-3 存在
ls "$AISHELL3_ROOT/train" | wc -l
# 应该有 200+ speaker 目录
```

四条全 PASS → 环境 ready。

---

## 6. 训练 vs 推理 —— 在 H800 上怎么分工

### 6.1 说清楚：两个都要在 H800 上跑

| 阶段 | 在哪跑 | 为什么 |
|---|---|---|
| **LoRA 训练** | H800 | 训练是 GPU 密集操作，H800 80GB 一次跑通 rank=32 |
| **CV3 推理（训练 val）** | H800 | 每个 val epoch 需要合成 10-20 条 wav 跑 SECS/WER eval。H800 合成一条几秒，不用拉到本地 |
| **CV3 推理（exp_002 eval）** | H800 | 6 条 t3 wavs 再跑一遍 CV3 版，给正式 baseline 表 |
| **WebUI 推理** | **你在新加坡的 Mac** | 本地 cosyvoice3_worker.py 已经通了；不要把 H800 当 server 用 |

**一句话**：计算密集的（训练 + val eval）在 H800；交互式的（WebUI 合成）在本地。

### 6.2 训练时 eval 流程

```
Training loop (H800):
  for step in range(total_steps):
      loss = forward(batch)     # train
      loss.backward()
      optimizer.step()
      
      if step % 100 == 0:
          # Generate validation audio ON H800
          syn_wav = model.synthesize(val_text, val_ref)
          
          # Run objective eval ON H800
          scores = evaluate_synthesis(syn_wav, val_gold_clip, val_target_text)
          
          log({secs: scores.secs, wer: scores.wer, f0: scores.f0_rmse_hz})
          
          if scores.secs < best_secs - 0.02:
              early_stop()
```

---

## 7. 训练数据集策略

### 7.1 现状盘点

| 数据集 | 状态 | 说话人 | 语言 | 规模 | 角色 |
|---|---|---|---|---|---|
| **ESD** | ✅ ingested (35000 chunks) | 20 (10 zh + 10 en) | 中英 | 350 句/人/情绪 | **Style 主力** — 同 speaker 多 emotion 配对 |
| **Trump WEF** | ✅ (206 chunks) | 1 | 英 | 10h | 不放进训练 (留在 held-out)，当 unseen eval |
| **AISHELL-3** | ⏳ 待下 (~20GB) | 218 | 中 | 85h | **Speaker 多样性主力** — 目标 #2 核心 |

### 7.2 为什么选 AISHELL-3（而不是其他中文数据集）

| 候选 | 说话人 | 时长 | 许可 | 为什么不选 / 选的原因 |
|---|---|---|---|---|
| **AISHELL-3** | 218 | 85h | Apache 2.0 ✅ | ⭐ **推荐** — 多说话人、高保真录音、干净标注、开放许可 |
| DiDiSpeech | ~600 | 800h | 学术 | 太大（下载 200GB+）、标注质量一般、是对话风格非朗读 |
| WenetSpeech4TTS | 多 | 12000h | CC | 数据太脏（YouTube 抓取）、下载 TB 级、不符合"干净评估"|
| aidatatang_200zh | 600 | 200h | Apache 2.0 ✓ | 可作为补充，但 200h 对 LoRA 太多（过拟合风险） |
| **ESD 中文** | **10** | **7h** | 研究 | ✅ 已经在了——同 speaker 多 emotion 才是核心 value |

**AISHELL-3 是最佳选择**：
- 218 个说话人 → 和你的 RESEARCH_PLAN "≥200 speakers" 目标完美对齐
- 录音室环境 → 干净音频，preprocessing 零成本
- 有完整中文文本 → 训练时 WER eval 有黄金标准
- Apache 2.0 许可证 → 没有版权隐患，适合公开发项目

### 7.3 最终训练数据集构成

```
Tier 1 数据（datasets/two_tier/tier1_train.jsonl + AISHELL-3）:

  ├── ESD (20 spk × 5 emo)
  │   ├── 中文: 0001-0010 (10 speakers)
  │   └── 英文: 0011-0020 (10 speakers)
  │
  └── AISHELL-3 (218 spk)
      └── 全部中文

总计: ~230 speakers
  中文: 228 spk (ESD 10 + AISHELL-3 218)
  英文: 10 spk (ESD 10)
  中文:英文样本比 = 配置文件控制，训练时设 2:1

eval unseen: ESD 0009/0010/0019/0020 (4 spk, held-out from ESD)
            + AISHELL-3 最后 10 spk (held-out from AISHELL-3)

eval gold: 每 speaker 2 clips (neutral/high-MOS)
```

### 7.4 中文数据集如果不想下 AISHELL-3 的备选

| 备选 | 说话人 | 为什么是第二选择 |
|---|---|---|
| **AISHELL-1** | 400 | 178h, 比 AISHELL-3 大但 speaker 多样性更好。wget 从 OpenSLR/33 下 |
| **thchs30** | 1 | 30h 单人，不够多样性 |
| **共同语音 (Common Voice zh-CN)** | 多 | 质量参差不齐，需要跑 M1 pipeline 清洗 |

---

## 8. 下一步

环境搭好（5 条验证全 PASS）后告诉我，接着我写：

1. `scripts/ingest_aishell3.py` — AISHELL-3 → M1 兼容 manifest
2. `scripts/train_lora.py` — LoRA 训练入口（PEFT + style-balanced batch + CV3 val eval）
3. `scripts/run_ablation.py` — 消融实验矩阵跑

---

## 附录 A: 常用命令速查

```bash
# 看 GPU 利用率
watch -n 1 nvidia-smi

# 训练后台跑，日志写文件
nohup python scripts/train_lora.py > logs/train_$(date +%Y%m%d_%H%M).log 2>&1 &

# 停止训练（优雅）
pkill -f train_lora.py

# 关机不释放（省钱）
# AutoDL 控制台 → 实例 → 关机不释放

# 重新开机
# 控制台 → 启动 → IP 可能会变，重新查 SSH 信息

# 网盘文件在哪
ls /root/autodl-fs/
# 这些关机不丢：
#   voice_story/  CosyVoice/  models/  data/
```

## 附录 B: 故障排查

| 问题 | 看哪 |
|---|---|
| GPU 没认 | `nvidia-smi` + `lsmod \| grep nvidia` |
| torch 版本不对 | `pip show torch \| grep Version` |
| 显存不够 | `nvidia-smi` 里看 `Memory-Usage`，80GB 应该不会 |
| CV3 加载失败 | 确认 `cosyvoice3.yaml` 在模型目录 |
| ESD_ROOT 没生效 | `echo $ESD_ROOT` + `source ~/.bashrc` |
| GitHub clone 失败 | `ssh -T git@github.com` 测连通性 |
