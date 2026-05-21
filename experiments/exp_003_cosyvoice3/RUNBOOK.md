# CosyVoice 3 Pipeline — MCGA Dataset — Remote H800 Commands

## 数据集: MCGA (Multi-task Classical Chinese Literary Genre Audio)
- HuggingFace: yxdu/MCGA
- 28 speakers (13男/15女), 119小时, 22,000条
- 古典文学中文语音 (赋/诗/文/词/曲, 先秦至清)

---

## Step 1: 连接远程服务器
```
ssh -p 17451 root@connect.westb.seetacloud.com
```

## Step 2: 上传新脚本 (从你本地终端运行)
```
scp -P 17451 experiments/exp_003_cosyvoice3/inference_eval.py \
    root@connect.westb.seetacloud.com:/root/voice_story/experiments/exp_003_cosyvoice3/
scp -P 17451 experiments/exp_003_cosyvoice3/train_lora.py \
    root@connect.westb.seetacloud.com:/root/voice_story/experiments/exp_003_cosyvoice3/
scp -P 17451 experiments/exp_003_cosyvoice3/prep_mcga.py \
    root@connect.westb.seetacloud.com:/root/voice_story/experiments/exp_003_cosyvoice3/
scp -P 17451 experiments/exp_003_cosyvoice3/setup_remote.sh \
    root@connect.westb.seetacloud.com:/root/voice_story/experiments/exp_003_cosyvoice3/
```

## Step 3: 一键 setup (下载 MCGA + 依赖 + NISQA)
```
cd /root/voice_story/experiments/exp_003_cosyvoice3
bash setup_remote.sh
```

## Step 4: 推理 + 4 维评测
```
cd /root/CosyVoice
python /root/voice_story/experiments/exp_003_cosyvoice3/inference_eval.py \
    --eval_set /root/autodl-tmp/datasets/mcga \
    --output_dir /root/autodl-tmp/exp003_outputs \
    --max_samples 50 2>&1 | tee /root/autodl-tmp/exp003_outputs/inference.log
```

## Step 5: 查看结果
```
cat /root/autodl-tmp/exp003_outputs/eval_results.json | python3 -m json.tool | head -80
```

## Step 6 (可选): 下载结果到本地
```
scp -P 17451 root@connect.westb.seetacloud.com:/root/autodl-tmp/exp003_outputs/eval_results.json .
```

---

## 文件说明

| 远程路径 | 用途 |
|---|---|
| `/root/autodl-tmp/datasets/mcga/wavs/` | MCGA 提取的 wav 文件 |
| `/root/autodl-tmp/datasets/mcga/eval_pairs.jsonl` | 评测对 (ref, text) |
| `/root/autodl-tmp/datasets/mcga/speakers.json` | 说话人分组信息 |
| `/root/autodl-tmp/exp003_outputs/` | 合成音频 + eval_results.json |
