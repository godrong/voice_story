# ===========================================================================
# CosyVoice 3 Pipeline — Remote H800 Commands
# 复制粘贴到 SSH 终端逐段执行
# ===========================================================================

# ============================
# Step 1: 连接 & 准备目录
# ============================
ssh -p 17451 root@connect.westb.seetacloud.com
# password: (在服务器提供商处查看)

mkdir -p /root/voice_story/experiments/exp_003_cosyvoice3
mkdir -p /root/autodl-tmp/exp003_outputs
mkdir -p /root/autodl-tmp/datasets

# ============================
# Step 2: 上传本地脚本 (从你本地终端运行)
# ============================
scp -P 17451 /Users/attention/Documents/projects/voice_story/experiments/exp_003_cosyvoice3/inference_eval.py \
    root@connect.westb.seetacloud.com:/root/voice_story/experiments/exp_003_cosyvoice3/
scp -P 17451 /Users/attention/Documents/projects/voice_story/experiments/exp_003_cosyvoice3/train_lora.py \
    root@connect.westb.seetacloud.com:/root/voice_story/experiments/exp_003_cosyvoice3/
scp -P 17451 /Users/attention/Documents/projects/voice_story/experiments/exp_003_cosyvoice3/setup_remote.sh \
    root@connect.westb.seetacloud.com:/root/voice_story/experiments/exp_003_cosyvoice3/

# ============================
# Step 3: 运行 setup (回到 SSH 终端)
# ============================
cd /root/voice_story/experiments/exp_003_cosyvoice3
bash setup_remote.sh

# ============================
# Step 4: 推理 + 4 维评测
# ============================
cd /root/CosyVoice
python /root/voice_story/experiments/exp_003_cosyvoice3/inference_eval.py \
    --eval_set /root/autodl-tmp/datasets/CV3-Eval/zh/test_zh \
    --output_dir /root/autodl-tmp/exp003_outputs \
    --max_samples 50 2>&1 | tee /root/autodl-tmp/exp003_outputs/inference.log

# ============================
# Step 5: 查看推理结果
# ============================
cat /root/autodl-tmp/exp003_outputs/eval_results.json

# ============================
# Step 6 (可选): 用预训练数据做 QLoRA 训练
# 注意: 需要先准备好 train.data.list 和 dev.data.list
# ============================
cd /root/CosyVoice
python /root/voice_story/experiments/exp_003_cosyvoice3/train_lora.py \
    --train_data /root/autodl-tmp/datasets/train.data.list \
    --cv_data /root/autodl-tmp/datasets/dev.data.list \
    --output_dir /root/autodl-tmp/exp003_lora \
    --batch_size 2 \
    --max_steps 5000 \
    --lora_rank 16 \
    --use_amp 2>&1 | tee /root/autodl-tmp/exp003_lora/train.log
