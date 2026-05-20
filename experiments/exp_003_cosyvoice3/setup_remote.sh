#!/bin/bash
# ===========================================================================
# CosyVoice 3 Pipeline — Remote H800 Setup Script
# 在远程 H800 上运行: bash setup_remote.sh
# ===========================================================================
set -e

echo "============================================"
echo "Step 1: Create directories"
echo "============================================"
mkdir -p /root/voice_story/experiments/exp_003_cosyvoice3
mkdir -p /root/autodl-tmp/exp003_outputs
mkdir -p /root/autodl-tmp/datasets
mkdir -p /root/CosyVoice/pretrained_models

echo "============================================"
echo "Step 2: Download CV3-Eval dataset (Chinese + English eval)"
echo "============================================"
cd /root/autodl-tmp/datasets
if [ ! -d "CV3-Eval" ]; then
    git clone https://github.com/FunAudioLLM/CV3-Eval.git || {
        echo "git clone failed, trying direct download..."
        wget -q https://github.com/FunAudioLLM/CV3-Eval/archive/refs/heads/main.zip
        unzip -o main.zip
        mv CV3-Eval-main CV3-Eval
    }
fi
echo "CV3-Eval structure:"
ls CV3-Eval/ 2>/dev/null || echo "(checking...)"

echo "============================================"
echo "Step 3: Install Python dependencies"
echo "============================================"
pip install nisqa jiwer soundfile librosa funasr faster-whisper -q 2>&1 | tail -3

echo "============================================"
echo "Step 4: Download NISQA weights"
echo "============================================"
python3 -c "
import urllib.request
url = 'https://github.com/gabrielmittag/NISQA/raw/master/weights/nisqa.tar'
dest = '/root/CosyVoice/pretrained_models/nisqa.tar'
if __import__('os').path.exists(dest):
    print('NISQA weights already downloaded')
else:
    print('Downloading NISQA weights (~1MB)...')
    urllib.request.urlretrieve(url, dest)
    print('NISQA weights ready')
"

echo "============================================"
echo "Step 5: Verify CosyVoice 3 model"
echo "============================================"
ls /root/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B/cosyvoice3.yaml && echo "Model OK"

echo "============================================"
echo "Setup complete!"
echo "============================================"
