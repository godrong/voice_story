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
echo "Step 2: Install Python dependencies"
echo "============================================"
pip install nisqa jiwer soundfile librosa funasr faster-whisper datasets -q 2>&1 | tail -5

echo "============================================"
echo "Step 3: Download NISQA weights"
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
echo "Step 4: Prep MCGA dataset from HuggingFace"
echo "============================================"
cd /root/voice_story/experiments/exp_003_cosyvoice3
python3 prep_mcga.py --output_dir /root/autodl-tmp/datasets/mcga

echo "============================================"
echo "Step 5: Verify CosyVoice 3 model"
echo "============================================"
ls /root/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B/cosyvoice3.yaml && echo "Model OK"

echo "============================================"
echo "Setup complete!"
echo "============================================"
