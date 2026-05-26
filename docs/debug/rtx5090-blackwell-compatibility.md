# RTX 5090 兼容性问题记录

> 2026-05-26，尝试将 CosyVoice 3 推理 pipeline 迁移到 RTX 5090 实例失败。
> 踩坑费已付，经验存档。

## 环境

| 项目 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 (32GB VRAM) |
| 架构 | Blackwell, compute capability sm_120 |
| 系统盘 | overlay 30GB |
| 基础镜像 | PyTorch 2.3.1+cu121, Python 3.10 |
| 目标 | CosyVoice 3 0.5B zero-shot 推理 + 4 维 eval |

## 问题 1: PyTorch 不支持 Blackwell 架构

```
RuntimeError: CUDA error: no kernel image is available for execution on the device

NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible
with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities
sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90.
```

**原因**: RTX 5090 是 Blackwell 架构 (sm_120)，需要 CUDA 12.5+。当前 PyTorch 2.3.1+cu121 最高支持 sm_90。

**解决方案**: 需要 PyTorch 2.6+ nightly + CUDA 12.5+（我们用了 nightly cu128）。

### PyTorch 版本对 RTX 5090 的支持

| PyTorch | CUDA | sm_120 支持 |
|---|---|---|
| 2.3.1+cu121 | 12.1 | ❌ |
| 2.5.1+cu124 | 12.4 | ❌ (CUDA 12.4 不含 sm_120 header) |
| 2.6+ nightly+cu128 | 12.8 | ✅ |

## 问题 2: cuDNN 版本不兼容

```
ImportError: libcudnn.so.9: cannot open shared object file
```

PyTorch 2.12 nightly CUDA 12.8 链接 cuDNN 9，但基础镜像只有 cuDNN 8。

## 问题 3: 系统盘仅 30GB

```
OSError: [Errno 28] No space left on device
```

| 占用 | 大小 |
|---|---|
| CosyVoice 3 预训练模型 | 9.3 GB |
| PyTorch 2.3.1 | 3.4 GB |
| 其余 Python 包 + 系统 | ~12 GB |
| 剩余 | ~5 GB |

安装 PyTorch nightly (~3.5GB 下载 + 解压) 会超出剩余空间。

## 结论

**RTX 5090 不适合跑 AutoDL 默认镜像的 CosyVoice 3**，核心原因是：

1. **Blackwell 架构需要重新构建整个 PyTorch 栈** (torch + cudnn + cublas)，不是 `pip install` 能干净解决的
2. **30GB 系统盘不够**装两个版本的 PyTorch

### 什么情况可以用 RTX 5090

- 用 Docker 镜像预装 PyTorch 2.6+ / CUDA 12.6+
- 系统盘 ≥ 50GB
- 或者自己构建 Dockerfile

### 什么情况不需要折腾

- H800 / A100 / 4090，PyTorch 开箱即用，模型已缓存，直接跑推理

## 投入

- 时间: ~2 小时（SSH 调试 + pip 安装 + 等待）
- 金钱: 5090 GPU 实例费
- 结论: 退回到 H800 实例
