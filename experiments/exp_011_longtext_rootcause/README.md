# exp_011: 长文本崩溃根因定位 / Long-text Crash Root Cause

P3 发现 CV3 在 200 字后 CER 崩溃（0.12→0.92→1.0）。本实验区分四个假说：
A) RoPE 位置外推失败  B) Attention 稀释  C) Token budget 截断  D) 自回归误差累积。

完整假说推理见 `docs/RESEARCH.md` §7。

## 脚本

| 脚本 | 问题 | 方法 | 规模 |
|------|------|------|------|
| `exp1_position_sweep.py` | 位置/上下文长度 vs 生成步数,谁是元凶? | 拼接 ref 加长前缀(3s→48s),target 固定 150 字;对照 800 字崩溃组 | 15 合成,~25min |
| `exp2_attention_probe.py` | 崩溃时模型内部发生了什么? | monkeypatch sdpa 记录每步 attention 熵/文本指针/logits 熵 | 2 次生成,~15min |

## 运行（上传到 GPU 后）

```bash
scp exp1_position_sweep.py exp2_attention_probe.py root@<host>:/root/autodl-tmp/
ssh root@<host> "/root/miniconda3/bin/python -u /root/autodl-tmp/exp1_position_sweep.py"
ssh root@<host> "/root/miniconda3/bin/python -u /root/autodl-tmp/exp2_attention_probe.py"
```

依赖（已在镜像中）：CV3 模型、ESD 数据 `/root/autodl-tmp/esd_cn/`、FunASR。

## 判读速查

| 实验 1 结果 | 结论 |
|------------|------|
| R16 ≈ CRASH | 位置/上下文长度是元凶 → 假说 A（试 RoPE 插值） |
| R16 ≈ R1 | 位置无辜,步数是元凶 → 假说 B/D（看实验 2） |
| 单调爬升 | 两者皆有贡献,剂量-响应给出权重 |

| 实验 2 指纹 | 结论 |
|------------|------|
| mass_text 塌缩 / cursor 停滞 | 模型"不再读稿"——直接病根 |
| + attention 熵升 | B: 注意力稀释 |
| + argmax 乱指远处 | A: 位置感丢失 |
| attention 正常 + logits 熵爬升 | D: 误差累积 |

## 输出

- `/root/autodl-tmp/exp011_position_sweep/results.json` — CER 表 + 自动判读
- `/root/autodl-tmp/exp011_attention_probe/fingerprint_report.json` + `probe_curves.png`
- 两目录下的 WAV 可下载人耳复核
