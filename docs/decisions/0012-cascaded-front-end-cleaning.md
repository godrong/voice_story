# ADR-0012: 前端清洗级联：Demucs + VoiceFixer

- **Date**: 2026-05-19
- **Status**: Accepted

## Context

ADR-0008 让 Demucs 强制开启来剥离 BGM / 音效。biggvoice（中文 LoL 游戏直播 3.4 分钟样本）作为反例暴露了"单 Demucs 不够"的问题：

实测数据（biggvoice 跑完 Demucs 之后）：
- 24 个 VAD chunk 全部 DNSMOS-OVR < 1.40，最高 1.36，最低 0.99
- 所有 chunk 被 `low_mos` 一刀切，manifest 行数 = 0
- 听感上 vocal stem 仍带：游戏音效残留、麦克风底噪、宽带嘶声
- 拿其中 mos_ovr=1.91 最佳的 chunk 当 CosyVoice 2 zero-shot 参考音直接合成，输出带"嘶哑感"

根因：`htdemucs` 的训练目标是"歌曲 + 乐器伴奏"分离，它对游戏音效 / 直播间噪声 / 麦克风噪音这类**非音乐性背景声**没有足够先验。Demucs 输出的 vocal stem 在主播类源上仍有可听噪音。

如果不在 Demucs 后加一道清洗：
1. **DNSMOS 评分失真**：全场 1.x，分布坍缩，过滤变成"全砍"，无法区分"真正好的 chunk"和"垃圾"
2. **ASR 错字偶发**：背景音干扰识别，prompt_text 不可靠，下游 zero-shot 合成质量被参考音的转写错误拖累
3. **下游 ref 还要单独洗**：使用者每次合成前要自己跑增强，pipeline 的承诺（"产出可直接用的 chunk"）破产

并行验证：把 mos_ovr=1.91 的 chunk 单独跑 VoiceFixer mode-0，OVR 跳到 2.66 / bak 从 1.52 跳到 3.06；再用增强后的 ref 重合成同 3 段文本，DNSMOS 从 2.31~2.95 上升到 2.86~3.58，听感"嘶哑感"消失。

## Decision

**Demucs 之后默认级联 VoiceFixer mode-0**，作为 M1 数据管线的 2.5 阶段。

完整链路：

```
raw_24khz ──► [Demucs htdemucs]   ──► vocals/       (去音乐性 BGM / SFX 主体)
              │
              ▼
              [VoiceFixer mode-0] ──► enhanced/     (去残余宽带噪声 + 修复带宽)
              │
              ▼
              [Silero VAD]        ──► chunks/       (从 enhanced 切，下游评分干净)
              │
              ▼
              [DNSMOS + ASR + filter] ──► manifest.jsonl
```

实现细节：
- 新建 [core/enhance.py](../../core/enhance.py)：`VoiceEnhancer` 类，惰性加载 VoiceFixer，输出固定 TARGET_SR=24 kHz mono（与 audio_io 一致）
- [agents/preprocess_agent.py](../../agents/preprocess_agent.py) 在 `Separator` 后插入 `VoiceEnhancer`，VAD 从 `enhanced/` 切
- [agents/state.py](../../agents/state.py) 加 `PipelineState.enhanced_files`，`ensure_dirs()` 多建一个 `enhanced/` 子目录
- CLI 增加 `--enhance / --skip-enhance` 与 `--enhance-mode {0,1,2}`，默认 `--enhance --enhance-mode 0`

## Alternatives

- **只换更强的 Demucs（htdemucs_ft）**：实测 SDR 提升约 0.3，但对非音乐性噪声没有质变改善；速度慢约 4 倍，时间预算不划算
- **只在 ref 选定后单独洗**（当前临时方案）：DNSMOS 评分仍失真，无法用作过滤门；管线的"产出可用 chunk"承诺破产
- **DeepFilterNet / RNNoise 替代 VoiceFixer**：仅降噪不重建带宽，对 MP3 压缩损失（高频缺失）无能为力；VoiceFixer 同时降噪 + 带宽重建对低质源更友好
- **不做二级清洗，让下游模型自己鲁棒**：意味着要等到 M6 引入更鲁棒模型才能解决，M1/M2/M3 期间数据集质量持续偏低
- **完全跳过 Demucs，只跑 VoiceFixer**：VoiceFixer 对纯人声背景下的轻量噪声合适，但对强 BGM 会"修复成怪异音色"；两步级联职责更清晰

## Consequences

### 正向

- DNSMOS / SNR / ASR 三个客观指标在脏源上重新有意义，过滤门槛能区分质量
- 下游 chunk 拿来直接当 ref 合成，无需用户额外预处理
- 管线职责分层清晰：Demucs 管"音乐去除"，VoiceFixer 管"语音修复"，VAD 管"切片"，DatasetAgent 管"过滤"
- biggvoice 类游戏直播源从"全砍"变成"有可用 chunk"，扩大了可用源的范围

### 负向 / 代价

- 增加约 ~30~60s/3-min 音频的 CPU 时间（VoiceFixer 推理）
- 首次安装多 ~500MB 模型权重（voicefixer 包自动下载）
- 录音棚级原始音源会被 VoiceFixer 略微"软化"高频细节（trade-off：人声变干净但带塑料感）→ 需要时 `--skip-enhance` 关闭
- 引入新依赖 `voicefixer`，pyproject 需加 extras

### 后续需要观察

- 已经跑过的 trump_smoke / trump_wef 数据集是否需要重跑（取决于：a）b 已生成数据是否还满足下游用途；b）VoiceFixer 对清干净源是否反向降质）
- 长音频（>10min）VoiceFixer 一次性吃进去是否爆内存；若爆，加分块 + 拼接
- 是否要为不同源类型记忆推荐 mode（演讲→0，直播→0 或 1，老式录音→2）；交给经验数据后再写新 ADR
- 在 H800 H 卡云训练时是否启用 CUDA path（`cuda=True`）；本地仍 CPU

## References

- [docs/PLAN.md](../PLAN.md) §3.A.3
- [ADR-0008](0008-demucs-always-on.md) — Demucs 强制开启
- [ADR-0009](0009-quality-thresholds-post-demucs.md) — DNSMOS 过滤门槛
- [core/enhance.py](../../core/enhance.py)
- [agents/preprocess_agent.py](../../agents/preprocess_agent.py)
- [datasets/biggvoice/README.md](../../datasets/biggvoice/README.md) — 反例数据集
- https://github.com/haoheliu/voicefixer
