# Experiment 002 — Reference 选择 + Instruct prompt A/B

## 目的

验证两个 CosyVoice 2 zero-shot 调节杠杆对"句尾僵硬 / 不像 Trump"的实际改善：
- **杠杆 A**：换更有"音高变化"的 reference（不再用 calm 开场白）
- **杠杆 C**：用 instruct mode 给自然语言风格指令（"句尾上扬""强调式演讲"等）

## 怎么用

### 1. 改你想测的内容

**`config.yaml`** 是唯一可编辑文件。4 个段落按编辑频率排序：
1. **`target_texts`** — 想合成的文本（每条会用所有 ref × instruct 组合跑）
2. **`matrix`** — 实际要跑的 (target × ref × instruct) 组合，注释行可跳过
3. **`instruct_prompts`** — 想试的风格指令（自然语言）
4. **`references`** — 参考音频候选（已挑好按 pitch_std 排）

### 2. 跑

```bash
conda run -n ai_study python experiments/exp_002_ref_and_instruct/runner.py
```

模型加载 ~140s（一次性，所有 matrix 任务复用）。每条任务推理约 10~30s。
默认 matrix 9 个任务 → 总耗时约 5~8 分钟。

### 3. 听 + 对比

合成结果落到 `outputs/<target_id>__<ref_id>__<instruct_id>.wav`：

```
outputs/
  t1_basic_test__r0_baseline_calm__none.wav         # 原 smoke 基线
  t1_basic_test__r1_emphatic_businessman__none.wav  # 仅换更好 ref
  t1_basic_test__r1_emphatic_businessman__en_rising.wav  # 换 ref + 英文指令
  t1_basic_test__r1_emphatic_businessman__zh_rising.wav  # 换 ref + 中文指令
  t2_trump_style__*.wav                             # Trump 风格文本
```

`results.md` 自动生成，含每个任务的耗时表 + 失败记录。

### 4. 怎么判断哪个变体最好

听感维度（按重要度）：
1. **句尾上扬** — Trump 标志性，0% 上扬 vs 80% 上扬一耳朵能分辨
2. **音色像不像** — 主观打 1~5 分，对 baseline 校准
3. **节奏 / 停顿** — 演讲式有强调感 vs 朗读式平滑
4. **发音准确** — 漏字 / 重复 token / 怪音

记下你的感受，下一轮 M3 我会把这些主观维度变成可测信号（end_pitch_slope / speaker_sim / WER）。

## 加新候选 ref（进阶）

如果想测 trump_wef manifest 里其它 chunk，先看 manifest 找：

```bash
conda run -n ai_study python -c "
import json
rows = [json.loads(l) for l in open('datasets/trump_wef/manifest.jsonl')]
# 按时长 / 文本关键词找
for r in rows:
    if 'America' in r['text'] and 6 <= r['duration'] <= 10:
        print(r['chunk_id'], r['duration'], r['text'][:80])
"
```

把找到的 chunk_id 粘到 `config.yaml` 的 `references` 段，加 audio 路径 + prompt_text（必须是 ref 中说的原文）。

## 加新风格指令（进阶）

CosyVoice 2 的 instruct mode 接受任意自然语言指令，runner 自动补 `<|endofprompt|>`。
比如想试"快速激动"：

```yaml
instruct_prompts:
  - id: en_excited_fast
    text: "Speak rapidly and excitedly, like a sports commentator."
```

然后 matrix 加一行 `[t1_basic_test, r1_emphatic_businessman, en_excited_fast]` 就行。

## 后续实验编号

- exp_001 (隐式)：M2 第一次 smoke 合成（v0.1.3 commit）
- **exp_002 (本目录)**：reference + instruct A/B
- exp_003 (待)：multi-reference（同时喂 2~3 段 ref，CosyVoice 2 支持）
- exp_004 (待)：text-level hack（标点 / 副语 / SSML 风）
