#!/usr/bin/env python3
"""exp_011-1: Position sweep — decouple absolute position from target length.
exp_011-1: 位置扫描 — 解耦"绝对位置"与"目标文本长度"。

Hypothesis test (假说检验):
  CV3 crashes on >200-char targets (P3: CER 0.12→0.92). Two families of causes:
    A) RoPE position out-of-distribution (绝对位置/上下文长度超训练分布)
    B/D) Generation-step count / autoregressive error accumulation (生成步数本身)

  Trick: lengthen the REF prefix (concat same-speaker utterances) to push the
  target's absolute positions up WITHOUT changing target length or step count.
  用拼接同说话人参考音频加长前缀，推高 target 绝对位置，而 target 内容不变。

Conditions (5 × 3 seeds = 15 syntheses):
  R1   : ref  1 utt (~3s)   + target 150 chars   ← baseline
  R4   : ref  4 utts (~12s) + target 150 chars
  R8   : ref  8 utts (~24s) + target 150 chars
  R16  : ref 16 utts (~48s) + target 150 chars   ← positions ≈ crash range
  CRASH: ref  1 utt (~3s)   + target 800 chars   ← P3 crash control

Readout (判读):
  R16 CER ≈ CRASH CER  → position/context-length is the culprit (hypothesis A/B-prefix)
  R16 CER ≈ R1 CER     → position innocent; step count is the culprit (hypothesis B-gen/D)
  Monotonic rise R1→R16 → dose-response for context length
"""

import sys, os, json, time, random
from pathlib import Path

import numpy as np
import torch
import torchaudio

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, "/root/CosyVoice")
sys.path.insert(0, "/root/CosyVoice/third_party/Matcha-TTS")
sys.stdout.reconfigure(line_buffering=True)

MD = "/root/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"
OUT = Path("/root/autodl-tmp/exp011_position_sweep")
AUDIO = OUT / "audio"
AUDIO.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = "You are a helpful assistant."
SPEAKER, EMOTION = "0002", "Neutral"   # same as P3 for comparability / 与 P3 一致
SEEDS = [42, 123, 456]
REF_UTT_COUNTS = [1, 4, 8, 16]

# Self-contained narrative text, no paralinguistic tokens.
# 自包含叙事文本（无副语言 token），重复拼接到 800 字复现 P3 崩溃。
BASE_STORY = (
    "老张今年六十岁，住在城南一条安静的小巷里。每天清晨五点半，他准时起床，"
    "先在院子里打一套太极拳，然后烧一壶开水，泡上一杯浓浓的龙井茶。"
    "他的老伴儿三年前去世了，儿子在外地工作，家里平时只有他一个人。"
    "但老张并不觉得孤单，他养了一只橘猫，名叫大黄，还种了满院子的花草。"
    "春天的时候，院子里的月季开得正艳，邻居们路过都要停下来看几眼。"
    "老张总是笑呵呵地剪下几枝，送给喜欢的人。他说，花开了就是让人看的，"
    "一个人看是看，大家看也是看，热闹一点总是好的。"
    "傍晚时分，他会搬一把竹椅坐在门口，看着巷子里的孩子们追逐打闹，"
    "听着远处菜市场收摊的吆喝声，慢慢地喝完最后一口茶。"
    "日子就这样一天天过去，平淡，却也安稳。"
)


def build_target(n_chars: int) -> str:
    """Repeat-and-trim base story to exactly n_chars, ending on a sentence.
    将基础文本重复拼接并裁剪到 n_chars，在句末截断。"""
    text = BASE_STORY
    while len(text) < n_chars:
        text += BASE_STORY
    seg = text[:n_chars]
    for delim in ("。", "！", "？"):
        idx = seg.rfind(delim)
        if idx > n_chars * 0.7:
            return seg[: idx + 1]
    return seg


def compute_cer(asr_text: str, target_text: str) -> float:
    """Char error rate via edit distance. 编辑距离字符错误率。"""
    a = asr_text.replace(" ", "")
    b = (target_text.replace(" ", "").replace("，", "").replace("。", "")
         .replace("？", "").replace("！", "").replace("、", ""))
    if not a or not b:
        return 1.0
    m, n = len(a), len(b)
    d = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (0 if a[i - 1] == b[j - 1] else 1))
    return round(d[m][n] / max(1, len(b)), 4)


def select_ref_pool(data_list: str, speaker: str, emotion: str):
    """All (wav, text) for one speaker+emotion, deterministic order.
    取该说话人该情感的全部样本，固定乱序。"""
    pool = []
    with open(data_list) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            fields = parts[0].split("_")
            if fields[1] == speaker and fields[2] == emotion:
                pool.append((parts[1], parts[2]))
    random.seed(42)
    random.shuffle(pool)
    return pool


def build_concat_ref(pool, n_utts: int, out_path: Path):
    """Concat n_utts wavs (0.2s silence gaps) → one ref wav + joined text.
    拼接 n_utts 条 wav（间隔 0.2s 静音）成单一参考音频，文本同步拼接。"""
    wavs, texts = [], []
    sr_ref = None
    for wav_path, text in pool[:n_utts]:
        wav, sr = torchaudio.load(wav_path)
        if sr_ref is None:
            sr_ref = sr
        elif sr != sr_ref:
            wav = torchaudio.functional.resample(wav, sr, sr_ref)
        wavs.append(wav)
        texts.append(text)
    gap = torch.zeros(1, int(0.2 * sr_ref))
    pieces = []
    for i, w in enumerate(wavs):
        pieces.append(w)
        if i < len(wavs) - 1:
            pieces.append(gap)
    full = torch.cat(pieces, dim=1)
    torchaudio.save(str(out_path), full, sr_ref)
    dur = full.shape[1] / sr_ref
    return str(out_path), "".join(texts), dur


def main():
    t0 = time.monotonic()
    from cosyvoice.cli.cosyvoice import AutoModel

    pool = select_ref_pool("/root/autodl-tmp/esd_cn/train.data.list",
                           SPEAKER, EMOTION)
    print(f"Ref pool: {len(pool)} utts for {SPEAKER}/{EMOTION}")

    target_150 = build_target(150)
    target_800 = build_target(800)
    print(f"Target 150: {len(target_150)} chars | Target 800: {len(target_800)} chars\n")

    print("Loading CV3 ...")
    model = AutoModel(model_dir=MD)
    frt, cv3m = model.frontend, model.model
    print("  Ready\n")

    # Build conditions / 构造实验条件
    conditions = []
    for n in REF_UTT_COUNTS:
        ref_wav, ref_text, dur = build_concat_ref(
            pool, n, OUT / f"ref_{n}utt.wav")
        conditions.append({
            "name": f"R{n}", "ref_wav": ref_wav, "ref_text": ref_text,
            "ref_dur_s": round(dur, 1), "target": target_150,
            "est_prefix_speech_tokens": int(dur * 25),
        })
    # Crash control reuses the 1-utt ref / 崩溃对照复用最短 ref
    conditions.append({
        "name": "CRASH", "ref_wav": conditions[0]["ref_wav"],
        "ref_text": conditions[0]["ref_text"],
        "ref_dur_s": conditions[0]["ref_dur_s"], "target": target_800,
        "est_prefix_speech_tokens": conditions[0]["est_prefix_speech_tokens"],
    })

    for c in conditions:
        print(f"  {c['name']:<6s} ref={c['ref_dur_s']:5.1f}s "
              f"(~{c['est_prefix_speech_tokens']} speech tok) "
              f"target={len(c['target'])} chars")
    print()

    records = []
    for cond in conditions:
        prompt = SYSTEM_PROMPT + "<|endofprompt|>" + cond["ref_text"]
        try:
            sentences = frt.text_normalize(cond["target"], split=False,
                                           text_frontend=True)
            mi = frt.frontend_zero_shot(str(sentences), prompt,
                                        cond["ref_wav"], model.sample_rate, "")
        except Exception as e:
            print(f"  {cond['name']} frontend FAILED: {str(e)[:120]}")
            for seed in SEEDS:
                records.append({**{k: cond[k] for k in
                                   ("name", "ref_dur_s", "est_prefix_speech_tokens")},
                                "seed": seed, "status": "frontend_error",
                                "error": str(e)[:150]})
            continue

        # Log actual prefix sizes / 记录真实前缀规模
        actual_speech_tok = int(mi["llm_prompt_speech_token"].shape[1]) \
            if "llm_prompt_speech_token" in mi else -1
        actual_prompt_text_tok = int(mi["prompt_text"].shape[1]) \
            if "prompt_text" in mi else -1

        for seed in SEEDS:
            torch.manual_seed(seed)
            torch.cuda.empty_cache()
            tag = f"{cond['name']}_seed{seed}"
            out_wav = AUDIO / f"{tag}.wav"
            rec = {
                "name": cond["name"], "seed": seed,
                "ref_dur_s": cond["ref_dur_s"],
                "prefix_speech_tokens": actual_speech_tok,
                "prefix_text_tokens": actual_prompt_text_tok,
                "target_chars": len(cond["target"]),
                "target_text": cond["target"],
                "wav_path": str(out_wav),
            }
            try:
                ts = time.monotonic()
                gen = cv3m.tts(**mi, stream=False)
                audio = torch.cat([j["tts_speech"] for j in gen], dim=1)
                torchaudio.save(str(out_wav), audio, model.sample_rate)
                rec["duration_s"] = round(audio.shape[1] / model.sample_rate, 1)
                rec["gen_speech_tokens_est"] = int(rec["duration_s"] * 25)
                # Hypothesis-C check: did generation hit the budget cap?
                # 假说 C 检查：生成是否触顶 text_len×20
                rec["budget_cap"] = len(cond["target"]) * 20
                rec["hit_budget_cap"] = rec["gen_speech_tokens_est"] >= rec["budget_cap"] * 0.95
                rec["elapsed_s"] = round(time.monotonic() - ts, 1)
                rec["status"] = "ok"
                print(f"  {tag}: {rec['duration_s']}s audio, "
                      f"{rec['elapsed_s']}s elapsed")
            except Exception as e:
                rec["status"] = "error"
                rec["error"] = str(e)[:150]
                print(f"  {tag} FAILED: {rec['error']}")
            records.append(rec)

    del model, frt, cv3m
    torch.cuda.empty_cache()

    # ASR eval / 语音识别评测
    ok = [r for r in records if r["status"] == "ok"]
    print(f"\nLoading FunASR ({len(ok)} files) ...")
    from funasr import AutoModel as FM
    asr = FM(model="paraformer-zh", disable_update=True)
    for r in ok:
        res = asr.generate(input=r["wav_path"])
        r["asr_text"] = res[0]["text"] if res else ""
        r["CER"] = compute_cer(r["asr_text"], r["target_text"])
    del asr
    torch.cuda.empty_cache()

    # Report / 汇总
    print(f"\n{'=' * 78}")
    print("EXP_011-1: POSITION SWEEP — does prefix length alone break CV3?")
    print(f"{'=' * 78}")
    print(f"{'Cond':<7s} {'RefDur':>7s} {'PrefixTok':>10s} {'TgtChars':>9s} "
          f"{'CER':>7s} {'±':>6s} {'AudioDur':>9s} {'N':>3s}")
    print("-" * 78)
    summary = {}
    for cond in conditions:
        items = [r for r in ok if r["name"] == cond["name"]]
        if not items:
            print(f"{cond['name']:<7s} {'—':>7s}  (all failed)")
            continue
        cers = [r["CER"] for r in items]
        durs = [r["duration_s"] for r in items]
        summary[cond["name"]] = {
            "CER_mean": round(float(np.mean(cers)), 4),
            "CER_std": round(float(np.std(cers, ddof=1)), 4) if len(cers) > 1 else 0.0,
            "prefix_speech_tokens": items[0]["prefix_speech_tokens"],
            "n": len(items),
        }
        s = summary[cond["name"]]
        print(f"{cond['name']:<7s} {items[0]['ref_dur_s']:>6.1f}s "
              f"{s['prefix_speech_tokens']:>10d} {items[0]['target_chars']:>9d} "
              f"{s['CER_mean']:>7.3f} {s['CER_std']:>6.3f} "
              f"{np.mean(durs):>8.1f}s {s['n']:>3d}")

    # Verdict / 自动判读
    print(f"\n{'=' * 78}\nVERDICT\n{'=' * 78}")
    if "R1" in summary and "R16" in summary and "CRASH" in summary:
        r1, r16, crash = (summary[k]["CER_mean"] for k in ("R1", "R16", "CRASH"))
        gap = crash - r1
        if gap < 0.1:
            print("CRASH control did not crash — P3 not reproduced, check setup!")
        elif r16 - r1 > 0.6 * gap:
            print(f"R16 ({r16:.3f}) ≈ CRASH ({crash:.3f}) >> R1 ({r1:.3f})")
            print("→ Long PREFIX alone breaks generation: position/context-length")
            print("  is the culprit. Hypothesis A (RoPE OOD) or B-prefix favored.")
            print("  Next: RoPE interpolation intervention to confirm A.")
        elif r16 - r1 < 0.2 * gap:
            print(f"R16 ({r16:.3f}) ≈ R1 ({r1:.3f}) << CRASH ({crash:.3f})")
            print("→ Position is INNOCENT. Crash tracks generation step count /")
            print("  target length. Hypothesis B-generation or D favored.")
            print("  Next: run exp2_attention_probe.py to separate B vs D.")
        else:
            print(f"R16 ({r16:.3f}) sits between R1 ({r1:.3f}) and CRASH ({crash:.3f})")
            print("→ Mixed: both context length and step count contribute.")
            print("  Dose-response across R1→R4→R8→R16 gives the weighting.")

    json.dump({"summary": summary, "runs": records},
              open(OUT / "results.json", "w"), ensure_ascii=False, indent=2)
    print(f"\nSaved: {OUT / 'results.json'}")
    print(f"Total: {(time.monotonic() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
