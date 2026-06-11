#!/usr/bin/env python3
"""exp_011-1b: Position sweep via TOKEN-level prefix concat (bypass 30s limit).
exp_011-1b: 在 speech-token 层拼接前缀，绕过 30s 音频限制的位置扫描。

Why (背景): CV3 speech tokenizer rejects audio >30s, so R8/R16 in exp1 failed.
Fix: extract speech tokens per ≤30s chunk, concat tokens → inflate ONLY the
LLM prompt prefix. Flow/vocoder inputs stay from a short ref → cleaner control.
仅加长 LLM 的 llm_prompt_speech_token 前缀；flow 侧固定用短 ref，变量更干净。

Conditions (4 × 3 seeds = 12 syntheses):
  R4tok : 4 utts (~15s, 2 chunks)  — machinery sanity check vs exp1's R4 (wav path)
  R8tok : 8 utts (~32s, 2 chunks)
  R16tok: 16 utts (~65s, 4 chunks)
  R24tok: 24 utts (~98s, 6 chunks) — prefix alone ≈ CRASH-range positions

Readout vs exp1: R1=0.000, R4=0.017, CRASH=0.698.
  R16/R24 CER → 0.7  ⇒ position/context length is the culprit (hypothesis A/B-prefix)
  R16/R24 CER ≈ 0.02 ⇒ position innocent ⇒ step count (hypothesis B-gen/D)
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
OUT = Path("/root/autodl-tmp/exp011_position_sweep")   # share dir with exp1
AUDIO = OUT / "audio"
AUDIO.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = "You are a helpful assistant."
SPEAKER, EMOTION = "0002", "Neutral"
SEEDS = [42, 123, 456]
CHUNK_UTTS = 4                      # 4 utts ≈ 15s < 30s per chunk
CONDITIONS = [("R4tok", 4), ("R8tok", 8), ("R16tok", 16), ("R24tok", 24)]

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


def build_target(n_chars):
    text = BASE_STORY
    while len(text) < n_chars:
        text += BASE_STORY
    seg = text[:n_chars]
    for delim in ("。", "！", "？"):
        idx = seg.rfind(delim)
        if idx > n_chars * 0.7:
            return seg[: idx + 1]
    return seg


def compute_cer(asr_text, target_text):
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


def select_ref_pool(data_list, speaker, emotion):
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


def build_chunk_wav(pool, lo, hi, out_path):
    """Concat utts [lo,hi) into one ≤30s wav. 拼接子块（保证 <30s）。"""
    wavs, texts, sr_ref = [], [], None
    for wav_path, text in pool[lo:hi]:
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
    return str(out_path), "".join(texts), full.shape[1] / sr_ref


def main():
    t0 = time.monotonic()
    from cosyvoice.cli.cosyvoice import AutoModel

    pool = select_ref_pool("/root/autodl-tmp/esd_cn/train.data.list",
                           SPEAKER, EMOTION)
    target = build_target(150)
    print(f"Pool {len(pool)} utts | target {len(target)} chars")

    print("Loading CV3 ...")
    model = AutoModel(model_dir=MD)
    frt, cv3m = model.frontend, model.model
    print("  Ready\n")

    sentences = frt.text_normalize(target, split=False, text_frontend=True)

    records = []
    for cond_name, n_utts in CONDITIONS:
        # 1) chunk-wise token extraction / 分块提取 speech token
        chunk_tokens, full_ref_text, total_dur = [], "", 0.0
        ok_chunks = True
        for ci, lo in enumerate(range(0, n_utts, CHUNK_UTTS)):
            hi = min(lo + CHUNK_UTTS, n_utts)
            cpath, ctext, cdur = build_chunk_wav(
                pool, lo, hi, OUT / f"{cond_name}_chunk{ci}.wav")
            total_dur += cdur
            full_ref_text += ctext
            try:
                cprompt = SYSTEM_PROMPT + "<|endofprompt|>" + ctext
                cmi = frt.frontend_zero_shot(str(sentences), cprompt,
                                             cpath, model.sample_rate, "")
                chunk_tokens.append(cmi["llm_prompt_speech_token"])
            except Exception as e:
                print(f"  {cond_name} chunk{ci} FAILED: {str(e)[:100]}")
                ok_chunks = False
                break
        if not ok_chunks:
            continue

        # 2) base inputs from first chunk (flow side stays short)
        #    基础输入来自第一个子块（flow 侧保持短 ref）
        base_prompt = SYSTEM_PROMPT + "<|endofprompt|>" + full_ref_text
        first_chunk_path = str(OUT / f"{cond_name}_chunk0.wav")
        mi = frt.frontend_zero_shot(str(sentences),
                                    SYSTEM_PROMPT + "<|endofprompt|>"
                                    + full_ref_text[:0]
                                    + full_ref_text,   # placeholder, replaced below
                                    first_chunk_path, model.sample_rate, "")

        # 3) splice long prefix into LLM inputs / 拼接长前缀注入 LLM 输入
        long_tok = torch.cat(chunk_tokens, dim=1)
        mi["llm_prompt_speech_token"] = long_tok
        if "llm_prompt_speech_token_len" in mi:
            mi["llm_prompt_speech_token_len"] = torch.tensor(
                [long_tok.shape[1]], dtype=torch.int32)
        ptok, plen = frt._extract_text_token(base_prompt)
        mi["prompt_text"] = ptok
        if "prompt_text_len" in mi:
            mi["prompt_text_len"] = plen

        prefix_speech = int(long_tok.shape[1])
        prefix_text = int(ptok.shape[1])
        print(f"  {cond_name}: ref {total_dur:.1f}s → prefix speech "
              f"{prefix_speech} tok + text {prefix_text} tok")

        # 4) synthesize / 合成
        for seed in SEEDS:
            torch.manual_seed(seed)
            torch.cuda.empty_cache()
            tag = f"{cond_name}_seed{seed}"
            out_wav = AUDIO / f"{tag}.wav"
            rec = {"name": cond_name, "seed": seed,
                   "ref_dur_s": round(total_dur, 1),
                   "prefix_speech_tokens": prefix_speech,
                   "prefix_text_tokens": prefix_text,
                   "target_chars": len(target), "target_text": target,
                   "wav_path": str(out_wav)}
            try:
                ts = time.monotonic()
                gen = cv3m.tts(**mi, stream=False)
                audio = torch.cat([j["tts_speech"] for j in gen], dim=1)
                torchaudio.save(str(out_wav), audio, model.sample_rate)
                rec["duration_s"] = round(audio.shape[1] / model.sample_rate, 1)
                rec["elapsed_s"] = round(time.monotonic() - ts, 1)
                rec["status"] = "ok"
                print(f"    {tag}: {rec['duration_s']}s audio, "
                      f"{rec['elapsed_s']}s elapsed")
            except Exception as e:
                rec["status"] = "error"
                rec["error"] = str(e)[:150]
                print(f"    {tag} FAILED: {rec['error']}")
            records.append(rec)

    del model, frt, cv3m
    torch.cuda.empty_cache()

    ok = [r for r in records if r["status"] == "ok"]
    print(f"\nLoading FunASR ({len(ok)} files) ...")
    from funasr import AutoModel as FM
    asr = FM(model="paraformer-zh", disable_update=True)
    for r in ok:
        res = asr.generate(input=r["wav_path"])
        r["asr_text"] = res[0]["text"] if res else ""
        r["CER"] = compute_cer(r["asr_text"], r["target_text"])
    del asr

    print(f"\n{'=' * 74}")
    print("EXP_011-1b: TOKEN-CONCAT POSITION SWEEP (cf. R1=0.000 R4=0.017 "
          "CRASH=0.698 from exp1)")
    print(f"{'=' * 74}")
    print(f"{'Cond':<8s} {'RefDur':>7s} {'PrefixSpeech':>13s} {'PrefixText':>11s} "
          f"{'CER':>7s} {'±':>6s} {'N':>3s}")
    print("-" * 74)
    summary = {}
    for cond_name, _ in CONDITIONS:
        items = [r for r in ok if r["name"] == cond_name]
        if not items:
            print(f"{cond_name:<8s} (all failed)")
            continue
        cers = [r["CER"] for r in items]
        summary[cond_name] = {
            "CER_mean": round(float(np.mean(cers)), 4),
            "CER_std": round(float(np.std(cers, ddof=1)), 4) if len(cers) > 1 else 0,
            "prefix_speech_tokens": items[0]["prefix_speech_tokens"],
            "prefix_text_tokens": items[0]["prefix_text_tokens"],
        }
        s = summary[cond_name]
        print(f"{cond_name:<8s} {items[0]['ref_dur_s']:>6.1f}s "
              f"{s['prefix_speech_tokens']:>13d} {s['prefix_text_tokens']:>11d} "
              f"{s['CER_mean']:>7.3f} {s['CER_std']:>6.3f} {len(items):>3d}")

    print(f"\n{'=' * 74}\nVERDICT\n{'=' * 74}")
    if summary:
        last = [c for c, _ in CONDITIONS if c in summary][-1]
        top_cer = summary[last]["CER_mean"]
        if top_cer > 0.4:
            print(f"{last} CER={top_cer:.3f} → 长前缀单独即可致崩。")
            print("→ 位置/上下文长度是元凶（假说 A: RoPE OOD 或 B-前缀稀释）。")
            print("  下一步：RoPE 插值干预实验确认 A。")
        elif top_cer < 0.1:
            print(f"{last} CER={top_cer:.3f} ≈ 基线 → 位置无辜。")
            print("→ 崩溃由生成步数/目标长度驱动（假说 B-生成 或 D 误差累积）。")
            print("  下一步：看 exp2 attention probe 指纹区分 B vs D。")
        else:
            print(f"{last} CER={top_cer:.3f} 居中 → 两因素皆有贡献。")
            print("  剂量-响应曲线 R4tok→R24tok 给出权重。")

    json.dump({"summary": summary, "runs": records},
              open(OUT / "results_1b.json", "w"), ensure_ascii=False, indent=2)
    print(f"\nSaved: {OUT / 'results_1b.json'}")
    print(f"Total: {(time.monotonic() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
