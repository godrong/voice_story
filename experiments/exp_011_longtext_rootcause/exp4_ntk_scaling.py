#!/usr/bin/env python3
"""exp_011d: NTK-aware frequency scaling — the non-toxic A-vs-B instrument.
exp_011d: NTK 频率选择性缩放 — "无毒版"的 A vs B 裁决仪器。

vs exp_011c (naive PI, failed): PI slowed ALL rotation including the
high-frequency "second hand" → destroyed adjacent-token resolution →
even healthy R1 crashed (CER 0.905) → instrument toxic, verdict void.

NTK scales each RoPE dimension progressively (NTK 按维度递进缩放):
    new_inv_freq[j] = inv_freq[j] * s^(-2j/(d-2))
    j=0   (highest freq, "秒针"): factor 1.0  → untouched, local order safe
    j=d/2-1 (lowest freq, "时针"): factor 1/s → far-range extended by s

Gate design (门控设计):
    R1_n2 / R1_n4 (healthy + scaling) MUST stay CER≈0.
    Gate passes → R16/R24 recovery is interpretable:
        recovery > 60% → Hypothesis A (RoPE distance OOD) confirmed
        recovery < 20% → Hypothesis B (attention dilution) confirmed
    Gate fails → instrument still toxic, report and stop.
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
OUT = Path("/root/autodl-tmp/exp011_ntk")
AUDIO = OUT / "audio"
AUDIO.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = "You are a helpful assistant."
SPEAKER, EMOTION = "0002", "Neutral"
SEEDS = [42, 123, 456]
CHUNK_UTTS = 4

# (name, n_ref_utts, ntk_scale)
CONDITIONS = [
    ("R1_n1",   1, 1.0),   # healthy baseline / 健康基线
    ("R1_n2",   1, 2.0),   # GATE: must stay ~0 / 门控
    ("R1_n4",   1, 4.0),   # GATE stronger / 强门控
    ("R16_n1", 16, 1.0),   # crash baseline / 崩溃基线 (~0.70)
    ("R16_n2", 16, 2.0),   # intervention / 干预
    ("R16_n4", 16, 4.0),   # stronger / 强干预
    ("R24_n4", 24, 4.0),   # strongest test / 最强测试
]

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


# ═══════════════════ THE CORE: NTK frequency scaler ═════════════════════
class NTKScaler:
    """Rescales the inv_freq buffer of every RotaryEmbedding in the LLM.
    重写 LLM 内所有 RotaryEmbedding 的 inv_freq 缓冲区。

    Per-dimension factor s^(-2j/(d-2)): j=0 untouched ("second hand"),
    j=d/2-1 slowed by s ("hour hand"). Engagement proof = printed
    inv_freq[-1] ratio, which must equal 1/s exactly.
    逐维系数：秒针不动，时针慢 s 倍。生效证明 = inv_freq[-1] 比值恒等于 1/s。
    """

    def __init__(self, llm_root: torch.nn.Module):
        self.mods = []
        for name, mod in llm_root.named_modules():
            if "RotaryEmbedding" in type(mod).__name__ and hasattr(mod, "inv_freq"):
                self.mods.append((name, mod, mod.inv_freq.detach().clone()))
        print(f"[NTK] found {len(self.mods)} rotary module(s): "
              f"{[n for n, _, _ in self.mods]}")

    def set_scale(self, s: float):
        for name, mod, orig in self.mods:
            half = orig.shape[0]                      # d/2 entries
            d = half * 2                              # head_dim
            j = torch.arange(half, dtype=torch.float64, device=orig.device)
            factor = float(s) ** (-2.0 * j / (d - 2))   # ← the entire trick
            new = (orig.to(torch.float64) * factor).to(orig.dtype)
            mod.inv_freq = new                        # plain attr or buffer
            # Invalidate any cached cos/sin from older transformers versions.
            # 清掉旧版 transformers 可能缓存的 cos/sin。
            for attr in ("cos_cached", "sin_cached", "_cos_cached", "_sin_cached"):
                if hasattr(mod, attr):
                    try:
                        delattr(mod, attr)
                    except Exception:
                        pass
            if hasattr(mod, "max_seq_len_cached"):
                mod.max_seq_len_cached = 0
            ratio = float(new[-1] / orig[-1])
            print(f"[NTK] ENGAGED scale={s}: {name} inv_freq[0] ratio="
                  f"{float(new[0]/orig[0]):.3f} (must be 1.0), "
                  f"inv_freq[-1] ratio={ratio:.3f} (must be {1/float(s):.3f})")
# ═════════════════════════════════════════════════════════════════════════


def build_target(n_chars):
    text = BASE_STORY
    while len(text) < n_chars:
        text += BASE_STORY
    seg = text[:n_chars]
    for dlm in ("。", "！", "？"):
        i = seg.rfind(dlm)
        if i > n_chars * 0.7:
            return seg[: i + 1]
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
            p = line.strip().split("\t")
            if len(p) < 3:
                continue
            f2 = p[0].split("_")
            if f2[1] == speaker and f2[2] == emotion:
                pool.append((p[1], p[2]))
    random.seed(42)
    random.shuffle(pool)
    return pool


def build_chunk_wav(pool, lo, hi, out_path):
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
    return str(out_path), "".join(texts)


def build_inputs(frt, sentences, pool, n_utts, sample_rate, tag):
    """Token-concat prefix builder (validated in exp1b). exp1b 验证过的拼接器。"""
    chunk_tokens, full_text = [], ""
    for ci, lo in enumerate(range(0, n_utts, CHUNK_UTTS)):
        hi = min(lo + CHUNK_UTTS, n_utts)
        cpath, ctext = build_chunk_wav(pool, lo, hi, OUT / f"{tag}_c{ci}.wav")
        full_text += ctext
        cmi = frt.frontend_zero_shot(
            str(sentences), SYSTEM_PROMPT + "<|endofprompt|>" + ctext,
            cpath, sample_rate, "")
        chunk_tokens.append(cmi["llm_prompt_speech_token"])
    mi = frt.frontend_zero_shot(
        str(sentences), SYSTEM_PROMPT + "<|endofprompt|>" + full_text,
        str(OUT / f"{tag}_c0.wav"), sample_rate, "")
    long_tok = torch.cat(chunk_tokens, dim=1)
    mi["llm_prompt_speech_token"] = long_tok
    if "llm_prompt_speech_token_len" in mi:
        mi["llm_prompt_speech_token_len"] = torch.tensor(
            [long_tok.shape[1]], dtype=torch.int32)
    ptok, plen = frt._extract_text_token(
        SYSTEM_PROMPT + "<|endofprompt|>" + full_text)
    mi["prompt_text"] = ptok
    if "prompt_text_len" in mi:
        mi["prompt_text_len"] = plen
    return mi, int(long_tok.shape[1]), int(ptok.shape[1])


def main():
    t0 = time.monotonic()
    from cosyvoice.cli.cosyvoice import AutoModel

    pool = select_ref_pool("/root/autodl-tmp/esd_cn/train.data.list",
                           SPEAKER, EMOTION)
    target = build_target(150)

    print("Loading CV3 ...")
    model = AutoModel(model_dir=MD)
    frt, cv3m = model.frontend, model.model
    print("  Ready")

    scaler = NTKScaler(cv3m.llm)
    if not scaler.mods:
        print("FATAL: no RotaryEmbedding with inv_freq found, abort.")
        sys.exit(1)

    sentences = frt.text_normalize(target, split=False, text_frontend=True)
    inputs = {}
    for n_utts in sorted({n for _, n, _ in CONDITIONS}):
        mi, sp, tx = build_inputs(frt, sentences, pool, n_utts,
                                  model.sample_rate, f"p{n_utts}")
        inputs[n_utts] = (mi, sp, tx)
        print(f"  prefix[{n_utts} utts]: speech {sp} + text {tx} tok")

    records = []
    for name, n_utts, scale in CONDITIONS:
        mi, sp, tx = inputs[n_utts]
        scaler.set_scale(scale)
        for seed in SEEDS:
            torch.manual_seed(seed)
            torch.cuda.empty_cache()
            tag = f"{name}_seed{seed}"
            out_wav = AUDIO / f"{tag}.wav"
            rec = {"name": name, "scale": scale, "seed": seed,
                   "prefix_speech_tokens": sp, "prefix_text_tokens": tx,
                   "target_text": target, "wav_path": str(out_wav)}
            try:
                gen = cv3m.tts(**mi, stream=False)
                audio = torch.cat([j["tts_speech"] for j in gen], dim=1)
                torchaudio.save(str(out_wav), audio, model.sample_rate)
                rec["duration_s"] = round(audio.shape[1] / model.sample_rate, 1)
                rec["status"] = "ok"
                print(f"  {tag}: {rec['duration_s']}s")
            except Exception as e:
                rec["status"] = "error"
                rec["error"] = str(e)[:150]
                print(f"  {tag} FAILED: {rec['error']}")
            records.append(rec)
    scaler.set_scale(1.0)

    del model, frt, cv3m
    torch.cuda.empty_cache()

    ok = [r for r in records if r["status"] == "ok"]
    print(f"\nASR on {len(ok)} files ...")
    from funasr import AutoModel as FM
    asr = FM(model="paraformer-zh", disable_update=True)
    for r in ok:
        res = asr.generate(input=r["wav_path"])
        r["asr_text"] = res[0]["text"] if res else ""
        r["CER"] = compute_cer(r["asr_text"], r["target_text"])
    del asr

    print(f"\n{'=' * 70}")
    print("EXP_011d: NTK-AWARE SCALING  (ref: R16 base 0.70, naive-PI was toxic)")
    print(f"{'=' * 70}")
    print(f"{'Cond':<9s} {'Scale':>6s} {'PrefixTok':>10s} {'CER':>7s} {'±':>6s} "
          f"{'Dur':>6s}")
    print("-" * 70)
    summary = {}
    for name, n_utts, scale in CONDITIONS:
        items = [r for r in ok if r["name"] == name]
        if not items:
            print(f"{name:<9s} (failed)")
            continue
        cers = [r["CER"] for r in items]
        durs = [r["duration_s"] for r in items]
        summary[name] = {"scale": scale,
                         "CER_mean": round(float(np.mean(cers)), 4),
                         "CER_std": round(float(np.std(cers, ddof=1)), 4)
                         if len(cers) > 1 else 0.0,
                         "dur_mean": round(float(np.mean(durs)), 1)}
        s = summary[name]
        print(f"{name:<9s} {scale:>6.1f} {items[0]['prefix_speech_tokens']:>10d} "
              f"{s['CER_mean']:>7.3f} {s['CER_std']:>6.3f} {s['dur_mean']:>5.1f}s")

    print(f"\n{'=' * 70}\nVERDICT\n{'=' * 70}")
    g = lambda k: summary.get(k, {}).get("CER_mean")
    # Gate: instrument must be non-toxic on healthy input.
    # 门控：仪器对健康输入必须无毒。
    gate2 = g("R1_n2") is not None and g("R1_n2") < 0.10
    gate4 = g("R1_n4") is not None and g("R1_n4") < 0.15
    print(f"GATE  R1_n2={g('R1_n2')} (<0.10? {gate2})   "
          f"R1_n4={g('R1_n4')} (<0.15? {gate4})")
    if not gate2:
        print("✗ 门控失败：NTK 缩放对健康样本仍有毒，本仪器也不合格。")
        print("  A vs B 需要换路（如注意力窗口化或微调后 PI）。")
    else:
        base = g("R16_n1")
        candidates = [g("R16_n2")] + ([g("R16_n4")] if gate4 else [])
        best = min(x for x in candidates if x is not None)
        recovery = (base - best) / max(base, 1e-6)
        print(f"R16: base={base}  n2={g('R16_n2')}  n4={g('R16_n4')}  "
              f"recovery={recovery:.0%}")
        if recovery > 0.6:
            print("→ 假说 A 实锤：频率缩放修复长上下文 → RoPE 距离 OOD 是病根。")
            print("  含义：免训练 NTK 缩放可扩展 CV3 长文本上限。")
        elif recovery < 0.2:
            print("→ 假说 B 实锤：距离量程已扩、局部精度未损，仍不恢复")
            print("  → 上下文稀释（人海）是病根，切分是唯一解。")
        else:
            print("→ 部分恢复：A 与 B 共同作用，recovery 幅度即 A 的占比。")
        if g("R24_n4") is not None:
            print(f"R24_n4: CER={g('R24_n4')} (exp1b 基线 0.989)")

    json.dump({"summary": summary, "runs": records},
              open(OUT / "results.json", "w"), ensure_ascii=False, indent=2)
    print(f"\nSaved: {OUT / 'results.json'}")
    print(f"Total: {(time.monotonic() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
