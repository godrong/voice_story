#!/usr/bin/env python3
"""exp_011c: RoPE position interpolation — the A-vs-B verdict experiment.
exp_011c: RoPE 位置插值干预 — 区分假说 A(RoPE 外推) vs B(注意力稀释)的裁决实验。

Logic (逻辑):
  A's input variable = token DISTANCES (位置差)
  B's input variable = token COUNT in softmax (竞争人数)
  Intervention: divide all position ids by `scale` → distances shrink,
  count untouched. 把位置编号整体除以 scale —— 只缩短距离，不减少人数。

  R16_s2 recovers  → A confirmed (scoring function was fed OOD distances)
  R16_s2 unchanged → B confirmed (crowding; position innocent)
  Canary R1_s8 MUST degrade, else the hook silently failed → abort analysis.
  金丝雀 R1_s8 必须劣化，否则说明干预没挂上，结论全部作废。
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
OUT = Path("/root/autodl-tmp/exp011_rope")
AUDIO = OUT / "audio"
AUDIO.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = "You are a helpful assistant."
SPEAKER, EMOTION = "0002", "Neutral"
SEEDS = [42, 123, 456]
CHUNK_UTTS = 4

# (name, n_ref_utts, rope_scale)
CONDITIONS = [
    ("R1_s1",   1, 1.0),   # healthy baseline / 健康基线
    ("R1_s2",   1, 2.0),   # scaling cost on healthy case / 压缩的副作用
    ("R1_s8",   1, 8.0),   # CANARY: must degrade / 金丝雀：必须劣化
    ("R16_s1", 16, 1.0),   # crash baseline / 崩溃基线 (expect ~0.70)
    ("R16_s2", 16, 2.0),   # THE intervention / 主干预 (2570→1285)
    ("R16_s3", 16, 3.0),   # stronger / 更强 (→857, 完全回到校准区)
    ("R24_s3", 24, 3.0),   # strongest test / 最强测试 (3590→1197)
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


# ═════════════════════ THE CORE: RoPE scaler ═════════════════════════════
class RopeScaler:
    """Wraps every RotaryEmbedding inside the LLM; divides position ids.
    包住 LLM 里所有 RotaryEmbedding 模块，把位置编号整体除以 scale。

    Engagement proof (生效证明): logs original→scaled max position on the
    first call of every generation. If this line never appears, the hook
    silently failed and all conclusions are void.
    每次生成的第一次调用都打印 原始→缩放后 的最大位置；没出现 = 干预未生效。
    """

    def __init__(self, llm_root: torch.nn.Module):
        self.scale = 1.0
        self._engaged_logged = False
        self.patched = []
        for name, mod in llm_root.named_modules():
            if "RotaryEmbedding" in type(mod).__name__:
                self._patch(name, mod)
        print(f"[RopeScaler] patched {len(self.patched)} rotary module(s): "
              f"{[n for n, _ in self.patched]}")

    def _patch(self, name, mod):
        orig_forward = mod.forward

        def scaled_forward(*args, **kwargs):
            # position_ids may arrive positionally (arg 1) or as kwarg.
            # 位置编号可能在第 2 个位置参数，也可能在 kwargs 里。
            args = list(args)
            pid = kwargs.get("position_ids", None)
            pid_in_kw = pid is not None
            if pid is None and len(args) >= 2 and torch.is_tensor(args[1]):
                pid = args[1]
            if pid is not None and self.scale != 1.0:
                scaled = pid.float() / self.scale     # ← the entire trick
                if not self._engaged_logged:
                    print(f"[RopeScaler] ENGAGED scale={self.scale}: "
                          f"max pos {int(pid.max())} → {float(scaled.max()):.1f}")
                    self._engaged_logged = True
                if pid_in_kw:
                    kwargs["position_ids"] = scaled
                else:
                    args[1] = scaled
            return orig_forward(*args, **kwargs)

        mod.forward = scaled_forward                  # instance attr shadows class
        self.patched.append((name, orig_forward))

    def set_scale(self, s):
        self.scale = float(s)
        self._engaged_logged = False
# ═════════════════════════════════════════════════════════════════════════


def build_target(n_chars):
    text = BASE_STORY
    while len(text) < n_chars:
        text += BASE_STORY
    seg = text[:n_chars]
    for d in ("。", "！", "？"):
        i = seg.rfind(d)
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
    """Token-concat prefix builder (same machinery as exp1b, validated).
    与 exp1b 相同的 token 拼接前缀构造（已验证）。"""
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
    print(f"Target {len(target)} chars")

    print("Loading CV3 ...")
    model = AutoModel(model_dir=MD)
    frt, cv3m = model.frontend, model.model
    print("  Ready")

    # Patch rotary modules INSIDE the LLM only (flow untouched).
    # 只 patch LLM 内部的 rotary，flow 声码器不受影响。
    scaler = RopeScaler(cv3m.llm)
    if not scaler.patched:
        print("FATAL: no RotaryEmbedding found — wrong module path, abort.")
        sys.exit(1)

    sentences = frt.text_normalize(target, split=False, text_frontend=True)

    # Pre-build inputs per unique prefix size / 每种前缀只构造一次
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
    print("EXP_011c: ROPE INTERPOLATION  (exp1b reference: R16=0.701 R24=0.989)")
    print(f"{'=' * 70}")
    print(f"{'Cond':<9s} {'Scale':>6s} {'PrefixTok':>10s} {'CER':>7s} {'±':>6s}")
    print("-" * 70)
    summary = {}
    for name, n_utts, scale in CONDITIONS:
        items = [r for r in ok if r["name"] == name]
        if not items:
            print(f"{name:<9s} (failed)")
            continue
        cers = [r["CER"] for r in items]
        summary[name] = {"scale": scale,
                         "CER_mean": round(float(np.mean(cers)), 4),
                         "CER_std": round(float(np.std(cers, ddof=1)), 4)
                         if len(cers) > 1 else 0.0}
        s = summary[name]
        print(f"{name:<9s} {scale:>6.1f} {items[0]['prefix_speech_tokens']:>10d} "
              f"{s['CER_mean']:>7.3f} {s['CER_std']:>6.3f}")

    print(f"\n{'=' * 70}\nVERDICT\n{'=' * 70}")
    g = lambda k: summary.get(k, {}).get("CER_mean")
    canary_fired = (g("R1_s8") or 0) > (g("R1_s1") or 0) + 0.15
    print(f"Canary R1_s8 degraded vs R1_s1: {canary_fired} "
          f"({g('R1_s1')} → {g('R1_s8')})")
    if not canary_fired:
        print("⚠ CANARY DID NOT FIRE — hook may be a silent no-op.")
        print("  All conclusions VOID. Check [RopeScaler] ENGAGED lines in log.")
    elif g("R16_s1") is not None and g("R16_s2") is not None:
        base, fixed2, fixed3 = g("R16_s1"), g("R16_s2"), g("R16_s3")
        best = min(x for x in (fixed2, fixed3) if x is not None)
        recovery = (base - best) / max(base, 1e-6)
        print(f"R16: scale1={base}  scale2={fixed2}  scale3={fixed3}  "
              f"recovery={recovery:.0%}")
        if recovery > 0.6:
            print("→ 假说 A 实锤：位置压缩大幅恢复 → RoPE 外推是病根。")
            print("  含义：免训练位置插值即可扩展 CV3 长文本上限。")
        elif recovery < 0.2:
            print("→ 假说 B 实锤：距离压回校准区也没用 → 上下文稀释是病根。")
            print("  含义：切分是唯一解，Agent 方案地位封顶。")
        else:
            print("→ 部分恢复：A 与 B 叠加，恢复幅度给出各自占比。")
        if g("R24_s3") is not None:
            print(f"R24_s3 (3590→1197): CER={g('R24_s3')} (exp1b 基线 0.989)")
        if g("R1_s2") is not None:
            print(f"压缩副作用 (R1_s2 vs R1_s1): {g('R1_s1')} → {g('R1_s2')}")

    json.dump({"summary": summary, "runs": records},
              open(OUT / "results.json", "w"), ensure_ascii=False, indent=2)
    print(f"\nSaved: {OUT / 'results.json'}")
    print(f"Total: {(time.monotonic() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
