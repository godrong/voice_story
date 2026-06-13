#!/usr/bin/env python3
"""exp_011f: Residual attribution — is the post-NTK residual attention dilution?
exp_011f: 残差归因 — NTK 修复后剩下的 ~30% 是不是注意力稀释(假说 B)?

TTS-specific insight (TTS 特有洞察):
  The script (target text) must stay visible the whole time — the model reads
  it word by word. What actually ACCUMULATES during long generation is the
  ALREADY-GENERATED speech tokens (~4000 for 800 chars). If dilution exists,
  THEY are the crowd. 稿子必须全程可见；真正越积越多的是已生成的 speech
  token——若稀释存在，人海就是它们。

The knife (手术刀): selective sliding window via sdpa monkeypatch —
  keep [sos][text][task][ref_speech] + last W generated tokens,
  mask generated tokens older than W. Removes B's variable (crowd size)
  WITHOUT touching A's variable (distances of remaining tokens).
  只屏蔽 W 之前的已生成 token：减少竞争人数，不改剩余 token 的距离。

Real-scenario conditions (真实场景: 短 ref ~3s + 800 字 target), 2×2 factorial:
  G_win  : 150-char + window      → gate: window must be non-toxic
  F0     : 800-char baseline      → expect ~0.92 (P3 crash)
  F1     : 800-char + NTK4        → fixes A only  (doubles as exp_011e preview)
  F2     : 800-char + window      → fixes B only  (A unfixed — also re-checks A)
  F3     : 800-char + NTK4 + win  → fixes both

Readout (判读): F3 vs F1 = does removing the crowd help AFTER distances are
fixed?  F1−F3 > 0.15 → B confirmed as residual;  < 0.05 → residual is not B.
"""

import sys, os, json, math, time, random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, "/root/CosyVoice")
sys.path.insert(0, "/root/CosyVoice/third_party/Matcha-TTS")
sys.stdout.reconfigure(line_buffering=True)

MD = "/root/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"
OUT = Path("/root/autodl-tmp/exp011_residual")
AUDIO = OUT / "audio"
AUDIO.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = "You are a helpful assistant."
SPEAKER, EMOTION = "0002", "Neutral"
SEEDS = [42, 123, 456]
WINDOW = 200          # keep last 200 generated speech tokens (~8s of audio)
QWEN_HEADS, QWEN_HEAD_DIM = 14, 64

# (name, target_chars, ntk_scale, use_window)
CONDITIONS = [
    ("G_win", 150, 1.0, True),    # gate: window non-toxic on healthy case
    ("F0",    800, 1.0, False),   # crash baseline
    ("F1",    800, 4.0, False),   # NTK only (fix A)
    ("F2",    800, 1.0, True),    # window only (fix B)
    ("F3",    800, 4.0, True),    # both
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


# ═══════════════ knife 1: NTK scaler (validated in exp_011d) ═════════════
class NTKScaler:
    """Frequency-selective range extension. 频率选择性扩程（exp_011d 已验证）。"""

    def __init__(self, llm_root):
        self.mods = []
        for name, mod in llm_root.named_modules():
            if "RotaryEmbedding" in type(mod).__name__ and hasattr(mod, "inv_freq"):
                self.mods.append((name, mod, mod.inv_freq.detach().clone()))
        print(f"[NTK] found {len(self.mods)} rotary module(s)")

    def set_scale(self, s):
        for name, mod, orig in self.mods:
            half = orig.shape[0]
            d = half * 2
            j = torch.arange(half, dtype=torch.float64, device=orig.device)
            factor = float(s) ** (-2.0 * j / (d - 2))
            mod.inv_freq = (orig.to(torch.float64) * factor).to(orig.dtype)
            print(f"[NTK] ENGAGED scale={s}: inv_freq[-1] ratio="
                  f"{float(mod.inv_freq[-1]/orig[-1]):.3f} (must be {1/float(s):.3f})")


# ═══════════════ knife 2: selective sliding window ══════════════════════
_ORIG_SDPA = F.scaled_dot_product_attention


class WindowedSDPA:
    """Masks generated speech tokens older than `window`; prefix untouched.
    屏蔽 window 之前的已生成 speech token；前缀（稿子/ref音色）原样保留。

    LLM decode step filter: q_len==1, 14 heads, head_dim 64 (same as exp2).
    prefix_len is learned from the first decode step's kv length.
    """

    def __init__(self, window):
        self.window = window
        self.prefix_len = None
        self.enabled = False
        self.engaged_logged = False
        self.masked_calls = 0

    def __call__(self, query, key, value, attn_mask=None, dropout_p=0.0,
                 is_causal=False, **kw):
        if not (self.enabled and query.dim() == 4 and query.size(-2) == 1
                and query.size(1) == QWEN_HEADS
                and query.size(-1) == QWEN_HEAD_DIM):
            return _ORIG_SDPA(query, key, value, attn_mask=attn_mask,
                              dropout_p=dropout_p, is_causal=is_causal, **kw)

        kvlen = key.size(-2)
        if self.prefix_len is None:
            self.prefix_len = kvlen - 1          # kv before first generated token
            print(f"[WIN] prefix_len={self.prefix_len} (text+ref preserved), "
                  f"window={self.window}")

        cut_hi = kvlen - self.window             # mask generated tokens [prefix, cut_hi)
        if cut_hi > self.prefix_len:
            if key.size(1) != query.size(1):     # GQA expand
                rep = query.size(1) // key.size(1)
                key = key.repeat_interleave(rep, dim=1)
                value = value.repeat_interleave(rep, dim=1)
            scores = (query @ key.transpose(-1, -2)) / math.sqrt(query.size(-1))
            if attn_mask is not None:
                scores = scores + attn_mask
            scores[..., self.prefix_len:cut_hi] = torch.finfo(scores.dtype).min
            self.masked_calls += 1
            if not self.engaged_logged:
                print(f"[WIN] ENGAGED: masking kv[{self.prefix_len}:{cut_hi}] "
                      f"of {kvlen} (crowd removed, distances untouched)")
                self.engaged_logged = True
            return scores.softmax(dim=-1) @ value
        return _ORIG_SDPA(query, key, value, attn_mask=attn_mask,
                          dropout_p=dropout_p, is_causal=is_causal, **kw)

    def reset(self):
        self.prefix_len = None
        self.engaged_logged = False
        self.masked_calls = 0
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


def select_ref(data_list, speaker, emotion):
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
    return pool[0]


def main():
    t0 = time.monotonic()
    from cosyvoice.cli.cosyvoice import AutoModel

    ref_wav, ref_text = select_ref("/root/autodl-tmp/esd_cn/train.data.list",
                                   SPEAKER, EMOTION)
    print(f"Ref: {ref_text} ({ref_wav})")

    print("Loading CV3 ...")
    model = AutoModel(model_dir=MD)
    frt, cv3m = model.frontend, model.model
    print("  Ready")

    ntk = NTKScaler(cv3m.llm)
    win = WindowedSDPA(WINDOW)

    prompt = SYSTEM_PROMPT + "<|endofprompt|>" + ref_text
    inputs = {}
    for n_chars in sorted({c for _, c, _, _ in CONDITIONS}):
        target = build_target(n_chars)
        sentences = frt.text_normalize(target, split=False, text_frontend=True)
        mi = frt.frontend_zero_shot(str(sentences), prompt, ref_wav,
                                    model.sample_rate, "")
        inputs[n_chars] = (mi, target)
        print(f"  target[{n_chars}]: {len(target)} chars")

    records = []
    for name, n_chars, scale, use_win in CONDITIONS:
        mi, target = inputs[n_chars]
        ntk.set_scale(scale)
        for seed in SEEDS:
            torch.manual_seed(seed)
            torch.cuda.empty_cache()
            win.reset()
            if use_win:
                F.scaled_dot_product_attention = win
                win.enabled = True
            tag = f"{name}_seed{seed}"
            out_wav = AUDIO / f"{tag}.wav"
            rec = {"name": name, "target_chars": n_chars, "ntk_scale": scale,
                   "window": WINDOW if use_win else None, "seed": seed,
                   "target_text": target, "wav_path": str(out_wav)}
            try:
                gen = cv3m.tts(**mi, stream=False)
                audio = torch.cat([j["tts_speech"] for j in gen], dim=1)
                torchaudio.save(str(out_wav), audio, model.sample_rate)
                rec["duration_s"] = round(audio.shape[1] / model.sample_rate, 1)
                rec["masked_calls"] = win.masked_calls
                rec["status"] = "ok"
                print(f"  {tag}: {rec['duration_s']}s "
                      f"(masked_calls={win.masked_calls})")
            except Exception as e:
                rec["status"] = "error"
                rec["error"] = str(e)[:150]
                print(f"  {tag} FAILED: {rec['error']}")
            finally:
                win.enabled = False
                F.scaled_dot_product_attention = _ORIG_SDPA
            records.append(rec)
    ntk.set_scale(1.0)

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

    print(f"\n{'=' * 72}")
    print("EXP_011f: RESIDUAL ATTRIBUTION  (real scenario: 3s ref + long target)")
    print(f"{'=' * 72}")
    print(f"{'Cond':<7s} {'Chars':>6s} {'NTK':>5s} {'Win':>5s} {'CER':>7s} "
          f"{'±':>6s} {'Dur':>7s}")
    print("-" * 72)
    summary = {}
    for name, n_chars, scale, use_win in CONDITIONS:
        items = [r for r in ok if r["name"] == name]
        if not items:
            print(f"{name:<7s} (failed)")
            continue
        cers = [r["CER"] for r in items]
        durs = [r["duration_s"] for r in items]
        summary[name] = {"CER_mean": round(float(np.mean(cers)), 4),
                         "CER_std": round(float(np.std(cers, ddof=1)), 4)
                         if len(cers) > 1 else 0.0,
                         "dur_mean": round(float(np.mean(durs)), 1)}
        s = summary[name]
        print(f"{name:<7s} {n_chars:>6d} {scale:>5.1f} "
              f"{'W' + str(WINDOW) if use_win else '—':>5s} "
              f"{s['CER_mean']:>7.3f} {s['CER_std']:>6.3f} {s['dur_mean']:>6.1f}s")

    print(f"\n{'=' * 72}\nVERDICT\n{'=' * 72}")
    g = lambda k: summary.get(k, {}).get("CER_mean")
    gate = g("G_win") is not None and g("G_win") < 0.10
    print(f"GATE G_win={g('G_win')} (<0.10? {gate})")
    if not gate:
        print("✗ 滑窗对健康样本有毒，B 判读无效。")
    elif all(g(k) is not None for k in ("F0", "F1", "F2", "F3")):
        f0, f1, f2, f3 = g("F0"), g("F1"), g("F2"), g("F3")
        print(f"F0 base={f0}  F1 NTK={f1}  F2 win={f2}  F3 both={f3}")
        print(f"NTK 在真实长文本的效果 (F0→F1): {f0:.3f} → {f1:.3f}")
        d_b = f1 - f3
        print(f"修好 A 之后再除掉人海 (F1→F3): Δ={d_b:+.3f}")
        if d_b > 0.15:
            print("→ B 确认：稀释是 NTK 后残差的实质成分。联合方案 = 最优解。")
        elif d_b < 0.05:
            print("→ B 否定：除掉人海无增益，残差另有来源")
            print("  （NTK scale 不足 / 训练分布其它差异）。")
        else:
            print("→ B 弱贡献：稀释只占残差一小部分。")
        print(f"复核 A (F0→F2, 只除人海不修距离): {f0:.3f} → {f2:.3f} "
              f"(若 A 是主因应改善有限)")

    json.dump({"summary": summary, "window": WINDOW, "runs": records},
              open(OUT / "results.json", "w"), ensure_ascii=False, indent=2)
    print(f"\nSaved: {OUT / 'results.json'}")
    print(f"Total: {(time.monotonic() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
