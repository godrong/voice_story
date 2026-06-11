#!/usr/bin/env python3
"""exp_011-2: Attention & entropy probe — fingerprint the crash mechanism.
exp_011-2: 注意力与熵探针 — 给崩溃机制取"指纹"。

Method (方法):
  Monkeypatch torch.nn.functional.scaled_dot_product_attention so every LLM
  decode step computes attention weights manually and records statistics.
  通过替换 sdpa 函数，在每个解码步手工计算 attention 权重并记录统计量。
  No model surgery, works regardless of transformers version.

Two runs, same ref, seed 42 (两次生成对比):
  HEALTHY: 150-char target (normal)   CRASH: 800-char target (P3 crash)

Per-step recordings (每步记录):
  - attention entropy per layer        注意力熵（B 假说指纹：熵升高=权重糊掉）
  - mass on target-text region         对目标文本区的注意力质量（"还在读稿吗"）
  - text cursor = argmax within text   文本指针（健康时应单调前进，崩溃时停滞/乱跳）
  - argmax absolute position           绝对聚焦位置（A 假说指纹：乱指远处）
  - logits entropy from llm_decoder    输出熵（D 假说指纹：随步数单调爬升）

Fingerprint table (指纹判读表):
  A RoPE OOD     : attention argmax scatters to irrelevant far positions
  B dilution     : attention entropy rises, mass_text collapses, flat-ish logits
  D accumulation : attention stays sharp, but logits entropy climbs with steps
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
OUT = Path("/root/autodl-tmp/exp011_attention_probe")
OUT.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = "You are a helpful assistant."
SPEAKER, EMOTION = "0002", "Neutral"
SEED = 42

QWEN_HEADS, QWEN_HEAD_DIM, QWEN_LAYERS = 14, 64, 24   # Qwen2-0.5B
SAMPLE_LAYERS = (0, 6, 12, 18, 23)
SNAPSHOT_EVERY = 25        # full attention vector snapshot interval

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
    text = BASE_STORY
    while len(text) < n_chars:
        text += BASE_STORY
    seg = text[:n_chars]
    for delim in ("。", "！", "？"):
        idx = seg.rfind(delim)
        if idx > n_chars * 0.7:
            return seg[: idx + 1]
    return seg


# ───────────────────────── attention probe ──────────────────────────────

class AttnProbe:
    """Replaces F.scaled_dot_product_attention; records decode-step stats.
    替换 sdpa；仅拦截 LLM 解码步（q_len==1, 14 heads, head_dim 64）。

    Step/layer indexing derives from kv length: kv_len changes → new step.
    步号由 kv 长度推断（kv 变长即进入下一步），层号是步内调用序。
    """

    def __init__(self, regions: dict):
        self.regions = regions          # {"text_lo","text_hi","speech_lo","speech_hi"}
        self.stats = {l: [] for l in SAMPLE_LAYERS}   # per-layer per-step dicts
        self.snapshots = []             # (step, layer, head-mean vec)
        self._cur_kvlen = -1
        self._layer_in_step = 0
        self._step = -1
        self.prefix_kvlen = None
        self.enabled = False

    def __call__(self, query, key, value, attn_mask=None, dropout_p=0.0,
                 is_causal=False, **kw):
        if not (self.enabled and query.dim() == 4 and query.size(-2) == 1
                and query.size(1) == QWEN_HEADS
                and query.size(-1) == QWEN_HEAD_DIM):
            return _ORIG_SDPA(query, key, value, attn_mask=attn_mask,
                              dropout_p=dropout_p, is_causal=is_causal, **kw)

        # GQA: expand kv heads if needed / GQA 时扩展 kv 头数
        if key.size(1) != query.size(1):
            rep = query.size(1) // key.size(1)
            key = key.repeat_interleave(rep, dim=1)
            value = value.repeat_interleave(rep, dim=1)

        scores = (query @ key.transpose(-1, -2)) / math.sqrt(query.size(-1))
        if attn_mask is not None:
            scores = scores + attn_mask
        w = scores.softmax(dim=-1)                  # (1, H, 1, K)
        out = w @ value

        kvlen = key.size(-2)
        if kvlen != self._cur_kvlen:                # new decode step
            self._cur_kvlen = kvlen
            self._layer_in_step = 0
            self._step += 1
            if self.prefix_kvlen is None:
                self.prefix_kvlen = kvlen - 1       # kv before first gen token
        layer = self._layer_in_step
        self._layer_in_step += 1

        if layer in self.stats:
            p = w.mean(dim=1)[0, 0].float()         # head-mean (K,)
            ent = float(-(p * (p + 1e-12).log()).sum())
            lo, hi = self.regions["text_lo"], self.regions["text_hi"]
            mass_text = float(p[lo:hi].sum()) if hi <= kvlen else 0.0
            cursor = int(p[lo:hi].argmax()) if hi <= kvlen and hi > lo else -1
            self.stats[layer].append({
                "step": self._step, "kvlen": kvlen, "entropy": ent,
                "mass_text": mass_text,
                "mass_last50": float(p[-50:].sum()),
                "argmax": int(p.argmax()),
                "cursor": cursor,
            })
            if self._step % SNAPSHOT_EVERY == 0:
                self.snapshots.append((self._step, layer,
                                       p.cpu().numpy().astype(np.float16)))
        return out


_ORIG_SDPA = F.scaled_dot_product_attention


# ───────────────────────── logits probe ─────────────────────────────────

class LogitsProbe:
    """Forward hook on llm_decoder Linear → per-step output entropy.
    挂在 llm_decoder 上的钩子，记录每步输出分布的熵与 top-1 概率。"""

    def __init__(self):
        self.entropy, self.top1 = [], []
        self.enabled = False

    def hook(self, module, inputs, output):
        if not self.enabled:
            return
        logits = output.float().flatten(0, -2)[-1]   # last position (vocab,)
        p = logits.softmax(-1)
        self.entropy.append(float(-(p * (p + 1e-12).log()).sum()))
        self.top1.append(float(p.max()))


# ───────────────────────── helpers ──────────────────────────────────────

def select_ref(data_list, speaker, emotion):
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
    return pool[0]


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


def trend(xs):
    """Linear-fit slope normalized by mean — 'how much does it climb'.
    线性拟合斜率/均值 — 衡量随步数爬升幅度。"""
    if len(xs) < 10:
        return 0.0
    t = np.arange(len(xs))
    k = np.polyfit(t, np.asarray(xs, dtype=np.float64), 1)[0]
    return float(k * len(xs) / (np.mean(xs) + 1e-9))


def main():
    t0 = time.monotonic()
    from cosyvoice.cli.cosyvoice import AutoModel
    import inspect

    ref_wav, ref_text = select_ref("/root/autodl-tmp/esd_cn/train.data.list",
                                   SPEAKER, EMOTION)
    print(f"Ref: {ref_wav} → {ref_text}")

    print("Loading CV3 ...")
    model = AutoModel(model_dir=MD)
    frt, cv3m = model.frontend, model.model
    print("  Ready")

    # Dump inference layout source for region-boundary verification.
    # 打印 inference 源码片段，人工核对序列布局 [sos][text][task][speech]。
    try:
        src = inspect.getsource(type(cv3m.llm).inference)
        concat_lines = [l.strip() for l in src.split("\n")
                        if "concat" in l or "lm_input" in l][:8]
        print("\n--- llm.inference layout lines (verify region order) ---")
        for l in concat_lines:
            print("   ", l)
        print("---\n")
    except Exception as e:
        print(f"(could not dump source: {e})")

    # Locate llm_decoder Linear / 定位输出头
    lp = LogitsProbe()
    decoder = None
    for name, mod in cv3m.llm.named_modules():
        if name.endswith("llm_decoder"):
            decoder = mod
            break
    if decoder is not None:
        decoder.register_forward_hook(lp.hook)
        print(f"Hooked llm_decoder: {decoder}")
    else:
        print("WARN: llm_decoder not found, logits probe disabled")

    from funasr import AutoModel as FM

    runs = {}
    for run_name, n_chars in (("HEALTHY", 150), ("CRASH", 800)):
        target = build_target(n_chars)
        prompt = SYSTEM_PROMPT + "<|endofprompt|>" + ref_text
        sentences = frt.text_normalize(target, split=False, text_frontend=True)
        mi = frt.frontend_zero_shot(str(sentences), prompt, ref_wav,
                                    model.sample_rate, "")

        prompt_text_len = int(mi["prompt_text"].shape[1])
        total_text_len = prompt_text_len + int(mi["text"].shape[1])
        speech_len = int(mi["llm_prompt_speech_token"].shape[1])
        # Layout assumption: [sos][prompt_text+target_text][task][ref_speech]
        regions = {
            "text_lo": 1 + prompt_text_len,           # target text start
            "text_hi": 1 + total_text_len,            # target text end
            "speech_lo": 2 + total_text_len,
            "speech_hi": 2 + total_text_len + speech_len,
        }
        print(f"\n=== {run_name}: {len(target)} chars | "
              f"prompt_text={prompt_text_len} tok, target_text="
              f"{total_text_len - prompt_text_len} tok, ref_speech={speech_len} tok")
        print(f"    regions: {regions}")

        probe = AttnProbe(regions)
        F.scaled_dot_product_attention = probe
        probe.enabled = True
        lp.enabled = True
        lp.entropy, lp.top1 = [], []

        torch.manual_seed(SEED)
        torch.cuda.empty_cache()
        ts = time.monotonic()
        gen = cv3m.tts(**mi, stream=False)
        audio = torch.cat([j["tts_speech"] for j in gen], dim=1)
        elapsed = time.monotonic() - ts

        probe.enabled = False
        lp.enabled = False
        F.scaled_dot_product_attention = _ORIG_SDPA

        wav_path = OUT / f"{run_name.lower()}.wav"
        torchaudio.save(str(wav_path), audio, model.sample_rate)
        n_steps = probe._step + 1
        print(f"    {audio.shape[1] / model.sample_rate:.1f}s audio, "
              f"{n_steps} decode steps, {elapsed:.0f}s elapsed "
              f"(prefix kv={probe.prefix_kvlen})")

        runs[run_name] = {
            "target": target, "wav_path": str(wav_path),
            "probe": probe, "logits_entropy": list(lp.entropy),
            "logits_top1": list(lp.top1), "n_steps": n_steps,
            "regions": regions,
        }

        # Save raw arrays / 保存原始数据
        np.savez_compressed(
            OUT / f"{run_name.lower()}_probe.npz",
            **{f"layer{l}_{k}": np.array([s[k] for s in probe.stats[l]])
               for l in SAMPLE_LAYERS if probe.stats[l]
               for k in ("step", "entropy", "mass_text", "mass_last50",
                         "argmax", "cursor")},
            logits_entropy=np.array(lp.entropy),
            logits_top1=np.array(lp.top1),
            snapshots_step=np.array([s[0] for s in probe.snapshots]),
            snapshots_layer=np.array([s[1] for s in probe.snapshots]),
        )
        np.save(OUT / f"{run_name.lower()}_snapshots.npy",
                np.array([s[2] for s in probe.snapshots], dtype=object),
                allow_pickle=True)

    # CER check / 复核两次生成的 CER
    asr = FM(model="paraformer-zh", disable_update=True)
    for rn, r in runs.items():
        res = asr.generate(input=r["wav_path"])
        r["asr_text"] = res[0]["text"] if res else ""
        r["CER"] = compute_cer(r["asr_text"], r["target"])
    del asr

    # ── Fingerprint analysis / 指纹分析 ──────────────────────────────────
    print(f"\n{'=' * 78}\nFINGERPRINT ANALYSIS\n{'=' * 78}")
    report = {}
    for rn, r in runs.items():
        probe = r["probe"]
        mid = QWEN_LAYERS // 2
        layer = mid if probe.stats.get(mid) else SAMPLE_LAYERS[2]
        st = probe.stats[12]
        ents = [s["entropy"] for s in st]
        masses = [s["mass_text"] for s in st]
        cursors = [s["cursor"] for s in st]
        q = max(1, len(ents) // 4)

        cursor_adv = float(np.mean(np.diff(cursors) > 0)) if len(cursors) > 2 else 0
        rep = {
            "CER": r["CER"],
            "n_steps": r["n_steps"],
            "attn_entropy_first_quarter": round(float(np.mean(ents[:q])), 3),
            "attn_entropy_last_quarter": round(float(np.mean(ents[-q:])), 3),
            "attn_entropy_trend": round(trend(ents), 3),
            "mass_text_first_quarter": round(float(np.mean(masses[:q])), 4),
            "mass_text_last_quarter": round(float(np.mean(masses[-q:])), 4),
            "cursor_advance_ratio": round(cursor_adv, 3),
            "logits_entropy_trend": round(trend(r["logits_entropy"]), 3),
            "logits_entropy_first_quarter":
                round(float(np.mean(r["logits_entropy"][:q])), 3)
                if r["logits_entropy"] else None,
            "logits_entropy_last_quarter":
                round(float(np.mean(r["logits_entropy"][-q:])), 3)
                if r["logits_entropy"] else None,
        }
        report[rn] = rep
        print(f"\n[{rn}] CER={rep['CER']}, steps={rep['n_steps']} (layer 12)")
        print(f"  attn entropy  : {rep['attn_entropy_first_quarter']} → "
              f"{rep['attn_entropy_last_quarter']} (trend {rep['attn_entropy_trend']})")
        print(f"  mass on text  : {rep['mass_text_first_quarter']} → "
              f"{rep['mass_text_last_quarter']}")
        print(f"  cursor advance: {rep['cursor_advance_ratio']} of steps")
        print(f"  logits entropy: {rep['logits_entropy_first_quarter']} → "
              f"{rep['logits_entropy_last_quarter']} "
              f"(trend {rep['logits_entropy_trend']})")

    # Auto verdict / 自动判读
    print(f"\n{'=' * 78}\nVERDICT\n{'=' * 78}")
    if "HEALTHY" in report and "CRASH" in report:
        h, c = report["HEALTHY"], report["CRASH"]
        d_attn_ent = c["attn_entropy_trend"] - h["attn_entropy_trend"]
        d_logit_ent = c["logits_entropy_trend"] - h["logits_entropy_trend"]
        mass_collapse = (c["mass_text_last_quarter"]
                         < 0.5 * c["mass_text_first_quarter"])
        cursor_broken = c["cursor_advance_ratio"] < 0.6 * h["cursor_advance_ratio"]

        print(f"Δattn-entropy-trend (CRASH−HEALTHY): {d_attn_ent:+.3f}")
        print(f"Δlogits-entropy-trend             : {d_logit_ent:+.3f}")
        print(f"mass_text collapsed in CRASH       : {mass_collapse}")
        print(f"text cursor broken in CRASH        : {cursor_broken}")
        print()
        if mass_collapse or cursor_broken:
            print("→ Model STOPS READING THE SCRIPT mid-generation:")
            print("  attention to target-text collapses / cursor stalls.")
            if d_attn_ent > 0.1:
                print("  + entropy rises → consistent with B (attention dilution).")
            else:
                print("  + entropy flat → focus shifts elsewhere, check argmax")
                print("    distribution (A: scattered far positions).")
        elif d_logit_ent > 0.1:
            print("→ Attention intact but output uncertainty climbs with steps:")
            print("  consistent with D (autoregressive error accumulation).")
        else:
            print("→ No clean fingerprint — inspect snapshots npz manually,")
            print("  and cross-check with exp1 position-sweep verdict.")

    json.dump({k: {kk: vv for kk, vv in v.items()} for k, v in report.items()},
              open(OUT / "fingerprint_report.json", "w"), indent=2,
              ensure_ascii=False)

    # Plots / 可视化
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        for rn, color in (("HEALTHY", "tab:green"), ("CRASH", "tab:red")):
            st = runs[rn]["probe"].stats[12]
            steps = [s["step"] for s in st]
            axes[0, 0].plot(steps, [s["entropy"] for s in st],
                            color=color, label=rn, alpha=0.7)
            axes[0, 1].plot(steps, [s["mass_text"] for s in st],
                            color=color, label=rn, alpha=0.7)
            axes[1, 0].plot(steps, [s["cursor"] for s in st],
                            color=color, label=rn, alpha=0.7, lw=0.8)
            axes[1, 1].plot(runs[rn]["logits_entropy"],
                            color=color, label=rn, alpha=0.7)
        axes[0, 0].set_title("Attention entropy (layer 12)")
        axes[0, 1].set_title("Attention mass on target text")
        axes[1, 0].set_title("Text cursor (argmax within text region)")
        axes[1, 1].set_title("Logits entropy per step")
        for ax in axes.flat:
            ax.legend()
            ax.set_xlabel("decode step")
        plt.tight_layout()
        plt.savefig(OUT / "probe_curves.png", dpi=130)
        print(f"\nPlots: {OUT / 'probe_curves.png'}")
    except Exception as e:
        print(f"(plotting skipped: {e})")

    print(f"\nSaved: {OUT}")
    print(f"Total: {(time.monotonic() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
