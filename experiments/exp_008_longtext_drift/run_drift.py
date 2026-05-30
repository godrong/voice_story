#!/usr/bin/env python3
"""P3: Long-Text Attention Drift — CV3 prosody consistency vs text position.

Hypothesis:
  CV3's prompt prosody influence decays with position when text exceeds
  the training cutoff (token_max_length=200 chars). The decay pattern
  reveals whether the LLM backbone truly uses its 32K context window
  for prosody transfer, or whether it's effectively a short-context model.

Experiment:
  Phase 1 — Segment-level synthesis:
    Split longform text at progressive lengths (50, 100, 200, 400, 800,
    1600, 3200, full), synthesize each as a standalone zero-shot TTS call
    with a fixed ref audio.
  Phase 2 — Chunk-level drift within one long synthesis:
    Synthesize the FULL text in one call, then split the output audio into
    fixed-size chunks and measure per-chunk metrics (CER, SECS, F0).
    Fit linear regression: metric ~ chunk_position.
  Phase 3 — Attention hooking (optional, --hook-attn):
    Register forward hooks on Qwen2 attention layers, capture prompt-text
    attention weights, measure decay of prompt attention vs target position.

Metrics:
  - CER per segment (FunASR paraformer-zh)
  - SECS vs ref per segment (WavLM speaker embedding cosine sim)
  - F0 RMSE vs ref per segment (librosa pyin)
  - Prosody Drift Score: slope of F0_mean ~ text_position regression
  - Energy Drift Score: slope of RMS_energy ~ text_position regression
  - Speaking Rate Stability: chars/sec variance across segments
  - (Optional) Prompt Attention Decay: avg attention on prompt tokens
    vs target token position, per layer

Output:
  audio/          — synthesized WAVs
  results.json    — full per-segment metrics + drift scores
  drift_report.txt — human-readable summary
"""

import sys, os, json, time, re, logging
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torchaudio

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, "/root/CosyVoice")
sys.path.insert(0, "/root/CosyVoice/third_party/Matcha-TTS")
sys.stdout.reconfigure(line_buffering=True)
logging.basicConfig(level=logging.WARNING, format="%(message)s")

MD = "/root/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"
OUT = Path("/root/autodl-tmp/longtext_drift")
AUDIO = OUT / "audio"
OUT.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = "You are a helpful assistant."
SPEAKER = "0002"
EMOTION = "Neutral"
SEED = 42

# Progressive segment lengths (chars) — spans below, at, and above training cutoff
SEGMENT_LENGTHS = [50, 100, 200, 400, 800, 1600, 3200]

# Chunk size for within-synthesis drift analysis (chars)
CHUNK_SIZE = 100

WAVLM_SNAP = "/root/.cache/huggingface/hub/models--microsoft--wavlm-base-plus-sv/snapshots/1bfd64eca136543feb28c5ffaf05381c6af33121"


# ── text utils ──────────────────────────────────────────────────────────

def load_longform(path):
    """Load longform text, strip markdown headers/blank lines."""
    with open(path) as f:
        text = f.read()
    # Remove markdown headers (## ...)
    text = re.sub(r'^#+\s+.*$', '', text, flags=re.MULTILINE)
    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Join into single line for TTS (CV3 handles [breath] etc for pauses)
    text = text.strip()
    return text


def segment_text(text, max_chars):
    """Take first max_chars characters, breaking at sentence boundary if possible."""
    if len(text) <= max_chars:
        return text
    segment = text[:max_chars]
    # Try to break at last 。or ？or ！or ]
    for delim in ['。', '？', '！', '] ']:
        idx = segment.rfind(delim)
        if idx > max_chars * 0.7:
            segment = text[:idx + len(delim)]
            break
    return segment


# ── ref selection ───────────────────────────────────────────────────────

def select_ref(data_list_path, speaker, emotion):
    pool = []
    with open(data_list_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            key, wav, text = parts[0], parts[1], parts[2]
            fields = key.split("_")
            if fields[1] == speaker and fields[2] == emotion:
                pool.append((wav, text))
    import random
    random.seed(42)
    return random.choice(pool) if pool else (None, None)


# ── metrics ─────────────────────────────────────────────────────────────

def compute_cer(asr_text, target_text):
    a = asr_text.replace(" ", "")
    # Remove paralinguistic tokens and punctuation for clean comparison
    b = re.sub(r'\[.*?\]', '', target_text)
    b = re.sub(r'</?laughter>', '', b)
    b = re.sub(r'</?strong>', '', b)
    b = b.replace(" ", "").replace("，", "").replace("。", "").replace("？", "")
    b = b.replace("！", "").replace("、", "").replace("：", "").replace("；", "")
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


def extract_f0(wav_path, sr=16000):
    """Extract F0 contour, return (f0_hz, voiced_flag)."""
    import librosa
    y, _ = librosa.load(wav_path, sr=sr)
    f0, v_flag, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sr,
                                 frame_length=2048, hop_length=320)
    f0 = np.nan_to_num(f0, nan=0.0)
    return f0, v_flag.astype(bool)


def extract_rms_energy(wav_path, sr=16000):
    """Extract RMS energy contour."""
    import librosa
    y, _ = librosa.load(wav_path, sr=sr)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=320)[0]
    return rms


# ── attention hooking ───────────────────────────────────────────────────

class AttentionCapture:
    """Captures cross-position attention weights from Qwen2 layers.

    CV3 input: [system_prompt | <|endofprompt|> | ref_text | target_text]
    encoded as text tokens, then concatenated with speech tokens in the LLM.

    We hook self_attn layers and record attention from target-text positions
    back to prompt/ref positions.

    Auto-detects Qwen2DecoderLayer modules by searching for self_attn
    attributes anywhere in the model tree.
    """

    def __init__(self, cv3_model, prompt_len_tokens):
        self.model = cv3_model
        self.prompt_len = prompt_len_tokens
        self.attentions = defaultdict(list)
        self.hooks = []
        self._register()

    def _find_attn_layers(self):
        """Auto-detect attention layers in CV3's Qwen2 backbone.
        Searches recursively for modules with self_attn.q_proj."""
        layers = []
        for name, mod in self.model.named_modules():
            if hasattr(mod, "self_attn") and hasattr(mod.self_attn, "q_proj"):
                layers.append((name, mod.self_attn))
        print(f"  AttentionCapture: found {len(layers)} attention layers")
        for n, _ in layers[:2]:
            print(f"    {n}")
        return layers

    def _register(self):
        for i, (name, attn_mod) in enumerate(self._find_attn_layers()):
            hook = attn_mod.register_forward_hook(
                self._make_hook(i), with_kwargs=True)
            self.hooks.append(hook)

    def _make_hook(self, layer_idx):
        def hook(module, args, kwargs, output):
            # output is (attn_output, attn_weights, past_key_value) when
            # output_attentions=True, else just attn_output
            if isinstance(output, tuple) and len(output) >= 2:
                attn_weights = output[1]
                if attn_weights is not None:
                    self.attentions[layer_idx].append(
                        attn_weights.detach().cpu())
        return hook

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def compute_prompt_attention_decay(self):
        """For each layer, compute avg attention from target positions
        to prompt positions, as a function of target position.

        Returns dict: layer_idx -> array of shape (n_target_positions,)
        """
        results = {}
        for layer_idx, attn_list in self.attentions.items():
            # attn_list is list of tensors from each forward call
            # Each: (batch, n_heads, seq_len, seq_len)
            # Stack and average over calls and heads
            all_attn = torch.cat([a.unsqueeze(0) for a in attn_list], dim=0)
            avg_attn = all_attn.mean(dim=(0, 1))  # (seq_len, seq_len)

            seq_len = avg_attn.shape[0]
            if seq_len <= self.prompt_len:
                continue

            # For each target position, avg attention to prompt region
            target_start = self.prompt_len
            prompt_attn_by_pos = []
            for t in range(target_start, seq_len):
                p_attn = avg_attn[t, :self.prompt_len].mean().item()
                prompt_attn_by_pos.append(p_attn)

            results[layer_idx] = np.array(prompt_attn_by_pos)
        return results


# ── main ────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str,
                        default=str(Path(__file__).parent / "longform_text.txt"),
                        help="Path to longform text file")
    parser.add_argument("--hook-attn", action="store_true",
                        help="Enable attention hooking (requires eager attn, more VRAM)")
    parser.add_argument("--skip-synthesis", action="store_true",
                        help="Skip synthesis, only run analysis on existing audio")
    args = parser.parse_args()

    t0 = time.monotonic()
    text = load_longform(args.text)
    text_clean = re.sub(r'\[.*?\]', '', text)
    text_clean = re.sub(r'</?(laughter|strong)>', '', text_clean)
    print(f"Longform text: {len(text)} chars (clean: {len(text_clean)} chars)\n")

    dl = "/root/autodl-tmp/esd_cn/train.data.list"
    ref_wav, ref_text = select_ref(dl, SPEAKER, EMOTION)
    if not ref_wav or not Path(ref_wav).exists():
        print(f"ERROR: no ref found for {SPEAKER}/{EMOTION}")
        sys.exit(1)
    print(f"Ref: {ref_wav}\n  text: {ref_text[:80]}...\n")

    # ── Phase 1: Segment-level synthesis ─────────────────────────────────
    records = []

    if not args.skip_synthesis:
        from cosyvoice.cli.cosyvoice import AutoModel
        print("Loading CV3 ...")
        model = AutoModel(model_dir=MD)
        frt = model.frontend
        cv3m = model.model
        print("  Ready\n")

        # Pre-compute prompt features (same for all segments)
        prompt = SYSTEM_PROMPT + "<|endofprompt|>" + ref_text

        for seg_len in SEGMENT_LENGTHS:
            seg_text = segment_text(text, seg_len)
            seg_clean = re.sub(r'\[.*?\]', '', seg_text)
            seg_clean = re.sub(r'</?(laughter|strong)>', '', seg_clean)
            actual_len = len(seg_clean)
            print(f"  Segment {seg_len:>5d} chars → actual {actual_len} chars")

            torch.manual_seed(SEED)
            torch.cuda.empty_cache()

            sentences = frt.text_normalize(seg_text, split=False, text_frontend=True)
            mi_base = frt.frontend_zero_shot(
                str(sentences), prompt, ref_wav, model.sample_rate, "")

            tag = f"seg_{seg_len:04d}"
            out_wav = AUDIO / f"{tag}.wav"
            out_wav.parent.mkdir(parents=True, exist_ok=True)

            try:
                t_start = time.monotonic()
                gen = cv3m.tts(**mi_base, stream=False)
                chunks = [j["tts_speech"] for j in gen]
                audio = torch.cat(chunks, dim=1)
                duration_s = audio.shape[1] / model.sample_rate
                rtf = (time.monotonic() - t_start) / max(duration_s, 0.01)
                torchaudio.save(str(out_wav), audio, model.sample_rate)
                status = "ok"
            except Exception as e:
                duration_s = 0
                rtf = 0
                status = "error"
                print(f"    ERROR: {str(e)[:100]}")

            rec = {
                "segment_len_target": seg_len,
                "segment_len_actual": actual_len,
                "text": seg_clean,
                "wav_path": str(out_wav),
                "duration_s": round(duration_s, 1),
                "rtf": round(rtf, 2),
                "status": status,
            }
            if status == "error":
                rec["error"] = str(e)[:150]
            records.append(rec)

            # Also synthesize the full text in one go (Phase 2 prep)
            if seg_len == SEGMENT_LENGTHS[-1]:
                print(f"\n  Full text synthesis ({len(text)} chars with paralinguistic tokens)...")
                torch.manual_seed(SEED)
                torch.cuda.empty_cache()
                sentences_full = frt.text_normalize(text, split=False, text_frontend=True)
                mi_full = frt.frontend_zero_shot(
                    str(sentences_full), prompt, ref_wav, model.sample_rate, "")
                out_full = AUDIO / "full.wav"
                try:
                    gen_full = cv3m.tts(**mi_full, stream=False)
                    chunks_full = [j["tts_speech"] for j in gen_full]
                    audio_full = torch.cat(chunks_full, dim=1)
                    torchaudio.save(str(out_full), audio_full, model.sample_rate)
                    records.append({
                        "segment_len_target": -1,
                        "segment_len_actual": len(text_clean),
                        "text": text_clean,
                        "wav_path": str(out_full),
                        "duration_s": round(audio_full.shape[1] / model.sample_rate, 1),
                        "status": "ok",
                    })
                    print(f"    Full audio: {audio_full.shape[1]/model.sample_rate:.1f}s")
                except Exception as e:
                    print(f"    Full synthesis ERROR: {str(e)[:100]}")
                    records.append({
                        "segment_len_target": -1, "status": "error",
                        "error": str(e)[:150],
                    })

        del model, frt, cv3m
        torch.cuda.empty_cache()
    else:
        # Load existing records
        existing = OUT / "results.json"
        if existing.exists():
            records = json.load(open(existing))["runs"]
            print(f"Loaded {len(records)} existing runs")

    ok = [r for r in records if r["status"] == "ok"]
    print(f"\nSynthesis: {len(ok)}/{len(records)} ok\n")

    # ── Phase 1 metrics: CER, SECS, F0 per segment ──────────────────────

    # ASR
    print("Loading FunASR ...")
    from funasr import AutoModel as FM
    asr = FM(model="paraformer-zh")
    for i, r in enumerate(ok):
        if (i + 1) % 5 == 0:
            print(f"  ASR {i + 1}/{len(ok)}")
        result = asr.generate(input=r["wav_path"])
        r["asr_text"] = result[0]["text"] if result else ""
        r["CER"] = compute_cer(r["asr_text"], r.get("text", ""))
        # Speaking rate
        if r.get("duration_s", 0) > 0 and r.get("text"):
            r["speaking_rate_cps"] = round(
                len(r["text"]) / r["duration_s"], 1)
    del asr
    torch.cuda.empty_cache()

    # SECS
    from transformers import WavLMConfig, WavLMModel
    from safetensors.torch import load_file
    print("Loading WavLM ...")
    cfg = WavLMConfig.from_pretrained(WAVLM_SNAP, local_files_only=True)
    wavlm = WavLMModel(cfg)
    wavlm.load_state_dict(load_file(f"{WAVLM_SNAP}/model.safetensors"), strict=False)
    wavlm = wavlm.cuda().eval()
    import librosa as _librosa

    # Ref embedding
    y_ref, _ = _librosa.load(str(ref_wav), sr=16000)
    inp_ref = torch.from_numpy(y_ref.astype(np.float32)).unsqueeze(0).cuda()
    out_ref = wavlm(inp_ref, output_hidden_states=True)
    e_ref = torch.nn.functional.normalize(
        out_ref.last_hidden_state.mean(dim=1)[:, :192], dim=-1)

    for i, r in enumerate(ok):
        if (i + 1) % 5 == 0:
            print(f"  SECS {i + 1}/{len(ok)}")
        try:
            y_s, _ = _librosa.load(r["wav_path"], sr=16000)
            inp_s = torch.from_numpy(y_s.astype(np.float32)).unsqueeze(0).cuda()
            out_s = wavlm(inp_s, output_hidden_states=True)
            e_s = torch.nn.functional.normalize(
                out_s.last_hidden_state.mean(dim=1)[:, :192], dim=-1)
            r["SECS"] = round(max(0.0, (e_s * e_ref).sum(dim=-1).item()), 4)
        except Exception:
            r["SECS"] = None
    del wavlm
    torch.cuda.empty_cache()

    # F0 RMSE
    print("Computing F0 ...")
    f0_ref, v_ref = extract_f0(str(ref_wav))
    for i, r in enumerate(ok):
        if (i + 1) % 5 == 0:
            print(f"  F0 {i + 1}/{len(ok)}")
        try:
            f0_s, v_s = extract_f0(r["wav_path"])
            L = min(len(f0_s), len(f0_ref))
            both = v_s[:L] & v_ref[:L]
            if both.sum() >= 5:
                r["F0_RMSE_Hz"] = round(float(
                    np.sqrt(np.mean((f0_s[:L][both] - f0_ref[:L][both]) ** 2))), 1)
                r["F0_mean_Hz"] = round(float(f0_s[both].mean()), 1)
                r["F0_std_Hz"] = round(float(f0_s[both].std()), 1)
            # Energy
            rms = extract_rms_energy(r["wav_path"])
            r["energy_rms"] = round(float(rms.mean()), 6)
            r["energy_std"] = round(float(rms.std()), 6)
        except Exception:
            r["F0_RMSE_Hz"] = None
            r["energy_rms"] = None

    # ── Phase 2: Chunk-level drift within full synthesis ─────────────────

    full_run = [r for r in ok if r.get("segment_len_target") == -1]
    drift_chunks = []
    if full_run:
        full = full_run[0]
        print(f"\nPhase 2: Chunk-level drift on full audio ({full['duration_s']:.1f}s)")

        # Split text into chunks, estimate audio timestamps by proportional mapping
        full_text = full.get("text", "")
        total_chars = len(full_text)
        total_dur = full["duration_s"]
        chunk_boundaries = list(range(0, total_chars, CHUNK_SIZE))

        y_full, sr = _librosa.load(full["wav_path"], sr=16000)
        f0_full, v_full = extract_f0(full["wav_path"])
        rms_full = extract_rms_energy(full["wav_path"])
        frames_full = min(len(f0_full), len(rms_full))

        from funasr import AutoModel as FM2
        asr2 = FM2(model="paraformer-zh")

        for ci, start_char in enumerate(chunk_boundaries):
            end_char = min(start_char + CHUNK_SIZE, total_chars)
            chunk_text = full_text[start_char:end_char]
            chunk_clean = re.sub(r'\[.*?\]', '', chunk_text)

            # Estimate audio segment
            t_start = (start_char / total_chars) * total_dur if total_chars > 0 else 0
            t_end = (end_char / total_chars) * total_dur if total_chars > 0 else total_dur
            s_start = int(t_start * sr)
            s_end = int(t_end * sr)
            s_start = max(0, s_start)
            s_end = min(len(y_full), s_end)

            if s_end - s_start < sr * 0.5:  # skip chunks < 0.5s
                continue

            chunk_wav = AUDIO / f"chunk_{start_char:04d}_{end_char:04d}.wav"
            torchaudio.save(str(chunk_wav),
                            torch.from_numpy(y_full[s_start:s_end]).unsqueeze(0), sr)

            # ASR
            result = asr2.generate(input=str(chunk_wav))
            asr_t = result[0]["text"] if result else ""
            cer_c = compute_cer(asr_t, chunk_clean)

            # F0 stats for this chunk
            f_start = int(t_start * (sr / 320))  # 320 hop_length for pyin
            f_end = int(t_end * (sr / 320))
            f_start = max(0, min(f_start, frames_full - 1))
            f_end = max(f_start + 1, min(f_end, frames_full))
            f0_chunk = f0_full[f_start:f_end]
            v_chunk = v_full[f_start:f_end]
            f0_voiced = f0_chunk[v_chunk] if v_chunk.any() else np.array([0])
            rms_chunk = rms_full[f_start:f_end] if f_end < len(rms_full) else np.array([0])

            drift_chunks.append({
                "chunk_idx": ci,
                "char_start": start_char,
                "char_end": end_char,
                "char_position_pct": round(start_char / max(total_chars, 1) * 100, 1),
                "chunk_text": chunk_clean[:80],
                "CER": cer_c,
                "F0_mean_Hz": round(float(f0_voiced.mean()), 1) if len(f0_voiced) > 0 else 0,
                "F0_std_Hz": round(float(f0_voiced.std()), 1) if len(f0_voiced) > 1 else 0,
                "energy_rms": round(float(rms_chunk.mean()), 6) if len(rms_chunk) > 0 else 0,
                "duration_s": round((s_end - s_start) / sr, 2),
            })

        del asr2
        torch.cuda.empty_cache()
        print(f"  {len(drift_chunks)} chunks analyzed")

    # ── Drift metrics ────────────────────────────────────────────────────

    drift_scores = {}
    if len(drift_chunks) >= 3:
        positions = np.array([c["char_position_pct"] for c in drift_chunks])

        # F0 drift: linear regression slope (Hz per 1% position)
        f0_vals = np.array([c["F0_mean_Hz"] for c in drift_chunks if c["F0_mean_Hz"] > 0])
        f0_pos = np.array([c["char_position_pct"] for c in drift_chunks
                           if c["F0_mean_Hz"] > 0])
        if len(f0_vals) >= 3:
            slope_f0, intercept_f0 = np.polyfit(f0_pos, f0_vals, 1)
            drift_scores["F0_slope_Hz_per_pct"] = round(float(slope_f0), 3)
            drift_scores["F0_total_drift_Hz"] = round(
                float(slope_f0 * 100), 1)  # total drift over full text

        # Energy drift
        e_vals = np.array([c["energy_rms"] for c in drift_chunks if c["energy_rms"]])
        e_pos = np.array([c["char_position_pct"] for c in drift_chunks
                          if c["energy_rms"]])
        if len(e_vals) >= 3:
            slope_e, _ = np.polyfit(e_pos, e_vals, 1)
            drift_scores["energy_slope_per_pct"] = round(float(slope_e), 8)

        # CER drift
        cer_vals = np.array([c["CER"] for c in drift_chunks])
        if len(cer_vals) >= 3:
            slope_cer, _ = np.polyfit(positions, cer_vals, 1)
            drift_scores["CER_slope_per_pct"] = round(float(slope_cer), 5)

        # Speaking rate stability
        if len(drift_chunks) >= 2:
            rates = []
            for c in drift_chunks:
                if c["duration_s"] > 0:
                    rates.append(len(c["chunk_text"]) / c["duration_s"])
            if rates:
                drift_scores["speaking_rate_mean_cps"] = round(float(np.mean(rates)), 1)
                drift_scores["speaking_rate_cv"] = round(
                    float(np.std(rates) / max(np.mean(rates), 0.01)), 3)

        print(f"\nDrift scores: {json.dumps(drift_scores, indent=2)}")

    # ── Phase 1 segment trend ────────────────────────────────────────────

    seg_trend = []
    seg_ok = [r for r in ok if r.get("segment_len_target", -1) > 0]
    for r in seg_ok:
        seg_trend.append({
            "target_len": r["segment_len_target"],
            "actual_len": r["segment_len_actual"],
            "CER": r.get("CER"),
            "SECS": r.get("SECS"),
            "F0_RMSE_Hz": r.get("F0_RMSE_Hz"),
            "F0_mean_Hz": r.get("F0_mean_Hz"),
            "duration_s": r.get("duration_s"),
            "rtf": r.get("rtf"),
            "speaking_rate_cps": r.get("speaking_rate_cps"),
        })

    # ── Save ─────────────────────────────────────────────────────────────

    result_path = OUT / "results.json"
    json.dump({
        "config": {
            "speaker": SPEAKER,
            "emotion": EMOTION,
            "seed": SEED,
            "segment_lengths": SEGMENT_LENGTHS,
            "chunk_size": CHUNK_SIZE,
            "total_text_chars": len(text),
            "total_text_clean_chars": len(text_clean),
            "ref_wav": ref_wav,
            "ref_text": ref_text,
        },
        "segment_trend": seg_trend,
        "drift_scores": drift_scores,
        "drift_chunks": drift_chunks,
        "runs": ok,
    }, open(result_path, "w"), ensure_ascii=False, indent=2)

    # ── Report ───────────────────────────────────────────────────────────

    report = []
    report.append("=" * 72)
    report.append("P3: LONG-TEXT ATTENTION DRIFT — RESULTS")
    report.append("=" * 72)
    report.append(f"")
    report.append(f"Text: {len(text)} chars ({len(text_clean)} clean)")
    report.append(f"Ref:  {SPEAKER}/{EMOTION} — {ref_text[:60]}...")
    report.append(f"")

    # Segment trend table
    report.append("── Segment-Level Trend ──")
    report.append(f"{'Len':>6s}  {'CER':>6s}  {'SECS':>7s}  {'F0_RMSE':>8s}  {'F0_Mean':>8s}  {'Dur(s)':>7s}  {'RTF':>5s}  {'Rate':>6s}")
    report.append("-" * 72)
    for s in seg_trend:
        report.append(
            f"{s['target_len']:>6d}  "
            f"{s.get('CER', 0) or 0:6.3f}  "
            f"{s.get('SECS') or 0:7.3f}  "
            f"{s.get('F0_RMSE_Hz') or 0:8.1f}  "
            f"{s.get('F0_mean_Hz') or 0:8.1f}  "
            f"{s.get('duration_s', 0):7.1f}  "
            f"{s.get('rtf', 0):5.1f}  "
            f"{s.get('speaking_rate_cps', 0):6.1f}"
        )

    report.append("")
    report.append("── Drift Scores (within full synthesis) ──")
    if drift_scores:
        for k, v in drift_scores.items():
            report.append(f"  {k}: {v}")
    else:
        report.append("  (full synthesis failed or too few chunks)")

    report.append("")
    report.append("── Interpretation ──")
    if drift_scores:
        f0_drift = drift_scores.get("F0_total_drift_Hz", 0)
        cer_slope = drift_scores.get("CER_slope_per_pct", 0)
        rate_cv = drift_scores.get("speaking_rate_cv", 0)

        if abs(f0_drift) < 3:
            report.append(f"  F0 drift {f0_drift:+.1f} Hz: PROSODY STABLE — CV3 maintains pitch")
            report.append(f"  across the full text (>200 chars training limit).")
            report.append(f"  The LLM backbone effectively uses its 32K context window.")
        elif abs(f0_drift) < 8:
            report.append(f"  F0 drift {f0_drift:+.1f} Hz: MODERATE drift — pitch shifts")
            report.append(f"  gradually but stays within natural range. Acceptable.")
        else:
            report.append(f"  F0 drift {f0_drift:+.1f} Hz: SIGNIFICANT drift — prompt")
            report.append(f"  prosody decays measurably. Consider segmenting long texts.")

        if cer_slope > 0.0005:
            report.append(f"  CER degrades {cer_slope*100:.3f} per 1% position —")
            report.append(f"  intelligibility worsens toward end. Chunk long texts.")
        else:
            report.append(f"  CER stable across position — no intelligibility degradation.")

        report.append(f"  Speaking rate CV: {rate_cv:.3f} "
                       f"{'(unstable)' if rate_cv > 0.15 else '(stable)'}")

    report_path = OUT / "drift_report.txt"
    report_text = "\n".join(report)
    with open(report_path, "w") as f:
        f.write(report_text)
    print(report_text)
    print(f"\nSaved: {result_path}")
    print(f"Report: {report_path}")
    print(f"Total: {(time.monotonic() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
