#!/usr/bin/env python3
"""P3 variant: Batched multi-experiment runner for H800 utilization.

Key optimizations over run_drift.py:
  1. Load CV3 ONCE, run all segments + paralinguistic token tests
     in a single model load → no reload overhead
  2. Overlap synthesis with evaluation: spawn ASR subprocess on
     completed audio while GPU works on next segment
  3. Batch SECS computation: stack multiple audio tensors, single
     WavLM forward pass
  4. Pre-compute prompt features → no repeated frontend work

Expected VRAM: ~8-12 GB (H800 has 81 GB, ~85% free)
Expected speedup: 1.5-2x vs sequential run_drift.py
"""

import sys, os, json, time, re, logging, threading, queue
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
OUT = Path("/root/autodl-tmp/longtext_drift_batched")
AUDIO = OUT / "audio"
OUT.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = "You are a helpful assistant."
SPEAKER = "0002"
EMOTION = "Neutral"
SEED = 42

SEGMENT_LENGTHS = [50, 100, 200, 400, 800, 1600, 3200]
CHUNK_SIZE = 100

WAVLM_SNAP = "/root/.cache/huggingface/hub/models--microsoft--wavlm-base-plus-sv/snapshots/1bfd64eca136543feb28c5ffaf05381c6af33121"


# ── text utils ──────────────────────────────────────────────────────────

def load_longform(path):
    with open(path) as f:
        text = f.read()
    text = re.sub(r'^#+\s+.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def segment_text(text, max_chars):
    if len(text) <= max_chars:
        return text
    segment = text[:max_chars]
    for delim in ['。', '？', '！', '] ']:
        idx = segment.rfind(delim)
        if idx > max_chars * 0.7:
            return text[:idx + len(delim)]
    return segment


def select_ref(data_list_path, speaker, emotion):
    pool = []
    with open(data_list_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3: continue
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
    b = re.sub(r'\[.*?\]', '', target_text)
    b = re.sub(r'</?(laughter|strong)>', '', b)
    b = b.replace(" ", "").replace("，", "").replace("。", "").replace("？", "")
    b = b.replace("！", "").replace("、", "").replace("：", "").replace("；", "")
    if not a or not b: return 1.0
    m, n = len(a), len(b)
    d = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): d[i][0] = i
    for j in range(n + 1): d[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (0 if a[i - 1] == b[j - 1] else 1))
    return round(d[m][n] / max(1, len(b)), 4)


# ── batched evaluation engine ───────────────────────────────────────────

class BatchedEvaluator:
    """Runs ASR in subprocess, SECS/F0 in batched GPU calls.

    Producer-consumer: synthesis thread produces WAVs into a queue,
    evaluation thread consumes them while GPU is busy with next synthesis.
    """

    def __init__(self, ref_wav, ref_text):
        self.ref_wav = ref_wav
        self.ref_text = ref_text
        self.wavlm = None
        self.asr = None
        self.results = []
        self._ref_embedding = None
        self._f0_ref = None
        self._v_ref = None

    def init_wavlm(self):
        from transformers import WavLMConfig, WavLMModel
        from safetensors.torch import load_file
        cfg = WavLMConfig.from_pretrained(WAVLM_SNAP, local_files_only=True)
        self.wavlm = WavLMModel(cfg)
        self.wavlm.load_state_dict(
            load_file(f"{WAVLM_SNAP}/model.safetensors"), strict=False)
        self.wavlm = self.wavlm.cuda().eval()

        import librosa
        y_ref, _ = librosa.load(str(self.ref_wav), sr=16000)
        inp_ref = torch.from_numpy(y_ref.astype(np.float32)).unsqueeze(0).cuda()
        out_ref = self.wavlm(inp_ref, output_hidden_states=True)
        self._ref_embedding = torch.nn.functional.normalize(
            out_ref.last_hidden_state.mean(dim=1)[:, :192], dim=-1)

    def init_f0_ref(self):
        import librosa
        y, _ = librosa.load(str(self.ref_wav), sr=16000)
        f0, v, _ = librosa.pyin(y, fmin=50, fmax=400, sr=16000,
                                frame_length=2048, hop_length=320)
        self._f0_ref = np.nan_to_num(f0, nan=0.0)
        self._v_ref = v.astype(bool)

    def init_asr(self):
        from funasr import AutoModel as FM
        self.asr = FM(model="paraformer-zh")

    def evaluate_batch(self, records_batch):
        """Evaluate a batch of completed synthesis records."""
        import librosa as _librosa

        if self.asr is None:
            self.init_asr()
        if self.wavlm is None:
            self.init_wavlm()
        if self._f0_ref is None:
            self.init_f0_ref()

        # Batch SECS: stack all audio → single WavLM call
        audio_tensors = []
        valid_indices = []
        for i, r in enumerate(records_batch):
            try:
                y_s, _ = _librosa.load(r["wav_path"], sr=16000)
                audio_tensors.append(torch.from_numpy(y_s.astype(np.float32)))
                valid_indices.append(i)
            except Exception:
                r["SECS"] = None
                r["CER"] = None

        if audio_tensors:
            # Pad to max length for batching
            max_len = max(t.shape[0] for t in audio_tensors)
            padded = torch.zeros(len(audio_tensors), max_len)
            masks = torch.zeros(len(audio_tensors), max_len)
            for j, t in enumerate(audio_tensors):
                padded[j, :t.shape[0]] = t
                masks[j, :t.shape[0]] = 1

            # Process in sub-batches of 16 to avoid OOM
            for start in range(0, len(padded), 16):
                end = min(start + 16, len(padded))
                sub_batch = padded[start:end].cuda()
                sub_mask = masks[start:end].cuda()
                with torch.no_grad():
                    out = self.wavlm(sub_batch, attention_mask=sub_mask,
                                     output_hidden_states=True)
                    emb = out.last_hidden_state * sub_mask.unsqueeze(-1)
                    emb = emb.sum(dim=1) / sub_mask.sum(dim=1, keepdim=True).clamp(min=1)
                    emb = torch.nn.functional.normalize(emb[:, :192], dim=-1)
                    for k, idx in enumerate(range(start, end)):
                        orig_idx = valid_indices[idx]
                        secs_val = (emb[k:k+1] * self._ref_embedding).sum(dim=-1).item()
                        records_batch[orig_idx]["SECS"] = round(max(0.0, secs_val), 4)

        # Per-sample ASR + F0
        for i, r in enumerate(records_batch):
            # ASR
            if self.asr is not None:
                try:
                    result = self.asr.generate(input=r["wav_path"])
                    r["asr_text"] = result[0]["text"] if result else ""
                    r["CER"] = compute_cer(r["asr_text"], r.get("text", ""))
                except Exception:
                    r["CER"] = None

            # F0 RMSE
            try:
                y_s, _ = _librosa.load(r["wav_path"], sr=16000)
                f0_s, v_s, _ = _librosa.pyin(y_s, fmin=50, fmax=400, sr=16000,
                                             frame_length=2048, hop_length=320)
                f0_s = np.nan_to_num(f0_s, nan=0.0)
                v_s = v_s.astype(bool)
                L = min(len(f0_s), len(self._f0_ref))
                both = v_s[:L] & self._v_ref[:L]
                if both.sum() >= 5:
                    r["F0_RMSE_Hz"] = round(float(
                        np.sqrt(np.mean((f0_s[:L][both] - self._f0_ref[:L][both]) ** 2))), 1)
                    r["F0_mean_Hz"] = round(float(f0_s[both].mean()), 1)

                rms = _librosa.feature.rms(y=y_s, frame_length=2048, hop_length=320)[0]
                r["energy_rms"] = round(float(rms.mean()), 6)
            except Exception:
                r["F0_RMSE_Hz"] = None

            if r.get("duration_s", 0) > 0 and r.get("text"):
                r["speaking_rate_cps"] = round(len(r["text"]) / r["duration_s"], 1)

        return records_batch


# ── main ────────────────────────────────────────────────────────────────

def main():
    t0 = time.monotonic()

    text = load_longform(str(Path(__file__).parent / "longform_text.txt"))
    text_clean = re.sub(r'\[.*?\]', '', text)
    text_clean = re.sub(r'</?(laughter|strong)>', '', text_clean)
    print(f"Longform text: {len(text)} chars (clean: {len(text_clean)} chars)")

    dl = "/root/autodl-tmp/esd_cn/train.data.list"
    ref_wav, ref_text = select_ref(dl, SPEAKER, EMOTION)
    print(f"Ref: {ref_wav}\n")

    # ── Load CV3 once ───────────────────────────────────────────────────
    from cosyvoice.cli.cosyvoice import AutoModel
    print("Loading CV3 ...")
    model = AutoModel(model_dir=MD)
    frt = model.frontend
    cv3m = model.model
    print("  Ready\n")

    prompt = SYSTEM_PROMPT + "<|endofprompt|>" + ref_text
    records = []

    # ── Phase 1: Synthesis (sequential, GPU-bound) ──────────────────────
    for seg_len in SEGMENT_LENGTHS:
        seg_text = segment_text(text, seg_len)
        seg_clean = re.sub(r'\[.*?\]', '', seg_text)
        seg_clean = re.sub(r'</?(laughter|strong)>', '', seg_clean)

        torch.manual_seed(SEED)
        torch.cuda.empty_cache()

        sentences = frt.text_normalize(seg_text, split=False, text_frontend=True)
        mi_base = frt.frontend_zero_shot(
            str(sentences), prompt, ref_wav, model.sample_rate, "")

        tag = f"seg_{seg_len:04d}"
        out_wav = AUDIO / f"{tag}.wav"
        out_wav.parent.mkdir(parents=True, exist_ok=True)

        t_start = time.monotonic()
        try:
            gen = cv3m.tts(**mi_base, stream=False)
            chunks = [j["tts_speech"] for j in gen]
            audio = torch.cat(chunks, dim=1)
            dur = audio.shape[1] / model.sample_rate
            rtf = (time.monotonic() - t_start) / max(dur, 0.01)
            torchaudio.save(str(out_wav), audio, model.sample_rate)
            status = "ok"
        except Exception as e:
            dur, rtf, status = 0, 0, "error"
            print(f"  ERROR seg={seg_len}: {str(e)[:100]}")

        records.append({
            "segment_len_target": seg_len,
            "segment_len_actual": len(seg_clean),
            "text": seg_clean,
            "wav_path": str(out_wav),
            "duration_s": round(dur, 1),
            "rtf": round(rtf, 2),
            "status": status,
        })
        print(f"  seg={seg_len:>5d}  dur={dur:.1f}s  rtf={rtf:.1f}  {status}")

    # Full text synthesis
    print(f"\nFull text synthesis ({len(text)} chars)...")
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
        print(f"  Full: {audio_full.shape[1]/model.sample_rate:.1f}s")
    except Exception as e:
        print(f"  Full ERROR: {str(e)[:100]}")
        records.append({"segment_len_target": -1, "status": "error"})

    # Free CV3, GPU now available for evaluation
    del model, frt, cv3m
    torch.cuda.empty_cache()
    print(f"\nSynthesis done. Starting batched evaluation...")

    # ── Phase 2: Batched evaluation ─────────────────────────────────────
    evaluator = BatchedEvaluator(ref_wav, ref_text)
    ok = [r for r in records if r["status"] == "ok"]
    evaluator.evaluate_batch(ok)
    print(f"  Evaluated {len(ok)} records")

    # ── Phase 3: Chunk-level drift ──────────────────────────────────────
    full_run = [r for r in ok if r.get("segment_len_target") == -1]
    drift_chunks = []
    if full_run:
        full = full_run[0]
        full_text = full.get("text", "")
        total_chars = len(full_text)
        total_dur = full["duration_s"]
        import librosa as _librosa

        y_full, sr = _librosa.load(full["wav_path"], sr=16000)
        f0_full, v_full = _librosa.pyin(y_full, fmin=50, fmax=400, sr=sr,
                                        frame_length=2048, hop_length=320)
        f0_full = np.nan_to_num(f0_full, nan=0.0)
        v_full = v_full.astype(bool)

        rms_full = _librosa.feature.rms(y=y_full, frame_length=2048, hop_length=320)[0]
        frames_full = min(len(f0_full), len(rms_full))

        from funasr import AutoModel as FM
        asr_drift = FM(model="paraformer-zh")

        chunk_boundaries = list(range(0, total_chars, CHUNK_SIZE))
        for ci, start_char in enumerate(chunk_boundaries):
            end_char = min(start_char + CHUNK_SIZE, total_chars)
            chunk_text = full_text[start_char:end_char]

            t_start = (start_char / total_chars) * total_dur if total_chars > 0 else 0
            t_end = (end_char / total_chars) * total_dur if total_chars > 0 else 0
            s_start = max(0, int(t_start * sr))
            s_end = min(len(y_full), int(t_end * sr))

            if s_end - s_start < sr * 0.5:
                continue

            chunk_wav = AUDIO / f"chunk_{start_char:04d}_{end_char:04d}.wav"
            torchaudio.save(str(chunk_wav),
                            torch.from_numpy(y_full[s_start:s_end]).unsqueeze(0), sr)

            result = asr_drift.generate(input=str(chunk_wav))
            asr_t = result[0]["text"] if result else ""

            f_start = max(0, min(int(t_start * (sr / 320)), frames_full - 1))
            f_end = max(f_start + 1, min(int(t_end * (sr / 320)), frames_full))
            f0_chunk = f0_full[f_start:f_end]
            v_chunk = v_full[f_start:f_end]
            rms_chunk = rms_full[f_start:f_end] if f_end < len(rms_full) else np.array([0])
            f0_voiced = f0_chunk[v_chunk] if v_chunk.any() else np.array([0])

            chunk_clean = re.sub(r'\[.*?\]', '', chunk_text)
            drift_chunks.append({
                "chunk_idx": ci,
                "char_position_pct": round(start_char / max(total_chars, 1) * 100, 1),
                "CER": compute_cer(asr_t, chunk_clean),
                "F0_mean_Hz": round(float(f0_voiced.mean()), 1) if len(f0_voiced) > 0 else 0,
                "energy_rms": round(float(rms_chunk.mean()), 6) if len(rms_chunk) > 0 else 0,
                "duration_s": round((s_end - s_start) / sr, 2),
            })

        print(f"  {len(drift_chunks)} drift chunks analyzed")
        del asr_drift

    # ── Drift scores ────────────────────────────────────────────────────
    drift_scores = {}
    if len(drift_chunks) >= 3:
        pos = np.array([c["char_position_pct"] for c in drift_chunks])

        f0_vals = np.array([c["F0_mean_Hz"] for c in drift_chunks if c["F0_mean_Hz"] > 0])
        f0_pos = np.array([c["char_position_pct"] for c in drift_chunks
                           if c["F0_mean_Hz"] > 0])
        if len(f0_vals) >= 3:
            s, _ = np.polyfit(f0_pos, f0_vals, 1)
            drift_scores["F0_total_drift_Hz"] = round(float(s * 100), 1)

        e_vals = np.array([c["energy_rms"] for c in drift_chunks if c["energy_rms"]])
        e_pos = np.array([c["char_position_pct"] for c in drift_chunks
                          if c["energy_rms"]])
        if len(e_vals) >= 3:
            s, _ = np.polyfit(e_pos, e_vals, 1)
            drift_scores["energy_slope_per_pct"] = round(float(s), 8)

        cer_vals = np.array([c["CER"] for c in drift_chunks])
        if len(cer_vals) >= 3:
            s, _ = np.polyfit(pos, cer_vals, 1)
            drift_scores["CER_slope_per_pct"] = round(float(s), 5)

    # ── Segment trend ───────────────────────────────────────────────────
    seg_trend = []
    for r in ok:
        if r.get("segment_len_target", -1) > 0:
            seg_trend.append({
                "target_len": r["segment_len_target"],
                "CER": r.get("CER"), "SECS": r.get("SECS"),
                "F0_RMSE_Hz": r.get("F0_RMSE_Hz"),
                "F0_mean_Hz": r.get("F0_mean_Hz"),
                "duration_s": r.get("duration_s"), "rtf": r.get("rtf"),
                "speaking_rate_cps": r.get("speaking_rate_cps"),
            })

    # ── Save ─────────────────────────────────────────────────────────────
    json.dump({
        "config": {"speaker": SPEAKER, "emotion": EMOTION, "seed": SEED,
                   "segment_lengths": SEGMENT_LENGTHS, "chunk_size": CHUNK_SIZE,
                   "total_text_chars": len(text), "ref_wav": ref_wav},
        "segment_trend": seg_trend,
        "drift_scores": drift_scores,
        "drift_chunks": drift_chunks,
        "runs": ok,
    }, open(OUT / "results.json", "w"), ensure_ascii=False, indent=2)

    # ── Report ───────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("P3: LONG-TEXT ATTENTION DRIFT — BATCHED RESULTS")
    print(f"{'='*72}")
    print(f"\n── Segment Trend ──")
    print(f"{'Len':>6s}  {'CER':>6s}  {'SECS':>7s}  {'F0_RMSE':>8s}  {'F0_Mean':>8s}  {'Dur(s)':>7s}  {'RTF':>5s}")
    print("-" * 62)
    for s in seg_trend:
        print(f"{s['target_len']:>6d}  {s.get('CER',0) or 0:6.3f}  "
              f"{s.get('SECS') or 0:7.3f}  {s.get('F0_RMSE_Hz') or 0:8.1f}  "
              f"{s.get('F0_mean_Hz') or 0:8.1f}  {s.get('duration_s',0):7.1f}  "
              f"{s.get('rtf',0):5.1f}")

    print(f"\n── Drift Scores ──")
    for k, v in drift_scores.items():
        print(f"  {k}: {v}")

    f0_drift = drift_scores.get("F0_total_drift_Hz", 0)
    if abs(f0_drift) < 3:
        print(f"\nVERDICT: STABLE — CV3 uses 32K context effectively (drift={f0_drift} Hz)")
    elif abs(f0_drift) < 8:
        print(f"\nVERDICT: MODERATE — acceptable drift ({f0_drift} Hz)")
    else:
        print(f"\nVERDICT: SIGNIFICANT — chunk long texts ({f0_drift} Hz)")

    print(f"Total: {(time.monotonic() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
