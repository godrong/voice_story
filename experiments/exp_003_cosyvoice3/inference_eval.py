#!/usr/bin/env python3
"""CosyVoice 3 zero-shot voice cloning inference + 4-dim objective evaluation.

Evaluates CosyVoice 3's base voice cloning capability using CV3-Eval or
any (ref_wav, text) pair dataset. Produces MOS-NISQA / WER / SECS / F0 RMSE
scores, matching the eval protocol from exp_002.

Run on remote H800:
  cd /root/CosyVoice
  python /root/voice_story/experiments/exp_003_cosyvoice3/inference_eval.py \
    --eval_set /root/autodl-tmp/cv3-eval/zh/test_zh \
    --output_dir /root/autodl-tmp/exp003_outputs
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

# Add CosyVoice to path
COSYVOICE_ROOT = Path("/root/CosyVoice")
sys.path.insert(0, str(COSYVOICE_ROOT))
sys.path.insert(0, str(COSYVOICE_ROOT / "third_party" / "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import AutoModel


# ---------------------------------------------------------------------------
# 4-dim objective eval (minimal copy from core/eval_tts.py, self-contained)
# ---------------------------------------------------------------------------

def _load_wavlm_sv():
    from transformers import AutoFeatureExtractor, WavLMForXVector
    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    model = WavLMForXVector.from_pretrained(
        "microsoft/wavlm-base-plus-sv", use_safetensors=True)
    model.eval()
    return fe, model


def compute_secs(syn_wav: Path, ref_wav: Path) -> float:
    """Speaker embedding cosine similarity."""
    import librosa
    syn_audio, syn_sr = sf.read(str(syn_wav))
    ref_audio, ref_sr = sf.read(str(ref_wav))
    if syn_sr != 16000:
        syn_audio = librosa.resample(syn_audio.astype(np.float32), orig_sr=syn_sr, target_sr=16000)
    if ref_sr != 16000:
        ref_audio = librosa.resample(ref_audio.astype(np.float32), orig_sr=ref_sr, target_sr=16000)

    fe, model = _load_wavlm_sv()
    with torch.no_grad():
        syn_inp = fe([syn_audio.astype(np.float32)], sampling_rate=16000, return_tensors="pt", padding=True)
        ref_inp = fe([ref_audio.astype(np.float32)], sampling_rate=16000, return_tensors="pt", padding=True)
        syn_emb = model(**syn_inp).embeddings
        ref_emb = model(**ref_inp).embeddings
    syn_emb = torch.nn.functional.normalize(syn_emb, dim=-1)
    ref_emb = torch.nn.functional.normalize(ref_emb, dim=-1)
    return float((syn_emb @ ref_emb.T).squeeze().item())


def compute_mos_nisqa(wav_path: Path) -> float:
    """NISQA primary naturalness MOS in [1,5]."""
    import io
    from contextlib import redirect_stderr, redirect_stdout
    from nisqa.NISQA_model import nisqaModel

    args = {
        "mode": "predict_file",
        "pretrained_model": "/root/CosyVoice/pretrained_models/nisqa.tar",
        "deg": str(wav_path),
        "output_dir": None,
        "tr_bs_val": 1,
        "tr_num_workers": 0,
        "tr_parallel": False,
        "ms_channel": None,
    }
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        model = nisqaModel(args)
        df = model.predict()
    return float(df.iloc[0]["mos_pred"])


def compute_f0_rmse(syn_wav: Path, ref_wav: Path) -> float:
    """F0 RMSE over jointly voiced frames (Hz)."""
    import librosa
    syn_audio, syn_sr = sf.read(str(syn_wav))
    ref_audio, ref_sr = sf.read(str(ref_wav))
    if syn_sr != 16000:
        syn_audio = librosa.resample(syn_audio.astype(np.float32), orig_sr=syn_sr, target_sr=16000)
    if ref_sr != 16000:
        ref_audio = librosa.resample(ref_audio.astype(np.float32), orig_sr=ref_sr, target_sr=16000)

    f0_syn, vsyn, _ = librosa.pyin(syn_audio, fmin=50, fmax=400, sr=16000)
    f0_ref, vref, _ = librosa.pyin(ref_audio, fmin=50, fmax=400, sr=16000)
    n = min(len(f0_syn), len(f0_ref))
    f0_syn, f0_ref = f0_syn[:n], f0_ref[:n]
    vsyn, vref = vsyn[:n], vref[:n]
    both = vsyn & vref & ~np.isnan(f0_syn) & ~np.isnan(f0_ref)
    if int(both.sum()) < 5:
        return float("nan")
    diff = f0_syn[both] - f0_ref[both]
    return float(np.sqrt(np.mean(diff ** 2)))


def compute_wer(syn_wav: Path, target_text: str, lang: str = "zh") -> tuple[float, str]:
    """ASR-cycle WER (en) or CER (zh)."""
    from jiwer import cer as jcer
    from jiwer import wer as jwer

    if lang == "zh":
        from funasr import AutoModel as FunASRModel
        asr_model = FunASRModel(model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
        result = asr_model.generate(input=str(syn_wav))
        asr_text = result[0]["text"].strip() if result else ""
        score = float(jcer(target_text.strip().replace(" ", ""), asr_text.replace(" ", "")))
        return score, asr_text
    else:
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(str(syn_wav))
        asr_text = result["text"].strip()
        score = float(jwer(target_text.lower(), asr_text.lower()))
        return score, asr_text


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_eval_set(eval_path: Path) -> list[dict]:
    """Load evaluation pairs from JSONL manifest or CV3-Eval directory.

    Supports:
      - Direct JSONL file: each line = {id, text, ref_wav, prompt_text?, lang?}
      - Directory with eval_pairs.jsonl (MCGA prepped format)
      - CV3-Eval directory: text file + prompt_wavs/ or wavs/
    """
    items = []

    # JSONL file directly
    if eval_path.is_file() and eval_path.suffix == ".jsonl":
        with open(eval_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    # Directory: check for eval_pairs.jsonl first
    jsonl = eval_path / "eval_pairs.jsonl"
    if jsonl.exists():
        with open(jsonl) as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    # CV3-Eval format: text file + prompt_wavs/ or wavs/
    if eval_path.is_dir():
        text_file = eval_path / "text"
        if text_file.exists():
            with open(text_file) as f:
                for line in f:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        utt_id, text = parts
                        ref_wav = eval_path / "prompt_wavs" / f"{utt_id}.wav"
                        if not ref_wav.exists():
                            ref_wav = next(eval_path.glob(f"{utt_id}*.wav"), None)
                        if ref_wav:
                            items.append({"id": utt_id, "text": text, "ref_wav": str(ref_wav), "lang": "zh"})
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="pretrained_models/Fun-CosyVoice3-0.5B")
    parser.add_argument("--eval_set", default="cv3-eval/zh/test_zh")
    parser.add_argument("--output_dir", default="outputs/exp003")
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--mode", default="zero_shot", choices=["zero_shot", "instruct"])
    parser.add_argument("--skip_wer", action="store_true")
    parser.add_argument("--skip_mos", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Loading CosyVoice 3 model ...")
    cosyvoice = AutoModel(model_dir=args.model_dir)
    print(f"Sample rate: {cosyvoice.sample_rate}")
    print("=" * 60)

    # Load eval set
    eval_items = load_eval_set(Path(args.eval_set))
    if not eval_items:
        # Fallback: use CosyVoice built-in assets
        print("No eval set found, using built-in zero_shot_prompt.wav as ref")
        ref_wav = Path("/root/CosyVoice/asset/zero_shot_prompt.wav")
        default_texts = [
            ("zh_001", "八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。", "zh"),
            ("zh_002", "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐。", "zh"),
            ("zh_003", "随着人工智能技术的不断发展，语音合成已经能够以假乱真。", "zh"),
            ("en_001", "CosyVoice is undergoing a comprehensive upgrade with enhanced voice cloning capabilities.", "en"),
        ]
        for uid, text, lang in default_texts:
            eval_items.append({"id": uid, "text": text, "ref_wav": str(ref_wav),
                               "prompt_text": "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
                               "lang": lang})
    else:
        eval_items = eval_items[:args.max_samples]

    print(f"Evaluating {len(eval_items)} samples ...")
    results = []
    prompt_text_default = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"

    for idx, item in enumerate(eval_items):
        uid = item.get("id", f"sample_{idx:04d}")
        text = item["text"]
        ref_wav = item["ref_wav"]
        lang = item.get("lang", "zh")
        prompt_text = item.get("prompt_text", prompt_text_default)
        instruct = item.get("instruct", None)

        syn_wav = output_dir / f"{uid}_syn.wav"
        t0 = time.time()

        try:
            if args.mode == "instruct" and instruct:
                gen = cosyvoice.inference_instruct2(
                    text, instruct, ref_wav, stream=False)
            else:
                gen = cosyvoice.inference_zero_shot(
                    text, prompt_text, ref_wav, stream=False)

            for i, out in enumerate(gen):
                torchaudio.save(str(syn_wav), out['tts_speech'], cosyvoice.sample_rate)
        except Exception as e:
            print(f"  [{uid}] SYNTH FAILED: {e}")
            results.append({"id": uid, "error": str(e)})
            continue

        synth_time = time.time() - t0
        print(f"  [{uid}] synthesized in {synth_time:.1f}s → {syn_wav}")

        # Compute metrics
        scores = {"id": uid, "synth_time_s": round(synth_time, 1)}

        if not args.skip_mos:
            try:
                scores["mos_nisqa"] = round(compute_mos_nisqa(syn_wav), 3)
            except Exception as e:
                print(f"    MOS failed: {e}")

        try:
            scores["secs"] = round(compute_secs(syn_wav, Path(ref_wav)), 4)
        except Exception as e:
            print(f"    SECS failed: {e}")

        try:
            scores["f0_rmse_hz"] = round(compute_f0_rmse(syn_wav, Path(ref_wav)), 2)
        except Exception as e:
            print(f"    F0 RMSE failed: {e}")

        if not args.skip_wer:
            try:
                wer_val, asr_text = compute_wer(syn_wav, text, lang)
                metric_name = "cer" if lang == "zh" else "wer"
                scores[metric_name] = round(wer_val, 4)
                scores["asr_text"] = asr_text
            except Exception as e:
                print(f"    WER failed: {e}")

        results.append(scores)
        print(f"    scores: { {k: v for k, v in scores.items() if k != 'asr_text'} }")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    valid = [r for r in results if "error" not in r]
    for metric in ["mos_nisqa", "secs", "f0_rmse_hz", "cer", "wer"]:
        vals = [r[metric] for r in valid if metric in r and r[metric] is not None
                and not (isinstance(r[metric], float) and np.isnan(r[metric]))]
        if vals:
            print(f"  {metric:15s}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
                  f"min={np.min(vals):.4f}  max={np.max(vals):.4f}  n={len(vals)}")

    # Save
    report_path = output_dir / "eval_results.json"
    with open(report_path, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {report_path}")


if __name__ == "__main__":
    main()
