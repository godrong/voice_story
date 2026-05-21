#!/usr/bin/env python3
"""Prepare MCGA dataset for CosyVoice 3 zero-shot voice cloning evaluation.

Downloads yxdu/MCGA from HuggingFace, extracts audio to wav files,
groups by speaker description, and builds eval pairs in the format
expected by inference_eval.py.

Output structure:
  /root/autodl-tmp/datasets/mcga/
    wavs/          — individual wav files ({id}.wav)
    eval_pairs.jsonl  — evaluation pairs for inference_eval.py
    speakers.json  — speaker metadata grouped by description

Usage:
  python prep_mcga.py --output_dir /root/autodl-tmp/datasets/mcga
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="/root/autodl-tmp/datasets/mcga")
    parser.add_argument("--min_duration", type=float, default=3.0,
                       help="Minimum audio duration in seconds")
    parser.add_argument("--max_duration", type=float, default=15.0,
                       help="Maximum audio duration in seconds")
    parser.add_argument("--pairs_per_speaker", type=int, default=10,
                       help="Max eval pairs per speaker group")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    wav_dir = output_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    print("Loading MCGA from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset("yxdu/MCGA", split="test")
    print(f"Loaded {len(ds)} samples")

    # Filter by duration
    ds = ds.filter(lambda x: args.min_duration <= x["time"] <= args.max_duration)
    print(f"After duration filter [{args.min_duration}s, {args.max_duration}s]: {len(ds)} samples")

    # Group by speaker description (sec_2) + gender
    speaker_groups = defaultdict(list)
    for i, sample in enumerate(ds):
        spk_key = f"{sample['gender']}_{sample.get('sec_1', 'unknown')}"
        # Use sec_2 for finer granularity if available
        if sample.get("sec_2") and len(sample["sec_2"]) > 0:
            spk_key = sample["sec_2"].strip()
        speaker_groups[spk_key].append(i)

    print(f"Speaker groups: {len(speaker_groups)}")
    for spk, indices in sorted(speaker_groups.items(), key=lambda x: -len(x[1])):
        print(f"  {spk}: {len(indices)} clips")

    # Write wav files and build pairs
    pairs = []
    speakers_meta = {}
    total_wavs = 0

    for spk_idx, (spk_key, indices) in enumerate(sorted(speaker_groups.items())):
        if len(indices) < 2:
            continue  # Need at least 2 clips for (ref, target) pair

        spk_clips = []
        for idx in indices:
            sample = ds[int(idx)]
            audio = sample["audio"]["array"]
            sr = sample["audio"]["sampling_rate"]

            # Resample to 24kHz (CosyVoice 3 native rate) if needed
            if sr != 24000:
                import librosa
                audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=24000)
                sr = 24000

            clip_id = sample["id"].replace("/", "_").replace(" ", "_")
            wav_path = wav_dir / f"{clip_id}.wav"
            sf.write(str(wav_path), audio, sr)
            total_wavs += 1

            spk_clips.append({
                "id": clip_id,
                "wav_path": str(wav_path),
                "text": sample["asr"],
                "gender": sample["gender"],
                "genre": sample["genre"],
                "author": sample["author"],
                "title": sample["title"],
                "duration": sample["time"],
            })

        # Build eval pairs: first clip as ref, others as targets
        max_pairs = min(args.pairs_per_speaker, len(spk_clips) - 1)
        spk_clips_sorted = sorted(spk_clips, key=lambda x: x["duration"], reverse=True)

        # Use longest clip as reference
        ref_clip = spk_clips_sorted[0]
        target_clips = spk_clips_sorted[1:1 + max_pairs]

        for target in target_clips:
            pairs.append({
                "id": f"{spk_idx:03d}_{target['id']}",
                "text": target["text"],
                "ref_wav": ref_clip["wav_path"],
                "prompt_text": "You are a helpful assistant.<|endofprompt|>"
                               f"请用中文朗读以下文本。",
                "lang": "zh",
                "speaker_desc": spk_key,
                "genre": target["genre"],
                "author": target["author"],
                "ref_duration": ref_clip["duration"],
            })

        speakers_meta[spk_key] = {
            "n_clips": len(spk_clips),
            "ref_clip_id": ref_clip["id"],
            "n_pairs": len(target_clips),
            "gender": spk_clips[0]["gender"],
        }

    # Save
    pairs_path = output_dir / "eval_pairs.jsonl"
    with open(pairs_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    speakers_path = output_dir / "speakers.json"
    with open(speakers_path, "w", encoding="utf-8") as f:
        json.dump(speakers_meta, f, ensure_ascii=False, indent=2)

    print(f"\nDone:")
    print(f"  Wavs:    {total_wavs} files in {wav_dir}")
    print(f"  Pairs:   {len(pairs)} in {pairs_path}")
    print(f"  Speaker groups: {len(speakers_meta)} in {speakers_path}")


if __name__ == "__main__":
    main()
