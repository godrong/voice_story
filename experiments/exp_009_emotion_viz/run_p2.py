#!/usr/bin/env python3
"""P2: Emotion Representation Visualization.

Analyze if different emotions have distinct acoustic signatures in CV3 outputs.
Uses WavLM embeddings from existing exp_005 WAVs.
PCA/t-SNE visualization + linear probe for emotion separability.
"""

import sys, os, json, numpy as np, torch
from pathlib import Path
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"

AUDIO = Path("/root/autodl-tmp/clean_multiemotion/audio")
OUT = Path("/root/autodl-tmp/emotion_viz")
OUT.mkdir(parents=True, exist_ok=True)
EMOTIONS = ["Neutral","Happy","Angry","Surprise","Sad"]
N_SAMPLES = 360  # balanced: 72 per emotion

# Load WavLM
from transformers import WavLMConfig, WavLMModel
from safetensors.torch import load_file
import librosa

SNAP = "/root/.cache/huggingface/hub/models--microsoft--wavlm-base-plus-sv/snapshots/1bfd64eca136543feb28c5ffaf05381c6af33121"
cfg = WavLMConfig.from_pretrained(SNAP, local_files_only=True)
wm = WavLMModel(cfg)
wm.load_state_dict(load_file(f"{SNAP}/model.safetensors"), strict=False)
wm = wm.cuda().eval()

# Extract embeddings
print("Extracting WavLM embeddings...")
embeddings = []
labels = []
count = 0
for emo in EMOTIONS:
    emo_wavs = list(AUDIO.glob(f"*/{emo}/*/*.wav"))
    np.random.seed(42)
    selected = np.random.choice(emo_wavs, min(72, len(emo_wavs)), replace=False)
    for wav_path in selected:
        try:
            y,_ = librosa.load(str(wav_path), sr=16000)
            inp = torch.from_numpy(y.astype(np.float32)).unsqueeze(0).cuda()
            out = wm(inp, output_hidden_states=True)
            emb = out.last_hidden_state.mean(dim=1).detach().cpu().numpy().flatten()
            embeddings.append(emb)
            labels.append(emo)
            count += 1
            if count % 50 == 0: print(f"  {count}/{N_SAMPLES}")
        except Exception as e:
            print(f"  SKIP {wav_path.name}: {e}")
del wm; torch.cuda.empty_cache()

X = np.array(embeddings)
y = np.array(labels)
print(f"Extracted {len(X)} embeddings, shape={X.shape}")

# PCA
print("\nPCA...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
print(f"  Explained variance: {pca.explained_variance_ratio_}")

# t-SNE
print("t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(StandardScaler().fit_transform(X[:200]))  # subset for speed
y_tsne = y[:200]

# Linear probe
print("Linear probe...")
clf = LogisticRegression(max_iter=1000)
scores = cross_val_score(clf, StandardScaler().fit_transform(X), y, cv=5)
print(f"  5-fold CV accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

# Per-emotion separability
print("\nPer-emotion probe accuracy:")
clf.fit(StandardScaler().fit_transform(X), y)
for emo in EMOTIONS:
    mask = y == emo
    acc = (clf.predict(StandardScaler().fit_transform(X[mask])) == y[mask]).mean()
    print(f"  {emo}: {acc:.3f}")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
colors = {'Neutral':'gray','Happy':'orange','Angry':'red','Surprise':'green','Sad':'blue'}
for emo in EMOTIONS:
    mask_pca = y == emo
    ax1.scatter(X_pca[mask_pca,0], X_pca[mask_pca,1], c=colors[emo], label=emo, alpha=0.5, s=10)
    mask_tsne = y_tsne == emo
    ax2.scatter(X_tsne[mask_tsne,0], X_tsne[mask_tsne,1], c=colors[emo], label=emo, alpha=0.5, s=10)
ax1.set_title(f'PCA (explained var: {pca.explained_variance_ratio_.sum():.2f})')
ax1.legend()
ax2.set_title('t-SNE')
ax2.legend()
plt.tight_layout()
plt.savefig(OUT / "emotion_viz.png", dpi=150)
print(f"\nSaved: {OUT / 'emotion_viz.png'}")

# Save data
results = {
    "pca_variance": pca.explained_variance_ratio_.tolist(),
    "probe_accuracy": float(np.mean(scores)),
    "probe_std": float(np.std(scores)),
    "per_emotion_accuracy": {},
    "n_samples": len(X),
}
clf.fit(StandardScaler().fit_transform(X), y)
for emo in EMOTIONS:
    mask = y == emo
    results["per_emotion_accuracy"][emo] = float((clf.predict(StandardScaler().fit_transform(X[mask])) == y[mask]).mean())

json.dump(results, open(OUT / "p2_results.json", "w"), indent=2)
print(f"Saved: {OUT / 'p2_results.json'}")
print("Done.")
