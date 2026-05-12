"""Speaker embedding + similarity-based filter for multi-speaker sources.

Given a short reference clip of the target speaker, this module computes
embeddings for every chunk and keeps those whose cosine similarity to the
reference is above a threshold. Lets us pull a single voice out of
panels / interviews / podcasts.

Default model: WeSpeaker `voxceleb_resnet34` (256-dim embedding, ~25MB).
Skipped entirely when SourceMeta.is_single_speaker is True (e.g. a
single-speaker speech corpus like Trump speeches), so we never pay the
cost when it can't possibly help.

说话人 embedding + 相似度过滤模块。

给定一段目标说话人的参考片段（5~10s），本模块对每个 chunk 提取 embedding，
保留与参考的余弦相似度高于阈值的部分。用途：从访谈 / 播客 / 圆桌等
多说话人源中只保留目标人物的片段。

默认模型：WeSpeaker voxceleb_resnet34（256 维 embedding，~25MB）。
SourceMeta.is_single_speaker=True（如演讲数据集）时整个步骤跳过，
避免在不可能有帮助的场景下白白消耗算力。
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from . import audio_io

logger = logging.getLogger(__name__)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors.

    两个 1-D 向量的余弦相似度（已处理零向量保护）。
    """
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class SpeakerFilter:
    """Filter chunks by speaker similarity to a reference clip.

    按与参考片段的说话人相似度过滤 chunks。

    Usage:
        sf = SpeakerFilter()
        sf.set_reference("path/to/target_5sec.wav")
        kept = sf.keep(chunk_paths, threshold=0.65)

    Args:
        model_name: WeSpeaker pretrained name.
                    WeSpeaker 预训练模型名。
        device: torch device. Auto-pick if None.
                设备名，None 则自动挑最快的。
    """

    def __init__(
        self,
        model_name: str = "voxceleb_resnet34",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self._model = None
        self._reference_emb: np.ndarray | None = None

    def _ensure_model(self):
        """Lazily load the WeSpeaker model.

        惰性加载 WeSpeaker；首次调用会从 modelscope 下载权重。
        """
        if self._model is not None:
            return self._model
        # Lazy import: wespeakerruntime triggers a torch model load.
        # 延迟导入：避免 import 时连带加载 wespeaker。
        try:
            import wespeaker
        except ImportError as e:
            raise ImportError(
                "wespeaker not installed. Add to extras: "
                "`pip install wespeaker` (uses ONNX runtime by default)."
            ) from e
        logger.info("SpeakerFilter: loading WeSpeaker %s on %s", self.model_name, self.device)
        # WeSpeaker loads its English voxceleb model by default; multilingual
        # users can swap to a CN-CommonVoice-trained model later via ADR.
        # WeSpeaker 默认加载 voxceleb 英文模型；后续如需中文可换 CN 模型。
        self._model = wespeaker.load_model(self.model_name)
        if hasattr(self._model, "set_device"):
            self._model.set_device(self.device)
        return self._model

    def embed(self, wav_path: Path | str) -> np.ndarray:
        """Compute a speaker embedding for one WAV file.

        计算一个 WAV 文件的说话人 embedding。

        Args:
            wav_path: Audio file path (any sample rate; WeSpeaker resamples).
                      音频路径（任意采样率，WeSpeaker 内部会重采样）。

        Returns:
            1-D numpy array (typically 256-dim) of the speaker embedding.
            1-D numpy 数组（一般 256 维）。
        """
        model = self._ensure_model()
        emb = model.extract_embedding(str(wav_path))
        if isinstance(emb, torch.Tensor):
            emb = emb.detach().cpu().numpy()
        return np.asarray(emb).reshape(-1)

    def set_reference(self, ref_wav: Path | str) -> None:
        """Set the target-speaker reference embedding.

        设置目标说话人的参考 embedding（后续 keep() 用它做余弦比较）。

        Args:
            ref_wav: Path to a clean 5~10s WAV of the target speaker.
                     一段干净的目标说话人 WAV（建议 5~10 秒）。
        """
        self._reference_emb = self.embed(ref_wav)
        logger.info("SpeakerFilter: reference set from %s", ref_wav)

    def keep(
        self, wav_paths: list[Path], *, threshold: float = 0.65,
    ) -> tuple[list[Path], dict[Path, float]]:
        """Return chunks whose similarity to the reference >= threshold.

        过滤 chunks，保留与参考相似度 >= threshold 的那些。

        Args:
            wav_paths: Candidate chunk WAV paths.
                       候选切片路径列表。
            threshold: Cosine similarity cutoff in [-1, 1].
                       余弦相似度阈值，建议 0.6~0.7。

        Returns:
            Tuple of (kept_paths, similarity_map). The map covers ALL
            input paths (not just kept) for diagnostic purposes.
            (保留下来的路径列表, 所有输入路径的相似度字典)。
            相似度字典覆盖全部输入，便于调试与排查阈值。

        Raises:
            ValueError: If `set_reference()` has not been called.
                        未先调用 set_reference() 时抛出。
        """
        if self._reference_emb is None:
            raise ValueError("SpeakerFilter: call set_reference() before keep()")
        scores: dict[Path, float] = {}
        kept: list[Path] = []
        for p in wav_paths:
            emb = self.embed(p)
            sim = _cosine(self._reference_emb, emb)
            scores[p] = sim
            if sim >= threshold:
                kept.append(p)
        logger.info(
            "SpeakerFilter: kept %d/%d (threshold=%.2f)",
            len(kept), len(wav_paths), threshold,
        )
        return kept, scores
