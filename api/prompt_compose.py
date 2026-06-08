"""Compose a structured VoiceConfig into a CosyVoice 3 instruct prompt.

CosyVoice 3's ``inference_instruct2`` expects a **natural-language
imperative** describing HOW to speak — the official examples are all short
Chinese commands like ``用四川话说这句话<|endofprompt|>`` or
``请用广东话表达。<|endofprompt|>``. Feeding it an ElevenLabs-style list of
attribute labels (``"Male, casual quality."``) makes the model fail to
recognise it as an instruction and **vocalise the labels verbatim**
(observed: output literally said "male casual").

So we render a Chinese imperative and deliberately use **delivery-only**
fields (emotion / quality / persona / description). ``gender`` and ``age``
are intentionally dropped: in instruct2 the timbre/identity comes from the
reference audio, so naming them in the instruction is useless AND a leak
source.

We do **not** append ``<|endofprompt|>`` here — the worker appends it.

把结构化 VoiceConfig 拼成 CosyVoice 3 的 instruct 指令。

CV3 的 ``inference_instruct2`` 要的是「怎么说」的**中文自然语言祈使句**
（官方示例全是 ``用四川话说这句话<|endofprompt|>`` 这种）。喂给它
ElevenLabs 风格的英文属性标签（``"Male, casual quality."``）会让模型识别
不出是指令，**把标签原文念出来**（实测输出真的念了 "male casual"）。

因此改为渲染中文祈使句，且只用「怎么说」的字段（emotion / quality /
persona / description）。**gender / age 故意丢弃**：instruct2 模式下音色与
身份由参考音频决定，写进指令既无用又会泄漏。

``<|endofprompt|>`` 由 worker 追加，这层不加。
"""

from __future__ import annotations

from .schemas import VoiceConfig

# Map the `quality` enum to a natural Chinese delivery clause.
# 把 quality 枚举映射成中文「音质 / 风格」短语。
_QUALITY_ZH = {
    "studio": "音质干净清晰",
    "broadcast": "带专业的播音腔",
    "casual": "语气轻松随意",
}


def compose_instruct(cfg: VoiceConfig) -> str:
    """Render a ``VoiceConfig`` as a CV3 Chinese imperative instruct prompt.

    把 ``VoiceConfig`` 渲染成 CV3 中文祈使句指令。

    Template (empty fields skipped)::

        请[以<persona>的口吻][，用<emotion>的情绪][，<quality 短语>]，说这句话。<description>

    Identity fields (``gender`` / ``age``) are NOT rendered — they come from
    the reference audio in instruct2 mode.

    Args:
        cfg: Validated VoiceConfig from the frontend.
            前端传来的、已校验的 VoiceConfig。

    Returns:
        Single-line Chinese imperative for ``LocalSubprocessTTS.synthesize(
        ..., instruct=...)``. Empty string if no delivery field is set.
        可直接传给 instruct 的中文单行祈使句；无任何「怎么说」字段时返回空串。
    """
    # Delivery clauses, ordered: persona → emotion → quality.
    # 「怎么说」从句，顺序：口吻 → 情绪 → 音质。
    clauses: list[str] = []

    if cfg.persona.strip():
        clauses.append(f"以{cfg.persona.strip()}的口吻")
    if cfg.emotion.strip():
        clauses.append(f"用{cfg.emotion.strip()}的情绪")
    if cfg.quality and cfg.quality in _QUALITY_ZH:
        clauses.append(_QUALITY_ZH[cfg.quality])

    description = cfg.description.strip()

    # Nothing to instruct — return empty so the server's 422 guard fires
    # rather than emitting a content-free "请说这句话。".
    # 没有任何可控字段就返回空串，交给 server 的 422 拦截，
    # 不要生成空洞的「请说这句话。」。
    if not clauses and not description:
        return ""

    sentence = ""
    if clauses:
        sentence = "请" + "，".join(clauses) + "，说这句话。"

    # Description is free-form Chinese; append after the imperative.
    # description 是自由中文，拼在祈使句之后；假定用户写了完整句子。
    if description:
        sentence = (sentence + description) if sentence else description

    return sentence
