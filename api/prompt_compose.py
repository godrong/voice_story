"""Compose a structured VoiceConfig into a CosyVoice 2 instruct prompt.

Strict ElevenLabs template, skipping empty fields rather than echoing
``"None"``. We deliberately do **not** append the ``<|endofprompt|>``
sentinel here — ``core/tts_worker.py`` (L170-171) appends it on the
worker side, so this layer stays a pure formatter that's easy to preview
in the UI.

把结构化的 VoiceConfig 拼成 CosyVoice 2 的 instruct 自然语言指令。

严格按 ElevenLabs 模板，空字段直接跳过（不渲染成 ``"None"``）。这里
故意 **不** 追加 ``<|endofprompt|>`` 哨兵：worker 侧
（``core/tts_worker.py`` L170-171）会自动补，所以这一层保持是纯格式
化函数，方便前端实时预览。
"""

from __future__ import annotations

from .schemas import VoiceConfig


def compose_instruct(cfg: VoiceConfig) -> str:
    """Render a ``VoiceConfig`` as a single-line instruct prompt.

    把 ``VoiceConfig`` 渲染成一行 instruct prompt。

    Template (parts whose source fields are empty are omitted):

        Native <Language>. <Gender>, <Age> age range. <Quality> quality.
        Persona: <persona>. Emotion: <emotion>.
        <description>

    Args:
        cfg: Validated VoiceConfig from the frontend.
            前端传来的、已校验的 VoiceConfig。

    Returns:
        Single-line prompt suitable for ``LocalSubprocessTTS.synthesize(
        ..., instruct=...)``. Empty string if every field is blank.
        可直接传给 ``LocalSubprocessTTS.synthesize(..., instruct=...)``
        的单行文本；若所有字段都为空则返回空串。
    """
    parts: list[str] = []

    if cfg.language:
        parts.append(f"Native {cfg.language}.")

    # Gender + age render as one sentence to match ElevenLabs's "<Gender>,
    # <Age range>." convention. Either alone is also valid.
    # gender 与 age 合成一句，对齐 ElevenLabs 的 "<Gender>, <Age range>."
    # 写法；任一独立出现也是合法的。
    ga_bits: list[str] = []
    if cfg.gender:
        ga_bits.append(cfg.gender.capitalize())
    if cfg.age:
        ga_bits.append(f"{cfg.age} age range")
    if ga_bits:
        parts.append(", ".join(ga_bits) + ".")

    if cfg.quality:
        parts.append(f"{cfg.quality.capitalize()} quality.")

    if cfg.persona.strip():
        parts.append(f"Persona: {cfg.persona.strip()}.")

    if cfg.emotion.strip():
        parts.append(f"Emotion: {cfg.emotion.strip()}.")

    if cfg.description.strip():
        # Description is free-form; assume the user wrote complete
        # sentences. Don't auto-punctuate — that would be one more
        # surprising behavior to debug.
        # description 是自由文本；假定用户写了完整句子。不主动补标点，
        # 避免又多一处会让用户迷惑的隐式行为。
        parts.append(cfg.description.strip())

    return " ".join(parts)
