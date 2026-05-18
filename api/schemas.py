"""Pydantic request/response models for the WebUI API.

Keep these models thin and stable — the Vue frontend serialises directly
to JSON over fetch, so any change here is a contract change. Validation
that's specific to CosyVoice (e.g. instruct text must end with the
``<|endofprompt|>`` token) belongs in ``core/tts_worker.py``, not here.

WebUI API 的 Pydantic 请求/响应模型。

模型要薄、要稳——前端 Vue 直接 fetch 反序列化这些字段，改一处就是
契约变更。CosyVoice 特有的校验（例如 instruct 必须以 ``<|endofprompt|>``
结尾）放在 ``core/tts_worker.py``，不要写到这一层。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Voice config — modelled after ElevenLabs' voice design prompting guide.
# 声音配置——参照 ElevenLabs voice design 的 prompting guide。
# https://elevenlabs.io/docs/eleven-creative/voices/voice-design#prompting-guide
# ---------------------------------------------------------------------------


class VoiceConfig(BaseModel):
    """ElevenLabs-style structured voice description.

    ElevenLabs 风格的结构化声音描述。

    Composes into a natural-language instruct prompt for CosyVoice 2 via
    :func:`api.prompt_compose.compose_instruct`. None / empty fields are
    skipped during composition rather than rendered as ``"None"``.

    通过 :func:`api.prompt_compose.compose_instruct` 拼成 CosyVoice 2
    可用的自然语言指令；None / 空串字段会被跳过而不是渲染成 "None"。
    """

    language: Literal["English", "Chinese"] = "English"
    gender: Literal["male", "female"] | None = None
    age: Literal["young", "middle", "old"] | None = None
    quality: Literal["studio", "broadcast", "casual"] | None = None
    persona: str = Field(default="", description="2-5 words, e.g. 'confident teacher'")
    emotion: str = Field(default="", description="2-3 adjectives, e.g. 'calm, warm'")
    description: str = Field(default="", description="1-2 sentences on timbre / pacing / delivery")


# ---------------------------------------------------------------------------
# Reference audio — built-in (from datasets/ manifests) or user-uploaded.
# 参考音频——内置（来自 datasets/ manifest）或用户上传。
# ---------------------------------------------------------------------------


class RefAudio(BaseModel):
    """Metadata for a reference audio entry shown in the UI dropdown.

    UI 下拉里展示的参考音频条目元信息。
    """

    ref_id: str
    source: Literal["builtin", "upload"]
    dataset: str | None = None
    audio_path: str
    prompt_text: str
    duration: float | None = None
    mos_ovr: float | None = None


class UploadRefResponse(BaseModel):
    """Response after a user uploads a reference audio file.

    用户上传参考音频后的响应。
    """

    ref_id: str
    prompt_text: str
    duration: float | None
    asr_lang: str | None
    asr_confidence: float | None


# ---------------------------------------------------------------------------
# Bilibili URL import — probe a URL, then import as an "upload"-equivalent ref.
# B 站 URL 导入——先 probe 拿元数据，再把片段下载落到 uploads/ 当作"上传"。
# ---------------------------------------------------------------------------


class BilibiliPartInfo(BaseModel):
    """One entry of a multi-P video / collection, surfaced to the UI for choice.

    多 P 视频 / 合集里的一条子项，给前端做选择用。
    """

    index: int
    title: str
    duration: float


class BilibiliProbeRequest(BaseModel):
    """Body of POST /api/bilibili/probe — just the URL.

    POST /api/bilibili/probe 的请求体——就一个 URL。
    """

    url: str = Field(min_length=1)


class BilibiliProbeResponse(BaseModel):
    """Response from POST /api/bilibili/probe — metadata, no download yet.

    POST /api/bilibili/probe 的响应——元数据，未下载。
    """

    bvid: str
    title: str
    uploader: str
    duration: float
    parts: list[BilibiliPartInfo]
    available_subtitles: list[str]  # lang codes, e.g. ["zh-CN", "en"]


class BilibiliImportRequest(BaseModel):
    """Body of POST /api/bilibili/import — kick off async download → ASR → ref.

    POST /api/bilibili/import 的请求体——异步下载 → ASR → 落成 ref。

    Time range, if provided, is applied as a clip on the chosen part.
    时间区间（如果给）应用到选中的那一 P。
    """

    url: str = Field(min_length=1)
    part_index: int | None = Field(default=None, ge=1)
    start_sec: float | None = Field(default=None, ge=0)
    end_sec: float | None = Field(default=None, gt=0)
    use_subtitle_as_prompt: bool = True


class StageState(BaseModel):
    """One stage in a pipeline visualisation card.

    流水线可视化卡片里的一个 stage。

    The frontend renders a vertical list of these — each shows a status
    icon, elapsed time, and a free-form `detail` line that the backend
    updates with the latest progress signal (yt-dlp's download line,
    "FunASR transcribing...", etc).
    前端渲染成竖排卡片：状态图标 + 名字 + 耗时 + detail（后端写进来的
    最新进度行，如 yt-dlp 下载行、"FunASR transcribing..." 等）。
    """

    name: str
    status: Literal["pending", "running", "done", "error"] = "pending"
    started_at: str | None = None
    finished_at: str | None = None
    elapsed_s: float | None = None
    detail: str = ""


class BilibiliImportJob(BaseModel):
    """Status of one async Bilibili import job — polled by the UI.

    一次 B 站异步导入任务的状态——前端轮询读取。

    Status flow:
        queued → downloading → transcribing → ready
                                            → error

    The legacy ``status`` + ``progress_hint`` fields remain for backward
    compatibility; the structured ``stages`` array is the source of truth
    that the new PipelineCard renders.
    legacy 的 ``status`` 与 ``progress_hint`` 保留兼容；前端的 PipelineCard
    用结构化的 ``stages`` 数组渲染。
    """

    job_id: str
    status: Literal["queued", "downloading", "transcribing", "ready", "error"]
    progress_hint: str = ""
    stages: list[StageState] = Field(default_factory=list)
    # Populated once status == "ready":
    ref_id: str | None = None
    prompt_text: str | None = None
    duration: float | None = None
    asr_lang: str | None = None
    asr_confidence: float | None = None
    prompt_source: Literal["subtitle", "asr"] | None = None
    # Populated once status == "error":
    error: str | None = None


# ---------------------------------------------------------------------------
# Synthesis — request body and response.
# 合成——请求体与响应。
# ---------------------------------------------------------------------------


class SynthesizeRequest(BaseModel):
    """Body of POST /api/synthesize.

    POST /api/synthesize 的请求体。

    ``voice_config`` is required when ``mode == "instruct"`` and ignored
    in ``zero_shot`` mode (where the reference audio + prompt_text carry
    all the style information).

    ``mode == "instruct"`` 时必填 ``voice_config``；``zero_shot`` 模式下
    被忽略（这种模式靠参考音频 + prompt_text 携带风格信息）。
    """

    text: str = Field(min_length=1)
    ref_id: str
    mode: Literal["zero_shot", "instruct"] = "zero_shot"
    voice_config: VoiceConfig | None = None


class SynthesizeResponse(BaseModel):
    """Response from POST /api/synthesize.

    POST /api/synthesize 的响应。
    """

    syn_id: str
    audio_url: str
    wall_time_s: float
    mode: str
    composed_instruct: str | None = None


# ---------------------------------------------------------------------------
# Feedback — the "reward signal" of the listening-policy loop.
# 反馈——听感策略闭环里的“奖励信号”。
# ---------------------------------------------------------------------------


class EvalScores(BaseModel):
    """Objective eval scores attached to a synthesis (subset shown to UI).

    一次合成的客观评测结果（UI 展示用的子集）。
    """

    mos_nisqa: float | None = None
    mos_p808: float | None = None
    wer: float | None = None
    cer: float | None = None
    secs: float | None = None
    f0_rmse_hz: float | None = None
    eval_time_s: float | None = None


class EvalResponse(BaseModel):
    """Response from GET /api/eval/{syn_id}.

    GET /api/eval/{syn_id} 的响应。

    ``status``: "running" (eval still in progress) / "done" (scores ready)
    / "error" (eval failed; error field populated) / "unknown" (no such
    syn_id or eval never started).
    """

    syn_id: str
    status: Literal["running", "done", "error", "unknown"]
    scores: EvalScores | None = None
    error: str | None = None


class FeedbackEntry(BaseModel):
    """One user-provided listening rating, persisted to feedback.jsonl.

    一条用户给出的听感评分，持久化到 feedback.jsonl。
    """

    syn_id: str
    rating: int = Field(ge=1, le=5)
    tags: list[str] = Field(default_factory=list)
    note: str = ""


class HistoryEntry(BaseModel):
    """One row in the history panel: a past synthesis + its feedback (if any).

    History 面板里的一行：一次过去的合成 + 它的反馈（如果有）。
    """

    syn_id: str
    timestamp: str
    text: str
    ref_id: str
    mode: str
    composed_instruct: str | None
    audio_url: str
    eval: EvalScores | None = None
    wall_time_s: float
    feedback: FeedbackEntry | None = None
