"""FastAPI server for the TTS playground WebUI.

Endpoints
---------
- ``GET  /``                          serves ``webui/index.html``
- ``GET  /static/*``                  serves ``webui/`` (CSS, JS, etc.)
- ``GET  /api/refs``                  list built-in reference audios
- ``POST /api/upload-ref``            upload + auto-ASR a new reference
- ``POST /api/synthesize``            synthesise + persist meta; spawns
                                      a background eval task
- ``GET  /api/audio/{syn_id}``        stream synthesised wav
- ``GET  /api/eval/{syn_id}``         objective eval scores (poll until ready)
- ``POST /api/feedback``              append one rating to feedback.jsonl
- ``GET  /api/history``               list past synthesises + feedback

The CosyVoice ``LocalSubprocessTTS`` worker is constructed once in the
lifespan handler and shared across requests — re-spawning per request
would mean a ~19 s cold-start every time.

WebUI 的 FastAPI 后端服务。

CosyVoice ``LocalSubprocessTTS`` worker 在 lifespan 里只构造一次并跨
请求复用——每次请求都重启 worker 等于每次都吃 ~19 秒冷启动。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from agents.dataset_agent import load_manifest
from core.tts import TTSError, get_tts_backend

from .prompt_compose import compose_instruct
from .schemas import (
    BilibiliImportJob,
    BilibiliImportRequest,
    BilibiliPartInfo,
    BilibiliProbeRequest,
    BilibiliProbeResponse,
    EvalResponse,
    EvalScores,
    FeedbackEntry,
    HistoryEntry,
    RefAudio,
    StageState,
    SynthesizeRequest,
    SynthesizeResponse,
    UploadRefResponse,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
WEBUI_DIR = REPO_ROOT / "webui"
OUTPUTS_DIR = REPO_ROOT / "outputs" / "webui"
SYN_DIR = OUTPUTS_DIR / "syn"
META_DIR = OUTPUTS_DIR / "meta"
UPLOADS_DIR = OUTPUTS_DIR / "uploads"
FEEDBACK_PATH = OUTPUTS_DIR / "feedback.jsonl"


# ---------------------------------------------------------------------------
# Lifespan: warm TTS worker once at startup, close cleanly at shutdown.
# 生命周期：启动时一次性热身 TTS worker，停服时干净关闭。
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Spin up the TTS worker once and tear it down at shutdown.

    启动时拉起 TTS worker，停服时干净关闭，避免僵尸进程。
    """
    for d in (SYN_DIR, META_DIR, UPLOADS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    logger.info("WebUI server starting; warming TTS worker...")
    app.state.tts = get_tts_backend("cosyvoice3", startup_timeout=300)
    logger.info("TTS worker ready (sample_rate=%s)", app.state.tts.sample_rate)
    app.state.transcriber = None  # lazy — first upload pays the cost
    # eval_jobs: syn_id -> "running" | "done" | "error"; "done" / "error"
    # are also persisted in the meta JSON, so the in-memory dict is just
    # a fast-path for in-flight evals.
    # eval_jobs：syn_id -> 状态字符串；"done"/"error" 也会写进 meta JSON，
    # 这个内存 dict 只是 in-flight eval 的快速通道。
    app.state.eval_jobs = {}
    # bilibili_jobs: job_id -> BilibiliImportJob model. In-memory only;
    # the resulting ref is persisted via the same upload_<id>.json sidecar
    # used by /api/upload-ref, so server restarts don't lose the ref.
    # bilibili_jobs：内存里的导入任务进度表；最终落地的 ref 走与上传一致的
    # uploads/upload_<id>.json，重启 server 不会丢已完成的 ref。
    app.state.bilibili_jobs = {}
    try:
        yield
    finally:
        logger.info("Shutting down TTS worker...")
        try:
            app.state.tts.close()
        except Exception as e:  # noqa: BLE001 — best effort during shutdown
            logger.warning("error closing TTS worker: %s", e)


app = FastAPI(title="voice_story TTS playground", lifespan=lifespan)

# Serve everything inside webui/ — CSS, JS, images. The root path ``/``
# is overridden below to return index.html explicitly so we don't get
# the StaticFiles directory listing.
# 把 webui/ 目录整个挂载为静态资源：CSS、JS、图片。根路径 / 下面单独
# 处理，返回 index.html 而不是 StaticFiles 的目录索引。
if WEBUI_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEBUI_DIR), name="static")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_utc_iso() -> str:
    """ISO-8601 timestamp with tz, e.g. ``2026-05-15T14:23:01+00:00``.

    带时区的 ISO-8601 时间戳。
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_syn_id() -> str:
    """Generate a synthesis id like ``syn_20260515T142301_a3f2``.

    生成形如 ``syn_20260515T142301_a3f2`` 的合成 id。
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"syn_{ts}_{uuid.uuid4().hex[:4]}"


def _new_upload_ref_id() -> str:
    """Generate a reference id for uploaded audio.

    为上传音频生成参考 id。
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"upload_{ts}_{uuid.uuid4().hex[:4]}"


def _resolve_ref(ref_id: str) -> tuple[Path, str]:
    """Look up ``(audio_path, prompt_text)`` for a built-in or uploaded ref.

    解析 ref_id 到 (音频路径, prompt_text)，覆盖内置与上传两类。

    Raises:
        HTTPException 404 if the ref_id is unknown.
        若 ref_id 不存在则抛 HTTPException 404。
    """
    if ref_id.startswith("upload_"):
        meta_path = UPLOADS_DIR / f"{ref_id}.json"
        if not meta_path.exists():
            raise HTTPException(status_code=404, detail=f"unknown upload ref {ref_id!r}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return Path(meta["audio_path"]), meta["prompt_text"]

    # Built-in: scan manifests under datasets/<name>/manifest.jsonl.
    # 内置：扫描 datasets/<name>/manifest.jsonl 找 chunk_id。
    for entry in _iter_builtin_refs():
        if entry.ref_id == ref_id:
            return REPO_ROOT / entry.audio_path, entry.prompt_text
    raise HTTPException(status_code=404, detail=f"unknown ref_id {ref_id!r}")


def _iter_builtin_refs():
    """Yield ``RefAudio`` entries for every chunk in datasets/.

    枚举 datasets/ 下所有 chunk，产出 ``RefAudio``。
    """
    datasets_root = REPO_ROOT / "datasets"
    if not datasets_root.exists():
        return
    for manifest_path in datasets_root.glob("*/manifest.jsonl"):
        dataset = manifest_path.parent.name
        for row in load_manifest(manifest_path):
            yield RefAudio(
                ref_id=row["chunk_id"],
                source="builtin",
                dataset=dataset,
                audio_path=row["audio_path"],
                prompt_text=row.get("text", ""),
                duration=row.get("duration"),
                mos_ovr=row.get("mos_ovr"),
            )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
async def root():
    """Serve the SPA entry point.

    单页应用入口。
    """
    index = WEBUI_DIR / "index.html"
    if not index.exists():
        return JSONResponse(
            {"error": f"webui/index.html missing at {index}"}, status_code=500
        )
    return FileResponse(index)


@app.get("/api/refs", response_model=list[RefAudio])
async def list_refs():
    """List every built-in reference audio plus any prior uploads.

    列出所有内置参考音频，以及历史上传过的 ref。
    """
    refs = list(_iter_builtin_refs())
    # Also include uploaded refs so a page refresh doesn't lose them.
    # 也包含历史上传，避免刷新页面就丢。
    if UPLOADS_DIR.exists():
        for meta_path in sorted(UPLOADS_DIR.glob("upload_*.json")):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            refs.append(RefAudio(
                ref_id=meta["ref_id"],
                source="upload",
                dataset=None,
                audio_path=meta["audio_path"],
                prompt_text=meta["prompt_text"],
                duration=meta.get("duration"),
                mos_ovr=None,
            ))
    return refs


@app.post("/api/upload-ref", response_model=UploadRefResponse)
async def upload_ref(
    file: UploadFile = File(...),
    prompt_text: str = Form(""),
):
    """Accept an uploaded audio file; auto-ASR if no ``prompt_text`` given.

    接收上传的音频；用户没填 ``prompt_text`` 时自动 ASR 转写。

    The audio is saved to ``outputs/webui/uploads/<ref_id>.<ext>`` and a
    sidecar JSON records the resolved prompt_text + duration so future
    requests can refer to it by ``ref_id`` alone.

    音频落到 ``outputs/webui/uploads/<ref_id>.<ext>``，旁路 JSON 记录
    解析出来的 prompt_text 与时长，后续请求只凭 ``ref_id`` 即可定位。
    """
    ref_id = _new_upload_ref_id()
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    audio_path = UPLOADS_DIR / f"{ref_id}{suffix}"
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    audio_bytes = await file.read()
    audio_path.write_bytes(audio_bytes)

    asr_lang: str | None = None
    asr_conf: float | None = None
    resolved_prompt = prompt_text.strip()
    if not resolved_prompt:
        transcriber = _get_or_init_transcriber(app)
        result = transcriber.transcribe(audio_path)
        resolved_prompt = result.text
        asr_lang = result.lang
        asr_conf = result.confidence

    duration: float | None = None
    try:
        import soundfile as sf  # already in core deps
        info = sf.info(str(audio_path))
        duration = float(info.frames) / float(info.samplerate)
    except Exception as e:  # noqa: BLE001 — duration is best-effort metadata
        logger.warning("failed to read duration of %s: %s", audio_path, e)

    meta_path = UPLOADS_DIR / f"{ref_id}.json"
    meta = {
        "ref_id": ref_id,
        "audio_path": str(audio_path),
        "prompt_text": resolved_prompt,
        "duration": duration,
        "asr_lang": asr_lang,
        "asr_confidence": asr_conf,
        "uploaded_at": _now_utc_iso(),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return UploadRefResponse(
        ref_id=ref_id,
        prompt_text=resolved_prompt,
        duration=duration,
        asr_lang=asr_lang,
        asr_confidence=asr_conf,
    )


# ---------------------------------------------------------------------------
# Bilibili URL import: probe → background download + ASR → ref in uploads/.
# B 站 URL 导入：probe → 后台下载 + ASR → 落到 uploads/ 当成普通 ref。
# ---------------------------------------------------------------------------


def _new_bilibili_job_id() -> str:
    """Generate a bilibili import job id like ``bili_20260518T142301_a3f2``.

    生成形如 ``bili_20260518T142301_a3f2`` 的 B 站导入任务 id。
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"bili_{ts}_{uuid.uuid4().hex[:4]}"


def _parse_srt_text(srt_path: Path, time_range: tuple[float, float] | None) -> str:
    """Crude SRT → plain text concatenation. Filter by time range if given.

    简单 SRT → 纯文本拼接。给了时间区间就只保留覆盖该段的字幕行。

    Not a full parser — just enough to recover Chinese/English prompt text
    from B 站 official subtitles. Skips index lines and timestamps.
    不是完整 parser，够把 B 站官方字幕转回 prompt_text 就行：
    跳过序号行与时间戳行，剩下的文本拼接起来。
    """
    text_lines: list[str] = []
    current_start = current_end = None
    for raw in srt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            current_start = current_end = None
            continue
        if "-->" in line:
            try:
                a, b = line.split("-->")
                current_start = _srt_ts(a.strip())
                current_end = _srt_ts(b.strip())
            except Exception:  # noqa: BLE001 — malformed line, skip
                current_start = current_end = None
            continue
        if line.isdigit() and current_start is None:
            continue  # SRT sequence index
        if time_range is not None and current_start is not None and current_end is not None:
            lo, hi = time_range
            if current_end < lo or current_start > hi:
                continue
        text_lines.append(line)
    return " ".join(text_lines).strip()


def _srt_ts(ts: str) -> float:
    """Parse ``HH:MM:SS,mmm`` (or ``HH:MM:SS.mmm``) into seconds.

    把 SRT 的 ``HH:MM:SS,mmm`` 时间戳转换成浮点秒数。
    """
    ts = ts.replace(",", ".")
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _bilibili_pick_url(info, part_index: int | None) -> str:
    """Decide which URL to feed `extract_audio` based on multi-P selection.

    根据用户选的 part_index 决定喂给 extract_audio 的 URL：
    - 单 P 视频：直接用 info.url
    - 多 P + 未选：默认第 1 P
    - 多 P + 选了：用对应 part 的子 URL
    """
    if len(info.parts) <= 1:
        return info.url
    idx = part_index or 1
    for p in info.parts:
        if p.index == idx:
            return p.url
    raise HTTPException(
        status_code=400,
        detail=f"part_index={part_index} not found; available: "
               f"{[p.index for p in info.parts]}",
    )


@app.post("/api/bilibili/probe", response_model=BilibiliProbeResponse)
async def bilibili_probe(req: BilibiliProbeRequest):
    """Probe a Bilibili URL — return metadata, no media bytes downloaded.

    探测 B 站 URL，返回元数据；不下载媒体字节。
    前端拿到 parts 列表与字幕语言后才让用户选择要不要导入、导入哪段。
    """
    from tools.bilibili_mcp import extractor
    try:
        info = await asyncio.to_thread(extractor.probe, req.url)
    except extractor.BilibiliExtractError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return BilibiliProbeResponse(
        bvid=info.bvid,
        title=info.title,
        uploader=info.uploader,
        duration=info.duration,
        parts=[
            BilibiliPartInfo(index=p.index, title=p.title, duration=p.duration)
            for p in info.parts
        ],
        available_subtitles=sorted(info.available_subtitles.keys()),
    )


# --- stage helpers ---------------------------------------------------------
# Tiny state-machine helpers for PipelineCard's `StageState` list. Pulled out
# so the actual task body reads like a story instead of dict updates.
# 小型状态机辅助函数：把 stage 更新从任务主体里抽出来，让正文像故事一样可读。


def _stage_start(stage: StageState, detail: str = "") -> StageState:
    """Mark a pending stage as running. 把 pending stage 推进到 running。"""
    return stage.model_copy(update={
        "status": "running",
        "started_at": _now_utc_iso(),
        "detail": detail or stage.detail,
    })


def _stage_finish(stage: StageState, *, ok: bool = True, detail: str | None = None) -> StageState:
    """Mark a running stage as done/error and record elapsed seconds.

    标 running stage 为 done/error，并记录耗时秒数。
    """
    now = datetime.now(timezone.utc)
    elapsed: float | None = None
    if stage.started_at:
        try:
            started = datetime.fromisoformat(stage.started_at)
            elapsed = (now - started).total_seconds()
        except ValueError:
            elapsed = None
    return stage.model_copy(update={
        "status": "done" if ok else "error",
        "finished_at": now.isoformat(timespec="seconds"),
        "elapsed_s": elapsed,
        "detail": detail if detail is not None else stage.detail,
    })


def _stage_detail(stage: StageState, detail: str) -> StageState:
    """Update only the detail line of a (typically running) stage.

    只更新某个（通常 running 状态的）stage 的 detail 行，
    供 yt-dlp 下载进度等流式信号写入。
    """
    return stage.model_copy(update={"detail": detail})


async def _bilibili_import_task(
    app: FastAPI, job_id: str, req: BilibiliImportRequest,
) -> None:
    """Run download + ASR off the event loop; update stages and ref sidecar.

    后台跑下载 + ASR；过程中维护 stages 数组供前端 PipelineCard 渲染，
    最终把成品落到 uploads/upload_<id>.json，让 /api/refs 自动收录。

    失败只更新 job state（status="error"），不抛出。
    """
    from tools.bilibili_mcp import extractor

    jobs: dict = app.state.bilibili_jobs

    # Initialise three pending stages up front so the UI can render the full
    # pipeline skeleton from the first poll, with empty slots filling in.
    # 一次性初始化三个 pending stage，前端第一次轮询就能拿到完整骨架。
    stages = [
        StageState(name="download", detail="等待开始"),
        StageState(name="transcribe", detail="等待开始"),
        StageState(name="persist", detail="等待开始"),
    ]
    jobs[job_id] = jobs[job_id].model_copy(update={
        "stages": stages, "progress_hint": "队列中",
    })

    def _commit(*, status: str | None = None, progress_hint: str | None = None, **extra):
        """Atomic update: write current `stages` + optional top-level fields back.

        原子地把当前 stages + 可选顶层字段写回 jobs[job_id]。
        """
        cur = jobs[job_id]
        update: dict = {"stages": list(stages)}
        if status is not None:
            update["status"] = status
        if progress_hint is not None:
            update["progress_hint"] = progress_hint
        update.update(extra)
        jobs[job_id] = cur.model_copy(update=update)

    try:
        time_range: tuple[float, float] | None = None
        if req.start_sec is not None and req.end_sec is not None:
            time_range = (float(req.start_sec), float(req.end_sec))
        elif (req.start_sec is None) ^ (req.end_sec is None):
            raise ValueError("start_sec and end_sec must be provided together.")

        # ---- Stage 1: Download (also includes the implicit probe) ----------
        stages[0] = _stage_start(stages[0], detail="探测视频元数据...")
        _commit(status="downloading", progress_hint="下载中")

        info = await asyncio.to_thread(extractor.probe, req.url)
        target_url = _bilibili_pick_url(info, req.part_index)
        stages[0] = _stage_detail(
            stages[0],
            f"yt-dlp 拉取 bestaudio · {info.title[:30]}...",
        )
        _commit()

        # yt-dlp progress hook — runs from a worker thread, so don't touch
        # async state directly; just append to `stages[0].detail`.
        # yt-dlp 进度回调在 worker 线程跑，直接同步写 stages[0].detail（dict
        # 本身就是线程安全的 read-modify-write 单步操作）。
        last_pct_logged = [-1.0]

        def _progress_hook(d):
            if d.get("status") == "downloading":
                total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                done = d.get("downloaded_bytes", 0)
                if total > 0:
                    pct = done * 100.0 / total
                    if pct - last_pct_logged[0] < 5:  # only update every ~5%
                        return
                    last_pct_logged[0] = pct
                    mb_done = done / (1024 * 1024)
                    mb_total = total / (1024 * 1024)
                    stages[0] = _stage_detail(
                        stages[0],
                        f"yt-dlp 下载 {pct:.0f}% · {mb_done:.1f}/{mb_total:.1f} MB",
                    )
                    _commit()
            elif d.get("status") == "finished":
                stages[0] = _stage_detail(stages[0], "yt-dlp 完成，ffmpeg 转 WAV 中...")
                _commit()

        sub_langs = ("zh-CN", "zh-Hans", "en")
        result = await asyncio.to_thread(
            extractor.extract_audio,
            target_url,
            UPLOADS_DIR,
            time_range=time_range,
            audio_format="wav",
            fetch_subtitles=req.use_subtitle_as_prompt,
            sub_langs=sub_langs,
            overwrite=False,
            progress_hooks=[_progress_hook],
        )
        clip_desc = f"{(time_range[1] - time_range[0]):.1f}s 切片" if time_range else "全片"
        stages[0] = _stage_finish(
            stages[0],
            detail=f"{clip_desc} · {result.audio_path.stat().st_size // 1024} KB WAV",
        )
        _commit()

        # ---- Stage 2: Transcribe (subtitle fast-path, ASR fallback) -------
        # Rename the extractor's output to upload_<id>.wav (canonical layout
        # for /api/refs). 把 BV 命名改成 upload_<id>.wav，让 /api/refs 收录。
        ref_id = _new_upload_ref_id()
        canonical_audio = UPLOADS_DIR / f"{ref_id}.wav"
        result.audio_path.rename(canonical_audio)

        prompt_text = ""
        prompt_source = "asr"
        asr_lang: str | None = None
        asr_conf: float | None = None

        if req.use_subtitle_as_prompt and result.subtitle_paths:
            stages[1] = _stage_start(stages[1], detail="解析官方字幕...")
            _commit(status="transcribing", progress_hint="解析字幕中")
            for srt in result.subtitle_paths:
                if srt.suffix.lower() != ".srt":
                    continue
                text = _parse_srt_text(srt, time_range)
                if text:
                    prompt_text = text
                    prompt_source = "subtitle"
                    break

        if not prompt_text:
            # Either no subtitle was downloaded, or it was empty / didn't
            # overlap the requested time range — fall back to ASR.
            # 字幕没拉到/没命中区间，走 ASR。
            stages[1] = _stage_start(
                stages[1],
                detail="字幕未命中 — 走 ASR（首次需下 FunASR 模型 ~944MB）",
            )
            _commit(status="transcribing", progress_hint="ASR 中")
            transcriber = _get_or_init_transcriber(app)
            stages[1] = _stage_detail(stages[1], "ASR 模型已就绪，识别音频...")
            _commit()
            asr_result = await asyncio.to_thread(transcriber.transcribe, canonical_audio)
            prompt_text = asr_result.text
            asr_lang = asr_result.lang
            asr_conf = asr_result.confidence
            prompt_source = "asr"

        stages[1] = _stage_finish(
            stages[1],
            detail=f"{prompt_source} · lang={asr_lang or '—'} · "
                   f"{len(prompt_text)} 字",
        )
        _commit()

        # ---- Stage 3: Persist (sidecar JSON + duration probe) -------------
        stages[2] = _stage_start(stages[2], detail="写 sidecar JSON 中...")
        _commit()

        duration: float | None = None
        try:
            import soundfile as sf
            sf_info = sf.info(str(canonical_audio))
            duration = float(sf_info.frames) / float(sf_info.samplerate)
        except Exception as e:  # noqa: BLE001 — best-effort metadata
            logger.warning("bilibili import: failed to read duration: %s", e)

        meta = {
            "ref_id": ref_id,
            "audio_path": str(canonical_audio),
            "prompt_text": prompt_text,
            "duration": duration,
            "asr_lang": asr_lang,
            "asr_confidence": asr_conf,
            "uploaded_at": _now_utc_iso(),
            "source": "bilibili",
            "bilibili": {
                "url": req.url,
                "bvid": info.bvid,
                "title": info.title,
                "uploader": info.uploader,
                "part_index": req.part_index,
                "time_range": list(time_range) if time_range else None,
                "subtitle_paths": [str(p) for p in result.subtitle_paths],
                "prompt_source": prompt_source,
            },
        }
        (UPLOADS_DIR / f"{ref_id}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        stages[2] = _stage_finish(
            stages[2],
            detail=f"ref_id={ref_id} · duration={duration:.1f}s" if duration
                   else f"ref_id={ref_id}",
        )
        _commit(
            status="ready",
            progress_hint="完成",
            ref_id=ref_id,
            prompt_text=prompt_text,
            duration=duration,
            asr_lang=asr_lang,
            asr_confidence=asr_conf,
            prompt_source=prompt_source,
        )
        logger.info(
            "[%s] bilibili import done: bvid=%s ref_id=%s prompt_source=%s",
            job_id, info.bvid, ref_id, prompt_source,
        )
    except Exception as e:  # noqa: BLE001 — surface the error to the UI, don't crash
        logger.exception("[%s] bilibili import failed: %s", job_id, e)
        # Mark the currently-running stage as error so the UI shows where it broke.
        # 把正在跑的 stage 标 error，让前端高亮卡在哪一段。
        for i, s in enumerate(stages):
            if s.status == "running":
                stages[i] = _stage_finish(s, ok=False, detail=str(e))
                break
        _commit(status="error", error=str(e), progress_hint="失败")


@app.post("/api/bilibili/import", response_model=BilibiliImportJob)
async def bilibili_import(req: BilibiliImportRequest):
    """Kick off an async Bilibili import; returns a job id to poll.

    启动一个异步 B 站导入任务；返回 job_id 给前端轮询状态。
    """
    job_id = _new_bilibili_job_id()
    job = BilibiliImportJob(job_id=job_id, status="queued", progress_hint="排队中...")
    app.state.bilibili_jobs[job_id] = job
    asyncio.create_task(_bilibili_import_task(app, job_id, req))
    return job


@app.get("/api/bilibili/import/{job_id}", response_model=BilibiliImportJob)
async def bilibili_import_status(job_id: str):
    """Return the current state of a Bilibili import job.

    返回某个 B 站导入任务的当前状态。前端拿到 status=="ready" 后
    应当刷新 /api/refs 并自动选中新的 ref_id。
    """
    job = app.state.bilibili_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id!r}")
    return job


def _get_or_init_transcriber(app: FastAPI):
    """Lazy-init the shared Transcriber (Whisper + FunASR routing).

    懒加载共享的 Transcriber；第一次调用承担 ASR 模型加载（~5-10s）。

    Used by both /api/upload-ref (ASR for uploaded refs) and the background
    eval task (ASR cycle for WER/CER).
    /api/upload-ref（参考音频转写）与 eval 后台任务（WER/CER cycle）
    共享同一个 Transcriber 实例。
    """
    if app.state.transcriber is None:
        from core.asr import Transcriber
        logger.info("Loading ASR transcriber for the first time...")
        app.state.transcriber = Transcriber()
    return app.state.transcriber


def _run_eval_blocking(
    syn_wav: Path, ref_wav: Path, target_text: str, transcriber,
) -> dict:
    """Synchronous wrapper around evaluate_synthesis; returns a JSON-safe dict.

    evaluate_synthesis 的同步包装；返回可 JSON 序列化的 dict。
    Called from a thread pool to avoid blocking the asyncio event loop.
    通过线程池调用，避免阻塞 asyncio 事件循环。
    """
    from core.eval_tts import evaluate_synthesis
    scores = evaluate_synthesis(
        syn_wav,
        ref_wav=ref_wav,
        target_text=target_text,
        lang="en",
        transcriber=transcriber,
    )
    # Strip transcript field for the API response (kept in meta JSON
    # but not echoed back via /api/eval to reduce payload).
    # 把 transcript 留在 meta JSON，不通过 /api/eval 回吐前端，减小载荷。
    from dataclasses import asdict
    return asdict(scores)


async def _eval_background_task(
    app: FastAPI, syn_id: str, syn_wav: Path, ref_wav: Path, target_text: str,
) -> None:
    """Run eval off the event loop, persist results into the meta JSON.

    把 eval 扔到线程池跑，完成后把结果合入对应 meta JSON。

    Failures are logged + recorded as state="error" but never propagated —
    the synthesis itself already returned successfully to the client.
    失败只记日志 + 标 state="error"，不抛出——合成 HTTP 响应已经回去了。
    """
    app.state.eval_jobs[syn_id] = "running"
    try:
        transcriber = _get_or_init_transcriber(app)
        eval_dict = await asyncio.to_thread(
            _run_eval_blocking, syn_wav, ref_wav, target_text, transcriber,
        )
        meta_path = META_DIR / f"{syn_id}.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["eval"] = eval_dict
            meta_path.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8",
            )
        app.state.eval_jobs[syn_id] = "done"
        logger.info(
            "[%s] eval done: NISQA=%.3f WER=%.3f SECS=%.3f F0=%.1fHz t=%.1fs",
            syn_id,
            eval_dict.get("mos_nisqa") or 0,
            eval_dict.get("wer") or 0,
            eval_dict.get("secs") or 0,
            eval_dict.get("f0_rmse_hz") or 0,
            eval_dict.get("eval_time_s") or 0,
        )
    except Exception as e:  # noqa: BLE001 — eval failures must not crash server
        logger.exception("eval failed for syn_id=%s: %s", syn_id, e)
        app.state.eval_jobs[syn_id] = f"error: {e}"


@app.post("/api/synthesize", response_model=SynthesizeResponse)
async def synthesize(req: SynthesizeRequest):
    """Run one synthesis through the shared TTS worker.

    用共享 TTS worker 跑一次合成。

    Blocking call (CosyVoice 2 single-instance constraint). Concurrent
    POSTs are serialised by FastAPI's single event loop + the worker's
    line protocol — fine for a local single-user tool.

    阻塞调用（CosyVoice 2 单实例约束）。并发 POST 在 FastAPI 单事件循环
    + worker 行协议下天然串行——本地单用户工具够用。
    """
    if req.mode == "instruct" and req.voice_config is None:
        raise HTTPException(status_code=422, detail="instruct mode requires voice_config")

    ref_audio_path, prompt_text = _resolve_ref(req.ref_id)
    if not ref_audio_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"ref audio file missing on disk: {ref_audio_path}",
        )

    composed = None
    if req.mode == "instruct":
        composed = compose_instruct(req.voice_config)  # type: ignore[arg-type]
        if not composed:
            raise HTTPException(
                status_code=422,
                detail="instruct mode requires at least one non-empty voice_config field",
            )

    syn_id = _new_syn_id()
    out_path = SYN_DIR / f"{syn_id}.wav"
    SYN_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    try:
        app.state.tts.synthesize(
            req.text,
            ref_audio_path,
            out_path=out_path,
            prompt_text=prompt_text,
            instruct=composed,
            mode=req.mode,
            normalize=True,
        )
    except TTSError as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}") from e
    wall = time.monotonic() - t0

    meta = {
        "syn_id": syn_id,
        "timestamp": _now_utc_iso(),
        "text": req.text,
        "ref_id": req.ref_id,
        "ref_audio_path": str(ref_audio_path),
        "prompt_text": prompt_text,
        "mode": req.mode,
        "voice_config": req.voice_config.model_dump() if req.voice_config else None,
        "composed_instruct": composed,
        "wall_time_s": wall,
        "output_path": str(out_path),
    }
    META_DIR.mkdir(parents=True, exist_ok=True)
    (META_DIR / f"{syn_id}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    # Fire-and-forget eval; result patched into meta when ready and made
    # available via GET /api/eval/{syn_id}.
    # 立即触发后台 eval；结果就绪后合入 meta，前端走 GET /api/eval/{syn_id}
    # 拉取。
    asyncio.create_task(
        _eval_background_task(app, syn_id, out_path, ref_audio_path, req.text)
    )

    return SynthesizeResponse(
        syn_id=syn_id,
        audio_url=f"/api/audio/{syn_id}",
        wall_time_s=wall,
        mode=req.mode,
        composed_instruct=composed,
    )


@app.get("/api/audio/{syn_id}")
async def get_audio(syn_id: str):
    """Stream a synthesised wav file.

    流式返回合成的 wav 文件。
    """
    wav_path = SYN_DIR / f"{syn_id}.wav"
    if not wav_path.exists():
        raise HTTPException(status_code=404, detail=f"unknown syn_id {syn_id!r}")
    return FileResponse(wav_path, media_type="audio/wav")


@app.get("/api/eval/{syn_id}", response_model=EvalResponse)
async def get_eval(syn_id: str):
    """Return objective eval scores for a synthesis (poll until done).

    返回一次合成的客观评测分；evaluating 时前端轮询。

    Status flow:
      1. /api/synthesize returns → background eval starts → status="running"
      2. Eval finishes → meta JSON patched with "eval" key → status="done"
      3. If eval raised → status="error"
    Status 流转：1) /api/synthesize 返回 → 后台 eval 开始（running）
    → 2) eval 写回 meta JSON（done） → 3) 异常时 error。
    """
    meta_path = META_DIR / f"{syn_id}.json"
    if not meta_path.exists():
        return EvalResponse(syn_id=syn_id, status="unknown")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if "eval" in meta:
        e = meta["eval"]
        return EvalResponse(
            syn_id=syn_id,
            status="done",
            scores=EvalScores(
                mos_nisqa=e.get("mos_nisqa"),
                mos_p808=e.get("mos_p808"),
                wer=e.get("wer"),
                cer=e.get("cer"),
                secs=e.get("secs"),
                f0_rmse_hz=e.get("f0_rmse_hz"),
                eval_time_s=e.get("eval_time_s"),
            ),
        )
    job_state = app.state.eval_jobs.get(syn_id, "running")
    if job_state.startswith("error"):
        return EvalResponse(syn_id=syn_id, status="error", error=job_state)
    return EvalResponse(syn_id=syn_id, status="running")


@app.post("/api/feedback")
async def post_feedback(entry: FeedbackEntry):
    """Append one feedback entry to ``outputs/webui/feedback.jsonl``.

    把一条反馈追加到 ``outputs/webui/feedback.jsonl``。
    """
    if not (META_DIR / f"{entry.syn_id}.json").exists():
        raise HTTPException(status_code=404, detail=f"unknown syn_id {entry.syn_id!r}")

    payload = entry.model_dump()
    payload["ts"] = _now_utc_iso()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with FEEDBACK_PATH.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return {"ok": True}


@app.get("/api/history", response_model=list[HistoryEntry])
async def list_history(limit: int = 20):
    """Return the most recent ``limit`` syntheses (newest first).

    返回最近 limit 次合成（最新在前）。

    Feedback is loaded from ``feedback.jsonl``; if a synthesis has
    multiple ratings (rare — user re-rated), the **last** one wins.

    反馈从 ``feedback.jsonl`` 读；若同一 syn_id 有多条评分（少见，用户
    重新打分），取**最后**一条。
    """
    if not META_DIR.exists():
        return []

    meta_files = sorted(META_DIR.glob("syn_*.json"), reverse=True)[:limit]
    feedback_by_id: dict[str, FeedbackEntry] = {}
    if FEEDBACK_PATH.exists():
        for line in FEEDBACK_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                feedback_by_id[row["syn_id"]] = FeedbackEntry(
                    syn_id=row["syn_id"],
                    rating=row["rating"],
                    tags=row.get("tags", []),
                    note=row.get("note", ""),
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("skipping malformed feedback line: %s", e)

    out: list[HistoryEntry] = []
    for mp in meta_files:
        m = json.loads(mp.read_text(encoding="utf-8"))
        eval_block = None
        if "eval" in m and m["eval"]:
            e = m["eval"]
            eval_block = EvalScores(
                mos_nisqa=e.get("mos_nisqa"),
                mos_p808=e.get("mos_p808"),
                wer=e.get("wer"),
                cer=e.get("cer"),
                secs=e.get("secs"),
                f0_rmse_hz=e.get("f0_rmse_hz"),
                eval_time_s=e.get("eval_time_s"),
            )
        out.append(HistoryEntry(
            syn_id=m["syn_id"],
            timestamp=m["timestamp"],
            text=m["text"],
            ref_id=m["ref_id"],
            mode=m["mode"],
            composed_instruct=m.get("composed_instruct"),
            audio_url=f"/api/audio/{m['syn_id']}",
            wall_time_s=m["wall_time_s"],
            feedback=feedback_by_id.get(m["syn_id"]),
            eval=eval_block,
        ))
    return out
