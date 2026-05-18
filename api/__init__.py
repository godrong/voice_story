"""FastAPI web service for the TTS playground WebUI.

Thin HTTP wrapper around `core.tts.LocalSubprocessTTS`. The CosyVoice
worker is held in app state so it can be reused across requests, avoiding
the ~19s cold start that hurts the per-request runner pattern used in
experiments/.

WebUI 的 FastAPI 后端。

对 ``core.tts.LocalSubprocessTTS`` 的薄 HTTP 封装。CosyVoice worker
在 app state 中常驻复用，避免每次请求都吃一次 ~19 秒的冷启动
（experiments/ 里的 runner 是单次跑，所以每次都冷启动）。
"""
