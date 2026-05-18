---
name: bilibili-extract
description: |
  Extract audio (and optionally official subtitles) from Bilibili videos.
  Trigger when the user mentions "B 站 / bilibili", a `BV...` id, or asks to
  扒/抠/拉取/下载 audio from a B 站 video URL. Use the bilibili-extract MCP
  tools (probe → extract / playlist / subs) — do NOT shell out to yt-dlp or
  curl directly. When the user wants the result to feed the voice_story
  pipeline, prefer the BilibiliSource path (see "Integration" below).
---

# Bilibili Audio Extraction

A workflow guide for extracting audio from Bilibili videos. Backed by the
`bilibili-extract` MCP server (4 tools: `bilibili_probe`,
`bilibili_extract_audio`, `bilibili_extract_playlist`,
`bilibili_fetch_subtitles`) and the `BilibiliSource` plugin at
[core/sources/bilibili.py](core/sources/bilibili.py).

## When to use

- 用户给一个 B 站 URL 或 BV 号，让你提取音频
- 用户说要做"参考音色"，想从某段视频里抠 30 秒
- 用户提到要下一个 UP 主的合集或多 P 视频
- 用户提到要拉某视频的官方字幕

## Workflow（固定顺序）

1. **先 probe，绝不要直接 extract**。
   调 `bilibili_probe(url)` 拿 title / uploader / duration / 分 P 列表 /
   可用字幕语言。**展示给用户确认**：
   - 分 P > 1 时问"全要还是哪几 P？"
   - 总时长 > 5 min 时问"要不要只取一段（比如做参考音色 30s 够）？"
   - 有官方字幕时告诉用户可以省一道 ASR。

2. **决策时间区间 / 字幕**。
   - 时间区间用浮点秒数：`start_sec=12.5, end_sec=42.0`，**不是** `HH:MM:SS`。
   - 字幕语言传 list，例如 `["zh-CN", "zh-Hans", "en"]`，按优先级。

3. **extract**。
   - 单视频：`bilibili_extract_audio(url, out_dir, ...)`
   - 多 P / 合集：`bilibili_extract_playlist(url, out_dir, parts=[...], ...)`
   - 只要字幕：`bilibili_fetch_subtitles(url, out_dir, sub_langs=[...])`

4. **校验产出**。每个工具返回 `{"ok": True, "data": ...}` 信封；
   失败时是 `{"ok": False, "error": "..."}`，**先读 error 文本里的排错提示**，
   再决定是更新 yt-dlp 还是换代理。

## Integration with voice_story pipeline

如果用户的目的是"把 B 站音频喂进 TTS 主管线"（做 manifest / 做声音克隆参考），
**不要用 MCP 工具单独下**。推荐改用 Source 形态：

```bash
conda run -n ai_study python -m cli ingest \
    --source bilibili \
    --source-kwargs '{"url":"https://www.bilibili.com/video/BVxxx","time_range":[10,40]}' \
    --name <speaker_name>
```

或在 Python 里：

```python
from core.sources import get_source
src = get_source("bilibili", name="trump_bili", url="BVxxx", time_range=(10, 40))
for path in src.fetch():
    print(path)  # 落到 inputs/trump_bili/raw/BVxxx_T10.0-40.0.wav
```

这条路径会自动接上 Demucs → VAD → 说话人筛 → ASR → manifest 整条管线
（[core/sources/bilibili.py](core/sources/bilibili.py) +
[agents/source_agent.py](agents/source_agent.py)）。

## Pitfalls

- **yt-dlp 经常被 B 站打**。看到 412/429/format-not-found 时第一步：
  `conda run -n ai_study pip install -U yt-dlp`。
- 不要用 `format='bestvideo+bestaudio'`。我们只要音频，`bestaudio/best` 够了
  （extractor 已经默认这样设置）。
- 时间区间 end 必须 > start，extractor 会校验。
- 默认 **不传 cookie**。会员专享 / UP 主限定视频会直接报"login required" —
  此时如实告诉用户该视频需要登录，**不要**尝试绕过。
- 多 P 视频里 `parts` 是 **1-based** 索引（与 B 站 ?p=N 一致）。
- 输出文件名是 `<BV>[_pN][_T<start>-<end>].<format>`，**重跑相同参数会复用**
  已有文件（除非传 `overwrite=True`）。

## Output convention

默认输出路径与 LocalSource 扫描目录对齐：

```
inputs/
└── <name>/
    └── raw/
        ├── BV1xx411x7xx.wav                  # 单视频全片
        ├── BV1xx411x7xx_T10.0-40.0.wav       # 时间区间切片
        ├── BV1xx411x7xx_p1.wav               # 多 P 第 1 P
        ├── BV1xx411x7xx_p1.zh-CN.srt         # 官方字幕（同前缀）
        └── ...
```

下游 SourceAgent 会原地把它们标准化成 24 kHz / 16-bit / mono WAV
（通过 [core/audio_io.py::to_standard_wav](core/audio_io.py#L122)）。

## Examples

**用户："帮我把 https://www.bilibili.com/video/BV1xx411x7xx 的音频抠出来"**

1. 调 `bilibili_probe("BV1xx411x7xx")`。返回 title="某演讲" duration=720s parts=1。
2. 告诉用户："这是一个 12 分钟的视频《某演讲》。你要全片还是某一段？"
3. 用户回："只要第 2-4 分钟"。
4. 调 `bilibili_extract_audio("BV1xx411x7xx", out_dir="inputs/_tmp/raw",
   start_sec=120, end_sec=240)`。
5. 检查信封 ok=True，把 `data.audio_path` 报给用户。

**用户："这个合集 https://space.bilibili.com/.../channel/seriesdetail?sid=...
   全部下了做训练集"**

1. 调 `bilibili_probe(url)`。返回 parts 列表（比如 8 个）。
2. 告诉用户分 P 标题和总时长，问"全要吗？"
3. 用户确认后建议改走 Source 形态（接主管线）而不是 MCP 工具，
   因为下完还要 Demucs/VAD/manifest，单纯 MCP 下完用户还要再跑一遍管线。
