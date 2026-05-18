# bilibili_mcp

B 站音频提取的独立模块。四层架构：

```
.claude/skills/bilibili-extract/SKILL.md   ← LLM 工作流引导
        ↓ 通过 MCP 协议调用
tools/bilibili_mcp/server.py               ← MCP stdio server（FastMCP）
        ↓ Python 函数调用
tools/bilibili_mcp/extractor.py            ← 纯函数核心（yt-dlp + ffmpeg）
        ↑ 同样被这一层消费
core/sources/bilibili.py                   ← Source Protocol 适配，喂主管线
```

## 安装

```bash
# 在项目根
conda run -n ai_study pip install -e '.[bilibili]'
# 或直接：
conda run -n ai_study pip install -U yt-dlp mcp
```

ffmpeg 需要在系统 PATH 中（`brew install ffmpeg`，与 voice_story 主管线共用）。

## 三种调用方式

### 1) 通过 Claude Code（MCP server）

`.mcp.json` 在项目根已配置好。重启 Claude Code 后会自动连接 `bilibili-extract`
server，4 个工具可见：

| 工具 | 用途 |
|---|---|
| `bilibili_probe` | 拿元数据 + 分 P 列表（不下载） |
| `bilibili_extract_audio` | 单视频音频提取（支持时间区间） |
| `bilibili_extract_playlist` | 多 P / 合集批量 |
| `bilibili_fetch_subtitles` | 只拉官方字幕 |

skill 文件 [`.claude/skills/bilibili-extract/SKILL.md`](../../.claude/skills/bilibili-extract/SKILL.md)
里定义了 LLM 调用顺序和典型对话。

### 2) 直接当 Python 库

```python
from tools.bilibili_mcp import extractor

info = extractor.probe("BV1xx411x7xx")
print(info.title, info.duration, len(info.parts))

r = extractor.extract_audio(
    "BV1xx411x7xx",
    out_dir="downloads",
    time_range=(10.0, 40.0),
    fetch_subtitles=True,
)
print(r.audio_path)         # downloads/BV1xx411x7xx_T10.0-40.0.wav
print(r.subtitle_paths)     # [downloads/BV1xx411x7xx_T10.0-40.0.zh-CN.srt, ...]
```

### 3) 通过 Source Protocol 喂主管线

```bash
conda run -n ai_study python -m cli ingest \
    --source bilibili \
    --source-kwargs '{"url":"BVxxx","time_range":[10,40]}' \
    --name my_speaker
```

落到 `inputs/my_speaker/raw/`，之后 SourceAgent 会自动标准化成 24 kHz 单声道 WAV
并接上 Demucs → VAD → 说话人筛 → ASR → manifest。

## 启动 MCP server 做调试

```bash
conda run -n ai_study python -m tools.bilibili_mcp.server
```

server 走 stdio；用 [MCP Inspector](https://github.com/modelcontextprotocol/inspector)
连过去能看到工具列表。

## 测试

```bash
# 离线（默认）：mock yt-dlp，CI 友好
conda run -n ai_study pytest tools/bilibili_mcp/tests/test_extractor_offline.py -v

# 在线烟测：手工跑，要联网，会真下载几秒钟的音频
conda run -n ai_study pytest tools/bilibili_mcp/tests/test_smoke_online.py -v -m online
```

## 已知问题

- **B 站经常更新风控**。报 `412/429/format not found` 时第一时间
  `pip install -U yt-dlp`。yt-dlp 的修复跟得很紧。
- **v1 不带 cookie**。会员/UP 主限定视频会失败。如果将来需要登录态，
  在 extractor.py 里加一个 `cookies_from_browser` 参数（yt-dlp 原生支持）。
- **不接 bilibili-api-python**。见 ADR-0003，cookie/WBI 签名维护成本太高。
