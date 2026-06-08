// components.js — Vue 3 components for the TTS playground (redesign v2).
//
// Layout: 3-zone bento (per ui-ux-pro-max-skill + A2UI pattern).
//   Zone 1 — VoiceSourcePanel  (B站 / YouTube / Upload / Built-in tabs)
//   Zone 2 — PipelineDAG       (Cytoscape.js DAG of agent stages)
//   Zone 3 — SynthesisPanel    (text + voice config + scores)
//
// Design tokens come from styles/base.css + styles/zones.css.
// Backend API contract is unchanged — same endpoints, same payloads.
//
// components.js ——TTS playground 的 Vue 3 组件（v2 重设）。
// 三区 bento + Cytoscape.js A2UI DAG，沿用 ui-ux-pro-max-skill 推荐的设计 token。
// 后端 API 契约不变，只换前端表现层。

const { defineComponent, computed, ref, watch, onMounted, onUnmounted } = Vue;


// ---------------------------------------------------------------------------
// PipelineDAG — Cytoscape.js A2UI-style DAG of agent stages.
//
// Stages from the backend are linear (download / transcribe / persist for
// bilibili import, or source / preprocess / dataset for full ingest). We
// render them as nodes + edges with live status colors and a click-to-inspect
// side panel.
//
// PipelineDAG ——A2UI 风格的 Cytoscape.js DAG。后端给的 stages 是线性的，
// 我们渲染成节点 + 边 + 状态色 + 点击侧栏看详情。
// ---------------------------------------------------------------------------

const STATE_COLOR = {
  pending: "#475569",
  running: "#6B8FAF",
  done:    "#22C55E",
  error:   "#EF4444",
};

export const PipelineDAG = defineComponent({
  name: "PipelineDAG",
  props: {
    title:  { type: String, default: "Pipeline" },
    stages: { type: Array,  required: true },
  },
  emits: ["node-select"],
  setup(props, { emit }) {
    const cyRef = ref(null);
    let cy = null;

    function buildElements(stages) {
      const els = [];
      stages.forEach((s, i) => {
        els.push({ data: {
          id: `s_${i}_${s.name}`,
          label: s.name,
          color: STATE_COLOR[s.status] || STATE_COLOR.pending,
          status: s.status,
          stageIdx: i,
        }});
        if (i > 0) {
          els.push({ data: {
            source: `s_${i-1}_${stages[i-1].name}`,
            target: `s_${i}_${s.name}`,
          }});
        }
      });
      return els;
    }

    function initCy() {
      if (!cyRef.value || cy) return;
      cy = cytoscape({
        container: cyRef.value,
        elements: buildElements(props.stages),
        style: [
          {
            selector: "node",
            style: {
              "background-color": "data(color)",
              label: "data(label)",
              color: "#F8FAFC",
              "text-valign": "bottom",
              "text-margin-y": 8,
              "font-family": "Fira Code, monospace",
              "font-size": "11px",
              "text-transform": "capitalize",
              width: 40, height: 40,
              "border-width": 2,
              "border-color": "#0F172A",
            },
          },
          {
            selector: 'node[status = "running"]',
            style: { "border-color": "#6B8FAF", "border-width": 3 },
          },
          {
            selector: "node:selected",
            style: { "border-color": "#22C55E", "border-width": 4 },
          },
          {
            selector: "edge",
            style: {
              "curve-style": "bezier",
              "target-arrow-shape": "triangle",
              "line-color": "#90A4AE",
              "target-arrow-color": "#90A4AE",
              opacity: 0.5,
              width: 2,
            },
          },
        ],
        layout: { name: "breadthfirst", directed: true, spacingFactor: 1.4, padding: 16 },
        wheelSensitivity: 0.2,
      });

      cy.on("tap", "node", (evt) => {
        const idx = evt.target.data("stageIdx");
        if (idx != null) emit("node-select", { index: idx, stage: props.stages[idx] });
      });

      // Pulse the currently-running node.
      // 给 running 节点加脉冲动画，作为 A2UI 的视觉心跳。
      pulseInterval = setInterval(() => {
        if (!cy) return;
        const running = cy.nodes('[status = "running"]');
        if (running.length === 0) return;
        running.animate(
          { style: { "border-width": 6 } },
          { duration: 600, complete: () => {
            running.animate({ style: { "border-width": 3 } }, { duration: 600 });
          }},
        );
      }, 1400);
    }

    let pulseInterval = null;

    function updateCy() {
      if (!cy) { initCy(); return; }
      // Diff: keep IDs stable per stage index so we can do a fast in-place
      // update without rebuilding the layout (which would jitter the graph).
      // 通过稳定 ID 做就地更新，不重建 layout，避免抖动。
      props.stages.forEach((s, i) => {
        const id = `s_${i}_${s.name}`;
        const n = cy.getElementById(id);
        if (n.empty()) {
          // Stage appeared (rare); rebuild fully.
          cy.elements().remove();
          cy.add(buildElements(props.stages));
          cy.layout({ name: "breadthfirst", directed: true, spacingFactor: 1.4 }).run();
          return;
        }
        n.data("color", STATE_COLOR[s.status] || STATE_COLOR.pending);
        n.data("status", s.status);
      });
      cy.style().update();
    }

    onMounted(() => initCy());
    onUnmounted(() => {
      if (pulseInterval) clearInterval(pulseInterval);
      if (cy) { cy.destroy(); cy = null; }
    });

    watch(() => props.stages, () => updateCy(), { deep: true });

    return { cyRef };
  },
  template: `
    <div>
      <div class="zone-title" style="margin-bottom: 6px;">{{ title }}</div>
      <div class="cy-canvas" ref="cyRef"></div>
      <div class="legend">
        <span><i class="dot" style="background:#475569"></i>pending</span>
        <span><i class="dot" style="background:#6B8FAF"></i>running</span>
        <span><i class="dot" style="background:#22C55E"></i>done</span>
        <span><i class="dot" style="background:#EF4444"></i>error</span>
        <span class="right">click a node for details</span>
      </div>
    </div>
  `,
});


// ---------------------------------------------------------------------------
// NodeDetailPanel — side panel for an A2UI DAG node.
//
// Shows status / elapsed / detail line. When no node is selected (or the
// stage hasn't started), shows a hint.
// 显示选中 stage 的详情；未选中时给提示。
// ---------------------------------------------------------------------------

export const NodeDetailPanel = defineComponent({
  name: "NodeDetailPanel",
  props: {
    stage:    { type: Object, default: null },
    jobMeta:  { type: Object, default: null },  // {job_id, status, name, ...}
  },
  setup(props) {
    const tick = ref(0);
    const timer = setInterval(() => { tick.value++; }, 1000);
    onUnmounted(() => clearInterval(timer));

    const elapsedStr = computed(() => {
      void tick.value;
      const s = props.stage;
      if (!s) return "";
      if (s.elapsed_s != null) return `${s.elapsed_s.toFixed(1)}s`;
      if (s.status === "running" && s.started_at) {
        const sec = ((Date.now() - new Date(s.started_at).getTime()) / 1000).toFixed(0);
        return `${sec}s…`;
      }
      return "—";
    });

    return { elapsedStr };
  },
  template: `
    <div class="node-detail">
      <template v-if="stage">
        <div class="nd-title" style="text-transform: capitalize;">{{ stage.name }}</div>
        <div class="nd-row"><span>status</span>
          <span class="v" :style="{color: stage.status === 'done' ? '#22C55E'
                                       : stage.status === 'running' ? '#6B8FAF'
                                       : stage.status === 'error' ? '#EF4444' : '#94A3B8'}">
            {{ stage.status }}
          </span>
        </div>
        <div class="nd-row"><span>elapsed</span><span class="v">{{ elapsedStr }}</span></div>
        <div class="nd-row" v-if="stage.started_at">
          <span>started</span>
          <span class="v">{{ stage.started_at.replace('T',' ').replace('+00:00','') }}</span>
        </div>
        <div class="nd-row" v-if="jobMeta && jobMeta.job_id">
          <span>job_id</span><span class="v">{{ jobMeta.job_id.slice(0, 28) }}…</span>
        </div>
        <div v-if="stage.detail" class="nd-log">{{ stage.detail }}</div>
      </template>
      <template v-else>
        <div class="nd-hint">No stage selected.<br/>Click a node on the left to inspect.</div>
        <template v-if="jobMeta && jobMeta.job_id">
          <hr style="border:none; border-top:1px solid #1F2A3D; margin: 10px 0;" />
          <div class="nd-row"><span>job</span><span class="v">{{ jobMeta.name }}</span></div>
          <div class="nd-row"><span>source</span><span class="v">{{ jobMeta.source }}</span></div>
          <div class="nd-row"><span>status</span><span class="v">{{ jobMeta.status }}</span></div>
        </template>
      </template>
    </div>
  `,
});


// ---------------------------------------------------------------------------
// SourceTabs — small tab strip used by VoiceSourcePanel.
// 子组件：源类型切换 tab。
// ---------------------------------------------------------------------------

export const SourceTabs = defineComponent({
  name: "SourceTabs",
  props: {
    active:        { type: String, required: true },
    uploadedCount: { type: Number, default: 0 },
  },
  emits: ["update:active"],
  setup(props, { emit }) {
    const tabs = [
      { key: "bilibili", label: "B站 URL" },
      { key: "youtube",  label: "YouTube" },
      { key: "record",   label: "🎙 录音" },
      { key: "upload",   label: "Upload" },
      { key: "uploaded", label: "Uploaded" },
      { key: "builtin",  label: "Built-in" },
    ];
    return { tabs, set: (k) => emit("update:active", k) };
  },
  template: `
    <div class="tabs">
      <button v-for="t in tabs" :key="t.key"
              class="tab"
              :class="{active: active === t.key}"
              @click="set(t.key)">
        {{ t.label }}
        <span v-if="t.key === 'uploaded' && uploadedCount" class="badge">{{ uploadedCount }}</span>
      </button>
    </div>
  `,
});


// ---------------------------------------------------------------------------
// ReferencePreview — show the currently-selected reference with waveform +
// prompt text. Pure presentation; play button toggles a hidden <audio>.
// 当前选中参考音频的预览卡（波形占位 + 文本 + 试听）。
// ---------------------------------------------------------------------------

export const ReferencePreview = defineComponent({
  name: "ReferencePreview",
  // NOTE: `ref` is a reserved attribute in Vue 3 (template refs). Don't name
  // a prop `ref` — bindings like `:ref="x"` get hijacked by the template-ref
  // system and the prop is never populated. Using `audio` here instead.
  // 注意：Vue 3 里 `ref` 是模板 ref 的保留属性名，不能当 prop 名用；
  // 否则 `:ref="x"` 会被模板 ref 系统抢走，props.ref 永远是 undefined。
  props: {
    audio: { type: Object, default: null },
  },
  setup(props) {
    const audioElRef = ref(null);
    const playing = ref(false);

    const audioUrl = computed(() => {
      const r = props.audio;
      if (!r) return "";
      // Uploaded / bilibili-imported refs point at outputs/webui/uploads/.
      // Built-in refs point at datasets/<name>/chunks/. We serve them via
      // /static/... since webui is mounted as the static root, but uploads
      // live under outputs/ which isn't mounted — fallback to file path.
      // 上传/B 站导入的产物走 /static 拿不到；当前 webui 没暴露 raw audio 端点，
      // 这里仅作为占位 UX，真正试听要后端加端点。先用空 src，按钮 disabled。
      return "";
    });

    function togglePlay() {
      if (!audioElRef.value) return;
      if (playing.value) audioElRef.value.pause();
      else audioElRef.value.play();
    }

    // Fake waveform — 60 random-but-stable bars driven by ref_id hash.
    // 假波形：60 个随机但稳定的竖条（按 ref_id hash 种子），仅展示用。
    const bars = computed(() => {
      const r = props.audio;
      if (!r) return [];
      let seed = 0;
      for (const c of r.ref_id || "") seed = (seed * 31 + c.charCodeAt(0)) & 0xffffffff;
      const out = [];
      for (let i = 0; i < 60; i++) {
        seed = (seed * 1103515245 + 12345) & 0x7fffffff;
        const h = 6 + (seed % 32);
        out.push({ x: i * 6, y: (44 - h) / 2, h });
      }
      return out;
    });

    return { audioElRef, audioUrl, playing, togglePlay, bars };
  },
  template: `
    <div class="vinfo" v-if="audio">
      <h4>
        <span>Active reference</span>
        <code style="font-size: 10px; color: var(--fg-muted);">
          {{ audio.ref_id }} · {{ audio.duration ? audio.duration.toFixed(1)+'s' : '—' }}
        </code>
      </h4>
      <div class="waveform">
        <svg viewBox="0 0 360 44" preserveAspectRatio="none">
          <g fill="#6B8FAF">
            <rect v-for="(b, i) in bars" :key="i"
                  :x="b.x" :y="b.y" width="3" :height="b.h" rx="1" />
          </g>
        </svg>
      </div>
      <div class="play-row">
        <button class="play-btn" disabled title="audio streaming endpoint TBD">▶</button>
        <span>{{ audio.duration ? audio.duration.toFixed(1) + 's' : '—' }}</span>
        <span style="margin-left:auto; max-width: 60%; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
          prompt: <span style="color: var(--fg);">{{ audio.prompt_text || '—' }}</span>
        </span>
      </div>
    </div>
    <div class="vinfo" v-else style="text-align:center; color: var(--fg-muted); font-style: italic;">
      No active reference. Pick one from a tab above.
    </div>
  `,
});


// ---------------------------------------------------------------------------
// VoiceSourcePanel — Zone 1. 4-tab source picker. Each tab owns the form for
// that source type; all backend chatter is routed through emit events so the
// parent (app.js) keeps its existing fetch + polling logic.
//
// VoiceSourcePanel ——Zone 1 的四 tab 源选择器，子表单 emit 意图给父层。
// ---------------------------------------------------------------------------

export const VoiceSourcePanel = defineComponent({
  name: "VoiceSourcePanel",
  components: { SourceTabs, ReferencePreview },
  props: {
    refs:           { type: Array,   required: true },  // small set: ?limit=200
    selectedRefId:  { type: String,  default: "" },
    // Bilibili pipeline state (from parent)
    bilibiliProbing:       { type: Boolean, default: false },
    bilibiliProbeResult:   { type: Object,  default: null },
    bilibiliImporting:     { type: Boolean, default: false },
    bilibiliError:         { type: String,  default: "" },
    // Local upload state
    uploading:             { type: Boolean, default: false },
  },
  emits: [
    "update:selectedRefId",
    "upload",
    "bilibili-probe", "bilibili-import",
    "youtube-extract",
    "builtin-search",
  ],
  setup(props, { emit }) {
    const activeTab = ref("bilibili");

    // B 站 form
    const bvUrl = ref("");
    const bvStart = ref(null);
    const bvEnd = ref(null);
    const bvUseSubtitle = ref(true);

    // YouTube form
    const ytUrl = ref("");
    const ytStart = ref(null);
    const ytEnd = ref(null);

    // Built-in search (debounced — re-fetches via emit)
    const builtinQuery = ref("");
    let builtinDebounce = null;
    watch(builtinQuery, (q) => {
      if (builtinDebounce) clearTimeout(builtinDebounce);
      builtinDebounce = setTimeout(() => emit("builtin-search", q.trim()), 250);
    });

    // local upload
    const onFile = (e) => {
      const f = e.target.files && e.target.files[0];
      if (f) emit("upload", f);
      e.target.value = "";
    };

    // --- mic recording ---
    // 用 MediaRecorder 抓 webm/opus，停止后用 AudioContext 解码 → 重编 WAV，
    // 复用 upload 通路（后端只要 wav/mp3 即可）。
    const REC_MAX_S = 60;  // 自动 stop 上限
    const recState = ref("idle");  // idle | requesting | recording | stopping | preview | error
    const recError = ref("");
    const recElapsedS = ref(0);
    const recPreviewUrl = ref("");
    const recPreviewBytes = ref(0);
    let mediaRecorder = null;
    let recordedChunks = [];
    let recordedBlob = null;        // raw webm blob from MediaRecorder
    let recordedWavBlob = null;     // re-encoded wav blob (uploaded)
    let micStream = null;
    let recTimer = null;
    let recStart0 = 0;

    function _stopMic() {
      if (recTimer) { clearInterval(recTimer); recTimer = null; }
      if (micStream) {
        micStream.getTracks().forEach(t => t.stop());
        micStream = null;
      }
    }

    function _resetRec() {
      _stopMic();
      mediaRecorder = null;
      recordedChunks = [];
      recordedBlob = null;
      recordedWavBlob = null;
      if (recPreviewUrl.value) URL.revokeObjectURL(recPreviewUrl.value);
      recPreviewUrl.value = "";
      recPreviewBytes.value = 0;
      recElapsedS.value = 0;
      recError.value = "";
    }

    async function onRecStart() {
      _resetRec();
      recState.value = "requesting";
      try {
        micStream = await navigator.mediaDevices.getUserMedia({
          audio: { channelCount: 1, sampleRate: 48000, echoCancellation: true, noiseSuppression: true },
        });
      } catch (e) {
        recState.value = "error";
        recError.value = `麦克风权限被拒：${e.message || e.name}`;
        return;
      }
      const mimeTypes = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", ""];
      const mt = mimeTypes.find(t => !t || MediaRecorder.isTypeSupported(t)) || "";
      mediaRecorder = new MediaRecorder(micStream, mt ? { mimeType: mt } : undefined);
      mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) recordedChunks.push(e.data); };
      mediaRecorder.onstop = onRecStopped;
      mediaRecorder.start();
      recState.value = "recording";
      recStart0 = Date.now();
      recElapsedS.value = 0;
      recTimer = setInterval(() => {
        recElapsedS.value = (Date.now() - recStart0) / 1000;
        if (recElapsedS.value >= REC_MAX_S) onRecStop();
      }, 100);
    }

    function onRecStop() {
      if (!mediaRecorder || mediaRecorder.state === "inactive") return;
      recState.value = "stopping";
      mediaRecorder.stop();
    }

    async function onRecStopped() {
      _stopMic();
      const mt = mediaRecorder?.mimeType || "audio/webm";
      recordedBlob = new Blob(recordedChunks, { type: mt });
      try {
        recordedWavBlob = await _blobToWav(recordedBlob);
      } catch (e) {
        recState.value = "error";
        recError.value = `编码 WAV 失败：${e.message || e}`;
        return;
      }
      recPreviewUrl.value = URL.createObjectURL(recordedWavBlob);
      recPreviewBytes.value = recordedWavBlob.size;
      recState.value = "preview";
    }

    function onRecUse() {
      if (!recordedWavBlob) return;
      // file name lines up with the upload tab — uploaded sidecar will key
      // off mime/extension on the server side.
      // 文件名与 Upload tab 风格保持一致；后端按 mime/ext 落 sidecar。
      const stamp = new Date().toISOString().replace(/[-:.TZ]/g, "").slice(0, 14);
      const file = new File([recordedWavBlob], `mic_${stamp}.wav`, { type: "audio/wav" });
      emit("upload", file);
      _resetRec();
      recState.value = "idle";
    }

    // --- WAV encoder: decode webm/opus via AudioContext → PCM16 LE → RIFF ---
    // 浏览器内把任意 codec 的录音重编成 PCM16 WAV；后端 soundfile/faster-whisper
    // 拿到 WAV 都能识别，避开 webm/opus 兼容性问题。
    async function _blobToWav(blob) {
      const ab = await blob.arrayBuffer();
      const AC = window.AudioContext || window.webkitAudioContext;
      const ctx = new AC();
      const buf = await ctx.decodeAudioData(ab.slice(0));
      const sampleRate = buf.sampleRate;
      const ch = buf.numberOfChannels;
      // Down-mix to mono if needed (cloning rarely benefits from stereo).
      // 多声道降到 mono；克隆基本只用得到 mono。
      const samples = buf.length;
      const mono = new Float32Array(samples);
      for (let c = 0; c < ch; c++) {
        const data = buf.getChannelData(c);
        for (let i = 0; i < samples; i++) mono[i] += data[i] / ch;
      }
      const wavBytes = _floatToWav(mono, sampleRate);
      await ctx.close();
      return new Blob([wavBytes], { type: "audio/wav" });
    }

    function _floatToWav(samples, sampleRate) {
      const len = samples.length;
      const buf = new ArrayBuffer(44 + len * 2);
      const v = new DataView(buf);
      const writeString = (off, s) => { for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i)); };
      // RIFF header
      writeString(0, "RIFF");
      v.setUint32(4, 36 + len * 2, true);
      writeString(8, "WAVE");
      // fmt chunk
      writeString(12, "fmt ");
      v.setUint32(16, 16, true);        // PCM chunk size
      v.setUint16(20, 1, true);         // format = PCM
      v.setUint16(22, 1, true);         // channels = mono
      v.setUint32(24, sampleRate, true);
      v.setUint32(28, sampleRate * 2, true);  // byte rate
      v.setUint16(32, 2, true);         // block align
      v.setUint16(34, 16, true);        // bits per sample
      // data chunk
      writeString(36, "data");
      v.setUint32(40, len * 2, true);
      // PCM samples
      let off = 44;
      for (let i = 0; i < len; i++) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        v.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
        off += 2;
      }
      return new Uint8Array(buf);
    }

    onUnmounted(() => _resetRec());

    const onProbe = () => {
      if (!bvUrl.value.trim()) return;
      emit("bilibili-probe", bvUrl.value.trim());
    };
    const onBiliImport = () => {
      if (!bvUrl.value.trim()) return;
      emit("bilibili-import", {
        url: bvUrl.value.trim(),
        start_sec: bvStart.value,
        end_sec: bvEnd.value,
        use_subtitle_as_prompt: bvUseSubtitle.value,
      });
    };
    const onYtExtract = () => {
      if (!ytUrl.value.trim()) return;
      emit("youtube-extract", {
        url: ytUrl.value.trim(),
        start_sec: ytStart.value,
        end_sec: ytEnd.value,
      });
    };

    // refs are server-side filtered (sources prefiltered, search applied).
    // Frontend just splits by source for the Uploaded vs Built-in tabs.
    // refs 已被后端按 limit/source/search 过滤；前端只按 source 分两组显示。
    const uploadedRefs = computed(() => props.refs.filter(r => r.source === "upload"));
    const uploadedCount = computed(() => uploadedRefs.value.length);
    const builtinRefs = computed(() => props.refs.filter(r => r.source === "builtin"));
    const selectedRef = computed(() =>
      props.refs.find(r => r.ref_id === props.selectedRefId) || null
    );

    return {
      activeTab,
      bvUrl, bvStart, bvEnd, bvUseSubtitle,
      ytUrl, ytStart, ytEnd,
      builtinQuery,
      onFile, onProbe, onBiliImport, onYtExtract,
      uploadedRefs, uploadedCount, builtinRefs, selectedRef,
      onPick: (refId) => emit("update:selectedRefId", refId),
      // mic recording state + actions
      recState, recError, recElapsedS, recPreviewUrl, recPreviewBytes,
      onRecStart, onRecStop, onRecUse,
      onRecRedo: () => { _resetRec(); recState.value = "idle"; },
      REC_MAX_S,
    };
  },
  template: `
    <section class="zone">
      <h3 class="zone-title">Zone 1 · Voice Source</h3>

      <SourceTabs v-model:active="activeTab" :uploadedCount="uploadedCount" />

      <!-- B 站 tab -->
      <template v-if="activeTab === 'bilibili'">
        <div class="field">
          <label class="field-label">URL / BV id</label>
          <input type="text" class="select" v-model="bvUrl"
                 :disabled="bilibiliProbing || bilibiliImporting"
                 placeholder="https://www.bilibili.com/video/BVxxx or BV id" />
        </div>
        <div style="display:flex; gap:6px;">
          <button class="btn-primary" style="background: var(--bg-elev); border-color: var(--bg-elev); color: var(--fg); font-weight: 500;"
                  :disabled="bilibiliProbing || bilibiliImporting || !bvUrl.trim()"
                  @click="onProbe">
            {{ bilibiliProbing ? 'Probing…' : 'Probe' }}
          </button>
        </div>
        <div v-if="bilibiliError" class="field-hint" style="color: var(--state-error); margin-top:6px;">
          {{ bilibiliError }}
        </div>

        <div class="vinfo" v-if="bilibiliProbeResult">
          <h4>{{ bilibiliProbeResult.title }}</h4>
          <div class="row">
            <span>UP <code>{{ bilibiliProbeResult.uploader }}</code></span>
            <span>{{ bilibiliProbeResult.duration.toFixed(1) }}s · {{ bilibiliProbeResult.parts.length }} part(s)</span>
            <span v-if="bilibiliProbeResult.available_subtitles.length">
              subs: {{ bilibiliProbeResult.available_subtitles.join(', ') }}
            </span>
            <span v-else style="color: var(--fg-muted)">no subs</span>
          </div>
          <div class="field" style="margin-top: 10px;">
            <label class="field-label">Clip range (sec) · 推荐 20–40s 做 ref</label>
            <div style="display:flex; gap:6px; align-items:center;">
              <input type="number" style="width:80px;" class="select" min="0" step="0.5"
                     v-model.number="bvStart" :disabled="bilibiliImporting" placeholder="start" />
              <span style="color: var(--fg-muted)">→</span>
              <input type="number" style="width:80px;" class="select" min="0" step="0.5"
                     v-model.number="bvEnd" :disabled="bilibiliImporting" placeholder="end" />
              <span style="color: var(--fg-muted); font-family: var(--font-mono); font-size:11px;">
                = {{ (bvEnd != null && bvStart != null) ? (bvEnd - bvStart).toFixed(1) + 's' : '—' }}
              </span>
            </div>
          </div>
          <div class="field" v-if="bilibiliProbeResult.available_subtitles.length">
            <label style="display:flex; align-items:center; gap:6px; font-weight:normal; font-size:12px;">
              <input type="checkbox" v-model="bvUseSubtitle" :disabled="bilibiliImporting" />
              Use official subtitle as prompt_text (skips ASR)
            </label>
          </div>
          <button class="btn-primary"
                  :disabled="bilibiliImporting" @click="onBiliImport">
            {{ bilibiliImporting ? 'Importing…' : 'Extract → reference' }}
          </button>
        </div>
      </template>

      <!-- YouTube tab -->
      <template v-if="activeTab === 'youtube'">
        <div class="field">
          <label class="field-label">YouTube URL</label>
          <input type="text" class="select" v-model="ytUrl"
                 placeholder="https://www.youtube.com/watch?v=..." />
        </div>
        <div class="field">
          <label class="field-label">Clip range (sec, optional)</label>
          <div style="display:flex; gap:6px; align-items:center;">
            <input type="number" style="width:80px;" class="select" min="0" step="0.5"
                   v-model.number="ytStart" placeholder="start" />
            <span style="color: var(--fg-muted)">→</span>
            <input type="number" style="width:80px;" class="select" min="0" step="0.5"
                   v-model.number="ytEnd" placeholder="end" />
          </div>
        </div>
        <button class="btn-primary" :disabled="!ytUrl.trim()" @click="onYtExtract">
          Extract → dataset
        </button>
        <div class="field-hint" style="margin-top:6px;">
          YouTube extraction goes through the full ingest pipeline (Demucs → VAD → ASR → manifest).
          Watch Zone 2 for live progress.
        </div>
      </template>

      <!-- Record tab — capture mic via getUserMedia + MediaRecorder, re-encode to WAV -->
      <template v-if="activeTab === 'record'">
        <div class="field-hint" style="margin-bottom: 10px;">
          直接用电脑麦克风录一段干净的 5–30 秒语音作为参考。
          自动停止上限 {{ REC_MAX_S }}s。
        </div>

        <!-- idle: big start button -->
        <div v-if="recState === 'idle'" style="display:flex; align-items:center; gap:12px;">
          <button class="btn-primary" @click="onRecStart">
            ● 开始录音
          </button>
          <span class="field-hint" style="margin:0;">需要授予浏览器麦克风权限</span>
        </div>

        <!-- requesting permission -->
        <div v-if="recState === 'requesting'" class="field-hint">
          正在请求麦克风权限…
        </div>

        <!-- recording: red pulse + timer + stop -->
        <div v-if="recState === 'recording'"
             style="display:flex; align-items:center; gap:14px; padding:10px;
                    background: var(--bg-elev); border-radius:8px; border: 1px solid var(--state-error);">
          <span style="width:14px; height:14px; border-radius:50%; background: var(--state-error);
                       animation: pulse 1.1s ease-in-out infinite;"></span>
          <span style="font-family: var(--font-mono); font-size: 14px;">
            录音中 · {{ recElapsedS.toFixed(1) }}s
            <span class="field-hint" style="margin-left:8px;">(自动停 {{ REC_MAX_S }}s)</span>
          </span>
          <button class="btn-primary" style="margin-left:auto; background: var(--state-error); border-color: var(--state-error);"
                  @click="onRecStop">
            ■ 停止
          </button>
        </div>

        <!-- stopping: re-encoding to WAV -->
        <div v-if="recState === 'stopping'" class="field-hint">
          停止录音 · 转 WAV 中…
        </div>

        <!-- preview: audio + use/redo -->
        <div v-if="recState === 'preview'"
             style="padding:10px; background: var(--bg-elev); border-radius:8px; border: 1px solid var(--border);">
          <div class="field-hint" style="margin-bottom: 8px;">
            录音预览 · {{ recElapsedS.toFixed(1) }}s · {{ (recPreviewBytes/1024).toFixed(0) }} KB (WAV)
          </div>
          <audio controls :src="recPreviewUrl" style="width:100%; margin-bottom:10px;"></audio>
          <div style="display:flex; gap:8px;">
            <button class="btn-primary" @click="onRecUse">
              使用这段录音 → 参考
            </button>
            <button class="btn-primary"
                    style="background: var(--bg); border-color: var(--border); color: var(--fg);"
                    @click="onRecRedo">
              重录
            </button>
          </div>
        </div>

        <!-- error -->
        <div v-if="recState === 'error'" class="field-hint" style="color: var(--state-error);">
          {{ recError }}
          <button class="btn-primary"
                  style="background: var(--bg); border-color: var(--border); color: var(--fg); margin-top:8px;"
                  @click="onRecRedo">重试</button>
        </div>
      </template>

      <!-- Upload tab -->
      <template v-if="activeTab === 'upload'">
        <div class="field">
          <label class="field-label">Upload local audio</label>
          <input type="file" class="file-input"
                 accept="audio/wav,audio/mpeg,audio/x-wav,.wav,.mp3"
                 :disabled="uploading"
                 @change="onFile" />
          <div v-if="uploading" class="field-hint">uploading + auto-ASR... (5–15s)</div>
        </div>
        <div class="field-hint">
          Drop a clean 20–60s clip of one speaker. ASR will auto-transcribe;
          you can edit the prompt_text afterwards.
        </div>
      </template>

      <!-- Uploaded tab — user's previously imported refs (cheap, small set) -->
      <template v-if="activeTab === 'uploaded'">
        <div class="field-hint" style="margin-bottom: 6px;">
          {{ uploadedCount }} uploaded / imported reference(s). Click to select.
        </div>
        <div v-if="!uploadedCount" class="field-hint" style="font-style: italic;">
          None yet. Use B 站 URL / YouTube / Upload tabs to add one.
        </div>
        <div v-for="r in uploadedRefs" :key="r.ref_id"
             style="padding:8px 10px; margin-bottom:4px; background: var(--bg); border-radius:6px; border: 1px solid var(--border); cursor:pointer;"
             :style="r.ref_id === selectedRefId ? 'border-color: var(--cta);' : ''"
             @click="onPick(r.ref_id)">
          <div style="font-family: var(--font-mono); font-size: 11px; color: var(--fg-muted);">
            {{ r.ref_id }} · {{ r.duration ? r.duration.toFixed(1)+'s' : '—' }}
            <span v-if="r.ref_id === selectedRefId" style="color: var(--cta); margin-left: 6px;">✓ selected</span>
          </div>
          <div style="font-size: 12px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
            {{ r.prompt_text || '(no prompt_text)' }}
          </div>
        </div>
      </template>

      <!-- Built-in tab — server-paginated. Search box debounces a refetch. -->
      <template v-if="activeTab === 'builtin'">
        <div class="field">
          <label class="field-label">Search built-in references</label>
          <input type="text" class="select" v-model="builtinQuery"
                 placeholder="search by ref_id / dataset / prompt_text..." />
          <div class="field-hint">
            Showing {{ builtinRefs.length }} matches (server-capped at 200).
            Refine with search to narrow down.
          </div>
        </div>
        <div v-for="r in builtinRefs.slice(0, 50)" :key="r.ref_id"
             style="padding:6px 10px; margin-bottom:3px; background: var(--bg); border-radius:6px; border: 1px solid var(--border); cursor:pointer;"
             :style="r.ref_id === selectedRefId ? 'border-color: var(--cta);' : ''"
             @click="onPick(r.ref_id)">
          <div style="font-family: var(--font-mono); font-size: 11px;">
            <span style="color: var(--accent-voice);">[{{ r.dataset || '?' }}]</span>
            {{ r.ref_id }}
            <span style="color: var(--fg-muted);">· {{ r.duration ? r.duration.toFixed(1)+'s' : '—' }}</span>
            <span v-if="r.ref_id === selectedRefId" style="color: var(--cta); margin-left: 6px;">✓</span>
          </div>
        </div>
        <div v-if="builtinRefs.length > 50" class="field-hint" style="text-align:center;">
          (showing first 50 of {{ builtinRefs.length }} — refine search to see more)
        </div>
      </template>

      <!-- Active reference preview, always shown -->
      <div style="margin-top: 12px;">
        <ReferencePreview :audio="selectedRef" />
      </div>
    </section>
  `,
});


// ---------------------------------------------------------------------------
// SynthesisPanel — Zone 3. Text input + voice config + Run + player + scores.
// Replaces the TextPanel / ConfigPanel / ActionBar / PlayerPanel trio with
// one denser bento card.
//
// SynthesisPanel ——Zone 3，把原来的 Text/Config/Action/Player 四个 panel
// 合并成一张密度更高的合成卡片，附带评分小卡片。
// ---------------------------------------------------------------------------

export const SynthesisPanel = defineComponent({
  name: "SynthesisPanel",
  props: {
    text:        { type: String, default: "" },
    mode:        { type: String, default: "zero_shot" },
    // NOTE: `config` is passed as a reactive object from the parent. We
    // mutate fields in place via `set-config-field` (not v-model:config),
    // because v-model:config would try to reassign a reactive const which
    // either breaks reactivity or throws.
    // 注意：parent 的 config 是 reactive 常量。我们通过 set-config-field 事件
    // 让 parent 原地改字段，不用 v-model:config（会试图整体重新赋值）。
    config:      { type: Object, required: true },
    composed:    { type: String, default: "" },
    selectedRefId: { type: String, default: "" },
    runStatus:   { type: String, default: "idle" },
    statusMessage: { type: String, default: "" },
    lastSyn:     { type: Object, default: null },
    evalStatus:  { type: String, default: "" },
    evalScores:  { type: Object, default: null },
  },
  emits: ["update:text", "update:mode", "set-config-field", "synthesize"],
  setup(props, { emit }) {
    const onText = (e) => emit("update:text", e.target.value);
    const onMode = (e) => emit("update:mode", e.target.value);
    // Single-field mutation event — parent does `config.k = v` in place to
    // preserve the reactive object identity.
    // 单字段事件——parent 原地 config.k = v，保持 reactive 对象 identity。
    const setCfg = (k, v) => emit("set-config-field", { key: k, value: v });

    // --- Paralinguistic token insertion ---
    // CV3 supports inline atom tokens (e.g. [breath]) and wrap tokens
    // (e.g. <strong>...</strong>, <laughter>...</laughter>). Atom inserts at
    // cursor; wrap surrounds selection (or inserts tag pair empty).
    // 副语言 token 插入：原子型直接插光标处，包裹型环绕选区或插空标签对。
    // Source of token names: RESEARCH.md §维度 4 副语言元素.
    const TOKEN_CHIPS = [
      { kind: "atom", token: "[breath]",        label: "呼吸"  },
      { kind: "atom", token: "[quick_breath]",  label: "短促呼吸" },
      { kind: "atom", token: "[laughter]",      label: "笑声"  },
      { kind: "atom", token: "[sigh]",          label: "叹气"  },
      { kind: "atom", token: "[cough]",         label: "咳嗽"  },
      { kind: "atom", token: "[mn]",            label: "嗯"   },
      { kind: "atom", token: "[lipsmack]",      label: "咂嘴"  },
      { kind: "wrap", open: "<strong>",   close: "</strong>",   label: "强调…"   },
      { kind: "wrap", open: "<laughter>", close: "</laughter>", label: "笑着说…" },
    ];
    const textareaRef = ref(null);

    function insertToken(chip) {
      const ta = textareaRef.value;
      const txt = props.text || "";
      const ss = ta ? ta.selectionStart : txt.length;
      const se = ta ? ta.selectionEnd   : txt.length;
      let next, caret;
      if (chip.kind === "atom") {
        next = txt.slice(0, ss) + chip.token + txt.slice(se);
        caret = ss + chip.token.length;
      } else {
        // wrap: surround selection (or insert empty pair + drop cursor inside)
        // wrap：环绕选区；无选区时插空标签对、光标停在中间方便接着打字
        const sel = txt.slice(ss, se);
        next = txt.slice(0, ss) + chip.open + sel + chip.close + txt.slice(se);
        caret = ss + chip.open.length + sel.length;
      }
      emit("update:text", next);
      // Restore focus + caret after Vue re-render flushes the new value.
      // 等 Vue flush 完 DOM 再恢复光标，否则 selectionStart 会被重置到末尾。
      setTimeout(() => {
        if (!textareaRef.value) return;
        textareaRef.value.focus();
        textareaRef.value.setSelectionRange(caret, caret);
      }, 0);
    }

    const evalDot = computed(() => {
      const e = props.evalStatus;
      if (e === "done") return "ok";
      if (e === "running") return "run";
      if (e === "error") return "err";
      return "";
    });

    // Compute the precise reason synth is disabled, so the user sees WHY.
    // 精确给出 synth 被禁用的原因，让用户一眼看到为啥点不了。
    const disabledReason = computed(() => {
      if (props.runStatus === "running") return "synthesising — please wait";
      const missing = [];
      if (!props.selectedRefId) missing.push("pick a reference (Zone 1)");
      if (!props.text || !props.text.trim()) missing.push("type target text");
      if (props.mode === "instruct" && !props.composed) {
        missing.push("fill at least one voice config field");
      }
      return missing.length ? "need: " + missing.join(" · ") : "";
    });
    const canRun = computed(() => !disabledReason.value);

    return {
      onText, onMode, setCfg, evalDot, disabledReason, canRun,
      TOKEN_CHIPS, insertToken, textareaRef,
    };
  },
  template: `
    <section class="zone zone-full">
      <h3 class="zone-title">Zone 3 · Synthesis</h3>
      <div class="synth-grid">
        <div>
          <div class="field">
            <label class="field-label">Target text</label>
            <!-- Paralinguistic token chips. Atom = insert at cursor;
                 Wrap = surround selection (or empty tag pair).
                 副语言 token 按钮条：原子插光标，包裹环绕选区。 -->
            <div class="token-chips">
              <button v-for="chip in TOKEN_CHIPS" :key="chip.token || chip.open"
                      type="button" class="token-chip"
                      :title="chip.kind === 'wrap' ? (chip.open + '...' + chip.close) : chip.token"
                      @click="insertToken(chip)">
                <span class="chip-label">{{ chip.label }}</span>
                <span class="chip-token">{{ chip.kind === 'wrap' ? chip.open : chip.token }}</span>
              </button>
            </div>
            <textarea class="textarea textarea-tall" ref="textareaRef"
                      :value="text" @input="onText"
                      placeholder="It's great to be back in Beijing, truly fantastic..."></textarea>
          </div>
          <div class="field">
            <label class="field-label">Mode</label>
            <select class="select" :value="mode" @change="onMode">
              <option value="zero_shot">zero_shot — copy ref voice directly</option>
              <option value="instruct">instruct — compose voice from config</option>
            </select>
          </div>
        </div>

        <div>
          <div class="card-title" style="margin-bottom: 4px;">Voice config (instruct 模式)</div>
          <div class="field-hint" style="margin-bottom: 8px;">
            只控制「怎么说」——音色 / 性别 / 年龄由参考音频决定，无法也无需在此设置。
          </div>
          <div class="field">
            <label class="field-label">风格 Quality</label>
            <select class="select" :value="config.quality || ''"
                    @change="e => setCfg('quality', e.target.value || null)">
              <option value="">—</option>
              <option value="studio">studio · 音质干净清晰</option>
              <option value="broadcast">broadcast · 带专业播音腔</option>
              <option value="casual">casual · 语气轻松随意</option>
            </select>
          </div>
          <div class="field">
            <label class="field-label">口吻 / 情绪 / 补充描述（自由文本）</label>
            <input class="select" :value="config.persona"
                   @input="e => setCfg('persona', e.target.value)"
                   placeholder="口吻：如 知心朋友、讲故事的人、新闻主播" />
            <input class="select" style="margin-top:6px;" :value="config.emotion"
                   @input="e => setCfg('emotion', e.target.value)"
                   placeholder="情绪：如 平静温暖、激动、低沉" />
            <input class="select" style="margin-top:6px;" :value="config.description"
                   @input="e => setCfg('description', e.target.value)"
                   placeholder="补充：如 语速放慢，像在哄睡（完整中文句子）" />
          </div>
          <div v-if="mode === 'instruct' && composed" class="field-hint" style="margin-top:4px;">
            实际指令：<code style="color: var(--accent-voice);">{{ composed }}</code>
          </div>
        </div>
      </div>

      <div style="display:flex; gap:8px; align-items:center; margin-top: 12px; flex-wrap: wrap;">
        <button class="btn-primary" style="padding: 10px 22px;"
                :disabled="!canRun"
                @click="$emit('synthesize')">
          {{ runStatus === 'running' ? '▶ Synthesising…' : '▶ Synthesize' }}
        </button>
        <span v-if="disabledReason"
              class="field-hint mono"
              style="color: var(--state-error); background: rgba(239,68,68,0.08); padding: 4px 8px; border-radius: 4px;">
          ⓘ {{ disabledReason }}
        </span>
        <span v-else class="field-hint mono" style="color: var(--fg-muted);">
          {{ statusMessage }}
        </span>
        <span style="margin-left: auto; font-family: var(--font-mono); font-size: 11px; color: var(--fg-muted);">
          <template v-if="lastSyn">
            last: {{ lastSyn.wall_time_s.toFixed(1) }}s ·
            <code>{{ lastSyn.syn_id.slice(-9) }}</code> ·
            <span :class="evalDot === 'ok' ? 'ok' : evalDot === 'err' ? 'err' : 'run'"
                  :style="{color: evalDot === 'ok' ? 'var(--state-done)'
                                : evalDot === 'err' ? 'var(--state-error)'
                                : 'var(--state-running)'}">
              {{ evalStatus === 'done' ? '✓ eval done' :
                 evalStatus === 'error' ? '✗ eval error' :
                 evalStatus === 'running' ? '⏳ eval running' : '— no eval' }}
            </span>
          </template>
        </span>
      </div>

      <!-- Player -->
      <div v-if="lastSyn" style="margin-top: 12px;">
        <audio controls :src="lastSyn.audio_url" style="width:100%; height:32px;"></audio>
      </div>

      <!-- Score cards -->
      <div class="scores" v-if="evalScores">
        <div class="score-card">
          <div class="v green">{{ evalScores.secs != null ? evalScores.secs.toFixed(3) : '—' }}</div>
          <div class="l">SECS</div>
        </div>
        <div class="score-card">
          <div class="v">{{ evalScores.mos_nisqa != null ? evalScores.mos_nisqa.toFixed(2) : '—' }}</div>
          <div class="l">MOS NISQA</div>
        </div>
        <div class="score-card">
          <div class="v blue">{{ evalScores.cer != null ? evalScores.cer.toFixed(2) : (evalScores.wer != null ? evalScores.wer.toFixed(2) : '—') }}</div>
          <div class="l">{{ evalScores.cer != null ? 'CER' : 'WER' }}</div>
        </div>
        <div class="score-card">
          <div class="v">{{ evalScores.f0_rmse_hz != null ? evalScores.f0_rmse_hz.toFixed(1) + ' Hz' : '—' }}</div>
          <div class="l">F0 RMSE</div>
        </div>
      </div>
    </section>
  `,
});


// ---------------------------------------------------------------------------
// FeedbackPanel — listening rating + tags. Light restyle of the v1 panel.
// FeedbackPanel ——听感打分，沿用 v1 逻辑，仅外观跟新主题对齐。
// ---------------------------------------------------------------------------

export const FeedbackPanel = defineComponent({
  name: "FeedbackPanel",
  props: {
    synId: { type: String, default: "" },
    saved: { type: Boolean, default: false },
  },
  emits: ["submit"],
  setup(props, { emit }) {
    const rating = ref(0);
    const tags = ref([]);
    const note = ref("");
    const ALL_TAGS = ["natural", "robotic", "mispronounce", "wrong-emotion", "great"];
    const toggleTag = (t) => {
      const i = tags.value.indexOf(t);
      if (i >= 0) tags.value.splice(i, 1); else tags.value.push(t);
    };
    const submit = () => {
      if (!props.synId || !rating.value) return;
      emit("submit", {
        syn_id: props.synId, rating: rating.value,
        tags: [...tags.value], note: note.value,
      });
    };
    watch(() => props.synId, () => {
      rating.value = 0; tags.value = []; note.value = "";
    });
    return { rating, tags, note, ALL_TAGS, toggleTag, submit };
  },
  template: `
    <div class="card">
      <div class="card-title">Feedback</div>
      <div v-if="!synId" class="field-hint">Synthesize first to leave feedback.</div>
      <template v-else>
        <div class="field">
          <label class="field-label">Rating</label>
          <div style="display:flex; gap:6px;">
            <button v-for="n in 5" :key="n"
                    class="tab"
                    :class="{active: rating === n}"
                    @click="rating = n">{{ n }}★</button>
          </div>
        </div>
        <div class="field">
          <label class="field-label">Tags (multi-select)</label>
          <div style="display:flex; gap:4px; flex-wrap: wrap;">
            <button v-for="t in ALL_TAGS" :key="t"
                    class="tab"
                    :class="{active: tags.includes(t)}"
                    @click="toggleTag(t)">{{ t }}</button>
          </div>
        </div>
        <div class="field">
          <label class="field-label">Note (optional)</label>
          <textarea class="textarea" v-model="note" placeholder="anything specific?"></textarea>
        </div>
        <button class="btn-primary"
                :disabled="!rating || saved"
                @click="submit">
          {{ saved ? 'Saved ✓' : 'Submit feedback' }}
        </button>
      </template>
    </div>
  `,
});


// ---------------------------------------------------------------------------
// HistoryPanel — past synth list. Click to replay.
// HistoryPanel ——历史合成列表，点击重听。
// ---------------------------------------------------------------------------

export const HistoryPanel = defineComponent({
  name: "HistoryPanel",
  props: {
    items: { type: Array, required: true },
  },
  emits: ["replay"],
  setup(props, { emit }) {
    return { onReplay: (it) => emit("replay", it) };
  },
  template: `
    <div class="card">
      <div class="card-title">History · {{ items.length }} entries</div>
      <div v-if="!items.length" class="field-hint">No synthesises yet.</div>
      <div v-for="it in items" :key="it.syn_id"
           style="padding:8px 0; border-bottom: 1px solid var(--border); display:flex; gap:8px; align-items:center;">
        <button class="tab" @click="onReplay(it)">▶ replay</button>
        <div style="flex:1; min-width: 0;">
          <div style="font-family: var(--font-mono); font-size: 11px; color: var(--fg-muted);">
            {{ it.timestamp.replace('T',' ').slice(0,19) }} ·
            ref={{ it.ref_id.slice(0, 32) }} · {{ it.mode }} ·
            {{ it.wall_time_s.toFixed(1) }}s
            <span v-if="it.eval && it.eval.secs != null"
                  :style="{color: 'var(--state-done)'}">
              · SECS {{ it.eval.secs.toFixed(3) }}
            </span>
          </div>
          <div style="overflow:hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 12px;">
            {{ it.text }}
          </div>
        </div>
      </div>
    </div>
  `,
});
