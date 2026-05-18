// components.js — Vue 3 components for the TTS playground.
// No SFC, no bundler — template strings rendered at runtime.
//
// components.js ——TTS playground 的 Vue 3 组件，用 template string 写，
// 不走 SFC、不走打包器，运行时直接渲染。

const { defineComponent, computed, ref, watch } = Vue;

// ---------------------------------------------------------------------------
// PipelineCard — vertical stage list for any backend job exposing a `stages`
// array of {name, status, started_at, finished_at, elapsed_s, detail}.
// Reused by both the Bilibili import job (live) and the TTS synth+eval flow
// (assembled client-side from existing state).
// PipelineCard ——通用的竖排 stage 卡片；输入是 stages 数组，
// 同时被 B 站 import（后端真实数据）和 TTS 合成（前端从已有状态拼出来）复用。
// ---------------------------------------------------------------------------

export const PipelineCard = defineComponent({
  name: "PipelineCard",
  props: {
    title: { type: String, default: "Pipeline" },
    stages: { type: Array, required: true },
  },
  setup(props) {
    // Force a re-render every second so the "elapsed" of a running stage
    // ticks even between polls. We don't read this value — assigning it
    // is what triggers Vue's reactivity.
    // 每秒强制 re-render 一次，让 running stage 的耗时即使在两次轮询之间
    // 也能"跳秒"——读不到这个值没关系，赋值动作触发响应即可。
    const tick = ref(0);
    const timer = setInterval(() => { tick.value++; }, 1000);
    Vue.onUnmounted(() => clearInterval(timer));

    const icon = (s) => ({
      pending: "○",
      running: "⏳",
      done:    "✓",
      error:   "✗",
    })[s.status] || "?";

    const iconColor = (s) => ({
      pending: "#aaa",
      running: "#1e88e5",
      done:    "#2e7d32",
      error:   "#c62828",
    })[s.status] || "#888";

    const fmtElapsed = (s) => {
      void tick.value;  // subscribe to ticks
      if (s.elapsed_s != null) return `${s.elapsed_s.toFixed(1)}s`;
      if (s.status === "running" && s.started_at) {
        const start = new Date(s.started_at).getTime();
        const sec = ((Date.now() - start) / 1000).toFixed(0);
        return `${sec}s…`;
      }
      return "";
    };

    return { icon, iconColor, fmtElapsed };
  },
  template: `
    <div class="card">
      <div class="card-title">{{ title }}</div>
      <div v-for="(s, i) in stages" :key="i"
           style="padding:6px 0;"
           :style="i < stages.length - 1 ? 'border-bottom:1px solid #eee;' : ''">
        <div style="display:flex; gap:8px; align-items:baseline;">
          <span style="font-family:monospace; width:1.2em; font-size:14px;"
                :style="{color: iconColor(s)}">{{ icon(s) }}</span>
          <span style="font-weight:600; flex:1; text-transform:capitalize;">{{ s.name }}</span>
          <span class="field-hint" style="font-size:11px;">{{ fmtElapsed(s) }}</span>
        </div>
        <div v-if="s.detail"
             class="field-hint"
             style="margin-left:1.9em; font-size:11px; word-break:break-word;">
          {{ s.detail }}
        </div>
      </div>
    </div>
  `,
});

// ---------------------------------------------------------------------------
// RefPanel — choose a built-in ref or upload a new one.
// RefPanel ——选用内置 ref 或上传新 ref。
// ---------------------------------------------------------------------------

export const RefPanel = defineComponent({
  name: "RefPanel",
  components: { PipelineCard },
  props: {
    refs: { type: Array, required: true },
    selectedRefId: { type: String, default: "" },
    promptText: { type: String, default: "" },
    uploading: { type: Boolean, default: false },
    // Bilibili import UI state, owned by the parent so all backend chatter
    // stays in app.js. bilibiliProbing / bilibiliImporting are flags;
    // bilibiliProbeResult is the /api/bilibili/probe response (or null);
    // bilibiliJob is the latest /api/bilibili/import/{id} state (or null).
    // B 站导入 UI 状态由父组件持有，后端通信集中在 app.js。
    bilibiliProbing: { type: Boolean, default: false },
    bilibiliProbeResult: { type: Object, default: null },
    bilibiliImporting: { type: Boolean, default: false },
    bilibiliJob: { type: Object, default: null },
    bilibiliError: { type: String, default: "" },
  },
  emits: [
    "update:selectedRefId", "update:promptText", "upload",
    "bilibili-probe", "bilibili-import", "bilibili-reset",
  ],
  setup(props, { emit }) {
    const onSelect = (e) => emit("update:selectedRefId", e.target.value);
    const onPrompt = (e) => emit("update:promptText", e.target.value);
    const onFile = (e) => {
      const file = e.target.files && e.target.files[0];
      if (file) emit("upload", file);
      e.target.value = ""; // allow re-upload same file
    };

    const selectedRef = computed(() =>
      props.refs.find((r) => r.ref_id === props.selectedRefId) || null
    );

    // Local UI state for the Bilibili form. Lives here because it's purely
    // form-input scratch — parent doesn't need to round-trip it.
    // 本地 UI 状态——只是 form 输入草稿，不需要父组件参与。
    const bvUrl = ref("");
    const bvPartIndex = ref(null);   // 1-based; null = let backend default
    const bvStartSec = ref(null);
    const bvEndSec = ref(null);
    const bvUseSubtitle = ref(true);

    const onProbe = () => {
      const u = bvUrl.value.trim();
      if (!u) return;
      emit("bilibili-probe", u);
    };

    const onImport = () => {
      const u = bvUrl.value.trim();
      if (!u) return;
      emit("bilibili-import", {
        url: u,
        part_index: bvPartIndex.value,
        start_sec: bvStartSec.value,
        end_sec: bvEndSec.value,
        use_subtitle_as_prompt: bvUseSubtitle.value,
      });
    };

    const onReset = () => {
      bvUrl.value = "";
      bvPartIndex.value = null;
      bvStartSec.value = null;
      bvEndSec.value = null;
      bvUseSubtitle.value = true;
      emit("bilibili-reset");
    };

    const probe = computed(() => props.bilibiliProbeResult);
    const isMultiP = computed(() => probe.value && probe.value.parts && probe.value.parts.length > 1);
    const hasSubs = computed(() =>
      probe.value && probe.value.available_subtitles && probe.value.available_subtitles.length > 0
    );

    const formatDuration = (s) => {
      if (s == null) return "?";
      const m = Math.floor(s / 60);
      const sec = (s - m * 60).toFixed(1);
      return m > 0 ? `${m}m${sec}s` : `${sec}s`;
    };

    return {
      onSelect, onPrompt, onFile, selectedRef,
      bvUrl, bvPartIndex, bvStartSec, bvEndSec, bvUseSubtitle,
      onProbe, onImport, onReset,
      probe, isMultiP, hasSubs, formatDuration,
    };
  },
  template: `
    <div class="card">
      <div class="card-title">Reference Audio</div>

      <div class="field">
        <label class="field-label">Built-in reference</label>
        <select class="select" :value="selectedRefId" @change="onSelect">
          <option value="">— pick one —</option>
          <option v-for="r in refs" :key="r.ref_id" :value="r.ref_id">
            [{{ r.source }}] {{ r.ref_id }} ({{ r.duration ? r.duration.toFixed(1)+'s' : '?' }})
          </option>
        </select>
      </div>

      <div class="field">
        <label class="field-label">Or upload a new wav/mp3</label>
        <input type="file"
               class="file-input"
               accept="audio/wav,audio/mpeg,audio/x-wav,.wav,.mp3"
               :disabled="uploading"
               @change="onFile" />
        <div v-if="uploading" class="field-hint">uploading + auto-ASR... (5-15s)</div>
      </div>

      <div class="field">
        <label class="field-label">Or paste a Bilibili URL</label>
        <div style="display:flex; gap:6px;">
          <input type="text"
                 class="select"
                 style="flex:1;"
                 v-model="bvUrl"
                 :disabled="bilibiliProbing || bilibiliImporting"
                 placeholder="https://www.bilibili.com/video/BVxxx or bare BV id" />
          <button type="button"
                  class="select"
                  style="cursor:pointer; padding:4px 12px;"
                  :disabled="bilibiliProbing || bilibiliImporting || !bvUrl.trim()"
                  @click="onProbe">
            {{ bilibiliProbing ? 'Probing...' : 'Probe' }}
          </button>
        </div>
        <div v-if="bilibiliError" class="field-hint" style="color:#c0392b;">
          {{ bilibiliError }}
        </div>

        <div v-if="probe" style="margin-top:8px; padding:8px; border:1px solid #ddd; border-radius:4px;">
          <div style="font-weight:600; margin-bottom:4px;">{{ probe.title }}</div>
          <div class="field-hint">
            UP: {{ probe.uploader }} ·
            total: {{ formatDuration(probe.duration) }} ·
            parts: {{ probe.parts.length }}
            <span v-if="hasSubs"> · subtitles: {{ probe.available_subtitles.join(', ') }}</span>
          </div>

          <div v-if="isMultiP" class="field" style="margin-top:6px;">
            <label class="field-label">Pick a part</label>
            <select class="select" v-model.number="bvPartIndex">
              <option :value="null">— part 1 (default) —</option>
              <option v-for="p in probe.parts" :key="p.index" :value="p.index">
                P{{ p.index }} · {{ formatDuration(p.duration) }} · {{ p.title }}
              </option>
            </select>
          </div>

          <div class="field" style="margin-top:6px;">
            <label class="field-label">Time range (optional, seconds)</label>
            <div style="display:flex; gap:6px; align-items:center;">
              <input type="number" class="select" style="width:90px;"
                     v-model.number="bvStartSec" min="0" step="0.5"
                     placeholder="start" />
              <span>→</span>
              <input type="number" class="select" style="width:90px;"
                     v-model.number="bvEndSec" min="0" step="0.5"
                     placeholder="end" />
              <span class="field-hint">tip: 20–40s clip is plenty for a voice reference</span>
            </div>
          </div>

          <div class="field" v-if="hasSubs" style="margin-top:6px;">
            <label style="display:flex; align-items:center; gap:6px; font-weight:normal;">
              <input type="checkbox" v-model="bvUseSubtitle" />
              Use official subtitle as prompt_text (skips ASR)
            </label>
          </div>

          <div style="display:flex; gap:6px; margin-top:8px;">
            <button type="button"
                    class="select"
                    style="cursor:pointer; padding:4px 12px;"
                    :disabled="bilibiliImporting"
                    @click="onImport">
              {{ bilibiliImporting ? 'Importing...' : 'Import as reference' }}
            </button>
            <button type="button"
                    class="select"
                    style="cursor:pointer; padding:4px 12px;"
                    :disabled="bilibiliImporting"
                    @click="onReset">
              Cancel
            </button>
          </div>

          <div v-if="bilibiliJob && bilibiliJob.stages && bilibiliJob.stages.length"
               style="margin-top:8px;">
            <PipelineCard
              :title="'B 站 import · ' + (bilibiliJob.progress_hint || bilibiliJob.status)"
              :stages="bilibiliJob.stages" />
          </div>
          <div v-else-if="bilibiliJob" class="field-hint" style="margin-top:6px;">
            [{{ bilibiliJob.status }}] {{ bilibiliJob.progress_hint }}
          </div>
        </div>
      </div>

      <div class="field">
        <label class="field-label">prompt_text (what the reference audio says)</label>
        <textarea class="textarea"
                  :value="promptText"
                  @input="onPrompt"
                  placeholder="auto-filled after upload; editable"></textarea>
        <div class="field-hint">
          zero_shot mode needs this to match the reference audio's actual words.
        </div>
      </div>
    </div>
  `,
});

// ---------------------------------------------------------------------------
// TextPanel — target text to synthesise.
// TextPanel ——要合成的目标文本。
// ---------------------------------------------------------------------------

export const TextPanel = defineComponent({
  name: "TextPanel",
  props: {
    text: { type: String, default: "" },
  },
  emits: ["update:text"],
  setup(props, { emit }) {
    return { onInput: (e) => emit("update:text", e.target.value) };
  },
  template: `
    <div class="card">
      <div class="card-title">Target Text</div>
      <textarea class="textarea textarea-tall"
                :value="text"
                @input="onInput"
                placeholder="e.g. It's great to be back in Beijing, truly fantastic..."></textarea>
    </div>
  `,
});

// ---------------------------------------------------------------------------
// VoiceConfigForm — ElevenLabs-style structured form.
// VoiceConfigForm ——ElevenLabs 风格的结构化表单。
// ---------------------------------------------------------------------------

export const VoiceConfigForm = defineComponent({
  name: "VoiceConfigForm",
  props: {
    config: { type: Object, required: true },
    composed: { type: String, default: "" },
  },
  emits: ["update:config"],
  setup(props, { emit }) {
    const set = (k, v) => emit("update:config", { ...props.config, [k]: v });
    const setRadio = (k, v) => set(k, props.config[k] === v ? null : v);
    return { set, setRadio };
  },
  template: `
    <div>
      <div class="field">
        <label class="field-label">Language</label>
        <select class="select"
                :value="config.language"
                @change="set('language', $event.target.value)">
          <option>English</option>
          <option>Chinese</option>
        </select>
      </div>

      <div class="field">
        <label class="field-label">Gender</label>
        <div class="field-radio-group">
          <button class="radio-btn"
                  :class="{ 'is-active': config.gender === 'male' }"
                  @click="setRadio('gender', 'male')">male</button>
          <button class="radio-btn"
                  :class="{ 'is-active': config.gender === 'female' }"
                  @click="setRadio('gender', 'female')">female</button>
        </div>
      </div>

      <div class="field">
        <label class="field-label">Age</label>
        <div class="field-radio-group">
          <button class="radio-btn"
                  :class="{ 'is-active': config.age === 'young' }"
                  @click="setRadio('age', 'young')">young</button>
          <button class="radio-btn"
                  :class="{ 'is-active': config.age === 'middle' }"
                  @click="setRadio('age', 'middle')">middle</button>
          <button class="radio-btn"
                  :class="{ 'is-active': config.age === 'old' }"
                  @click="setRadio('age', 'old')">old</button>
        </div>
      </div>

      <div class="field">
        <label class="field-label">Quality</label>
        <div class="field-radio-group">
          <button class="radio-btn"
                  :class="{ 'is-active': config.quality === 'studio' }"
                  @click="setRadio('quality', 'studio')">studio</button>
          <button class="radio-btn"
                  :class="{ 'is-active': config.quality === 'broadcast' }"
                  @click="setRadio('quality', 'broadcast')">broadcast</button>
          <button class="radio-btn"
                  :class="{ 'is-active': config.quality === 'casual' }"
                  @click="setRadio('quality', 'casual')">casual</button>
        </div>
      </div>

      <div class="field">
        <label class="field-label">Persona (2–5 words)</label>
        <input class="input"
               :value="config.persona"
               @input="set('persona', $event.target.value)"
               placeholder="confident teacher" />
      </div>

      <div class="field">
        <label class="field-label">Emotion (2–3 adjectives)</label>
        <input class="input"
               :value="config.emotion"
               @input="set('emotion', $event.target.value)"
               placeholder="calm, warm" />
      </div>

      <div class="field">
        <label class="field-label">Description (1–2 sentences on timbre / pacing / delivery)</label>
        <textarea class="textarea"
                  :value="config.description"
                  @input="set('description', $event.target.value)"
                  placeholder="Slow tempo with clear consonants. Slight rising tone at sentence endings."></textarea>
      </div>

      <div class="field">
        <label class="field-label">Composed instruct (live preview)</label>
        <div class="composed-preview" :class="{ 'is-empty': !composed }">{{ composed }}</div>
      </div>
    </div>
  `,
});

// ---------------------------------------------------------------------------
// ConfigPanel — mode toggle + VoiceConfigForm (only for instruct).
// ConfigPanel ——mode 切换 + VoiceConfigForm（仅 instruct 模式）。
// ---------------------------------------------------------------------------

export const ConfigPanel = defineComponent({
  name: "ConfigPanel",
  components: { VoiceConfigForm },
  props: {
    mode: { type: String, default: "zero_shot" },
    config: { type: Object, required: true },
    composed: { type: String, default: "" },
  },
  emits: ["update:mode", "update:config"],
  template: `
    <div class="card">
      <div class="card-title">Voice Config</div>

      <div class="mode-toggle">
        <button class="radio-btn"
                :class="{ 'is-active': mode === 'zero_shot' }"
                @click="$emit('update:mode', 'zero_shot')">zero_shot</button>
        <button class="radio-btn"
                :class="{ 'is-active': mode === 'instruct' }"
                @click="$emit('update:mode', 'instruct')">instruct</button>
      </div>

      <div v-if="mode === 'zero_shot'" class="field-hint">
        zero_shot copies the reference audio's voice + style directly.
        No voice config needed — just pick a good reference.
      </div>

      <VoiceConfigForm v-else
                       :config="config"
                       :composed="composed"
                       @update:config="$emit('update:config', $event)" />
    </div>
  `,
});

// ---------------------------------------------------------------------------
// ActionBar — Synthesize button + status pill.
// ActionBar ——合成按钮 + 状态指示。
// ---------------------------------------------------------------------------

export const ActionBar = defineComponent({
  name: "ActionBar",
  props: {
    status: { type: String, default: "idle" }, // idle / running / ok / error
    statusMessage: { type: String, default: "" },
    canRun: { type: Boolean, default: false },
  },
  emits: ["synthesize"],
  template: `
    <div class="card">
      <div style="display:flex; align-items:center; gap:14px; flex-wrap:wrap;">
        <button class="btn"
                :disabled="!canRun || status === 'running'"
                @click="$emit('synthesize')">
          {{ status === 'running' ? 'Synthesizing...' : 'Synthesize' }}
        </button>
        <span class="status-pill" :class="'status-' + status">{{ status }}</span>
        <span v-if="statusMessage" style="color:var(--fg-muted); font-size:12px;">
          {{ statusMessage }}
        </span>
      </div>
    </div>
  `,
});

// ---------------------------------------------------------------------------
// PlayerPanel — audio playback + meta.
// PlayerPanel ——音频播放 + 元信息。
// ---------------------------------------------------------------------------

export const PlayerPanel = defineComponent({
  name: "PlayerPanel",
  props: {
    syn: { type: Object, default: null }, // {syn_id, audio_url, wall_time_s, mode, composed_instruct}
    evalStatus: { type: String, default: "" }, // "", "running", "done", "error"
    evalScores: { type: Object, default: null }, // EvalScores
  },
  setup() {
    // Quality thresholds for color-coding (based on community benchmarks
    // and our exp_002 baseline observations).
    // 阈值用来给单元格上色；参考 community + exp_002 baseline。
    const tier = (key, val) => {
      if (val === null || val === undefined) return "";
      if (key === "mos") return val >= 4.0 ? "good" : val >= 3.5 ? "ok" : "bad";
      if (key === "wer") return val <= 0.10 ? "good" : val <= 0.25 ? "ok" : "bad";
      if (key === "secs") return val >= 0.85 ? "good" : val >= 0.70 ? "ok" : "bad";
      if (key === "f0") return val <= 30 ? "good" : val <= 60 ? "ok" : "bad";
      return "";
    };
    const fmt = (v, n = 3) => (v === null || v === undefined) ? "—" : v.toFixed(n);
    return { tier, fmt };
  },
  template: `
    <div class="card" v-if="syn">
      <div class="card-title">Player</div>
      <div class="player-row">
        <audio class="player-audio" :src="syn.audio_url" controls preload="auto"></audio>
      </div>
      <div class="meta-card">
        <div><span class="k">syn_id:</span> <span class="v">{{ syn.syn_id }}</span></div>
        <div><span class="k">mode:</span>   <span class="v">{{ syn.mode }}</span></div>
        <div><span class="k">wall:</span>   <span class="v">{{ syn.wall_time_s.toFixed(1) }}s</span></div>
        <div v-if="syn.composed_instruct">
          <span class="k">instruct:</span> <span class="v">{{ syn.composed_instruct }}</span>
        </div>
      </div>

      <div class="eval-card">
        <div class="eval-title">
          Objective Eval
          <span class="status-pill" :class="'status-' + (evalStatus || 'idle')">{{ evalStatus || "idle" }}</span>
        </div>
        <div v-if="evalScores" class="eval-grid">
          <div class="eval-cell" :class="'tier-' + tier('mos', evalScores.mos_nisqa)">
            <div class="eval-k">MOS-NISQA</div>
            <div class="eval-v">{{ fmt(evalScores.mos_nisqa, 2) }}</div>
            <div class="eval-hint">naturalness · ≥4 good</div>
          </div>
          <div class="eval-cell" :class="'tier-' + tier('wer', evalScores.wer)">
            <div class="eval-k">WER</div>
            <div class="eval-v">{{ fmt(evalScores.wer, 3) }}</div>
            <div class="eval-hint">intelligibility · ≤0.10 good</div>
          </div>
          <div class="eval-cell" :class="'tier-' + tier('secs', evalScores.secs)">
            <div class="eval-k">SECS</div>
            <div class="eval-v">{{ fmt(evalScores.secs, 3) }}</div>
            <div class="eval-hint">speaker sim · ≥0.85 good</div>
          </div>
          <div class="eval-cell" :class="'tier-' + tier('f0', evalScores.f0_rmse_hz)">
            <div class="eval-k">F0 RMSE</div>
            <div class="eval-v">{{ fmt(evalScores.f0_rmse_hz, 1) }}<span class="unit"> Hz</span></div>
            <div class="eval-hint">prosody · ≤30 good</div>
          </div>
        </div>
        <div v-else-if="evalStatus === 'running'" class="eval-running">
          evaluating... (~10–40s, runs in background)
        </div>
        <div v-else-if="evalStatus === 'error'" class="eval-error">
          eval failed — check server logs
        </div>
      </div>
    </div>
  `,
});

// ---------------------------------------------------------------------------
// FeedbackPanel — star rating + tag chips + note.
// FeedbackPanel ——星标评分 + 标签 + 备注。
// ---------------------------------------------------------------------------

const FEEDBACK_TAGS = [
  "natural", "robotic", "style_off", "accent_off",
  "pace_off", "muffled", "artifact", "good_match",
];

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

    // Reset when a new syn comes in.
    watch(() => props.synId, () => { rating.value = 0; tags.value = []; note.value = ""; });

    const toggleTag = (t) => {
      if (tags.value.includes(t)) tags.value = tags.value.filter((x) => x !== t);
      else tags.value = [...tags.value, t];
    };

    const submit = () => {
      if (!rating.value) return;
      emit("submit", { syn_id: props.synId, rating: rating.value, tags: tags.value, note: note.value });
    };

    return { rating, tags, note, toggleTag, submit, FEEDBACK_TAGS };
  },
  template: `
    <div class="card" v-if="synId">
      <div class="card-title">Feedback (this is the "reward signal" for listening policy)</div>

      <div class="field">
        <label class="field-label">Rating</label>
        <div class="stars">
          <span v-for="n in 5" :key="n"
                class="star"
                :class="{ 'is-active': n <= rating }"
                @click="rating = n">★</span>
        </div>
        <div class="field-hint">5 = ship-ready · 4 = natural but off-style · 3 = usable with artifacts · 2 = degraded · 1 = unusable</div>
      </div>

      <div class="field">
        <label class="field-label">Tags</label>
        <div class="tag-row">
          <span v-for="t in FEEDBACK_TAGS" :key="t"
                class="tag-chip"
                :class="{ 'is-active': tags.includes(t) }"
                @click="toggleTag(t)">{{ t }}</span>
        </div>
      </div>

      <div class="field">
        <label class="field-label">Note (optional)</label>
        <textarea class="textarea" v-model="note"
                  placeholder="e.g. 'slightly too calm', 'accent leaks in the second sentence'"></textarea>
      </div>

      <div class="feedback-actions">
        <button class="btn" :disabled="!rating" @click="submit">Submit Feedback</button>
        <span v-if="saved" class="feedback-saved">✓ saved</span>
      </div>
    </div>
  `,
});

// ---------------------------------------------------------------------------
// HistoryPanel — list past syntheses with quick replay.
// HistoryPanel ——历史合成列表，可快速重听。
// ---------------------------------------------------------------------------

export const HistoryPanel = defineComponent({
  name: "HistoryPanel",
  props: {
    items: { type: Array, default: () => [] },
  },
  emits: ["replay"],
  setup(_, { emit }) {
    return { replay: (item) => emit("replay", item) };
  },
  template: `
    <div class="card" v-if="items.length">
      <div class="card-title">History</div>
      <div v-for="it in items" :key="it.syn_id" class="history-row">
        <div>
          <div class="history-text">{{ it.text }}</div>
          <div class="history-meta">
            {{ it.timestamp.slice(0,16).replace('T',' ') }} ·
            {{ it.mode }} ·
            ref={{ it.ref_id.slice(0,40) }} ·
            {{ it.wall_time_s.toFixed(1) }}s
          </div>
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
          <span class="history-rating" v-if="it.feedback">
            {{ '★'.repeat(it.feedback.rating) }}{{ '☆'.repeat(5 - it.feedback.rating) }}
          </span>
          <button class="radio-btn" @click="replay(it)">replay</button>
        </div>
      </div>
    </div>
  `,
});
