// app.js — top-level Vue 3 app (redesign v2: 3-zone bento + A2UI DAG).
//
// Layout:
//   <header>              project title + worker status
//   Zone 1 (left)         VoiceSourcePanel — tabs: B站 / YouTube / Upload / Built-in
//   Zone 2 (right)        PipelineDAG + NodeDetailPanel — A2UI viz of latest job
//   Zone 3 (full)         SynthesisPanel — text/config/run/player/scores
//   <bottom-row>          FeedbackPanel + HistoryPanel
//
// All fetch + polling logic from v1 is preserved verbatim. Only the
// template and the components consumed have changed.
//
// app.js ——v2 顶层（三区 bento + A2UI DAG）。
// v1 的 fetch/polling 逻辑原样保留；只换模板和组件。

import {
  PipelineDAG,
  NodeDetailPanel,
  VoiceSourcePanel,
  ReferencePreview,
  SynthesisPanel,
  FeedbackPanel,
  HistoryPanel,
} from "/static/components.js";

const { createApp, reactive, ref, computed, onMounted, watch } = Vue;

// ---------------------------------------------------------------------------
// composeInstructPreview — mirrors api/prompt_compose.py.compose_instruct.
// 与 api/prompt_compose.py 保持一致；让用户实时看到拼出的 instruct 文本。
// ---------------------------------------------------------------------------

function composeInstructPreview(cfg) {
  const parts = [];
  if (cfg.language) parts.push(`Native ${cfg.language}.`);

  const ga = [];
  if (cfg.gender) ga.push(cfg.gender[0].toUpperCase() + cfg.gender.slice(1));
  if (cfg.age) ga.push(`${cfg.age} age range`);
  if (ga.length) parts.push(ga.join(", ") + ".");

  if (cfg.quality) {
    parts.push(`${cfg.quality[0].toUpperCase() + cfg.quality.slice(1)} quality.`);
  }
  if ((cfg.persona || "").trim()) parts.push(`Persona: ${cfg.persona.trim()}.`);
  if ((cfg.emotion || "").trim()) parts.push(`Emotion: ${cfg.emotion.trim()}.`);
  if ((cfg.description || "").trim()) parts.push(cfg.description.trim());

  return parts.join(" ");
}

const App = {
  components: {
    PipelineDAG, NodeDetailPanel, VoiceSourcePanel, ReferencePreview,
    SynthesisPanel, FeedbackPanel, HistoryPanel,
  },
  setup() {
    // -------- core synth state --------
    const refs = ref([]);
    const selectedRefId = ref("");
    const promptText = ref("");
    const text = ref("");
    const mode = ref("zero_shot");
    const config = reactive({
      language: "English", gender: null, age: null, quality: null,
      persona: "", emotion: "", description: "",
    });

    const runStatus = ref("idle");
    const statusMessage = ref("");
    const uploading = ref(false);
    const lastSyn = ref(null);
    const feedbackSaved = ref(false);
    const history = ref([]);
    const evalStatus = ref("");
    const evalScores = ref(null);
    let evalPollTimer = null;

    // -------- bilibili import state --------
    const bilibiliProbing = ref(false);
    const bilibiliProbeResult = ref(null);
    const bilibiliImporting = ref(false);
    const bilibiliJob = ref(null);
    const bilibiliError = ref("");
    let bilibiliPollTimer = null;

    // -------- ingest (dataset) state --------
    const ingesting = ref(false);
    const ingestJob = ref(null);
    const ingestError = ref("");
    let ingestPollTimer = null;

    // -------- DAG node selection (Zone 2) --------
    const selectedDagStage = ref(null);

    // The DAG renders whichever job is most relevant. Priority:
    //   ingest > bilibili (because ingest is the bigger pipeline)
    // 优先级：完整 ingest > B 站单段 import。
    const activeJob = computed(() => {
      if (ingestJob.value && ingestJob.value.stages?.length) {
        return {
          stages: ingestJob.value.stages,
          meta: {
            job_id: ingestJob.value.job_id,
            name: ingestJob.value.name,
            source: ingestJob.value.source,
            status: ingestJob.value.status,
          },
          title: `Ingest · ${ingestJob.value.name} (${ingestJob.value.source})`,
        };
      }
      if (bilibiliJob.value && bilibiliJob.value.stages?.length) {
        return {
          stages: bilibiliJob.value.stages,
          meta: {
            job_id: bilibiliJob.value.job_id,
            name: "(B 站 import)",
            source: "bilibili",
            status: bilibiliJob.value.status,
          },
          title: `B 站 import · ${bilibiliJob.value.progress_hint || bilibiliJob.value.status}`,
        };
      }
      return null;
    });

    // When stages update from polling, refresh the selected stage's pointer
    // so the side panel shows live elapsed/detail.
    // stages 在轮询里被替换；保持选中索引以让 side panel 同步刷新。
    watch(activeJob, (j) => {
      if (!j || selectedDagStage.value == null) return;
      const idx = j.stages.findIndex(s => s.name === selectedDagStage.value.name);
      if (idx >= 0) selectedDagStage.value = j.stages[idx];
    }, { deep: true });

    const composed = computed(() => composeInstructPreview(config));

    const canRun = computed(() => {
      if (!selectedRefId.value || !text.value.trim()) return false;
      if (mode.value === "instruct" && !composed.value) return false;
      return true;
    });

    // -------- ref selection wiring --------
    let promptDirty = false;
    watch(selectedRefId, (id) => {
      const r = refs.value.find((x) => x.ref_id === id);
      if (r && !promptDirty) promptText.value = r.prompt_text || "";
    });
    watch(promptText, () => { promptDirty = true; });

    // -------- API: refs / history --------
    // Built-in search query — debounced from VoiceSourcePanel. Empty = no
    // filter, just first 200 capped by backend.
    // built-in 搜索词；空字符串 = 不过滤，后端默认 cap 200。
    const builtinSearch = ref("");

    async function loadRefs() {
      try {
        const params = new URLSearchParams();
        params.set("limit", "200");
        if (builtinSearch.value) params.set("search", builtinSearch.value);
        const r = await fetch("/api/refs?" + params.toString());
        refs.value = await r.json();
      } catch (e) { console.error("loadRefs failed:", e); }
    }
    function onBuiltinSearch(q) {
      builtinSearch.value = q;
      loadRefs();
    }
    async function loadHistory() {
      try {
        const r = await fetch("/api/history?limit=20");
        history.value = await r.json();
      } catch (e) { console.error("loadHistory failed:", e); }
    }

    // -------- upload --------
    async function onUpload(file) {
      uploading.value = true;
      statusMessage.value = "uploading + transcribing reference audio...";
      try {
        const form = new FormData();
        form.append("file", file);
        const r = await fetch("/api/upload-ref", { method: "POST", body: form });
        if (!r.ok) throw new Error(`upload failed: ${r.status} ${await r.text()}`);
        const result = await r.json();
        await loadRefs();
        promptDirty = false;
        selectedRefId.value = result.ref_id;
        promptText.value = result.prompt_text;
        statusMessage.value = `uploaded; ASR lang=${result.asr_lang}`;
      } catch (e) {
        console.error(e);
        statusMessage.value = `upload error: ${e.message}`;
      } finally {
        uploading.value = false;
      }
    }

    // -------- bilibili probe + import (unchanged from v1) --------
    function stopBilibiliPoll() {
      if (bilibiliPollTimer) { clearTimeout(bilibiliPollTimer); bilibiliPollTimer = null; }
    }
    async function onBilibiliProbe(url) {
      bilibiliError.value = "";
      bilibiliProbeResult.value = null;
      bilibiliJob.value = null;
      stopBilibiliPoll();
      bilibiliProbing.value = true;
      try {
        const r = await fetch("/api/bilibili/probe", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ url }),
        });
        if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
        bilibiliProbeResult.value = await r.json();
      } catch (e) {
        bilibiliError.value = `probe failed: ${e.message}`;
      } finally {
        bilibiliProbing.value = false;
      }
    }
    async function pollBilibiliJob(jobId, attempt = 0) {
      if (attempt > 90) {
        bilibiliError.value = "import timed out after ~3 min";
        bilibiliImporting.value = false;
        return;
      }
      try {
        const r = await fetch(`/api/bilibili/import/${jobId}`);
        if (!r.ok) throw new Error(`${r.status}`);
        const data = await r.json();
        bilibiliJob.value = data;
        if (data.status === "ready") {
          bilibiliImporting.value = false;
          await loadRefs();
          promptDirty = false;
          selectedRefId.value = data.ref_id;
          promptText.value = data.prompt_text || "";
          statusMessage.value = `imported from Bilibili (prompt_source=${data.prompt_source})`;
          return;
        }
        if (data.status === "error") {
          bilibiliImporting.value = false;
          bilibiliError.value = `import failed: ${data.error || "unknown error"}`;
          return;
        }
      } catch (e) { console.warn("bilibili poll error:", e.message); }
      bilibiliPollTimer = setTimeout(() => pollBilibiliJob(jobId, attempt + 1), 2000);
    }
    async function onBilibiliImport(req) {
      bilibiliError.value = "";
      bilibiliJob.value = null;
      stopBilibiliPoll();
      bilibiliImporting.value = true;
      try {
        const r = await fetch("/api/bilibili/import", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify(req),
        });
        if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
        const job = await r.json();
        bilibiliJob.value = job;
        pollBilibiliJob(job.job_id);
      } catch (e) {
        bilibiliError.value = `import failed: ${e.message}`;
        bilibiliImporting.value = false;
      }
    }

    // -------- dataset ingest (used for YouTube too) --------
    function stopIngestPoll() {
      if (ingestPollTimer) { clearTimeout(ingestPollTimer); ingestPollTimer = null; }
    }
    async function pollIngestJob(jobId, attempt = 0) {
      if (attempt > 900) {
        ingestError.value = "ingest timed out after ~30 min";
        ingesting.value = false;
        return;
      }
      try {
        const r = await fetch(`/api/dataset/ingest/${jobId}`);
        if (!r.ok) throw new Error(`${r.status}`);
        const data = await r.json();
        ingestJob.value = data;
        if (data.status === "done") {
          ingesting.value = false;
          statusMessage.value =
            `ingest done: ${data.n_chunks} chunks @ ${data.total_duration_s?.toFixed(1)}s → ${data.manifest_path}`;
          return;
        }
        if (data.status === "error") {
          ingesting.value = false;
          ingestError.value = `ingest failed: ${data.error || "unknown error"}`;
          return;
        }
      } catch (e) { console.warn("ingest poll error:", e.message); }
      ingestPollTimer = setTimeout(() => pollIngestJob(jobId, attempt + 1), 2000);
    }
    async function onIngestRun(req) {
      ingestError.value = "";
      ingestJob.value = null;
      stopIngestPoll();
      ingesting.value = true;
      try {
        const r = await fetch("/api/dataset/ingest", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify(req),
        });
        if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
        const job = await r.json();
        ingestJob.value = job;
        pollIngestJob(job.job_id);
      } catch (e) {
        ingestError.value = `ingest failed: ${e.message}`;
        ingesting.value = false;
      }
    }
    // YouTube tab → ingest endpoint with auto-generated dataset name.
    // YouTube tab 复用完整 ingest 端点；自动生成数据集名。
    function onYoutubeExtract(payload) {
      const name = "yt_" + Date.now().toString(36);
      onIngestRun({
        source: "youtube",
        name,
        url: payload.url,
        start_sec: payload.start_sec,
        end_sec: payload.end_sec,
        is_single_speaker: true,
        needs_separation: true,
      });
    }

    // -------- DAG node click --------
    function onDagNodeSelect(payload) {
      selectedDagStage.value = payload.stage;
    }

    // -------- synthesize (unchanged) --------
    function stopEvalPoll() {
      if (evalPollTimer) { clearTimeout(evalPollTimer); evalPollTimer = null; }
    }
    async function pollEval(synId, attempt = 0) {
      if (attempt > 90 || !lastSyn.value || lastSyn.value.syn_id !== synId) return;
      try {
        const r = await fetch(`/api/eval/${synId}`);
        if (!r.ok) throw new Error(`${r.status}`);
        const data = await r.json();
        evalStatus.value = data.status;
        if (data.status === "done") { evalScores.value = data.scores; loadHistory(); return; }
        if (data.status === "error") { evalScores.value = null; return; }
      } catch (e) { console.warn("eval poll error:", e.message); }
      evalPollTimer = setTimeout(() => pollEval(synId, attempt + 1), 2000);
    }

    async function onSynthesize() {
      runStatus.value = "running";
      statusMessage.value = "synthesising (first request takes ~60s)...";
      feedbackSaved.value = false;
      stopEvalPoll();
      evalStatus.value = ""; evalScores.value = null;
      try {
        const body = { text: text.value, ref_id: selectedRefId.value, mode: mode.value };
        if (mode.value === "instruct") body.voice_config = { ...config };
        const r = await fetch("/api/synthesize", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
        lastSyn.value = await r.json();
        runStatus.value = "ok";
        statusMessage.value = `done in ${lastSyn.value.wall_time_s.toFixed(1)}s`;
        loadHistory();
        evalStatus.value = "running";
        pollEval(lastSyn.value.syn_id);
      } catch (e) {
        runStatus.value = "error";
        statusMessage.value = `error: ${e.message}`;
      }
    }

    async function onFeedback(entry) {
      try {
        const r = await fetch("/api/feedback", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify(entry),
        });
        if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
        feedbackSaved.value = true;
        loadHistory();
      } catch (e) { statusMessage.value = `feedback save failed: ${e.message}`; }
    }

    function onReplay(item) {
      stopEvalPoll();
      lastSyn.value = {
        syn_id: item.syn_id, audio_url: item.audio_url,
        wall_time_s: item.wall_time_s, mode: item.mode,
        composed_instruct: item.composed_instruct,
      };
      feedbackSaved.value = !!item.feedback;
      if (item.eval) { evalStatus.value = "done"; evalScores.value = item.eval; }
      else { evalStatus.value = ""; evalScores.value = null; }
    }

    onMounted(() => { loadRefs(); loadHistory(); });

    // -------- config field mutation (single source of reactivity truth) --------
    // Children must NOT use v-model:config (would try to reassign a reactive
    // const). They emit `set-config-field` instead; we mutate in place.
    // 子组件用 set-config-field 触发我们就地改字段——绝对不要 v-model:config。
    function onSetConfigField({ key, value }) { config[key] = value; }

    return {
      // state
      refs, selectedRefId, promptText, text, mode, config,
      runStatus, statusMessage, uploading,
      lastSyn, feedbackSaved, history,
      evalStatus, evalScores,
      bilibiliProbing, bilibiliProbeResult, bilibiliImporting,
      bilibiliJob, bilibiliError,
      ingesting, ingestJob, ingestError,
      activeJob, selectedDagStage,
      // computed
      composed, canRun,
      // handlers
      onUpload, onSynthesize, onFeedback, onReplay,
      onBilibiliProbe, onBilibiliImport,
      onYoutubeExtract, onDagNodeSelect,
      onBuiltinSearch, onSetConfigField,
    };
  },

  template: `
    <div>
      <header class="app-header">
        <h1>
          <span class="accent">▸</span> voice_story
          <span class="sub">TTS · v0.1.4 · dark</span>
        </h1>
        <div class="status">
          TTS worker: <span class="ok">ready</span>
        </div>
      </header>

      <div v-if="statusMessage" class="app-sub">{{ statusMessage }}</div>

      <main class="zones">
        <!-- Zone 1 -->
        <VoiceSourcePanel
          :refs="refs"
          v-model:selectedRefId="selectedRefId"
          :bilibiliProbing="bilibiliProbing"
          :bilibiliProbeResult="bilibiliProbeResult"
          :bilibiliImporting="bilibiliImporting"
          :bilibiliError="bilibiliError"
          :uploading="uploading"
          @upload="onUpload"
          @bilibili-probe="onBilibiliProbe"
          @bilibili-import="onBilibiliImport"
          @youtube-extract="onYoutubeExtract"
          @builtin-search="onBuiltinSearch" />

        <!-- Zone 2 -->
        <section class="zone">
          <h3 class="zone-title">Zone 2 · Pipeline Graph (A2UI)</h3>
          <div class="dag-wrap">
            <div v-if="activeJob">
              <PipelineDAG
                :title="activeJob.title"
                :stages="activeJob.stages"
                @node-select="onDagNodeSelect" />
            </div>
            <div v-else class="cy-canvas"
                 style="display:flex; align-items:center; justify-content:center; color: var(--fg-muted);
                        font-family: var(--font-mono); font-size: 12px; text-align:center; padding: 20px;">
              No active pipeline. Trigger an import or ingest<br/>from Zone 1 to see the agent graph here.
            </div>
            <NodeDetailPanel
              :stage="selectedDagStage"
              :jobMeta="activeJob ? activeJob.meta : null" />
          </div>
        </section>

        <!-- Zone 3 -->
        <SynthesisPanel
          v-model:text="text"
          v-model:mode="mode"
          :config="config"
          :composed="composed"
          :selectedRefId="selectedRefId"
          :runStatus="runStatus"
          :statusMessage="statusMessage"
          :lastSyn="lastSyn"
          :evalStatus="evalStatus"
          :evalScores="evalScores"
          @set-config-field="onSetConfigField"
          @synthesize="onSynthesize" />
      </main>

      <!-- Bottom row: feedback + history -->
      <div class="bottom-row">
        <FeedbackPanel
          :synId="lastSyn ? lastSyn.syn_id : ''"
          :saved="feedbackSaved"
          @submit="onFeedback" />

        <HistoryPanel
          :items="history"
          @replay="onReplay" />
      </div>
    </div>
  `,
};

createApp(App).mount("#app");
