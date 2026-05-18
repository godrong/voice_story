// app.js — top-level Vue 3 app for the TTS playground.
// Wires components, owns the reactive store, calls the FastAPI backend.
//
// app.js ——TTS playground 的顶层 Vue 3 应用。
// 把组件粘合起来，持有响应式 store，调 FastAPI 后端。

import {
  RefPanel,
  TextPanel,
  ConfigPanel,
  ActionBar,
  PlayerPanel,
  FeedbackPanel,
  HistoryPanel,
  PipelineCard,
} from "/static/components.js";

const { createApp, reactive, ref, computed, onMounted, watch } = Vue;

// ---------------------------------------------------------------------------
// composeInstructPreview — mirrors api/prompt_compose.py.compose_instruct.
// composeInstructPreview ——和 api/prompt_compose.py.compose_instruct
// 保持完全一致；放在前端让用户实时看到拼出来的指令。
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

// ---------------------------------------------------------------------------
// App — single root component.
// App ——单根组件。
// ---------------------------------------------------------------------------

const App = {
  components: { RefPanel, TextPanel, ConfigPanel, ActionBar, PlayerPanel, FeedbackPanel, HistoryPanel, PipelineCard },
  setup() {
    const refs = ref([]);
    const selectedRefId = ref("");
    const promptText = ref("");
    const text = ref("");
    const mode = ref("zero_shot");
    const config = reactive({
      language: "English",
      gender: null,
      age: null,
      quality: null,
      persona: "",
      emotion: "",
      description: "",
    });

    const status = ref("idle"); // idle / running / ok / error
    const statusMessage = ref("");
    const uploading = ref(false);
    const lastSyn = ref(null);
    const feedbackSaved = ref(false);
    const history = ref([]);
    const evalStatus = ref(""); // "", "running", "done", "error"
    const evalScores = ref(null);
    let evalPollTimer = null;

    // Bilibili import state. All backend chatter lives here; RefPanel just
    // emits intent events and renders these.
    // B 站导入相关状态，集中在这里——所有 fetch 都从这里发，RefPanel 只 emit 意图。
    const bilibiliProbing = ref(false);
    const bilibiliProbeResult = ref(null);
    const bilibiliImporting = ref(false);
    const bilibiliJob = ref(null);
    const bilibiliError = ref("");
    let bilibiliPollTimer = null;

    const composed = computed(() => composeInstructPreview(config));

    // Derived TTS pipeline stages — synth is sync so it's always done by the
    // time we have lastSyn; eval mirrors evalStatus + evalScores.
    // 派生的 TTS pipeline stages —— synth 是同步调用所以拿到 lastSyn 时
    // 一定 done；eval 跟 evalStatus / evalScores 同步。
    const ttsStages = computed(() => {
      if (!lastSyn.value) return [];
      const synStage = {
        name: "synthesize",
        status: "done",
        elapsed_s: lastSyn.value.wall_time_s,
        detail: `mode=${lastSyn.value.mode} · syn_id=${lastSyn.value.syn_id.slice(0, 24)}...`,
      };
      const es = evalStatus.value;
      const sc = evalScores.value;
      const evalDetail = sc
        ? `NISQA=${(sc.mos_nisqa ?? 0).toFixed(2)} · WER=${(sc.wer ?? 0).toFixed(2)}` +
          ` · SECS=${(sc.secs ?? 0).toFixed(2)}` +
          (sc.f0_rmse_hz != null ? ` · F0=${sc.f0_rmse_hz.toFixed(1)}Hz` : "")
        : (es === "running" ? "WER/CER + neural MOS + SECS + F0 在跑..." : "");
      const evalStage = {
        name: "evaluate",
        status: es === "done" ? "done"
              : es === "error" ? "error"
              : es === "running" ? "running"
              : "pending",
        elapsed_s: sc?.eval_time_s,
        detail: evalDetail,
      };
      return [synStage, evalStage];
    });

    const canRun = computed(() => {
      if (!selectedRefId.value || !text.value.trim()) return false;
      if (mode.value === "instruct" && !composed.value) return false;
      return true;
    });

    // Keep prompt_text in sync with the selected ref unless the user
    // has manually edited it.
    // 选择 ref 时同步它的 prompt_text；用户手动改过的不覆盖。
    let promptDirty = false;
    watch(selectedRefId, (id) => {
      const r = refs.value.find((x) => x.ref_id === id);
      if (r && !promptDirty) promptText.value = r.prompt_text || "";
    });
    watch(promptText, () => { promptDirty = true; });

    async function loadRefs() {
      try {
        const r = await fetch("/api/refs");
        refs.value = await r.json();
      } catch (e) {
        console.error("loadRefs failed:", e);
      }
    }

    async function loadHistory() {
      try {
        const r = await fetch("/api/history?limit=20");
        history.value = await r.json();
      } catch (e) {
        console.error("loadHistory failed:", e);
      }
    }

    async function onUpload(file) {
      uploading.value = true;
      statusMessage.value = "uploading + transcribing reference audio...";
      try {
        const form = new FormData();
        form.append("file", file);
        // Don't pass prompt_text — let the server run ASR.
        // 不传 prompt_text，让后端跑 ASR。
        const r = await fetch("/api/upload-ref", { method: "POST", body: form });
        if (!r.ok) throw new Error(`upload failed: ${r.status} ${await r.text()}`);
        const result = await r.json();
        await loadRefs();
        promptDirty = false;
        selectedRefId.value = result.ref_id;
        promptText.value = result.prompt_text;
        statusMessage.value = `uploaded; ASR detected lang=${result.asr_lang}`;
      } catch (e) {
        console.error(e);
        statusMessage.value = `upload error: ${e.message}`;
      } finally {
        uploading.value = false;
      }
    }

    function stopBilibiliPoll() {
      if (bilibiliPollTimer) {
        clearTimeout(bilibiliPollTimer);
        bilibiliPollTimer = null;
      }
    }

    async function onBilibiliProbe(url) {
      bilibiliError.value = "";
      bilibiliProbeResult.value = null;
      bilibiliJob.value = null;
      stopBilibiliPoll();
      bilibiliProbing.value = true;
      try {
        const r = await fetch("/api/bilibili/probe", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ url }),
        });
        if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
        bilibiliProbeResult.value = await r.json();
      } catch (e) {
        console.error("bilibili probe failed:", e);
        bilibiliError.value = `probe failed: ${e.message}`;
      } finally {
        bilibiliProbing.value = false;
      }
    }

    async function pollBilibiliJob(jobId, attempt = 0) {
      // Cap polling at ~3 min (90 × 2s). Import normally finishes in 10-60s.
      // 总计最多 ~3 分钟；正常 10-60s 内能拿到 ready。
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
          // Refresh refs list and auto-select the new one.
          // 刷新 refs 列表并自动选中新导入的那条。
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
      } catch (e) {
        console.warn("bilibili poll error:", e.message);
      }
      bilibiliPollTimer = setTimeout(() => pollBilibiliJob(jobId, attempt + 1), 2000);
    }

    async function onBilibiliImport(req) {
      bilibiliError.value = "";
      bilibiliJob.value = null;
      stopBilibiliPoll();
      bilibiliImporting.value = true;
      try {
        const r = await fetch("/api/bilibili/import", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(req),
        });
        if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
        const job = await r.json();
        bilibiliJob.value = job;
        pollBilibiliJob(job.job_id);
      } catch (e) {
        console.error("bilibili import failed:", e);
        bilibiliError.value = `import failed: ${e.message}`;
        bilibiliImporting.value = false;
      }
    }

    function onBilibiliReset() {
      stopBilibiliPoll();
      bilibiliProbeResult.value = null;
      bilibiliJob.value = null;
      bilibiliError.value = "";
      bilibiliImporting.value = false;
      bilibiliProbing.value = false;
    }

    function stopEvalPoll() {
      if (evalPollTimer) {
        clearTimeout(evalPollTimer);
        evalPollTimer = null;
      }
    }

    async function pollEval(synId, attempt = 0) {
      // Cap polling at ~3 min (~90 × 2s). Eval normally finishes in 10-40s.
      // 总计最多 ~3 分钟（90 × 2s）；正常 eval 应在 10-40s 内出。
      if (attempt > 90 || !lastSyn.value || lastSyn.value.syn_id !== synId) return;
      try {
        const r = await fetch(`/api/eval/${synId}`);
        if (!r.ok) throw new Error(`${r.status}`);
        const data = await r.json();
        evalStatus.value = data.status;
        if (data.status === "done") {
          evalScores.value = data.scores;
          // Refresh history so the eval shows up there too.
          // 顺手刷新 history。
          loadHistory();
          return;
        }
        if (data.status === "error") {
          evalScores.value = null;
          return;
        }
      } catch (e) {
        console.warn("eval poll error:", e.message);
      }
      evalPollTimer = setTimeout(() => pollEval(synId, attempt + 1), 2000);
    }

    async function onSynthesize() {
      status.value = "running";
      statusMessage.value = "synthesising (first request takes ~60s)...";
      feedbackSaved.value = false;
      stopEvalPoll();
      evalStatus.value = "";
      evalScores.value = null;
      try {
        const body = {
          text: text.value,
          ref_id: selectedRefId.value,
          mode: mode.value,
        };
        if (mode.value === "instruct") {
          body.voice_config = { ...config };
        }
        const r = await fetch("/api/synthesize", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
        lastSyn.value = await r.json();
        status.value = "ok";
        statusMessage.value = `done in ${lastSyn.value.wall_time_s.toFixed(1)}s`;
        loadHistory();
        // Start polling for objective eval (server kicks off automatically).
        // 立即开始轮询客观 eval（服务端已自动后台跑）。
        evalStatus.value = "running";
        pollEval(lastSyn.value.syn_id);
      } catch (e) {
        console.error(e);
        status.value = "error";
        statusMessage.value = `error: ${e.message}`;
      }
    }

    async function onFeedback(entry) {
      try {
        const r = await fetch("/api/feedback", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(entry),
        });
        if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
        feedbackSaved.value = true;
        loadHistory();
      } catch (e) {
        console.error(e);
        statusMessage.value = `feedback save failed: ${e.message}`;
      }
    }

    function onReplay(item) {
      stopEvalPoll();
      lastSyn.value = {
        syn_id: item.syn_id,
        audio_url: item.audio_url,
        wall_time_s: item.wall_time_s,
        mode: item.mode,
        composed_instruct: item.composed_instruct,
      };
      feedbackSaved.value = !!item.feedback;
      // History items carry their own eval block when available.
      // history 项自带 eval block（如果有）。
      if (item.eval) {
        evalStatus.value = "done";
        evalScores.value = item.eval;
      } else {
        evalStatus.value = "";
        evalScores.value = null;
      }
    }

    onMounted(() => {
      loadRefs();
      loadHistory();
    });

    return {
      refs, selectedRefId, promptText, text, mode, config,
      status, statusMessage, uploading, lastSyn, feedbackSaved, history,
      evalStatus, evalScores,
      bilibiliProbing, bilibiliProbeResult, bilibiliImporting,
      bilibiliJob, bilibiliError,
      ttsStages,
      composed, canRun,
      onUpload, onSynthesize, onFeedback, onReplay,
      onBilibiliProbe, onBilibiliImport, onBilibiliReset,
    };
  },

  template: `
    <div>
      <div class="app-title">voice_story — TTS Playground</div>
      <div class="app-sub">
        Upload reference audio → describe the voice → synthesise → rate.
        Ratings feed <code>outputs/webui/feedback.jsonl</code>, the data source for the listening policy.
      </div>

      <div class="grid-2">
        <RefPanel
          :refs="refs"
          v-model:selectedRefId="selectedRefId"
          v-model:promptText="promptText"
          :uploading="uploading"
          :bilibiliProbing="bilibiliProbing"
          :bilibiliProbeResult="bilibiliProbeResult"
          :bilibiliImporting="bilibiliImporting"
          :bilibiliJob="bilibiliJob"
          :bilibiliError="bilibiliError"
          @upload="onUpload"
          @bilibili-probe="onBilibiliProbe"
          @bilibili-import="onBilibiliImport"
          @bilibili-reset="onBilibiliReset" />

        <TextPanel v-model:text="text" />
      </div>

      <ConfigPanel
        v-model:mode="mode"
        v-model:config="config"
        :composed="composed" />

      <ActionBar
        :status="status"
        :statusMessage="statusMessage"
        :canRun="canRun"
        @synthesize="onSynthesize" />

      <PlayerPanel :syn="lastSyn" :evalStatus="evalStatus" :evalScores="evalScores" />

      <PipelineCard
        v-if="ttsStages.length"
        title="TTS pipeline"
        :stages="ttsStages" />

      <FeedbackPanel
        :synId="lastSyn ? lastSyn.syn_id : ''"
        :saved="feedbackSaved"
        @submit="onFeedback" />

      <HistoryPanel
        :items="history"
        @replay="onReplay" />
    </div>
  `,
};

createApp(App).mount("#app");
