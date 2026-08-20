/* Voice-RAG frontend. No build step, no framework -- the repo has no JS
   tooling and adding a bundler on submission day risks a build failure at the
   worst possible moment. */

const DEFAULT_BACKEND = "http://localhost:8000";
const MAX_RECORD_MS = 25_000;   // Sarvam's REST endpoint caps at ~30s

const $ = (id) => document.getElementById(id);
let backend = localStorage.getItem("backendUrl") || DEFAULT_BACKEND;
let recorder = null, chunks = [], timerId = null, startedAt = 0, busy = false;

/* ---------------- backend config ---------------- */
$("backendUrl").value = backend;

function setCfgState(txt, cls) {
  const el = $("cfgState");
  el.textContent = txt;
  el.className = "pill " + cls;
}

async function testBackend() {
  setCfgState("checking…", "pill-dim");
  try {
    const r = await fetch(backend.replace(/\/$/, "") + "/health", { mode: "cors" });
    const d = await r.json();
    if (d.status === "ok" || d.status === "degraded") {
      setCfgState(`${d.status} · ${(d.points ?? 0).toLocaleString()} chunks`, "pill-ok");
      return true;
    }
    setCfgState("unreachable", "pill-bad");
  } catch (e) {
    setCfgState("unreachable", "pill-bad");
    $("cfg").open = true;
  }
  return false;
}

$("saveUrl").onclick = async () => {
  backend = $("backendUrl").value.trim().replace(/\/$/, "") || DEFAULT_BACKEND;
  localStorage.setItem("backendUrl", backend);
  if (await testBackend()) $("cfg").open = false;
};

/* ---------------- UI helpers ---------------- */
function showStatus(msg) {
  $("statusText").textContent = msg;
  $("status").classList.remove("hidden");
  $("result").classList.add("hidden");
}
const hideStatus = () => $("status").classList.add("hidden");

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* The three states the demo has to prove visually. Derived from fields the API
   has returned since Day 2: scope.in_scope and groundedness.replaced. */
function classify(d) {
  if (d.scope && d.scope.in_scope === false) {
    return {
      cls: "state-out-of-scope",
      badge: "⚠ Out of scope",
      note: `Blocked by the ${d.scope.stage === "rerank" ? "post-retrieval relevance" : "pre-retrieval scope"} guardrail — generation was never called.`,
    };
  }
  if (d.groundedness && d.groundedness.replaced) {
    return {
      cls: "state-ungrounded",
      badge: "⛔ Insufficient grounding",
      note: "The post-generation guardrail found no retrieved passage supporting the answer, so it was withheld.",
    };
  }
  return { cls: "state-grounded", badge: "✓ Grounded answer", note: "" };
}

function render(d) {
  hideStatus();
  const s = classify(d);
  const g = d.groundedness || {}, sc = d.scope || {}, t = d.timing || {};
  const conf = d.confidence ?? 0, ground = g.score ?? 0;

  let html = `<div class="card ${s.cls}"><div class="badge">${s.badge}</div>`;

  if (d.transcript) {
    const lang = d.detected_language
      ? `<span class="pill pill-lang">${esc(d.detected_language)}</span>`
      : (d.raw_detected_language
          ? `<span class="pill">${esc(d.raw_detected_language)} · not indexed, searched all</span>` : "");
    html += `<div class="transcript"><span class="lbl">Transcribed by Sarvam${lang ? "" : ""}</span>${esc(d.transcript)} ${lang}</div>`;
  }

  html += `<p class="answer">${esc(d.answer)}</p>`;
  if (s.note) html += `<p class="hint">${esc(s.note)}</p>`;

  html += `<div class="metrics">
    <div class="metric"><div class="k">Confidence</div><div class="v">${(conf * 100).toFixed(0)}%</div>
      <div class="bar"><i style="width:${Math.round(conf * 100)}%"></i></div></div>
    <div class="metric"><div class="k">Groundedness</div><div class="v">${(ground * 100).toFixed(0)}%</div>
      <div class="bar"><i style="width:${Math.round(ground * 100)}%"></i></div></div>
    <div class="metric"><div class="k">Citations</div><div class="v">${(d.citations || []).length}</div></div>
    <div class="metric"><div class="k">Total</div><div class="v">${Math.round(t.total_ms || 0)}<span style="font-size:12px"> ms</span></div></div>
  </div>`;

  if (sc.stage && sc.stage !== "disabled") {
    html += `<div class="hint">Guardrail: <code>${esc(sc.stage)}</code> scored
      ${(sc.score ?? 0).toFixed(2)} against threshold ${(sc.threshold ?? 0).toFixed(2)}${
      g.dropped ? ` · ${g.dropped} unsupported sentence(s) removed` : ""}</div>`;
  }

  const ctx = d.contexts || [];
  if (ctx.length) {
    html += `<details class="cites"><summary>${ctx.length} retrieved chunk${ctx.length > 1 ? "s" : ""}${
      (d.citations || []).length ? ` · ${d.citations.length} cited` : ""}</summary>`;
    for (const c of ctx) {
      const cited = (d.citations || []).includes(c.chunk_id);
      html += `<div class="cite"><div class="id">${cited ? "★ " : ""}${esc(c.chunk_id)}${
        c.rerank_score != null ? ` · rerank ${c.rerank_score.toFixed(3)}` : ""}</div>
        <div class="tx">${esc((c.text || "").slice(0, 320))}</div></div>`;
    }
    html += `</details>`;
  }

  const parts = [];
  if (t.stt_ms) parts.push(`stt ${Math.round(t.stt_ms)}`);
  parts.push(`scope ${Math.round(t.scope_ms || 0)}`,
             `retrieval ${Math.round(t.retrieval_ms || 0)}`,
             `gen ${Math.round(t.generation_ms || 0)}`,
             `ground ${Math.round(t.groundedness_ms || 0)}`);
  html += `<div class="timing">${parts.join("  ·  ")} ms${d.provider ? `  ·  via ${esc(d.provider)}` : ""}</div>`;

  html += `</div>`;
  $("result").innerHTML = html;
  $("result").classList.remove("hidden");
}

function renderError(msg) {
  hideStatus();
  $("result").innerHTML =
    `<div class="card state-error"><div class="badge">Request failed</div>
     <p class="answer">${esc(msg)}</p>
     <p class="hint">If this persists, check the Backend URL above — the tunnel URL rotates.</p></div>`;
  $("result").classList.remove("hidden");
}

async function send(path, opts, statusMsg) {
  if (busy) return;
  busy = true;
  showStatus(statusMsg);
  try {
    const r = await fetch(backend.replace(/\/$/, "") + path, { mode: "cors", ...opts });
    let d = null;
    try { d = await r.json(); } catch { /* non-JSON error page */ }
    if (!r.ok) {
      renderError(d && d.detail ? d.detail : `HTTP ${r.status}`);
    } else {
      render(d);
    }
  } catch (e) {
    renderError(`Could not reach the backend (${e.message}). It may be asleep or the URL may have changed.`);
  } finally {
    busy = false;
  }
}

/* ---------------- text ---------------- */
function ask(text) {
  if (!text.trim()) return;
  send("/query", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ text }),
  }, "Retrieving and generating…");
}

$("textForm").onsubmit = (e) => { e.preventDefault(); ask($("textInput").value); };
document.querySelectorAll(".chip").forEach((c) => {
  c.onclick = () => { $("textInput").value = c.dataset.q; ask(c.dataset.q); };
});

/* ---------------- mic ---------------- */
function stopRecording() {
  if (recorder && recorder.state !== "inactive") recorder.stop();
}

$("mic").onclick = async () => {
  if (recorder && recorder.state === "recording") { stopRecording(); return; }

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    // Mic denied or unavailable -- the text box is the documented fallback.
    $("micLabel").textContent = "Microphone unavailable — use the text box below";
    return;
  }

  // webm/opus is what Chrome gives; Safari gives mp4. Both are Sarvam-supported.
  const mime = MediaRecorder.isTypeSupported("audio/webm") ? "audio/webm" : "";
  recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
  chunks = [];
  recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);

  recorder.onstop = () => {
    clearInterval(timerId);
    stream.getTracks().forEach((t) => t.stop());
    $("mic").classList.remove("recording");
    $("micIcon").textContent = "🎙";
    $("micLabel").textContent = "Tap to ask by voice";
    $("micTimer").textContent = "";

    const blob = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
    if (blob.size < 1200) { renderError("Recording was too short to transcribe."); return; }
    const ext = (recorder.mimeType || "").includes("mp4") ? "m4a" : "webm";
    const fd = new FormData();
    fd.append("file", blob, `q.${ext}`);
    send("/query-audio", { method: "POST", body: fd }, "Transcribing, then retrieving…");
  };

  recorder.start();
  startedAt = Date.now();
  $("mic").classList.add("recording");
  $("micIcon").textContent = "⏹";
  $("micLabel").textContent = "Listening… tap to stop";
  timerId = setInterval(() => {
    const s = (Date.now() - startedAt) / 1000;
    $("micTimer").textContent = `${s.toFixed(1)}s / 25s`;
    if (s * 1000 >= MAX_RECORD_MS) stopRecording();   // hard cap: Sarvam's limit
  }, 100);
};

testBackend();
