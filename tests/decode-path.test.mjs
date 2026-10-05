// sc-24139: weight-format gating, the CUDA-graph control, and the decode-path status view, all
// driven by the runtime's own reports in `engine_status`.
import assert from "node:assert/strict";
import test from "node:test";
import React from "react";
import ReactDOMServer from "react-dom/server";
import { readFile } from "node:fs/promises";
import {
  applyDecodeEvent,
  cudaGraphsControl,
  decodePathRows,
  dismissSpeculativeNotice,
  effectiveCudaGraphs,
  enableSpeculativeAuto,
  graphsReloadPending,
  LAST_GENERATION,
  lossyWeightsBadge,
  NVFP4_LOSSY_NOTE,
  selectedWeightFormat,
  serveAction,
  speculativeNotice,
  weightFormatOptions,
} from "../src/state/decodePath.js";
import { DecodePathStatus, SpeculativeNotice } from "../src/components/DecodePathStatus.js";
import { modelSubtitle } from "../src/state/models.js";

const SM120 = {
  backend: "candle-cuda",
  device: "cuda:0",
  compute_capability: "sm_120",
  nvfp4: { supported: true, reason: null },
  cuda_graphs: { supported: true, reason: null },
};
const SM89 = {
  ...SM120,
  compute_capability: "sm_89",
  nvfp4: {
    supported: false,
    reason: "nvfp4: NVFP4 projections need compute capability >= sm_120 (cuBLASLt block-scaled FP4 GEMM); this GPU is sm_89",
  },
};
const CPU = {
  backend: "candle-cpu",
  device: "cpu",
  compute_capability: null,
  nvfp4: { supported: false, reason: "nvfp4: NVFP4 projections need a CUDA device with compute capability >= sm_120; the load device is Cpu" },
  cuda_graphs: { supported: false, reason: "cuda_graphs: cuda_feature_off: CUDA graphs need a CUDA load device; this runtime loads on cpu" },
};
const MLX = {
  backend: "mlx",
  device: "metal",
  compute_capability: null,
  nvfp4: { supported: false, reason: "nvfp4: NVFP4 weights need the Candle CUDA backend on a compute capability >= sm_120 GPU; this runtime is mlx" },
  cuda_graphs: { supported: false, reason: "cuda_graphs: CUDA graphs need the Candle CUDA backend; this runtime is mlx" },
};

const DECODE = {
  path: "mtp",
  proposer: "mtp",
  draft_tokens: 3,
  sampler: "device",
  kv_cache: "static",
  attention: "gqa",
  cuda_graphs: { enabled: true, path: "eager", replayed: 0, eager: 12, captured: 0, fallback_reason: "deltanet_state_unstable" },
  nvfp4_projections: { path: "mixed", reason: "rows" },
  fused_primitives: { path: "fused", reason: null },
  target_forwards: 5,
  proposed_tokens: 9,
  accepted_tokens: 6,
  replay_forwards: 1,
};

function status(capabilities, loaded) {
  return { execution_backend: capabilities.backend, backend_capabilities: capabilities, loaded, providers: [] };
}

test("NVFP4 is selectable on sm_120 and disabled with the runtime's reason elsewhere", () => {
  const blackwell = weightFormatOptions(SM120).find((option) => option.id === "nvfp4");
  assert.equal(blackwell.disabled, false);
  assert.equal(blackwell.reason, null);
  assert.equal(blackwell.value, "nvfp4");
  for (const caps of [SM89, CPU, MLX]) {
    const option = weightFormatOptions(caps).find((item) => item.id === "nvfp4");
    assert.equal(option.disabled, true, caps.backend);
    assert.equal(option.reason, caps.nvfp4.reason);
  }
  // A runtime that reported nothing still gets a reason, never a silently enabled control.
  const unknown = weightFormatOptions(undefined).find((option) => option.id === "nvfp4");
  assert.equal(unknown.disabled, true);
  assert.match(unknown.reason, /did not report/);
  // The other formats are never gated.
  for (const option of weightFormatOptions(CPU).filter((item) => item.id !== "nvfp4")) {
    assert.equal(option.disabled, false, option.id);
  }
});

test("the weight format submitted is bf16 | Q8 | NVFP4 and never a disabled choice", () => {
  assert.equal(selectedWeightFormat(weightFormatOptions(SM120), "nvfp4").value, "nvfp4");
  assert.equal(selectedWeightFormat(weightFormatOptions(SM120), "q8").value, "q8");
  assert.equal(selectedWeightFormat(weightFormatOptions(SM120), "dense").value, null);
  assert.equal(selectedWeightFormat(weightFormatOptions(SM89), "nvfp4").value, null,
    "a stale NVFP4 selection on an unsupported device falls back to bf16");
  // main (sc-23935) labels a load-time format after its source: "Safetensors · Q8 load".
  assert.equal(modelSubtitle({ quantize: "nvfp4" }), "Safetensors · NVFP4 load (lossy)");
});

test("the CUDA-graph toggle is gated on the runtime and flags a pending reload", () => {
  assert.deepEqual(
    { disabled: cudaGraphsControl(CPU, false, null).disabled, reason: cudaGraphsControl(CPU, false, null).reason },
    { disabled: true, reason: CPU.cuda_graphs.reason },
  );
  assert.equal(cudaGraphsControl(MLX, true, null).disabled, true);
  const idle = cudaGraphsControl(SM120, true, null);
  assert.equal(idle.disabled, false);
  assert.equal(idle.pendingReload, false);
  assert.match(idle.note, /Applies when a model is loaded/);
  const stale = cudaGraphsControl(SM120, true, { cuda_graphs: false });
  assert.equal(stale.pendingReload, true);
  assert.match(stale.note, /loaded with CUDA graphs off; reload it/);
  assert.equal(cudaGraphsControl(SM120, false, { cuda_graphs: false }).pendingReload, false);
  // A model loaded where the switch is unavailable (cuda_graphs: null) never asks for a reload.
  assert.equal(cudaGraphsControl(CPU, true, { cuda_graphs: null }).pendingReload, false);
});

test("the status rows name the proposer, graph fallback, NVFP4 path, and sampler", () => {
  const loaded = {
    name: "Qwen3.8-27B NVFP4",
    quantize: "nvfp4",
    cuda_graphs: true,
    load_report: { requested: "nvfp4", projections: [{ kind: "nvfp4", count: 448, params: 1, resident_bytes: 1 }, { kind: "dense", count: 2, params: 1, resident_bytes: 1 }] },
    last_decode: DECODE,
  };
  const rows = Object.fromEntries(decodePathRows(status(SM120, loaded)).map((row) => [row.key, row]));
  assert.equal(rows.backend.value, "candle-cuda · cuda:0 · sm_120");
  assert.equal(rows.weights.value, "NVFP4 (lossy)");
  assert.equal(rows.weights.detail, `Resident projections: nvfp4 × 448, dense × 2 · ${NVFP4_LOSSY_NOTE}`);
  assert.equal(rows.proposer.value, "mtp · 3 drafts");
  assert.equal(rows.proposer.detail, "Accepted 6 of 9 drafts in 5 forwards");
  assert.equal(rows.cuda_graphs.value, "eager");
  assert.equal(rows.cuda_graphs.detail, "0 captured · fallback: deltanet_state_unstable");
  assert.equal(rows.nvfp4.value, "mixed");
  assert.equal(rows.nvfp4.detail, "fallback: rows");
  assert.equal(rows.sampler.value, "device");
  assert.equal(rows.kv_cache.value, "static · gqa attention");
});

test("an Auto request without an MTP head, graphs off, and a host sampler read as such", () => {
  const decode = {
    ...DECODE,
    path: "reference",
    proposer: "none",
    draft_tokens: null,
    proposed_tokens: 0,
    accepted_tokens: 0,
    sampler: "host:penalty",
    cuda_graphs: { enabled: false, path: "none", replayed: 0, eager: 0, captured: 0, fallback_reason: null },
    nvfp4_projections: { path: "none", reason: null },
  };
  const rows = Object.fromEntries(
    decodePathRows(status(SM120, { quantize: null, cuda_graphs: false, load_report: null, last_decode: decode }))
      .map((row) => [row.key, row]),
  );
  assert.equal(rows.proposer.value, "none (token-at-a-time)");
  assert.equal(rows.cuda_graphs.value, "off");
  assert.equal(rows.nvfp4.value, "none (no NVFP4 weights)");
  assert.equal(rows.sampler.value, "host:penalty");
  assert.equal(rows.weights.value, "checkpoint encoding");
  // Graphs on but never reached by the path: said so, not shown as "off".
  const unused = decodePathRows(status(SM120, {
    quantize: null, cuda_graphs: true, load_report: null,
    last_decode: { ...decode, cuda_graphs: { ...decode.cuda_graphs, enabled: true } },
  })).find((row) => row.key === "cuda_graphs");
  assert.equal(unused.value, "on, not used by this decode path");
});

test("before a generation the view says the path is not measured yet", () => {
  const rows = decodePathRows(status(CPU, {
    quantize: "q8", cuda_graphs: null, load_report: null, last_decode: null, decode_reported: null,
  }));
  assert.deepEqual(rows.map((row) => row.key), ["backend", "weights", "graph_switch", "decode"]);
  assert.equal(rows[3].value, "not measured yet");
  // It promises nothing a runtime may never deliver.
  assert.doesNotMatch(rows[3].detail, /after the next generation/);
  assert.match(rows[3].detail, /where the runtime reports it/);
  assert.deepEqual(decodePathRows(status(MLX, null)).map((row) => row.key), ["backend"]);
});

test("a finished generation the runtime did not report on reads 'not reported', not 'not measured yet'", () => {
  const rows = decodePathRows(status(MLX, {
    quantize: null, cuda_graphs: null, load_report: null, last_decode: null, decode_reported: false,
  }));
  const decode = rows.find((row) => row.key === "decode");
  assert.equal(decode.value, "not reported");
  assert.match(decode.detail, /does not report its decode path/);
  assert.equal(decode.section, LAST_GENERATION);
});

test("the decode-path status view renders every reported path", () => {
  const html = ReactDOMServer.renderToStaticMarkup(React.createElement(DecodePathStatus, {
    engineStatus: status(SM120, {
      quantize: "nvfp4",
      cuda_graphs: true,
      load_report: { requested: "nvfp4", projections: [{ kind: "nvfp4", count: 448, params: 1, resident_bytes: 1 }] },
      last_decode: DECODE,
    }),
  }));
  for (const text of [
    "Decode path",
    "Proposer",
    "mtp · 3 drafts",
    "CUDA graphs",
    "fallback: deltanet_state_unstable",
    "NVFP4 projections",
    "mixed",
    "Sampler",
    "device",
    "sm_120",
    "nvfp4 × 448",
  ]) {
    assert.ok(html.includes(text), `missing ${text} in ${html}`);
  }
  assert.ok(html.includes('data-row="cuda_graphs"'));
});

test("NVFP4 is labelled lossy in the picker itself, with the measured perplexity cost visible", () => {
  for (const caps of [SM120, SM89, CPU]) {
    const nvfp4 = weightFormatOptions(caps).find((option) => option.id === "nvfp4");
    assert.match(nvfp4.label, /lossy/i, "the label, not a tooltip, says lossy");
    assert.equal(nvfp4.note, NVFP4_LOSSY_NOTE);
  }
  assert.ok(NVFP4_LOSSY_NOTE.includes("+7.37% perplexity"), NVFP4_LOSSY_NOTE);
  assert.ok(NVFP4_LOSSY_NOTE.includes("never preselected"), NVFP4_LOSSY_NOTE);
  // Never the default: the picker starts at dense and every other format carries no lossy note.
  assert.equal(selectedWeightFormat(weightFormatOptions(SM120), undefined).id, "dense");
  for (const option of weightFormatOptions(SM120).filter((item) => item.id !== "nvfp4")) {
    assert.equal(option.note, null, option.id);
  }
});

test("the pending reload compares the SAVED setting with the switch the load settled, and the served row offers Reload", () => {
  // Saved on, served model settled off: reload pending, and the served row can act on it.
  assert.equal(graphsReloadPending(SM120, true, { cuda_graphs: false }), true);
  assert.deepEqual(serveAction(true, true), { label: "Reload", disabled: false });
  assert.match(cudaGraphsControl(SM120, true, { cuda_graphs: false }).note, /reload it from Models/);
  // Saved matches the settled switch: nothing to do.
  assert.equal(graphsReloadPending(SM120, false, { cuda_graphs: false }), false);
  assert.deepEqual(serveAction(true, false), { label: "Serving", disabled: true });
  assert.deepEqual(serveAction(false, true), { label: "Serve", disabled: false });
  // A provider that does not take the switch (settled null) never asks for a reload.
  assert.equal(graphsReloadPending(SM120, true, { cuda_graphs: null }), false);
  // Where the runtime reports the switch unavailable, neither does anything else.
  assert.equal(graphsReloadPending(CPU, true, { cuda_graphs: false }), false);
});

test("the view shows the decode implementation, fused primitives, graphs captured, and the settled switch", () => {
  const loaded = {
    quantize: "nvfp4",
    cuda_graphs: true,
    load_report: { requested: "nvfp4", projections: [], cuda_graphs: true },
    last_decode: {
      ...DECODE,
      cuda_graphs: { ...DECODE.cuda_graphs, path: "mixed", replayed: 7, eager: 2, captured: 3 },
      fused_primitives: { path: "mixed", reason: "dtype" },
    },
    decode_reported: true,
  };
  const rows = Object.fromEntries(decodePathRows(status(SM120, loaded)).map((row) => [row.key, row]));
  assert.equal(rows.implementation.label, "Decode implementation");
  assert.equal(rows.implementation.value, "mtp");
  assert.equal(rows.fused.label, "Fused primitives");
  assert.equal(rows.fused.value, "mixed");
  assert.equal(rows.fused.detail, "fallback: dtype");
  assert.equal(rows.cuda_graphs.value, "mixed (7 replayed, 2 eager)");
  assert.equal(rows.cuda_graphs.detail, "3 captured · fallback: deltanet_state_unstable");
  assert.equal(rows.graph_switch.value, "on");
  const unused = decodePathRows(status(SM120, { ...loaded, cuda_graphs: null }))
    .find((row) => row.key === "graph_switch");
  assert.equal(unused.value, "not used");
  // Every per-generation row is labelled as the last generation; the host and load rows are not.
  for (const row of Object.values(rows)) {
    const perGeneration = !["backend", "weights", "graph_switch", "draft", "load_fallbacks", "prefix_cache_budget"].includes(row.key);
    assert.equal(row.section, perGeneration ? LAST_GENERATION : null, row.key);
  }
});

test("the Rust wire shape renders: the view reads tests/engine-status-wire.json as the engine serializes it", async () => {
  const wire = JSON.parse(await readFile(new URL("engine-status-wire.json", import.meta.url)));
  const rows = Object.fromEntries(decodePathRows(wire).map((row) => [row.key, row]));
  assert.equal(rows.backend.value, "candle-cuda · cuda:0 · sm_120");
  assert.equal(rows.weights.value, "NVFP4 (lossy)");
  assert.equal(rows.weights.detail, `Resident projections: nvfp4 × 448 · ${NVFP4_LOSSY_NOTE}`);
  assert.equal(rows.graph_switch.value, "on");
  assert.equal(rows.draft.value, "resident");
  assert.equal(rows.draft.detail, "/models/qwen3-0.6b");
  assert.match(rows.load_fallbacks.value, /^cuda_graphs: positions_host_scalar; mtp_head: /);
  assert.match(rows.load_fallbacks.value, /qwen3\.8-27b-mtp/);
  assert.equal(rows.graph_path.value, "eager");
  // AC3 at load, from the engine's own wire shape: the runtime's load fallback is the toggle's reason.
  assert.equal(effectiveCudaGraphs(null, wire.cuda_graphs_default), true);
  const toggle = cudaGraphsControl(wire.backend_capabilities, false, wire.loaded);
  assert.equal(toggle.disabled, true);
  assert.equal(toggle.reason, "cuda_graphs: positions_host_scalar");
  assert.equal(rows.prefix_cache_budget.value, "1.0 GB");
  assert.equal(rows.prefix_cache_budget.detail, "Requested 1.0 GB; settled by the runtime at load.");
  assert.equal(rows.implementation.value, "mtp");
  assert.equal(rows.proposer.value, "mtp · 2 drafts");
  assert.equal(rows.proposer.detail, "Accepted 3 of 4 drafts in 3 forwards");
  assert.equal(rows.accepted_length.value, "1.50 drafts per verify step");
  assert.equal(rows.prefix_cache.value, "miss");
  assert.equal(rows.fallbacks.value, "none");
  assert.equal(rows.cuda_graphs.value, "eager");
  assert.equal(rows.cuda_graphs.detail, "0 captured · fallback: deltanet_state_unstable");
  assert.equal(rows.nvfp4.value, "mixed");
  assert.equal(rows.nvfp4.detail, "fallback: rows");
  assert.equal(rows.fused.value, "fused");
  assert.equal(rows.sampler.value, "device");
  assert.equal(rows.kv_cache.value, "static · gqa attention");
  const html = ReactDOMServer.renderToStaticMarkup(React.createElement(DecodePathStatus, { engineStatus: wire }));
  assert.ok(html.includes(`<p class="decode-path-section">${LAST_GENERATION}</p>`), html);
});

test("a pushed decode status updates the served model without polling, and only that model", () => {
  const before = status(SM120, {
    source: "/m/a", quantize: null, cuda_graphs: false, load_report: null, last_decode: null, decode_reported: null,
  });
  const pushed = applyDecodeEvent(before, { source: "/m/a", last_decode: DECODE, decode_reported: true });
  assert.deepEqual(pushed.loaded.last_decode, DECODE);
  assert.equal(pushed.loaded.decode_reported, true);
  assert.equal(before.loaded.last_decode, null, "the previous status is not mutated");
  // A push for another model (the served model changed meanwhile) is ignored.
  assert.equal(applyDecodeEvent(before, { source: "/m/b", last_decode: DECODE, decode_reported: true }), before);
  assert.equal(applyDecodeEvent(null, { source: "/m/a" }), null);
  // An unreported generation lands as such.
  const unreported = applyDecodeEvent(before, { source: "/m/a", last_decode: null, decode_reported: false });
  assert.equal(decodePathRows(unreported).find((row) => row.key === "decode").value, "not reported");
});

const CARRIED_OVER = {
  sampling: { speculative: "off" },
  notices: { speculativeOffCarriedOver: true, speculativeNoticeDismissed: false },
};

test("the speculative notice shows once for a carried-over off where the runtime's default is not off", () => {
  const shown = speculativeNotice(CARRIED_OVER, "auto");
  assert.equal(shown.show, true);
  assert.equal(shown.message, "Speculative decoding is available — turn on Auto");
  assert.equal(shown.actionLabel, "Turn on Auto");
  // Hidden: the runtime's default is off (or unknown), an off that was not carried over, another
  // saved option, dismissed.
  assert.equal(speculativeNotice(CARRIED_OVER, "off").show, false);
  assert.equal(speculativeNotice(CARRIED_OVER, undefined).show, false);
  assert.equal(
    speculativeNotice({ ...CARRIED_OVER, notices: { speculativeOffCarriedOver: false } }, "auto").show,
    false,
  );
  assert.equal(speculativeNotice({ ...CARRIED_OVER, sampling: { speculative: "auto" } }, "auto").show, false);
  assert.equal(speculativeNotice(undefined, "auto").show, false);
  assert.equal(speculativeNotice(dismissSpeculativeNotice(CARRIED_OVER), "auto").show, false);
});

test("the notice's one-click action turns on Auto; dismissing persists without touching the saved option", () => {
  const auto = enableSpeculativeAuto(CARRIED_OVER);
  assert.equal(auto.sampling.speculative, "auto");
  assert.equal(auto.notices.speculativeOffCarriedOver, false);
  assert.equal(speculativeNotice(auto, "auto").show, false);
  const dismissed = dismissSpeculativeNotice(CARRIED_OVER);
  assert.equal(dismissed.sampling.speculative, "off", "a dismissal never changes the saved choice");
  assert.equal(dismissed.notices.speculativeNoticeDismissed, true);
  assert.equal(dismissed.notices.speculativeOffCarriedOver, true);
  assert.equal(CARRIED_OVER.notices.speculativeNoticeDismissed, false, "the input is not mutated");
});

test("the notice renders in the decode-path panel with its action and dismiss buttons", () => {
  const notice = { appSettings: CARRIED_OVER, speculativeDefault: "auto", onEnableAuto() {}, onDismiss() {} };
  const html = ReactDOMServer.renderToStaticMarkup(React.createElement(DecodePathStatus, {
    engineStatus: status(SM120, null),
    notice,
  }));
  assert.ok(html.includes("Speculative decoding is available — turn on Auto"), html);
  assert.ok(html.includes(">Turn on Auto</button>"), html);
  assert.ok(html.includes(">Dismiss</button>"), html);
  const hidden = ReactDOMServer.renderToStaticMarkup(React.createElement(SpeculativeNotice, {
    ...notice,
    appSettings: dismissSpeculativeNotice(CARRIED_OVER),
  }));
  assert.equal(hidden, "");
});

// The decode report the MLX runtime returns for a chat response under `auto` on a model without an
// MTP head, in the shape `src-tauri/tests/mlx_decode_status.rs` reads back from a real MLX
// generation (pinned runtime, tiny Llama snapshot).
const MLX_DECODE = {
  path: "prompt_lookup",
  proposer: "prompt_lookup",
  draft_tokens: 4,
  sampler: "device",
  kv_cache: "growing",
  attention: "gqa",
  cuda_graphs: { enabled: false, path: "none", replayed: 0, eager: 0, captured: 0, fallback_reason: null },
  nvfp4_projections: { path: "none", reason: null },
  fused_primitives: { path: "none", reason: null },
  target_forwards: 7,
  prefill_forwards: 1,
  proposed_tokens: 9,
  accepted_tokens: 5,
  verify_steps: 6,
  mean_accepted_length: 5 / 6,
  replay_forwards: 0,
  prefix_hit_tokens: 0,
  prefix_cache: { path: "miss", reason: null },
  fallbacks: [],
};

test("sc-24445 AC2: on MLX a chat response shows the proposer, accepted length and sampler path", () => {
  const engineStatus = {
    ...status(MLX, {
      source: "/models/tiny", quantize: null, cuda_graphs: null, load_report: null,
      last_decode: null, decode_reported: null,
    }),
    speculative_default: "off",
  };
  // The finished generation arrives as an `engine://decode` push.
  const pushed = applyDecodeEvent(engineStatus, { source: "/models/tiny", last_decode: MLX_DECODE, decode_reported: true });
  const html = ReactDOMServer.renderToStaticMarkup(React.createElement(DecodePathStatus, { engineStatus: pushed }));
  const rows = Object.fromEntries(decodePathRows(pushed).map((row) => [row.key, row]));
  assert.equal(rows.backend.value, "mlx · metal");
  assert.equal(rows.proposer.value, "prompt_lookup · 4 drafts");
  assert.equal(rows.accepted_length.value, "0.83 drafts per verify step");
  assert.equal(rows.accepted_length.detail, "5 accepted over 6 verify steps");
  assert.equal(rows.sampler.value, "device");
  for (const text of [
    'data-row="proposer"', "prompt_lookup · 4 drafts",
    'data-row="accepted_length"', "Accepted length", "0.83 drafts per verify step",
    'data-row="sampler"', "Sampler", "device",
  ]) {
    assert.ok(html.includes(text), `missing ${text} in ${html}`);
  }
  assert.ok(!html.includes("not reported"), html);
});

test("fallbacks, the prefix cache and (when reported) the graph path are named", () => {
  const decode = {
    ...MLX_DECODE,
    proposer: "none",
    draft_tokens: null,
    mean_accepted_length: null,
    sampler: "host:penalty",
    prefix_hit_tokens: 120,
    prefix_cache: { path: "hit", reason: null },
    fallbacks: ["speculative: `prompt_lookup` depth 12 clamped to 7 (advertised 1..=7)"],
    graph_path: "captured",
  };
  const rows = Object.fromEntries(decodePathRows(status(MLX, { quantize: null, last_decode: decode })).map((row) => [row.key, row]));
  assert.equal(rows.accepted_length.value, "n/a");
  assert.equal(rows.accepted_length.detail, "No proposer ran.");
  assert.equal(rows.sampler.value, "host:penalty");
  assert.equal(rows.prefix_cache.value, "hit · 120 prompt tokens reused");
  assert.match(rows.fallbacks.value, /clamped to 7/);
  assert.equal(rows.graph_path.value, "captured");
  // A report from a runtime without these fields renders without them.
  const older = decodePathRows(status(SM120, { quantize: null, last_decode: DECODE })).map((row) => row.key);
  for (const key of ["accepted_length", "graph_path", "prefix_cache", "fallbacks"]) {
    assert.ok(!older.includes(key), key);
  }
});

test("sc-24445 AC3: the CUDA-graph toggle is disabled with the runtime's reason when the served model cannot capture", () => {
  const eagerDecode = {
    ...DECODE,
    cuda_graphs: { enabled: true, path: "eager", replayed: 0, eager: 12, captured: 0, fallback_reason: "moe_router_host_read" },
  };
  const moe = { name: "Qwen3.5 MoE", cuda_graphs: true, load_report: { cuda_graphs: true, projections: [] }, last_decode: eagerDecode };
  // Saved OFF: turning it on is refused, with the runtime's reason.
  const control = cudaGraphsControl(SM120, false, moe);
  assert.equal(control.disabled, true);
  assert.match(control.reason, /moe_router_host_read/);
  assert.match(control.reason, /^cuda_graphs: /);
  assert.equal(control.pendingReload, false);
  assert.match(control.note, /saved setting is kept/);
  // Saved ON: the toggle stays live so the user can turn it off; the reason is still shown.
  const savedOn = cudaGraphsControl(SM120, true, moe);
  assert.equal(savedOn.disabled, false);
  assert.match(savedOn.reason, /moe_router_host_read/);
  // The runtime's graph path decides when it reports one.
  assert.equal(cudaGraphsControl(SM120, false, { ...moe, last_decode: { ...eagerDecode, graph_path: "captured" } }).disabled, false);
  // A load fallback naming the switch (`cuda_graphs: …`, the runtime's load-time format) is the
  // runtime's reason verbatim, before any generation.
  // The exact string candle-llm pushes at load for a Qwen3.5/3.8 decoder (`cuda_graphs: <graph_support reason>`).
  const fallback = "cuda_graphs: positions_host_scalar";
  const atLoad = cudaGraphsControl(SM120, false, { cuda_graphs: true, load_report: { cuda_graphs: true, fallbacks: [fallback] } });
  assert.equal(atLoad.disabled, true);
  assert.equal(atLoad.reason, fallback);
  // A load that settled no switch carries no runtime reason, so ChatWorks refuses nothing on its own.
  const unrouted = cudaGraphsControl(SM120, false, { cuda_graphs: null, load_report: { cuda_graphs: null, projections: [] } });
  assert.equal(unrouted.disabled, false);
  assert.equal(unrouted.reason, null);
  // A model that captured (some steps replayed), or one not yet measured, keeps the toggle live.
  const captured = { ...moe, last_decode: { ...DECODE, cuda_graphs: { ...DECODE.cuda_graphs, path: "mixed", replayed: 7 } } };
  assert.equal(cudaGraphsControl(SM120, false, captured).disabled, false);
  assert.equal(cudaGraphsControl(SM120, false, { ...moe, last_decode: null }).disabled, false);
  // The host's own refusal still wins where the switch is unavailable.
  assert.equal(cudaGraphsControl(MLX, true, moe).reason, MLX.cuda_graphs.reason);
  assert.equal(cudaGraphsControl(MLX, true, moe).disabled, true);
});

test("epic sc-24432 load options: the status names a refused draft with the runtime's reason, and the settled prefix-cache budget", () => {
  const refused = status(MLX, {
    quantize: null,
    cuda_graphs: null,
    draft_source: "/m/other-vocab",
    load_report: {
      cuda_graphs: null,
      projections: [],
      prefix_cache_bytes: 0,
      draft: { source: "/m/other-vocab", refusal: "draft vocabulary (151936) is not the target's (248320)" },
    },
    last_decode: null,
  });
  const rows = Object.fromEntries(decodePathRows(refused).map((row) => [row.key, row]));
  assert.equal(rows.draft.value, "refused");
  assert.equal(rows.draft.detail, "/m/other-vocab: draft vocabulary (151936) is not the target's (248320)");
  assert.equal(rows.prefix_cache_budget.value, "off");
  assert.equal(rows.prefix_cache_budget.detail, "Requested the runtime default; settled by the runtime at load.");
  const html = ReactDOMServer.renderToStaticMarkup(React.createElement(DecodePathStatus, { engineStatus: refused }));
  for (const text of ['data-row="draft"', "Draft model", "refused", 'data-row="prefix_cache_budget"', "Prefix cache budget"]) {
    assert.ok(html.includes(text), `missing ${text} in ${html}`);
  }
  // No draft named and no prefix cache in the provider: neither row appears.
  const keys = decodePathRows(status(MLX, { quantize: null, load_report: { projections: [], prefix_cache_bytes: null, draft: null } }))
    .map((row) => row.key);
  assert.ok(!keys.includes("draft") && !keys.includes("prefix_cache_budget"), keys.join(","));
});

test("Candle's length-aware decode attention reads as such; other labels keep the runtime's name", () => {
  const rows = (attention) => Object.fromEntries(decodePathRows(status(SM120, { quantize: null, last_decode: { ...DECODE, attention } }))
    .map((row) => [row.key, row]));
  assert.equal(rows("decode_attention").kv_cache.value, "static · length-aware decode attention");
  assert.equal(rows("expanded").kv_cache.value, "static · expanded attention");
  assert.equal(rows("gqa").kv_cache.value, "static · gqa attention");
});

test("a load with nothing unattached shows no load-fallbacks row", () => {
  const keys = decodePathRows(status(SM120, { quantize: null, load_report: { projections: [], fallbacks: [] } }))
    .map((row) => row.key);
  assert.ok(!keys.includes("load_fallbacks"), keys.join(","));
});

test("resident NVFP4 weights are labelled lossy beside the served model's name and in the Weights row", () => {
  const nvfp4 = {
    name: "Qwen3.8-27B NVFP4",
    quantize: "nvfp4",
    load_report: { requested: "nvfp4", projections: [{ kind: "nvfp4", count: 448, params: 1, resident_bytes: 1 }] },
  };
  assert.deepEqual(lossyWeightsBadge(nvfp4), { label: "NVFP4 · lossy", title: NVFP4_LOSSY_NOTE });
  // Without a load report, the requested format decides.
  assert.deepEqual(lossyWeightsBadge({ quantize: "nvfp4" }), { label: "NVFP4 · lossy", title: NVFP4_LOSSY_NOTE });
  const dense = {
    name: "Qwen3.8-27B",
    quantize: null,
    load_report: { requested: null, projections: [{ kind: "dense", count: 505, params: 1, resident_bytes: 1 }] },
  };
  assert.equal(lossyWeightsBadge(dense), null);
  assert.equal(lossyWeightsBadge(null), null);
  const weights = (loaded) => decodePathRows(status(SM120, loaded)).find((row) => row.key === "weights");
  assert.equal(weights(dense).value, "checkpoint encoding");
  assert.equal(weights(dense).detail, "Resident projections: dense × 505");
  assert.equal(weights({ ...nvfp4, load_report: null }).value, "NVFP4 (lossy)");
  assert.equal(weights({ ...nvfp4, load_report: null }).detail, NVFP4_LOSSY_NOTE);
});

test("sc-24446: an unset CUDA-graph setting follows the runtime's default; a saved choice wins", () => {
  assert.equal(effectiveCudaGraphs(null, true), true);
  assert.equal(effectiveCudaGraphs(undefined, true), true);
  assert.equal(effectiveCudaGraphs(null, false), false);
  assert.equal(effectiveCudaGraphs(null, undefined), false);
  assert.equal(effectiveCudaGraphs(false, true), false);
  assert.equal(effectiveCudaGraphs(true, false), true);
  // Served with graphs on under an unset setting whose default is on: nothing to reload.
  assert.equal(graphsReloadPending(SM120, effectiveCudaGraphs(null, true), { cuda_graphs: true }), false);
  assert.equal(graphsReloadPending(SM120, effectiveCudaGraphs(false, true), { cuda_graphs: true }), true);
});

test("sc-24446: a monitor demotion is named on the proposer row", () => {
  const base = {
    path: "prompt_lookup", proposer: "prompt_lookup", draft_tokens: 4, proposed_tokens: 8,
    accepted_tokens: 1, target_forwards: 6, verify_steps: 4, mean_accepted_length: 0.25,
    cuda_graphs: { enabled: false, path: "none" }, graph_path: "none", fallbacks: [],
  };
  const demoted = Object.fromEntries(decodePathRows({ loaded: { last_decode: { ...base, speculative_demoted_at: 24 } } })
    .map((row) => [row.key, row]));
  assert.equal(demoted.proposer.value, "prompt_lookup · 4 drafts → plain");
  assert.equal(demoted.proposer.detail, "Accepted 1 of 8 drafts in 6 forwards · Demoted to plain decoding at token 24");
  const kept = Object.fromEntries(decodePathRows({ loaded: { last_decode: { ...base, speculative_demoted_at: null } } })
    .map((row) => [row.key, row]));
  assert.equal(kept.proposer.value, "prompt_lookup · 4 drafts");
  assert.equal(kept.proposer.detail, "Accepted 1 of 8 drafts in 6 forwards");
});
