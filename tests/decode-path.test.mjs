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
    const perGeneration = !["backend", "weights", "graph_switch"].includes(row.key);
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
  assert.equal(rows.implementation.value, "mtp");
  assert.equal(rows.proposer.value, "mtp · 2 drafts");
  assert.equal(rows.proposer.detail, "Accepted 3 of 4 drafts in 2 forwards");
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
  sampling: { mtpMode: "off" },
  notices: { speculativeOffCarriedOver: true, speculativeNoticeDismissed: false },
};

test("the speculative notice shows once for a carried-over off on CUDA, and hides otherwise", () => {
  const shown = speculativeNotice(CARRIED_OVER, "candle-cuda");
  assert.equal(shown.show, true);
  assert.equal(shown.message, "Speculative decoding is available — turn on Auto");
  assert.equal(shown.actionLabel, "Turn on Auto");
  // Hidden: another backend, an off that was not carried over, a mode other than off, dismissed.
  assert.equal(speculativeNotice(CARRIED_OVER, "candle-cpu").show, false);
  assert.equal(speculativeNotice(CARRIED_OVER, "mlx").show, false);
  assert.equal(
    speculativeNotice({ ...CARRIED_OVER, notices: { speculativeOffCarriedOver: false } }, "candle-cuda").show,
    false,
  );
  assert.equal(speculativeNotice({ ...CARRIED_OVER, sampling: { mtpMode: "auto" } }, "candle-cuda").show, false);
  assert.equal(speculativeNotice(undefined, "candle-cuda").show, false);
  assert.equal(speculativeNotice(dismissSpeculativeNotice(CARRIED_OVER), "candle-cuda").show, false);
});

test("the notice's one-click action turns on Auto; dismissing persists without touching the saved mode", () => {
  const auto = enableSpeculativeAuto(CARRIED_OVER);
  assert.equal(auto.sampling.mtpMode, "auto");
  assert.equal(auto.notices.speculativeOffCarriedOver, false);
  assert.equal(speculativeNotice(auto, "candle-cuda").show, false);
  const dismissed = dismissSpeculativeNotice(CARRIED_OVER);
  assert.equal(dismissed.sampling.mtpMode, "off", "a dismissal never changes the saved choice");
  assert.equal(dismissed.notices.speculativeNoticeDismissed, true);
  assert.equal(dismissed.notices.speculativeOffCarriedOver, true);
  assert.equal(CARRIED_OVER.notices.speculativeNoticeDismissed, false, "the input is not mutated");
});

test("the notice renders in the decode-path panel with its action and dismiss buttons", () => {
  const notice = { appSettings: CARRIED_OVER, executionBackend: "candle-cuda", onEnableAuto() {}, onDismiss() {} };
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
