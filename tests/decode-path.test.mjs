// sc-24139: weight-format gating, the CUDA-graph control, and the decode-path status view, all
// driven by the runtime's own reports in `engine_status`.
import assert from "node:assert/strict";
import test from "node:test";
import React from "react";
import ReactDOMServer from "react-dom/server";
import {
  cudaGraphsControl,
  decodePathRows,
  selectedWeightFormat,
  weightFormatOptions,
} from "../src/state/decodePath.js";
import { DecodePathStatus } from "../src/components/DecodePathStatus.js";
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
  assert.equal(modelSubtitle({ quantize: "nvfp4" }), "NVFP4");
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
  assert.equal(rows.weights.value, "NVFP4");
  assert.equal(rows.weights.detail, "Resident projections: nvfp4 × 448, dense × 2");
  assert.equal(rows.proposer.value, "mtp · 3 drafts");
  assert.equal(rows.proposer.detail, "Accepted 6 of 9 drafts in 5 forwards");
  assert.equal(rows.cuda_graphs.value, "eager");
  assert.equal(rows.cuda_graphs.detail, "fallback: deltanet_state_unstable");
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
  const rows = decodePathRows(status(CPU, { quantize: "q8", cuda_graphs: null, load_report: null, last_decode: null }));
  assert.deepEqual(rows.map((row) => row.key), ["backend", "weights", "decode"]);
  assert.equal(rows[2].value, "not measured yet");
  assert.deepEqual(decodePathRows(status(MLX, null)).map((row) => row.key), ["backend"]);
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
