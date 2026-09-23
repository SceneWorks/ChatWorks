// View models for the runtime-backed decode controls and the decode-path status (sc-24139).
// Every value shown comes from the runtime's own reports in `engine_status`:
// `backend_capabilities` (what this host can serve, with the runtime's reason when it cannot),
// `loaded.load_report` (what the load produced) and `loaded.last_decode` (which path the most
// recent generation took). Nothing here guesses a path the runtime did not report.

export const WEIGHT_FORMATS = [
  { id: "dense", label: "Dense (bf16)", value: null },
  { id: "q4", label: "Quantize Q4", value: "q4" },
  { id: "q8", label: "Quantize Q8", value: "q8" },
  { id: "nvfp4", label: "NVFP4 (Blackwell)", value: "nvfp4" },
];

const UNREPORTED = "The inference runtime did not report this capability.";

function feature(capabilities, name) {
  const support = capabilities?.[name];
  const supported = support?.supported === true;
  return { supported, reason: supported ? null : support?.reason ?? UNREPORTED };
}

/// The weight-format choices, NVFP4 disabled with the runtime's reason unless the runtime reports
/// it available (a compute capability >= sm_120 GPU on the Candle CUDA build).
export function weightFormatOptions(capabilities) {
  const nvfp4 = feature(capabilities, "nvfp4");
  return WEIGHT_FORMATS.map((option) =>
    option.id === "nvfp4"
      ? { ...option, disabled: !nvfp4.supported, reason: nvfp4.reason }
      : { ...option, disabled: false, reason: null },
  );
}

/// The weight format to submit: a disabled choice never reaches a load request.
export function selectedWeightFormat(options, id) {
  const option = options.find((item) => item.id === id && !item.disabled);
  return option ?? options[0];
}

/// The CUDA-graph toggle: disabled with the runtime's reason where the switch is unavailable, and
/// flagged when the served model was loaded under a different setting (graphs are a load option,
/// so a change applies on the next load).
export function cudaGraphsControl(capabilities, setting, loaded) {
  const graphs = feature(capabilities, "cuda_graphs");
  const loadedWith = loaded?.cuda_graphs;
  const pendingReload = graphs.supported && loaded != null && typeof loadedWith === "boolean"
    && loadedWith !== Boolean(setting);
  return {
    disabled: !graphs.supported,
    reason: graphs.reason,
    pendingReload,
    note: pendingReload
      ? `The served model was loaded with CUDA graphs ${loadedWith ? "on" : "off"}; reload it from Models to apply.`
      : "Applies when a model is loaded.",
  };
}

const FORMAT_NAMES = { q4: "Q4", q8: "Q8", nvfp4: "NVFP4" };

function weightsValue(loaded) {
  const requested = loaded.quantize ? FORMAT_NAMES[loaded.quantize] ?? loaded.quantize : "checkpoint encoding";
  const projections = loaded.load_report?.projections ?? [];
  if (!projections.length) return { value: requested, detail: null };
  const resident = projections.map((item) => `${item.kind} × ${item.count}`).join(", ");
  return { value: requested, detail: `Resident projections: ${resident}` };
}

function proposerValue(decode) {
  if (decode.proposer === "none") {
    return { value: "none (token-at-a-time)", detail: null };
  }
  const drafts = decode.draft_tokens ? ` · ${decode.draft_tokens} drafts` : "";
  const detail = decode.proposed_tokens > 0
    ? `Accepted ${decode.accepted_tokens} of ${decode.proposed_tokens} drafts in ${decode.target_forwards} forwards`
    : null;
  return { value: `${decode.proposer}${drafts}`, detail };
}

function graphsValue(graphs) {
  if (!graphs.enabled) return { value: "off", detail: null };
  const reason = graphs.fallback_reason ? `fallback: ${graphs.fallback_reason}` : null;
  switch (graphs.path) {
    case "graph":
      return { value: `replayed (${graphs.replayed} steps)`, detail: reason };
    case "mixed":
      return { value: `mixed (${graphs.replayed} replayed, ${graphs.eager} eager)`, detail: reason };
    case "eager":
      return { value: "eager", detail: reason };
    default:
      return { value: "on, not used by this decode path", detail: reason };
  }
}

function pathValue(report, noneLabel) {
  if (!report || report.path === "none") return { value: noneLabel, detail: null };
  return { value: report.path, detail: report.reason ? `fallback: ${report.reason}` : null };
}

/// The rows of the decode-path status view, in display order.
export function decodePathRows(engineStatus) {
  const capabilities = engineStatus?.backend_capabilities;
  const loaded = engineStatus?.loaded;
  const backend = [engineStatus?.execution_backend, capabilities?.device, capabilities?.compute_capability]
    .filter(Boolean)
    .join(" · ");
  const rows = [{ key: "backend", label: "Backend", value: backend || "unknown", detail: null }];
  if (!loaded) return rows;
  rows.push({ key: "weights", label: "Weights", ...weightsValue(loaded) });
  const decode = loaded.last_decode;
  if (!decode) {
    rows.push({
      key: "decode",
      label: "Decode path",
      value: "not measured yet",
      detail: "Reported by the runtime after the next generation.",
    });
    return rows;
  }
  rows.push({ key: "proposer", label: "Proposer", ...proposerValue(decode) });
  rows.push({ key: "cuda_graphs", label: "CUDA graphs", ...graphsValue(decode.cuda_graphs) });
  rows.push({ key: "nvfp4", label: "NVFP4 projections", ...pathValue(decode.nvfp4_projections, "none (no NVFP4 weights)") });
  rows.push({ key: "sampler", label: "Sampler", value: decode.sampler, detail: null });
  rows.push({ key: "kv_cache", label: "KV cache", value: `${decode.kv_cache} · ${decode.attention} attention`, detail: null });
  return rows;
}
