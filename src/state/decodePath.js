// View models for the runtime-backed decode controls and the decode-path status (sc-24139).
// Every value shown comes from the runtime's own reports in `engine_status`:
// `backend_capabilities` (what this host can serve, with the runtime's reason when it cannot),
// `loaded.load_report` (what the load produced, including the CUDA-graph switch it settled) and
// `loaded.last_decode` (which path the most recent generation took). Nothing here guesses a path
// the runtime did not report.

/// NVFP4 is lossy: on Qwen3.8-27B it measured +7.37% perplexity against bf16, above the epic's 2%
/// gate for a default-eligible format. It is offered, labelled as such, and never preselected.
export const NVFP4_LOSSY_NOTE =
  "NVFP4 is lossy: +7.37% perplexity vs bf16 on Qwen3.8-27B (above the 2% bar for a default), so it is never preselected.";

export const WEIGHT_FORMATS = [
  { id: "dense", label: "Dense (bf16)", value: null, note: null },
  { id: "q4", label: "Quantize Q4", value: "q4", note: null },
  { id: "q8", label: "Quantize Q8", value: "q8", note: null },
  { id: "nvfp4", label: "NVFP4 (Blackwell, lossy)", value: "nvfp4", note: NVFP4_LOSSY_NOTE },
];

const UNREPORTED = "The inference runtime did not report this capability.";

function feature(capabilities, name) {
  const support = capabilities?.[name];
  const supported = support?.supported === true;
  return { supported, reason: supported ? null : support?.reason ?? UNREPORTED };
}

/// The weight-format choices, NVFP4 disabled with the runtime's reason unless the runtime reports
/// it available (a compute capability >= sm_120 GPU on the Candle CUDA build). Whether a given
/// snapshot can be served as NVFP4 is asked of the runtime when it is imported or added.
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

/// Whether the served model runs under a different CUDA-graph switch than the SAVED setting:
/// graphs are a load option, so it applies only after a reload. Compared against the switch the
/// runtime settled at load (`loaded.cuda_graphs`, from its load report) — a provider that does
/// not use the switch (`null`) never asks for a reload.
export function graphsReloadPending(capabilities, savedSetting, loaded) {
  const loadedWith = loaded?.cuda_graphs;
  return feature(capabilities, "cuda_graphs").supported
    && loaded != null
    && typeof loadedWith === "boolean"
    && loadedWith !== Boolean(savedSetting);
}

/// The served-model row's action on the Models screen: a model served as saved reads "Serving"
/// (nothing to do); one served under a CUDA-graph switch other than the saved setting offers
/// "Reload" — graphs are a load option, so reloading is how the saved setting takes effect.
export function serveAction(isServed, reloadPending) {
  if (isServed && reloadPending) return { label: "Reload", disabled: false };
  if (isServed) return { label: "Serving", disabled: true };
  return { label: "Serve", disabled: false };
}

/// Why the SERVED model cannot capture CUDA graphs, in the runtime's words, or `null` (sc-24445).
/// The runtime says so three ways: a load fallback naming `cuda_graphs` (its load report), a load
/// that settled no switch at all for a provider that never routes decode steps through the graph
/// runner (`cuda_graphs: null` on a host that offers the switch), and a generation that ran with
/// graphs on yet replayed nothing — eager, with the runtime's fallback reason (for example
/// `moe_router_host_read`).
export function servedModelGraphRefusal(loaded) {
  if (!loaded) return null;
  const loadFallback = (loaded.load_report?.fallbacks ?? []).find((item) => item.startsWith("cuda_graphs"));
  if (loadFallback) return loadFallback;
  if (loaded.load_report && loaded.cuda_graphs == null) {
    return "cuda_graphs: the served model's provider does not route decode steps through the CUDA-graph runner (its load settled no switch)";
  }
  const graphs = loaded.last_decode?.cuda_graphs;
  const eager = loaded.last_decode?.graph_path != null
    ? loaded.last_decode.graph_path === "eager"
    : graphs?.path === "eager";
  if (graphs?.enabled && eager && graphs.replayed === 0 && graphs.fallback_reason) {
    return `cuda_graphs: the served model cannot capture (${graphs.fallback_reason}); its steps run eagerly`;
  }
  return null;
}

/// The CUDA-graph toggle: disabled with the runtime's reason where the switch is unavailable on
/// this host or the served model cannot capture (sc-24445), and flagged when the served model was
/// loaded under a different saved setting (the Models screen then offers "Reload" on the served
/// model).
export function cudaGraphsControl(capabilities, savedSetting, loaded) {
  const graphs = feature(capabilities, "cuda_graphs");
  const modelRefusal = graphs.supported ? servedModelGraphRefusal(loaded) : null;
  const pendingReload = !modelRefusal && graphsReloadPending(capabilities, savedSetting, loaded);
  return {
    disabled: !graphs.supported || modelRefusal != null,
    reason: graphs.supported ? modelRefusal : graphs.reason,
    pendingReload,
    note: pendingReload
      ? `The served model was loaded with CUDA graphs ${loaded.cuda_graphs ? "on" : "off"}; reload it from Models to apply.`
      : modelRefusal
        ? "The saved setting is kept and applies to the next model that can capture."
        : "Applies when a model is loaded.",
  };
}

const FORMAT_NAMES = { q4: "Q4", q8: "Q8", nvfp4: "NVFP4" };

/// Whether the served model holds NVFP4 weights: its resident projections as the runtime counted
/// them, else (no load report) the format the load requested.
function holdsNvfp4(loaded) {
  const projections = loaded?.load_report?.projections ?? [];
  return projections.length
    ? projections.some((item) => item.kind === "nvfp4")
    : loaded?.quantize === "nvfp4";
}

/// The badge beside the served model's name when its weights are NVFP4, which is lossy
/// (sc-24140 feature-end review); `null` otherwise.
export function lossyWeightsBadge(loaded) {
  return holdsNvfp4(loaded) ? { label: "NVFP4 · lossy", title: NVFP4_LOSSY_NOTE } : null;
}

function weightsValue(loaded) {
  const requested = loaded.quantize ? FORMAT_NAMES[loaded.quantize] ?? loaded.quantize : "checkpoint encoding";
  const lossy = holdsNvfp4(loaded);
  const value = lossy ? `${loaded.quantize === "nvfp4" ? requested : "NVFP4"} (lossy)` : requested;
  const projections = loaded.load_report?.projections ?? [];
  const resident = projections.length
    ? `Resident projections: ${projections.map((item) => `${item.kind} × ${item.count}`).join(", ")}`
    : null;
  const detail = [resident, lossy ? NVFP4_LOSSY_NOTE : null].filter(Boolean).join(" · ") || null;
  return { value, detail };
}

function graphSwitchValue(loaded) {
  if (typeof loaded.cuda_graphs !== "boolean") {
    return { value: "not used", detail: "This runtime/provider does not take the CUDA-graph switch." };
  }
  return { value: loaded.cuda_graphs ? "on" : "off", detail: "Settled by the runtime at load." };
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

/// The realized mean accepted length, as the runtime computed it (`accepted / verify_steps`).
function acceptedLengthValue(decode) {
  const mean = decode.mean_accepted_length;
  if (typeof mean !== "number") {
    return { value: "n/a", detail: decode.proposer === "none" ? "No proposer ran." : "No verify step ran." };
  }
  return {
    value: `${mean.toFixed(2)} drafts per verify step`,
    detail: `${decode.accepted_tokens} accepted over ${decode.verify_steps} verify steps`,
  };
}

function prefixCacheValue(decode) {
  const cache = decode.prefix_cache;
  const value = cache.path === "hit" ? `hit · ${decode.prefix_hit_tokens} prompt tokens reused` : cache.path || "none";
  return { value, detail: cache.reason ?? null };
}

function graphsValue(graphs) {
  if (!graphs.enabled) return { value: "off", detail: null };
  const detail = [
    `${graphs.captured} captured`,
    graphs.fallback_reason ? `fallback: ${graphs.fallback_reason}` : null,
  ].filter(Boolean).join(" · ");
  switch (graphs.path) {
    case "graph":
      return { value: `replayed (${graphs.replayed} steps)`, detail };
    case "mixed":
      return { value: `mixed (${graphs.replayed} replayed, ${graphs.eager} eager)`, detail };
    case "eager":
      return { value: "eager", detail };
    default:
      return { value: "on, not used by this decode path", detail };
  }
}

function pathValue(report, noneLabel) {
  if (!report || report.path === "none") return { value: noneLabel, detail: null };
  return { value: report.path, detail: report.reason ? `fallback: ${report.reason}` : null };
}

/// The section the per-generation rows belong to: they describe the most recent generation that
/// finished, never one still streaming.
export const LAST_GENERATION = "Last generation";

/// The rows of the decode-path status view, in display order. Each row names its `section`:
/// `null` for the host and the load, `LAST_GENERATION` for what the most recent generation ran.
export function decodePathRows(engineStatus) {
  const capabilities = engineStatus?.backend_capabilities;
  const loaded = engineStatus?.loaded;
  const backend = [engineStatus?.execution_backend, capabilities?.device, capabilities?.compute_capability]
    .filter(Boolean)
    .join(" · ");
  const rows = [{ key: "backend", label: "Backend", value: backend || "unknown", detail: null, section: null }];
  if (!loaded) return rows;
  rows.push({ key: "weights", label: "Weights", ...weightsValue(loaded), section: null });
  rows.push({ key: "graph_switch", label: "CUDA-graph switch", ...graphSwitchValue(loaded), section: null });
  const decode = loaded.last_decode;
  const last = (row) => rows.push({ ...row, section: LAST_GENERATION });
  if (!decode) {
    last(loaded.decode_reported === false
      ? {
        key: "decode",
        label: "Decode path",
        value: "not reported",
        detail: "This runtime/provider does not report its decode path.",
      }
      : {
        key: "decode",
        label: "Decode path",
        value: "not measured yet",
        detail: "Shown when a generation finishes, where the runtime reports it.",
      });
    return rows;
  }
  last({ key: "implementation", label: "Decode implementation", value: decode.path, detail: null });
  last({ key: "proposer", label: "Proposer", ...proposerValue(decode) });
  if ("mean_accepted_length" in decode) {
    last({ key: "accepted_length", label: "Accepted length", ...acceptedLengthValue(decode) });
  }
  last({ key: "cuda_graphs", label: "CUDA graphs", ...graphsValue(decode.cuda_graphs) });
  if (decode.graph_path != null) {
    last({ key: "graph_path", label: "Graph path", value: decode.graph_path, detail: null });
  }
  last({ key: "nvfp4", label: "NVFP4 projections", ...pathValue(decode.nvfp4_projections, "none (no NVFP4 weights)") });
  last({ key: "fused", label: "Fused primitives", ...pathValue(decode.fused_primitives, "none") });
  last({ key: "sampler", label: "Sampler", value: decode.sampler, detail: null });
  last({ key: "kv_cache", label: "KV cache", value: `${decode.kv_cache} · ${decode.attention} attention`, detail: null });
  if (decode.prefix_cache) last({ key: "prefix_cache", label: "Prefix cache", ...prefixCacheValue(decode) });
  if (Array.isArray(decode.fallbacks)) {
    last({
      key: "fallbacks",
      label: "Fallbacks",
      value: decode.fallbacks.length ? decode.fallbacks.join("; ") : "none",
      detail: null,
    });
  }
  return rows;
}

/// Apply an `engine://decode` push (a finished generation, from the desktop or an API client) to
/// the engine status the view renders. A push for a model that is no longer the served one is
/// ignored.
export function applyDecodeEvent(engineStatus, payload) {
  const loaded = engineStatus?.loaded;
  if (!loaded || !payload || payload.source !== loaded.source) return engineStatus;
  return {
    ...engineStatus,
    loaded: { ...loaded, last_decode: payload.last_decode ?? null, decode_reported: payload.decode_reported ?? null },
  };
}

/// The one-time "speculative decoding is available" notice (sc-24139 feature-end review): shown
/// while speculative decoding is `off` only because an older settings file carried `off` over and
/// the runtime's own default (`engine_status.speculative_default`, sc-24445) is not `off`, until the
/// user turns Auto on or dismisses it.
export function speculativeNotice(appSettings, speculativeDefault) {
  const notices = appSettings?.notices ?? {};
  const show = speculativeDefault != null
    && speculativeDefault !== "off"
    && appSettings?.sampling?.speculative === "off"
    && notices.speculativeOffCarriedOver === true
    && notices.speculativeNoticeDismissed !== true;
  return {
    show,
    message: "Speculative decoding is available — turn on Auto",
    detail: "Your earlier settings kept it off. Auto uses the model's own MTP head where it has one, prompt lookup where it does not, and ordinary decoding where neither runs.",
    actionLabel: "Turn on Auto",
    dismissLabel: "Dismiss",
  };
}

/// The settings after the notice's one-click action: speculative decoding on Auto (the runtime
/// then clears the carried-over flag for good).
export function enableSpeculativeAuto(appSettings) {
  return {
    ...appSettings,
    sampling: { ...appSettings.sampling, speculative: "auto" },
    notices: { ...appSettings.notices, speculativeOffCarriedOver: false },
  };
}

/// The settings after the notice is dismissed; the saved `off` is left as it is.
export function dismissSpeculativeNotice(appSettings) {
  return {
    ...appSettings,
    notices: { ...appSettings.notices, speculativeNoticeDismissed: true },
  };
}
