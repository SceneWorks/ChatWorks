// Optional controls retain the model default until the user chooses an override.
//
// Speculative decoding is the runtime's proposer-agnostic option (sc-24445):
// `"off" | "auto" | {proposer: "mtp" | "prompt_lookup" | "draft_model" | "block_draft", depth}`. In the form it is
// `speculativeMode` (`""`, `off`, `auto` or a proposer) plus `speculativeDepth`. An unchosen mode is
// `""`, never a UI constant: a request then omits `speculative` (the server applies the saved
// setting) and a saved setting stores `null` (the runtime's own default applies, epic sc-24432 E5).

/// The proposers a speculative request can name, in the runtime's order.
export const SPECULATIVE_PROPOSERS = [
  ["mtp", "MTP head"],
  ["prompt_lookup", "Prompt lookup"],
  ["draft_model", "Draft model"],
  // A DSpine / DFlash2 block draft installed beside a compatible target model (SW-F360 / SW-F357).
  ["block_draft", "Block draft"],
];

/// The draft depth a proposer starts at when the form has none (the legacy MTP default).
const DEFAULT_SPECULATIVE_DEPTH = 3;

function speculativeForm(speculative) {
  if (speculative === "off" || speculative === "auto") {
    return { speculativeMode: speculative, speculativeDepth: String(DEFAULT_SPECULATIVE_DEPTH) };
  }
  if (speculative && typeof speculative === "object" && speculative.proposer) {
    return {
      speculativeMode: speculative.proposer,
      speculativeDepth: String(speculative.depth ?? DEFAULT_SPECULATIVE_DEPTH),
    };
  }
  return { speculativeMode: "", speculativeDepth: String(DEFAULT_SPECULATIVE_DEPTH) };
}

/// What the loaded model advertises for `proposer` (`max_depth`, `recommended_depth`), or `null`.
export function proposerCapability(capabilities, proposer) {
  return capabilities?.speculative?.find((item) => item.proposer === proposer) ?? null;
}

/// A depth clamped to `1..=max_depth` of the proposer's capability (`1..` when none is known).
export function clampSpeculativeDepth(depth, capability) {
  const number = Number(depth);
  const value = Math.max(1, Math.round(Number.isFinite(number) ? number : DEFAULT_SPECULATIVE_DEPTH));
  return capability ? Math.min(value, Math.max(1, capability.max_depth)) : value;
}

/// Why the saved depth is not the one the model runs, or `null`: a saved depth above the proposer's
/// advertised `max_depth` stays the user's saved preference and runs clamped (sc-24445).
export function speculativeDepthNote(depth, capability) {
  const number = Number(depth);
  if (!capability || depth === "" || depth == null || !Number.isInteger(number) || number < 1) return null;
  const runs = clampSpeculativeDepth(number, capability);
  return runs === number ? null : `Runs at ${runs} on this model (it advertises up to ${capability.max_depth}).`;
}

/// The speculative choices the control offers: the inherited default, `off`, `auto`, then every
/// proposer the loaded model advertises (all of them where no model is in view, as in Settings).
/// A saved proposer the model does not advertise stays visible, marked unavailable; a request
/// with it runs `off` ([`speculativeRequest`]).
export function speculativeOptions(capabilities, selected = "", defaults = false, runtimeDefault = null) {
  const advertised = defaults
    ? SPECULATIVE_PROPOSERS
    : SPECULATIVE_PROPOSERS.filter(([proposer]) => proposerCapability(capabilities, proposer));
  const inherited = defaults
    ? `Runtime default${runtimeDefault == null ? "" : ` (${describeSpeculative(runtimeDefault)})`}`
    : "App setting";
  const autoLabel = !defaults && advertised.length
    ? `Auto (adaptive: ${advertised.map(([, label]) => label).join(", ")})`
    : "Auto (adaptive)";
  const options = [
    ["", inherited],
    ["off", "Off"],
    ["auto", autoLabel],
    ...advertised,
  ];
  if (selected && !options.some(([value]) => value === selected)) {
    const label = SPECULATIVE_PROPOSERS.find(([value]) => value === selected)?.[1] ?? selected;
    options.push([selected, `${label} (not available for this model)`]);
  }
  return options;
}

/// Why the chosen proposer will not run on the served model, or `null` when it is advertised (or
/// no proposer is chosen). A draft-backed proposer names the load's own reason: the draft the
/// runtime refused, or that no draft was loaded. The request then runs `off` ([`speculativeRequest`]).
export function proposerUnavailableNote(mode, loaded) {
  if (!mode || mode === "off" || mode === "auto" || !loaded) return null;
  if (proposerCapability(loaded.provider?.capabilities, mode)) return null;
  const draftBacked = mode === "draft_model" || mode === "block_draft";
  const draft = loaded.load_report?.draft;
  let why = "this model does not advertise it";
  if (draftBacked && draft?.refusal) {
    why = `the runtime refused its draft (${draft.refusal})`;
  } else if (draftBacked && !loaded.draft_source) {
    why = mode === "block_draft"
      ? "no block draft is loaded with it — install one under Models, then reload"
      : "no draft model is loaded with it — choose one under Models, then reload";
  } else if (draftBacked && draft && !draft.refusal) {
    why = `its draft is not a ${mode === "block_draft" ? "block draft" : "draft model"}`;
  }
  return `Not available: ${why}. Decoding runs without a proposer.`;
}

/// A speculative option as a short label: `off`, `auto`, or `prompt_lookup · depth 4`.
export function describeSpeculative(speculative) {
  if (typeof speculative === "string") return speculative;
  if (speculative?.proposer) return `${speculative.proposer} · depth ${speculative.depth}`;
  return "unknown";
}

/// Exact prefix-checkpoint choices: `""` (stored as `null`) follows the inherited default,
/// `"off"` preserves ordinary exact-prefix behavior, and `"auto"` captures multi-boundary and
/// document checkpoints.
export function prefixCheckpointOptions(
  defaults = false,
  runtimeDefault = null,
  appSetting = null,
) {
  const inherited = defaults
    ? `Runtime default${runtimeDefault == null ? "" : ` (${runtimeDefault})`}`
    : appSetting
      ? `App setting (${appSetting})`
      : `Runtime default${runtimeDefault == null ? "" : ` (${runtimeDefault})`}`;
  return [
    ["", inherited],
    ["off", "Off (ordinary prefix)"],
    ["auto", "Auto (edits & documents)"],
  ];
}

/// Availability and reason for the prefix-checkpoints policy control, capability-gated on
/// `loaded.provider.capabilities.prefix_checkpoints`.
export function prefixCheckpointsControl(capabilities, loaded = null, defaults = false) {
  if (defaults) {
    const supported = loaded?.provider?.capabilities?.prefix_checkpoints != null;
    if (loaded && !supported) {
      return {
        disabled: false,
        reason: null,
        note: "The served model does not advertise exact prefix checkpoints; this default applies to models that support them.",
      };
    }
    return {
      disabled: false,
      reason: null,
      note: "Reuses exact prefix checkpoints across edited turns, branches, and pinned documents.",
    };
  }
  if (!loaded) {
    return {
      disabled: true,
      reason: "Unavailable: load a model that supports exact prefix checkpoints.",
      note: null,
    };
  }
  const support = capabilities?.prefix_checkpoints;
  if (!support) {
    return {
      disabled: true,
      reason: "Unavailable: the loaded model does not advertise exact prefix checkpoints.",
      note: null,
    };
  }
  const docNote = support.supports_document_span ? "; supports pinned documents" : "";
  return {
    disabled: false,
    reason: null,
    note: `Up to ${support.max_checkpoints_per_request} boundaries per request${docNote}.`,
  };
}

export function generationParams(value = {}) {
  return {
    reasoningEffort: value.reasoningEffort ?? "",
    preserveThinking: value.preserveThinking == null ? "" : String(value.preserveThinking),
    ...speculativeForm(value.speculative),
    prefixCheckpoints: value.prefixCheckpoints === "off" || value.prefixCheckpoints === "auto"
      ? value.prefixCheckpoints : "",
    topK: String(value.topK ?? ""),
    presencePenalty: String(value.presencePenalty ?? ""),
    repetitionPenalty: String(value.repetitionPenalty ?? ""),
    repetitionContext: String(value.repetitionContext ?? ""),
    seed: String(value.seed ?? ""),
  };
}

export function applySamplingPreset(params, preset) {
  if (!preset) return params;
  return {
    ...params,
    temperature: String(preset.temperature),
    topP: String(preset.top_p),
    topK: String(preset.top_k),
    presencePenalty: String(preset.presence_penalty),
    repetitionPenalty: String(preset.repetition_penalty),
    repetitionContext: String(preset.repetition_context),
  };
}

function optionalNumber(value) {
  if (value == null || String(value).trim() === "") return null;
  const number = Number(value);
  if (!Number.isFinite(number)) throw new Error("Generation settings must contain valid numbers.");
  return number;
}

export function generationSettings(params) {
  const seed = optionalNumber(params.seed);
  if (seed != null && (!Number.isSafeInteger(seed) || seed < 0)) {
    throw new Error("Seed must be a non-negative integer no larger than 9007199254740991.");
  }
  return {
    reasoningEffort: params.reasoningEffort || null,
    preserveThinking: params.preserveThinking === "" || params.preserveThinking == null
      ? null : params.preserveThinking === true || params.preserveThinking === "true",
    // Omitted (undefined) when unchosen, so the inherited default applies instead of a UI guess.
    speculative: speculativeSetting(params),
    prefixCheckpoints: prefixCheckpointSetting(params),
    topK: optionalNumber(params.topK),
    presencePenalty: optionalNumber(params.presencePenalty),
    repetitionPenalty: optionalNumber(params.repetitionPenalty),
    repetitionContext: optionalNumber(params.repetitionContext),
    seed,
  };
}

function prefixCheckpointSetting(params) {
  const mode = params.prefixCheckpoints;
  if (mode === "off" || mode === "auto") return mode;
  return undefined;
}

function speculativeSetting(params) {
  const mode = params.speculativeMode;
  if (!mode) return undefined;
  if (mode === "off" || mode === "auto") return mode;
  const depth = optionalNumber(params.speculativeDepth) ?? DEFAULT_SPECULATIVE_DEPTH;
  if (!Number.isInteger(depth) || depth < 1) {
    throw new Error("Speculative depth must be a whole number of at least 1.");
  }
  return { proposer: mode, depth };
}

/// The request's speculative option against the loaded model: `off` and `auto` always (the runtime
/// resolves `auto` to what the model offers and names any fallback); a proposer only when the model
/// advertises it, its depth clamped to that proposer's `max_depth`. A proposer the model does not
/// advertise runs `off` — the same rule the server applies to an inherited one.
export function speculativeRequest(speculative, capabilities) {
  if (speculative === undefined || typeof speculative === "string") return speculative;
  const capability = proposerCapability(capabilities, speculative.proposer);
  if (!capability) return "off";
  return { proposer: speculative.proposer, depth: clampSpeculativeDepth(speculative.depth, capability) };
}

/// Leading `messages` count sent on `/v1/chat/completions` covered by a pinned conversation
/// message (`0..=pinnedDocumentIndex`), accounting for any prepended system prompt message.
export function pinnedDocumentMessagesCount({
  pinnedDocumentIndex,
  messagesCount,
  hasSystemPrompt = false,
}) {
  if (!Number.isInteger(pinnedDocumentIndex) || pinnedDocumentIndex < 0 || pinnedDocumentIndex >= messagesCount) {
    return null;
  }
  return (hasSystemPrompt ? 1 : 0) + pinnedDocumentIndex + 1;
}

/// Update a pinned document message index after `handleRewind(rewindIndex)` slices the transcript
/// to `messages.slice(0, rewindIndex)`: clears the pin when truncated at or before it.
export function rewindPinnedDocumentIndex(pinnedDocumentIndex, rewindIndex) {
  if (!Number.isInteger(pinnedDocumentIndex) || pinnedDocumentIndex < 0) return null;
  return rewindIndex <= pinnedDocumentIndex ? null : pinnedDocumentIndex;
}

/// Build the request's `prefix_checkpoints` field against `capabilities`. Disabled (`"off"`)
/// always omits any document span so ordinary exact-prefix behavior is preserved.
export function prefixCheckpointsRequest(mode, capabilities, documentMessagesCount = null) {
  const support = capabilities?.prefix_checkpoints;
  if (!support) return undefined;
  if (mode === "off") return "off";
  if (Number.isInteger(documentMessagesCount) && documentMessagesCount > 0 && support.supports_document_span !== false) {
    return { mode: "auto", document: { messages: { count: documentMessagesCount } } };
  }
  if (mode === "auto") return "auto";
  return undefined;
}

export function generationOverrides(params, capabilities = {}, options = {}) {
  const values = generationSettings(params);
  const body = { model_defaults: ["reasoning_effort", "preserve_thinking"] };
  for (const [key, wire] of [["topK", "top_k"], ["presencePenalty", "presence_penalty"], ["repetitionPenalty", "repetition_penalty"],
    ["repetitionContext", "repetition_context"], ["seed", "seed"]]) {
    if (values[key] != null) body[wire] = values[key];
  }
  if (capabilities.supports_reasoning_effort && !params.disableThinking && values.reasoningEffort) {
    body.reasoning_effort = values.reasoningEffort;
  }
  if (capabilities.supports_preserve_thinking && values.preserveThinking != null) {
    body.preserve_thinking = values.preserveThinking;
  }
  const speculative = speculativeRequest(values.speculative, capabilities);
  if (speculative !== undefined) body.speculative = speculative;
  const prefixCheckpoints = prefixCheckpointsRequest(
    values.prefixCheckpoints,
    capabilities,
    options.documentMessagesCount,
  );
  if (prefixCheckpoints !== undefined) body.prefix_checkpoints = prefixCheckpoints;
  return body;
}
