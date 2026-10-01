// Optional controls retain the model default until the user chooses an override.
//
// Speculative decoding is the runtime's proposer-agnostic option (sc-24445):
// `"off" | "auto" | {proposer: "mtp" | "prompt_lookup" | "draft_model", depth}`. In the form it is
// `speculativeMode` (`""`, `off`, `auto` or a proposer) plus `speculativeDepth`. An unchosen mode is
// `""`, never a UI constant: a request then omits `speculative` (the server applies the saved
// setting) and a saved setting stores `null` (the runtime's own default applies, epic sc-24432 E5).

/// The proposers a speculative request can name, in the runtime's order.
export const SPECULATIVE_PROPOSERS = [
  ["mtp", "MTP head"],
  ["prompt_lookup", "Prompt lookup"],
  ["draft_model", "Draft model"],
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

/// The speculative choices the control offers: the inherited default, `off`, `auto`, then every
/// proposer the loaded model advertises (all of them where no model is in view, as in Settings).
/// A saved proposer the model does not advertise stays visible, marked unavailable, and is never
/// sent.
export function speculativeOptions(capabilities, selected = "", defaults = false, runtimeDefault = null) {
  const advertised = defaults
    ? SPECULATIVE_PROPOSERS
    : SPECULATIVE_PROPOSERS.filter(([proposer]) => proposerCapability(capabilities, proposer));
  const inherited = defaults
    ? `Runtime default${runtimeDefault == null ? "" : ` (${describeSpeculative(runtimeDefault)})`}`
    : "App setting";
  const options = [
    ["", inherited],
    ["off", "Off"],
    ["auto", "Auto (MTP head, else prompt lookup)"],
    ...advertised,
  ];
  if (selected && !options.some(([value]) => value === selected)) {
    const label = SPECULATIVE_PROPOSERS.find(([value]) => value === selected)?.[1] ?? selected;
    options.push([selected, `${label} (not available for this model)`]);
  }
  return options;
}

/// A speculative option as a short label: `off`, `auto`, or `prompt_lookup · depth 4`.
export function describeSpeculative(speculative) {
  if (typeof speculative === "string") return speculative;
  if (speculative?.proposer) return `${speculative.proposer} · depth ${speculative.depth}`;
  return "unknown";
}

export function generationParams(value = {}) {
  return {
    reasoningEffort: value.reasoningEffort ?? "",
    preserveThinking: value.preserveThinking == null ? "" : String(value.preserveThinking),
    ...speculativeForm(value.speculative),
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
    topK: optionalNumber(params.topK),
    presencePenalty: optionalNumber(params.presencePenalty),
    repetitionPenalty: optionalNumber(params.repetitionPenalty),
    repetitionContext: optionalNumber(params.repetitionContext),
    seed,
  };
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
/// advertises it, its depth clamped to that proposer's `max_depth`.
export function speculativeRequest(speculative, capabilities) {
  if (speculative === undefined || typeof speculative === "string") return speculative;
  const capability = proposerCapability(capabilities, speculative.proposer);
  if (!capability) return undefined;
  return { proposer: speculative.proposer, depth: clampSpeculativeDepth(speculative.depth, capability) };
}

export function generationOverrides(params, capabilities = {}) {
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
  return body;
}
