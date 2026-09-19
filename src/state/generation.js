// Optional controls retain the model default until the user chooses an override.
export function generationParams(value = {}) {
  return {
    reasoningEffort: value.reasoningEffort ?? "",
    preserveThinking: value.preserveThinking == null ? "" : String(value.preserveThinking),
    mtpMode: value.mtpMode ?? "off",
    mtpDraftTokens: String(value.mtpDraftTokens ?? 3),
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
    mtpMode: params.mtpMode ?? "off",
    mtpDraftTokens: optionalNumber(params.mtpDraftTokens) ?? 3,
    topK: optionalNumber(params.topK),
    presencePenalty: optionalNumber(params.presencePenalty),
    repetitionPenalty: optionalNumber(params.repetitionPenalty),
    repetitionContext: optionalNumber(params.repetitionContext),
    seed,
  };
}

export function generationOverrides(params, capabilities = {}) {
  const values = generationSettings(params);
  const body = {};
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
  if (capabilities.mtp && values.mtpMode !== "off") {
    body.mtp = values.mtpMode === "enabled"
      ? { mode: "enabled", draft_tokens: values.mtpDraftTokens }
      : { mode: "auto" };
  }
  return body;
}
