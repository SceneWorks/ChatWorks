export function formatBytes(bytes) {
  if (!bytes && bytes !== 0) return "";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value >= 10 || unit === 0 ? 0 : 1)} ${units[unit]}`;
}

export function modelSubtitle(model) {
  const parts = [];
  parts.push(modelWeightLabel(model));
  if (model.sizeBytes) parts.push(formatBytes(model.sizeBytes));
  return parts.join(" · ");
}

export function modelWeightLabel(model) {
  if (model.pack === "bonsai2-packed" || model.format === "gguf-prism-packed") return "Bonsai 2 packed";
  if (model.format === "gguf") return "GGUF";
  const sourceBits = Number.isInteger(model.sourceBits) && model.sourceBits >= 1 && model.sourceBits <= 8
    ? model.sourceBits : null;
  const loadQuantization = { q4: "Q4 load", q8: "Q8 load", nvfp4: "NVFP4 load (lossy)" }[model.quantize] ?? null;
  if (loadQuantization) return `${sourceBits ? `${sourceBits}-bit source` : "Safetensors"} · ${loadQuantization}`;
  return sourceBits ? `${sourceBits}-bit` : "Safetensors";
}

/// The Models screen's notice after a load. A load that had to unload the served model first
/// (the engine's `load_transition`: a reload of the same model, or one that did not fit beside
/// it) says so (sc-24140 feature-end review).
export function loadNotice(modelName, status) {
  const transition = status?.load_transition;
  if (transition?.reason === "reload") {
    return `${modelName} reloaded. The served copy was unloaded first, so two copies never had to fit in memory.`;
  }
  if (transition?.reason === "memory") {
    return `${modelName} is now the served model. ${transition.released} was unloaded first because both did not fit in memory.`;
  }
  return `${modelName} is now the served model.`;
}

export function isExactGgufUrl(value) {
  return /huggingface\.co\/[^/]+\/[^/]+\/(?:blob|resolve)\/[^/]+\/.+\.gguf(?:[?#].*)?$/i.test(value.trim());
}

export async function unloadServedModel({ invoke, busy, refreshStatus }) {
  if (busy) throw new Error("Stop generation before unloading the model.");
  // Also stop requests from external API clients, then let the engine's serial queue finish cleanup.
  await invoke("stop_generation");
  await invoke("unload_model");
  await refreshStatus();
}
