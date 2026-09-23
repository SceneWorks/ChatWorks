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
  const loadQuantization = model.quantize === "q4" ? "Q4 load" : model.quantize === "q8" ? "Q8 load" : null;
  if (loadQuantization) return `${sourceBits ? `${sourceBits}-bit source` : "Safetensors"} · ${loadQuantization}`;
  return sourceBits ? `${sourceBits}-bit` : "Safetensors";
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
