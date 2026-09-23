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
  if (model.pack === "bonsai2-packed" || model.format === "gguf-prism-packed") parts.push("Bonsai 2 packed");
  else if (model.format === "gguf") parts.push("GGUF");
  else if (model.quantize === "q4") parts.push("Q4");
  else if (model.quantize === "q8") parts.push("Q8");
  else parts.push("Dense");
  if (model.sizeBytes) parts.push(formatBytes(model.sizeBytes));
  return parts.join(" · ");
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
