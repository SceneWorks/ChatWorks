import assert from "node:assert/strict";
import test from "node:test";
import { loadNotice, modelSubtitle, modelWeightLabel } from "./models.js";

test("cached source labels follow the detected format and quantization", () => {
  assert.equal(modelWeightLabel({ format: "hf-safetensors", sourceBits: 4 }), "4-bit");
  assert.equal(modelWeightLabel({ format: "hf-safetensors" }), "Safetensors");
  assert.equal(modelWeightLabel({ format: "hf-safetensors", pack: "bonsai2-packed", sourceBits: 2 }), "Bonsai 2 packed");
  assert.equal(modelWeightLabel({ format: "gguf" }), "GGUF");
});

test("adopted source bits remain visible in the registered model subtitle", () => {
  assert.equal(modelSubtitle({ format: "hf-safetensors", sourceBits: 4, sizeBytes: 4096 }), "4-bit · 4.0 KB");
  assert.equal(modelSubtitle({ format: "hf-safetensors", quantize: "q8", sourceBits: 4 }), "4-bit source · Q8 load");
  assert.equal(modelSubtitle({ format: "hf-safetensors", quantize: "q8" }), "Safetensors · Q8 load");
});

test("an NVFP4 load keeps the source label, names the load format and says it is lossy", () => {
  assert.equal(modelWeightLabel({ format: "hf-safetensors", quantize: "nvfp4" }), "Safetensors · NVFP4 load (lossy)");
  assert.equal(modelSubtitle({ format: "hf-safetensors", quantize: "nvfp4", sourceBits: 16 }), "Safetensors · NVFP4 load (lossy)");
  assert.equal(modelWeightLabel({ format: "hf-safetensors", quantize: "q8" }).includes("lossy"), false);
});

test("a load that unloaded the served model first says so", () => {
  assert.equal(loadNotice("Qwen3-8B", { loaded: {} }), "Qwen3-8B is now the served model.");
  assert.equal(
    loadNotice("Qwen3.8-27B", { load_transition: { released: "Qwen3.8-27B", reason: "reload" } }),
    "Qwen3.8-27B reloaded. The served copy was unloaded first, so two copies never had to fit in memory.",
  );
  assert.equal(
    loadNotice("Qwen3-8B", { load_transition: { released: "Qwen3.8-27B", reason: "memory" } }),
    "Qwen3-8B is now the served model. Qwen3.8-27B was unloaded first because both did not fit in memory.",
  );
});
