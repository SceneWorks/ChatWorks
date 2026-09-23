import assert from "node:assert/strict";
import test from "node:test";
import { modelSubtitle, modelWeightLabel } from "./models.js";

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
