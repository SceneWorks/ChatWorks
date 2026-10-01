import assert from "node:assert/strict";
import test from "node:test";
import {
  draftModelOptions, loadNotice, modelSubtitle, modelWeightLabel, prefixCacheBytesFromField, prefixCacheField,
  servedWithLoadOptions,
} from "./models.js";

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

test("epic sc-24432 load options: the prefix-cache field maps MiB to the load's byte budget", () => {
  assert.equal(prefixCacheBytesFromField(""), null, "blank keeps the runtime's default");
  assert.equal(prefixCacheBytesFromField("0"), 0, "0 turns the cache off");
  assert.equal(prefixCacheBytesFromField("512"), 512 * 1024 * 1024);
  assert.throws(() => prefixCacheBytesFromField("-1"), /at least 0/);
  assert.throws(() => prefixCacheBytesFromField("lots"), /at least 0/);
  assert.equal(prefixCacheField(null), "");
  assert.equal(prefixCacheField(0), "0");
  assert.equal(prefixCacheField(512 * 1024 * 1024), "512");
});

test("epic sc-24432 load options: draft choices are the other registered models, and a reload is offered when they change", () => {
  const target = { id: "big", name: "Qwen3-32B", localPath: "/m/qwen3-32b" };
  const models = [target, { id: "small", name: "Qwen3-0.6B", localPath: "/m/qwen3-0.6b" }];
  assert.deepEqual(draftModelOptions(models, target), [["", "No draft model"], ["/m/qwen3-0.6b", "Qwen3-0.6B"]]);
  const stale = { ...target, draftSource: "/gone/tiny-draft" };
  assert.deepEqual(draftModelOptions(models, stale).at(-1), ["/gone/tiny-draft", "tiny-draft"]);
  const loaded = { draft_source: "/m/qwen3-0.6b", prefix_cache_bytes: 0 };
  assert.equal(servedWithLoadOptions({ ...target, draftSource: "/m/qwen3-0.6b", prefixCacheBytes: 0 }, loaded), true);
  assert.equal(servedWithLoadOptions({ ...target, prefixCacheBytes: 0 }, loaded), false, "draft removed");
  assert.equal(servedWithLoadOptions({ ...target, draftSource: "/m/qwen3-0.6b" }, loaded), false, "budget changed");
  assert.equal(servedWithLoadOptions(target, {}), true, "no options either side");
});
