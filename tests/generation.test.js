import assert from "node:assert/strict";
import test from "node:test";
import { chatRequestBody, toOpenAiMessage } from "../src/api/sse.js";
import { paramsFromConversation, paramsToConversation } from "../src/state/conversations.js";
import {
  applySamplingPreset, clampSpeculativeDepth, generationParams, generationSettings, speculativeOptions,
} from "../src/state/generation.js";
import { appendAttachmentPlaceholders, settleAttachment, registerPreparation } from "../src/state/attachments.js";
import { isExactGgufUrl, modelSubtitle, unloadServedModel } from "../src/state/models.js";
import { prepareRemoteMedia } from "../src/state/media.js";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const capabilities = {
  supports_thinking: true,
  supports_reasoning_effort: true,
  supports_preserve_thinking: true,
  speculative: [
    { proposer: "mtp", max_depth: 7, recommended_depth: 3 },
    { proposer: "prompt_lookup", max_depth: 7, recommended_depth: 4 },
  ],
};
function request(params, caps = capabilities) {
  return chatRequestBody({
    engineStatus: { loaded: { name: "fixture", provider: { capabilities: caps } } },
    messages: [{ role: "user", content: "Hello" }],
    params: { systemPrompt: "", temperature: "0.7", topP: "0.9", maxTokens: "16",
      disableThinking: false, ...generationParams(), ...params },
    thinkingCapable: caps.supports_thinking,
  });
}

test("model defaults remain omitted and an unchosen speculative option is left to the backend", () => {
  const body = request({});
  // No UI constant: the server applies the saved setting (null there: the runtime's default).
  assert.equal(Object.hasOwn(body, "speculative"), false);
  assert.equal(Object.hasOwn(body, "mtp"), false, "the legacy field is never sent");
  assert.equal(request({ speculativeMode: "off" }).speculative, "off");
  assert.equal(request({ speculativeMode: "auto" }).speculative, "auto");
  assert.deepEqual(body.model_defaults, ["reasoning_effort", "preserve_thinking"]);
  for (const field of ["reasoning_effort", "preserve_thinking", "top_k", "seed"]) {
    assert.equal(Object.hasOwn(body, field), false, field);
  }
});

test("explicit native controls survive conversation persistence and request mapping", () => {
  const selected = { systemPrompt: "", temperature: "0.7", topP: "0.9", maxTokens: "16",
    disableThinking: false, reasoningEffort: "low", preserveThinking: "false",
    speculativeMode: "mtp", speculativeDepth: "5", topK: "20", presencePenalty: "1.5", repetitionPenalty: "1.1",
    repetitionContext: "64", seed: "42" };
  const saved = paramsToConversation(selected);
  assert.deepEqual(saved.speculative, { proposer: "mtp", depth: 5 });
  const restored = paramsFromConversation(saved);
  assert.deepEqual(restored, selected);
  const body = request(restored);
  assert.equal(body.reasoning_effort, "low");
  assert.equal(body.preserve_thinking, false);
  assert.deepEqual(body.speculative, { proposer: "mtp", depth: 5 });
  assert.equal(body.top_k, 20);
  assert.equal(body.presence_penalty, 1.5);
  assert.equal(body.repetition_penalty, 1.1);
  assert.equal(body.repetition_context, 64);
  assert.equal(body.seed, 42);
});

test("sc-24445: the depth is clamped to the proposer's advertised max_depth, and an unadvertised proposer is never sent", () => {
  assert.deepEqual(request({ speculativeMode: "prompt_lookup", speculativeDepth: "12" }).speculative,
    { proposer: "prompt_lookup", depth: 7 });
  assert.deepEqual(request({ speculativeMode: "prompt_lookup", speculativeDepth: "2" }).speculative,
    { proposer: "prompt_lookup", depth: 2 });
  assert.equal(Object.hasOwn(request({ speculativeMode: "draft_model", speculativeDepth: "2" }), "speculative"), false);
  assert.equal(clampSpeculativeDepth(0, { max_depth: 7 }), 1);
  assert.equal(clampSpeculativeDepth(9, { max_depth: 7 }), 7);
  assert.equal(clampSpeculativeDepth(9, null), 9, "no model in view: only the lower bound");
  assert.throws(() => generationSettings({ ...generationParams(), speculativeMode: "mtp", speculativeDepth: "0" }), /at least 1/);
});

test("sc-24445: the control offers the inherited default, off, auto and only the advertised proposers", () => {
  const values = (options) => options.map(([value]) => value);
  assert.deepEqual(values(speculativeOptions(capabilities)), ["", "off", "auto", "mtp", "prompt_lookup"]);
  assert.deepEqual(values(speculativeOptions({})), ["", "off", "auto"]);
  // Settings (no model in view) offers every proposer and names the runtime's default.
  const settings = speculativeOptions(null, "", true, "off");
  assert.deepEqual(values(settings), ["", "off", "auto", "mtp", "prompt_lookup", "draft_model"]);
  assert.equal(settings[0][1], "Runtime default (off)");
  // A saved proposer the model does not advertise stays visible, marked unavailable.
  const unavailable = speculativeOptions({}, "draft_model");
  assert.equal(unavailable.at(-1)[0], "draft_model");
  assert.match(unavailable.at(-1)[1], /not available for this model/);
  assert.deepEqual(generationParams({ speculative: { proposer: "prompt_lookup", depth: 6 } }).speculativeMode, "prompt_lookup");
  assert.equal(generationParams({ speculative: null }).speculativeMode, "");
});

test("off and auto reach any model; effort and preservation stay capability-gated", () => {
  const selected = { reasoningEffort: "xhigh", preserveThinking: "true", speculativeMode: "auto" };
  const unsupported = request(selected, { supports_thinking: true });
  for (const field of ["reasoning_effort", "preserve_thinking"]) {
    assert.equal(Object.hasOwn(unsupported, field), false);
  }
  // The runtime resolves `auto` to what the model offers (or plain decoding, named) itself.
  assert.equal(unsupported.speculative, "auto");
  assert.equal(Object.hasOwn(request({ ...selected, disableThinking: true }), "reasoning_effort"), false);
  assert.equal(request(selected).speculative, "auto");
});

test("assistant reasoning survives ordinary and tool-call history", () => {
  assert.deepEqual(toOpenAiMessage({ role: "assistant", content: "Answer", thinking: "Reason" }),
    { role: "assistant", content: "Answer", reasoning_content: "Reason" });
  const message = toOpenAiMessage({ role: "assistant", content: "", thinking: "Need tool",
    tool_calls: [{ name: "lookup", arguments: { query: "test" } }] });
  assert.equal(message.reasoning_content, "Need tool");
  assert.equal(message.tool_calls[0].function.arguments, '{"query":"test"}');
  assert.equal(Object.hasOwn(toOpenAiMessage({ role: "user", content: "Hi", thinking: "Ignore" }),
    "reasoning_content"), false);
});

test("mixed media preserves its attachment order on the OpenAI wire", () => {
  const message = toOpenAiMessage({
    role: "user",
    content: "Describe the sequence",
    media: [
      { type: "video", frames: ["data:image/jpeg;base64,AA=="], timestamps: [0], fps: 1 },
      { type: "image", url: "data:image/jpeg;base64,BB==" },
      { type: "video", frames: ["data:image/jpeg;base64,CC=="], timestamps: [2], fps: 1 },
    ],
  });
  assert.deepEqual(message.content.map((part) => part.type), ["video_url", "image_url", "video_url", "text"]);
  assert.deepEqual(message.content[0].video_url.timestamps, [0]);
  assert.equal(message.content[1].image_url.url, "data:image/jpeg;base64,BB==");
});

test("a seed beyond JavaScript precision cannot silently change the requested run", () => {
  assert.throws(() => request({ seed: "9007199254740993" }), /Seed must/);
  assert.throws(() => request({ seed: "-1" }), /Seed must/);
  assert.equal(request({ seed: "0" }).seed, 0);
});

test("model changes preserve explicit sampling until the user applies a recommendation", () => {
  const explicit = {
    temperature: "0.13", topP: "0.77", topK: "19", presencePenalty: "0.4",
    repetitionPenalty: "1.23", repetitionContext: "91", maxTokens: "42",
  };
  assert.deepEqual({ ...explicit }, explicit);
  const recommended = applySamplingPreset(explicit, {
    temperature: 0.6, top_p: 0.95, top_k: 40, presence_penalty: 1,
    repetition_penalty: 1.05, repetition_context: 128,
  });
  assert.equal(explicit.temperature, "0.13");
  assert.equal(recommended.temperature, "0.6");
  assert.equal(recommended.maxTokens, "42");
});

test("attachment placeholders preserve enqueue order when preparation resolves in reverse", () => {
  let attachments = appendAttachmentPlaceholders([], [
    { id: 1, type: "video", name: "slow.mp4" },
    { id: 2, type: "image", name: "fast.jpg" },
  ]);
  attachments = settleAttachment(attachments, 2, { type: "image", url: "image" });
  attachments = settleAttachment(attachments, 1, { type: "video", frames: ["video"], timestamps: [0], fps: 1 });
  assert.deepEqual(attachments.map((item) => item.type), ["video", "image"]);
  const wire = toOpenAiMessage({ role: "user", content: "ordered", media: attachments });
  assert.deepEqual(wire.content.map((part) => part.type), ["video_url", "image_url", "text"]);
});

test("packed GGUF imports are identified and labeled by their real format", () => {
  assert.equal(isExactGgufUrl("https://huggingface.co/prism/repo/blob/rev/PQ2_0.gguf"), true);
  assert.equal(modelSubtitle({ format: "gguf-prism-packed", pack: "bonsai2-packed", quantize: null }), "Bonsai 2 packed");
  assert.equal(modelSubtitle({ format: "gguf", pack: null, quantize: null }), "GGUF");
});

test("Windows sidecar output paths include the executable extension", () => {
  const root = fileURLToPath(new URL("..", import.meta.url));
  const output = execFileSync("bash", ["scripts/provision-ffmpeg-sidecars.sh", "--print-output-paths"], {
    cwd: root,
    env: { ...process.env, MEDIA_SIDECAR_TARGET: "x86_64-pc-windows-msvc" },
    encoding: "utf8",
  });
  assert.match(output, /ffmpeg-x86_64-pc-windows-msvc\.exe/);
  assert.match(output, /ffprobe-x86_64-pc-windows-msvc\.exe/);
});

test("remote UI media is routed through native staging without browser fetch or decode", async () => {
  const calls = [];
  const prepared = await prepareRemoteMedia(async (command, payload) => {
    calls.push({ command, payload });
    if (command === "begin_media_preparation") return "1";
    return { type: "image", url: "data:image/jpeg;base64,AA==" };
  }, "https://media.example/no-cors.jpg", "image");
  assert.deepEqual(calls, [{ command: "begin_media_preparation", payload: undefined }, {
    command: "prepare_remote_media",
    payload: { id: "1", source: "https://media.example/no-cors.jpg", kind: "image" },
  }]);
  assert.equal(prepared.url, "data:image/jpeg;base64,AA==");
});

test("desktop wire fixture explicitly clears global controls and leaves an unchosen speculative option to the backend after restore", async () => {
  const { readFile } = await import("node:fs/promises");
  const fixture = JSON.parse(await readFile(new URL("generation-wire.json", import.meta.url)));
  const saved = paramsToConversation({ ...generationParams(), disableThinking: true });
  // Nothing pins a speculative option the user never chose: the inherited default fills it in.
  const savedWire = JSON.parse(JSON.stringify(saved));
  assert.equal(Object.hasOwn(savedWire, "speculative"), false);
  assert.equal(Object.hasOwn(savedWire, "mtpMode"), false);
  const restored = paramsFromConversation(saved);
  const body = request(restored);
  const { model_defaults: modelDefaults, ...rest } = body;
  assert.deepEqual({ model_defaults: modelDefaults, ...(Object.hasOwn(rest, "speculative") ? { speculative: rest.speculative } : {}) }, fixture);
  assert.equal(body.disable_thinking, true);
  // An explicit Off survives persistence.
  const off = paramsFromConversation(paramsToConversation({ ...generationParams(), speculativeMode: "off" }));
  assert.equal(request(off).speculative, "off");
});

test("cancel before native registration finishes never starts preparation", async () => {
  const controller = new AbortController();
  const calls = [];
  let register;
  const result = prepareRemoteMedia((command, payload) => {
    calls.push([command, payload]);
    if (command === "begin_media_preparation") return new Promise((resolve) => { register = resolve; });
    return Promise.resolve();
  }, "https://example.com/video.mp4", "video", controller.signal);
  controller.abort(); register("operation");
  await assert.rejects(result, { name: "AbortError" });
  assert.deepEqual(calls.map(([command]) => command), ["begin_media_preparation", "cancel_media_preparation"]);
});

test("removing a pending native operation cancels its own ID and settles once", async () => {
  const controller = new AbortController();
  let rejectWork;
  let cancelled = 0;
  const result = prepareRemoteMedia(async (command, payload) => {
    if (command === "begin_media_preparation") return "operation";
    if (command === "prepare_remote_media") return new Promise((_, reject) => { rejectWork = reject; });
    assert.equal(payload.id, "operation"); cancelled += 1;
    rejectWork(new DOMException("cancelled", "AbortError"));
  }, "https://example.com/video.mp4", "video", controller.signal);
  await new Promise((resolve) => setImmediate(resolve));
  controller.abort(); controller.abort();
  await assert.rejects(result, { name: "AbortError" });
  assert.equal(cancelled, 1);
});


test("remove and conversation cleanup release pending slots once before late completion", () => {
  const operations = new Map();
  let pending = 2;
  const first = registerPreparation(operations, 1, () => pending--);
  const second = registerPreparation(operations, 2, () => pending--);
  first.cancel();
  assert.equal(pending, 1);
  assert.equal(first.controller.signal.aborted, true);
  for (const entry of operations.values()) entry.cancel();
  assert.equal(pending, 0); // Send is available even while old work's promise settles.
  first.release(); second.release();
  assert.equal(pending, 0);
  assert.equal(operations.size, 0);
});


test("unload refuses active UI generation and preserves errors without claiming refreshed status", async () => {
  const calls = [];
  const invoke = async (command) => { calls.push(command); if (command === "unload_model") throw new Error("unload failed"); };
  const refreshStatus = async () => { calls.push("refresh"); };
  await assert.rejects(unloadServedModel({ invoke, busy: true, refreshStatus }), /Stop generation/);
  assert.deepEqual(calls, []);
  await assert.rejects(unloadServedModel({ invoke, busy: false, refreshStatus }), /unload failed/);
  assert.deepEqual(calls, ["stop_generation", "unload_model"]);
  calls.length = 0;
  await unloadServedModel({ invoke: async (command) => { calls.push(command); }, busy: false, refreshStatus });
  assert.deepEqual(calls, ["stop_generation", "unload_model", "refresh"]);
});


test("browser video metadata wait aborts and releases its decoder source", async () => {
  const { sampleVideoAttachment } = await import("../src/media/video.js");
  const previous = globalThis.document;
  const actions = [];
  const video = { pause: () => actions.push("pause"), removeAttribute: (name) => actions.push(`remove:${name}`), load: () => actions.push("load") };
  globalThis.document = { createElement: () => video };
  try {
    const controller = new AbortController();
    const work = sampleVideoAttachment("https://example.com/stalled.mp4", controller.signal);
    controller.abort();
    await assert.rejects(work, { name: "AbortError" });
    assert.ok(actions.includes("pause"));
    assert.ok(actions.includes("remove:src"));
    assert.ok(actions.includes("load"));
  } finally { globalThis.document = previous; }
});

test("local video decoder failure names the file without a remote CORS explanation", async () => {
  const { sampleVideoAttachment } = await import("../src/media/video.js");
  const previousDocument = globalThis.document;
  const previousCreate = URL.createObjectURL;
  const previousRevoke = URL.revokeObjectURL;
  const revoked = [];
  const video = {
    pause() {}, removeAttribute() {}, load() {},
    set src(value) { if (value) queueMicrotask(() => this.onerror?.()); },
  };
  globalThis.document = { createElement: () => video };
  URL.createObjectURL = () => "blob:chatworks-local-fixture";
  URL.revokeObjectURL = (value) => revoked.push(value);
  try {
    await assert.rejects(sampleVideoAttachment({ name: "clip with space.mp4" }), (error) =>
      error.message.includes("clip with space.mp4") && !error.message.includes("CORS"));
    assert.deepEqual(revoked, ["blob:chatworks-local-fixture"]);
  } finally {
    globalThis.document = previousDocument;
    URL.createObjectURL = previousCreate;
    URL.revokeObjectURL = previousRevoke;
  }
});
