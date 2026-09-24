import assert from "node:assert/strict";
import test from "node:test";
import { chatRequestBody, toOpenAiMessage } from "../src/api/sse.js";
import { paramsFromConversation, paramsToConversation } from "../src/state/conversations.js";
import { applySamplingPreset, generationParams } from "../src/state/generation.js";
import { appendAttachmentPlaceholders, settleAttachment, registerPreparation } from "../src/state/attachments.js";
import { isExactGgufUrl, modelSubtitle, unloadServedModel } from "../src/state/models.js";
import { prepareRemoteMedia } from "../src/state/media.js";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const capabilities = {
  supports_thinking: true,
  supports_reasoning_effort: true,
  supports_preserve_thinking: true,
  mtp: { max_draft_tokens: 8, recommended_draft_tokens: 3 },
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

test("model defaults remain omitted and an unknown MTP mode is left to the backend", () => {
  const body = request({});
  // No UI constant: the server applies the saved setting (the build's default: auto on CUDA).
  assert.equal(Object.hasOwn(body, "mtp"), false);
  assert.deepEqual(request({ mtpMode: "off" }).mtp, { mode: "off" });
  assert.deepEqual(request({ mtpMode: "auto" }).mtp, { mode: "auto" });
  assert.deepEqual(body.model_defaults, ["reasoning_effort", "preserve_thinking"]);
  for (const field of ["reasoning_effort", "preserve_thinking", "top_k", "seed"]) {
    assert.equal(Object.hasOwn(body, field), false, field);
  }
});

test("explicit native controls survive conversation persistence and request mapping", () => {
  const selected = { systemPrompt: "", temperature: "0.7", topP: "0.9", maxTokens: "16",
    disableThinking: false, reasoningEffort: "low", preserveThinking: "false",
    mtpMode: "enabled", mtpDraftTokens: "5", topK: "20", presencePenalty: "1.5", repetitionPenalty: "1.1",
    repetitionContext: "64", seed: "42" };
  const restored = paramsFromConversation(paramsToConversation(selected));
  assert.deepEqual(restored, selected);
  const body = request(restored);
  assert.equal(body.reasoning_effort, "low");
  assert.equal(body.preserve_thinking, false);
  assert.deepEqual(body.mtp, { mode: "enabled", draft_tokens: 5 });
  assert.equal(body.top_k, 20);
  assert.equal(body.presence_penalty, 1.5);
  assert.equal(body.repetition_penalty, 1.1);
  assert.equal(body.repetition_context, 64);
  assert.equal(body.seed, 42);
});

test("thinking alone does not advertise effort, preservation, or MTP", () => {
  const selected = { reasoningEffort: "xhigh", preserveThinking: "true", mtpMode: "auto" };
  const unsupported = request(selected, { supports_thinking: true });
  for (const field of ["reasoning_effort", "preserve_thinking", "mtp"]) {
    assert.equal(Object.hasOwn(unsupported, field), false);
  }
  assert.equal(Object.hasOwn(request({ ...selected, disableThinking: true }), "reasoning_effort"), false);
  assert.deepEqual(request(selected).mtp, { mode: "auto" });
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

test("desktop wire fixture explicitly clears global controls and leaves an unknown MTP mode to the backend after restore", async () => {
  const { readFile } = await import("node:fs/promises");
  const fixture = JSON.parse(await readFile(new URL("generation-wire.json", import.meta.url)));
  const saved = paramsToConversation({ ...generationParams(), disableThinking: true });
  // Nothing pins a speculative mode the user never chose: the backend's default fills it in.
  assert.equal(Object.hasOwn(JSON.parse(JSON.stringify(saved)), "mtpMode"), false);
  const restored = paramsFromConversation(saved);
  const body = request(restored);
  const { model_defaults: modelDefaults, ...rest } = body;
  assert.deepEqual({ model_defaults: modelDefaults, ...(Object.hasOwn(rest, "mtp") ? { mtp: rest.mtp } : {}) }, fixture);
  assert.equal(body.disable_thinking, true);
  // An explicit Off survives persistence.
  const off = paramsFromConversation(paramsToConversation({ ...generationParams(), mtpMode: "off" }));
  assert.deepEqual(request(off).mtp, { mode: "off" });
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
