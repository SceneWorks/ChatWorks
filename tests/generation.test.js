import assert from "node:assert/strict";
import test from "node:test";
import { chatRequestBody, toOpenAiMessage } from "../src/api/sse.js";
import { paramsFromConversation, paramsToConversation } from "../src/state/conversations.js";
import { generationParams } from "../src/state/generation.js";

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

test("model defaults remain omitted and MTP defaults off", () => {
  const body = request({});
  for (const field of ["reasoning_effort", "preserve_thinking", "mtp", "top_k", "seed"]) {
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
