import { parseNumber } from "../media/image";

export function buildLocalApiBase(serverStatus) {
  if (!serverStatus?.running) return "http://127.0.0.1:8000";
  const host = serverStatus.host === "0.0.0.0" || serverStatus.host === "::" ? "127.0.0.1" : serverStatus.host;
  return `http://${host}:${serverStatus.port}`;
}

export function supportsThinking(engineStatus) {
  return engineStatus?.loaded?.provider?.capabilities?.supports_thinking === true;
}

export function supportsVision(engineStatus) {
  return engineStatus?.loaded?.provider?.capabilities?.supports_vision === true;
}

export function supportsVideo(engineStatus) {
  return engineStatus?.loaded?.provider?.capabilities?.supports_video === true;
}

export function supportsTools(engineStatus) {
  return engineStatus?.loaded?.provider?.capabilities?.supports_tools === true;
}

/// Parse an OpenAI tool call's `arguments` (a JSON-encoded string) into an object for `execute_tool`.
export function parseToolArguments(raw) {
  if (raw && typeof raw === "object") return raw;
  if (typeof raw !== "string" || !raw.trim()) return {};
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

export function stripThinkBlocks(value) {
  return value.replace(/<think>[\s\S]*?<\/think>/gi, "").replace(/<think>[\s\S]*$/i, "").trimStart();
}

export function readSseMessages(buffer, onData) {
  let remaining = buffer;
  let index = remaining.indexOf("\n\n");
  while (index >= 0) {
    const rawEvent = remaining.slice(0, index);
    remaining = remaining.slice(index + 2);
    const data = rawEvent
      .split("\n")
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trim())
      .join("\n");
    if (data) onData(data);
    index = remaining.indexOf("\n\n");
  }
  return remaining;
}

/// Map an in-app message to the OpenAI wire shape: a `tool` result turn, an assistant turn carrying
/// `tool_calls` (content `null`), a vision turn with `image_url` / `video_url` parts, or a plain text
/// turn. Video parts (sc-8081) carry pre-sampled `frames` + per-frame `timestamps` (Text–Timestamp
/// Alignment); visuals come before text, matching the Qwen3-VL convention.
export function toOpenAiMessage({ role, content, images, videos, tool_calls: toolCalls }) {
  if (role === "tool") {
    return { role: "tool", content: content ?? "" };
  }
  if (role === "assistant" && toolCalls && toolCalls.length) {
    return {
      role: "assistant",
      content: content ? content : null,
      tool_calls: toolCalls.map((call, index) => ({
        id: call.id ?? `call_${index}`,
        type: "function",
        function: {
          name: call.name,
          arguments:
            typeof call.arguments === "string"
              ? call.arguments
              : JSON.stringify(call.arguments ?? {}),
        },
      })),
    };
  }
  if ((images && images.length) || (videos && videos.length)) {
    const parts = [];
    for (const url of images ?? []) parts.push({ type: "image_url", image_url: { url } });
    for (const video of videos ?? []) {
      parts.push({
        type: "video_url",
        video_url: { frames: video.frames, timestamps: video.timestamps, fps: video.fps },
      });
    }
    if (content) parts.push({ type: "text", text: content });
    return { role, content: parts };
  }
  return { role, content };
}

/// Extract the plain-text content of an in-app message for Copy/Rewind (sc-8147). `content` is
/// normally a string, but a saved/loaded vision turn may carry an array of OpenAI content parts;
/// only `text` parts count as text (image_url/video_url parts are visual). Returns "" for
/// image/video-only turns so Copy/Rewind can be disabled on them (decision 7).
export function messageTextContent(message) {
  const content = message?.content;
  if (content == null) return "";
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => (part && part.type === "text" && typeof part.text === "string" ? part.text : ""))
      .join("\n");
  }
  return "";
}

export function chatRequestBody({ engineStatus, messages, params, thinkingCapable, tools }) {
  const requestMessages = [];
  if (params.systemPrompt.trim()) {
    requestMessages.push({ role: "system", content: params.systemPrompt.trim() });
  }
  requestMessages.push(...messages.map(toOpenAiMessage));
  const body = {
    model: engineStatus?.loaded?.name ?? "chatworks",
    messages: requestMessages,
    stream: true,
  };
  const temperature = parseNumber(params.temperature);
  const topP = parseNumber(params.topP);
  const maxTokens = parseNumber(params.maxTokens);
  if (temperature !== undefined) body.temperature = temperature;
  if (topP !== undefined) body.top_p = topP;
  if (maxTokens !== undefined) body.max_tokens = maxTokens;
  if (thinkingCapable) body.disable_thinking = params.disableThinking;
  if (tools && tools.length) body.tools = tools;
  return body;
}

/// POST one chat completion and consume its SSE stream, calling `onUpdate` as content/reasoning
/// arrive. Returns the final `{content, thinking, toolCalls, finishReason}`. The local server emits
/// each tool call whole in the final chunk's `delta.tool_calls`, so calls need no fragment assembly.
export async function streamChatCompletion({ url, headers, body, onUpdate }) {
  const response = await fetch(url, { method: "POST", headers, body: JSON.stringify(body) });
  if (!response.ok) {
    const errorBody = await response.json().catch(() => null);
    if (response.status === 413) {
      throw new Error("Image attachments are too large for the local OpenAI server.");
    }
    throw new Error(errorBody?.error?.message ?? `OpenAI API returned HTTP ${response.status}`);
  }
  if (!response.body) throw new Error("OpenAI API did not return a stream");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let content = "";
  let thinking = "";
  const toolCalls = [];
  let finishReason = null;
  let done = false;
  while (!done) {
    const chunk = await reader.read();
    done = chunk.done;
    buffer += decoder.decode(chunk.value ?? new Uint8Array(), { stream: !done });
    buffer = readSseMessages(buffer, (data) => {
      if (data === "[DONE]") return;
      const eventData = JSON.parse(data);
      if (eventData.error) throw new Error(eventData.error.message);
      const choice = eventData.choices?.[0];
      if (!choice) return;
      const callDeltas = choice.delta?.tool_calls;
      if (Array.isArray(callDeltas)) {
        for (const callDelta of callDeltas) {
          const fn = callDelta.function ?? {};
          toolCalls.push({ id: callDelta.id, name: fn.name ?? "", arguments: fn.arguments ?? "" });
        }
      }
      if (choice.finish_reason) finishReason = choice.finish_reason;
      const contentDelta = choice.delta?.content ?? "";
      const thinkingDelta = choice.delta?.reasoning_content ?? "";
      if (contentDelta || thinkingDelta) {
        content += contentDelta;
        thinking += thinkingDelta;
        onUpdate({ content, thinking });
      }
    });
  }
  return { content, thinking, toolCalls, finishReason };
}
