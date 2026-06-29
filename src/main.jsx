import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import ReactDOM from "react-dom/client";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import "@sceneworks/ui/theme.css";
import "@sceneworks/ui/shell.css";
import {
  ACCENTS,
  CompactSelector,
  DEFAULT_ACCENT,
  ErrorBoundary,
  Icon,
  Logo,
  Markdown,
  StatusDot,
} from "@sceneworks/ui";
import "./styles.css";

const AppContext = createContext(null);

const DEFAULT_APP_SETTINGS = {
  server: {
    host: "127.0.0.1",
    port: 8000,
    allowLan: false,
    authEnabled: false,
  },
  sampling: {
    systemPrompt: "You are a helpful local assistant.",
    temperature: 0.7,
    topP: 0.9,
    maxTokens: 512,
    disableThinking: true,
  },
};

const VIEWS = {
  Chat: {
    title: "Chat",
    blurb: "Talk to the currently served local model.",
  },
  Models: {
    title: "Models",
    blurb: "Import, convert, and select the one model ChatWorks serves.",
  },
  Settings: {
    title: "Settings",
    blurb: "Configure the LAN API, auth, and default sampling profile.",
  },
};

const navSections = [
  {
    label: "Serve",
    items: [
      { id: "Chat", icon: Icon.Sparkle, label: "Chat" },
      { id: "Models", icon: Icon.Model, label: "Models" },
    ],
  },
  {
    label: "App",
    items: [{ id: "Settings", icon: Icon.Sliders, label: "Settings" }],
  },
];

function readStoredValue(key, fallback) {
  try {
    return window.localStorage.getItem(key) ?? fallback;
  } catch {
    return fallback;
  }
}

function AppProvider({ children }) {
  const [activeView, setActiveView] = useState(() => readStoredValue("chatworks-active-view", "Chat"));
  const [theme, setTheme] = useState(() => readStoredValue("chatworks-theme", "dark"));
  const [accent, setAccent] = useState(() => readStoredValue("chatworks-accent", DEFAULT_ACCENT));
  const [engineStatus, setEngineStatus] = useState(null);
  const [appSettings, setAppSettings] = useState(DEFAULT_APP_SETTINGS);
  const [apiAuthToken, setApiAuthToken] = useState(null);

  const refreshAppSettings = useCallback(() => {
    return Promise.all([
      invoke("load_app_settings"),
      invoke("api_auth_token").catch(() => null),
    ])
      .then(([settings, token]) => {
        setAppSettings(settings);
        setApiAuthToken(token);
        return settings;
      })
      .catch(() => DEFAULT_APP_SETTINGS);
  }, []);

  const refreshEngineStatus = useCallback(() => {
    return invoke("engine_status")
      .then((status) => {
        setEngineStatus(status);
        return status;
      })
      .catch(() => {
        setEngineStatus(null);
        return null;
      });
  }, []);

  useEffect(() => {
    const nextView = VIEWS[activeView] ? activeView : "Chat";
    if (nextView !== activeView) setActiveView(nextView);
  }, [activeView]);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    window.localStorage.setItem("chatworks-theme", theme);
  }, [theme]);

  useEffect(() => {
    document.documentElement.setAttribute("data-accent", accent);
    window.localStorage.setItem("chatworks-accent", accent);
  }, [accent]);

  useEffect(() => {
    window.localStorage.setItem("chatworks-active-view", activeView);
  }, [activeView]);

  useEffect(() => {
    refreshEngineStatus();
    refreshAppSettings();
  }, [refreshAppSettings, refreshEngineStatus]);

  const value = useMemo(
    () => ({
      activeView,
      setActiveView,
      theme,
      setTheme,
      accent,
      setAccent,
      engineStatus,
      refreshEngineStatus,
      appSettings,
      setAppSettings,
      apiAuthToken,
      setApiAuthToken,
      refreshAppSettings,
    }),
    [accent, activeView, apiAuthToken, appSettings, engineStatus, refreshAppSettings, refreshEngineStatus, theme],
  );

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

function useApp() {
  const context = useContext(AppContext);
  if (!context) throw new Error("useApp must be used inside AppProvider");
  return context;
}

function paramsFromSettings(sampling) {
  return {
    systemPrompt: sampling.systemPrompt ?? "",
    temperature: String(sampling.temperature ?? ""),
    topP: String(sampling.topP ?? ""),
    maxTokens: String(sampling.maxTokens ?? ""),
    disableThinking: Boolean(sampling.disableThinking),
  };
}

/// Cap for frontend-derived conversation titles. Matches the backend preview cap
/// (conversations.rs `PREVIEW_MAX_CHARS`) so a frontend-derived title and the server-derived
/// preview/title stay byte-consistent.
const CONVERSATION_TITLE_MAX_CHARS = 80;

/// Convert the frontend's per-session `params` — stringified numbers bound to the Sampling inputs —
/// into the typed `ConversationParams` shape the Tauri `save_conversation` command expects
/// (systemPrompt: String, temperature/topP: f32, maxTokens: u32, disableThinking: bool). The Rust
/// serde types reject JSON strings, so the conversion is mandatory before persisting.
function paramsToConversation(params) {
  return {
    systemPrompt: params.systemPrompt ?? "",
    temperature: parseNumber(params.temperature) ?? 0,
    topP: parseNumber(params.topP) ?? 0,
    maxTokens: parseNumber(params.maxTokens) ?? 0,
    disableThinking: Boolean(params.disableThinking),
  };
}

/// Inverse of `paramsToConversation`: restore typed conversation params back into the string-input
/// shape the Sampling panel binds to. Mirrors `paramsFromSettings` so a loaded conversation drops
/// into the panel exactly like a fresh chat seeded from the app defaults.
function paramsFromConversation(params) {
  const p = params ?? {};
  return {
    systemPrompt: p.systemPrompt ?? "",
    temperature: String(p.temperature ?? ""),
    topP: String(p.topP ?? ""),
    maxTokens: String(p.maxTokens ?? ""),
    disableThinking: Boolean(p.disableThinking),
  };
}

/// Derive a conversation title from the first user message: collapse whitespace and cap at
/// `CONVERSATION_TITLE_MAX_CHARS` code points with an ellipsis. This is the lazy-save title used on
/// the first send of a new chat; once set it is preserved across upserts.
function deriveConversationTitle(messages) {
  for (const message of messages ?? []) {
    if (message?.role !== "user") continue;
    const text = String(message.content ?? "").replace(/\s+/g, " ").trim();
    if (!text) continue;
    return truncateForTitle(text, CONVERSATION_TITLE_MAX_CHARS);
  }
  return "New conversation";
}

function truncateForTitle(text, maxChars) {
  const chars = Array.from(text);
  if (chars.length <= maxChars) return text;
  return `${chars.slice(0, maxChars).join("")}\u{2026}`;
}

const ConversationsContext = createContext(null);
const ChatStateContext = createContext(null);

/// Owns the full conversation lifecycle and the per-chat ephemeral state, calling story A's five
/// Tauri commands (`list_conversations`, `get_conversation`, `save_conversation`,
/// `rename_conversation`, `delete_conversation`) via `invoke`.
///
/// The state is split across two contexts on purpose so the high-frequency transcript updates
/// (streaming tokens) do not re-render subscribers that only care about the metadata cache / active
/// id (e.g. the history nav in story C):
///   - `ConversationsContext` (rarely changes): active id, metadata cache, and the lifecycle
///     actions (`selectConversation`, `startNewChat`, `persistConversation`, `renameConversation`,
///     `deleteConversation`, `refreshConversations`).
///   - `ChatStateContext` (changes every token): `messages`, `draft`, `params`, `attachments`,
///     `videoAttachments` and their setters.
///
/// App start opens a fresh, unsaved new chat (activeConversationId === null) and loads the history
/// metadata cache independently — there is no auto-resume.
function ConversationsProvider({ children }) {
  const { appSettings } = useApp();
  const defaultParams = useMemo(() => paramsFromSettings(appSettings.sampling), [appSettings]);

  const [activeConversationId, setActiveConversationId] = useState(null);
  const [conversations, setConversations] = useState([]);
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState("");
  const [params, setParams] = useState(defaultParams);
  const [attachments, setAttachments] = useState([]);
  const [videoAttachments, setVideoAttachments] = useState([]);
  // `busy` is the active-stream flag. It lives here (not in ChatScreen) so the history nav — a
  // sibling of ChatScreen in the shell — can read it and hard-block conversation switching while a
  // response is streaming (story C). It only flips at stream boundaries, so exposing it through
  // ConversationsContext does not add per-token re-renders to nav subscribers.
  const [busy, setBusy] = useState(false);

  // Refs let the action callbacks read the latest state without depending on it, which keeps the
  // ConversationsContext value referentially stable across streaming-driven `messages` updates.
  const activeIdRef = useRef(activeConversationId);
  const conversationsRef = useRef(conversations);
  const messagesRef = useRef(messages);
  const paramsRef = useRef(params);
  const busyRef = useRef(busy);
  useEffect(() => {
    activeIdRef.current = activeConversationId;
  }, [activeConversationId]);
  useEffect(() => {
    conversationsRef.current = conversations;
  }, [conversations]);
  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);
  useEffect(() => {
    paramsRef.current = params;
  }, [params]);
  useEffect(() => {
    busyRef.current = busy;
  }, [busy]);

  // Metadata cache (from `list_conversations`) so the nav renders without a refetch; refreshed
  // after every save/rename/delete.
  const refreshConversations = useCallback(() => {
    return invoke("list_conversations")
      .then((list) => {
        setConversations(Array.isArray(list) ? list : []);
        return list;
      })
      .catch(() => {
        setConversations([]);
        return [];
      });
  }, []);

  useEffect(() => {
    refreshConversations();
  }, [refreshConversations]);

  // A fresh new chat tracks the app default sampling profile; a loaded conversation owns the params
  // it was run with, so defaults are only re-applied while there is no active conversation.
  useEffect(() => {
    if (activeConversationId === null) {
      setParams(defaultParams);
    }
  }, [defaultParams, activeConversationId]);

  /// Reset to a fresh, unsaved new chat: clears messages/draft/attachments, clears the active id,
  /// and resets params to the app defaults. The saved history is untouched.
  const startNewChat = useCallback(() => {
    setActiveConversationId(null);
    setMessages([]);
    setDraft("");
    setAttachments([]);
    setVideoAttachments([]);
    setParams(paramsFromSettings(appSettings.sampling));
  }, [appSettings]);

  /// Load a conversation: `get_conversation(id)` → messages into the transcript + params restored
  /// into the Sampling panel, and set as active. Throws on failure so the caller (story C) can
  /// surface the error. Hard-blocks while a response is streaming — switching the transcript
  /// mid-stream would discard the in-flight assistant turn; the nav also disables row selection
  /// while busy, so this is a defensive backstop.
  const selectConversation = useCallback(async (id) => {
    if (busyRef.current) return;
    const conversation = await invoke("get_conversation", { id });
    setActiveConversationId(conversation.id);
    setMessages(Array.isArray(conversation.messages) ? conversation.messages : []);
    setDraft("");
    setAttachments([]);
    setVideoAttachments([]);
    setParams(paramsFromConversation(conversation.params));
    return conversation;
  }, []);

  /// Persist the active conversation. On the first send of a new chat this lazily creates it with a
  /// `crypto.randomUUID()` id, a title derived from the first user message, and the active params;
  /// on subsequent turns / rewind it upserts the same id and the backend bumps `updatedAt`.
  /// Callers (the send loop, story D's rewind) should pass `messages`/`params` explicitly so the
  /// committed transcript is captured even when React state has not flushed yet; otherwise the
  /// latest state (via refs) is used. Always refreshes the metadata cache on success.
  const persistConversation = useCallback(async ({ id, messages: msgs, params: p, title } = {}) => {
    const resolvedId = id ?? activeIdRef.current ?? crypto.randomUUID();
    const finalMessages = msgs ?? messagesRef.current;
    const finalParams = p ?? paramsRef.current;
    const existing = conversationsRef.current.find((entry) => entry.id === resolvedId);
    const finalTitle = title ?? existing?.title ?? deriveConversationTitle(finalMessages);
    const record = {
      id: resolvedId,
      title: finalTitle,
      createdAt: 0,
      updatedAt: 0,
      params: paramsToConversation(finalParams),
      messages: finalMessages,
    };
    const saved = await invoke("save_conversation", { conversation: record });
    setActiveConversationId(saved.id);
    await refreshConversations();
    return saved;
  }, [refreshConversations]);

  const renameConversation = useCallback(
    async (id, title) => {
      const meta = await invoke("rename_conversation", { id, title });
      await refreshConversations();
      return meta;
    },
    [refreshConversations],
  );

  const deleteConversation = useCallback(
    async (id) => {
      await invoke("delete_conversation", { id });
      if (id === activeIdRef.current) {
        startNewChat();
      }
      await refreshConversations();
    },
    [refreshConversations, startNewChat],
  );

  const conversationsValue = useMemo(
    () => ({
      activeConversationId,
      conversations,
      busy,
      setBusy,
      selectConversation,
      startNewChat,
      persistConversation,
      renameConversation,
      deleteConversation,
      refreshConversations,
    }),
    [
      activeConversationId,
      conversations,
      busy,
      setBusy,
      selectConversation,
      startNewChat,
      persistConversation,
      renameConversation,
      deleteConversation,
      refreshConversations,
    ],
  );

  const chatStateValue = useMemo(
    () => ({
      messages,
      setMessages,
      draft,
      setDraft,
      params,
      setParams,
      attachments,
      setAttachments,
      videoAttachments,
      setVideoAttachments,
    }),
    [messages, draft, params, attachments, videoAttachments],
  );

  return (
    <ConversationsContext.Provider value={conversationsValue}>
      <ChatStateContext.Provider value={chatStateValue}>{children}</ChatStateContext.Provider>
    </ConversationsContext.Provider>
  );
}

function useConversations() {
  const context = useContext(ConversationsContext);
  if (!context) throw new Error("useConversations must be used inside ConversationsProvider");
  return context;
}

function useChatState() {
  const context = useContext(ChatStateContext);
  if (!context) throw new Error("useChatState must be used inside ConversationsProvider");
  return context;
}

function buildLocalApiBase(serverStatus) {
  if (!serverStatus?.running) return "http://127.0.0.1:8000";
  const host = serverStatus.host === "0.0.0.0" || serverStatus.host === "::" ? "127.0.0.1" : serverStatus.host;
  return `http://${host}:${serverStatus.port}`;
}

function supportsThinking(engineStatus) {
  return engineStatus?.loaded?.provider?.capabilities?.supports_thinking === true;
}

function supportsVision(engineStatus) {
  return engineStatus?.loaded?.provider?.capabilities?.supports_vision === true;
}

function supportsVideo(engineStatus) {
  return engineStatus?.loaded?.provider?.capabilities?.supports_video === true;
}

function supportsTools(engineStatus) {
  return engineStatus?.loaded?.provider?.capabilities?.supports_tools === true;
}

/// Whether the loaded model can run a quantized KV cache (sc-8533). DISTINCT from weight quantization
/// (Q4/Q8): this compresses the per-step key/value cache at runtime, not the model weights at load.
/// The MLX llama backend advertises true for the generic causal decoder; candle and the hybrid
/// Qwen3.6 decoder advertise false, so the KV-cache-quant control stays hidden there.
function supportsKvCacheQuant(engineStatus) {
  return engineStatus?.loaded?.provider?.capabilities?.supports_kv_cache_quant === true;
}

/// The maximum number of model→tool→model round-trips in a single send, to bound runaway loops.
const MAX_TOOL_STEPS = 8;
const IMAGE_ATTACHMENT_MAX_DIMENSION = 1536;
const IMAGE_ATTACHMENT_MAX_BYTES = 8 * 1024 * 1024;
const IMAGE_ATTACHMENT_QUALITY_STEPS = [0.86, 0.76, 0.66, 0.56];

// Video frame sampling (sc-8081): the frontend samples a small number of evenly-spaced frames from
// an attached video client-side (no native decoder needed) and sends them as a `video_url` part with
// per-frame timestamps. Frames are downscaled like images. Keeping the count small bounds prompt size
// and keeps inference responsive.
const VIDEO_ATTACHMENT_MAX_FRAMES = 8;
const VIDEO_FRAME_MAX_DIMENSION = 768;
const VIDEO_FRAME_QUALITY = 0.7;

/// Parse an OpenAI tool call's `arguments` (a JSON-encoded string) into an object for `execute_tool`.
function parseToolArguments(raw) {
  if (raw && typeof raw === "object") return raw;
  if (typeof raw !== "string" || !raw.trim()) return {};
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

function stripThinkBlocks(value) {
  return value.replace(/<think>[\s\S]*?<\/think>/gi, "").replace(/<think>[\s\S]*$/i, "").trimStart();
}

function parseNumber(value) {
  if (value === "") return undefined;
  const number = Number(value);
  return Number.isFinite(number) ? number : undefined;
}

function canvasToBlob(canvas, type, quality) {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (blob) {
          resolve(blob);
        } else {
          reject(new Error("Could not encode image attachment."));
        }
      },
      type,
      quality,
    );
  });
}

function readBlobAsDataUrl(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("Could not read image attachment."));
    reader.readAsDataURL(blob);
  });
}

async function loadDrawableImage(file) {
  if (typeof createImageBitmap === "function") {
    try {
      const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
      return {
        source: bitmap,
        width: bitmap.width,
        height: bitmap.height,
        close: () => bitmap.close?.(),
      };
    } catch {
      // Fall through to the HTMLImageElement decoder for formats createImageBitmap cannot open.
    }
  }

  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () =>
      resolve({
        source: image,
        width: image.naturalWidth,
        height: image.naturalHeight,
        close: () => URL.revokeObjectURL(url),
      });
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error(`Could not decode ${file.name || "image attachment"}.`));
    };
    image.src = url;
  });
}

async function normalizeImageAttachment(file) {
  const drawable = await loadDrawableImage(file);
  try {
    const scale = Math.min(1, IMAGE_ATTACHMENT_MAX_DIMENSION / Math.max(drawable.width, drawable.height));
    const width = Math.max(1, Math.round(drawable.width * scale));
    const height = Math.max(1, Math.round(drawable.height * scale));
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("Could not prepare image attachment.");
    context.fillStyle = "#fff";
    context.fillRect(0, 0, width, height);
    context.drawImage(drawable.source, 0, 0, width, height);

    for (const quality of IMAGE_ATTACHMENT_QUALITY_STEPS) {
      const blob = await canvasToBlob(canvas, "image/jpeg", quality);
      if (blob.size <= IMAGE_ATTACHMENT_MAX_BYTES) return readBlobAsDataUrl(blob);
    }
  } finally {
    drawable.close?.();
  }

  throw new Error(`${file.name || "Image attachment"} is too large after compression.`);
}

/// Sample up to `VIDEO_ATTACHMENT_MAX_FRAMES` evenly-spaced frames from a video file, client-side,
/// using a hidden `<video>` element + canvas (no native decoder). Returns `{ frames, timestamps, fps
/// }` where `frames` are downscaled JPEG data URLs in temporal order and `timestamps` are the
/// wall-clock seconds of each sampled frame — exactly the `video_url` shape the local server expects
/// (sc-8081). `fps` is the *sampled* rate (frames per second over the captured span), forwarded so
/// the server can derive timestamps if needed.
async function sampleVideoAttachment(file) {
  const url = URL.createObjectURL(file);
  const video = document.createElement("video");
  video.preload = "auto";
  video.muted = true;
  video.playsInline = true;
  video.src = url;

  const ready = new Promise((resolve, reject) => {
    video.onloadedmetadata = () => resolve();
    video.onerror = () => reject(new Error(`Could not decode ${file.name || "video attachment"}.`));
  });

  try {
    await ready;
    const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 0;
    const count = Math.max(1, Math.min(VIDEO_ATTACHMENT_MAX_FRAMES, duration > 0 ? VIDEO_ATTACHMENT_MAX_FRAMES : 1));
    // Even sampling across the duration (midpoints of `count` equal segments) so frames span the clip.
    const times = duration > 0
      ? Array.from({ length: count }, (_, i) => ((i + 0.5) / count) * duration)
      : [0];

    const vw = video.videoWidth || 1;
    const vh = video.videoHeight || 1;
    const scale = Math.min(1, VIDEO_FRAME_MAX_DIMENSION / Math.max(vw, vh));
    const width = Math.max(1, Math.round(vw * scale));
    const height = Math.max(1, Math.round(vh * scale));
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("Could not prepare video frame canvas.");

    const seekTo = (t) =>
      new Promise((resolve, reject) => {
        const onSeeked = () => {
          video.removeEventListener("seeked", onSeeked);
          resolve();
        };
        video.addEventListener("seeked", onSeeked);
        video.onerror = () => reject(new Error("Could not seek video for frame sampling."));
        // Clamp to just inside the duration to avoid a seek past the end never firing `seeked`.
        video.currentTime = Math.min(t, Math.max(0, (duration || t) - 0.01));
      });

    const frames = [];
    const timestamps = [];
    for (const t of times) {
      await seekTo(t);
      context.drawImage(video, 0, 0, width, height);
      const blob = await canvasToBlob(canvas, "image/jpeg", VIDEO_FRAME_QUALITY);
      frames.push(await readBlobAsDataUrl(blob));
      timestamps.push(Number(video.currentTime.toFixed(3)));
    }
    const span = timestamps.length > 1 ? timestamps[timestamps.length - 1] - timestamps[0] : 0;
    const fps = span > 0 ? (timestamps.length - 1) / span : 1;
    return { frames, timestamps, fps };
  } finally {
    URL.revokeObjectURL(url);
  }
}

function readSseMessages(buffer, onData) {
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
function toOpenAiMessage({ role, content, images, videos, tool_calls: toolCalls }) {
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
function messageTextContent(message) {
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

function chatRequestBody({ engineStatus, messages, params, thinkingCapable, tools }) {
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
async function streamChatCompletion({ url, headers, body, onUpdate }) {
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

function MessageContent({ content, thinking, stripThinking }) {
  const visibleContent = stripThinking ? stripThinkBlocks(content) : content;
  if (!visibleContent.trim()) return <p className="thinking-hidden">Thinking hidden.</p>;
  return (
    <>
      {!stripThinking && thinking ? (
        <details className="thinking-block">
          <summary>Reasoning</summary>
          <Markdown content={thinking} />
        </details>
      ) : null}
      <Markdown content={visibleContent} />
    </>
  );
}

/// Pretty-print a tool call's arguments (a JSON string or object) for display.
function formatToolArguments(args) {
  const value = parseToolArguments(args);
  const text = JSON.stringify(value);
  return text === "{}" ? "" : JSON.stringify(value, null, 0);
}

/// Render the tool calls an assistant turn requested.
function ToolCallList({ calls }) {
  return (
    <div className="tool-calls">
      {calls.map((call, index) => (
        <div className="tool-call" key={index}>
          <span className="tool-call-icon" aria-hidden="true">🛠</span>
          <code className="tool-call-sig">
            {call.name}({formatToolArguments(call.arguments)})
          </code>
        </div>
      ))}
    </div>
  );
}

/// Render a tool-result turn (the executed output, an error, or a denial).
function ToolResult({ message }) {
  const status = message.denied ? "denied" : message.isError ? "error" : "ok";
  return (
    <div className={`tool-result tool-result-${status}`}>
      <div className="tool-result-head">
        {message.name ? <code>{message.name}</code> : null}
        <span className="tool-result-tag">{status}</span>
      </div>
      <pre className="tool-result-body">{message.content}</pre>
    </div>
  );
}

/// `Copy` and `Rewind` are not part of `@sceneworks/ui` (sc-8147). These inline SVGs mirror the
/// package's icon base (24×24 viewBox, `currentColor` stroke, round joins, strokeWidth 1.7) so they
/// render identically alongside `Icon.*` glyphs. A `Check` glyph backs the Copy "Copied" state.
function CopyIcon({ size = 18, ...rest }) {
  return (
    <svg
      aria-hidden="true"
      fill="none"
      height={size}
      stroke="currentColor"
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="1.7"
      viewBox="0 0 24 24"
      width={size}
      {...rest}
    >
      <path d="M9 4h6v2H9z M8 6h8a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2" />
    </svg>
  );
}

function RewindIcon({ size = 18, ...rest }) {
  return (
    <svg
      aria-hidden="true"
      fill="none"
      height={size}
      stroke="currentColor"
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="1.7"
      viewBox="0 0 24 24"
      width={size}
      {...rest}
    >
      <path d="M9 6L4 12l5 6 M4 12h9a6 6 0 0 1 6 6" />
    </svg>
  );
}

function CheckIcon({ size = 18, ...rest }) {
  return (
    <svg
      aria-hidden="true"
      fill="none"
      height={size}
      stroke="currentColor"
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="1.7"
      viewBox="0 0 24 24"
      width={size}
      {...rest}
    >
      <path d="M5 12l5 5L19 7" />
    </svg>
  );
}

/// Per-message Copy + Rewind actions rendered at the foot of every bubble (sc-8147). Copy writes the
/// message's text content to the clipboard (disabled on image/video-only turns); Rewind asks the
/// parent to drop this message and everything after it and load its text into the composer (disabled
/// mid-stream and on text-less turns). Extracted into its own component so the Copy button can hold
/// a local "Copied" confirmation without per-bubble state in the parent's message map.
function MessageActions({ message, index, onRewind, busy }) {
  const [copied, setCopied] = useState(false);
  const text = messageTextContent(message);
  const hasText = text.trim().length > 0;
  const canRewind = hasText && !busy;

  async function handleCopy() {
    if (!hasText) return;
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  }

  return (
    <div className="message-actions">
      <button
        type="button"
        className="message-action-btn"
        onClick={handleCopy}
        disabled={!hasText}
        aria-label="Copy message text"
        title={hasText ? "Copy message text" : "No text to copy"}
      >
        {copied ? <CheckIcon /> : <CopyIcon />}
      </button>
      <button
        type="button"
        className="message-action-btn"
        onClick={() => onRewind(index)}
        disabled={!canRewind}
        aria-label="Rewind to this message"
        title={
          busy ? "Wait for the response to finish" : hasText ? "Rewind to this message" : "No text to load"
        }
      >
        <RewindIcon />
      </button>
    </div>
  );
}

function ChatScreen() {
  const { engineStatus, refreshEngineStatus, appSettings, apiAuthToken } = useApp();
  const { activeConversationId, persistConversation, startNewChat, busy, setBusy } = useConversations();
  const {
    messages,
    setMessages,
    draft,
    setDraft,
    params,
    setParams,
    attachments,
    setAttachments,
    videoAttachments,
    setVideoAttachments,
  } = useChatState();
  const [serverStatus, setServerStatus] = useState(null);
  const [error, setError] = useState(null);
  const [pendingAttachments, setPendingAttachments] = useState(0);
  const [toolSpecs, setToolSpecs] = useState([]); // OpenAI function-tool defs from the backend
  const [toolsEnabled, setToolsEnabled] = useState(true); // offer tools when the model supports them
  const [pendingApproval, setPendingApproval] = useState(null); // {calls, decisions, resolve}
  const thinkingCapable = supportsThinking(engineStatus);
  const visionCapable = supportsVision(engineStatus);
  const videoCapable = supportsVideo(engineStatus);
  const toolsCapable = supportsTools(engineStatus);
  const apiBase = buildLocalApiBase(serverStatus);
  const canSend =
    Boolean(engineStatus?.loaded) &&
    !busy &&
    pendingAttachments === 0 &&
    (Boolean(draft.trim()) || attachments.length > 0 || videoAttachments.length > 0);

  // Load the built-in tool definitions once; the chat loop offers them when tools are enabled.
  useEffect(() => {
    invoke("list_builtin_tools")
      .then((specs) => setToolSpecs(Array.isArray(specs) ? specs : []))
      .catch(() => setToolSpecs([]));
  }, []);

  // Resolve the pending approval promise once every proposed call has an Approve/Deny decision.
  useEffect(() => {
    if (pendingApproval && pendingApproval.decisions.every((decision) => decision !== null)) {
      pendingApproval.resolve(pendingApproval.decisions);
      setPendingApproval(null);
    }
  }, [pendingApproval]);

  // Open the approval panel for a turn's tool calls; resolves with a per-call approve/deny array.
  const requestApproval = useCallback((calls) => {
    return new Promise((resolve) => {
      setPendingApproval({ calls, decisions: calls.map(() => null), resolve });
    });
  }, []);

  const decideApproval = useCallback((index, approved) => {
    setPendingApproval((current) => {
      if (!current) return current;
      const decisions = current.decisions.slice();
      decisions[index] = approved;
      return { ...current, decisions };
    });
  }, []);

  const refreshServerStatus = useCallback(() => {
    return invoke("openai_server_status")
      .then((status) => {
        setServerStatus(status);
        return status;
      })
      .catch(() => {
        setServerStatus(null);
        return null;
      });
  }, []);

  useEffect(() => {
    refreshServerStatus();
  }, [refreshServerStatus]);

  /// Rewind (sc-8147, decision 3): drop message `index` and every message after it, load that
  /// message's text into the composer, and persist the trimmed transcript so the trim survives a
  /// relaunch. Blocked mid-stream and on text-less turns (the buttons are disabled then; this is a
  /// defensive backstop). The in-memory trim always runs even when there is no saved conversation
  /// yet (unsaved new chat), matching the existing lazy-save semantics.
  const handleRewind = useCallback(
    (index) => {
      if (busy) return;
      const target = messages[index];
      const text = target ? messageTextContent(target).trim() : "";
      if (!text) return;
      const trimmed = messages.slice(0, index);
      setMessages(trimmed);
      setDraft(text);
      if (activeConversationId !== null) {
        persistConversation({ messages: trimmed }).catch((err) => {
          setError(err instanceof Error ? err.message : String(err));
        });
      }
    },
    [busy, messages, setMessages, setDraft, activeConversationId, persistConversation],
  );

  async function handleSubmit(eventArg) {
    eventArg.preventDefault();
    if (!canSend) return;
    setBusy(true);
    setError(null);
    const userMessage = {
      role: "user",
      content: draft.trim(),
      images: attachments,
      videos: videoAttachments,
    };
    // `conversation` is the committed transcript; the in-flight assistant turn is appended for
    // rendering and only committed once it finishes streaming.
    let conversation = [...messages, userMessage];
    setMessages(conversation);
    setDraft("");
    setAttachments([]);
    setVideoAttachments([]);

    // Lazy save / upsert: on the first send of a new chat this creates the conversation
    // (`save_conversation` with a `crypto.randomUUID()` id, a title derived from the first user
    // message, and the active per-session `params`); on subsequent commits it upserts the same id
    // and the backend bumps `updatedAt`. Capturing the user message here means the conversation
    // survives an interrupted stream. `conversationId` is tracked locally because the context's
    // `activeConversationId` does not flush within this handler.
    let conversationId = activeConversationId ?? crypto.randomUUID();
    const persist = async (transcript) => {
      try {
        const saved = await persistConversation({
          id: conversationId,
          messages: transcript,
          params,
        });
        conversationId = saved.id;
      } catch (cause) {
        // Persistence failures must not abort the in-flight chat; surface a soft error.
        setError(`Could not save conversation: ${String(cause?.message ?? cause)}`);
      }
    };
    await persist(conversation);

    try {
      const status = await refreshServerStatus();
      const nextEngineStatus = await refreshEngineStatus();
      const activeEngineStatus = nextEngineStatus ?? engineStatus;
      const url = `${buildLocalApiBase(status)}/v1/chat/completions`;
      const headers = { "Content-Type": "application/json" };
      if (appSettings.server.authEnabled && apiAuthToken) {
        headers.Authorization = `Bearer ${apiAuthToken}`;
      }
      const offerTools = toolsEnabled && supportsTools(activeEngineStatus) && toolSpecs.length > 0;

      let hitStepLimit = true;
      for (let step = 0; step < MAX_TOOL_STEPS; step += 1) {
        const committed = conversation;
        const assistantMessage = { role: "assistant", content: "", thinking: "" };
        setMessages([...committed, assistantMessage]);

        const result = await streamChatCompletion({
          url,
          headers,
          body: chatRequestBody({
            engineStatus: activeEngineStatus,
            messages: committed,
            params,
            thinkingCapable,
            tools: offerTools ? toolSpecs : null,
          }),
          onUpdate: ({ content, thinking }) =>
            setMessages([...committed, { ...assistantMessage, content, thinking }]),
        });

        // Commit the assistant turn (with any tool calls it requested).
        const finalAssistant = { role: "assistant", content: result.content, thinking: result.thinking };
        if (result.toolCalls.length) finalAssistant.tool_calls = result.toolCalls;
        conversation = [...committed, finalAssistant];
        setMessages(conversation);
        await persist(conversation);

        if (!result.toolCalls.length) {
          hitStepLimit = false;
          break;
        }

        // Human-in-the-loop: approve/deny each call, then execute the approved ones in the backend.
        const decisions = await requestApproval(result.toolCalls);
        const toolMessages = [];
        for (let i = 0; i < result.toolCalls.length; i += 1) {
          const call = result.toolCalls[i];
          if (!decisions[i]) {
            toolMessages.push({
              role: "tool",
              name: call.name,
              content: "Tool call denied by the user.",
              denied: true,
            });
            continue;
          }
          try {
            const output = await invoke("execute_tool", {
              name: call.name,
              arguments: parseToolArguments(call.arguments),
            });
            toolMessages.push({ role: "tool", name: call.name, content: String(output) });
          } catch (cause) {
            toolMessages.push({
              role: "tool",
              name: call.name,
              content: `Error: ${String(cause?.message ?? cause)}`,
              isError: true,
            });
          }
        }
        conversation = [...conversation, ...toolMessages];
        setMessages(conversation);
        await persist(conversation);
        // Loop: re-send the transcript (now with the tool results) for the model's next turn.
      }

      if (hitStepLimit) {
        setError(`Stopped after the tool-call step limit (${MAX_TOOL_STEPS}).`);
      }
    } catch (cause) {
      setError(String(cause?.message ?? cause));
      setMessages(conversation); // drop the in-flight assistant placeholder, keep committed turns
      await persist(conversation);
    } finally {
      setBusy(false);
    }
  }

  function updateParam(key, value) {
    setParams((current) => ({ ...current, [key]: value }));
  }

  function addImageFiles(fileList) {
    const files = Array.from(fileList || []).filter((file) => file && file.type.startsWith("image/"));
    if (!files.length) return;
    setError(null);
    setPendingAttachments((current) => current + files.length);
    files.forEach((file) => {
      normalizeImageAttachment(file)
        .then((url) => setAttachments((current) => [...current, url]))
        .catch((cause) => setError(String(cause?.message ?? cause)))
        .finally(() => setPendingAttachments((current) => Math.max(0, current - 1)));
    });
  }

  function addVideoFiles(fileList) {
    const files = Array.from(fileList || []).filter((file) => file && file.type.startsWith("video/"));
    if (!files.length) return;
    setError(null);
    setPendingAttachments((current) => current + files.length);
    files.forEach((file) => {
      sampleVideoAttachment(file)
        .then((sampled) =>
          setVideoAttachments((current) => [
            ...current,
            { name: file.name || "video", ...sampled },
          ]),
        )
        .catch((cause) => setError(String(cause?.message ?? cause)))
        .finally(() => setPendingAttachments((current) => Math.max(0, current - 1)));
    });
  }

  return (
    <section className="chat-layout">
      <div className="panel chat-panel">
        <div className="panel-head chat-head">
          <div>
            <p className="eyebrow">Streaming chat</p>
            <h2>{engineStatus?.loaded ? engineStatus.loaded.name : "Load a model to chat"}</h2>
            <p className="view-copy">Dogfoods {apiBase}/v1/chat/completions over SSE.</p>
          </div>
          <span className={serverStatus?.running ? "status-pill" : "status-pill warning"}>
            <StatusDot ok={Boolean(serverStatus?.running)} />
            {serverStatus?.running ? "API online" : "API offline"}
          </span>
        </div>

        <div className="message-list" aria-live="polite">
          {messages.length ? (
            messages.map((message, index) => {
              const hasToolCalls = Boolean(message.tool_calls && message.tool_calls.length);
              return (
                <article className={`message-bubble ${message.role}`} key={`${message.role}-${index}`}>
                  <div className="message-role">{message.role}</div>
                  {message.images && message.images.length ? (
                    <div className="message-images">
                      {message.images.map((url, imageIndex) => (
                        <img key={imageIndex} className="message-image" src={url} alt={`attachment ${imageIndex + 1}`} />
                      ))}
                    </div>
                  ) : null}
                  {message.videos && message.videos.length ? (
                    <div className="message-images">
                      {message.videos.map((video, videoIndex) => (
                        <img
                          key={videoIndex}
                          className="message-image"
                          src={video.frames?.[0]}
                          alt={`video ${videoIndex + 1} (${video.frames?.length ?? 0} frames)`}
                          title={`${video.frames?.length ?? 0} sampled frames`}
                        />
                      ))}
                    </div>
                  ) : null}
                  {message.role === "tool" ? (
                    <ToolResult message={message} />
                  ) : (
                    <>
                      {message.content || !hasToolCalls ? (
                        <MessageContent
                          content={message.content}
                          thinking={message.thinking}
                          stripThinking={thinkingCapable && params.disableThinking}
                        />
                      ) : null}
                      {hasToolCalls ? <ToolCallList calls={message.tool_calls} /> : null}
                    </>
                  )}
                  <MessageActions message={message} index={index} onRewind={handleRewind} busy={busy} />
                </article>
              );
            })
          ) : (
            <div className="empty-panel">Ask a question to start a multi-turn chat with the served model.</div>
          )}
        </div>

        {pendingApproval ? (
          <div className="tool-approval" role="alertdialog" aria-label="Approve tool calls">
            <p className="tool-approval-title">The model wants to run a tool. Approve to execute it locally.</p>
            {pendingApproval.calls.map((call, index) => (
              <div className="tool-approval-row" key={index}>
                <code className="tool-call-sig">
                  {call.name}({formatToolArguments(call.arguments)})
                </code>
                {pendingApproval.decisions[index] === null ? (
                  <span className="tool-approval-actions">
                    <button className="primary-btn" type="button" onClick={() => decideApproval(index, true)}>
                      Approve
                    </button>
                    <button className="ghost-btn" type="button" onClick={() => decideApproval(index, false)}>
                      Deny
                    </button>
                  </span>
                ) : (
                  <span className="tool-approval-decided">
                    {pendingApproval.decisions[index] ? "Approved" : "Denied"}
                  </span>
                )}
              </div>
            ))}
          </div>
        ) : null}

        {error ? <p className="form-error" role="alert">{error}</p> : null}

        <form className="composer" onSubmit={handleSubmit}>
          {visionCapable && attachments.length ? (
            <div className="composer-attachments">
              {attachments.map((url, index) => (
                <div className="composer-thumb" key={index}>
                  <img src={url} alt={`attachment ${index + 1}`} />
                  <button
                    type="button"
                    aria-label="Remove image"
                    onClick={() => setAttachments((current) => current.filter((_, i) => i !== index))}
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          ) : null}
          {videoCapable && videoAttachments.length ? (
            <div className="composer-attachments">
              {videoAttachments.map((video, index) => (
                <div className="composer-thumb composer-thumb-video" key={index}>
                  {/* First sampled frame as the video thumbnail; badge shows the frame count. */}
                  <img src={video.frames[0]} alt={`video ${index + 1}`} />
                  <span className="composer-thumb-badge">{video.frames.length}f</span>
                  <button
                    type="button"
                    aria-label="Remove video"
                    onClick={() =>
                      setVideoAttachments((current) => current.filter((_, i) => i !== index))
                    }
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          ) : null}
          <textarea
            disabled={!engineStatus?.loaded || busy}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                event.currentTarget.form?.requestSubmit();
              }
            }}
            onPaste={
              visionCapable
                ? (event) => {
                    const files = Array.from(event.clipboardData?.items ?? [])
                      .filter((item) => item.type.startsWith("image/"))
                      .map((item) => item.getAsFile());
                    if (files.length) {
                      event.preventDefault();
                      addImageFiles(files);
                    }
                  }
                : undefined
            }
            placeholder={engineStatus?.loaded ? "Message the local model…" : "Load a model from Models first"}
            rows={3}
            value={draft}
          />
          <div className="composer-actions">
            {visionCapable ? (
              <label className="ghost-btn" title="Attach image">
                <input
                  type="file"
                  accept="image/*"
                  multiple
                  style={{ display: "none" }}
                  disabled={!engineStatus?.loaded || busy}
                  onChange={(event) => {
                    addImageFiles(event.target.files);
                    event.target.value = "";
                  }}
                />
                {pendingAttachments ? "Preparing..." : "Image"}
              </label>
            ) : null}
            {videoCapable ? (
              <label className="ghost-btn" title="Attach video (sampled into frames)">
                <input
                  type="file"
                  accept="video/*"
                  style={{ display: "none" }}
                  disabled={!engineStatus?.loaded || busy}
                  onChange={(event) => {
                    addVideoFiles(event.target.files);
                    event.target.value = "";
                  }}
                />
                {pendingAttachments ? "Preparing..." : "Video"}
              </label>
            ) : null}
            <button className="primary-btn" disabled={!canSend} type="submit">
              {busy ? "Streaming…" : "Send"}
            </button>
          </div>
        </form>
      </div>

      <aside className="panel chat-settings">
        <div className="panel-head">
          <p className="eyebrow">Conversation overrides</p>
          <h2>Sampling</h2>
          <p className="view-copy">Applies only to this chat session.</p>
        </div>
        <div className="field">
          <label htmlFor="system-prompt">System prompt</label>
          <textarea
            id="system-prompt"
            onChange={(event) => updateParam("systemPrompt", event.target.value)}
            rows={5}
            value={params.systemPrompt}
          />
        </div>
        <div className="field-grid">
          <div className="field">
            <label htmlFor="temperature">Temperature</label>
            <input
              id="temperature"
              inputMode="decimal"
              onChange={(event) => updateParam("temperature", event.target.value)}
              type="number"
              step="0.1"
              min="0"
              max="2"
              value={params.temperature}
            />
          </div>
          <div className="field">
            <label htmlFor="top-p">Top P</label>
            <input
              id="top-p"
              inputMode="decimal"
              onChange={(event) => updateParam("topP", event.target.value)}
              type="number"
              step="0.05"
              min="0"
              max="1"
              value={params.topP}
            />
          </div>
          <div className="field">
            <label htmlFor="max-tokens">Max tokens</label>
            <input
              id="max-tokens"
              inputMode="numeric"
              onChange={(event) => updateParam("maxTokens", event.target.value)}
              type="number"
              step="1"
              min="1"
              value={params.maxTokens}
            />
          </div>
        </div>
        {thinkingCapable ? (
          <label className="toggle-row">
            <input
              checked={params.disableThinking}
              onChange={(event) => updateParam("disableThinking", event.target.checked)}
              type="checkbox"
            />
            <span>
              Disable thinking
              <small>No-think request flag and hidden &lt;think&gt; output.</small>
            </span>
          </label>
        ) : null}
        {toolsCapable ? (
          <label className="toggle-row">
            <input
              checked={toolsEnabled}
              onChange={(event) => setToolsEnabled(event.target.checked)}
              type="checkbox"
            />
            <span>
              Enable tools
              <small>
                Offer {toolSpecs.length} built-in tool{toolSpecs.length === 1 ? "" : "s"}; each call needs your approval.
              </small>
            </span>
          </label>
        ) : null}
        <button className="ghost-btn" disabled={busy || !messages.length} onClick={startNewChat} type="button">
          Clear conversation
        </button>
      </aside>
    </section>
  );
}

const QUANTIZE_OPTIONS = [
  { id: "dense", label: "Dense (full precision)", value: null },
  { id: "q4", label: "Quantize Q4", value: "q4" },
  { id: "q8", label: "Quantize Q8", value: "q8" },
];

/// KV-cache quantization (sc-8533), runtime KV-cache compression — kept SEPARATE from the weight
/// quantization (Q4/Q8) above. Currently RVQ is the only method; bit-width is one of {1, 2, 4}.
const KV_CACHE_QUANT_METHODS = [{ id: "rvq", label: "RVQ", value: "rvq" }];
const KV_CACHE_QUANT_BITS = [1, 2, 4];

function formatBytes(bytes) {
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

function modelSubtitle(model) {
  const parts = [];
  if (model.quantize === "q4") parts.push("Q4");
  else if (model.quantize === "q8") parts.push("Q8");
  else parts.push("Dense");
  if (model.sizeBytes) parts.push(formatBytes(model.sizeBytes));
  return parts.join(" · ");
}

/// Runtime KV-cache-quantization control (sc-8533) for the served model. Rendered ONLY when the
/// loaded model advertises `supports_kv_cache_quant` — hidden entirely for backends/models that do
/// not honor it (candle, the hybrid Qwen3.6 decoder). DISTINCT from the weight-quant (Q4/Q8) import
/// control above: this compresses the per-step KV cache at runtime via `set_model_kv_cache_quant`.
/// Default OFF (dense). Toggling off sends `null`, restoring the dense cache.
function KvCacheQuantPanel({ engineStatus, servedModelId, onApplied }) {
  const active = engineStatus?.loaded?.kv_cache_quant ?? null;
  const enabled = active !== null;
  // Form state seeds from the active setting (method/bits) and falls back to RVQ @ 4 bits.
  const [method, setMethod] = useState(active?.method ?? KV_CACHE_QUANT_METHODS[0].value);
  const [bits, setBits] = useState(active?.bits ?? 4);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  // Re-seed the form whenever the served model / its active KV-quant changes so the controls reflect
  // the real loaded state (e.g. after a model swap).
  useEffect(() => {
    setMethod(active?.method ?? KV_CACHE_QUANT_METHODS[0].value);
    setBits(active?.bits ?? 4);
    setError(null);
  }, [active?.method, active?.bits, servedModelId]);

  // Don't render unless a model is served AND it advertises KV-cache-quant support.
  if (!servedModelId || !supportsKvCacheQuant(engineStatus)) return null;

  async function apply(kvCacheQuant) {
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      await invoke("set_model_kv_cache_quant", { modelId: servedModelId, kvCacheQuant });
      await onApplied?.();
    } catch (cause) {
      setError(String(cause));
    } finally {
      setBusy(false);
    }
  }

  function handleToggle(event) {
    // Toggling ON applies the current method/bits; toggling OFF sends null (restores dense).
    apply(event.target.checked ? { method, bits } : null);
  }

  function handleMethod(value) {
    setMethod(value);
    if (enabled) apply({ method: value, bits });
  }

  function handleBits(value) {
    const nextBits = Number(value);
    setBits(nextBits);
    if (enabled) apply({ method, bits: nextBits });
  }

  return (
    <div className="panel">
      <div className="panel-head">
        <p className="eyebrow">Runtime</p>
        <h2>KV-cache quantization</h2>
        <p className="view-copy">
          Compress the per-step key/value cache during generation to fit longer contexts in memory.
          Separate from the model&apos;s weight quantization (Q4/Q8) — this is applied at runtime to the
          served model and reloads it with the new setting. Default off (dense cache).
        </p>
      </div>
      <label className="toggle-row">
        <input
          type="checkbox"
          checked={enabled}
          disabled={busy}
          onChange={handleToggle}
          aria-label="Enable KV-cache quantization"
        />
        <span>
          Quantize KV cache
          <small>{enabled ? "Enabled — running a quantized KV cache." : "Disabled — dense KV cache."}</small>
        </span>
      </label>
      {enabled ? (
        <>
          <div className="field">
            <label htmlFor="kv-quant-method">Method</label>
            <select
              id="kv-quant-method"
              value={method}
              disabled={busy}
              onChange={(event) => handleMethod(event.target.value)}
            >
              {KV_CACHE_QUANT_METHODS.map((option) => (
                <option key={option.id} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label htmlFor="kv-quant-bits">Bit-width</label>
            <select
              id="kv-quant-bits"
              value={String(bits)}
              disabled={busy}
              onChange={(event) => handleBits(event.target.value)}
            >
              {KV_CACHE_QUANT_BITS.map((option) => (
                <option key={option} value={String(option)}>
                  {option}-bit
                </option>
              ))}
            </select>
          </div>
        </>
      ) : null}
      {busy ? <p className="view-copy" aria-live="polite">Applying…</p> : null}
      {error ? <p className="form-error" role="alert">{error}</p> : null}
    </div>
  );
}

function ModelsScreen() {
  const { engineStatus, refreshEngineStatus } = useApp();
  const [registry, setRegistry] = useState({ models: [], selectedId: null });
  const [sourceUrl, setSourceUrl] = useState("");
  const [quantizeId, setQuantizeId] = useState("dense");
  const [tokenStatus, setTokenStatus] = useState({ present: false });
  const [tokenInput, setTokenInput] = useState("");
  const [progress, setProgress] = useState(null);
  const [busy, setBusy] = useState(false);
  const [cacheBusy, setCacheBusy] = useState(false);
  const [cachedModels, setCachedModels] = useState([]);
  const [adoptingPath, setAdoptingPath] = useState("");
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState(null);
  const [loadingId, setLoadingId] = useState("");

  const loadedSource = engineStatus?.loaded?.source ?? null;
  const selectedModel = registry.models.find((model) => model.id === registry.selectedId) ?? null;
  // The registry id of the currently served model (matched by source + weight-quant), used to key the
  // runtime KV-cache-quant command. `null` when nothing is served.
  const servedModelId =
    registry.models.find(
      (model) =>
        loadedSource &&
        model.localPath === loadedSource &&
        (model.quantize ?? null) === (engineStatus?.loaded?.quantize ?? null),
    )?.id ?? null;

  const refreshRegistry = useCallback(() => {
    return invoke("list_registered_models")
      .then((next) => {
        setRegistry(next);
        return next;
      })
      .catch((cause) => {
        setError(String(cause));
        return null;
      });
  }, []);

  // Re-pull the registry and engine status after a runtime KV-cache-quant change so the panel
  // reflects the reloaded model's active setting.
  const refreshServedModel = useCallback(async () => {
    await refreshRegistry();
    await refreshEngineStatus();
  }, [refreshRegistry, refreshEngineStatus]);

  useEffect(() => {
    refreshRegistry();
    invoke("hf_token_status")
      .then(setTokenStatus)
      .catch(() => setTokenStatus({ present: false }));
  }, [refreshRegistry]);

  useEffect(() => {
    const unlistenPromise = listen("models://import-progress", (event) => {
      setProgress(event.payload);
    });
    return () => {
      unlistenPromise.then((unlisten) => unlisten());
    };
  }, []);

  async function handleImport(eventArg) {
    eventArg.preventDefault();
    if (!sourceUrl.trim() || busy) return;
    setBusy(true);
    setError(null);
    setNotice(null);
    setProgress(null);
    const option = QUANTIZE_OPTIONS.find((item) => item.id === quantizeId) ?? QUANTIZE_OPTIONS[0];
    try {
      const next = await invoke("import_hf_model", {
        request: { sourceUrl: sourceUrl.trim(), quantize: option.value },
      });
      setRegistry(next);
      setNotice("Model imported and added to the registry.");
      setSourceUrl("");
    } catch (cause) {
      setError(String(cause));
    } finally {
      setBusy(false);
    }
  }

  async function handleScanCache() {
    if (cacheBusy) return;
    setCacheBusy(true);
    setError(null);
    setNotice(null);
    try {
      const models = await invoke("list_cached_hf_models");
      setCachedModels(models);
      setNotice(models.length ? `Found ${models.length} supported cached model${models.length === 1 ? "" : "s"}.` : "No supported cached HuggingFace models found.");
    } catch (cause) {
      setError(String(cause));
    } finally {
      setCacheBusy(false);
    }
  }

  async function handleAdoptCached(candidate) {
    if (adoptingPath) return;
    setAdoptingPath(candidate.localPath);
    setError(null);
    setNotice(null);
    const option = QUANTIZE_OPTIONS.find((item) => item.id === quantizeId) ?? QUANTIZE_OPTIONS[0];
    try {
      const next = await invoke("adopt_cached_hf_model", {
        request: { localPath: candidate.localPath, quantize: option.value },
      });
      setRegistry(next);
      setNotice(`${candidate.name} added from the HuggingFace cache.`);
    } catch (cause) {
      setError(String(cause));
    } finally {
      setAdoptingPath("");
    }
  }

  async function handleSelect(model) {
    if (loadingId) return;
    setLoadingId(model.id);
    setError(null);
    setNotice(null);
    try {
      await invoke("load_registered_model", { modelId: model.id });
      await refreshRegistry();
      await refreshEngineStatus();
      setNotice(`${model.name} is now the served model.`);
    } catch (cause) {
      setError(String(cause));
    } finally {
      setLoadingId("");
    }
  }

  async function handleSaveToken() {
    if (!tokenInput.trim()) return;
    try {
      const status = await invoke("set_hf_token", { request: { token: tokenInput.trim() } });
      setTokenStatus(status);
      setTokenInput("");
      setNotice("HuggingFace token saved to the keychain.");
    } catch (cause) {
      setError(String(cause));
    }
  }

  async function handleClearToken() {
    try {
      const status = await invoke("clear_hf_token");
      setTokenStatus(status);
      setNotice("HuggingFace token removed.");
    } catch (cause) {
      setError(String(cause));
    }
  }

  const showProgress = busy || (progress && progress.stage !== "done" && progress.stage !== "error");
  const progressPct = progress ? Math.round(Math.min(Math.max(progress.progress, 0), 1) * 100) : 0;

  return (
    <section className="screen-stack">
      <form className="panel" onSubmit={handleImport}>
        <div className="panel-head">
          <p className="eyebrow">Import</p>
          <h2>Add a model from HuggingFace</h2>
          <p className="view-copy">
            Paste a HuggingFace model URL or <code>owner/repo</code>. ChatWorks downloads the snapshot,
            prepares it for local inference, and adds it to your local registry.
          </p>
        </div>
        <div className="field">
          <label htmlFor="hf-url">HuggingFace URL or repo</label>
          <input
            autoComplete="off"
            disabled={busy}
            id="hf-url"
            name="hf-url"
            onChange={(event) => setSourceUrl(event.target.value)}
            placeholder="https://huggingface.co/Qwen/Qwen3-0.6B"
            spellCheck={false}
            type="text"
            value={sourceUrl}
          />
        </div>
        <div className="field">
          <span className="field-label">Conversion</span>
          <div className="segmented" role="radiogroup" aria-label="Quantization">
            {QUANTIZE_OPTIONS.map((option) => (
              <button
                aria-checked={quantizeId === option.id}
                className={quantizeId === option.id ? "segmented-item active" : "segmented-item"}
                disabled={busy}
                key={option.id}
                onClick={() => setQuantizeId(option.id)}
                role="radio"
                type="button"
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>
        <div className="panel-actions">
          <button className="primary-btn" disabled={busy || !sourceUrl.trim()} type="submit">
            {busy ? "Importing…" : "Import model"}
          </button>
          {showProgress && progress ? (
            <div className="import-progress" aria-live="polite">
              <div className="progress-track">
                <div className="progress-fill" style={{ width: `${progressPct}%` }} />
              </div>
              <span className="progress-label">
                {progress.message}
                {progress.totalBytes
                  ? ` — ${formatBytes(progress.downloadedBytes)} / ${formatBytes(progress.totalBytes)}`
                  : ""}
              </span>
            </div>
          ) : null}
        </div>
        {error ? <p className="form-error" role="alert">{error}</p> : null}
        {notice ? <p className="form-notice">{notice}</p> : null}
      </form>

      <div className="panel">
        <div className="panel-head">
          <p className="eyebrow">Cache</p>
          <h2>Adopt cached HuggingFace models</h2>
          <p className="view-copy">
            Scan your local HuggingFace cache and add supported snapshots to ChatWorks without downloading them again.
          </p>
        </div>
        <div className="panel-actions">
          <button className="ghost-btn" disabled={cacheBusy} onClick={handleScanCache} type="button">
            {cacheBusy ? "Scanning…" : "Scan HuggingFace cache"}
          </button>
        </div>
        {cachedModels.length ? (
          <ul className="model-list">
            {cachedModels.map((model) => {
              const alreadyRegistered = registry.models.some((entry) => entry.localPath === model.localPath);
              return (
                <li className="model-row" key={model.localPath}>
                  <div className="model-row-main">
                    <span className="model-row-name">{model.name}</span>
                    <span className="model-row-meta">
                      {model.repo} · {model.providerFamily} · {model.supportsVision ? "Vision" : "Text"}
                    </span>
                  </div>
                  <span className="model-row-meta">{formatBytes(model.sizeBytes)}</span>
                  <button
                    className="ghost-btn"
                    disabled={Boolean(adoptingPath) || alreadyRegistered}
                    onClick={() => handleAdoptCached(model)}
                    type="button"
                  >
                    {alreadyRegistered ? "Registered" : adoptingPath === model.localPath ? "Adding…" : "Add"}
                  </button>
                </li>
              );
            })}
          </ul>
        ) : null}
      </div>

      <div className="panel">
        <div className="panel-head">
          <p className="eyebrow">Registry</p>
          <h2>Local models</h2>
          <p className="view-copy">Pick the one model ChatWorks serves over the OpenAI-compatible API.</p>
        </div>
        <CompactSelector
          items={registry.models}
          selectedId={registry.selectedId ?? ""}
          onSelect={handleSelect}
          getSubtitle={modelSubtitle}
          busyId={loadingId}
          label="Served model"
          placeholder="No model selected"
          emptyLabel="Import a model to get started"
        />
        {registry.models.length ? (
          <ul className="model-list">
            {registry.models.map((model) => {
              const isServed =
                loadedSource &&
                model.localPath === loadedSource &&
                (model.quantize ?? null) === (engineStatus?.loaded?.quantize ?? null);
              return (
                <li className={isServed ? "model-row served" : "model-row"} key={model.id}>
                  <div className="model-row-main">
                    <span className="model-row-name">
                      <StatusDot ok={Boolean(isServed)} />
                      {model.name}
                    </span>
                    <span className="model-row-meta">{model.repo}</span>
                  </div>
                  <span className="model-row-meta">{modelSubtitle(model)}</span>
                  <button
                    className="ghost-btn"
                    disabled={Boolean(loadingId) || isServed}
                    onClick={() => handleSelect(model)}
                    type="button"
                  >
                    {loadingId === model.id ? "Loading…" : isServed ? "Serving" : "Serve"}
                  </button>
                </li>
              );
            })}
          </ul>
        ) : (
          <p className="empty-panel">No models imported yet.</p>
        )}
        {selectedModel ? (
          <p className="view-copy">
            Selected: <strong>{selectedModel.name}</strong> ({selectedModel.repo})
          </p>
        ) : null}
      </div>

      <KvCacheQuantPanel
        engineStatus={engineStatus}
        servedModelId={servedModelId}
        onApplied={refreshServedModel}
      />

      <div className="panel">
        <div className="panel-head">
          <p className="eyebrow">Credentials</p>
          <h2>HuggingFace token</h2>
          <p className="view-copy">
            Optional. Stored in the OS keychain and used for gated or private repositories.
            {tokenStatus.present ? " A token is currently saved." : " No token saved."}
          </p>
        </div>
        <div className="field-inline">
          <input
            autoComplete="off"
            aria-label="HuggingFace token"
            onChange={(event) => setTokenInput(event.target.value)}
            placeholder="hf_…"
            spellCheck={false}
            type="password"
            value={tokenInput}
          />
          <button className="ghost-btn" disabled={!tokenInput.trim()} onClick={handleSaveToken} type="button">
            Save
          </button>
          {tokenStatus.present ? (
            <button className="ghost-btn danger" onClick={handleClearToken} type="button">
              Remove
            </button>
          ) : null}
        </div>
      </div>
    </section>
  );
}

function settingsToForm(settings) {
  return {
    host: settings.server.host,
    port: String(settings.server.port),
    allowLan: Boolean(settings.server.allowLan),
    authEnabled: Boolean(settings.server.authEnabled),
    systemPrompt: settings.sampling.systemPrompt,
    temperature: String(settings.sampling.temperature),
    topP: String(settings.sampling.topP),
    maxTokens: String(settings.sampling.maxTokens),
    disableThinking: Boolean(settings.sampling.disableThinking),
  };
}

function SettingsScreen() {
  const { appSettings, setAppSettings, apiAuthToken, setApiAuthToken, refreshAppSettings } = useApp();
  const [form, setForm] = useState(() => settingsToForm(appSettings));
  const [serverStatus, setServerStatus] = useState(null);
  const [tokenInput, setTokenInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState(null);

  useEffect(() => {
    setForm(settingsToForm(appSettings));
  }, [appSettings]);

  useEffect(() => {
    invoke("openai_server_status")
      .then(setServerStatus)
      .catch(() => setServerStatus(null));
  }, []);

  function updateForm(key, value) {
    setForm((current) => ({ ...current, [key]: value }));
  }

  function buildSettings(nextForm = form) {
    return {
      server: {
        host: nextForm.host.trim(),
        port: Number(nextForm.port),
        allowLan: nextForm.allowLan,
        authEnabled: nextForm.authEnabled,
      },
      sampling: {
        systemPrompt: nextForm.systemPrompt,
        temperature: Number(nextForm.temperature),
        topP: Number(nextForm.topP),
        maxTokens: Number(nextForm.maxTokens),
        disableThinking: nextForm.disableThinking,
      },
    };
  }

  async function saveSettings(nextForm = form, tokenOverride) {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const tokenValue = tokenOverride ?? (tokenInput.trim() ? tokenInput.trim() : null);
      const [nextSettings, nextStatus] = await invoke("save_app_settings", {
        settings: buildSettings(nextForm),
        apiAuthToken: tokenValue,
      });
      setAppSettings(nextSettings);
      setServerStatus(nextStatus);
      setTokenInput("");
      const nextToken = await invoke("api_auth_token").catch(() => null);
      setApiAuthToken(nextToken);
      setNotice("Settings saved and the API server was reconfigured.");
      return nextSettings;
    } catch (cause) {
      setError(String(cause));
      return null;
    } finally {
      setBusy(false);
    }
  }

  async function handleSubmit(eventArg) {
    eventArg.preventDefault();
    await saveSettings();
  }

  async function handleClearApiToken() {
    const nextForm = { ...form, authEnabled: false };
    setForm(nextForm);
    await saveSettings(nextForm, "");
  }

  const apiAuthPresent = Boolean(apiAuthToken);
  const lanWarning = form.host === "0.0.0.0" || form.host === "::" || form.allowLan;

  return (
    <section className="settings-layout">
      <form className="panel settings-card" onSubmit={handleSubmit}>
        <div className="panel-head">
          <p className="eyebrow">OpenAI API</p>
          <h2>Server</h2>
          <p className="view-copy">
            Defaults to loopback. Binding to <code>0.0.0.0</code> exposes the API on your LAN and requires explicit opt-in.
          </p>
        </div>
        <div className="field-grid server-grid">
          <div className="field">
            <label htmlFor="bind-host">Bind host</label>
            <input
              autoComplete="off"
              id="bind-host"
              onChange={(event) => updateForm("host", event.target.value)}
              spellCheck={false}
              type="text"
              value={form.host}
            />
          </div>
          <div className="field">
            <label htmlFor="bind-port">Port</label>
            <input
              id="bind-port"
              inputMode="numeric"
              min="1"
              max="65535"
              onChange={(event) => updateForm("port", event.target.value)}
              step="1"
              type="number"
              value={form.port}
            />
          </div>
        </div>
        <label className="toggle-row">
          <input
            checked={form.allowLan}
            onChange={(event) => updateForm("allowLan", event.target.checked)}
            type="checkbox"
          />
          <span>
            Allow LAN exposure
            <small>Required before binding to 0.0.0.0 or ::. Use an auth token for shared networks.</small>
          </span>
        </label>
        {lanWarning ? (
          <p className="warning-card">
            LAN clients can reach this server at the selected bind address. Keep this off unless you trust the network.
          </p>
        ) : null}
        <div className="field">
          <label htmlFor="api-auth-token">API auth token</label>
          <input
            autoComplete="off"
            id="api-auth-token"
            onChange={(event) => setTokenInput(event.target.value)}
            placeholder={apiAuthPresent ? "Token saved in keychain" : "Optional bearer token"}
            spellCheck={false}
            type="password"
            value={tokenInput}
          />
        </div>
        <label className="toggle-row">
          <input
            checked={form.authEnabled}
            onChange={(event) => updateForm("authEnabled", event.target.checked)}
            type="checkbox"
          />
          <span>
            Require bearer token
            <small>{apiAuthPresent ? "A token is saved in the OS keychain." : "Save a token before enabling auth."}</small>
          </span>
        </label>

        <div className="panel-head section-head">
          <p className="eyebrow">Defaults</p>
          <h2>Sampling</h2>
          <p className="view-copy">Used by Chat and by OpenAI API calls that omit these fields.</p>
        </div>
        <div className="field">
          <label htmlFor="default-system-prompt">System prompt</label>
          <textarea
            id="default-system-prompt"
            onChange={(event) => updateForm("systemPrompt", event.target.value)}
            rows={5}
            value={form.systemPrompt}
          />
        </div>
        <div className="field-grid">
          <div className="field">
            <label htmlFor="default-temperature">Temperature</label>
            <input
              id="default-temperature"
              inputMode="decimal"
              max="2"
              min="0"
              onChange={(event) => updateForm("temperature", event.target.value)}
              step="0.1"
              type="number"
              value={form.temperature}
            />
          </div>
          <div className="field">
            <label htmlFor="default-top-p">Top P</label>
            <input
              id="default-top-p"
              inputMode="decimal"
              max="1"
              min="0"
              onChange={(event) => updateForm("topP", event.target.value)}
              step="0.05"
              type="number"
              value={form.topP}
            />
          </div>
          <div className="field">
            <label htmlFor="default-max-tokens">Max tokens</label>
            <input
              id="default-max-tokens"
              inputMode="numeric"
              min="1"
              onChange={(event) => updateForm("maxTokens", event.target.value)}
              step="1"
              type="number"
              value={form.maxTokens}
            />
          </div>
        </div>
        <label className="toggle-row">
          <input
            checked={form.disableThinking}
            onChange={(event) => updateForm("disableThinking", event.target.checked)}
            type="checkbox"
          />
          <span>
            Disable thinking by default
            <small>Applied when a thinking-capable loaded model supports no-think mode.</small>
          </span>
        </label>
        <div className="panel-actions">
          <button className="primary-btn" disabled={busy} type="submit">
            {busy ? "Saving…" : "Save settings"}
          </button>
          {apiAuthPresent ? (
            <button className="ghost-btn danger" disabled={busy} onClick={handleClearApiToken} type="button">
              Remove token
            </button>
          ) : null}
          <button className="ghost-btn" disabled={busy} onClick={refreshAppSettings} type="button">
            Reload
          </button>
        </div>
        {error ? <p className="form-error" role="alert">{error}</p> : null}
        {notice ? <p className="form-notice">{notice}</p> : null}
      </form>

      <aside className="panel settings-summary">
        <div className="panel-head">
          <p className="eyebrow">Status</p>
          <h2>{serverStatus?.running ? "API online" : "API offline"}</h2>
          <p className="view-copy">{serverStatus?.bound_addr ?? "Server is not bound."}</p>
        </div>
        <span className={serverStatus?.running ? "status-pill" : "status-pill warning"}>
          <StatusDot ok={Boolean(serverStatus?.running)} />
          {serverStatus?.running ? `${serverStatus.host}:${serverStatus.port}` : "Stopped"}
        </span>
        <span className={serverStatus?.auth_required ? "status-pill" : "status-pill warning"}>
          <StatusDot ok={Boolean(serverStatus?.auth_required)} />
          {serverStatus?.auth_required ? "Auth required" : "Auth off"}
        </span>
        {serverStatus?.last_error ? <p className="form-error">{serverStatus.last_error}</p> : null}
      </aside>
    </section>
  );
}

/// Page size for the history list ("Show more…" reveals the next batch) and the localStorage key
/// that remembers the Chat group's expand/collapse state across sessions.
const CHAT_NAV_PAGE_SIZE = 10;
const CHAT_NAV_EXPANDED_KEY = "chatworks-chat-nav-expanded";

/// A single conversation row in the Chat history list: title (click to select), active highlight,
/// inline rename, and delete. Inline rename turns the title into a text input that commits on
/// Enter/blur and cancels on Escape.
function ConversationRow({ conversation, active, busy, onSelect, onRename, onDelete }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(conversation.title);
  const inputRef = useRef(null);

  useEffect(() => {
    if (!editing) return;
    setDraft(conversation.title);
    const el = inputRef.current;
    if (el) {
      el.focus();
      el.select();
    }
  }, [editing, conversation.title]);

  const commitRename = () => {
    setEditing(false);
    const next = draft.trim();
    if (next && next !== conversation.title) onRename(conversation.id, next);
  };
  const cancelRename = () => setEditing(false);

  return (
    <div className={"conv-row" + (active ? " active" : "") + (editing ? " editing" : "")}>
      {editing ? (
        <input
          className="conv-title-input"
          onBlur={commitRename}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              commitRename();
            } else if (event.key === "Escape") {
              event.preventDefault();
              cancelRename();
            }
          }}
          ref={inputRef}
          type="text"
          value={draft}
        />
      ) : (
        <button
          className="conv-title"
          disabled={busy}
          onClick={() => onSelect(conversation.id)}
          title={conversation.title}
          type="button"
        >
          {conversation.title || "New chat"}
        </button>
      )}
      <div className="conv-actions">
        <button
          className="conv-action"
          onClick={() => setEditing(true)}
          title="Rename"
          type="button"
        >
          <Icon.Editor />
        </button>
        <button
          className="conv-action delete"
          disabled={busy && active}
          onClick={() => onDelete(conversation)}
          title="Delete"
          type="button"
        >
          <span aria-hidden="true">×</span>
        </button>
      </div>
    </div>
  );
}

/// Collapsible "Chat" parent group in the sidebar (story C). The header row toggles expand/collapse
/// and navigates to the Chat view; a "+ New chat" button starts a fresh conversation. Expanded, it
/// shows the most recent conversations from the metadata cache with client-side paging, inline
/// rename, delete, and active-conversation highlight. Selection is hard-blocked while a response is
/// streaming (`busy`), matching the "Clear conversation" guard.
function ChatNavGroup() {
  const { activeView, setActiveView } = useApp();
  const {
    activeConversationId,
    conversations,
    busy,
    selectConversation,
    startNewChat,
    renameConversation,
    deleteConversation,
  } = useConversations();

  const [expanded, setExpanded] = useState(
    () => readStoredValue(CHAT_NAV_EXPANDED_KEY, "true") !== "false",
  );
  const [visibleCount, setVisibleCount] = useState(CHAT_NAV_PAGE_SIZE);

  useEffect(() => {
    try {
      window.localStorage.setItem(CHAT_NAV_EXPANDED_KEY, expanded ? "true" : "false");
    } catch {
      /* localStorage unavailable — keep state in-memory only */
    }
  }, [expanded]);

  // The backend already returns the cache sorted by `updatedAt` desc; sort defensively so the nav
  // order stays stable regardless of any future refresh ordering.
  const sorted = useMemo(
    () =>
      [...conversations].sort(
        (a, b) => (b.updatedAt ?? 0) - (a.updatedAt ?? 0) || String(a.id).localeCompare(String(b.id)),
      ),
    [conversations],
  );
  const visible = sorted.slice(0, visibleCount);
  const hasMore = visibleCount < sorted.length;

  const goChat = useCallback(() => setActiveView("Chat"), [setActiveView]);

  const handleHeaderClick = useCallback(() => {
    setExpanded((prev) => !prev);
    goChat();
  }, [goChat]);

  const handleNewChat = useCallback(() => {
    startNewChat();
    goChat();
  }, [startNewChat, goChat]);

  const handleSelect = useCallback(
    (id) => {
      selectConversation(id);
      goChat();
    },
    [selectConversation, goChat],
  );

  const handleRename = useCallback(
    (id, title) => {
      renameConversation(id, title).catch(() => {
        /* refreshConversations already ran inside the action; surface nothing in the nav */
      });
    },
    [renameConversation],
  );

  const handleDelete = useCallback(
    (conversation) => {
      const label = conversation.title || "this conversation";
      if (!window.confirm(`Delete "${label}"? This cannot be undone.`)) return;
      deleteConversation(conversation.id).catch(() => {
        /* ignore — the cache refresh inside the action keeps the list consistent */
      });
    },
    [deleteConversation],
  );

  return (
    <div className={"chat-nav-group" + (expanded ? " expanded" : "")}>
      <div className="chat-nav-header">
        <button
          aria-expanded={expanded}
          className={"chat-nav-toggle" + (activeView === "Chat" ? " is-active" : "")}
          onClick={handleHeaderClick}
          title="Chat"
          type="button"
        >
          <Icon.ChevDown className="chat-nav-chevron" />
          <span className="nav-label">Chat</span>
        </button>
        <button
          className="chat-nav-new icon-btn"
          disabled={busy}
          onClick={handleNewChat}
          title="New chat"
          type="button"
        >
          <Icon.Plus />
        </button>
      </div>
      {expanded ? (
        <div className="chat-nav-list">
          {sorted.length === 0 ? (
            <p className="chat-nav-empty">No conversations yet.</p>
          ) : (
            <>
              {visible.map((conversation) => (
                <ConversationRow
                  active={conversation.id === activeConversationId}
                  busy={busy}
                  conversation={conversation}
                  key={conversation.id}
                  onDelete={handleDelete}
                  onRename={handleRename}
                  onSelect={handleSelect}
                />
              ))}
              {hasMore ? (
                <button
                  className="chat-nav-more"
                  onClick={() => setVisibleCount((count) => count + CHAT_NAV_PAGE_SIZE)}
                  type="button"
                >
                  Show more…
                </button>
              ) : null}
            </>
          )}
        </div>
      ) : null}
    </div>
  );
}

function ActiveScreen() {
  const { activeView } = useApp();
  if (activeView === "Models") return <ModelsScreen />;
  if (activeView === "Settings") return <SettingsScreen />;
  return <ChatScreen />;
}

function AppShell() {
  const { activeView, setActiveView, theme, setTheme, accent, setAccent, engineStatus } = useApp();
  const titleInfo = VIEWS[activeView] ?? VIEWS.Chat;
  const loadedModel = engineStatus?.loaded;

  return (
    <main className="app">
      <aside className="sidebar" aria-label="Primary">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            <Logo size={32} />
          </span>
          <div>
            <h1>
              Chat<span className="light">Works</span>
            </h1>
            <p>Local LLM serving</p>
          </div>
        </div>

        <div className="sidebar-nav">
          {navSections.map((section) => (
            <div className="sidebar-section" key={section.label}>
              <div className="sidebar-section-title">{section.label}</div>
              <nav className="nav-list">
                {section.items.map((item) => {
                  if (item.id === "Chat") {
                    return <ChatNavGroup key={item.id} />;
                  }
                  const IconComponent = item.icon;
                  return (
                    <button
                      className={activeView === item.id ? "nav-item active" : "nav-item"}
                      key={item.id}
                      onClick={() => setActiveView(item.id)}
                      title={item.label}
                      type="button"
                    >
                      <IconComponent />
                      <span className="nav-label">{item.label}</span>
                    </button>
                  );
                })}
              </nav>
            </div>
          ))}
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div className="topbar-title">
            <h1>{titleInfo.title}</h1>
            <p>{titleInfo.blurb}</p>
          </div>
          <span className="topbar-spacer" />
          <div className="topbar-status">
            <span className={loadedModel ? "status-pill" : "status-pill warning"}>
              <StatusDot ok={Boolean(loadedModel)} />
              {loadedModel ? loadedModel.name : "No model loaded"}
            </span>
            <span className="status-pill">Engine actor ready</span>
          </div>
          <div className="accent-picker" role="group" aria-label="Accent color">
            {ACCENTS.map((option) => (
              <button
                aria-label={option.name}
                aria-pressed={accent === option.id}
                className={accent === option.id ? "accent-swatch active" : "accent-swatch"}
                key={option.id}
                onClick={() => setAccent(option.id)}
                style={{ "--sw": option.swatch }}
                title={option.name}
                type="button"
              />
            ))}
          </div>
          <button
            className="icon-btn"
            onClick={() => setTheme(theme === "light" ? "dark" : "light")}
            title={theme === "light" ? "Switch to dark mode" : "Switch to light mode"}
            type="button"
          >
            {theme === "light" ? <Icon.Moon /> : <Icon.Sun />}
          </button>
        </header>

        <ActiveScreen />
      </section>
    </main>
  );
}

function Root() {
  return (
    <React.StrictMode>
      <ErrorBoundary>
        <AppProvider>
          <ConversationsProvider>
            <AppShell />
          </ConversationsProvider>
        </AppProvider>
      </ErrorBoundary>
    </React.StrictMode>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<Root />);
