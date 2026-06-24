import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
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

function stripThinkBlocks(value) {
  return value.replace(/<think>[\s\S]*?<\/think>/gi, "").replace(/<think>[\s\S]*$/i, "").trimStart();
}

function parseNumber(value) {
  if (value === "") return undefined;
  const number = Number(value);
  return Number.isFinite(number) ? number : undefined;
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

function chatRequestBody({ engineStatus, messages, params, thinkingCapable }) {
  const requestMessages = [];
  if (params.systemPrompt.trim()) {
    requestMessages.push({ role: "system", content: params.systemPrompt.trim() });
  }
  requestMessages.push(
    ...messages.map(({ role, content, images }) => {
      // Vision turns send OpenAI content parts (image_url data URLs + text); text turns stay strings.
      if (images && images.length) {
        const parts = images.map((url) => ({ type: "image_url", image_url: { url } }));
        if (content) parts.push({ type: "text", text: content });
        return { role, content: parts };
      }
      return { role, content };
    }),
  );
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
  return body;
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

function ChatScreen() {
  const { engineStatus, refreshEngineStatus, appSettings, apiAuthToken } = useApp();
  const [serverStatus, setServerStatus] = useState(null);
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState("");
  const [params, setParams] = useState(() => paramsFromSettings(appSettings.sampling));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [attachments, setAttachments] = useState([]); // image data URLs for the next turn
  const thinkingCapable = supportsThinking(engineStatus);
  const visionCapable = supportsVision(engineStatus);
  const apiBase = buildLocalApiBase(serverStatus);
  const canSend =
    Boolean(engineStatus?.loaded) && !busy && (Boolean(draft.trim()) || attachments.length > 0);

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

  useEffect(() => {
    setParams(paramsFromSettings(appSettings.sampling));
  }, [appSettings]);

  async function handleSubmit(eventArg) {
    eventArg.preventDefault();
    if (!canSend) return;
    setBusy(true);
    setError(null);
    const userMessage = { role: "user", content: draft.trim(), images: attachments };
    const nextMessages = [...messages, userMessage];
    const assistantMessage = { role: "assistant", content: "", thinking: "" };
    setMessages([...nextMessages, assistantMessage]);
    setDraft("");
    setAttachments([]);
    try {
      const status = await refreshServerStatus();
      const nextEngineStatus = await refreshEngineStatus();
      const headers = { "Content-Type": "application/json" };
      if (appSettings.server.authEnabled && apiAuthToken) {
        headers.Authorization = `Bearer ${apiAuthToken}`;
      }
      const response = await fetch(`${buildLocalApiBase(status)}/v1/chat/completions`, {
        method: "POST",
        headers,
        body: JSON.stringify(
          chatRequestBody({
            engineStatus: nextEngineStatus ?? engineStatus,
            messages: nextMessages,
            params,
            thinkingCapable,
          }),
        ),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.error?.message ?? `OpenAI API returned HTTP ${response.status}`);
      }
      if (!response.body) throw new Error("OpenAI API did not return a stream");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let content = "";
      let thinking = "";
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
          const contentDelta = choice?.delta?.content ?? "";
          const thinkingDelta = choice?.delta?.reasoning_content ?? "";
          if (!contentDelta && !thinkingDelta) return;
          content += contentDelta;
          thinking += thinkingDelta;
          setMessages([...nextMessages, { ...assistantMessage, content, thinking }]);
        });
      }
    } catch (cause) {
      setError(String(cause?.message ?? cause));
      setMessages(nextMessages);
    } finally {
      setBusy(false);
    }
  }

  function updateParam(key, value) {
    setParams((current) => ({ ...current, [key]: value }));
  }

  function addImageFiles(fileList) {
    const files = Array.from(fileList || []).filter((file) => file && file.type.startsWith("image/"));
    files.forEach((file) => {
      const reader = new FileReader();
      reader.onload = () => setAttachments((current) => [...current, reader.result]); // data: URL
      reader.readAsDataURL(file);
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
            messages.map((message, index) => (
              <article className={`message-bubble ${message.role}`} key={`${message.role}-${index}`}>
                <div className="message-role">{message.role}</div>
                {message.images && message.images.length ? (
                  <div className="message-images">
                    {message.images.map((url, imageIndex) => (
                      <img key={imageIndex} className="message-image" src={url} alt={`attachment ${imageIndex + 1}`} />
                    ))}
                  </div>
                ) : null}
                <MessageContent
                  content={message.content}
                  thinking={message.thinking}
                  stripThinking={thinkingCapable && params.disableThinking}
                />
              </article>
            ))
          ) : (
            <div className="empty-panel">Ask a question to start a multi-turn chat with the served model.</div>
          )}
        </div>

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
                Image
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
        <button className="ghost-btn" disabled={busy || !messages.length} onClick={() => setMessages([])} type="button">
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

        {navSections.map((section) => (
          <div className="sidebar-section" key={section.label}>
            <div className="sidebar-section-title">{section.label}</div>
            <nav className="nav-list">
              {section.items.map((item) => {
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
          <AppShell />
        </AppProvider>
      </ErrorBoundary>
    </React.StrictMode>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<Root />);
