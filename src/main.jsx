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
  StatusDot,
} from "@sceneworks/ui";
import "./styles.css";

const AppContext = createContext(null);

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
  }, [refreshEngineStatus]);

  const value = useMemo(
    () => ({ activeView, setActiveView, theme, setTheme, accent, setAccent, engineStatus, refreshEngineStatus }),
    [accent, activeView, engineStatus, refreshEngineStatus, theme],
  );

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

function useApp() {
  const context = useContext(AppContext);
  if (!context) throw new Error("useApp must be used inside AppProvider");
  return context;
}

function EmptyScreen({ eyebrow, title, children }) {
  return (
    <section className="screen-stack">
      <div className="hero-panel">
        <p className="eyebrow">{eyebrow}</p>
        <h2>{title}</h2>
        <p className="view-copy">{children}</p>
      </div>
      <div className="empty-panel">This screen is scaffolded for the next ChatWorks slice.</div>
    </section>
  );
}

function ChatScreen() {
  return (
    <EmptyScreen eyebrow="Streaming chat" title="Chat UI shell">
      The chat surface will dogfood the local OpenAI-compatible API once the server slice lands.
    </EmptyScreen>
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
            prepares it for MLX, and adds it to your local registry.
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

function SettingsScreen() {
  return (
    <EmptyScreen eyebrow="Server defaults" title="Settings shell">
      Bind address, port, auth token, and default sampling controls will be persisted from here.
    </EmptyScreen>
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
