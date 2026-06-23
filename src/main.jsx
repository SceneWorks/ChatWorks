import React, { createContext, useContext, useEffect, useMemo, useState } from "react";
import ReactDOM from "react-dom/client";
import { invoke } from "@tauri-apps/api/core";
import "@sceneworks/ui/theme.css";
import "@sceneworks/ui/shell.css";
import { ACCENTS, DEFAULT_ACCENT, ErrorBoundary, Icon, Logo, StatusDot } from "@sceneworks/ui";
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
    let mounted = true;
    invoke("engine_status")
      .then((status) => {
        if (mounted) setEngineStatus(status);
      })
      .catch(() => {
        if (mounted) setEngineStatus(null);
      });
    return () => {
      mounted = false;
    };
  }, []);

  const value = useMemo(
    () => ({ activeView, setActiveView, theme, setTheme, accent, setAccent, engineStatus }),
    [accent, activeView, engineStatus, theme],
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

function ModelsScreen() {
  return (
    <EmptyScreen eyebrow="Model registry" title="Models shell">
      HuggingFace import, MLX conversion, quantization, and serve selection will live here.
    </EmptyScreen>
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
