import React from "react";
import ReactDOM from "react-dom/client";
import { ACCENTS, ErrorBoundary, Icon, Logo, StatusDot } from "@sceneworks/ui";
import "@sceneworks/ui/theme.css";
import "@sceneworks/ui/shell.css";
import "./styles.css";
import { AppProvider, navSections, useApp, VIEWS } from "./state/AppContext";
import { ConversationsProvider } from "./state/ConversationsContext";
import { ChatNavGroup } from "./components/ChatNavGroup";
import { ChatScreen } from "./screens/ChatScreen";
import { ModelsScreen } from "./screens/ModelsScreen";
import { SettingsScreen } from "./screens/SettingsScreen";

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
