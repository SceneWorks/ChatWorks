import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { DEFAULT_ACCENT, Icon } from "@sceneworks/ui";

export const AppContext = createContext(null);

export const DEFAULT_APP_SETTINGS = {
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

export const VIEWS = {
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

export const navSections = [
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

export function readStoredValue(key, fallback) {
  try {
    return window.localStorage.getItem(key) ?? fallback;
  } catch {
    return fallback;
  }
}

export function AppProvider({ children }) {
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

export function useApp() {
  const context = useContext(AppContext);
  if (!context) throw new Error("useApp must be used inside AppProvider");
  return context;
}

export function paramsFromSettings(sampling) {
  return {
    systemPrompt: sampling.systemPrompt ?? "",
    temperature: String(sampling.temperature ?? ""),
    topP: String(sampling.topP ?? ""),
    maxTokens: String(sampling.maxTokens ?? ""),
    disableThinking: Boolean(sampling.disableThinking),
  };
}
