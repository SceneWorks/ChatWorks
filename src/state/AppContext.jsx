import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { DEFAULT_ACCENT, Icon } from "@sceneworks/ui";
import { generationParams } from "./generation.js";
import { applyDecodeEvent } from "./decodePath.js";
import { loadAppSettingsOrDefaults, saveAppCredentialState } from "./credentials.js";

export const AppContext = createContext(null);

/// A placeholder until the backend answers: it carries no speculative mode, so nothing saved from
/// it can pin one (the backend fills in this build's default).
export const DEFAULT_APP_SETTINGS = {
  server: {
    host: "127.0.0.1",
    port: 8000,
    allowLan: false,
    authEnabled: false,
    allowLocalFiles: false,
  },
  sampling: {
    systemPrompt: "You are a helpful local assistant.",
    temperature: 0.7,
    topP: 0.9,
    maxTokens: 512,
    disableThinking: true,
  },
  runtime: {
    cudaGraphs: false,
  },
  notices: {
    speculativeOffCarriedOver: false,
    speculativeNoticeDismissed: false,
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
  const [apiAuthError, setApiAuthError] = useState(null);

  const refreshAppSettings = useCallback(() => {
    return loadAppSettingsOrDefaults(invoke).then(({ settings, token, error }) => {
      if (settings) setAppSettings(settings);
      setApiAuthToken(token);
      setApiAuthError(error);
      return settings ?? DEFAULT_APP_SETTINGS;
    });
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

  // Every finished generation — this window's or an API client's — pushes the served model's
  // decode status, so the decode-path view updates without polling (sc-24139).
  useEffect(() => {
    const unlistenPromise = listen("engine://decode", (event) => {
      setEngineStatus((current) => applyDecodeEvent(current, event.payload));
    });
    return () => {
      unlistenPromise.then((unlisten) => unlisten());
    };
  }, []);

  /// Save a settings change made outside the Settings form (the speculative notice's actions),
  /// keeping the API token as it is.
  const updateAppSettings = useCallback(async (transform) => {
    const { settings: nextSettings, token } = await saveAppCredentialState(invoke, transform(appSettings), null);
    setAppSettings(nextSettings);
    setApiAuthToken(token);
    setApiAuthError(null);
    return nextSettings;
  }, [appSettings]);

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
      updateAppSettings,
      apiAuthToken,
      setApiAuthToken,
      apiAuthError,
      setApiAuthError,
      refreshAppSettings,
    }),
    [
      accent, activeView, apiAuthError, apiAuthToken, appSettings, engineStatus, refreshAppSettings,
      refreshEngineStatus, theme, updateAppSettings,
    ],
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
    ...generationParams(sampling),
  };
}
