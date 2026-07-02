import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { StatusDot } from "@sceneworks/ui";
import { useApp } from "../state/AppContext";

export function settingsToForm(settings) {
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

export function SettingsScreen() {
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
