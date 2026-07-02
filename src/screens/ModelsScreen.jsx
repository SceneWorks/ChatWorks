import { useCallback, useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { CompactSelector, StatusDot } from "@sceneworks/ui";
import { useApp } from "../state/AppContext";

export const QUANTIZE_OPTIONS = [
  { id: "dense", label: "Dense (full precision)", value: null },
  { id: "q4", label: "Quantize Q4", value: "q4" },
  { id: "q8", label: "Quantize Q8", value: "q8" },
];

export function formatBytes(bytes) {
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

export function modelSubtitle(model) {
  const parts = [];
  if (model.quantize === "q4") parts.push("Q4");
  else if (model.quantize === "q8") parts.push("Q8");
  else parts.push("Dense");
  if (model.sizeBytes) parts.push(formatBytes(model.sizeBytes));
  return parts.join(" · ");
}

export function ModelsScreen() {
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
