import { useCallback, useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { CompactSelector, StatusDot } from "@sceneworks/ui";
import { useApp } from "../state/AppContext";
import { useConversations } from "../state/ConversationsContext";
import { formatBytes, isExactGgufUrl, loadNotice, modelSubtitle, modelWeightLabel, unloadServedModel } from "../state/models.js";
import { checkHfCredentialStatus } from "../state/credentials.js";
import {
  dismissSpeculativeNotice,
  enableSpeculativeAuto,
  graphsReloadPending,
  selectedWeightFormat,
  serveAction,
  weightFormatOptions,
} from "../state/decodePath.js";
import { DecodePathStatus } from "../components/DecodePathStatus.js";

export function ModelsScreen() {
  const { engineStatus, refreshEngineStatus, appSettings, updateAppSettings } = useApp();
  const { busy: generationBusy } = useConversations();
  const [unloading, setUnloading] = useState(false);
  const [registry, setRegistry] = useState({ models: [], selectedId: null });
  const [sourceUrl, setSourceUrl] = useState("");
  const [projectorUrl, setProjectorUrl] = useState("");
  const [quantizeId, setQuantizeId] = useState("dense");
  const [tokenStatus, setTokenStatus] = useState(null);
  const [tokenInput, setTokenInput] = useState("");
  const [progress, setProgress] = useState(null);
  const [busy, setBusy] = useState(false);
  const [cacheBusy, setCacheBusy] = useState(false);
  const [cachedModels, setCachedModels] = useState([]);
  const [adoptingPath, setAdoptingPath] = useState("");
  const [projectorSelections, setProjectorSelections] = useState({});
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState(null);
  const [loadingId, setLoadingId] = useState("");

  const loadedSource = engineStatus?.loaded?.source ?? null;
  // The weight formats this runtime can load; NVFP4 is disabled with the runtime's own reason
  // unless it reports a compute capability >= sm_120 CUDA device (sc-24139).
  const weightFormats = weightFormatOptions(engineStatus?.backend_capabilities);
  const unavailableFormats = weightFormats.filter((option) => option.disabled);
  const formatNotes = weightFormats.filter((option) => option.note && !option.disabled);
  // The served model runs under a CUDA-graph switch other than the saved setting: its row offers
  // "Reload" (graphs are a load option).
  const reloadPending = graphsReloadPending(
    engineStatus?.backend_capabilities,
    appSettings?.runtime?.cudaGraphs,
    engineStatus?.loaded,
  );
  const selectedModel = registry.models.find((model) => model.id === registry.selectedId) ?? null;
  const exactGgufImport = isExactGgufUrl(sourceUrl);

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
  }, [refreshRegistry]);

  async function checkTokenStatus() {
    try {
      setTokenStatus(await checkHfCredentialStatus(invoke));
      setError(null);
    } catch (cause) {
      setTokenStatus(null);
      setError(`Could not read HuggingFace credential: ${String(cause)}`);
    }
  }

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
    const option = selectedWeightFormat(weightFormats, quantizeId);
    try {
      const next = await invoke("import_hf_model", {
        request: {
          sourceUrl: sourceUrl.trim(),
          quantize: exactGgufImport ? null : option.value,
          projectorSource: projectorUrl.trim() || null,
        },
      });
      setRegistry(next);
      setNotice("Model imported and added to the registry.");
      setSourceUrl("");
      setProjectorUrl("");
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
      setNotice(models.length ? `Found ${models.length} cached model${models.length === 1 ? "" : "s"}.` : "No supported cached HuggingFace models found.");
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
    const option = selectedWeightFormat(weightFormats, quantizeId);
    const storedEncoding = candidate.pack === "bonsai2-packed" || candidate.format?.startsWith("gguf");
    try {
      const next = await invoke("adopt_cached_hf_model", {
        request: {
          localPath: candidate.localPath,
          quantize: storedEncoding ? null : option.value,
          projectorSource: candidate.projectorSource,
        },
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
    if (loadingId || unloading || generationBusy) return;
    setLoadingId(model.id);
    setError(null);
    setNotice(null);
    try {
      const status = await invoke("load_registered_model", {
        modelId: model.id,
        projectorSource: projectorSelections[model.id] ?? model.projectorSource ?? null,
      });
      await refreshRegistry();
      await refreshEngineStatus();
      setNotice(loadNotice(model.name, status));
    } catch (cause) {
      setError(String(cause));
      // A failed reload may already have unloaded the served model: show what is served now.
      await refreshEngineStatus();
    } finally {
      setLoadingId("");
    }
  }

  async function handleUnload() {
    if (unloading || loadingId || generationBusy) return;
    setUnloading(true);
    setError(null);
    setNotice(null);
    try {
      await unloadServedModel({ invoke, busy: generationBusy, refreshStatus: refreshEngineStatus });
      setNotice("Model unloaded. Select a registered model to load it again.");
    } catch (cause) {
      setError(String(cause));
    } finally {
      setUnloading(false);
    }
  }

  async function handleSaveToken() {
    if (!tokenInput.trim()) return;
    try {
      const status = await invoke("set_hf_token", { request: { token: tokenInput.trim() } });
      setTokenStatus(status);
      setError(null);
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
      setError(null);
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
          {engineStatus?.execution_backend === "candle-cpu" ? (
            <p className="view-copy">
              This build uses Candle CPU. Qwen3.8-27B and Bonsai 2 require Apple MLX or Candle CUDA; other supported models can still use CPU.
            </p>
          ) : null}
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
        {exactGgufImport ? (
          <div className="field">
            <label htmlFor="hf-projector-url">Companion projector URL (optional)</label>
            <input
              autoComplete="off"
              disabled={busy}
              id="hf-projector-url"
              onChange={(event) => setProjectorUrl(event.target.value)}
              placeholder="https://huggingface.co/owner/repo/blob/revision/mmproj-F16.gguf"
              spellCheck={false}
              type="url"
              value={projectorUrl}
            />
            <small>Choose one exact mmproj artifact from the same repository and revision. Empty remains text-only.</small>
          </div>
        ) : null}
        <div className="field">
          <span className="field-label">Weight format</span>
          {exactGgufImport ? <p className="view-copy">Existing GGUF encoding (conversion unavailable)</p> : null}
          <div className="segmented" role="radiogroup" aria-label="Weight format">
            {weightFormats.map((option) => (
              <button
                aria-checked={quantizeId === option.id}
                className={quantizeId === option.id ? "segmented-item active" : "segmented-item"}
                disabled={busy || exactGgufImport || option.disabled}
                key={option.id}
                onClick={() => setQuantizeId(option.id)}
                role="radio"
                title={option.reason ?? undefined}
                type="button"
              >
                {option.label}
              </button>
            ))}
          </div>
          {formatNotes.map((option) => (
            <small className="field-note lossy" key={`${option.id}-note`}>
              {option.note}
            </small>
          ))}
          {unavailableFormats.map((option) => (
            <small className="field-note" key={option.id}>
              {option.label} unavailable: {option.reason}
            </small>
          ))}
          <small className="field-note">Applied when the model loads; the same choice applies to cached models you add below.</small>
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
              const selectedProjector = projectorSelections[model.localPath] ?? "";
              return (
                <li className="model-row" key={model.localPath}>
                  <div className="model-row-main">
                    <span className="model-row-name">{model.name}</span>
                    <span className="model-row-meta">
                      {model.repo} · {model.modelFamily} · {modelWeightLabel(model)} · {model.supportsVision ? "Vision" : "Text"}
                    </span>
                    {model.unavailableReason ? <span className="model-row-meta">{model.unavailableReason}</span> : null}
                  </div>
                  <span className="model-row-meta">{formatBytes(model.sizeBytes)}</span>
                  {model.projectorSources?.length ? (
                    <select
                      aria-label={`Projector for ${model.name}`}
                      disabled={Boolean(adoptingPath) || alreadyRegistered || Boolean(model.unavailableReason)}
                      onChange={(event) => setProjectorSelections((current) => ({ ...current, [model.localPath]: event.target.value }))}
                      value={selectedProjector}
                    >
                      <option value="">Text only (no projector)</option>
                      {model.projectorSources.map((source) => <option key={source} value={source}>{source.split("/").at(-1)}</option>)}
                    </select>
                  ) : null}
                  <button
                    className="ghost-btn"
                    disabled={Boolean(adoptingPath) || alreadyRegistered || Boolean(model.unavailableReason)}
                    onClick={() => handleAdoptCached({ ...model, projectorSource: selectedProjector || null })}
                    type="button"
                  >
                    {alreadyRegistered ? "Registered" : model.unavailableReason ? "Unavailable" : adoptingPath === model.localPath ? "Adding…" : "Add"}
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
        {loadedSource ? <button className="ghost-btn" type="button"
          disabled={unloading || Boolean(loadingId) || generationBusy}
          onClick={handleUnload}>{unloading ? "Unloading…" : "Unload model"}</button> : null}
        <CompactSelector
          items={registry.models}
          selectedId={loadedSource ? registry.selectedId ?? "" : ""}
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
              const selectedProjector = projectorSelections[model.id] ?? model.projectorSource ?? "";
              const isServed =
                loadedSource &&
                model.localPath === loadedSource &&
                (model.quantize ?? null) === (engineStatus?.loaded?.quantize ?? null) &&
                (selectedProjector || null) === (engineStatus?.loaded?.projector_source ?? null);
              const action = serveAction(Boolean(isServed), reloadPending);
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
                  {model.projectorSources?.length ? (
                    <select
                      aria-label={`Projector for ${model.name}`}
                      disabled={Boolean(loadingId)}
                      onChange={(event) => setProjectorSelections((current) => ({ ...current, [model.id]: event.target.value }))}
                      value={selectedProjector}
                    >
                      <option value="">Text only (no projector)</option>
                      {model.projectorSources.map((source) => <option key={source} value={source}>{source.split("/").at(-1)}</option>)}
                    </select>
                  ) : null}
                  <button
                    className="ghost-btn"
                    disabled={Boolean(loadingId) || action.disabled}
                    onClick={() => handleSelect(model)}
                    title={action.label === "Reload" ? "Reload with the saved CUDA-graph setting" : undefined}
                    type="button"
                  >
                    {loadingId === model.id ? "Loading…" : action.label}
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
        <DecodePathStatus
          engineStatus={engineStatus}
          title="Served model decode path"
          notice={{
            appSettings,
            executionBackend: engineStatus?.execution_backend,
            onEnableAuto: () => updateAppSettings(enableSpeculativeAuto).catch((cause) => setError(String(cause))),
            onDismiss: () => updateAppSettings(dismissSpeculativeNotice).catch((cause) => setError(String(cause))),
          }}
        />
      </div>

      <div className="panel">
        <div className="panel-head">
          <p className="eyebrow">Credentials</p>
          <h2>HuggingFace token</h2>
          <p className="view-copy">
            Optional. Stored in the OS keychain and used for gated or private repositories.
            {tokenStatus === null ? " Check its status when needed." : tokenStatus.present ? " A token is currently saved." : " No token saved."}
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
          <button className="ghost-btn" onClick={checkTokenStatus} type="button">Check saved token</button>
          {tokenStatus?.present ? (
            <button className="ghost-btn danger" onClick={handleClearToken} type="button">
              Remove
            </button>
          ) : null}
        </div>
      </div>
    </section>
  );
}
