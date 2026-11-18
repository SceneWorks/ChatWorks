import { BLOCK_DRAFT_PRECISION_OPTIONS, blockDraftAction, savedBlockDraft } from "../state/models.js";

/// A registered model's block drafts (SW-F360): each catalog draft with its install / use action,
/// or the reason this model cannot use it, and the precision a saved block draft loads at. Changes
/// apply on the model's next load.
export function BlockDraftControls({ model, drafts, disabled, installing, onInstall, onUse, onPrecision }) {
  if (!drafts?.length) return null;
  const saved = savedBlockDraft(model, drafts);
  return (
    <details className="block-drafts">
      <summary>Block drafts{saved ? ` · ${saved.label}` : ""}</summary>
      <ul>
        {drafts.map((draft) => {
          const action = blockDraftAction(draft, model);
          return (
            <li key={draft.format}>
              <span title={`${draft.repo}@${draft.revision.slice(0, 12)} · ${draft.license}`}>{draft.label}</span>
              {action.kind === "unavailable" || action.kind === "selected" ? (
                <small className="field-note">{action.label}</small>
              ) : (
                <button
                  className="ghost-btn"
                  disabled={disabled || Boolean(installing)}
                  onClick={() => (action.kind === "install" ? onInstall(draft) : onUse(draft))}
                  type="button"
                >
                  {installing === draft.format ? "Installing…" : action.label}
                </button>
              )}
            </li>
          );
        })}
      </ul>
      {saved ? (
        <select
          aria-label={`Block draft precision for ${model.name}`}
          disabled={disabled}
          onChange={(event) => onPrecision(event.target.value || null)}
          title="The precision the block draft's projections load at. Applies on the next load."
          value={model.draftQuantize ?? ""}
        >
          {BLOCK_DRAFT_PRECISION_OPTIONS.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
        </select>
      ) : null}
    </details>
  );
}
