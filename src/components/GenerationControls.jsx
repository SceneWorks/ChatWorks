export function GenerationControls({ params, onChange, capabilities, prefix = "generation" }) {
  const defaults = capabilities == null;
  const field = (name, label, options, disabled = false) => (
    <div className="field" key={name}>
      <label htmlFor={`${prefix}-${name}`}>{label}</label>
      <select id={`${prefix}-${name}`} value={params[name] ?? ""} disabled={disabled}
        onChange={(event) => onChange(name, event.target.value)}>
        {options.map(([value, text]) => <option value={value} key={value}>{text}</option>)}
      </select>
    </div>
  );
  return (
    <div className="field-grid">
      {(defaults || capabilities.supports_reasoning_effort) && field("reasoningEffort", "Reasoning effort",
        [["", "Model default"], ["low", "Low"], ["medium", "Medium"], ["xhigh", "Extra high"]], params.disableThinking)}
      {(defaults || capabilities.supports_preserve_thinking) && field("preserveThinking", "Prior reasoning",
        [["", "Model default"], ["true", "Keep in conversation"], ["false", "Omit from future prompts"]])}
      {(defaults || capabilities.mtp) && field("mtpMode", "Multi-token prediction",
        [["off", "Off"], ["auto", "Automatic"], ["enabled", "Choose draft count"]])}
      {(defaults || capabilities.mtp) && params.mtpMode === "enabled" && (
        <div className="field">
          <label htmlFor={`${prefix}-mtp-drafts`}>Draft tokens</label>
          <input id={`${prefix}-mtp-drafts`} type="number" min="1" step="1"
            max={capabilities?.mtp?.max_draft_tokens} value={params.mtpDraftTokens ?? "3"}
            onChange={(event) => onChange("mtpDraftTokens", event.target.value)} />
        </div>
      )}
      {[["topK", "Top K", 0, 1], ["repetitionPenalty", "Repetition penalty", 0.01, 0.05],
        ["repetitionContext", "Repetition window", 0, 1], ["seed", "Seed", 0, 1]].map(([name, label, min, step]) => (
        <div className="field" key={name}>
          <label htmlFor={`${prefix}-${name}`}>{label}</label>
          <input id={`${prefix}-${name}`} type="number" min={min} step={step}
            max={name === "seed" ? Number.MAX_SAFE_INTEGER : undefined} placeholder="Model default"
            value={params[name] ?? ""} onChange={(event) => onChange(name, event.target.value)} />
        </div>
      ))}
    </div>
  );
}
