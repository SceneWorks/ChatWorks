# Changelog

Release notes for ChatWorks. The newest unreleased changes come first.

## Unreleased

### Global decode speedups (epic sc-24432)

- Speculative decoding uses the runtime's proposer-agnostic option (`off`, `auto`, or a proposer
  — MTP head, prompt lookup, draft model — with a depth clamped to what the model advertises).
  Saved settings and conversations keep their choice: `mtpMode: off` loads as `off`, `enabled`
  with N draft tokens as the MTP proposer at depth N, and legacy `mtpMode: auto` now means the
  runtime's `auto` — the model's MTP head where it has one, otherwise prompt lookup — instead of
  "MTP or nothing". A setting that was never chosen follows the runtime's own default.
- Seeded non-greedy (temperature > 0) outputs change: sampling moved to the device, so the same
  seed draws different tokens than before. The output distribution is unchanged; greedy output
  is unaffected.
- On Apple Silicon (MLX) the decode-path status now shows what the runtime measured for each chat
  response: the proposer that ran, its mean accepted length, the sampler path, the prefix cache,
  and every fallback.
- The CUDA-graphs toggle cannot be turned on, with the runtime's reason, while the served model
  cannot capture graphs; a saved "on" can still be turned off.
- A saved draft depth above what the loaded model advertises is kept as saved; the control notes
  the depth it runs at on that model. A saved proposer the model does not advertise runs `off`.
- Conversations saved with a legacy speculative value the old schema refused (`enabled` with 0
  draft tokens, or an unknown mode) open and use the app setting instead of failing to load.
- Models can be loaded with a draft model (another registered model, for draft-model
  speculation), a companion MTP head (a predictor-only artifact for a model that ships none, such
  as a packed checkpoint) and a prefix-cache budget (blank = the runtime's default, 0 = off), set
  per model on the Models screen. The decode-path status shows whether the draft is resident or
  refused (with the runtime's reason), every accelerator the load did not attach (`mtp_head: …`,
  `cuda_graphs: …`), the graph path a generation took, and the prefix-cache budget the load
  settled. Candle's length-aware decode attention is shown by name.
- The CUDA-graphs toggle now knows at load time when the served model's decoder cannot be
  captured (the runtime's `cuda_graphs: …` load fallback), before any generation runs.
- **Downgrade note:** settings are now written as schema version 2 with a `speculative` field
  instead of `mtpMode` / `mtpDraftTokens`. An older ChatWorks reading a version-2 file finds no
  `mtpMode`, falls back to its own default, and its next save drops `speculative` — the choice is
  lost on a downgrade. Conversations behave the same way.
