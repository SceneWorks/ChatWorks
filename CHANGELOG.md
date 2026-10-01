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
- The CUDA-graphs toggle is disabled, with the runtime's reason, while the served model cannot
  capture graphs.
