# Full Codebase Review — ChatWorks — 2026-07-02

## Executive summary

- **Repository at a glance:** Tauri desktop app (Rust backend + React/JSX frontend) that serves
  local LLMs over an OpenAI-compatible HTTP/SSE API. ~41 tracked files, ~9.2k LOC (≈6.6k Rust,
  ≈2.5k JSX, plus CSS). Stack: `axum` + `tokio`, `keyring`, `reqwest`, `core-llm`/`mlx-llm`/
  `candle-llm` (git deps), React 18 + `@sceneworks/ui`.
- **Coverage:** Whole tree reviewed — all 6 Rust modules (`app_settings`, `conversations`,
  `engine`, `model_registry`, `server`, `tools`), `main.rs`/`lib.rs`/`build.rs`, the full
  `src/main.jsx` frontend, all configs (`Cargo.toml`, `tauri.conf.json`, `capabilities/default.json`,
  `Entitlements.plist`, `vite.config.js`, `eslint.config.js`), the docs (`README`, `WINDOWS`,
  `docs/VIDEO_API`), and a representative integration test (`qwen3vl_tools_server.rs`).
  **Excluded:** `node_modules/`, `target/`, `dist/` (build outputs), the 7 other
  `src-tauri/tests/qwen*_*.rs` files (skimming confirms they share the pattern of the one read in
  full — env-gated real-weights tests), and binary icons.
- **Headline:** This is a notably well-engineered small codebase — atomic writes, careful path
  validation, sidecar-based conversation listing, a clean engine actor, and strong test coverage of
  the security-sensitive paths. No Critical issues. The top risks are (1) a supply-chain posture
  that pins three core inference deps to a mutable git branch, (2) a couple of unbounded resource
  surfaces on the HTTP API (image decode, body size, concurrency) that matter mostly when the
  server is LAN-exposed, and (3) a block of now-stale "DEV-PIN" comments that contradict the actual
  dependency declarations after the recent sc-8528 backout.
- **Counts:** Critical: 0 | High: 1 | Medium: 6 | Low: 6 | Info: 2.

---

## Critical findings

_(None.)_

---

## High findings

#### [F-001] Core inference deps pinned to mutable `branch = "main"` (supply chain)
- **Category:** security
- **Severity:** High
- **Location:** `src-tauri/Cargo.toml:29` (`core-llm`), `src-tauri/Cargo.toml:49` (`mlx-llm`),
  `src-tauri/Cargo.toml:58` (`candle-llm`)
- **Finding:** All three inference-critical crates are declared as `git = "...", branch = "main"`
  rather than `rev = "<sha>"` or a tag. A branch spec means the next time the lockfile is
  regenerated (fresh clone without `Cargo.lock`, `cargo update -p core-llm`, or a dependency
  bump) cargo silently advances to whatever `main` HEAD is at that moment — including, in the worst
  case, a compromised or force-pushed upstream. `Cargo.lock` pins the *current* resolution, but a
  branch pin makes "update the lock" a trust event in a way a `rev`/tag pin does not.
- **Impact:** These three crates *are* the inference path — the model weights loader, the chat
  template, the sampler. A silent drift here is the highest-blast-radius supply-chain event in the
  repo. The dependency-unification rationale (documented in the comment block) is real, but it is
  satisfied equally well by a pinned `rev`, which the backends could share.
- **Suggested fix:** Pin each of the three to an explicit `rev = "<sha>"` (or a signed tag) and bump
  intentionally. Keep `Cargo.lock` as the second layer. Add a short CI step (or a `cargo deny`
  / `cargo vet` config) that fails if a git dep is on a bare `branch`.
- **Confidence:** Medium — the lockfile does prevent *accidental* drift today; this is about the
  next deliberate lockfile refresh and the trust model around it.

---

## Medium findings

#### [F-002] No server-side cap on decoded image/video dimensions (OOM surface)
- **Category:** security
- **Severity:** Medium
- **Location:** `src-tauri/src/engine.rs:443-455` (`decode_image`), `src-tauri/src/server.rs:28`
  (`OPENAI_JSON_BODY_LIMIT_BYTES = 64 * 1024 * 1024`)
- **Finding:** The OpenAI API path decodes any submitted image to a full RGB8 buffer via
  `image::load_from_memory` with no pixel-dimension bound. The 64 MiB JSON body limit allows a
  ~48 MiB base64 payload, which can encode a compressed image that decodes to a multi-gigabyte RGB
  buffer (decompression bomb). The frontend self-limits attachments to 1536px / 8 MiB
  (`src/main.jsx:465-467`), but the HTTP API — the documented OpenAI-compatible surface — does not.
- **Impact:** On a loopback-only server this is self-inflicted at worst. But the server can be
  LAN-exposed (`allow_lan`) with auth optional, and a single oversized `image_url`/`video_url` part
  can force OOM or a stall during inference. `VideoRef` compounds this: up to N frames, each
  unbounded.
- **Suggested fix:** Before `decode_image`, cap total decoded pixels per request (e.g. reject if
  `width*height > MAX_IMAGE_PIXELS`, matching the `image` crate's `set_limits` / a chosen budget),
  and cap frames-per-video and total-frames-per-request. Apply it in `engine.rs` so both the IPC
  and HTTP paths share the guard.
- **Confidence:** High

#### [F-003] CORS allows any origin on the OpenAI server
- **Category:** security
- **Severity:** Medium
- **Location:** `src-tauri/src/server.rs:281-300` (`apply_cors_headers` hardcodes
  `ACCESS_CONTROL_ALLOW_ORIGIN: *`)
- **Finding:** Every response — including preflight — carries `Access-Control-Allow-Origin: *`.
  When the server is bound to the LAN (`allow_lan=true`) and a bearer token is set, any web page in
  any browser on a reachable client can issue authenticated requests to the API so long as it can
  learn/guess the token; with auth disabled (the default), any such page can drive the loaded model
  and read its output.
- **Impact:** Loopback (the default) contains this. The risk appears specifically when a user opts
  into LAN serving, which the UI warns about but does not restrict the CORS policy for.
- **Suggested fix:** When `auth_token` is set, reflect the request `Origin` only when it is
  explicitly allow-listed, or drop the wildcard in favor of a configured origin list. At minimum,
  when `allow_lan` is false, the `*` is unnecessary (loopback pages don't need CORS) and can be
  tightened. Document the tradeoff next to the header.
- **Confidence:** Medium — partly a deliberate "local server" convention, hence Medium not High.

#### [F-004] `CancelFlag` is created but never signaled — no mid-stream cancellation
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `src-tauri/src/engine.rs:295` (`cancel: CancelFlag::new()` inside
  `GenerateRequest::into_core`); the engine actor at `src-tauri/src/engine.rs:117-140` never holds a
  handle to signal it.
- **Finding:** Each request builds a fresh `CancelFlag` that is dropped with the request; nothing in
  `EngineHandle`/`EngineActor` exposes a cancel/stop path, and the streaming SSE handler
  (`server.rs:346-403`) likewise has no way to react to a dropped client mid-generation. The only
  bounds on a runaway generation are `max_new_tokens` and the frontend's `MAX_TOOL_STEPS` loop.
- **Impact:** A long generation (especially a high `max_new_tokens` or a stuck provider) keeps the
  engine thread and GPU/Metal busy until completion even if the user navigates away or closes the
  stream. For a desktop LLM host this is a real resource/UX issue.
- **Suggested fix:** Store the `CancelFlag` handle on `EngineActor` (keyed by an in-flight request
  id) and add an `EngineHandle::cancel()` (or a Tauri `stop_generation` command) plus an axum
  handler that cancels on client disconnect (`Sse`/`ReceiverStream` can observe drop). Plumb the
  existing flag rather than allocating a dead one.
- **Confidence:** High that the flag is unused; Medium on the provider actually honoring it (depends
  on `core-llm`'s contract — verify before claiming end-to-end cancellation).

#### [F-005] Stale "DEV-PIN" comments contradict the actual `branch = "main"` pins
- **Category:** readability (with a maintainability hazard)
- **Severity:** Medium
- **Location:** `src-tauri/Cargo.toml:26-28` (core-llm), `src-tauri/Cargo.toml:46-47` (mlx-llm),
  `src-tauri/Cargo.toml:52-56` (candle-llm)
- **Finding:** Commit `8051629` ("fix(deps): repin core-llm/mlx-llm/candle to branch=main
  (sc-8528 backout; #29 merged a stale dev-pin)") flipped all three deps back to `branch = "main"`,
  but the surrounding comments still say things like *"Temporarily pinned to the field-removed
  core-llm backout branch"* and *"COORDINATOR flip back to branch='main' after the core-llm +
  mlx-llm backout PRs merge."* The comments now describe the opposite of what the code does.
- **Impact:** A future reader (or a coordinator bot) will either act on the stale instruction
  ("flip back to main" — already done) or be confused about which state is intended. This is exactly
  the class of stale TODO that the skill's rubric flags.
- **Suggested fix:** Delete the DEV-PIN/sc-8528 comment blocks, or replace them with a one-line
  note that the deps unify on `core-llm` `branch = "main"` (and, per F-001, ideally a `rev`).
- **Confidence:** High

#### [F-006] `unsafe_code = "allow"` undercuts the workspace forbid without explanation
- **Category:** readability / governance
- **Severity:** Medium
- **Location:** `src-tauri/Cargo.toml:60-61` (`[lints.rust] unsafe_code = "allow"`) overriding
  `Cargo.toml:12-13` (`[workspace.lints.rust] unsafe_code = "forbid"`)
- **Finding:** The workspace declares `unsafe_code = "forbid"` as the safety posture, but the
  `src-tauri` crate — the one that handles untrusted HTTP input, image decoding, file I/O, and
  native FFI adjacency — silently relaxes it to `allow`. No `unsafe` blocks are present in the
  crate's own source today, so the override appears to exist for macro-generated code (e.g. Tauri),
  but it is undocumented.
- **Impact:** The forbid posture stops being a guardrail precisely where the most security-sensitive
  code lives, and a future contributor can add an `unsafe` block here without any signal that it
  contradicts the project's stated posture.
- **Suggested fix:** Either document *why* the allow is needed (which macro/dependency forces it)
  next to the line, or scope the relaxation to the minimum (e.g. `#![allow(unsafe_code)]` in just
  the module that needs it, with a comment). If nothing currently requires it, remove the override
  and let forbid apply.
- **Confidence:** High that no crate-source `unsafe` exists; Medium on the cause.

#### [F-007] The entire frontend is a single 2,492-line file
- **Category:** readability
- **Severity:** Medium
- **Location:** `src/main.jsx` (2,492 lines)
- **Finding:** All contexts (`AppProvider`, `ConversationsProvider`, `ChatStateContext`), every
  screen (`ChatScreen`, `ModelsScreen`, `SettingsScreen`), the SSE/network layer
  (`streamChatCompletion`, `readSseMessages`), media helpers (`normalizeImageAttachment`,
  `sampleVideoAttachment`), inline SVG icons, and the tool-approval UI live in one module. Comments
  are good and the code is internally well-organized, but the file is past the point where navigation
  and merge-conflict surface scale well.
- **Impact:** Slows review and raises the chance of unrelated changes colliding. The two-context
  split (rare metadata vs. per-token chat state) is a thoughtful design that would read even better
  as its own module.
- **Suggested fix:** Extract along the existing seams: `src/state/` (contexts + the param/title
  helpers), `src/api/` (SSE + `chatRequestBody`/`toOpenAiMessage`), `src/media/` (image/video
  sampling), and `src/screens/` + `src/components/`. No behavior change required.
- **Confidence:** High (subjective but conventional)

---

## Low findings

#### [F-008] Bearer-token check uses non-constant-time string comparison
- **Category:** security
- **Severity:** Low
- **Location:** `src-tauri/src/server.rs:414-426` (`authorize`), specifically `actual == expected`
  at line 423
- **Finding:** The auth check builds `format!("Bearer {token}")` and compares it to the incoming
  header with `==`, which short-circuits on the first mismatched byte.
- **Impact:** For a loopback-only desktop server this is negligible. If the server is LAN-exposed
  and an attacker can make many timed requests, a timing oracle could in principle shorten a
  brute-force of the token; network jitter makes this impractical but not impossible. Low real-world
  risk here, but it's a cheap fix and the kind of thing worth doing correctly in auth code.
- **Suggested fix:** Compare the header value with a constant-time equality (`subtle::ConstantTimeEq`,
  or compare SHA-256 digests). At minimum compare the raw token bytes, not the formatted string.
- **Confidence:** High

#### [F-009] Conversation id is validated trimmed but used untrimmed on the path
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `src-tauri/src/conversations.rs:115-130` (`validate_id` trims before its path
  checks) vs. `:196-216` (`save_conversation_in_dir` calls `validate_id(&conversation.id)?` then
  builds the path from the *untrimmed* `conversation.id`); same shape in `get_/rename_/
  delete_conversation_in_dir`
- **Finding:** `validate_id` deliberately trims so that padded dangerous values like `" .. "` are
  rejected, and its own test says *"callers trim on use."* But the callers don't trim — they pass
  the raw id into `conversation_file_path`/`conversation_meta_path`. So an id like `"  abc  "`
  passes validation and is then written as `"  abc  .json"`, which a subsequent `get_conversation(
  "abc")` won't find.
- **Impact:** The frontend always uses `crypto.randomUUID()` (no surrounding whitespace), so this
  never fires in practice. It's a latent contract inconsistency: the safety check and the storage
  key disagree on what the canonical id is.
- **Suggested fix:** Canonicalize once at the boundary — either trim `conversation.id` in
  `save_conversation_in_dir` (and `id` in the other three) right after validation, or change
  `validate_id` to return the trimmed form and use that everywhere downstream.
- **Confidence:** High

#### [F-010] Three near-identical atomic-write helpers
- **Category:** redundant
- **Severity:** Low
- **Location:** `src-tauri/src/app_settings.rs:151-162` (`write_settings`),
  `src-tauri/src/conversations.rs:243-254` (`write_json_atomic`),
  `src-tauri/src/model_registry.rs:817-828` (`write_registry`)
- **Finding:** All three do the same dance: `create_dir_all(parent)` → write to a `.tmp` sibling →
  `fs::rename`. They differ only in serialization (`to_string_pretty` of a concrete type vs. a
  generic `T: Serialize`) and the exact temp suffix string.
- **Impact:** Minor drift risk (e.g., `write_registry` uses `with_extension("json.tmp")`, which
  replaces an existing extension, while `write_json_atomic` appends `.tmp` preserving it). One
  shared helper would make the atomic-write invariant explicit and testable.
- **Suggested fix:** Promote `conversations::write_json_atomic<T: Serialize>` to a small shared
  module (e.g. `src/fsutil.rs`) and call it from all three sites.
- **Confidence:** High

#### [F-011] Duplicated timestamp helpers across modules
- **Category:** redundant
- **Severity:** Low
- **Location:** `src-tauri/src/conversations.rs:336-341` (`now_secs`),
  `src-tauri/src/model_registry.rs:939-944` (`now_secs`),
  `src-tauri/src/server.rs:1153-1165` (`created_timestamp` + `timestamp_nanos`)
- **Finding:** At least two identical `now_secs()` functions and a couple of near-identical epoch
  formatters are copy-pasted per module.
- **Impact:** Pure noise; trivial to consolidate alongside F-010's shared module.
- **Suggested fix:** One `now_secs()` (and the RFC-3339 formatter already in `tools.rs` could move
  to the same place if the server ever wants shared formatting).
- **Confidence:** High

#### [F-012] Duplicated `FakeProvider` test scaffolding
- **Category:** redundant
- **Severity:** Low
- **Location:** `src-tauri/src/engine.rs:675-736` (`FakeProvider`, `fake_loader`) and
  `src-tauri/src/server.rs:1176-1338` (`FakeProvider`, `fake_loader`, `loaded_fake_engine`,
  `FakeToolProvider`, …)
- **Finding:** Both test modules independently define a `FakeProvider` implementing `TextLlm`, with
  the server version adding a tool variant. They're close enough that the engine one could be reused
  (or a shared `src/test_support.rs` gated under `#[cfg(test)]` could host them).
- **Impact:** Test-only duplication; the two copies have already drifted slightly (descriptor
  capabilities, token ids). A shared fake keeps them honest.
- **Suggested fix:** Move the fakes behind a `#[cfg(test)]` pub(crate) test-support module and
  import from both `#[cfg(test)]` blocks.
- **Confidence:** High

#### [F-013] Conversation list is re-read from disk on every persist within a turn
- **Category:** efficiency
- **Severity:** Low
- **Location:** `src-tauri/src/conversations.rs:135-185` (`list_conversations_in_dir` opens every
  `.meta` sidecar) called via `main.rs:187` (`list_conversations`) ← `src/main.jsx:281-291`
  (`refreshConversations`), which `persistConversation` (`src/main.jsx:339-357`) awaits after every
  save
- **Finding:** Each `persistConversation` calls `refreshConversations`, and a single send can persist
  many times (user message, each assistant turn, each tool-result batch — up to `MAX_TOOL_STEPS=8`).
  Each persist triggers a full `read_dir` + per-sidecar `read_to_string` + parse of *all*
  conversations.
- **Impact:** Bounded (persist is at turn boundaries, not per token) and fine at hobby scale, but it
  scales linearly with history length and is O(turns × conversations) per send.
- **Suggested fix:** Either (a) have `save_conversation` return the updated `ConversationMetadata`
  and update the cache in place instead of re-listing, or (b) debounce `refreshConversations` to the
  end of the send loop. The sidecar design already makes (a) cheap.
- **Confidence:** High

---

## Informational

#### [F-014] `max_new_tokens` is not clamped at the engine boundary
- **Category:** Info
- **Severity:** Info
- **Location:** `src-tauri/src/engine.rs:274-298` (`GenerateRequest::into_core` passes
  `max_new_tokens` through unchanged)
- **Finding:** The engine doesn't clamp the request's `max_new_tokens` against the loaded
  provider's `TextLlmCapabilities::max_new_tokens`; it relies on the provider's `validate` to
  reject an over-limit value (the `FakeProvider` test exercises exactly this). That's a reasonable
  boundary choice; this note exists only so a reader knows the clamp is intentionally downstream.
- **Confidence:** High

#### [F-015] Test temp dirs are not cleaned up on panic
- **Category:** Info
- **Severity:** Info
- **Location:** `src-tauri/src/conversations.rs:872-878` (`test_dir`) and
  `src-tauri/src/model_registry.rs:1271-1276` (`snapshot_dir`); every test does
  `let _ = fs::remove_dir_all(dir);` only as its last statement
- **Finding:** If a test fails (panic) before its final `remove_dir_all`, the temp dir leaks under
  `std::env::temp_dir()`. Each is namespaced with `std::process::id()` so they don't collide, but
  they accumulate across failed runs.
- **Impact:** Negligible for CI; mildly annoying for local TDD. A `Drop` guard (or the `tempfile`
  crate) would make cleanup automatic. Purely a polish item.
- **Confidence:** High

---

## Themes and systemic observations

1. **Dependency trust model (F-001).** The single highest-leverage hardening is moving the three
   `core-llm`/`mlx-llm`/`candle-llm` git deps off a bare `branch = "main"`. The unification
   problem the comments describe is real and worth preserving, but a shared `rev` solves it without
   making every lockfile refresh a trust event. This is the one finding worth treating as structural
   rather than per-finding.

2. **The HTTP API is the trust boundary, not the frontend.** Several findings (F-002 image decode,
   F-003 CORS, F-008 token compare) are low-impact when the server is loopback-only and
   self-inflicted by the app's own frontend, but become real the moment a user enables `allow_lan`.
   The code already gates `0.0.0.0` behind an explicit opt-in and warns the user; the remaining work
   is to make the API defensive *as if* it were always exposed, since that's the documented
   OpenAI-compatible surface third-party callers use.

3. **Duplication is the main maintainability drag (F-010, F-011, F-012, F-013).** Nothing dramatic —
   a shared `fsutil`/`time`/test-support module and a `return-metadata-from-save` tweak would clear
   most of it. The sidecar-based conversation store is genuinely well designed; the duplication is
   incidental, not architectural.

4. **Safety posture is mostly good but under-documented in two spots (F-006, F-009).** `unsafe` is
   forbidden at workspace level and path-traversal is handled carefully with tests; the two gaps
   (the unexplained crate-level `allow` and the trim/use mismatch) are consistency issues, not deep
   flaws.

5. **Comment quality is a genuine strength.** Nearly every non-obvious decision — sidecar
   extensions, video frame shape, tool-call delta granularity, core-llm unification — has a comment
   explaining *why*. F-005 (stale DEV-PINs) stands out precisely because the rest of the comment
   hygiene is this good.

---

## Coverage notes

- **Reviewed in full:** all tracked source under `src-tauri/src/` (6 modules + `main.rs`/`lib.rs`/
  `build.rs`), the complete `src/main.jsx`, `src/styles.css` (size/skim only — 907 lines of styling,
  no logic), all manifests/configs (`Cargo.toml`, `package.json`, `tauri.conf.json`,
  `capabilities/default.json`, `Entitlements.plist`, `vite.config.js`, `eslint.config.js`), and all
  Markdown docs.
- **Integration tests:** `src-tauri/tests/qwen3vl_tools_server.rs` read in full; the other seven
  (`qwen35_*`, `qwen3vl_*`) were skimmed and confirmed to follow the same env-gated real-weights
  pattern (`MLX_LLM_QWEN3VL_MODEL` / `MLX_LLM_QWEN35_MODEL` gates, `#[ignore]`, macOS-only). If you
  want, a follow-up can deep-read each.
- **Excluded (build outputs / vendored):** `node_modules/`, `target/`, `dist/`, binary icons under
  `src-tauri/icons/`.
- **Not statically verifiable:** runtime behavior of the MLX/Candle backends (they live in the
  upstream crates), and any `core-llm` contract behavior asserted in F-004 (provider honoring the
  cancel flag) — both flagged Medium/Low confidence where they affect a recommendation.
