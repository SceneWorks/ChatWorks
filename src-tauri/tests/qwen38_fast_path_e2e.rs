//! Epic sc-24128 acceptance test 5 (story sc-24140): ChatWorks end to end on the pinned inference
//! runtime **with the shipped defaults**. Qwen3.8-27B streams, calls a tool, cancels mid-stream
//! and unloads through the same engine and OpenAI server the desktop app runs. The Tauri shell is
//! the only part left out; `docs/AT5_FAST_PATH_E2E.md` has a manual checklist for it.
//!
//! What runs, in order (step numbers match the epic's AT5 and the sealed log):
//!
//! 1. Load Qwen3.8-27B the way the Models screen does ([`app_load_request`] under
//!    `AppSettings::default()` and the linked runtime's backend capabilities): dense (`quantize:
//!    None`), CUDA graphs settled off, speculative decoding `auto`.
//! 2. Streaming chat completion with **no `mtp` field**: the server's default resolves it to the
//!    MTP head (K = 3, static KV cache, CUDA graphs off with a named reason), and the answer is
//!    right.
//! 3. A tool call, streamed and non-streamed: `finish_reason = tool_calls`, and the arguments are
//!    JSON matching the tool schema, still decoded on the MTP path.
//! 4. Mid-stream cancel three ways: (a) the HTTP client drops the SSE stream (server watch ->
//!    cancel), (b) a caller-owned `CancelFlag` tripped from the token callback, (c) the desktop
//!    Stop button's `EngineHandle::cancel`. After that, a greedy request is token-identical to the
//!    same request on the fresh load, so no cache or ring state leaked from a cancelled verify.
//! 5. The Models screen's **Reload** (the served source loaded again while resident) releases the
//!    resident copy first, so it succeeds with one copy on the device. Then `unload_model()` gives
//!    device memory back to within [`MEMORY_TOLERANCE_MIB`] of the pre-load level, and serving the
//!    model again reproduces the fresh-load reference token for token.
//! 6. If `CHATWORKS_QWEN3_8B_SNAPSHOT` is set, a model with no MTP head (Qwen3-8B) serves the step-2
//!    request under the default temperature + top-p: `proposer = none`, static KV, device sampler.
//!    The product contract carries only the sampler *label*, so on the CUDA build the same request
//!    also runs through the pinned runtime's llama provider, whose own record counts the logits rows
//!    copied to the host (`logits_to_host == 0`).
//! 7. Every check's observed values go into a sealed JSON record at `CHATWORKS_E2E_OUTPUT`, which
//!    must be outside the repository. The record is labelled [`HARDWARE_LABEL`] and carries the
//!    ChatWorks commit and the inference pin from `Cargo.lock`. It is written once
//!    (`create_new`), stamped with a SHA-256 over its content and made read-only.
//!
//! Soft checks do not stop the run, so one failure still leaves a complete record. The test fails
//! at the end if any check failed. A broken prerequisite (the model does not load, the server does
//! not start) aborts, and the record is still sealed with `outcome = "aborted"`.
//!
//! ```text
//! # Windows (PowerShell, MSVC vcvars loaded), one GPU pinned by PCI bus order:
//! $env:CUDA_DEVICE_ORDER = "PCI_BUS_ID"; $env:CUDA_VISIBLE_DEVICES = "0"
//! $env:CHATWORKS_QWEN38_SNAPSHOT = "E:\huggingface\hub\models--Qwen--Qwen3.8-27B\snapshots\<rev>"
//! $env:CHATWORKS_QWEN3_8B_SNAPSHOT = "E:\huggingface\hub\models--Qwen--Qwen3-8B\snapshots\<rev>"  # optional
//! $env:CHATWORKS_E2E_OUTPUT = "C:\evidence\at5-<date>.json"   # outside the repo; never overwritten
//! cargo test --no-default-features --features cuda --test qwen38_fast_path_e2e -- --ignored --nocapture
//! ```

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use chatworks::app_settings::AppSettings;
use chatworks::core_llm::CancelFlag;
use chatworks::engine::{
    DecodeReportPayload, DecodeStatusPayload, EngineHandle, EngineStatus, GenerateRequest,
    GenerateResponse, LoadTransitionReason, StreamPayload,
};
use chatworks::model_registry::{app_load_request, ModelEntry};
use chatworks::server::{OpenAiServerConfig, OpenAiServerHandle};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

/// The hardware this record is taken on (epic sc-24128's reference box).
const HARDWARE_LABEL: &str = "RTX Pro 6000 / sm_120";
/// How close device memory used must come back to its pre-load level after an unload.
const MEMORY_TOLERANCE_MIB: i64 = 1024;
/// How long an unload may take to hand device memory back before the check reads it.
const MEMORY_SETTLE: Duration = Duration::from_secs(20);
/// Content chunks (tokens) read before a stream is cancelled.
const CANCEL_AFTER_TOKENS: usize = 8;
/// The cancelled requests' token budget; a cancel must stop well short of it.
const CANCEL_MAX_TOKENS: u32 = 1024;
/// The greedy determinism probe's token budget.
const PROBE_MAX_TOKENS: u32 = 64;

const CAPITAL_PROMPT: &str = "What is the capital of France? Answer in one short sentence.";
const CAPITAL_KEYWORD: &str = "paris";
const LONG_PROMPT: &str =
    "Write a detailed, multi-paragraph essay on the history of the printing press, from Gutenberg \
     to the present day.";
const PROBE_PROMPT: &str =
    "List the first twenty prime numbers, separated by commas, then name the largest of them.";
const TOOL_PROMPT: &str = "What is the weather in Paris right now? Use the get_weather tool.";

#[test]
#[ignore = "needs a CUDA build, a pinned GPU, CHATWORKS_QWEN38_SNAPSHOT and CHATWORKS_E2E_OUTPUT"]
fn qwen38_fast_path_end_to_end() {
    let snapshot = required_env("CHATWORKS_QWEN38_SNAPSHOT");
    let output = sealed_output_path(&required_env("CHATWORKS_E2E_OUTPUT"));
    let gpu = pinned_gpu();
    let mut rec = Record::new(output);
    rec.meta("record", json!("epic sc-24128 AT5 (story sc-24140)"));
    rec.meta("hardware_label", json!(HARDWARE_LABEL));
    rec.meta("chatworks", chatworks_revision());
    rec.meta("inference_pin", inference_pin());
    rec.meta("pinned_gpu", gpu_identity(&gpu));
    rec.meta(
        "models",
        json!({
            "qwen3.8-27b": snapshot,
            "qwen3-8b": std::env::var("CHATWORKS_QWEN3_8B_SNAPSHOT").ok(),
        }),
    );

    // The settings a fresh install ships with. The server and the load request are both built
    // from them, the same way `main.rs` does for the desktop app.
    let settings = AppSettings::default();
    rec.meta(
        "shipped_settings",
        json!({
            "sampling": to_json(&settings.sampling),
            "runtime": to_json(&settings.runtime),
        }),
    );

    let engine = EngineHandle::spawn();
    // Every finished generation's decode status, as the desktop's `engine://decode` push sees it.
    // Case 4(a)'s cancelled stream is only visible here.
    let decode_events: Arc<Mutex<Vec<DecodeStatusPayload>>> = Arc::default();
    let sink = Arc::clone(&decode_events);
    engine.observe_generations(move |status| {
        if let Ok(mut events) = sink.lock() {
            events.push(status);
        }
    });

    let initial = engine_status(&engine);
    rec.meta("execution_backend", json!(initial.execution_backend));
    rec.meta(
        "backend_capabilities",
        to_json(&initial.backend_capabilities),
    );
    rec.require(
        "0",
        "the linked runtime is Candle CUDA",
        initial.execution_backend == "candle-cuda",
        json!(initial.execution_backend),
    );
    rec.check(
        "0",
        "the load device is sm_120 (the record's hardware label)",
        initial.backend_capabilities.compute_capability.as_deref() == Some("sm_120"),
        to_json(&initial.backend_capabilities.compute_capability),
    );
    rec.require(
        "0",
        "no model is loaded before step 1",
        initial.loaded.is_none(),
        json!(initial.loaded.is_some()),
    );
    let memory_before_load = gpu_memory_used_mib(&gpu);
    rec.meta("device_memory_before_load_mib", json!(memory_before_load));

    // ---- Step 1: load with the shipped defaults ------------------------------------------------
    rec.check(
        "1",
        "sampling_defaults.mtp_mode == auto",
        settings.sampling.mtp_mode == "auto",
        json!(settings.sampling.mtp_mode),
    );
    rec.check(
        "1",
        "sampling_defaults.mtp_draft_tokens == 3",
        settings.sampling.mtp_draft_tokens == 3,
        json!(settings.sampling.mtp_draft_tokens),
    );
    rec.check(
        "1",
        "runtime.cuda_graphs ships off",
        !settings.runtime.cuda_graphs,
        json!(settings.runtime.cuda_graphs),
    );
    let qwen38 = registry_entry(&snapshot, "Qwen3.8-27B", "Qwen/Qwen3.8-27B");
    let request = app_load_request(&qwen38, None, || Ok(settings.clone()))
        .unwrap_or_else(|error| rec.abort("1", &format!("load request: {error}")));
    rec.observe("1", "load_request", to_json(&request));
    rec.check(
        "1",
        "load request is dense (quantize: None)",
        request.quantize.is_none(),
        to_json(&request.quantize),
    );
    rec.check(
        "1",
        "load request sends the saved CUDA-graph switch (Some(false)) where the runtime honours it",
        request.cuda_graphs
            == initial
                .backend_capabilities
                .cuda_graphs
                .supported
                .then_some(false),
        json!({
            "cuda_graphs": request.cuda_graphs,
            "runtime_supports_switch": initial.backend_capabilities.cuda_graphs.supported,
        }),
    );
    let started = Instant::now();
    let status = load(&mut rec, &engine, request.clone(), "1");
    rec.observe("1", "load_seconds", json!(started.elapsed().as_secs_f64()));
    check_qwen38_load(&mut rec, "1", &status);
    let memory_loaded = gpu_memory_used_mib(&gpu);
    rec.observe("1", "device_memory_loaded_mib", json!(memory_loaded));

    // The fresh-load reference for step 4's clean-state check: the first generation this load runs.
    let reference = greedy_probe(&engine, &settings);
    rec.observe("4", "fresh_load_probe", reference.summary());
    rec.check(
        "4",
        "the fresh-load greedy probe decodes on the MTP path",
        reference.proposer() == Some("mtp"),
        json!(reference.proposer()),
    );

    let server = OpenAiServerHandle::new();
    let server_status = server
        .start(
            OpenAiServerConfig {
                port: 0,
                sampling_defaults: settings.sampling.clone(),
                ..OpenAiServerConfig::default()
            },
            engine.clone(),
        )
        .unwrap_or_else(|error| rec.abort("server", &format!("start OpenAI server: {error}")));
    let addr = server_status
        .bound_addr
        .unwrap_or_else(|| rec.abort("server", "the OpenAI server reported no bound address"));
    let url = format!("http://{addr}/v1/chat/completions");
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(900))
        .build()
        .expect("HTTP client");

    // ---- Step 2: streaming chat completion with no `mtp` field ---------------------------------
    let body = json!({
        "model": "qwen3.8-27b",
        "stream": true,
        "messages": [{"role": "user", "content": CAPITAL_PROMPT}],
    });
    rec.observe("2", "request", body.clone());
    rec.check(
        "2",
        "the request carries no mtp field (server default applies)",
        body.get("mtp").is_none(),
        json!(body.get("mtp")),
    );
    let stream = stream_chat(&client, &url, &body, None);
    rec.observe("2", "stream", stream.summary());
    check_stream_basics(&mut rec, "2", &stream, "stop");
    check_streamed_text(&mut rec, "2", &stream);
    rec.check(
        "2",
        "the answer is coherent (names Paris)",
        stream.content.to_lowercase().contains(CAPITAL_KEYWORD),
        json!(stream.content),
    );
    let decode = stream.decode();
    check_mtp_decode(&mut rec, "2", decode.as_ref());
    rec.check(
        "2",
        "kv_cache == static",
        decode.as_ref().map(|d| d.kv_cache.as_str()) == Some("static"),
        json!(decode.as_ref().map(|d| &d.kv_cache)),
    );
    // Graphs ship off: the switch is off for the generation and no step goes through the graph
    // runner (`path = none`), so there is no eager fallback to name. With the switch on, the
    // runtime names the reason for every eager step (`fallback_reason`); the record keeps it.
    rec.check(
        "2",
        "cuda_graphs disabled (switch off, no step through the graph runner)",
        decode.as_ref().is_some_and(|d| {
            let graphs = &d.cuda_graphs;
            !graphs.enabled && graphs.path == "none" && graphs.replayed == 0 && graphs.captured == 0
        }),
        json!(decode.as_ref().map(|d| to_json(&d.cuda_graphs))),
    );
    let accepted = stream
        .terminal_field("chatworks_mtp")
        .and_then(|mtp| mtp["accepted_tokens"].as_u64());
    rec.check(
        "2",
        "chatworks_mtp.accepted_tokens > 0",
        accepted.is_some_and(|n| n > 0),
        json!(stream.terminal_field("chatworks_mtp")),
    );

    // ---- Step 3: tool call, non-streamed and streamed ------------------------------------------
    let tools = weather_tool();
    let tool_body = |stream: bool| {
        json!({
            "model": "qwen3.8-27b",
            "stream": stream,
            "max_tokens": 256,
            "tools": tools,
            "messages": [{"role": "user", "content": TOOL_PROMPT}],
        })
    };
    let response: Value = client
        .post(&url)
        .json(&tool_body(false))
        .send()
        .and_then(|response| response.json())
        .unwrap_or_else(|error| rec.abort("3", &format!("non-streamed tool call: {error}")));
    rec.observe("3", "non_streamed_response", response.clone());
    let choice = &response["choices"][0];
    rec.check(
        "3",
        "non-streamed: finish_reason == tool_calls",
        choice["finish_reason"] == "tool_calls",
        choice["finish_reason"].clone(),
    );
    check_weather_call(
        &mut rec,
        "3",
        "non-streamed",
        &choice["message"]["tool_calls"],
    );
    let non_streamed_decode = decode_from(&response["chatworks_decode"]);
    check_mtp_decode(&mut rec, "3", non_streamed_decode.as_ref());

    let tool_stream = stream_chat(&client, &url, &tool_body(true), None);
    rec.observe("3", "streamed", tool_stream.summary());
    check_stream_basics(&mut rec, "3", &tool_stream, "tool_calls");
    let streamed_calls = tool_stream
        .terminal
        .as_ref()
        .map(|chunk| chunk["choices"][0]["delta"]["tool_calls"].clone())
        .unwrap_or(Value::Null);
    check_weather_call(&mut rec, "3", "streamed", &streamed_calls);
    check_mtp_decode(&mut rec, "3", tool_stream.decode().as_ref());

    // ---- Step 4: mid-stream cancel -------------------------------------------------------------
    // (a) The HTTP client drops the SSE response after a few tokens; the server's watch trips the
    // request's cancel flag. The cancelled generation is seen through the decode push.
    let events_before = decode_events.lock().map(|events| events.len()).unwrap_or(0);
    let cancel_body = json!({
        "model": "qwen3.8-27b",
        "stream": true,
        "max_tokens": CANCEL_MAX_TOKENS,
        "messages": [{"role": "user", "content": LONG_PROMPT}],
    });
    let dropped = stream_chat(&client, &url, &cancel_body, Some(CANCEL_AFTER_TOKENS));
    let dropped_at = Instant::now();
    // The engine actor is serial, so a status reply means the cancelled generation has returned.
    let after_drop = engine_status(&engine);
    let settle = dropped_at.elapsed();
    let cancelled_event = decode_events
        .lock()
        .ok()
        .and_then(|events| events.get(events_before).cloned());
    let dropped_decode = cancelled_event
        .as_ref()
        .and_then(|event| event.last_decode.clone());
    // Each target forward yields at most K + 1 tokens, so this bounds what the generation produced.
    let token_bound = dropped_decode
        .as_ref()
        .map(|decode| decode.target_forwards * (u64::from(decode.draft_tokens.unwrap_or(0)) + 1));
    rec.observe(
        "4a",
        "client_drop",
        json!({
            "content_chunks_read": dropped.content_chunks,
            "done_seen_before_drop": dropped.done,
            "engine_idle_after_seconds": settle.as_secs_f64(),
            "cancelled_generation_decode": to_json(&dropped_decode),
            "generated_token_upper_bound": token_bound,
        }),
    );
    rec.check(
        "4a",
        "the client read tokens and dropped the stream before [DONE]",
        dropped.content_chunks >= CANCEL_AFTER_TOKENS && !dropped.done,
        json!({"chunks": dropped.content_chunks, "done": dropped.done}),
    );
    rec.check(
        "4a",
        "the dropped stream's generation stopped well short of max_tokens",
        token_bound.is_some_and(|bound| bound < u64::from(CANCEL_MAX_TOKENS) / 2),
        json!({"upper_bound": token_bound, "max_tokens": CANCEL_MAX_TOKENS}),
    );
    rec.check(
        "4a",
        "the cancelled generation ran on the MTP path",
        dropped_decode.as_ref().map(|d| d.proposer.as_str()) == Some("mtp"),
        json!(dropped_decode.as_ref().map(|d| &d.proposer)),
    );
    rec.check(
        "4a",
        "the model is still served after the drop",
        after_drop.loaded.is_some(),
        json!(after_drop.loaded.is_some()),
    );
    rec.check(
        "4a",
        "no generation is left in flight (Stop has nothing to cancel)",
        !engine.cancel(),
        json!("engine.cancel() while idle"),
    );

    // (b) A caller-owned flag tripped from the token callback (the server's own cancel path).
    let flag = CancelFlag::new();
    let tripped = flag.clone();
    let mut seen = 0_usize;
    let flagged = engine.generate_with_cancel(cancel_request(&settings), flag, |event| {
        if matches!(event, StreamPayload::Token { .. }) {
            seen += 1;
            if seen == CANCEL_AFTER_TOKENS {
                tripped.cancel();
            }
        }
    });
    check_cancelled(&mut rec, "4b", "generate_with_cancel", flagged, seen);

    // (c) The desktop Stop button: its `stop_generation` command calls `EngineHandle::cancel`.
    let stopper = engine.clone();
    let mut seen = 0_usize;
    let mut stop_hit = None;
    let stopped = engine.generate(cancel_request(&settings), |event| {
        if matches!(event, StreamPayload::Token { .. }) {
            seen += 1;
            if seen == CANCEL_AFTER_TOKENS {
                stop_hit = Some(stopper.cancel());
            }
        }
    });
    rec.check(
        "4c",
        "Stop found the in-flight generation",
        stop_hit == Some(true),
        json!(stop_hit),
    );
    check_cancelled(
        &mut rec,
        "4c",
        "Stop button (EngineHandle::cancel)",
        stopped,
        seen,
    );

    // Clean state after three mid-verify cancels: token-identical to the fresh load.
    let after_cancel = greedy_probe(&engine, &settings);
    rec.observe("4", "after_cancel_probe", after_cancel.summary());
    check_same_tokens(
        &mut rec,
        "4",
        "after the cancels",
        &reference,
        &after_cancel,
    );

    // ---- Step 5: Reload in place, unload, device memory, serve again ----------------------------
    // (a) Models -> Reload: the served source is loaded again while it is resident (what the
    // screen offers after a CUDA-graph setting change). The engine releases the resident copy
    // first; two bf16 copies (2 x ~52 GB) would not fit on a 96 GB card.
    let started = Instant::now();
    let in_place = engine.load_model(request.clone());
    let memory_reloaded = gpu_memory_used_mib(&gpu);
    match &in_place {
        Ok(status) => {
            rec.observe(
                "5",
                "reload_in_place_seconds",
                json!(started.elapsed().as_secs_f64()),
            );
            rec.check(
                "5",
                "Reload in place: the served copy was released before the load",
                status.load_transition.as_ref().is_some_and(|transition| {
                    transition.reason == LoadTransitionReason::Reload
                        && transition.released_source == request.source
                }),
                to_json(&status.load_transition),
            );
            check_qwen38_load(&mut rec, "5", status);
            rec.check(
                "5",
                "Reload in place: one copy resident (device memory within 1 GiB of the first load)",
                (memory_reloaded - memory_loaded).abs() <= MEMORY_TOLERANCE_MIB,
                json!({
                    "first_load_mib": memory_loaded,
                    "after_reload_mib": memory_reloaded,
                    "tolerance_mib": MEMORY_TOLERANCE_MIB,
                }),
            );
        }
        Err(error) => {
            rec.check(
                "5",
                "Reload in place: the served model reloads",
                false,
                json!(error),
            );
        }
    }
    if engine_status(&engine).loaded.is_none() {
        load(&mut rec, &engine, request.clone(), "5");
    }
    let in_place_probe = greedy_probe(&engine, &settings);
    rec.observe("5", "reload_in_place_probe", in_place_probe.summary());
    check_same_tokens(
        &mut rec,
        "5",
        "after Reload in place",
        &reference,
        &in_place_probe,
    );

    // (b) Models -> Unload model, then (c) serve it again and generate once.
    let unloaded = engine
        .unload_model()
        .unwrap_or_else(|error| rec.abort("5", &format!("unload: {error}")));
    rec.check(
        "5",
        "status.loaded is None after unload",
        unloaded.loaded.is_none(),
        json!(unloaded.loaded.as_ref().map(|model| &model.name)),
    );
    let memory_unloaded = settled_memory(&gpu, memory_before_load);
    rec.check(
        "5",
        "device memory returns to within 1 GiB of the pre-load level",
        (memory_unloaded - memory_before_load).abs() <= MEMORY_TOLERANCE_MIB,
        json!({
            "before_load_mib": memory_before_load,
            "loaded_mib": memory_loaded,
            "after_unload_mib": memory_unloaded,
            "tolerance_mib": MEMORY_TOLERANCE_MIB,
        }),
    );
    let reload = load(&mut rec, &engine, request, "5");
    check_qwen38_load(&mut rec, "5", &reload);
    let reloaded = greedy_probe(&engine, &settings);
    rec.observe("5", "reload_probe", reloaded.summary());
    check_same_tokens(
        &mut rec,
        "5",
        "after an unload and a new load",
        &reference,
        &reloaded,
    );

    // ---- Step 6: a model with no MTP head under temperature + top-p ----------------------------
    match std::env::var("CHATWORKS_QWEN3_8B_SNAPSHOT") {
        Ok(qwen3) if !qwen3.trim().is_empty() => {
            // One large model resident at a time: unload the 27B before loading the next one.
            let _ = engine
                .unload_model()
                .unwrap_or_else(|error| rec.abort("6", &format!("unload 27B: {error}")));
            let entry = registry_entry(&qwen3, "Qwen3-8B", "Qwen/Qwen3-8B");
            let request = app_load_request(&entry, None, || Ok(settings.clone()))
                .unwrap_or_else(|error| rec.abort("6", &format!("load request: {error}")));
            let status = load(&mut rec, &engine, request.clone(), "6");
            let loaded = status.loaded.as_ref();
            rec.observe(
                "6",
                "provider",
                json!(loaded.map(|model| to_json(&model.provider))),
            );
            let mut llama_body = body.clone();
            llama_body["model"] = json!("qwen3-8b");
            let stream = stream_chat(&client, &url, &llama_body, None);
            rec.observe("6", "request", llama_body);
            rec.observe(
                "6",
                "request_sampling",
                json!({
                    "temperature": settings.sampling.temperature,
                    "top_p": settings.sampling.top_p,
                    "mtp_mode": settings.sampling.mtp_mode,
                }),
            );
            rec.observe("6", "stream", stream.summary());
            check_stream_basics(&mut rec, "6", &stream, "stop");
            check_streamed_text(&mut rec, "6", &stream);
            rec.check(
                "6",
                "the answer is coherent (names Paris)",
                stream.content.to_lowercase().contains(CAPITAL_KEYWORD),
                json!(stream.content),
            );
            let decode = stream.decode();
            rec.check(
                "6",
                "proposer == none (auto on a model without an MTP head)",
                decode.as_ref().map(|d| d.proposer.as_str()) == Some("none"),
                json!(decode.as_ref().map(|d| &d.proposer)),
            );
            rec.check(
                "6",
                "kv_cache == static",
                decode.as_ref().map(|d| d.kv_cache.as_str()) == Some("static"),
                json!(decode.as_ref().map(|d| &d.kv_cache)),
            );
            rec.check(
                "6",
                "sampler == device (the product contract's label)",
                decode.as_ref().map(|d| d.sampler.as_str()) == Some("device"),
                json!(decode.as_ref().map(|d| &d.sampler)),
            );
            // The label is only as truthful as the runtime's recording of its draws; the counter
            // that proves no logits row reached the host lives in the runtime's own record.
            #[cfg(feature = "cuda")]
            runtime_sampler_cross_check(&mut rec, &engine, &request, &settings);
            #[cfg(not(feature = "cuda"))]
            let _ = request;
        }
        _ => rec.observe(
            "6",
            "skipped",
            json!("CHATWORKS_QWEN3_8B_SNAPSHOT is not set"),
        ),
    }

    server.stop().ok();
    let last = engine
        .unload_model()
        .unwrap_or_else(|error| rec.abort("end", &format!("final unload: {error}")));
    let memory_end = settled_memory(&gpu, memory_before_load);
    rec.observe(
        "end",
        "final_unload",
        json!({
            "loaded": last.loaded.is_some(),
            "device_memory_mib": memory_end,
        }),
    );

    // ---- Step 7: seal the record ---------------------------------------------------------------
    let failures = rec.failures();
    let path = rec.seal("complete");
    println!("\nAT5 record sealed at {}", path.display());
    assert!(
        failures.is_empty(),
        "AT5 checks failed ({}):\n  {}\nrecord: {}",
        failures.len(),
        failures.join("\n  "),
        path.display()
    );
}

// ---- Engine and server helpers --------------------------------------------------------------

/// Step 6's `logits_to_host == 0`, read where it is counted. The product contract
/// (`core_llm::DecodeReport`) carries the sampler *label*, which is only as truthful as the
/// runtime's recording of its draws: before inference #1036 a stochastic step of the engine with
/// no drafts (the llama default path) drew on the host without recording it, so the label still
/// read `device`. The runtime's own record counts every logits row copied to the host, so this
/// loads the same snapshot through the pinned runtime's llama provider (after releasing the
/// engine's copy), runs the request the server built for step 6 (the shipped temperature + top-p,
/// MTP resolved off for a provider without a head) and reads that counter.
#[cfg(feature = "cuda")]
fn runtime_sampler_cross_check(
    rec: &mut Record,
    engine: &EngineHandle,
    request: &chatworks::engine::LoadModelRequest,
    settings: &AppSettings,
) {
    use chatworks::core_llm::{
        LoadSpec, Message, MtpMode, Sampling, TextLlm, TextLlmRequest, ThinkingMode,
    };
    use runtime_cuda::llm::LlamaProvider;

    // One copy resident at a time.
    let _ = engine.unload_model();
    let spec = LoadSpec {
        source: request.source.clone(),
        projector_source: None,
        quantize: None,
        cuda_graphs: request.cuda_graphs,
    };
    let provider = match LlamaProvider::load(&spec) {
        Ok(provider) => provider,
        Err(error) => {
            rec.check(
                "6",
                "runtime cross-check: the llama provider loads the snapshot",
                false,
                json!(error.to_string()),
            );
            return;
        }
    };
    let sampling = &settings.sampling;
    let core = TextLlmRequest {
        messages: vec![
            Message::system(sampling.system_prompt.clone()),
            Message::user(CAPITAL_PROMPT),
        ],
        sampling: Sampling {
            temperature: sampling.temperature,
            top_p: sampling.top_p,
            ..Sampling::default()
        },
        max_new_tokens: sampling.max_tokens,
        thinking: if sampling.disable_thinking {
            ThinkingMode::Disabled
        } else {
            ThinkingMode::Auto
        },
        mtp: MtpMode::Off,
        ..TextLlmRequest::default()
    };
    let output = provider.generate(&core, &mut |_| {});
    let record = provider.last_decode_record();
    let observed = json!({
        "generated": output.as_ref().ok().map(|out| &out.text),
        "error": output.as_ref().err().map(ToString::to_string),
        "path": record.map(|r| r.path.label()),
        "sampler": record.map(|r| r.sampler.label()),
        "device_draws": record.map(|r| r.sampler.device_draws),
        "host_draws": record.map(|r| r.sampler.host_draws),
        "logits_to_host": record.map(|r| r.sampler.logits_to_host),
        "generated_tokens": record.map(|r| r.generated_tokens),
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
    });
    rec.observe("6", "runtime_cross_check", observed.clone());
    rec.check(
        "6",
        "runtime cross-check: logits_to_host == 0 under temperature + top-p",
        output.is_ok() && record.is_some_and(|r| r.sampler.logits_to_host == 0),
        observed,
    );
}

fn engine_status(engine: &EngineHandle) -> EngineStatus {
    engine.status().expect("engine status")
}

fn load(
    rec: &mut Record,
    engine: &EngineHandle,
    request: chatworks::engine::LoadModelRequest,
    step: &str,
) -> EngineStatus {
    engine
        .load_model(request)
        .unwrap_or_else(|error| rec.abort(step, &format!("load: {error}")))
}

fn check_qwen38_load(rec: &mut Record, step: &str, status: &EngineStatus) {
    let loaded = status
        .loaded
        .as_ref()
        .unwrap_or_else(|| rec.abort(step, "the load returned no served model"));
    rec.observe(step, "loaded", to_json(loaded));
    rec.check(
        step,
        "the settled cuda_graphs == Some(false)",
        loaded.cuda_graphs == Some(false),
        json!(loaded.cuda_graphs),
    );
    let report = loaded.load_report.as_ref();
    let dense = report.is_some_and(|report| {
        report.requested.is_none()
            && !report.projections.is_empty()
            && report
                .projections
                .iter()
                .all(|projection| projection.kind == "dense")
    });
    rec.check(
        step,
        "a dense load report (nothing requested, only dense projections)",
        dense,
        json!(report.map(to_json)),
    );
    rec.check(
        step,
        "the provider advertises an MTP head and tool calling",
        loaded.provider.capabilities.mtp.is_some() && loaded.provider.capabilities.supports_tools,
        json!({
            "mtp": to_json(&loaded.provider.capabilities.mtp),
            "supports_tools": loaded.provider.capabilities.supports_tools,
        }),
    );
}

/// The desktop's generation request shape for the cancel cases: greedy, with the shipped
/// speculative mode.
fn cancel_request(settings: &AppSettings) -> GenerateRequest {
    engine_request(settings, LONG_PROMPT, CANCEL_MAX_TOKENS)
}

fn engine_request(settings: &AppSettings, prompt: &str, max_new_tokens: u32) -> GenerateRequest {
    let mtp = match settings.sampling.mtp_mode.as_str() {
        "enabled" => json!({"mode": "enabled", "draft_tokens": settings.sampling.mtp_draft_tokens}),
        mode => json!({ "mode": mode }),
    };
    serde_json::from_value(json!({
        "messages": [{"role": "user", "content": prompt}],
        "sampling": {"temperature": 0.0},
        "max_new_tokens": max_new_tokens,
        "disable_thinking": settings.sampling.disable_thinking,
        "mtp": mtp,
    }))
    .expect("engine request")
}

struct Probe {
    tokens: Vec<u32>,
    text: String,
    response: Result<GenerateResponse, String>,
}

impl Probe {
    fn proposer(&self) -> Option<&str> {
        self.response
            .as_ref()
            .ok()
            .and_then(|response| response.decode.as_ref())
            .map(|decode| decode.proposer.as_str())
    }

    fn summary(&self) -> Value {
        json!({
            "tokens": self.tokens.len(),
            "token_ids": self.tokens,
            "text": self.text,
            "finish_reason": self.response.as_ref().ok().map(|r| &r.finish_reason),
            "mtp": self.response.as_ref().ok().map(|r| to_json(&r.mtp)),
            "decode": self.response.as_ref().ok().map(|r| to_json(&r.decode)),
            "error": self.response.as_ref().err(),
        })
    }
}

/// A greedy request whose token ids must not depend on what the model served before it.
fn greedy_probe(engine: &EngineHandle, settings: &AppSettings) -> Probe {
    let mut tokens = Vec::new();
    let mut text = String::new();
    let response = engine.generate(
        engine_request(settings, PROBE_PROMPT, PROBE_MAX_TOKENS),
        |event| {
            if let StreamPayload::Token {
                id, text: piece, ..
            } = event
            {
                tokens.push(id);
                text.push_str(&piece);
            }
        },
    );
    Probe {
        tokens,
        text,
        response,
    }
}

fn check_same_tokens(rec: &mut Record, step: &str, when: &str, reference: &Probe, probe: &Probe) {
    let first_difference = reference
        .tokens
        .iter()
        .zip(&probe.tokens)
        .position(|(a, b)| a != b);
    rec.check(
        step,
        &format!("a greedy request {when} is token-identical to the fresh load"),
        !reference.tokens.is_empty() && reference.tokens == probe.tokens && probe.response.is_ok(),
        json!({
            "reference_tokens": reference.tokens.len(),
            "tokens": probe.tokens.len(),
            "first_difference": first_difference,
            "error": probe.response.as_ref().err(),
        }),
    );
}

fn check_cancelled(
    rec: &mut Record,
    step: &str,
    how: &str,
    result: Result<GenerateResponse, String>,
    tokens_seen: usize,
) {
    let observed = match &result {
        Ok(response) => json!({
            "finish_reason": response.finish_reason,
            "generated_tokens": response.usage.generated_tokens,
            "tokens_streamed": tokens_seen,
            "max_new_tokens": CANCEL_MAX_TOKENS,
            "decode": to_json(&response.decode),
            "mtp": to_json(&response.mtp),
        }),
        Err(error) => json!({ "error": error, "tokens_streamed": tokens_seen }),
    };
    rec.observe(step, "cancel", observed.clone());
    let response = result.as_ref().ok();
    rec.check(
        step,
        &format!("{how}: finish_reason == cancelled"),
        response.is_some_and(|r| r.finish_reason == "cancelled"),
        observed.clone(),
    );
    rec.check(
        step,
        &format!("{how}: stopped with fewer than max_new_tokens"),
        response.is_some_and(|r| r.usage.generated_tokens < CANCEL_MAX_TOKENS),
        json!(response.map(|r| r.usage.generated_tokens)),
    );
    rec.check(
        step,
        &format!("{how}: cancelled on the MTP path"),
        response
            .and_then(|r| r.decode.as_ref())
            .is_some_and(|d| d.proposer == "mtp"),
        json!(response
            .and_then(|r| r.decode.as_ref())
            .map(|d| &d.proposer)),
    );
}

fn check_mtp_decode(rec: &mut Record, step: &str, decode: Option<&DecodeReportPayload>) {
    rec.check(
        step,
        "decode proposer == mtp with draft_tokens == 3",
        decode.is_some_and(|d| d.proposer == "mtp" && d.draft_tokens == Some(3)),
        json!(decode.map(|d| json!({"proposer": d.proposer, "draft_tokens": d.draft_tokens}))),
    );
}

/// A text answer streams token by token. (A tool-call turn streams no deltas: the provider
/// surfaces the parsed call whole in the terminal chunk.)
fn check_streamed_text(rec: &mut Record, step: &str, stream: &SseTranscript) {
    rec.check(
        step,
        "more than one SSE delta before the terminal chunk",
        stream.deltas > 1,
        json!(stream.deltas),
    );
}

fn check_stream_basics(rec: &mut Record, step: &str, stream: &SseTranscript, finish: &str) {
    rec.check(
        step,
        "the stream ends with [DONE]",
        stream.done,
        json!(stream.done),
    );
    let finish_reason = stream
        .terminal
        .as_ref()
        .map(|chunk| chunk["choices"][0]["finish_reason"].clone());
    rec.check(
        step,
        &format!("the terminal chunk finishes with {finish}"),
        finish_reason.as_ref().and_then(Value::as_str) == Some(finish),
        json!(finish_reason),
    );
    rec.check(
        step,
        "the terminal chunk carries chatworks_decode",
        stream.decode().is_some(),
        json!(stream.terminal_field("chatworks_decode")),
    );
    rec.check(
        step,
        "no error event in the stream",
        stream.errors.is_empty(),
        json!(stream.errors),
    );
}

fn weather_tool() -> Value {
    json!([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city to get the weather for, e.g. 'Paris'"
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    }])
}

/// The call is `get_weather`, and its arguments are a JSON object that matches the schema: a
/// string `location` naming Paris, an optional `unit` from the enum, and nothing else.
fn check_weather_call(rec: &mut Record, step: &str, mode: &str, calls: &Value) {
    let call = &calls[0];
    let arguments = call["function"]["arguments"].as_str();
    let parsed = arguments.and_then(|text| serde_json::from_str::<Value>(text).ok());
    let matches_schema = parsed
        .as_ref()
        .and_then(Value::as_object)
        .is_some_and(|args| {
            let location_ok = args
                .get("location")
                .and_then(Value::as_str)
                .is_some_and(|city| city.to_lowercase().contains("paris"));
            let unit_ok = args.get("unit").map_or(true, |unit| {
                matches!(unit.as_str(), Some("celsius" | "fahrenheit"))
            });
            let known_keys = args.keys().all(|key| key == "location" || key == "unit");
            location_ok && unit_ok && known_keys
        });
    rec.check(
        step,
        &format!("{mode}: one get_weather function call"),
        calls.as_array().is_some_and(|calls| !calls.is_empty())
            && call["type"] == "function"
            && call["function"]["name"] == "get_weather",
        calls.clone(),
    );
    rec.check(
        step,
        &format!("{mode}: arguments parse as JSON matching the tool schema"),
        matches_schema,
        json!({"arguments": arguments, "parsed": parsed}),
    );
}

#[derive(Default)]
struct SseTranscript {
    /// Chunks before the terminal one that carried a content or reasoning delta.
    deltas: usize,
    content_chunks: usize,
    content: String,
    reasoning: String,
    terminal: Option<Value>,
    errors: Vec<Value>,
    done: bool,
}

impl SseTranscript {
    fn terminal_field(&self, key: &str) -> Option<Value> {
        self.terminal
            .as_ref()
            .map(|chunk| chunk[key].clone())
            .filter(|value| !value.is_null())
    }

    fn decode(&self) -> Option<DecodeReportPayload> {
        self.terminal_field("chatworks_decode")
            .and_then(|value| decode_from(&value))
    }

    fn summary(&self) -> Value {
        json!({
            "deltas": self.deltas,
            "content": self.content,
            "reasoning": self.reasoning,
            "terminal_chunk": self.terminal,
            "errors": self.errors,
            "done": self.done,
        })
    }
}

/// `DecodeReportPayload` is serialize-only on the wire; read the fields the checks need back out.
fn decode_from(value: &Value) -> Option<DecodeReportPayload> {
    let text = |key: &str| value[key].as_str().map(str::to_string);
    let count = |key: &str| value[key].as_u64().unwrap_or(0);
    let path = |key: &str| chatworks::engine::PathReportPayload {
        path: value[key]["path"].as_str().unwrap_or_default().to_string(),
        reason: value[key]["reason"].as_str().map(str::to_string),
    };
    let graphs = &value["cuda_graphs"];
    Some(DecodeReportPayload {
        path: text("path")?,
        proposer: text("proposer")?,
        draft_tokens: value["draft_tokens"]
            .as_u64()
            .and_then(|n| u32::try_from(n).ok()),
        sampler: text("sampler")?,
        kv_cache: text("kv_cache")?,
        attention: text("attention").unwrap_or_default(),
        cuda_graphs: chatworks::engine::CudaGraphsReportPayload {
            enabled: graphs["enabled"].as_bool()?,
            path: graphs["path"].as_str().unwrap_or_default().to_string(),
            replayed: graphs["replayed"].as_u64().unwrap_or(0),
            eager: graphs["eager"].as_u64().unwrap_or(0),
            captured: graphs["captured"].as_u64().unwrap_or(0),
            fallback_reason: graphs["fallback_reason"].as_str().map(str::to_string),
        },
        nvfp4_projections: path("nvfp4_projections"),
        fused_primitives: path("fused_primitives"),
        target_forwards: count("target_forwards"),
        proposed_tokens: count("proposed_tokens"),
        accepted_tokens: count("accepted_tokens"),
        replay_forwards: count("replay_forwards"),
    })
}

/// POST a streaming chat completion and read its SSE events. With `drop_after`, the response is
/// dropped (the client disconnects) once that many content chunks have arrived.
fn stream_chat(
    client: &reqwest::blocking::Client,
    url: &str,
    body: &Value,
    drop_after: Option<usize>,
) -> SseTranscript {
    let response = client
        .post(url)
        .json(body)
        .send()
        .expect("streaming request");
    assert!(
        response.status().is_success(),
        "streaming request failed: {}",
        response.status()
    );
    let mut transcript = SseTranscript::default();
    let mut reader = BufReader::new(response);
    let mut line = String::new();
    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        let Some(data) = line.trim_end().strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data == "[DONE]" {
            transcript.done = true;
            break;
        }
        let Ok(chunk) = serde_json::from_str::<Value>(data) else {
            transcript.errors.push(json!(data));
            continue;
        };
        if chunk.get("error").is_some() {
            transcript.errors.push(chunk);
            continue;
        }
        let choice = &chunk["choices"][0];
        if !choice["finish_reason"].is_null() {
            transcript.terminal = Some(chunk);
            continue;
        }
        let delta = &choice["delta"];
        if let Some(piece) = delta["content"].as_str() {
            transcript.content.push_str(piece);
            transcript.content_chunks += 1;
            transcript.deltas += 1;
        } else if let Some(piece) = delta["reasoning_content"].as_str() {
            transcript.reasoning.push_str(piece);
            transcript.deltas += 1;
        }
        if drop_after.is_some_and(|limit| transcript.content_chunks >= limit) {
            break;
        }
    }
    // Dropping the reader closes the connection without reading the rest of the stream.
    drop(reader);
    transcript
}

/// A registry entry for a local snapshot, as the Models screen registers an adopted HF cache
/// snapshot: dense (no load-time quantization), no projector.
fn registry_entry(snapshot: &str, name: &str, repo: &str) -> ModelEntry {
    serde_json::from_value(json!({
        "id": format!("at5-{name}"),
        "name": name,
        "repo": repo,
        "revision": Path::new(snapshot)
            .file_name()
            .and_then(|revision| revision.to_str())
            .unwrap_or("main"),
        "sourceUrl": format!("https://huggingface.co/{repo}"),
        "localPath": snapshot,
        "importedAt": 0,
        "fileCount": 0,
    }))
    .expect("registry entry")
}

fn to_json<T: serde::Serialize>(value: &T) -> Value {
    serde_json::to_value(value).unwrap_or_else(|error| json!(format!("unserializable: {error}")))
}

// ---- Environment, device and provenance ---------------------------------------------------

fn required_env(name: &str) -> String {
    std::env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| panic!("set {name} (see the module docs)"))
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

/// The record's path: a new file whose directory exists and lies outside the repository.
fn sealed_output_path(raw: &str) -> PathBuf {
    let path = PathBuf::from(raw);
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| panic!("CHATWORKS_E2E_OUTPUT must be an absolute file path: {raw}"));
    let parent = parent
        .canonicalize()
        .unwrap_or_else(|error| panic!("CHATWORKS_E2E_OUTPUT's directory must exist: {error}"));
    let repo = repo_root().canonicalize().expect("repository root");
    assert!(
        !parent.starts_with(&repo),
        "CHATWORKS_E2E_OUTPUT must be outside the repository ({})",
        repo.display()
    );
    assert!(
        !path.exists(),
        "CHATWORKS_E2E_OUTPUT already exists; a sealed record is never overwritten: {raw}"
    );
    parent.join(path.file_name().expect("record file name"))
}

/// The GPU the process is pinned to, as `nvidia-smi -i` names it. The pin is required so the
/// memory check reads the device the runtime loads onto (`cuda:0` = the first visible device).
fn pinned_gpu() -> String {
    let order = std::env::var("CUDA_DEVICE_ORDER").unwrap_or_default();
    let visible = std::env::var("CUDA_VISIBLE_DEVICES").unwrap_or_default();
    let first = visible
        .split(',')
        .next()
        .unwrap_or_default()
        .trim()
        .to_string();
    assert!(
        order == "PCI_BUS_ID" && !first.is_empty(),
        "pin the GPU: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<nvidia-smi index or UUID>"
    );
    first
}

fn nvidia_smi(gpu: &str, fields: &str) -> Option<String> {
    let output = Command::new("nvidia-smi")
        .args([
            &format!("--query-gpu={fields}"),
            "--format=csv,noheader,nounits",
            "-i",
            gpu,
        ])
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn gpu_identity(gpu: &str) -> Value {
    let fields = "name,pci.bus_id,uuid,driver_version,memory.total,compute_cap";
    let row = nvidia_smi(gpu, fields).unwrap_or_default();
    let values: Vec<&str> = row.split(',').map(str::trim).collect();
    let mut identity = Map::new();
    identity.insert("nvidia_smi_index".into(), json!(gpu));
    for (key, value) in fields.split(',').zip(values) {
        identity.insert(key.to_string(), json!(value));
    }
    Value::Object(identity)
}

fn gpu_memory_used_mib(gpu: &str) -> i64 {
    nvidia_smi(gpu, "memory.used")
        .and_then(|value| value.parse().ok())
        .unwrap_or_else(|| panic!("nvidia-smi could not read memory.used for GPU {gpu}"))
}

/// Device memory used once an unload has settled: freed allocations can reach the driver a little
/// after the provider drops, so poll until the reading is back within tolerance or the settle time
/// runs out (then report the last reading).
fn settled_memory(gpu: &str, baseline: i64) -> i64 {
    let deadline = Instant::now() + MEMORY_SETTLE;
    loop {
        let used = gpu_memory_used_mib(gpu);
        if (used - baseline).abs() <= MEMORY_TOLERANCE_MIB || Instant::now() >= deadline {
            return used;
        }
        std::thread::sleep(Duration::from_millis(500));
    }
}

fn git(args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn chatworks_revision() -> Value {
    json!({
        "commit": git(&["rev-parse", "HEAD"]),
        "branch": git(&["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": git(&["status", "--porcelain", "--untracked-files=no"])
            .map(|status| !status.is_empty()),
    })
}

/// The inference source `Cargo.lock` resolved for the CUDA runtime bundle.
fn inference_pin() -> Value {
    let lock = std::fs::read_to_string(repo_root().join("Cargo.lock")).unwrap_or_default();
    let source = lock
        .split("[[package]]")
        .find(|package| package.contains("name = \"runtime-cuda\""))
        .and_then(|package| {
            package
                .lines()
                .find_map(|line| line.trim().strip_prefix("source = "))
        })
        .map(|source| source.trim_matches('"').to_string());
    let commit = source
        .as_deref()
        .and_then(|source| source.rsplit_once('#'))
        .map(|(_, commit)| commit.to_string());
    json!({ "package": "runtime-cuda", "source": source, "commit": commit })
}

// ---- The sealed record --------------------------------------------------------------------

struct Record {
    output: PathBuf,
    meta: Map<String, Value>,
    observations: Map<String, Value>,
    checks: Vec<Value>,
    sealed: bool,
}

impl Record {
    fn new(output: PathBuf) -> Self {
        Self {
            output,
            meta: Map::new(),
            observations: Map::new(),
            checks: Vec::new(),
            sealed: false,
        }
    }

    fn meta(&mut self, key: &str, value: Value) {
        self.meta.insert(key.to_string(), value);
    }

    fn observe(&mut self, step: &str, key: &str, value: Value) {
        let entry = self
            .observations
            .entry(format!("step_{step}"))
            .or_insert_with(|| Value::Object(Map::new()));
        if let Value::Object(map) = entry {
            map.insert(key.to_string(), value);
        }
    }

    fn check(&mut self, step: &str, name: &str, pass: bool, observed: Value) -> bool {
        println!(
            "[{}] step {step}: {name} -> {observed}",
            if pass { "PASS" } else { "FAIL" }
        );
        self.checks.push(json!({
            "step": step,
            "check": name,
            "pass": pass,
            "observed": observed,
        }));
        pass
    }

    fn require(&mut self, step: &str, name: &str, pass: bool, observed: Value) {
        if !self.check(step, name, pass, observed) {
            self.abort(step, name);
        }
    }

    fn abort(&mut self, step: &str, reason: &str) -> ! {
        self.check(step, reason, false, json!("aborted the run"));
        self.seal("aborted");
        panic!("AT5 aborted at step {step}: {reason}");
    }

    fn failures(&self) -> Vec<String> {
        self.checks
            .iter()
            .filter(|check| check["pass"] == false)
            .map(|check| {
                format!(
                    "step {}: {}",
                    check["step"].as_str().unwrap_or("?"),
                    check["check"].as_str().unwrap_or("?")
                )
            })
            .collect()
    }

    /// Write the record once: the SHA-256 of the record's canonical JSON (compact, every object's
    /// keys sorted) is stored beside it, the file is created fresh (never overwritten) and left
    /// read-only. The record is written in that canonical key order too, so the seal can be checked
    /// from the file alone, whichever `serde_json` map ordering the build links.
    fn seal(&mut self, outcome: &str) -> PathBuf {
        if self.sealed {
            return self.output.clone();
        }
        self.sealed = true;
        let failed = self.failures();
        let record = json!({
            "outcome": outcome,
            "sealed_at_unix": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|elapsed| elapsed.as_secs())
                .unwrap_or(0),
            "meta": self.meta,
            "summary": {
                "checks": self.checks.len(),
                "passed": self.checks.len() - failed.len(),
                "failed": failed.len(),
                "failed_checks": failed,
            },
            "checks": self.checks,
            "observations": self.observations,
        });
        let record = canonical(record);
        let digest = Sha256::digest(serde_json::to_vec(&record).expect("record JSON"));
        let sealed = json!({
            "seal": {
                "sha256": digest.iter().map(|byte| format!("{byte:02x}")).collect::<String>(),
                "over": "the compact JSON of `record`, every object's keys sorted, non-ASCII unescaped",
            },
            "record": record,
        });
        let body = serde_json::to_string_pretty(&sealed).expect("sealed JSON");
        let written = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&self.output)
            .and_then(|mut file| std::io::Write::write_all(&mut file, body.as_bytes()));
        match written {
            Ok(()) => {
                if let Ok(metadata) = std::fs::metadata(&self.output) {
                    let mut permissions = metadata.permissions();
                    permissions.set_readonly(true);
                    let _ = std::fs::set_permissions(&self.output, permissions);
                }
            }
            Err(error) => eprintln!(
                "AT5 record could not be written to {}: {error}",
                self.output.display()
            ),
        }
        self.output.clone()
    }
}

/// `value` with every object's keys in sorted order. A `serde_json` map keeps insertion order
/// when a dependency enables its `preserve_order` feature (Tauri's graph does), so the seal sorts
/// explicitly rather than relying on the map type.
fn canonical(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut entries: Vec<(String, Value)> = map.into_iter().collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            Value::Object(
                entries
                    .into_iter()
                    .map(|(key, value)| (key, canonical(value)))
                    .collect(),
            )
        }
        Value::Array(items) => Value::Array(items.into_iter().map(canonical).collect()),
        other => other,
    }
}

impl Drop for Record {
    fn drop(&mut self) {
        if !self.sealed {
            self.seal(if std::thread::panicking() {
                "aborted"
            } else {
                "incomplete"
            });
        }
    }
}
