//! Real-weights end-to-end test for Qwen3-VL tool ("function") calling **through the ChatWorks
//! OpenAI server** (sc-8082). Mirrors `qwen35_tools_server.rs`, but for the VLM checkpoint:
//! proves the full app path on Qwen3-VL — an OpenAI chat request carrying a `tools` offer →
//! HTTP server → `core_llm::ToolSpec` → engine → `mlx-llama` Qwen3-VL → a parsed `<tool_call>`
//! block surfaced back as an OpenAI `tool_calls` response with `finish_reason` `tool_calls`. The
//! tools contract (server.rs deltas, tools.rs execution) is provider-agnostic, so this is the same
//! path qwen35 exercises; this test confirms it holds for the qwen3_vl provider too.
//!
//! Point `MLX_LLM_QWEN3VL_MODEL` at the Qwen3-VL-8B-Instruct snapshot revision dir (the
//! `…/snapshots/<rev>/` directory holding `config.json` + weights):
//!
//! ```text
//! MLX_LLM_QWEN3VL_MODEL=/path/to/Qwen3-VL-8B-Instruct \
//!   cargo test --release --test qwen3vl_tools_server -- --ignored --nocapture
//! ```
//!
//! The MLX backend is macOS-only, so `#[ignore]` plus the env-var gate keep CI green elsewhere:
//! the test is opt-in and skips cleanly whenever `MLX_LLM_QWEN3VL_MODEL` is unset.

use chatworks::engine::{EngineHandle, LoadModelRequest};
use chatworks::server::{OpenAiServerConfig, OpenAiServerHandle};

#[test]
#[ignore = "needs a Qwen3-VL-8B-Instruct VLM snapshot via MLX_LLM_QWEN3VL_MODEL"]
fn qwen3vl_tool_calling_over_openai_server() {
    // Unset env var → skip cleanly (the harness runs this only under `--ignored`, and even then a
    // missing path must not be treated as a failure). The MLX provider is macOS-only, so non-macOS
    // hosts never have a runnable snapshot and fall through the same skip.
    let Ok(model) = std::env::var("MLX_LLM_QWEN3VL_MODEL") else {
        eprintln!("skipping: MLX_LLM_QWEN3VL_MODEL not set");
        return;
    };

    let engine = EngineHandle::spawn();
    let status = engine
        .load_model(LoadModelRequest {
            source: model,
            display_name: None,
            quantize: None,
        })
        .expect("load Qwen3-VL");
    // The provider must advertise tool calling once loaded (the qwen3_vl chat template renders a
    // `tools` section and emits parseable `<tool_call>` blocks — established by story E).
    assert!(
        status
            .loaded
            .as_ref()
            .map(|m| m.provider.capabilities.supports_tools)
            .unwrap_or(false),
        "loaded Qwen3-VL must report supports_tools"
    );
    // The same checkpoint is a VLM, so it must still advertise vision — tools do not displace it.
    assert!(
        status
            .loaded
            .as_ref()
            .map(|m| m.provider.capabilities.supports_vision)
            .unwrap_or(false),
        "loaded Qwen3-VL must also report supports_vision"
    );

    let server = OpenAiServerHandle::new();
    let server_status = server
        .start(
            OpenAiServerConfig {
                port: 0, // OS-assigned
                ..Default::default()
            },
            engine,
        )
        .expect("start OpenAI server");
    let addr = server_status.bound_addr.expect("bound addr");
    let url = format!("http://{addr}/v1/chat/completions");
    let client = reqwest::blocking::Client::new();

    let body = serde_json::json!({
        "model": "qwen3-vl",
        "stream": false,
        "max_tokens": 256,
        "temperature": 0.0,
        "tools": [{
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
                        }
                    },
                    "required": ["location"]
                }
            }
        }],
        "messages": [
            {"role": "user", "content": "What is the weather in Paris? Use the get_weather tool."}
        ]
    });
    let resp: serde_json::Value = client
        .post(&url)
        .json(&body)
        .send()
        .expect("send")
        .json()
        .expect("json");
    println!(
        "\n=== ChatWorks OpenAI tool call (Qwen3-VL) ===\n{}\n",
        serde_json::to_string_pretty(&resp).unwrap_or_default()
    );

    let choice = &resp["choices"][0];
    assert_eq!(
        choice["finish_reason"], "tool_calls",
        "a tool-call turn must finish with finish_reason=tool_calls"
    );

    let call = &choice["message"]["tool_calls"][0];
    assert_eq!(
        call["type"], "function",
        "tool call must be a function call"
    );
    assert_eq!(
        call["function"]["name"], "get_weather",
        "the model must call get_weather"
    );

    // OpenAI carries the arguments as a JSON-encoded string; it must decode to a Paris-grounded call.
    let arguments = call["function"]["arguments"]
        .as_str()
        .expect("function.arguments is a JSON string");
    let parsed: serde_json::Value =
        serde_json::from_str(arguments).expect("arguments decode as JSON");
    let location = parsed["location"].as_str().unwrap_or_default();
    assert!(
        location.to_lowercase().contains("paris"),
        "the call must ground on Paris, got arguments: {arguments:?}"
    );

    server.stop().expect("stop server");
}
