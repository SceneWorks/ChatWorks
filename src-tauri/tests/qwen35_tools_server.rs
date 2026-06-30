//! Real-weights end-to-end test for Qwen3.6 tool ("function") calling **through the ChatWorks
//! OpenAI server** (sc-7771). Proves the full app path: an OpenAI chat request carrying a `tools`
//! offer → HTTP server → `core_llm::ToolSpec` → engine → `mlx-llama` Qwen3.6 → a parsed
//! `<tool_call>` block surfaced back as an OpenAI `tool_calls` response with `finish_reason`
//! `tool_calls`. Point `MLX_LLM_QWEN35_MODEL` at the 27B snapshot:
//!
//! ```text
//! MLX_LLM_QWEN35_MODEL=/path/to/Qwen3.6-27B \
//!   cargo test --test qwen35_tools_server -- --ignored --nocapture
//! ```

use chatworks::engine::{EngineHandle, LoadModelRequest};
use chatworks::server::{OpenAiServerConfig, OpenAiServerHandle};

#[test]
#[ignore = "needs a Qwen3.6 27B snapshot via MLX_LLM_QWEN35_MODEL"]
fn qwen35_tool_calling_over_openai_server() {
    let model = std::env::var("MLX_LLM_QWEN35_MODEL").expect("set MLX_LLM_QWEN35_MODEL");

    let engine = EngineHandle::spawn();
    let status = engine
        .load_model(LoadModelRequest {
            source: model,
            display_name: None,
            quantize: None,
            kv_cache_quant: None,
        })
        .expect("load 27B");
    // The provider must advertise tool calling once loaded (Qwen3.6's template renders a `tools`
    // section and it emits parseable `<tool_call>` blocks).
    assert!(
        status
            .loaded
            .as_ref()
            .map(|m| m.provider.capabilities.supports_tools)
            .unwrap_or(false),
        "loaded Qwen3.6 must report supports_tools"
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
        "model": "qwen3.6",
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
        "\n=== ChatWorks OpenAI tool call ===\n{}\n",
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
