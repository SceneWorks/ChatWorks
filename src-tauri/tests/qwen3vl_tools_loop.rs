//! Real-weights end-to-end test of the full tool-calling LOOP the chat UI drives, for Qwen3-VL
//! (sc-8082). Mirrors `qwen35_tools_loop.rs`: model emits a tool call → the app executes a built-in
//! tool → the result re-enters the conversation → the model produces a grounded final answer.
//! Reuses the real `chatworks::tools` executor (what the `execute_tool` command runs). The loop is
//! provider-agnostic; this confirms it holds for the qwen3_vl VLM provider.
//!
//! Point `MLX_LLM_QWEN3VL_MODEL` at the Qwen3-VL-8B-Instruct snapshot revision dir:
//!
//! ```text
//! MLX_LLM_QWEN3VL_MODEL=/path/to/Qwen3-VL-8B-Instruct \
//!   cargo test --release --test qwen3vl_tools_loop -- --ignored --nocapture
//! ```
//!
//! The MLX backend is macOS-only, so `#[ignore]` plus the env-var gate keep CI green elsewhere.

use chatworks::engine::{EngineHandle, LoadModelRequest};
use chatworks::server::{OpenAiServerConfig, OpenAiServerHandle};
use chatworks::tools::{builtin_tool_specs, execute_builtin_tool};

#[test]
#[ignore = "needs a Qwen3-VL-8B-Instruct VLM snapshot via MLX_LLM_QWEN3VL_MODEL"]
fn qwen3vl_tool_loop_over_openai_server() {
    // Unset env var → skip cleanly (opt-in under `--ignored`; non-macOS hosts fall through too).
    let Ok(model) = std::env::var("MLX_LLM_QWEN3VL_MODEL") else {
        eprintln!("skipping: MLX_LLM_QWEN3VL_MODEL not set");
        return;
    };

    let engine = EngineHandle::spawn();
    engine
        .load_model(LoadModelRequest {
            source: model,
            display_name: None,
            quantize: None,
            kv_cache_quant: None,
        })
        .expect("load Qwen3-VL");

    let server = OpenAiServerHandle::new();
    let server_status = server
        .start(
            OpenAiServerConfig {
                port: 0,
                ..Default::default()
            },
            engine,
        )
        .expect("start OpenAI server");
    let addr = server_status.bound_addr.expect("bound addr");
    let url = format!("http://{addr}/v1/chat/completions");
    let client = reqwest::blocking::Client::new();

    // Only the calculator is offered, so the model must use it to answer.
    let tools = builtin_tool_specs();
    let mut messages = vec![serde_json::json!({
        "role": "user",
        "content": "Use the calculator tool to compute 12.5 * 8, then tell me the result."
    })];

    // Round 1: the model should request a calculator call.
    let first: serde_json::Value = client
        .post(&url)
        .json(&serde_json::json!({
            "model": "qwen3-vl", "stream": false, "max_tokens": 256, "temperature": 0.0,
            "tools": tools, "messages": messages
        }))
        .send()
        .expect("send round 1")
        .json()
        .expect("json round 1");
    println!("\n=== round 1 (expect tool_calls) ===\n{}\n", serde_json::to_string_pretty(&first).unwrap_or_default());

    let choice = &first["choices"][0];
    assert_eq!(choice["finish_reason"], "tool_calls", "round 1 must request a tool call");
    let call = &choice["message"]["tool_calls"][0];
    assert_eq!(call["function"]["name"], "calculator", "must call the calculator");
    let raw_arguments = call["function"]["arguments"].as_str().expect("arguments string").to_string();

    // Execute the call with the REAL built-in executor (what the `execute_tool` command runs).
    let parsed_arguments: serde_json::Value =
        serde_json::from_str(&raw_arguments).expect("arguments decode as JSON");
    let result = execute_builtin_tool("calculator", &parsed_arguments).expect("calculator runs");
    println!("[tool] calculator({raw_arguments}) => {result}");
    assert_eq!(result, "100", "12.5 * 8 must evaluate to 100");

    // Re-enter the result: echo the assistant tool-call turn, then the tool result turn.
    messages.push(serde_json::json!({
        "role": "assistant",
        "content": serde_json::Value::Null,
        "tool_calls": [{
            "id": call["id"].as_str().unwrap_or("call_0"),
            "type": "function",
            "function": {"name": "calculator", "arguments": raw_arguments}
        }]
    }));
    messages.push(serde_json::json!({"role": "tool", "content": result}));

    // Round 2: with the tool result in context, the model should answer with the number.
    let second: serde_json::Value = client
        .post(&url)
        .json(&serde_json::json!({
            "model": "qwen3-vl", "stream": false, "max_tokens": 256, "temperature": 0.0,
            "tools": tools, "messages": messages
        }))
        .send()
        .expect("send round 2")
        .json()
        .expect("json round 2");
    println!("\n=== round 2 (expect final answer) ===\n{}\n", serde_json::to_string_pretty(&second).unwrap_or_default());

    let answer = second["choices"][0]["message"]["content"].as_str().unwrap_or("");
    assert!(
        answer.contains("100"),
        "the model must report the tool result (100) in its final answer, got: {answer:?}"
    );

    server.stop().expect("stop server");
}
