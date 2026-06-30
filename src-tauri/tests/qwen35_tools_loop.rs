//! Real-weights end-to-end test of the full tool-calling LOOP the chat UI drives (sc-7772a):
//! model emits a tool call → the app executes a built-in tool → the result re-enters the
//! conversation → the model produces a grounded final answer. Mirrors `src/main.jsx`'s loop but in
//! Rust, reusing the real `chatworks::tools` executor. Point `MLX_LLM_QWEN35_MODEL` at the 27B:
//!
//! ```text
//! MLX_LLM_QWEN35_MODEL=/path/to/Qwen3.6-27B \
//!   cargo test --test qwen35_tools_loop -- --ignored --nocapture
//! ```

use chatworks::engine::{EngineHandle, LoadModelRequest};
use chatworks::server::{OpenAiServerConfig, OpenAiServerHandle};
use chatworks::tools::{builtin_tool_specs, execute_builtin_tool};

#[test]
#[ignore = "needs a Qwen3.6 27B snapshot via MLX_LLM_QWEN35_MODEL"]
fn qwen35_tool_loop_over_openai_server() {
    let model = std::env::var("MLX_LLM_QWEN35_MODEL").expect("set MLX_LLM_QWEN35_MODEL");

    let engine = EngineHandle::spawn();
    engine
        .load_model(LoadModelRequest {
            source: model,
            display_name: None,
            quantize: None,
        })
        .expect("load 27B");

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
            "model": "qwen3.6", "stream": false, "max_tokens": 256, "temperature": 0.0,
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
            "model": "qwen3.6", "stream": false, "max_tokens": 256, "temperature": 0.0,
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
