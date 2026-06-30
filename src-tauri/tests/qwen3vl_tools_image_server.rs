//! Real-weights end-to-end test for the sc-8082 EXTENSION: Qwen3-VL tool ("function") calling in the
//! SAME TURN as image input, **through the ChatWorks OpenAI server**. This is the capability this
//! story adds on top of the qwen35 tool-calling mirror — confirming that a single request carrying
//! BOTH an `image_url` content part AND a `tools` offer yields a valid `tool_calls` response that is
//! grounded on the image.
//!
//! The app path under test: an OpenAI chat request with mixed image+text content AND tools → HTTP
//! server → image decode to `Content::Image` (placed before text) + `core_llm::ToolSpec` → engine →
//! `mlx-llama` Qwen3-VL VLM → a parsed `<tool_call>` block surfaced back as an OpenAI `tool_calls`
//! response with `finish_reason` `tool_calls`. The image and tools paths are independent in the
//! engine (`GenerateMessage` carries both `images` and the request `tools`), so this exercises them
//! together for the first time.
//!
//! Point `MLX_LLM_QWEN3VL_MODEL` at the Qwen3-VL-8B-Instruct snapshot revision dir:
//!
//! ```text
//! MLX_LLM_QWEN3VL_MODEL=/path/to/Qwen3-VL-8B-Instruct \
//!   cargo test --release --test qwen3vl_tools_image_server -- --ignored --nocapture
//! ```
//!
//! The MLX backend is macOS-only, so `#[ignore]` plus the env-var gate keep CI green elsewhere.

use base64::Engine as _;
use chatworks::engine::{EngineHandle, LoadModelRequest};
use chatworks::server::{OpenAiServerConfig, OpenAiServerHandle};

/// A solid-color RGB image encoded as a PNG `data:` URL (what an OpenAI vision client sends).
fn solid_png_data_url(rgb: [u8; 3]) -> String {
    let img = image::RgbImage::from_pixel(256, 256, image::Rgb(rgb));
    let mut png = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .expect("encode png");
    format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(&png)
    )
}

#[test]
#[ignore = "needs a Qwen3-VL-8B-Instruct VLM snapshot via MLX_LLM_QWEN3VL_MODEL"]
fn qwen3vl_tool_calling_with_image_over_openai_server() {
    // Unset env var → skip cleanly (opt-in under `--ignored`; non-macOS hosts fall through too).
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
            kv_cache_quant: None,
        })
        .expect("load Qwen3-VL");
    // The same VLM checkpoint must advertise BOTH capabilities — that is precisely what makes
    // image+tools-in-one-turn meaningful.
    assert!(
        status
            .loaded
            .as_ref()
            .map(|m| m.provider.capabilities.supports_tools)
            .unwrap_or(false),
        "loaded Qwen3-VL must report supports_tools"
    );
    assert!(
        status
            .loaded
            .as_ref()
            .map(|m| m.provider.capabilities.supports_vision)
            .unwrap_or(false),
        "loaded Qwen3-VL must report supports_vision"
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

    // A `log_color` tool whose single argument the model can only fill by LOOKING at the image. The
    // prompt forces a tool call, so a valid response both finishes with `tool_calls` AND grounds the
    // `color` argument on the image — proving image and tools coexist in one turn.
    let tools = serde_json::json!([{
        "type": "function",
        "function": {
            "name": "log_color",
            "description": "Record the dominant color shown in an image.",
            "parameters": {
                "type": "object",
                "properties": {
                    "color": {
                        "type": "string",
                        "description": "The single dominant color of the image, e.g. 'red' or 'blue'."
                    }
                },
                "required": ["color"]
            }
        }
    }]);

    for (rgb, want) in [([205u8, 35, 35], "red"), ([35u8, 70, 200], "blue")] {
        let body = serde_json::json!({
            "model": "qwen3-vl",
            "stream": false,
            "max_tokens": 256,
            "temperature": 0.0,
            "tools": tools,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": solid_png_data_url(rgb)}},
                    {"type": "text", "text": "Identify the dominant color of this image and record it by calling the log_color tool."}
                ]
            }]
        });
        let resp: serde_json::Value = client
            .post(&url)
            .json(&body)
            .send()
            .expect("send")
            .json()
            .expect("json");
        println!(
            "\n=== ChatWorks OpenAI image+tools in one turn ({want}) ===\n{}\n",
            serde_json::to_string_pretty(&resp).unwrap_or_default()
        );

        let choice = &resp["choices"][0];
        assert_eq!(
            choice["finish_reason"], "tool_calls",
            "an image+tools turn must finish with finish_reason=tool_calls ({want})"
        );

        let call = &choice["message"]["tool_calls"][0];
        assert_eq!(call["type"], "function", "tool call must be a function call");
        assert_eq!(
            call["function"]["name"], "log_color",
            "the model must call log_color ({want})"
        );

        // The call's `color` argument can only be filled correctly by reading the image, so a
        // grounded call proves image AND tools were both honored in the same turn.
        let arguments = call["function"]["arguments"]
            .as_str()
            .expect("function.arguments is a JSON string");
        let parsed: serde_json::Value =
            serde_json::from_str(arguments).expect("arguments decode as JSON");
        let color = parsed["color"].as_str().unwrap_or_default();
        assert!(
            color.to_lowercase().contains(want),
            "the tool call must ground its color on the {want} image, got arguments: {arguments:?}"
        );
    }

    server.stop().expect("stop server");
}
