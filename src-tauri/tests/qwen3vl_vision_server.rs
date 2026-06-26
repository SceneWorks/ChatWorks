//! Real-weights end-to-end test for Qwen3-VL vision **through the ChatWorks OpenAI server** (sc-8079).
//! Proves the full app path: an OpenAI `image_url` chat request → HTTP server → image decode →
//! `Content::Image` → engine → `mlx-llama` Qwen3-VL VLM → grounded answer. Point
//! `MLX_LLM_QWEN3VL_MODEL` at the Qwen3-VL-8B-Instruct snapshot revision dir (the
//! `…/snapshots/<rev>/` directory holding `config.json` + weights):
//!
//! ```text
//! MLX_LLM_QWEN3VL_MODEL=/path/to/Qwen3-VL-8B-Instruct \
//!   cargo test --test qwen3vl_vision_server -- --ignored --nocapture
//! ```
//!
//! The MLX backend is macOS-only, so `#[ignore]` plus the env-var gate keep CI green elsewhere:
//! the test is opt-in and skips cleanly whenever `MLX_LLM_QWEN3VL_MODEL` is unset.

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
fn qwen3vl_vision_over_openai_server() {
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
    // The provider must advertise vision once loaded (the VLM checkpoint carries the `vision_config`).
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

    for (rgb, want) in [([205u8, 35, 35], "red"), ([35u8, 70, 200], "blue")] {
        let body = serde_json::json!({
            "model": "qwen3-vl",
            "stream": false,
            "max_tokens": 24,
            "temperature": 0.0,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": solid_png_data_url(rgb)}},
                    {"type": "text", "text": "What is the dominant color of this image? Answer with one word."}
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
        let content = resp["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("");
        println!("\n=== ChatWorks OpenAI vision ({want}) ===\n[answer] {content:?}\n");
        assert!(
            content.to_lowercase().contains(want),
            "answer must ground on the {want} image, got: {content:?}"
        );
    }

    server.stop().expect("stop server");
}
