//! Real-weights end-to-end test for Qwen3.6 vision **through the ChatWorks OpenAI server** (sc-7635).
//! Proves the full app path: an OpenAI `image_url` chat request → HTTP server → image decode →
//! `Content::Image` → engine → `mlx-llama` Qwen3.6 VLM → grounded answer. Point
//! `MLX_LLM_QWEN35_MODEL` at the 27B snapshot:
//!
//! ```text
//! MLX_LLM_QWEN35_MODEL=/path/to/Qwen3.6-27B \
//!   cargo test --test qwen35_vision_server -- --ignored --nocapture
//! ```

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
#[ignore = "needs a Qwen3.6 27B VLM snapshot via MLX_LLM_QWEN35_MODEL"]
fn qwen35_vision_over_openai_server() {
    let model = std::env::var("MLX_LLM_QWEN35_MODEL").expect("set MLX_LLM_QWEN35_MODEL");

    let engine = EngineHandle::spawn();
    let status = engine
        .load_model(LoadModelRequest {
            source: model,
            display_name: None,
            quantize: None,
        })
        .expect("load 27B");
    // The provider must advertise vision once loaded (the VLM checkpoint carries model.visual.*).
    assert!(
        status
            .loaded
            .as_ref()
            .map(|m| m.provider.capabilities.supports_vision)
            .unwrap_or(false),
        "loaded Qwen3.6 must report supports_vision"
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
            "model": "qwen3.6",
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
