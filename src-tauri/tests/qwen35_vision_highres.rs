//! Regression: high-resolution images must not crash the Qwen3.6 27B VLM prefill.
//!
//! A high-res image expands to thousands of vision tokens; the gated-delta (linear-attention) prefill
//! used to build the whole per-timestep graph before a single eval, exhausting the Metal allocator's
//! buffer limit (`[metal::malloc] Resource limit (499000) exceeded`) and surfacing as a misleading
//! "expected a non-empty mlx_array". Fixed in mlx-llm by flushing the recurrence every N steps. A
//! 1536x1536 image (~2304 vision tokens) reproduced the crash; 2048x2048 (~4096) is well past the
//! limit. Both must now answer.
//!
//! ```text
//! MLX_LLM_QWEN35_MODEL=/path/to/Qwen3.6-27B \
//!   cargo test --release --test qwen35_vision_sweep -- --ignored --nocapture
//! ```

use base64::Engine as _;
use chatworks::engine::{
    EngineHandle, GenerateMessage, GenerateRequest, LoadModelRequest, SamplingRequest,
    ThinkingRequest,
};

fn gradient_png_data_url(w: u32, h: u32) -> String {
    let mut img = image::RgbImage::new(w, h);
    for (x, y, px) in img.enumerate_pixels_mut() {
        *px = image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8]);
    }
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
fn qwen35_vision_high_resolution_prefill_does_not_crash() {
    let model = std::env::var("MLX_LLM_QWEN35_MODEL").expect("set MLX_LLM_QWEN35_MODEL");
    let engine = EngineHandle::spawn();
    engine
        .load_model(LoadModelRequest {
            source: model,
            display_name: None,
            quantize: None,
        })
        .expect("load 27B");

    // Each previously crashed the gated-delta prefill once the image's vision-token count pushed the
    // decoder's live Metal buffer count past the allocator limit.
    for &(w, h) in &[(1536u32, 1536u32), (2048, 2048)] {
        let req = GenerateRequest {
            messages: vec![GenerateMessage {
                role: "user".into(),
                content: "What is in this image? One word.".into(),
                images: vec![gradient_png_data_url(w, h)],
                videos: vec![],
                tool_calls: vec![],
            }],
            sampling: SamplingRequest::default(),
            max_new_tokens: 1,
            seed: None,
            stop: vec![],
            thinking: ThinkingRequest::default(),
            tools: vec![],
        };
        engine
            .generate(req, |_ev| {})
            .unwrap_or_else(|e| panic!("{w}x{h} image prefill must not crash, got: {e}"));
    }
}
