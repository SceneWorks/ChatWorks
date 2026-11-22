//! SW-7236 AC1/AC2/AC3: Verify exact prefix checkpoints end-to-end on the pinned MLX runtime
//! through both ChatWorks' desktop (`EngineHandle`) rewind flow and the OpenAI HTTP server
//! (`OpenAiServerHandle`) conversation-branch + repeated-document flow.
//!
//! Uses a tiny synthetic Llama snapshot with an explicit multi-turn Jinja template in
//! `tokenizer_config.json` so `Messages { count }` and message boundaries align to exact
//! prompt token prefixes.

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use std::fs;
use std::path::{Path, PathBuf};

use chatworks::app_settings::SamplingDefaults;
use chatworks::core_llm::PrefixCheckpointMode;
use chatworks::engine::{EngineHandle, GenerateRequest, LoadModelRequest};
use chatworks::server::{OpenAiServerConfig, OpenAiServerHandle};
use serde_json::{json, Value};

const HIDDEN: usize = 8;
const VOCAB: usize = 24;
const INTERMEDIATE: usize = 16;
const Q_DIM: usize = 8;
const KV_DIM: usize = 4;

type Tensor = (Vec<usize>, Vec<f32>);

struct Weights(u64);

impl Weights {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.4
    }

    fn tensor(&mut self, shape: &[usize]) -> Tensor {
        let len = shape.iter().product();
        (shape.to_vec(), (0..len).map(|_| self.next()).collect())
    }
}

fn ones(shape: &[usize]) -> Tensor {
    (shape.to_vec(), vec![1.0; shape.iter().product()])
}

fn write_safetensors(path: &Path, tensors: &[(String, Tensor)]) {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, (shape, values)) in tensors {
        let start = data.len();
        for value in values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        header.insert(
            name.clone(),
            json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}),
        );
    }
    let mut header = serde_json::to_vec(&Value::Object(header)).unwrap();
    while header.len() % 8 != 0 {
        header.push(b' ');
    }
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(&header);
    file.extend_from_slice(&data);
    fs::write(path, file).unwrap();
}

struct Snapshot(PathBuf);

impl Drop for Snapshot {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn tiny_llama_snapshot(tag: &str) -> Snapshot {
    let dir = std::env::temp_dir().join(format!(
        "chatworks-mlx-prefix-checkpoints-{tag}-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    fs::write(
        dir.join("config.json"),
        json!({
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": HIDDEN,
            "intermediate_size": INTERMEDIATE,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "vocab_size": VOCAB,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "tie_word_embeddings": false,
            "eos_token_id": 99
        })
        .to_string(),
    )
    .unwrap();
    let vocab_words = [
        "<unk>",
        "<eos>",
        "system:",
        "user:",
        "assistant:",
        "doc",
        "clause",
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "zeta",
        "reply",
        "one",
        "two",
        "three",
        "four",
        "five",
        "q1",
        "q2",
        "q3",
        "branch",
        "other",
    ];
    assert_eq!(vocab_words.len(), VOCAB);
    let vocab: serde_json::Map<String, Value> = vocab_words
        .iter()
        .enumerate()
        .map(|(i, word)| ((*word).to_string(), json!(i)))
        .collect();
    fs::write(
        dir.join("tokenizer.json"),
        json!({
            "version": "1.0",
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": vocab,
                "unk_token": "<unk>"
            }
        })
        .to_string(),
    )
    .unwrap();
    // Explicit per-message whitespace-delimited Jinja template so render_prefix_boundary produces
    // an exact prefix of full prompt tokens at every message boundary.
    fs::write(
        dir.join("tokenizer_config.json"),
        json!({
            "chat_template": "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }} {% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
        })
        .to_string(),
    )
    .unwrap();
    let mut weights = Weights(0xCAFE_BABE);
    let mut tensors = vec![
        (
            "model.embed_tokens.weight".to_string(),
            weights.tensor(&[VOCAB, HIDDEN]),
        ),
        ("model.norm.weight".to_string(), ones(&[HIDDEN])),
        (
            "lm_head.weight".to_string(),
            weights.tensor(&[VOCAB, HIDDEN]),
        ),
    ];
    for layer in 0..2 {
        let name = |suffix: &str| format!("model.layers.{layer}.{suffix}");
        tensors.push((name("input_layernorm.weight"), ones(&[HIDDEN])));
        tensors.push((name("post_attention_layernorm.weight"), ones(&[HIDDEN])));
        tensors.push((
            name("self_attn.q_proj.weight"),
            weights.tensor(&[Q_DIM, HIDDEN]),
        ));
        tensors.push((
            name("self_attn.k_proj.weight"),
            weights.tensor(&[KV_DIM, HIDDEN]),
        ));
        tensors.push((
            name("self_attn.v_proj.weight"),
            weights.tensor(&[KV_DIM, HIDDEN]),
        ));
        tensors.push((
            name("self_attn.o_proj.weight"),
            weights.tensor(&[HIDDEN, Q_DIM]),
        ));
        tensors.push((
            name("mlp.gate_proj.weight"),
            weights.tensor(&[INTERMEDIATE, HIDDEN]),
        ));
        tensors.push((
            name("mlp.up_proj.weight"),
            weights.tensor(&[INTERMEDIATE, HIDDEN]),
        ));
        tensors.push((
            name("mlp.down_proj.weight"),
            weights.tensor(&[HIDDEN, INTERMEDIATE]),
        ));
    }
    write_safetensors(&dir.join("model.safetensors"), &tensors);
    Snapshot(dir)
}

fn load_tiny_model(engine: &EngineHandle, dir: &Path) -> chatworks::engine::EngineStatus {
    engine
        .load_model(LoadModelRequest {
            source: dir.to_string_lossy().to_string(),
            display_name: Some("tiny-llama-checkpoints".to_string()),
            quantize: None,
            projector_source: None,
            cuda_graphs: None,
            draft_source: None,
            prefix_cache_bytes: None,
            mtp_head_source: None,
            draft_quantize: None,
        })
        .expect("MLX runtime loads tiny snapshot")
}

#[test]
fn desktop_rewind_and_pinned_document_reuse_checkpoints_on_pinned_mlx_runtime() {
    let snapshot = tiny_llama_snapshot("desktop");
    let engine = EngineHandle::spawn();
    let status = load_tiny_model(&engine, &snapshot.0);
    assert_eq!(status.execution_backend, "mlx");
    assert_eq!(status.prefix_checkpoints_default, PrefixCheckpointMode::Off);
    let loaded = status.loaded.expect("served model");
    let pc_caps = loaded
        .provider
        .capabilities
        .prefix_checkpoints
        .expect("MLX provider advertises prefix_checkpoints capabilities");
    assert!(pc_caps.supports_document_span);
    assert!(pc_caps.max_checkpoints_per_request >= 1);

    // Turn 1: Initial conversation with `auto` captures protected SharedRoot + message boundaries.
    let turn1: GenerateRequest = serde_json::from_value(json!({
        "messages": [
            {"role": "user", "content": "doc clause alpha beta gamma delta"},
            {"role": "assistant", "content": "reply one two"},
            {"role": "user", "content": "q1 three"}
        ],
        "sampling": {"temperature": 0.0},
        "max_new_tokens": 4,
        "seed": 11,
        "speculative": "off",
        "prefix_checkpoints": "auto"
    }))
    .unwrap();
    let r1 = engine.generate(turn1, |_| {}).expect("turn1");
    let pc1 = r1
        .decode
        .as_ref()
        .and_then(|d| d.prefix_checkpoints.as_ref())
        .expect("turn1 prefix checkpoint telemetry");
    assert_eq!(pc1.policy, "auto");
    assert_eq!(pc1.reused_tokens, 0);
    assert!(pc1.reread_tokens > 0);
    assert!(pc1.resident_bytes > 0);
    assert!(
        pc1.protected_bytes > 0,
        "auto mode protects root checkpoint: {pc1:?}"
    );
    assert!(pc1.fallbacks.is_empty(), "{pc1:?}");

    // Desktop Rewind at message index 2 (edited final user question): reuses safe prefix cleanly.
    let rewind_edit: GenerateRequest = serde_json::from_value(json!({
        "messages": [
            {"role": "user", "content": "doc clause alpha beta gamma delta"},
            {"role": "assistant", "content": "reply one two"},
            {"role": "user", "content": "q2 four"}
        ],
        "sampling": {"temperature": 0.0},
        "max_new_tokens": 4,
        "seed": 11,
        "speculative": "off",
        "prefix_checkpoints": "auto"
    }))
    .unwrap();
    let r2 = engine.generate(rewind_edit, |_| {}).expect("rewind_edit");
    let d2 = r2.decode.expect("rewind decode report");
    let pc2 = d2
        .prefix_checkpoints
        .expect("rewind prefix checkpoint telemetry");
    assert_eq!(pc2.policy, "auto");
    assert!(
        pc2.reused_tokens > 0,
        "desktop Rewind must reuse cached prefix: {pc2:?}"
    );
    assert!(pc2.reread_tokens > 0);
    assert!(pc2.matched_kind.is_some(), "{pc2:?}");
    assert_eq!(
        engine
            .status()
            .unwrap()
            .loaded
            .unwrap()
            .last_decode
            .unwrap()
            .prefix_checkpoints,
        Some(pc2.clone())
    );

    // Reloading unloads prior KV state so `off` starts completely clean with 0 protected bytes.
    load_tiny_model(&engine, &snapshot.0);
    let off_req: GenerateRequest = serde_json::from_value(json!({
        "messages": [
            {"role": "user", "content": "doc clause alpha beta gamma delta"},
            {"role": "assistant", "content": "reply one two"},
            {"role": "user", "content": "q1 three"}
        ],
        "sampling": {"temperature": 0.0},
        "max_new_tokens": 4,
        "seed": 11,
        "speculative": "off",
        "prefix_checkpoints": "off"
    }))
    .unwrap();
    let r_off = engine.generate(off_req, |_| {}).expect("off_req");
    let pc_off = r_off
        .decode
        .and_then(|d| d.prefix_checkpoints)
        .expect("off telemetry");
    assert_eq!(pc_off.policy, "off");
    assert_eq!(
        pc_off.protected_bytes, 0,
        "disabling prefix checkpoints preserves ordinary single-boundary behavior without protected checkpoints: {pc_off:?}"
    );
    engine.unload_model().unwrap();
}

#[test]
fn openai_api_branch_and_repeated_document_reuse_checkpoints_on_pinned_mlx_runtime() {
    let snapshot = tiny_llama_snapshot("api");
    let engine = EngineHandle::spawn();
    load_tiny_model(&engine, &snapshot.0);

    let mut sampling_defaults = SamplingDefaults::default();
    sampling_defaults.system_prompt = "system: alpha beta".to_string();
    let server = OpenAiServerHandle::new();
    let server_status = server
        .start(
            OpenAiServerConfig {
                port: 0,
                sampling_defaults,
                ..Default::default()
            },
            engine,
        )
        .expect("start OpenAI server");
    let addr = server_status.bound_addr.expect("bound addr");
    let url = format!("http://{addr}/v1/chat/completions");
    let client = reqwest::blocking::Client::new();

    // API turn 1 with caller-supplied document span (`messages.count = 1`) and no explicit system
    // message: server prepends SamplingDefaults::system_prompt and shifts count 1 -> 2 accurately.
    let doc_q1: Value = client
        .post(&url)
        .json(&json!({
            "model": "tiny-llama-checkpoints",
            "stream": false,
            "max_tokens": 4,
            "temperature": 0.0,
            "speculative": "off",
            "prefix_checkpoints": {
                "mode": "auto",
                "document": { "messages": { "count": 1 } }
            },
            "messages": [
                {"role": "user", "content": "doc clause alpha beta gamma delta epsilon zeta"},
                {"role": "assistant", "content": "reply one two"},
                {"role": "user", "content": "q1 three"}
            ]
        }))
        .send()
        .expect("doc_q1")
        .json()
        .expect("doc_q1 json");
    let pc1 = &doc_q1["chatworks_decode"]["prefix_checkpoints"];
    assert_eq!(pc1["policy"], "document", "{doc_q1}");
    assert_eq!(pc1["reused_tokens"], 0);
    assert!(pc1["protected_bytes"].as_u64().unwrap() > 0, "{pc1}");
    assert_eq!(pc1["fallbacks"], json!([]), "{pc1}");

    // Repeated question reusing the pinned document on a completely different branch (no Turn 2):
    // must reuse the protected document prefix cleanly across the API.
    let doc_q2: Value = client
        .post(&url)
        .json(&json!({
            "model": "tiny-llama-checkpoints",
            "stream": false,
            "max_tokens": 4,
            "temperature": 0.0,
            "speculative": "off",
            "prefix_checkpoints": {
                "mode": "auto",
                "document": { "messages": { "count": 1 } }
            },
            "messages": [
                {"role": "user", "content": "doc clause alpha beta gamma delta epsilon zeta"},
                {"role": "user", "content": "q2 branch four"}
            ]
        }))
        .send()
        .expect("doc_q2")
        .json()
        .expect("doc_q2 json");
    let pc2 = &doc_q2["chatworks_decode"]["prefix_checkpoints"];
    assert_eq!(pc2["policy"], "document", "{doc_q2}");
    assert!(
        pc2["reused_tokens"].as_u64().unwrap() > 0,
        "repeated document question should reuse cached prefix: {pc2}"
    );
    assert!(pc2["matched_kind"].is_string(), "{pc2}");
    assert!(pc2["reread_tokens"].as_u64().unwrap() > 0, "{pc2}");
    assert!(pc2["restore_ms"].as_f64().is_some(), "{pc2}");
    assert!(pc2["capture_ms"].as_f64().is_some(), "{pc2}");
    assert!(pc2["resident_bytes"].as_u64().unwrap() > 0, "{pc2}");
    let protected_after_doc = pc2["protected_bytes"].as_u64().unwrap();

    // Disabling prefix_checkpoints over the API preserves ordinary behavior and adds no protected checkpoints.
    let off_api: Value = client
        .post(&url)
        .json(&json!({
            "model": "tiny-llama-checkpoints",
            "stream": false,
            "max_tokens": 4,
            "temperature": 0.0,
            "speculative": "off",
            "prefix_checkpoints": "off",
            "messages": [
                {"role": "user", "content": "other gamma delta"},
                {"role": "assistant", "content": "reply two"},
                {"role": "user", "content": "q1"}
            ]
        }))
        .send()
        .expect("off_api")
        .json()
        .expect("off_api json");
    let pc_off = &off_api["chatworks_decode"]["prefix_checkpoints"];
    assert_eq!(pc_off["policy"], "off", "{off_api}");
    assert_eq!(pc_off["protected_bytes"], protected_after_doc, "{pc_off}");

    // Streaming branch follow-up emits `chatworks_decode.prefix_checkpoints` on the final chunk.
    let stream_text = client
        .post(&url)
        .json(&json!({
            "model": "tiny-llama-checkpoints",
            "stream": true,
            "max_tokens": 4,
            "temperature": 0.0,
            "speculative": "off",
            "prefix_checkpoints": "auto",
            "messages": [
                {"role": "user", "content": "doc clause alpha beta gamma delta epsilon zeta"},
                {"role": "user", "content": "q3 five"}
            ]
        }))
        .send()
        .expect("stream branch")
        .text()
        .expect("stream body");
    assert!(
        stream_text.contains("\"prefix_checkpoints\""),
        "{stream_text}"
    );
    assert!(stream_text.contains("\"reused_tokens\""), "{stream_text}");
    assert!(stream_text.contains("data: [DONE]"));

    server.stop().expect("stop server");
}
