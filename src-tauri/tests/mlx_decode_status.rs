//! sc-24445 AC2: on the MLX build a finished chat generation reaches the decode-path status with
//! the proposer that ran, its realized accepted length and the sampler path — measured by the
//! linked runtime, not a fake. A tiny random-weight Llama snapshot (written here, no downloads)
//! is served through the same `EngineHandle` the desktop app and the OpenAI server use, under the
//! runtime's `auto` speculative option, which resolves to prompt lookup on a model without an MTP
//! head. `tests/decode-path.test.mjs` renders the same payload shape in `DecodePathStatus`.

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use std::fs;
use std::path::{Path, PathBuf};

use chatworks::engine::{EngineHandle, GenerateRequest, LoadModelRequest};
use serde_json::{json, Value};

const HIDDEN: usize = 8;
const VOCAB: usize = 4;
const INTERMEDIATE: usize = 16;
const Q_DIM: usize = 8;
const KV_DIM: usize = 4;

/// A named F32 tensor's shape and values.
type Tensor = (Vec<usize>, Vec<f32>);

/// A deterministic weight stream (SplitMix64), so the snapshot is the same on every run.
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

/// Write `tensors` as a little-endian F32 safetensors file.
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

/// A tiny but complete Llama snapshot directory, removed on drop.
struct Snapshot(PathBuf);

impl Drop for Snapshot {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn tiny_llama_snapshot() -> Snapshot {
    let dir = std::env::temp_dir().join(format!("chatworks-mlx-decode-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    // eos_token_id = 99 is outside the vocabulary, so a generation runs to `max_new_tokens`.
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
                "vocab": {"<unk>": 0, "hello": 1, "world": 2, "<eos>": 3},
                "unk_token": "<unk>"
            }
        })
        .to_string(),
    )
    .unwrap();
    let mut weights = Weights(0xC0FFEE);
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

#[test]
fn an_mlx_chat_response_reports_proposer_accepted_length_and_sampler() {
    let snapshot = tiny_llama_snapshot();
    let engine = EngineHandle::spawn();
    let status = engine
        .load_model(LoadModelRequest {
            source: snapshot.0.to_string_lossy().to_string(),
            display_name: Some("tiny-llama".to_string()),
            quantize: None,
            projector_source: None,
            cuda_graphs: None,
        })
        .expect("the MLX runtime loads the tiny snapshot");
    assert_eq!(status.execution_backend, "mlx");
    let loaded = status.loaded.expect("a served model");
    assert!(
        !loaded.provider.capabilities.speculative.is_empty(),
        "the MLX provider advertises a proposer: {:?}",
        loaded.provider.capabilities.speculative
    );

    let request: GenerateRequest = serde_json::from_value(json!({
        "messages": [{"role": "user", "content": "hello world hello world hello world hello"}],
        "sampling": {"temperature": 0.0},
        "max_new_tokens": 12,
        "seed": 7,
        "speculative": "auto",
    }))
    .unwrap();
    let response = engine.generate(request, |_| {}).expect("generation");
    let decode = response
        .decode
        .expect("the MLX runtime reports its decode path");
    // `auto` on a model without an MTP head runs prompt lookup (epic sc-24432 E4).
    assert_eq!(decode.proposer, "prompt_lookup", "{decode:?}");
    assert!(decode.verify_steps > 0, "{decode:?}");
    let accepted = decode
        .mean_accepted_length
        .expect("a proposer ran, so the runtime reports a mean accepted length");
    assert!(
        (accepted - decode.accepted_tokens as f64 / decode.verify_steps as f64).abs() < 1e-9,
        "{decode:?}"
    );
    assert!(
        decode.sampler == "device" || decode.sampler.starts_with("host:"),
        "{decode:?}"
    );

    // The status the decode-path view renders carries the same report.
    let loaded = engine.status().unwrap().loaded.unwrap();
    assert_eq!(loaded.decode_reported, Some(true));
    assert_eq!(loaded.last_decode.as_ref(), Some(&decode));
    let wire = serde_json::to_value(&loaded).unwrap();
    for key in [
        "proposer",
        "mean_accepted_length",
        "verify_steps",
        "sampler",
        "fallbacks",
    ] {
        assert!(
            wire["last_decode"].get(key).is_some(),
            "{key} in {}",
            wire["last_decode"]
        );
    }
    engine.unload_model().unwrap();
}
