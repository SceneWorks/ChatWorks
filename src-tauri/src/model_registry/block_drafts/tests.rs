use super::*;
use crate::fsutil::TempDir;

const DSPINE_CONFIG: &str = include_str!("dspine-config.json");
const DFLASH2_CONFIG: &str = include_str!("dflash2-config.json");

fn write(path: &Path, body: &str) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, body).unwrap();
}

/// A registered model whose snapshot holds `config`.
fn entry_with_config(dir: &Path, config: serde_json::Value) -> ModelEntry {
    let snapshot = dir.join("target");
    write(&snapshot.join("config.json"), &config.to_string());
    serde_json::from_value(serde_json::json!({
        "id": "target", "name": "Target", "repo": "Qwen/Target", "revision": "main",
        "sourceUrl": "https://huggingface.co/Qwen/Target",
        "localPath": snapshot.to_string_lossy(), "importedAt": 1, "fileCount": 2,
    }))
    .unwrap()
}

fn qwen38_vlm() -> serde_json::Value {
    serde_json::json!({
        "model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"],
        "vision_config": {},
        "text_config": {"model_type": "qwen3_5_text", "hidden_size": 5120,
            "num_hidden_layers": 64, "vocab_size": 248320},
    })
}

fn option<'a>(options: &'a [BlockDraftOption], format: &str) -> &'a BlockDraftOption {
    options
        .iter()
        .find(|option| option.format == format)
        .unwrap()
}

/// AC1: a Qwen3.8-27B target on MLX can use every draft the runtime reads; a draft the runtime
/// does not read, another backend, a packed checkpoint and another geometry are each explained.
#[test]
fn a_qwen38_target_can_use_the_runtimes_drafts_and_every_other_case_is_explained() {
    let dir = TempDir::new("block-draft-options");
    let snapshots = dir.path().join("snapshots");
    let target = entry_with_config(dir.path(), qwen38_vlm());

    let options = options_for(&target, &snapshots, "mlx", &["dspine"]);
    assert_eq!(options.len(), BLOCK_DRAFT_CATALOG.len());
    let dspine = option(&options, "dspine");
    assert_eq!(dspine.unavailable, None);
    assert_eq!(dspine.installed_path, None);
    assert_eq!(dspine.repo, "SceneWorks/qwen3.8-27b-dspine-b16");
    assert_eq!(dspine.revision, "8e450d7faf38238b64580163a7d3dd7e44af759d");
    assert!(option(&options, "dflash2")
        .unavailable
        .as_deref()
        .unwrap()
        .contains("does not read DFlash2"));
    let both = options_for(&target, &snapshots, "mlx", &["dspine", "dflash2"]);
    assert!(both.iter().all(|option| option.unavailable.is_none()));

    // Installed once its config and weights are on disk.
    let installed = snapshot_path(&snapshots, catalog_entry("dspine").unwrap());
    write(&installed.join("config.json"), DSPINE_CONFIG);
    assert_eq!(
        option(
            &options_for(&target, &snapshots, "mlx", &["dspine"]),
            "dspine"
        )
        .installed_path,
        None
    );
    write(&installed.join("model.safetensors"), "");
    assert_eq!(
        option(
            &options_for(&target, &snapshots, "mlx", &["dspine"]),
            "dspine"
        )
        .installed_path
        .as_deref(),
        Some(installed.to_string_lossy().as_ref())
    );

    let candle = options_for(&target, &snapshots, "candle-cuda", &["dspine"]);
    assert!(option(&candle, "dspine")
        .unavailable
        .as_deref()
        .unwrap()
        .contains("MLX backend"));

    let mut packed = target.clone();
    packed.pack = Some("bonsai2-packed".to_string());
    assert!(option(
        &options_for(&packed, &snapshots, "mlx", &["dspine"]),
        "dspine"
    )
    .unavailable
    .as_deref()
    .unwrap()
    .contains("packed or GGUF"));

    // A text-only qwen3_5_text fine-tune with the 27B geometry is a target too.
    let text_only = entry_with_config(
        &dir.path().join("text"),
        serde_json::json!({"model_type": "qwen3_5_text", "hidden_size": 5120,
            "num_hidden_layers": 64, "vocab_size": 248320}),
    );
    assert_eq!(
        option(
            &options_for(&text_only, &snapshots, "mlx", &["dspine"]),
            "dspine"
        )
        .unavailable,
        None
    );

    let small = entry_with_config(
        &dir.path().join("small"),
        serde_json::json!({"model_type": "qwen3_5_text", "hidden_size": 2048,
            "num_hidden_layers": 24, "vocab_size": 248320}),
    );
    let why = option(
        &options_for(&small, &snapshots, "mlx", &["dspine"]),
        "dspine",
    )
    .unavailable
    .clone()
    .unwrap();
    assert!(why.contains("hidden size 2048, 24 layers"), "{why}");

    let llama = entry_with_config(
        &dir.path().join("llama"),
        serde_json::json!({"model_type": "llama", "hidden_size": 4096}),
    );
    assert!(option(
        &options_for(&llama, &snapshots, "mlx", &["dspine"]),
        "dspine"
    )
    .unavailable
    .as_deref()
    .unwrap()
    .contains("not one"));
}

/// The runtime this build links reads its own formats; the catalog names each of them.
#[test]
fn the_catalog_covers_every_format_the_runtime_reads() {
    for format in runtime_formats() {
        assert!(
            catalog_entry(format).is_ok(),
            "{format} has no catalog draft"
        );
    }
    assert!(catalog_entry("medusa").is_err());
}

/// An installed draft is checked by the runtime's own reader: the published DSpine config is a
/// block draft it reads; a config outside the runtime's formats, or a malformed one, is refused
/// with the runtime's reason.
#[test]
fn an_installed_draft_is_checked_by_the_runtimes_reader() {
    let dir = TempDir::new("block-draft-check");
    let dspine = dir.path().join("dspine");
    write(&dspine.join("config.json"), DSPINE_CONFIG);
    assert_eq!(
        check_installed_draft(&dspine).unwrap(),
        dspine.to_string_lossy()
    );

    let dflash2 = dir.path().join("dflash2");
    write(&dflash2.join("config.json"), DFLASH2_CONFIG);
    let reads_dflash2 = runtime_formats().contains(&"dflash2");
    assert_eq!(check_installed_draft(&dflash2).is_ok(), reads_dflash2);

    let broken = dir.path().join("broken");
    write(
        &broken.join("config.json"),
        r#"{"architectures": ["DSpineDraftModel"]}"#,
    );
    assert!(check_installed_draft(&broken)
        .unwrap_err()
        .starts_with("block draft: format:"));

    let plain = dir.path().join("plain");
    write(
        &plain.join("config.json"),
        r#"{"architectures": ["Qwen3ForCausalLM"]}"#,
    );
    assert!(check_installed_draft(&plain)
        .unwrap_err()
        .contains("not a block draft"));
}

/// Installing saves the draft as the model's `draft_source`, keeping its saved precision, so
/// the next load names it through `LoadSpec::draft_source` / `draft_quantize`.
#[test]
fn an_installed_draft_becomes_the_models_draft_source() {
    let dir = TempDir::new("block-draft-set");
    let mut target = entry_with_config(dir.path(), qwen38_vlm());
    target.draft_quantize = Some(crate::engine::QuantizeRequest::Q4);
    let manifest = dir.path().join("manifest.json");
    write_registry(
        &manifest,
        &ModelRegistry {
            models: vec![target],
            selected_id: None,
        },
    )
    .unwrap();
    let registry = set_block_draft(&manifest, "target", "/snapshots/dspine").unwrap();
    let saved = &registry.models[0];
    assert_eq!(saved.draft_source.as_deref(), Some("/snapshots/dspine"));
    assert_eq!(
        saved.draft_quantize,
        Some(crate::engine::QuantizeRequest::Q4)
    );
    assert_eq!(
        read_registry(&manifest).unwrap().models[0].draft_source,
        saved.draft_source
    );
    assert!(set_block_draft(&manifest, "missing", "/x").is_err());
}
