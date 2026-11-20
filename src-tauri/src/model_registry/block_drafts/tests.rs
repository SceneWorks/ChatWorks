use super::*;
use crate::fsutil::TempDir;

const DSPINE_CONFIG: &str = BLOCK_DRAFT_CATALOG[0].config;
const DFLASH2_CONFIG: &str = BLOCK_DRAFT_CATALOG[1].config;

/// The Qwen3.8-27B geometry the catalog's drafts were trained against.
const QWEN38_27B: BlockDraftTarget = BlockDraftTarget {
    hidden_size: 5120,
    num_layers: 64,
    vocab_size: 248_320,
    tokenizer_vocab: 248_320,
};

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

    // Installed once its pinned config and weights are on disk and were verified on download.
    let dspine_entry = catalog_entry("dspine").unwrap();
    let installed = snapshot_path(&snapshots, dspine_entry);
    let installed_path = |snapshots: &Path| {
        option(
            &options_for(&target, snapshots, "mlx", &["dspine"]),
            "dspine",
        )
        .installed_path
        .clone()
    };
    write(&installed.join("config.json"), DSPINE_CONFIG);
    write(&installed.join("model.safetensors"), "");
    assert_eq!(installed_path(&snapshots), None, "no verified marker");
    write(&installed.join(VERIFIED_MARKER), dspine_entry.revision);
    write(&installed.join("config.json"), DFLASH2_CONFIG);
    assert_eq!(
        installed_path(&snapshots),
        None,
        "a config other than the pinned one"
    );
    write(&installed.join("config.json"), DSPINE_CONFIG);
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

    // The runtime's per-format rule for a packed (Prism) head: DSpine refused, DFlash2 fits.
    let mut packed = target.clone();
    packed.pack = Some("bonsai2-packed".to_string());
    let packed_options = options_for(&packed, &snapshots, "mlx", &["dspine", "dflash2"]);
    assert!(option(&packed_options, "dspine")
        .unavailable
        .as_deref()
        .unwrap()
        .contains("packed (Prism)"));
    assert_eq!(option(&packed_options, "dflash2").unavailable, None);

    let mut gguf = target.clone();
    gguf.format = "gguf".to_string();
    assert!(option(
        &options_for(&gguf, &snapshots, "mlx", &["dspine"]),
        "dspine"
    )
    .unavailable
    .as_deref()
    .unwrap()
    .contains("GGUF"));

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
    assert!(why.contains("hidden_size 5120 != target 2048"), "{why}");
    assert!(why.contains("do not fit a 24-layer target"), "{why}");

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
    .contains("qwen3_5) targets only"));
}

/// The target's tokenizer counts: a mask token outside it is the runtime's refusal, named.
#[test]
fn a_mask_token_outside_the_targets_tokenizer_is_refused() {
    let dir = TempDir::new("block-draft-tokenizer");
    let snapshots = dir.path().join("snapshots");
    let target = entry_with_config(dir.path(), qwen38_vlm());
    write(
        &Path::new(&target.local_path).join("tokenizer.json"),
        &serde_json::json!({
            "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
            "normalizer": null, "pre_tokenizer": null, "post_processor": null, "decoder": null,
            "model": {"type": "WordLevel", "vocab": {"<unk>": 0, "a": 1}, "unk_token": "<unk>"},
        })
        .to_string(),
    );
    assert_eq!(draft_target(&target).unwrap().tokenizer_vocab, 2);
    let options = options_for(&target, &snapshots, "mlx", &["dspine", "dflash2"]);
    for option in options {
        let why = option.unavailable.unwrap();
        assert!(why.contains("mask_token_id 248070 is outside"), "{why}");
    }
}

/// The bundled configs are the pinned re-host revisions' `config.json`, byte for byte (sha256
/// of `SceneWorks/qwen3.8-27b-dspine-b16@8e450d7f…` and `SceneWorks/qwen3.8-27b-dflash2@9030fabf…`),
/// and each parses as its own format.
#[test]
fn the_bundled_configs_are_the_pinned_revisions() {
    use sha2::{Digest, Sha256};
    let digest = |text: &str| super::super::hex(&Sha256::digest(text.as_bytes()));
    assert_eq!(
        digest(DSPINE_CONFIG),
        "6b7e45dde6e2485c945bbec478c1e82cbd7610ed961f892e21297bb2491effd3"
    );
    assert_eq!(
        digest(DFLASH2_CONFIG),
        "873e3556509b0da06e29654ba00d4944888d4b5e8a33afde25f7eb27d321e980"
    );
    for draft in &BLOCK_DRAFT_CATALOG {
        let config = pinned_config(draft).unwrap();
        assert_eq!(config.format.label(), draft.format);
        assert!(
            config.mismatches(&QWEN38_27B).is_empty(),
            "{}",
            draft.format
        );
    }
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
    let dspine_entry = catalog_entry("dspine").unwrap();
    let dflash2_entry = catalog_entry("dflash2").unwrap();
    let dspine = dir.path().join("dspine");
    write(&dspine.join("config.json"), DSPINE_CONFIG);
    assert_eq!(
        check_installed_draft(&dspine, dspine_entry, &QWEN38_27B).unwrap(),
        dspine.to_string_lossy()
    );
    // The downloaded config is checked against the target, with the runtime's reasons.
    let small = BlockDraftTarget {
        hidden_size: 2048,
        num_layers: 24,
        ..QWEN38_27B
    };
    let why = check_installed_draft(&dspine, dspine_entry, &small).unwrap_err();
    assert!(why.contains("hidden_size 5120 != target 2048"), "{why}");
    assert!(check_installed_draft(&dspine, dflash2_entry, &QWEN38_27B)
        .unwrap_err()
        .contains("holds a dspine draft, not DFlash2"));

    let dflash2 = dir.path().join("dflash2");
    write(&dflash2.join("config.json"), DFLASH2_CONFIG);
    assert_eq!(
        check_installed_draft(&dflash2, dflash2_entry, &QWEN38_27B).unwrap(),
        dflash2.to_string_lossy()
    );

    let broken = dir.path().join("broken");
    write(
        &broken.join("config.json"),
        r#"{"architectures": ["DSpineDraftModel"]}"#,
    );
    assert!(check_installed_draft(&broken, dspine_entry, &QWEN38_27B)
        .unwrap_err()
        .starts_with("block draft: format:"));

    let plain = dir.path().join("plain");
    write(
        &plain.join("config.json"),
        r#"{"architectures": ["Qwen3ForCausalLM"]}"#,
    );
    assert!(check_installed_draft(&plain, dspine_entry, &QWEN38_27B)
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

/// A Prism/Bonsai MLX target (`prism_hadamard_qwen35` wrapping Qwen3.8-27B text geometry) is read
/// through its `text_config`: DFlash2 fits its packed head, DSpine keeps the packed-head refusal.
#[test]
fn a_prism_target_reads_its_text_config_and_only_dspine_is_refused() {
    let dir = TempDir::new("block-draft-prism");
    let snapshots = dir.path().join("snapshots");
    let mut prism = entry_with_config(
        dir.path(),
        serde_json::json!({
            "schema_version": 2, "model_type": "prism_hadamard_qwen35",
            "base_model_type": "qwen3_5",
            "components": {"text": true, "vision": true, "mtp": false},
            "quantization": {"bits": 2, "group_size": 128, "mode": "affine"},
            "vision_config": {"model_type": "qwen3_5"},
            "text_config": {"model_type": "qwen3_5_text", "hidden_size": 5120,
                "num_hidden_layers": 64, "vocab_size": 248320},
        }),
    );
    prism.pack = Some("bonsai2-packed".to_string());
    assert_eq!(draft_target(&prism).unwrap(), QWEN38_27B);

    let options = options_for(&prism, &snapshots, "mlx", &["dspine", "dflash2"]);
    assert_eq!(option(&options, "dflash2").unavailable, None);
    assert!(option(&options, "dspine")
        .unavailable
        .as_deref()
        .unwrap()
        .contains("packed (Prism)"));
}

/// A retry after an interrupted download: each missing file's partial is removed before the
/// free-space check, so the check asks only for the files' listed sizes on a reclaimed volume.
#[test]
fn a_retry_removes_partials_before_measuring_the_missing_bytes() {
    let dir = TempDir::new("block-draft-partials");
    let sibling = |name: &str, size: u64| -> HfSibling {
        serde_json::from_value(serde_json::json!({"rfilename": name, "size": size})).unwrap()
    };
    let files = [sibling("config.json", 4), sibling("model.safetensors", 10)];
    write(&dir.path().join("config.json"), "{ab}");
    let weights = dir.path().join("model.safetensors");
    let partial = part_path(&weights);
    assert_eq!(partial, dir.path().join("model.safetensors.part"));
    write(&partial, "123456");
    // A complete file's stray partial is not one this download will write; it is left alone.
    write(&part_path(&dir.path().join("config.json")), "x");

    assert_eq!(clear_partials_and_measure_missing(dir.path(), &files), 10);
    assert!(!partial.exists(), "the interrupted partial is reclaimed");
    assert!(dir.path().join("config.json").exists());
    assert!(dir.path().join("config.json.part").exists());
}

/// A load that read the manifest before a draft install finished records only what it owns:
/// the install's `draft_source` survives, and the load still selects the model and saves the
/// projector it was given.
#[test]
fn a_load_finishing_after_a_draft_install_keeps_the_installed_draft() {
    let dir = TempDir::new("block-draft-load-race");
    let target = entry_with_config(dir.path(), qwen38_vlm());
    let manifest = dir.path().join("manifest.json");
    write_registry(
        &manifest,
        &ModelRegistry {
            models: vec![target],
            selected_id: None,
        },
    )
    .unwrap();
    // The load reads the manifest, the install saves the draft, then the load finishes.
    let _read_at_load_start = read_registry(&manifest).unwrap();
    set_block_draft(&manifest, "target", "/snapshots/dflash2").unwrap();
    super::super::record_loaded_model(&manifest, "target", Some(Some("/mmproj".into()))).unwrap();

    let saved = read_registry(&manifest).unwrap();
    assert_eq!(
        saved.models[0].draft_source.as_deref(),
        Some("/snapshots/dflash2")
    );
    assert_eq!(saved.models[0].projector_source.as_deref(), Some("/mmproj"));
    assert_eq!(saved.selected_id.as_deref(), Some("target"));

    // Without a projector argument the saved association is kept; a removed model is not re-added.
    super::super::record_loaded_model(&manifest, "target", None).unwrap();
    assert_eq!(
        read_registry(&manifest).unwrap().models[0]
            .projector_source
            .as_deref(),
        Some("/mmproj")
    );
    super::super::record_loaded_model(&manifest, "gone", None).unwrap();
    assert_eq!(read_registry(&manifest).unwrap().models.len(), 1);
}
