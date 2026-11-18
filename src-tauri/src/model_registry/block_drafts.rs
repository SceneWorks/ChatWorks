//! Block drafts (feature SW-F360, story SW-7199): the draft checkpoints ChatWorks can install for
//! a registered model's `block_draft` proposer, whether a model can use one, and the install path.
//!
//! A block draft is not a model of its own: it is installed beside a registered target, saved as
//! that target's `draft_source` and loaded with it through the runtime's `LoadSpec::draft_source`
//! (at `LoadSpec::draft_quantize` precision). The runtime decides at load whether the draft fits
//! the target and names a refusal in the load report; ChatWorks only explains up front what it
//! can see without loading (the build's backend, the target's geometry, the draft's format).

use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;
use tauri::AppHandle;

use super::{
    app_data_dir, cached_file_matches, emit_progress, ensure_free_space, env_hf_token,
    fetch_hf_files, file_is_complete, import_token, read_hf_token, read_registry, registry_path,
    snapshot_dir_name, sum_known_sizes, write_registry, DownloadContext, HfModelRef,
    ImportProgress, ModelEntry, ModelRegistry, SnapshotDownload,
};
use crate::core_llm::{
    block_draft_refusal_reason, BlockDraftConfig, BlockDraftFormat, BlockDraftRefusal,
    BlockDraftTarget, Tokenizer,
};

/// One installable block draft: a public re-host pinned to an immutable revision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BlockDraftCatalogEntry {
    /// The runtime's format label (`BlockDraftFormat::label`), also the catalog id.
    pub format: &'static str,
    pub label: &'static str,
    pub repo: &'static str,
    pub revision: &'static str,
    pub license: &'static str,
    /// The re-host's `config.json` at `revision`, byte for byte: what ChatWorks checks a target
    /// against before downloading, and what the downloaded config must equal.
    pub config: &'static str,
}

/// The block drafts ChatWorks offers. Both are drafted for Qwen3.8-27B targets.
pub(crate) const BLOCK_DRAFT_CATALOG: [BlockDraftCatalogEntry; 2] = [
    BlockDraftCatalogEntry {
        format: "dspine",
        label: "DSpine (b16)",
        repo: "SceneWorks/qwen3.8-27b-dspine-b16",
        revision: "8e450d7faf38238b64580163a7d3dd7e44af759d",
        license: "Apache-2.0",
        config: include_str!("block_drafts/dspine-config.json"),
    },
    BlockDraftCatalogEntry {
        format: "dflash2",
        label: "DFlash2",
        repo: "SceneWorks/qwen3.8-27b-dflash2",
        revision: "9030fabfb5e3ca0c5c180f84927af86aa9f9a2e9",
        license: "Apache-2.0",
        config: include_str!("block_drafts/dflash2-config.json"),
    },
];

/// Free space a draft download leaves on its volume, beyond the draft itself.
const DOWNLOAD_RESERVE_BYTES: u64 = 2 << 30;

/// Written into a draft's snapshot once every file is on disk and matches the listing; its body
/// is the revision. Without it a snapshot is not installed (an interrupted or foreign copy).
const VERIFIED_MARKER: &str = ".chatworks-verified";

/// A catalog draft as the Models screen shows it for one registered model.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BlockDraftOption {
    pub format: String,
    pub label: String,
    pub repo: String,
    pub revision: String,
    pub license: String,
    /// The installed snapshot directory, when every file is on disk.
    pub installed_path: Option<String>,
    /// Why this model cannot use this draft, or `None` when it can.
    pub unavailable: Option<String>,
}

/// The catalog against the registered model `model_id`.
pub fn block_draft_options(
    app: &AppHandle,
    model_id: &str,
) -> Result<Vec<BlockDraftOption>, String> {
    let registry = read_registry(&registry_path(app)?)?;
    let entry = find_entry(&registry, model_id)?;
    let snapshots = snapshots_dir(app)?;
    Ok(options_for(
        entry,
        &snapshots,
        crate::inference_runtime::execution_backend(),
        runtime_formats(),
    ))
}

fn options_for(
    entry: &ModelEntry,
    snapshots: &Path,
    backend: &str,
    runtime_formats: &[&str],
) -> Vec<BlockDraftOption> {
    let target = draft_target(entry);
    BLOCK_DRAFT_CATALOG
        .iter()
        .map(|draft| {
            let dir = snapshot_path(snapshots, draft);
            BlockDraftOption {
                format: draft.format.to_string(),
                label: draft.label.to_string(),
                repo: draft.repo.to_string(),
                revision: draft.revision.to_string(),
                license: draft.license.to_string(),
                installed_path: is_installed(&dir, draft)
                    .then(|| dir.to_string_lossy().to_string()),
                unavailable: unavailable_reason(entry, &target, draft, backend, runtime_formats),
            }
        })
        .collect()
}

/// The block-draft formats the linked runtime reads.
fn runtime_formats() -> &'static [&'static str] {
    const LABELS: [&str; BlockDraftFormat::ALL.len()] = {
        let mut labels = [""; BlockDraftFormat::ALL.len()];
        let mut index = 0;
        while index < labels.len() {
            labels[index] = BlockDraftFormat::ALL[index].label();
            index += 1;
        }
        labels
    };
    &LABELS
}

/// Why `entry` (whose block-draft geometry is `target`) cannot use `draft` on this build, from
/// what is visible without loading: the backend, the runtime's formats, the checkpoint kind and
/// the runtime's own fit rule (`BlockDraftConfig::mismatches`) between the draft's pinned config
/// and the target's. `None` when it can; the runtime re-checks the draft's tensors at load.
fn unavailable_reason(
    entry: &ModelEntry,
    target: &Result<BlockDraftTarget, String>,
    draft: &BlockDraftCatalogEntry,
    backend: &str,
    runtime_formats: &[&str],
) -> Option<String> {
    if backend != "mlx" {
        return Some(format!(
            "Block drafts run on the MLX backend in this inference runtime; this build uses {backend}."
        ));
    }
    if !runtime_formats.contains(&draft.format) {
        return Some(format!(
            "This build's inference runtime does not read {} drafts.",
            draft.label
        ));
    }
    if entry.format.starts_with("gguf") {
        return Some(format!(
            "{} drafts read a dense safetensors Qwen3.5/3.8 target; this model is a GGUF checkpoint.",
            draft.label
        ));
    }
    // The runtime's rule: DSpine's refinement scores candidates against the LM head's dense rows,
    // which a packed (Prism) head does not have; DFlash2's selector reads only its logits.
    if entry.pack.is_some() && draft.format == BlockDraftFormat::DSpine.label() {
        return Some(format!(
            "{} cannot draft for a packed (Prism) checkpoint: its packed LM head has no dense rows \
             for the refinement.",
            draft.label
        ));
    }
    let target = match target {
        Ok(target) => target,
        Err(why) => return Some(format!("{}: {why}", draft.label)),
    };
    let config = match pinned_config(draft) {
        Ok(config) => config,
        Err(why) => return Some(format!("{}: {why}", draft.label)),
    };
    let mismatches = config.mismatches(target);
    (!mismatches.is_empty()).then(|| {
        format!(
            "{} does not fit this model: {}",
            draft.label,
            block_draft_refusal_reason(&mismatches)
        )
    })
}

/// The draft's pinned config as the runtime parses it.
fn pinned_config(draft: &BlockDraftCatalogEntry) -> Result<BlockDraftConfig, String> {
    let raw: serde_json::Value =
        serde_json::from_str(draft.config).map_err(|error| error.to_string())?;
    BlockDraftConfig::from_json(&raw).map_err(|refusal| refusal.to_string())
}

/// What a registered model offers a block draft (`BlockDraftTarget`), read as the runtime reads
/// it at load: a Qwen3.5/3.8 hybrid (`qwen3_5`) target's text geometry from `config.json` (the
/// VLM wrapper's `text_config` or a text-only fine-tune's root) and its tokenizer's vocabulary
/// (added tokens included) from `tokenizer.json`, or the config's `vocab_size` without one.
fn draft_target(entry: &ModelEntry) -> Result<BlockDraftTarget, String> {
    let snapshot = Path::new(&entry.local_path);
    let not_a_target = || {
        BlockDraftRefusal::TargetFamily(
            "block drafts read Qwen3.5/3.8 hybrid (qwen3_5) targets only".into(),
        )
        .to_string()
    };
    let body = fs::read_to_string(snapshot.join("config.json")).map_err(|_| not_a_target())?;
    let config: serde_json::Value = serde_json::from_str(&body).map_err(|_| not_a_target())?;
    let text = match config.get("text_config") {
        Some(text) if config["model_type"] == "qwen3_5" => text,
        _ => &config,
    };
    if text["model_type"] != "qwen3_5_text" {
        return Err(not_a_target());
    }
    let field = |name: &str| {
        text[name]
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| format!("the target's config has no {name}"))
    };
    let vocab_size = field("vocab_size")?;
    let tokenizer_vocab = Tokenizer::from_file(snapshot.join("tokenizer.json"))
        .map_or(vocab_size, |tokenizer| tokenizer.vocab_size());
    Ok(BlockDraftTarget {
        hidden_size: field("hidden_size")?,
        num_layers: field("num_hidden_layers")?,
        vocab_size,
        tokenizer_vocab,
    })
}

fn find_entry<'a>(registry: &'a ModelRegistry, model_id: &str) -> Result<&'a ModelEntry, String> {
    registry
        .models
        .iter()
        .find(|model| model.id == model_id)
        .ok_or_else(|| format!("model {model_id:?} is not in the registry"))
}

fn catalog_entry(format: &str) -> Result<&'static BlockDraftCatalogEntry, String> {
    BLOCK_DRAFT_CATALOG
        .iter()
        .find(|draft| draft.format == format)
        .ok_or_else(|| format!("unknown block draft {format:?}"))
}

fn snapshots_dir(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(app_data_dir(app)?.join("models").join("snapshots"))
}

fn model_ref(draft: &BlockDraftCatalogEntry) -> HfModelRef {
    HfModelRef {
        repo: draft.repo.to_string(),
        revision: draft.revision.to_string(),
        file_name: None,
    }
}

fn snapshot_path(snapshots: &Path, draft: &BlockDraftCatalogEntry) -> PathBuf {
    snapshots.join(snapshot_dir_name(&model_ref(draft)))
}

/// Installed: the pinned config and the single `model.safetensors` every catalog draft
/// publishes, verified against the listing when they were downloaded (the marker).
fn is_installed(dir: &Path, draft: &BlockDraftCatalogEntry) -> bool {
    dir.join("model.safetensors").is_file()
        && fs::read_to_string(dir.join("config.json")).is_ok_and(|config| config == draft.config)
        && fs::read_to_string(dir.join(VERIFIED_MARKER))
            .is_ok_and(|revision| revision == draft.revision)
}

/// Download `format`'s pinned draft (if not on disk), check it is a block draft the runtime
/// reads, and save it as `model_id`'s draft: the next load attaches it.
pub async fn install_block_draft(
    app: AppHandle,
    model_id: String,
    format: String,
    job_id: String,
) -> Result<ModelRegistry, String> {
    let result = install_block_draft_inner(&app, &model_id, &format, &job_id).await;
    if let Err(error) = &result {
        emit_progress(
            &app,
            ImportProgress {
                job_id,
                stage: "error".to_string(),
                message: error.clone(),
                progress: 1.0,
                downloaded_bytes: 0,
                total_bytes: None,
            },
        );
    }
    result
}

async fn install_block_draft_inner(
    app: &AppHandle,
    model_id: &str,
    format: &str,
    job_id: &str,
) -> Result<ModelRegistry, String> {
    let draft = catalog_entry(format)?;
    let manifest = registry_path(app)?;
    let entry = find_entry(&read_registry(&manifest)?, model_id)?.clone();
    let target = draft_target(&entry);
    if let Some(why) = unavailable_reason(
        &entry,
        &target,
        draft,
        crate::inference_runtime::execution_backend(),
        runtime_formats(),
    ) {
        return Err(why);
    }
    let target = target?;
    let dir = snapshot_path(&snapshots_dir(app)?, draft);
    download_draft(app, draft, &dir, job_id).await?;
    let dir = check_installed_draft(&dir, draft, &target)?;
    let registry = set_block_draft(&manifest, model_id, &dir)?;
    emit_progress(
        app,
        ImportProgress {
            job_id: job_id.to_string(),
            stage: "done".to_string(),
            message: format!("{} installed; it loads with the model", draft.label),
            progress: 1.0,
            downloaded_bytes: 0,
            total_bytes: None,
        },
    );
    Ok(registry)
}

async fn download_draft(
    app: &AppHandle,
    draft: &BlockDraftCatalogEntry,
    dir: &Path,
    job_id: &str,
) -> Result<(), String> {
    let model_ref = model_ref(draft);
    let token = import_token(
        crate::profile::root()?.is_some(),
        read_hf_token(),
        env_hf_token,
    )?;
    let client = reqwest::Client::new();
    let files = fetch_hf_files(&client, &model_ref, None, token.as_deref()).await?;
    if !files.iter().any(|file| file.rfilename == "config.json") {
        return Err(format!(
            "{}@{} has no config.json",
            draft.repo, draft.revision
        ));
    }
    let _downloading = SnapshotDownload::claim(dir)?;
    let total_bytes = sum_known_sizes(&files);
    fs::create_dir_all(dir).map_err(|error| error.to_string())?;
    let missing = files
        .iter()
        .filter(|file| !file_is_complete(&dir.join(&file.rfilename), file.size))
        .map(|file| file.size.unwrap_or(0))
        .fold(0_u64, u64::saturating_add);
    ensure_free_space(dir, missing, DOWNLOAD_RESERVE_BYTES)?;
    // The marker is rewritten only once every file below matches the listing.
    let _ = fs::remove_file(dir.join(VERIFIED_MARKER));
    let download = DownloadContext {
        app,
        client: &client,
        model_ref: &model_ref,
        token: token.as_deref(),
        job_id,
        total_bytes,
    };
    let mut downloaded_bytes = 0_u64;
    for file in &files {
        let target = dir.join(&file.rfilename);
        if cached_file_matches(&target, file).await {
            downloaded_bytes = downloaded_bytes.saturating_add(file.size.unwrap_or(0));
            continue;
        }
        download
            .download_file(file, &target, &mut downloaded_bytes)
            .await?;
    }
    let config = fs::read_to_string(dir.join("config.json")).map_err(|error| error.to_string())?;
    if config != draft.config {
        return Err(format!(
            "{}@{} config.json differs from the config this build pins for it",
            draft.repo, draft.revision
        ));
    }
    fs::write(dir.join(VERIFIED_MARKER), draft.revision).map_err(|error| error.to_string())?;
    Ok(())
}

/// The runtime's own reading of an installed draft: `draft`'s block-draft format, with a config
/// inside that format's contract that fits `target` (`BlockDraftConfig::mismatches`, the rule
/// the runtime applies at load). Returns the directory as saved.
fn check_installed_draft(
    dir: &Path,
    draft: &BlockDraftCatalogEntry,
    target: &BlockDraftTarget,
) -> Result<String, String> {
    match BlockDraftFormat::of_dir(dir) {
        Some(format) if format.label() == draft.format => {}
        Some(format) => {
            return Err(format!(
                "{} holds a {} draft, not {}",
                dir.display(),
                format.label(),
                draft.label
            ))
        }
        None => {
            return Err(format!(
                "{} is not a block draft this build's inference runtime reads",
                dir.display()
            ))
        }
    }
    let config = BlockDraftConfig::read(dir).map_err(|refusal| refusal.to_string())?;
    let mismatches = config.mismatches(target);
    if !mismatches.is_empty() {
        return Err(block_draft_refusal_reason(&mismatches));
    }
    Ok(dir.to_string_lossy().to_string())
}

fn set_block_draft(manifest: &Path, model_id: &str, dir: &str) -> Result<ModelRegistry, String> {
    let mut registry = read_registry(manifest)?;
    let entry = registry
        .models
        .iter_mut()
        .find(|model| model.id == model_id)
        .ok_or_else(|| format!("model {model_id:?} is not in the registry"))?;
    entry.draft_source = Some(dir.to_string());
    write_registry(manifest, &registry)?;
    Ok(registry)
}

#[cfg(test)]
mod tests;
