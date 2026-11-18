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
    app_data_dir, emit_progress, env_hf_token, fetch_hf_files, file_is_complete, import_token,
    read_hf_token, read_registry, registry_path, snapshot_dir_name, sum_known_sizes,
    write_registry, DownloadContext, HfModelRef, ImportProgress, ModelEntry, ModelRegistry,
};
use crate::core_llm::{BlockDraftConfig, BlockDraftFormat};

/// One installable block draft: a public re-host pinned to an immutable revision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BlockDraftCatalogEntry {
    /// The runtime's format label (`BlockDraftFormat::label`), also the catalog id.
    pub format: &'static str,
    pub label: &'static str,
    pub repo: &'static str,
    pub revision: &'static str,
    pub license: &'static str,
}

/// The block drafts ChatWorks offers. Both are drafted for Qwen3.8-27B targets.
pub(crate) const BLOCK_DRAFT_CATALOG: [BlockDraftCatalogEntry; 2] = [
    BlockDraftCatalogEntry {
        format: "dspine",
        label: "DSpine (b16)",
        repo: "SceneWorks/qwen3.8-27b-dspine-b16",
        revision: "8e450d7faf38238b64580163a7d3dd7e44af759d",
        license: "Apache-2.0",
    },
    BlockDraftCatalogEntry {
        format: "dflash2",
        label: "DFlash2",
        repo: "SceneWorks/qwen3.8-27b-dflash2",
        revision: "9030fabfb5e3ca0c5c180f84927af86aa9f9a2e9",
        license: "Apache-2.0",
    },
];

/// The target the catalog's drafts were trained against: a Qwen3.8-27B decoder (its text
/// geometry). The runtime re-checks the full contract (mask token, layer taps, tensors) at load.
const TARGET_NAME: &str = "Qwen3.8-27B";
const TARGET_HIDDEN_SIZE: u64 = 5120;
const TARGET_LAYERS: u64 = 64;
const TARGET_VOCAB: u64 = 248_320;

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
                installed_path: is_installed(&dir).then(|| dir.to_string_lossy().to_string()),
                unavailable: unavailable_reason(entry, draft, backend, runtime_formats),
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

/// Why `entry` cannot use `draft` on this build, from what is visible without loading: the
/// backend, the runtime's formats and the target's stored geometry. `None` when it can.
fn unavailable_reason(
    entry: &ModelEntry,
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
    if entry.format.starts_with("gguf") || entry.pack.is_some() {
        return Some(format!(
            "{} drafts read a dense safetensors {TARGET_NAME} target; this model is a packed or GGUF checkpoint.",
            draft.label
        ));
    }
    match target_geometry(Path::new(&entry.local_path)) {
        Some((hidden, layers, vocab))
            if hidden == TARGET_HIDDEN_SIZE && layers == TARGET_LAYERS && vocab == TARGET_VOCAB =>
        {
            None
        }
        Some((hidden, layers, vocab)) => Some(format!(
            "{} is drafted for {TARGET_NAME} targets (hidden size {TARGET_HIDDEN_SIZE}, \
             {TARGET_LAYERS} layers, vocabulary {TARGET_VOCAB}); this model has hidden size \
             {hidden}, {layers} layers, vocabulary {vocab}.",
            draft.label
        )),
        None => Some(format!(
            "{} is drafted for {TARGET_NAME} (qwen3_5) targets; this model is not one.",
            draft.label
        )),
    }
}

/// A Qwen3.5/3.8 target's text geometry (`hidden_size`, `num_hidden_layers`, `vocab_size`) from
/// its `config.json`, the VLM wrapper's `text_config` or a text-only fine-tune's root.
fn target_geometry(snapshot: &Path) -> Option<(u64, u64, u64)> {
    let body = fs::read_to_string(snapshot.join("config.json")).ok()?;
    let config: serde_json::Value = serde_json::from_str(&body).ok()?;
    let text = match config.get("text_config") {
        Some(text) if config["model_type"] == "qwen3_5" => text,
        _ => &config,
    };
    if text["model_type"] != "qwen3_5_text" {
        return None;
    }
    Some((
        text["hidden_size"].as_u64()?,
        text["num_hidden_layers"].as_u64()?,
        text["vocab_size"].as_u64()?,
    ))
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

/// Installed: the config and the single `model.safetensors` every catalog draft publishes.
fn is_installed(dir: &Path) -> bool {
    dir.join("config.json").is_file() && dir.join("model.safetensors").is_file()
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
    if let Some(why) = unavailable_reason(
        &entry,
        draft,
        crate::inference_runtime::execution_backend(),
        runtime_formats(),
    ) {
        return Err(why);
    }
    let dir = snapshot_path(&snapshots_dir(app)?, draft);
    download_draft(app, draft, &dir, job_id).await?;
    let dir = check_installed_draft(&dir)?;
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
    let total_bytes = sum_known_sizes(&files);
    fs::create_dir_all(dir).map_err(|error| error.to_string())?;
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
        if file_is_complete(&target, file.size) {
            downloaded_bytes = downloaded_bytes.saturating_add(file.size.unwrap_or(0));
            continue;
        }
        download
            .download_file(file, &target, &mut downloaded_bytes)
            .await?;
    }
    Ok(())
}

/// The runtime's own reading of an installed draft: a block-draft format it reads, with a config
/// inside that format's contract. Returns the directory as saved.
fn check_installed_draft(dir: &Path) -> Result<String, String> {
    if BlockDraftFormat::of_dir(dir).is_none() {
        return Err(format!(
            "{} is not a block draft this build's inference runtime reads",
            dir.display()
        ));
    }
    BlockDraftConfig::read(dir).map_err(|refusal| refusal.to_string())?;
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
