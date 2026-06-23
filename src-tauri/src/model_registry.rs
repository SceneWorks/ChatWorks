use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager};

use crate::engine::{EngineHandle, EngineStatus, LoadModelRequest, QuantizeRequest};

const HF_HOST: &str = "huggingface.co";
const HF_KEYCHAIN_SERVICE: &str = "net.trefry.chatworks.huggingface";
const HF_KEYCHAIN_USER: &str = "token";
const PROGRESS_EVENT: &str = "models://import-progress";

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImportHfModelRequest {
    pub source_url: String,
    #[serde(default)]
    pub quantize: Option<QuantizeRequest>,
    #[serde(default)]
    pub job_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SetHfTokenRequest {
    pub token: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HfTokenStatus {
    pub present: bool,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelRegistry {
    #[serde(default)]
    pub models: Vec<ModelEntry>,
    #[serde(default)]
    pub selected_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelEntry {
    pub id: String,
    pub name: String,
    pub repo: String,
    pub revision: String,
    pub source_url: String,
    pub local_path: String,
    #[serde(default)]
    pub quantize: Option<QuantizeRequest>,
    pub imported_at: u64,
    pub file_count: usize,
    #[serde(default)]
    pub size_bytes: Option<u64>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImportProgress {
    pub job_id: String,
    pub stage: String,
    pub message: String,
    pub progress: f32,
    pub downloaded_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_bytes: Option<u64>,
}

#[derive(Clone, Debug)]
struct HfModelRef {
    repo: String,
    revision: String,
}

#[derive(Clone, Debug, Deserialize)]
struct HfApiModel {
    siblings: Vec<HfSibling>,
}

#[derive(Clone, Debug, Deserialize)]
struct HfSibling {
    rfilename: String,
    #[serde(default)]
    size: Option<u64>,
}

pub fn list_registered_models(app: &AppHandle) -> Result<ModelRegistry, String> {
    read_registry(&registry_path(app)?)
}

pub async fn import_hf_model(
    app: AppHandle,
    request: ImportHfModelRequest,
) -> Result<ModelRegistry, String> {
    let job_id = request.job_id.clone().unwrap_or_else(make_job_id);
    match import_hf_model_inner(&app, request, &job_id).await {
        Ok(registry) => Ok(registry),
        Err(error) => {
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
            Err(error)
        }
    }
}

pub fn load_registered_model(
    app: &AppHandle,
    engine: &EngineHandle,
    model_id: String,
) -> Result<EngineStatus, String> {
    let manifest = registry_path(app)?;
    let mut registry = read_registry(&manifest)?;
    let entry = registry
        .models
        .iter()
        .find(|model| model.id == model_id)
        .cloned()
        .ok_or_else(|| format!("model {model_id:?} is not in the registry"))?;
    let status = engine.load_model(LoadModelRequest {
        source: entry.local_path.clone(),
        display_name: Some(entry.name.clone()),
        quantize: entry.quantize,
    })?;
    registry.selected_id = Some(entry.id);
    write_registry(&manifest, &registry)?;
    Ok(status)
}

pub fn hf_token_status() -> HfTokenStatus {
    HfTokenStatus {
        present: read_hf_token().ok().flatten().is_some(),
    }
}

pub fn set_hf_token(request: SetHfTokenRequest) -> Result<HfTokenStatus, String> {
    let token = request.token.trim();
    if token.is_empty() {
        return Err("HuggingFace token is required".to_string());
    }
    let entry = keyring::Entry::new(HF_KEYCHAIN_SERVICE, HF_KEYCHAIN_USER)
        .map_err(|error| error.to_string())?;
    entry
        .set_password(token)
        .map_err(|error| error.to_string())?;
    Ok(hf_token_status())
}

pub fn clear_hf_token() -> Result<HfTokenStatus, String> {
    let entry = keyring::Entry::new(HF_KEYCHAIN_SERVICE, HF_KEYCHAIN_USER)
        .map_err(|error| error.to_string())?;
    match entry.delete_credential() {
        Ok(()) => Ok(hf_token_status()),
        Err(keyring::Error::NoEntry) => Ok(hf_token_status()),
        Err(error) => Err(error.to_string()),
    }
}

async fn import_hf_model_inner(
    app: &AppHandle,
    request: ImportHfModelRequest,
    job_id: &str,
) -> Result<ModelRegistry, String> {
    let model_ref = HfModelRef::parse(&request.source_url)?;
    let data_dir = app_data_dir(app)?;
    let snapshots_dir = data_dir.join("models").join("snapshots");
    let snapshot_dir = snapshots_dir.join(snapshot_dir_name(&model_ref));
    let manifest = registry_path(app)?;
    let token = read_hf_token().ok().flatten().or_else(env_hf_token);
    let client = reqwest::Client::new();

    emit_progress(
        app,
        ImportProgress {
            job_id: job_id.to_string(),
            stage: "queued".to_string(),
            message: format!("Resolving {}", model_ref.repo),
            progress: 0.0,
            downloaded_bytes: 0,
            total_bytes: None,
        },
    );

    let files = fetch_hf_files(&client, &model_ref, token.as_deref()).await?;
    if files.is_empty() {
        return Err("no loadable model files found in the HuggingFace repo".to_string());
    }
    let total_bytes = sum_known_sizes(&files);
    fs::create_dir_all(&snapshot_dir).map_err(|error| error.to_string())?;

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
        let target = snapshot_dir.join(&file.rfilename);
        if file_is_complete(&target, file.size) {
            downloaded_bytes = downloaded_bytes.saturating_add(file.size.unwrap_or(0));
            emit_progress(
                app,
                ImportProgress {
                    job_id: job_id.to_string(),
                    stage: "download".to_string(),
                    message: format!("Using cached {}", file.rfilename),
                    progress: progress(downloaded_bytes, total_bytes),
                    downloaded_bytes,
                    total_bytes,
                },
            );
            continue;
        }
        download
            .download_file(file, &target, &mut downloaded_bytes)
            .await?;
    }

    emit_progress(
        app,
        ImportProgress {
            job_id: job_id.to_string(),
            stage: "convert".to_string(),
            message: "Preparing MLX snapshot".to_string(),
            progress: 0.96,
            downloaded_bytes,
            total_bytes,
        },
    );
    validate_snapshot(&snapshot_dir)?;

    let mut registry = read_registry(&manifest)?;
    let entry = ModelEntry {
        id: model_id(&model_ref, request.quantize),
        name: model_name(&model_ref, request.quantize),
        repo: model_ref.repo.clone(),
        revision: model_ref.revision.clone(),
        source_url: model_ref.source_url(),
        local_path: snapshot_dir.to_string_lossy().to_string(),
        quantize: request.quantize,
        imported_at: now_secs(),
        file_count: files.len(),
        size_bytes: total_bytes,
    };
    upsert_model(&mut registry, entry);
    write_registry(&manifest, &registry)?;

    emit_progress(
        app,
        ImportProgress {
            job_id: job_id.to_string(),
            stage: "done".to_string(),
            message: "Model added to registry".to_string(),
            progress: 1.0,
            downloaded_bytes,
            total_bytes,
        },
    );

    Ok(registry)
}

async fn fetch_hf_files(
    client: &reqwest::Client,
    model_ref: &HfModelRef,
    token: Option<&str>,
) -> Result<Vec<HfSibling>, String> {
    let url = format!(
        "https://{}/api/models/{}/revision/{}?blobs=true",
        HF_HOST, model_ref.repo, model_ref.revision
    );
    let mut request = client.get(url);
    if let Some(token) = token {
        request = request.bearer_auth(token);
    }
    let response = request.send().await.map_err(|error| error.to_string())?;
    if response.status() == reqwest::StatusCode::UNAUTHORIZED
        || response.status() == reqwest::StatusCode::FORBIDDEN
    {
        return Err("HuggingFace access denied; save a token for gated/private models".to_string());
    }
    let payload = response
        .error_for_status()
        .map_err(|error| error.to_string())?
        .json::<HfApiModel>()
        .await
        .map_err(|error| error.to_string())?;
    let mut files = Vec::new();
    for file in payload.siblings {
        if !is_loadable_model_file(&file.rfilename) {
            continue;
        }
        validate_hf_file_name(&file.rfilename)?;
        files.push(file);
    }
    files.sort_by(|a, b| a.rfilename.cmp(&b.rfilename));
    Ok(files)
}

struct DownloadContext<'a> {
    app: &'a AppHandle,
    client: &'a reqwest::Client,
    model_ref: &'a HfModelRef,
    token: Option<&'a str>,
    job_id: &'a str,
    total_bytes: Option<u64>,
}

impl DownloadContext<'_> {
    async fn download_file(
        &self,
        file: &HfSibling,
        target: &Path,
        downloaded_bytes: &mut u64,
    ) -> Result<(), String> {
        let url = format!(
            "https://{}/{}/resolve/{}/{}",
            HF_HOST, self.model_ref.repo, self.model_ref.revision, file.rfilename
        );
        let mut request = self.client.get(url);
        if let Some(token) = self.token {
            request = request.bearer_auth(token);
        }
        let response = request
            .send()
            .await
            .map_err(|error| error.to_string())?
            .error_for_status()
            .map_err(|error| error.to_string())?;
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        let part_path = target.with_extension(format!(
            "{}.part",
            target
                .extension()
                .and_then(|ext| ext.to_str())
                .unwrap_or("download")
        ));
        let mut output = fs::File::create(&part_path).map_err(|error| error.to_string())?;
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|error| error.to_string())?;
            output
                .write_all(&chunk)
                .map_err(|error| error.to_string())?;
            *downloaded_bytes = downloaded_bytes.saturating_add(chunk.len() as u64);
            emit_progress(
                self.app,
                ImportProgress {
                    job_id: self.job_id.to_string(),
                    stage: "download".to_string(),
                    message: format!("Downloading {}", file.rfilename),
                    progress: progress(*downloaded_bytes, self.total_bytes),
                    downloaded_bytes: *downloaded_bytes,
                    total_bytes: self.total_bytes,
                },
            );
        }
        output.flush().map_err(|error| error.to_string())?;
        fs::rename(&part_path, target).map_err(|error| error.to_string())?;
        Ok(())
    }
}

impl HfModelRef {
    fn parse(input: &str) -> Result<Self, String> {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Err("HuggingFace URL is required".to_string());
        }
        let without_scheme = trimmed
            .strip_prefix("https://huggingface.co/")
            .or_else(|| trimmed.strip_prefix("http://huggingface.co/"))
            .or_else(|| trimmed.strip_prefix("hf://"))
            .unwrap_or(trimmed);
        let path = without_scheme
            .split(['?', '#'])
            .next()
            .unwrap_or(without_scheme)
            .trim_matches('/');
        let segments: Vec<_> = path
            .split('/')
            .filter(|segment| !segment.is_empty())
            .collect();
        if segments.is_empty() {
            return Err("HuggingFace repo path is required".to_string());
        }
        let marker = segments
            .iter()
            .position(|segment| matches!(*segment, "tree" | "blob" | "resolve"));
        let (repo_segments, revision) = if let Some(index) = marker {
            if index == 0 {
                return Err("HuggingFace repo path is required".to_string());
            }
            let revision = segments.get(index + 1).copied().unwrap_or("main");
            (&segments[..index], revision)
        } else if segments.len() >= 2 {
            (&segments[..2], "main")
        } else {
            (&segments[..1], "main")
        };
        let repo = repo_segments.join("/");
        validate_hf_path(&repo, "repo")?;
        validate_hf_path(revision, "revision")?;
        Ok(Self {
            repo,
            revision: revision.to_string(),
        })
    }

    fn source_url(&self) -> String {
        format!("https://{}/{}/tree/{}", HF_HOST, self.repo, self.revision)
    }
}

fn validate_hf_path(value: &str, label: &str) -> Result<(), String> {
    if value.contains("..")
        || value.starts_with('/')
        || value.chars().any(|ch| ch.is_control() || ch == '\\')
    {
        return Err(format!("invalid HuggingFace {label}"));
    }
    Ok(())
}

fn is_loadable_model_file(name: &str) -> bool {
    name == "config.json"
        || name == "tokenizer.json"
        || name == "tokenizer_config.json"
        || name == "special_tokens_map.json"
        || name == "generation_config.json"
        || name.ends_with(".safetensors")
        || name.ends_with(".safetensors.index.json")
}

fn validate_hf_file_name(name: &str) -> Result<(), String> {
    if name.is_empty()
        || name.starts_with('/')
        || name.contains("..")
        || name.chars().any(|ch| ch.is_control() || ch == '\\')
    {
        return Err(format!("invalid HuggingFace filename {name:?}"));
    }
    Ok(())
}

fn validate_snapshot(path: &Path) -> Result<(), String> {
    if !path.join("config.json").is_file() {
        return Err("downloaded snapshot is missing config.json".to_string());
    }
    if !path.join("tokenizer.json").is_file() {
        return Err("downloaded snapshot is missing tokenizer.json".to_string());
    }
    let has_safetensors = fs::read_dir(path)
        .map_err(|error| error.to_string())?
        .filter_map(Result::ok)
        .any(|entry| entry.path().extension().and_then(|ext| ext.to_str()) == Some("safetensors"));
    if !has_safetensors {
        return Err("downloaded snapshot is missing safetensors weights".to_string());
    }
    Ok(())
}

fn app_data_dir(app: &AppHandle) -> Result<PathBuf, String> {
    app.path().app_data_dir().map_err(|error| error.to_string())
}

fn registry_path(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(app_data_dir(app)?.join("models").join("manifest.json"))
}

fn read_registry(path: &Path) -> Result<ModelRegistry, String> {
    if !path.exists() {
        return Ok(ModelRegistry::default());
    }
    let body = fs::read_to_string(path).map_err(|error| error.to_string())?;
    serde_json::from_str(&body).map_err(|error| error.to_string())
}

fn write_registry(path: &Path, registry: &ModelRegistry) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let tmp = path.with_extension("json.tmp");
    fs::write(
        &tmp,
        serde_json::to_string_pretty(registry).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    fs::rename(tmp, path).map_err(|error| error.to_string())
}

fn upsert_model(registry: &mut ModelRegistry, entry: ModelEntry) {
    if let Some(existing) = registry
        .models
        .iter_mut()
        .find(|model| model.id == entry.id)
    {
        *existing = entry;
    } else {
        registry.models.push(entry);
    }
    registry.models.sort_by(|a, b| a.name.cmp(&b.name));
}

fn file_is_complete(path: &Path, size: Option<u64>) -> bool {
    let Ok(metadata) = fs::metadata(path) else {
        return false;
    };
    size.map_or(true, |expected| metadata.len() == expected)
}

fn sum_known_sizes(files: &[HfSibling]) -> Option<u64> {
    let mut total = 0_u64;
    for file in files {
        total = total.checked_add(file.size?)?;
    }
    Some(total)
}

fn progress(downloaded: u64, total: Option<u64>) -> f32 {
    match total {
        Some(0) | None => 0.0,
        Some(total) => (downloaded as f32 / total as f32).clamp(0.0, 0.95),
    }
}

fn read_hf_token() -> Result<Option<String>, keyring::Error> {
    let entry = keyring::Entry::new(HF_KEYCHAIN_SERVICE, HF_KEYCHAIN_USER)?;
    match entry.get_password() {
        Ok(token) if token.trim().is_empty() => Ok(None),
        Ok(token) => Ok(Some(token)),
        Err(keyring::Error::NoEntry) => Ok(None),
        Err(error) => Err(error),
    }
}

fn env_hf_token() -> Option<String> {
    std::env::var("HF_TOKEN")
        .ok()
        .or_else(|| std::env::var("HUGGINGFACE_TOKEN").ok())
        .map(|token| token.trim().to_string())
        .filter(|token| !token.is_empty())
}

fn emit_progress(app: &AppHandle, payload: ImportProgress) {
    let _ = app.emit(PROGRESS_EVENT, payload);
}

fn model_id(model_ref: &HfModelRef, quantize: Option<QuantizeRequest>) -> String {
    let suffix = match quantize {
        Some(QuantizeRequest::Q4) => "q4",
        Some(QuantizeRequest::Q8) => "q8",
        None => "dense",
    };
    format!(
        "{}--{}--{}",
        safe_name(&model_ref.repo),
        safe_name(&model_ref.revision),
        suffix
    )
}

fn model_name(model_ref: &HfModelRef, quantize: Option<QuantizeRequest>) -> String {
    let base = model_ref
        .repo
        .split('/')
        .next_back()
        .unwrap_or(model_ref.repo.as_str());
    match quantize {
        Some(QuantizeRequest::Q4) => format!("{base} Q4"),
        Some(QuantizeRequest::Q8) => format!("{base} Q8"),
        None => base.to_string(),
    }
}

fn snapshot_dir_name(model_ref: &HfModelRef) -> String {
    format!(
        "{}--{}",
        safe_name(&model_ref.repo),
        safe_name(&model_ref.revision)
    )
}

fn safe_name(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                ch
            } else {
                '-'
            }
        })
        .collect()
}

fn make_job_id() -> String {
    format!("import-{}", now_secs())
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_huggingface_urls() {
        let parsed = HfModelRef::parse("https://huggingface.co/Qwen/Qwen3-0.6B/tree/main").unwrap();
        assert_eq!(parsed.repo, "Qwen/Qwen3-0.6B");
        assert_eq!(parsed.revision, "main");

        let parsed = HfModelRef::parse("hf://meta-llama/Llama-3.2-1B-Instruct").unwrap();
        assert_eq!(parsed.repo, "meta-llama/Llama-3.2-1B-Instruct");
        assert_eq!(parsed.revision, "main");

        let parsed = HfModelRef::parse("gpt2").unwrap();
        assert_eq!(parsed.repo, "gpt2");
        assert_eq!(parsed.revision, "main");
    }

    #[test]
    fn filters_loadable_files() {
        assert!(is_loadable_model_file("config.json"));
        assert!(is_loadable_model_file("model-00001-of-00002.safetensors"));
        assert!(is_loadable_model_file("model.safetensors.index.json"));
        assert!(!is_loadable_model_file("model.gguf"));
        assert!(!is_loadable_model_file("README.md"));
    }

    #[test]
    fn rejects_unsafe_hf_file_names() {
        assert!(validate_hf_file_name("model.safetensors").is_ok());
        assert!(validate_hf_file_name("subdir/model.safetensors").is_ok());
        assert!(validate_hf_file_name("../model.safetensors").is_err());
        assert!(validate_hf_file_name("subdir\\model.safetensors").is_err());
        assert!(validate_hf_file_name("/tmp/model.safetensors").is_err());
    }

    #[test]
    fn registry_upsert_replaces_existing_entry() {
        let mut registry = ModelRegistry::default();
        let model_ref = HfModelRef::parse("Qwen/Qwen3-0.6B").unwrap();
        let id = model_id(&model_ref, None);
        upsert_model(
            &mut registry,
            ModelEntry {
                id: id.clone(),
                name: "old".to_string(),
                repo: model_ref.repo.clone(),
                revision: model_ref.revision.clone(),
                source_url: model_ref.source_url(),
                local_path: "/tmp/old".to_string(),
                quantize: None,
                imported_at: 1,
                file_count: 3,
                size_bytes: Some(10),
            },
        );
        upsert_model(
            &mut registry,
            ModelEntry {
                id,
                name: "new".to_string(),
                repo: model_ref.repo.clone(),
                revision: model_ref.revision.clone(),
                source_url: model_ref.source_url(),
                local_path: "/tmp/new".to_string(),
                quantize: None,
                imported_at: 2,
                file_count: 4,
                size_bytes: Some(20),
            },
        );
        assert_eq!(registry.models.len(), 1);
        assert_eq!(registry.models[0].name, "new");
        assert_eq!(registry.models[0].file_count, 4);
    }
}
