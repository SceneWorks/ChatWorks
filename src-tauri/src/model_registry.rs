use std::collections::HashSet;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::core_llm::LoadSpec;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter};

use crate::engine::{EngineHandle, EngineStatus, LoadModelRequest, QuantizeRequest};
use crate::fsutil::{now_secs, write_json_atomic};

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
    /// Optional exact companion projector source. A Prism GGUF stays text-only until the user
    /// chooses one; no sibling is ever inferred by filename.
    #[serde(default)]
    pub projector_source: Option<String>,
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
    /// Immutable source encoding recognized by the linked runtime loader.
    #[serde(default = "default_model_format")]
    pub format: String,
    /// Native packed family label, populated only from required pack sidecars (never config naming).
    #[serde(default)]
    pub pack: Option<String>,
    /// Weightless loader route selected by `can_load` for this exact snapshot.
    #[serde(default)]
    pub provider_id: Option<String>,
    /// User-selected, persisted companion projector for a GGUF language file. This is never
    /// populated implicitly from a directory containing several projector variants.
    #[serde(default)]
    pub projector_source: Option<String>,
    /// Valid companion projector choices observed in the same snapshot. Kept for an explicit UI
    /// selection only; `projector_source` remains the one actually loaded.
    #[serde(default)]
    pub projector_sources: Vec<String>,
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

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CachedModelCandidate {
    pub id: String,
    pub name: String,
    pub repo: String,
    pub revision: String,
    pub local_path: String,
    pub provider_id: String,
    pub provider_family: String,
    pub supports_vision: bool,
    pub format: String,
    pub pack: Option<String>,
    pub file_count: usize,
    pub size_bytes: Option<u64>,
    /// Explicit projector artifacts found beside this GGUF. The UI requires the user to choose;
    /// an empty selection means text-only loading.
    #[serde(default)]
    pub projector_sources: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AdoptCachedModelRequest {
    pub local_path: String,
    #[serde(default)]
    pub quantize: Option<QuantizeRequest>,
    #[serde(default)]
    pub projector_source: Option<String>,
}

#[derive(Clone, Debug)]
struct HfModelRef {
    repo: String,
    revision: String,
    /// An exact GGUF file selected by a Hugging Face `blob` / `resolve` URL.
    file_name: Option<String>,
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

pub fn list_cached_hf_models() -> Result<Vec<CachedModelCandidate>, String> {
    let mut candidates = Vec::new();
    let mut seen = HashSet::new();
    for cache_dir in hf_cache_dirs() {
        for snapshot in cached_snapshot_dirs(&cache_dir)? {
            for source in cached_model_sources(&snapshot)? {
                let key = source.to_string_lossy().to_string();
                if !seen.insert(key) {
                    continue;
                }
                if let Some(candidate) = cached_model_candidate(&source)? {
                    candidates.push(candidate);
                }
            }
        }
    }
    candidates.sort_by(|a, b| {
        a.name
            .cmp(&b.name)
            .then_with(|| a.revision.cmp(&b.revision))
    });
    Ok(candidates)
}

pub fn adopt_cached_hf_model(
    app: &AppHandle,
    request: AdoptCachedModelRequest,
) -> Result<ModelRegistry, String> {
    let path = PathBuf::from(request.local_path.trim());
    let candidate = cached_model_candidate(&path)?.ok_or_else(|| {
        "cached snapshot is not supported by the linked inference providers".to_string()
    })?;
    validate_quantize_request(
        &candidate.format,
        candidate.pack.as_deref(),
        request.quantize,
    )?;
    let model_ref = HfModelRef {
        repo: candidate.repo.clone(),
        revision: candidate.revision.clone(),
        file_name: source_file_name(Path::new(&candidate.local_path)),
    };
    let projector_source = validate_projector_source(
        Path::new(&candidate.local_path),
        request.projector_source.as_deref(),
    )?;
    let entry = ModelEntry {
        id: model_id(&model_ref, request.quantize),
        name: model_name(&model_ref, request.quantize),
        repo: candidate.repo,
        revision: candidate.revision,
        source_url: model_ref.source_url(),
        local_path: candidate.local_path,
        quantize: request.quantize,
        imported_at: now_secs(),
        file_count: candidate.file_count,
        size_bytes: candidate.size_bytes,
        format: candidate.format,
        pack: candidate.pack,
        provider_id: Some(candidate.provider_id),
        projector_source: projector_source.clone(),
        projector_sources: candidate.projector_sources,
    };
    let manifest = registry_path(app)?;
    let mut registry = read_registry(&manifest)?;
    upsert_model(&mut registry, entry);
    write_registry(&manifest, &registry)?;
    Ok(registry)
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
    projector_source: Option<String>,
) -> Result<EngineStatus, String> {
    let manifest = registry_path(app)?;
    let mut registry = read_registry(&manifest)?;
    let entry_index = registry
        .models
        .iter()
        .position(|model| model.id == model_id)
        .ok_or_else(|| format!("model {model_id:?} is not in the registry"))?;
    let requested_projector = projector_source.clone();
    let entry = registry.models[entry_index].clone();
    let snapshot = Path::new(&entry.local_path);
    validate_model_source(snapshot)?;
    recognized_model_format(snapshot)?;
    if matching_provider(snapshot)?.is_none() {
        return Err(unsupported_model_source_error(
            snapshot,
            "registered model is no longer supported",
        ));
    }
    let projector_source = validate_projector_source(
        snapshot,
        requested_projector
            .as_deref()
            .or(entry.projector_source.as_deref()),
    )?;
    let status = engine.load_model(LoadModelRequest {
        source: entry.local_path.clone(),
        display_name: Some(entry.name.clone()),
        quantize: entry.quantize,
        projector_source: projector_source.clone(),
    })?;
    // A present empty string from the picker is an intentional "Text only" selection. Keep it
    // distinct from an omitted command argument, which preserves an existing association.
    if requested_projector.is_some() {
        registry.models[entry_index].projector_source = projector_source;
    }
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
    let entry = crate::profile::credential(HF_KEYCHAIN_SERVICE, HF_KEYCHAIN_USER)
        .map_err(|error| error.to_string())?;
    entry
        .set_password(token)
        .map_err(|error| error.to_string())?;
    Ok(hf_token_status())
}

pub fn clear_hf_token() -> Result<HfTokenStatus, String> {
    let entry = crate::profile::credential(HF_KEYCHAIN_SERVICE, HF_KEYCHAIN_USER)
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
    let projector_ref = parse_projector_ref(&model_ref, request.projector_source.as_deref())?;
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

    let files = fetch_hf_files(
        &client,
        &model_ref,
        projector_ref.as_ref(),
        token.as_deref(),
    )
    .await?;
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
            validate_config_file_if_available(&snapshot_dir, &file.rfilename)?;
            continue;
        }
        download
            .download_file(file, &target, &mut downloaded_bytes)
            .await?;
        validate_config_file_if_available(&snapshot_dir, &file.rfilename)?;
    }

    emit_progress(
        app,
        ImportProgress {
            job_id: job_id.to_string(),
            stage: "convert".to_string(),
            message: "Preparing model snapshot".to_string(),
            progress: 0.96,
            downloaded_bytes,
            total_bytes,
        },
    );
    let source = imported_model_source(&snapshot_dir, &model_ref)?;
    validate_model_source(&source)?;
    let provider = matching_provider(&source)?.ok_or_else(|| {
        unsupported_model_source_error(&source, "downloaded model is not supported")
    })?;
    let (format, pack) = recognized_model_format(&source)?;
    validate_quantize_request(&format, pack.as_deref(), request.quantize)?;
    let projector_source = projector_ref
        .as_ref()
        .and_then(|projector| projector.file_name.as_deref())
        .map(|name| snapshot_dir.join(name));
    let projector_source = validate_projector_source(
        &source,
        projector_source.as_ref().and_then(|path| path.to_str()),
    )?;

    let mut registry = read_registry(&manifest)?;
    let entry = ModelEntry {
        id: model_id(&model_ref, request.quantize),
        name: model_name(&model_ref, request.quantize),
        repo: model_ref.repo.clone(),
        revision: model_ref.revision.clone(),
        source_url: model_ref.source_url(),
        local_path: source.to_string_lossy().to_string(),
        quantize: request.quantize,
        imported_at: now_secs(),
        file_count: files.len(),
        size_bytes: total_bytes,
        format,
        pack,
        provider_id: Some(provider.id),
        projector_source,
        projector_sources: if source.is_file() {
            sibling_projector_sources(&source)?
        } else {
            Vec::new()
        },
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
    projector_ref: Option<&HfModelRef>,
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
    select_import_files(files, model_ref, projector_ref)
}

/// Import one direct GGUF at a time. This prevents a repo that publishes PQ2, PTQ1, and projector
/// files from silently downloading every variant or loading an arbitrary one.
fn select_import_files(
    files: Vec<HfSibling>,
    model_ref: &HfModelRef,
    projector_ref: Option<&HfModelRef>,
) -> Result<Vec<HfSibling>, String> {
    if let Some(file_name) = &model_ref.file_name {
        if !is_gguf_model_file(file_name) {
            return Err(
                "file-specific HuggingFace imports require a non-projector .gguf model file"
                    .to_string(),
            );
        }
        let model = files
            .iter()
            .find(|file| file.rfilename == *file_name)
            .cloned()
            .ok_or_else(|| format!("HuggingFace revision does not contain {file_name:?}"))?;
        let mut selected = vec![model];
        if let Some(projector_name) = projector_ref.and_then(|value| value.file_name.as_deref()) {
            let projector = files
                .iter()
                .find(|file| file.rfilename == projector_name)
                .cloned()
                .ok_or_else(|| {
                    format!("HuggingFace revision does not contain {projector_name:?}")
                })?;
            selected.push(projector);
        }
        return Ok(selected);
    }

    let has_safetensors = files
        .iter()
        .any(|file| file.rfilename.ends_with(".safetensors"));
    if has_safetensors {
        return Ok(files
            .into_iter()
            .filter(|file| !file.rfilename.ends_with(".gguf"))
            .collect());
    }

    let gguf_files: Vec<_> = files
        .into_iter()
        .filter(|file| is_gguf_model_file(&file.rfilename))
        .collect();
    match gguf_files.as_slice() {
        [file] => Ok(vec![file.clone()]),
        [] => Ok(Vec::new()),
        files => Err(format!(
            "HuggingFace repo publishes multiple GGUF model files ({}); import one using its exact https://huggingface.co/<repo>/blob/<revision>/<file>.gguf URL",
            files
                .iter()
                .map(|file| file.rfilename.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

fn parse_projector_ref(
    model_ref: &HfModelRef,
    requested: Option<&str>,
) -> Result<Option<HfModelRef>, String> {
    let Some(requested) = requested.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(None);
    };
    if model_ref.file_name.is_none() {
        return Err(
            "a projector can only accompany an exact HuggingFace GGUF language-file URL"
                .to_string(),
        );
    }
    let projector = HfModelRef::parse(requested)?;
    let projector_name = projector.file_name.as_deref().ok_or_else(|| {
        "projector source must be an exact HuggingFace blob/resolve .gguf URL".to_string()
    })?;
    if !is_projector_file(projector_name) {
        return Err("projector source must name an mmproj .gguf artifact".to_string());
    }
    if projector.repo != model_ref.repo || projector.revision != model_ref.revision {
        return Err(
            "projector must come from the same HuggingFace repository and revision as the language GGUF"
                .to_string(),
        );
    }
    Ok(Some(projector))
}

fn validate_quantize_request(
    format: &str,
    pack: Option<&str>,
    quantize: Option<QuantizeRequest>,
) -> Result<(), String> {
    if quantize.is_some() && (format == "gguf-prism-packed" || pack == Some("bonsai2-packed")) {
        return Err(
            "packed Bonsai/Prism artifacts cannot be quantized again; choose Native packed"
                .to_string(),
        );
    }
    Ok(())
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
        let (repo_segments, revision, file_name) = if let Some(index) = marker {
            if index == 0 {
                return Err("HuggingFace repo path is required".to_string());
            }
            let revision = segments.get(index + 1).copied().unwrap_or("main");
            let file_name = match segments[index] {
                "blob" | "resolve" if segments.len() > index + 2 => {
                    Some(segments[index + 2..].join("/"))
                }
                _ => None,
            };
            (&segments[..index], revision, file_name)
        } else if segments.len() >= 2 {
            (&segments[..2], "main", None)
        } else {
            (&segments[..1], "main", None)
        };
        let repo = repo_segments.join("/");
        validate_hf_path(&repo, "repo")?;
        validate_hf_path(revision, "revision")?;
        if let Some(file_name) = &file_name {
            validate_hf_file_name(file_name)?;
        }
        Ok(Self {
            repo,
            revision: revision.to_string(),
            file_name,
        })
    }

    fn source_url(&self) -> String {
        match &self.file_name {
            Some(file_name) => format!(
                "https://{}/{}/blob/{}/{}",
                HF_HOST, self.repo, self.revision, file_name
            ),
            None => format!("https://{}/{}/tree/{}", HF_HOST, self.repo, self.revision),
        }
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
        || name == "hadamard.json"
        || name == "PACK-RUNTIME"
        || name.ends_with(".safetensors")
        || name.ends_with(".safetensors.index.json")
        || name.ends_with(".gguf")
}

fn is_gguf_model_file(name: &str) -> bool {
    name.ends_with(".gguf") && !name.to_ascii_lowercase().contains("mmproj")
}

fn is_projector_file(name: &str) -> bool {
    name.ends_with(".gguf") && name.to_ascii_lowercase().contains("mmproj")
}

/// Projectors are a user choice: accept only an explicit GGUF `mmproj` artifact from the same
/// snapshot directory as a direct language GGUF. This rejects arbitrary path injection while
/// retaining both published BF16 and Q8 variants for an intentional selection.
fn validate_projector_source(
    model: &Path,
    requested: Option<&str>,
) -> Result<Option<String>, String> {
    let Some(requested) = requested.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(None);
    };
    if !model.is_file() {
        return Err("a projector can only be paired with a direct GGUF language file".to_string());
    }
    let projector = Path::new(requested);
    if !projector.is_file() || !is_projector_file(&projector.to_string_lossy()) {
        return Err("projector must be an existing mmproj .gguf file".to_string());
    }
    if projector.parent() != model.parent() {
        return Err(
            "projector must come from the same snapshot directory as its GGUF language model"
                .to_string(),
        );
    }
    let mut header = [0_u8; 4];
    std::io::Read::read_exact(
        &mut fs::File::open(projector).map_err(|error| error.to_string())?,
        &mut header,
    )
    .map_err(|error| error.to_string())?;
    if header != *b"GGUF" {
        return Err("projector GGUF has an invalid header or is corrupt".to_string());
    }
    Ok(Some(projector.to_string_lossy().to_string()))
}

fn sibling_projector_sources(model: &Path) -> Result<Vec<String>, String> {
    let Some(directory) = model.parent() else {
        return Ok(Vec::new());
    };
    let mut sources = fs::read_dir(directory)
        .map_err(|error| error.to_string())?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_file() && is_projector_file(&path.to_string_lossy()))
        .filter_map(|path| validate_projector_source(model, path.to_str()).ok())
        .flatten()
        .collect::<Vec<_>>();
    sources.sort();
    Ok(sources)
}

fn source_file_name(path: &Path) -> Option<String> {
    path.is_file()
        .then(|| path.file_name()?.to_str().map(str::to_string))
        .flatten()
}

fn default_model_format() -> String {
    "hf-safetensors".to_string()
}

/// Labels a Prism/Bonsai pack only when its immutable pack sidecars are present. Model config
/// strings remain insufficient: the linked loader's `can_load` decides support separately.
fn recognized_model_format(path: &Path) -> Result<(String, Option<String>), String> {
    if path.is_file() {
        if !is_gguf_model_file(&path.to_string_lossy()) {
            return Err("direct model source must be a non-projector .gguf file".to_string());
        }
        return Ok((
            "gguf-prism-packed".to_string(),
            Some("bonsai2-packed".to_string()),
        ));
    }
    let hadamard = path.join("hadamard.json").is_file();
    let runtime = path.join("PACK-RUNTIME").is_file();
    match (hadamard, runtime) {
        (false, false) => Ok((default_model_format(), None)),
        (true, true) => Ok(("hf-safetensors".to_string(), Some("bonsai2-packed".to_string()))),
        _ => Err("Prism/Bonsai snapshot has incomplete packing sidecars (need hadamard.json and PACK-RUNTIME)".to_string()),
    }
}

fn imported_model_source(snapshot_dir: &Path, model_ref: &HfModelRef) -> Result<PathBuf, String> {
    if let Some(file_name) = &model_ref.file_name {
        let source = snapshot_dir.join(file_name);
        if !source.is_file() {
            return Err(format!("downloaded GGUF file {file_name:?} is missing"));
        }
        return Ok(source);
    }

    let gguf_files: Vec<_> = fs::read_dir(snapshot_dir)
        .map_err(|error| error.to_string())?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_file() && is_gguf_model_file(&path.to_string_lossy()))
        .collect();
    match gguf_files.as_slice() {
        [source] => Ok(source.clone()),
        [] => Ok(snapshot_dir.to_path_buf()),
        // `select_import_files` rejects this before any download. Keep this guard so a partially
        // populated cache cannot pick an arbitrary quantization variant on retry.
        _ => Err(
            "downloaded snapshot has multiple GGUF model files; use an exact blob URL".to_string(),
        ),
    }
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

fn validate_model_source(path: &Path) -> Result<(), String> {
    if path.is_file() {
        if !is_gguf_model_file(&path.to_string_lossy()) {
            return Err("direct model source must be a non-projector .gguf file".to_string());
        }
        let size = fs::metadata(path).map_err(|error| error.to_string())?.len();
        if size < 4 {
            return Err("GGUF model file is empty or corrupt".to_string());
        }
        let mut header = [0_u8; 4];
        std::io::Read::read_exact(
            &mut fs::File::open(path).map_err(|error| error.to_string())?,
            &mut header,
        )
        .map_err(|error| error.to_string())?;
        if header != *b"GGUF" {
            return Err("GGUF model file has an invalid header or is corrupt".to_string());
        }
        return Ok(());
    }
    validate_snapshot(path)
}

fn unsupported_model_source_error(path: &Path, prefix: &str) -> String {
    if path.is_file() && is_gguf_model_file(&path.to_string_lossy()) {
        format!("{prefix}: GGUF must be a complete Qwen3.5 Prism/Bonsai model with valid packed metadata")
    } else {
        format!("{prefix} by the linked inference providers")
    }
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
    validate_text_snapshot_config(path)
}

fn validate_config_file_if_available(snapshot_dir: &Path, file_name: &str) -> Result<(), String> {
    if file_name == "config.json" {
        validate_text_snapshot_config(snapshot_dir)?;
    }
    Ok(())
}

fn validate_text_snapshot_config(path: &Path) -> Result<(), String> {
    let config_path = path.join("config.json");
    let body = fs::read_to_string(config_path).map_err(|error| error.to_string())?;
    let config =
        serde_json::from_str::<serde_json::Value>(&body).map_err(|error| error.to_string())?;
    // A `vision_config` wrapper is supported when a linked provider can serve it: LLaVA/JoyCaption,
    // Qwen3.6 (`qwen3_5`) which the `mlx-llama` provider now handles as a full VLM (sc-7633), or
    // Qwen3-VL (`qwen3_vl`) which the `mlx-llama` provider also serves (sc-8078).
    // Anything else multimodal is still rejected with the clear error (sc-7618).
    if config.get("vision_config").is_some()
        && !is_joycaption_config(&config)
        && !is_qwen35_vision_config(&config)
        && !is_qwen3vl_vision_config(&config)
    {
        let model_type = config
            .get("model_type")
            .and_then(|value| value.as_str())
            .unwrap_or("unknown");
        return Err(format!(
            "unsupported model type {model_type}: this multimodal/VLM snapshot is not supported by the linked inference providers"
        ));
    }
    Ok(())
}

fn cached_model_sources(snapshot: &Path) -> Result<Vec<PathBuf>, String> {
    let mut sources = Vec::new();
    // A safetensors snapshot remains one directory source. GGUF variants are independently
    // selectable files, excluding companion multimodal projectors.
    let has_safetensors = fs::read_dir(snapshot)
        .map_err(|error| error.to_string())?
        .filter_map(Result::ok)
        .any(|entry| entry.path().extension().and_then(|ext| ext.to_str()) == Some("safetensors"));
    if has_safetensors {
        sources.push(snapshot.to_path_buf());
    }
    for entry in fs::read_dir(snapshot).map_err(|error| error.to_string())? {
        let path = entry.map_err(|error| error.to_string())?.path();
        if path.is_file() && is_gguf_model_file(&path.to_string_lossy()) {
            sources.push(path);
        }
    }
    Ok(sources)
}

fn cached_model_candidate(path: &Path) -> Result<Option<CachedModelCandidate>, String> {
    if let Err(error) = validate_model_source(path) {
        if error.starts_with("unsupported model type ") {
            return Ok(None);
        }
        return Err(error);
    }
    let Some((repo, revision)) = parse_hf_cache_snapshot(path) else {
        return Ok(None);
    };
    let Some(provider) = matching_provider(path)? else {
        return Ok(None);
    };
    if path.is_dir() && provider.capabilities.supports_vision && !is_joycaption_snapshot(path)? {
        return Ok(None);
    }
    let model_ref = HfModelRef {
        repo,
        revision,
        file_name: source_file_name(path),
    };
    let file_count = snapshot_file_count(path)?;
    let size_bytes = snapshot_size_bytes(path);
    let (format, pack) = recognized_model_format(path)?;
    Ok(Some(CachedModelCandidate {
        id: model_id(&model_ref, None),
        name: model_name(&model_ref, None),
        repo: model_ref.repo,
        revision: model_ref.revision,
        local_path: path.to_string_lossy().to_string(),
        provider_id: provider.id,
        provider_family: provider.family,
        // The static provider descriptor is weightless, so the `mlx-llama` Qwen3.6/Qwen3-VL VLM
        // advertises vision only once loaded; detect it from the snapshot config so the UI offers
        // image input before load.
        supports_vision: path.is_dir()
            && (provider.capabilities.supports_vision
                || is_qwen35_vision_snapshot(path)?
                || is_qwen3vl_vision_snapshot(path)?),
        format,
        pack,
        file_count,
        size_bytes,
        projector_sources: if path.is_file() {
            sibling_projector_sources(path)?
        } else {
            Vec::new()
        },
    }))
}

fn matching_provider(path: &Path) -> Result<Option<crate::core_llm::TextLlmDescriptor>, String> {
    let source = path.to_string_lossy().to_string();
    let spec = LoadSpec {
        source,
        projector_source: None,
        quantize: None,
    };
    Ok(crate::inference_runtime::textllms()
        .find(|registration| (registration.can_load)(&spec))
        .map(|registration| (registration.descriptor)()))
}

fn is_joycaption_snapshot(path: &Path) -> Result<bool, String> {
    let body = fs::read_to_string(path.join("config.json")).map_err(|error| error.to_string())?;
    let config =
        serde_json::from_str::<serde_json::Value>(&body).map_err(|error| error.to_string())?;
    Ok(is_joycaption_config(&config))
}

fn is_joycaption_config(config: &serde_json::Value) -> bool {
    let architecture = config
        .get("architectures")
        .and_then(|value| value.as_array())
        .and_then(|items| items.first())
        .and_then(|value| value.as_str())
        .unwrap_or_default()
        .to_lowercase();
    let model_type = config
        .get("model_type")
        .and_then(|value| value.as_str())
        .unwrap_or_default()
        .to_lowercase();
    architecture.contains("llava") || model_type.contains("llava")
}

/// Whether `path` is a Qwen3.6 (`qwen3_5`) vision wrapper — a `vision_config` carried alongside the
/// `qwen3_5` text decoder, which `mlx-llama` serves as a full VLM (sc-7633).
fn is_qwen35_vision_snapshot(path: &Path) -> Result<bool, String> {
    let body = fs::read_to_string(path.join("config.json")).map_err(|error| error.to_string())?;
    let config =
        serde_json::from_str::<serde_json::Value>(&body).map_err(|error| error.to_string())?;
    Ok(is_qwen35_vision_config(&config))
}

fn is_qwen35_vision_config(config: &serde_json::Value) -> bool {
    let model_type = config
        .get("model_type")
        .and_then(|value| value.as_str())
        .unwrap_or_default()
        .to_lowercase();
    model_type.starts_with("qwen3_5") && config.get("vision_config").is_some()
}

/// Whether `path` is a Qwen3-VL (`qwen3_vl`) vision wrapper — a `vision_config` carried alongside
/// the `qwen3_vl` text decoder, which `mlx-llama` serves as a full VLM (sc-8078).
fn is_qwen3vl_vision_snapshot(path: &Path) -> Result<bool, String> {
    let body = fs::read_to_string(path.join("config.json")).map_err(|error| error.to_string())?;
    let config =
        serde_json::from_str::<serde_json::Value>(&body).map_err(|error| error.to_string())?;
    Ok(is_qwen3vl_vision_config(&config))
}

fn is_qwen3vl_vision_config(config: &serde_json::Value) -> bool {
    // Gate on the top-level `model_type` (`qwen3_vl`), where `vision_config` sits. The nested
    // `text_config.model_type` is `qwen3_vl_text` (also `starts_with "qwen3_vl"`), but the relevant
    // gate is the top-level config that carries the `vision_config` wrapper.
    let model_type = config
        .get("model_type")
        .and_then(|value| value.as_str())
        .unwrap_or_default()
        .to_lowercase();
    model_type.starts_with("qwen3_vl") && config.get("vision_config").is_some()
}

fn hf_cache_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(value) = env_path("HUGGINGFACE_HUB_CACHE") {
        dirs.push(value);
    }
    if let Some(value) = env_path("HF_HOME") {
        dirs.push(value.join("hub"));
    }
    dirs.push(home_dir().join(".cache").join("huggingface").join("hub"));
    dedupe_paths(dirs)
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .map(PathBuf::from)
        .filter(|path| !path.as_os_str().is_empty())
}

fn home_dir() -> PathBuf {
    // `HOME` on macOS/Linux; `USERPROFILE` on Windows, where the HuggingFace cache lives under
    // `%USERPROFILE%\.cache\huggingface\hub` (huggingface_hub resolves `~` the same way).
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

fn dedupe_paths(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut seen = HashSet::new();
    paths
        .into_iter()
        .filter(|path| seen.insert(path.to_string_lossy().to_string()))
        .collect()
}

fn cached_snapshot_dirs(cache_dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut snapshots = Vec::new();
    if !cache_dir.is_dir() {
        return Ok(snapshots);
    }
    for model_dir in fs::read_dir(cache_dir).map_err(|error| error.to_string())? {
        let model_dir = model_dir.map_err(|error| error.to_string())?.path();
        if !model_dir
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("models--"))
        {
            continue;
        }
        let snapshots_dir = model_dir.join("snapshots");
        if !snapshots_dir.is_dir() {
            continue;
        }
        for snapshot in fs::read_dir(snapshots_dir).map_err(|error| error.to_string())? {
            let snapshot = snapshot.map_err(|error| error.to_string())?.path();
            if snapshot.is_dir() {
                snapshots.push(snapshot);
            }
        }
    }
    Ok(snapshots)
}

fn parse_hf_cache_snapshot(path: &Path) -> Option<(String, String)> {
    let snapshot = if path.is_file() { path.parent()? } else { path };
    let revision = snapshot.file_name()?.to_str()?.to_string();
    let snapshots_dir = snapshot.parent()?;
    if snapshots_dir.file_name()?.to_str()? != "snapshots" {
        return None;
    }
    let model_dir = snapshots_dir.parent()?;
    let encoded = model_dir.file_name()?.to_str()?.strip_prefix("models--")?;
    let repo = encoded.replace("--", "/");
    Some((repo, revision))
}

fn snapshot_file_count(path: &Path) -> Result<usize, String> {
    if path.is_file() {
        return Ok(1);
    }
    Ok(fs::read_dir(path)
        .map_err(|error| error.to_string())?
        .filter_map(Result::ok)
        .count())
}

fn snapshot_size_bytes(path: &Path) -> Option<u64> {
    if path.is_file() {
        return path.metadata().ok().map(|metadata| metadata.len());
    }
    let mut total = 0_u64;
    for entry in fs::read_dir(path).ok()?.filter_map(Result::ok) {
        let metadata = entry.metadata().ok()?;
        if metadata.is_file() {
            total = total.checked_add(metadata.len())?;
        }
    }
    Some(total)
}

fn app_data_dir(app: &AppHandle) -> Result<PathBuf, String> {
    crate::profile::data_dir(app)
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
    // Delegates to the shared atomic-write helper (code-review F-010). The previous local copy used
    // `with_extension("json.tmp")` (which replaces an extension) rather than appending `.tmp`; the
    // shared helper appends `.tmp` preserving the extension, and for `manifest.json` both land on
    // `manifest.json.tmp`, so the on-disk temp name is unchanged here.
    write_json_atomic(path, registry)
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
    let entry = crate::profile::credential(HF_KEYCHAIN_SERVICE, HF_KEYCHAIN_USER)?;
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
    let base = format!(
        "{}--{}--{}",
        safe_name(&model_ref.repo),
        safe_name(&model_ref.revision),
        suffix
    );
    match model_ref.file_name.as_deref() {
        Some(file_name) => format!("{base}--{}", safe_name(file_name)),
        None => base,
    }
}

fn model_name(model_ref: &HfModelRef, quantize: Option<QuantizeRequest>) -> String {
    let base = model_ref
        .repo
        .split('/')
        .next_back()
        .unwrap_or(model_ref.repo.as_str());
    let source = model_ref
        .file_name
        .as_deref()
        .and_then(|name| Path::new(name).file_name())
        .and_then(|name| name.to_str())
        .map(|name| format!(" ({name})"))
        .unwrap_or_default();
    match quantize {
        Some(QuantizeRequest::Q4) => format!("{base}{source} Q4"),
        Some(QuantizeRequest::Q8) => format!("{base}{source} Q8"),
        None => format!("{base}{source}"),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fsutil::TempDir;

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
        assert!(is_loadable_model_file("model.gguf"));
        assert!(!is_loadable_model_file("README.md"));
    }

    #[test]
    fn recognizes_direct_prism_gguf_and_rejects_projectors() {
        assert!(is_loadable_model_file("Ternary-Bonsai-2-27B-PQ2_0.gguf"));
        assert!(is_gguf_model_file("Ternary-Bonsai-2-27B-PTQ1_0.gguf"));
        assert!(!is_gguf_model_file("mmproj-Ternary-Bonsai-2-27B-F16.gguf"));

        let dir = snapshot_dir("prism-gguf");
        let source = dir.path().join("Ternary-Bonsai-2-27B-PQ2_0.gguf");
        write_minimal_prism_gguf(&source);
        assert!(validate_model_source(&source).is_ok());
        assert_eq!(
            recognized_model_format(&source).unwrap(),
            (
                "gguf-prism-packed".to_string(),
                Some("bonsai2-packed".to_string())
            )
        );
    }

    #[test]
    fn direct_gguf_requires_valid_header() {
        let dir = snapshot_dir("invalid-gguf");
        let source = dir.path().join("broken.gguf");
        fs::write(&source, b"bad!").unwrap();
        assert_eq!(
            validate_model_source(&source).unwrap_err(),
            "GGUF model file has an invalid header or is corrupt"
        );
    }

    #[test]
    fn parses_and_selects_exact_gguf_huggingface_file() {
        let model = HfModelRef::parse(
            "https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/blob/6ed5e12/PQ2_0.gguf",
        )
        .unwrap();
        assert_eq!(model.file_name.as_deref(), Some("PQ2_0.gguf"));
        assert!(model.source_url().ends_with("/blob/6ed5e12/PQ2_0.gguf"));

        let files = vec![
            HfSibling {
                rfilename: "PQ2_0.gguf".to_string(),
                size: Some(1),
            },
            HfSibling {
                rfilename: "PTQ1_0.gguf".to_string(),
                size: Some(1),
            },
            HfSibling {
                rfilename: "mmproj-F16.gguf".to_string(),
                size: Some(1),
            },
        ];
        let projector = parse_projector_ref(
            &model,
            Some("https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/blob/6ed5e12/mmproj-F16.gguf"),
        )
        .unwrap();
        let selected = select_import_files(files.clone(), &model, projector.as_ref()).unwrap();
        assert_eq!(
            selected
                .iter()
                .map(|file| file.rfilename.as_str())
                .collect::<Vec<_>>(),
            vec!["PQ2_0.gguf", "mmproj-F16.gguf"]
        );
        let text_only = select_import_files(files.clone(), &model, None).unwrap();
        assert_eq!(text_only.len(), 1);
        assert_eq!(selected.len(), 2);
        let repo = HfModelRef::parse("prism-ml/Ternary-Bonsai-2-27B-gguf").unwrap();
        assert!(select_import_files(files, &repo, None)
            .unwrap_err()
            .contains("multiple GGUF"));
    }

    #[test]
    fn projector_import_requires_one_explicit_same_revision_artifact() {
        let model =
            HfModelRef::parse("https://huggingface.co/prism/model/blob/rev-a/PQ2_0.gguf").unwrap();
        assert!(parse_projector_ref(&model, None).unwrap().is_none());
        assert!(parse_projector_ref(
            &model,
            Some("https://huggingface.co/prism/model/blob/rev-a/mmproj-Q8.gguf")
        )
        .is_ok());
        assert!(parse_projector_ref(
            &model,
            Some("https://huggingface.co/prism/model/blob/rev-b/mmproj-Q8.gguf")
        )
        .is_err());
        assert!(parse_projector_ref(
            &model,
            Some("https://huggingface.co/prism/model/blob/rev-a/PTQ1_0.gguf")
        )
        .is_err());
    }

    #[test]
    fn packed_bonsai_artifacts_reject_requantization() {
        assert!(validate_quantize_request(
            "gguf-prism-packed",
            Some("bonsai2-packed"),
            Some(QuantizeRequest::Q4)
        )
        .is_err());
        assert!(validate_quantize_request(
            "hf-safetensors",
            Some("bonsai2-packed"),
            Some(QuantizeRequest::Q8)
        )
        .is_err());
        assert!(
            validate_quantize_request("gguf-prism-packed", Some("bonsai2-packed"), None).is_ok()
        );
    }

    #[test]
    fn generic_single_gguf_snapshot_loads_the_file_not_its_directory() {
        let dir = snapshot_dir("single-gguf-source");
        let source = dir.path().join("PQ2_0.gguf");
        write_minimal_prism_gguf(&source);
        let model = HfModelRef::parse("prism-ml/Ternary-Bonsai-2-27B-gguf").unwrap();
        assert_eq!(imported_model_source(dir.path(), &model).unwrap(), source);
    }

    #[test]
    fn cached_prism_gguf_is_a_separate_selectable_source() {
        let root = snapshot_dir("hf-prism-gguf");
        let snapshot = root
            .join("models--prism-ml--Ternary-Bonsai-2-27B-gguf")
            .join("snapshots")
            .join("6ed5e12");
        fs::create_dir_all(&snapshot).unwrap();
        let pq2 = snapshot.join("PQ2_0.gguf");
        write_minimal_prism_gguf(&pq2);
        write_minimal_prism_gguf(&snapshot.join("mmproj-F16.gguf"));
        write_minimal_prism_gguf(&snapshot.join("mmproj-Q8.gguf"));
        let sources = cached_model_sources(&snapshot).unwrap();
        assert_eq!(sources, vec![pq2.clone()]);
        let candidate = cached_model_candidate(&pq2)
            .unwrap()
            .expect("native GGUF provider");
        assert_eq!(candidate.format, "gguf-prism-packed");
        assert_eq!(candidate.pack.as_deref(), Some("bonsai2-packed"));
        assert_eq!(candidate.file_count, 1);
        assert_eq!(candidate.local_path, pq2.to_string_lossy());
        assert_eq!(candidate.projector_sources.len(), 2);
    }

    #[test]
    fn projector_must_be_an_explicit_same_snapshot_gguf() {
        let dir = snapshot_dir("projector-association");
        let model = dir.path().join("PQ2_0.gguf");
        let projector = dir.path().join("mmproj-F16.gguf");
        write_minimal_prism_gguf(&model);
        write_minimal_prism_gguf(&projector);
        assert_eq!(validate_projector_source(&model, None).unwrap(), None);
        assert_eq!(
            validate_projector_source(&model, projector.to_str()).unwrap(),
            Some(projector.to_string_lossy().to_string())
        );
        assert!(validate_projector_source(&model, Some("/tmp/not-a-projector.gguf")).is_err());
    }

    #[test]
    fn recognizes_only_complete_bonsai_packing_sidecars() {
        let dir = snapshot_dir("bonsai-sidecars");
        assert_eq!(
            recognized_model_format(dir.path()).unwrap(),
            ("hf-safetensors".to_string(), None)
        );
        write_snapshot_file(&dir, "hadamard.json", "{}");
        assert!(recognized_model_format(dir.path()).is_err());
        write_snapshot_file(&dir, "PACK-RUNTIME", "ptq1");
        assert_eq!(
            recognized_model_format(dir.path()).unwrap().1.as_deref(),
            Some("bonsai2-packed")
        );
        assert!(is_loadable_model_file("hadamard.json"));
        assert!(is_loadable_model_file("PACK-RUNTIME"));
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
    fn accepts_text_only_snapshot_config() {
        let dir = snapshot_dir("text-only");
        write_snapshot_file(
            &dir,
            "config.json",
            r#"{"model_type":"qwen3","hidden_size":8}"#,
        );
        write_snapshot_file(&dir, "tokenizer.json", "{}");
        write_snapshot_file(&dir, "model.safetensors", "weights");

        assert!(validate_snapshot(dir.path()).is_ok());
    }

    #[test]
    fn rejects_unsupported_multimodal_snapshot_config() {
        // A VLM the linked providers don't serve (not LLaVA/JoyCaption, not Qwen3.6) is still
        // rejected with the clear error (sc-7618).
        let dir = snapshot_dir("multimodal");
        write_snapshot_file(
            &dir,
            "config.json",
            r#"{"model_type":"gemma3","text_config":{"model_type":"gemma3_text"},"vision_config":{"model_type":"siglip"}}"#,
        );
        write_snapshot_file(&dir, "tokenizer.json", "{}");
        write_snapshot_file(&dir, "model.safetensors", "weights");

        let error = validate_snapshot(dir.path()).unwrap_err();
        assert!(error.contains("unsupported model type gemma3"));
        assert!(error.contains("not supported by the linked inference providers"));
    }

    #[test]
    fn accepts_qwen35_vision_snapshot_config() {
        // Qwen3.6 (`qwen3_5`) carries a `vision_config`; the `mlx-llama` provider now serves it as a
        // full VLM (sc-7633), so validation must accept it (no longer the sc-7618 reject).
        let dir = snapshot_dir("qwen35-vision");
        write_snapshot_file(
            &dir,
            "config.json",
            r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5","text_config":{"model_type":"qwen3_5_text"},"vision_config":{"model_type":"qwen3_5"}}"#,
        );
        write_snapshot_file(&dir, "tokenizer.json", "{}");
        write_snapshot_file(&dir, "model.safetensors", "weights");

        assert!(validate_snapshot(dir.path()).is_ok());
        assert!(is_qwen35_vision_snapshot(dir.path()).unwrap());
    }

    #[test]
    fn accepts_qwen3vl_vision_snapshot_config() {
        // Qwen3-VL (`qwen3_vl`) carries a `vision_config`; the `mlx-llama` provider serves it as a
        // full VLM (sc-8078), so validation must accept it (no longer the sc-7618 reject). The
        // fixture mirrors the real `Qwen/Qwen3-VL-8B-Instruct` config (rev 0c351dd0): top-level
        // `model_type` `qwen3_vl`, `text_config.model_type` `qwen3_vl_text`, `vision_config` present.
        let dir = snapshot_dir("qwen3vl-vision");
        write_snapshot_file(
            &dir,
            "config.json",
            r#"{"architectures":["Qwen3VLForConditionalGeneration"],"model_type":"qwen3_vl","text_config":{"model_type":"qwen3_vl_text"},"vision_config":{"model_type":"qwen3_vl"}}"#,
        );
        write_snapshot_file(&dir, "tokenizer.json", "{}");
        write_snapshot_file(&dir, "model.safetensors", "weights");

        assert!(validate_snapshot(dir.path()).is_ok());
        assert!(is_qwen3vl_vision_snapshot(dir.path()).unwrap());
    }

    #[test]
    fn rejects_unsupported_vlm_despite_qwen3vl_text_config() {
        // A generic VLM whose top-level `model_type` is unknown is still rejected even if its
        // `vision_config` is present — the gate keys off the top-level `model_type` (sc-8078),
        // not the nested text config.
        let dir = snapshot_dir("unknown-vlm");
        write_snapshot_file(
            &dir,
            "config.json",
            r#"{"architectures":["SomeVlmForConditionalGeneration"],"model_type":"some_vlm","text_config":{"model_type":"qwen3_vl_text"},"vision_config":{"model_type":"siglip"}}"#,
        );
        write_snapshot_file(&dir, "tokenizer.json", "{}");
        write_snapshot_file(&dir, "model.safetensors", "weights");

        let error = validate_snapshot(dir.path()).unwrap_err();
        assert!(error.contains("unsupported model type some_vlm"));
        assert!(error.contains("not supported by the linked inference providers"));
        assert!(!is_qwen3vl_vision_snapshot(dir.path()).unwrap());
    }

    #[test]
    fn accepts_joycaption_snapshot_config() {
        let dir = snapshot_dir("joycaption");
        write_snapshot_file(
            &dir,
            "config.json",
            r#"{"architectures":["LlavaForConditionalGeneration"],"model_type":"llava","text_config":{"model_type":"llama","hidden_size":8},"vision_config":{"hidden_size":8}}"#,
        );
        write_snapshot_file(&dir, "tokenizer.json", "{}");
        write_snapshot_file(&dir, "model.safetensors", "weights");

        assert!(validate_snapshot(dir.path()).is_ok());
    }

    #[test]
    fn parses_hf_cache_snapshot_paths() {
        let path = PathBuf::from(
            "/tmp/hub/models--fancyfeast--llama-joycaption-beta-one-hf-llava/snapshots/abc123",
        );
        let (repo, revision) = parse_hf_cache_snapshot(&path).unwrap();
        assert_eq!(repo, "fancyfeast/llama-joycaption-beta-one-hf-llava");
        assert_eq!(revision, "abc123");
        assert!(parse_hf_cache_snapshot(Path::new("/tmp/not-a-snapshot")).is_none());
    }

    #[test]
    fn finds_cached_snapshot_dirs() {
        let root = snapshot_dir("hf-cache");
        let snapshot = root
            .join("models--Qwen--Qwen3-0.6B")
            .join("snapshots")
            .join("rev1");
        fs::create_dir_all(&snapshot).unwrap();
        fs::create_dir_all(root.join("not-a-model").join("snapshots").join("ignored")).unwrap();

        let found = cached_snapshot_dirs(root.path()).unwrap();
        assert_eq!(found, vec![snapshot]);
    }

    #[test]
    fn builds_cached_candidate_for_supported_snapshot() {
        let root = snapshot_dir("hf-candidate");
        let snapshot = root
            .join("models--Qwen--Qwen3-0.6B")
            .join("snapshots")
            .join("rev1");
        fs::create_dir_all(&snapshot).unwrap();
        fs::write(
            snapshot.join("config.json"),
            r#"{"architectures":["Qwen3ForCausalLM"],"model_type":"qwen3","hidden_size":8}"#,
        )
        .unwrap();
        fs::write(snapshot.join("tokenizer.json"), "{}").unwrap();
        fs::write(snapshot.join("model.safetensors"), "weights").unwrap();

        let candidate = cached_model_candidate(&snapshot).unwrap().unwrap();
        assert_eq!(candidate.repo, "Qwen/Qwen3-0.6B");
        assert_eq!(candidate.revision, "rev1");
        // The matched provider id depends on the platform's linked backend.
        #[cfg(target_os = "macos")]
        assert_eq!(candidate.provider_id, "mlx-llama");
        #[cfg(not(target_os = "macos"))]
        assert_eq!(candidate.provider_id, "candle-llama");
        assert!(!candidate.supports_vision);
    }

    #[test]
    fn skips_unsupported_vision_candidate() {
        // A VLM the providers don't serve is dropped from the cached-model list (the validate reject
        // maps to `Ok(None)`), so it never shows up as selectable.
        let root = snapshot_dir("hf-unsupported-vlm");
        let snapshot = root
            .join("models--google--gemma-3-vlm")
            .join("snapshots")
            .join("rev1");
        fs::create_dir_all(&snapshot).unwrap();
        fs::write(
            snapshot.join("config.json"),
            r#"{"architectures":["Gemma3ForConditionalGeneration"],"model_type":"gemma3","text_config":{"model_type":"gemma3_text"},"vision_config":{"model_type":"siglip"}}"#,
        )
        .unwrap();
        fs::write(snapshot.join("tokenizer.json"), "{}").unwrap();
        fs::write(snapshot.join("model.safetensors"), "weights").unwrap();

        assert!(cached_model_candidate(&snapshot).unwrap().is_none());
    }

    // Qwen3.6 (`qwen3_5`) vision is now served as a full VLM by BOTH backends: the `mlx-llama`
    // provider on macOS (sc-7633) and the `candle-llama` provider on Windows/Linux (sc-7634). A
    // cached Qwen3.6 VLM snapshot is therefore selectable and advertises vision on every platform —
    // only the matched provider id differs.
    #[test]
    fn builds_vision_candidate_for_qwen35_snapshot() {
        // A cached Qwen3.6 VLM snapshot is selectable and advertises vision (so the UI offers image
        // input), routed to the platform's linked backend.
        let root = snapshot_dir("hf-qwen35-vlm");
        let snapshot = root
            .join("models--Qwen--Qwen3.6-27B")
            .join("snapshots")
            .join("rev1");
        fs::create_dir_all(&snapshot).unwrap();
        fs::write(
            snapshot.join("config.json"),
            r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5","text_config":{"model_type":"qwen3_5_text","hidden_size":8},"vision_config":{"model_type":"qwen3_5"}}"#,
        )
        .unwrap();
        fs::write(snapshot.join("tokenizer.json"), "{}").unwrap();
        fs::write(snapshot.join("model.safetensors"), "weights").unwrap();

        let candidate = cached_model_candidate(&snapshot).unwrap().unwrap();
        assert_eq!(candidate.repo, "Qwen/Qwen3.6-27B");
        #[cfg(target_os = "macos")]
        assert_eq!(candidate.provider_id, "mlx-llama");
        #[cfg(not(target_os = "macos"))]
        assert_eq!(candidate.provider_id, "candle-llama");
        assert!(candidate.supports_vision);
    }

    // Qwen3-VL (`qwen3_vl`) vision is now served as a full VLM by BOTH backends: the `mlx-llama`
    // provider on macOS (sc-8078) and the `candle-llama` provider on Windows/Linux (sc-8080 image +
    // sc-8472 video). A cached Qwen3-VL snapshot is therefore selectable and advertises vision on
    // every platform — only the matched provider id differs.
    #[test]
    fn builds_vision_candidate_for_qwen3vl_snapshot() {
        // A cached Qwen3-VL snapshot is selectable and advertises vision (so the UI offers image
        // input), routed to the platform's linked backend.
        let root = snapshot_dir("hf-qwen3vl-vlm");
        let snapshot = root
            .join("models--Qwen--Qwen3-VL-8B-Instruct")
            .join("snapshots")
            .join("rev1");
        fs::create_dir_all(&snapshot).unwrap();
        fs::write(
            snapshot.join("config.json"),
            r#"{"architectures":["Qwen3VLForConditionalGeneration"],"model_type":"qwen3_vl","text_config":{"model_type":"qwen3_vl_text","hidden_size":8},"vision_config":{"model_type":"qwen3_vl"}}"#,
        )
        .unwrap();
        fs::write(snapshot.join("tokenizer.json"), "{}").unwrap();
        fs::write(snapshot.join("model.safetensors"), "weights").unwrap();

        let candidate = cached_model_candidate(&snapshot).unwrap().unwrap();
        assert_eq!(candidate.repo, "Qwen/Qwen3-VL-8B-Instruct");
        #[cfg(target_os = "macos")]
        assert_eq!(candidate.provider_id, "mlx-llama");
        #[cfg(not(target_os = "macos"))]
        assert_eq!(candidate.provider_id, "candle-llama");
        assert!(candidate.supports_vision);
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
                format: default_model_format(),
                pack: None,
                provider_id: None,
                projector_source: None,
                projector_sources: Vec::new(),
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
                format: default_model_format(),
                pack: None,
                provider_id: None,
                projector_source: Some("/tmp/mmproj-F16.gguf".to_string()),
                projector_sources: vec!["/tmp/mmproj-F16.gguf".to_string()],
            },
        );
        assert_eq!(registry.models.len(), 1);
        assert_eq!(registry.models[0].name, "new");
        assert_eq!(registry.models[0].file_count, 4);

        let dir = TempDir::new("registry-projector-roundtrip");
        let manifest = dir.path().join("manifest.json");
        write_registry(&manifest, &registry).unwrap();
        let restored = read_registry(&manifest).unwrap();
        assert_eq!(
            restored.models[0].projector_source.as_deref(),
            Some("/tmp/mmproj-F16.gguf")
        );
        assert_eq!(
            restored.models[0].projector_sources,
            vec!["/tmp/mmproj-F16.gguf"]
        );
    }

    fn snapshot_dir(name: &str) -> TempDir {
        TempDir::new(&format!("registry-{name}"))
    }

    fn write_snapshot_file(dir: &Path, name: &str, body: &str) {
        fs::write(dir.join(name), body).unwrap();
    }

    /// A metadata-only GGUF accepted by the weightless Prism route. It has no tensors and is used
    /// solely to verify registry discovery without loading real weights.
    fn write_minimal_prism_gguf(path: &Path) {
        fn string(out: &mut Vec<u8>, value: &str) {
            out.extend_from_slice(&(value.len() as u64).to_le_bytes());
            out.extend_from_slice(value.as_bytes());
        }
        let mut body = Vec::new();
        body.extend_from_slice(b"GGUF");
        body.extend_from_slice(&3_u32.to_le_bytes());
        body.extend_from_slice(&0_u64.to_le_bytes());
        body.extend_from_slice(&2_u64.to_le_bytes());
        string(&mut body, "general.architecture");
        body.extend_from_slice(&8_u32.to_le_bytes()); // GGUF string
        string(&mut body, "qwen35");
        string(&mut body, "prism.hadamard.version");
        body.extend_from_slice(&4_u32.to_le_bytes()); // GGUF u32
        body.extend_from_slice(&1_u32.to_le_bytes());
        fs::write(path, body).unwrap();
    }
}
