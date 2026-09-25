use std::convert::Infallible;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::{mpsc, Mutex};
use std::thread;

use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio::runtime::Runtime;
use tokio::sync::{mpsc as tokio_mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;

use crate::app_settings::SamplingDefaults;
use crate::core_llm::CancelFlag;
use crate::engine::{
    ConstraintRequest, EngineHandle, GenerateMedia, GenerateMessage, GenerateRequest,
    GenerateResponse, GenerateTool, GenerateToolCall, GenerateVideo, LoadedModelStatus, MtpRequest,
    ReasoningEffortRequest, SamplingRequest, StreamChannel, StreamPayload, ThinkingRequest,
    UsagePayload,
};
use crate::fsutil::{now_nanos, now_secs};

pub const DEFAULT_OPENAI_HOST: &str = "127.0.0.1";
pub const DEFAULT_OPENAI_PORT: u16 = 8000;
const OPENAI_JSON_BODY_LIMIT_BYTES: usize = 64 * 1024 * 1024;

pub type ServerResult<T> = Result<T, String>;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct OpenAiServerConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default)]
    pub allow_lan: bool,
    #[serde(default)]
    pub allow_local_files: bool,
    #[serde(default)]
    pub auth_token: Option<String>,
    #[serde(default)]
    pub sampling_defaults: SamplingDefaults,
}

impl Default for OpenAiServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            allow_lan: false,
            allow_local_files: false,
            auth_token: None,
            sampling_defaults: SamplingDefaults::default(),
        }
    }
}

fn default_host() -> String {
    DEFAULT_OPENAI_HOST.to_string()
}

fn default_port() -> u16 {
    DEFAULT_OPENAI_PORT
}

#[derive(Debug, Default)]
pub struct OpenAiServerHandle {
    state: Mutex<ServerState>,
}

impl OpenAiServerHandle {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn start(
        &self,
        config: OpenAiServerConfig,
        engine: EngineHandle,
    ) -> ServerResult<OpenAiServerStatus> {
        let auth_token = normalize_token(config.auth_token.clone());
        let server_config = OpenAiServerConfig {
            auth_token,
            ..config
        };
        let bind = validate_config(&server_config)?;
        self.stop()?;

        let (ready_tx, ready_rx) = mpsc::channel();
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let thread_config = server_config.clone();
        let join = thread::Builder::new()
            .name("chatworks-openai".to_string())
            .spawn(move || run_server_thread(bind, thread_config, engine, shutdown_rx, ready_tx))
            .map_err(|error| error.to_string())?;

        match ready_rx
            .recv()
            .map_err(|_| "server thread stopped before binding".to_string())?
        {
            Ok(bound_addr) => {
                let mut state = self
                    .state
                    .lock()
                    .map_err(|_| "server state lock poisoned".to_string())?;
                state.task = Some(ServerTask {
                    config: server_config,
                    bound_addr,
                    shutdown_tx: Some(shutdown_tx),
                    join: Some(join),
                });
                state.last_error = None;
                Ok(state.status())
            }
            Err(error) => {
                let _ = join.join();
                let mut state = self
                    .state
                    .lock()
                    .map_err(|_| "server state lock poisoned".to_string())?;
                state.last_error = Some(error.clone());
                Err(error)
            }
        }
    }

    pub fn stop(&self) -> ServerResult<OpenAiServerStatus> {
        let task = {
            let mut state = self
                .state
                .lock()
                .map_err(|_| "server state lock poisoned".to_string())?;
            state.task.take()
        };

        if let Some(mut task) = task {
            if let Some(shutdown_tx) = task.shutdown_tx.take() {
                let _ = shutdown_tx.send(());
            }
            if let Some(join) = task.join.take() {
                let _ = join.join();
            }
        }

        let state = self
            .state
            .lock()
            .map_err(|_| "server state lock poisoned".to_string())?;
        Ok(state.status())
    }

    pub fn status(&self) -> ServerResult<OpenAiServerStatus> {
        let state = self
            .state
            .lock()
            .map_err(|_| "server state lock poisoned".to_string())?;
        Ok(state.status())
    }
}

#[derive(Debug, Default)]
struct ServerState {
    task: Option<ServerTask>,
    last_error: Option<String>,
}

impl ServerState {
    fn status(&self) -> OpenAiServerStatus {
        if let Some(task) = &self.task {
            OpenAiServerStatus {
                running: true,
                host: task.config.host.clone(),
                port: task.config.port,
                bound_addr: Some(task.bound_addr.to_string()),
                allow_lan: task.config.allow_lan,
                auth_required: task.config.auth_token.is_some(),
                last_error: self.last_error.clone(),
            }
        } else {
            OpenAiServerStatus {
                running: false,
                host: DEFAULT_OPENAI_HOST.to_string(),
                port: DEFAULT_OPENAI_PORT,
                bound_addr: None,
                allow_lan: false,
                auth_required: false,
                last_error: self.last_error.clone(),
            }
        }
    }
}

#[derive(Debug)]
struct ServerTask {
    config: OpenAiServerConfig,
    bound_addr: SocketAddr,
    shutdown_tx: Option<oneshot::Sender<()>>,
    join: Option<thread::JoinHandle<()>>,
}

#[derive(Clone, Debug, Serialize)]
pub struct OpenAiServerStatus {
    pub running: bool,
    pub host: String,
    pub port: u16,
    pub bound_addr: Option<String>,
    pub allow_lan: bool,
    pub auth_required: bool,
    pub last_error: Option<String>,
}

fn run_server_thread(
    bind: SocketAddr,
    config: OpenAiServerConfig,
    engine: EngineHandle,
    shutdown_rx: oneshot::Receiver<()>,
    ready_tx: mpsc::Sender<Result<SocketAddr, String>>,
) {
    let result = Runtime::new()
        .map_err(|error| error.to_string())
        .and_then(|runtime| {
            runtime.block_on(run_server(bind, config, engine, shutdown_rx, ready_tx))
        });
    if let Err(error) = result {
        eprintln!("ChatWorks OpenAI server stopped: {error}");
    }
}

async fn run_server(
    bind: SocketAddr,
    config: OpenAiServerConfig,
    engine: EngineHandle,
    shutdown_rx: oneshot::Receiver<()>,
    ready_tx: mpsc::Sender<Result<SocketAddr, String>>,
) -> Result<(), String> {
    let listener = match tokio::net::TcpListener::bind(bind).await {
        Ok(listener) => listener,
        Err(error) => {
            let message = error.to_string();
            let _ = ready_tx.send(Err(message.clone()));
            return Err(message);
        }
    };
    let bound_addr = listener.local_addr().map_err(|error| error.to_string())?;
    let _ = ready_tx.send(Ok(bound_addr));
    axum::serve(
        listener,
        openai_router(
            engine,
            config.auth_token,
            config.sampling_defaults,
            config.allow_lan,
            config.allow_local_files,
        ),
    )
    .with_graceful_shutdown(async {
        let _ = shutdown_rx.await;
    })
    .await
    .map_err(|error| error.to_string())
}

fn openai_router(
    engine: EngineHandle,
    auth_token: Option<String>,
    sampling_defaults: SamplingDefaults,
    allow_lan: bool,
    allow_local_files: bool,
) -> Router {
    Router::new()
        .route("/v1/models", get(models).options(cors_preflight))
        .route(
            "/v1/chat/completions",
            post(chat_completions).options(cors_preflight),
        )
        .with_state(ApiState {
            engine,
            auth_token,
            sampling_defaults,
            allow_local_files,
        })
        .layer(DefaultBodyLimit::max(OPENAI_JSON_BODY_LIMIT_BYTES))
        .layer(axum::middleware::from_fn_with_state(
            allow_lan,
            apply_cors_headers,
        ))
}

async fn cors_preflight() -> StatusCode {
    StatusCode::NO_CONTENT
}

/// The origins that are ALWAYS allowed to call the OpenAI API, independent of `allow_lan`: the
/// ChatWorks webview itself. The app's chat screen uses a browser `fetch` from the webview to the
/// local server, so without CORS these origins' preflight would fail and every in-app send would
/// die with "Failed to fetch" (this is exactly the regression the first F-003 attempt introduced).
/// Packaged webviews use `tauri://localhost` on macOS and `http://tauri.localhost` on Windows;
/// `http://127.0.0.1:5173` is the Vite dev origin (see `tauri.conf.json` `devUrl`).
const APP_WEBVIEW_ORIGINS: &[&str] = &[
    "tauri://localhost",
    "http://tauri.localhost",
    "http://127.0.0.1:5173",
];

/// CORS policy for the OpenAI surface (code-review F-003).
///
/// Two cases:
/// - The ChatWorks webview's own origins (`APP_WEBVIEW_ORIGINS`) are always granted, regardless of
///   `allow_lan` — the in-app chat is a cross-origin browser `fetch`, so without an explicit grant
///   the preflight fails. This is the documented, trusted caller and is not gated behind `allow_lan`.
/// - When the user opts into LAN serving (`allow_lan`), the documented OpenAI-compatible surface is
///   opened to any origin (third-party LAN clients), matching the previous permissive behavior. We
///   reflect the request `Origin` and add `Vary: Origin` so a shared cache can't serve one origin's
///   grant to another. (No `Access-Control-Allow-Credentials` is emitted, so cookies/credentialed
///   requests are not enabled; auth uses a bearer header, not cookies.)
/// - When `allow_lan` is off and the origin is not a webview origin, no CORS headers are emitted:
///   a random browser page on the host can no longer drive the loopback model.
async fn apply_cors_headers(
    State(allow_lan): State<bool>,
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    // Capture the request Origin before the inner service consumes the request.
    let origin = request
        .headers()
        .get(axum::http::header::ORIGIN)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string);
    let allow = match origin.as_deref() {
        // The app's own webview is always allowed, even on loopback.
        Some(origin) if APP_WEBVIEW_ORIGINS.contains(&origin) => AllowOrigin::Webview,
        // LAN mode opens the documented OpenAI surface to any origin (third-party clients).
        Some(_) if allow_lan => AllowOrigin::Any,
        // Loopback, non-webview origin: no grant (tightens the old static `*`).
        _ => AllowOrigin::Deny,
    };
    let mut response = next.run(request).await;
    let headers = response.headers_mut();
    match allow {
        AllowOrigin::Webview => {
            let origin = origin.as_deref().expect("webview grant implies an origin");
            headers.insert(
                axum::http::header::ACCESS_CONTROL_ALLOW_ORIGIN,
                HeaderValue::from_str(origin).expect("webview origin is valid header"),
            );
            cors_methods_and_headers(headers);
        }
        AllowOrigin::Any => {
            headers.insert(
                axum::http::header::ACCESS_CONTROL_ALLOW_ORIGIN,
                HeaderValue::from_static("*"),
            );
            cors_methods_and_headers(headers);
            // The grant depends on the request Origin once LAN is on, so vary caches by it.
            headers.insert(axum::http::header::VARY, HeaderValue::from_static("Origin"));
        }
        AllowOrigin::Deny => {}
    }
    response
}

/// The methods + headers every granted preflight answer carries.
fn cors_methods_and_headers(headers: &mut HeaderMap) {
    headers.insert(
        axum::http::header::ACCESS_CONTROL_ALLOW_METHODS,
        HeaderValue::from_static("GET, POST, OPTIONS"),
    );
    headers.insert(
        axum::http::header::ACCESS_CONTROL_ALLOW_HEADERS,
        HeaderValue::from_static("authorization, content-type"),
    );
}

enum AllowOrigin {
    /// The request is from the app's own webview (always granted).
    Webview,
    /// LAN mode: grant any origin (the documented OpenAI surface).
    Any,
    /// Loopback, non-webview origin: no CORS grant.
    Deny,
}

#[derive(Clone)]
struct ApiState {
    engine: EngineHandle,
    auth_token: Option<String>,
    sampling_defaults: SamplingDefaults,
    allow_local_files: bool,
}

async fn models(
    State(state): State<ApiState>,
    headers: HeaderMap,
) -> Result<Json<OpenAiModelsResponse>, ApiError> {
    authorize(&headers, state.auth_token.as_deref())?;
    let status = tokio::task::spawn_blocking(move || state.engine.status())
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?
        .map_err(ApiError::engine)?;
    let data = status.loaded.into_iter().map(OpenAiModel::from).collect();
    Ok(Json(OpenAiModelsResponse {
        object: "list",
        data,
    }))
}

async fn chat_completions(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(request): Json<OpenAiChatRequest>,
) -> Result<Response, ApiError> {
    authorize(&headers, state.auth_token.as_deref())?;
    request.authorize_local_media(state.allow_local_files)?;
    let status_engine = state.engine.clone();
    let status = tokio::task::spawn_blocking(move || status_engine.status())
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?
        .map_err(ApiError::engine)?;
    let capabilities = &status
        .loaded
        .ok_or_else(|| ApiError::bad_request("load a model before generating"))?
        .provider
        .capabilities;
    let defaults = request.resolve_inherited_defaults(&state.sampling_defaults, capabilities);
    if request.stream {
        let stream = stream_chat_completion(state.engine, request, &defaults)?;
        Ok(stream.into_response())
    } else {
        let model = request.model_name();
        let generate_request = request.into_generate(&defaults)?;
        let response =
            tokio::task::spawn_blocking(move || state.engine.generate(generate_request, |_| {}))
                .await
                .map_err(|error| ApiError::internal(error.to_string()))?
                .map_err(ApiError::engine)?;
        Ok(Json(OpenAiChatResponse::from_generate(model, response)).into_response())
    }
}

fn stream_chat_completion(
    engine: EngineHandle,
    request: OpenAiChatRequest,
    sampling_defaults: &SamplingDefaults,
) -> Result<impl IntoResponse, ApiError> {
    let model = request.model_name();
    let id = completion_id();
    let created = created_timestamp();
    let generate_request = request.into_generate(sampling_defaults)?;
    let (tx, rx) = tokio_mpsc::channel::<Result<Event, Infallible>>(32);
    let cancel = CancelFlag::new();
    let worker_cancel = cancel.clone();
    let watch_tx = tx.clone();
    let (done_tx, done_rx) = oneshot::channel();

    tokio::task::spawn_blocking(move || {
        let result =
            engine.generate_with_cancel(generate_request, worker_cancel.clone(), |payload| {
                if let StreamPayload::Token { text, channel, .. } = payload {
                    let chunk = match channel {
                        StreamChannel::Content => {
                            OpenAiChatChunk::token(id.clone(), created, model.clone(), text)
                        }
                        StreamChannel::Thinking => {
                            OpenAiChatChunk::reasoning(id.clone(), created, model.clone(), text)
                        }
                    };
                    // The response watcher handles disconnects before the first token. A failed send
                    // also trips this request's flag, without touching a later engine request.
                    if tx.blocking_send(Ok(sse_json(&chunk))).is_err() {
                        worker_cancel.cancel();
                    }
                }
            });

        match result {
            Ok(response) => {
                // The provider surfaces tool calls only at end-of-generation, so emit them whole in
                // the final chunk and finish with `tool_calls` (end granularity is correct here).
                let (finish_reason, tool_calls) = if response.tool_calls.is_empty() {
                    (response.finish_reason, Vec::new())
                } else {
                    (
                        "tool_calls".to_string(),
                        tool_calls_delta(response.tool_calls),
                    )
                };
                let finish = OpenAiChatChunk::finish(
                    id.clone(),
                    created,
                    model.clone(),
                    finish_reason,
                    Some(OpenAiUsage::from(response.usage)),
                    tool_calls,
                    NativeTelemetry {
                        mtp: response.mtp,
                        timings: response.timings,
                        decode: response.decode,
                    },
                );
                let _ = tx.blocking_send(Ok(sse_json(&finish)));
                let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
            }
            Err(error) => {
                let _ = tx.blocking_send(Ok(sse_json(&OpenAiErrorBody::server(error))));
                let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
            }
        }
        let _ = done_tx.send(());
    });

    tokio::spawn(async move {
        tokio::select! {
            _ = watch_tx.closed() => cancel.cancel(),
            _ = done_rx => {},
        }
    });

    Ok(Sse::new(ReceiverStream::new(rx)).keep_alive(KeepAlive::default()))
}

fn sse_json<T: Serialize>(value: &T) -> Event {
    match serde_json::to_string(value) {
        Ok(data) => Event::default().data(data),
        Err(error) => Event::default().data(format!(
            "{{\"error\":{{\"message\":\"failed to serialize SSE event: {error}\"}}}}"
        )),
    }
}

fn authorize(headers: &HeaderMap, token: Option<&str>) -> Result<(), ApiError> {
    let Some(token) = token else {
        return Ok(());
    };
    // Compare the RAW token bytes (not the `Bearer `-prefixed string), as F-008 asks for "at
    // minimum". The header value is `Bearer <token>`; strip the fixed scheme prefix first. A missing
    // or non-Bearer header is rejected before any secret-dependent work.
    let actual = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .map(str::as_bytes);
    let ok = match actual {
        // Constant-time compare: no short-circuit on the first mismatched byte. A residual
        // length signal remains (the loop bound is the secret's length), which is acceptable for a
        // LAN server with sub-µs deltas; the prefix-stripped comparison is the meaningful fix over
        // `==` on the formatted string (code-review F-008).
        Some(actual) => constant_time_eq(token.as_bytes(), actual),
        None => false,
    };
    if ok {
        Ok(())
    } else {
        Err(ApiError::auth("missing or invalid bearer token"))
    }
}

/// Constant-time byte-slice equality: returns `true` only when `a == b`, but never short-circuits on
/// the first mismatched byte. Iterates over `secret.len()` bytes (folding in a length mismatch up
/// front) so the work is independent of WHERE the values differ. NOTE: the iteration count is
/// `secret.len()` — callers comparing a secret should pass it as `secret` and understand the secret
/// length influences the loop bound (a residual length signal; acceptable for this LAN-auth use
/// case, code-review F-008).
fn constant_time_eq(secret: &[u8], candidate: &[u8]) -> bool {
    let mut diff = (secret.len() ^ candidate.len()) as u8;
    for (i, &sb) in secret.iter().enumerate() {
        // Index `candidate` defensively; an out-of-bounds (candidate shorter than secret) is already
        // captured by the length diff above, so just XOR a poison byte to keep the loop secret-bound.
        let cb = candidate.get(i).copied().unwrap_or(0xFF);
        diff |= sb ^ cb;
    }
    diff == 0
}

fn validate_config(config: &OpenAiServerConfig) -> ServerResult<SocketAddr> {
    let host = config
        .host
        .parse::<IpAddr>()
        .map_err(|_| format!("invalid bind host '{}'", config.host))?;
    if is_unspecified(host) && !config.allow_lan {
        return Err("binding to 0.0.0.0 requires allow_lan=true".to_string());
    }
    if config.allow_local_files && config.auth_token.is_none() {
        return Err("local media access requires a bearer token".to_string());
    }
    Ok(SocketAddr::new(host, config.port))
}

fn is_unspecified(host: IpAddr) -> bool {
    match host {
        IpAddr::V4(value) => value == Ipv4Addr::UNSPECIFIED,
        IpAddr::V6(value) => value == Ipv6Addr::UNSPECIFIED,
    }
}

fn normalize_token(token: Option<String>) -> Option<String> {
    token.and_then(|value| {
        let trimmed = value.trim().to_string();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
    })
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    error_type: &'static str,
    message: String,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            error_type: "invalid_request_error",
            message: message.into(),
        }
    }

    fn auth(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            error_type: "authentication_error",
            message: message.into(),
        }
    }

    fn engine(message: String) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            error_type: "invalid_request_error",
            message,
        }
    }

    fn internal(message: String) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            error_type: "server_error",
            message,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(OpenAiErrorBody {
                error: OpenAiError {
                    message: self.message,
                    error_type: self.error_type,
                    code: None,
                },
            }),
        )
            .into_response()
    }
}

#[derive(Serialize)]
struct OpenAiErrorBody {
    error: OpenAiError,
}

impl OpenAiErrorBody {
    fn server(message: String) -> Self {
        Self {
            error: OpenAiError {
                message,
                error_type: "server_error",
                code: None,
            },
        }
    }
}

#[derive(Serialize)]
struct OpenAiError {
    message: String,
    #[serde(rename = "type")]
    error_type: &'static str,
    code: Option<String>,
}

#[derive(Serialize)]
struct OpenAiModelsResponse {
    object: &'static str,
    data: Vec<OpenAiModel>,
}

#[derive(Serialize)]
struct OpenAiModel {
    id: String,
    object: &'static str,
    created: u64,
    owned_by: &'static str,
}

impl From<LoadedModelStatus> for OpenAiModel {
    fn from(value: LoadedModelStatus) -> Self {
        Self {
            id: value.name,
            object: "model",
            created: created_timestamp(),
            owned_by: "chatworks",
        }
    }
}

#[derive(Deserialize)]
struct OpenAiChatRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<OpenAiChatMessage>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    presence_penalty: Option<f32>,
    #[serde(default)]
    repetition_penalty: Option<f32>,
    #[serde(default)]
    repetition_context: Option<usize>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    max_completion_tokens: Option<u32>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    stop: Option<StopValue>,
    #[serde(default)]
    disable_thinking: Option<bool>,
    #[serde(default)]
    enable_thinking: Option<bool>,
    #[serde(default)]
    reasoning_effort: Option<ReasoningEffortRequest>,
    /// Explicitly clear application overrides and use the model default.
    #[serde(default)]
    model_defaults: Vec<ModelDefaultControl>,
    #[serde(default)]
    preserve_thinking: Option<bool>,
    #[serde(default)]
    mtp: Option<MtpRequest>,
    #[serde(default)]
    response_format: Option<OpenAiResponseFormat>,
    /// Tools / functions offered to the model, in the OpenAI function-tool shape
    /// (`{"type":"function","function":{"name","description","parameters"}}`). Threaded to the
    /// provider, which rejects them with a 400 if it does not advertise tool support.
    #[serde(default)]
    tools: Option<Vec<OpenAiTool>>,
}

#[derive(Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
enum ModelDefaultControl {
    ReasoningEffort,
    PreserveThinking,
    Mtp,
}

impl OpenAiChatRequest {
    fn resolve_inherited_defaults(
        &self,
        defaults: &SamplingDefaults,
        caps: &crate::engine::CapabilitySummary,
    ) -> SamplingDefaults {
        let mut resolved = defaults.clone();
        let disabled = self
            .disable_thinking
            .unwrap_or(defaults.disable_thinking && self.enable_thinking != Some(true))
            || self.enable_thinking == Some(false);
        if disabled
            || !caps.supports_reasoning_effort
            || self
                .model_defaults
                .contains(&ModelDefaultControl::ReasoningEffort)
        {
            resolved.reasoning_effort = None;
        }
        if !caps.supports_preserve_thinking
            || self
                .model_defaults
                .contains(&ModelDefaultControl::PreserveThinking)
        {
            resolved.preserve_thinking = None;
        }
        if caps.mtp.is_none() || self.model_defaults.contains(&ModelDefaultControl::Mtp) {
            resolved.mtp_mode = "off".to_string();
        }
        resolved
    }

    #[cfg(test)]
    fn into_generate_for_engine(
        self,
        defaults: &SamplingDefaults,
        engine: &EngineHandle,
    ) -> Result<GenerateRequest, ApiError> {
        let status = engine.status().map_err(ApiError::engine)?;
        let caps = &status
            .loaded
            .ok_or_else(|| ApiError::bad_request("load a model before generating"))?
            .provider
            .capabilities;
        let resolved = self.resolve_inherited_defaults(defaults, caps);
        self.into_generate(&resolved)
    }

    fn model_name(&self) -> String {
        self.model
            .clone()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| "chatworks".to_string())
    }

    fn authorize_local_media(&self, allow_local_files: bool) -> Result<(), ApiError> {
        let requests_local_file = self.messages.iter().any(|message| {
            message
                .content
                .as_ref()
                .is_some_and(OpenAiMessageContent::contains_local_media)
        });
        if requests_local_file && !allow_local_files {
            return Err(ApiError::bad_request(
                "local media paths are disabled for the HTTP API; enable authenticated local media access in Settings or use the desktop file picker",
            ));
        }
        Ok(())
    }

    fn into_generate(self, defaults: &SamplingDefaults) -> Result<GenerateRequest, ApiError> {
        if self.messages.is_empty() {
            return Err(ApiError::bad_request("messages must not be empty"));
        }
        let has_system = self
            .messages
            .iter()
            .any(|message| matches!(message.role.as_str(), "system" | "developer"));
        let mut messages = Vec::new();
        if !has_system && !defaults.system_prompt.trim().is_empty() {
            messages.push(GenerateMessage {
                role: "system".to_string(),
                content: defaults.system_prompt.clone(),
                images: Vec::new(),
                videos: Vec::new(),
                media: Vec::new(),
                tool_calls: Vec::new(),
                thinking: None,
            });
        }
        for message in self.messages {
            messages.push(message.into_generate()?);
        }
        let tools = self
            .tools
            .unwrap_or_default()
            .into_iter()
            .map(OpenAiTool::into_generate)
            .collect::<Result<Vec<_>, _>>()?;
        let disable_thinking = self
            .disable_thinking
            .unwrap_or(defaults.disable_thinking && self.enable_thinking != Some(true));
        Ok(GenerateRequest {
            messages,
            sampling: SamplingRequest {
                temperature: Some(self.temperature.unwrap_or(defaults.temperature)),
                top_p: Some(self.top_p.unwrap_or(defaults.top_p)),
                top_k: self.top_k.or(defaults.top_k),
                presence_penalty: self.presence_penalty.or(defaults.presence_penalty),
                repetition_penalty: self.repetition_penalty.or(defaults.repetition_penalty),
                repetition_context: self.repetition_context.or(defaults.repetition_context),
            },
            max_new_tokens: self
                .max_completion_tokens
                .or(self.max_tokens)
                .unwrap_or(defaults.max_tokens),
            seed: self.seed.or(defaults.seed),
            stop: self.stop.map(StopValue::into_vec).unwrap_or_default(),
            thinking: ThinkingRequest::Auto,
            enable_thinking: self.enable_thinking,
            disable_thinking: Some(disable_thinking),
            reasoning_effort: self.reasoning_effort.or_else(|| {
                if disable_thinking
                    || self.enable_thinking == Some(false)
                    || self
                        .model_defaults
                        .contains(&ModelDefaultControl::ReasoningEffort)
                {
                    return None;
                }
                defaults
                    .reasoning_effort
                    .as_deref()
                    .and_then(parse_reasoning_effort)
            }),
            preserve_thinking: self.preserve_thinking.or(
                if self
                    .model_defaults
                    .contains(&ModelDefaultControl::PreserveThinking)
                {
                    None
                } else {
                    defaults.preserve_thinking
                },
            ),
            mtp: self.mtp.unwrap_or_else(|| {
                if self.model_defaults.contains(&ModelDefaultControl::Mtp) {
                    MtpRequest::Off
                } else {
                    mtp_from_defaults(defaults)
                }
            }),
            constraint: response_format_constraint(self.response_format)?,
            tools,
        })
    }
}

fn parse_reasoning_effort(value: &str) -> Option<ReasoningEffortRequest> {
    match value {
        "low" => Some(ReasoningEffortRequest::Low),
        "medium" => Some(ReasoningEffortRequest::Medium),
        "xhigh" => Some(ReasoningEffortRequest::Xhigh),
        _ => None,
    }
}

fn mtp_from_defaults(defaults: &SamplingDefaults) -> MtpRequest {
    match defaults.mtp_mode.as_str() {
        "auto" => MtpRequest::Auto,
        "enabled" => MtpRequest::Enabled {
            draft_tokens: defaults.mtp_draft_tokens,
        },
        _ => MtpRequest::Off,
    }
}

#[derive(Deserialize)]
struct OpenAiResponseFormat {
    #[serde(rename = "type")]
    kind: String,
}
fn response_format_constraint(
    format: Option<OpenAiResponseFormat>,
) -> Result<Option<ConstraintRequest>, ApiError> {
    match format.map(|value| value.kind) {
        None => Ok(None),
        Some(value) if value == "text" => Ok(None),
        Some(value) if value == "json_object" => Ok(Some(ConstraintRequest::Json)),
        Some(value) => Err(ApiError::bad_request(format!(
            "unsupported response_format type '{value}' (only json_object is supported)"
        ))),
    }
}

/// An offered tool in the OpenAI function-tool shape. Only `type: "function"` is supported.
#[derive(Deserialize)]
struct OpenAiTool {
    #[serde(rename = "type", default)]
    kind: Option<String>,
    function: OpenAiFunctionDef,
}

#[derive(Deserialize)]
struct OpenAiFunctionDef {
    name: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    parameters: Option<Value>,
}

impl OpenAiTool {
    fn into_generate(self) -> Result<GenerateTool, ApiError> {
        if let Some(kind) = self.kind.as_deref() {
            if kind != "function" {
                return Err(ApiError::bad_request(format!(
                    "unsupported tool type '{kind}' (only 'function' is supported)"
                )));
            }
        }
        Ok(GenerateTool {
            name: self.function.name,
            description: self.function.description.unwrap_or_default(),
            parameters: self
                .function
                .parameters
                .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}})),
        })
    }
}

/// An inbound chat message. `content` is optional: an assistant tool-call turn carries `tool_calls`
/// with `content: null`. OpenAI's `tool_call_id` (on `tool`-role result turns) and a tool call's
/// `id` / `type` are accepted but not forwarded — the core contract carries no call id and Qwen3.6's
/// chat template renders tool results positionally, so the id never reaches the rendered prompt.
#[derive(Deserialize)]
struct OpenAiChatMessage {
    role: String,
    #[serde(default)]
    content: Option<OpenAiMessageContent>,
    #[serde(default)]
    tool_calls: Vec<OpenAiToolCall>,
    #[serde(default)]
    reasoning_content: Option<String>,
}

impl OpenAiChatMessage {
    fn into_generate(self) -> Result<GenerateMessage, ApiError> {
        let (content, images, videos, media) = match self.content {
            Some(content) => content.into_parts()?,
            None => (String::new(), Vec::new(), Vec::new(), Vec::new()),
        };
        Ok(GenerateMessage {
            role: self.role,
            content,
            images,
            videos,
            media,
            tool_calls: self
                .tool_calls
                .into_iter()
                .map(OpenAiToolCall::into_generate)
                .collect::<Result<Vec<_>, _>>()?,
            thinking: self.reasoning_content,
        })
    }
}

/// A prior assistant turn's tool call in the OpenAI shape; its `arguments` is a JSON-encoded string.
#[derive(Deserialize)]
struct OpenAiToolCall {
    function: OpenAiFunctionCall,
}

#[derive(Deserialize)]
struct OpenAiFunctionCall {
    name: String,
    #[serde(default)]
    arguments: Option<String>,
}

impl OpenAiToolCall {
    fn into_generate(self) -> Result<GenerateToolCall, ApiError> {
        Ok(GenerateToolCall {
            name: self.function.name,
            arguments: parse_tool_arguments(self.function.arguments)?,
        })
    }
}

/// Decode an OpenAI tool call's JSON-encoded `arguments` string into the argument map. Absent or
/// empty ⇒ no arguments; a string that is not a JSON object is a 400 (rather than a silent guess).
fn parse_tool_arguments(raw: Option<String>) -> Result<Map<String, Value>, ApiError> {
    match raw {
        None => Ok(Map::new()),
        Some(value) if value.trim().is_empty() => Ok(Map::new()),
        Some(value) => serde_json::from_str::<Value>(&value)
            .ok()
            .and_then(|parsed| parsed.as_object().cloned())
            .ok_or_else(|| {
                ApiError::bad_request("tool_call function.arguments must be a JSON object string")
            }),
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum OpenAiMessageContent {
    Text(String),
    Parts(Vec<OpenAiContentPart>),
}

impl OpenAiMessageContent {
    fn contains_local_media(&self) -> bool {
        match self {
            Self::Text(_) => false,
            Self::Parts(parts) => parts.iter().any(|part| match part.kind.as_str() {
                "image_url" => part
                    .image_url
                    .as_ref()
                    .is_some_and(|image| is_local_media_source(&image.url)),
                "video_url" => part
                    .video_url
                    .as_ref()
                    .and_then(|video| video.url.as_deref())
                    .is_some_and(is_local_media_source),
                _ => false,
            }),
        }
    }

    /// Split OpenAI content into concatenated text, the ordered image-URL attachments, and the
    /// ordered video attachments. A plain string is text with no images/videos (the text path stays
    /// byte-identical).
    ///
    /// **Video representation.** There is no standard OpenAI `image_url` analog for video. A
    /// `video_url` accepts either pre-sampled frames plus timestamps, or one explicit file/URL in
    /// `video_url.url`. The latter is staged under bounded download/file limits and decoded by the
    /// app's FFmpeg media component before it reaches the same temporal frame path.
    /// ```json
    /// { "type": "video_url",
    ///   "video_url": {
    ///     "frames": ["data:image/jpeg;base64,…", "data:image/jpeg;base64,…"],
    ///     "timestamps": [0.0, 0.5],   // optional; seconds, one per frame
    ///     "fps": 2.0                  // optional; used to derive timestamps when absent
    ///   } }
    /// ```
    /// The ChatWorks frontend may sample frames client-side, while an explicit `video_url.url`
    /// uses the bundled FFmpeg sidecar. Each frame is an image data URL decoded exactly like an
    /// `image_url`. If `timestamps` is omitted it is
    /// derived from `fps` (`i / fps`) or, lacking both, frame index seconds (`i`, i.e. 1 fps) — the
    /// engine forwards these straight into `VideoRef`, which drives Text–Timestamp Alignment. A
    /// single `url` is intentionally not combined with `frames`; frames take precedence to retain
    /// backwards-compatible caller control of exact sampling.
    #[allow(clippy::type_complexity)]
    fn into_parts(
        self,
    ) -> Result<(String, Vec<String>, Vec<GenerateVideo>, Vec<GenerateMedia>), ApiError> {
        match self {
            Self::Text(value) => Ok((value, Vec::new(), Vec::new(), Vec::new())),
            Self::Parts(parts) => {
                let mut text = String::new();
                let mut images = Vec::new();
                let mut videos = Vec::new();
                let mut media = Vec::new();
                for part in parts {
                    match part.kind.as_str() {
                        "text" => text.push_str(&part.text.unwrap_or_default()),
                        "image_url" => {
                            let url = part.image_url.map(|image| image.url).ok_or_else(|| {
                                ApiError::bad_request("image_url part is missing its url")
                            })?;
                            images.push(url.clone());
                            if is_media_source_url(&url) {
                                media.push(GenerateMedia::ImageSource { url });
                            } else {
                                media.push(GenerateMedia::Image { url });
                            }
                        }
                        "video_url" => {
                            let video = part.video_url.ok_or_else(|| {
                                ApiError::bad_request("video_url part is missing its video_url")
                            })?;
                            if video.frames.is_empty() {
                                let url = video.url.ok_or_else(|| ApiError::bad_request(
                                    "video_url part must carry frames or a file/URL in video_url.url",
                                ))?;
                                media.push(GenerateMedia::VideoSource { url });
                                continue;
                            }
                            if video.fps.is_some_and(|fps| !fps.is_finite() || fps <= 0.0) {
                                return Err(ApiError::bad_request(
                                    "video_url fps must be finite and greater than zero",
                                ));
                            }
                            // Derive timestamps when absent: explicit > fps-derived > 1-fps index.
                            let n = video.frames.len();
                            let timestamps = match video.timestamps {
                                Some(ts) if ts.len() == n => {
                                    validate_video_timestamps(&ts)?;
                                    ts
                                }
                                Some(ts) => {
                                    return Err(ApiError::bad_request(format!(
                                        "video_url timestamps length {} != frame count {n}",
                                        ts.len()
                                    )));
                                }
                                None => {
                                    let fps = video.fps.unwrap_or(1.0);
                                    (0..n).map(|i| i as f32 / fps).collect()
                                }
                            };
                            let generated = GenerateVideo {
                                frames: video.frames,
                                timestamps,
                            };
                            media.push(GenerateMedia::Video {
                                frames: generated.frames.clone(),
                                timestamps: generated.timestamps.clone(),
                            });
                            videos.push(generated);
                        }
                        other => {
                            return Err(ApiError::bad_request(format!(
                                "unsupported content part type '{other}'"
                            )));
                        }
                    }
                }
                Ok((text, images, videos, media))
            }
        }
    }
}

fn validate_video_timestamps(timestamps: &[f32]) -> Result<(), ApiError> {
    if timestamps
        .iter()
        .any(|timestamp| !timestamp.is_finite() || *timestamp < 0.0)
    {
        return Err(ApiError::bad_request(
            "video_url timestamps must be finite and non-negative",
        ));
    }
    if timestamps.windows(2).any(|pair| pair[1] < pair[0]) {
        return Err(ApiError::bad_request(
            "video_url timestamps must be monotonically nondecreasing",
        ));
    }
    Ok(())
}

fn is_media_source_url(value: &str) -> bool {
    let value = value.trim();
    value.starts_with("http://")
        || value.starts_with("https://")
        || value.starts_with("file:")
        || value.starts_with('/')
        || (value
            .as_bytes()
            .get(1)
            .is_some_and(|separator| *separator == b':')
            && value
                .as_bytes()
                .first()
                .is_some_and(u8::is_ascii_alphabetic))
}

fn is_local_media_source(value: &str) -> bool {
    let value = value.trim();
    value.starts_with("file:")
        || value.starts_with('/')
        || (value
            .as_bytes()
            .get(1)
            .is_some_and(|separator| *separator == b':')
            && value
                .as_bytes()
                .first()
                .is_some_and(u8::is_ascii_alphabetic))
}

#[derive(Deserialize)]
struct OpenAiContentPart {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    image_url: Option<OpenAiImageUrl>,
    #[serde(default)]
    video_url: Option<OpenAiVideoUrl>,
}

/// The OpenAI vision part: `{"type":"image_url","image_url":{"url":"data:image/png;base64,…"}}`.
#[derive(Deserialize)]
struct OpenAiImageUrl {
    url: String,
}

/// The ChatWorks video part (sc-8081): pre-sampled frames + optional per-frame timestamps. See
/// [`OpenAiMessageContent::into_parts`] for the decision rationale and the wire shape.
#[derive(Deserialize)]
struct OpenAiVideoUrl {
    /// A local file path, `file://` URI, or HTTP(S) URL. It is decoded by the native media
    /// component into bounded timestamped frames. Mutually exclusive with `frames`.
    #[serde(default)]
    url: Option<String>,
    /// Sampled frames, in temporal order, each a `data:image/…;base64,…` URL (or bare base64).
    #[serde(default)]
    frames: Vec<String>,
    /// Optional per-frame timestamps in seconds (one per frame). Derived from `fps` / frame index
    /// when absent.
    #[serde(default)]
    timestamps: Option<Vec<f32>>,
    /// Optional sampling rate; used to derive timestamps when `timestamps` is absent.
    #[serde(default)]
    fps: Option<f32>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum StopValue {
    One(String),
    Many(Vec<String>),
}

impl StopValue {
    fn into_vec(self) -> Vec<String> {
        match self {
            Self::One(value) => vec![value],
            Self::Many(values) => values,
        }
    }
}

#[derive(Serialize)]
struct OpenAiChatResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<OpenAiChatChoice>,
    usage: OpenAiUsage,
    /// ChatWorks extension: native MTP counters, absent when ordinary autoregressive decode ran.
    #[serde(skip_serializing_if = "Option::is_none")]
    chatworks_mtp: Option<crate::engine::MtpStatsPayload>,
    /// ChatWorks extension: synchronized backend phase timings, absent when unavailable.
    #[serde(skip_serializing_if = "Option::is_none")]
    chatworks_timings: Option<crate::engine::GenerationTimingsPayload>,
    /// ChatWorks extension: which decode path served the generation (proposer, sampler, CUDA
    /// graphs, NVFP4 projection path), absent when the runtime does not report one.
    #[serde(skip_serializing_if = "Option::is_none")]
    chatworks_decode: Option<crate::engine::DecodeReportPayload>,
}

impl OpenAiChatResponse {
    fn from_generate(model: String, response: GenerateResponse) -> Self {
        let GenerateResponse {
            text,
            thinking,
            tool_calls,
            usage,
            finish_reason,
            mtp,
            timings,
            decode,
        } = response;
        let has_tool_calls = !tool_calls.is_empty();
        // A tool-call turn finishes with `tool_calls`, overriding the engine's stop/length reason.
        let finish_reason = if has_tool_calls {
            "tool_calls".to_string()
        } else {
            finish_reason
        };
        // OpenAI sets `content` to null on a pure tool-call turn (no preamble text); a turn that
        // produced answer text before the call keeps that text.
        let content = if text.is_empty() && has_tool_calls {
            None
        } else {
            Some(text)
        };
        Self {
            id: completion_id(),
            object: "chat.completion",
            created: created_timestamp(),
            model,
            choices: vec![OpenAiChatChoice {
                index: 0,
                message: Some(OpenAiResponseMessage {
                    role: "assistant",
                    content,
                    reasoning_content: thinking,
                    tool_calls: tool_calls_message(tool_calls),
                }),
                delta: None,
                finish_reason: Some(finish_reason),
            }],
            usage: OpenAiUsage::from(usage),
            chatworks_mtp: mtp,
            chatworks_timings: timings,
            chatworks_decode: decode,
        }
    }
}

#[derive(Serialize)]
struct NativeTelemetry {
    mtp: Option<crate::engine::MtpStatsPayload>,
    timings: Option<crate::engine::GenerationTimingsPayload>,
    decode: Option<crate::engine::DecodeReportPayload>,
}

#[derive(Serialize)]
struct OpenAiChatChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<OpenAiChatChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<OpenAiUsage>,
    /// ChatWorks extension: native MTP counters, emitted only on the terminal stream chunk.
    #[serde(skip_serializing_if = "Option::is_none")]
    chatworks_mtp: Option<crate::engine::MtpStatsPayload>,
    /// ChatWorks extension: synchronized backend phase timings, emitted only on the terminal stream chunk.
    #[serde(skip_serializing_if = "Option::is_none")]
    chatworks_timings: Option<crate::engine::GenerationTimingsPayload>,
    /// ChatWorks extension: the decode path that served the generation, emitted only on the
    /// terminal stream chunk.
    #[serde(skip_serializing_if = "Option::is_none")]
    chatworks_decode: Option<crate::engine::DecodeReportPayload>,
}

impl OpenAiChatChunk {
    fn token(id: String, created: u64, model: String, content: String) -> Self {
        Self {
            id,
            object: "chat.completion.chunk",
            created,
            model,
            choices: vec![OpenAiChatChoice {
                index: 0,
                message: None,
                delta: Some(OpenAiDelta {
                    content: Some(content),
                    reasoning_content: None,
                    tool_calls: Vec::new(),
                }),
                finish_reason: None,
            }],
            usage: None,
            chatworks_mtp: None,
            chatworks_timings: None,
            chatworks_decode: None,
        }
    }

    fn reasoning(id: String, created: u64, model: String, content: String) -> Self {
        Self {
            id,
            object: "chat.completion.chunk",
            created,
            model,
            choices: vec![OpenAiChatChoice {
                index: 0,
                message: None,
                delta: Some(OpenAiDelta {
                    content: None,
                    reasoning_content: Some(content),
                    tool_calls: Vec::new(),
                }),
                finish_reason: None,
            }],
            usage: None,
            chatworks_mtp: None,
            chatworks_timings: None,
            chatworks_decode: None,
        }
    }

    fn finish(
        id: String,
        created: u64,
        model: String,
        finish_reason: String,
        usage: Option<OpenAiUsage>,
        tool_calls: Vec<OpenAiToolCallDelta>,
        telemetry: NativeTelemetry,
    ) -> Self {
        Self {
            id,
            object: "chat.completion.chunk",
            created,
            model,
            choices: vec![OpenAiChatChoice {
                index: 0,
                message: None,
                delta: Some(OpenAiDelta {
                    content: None,
                    reasoning_content: None,
                    tool_calls,
                }),
                finish_reason: Some(finish_reason),
            }],
            usage,
            chatworks_mtp: telemetry.mtp,
            chatworks_timings: telemetry.timings,
            chatworks_decode: telemetry.decode,
        }
    }
}

#[derive(Serialize)]
struct OpenAiChatChoice {
    index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<OpenAiResponseMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    delta: Option<OpenAiDelta>,
    finish_reason: Option<String>,
}

#[derive(Serialize)]
struct OpenAiResponseMessage {
    role: &'static str,
    /// The answer text. Serialized as `null` (present, not omitted) on a pure tool-call turn, matching
    /// OpenAI; a text turn (incl. an empty one) carries the string, so the text path stays byte-identical.
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    /// The model's tool calls, in the OpenAI shape. Skipped (not emitted) when empty so a text-only
    /// response stays byte-identical.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<OpenAiToolCallOut>,
}

#[derive(Serialize)]
struct OpenAiDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    /// Tool calls, surfaced in one delta on the final chunk (the provider produces calls only at
    /// end-of-generation). Skipped when empty so text/reasoning chunks stay byte-identical.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<OpenAiToolCallDelta>,
}

/// A model tool call on a non-streaming response message (OpenAI `message.tool_calls[*]`).
#[derive(Serialize)]
struct OpenAiToolCallOut {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    function: OpenAiFunctionCallOut,
}

/// A model tool call on a streaming delta (OpenAI `delta.tool_calls[*]`) — adds the `index` that
/// correlates fragments across chunks (we emit each call whole in one chunk, so it is just its slot).
#[derive(Serialize)]
struct OpenAiToolCallDelta {
    index: u32,
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    function: OpenAiFunctionCallOut,
}

#[derive(Serialize)]
struct OpenAiFunctionCallOut {
    name: String,
    /// The arguments as the JSON-encoded string OpenAI carries on the wire.
    arguments: String,
}

impl OpenAiFunctionCallOut {
    fn from_call(call: GenerateToolCall) -> Self {
        Self {
            arguments: serde_json::to_string(&call.arguments).unwrap_or_else(|_| "{}".to_string()),
            name: call.name,
        }
    }
}

/// Synthesize the OpenAI call id (the provider does not assign one); the index keeps it unique within
/// a single response even at nanosecond granularity.
fn tool_call_id(index: usize) -> String {
    format!("call_{}_{index}", now_nanos())
}

/// The model's tool calls as non-streaming `message.tool_calls`.
fn tool_calls_message(calls: Vec<GenerateToolCall>) -> Vec<OpenAiToolCallOut> {
    calls
        .into_iter()
        .enumerate()
        .map(|(index, call)| OpenAiToolCallOut {
            id: tool_call_id(index),
            kind: "function",
            function: OpenAiFunctionCallOut::from_call(call),
        })
        .collect()
}

/// The model's tool calls as streaming `delta.tool_calls`.
fn tool_calls_delta(calls: Vec<GenerateToolCall>) -> Vec<OpenAiToolCallDelta> {
    calls
        .into_iter()
        .enumerate()
        .map(|(index, call)| OpenAiToolCallDelta {
            index: index as u32,
            id: tool_call_id(index),
            kind: "function",
            function: OpenAiFunctionCallOut::from_call(call),
        })
        .collect()
}

#[derive(Serialize)]
struct OpenAiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

impl From<UsagePayload> for OpenAiUsage {
    fn from(value: UsagePayload) -> Self {
        Self {
            prompt_tokens: value.prompt_tokens,
            completion_tokens: value.generated_tokens,
            total_tokens: value.total_tokens,
        }
    }
}

fn completion_id() -> String {
    format!("chatcmpl-{}", now_nanos())
}

/// Unix epoch seconds for OpenAI `created` timestamps (shared `now_secs`, F-011).
fn created_timestamp() -> u64 {
    now_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    // The weightless fakes are shared with the engine tests via `test_support` so the two can't
    // drift apart (code-review F-012).
    use crate::core_llm::{
        FinishReason, LoadSpec, StreamEvent, TextLlm, TextLlmDescriptor, TextLlmOutput,
        TextLlmRequest, Usage,
    };
    use crate::test_support::{
        fake_loader, fake_telemetry_loader, fake_tool_loader, thinking_descriptor,
    };
    use serde_json::{json, Value};
    use std::sync::OnceLock;
    use std::time::{Duration, Instant};

    static CANCEL_EVENTS: OnceLock<Mutex<Option<tokio_mpsc::UnboundedSender<&'static str>>>> =
        OnceLock::new();

    struct BlockingBeforeFirstToken {
        descriptor: TextLlmDescriptor,
        events: tokio_mpsc::UnboundedSender<&'static str>,
    }

    impl TextLlm for BlockingBeforeFirstToken {
        fn descriptor(&self) -> &TextLlmDescriptor {
            &self.descriptor
        }

        fn validate(&self, request: &TextLlmRequest) -> crate::core_llm::Result<()> {
            self.descriptor
                .capabilities
                .validate_request(&self.descriptor.id, request)
        }

        fn generate(
            &self,
            request: &TextLlmRequest,
            _on_event: &mut dyn FnMut(StreamEvent),
        ) -> crate::core_llm::Result<TextLlmOutput> {
            self.validate(request)?;
            let prompt = match &request.messages[0].content[0] {
                crate::core_llm::Content::Text(text) => text.as_str(),
                _ => panic!("expected text prompt"),
            };
            if prompt == "queued" {
                let _ = self.events.send("queued provider invoked");
            }
            if prompt == "hold" {
                let _ = self.events.send("started before first token");
                let deadline = Instant::now() + Duration::from_secs(5);
                while !request.cancel.is_cancelled() && Instant::now() < deadline {
                    std::thread::sleep(Duration::from_millis(5));
                }
                if request.cancel.is_cancelled() {
                    let _ = self.events.send("cancel observed");
                }
            }
            let usage = Usage {
                prompt_tokens: 1,
                generated_tokens: 0,
            };
            Ok(TextLlmOutput {
                text: "ok".to_string(),
                thinking: None,
                tool_calls: Vec::new(),
                usage,
                mtp: None,
                timings: None,
                decode: None,
                finish_reason: Some(if request.cancel.is_cancelled() {
                    FinishReason::Cancelled
                } else {
                    FinishReason::Stop
                }),
            })
        }
    }

    fn blocking_loader(_: &LoadSpec) -> crate::core_llm::Result<Box<dyn TextLlm>> {
        let events = CANCEL_EVENTS
            .get()
            .unwrap()
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .clone();
        Ok(Box::new(BlockingBeforeFirstToken {
            descriptor: thinking_descriptor("blocking", 8),
            events,
        }))
    }

    fn loaded_fake_engine() -> EngineHandle {
        let engine = EngineHandle::spawn_with_loader(fake_loader);
        engine
            .load_model(crate::engine::LoadModelRequest {
                source: "/tmp/fake-model".to_string(),
                display_name: Some("fake-model".to_string()),
                quantize: None,
                projector_source: None,
                cuda_graphs: None,
            })
            .unwrap();
        engine
    }

    fn loaded_telemetry_engine() -> EngineHandle {
        let engine = EngineHandle::spawn_with_loader(fake_telemetry_loader);
        engine
            .load_model(crate::engine::LoadModelRequest {
                source: "/tmp/fake-telemetry".to_string(),
                display_name: Some("fake-telemetry".to_string()),
                quantize: None,
                projector_source: None,
                cuda_graphs: None,
            })
            .unwrap();
        engine
    }

    fn loaded_tool_engine() -> EngineHandle {
        let engine = EngineHandle::spawn_with_loader(fake_tool_loader);
        engine
            .load_model(crate::engine::LoadModelRequest {
                source: "/tmp/fake-tools".to_string(),
                display_name: Some("fake-tools".to_string()),
                quantize: None,
                projector_source: None,
                cuda_graphs: None,
            })
            .unwrap();
        engine
    }

    /// The OpenAI `get_weather` function tool used by the tool-calling tests.
    fn weather_tool() -> Value {
        json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"]
                }
            }
        })
    }

    fn test_sampling_defaults() -> SamplingDefaults {
        SamplingDefaults {
            system_prompt: "".to_string(),
            disable_thinking: false,
            ..Default::default()
        }
    }

    #[test]
    fn rejects_unspecified_bind_without_lan_opt_in() {
        let config = OpenAiServerConfig {
            host: "0.0.0.0".to_string(),
            ..Default::default()
        };
        assert_eq!(
            validate_config(&config).unwrap_err(),
            "binding to 0.0.0.0 requires allow_lan=true"
        );
    }

    #[test]
    fn accepts_unspecified_bind_with_lan_opt_in() {
        let config = OpenAiServerConfig {
            host: "0.0.0.0".to_string(),
            allow_lan: true,
            ..Default::default()
        };
        assert_eq!(
            validate_config(&config).unwrap(),
            SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), DEFAULT_OPENAI_PORT)
        );
    }

    #[test]
    fn local_media_api_policy_requires_authentication_and_explicit_opt_in() {
        let request: OpenAiChatRequest = serde_json::from_value(json!({
            "messages": [{"role": "user", "content": [{
                "type": "image_url", "image_url": {"url": "file:///tmp/private.png"}
            }]}]
        }))
        .unwrap();
        let error = request.authorize_local_media(false).unwrap_err();
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert!(request.authorize_local_media(true).is_ok());

        let alternate_file_uri: OpenAiChatRequest = serde_json::from_value(json!({
            "messages": [{"role": "user", "content": [{
                "type": "video_url", "video_url": {"url": "file:/tmp/private.mp4"}
            }]}]
        }))
        .unwrap();
        assert!(alternate_file_uri.authorize_local_media(false).is_err());

        let unauthenticated = OpenAiServerConfig {
            host: "0.0.0.0".to_string(),
            allow_lan: true,
            allow_local_files: true,
            ..Default::default()
        };
        assert!(validate_config(&unauthenticated).is_err());
        let authenticated = OpenAiServerConfig {
            auth_token: Some("secret".to_string()),
            ..unauthenticated
        };
        assert!(validate_config(&authenticated).is_ok());
        assert!(authorize(&HeaderMap::new(), authenticated.auth_token.as_deref()).is_err());
    }

    #[test]
    fn non_streaming_response_preserves_native_telemetry() {
        let response = OpenAiChatResponse::from_generate(
            "test".to_string(),
            GenerateResponse {
                text: "ok".to_string(),
                thinking: None,
                tool_calls: Vec::new(),
                usage: crate::engine::UsagePayload {
                    prompt_tokens: 1,
                    generated_tokens: 1,
                    total_tokens: 2,
                },
                finish_reason: "stop".to_string(),
                mtp: Some(crate::engine::MtpStatsPayload {
                    proposed_tokens: 4,
                    accepted_tokens: 3,
                    target_forwards: 2,
                }),
                timings: Some(crate::engine::GenerationTimingsPayload {
                    prefill_ms: 12,
                    decode_ms: 34,
                }),
                decode: Some(crate::engine::DecodeReportPayload::from(
                    crate::test_support::fake_decode_report(),
                )),
            },
        );
        let json = serde_json::to_value(response).unwrap();
        assert_eq!(json["chatworks_mtp"]["accepted_tokens"], 3);
        assert_eq!(json["chatworks_timings"]["prefill_ms"], 12);
        assert_eq!(json["chatworks_decode"]["proposer"], "mtp");
        assert_eq!(
            json["chatworks_decode"]["cuda_graphs"]["fallback_reason"],
            "deltanet_state_unstable"
        );
    }

    #[test]
    fn streaming_terminal_chunk_preserves_native_telemetry() {
        let response = OpenAiChatChunk::finish(
            "chatcmpl-test".to_string(),
            1,
            "test".to_string(),
            "stop".to_string(),
            Some(OpenAiUsage {
                prompt_tokens: 1,
                completion_tokens: 1,
                total_tokens: 2,
            }),
            Vec::new(),
            NativeTelemetry {
                mtp: Some(crate::engine::MtpStatsPayload {
                    proposed_tokens: 4,
                    accepted_tokens: 3,
                    target_forwards: 2,
                }),
                timings: Some(crate::engine::GenerationTimingsPayload {
                    prefill_ms: 12,
                    decode_ms: 34,
                }),
                decode: Some(crate::engine::DecodeReportPayload::from(
                    crate::test_support::fake_decode_report(),
                )),
            },
        );
        let json = serde_json::to_value(response).unwrap();
        assert_eq!(json["chatworks_mtp"]["accepted_tokens"], 3);
        assert_eq!(json["chatworks_timings"]["decode_ms"], 34);
        assert_eq!(json["chatworks_decode"]["sampler"], "device");
        assert_eq!(
            json["chatworks_decode"]["nvfp4_projections"]["path"],
            "mixed"
        );
    }

    #[test]
    fn inherited_controls_resolve_through_loaded_engine_but_explicit_unsupported_controls_error() {
        let engine = loaded_fake_engine();
        let defaults = SamplingDefaults {
            system_prompt: String::new(),
            max_tokens: 8,
            reasoning_effort: Some("xhigh".into()),
            preserve_thinking: Some(true),
            mtp_mode: "enabled".into(),
            ..Default::default()
        };
        let base = json!({"messages":[{"role":"user","content":"hello"}]});
        let request: OpenAiChatRequest = serde_json::from_value(base.clone()).unwrap();
        let generated = request
            .into_generate_for_engine(&defaults, &engine)
            .unwrap();
        assert!(engine.generate(generated, |_| {}).is_ok());
        for (key, value, expected) in [
            ("reasoning_effort", json!("low"), "reasoning_effort"),
            ("preserve_thinking", json!(true), "preserve_thinking"),
            ("mtp", json!({"mode":"enabled", "draft_tokens":3}), "MTP"),
        ] {
            let mut wire = base.clone();
            wire[key] = value;
            let request: OpenAiChatRequest = serde_json::from_value(wire).unwrap();
            let generated = request
                .into_generate_for_engine(&defaults, &engine)
                .unwrap();
            assert!(engine
                .generate(generated, |_| {})
                .unwrap_err()
                .contains(expected));
        }
    }

    #[test]
    fn effective_thinking_filters_only_inherited_effort() {
        let mut caps = crate::engine::CapabilitySummary::from(
            crate::test_support::thinking_descriptor("fixture", 8).capabilities,
        );
        caps.supports_reasoning_effort = true;
        let defaults = SamplingDefaults {
            disable_thinking: true,
            reasoning_effort: Some("xhigh".into()),
            ..Default::default()
        };
        for (control, disabled, has_effort) in [
            (json!({"disable_thinking":true}), true, false),
            (json!({"enable_thinking":true}), false, true),
        ] {
            let mut wire = control;
            wire["messages"] = json!([{"role":"user", "content":"hello"}]);
            let request: OpenAiChatRequest = serde_json::from_value(wire).unwrap();
            let resolved = request.resolve_inherited_defaults(&defaults, &caps);
            let generated = request.into_generate(&resolved).unwrap();
            assert_eq!(generated.disable_thinking, Some(disabled));
            assert_eq!(generated.reasoning_effort.is_some(), has_effort);
        }
    }

    #[test]
    fn desktop_controls_clear_nontrivial_defaults_across_capability_switches() {
        let mut wire: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/generation-wire.json")).unwrap();
        wire["messages"] = json!([{"role":"user", "content":"hello"}]);
        wire["disable_thinking"] = json!(true);
        let defaults = SamplingDefaults {
            system_prompt: String::new(),
            reasoning_effort: Some("xhigh".into()),
            preserve_thinking: Some(true),
            mtp_mode: "enabled".into(),
            ..Default::default()
        };
        for supported in [true, false] {
            let mut caps = crate::test_support::thinking_descriptor("fixture", 8).capabilities;
            caps.supports_reasoning_effort = supported;
            caps.supports_preserve_thinking = supported;
            let summary = crate::engine::CapabilitySummary::from(caps);
            let request: OpenAiChatRequest = serde_json::from_value(wire.clone()).unwrap();
            let resolved = request.resolve_inherited_defaults(&defaults, &summary);
            let output = request.into_generate(&resolved).unwrap();
            assert!(output.reasoning_effort.is_none());
            assert!(output.preserve_thinking.is_none());
            assert!(matches!(output.mtp, MtpRequest::Off));
        }
        // An ordinary external API client inherits only controls supported by this model.
        let caps = crate::engine::CapabilitySummary::from(
            crate::test_support::thinking_descriptor("fixture", 8).capabilities,
        );
        let request: OpenAiChatRequest =
            serde_json::from_value(json!({"messages":[{"role":"user","content":"hello"}]}))
                .unwrap();
        let resolved = request.resolve_inherited_defaults(&defaults, &caps);
        let output = request.into_generate(&resolved).unwrap();
        assert!(output.reasoning_effort.is_none());
        assert!(output.preserve_thinking.is_none());
        assert!(matches!(output.mtp, MtpRequest::Off));
        // Explicit unsupported intent is retained, for actionable native capability validation.
        wire["reasoning_effort"] = json!("low");
        wire["preserve_thinking"] = json!(true);
        wire["mtp"] = json!({"mode":"auto"});
        let request: OpenAiChatRequest = serde_json::from_value(wire).unwrap();
        let resolved = request.resolve_inherited_defaults(&defaults, &caps);
        let output = request.into_generate(&resolved).unwrap();
        assert!(matches!(
            output.reasoning_effort,
            Some(ReasoningEffortRequest::Low)
        ));
        assert_eq!(output.preserve_thinking, Some(true));
        assert!(matches!(output.mtp, MtpRequest::Auto));
    }

    #[test]
    fn maps_chat_request_to_engine_request() {
        let request: OpenAiChatRequest = serde_json::from_value(json!({
            "model": "fake",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 7,
            "seed": 42,
            "stop": ["END"]
        }))
        .unwrap();

        let defaults = SamplingDefaults {
            system_prompt: "".to_string(),
            disable_thinking: false,
            ..Default::default()
        };
        let generate = request.into_generate(&defaults).unwrap();
        assert_eq!(generate.messages.len(), 1);
        assert_eq!(generate.messages[0].role, "user");
        assert_eq!(generate.messages[0].content, "hello");
        assert_eq!(generate.sampling.temperature, Some(0.2));
        assert_eq!(generate.sampling.top_p, Some(0.9));
        assert_eq!(generate.max_new_tokens, 7);
        assert_eq!(generate.seed, Some(42));
        assert_eq!(generate.stop, vec!["END"]);
        assert!(matches!(generate.thinking, ThinkingRequest::Auto));
    }

    #[test]
    fn maps_native_qwen_controls_and_preserves_reasoning_history() {
        let request: OpenAiChatRequest = serde_json::from_value(json!({
            "messages": [{"role": "assistant", "content": "answer", "reasoning_content": "trace"}],
            "top_k": 12, "presence_penalty": 1.5, "repetition_penalty": 1.1, "repetition_context": 32,
            "reasoning_effort": "low", "preserve_thinking": true,
            "mtp": {"mode": "enabled", "draft_tokens": 3},
            "response_format": {"type": "json_object"}
        })).unwrap();
        let generate = request.into_generate(&test_sampling_defaults()).unwrap();
        assert_eq!(generate.messages[0].thinking.as_deref(), Some("trace"));
        assert_eq!(generate.sampling.top_k, Some(12));
        assert_eq!(generate.sampling.presence_penalty, Some(1.5));
        assert_eq!(generate.sampling.repetition_penalty, Some(1.1));
        assert_eq!(generate.sampling.repetition_context, Some(32));
        assert!(matches!(
            generate.reasoning_effort,
            Some(ReasoningEffortRequest::Low)
        ));
        assert_eq!(generate.preserve_thinking, Some(true));
        assert!(matches!(
            generate.mtp,
            MtpRequest::Enabled { draft_tokens: 3 }
        ));
        assert!(matches!(generate.constraint, Some(ConstraintRequest::Json)));
    }

    #[test]
    fn rejects_contradictory_thinking_flags_and_unknown_response_format() {
        let conflict: OpenAiChatRequest = serde_json::from_value(json!({
            "messages": [{"role": "user", "content": "hello"}],
            "enable_thinking": true, "disable_thinking": true
        }))
        .unwrap();
        let generated = conflict.into_generate(&test_sampling_defaults()).unwrap();
        assert_eq!(generated.enable_thinking, Some(true));
        assert_eq!(generated.disable_thinking, Some(true));
        let unsupported: OpenAiChatRequest = serde_json::from_value(json!({
            "messages": [{"role": "user", "content": "hello"}],
            "response_format": {"type": "json_schema"}
        }))
        .unwrap();
        assert!(unsupported
            .into_generate(&test_sampling_defaults())
            .is_err());
    }

    /// A `video_url` content part with pre-sampled frames + explicit timestamps parses into a
    /// `GenerateVideo` carrying the frames and timestamps verbatim, alongside the text (sc-8081).
    #[test]
    fn parses_video_url_content_part_with_timestamps() {
        let request: OpenAiChatRequest = serde_json::from_value(json!({
            "model": "fake",
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {
                    "frames": ["data:image/jpeg;base64,AAA", "data:image/jpeg;base64,BBB"],
                    "timestamps": [0.0, 0.5]
                }},
                {"type": "text", "text": "what happens"}
            ]}],
            "max_tokens": 8
        }))
        .unwrap();
        let defaults = SamplingDefaults {
            system_prompt: String::new(),
            ..Default::default()
        };
        let generate = request.into_generate(&defaults).unwrap();
        let msg = &generate.messages[0];
        assert_eq!(msg.content, "what happens");
        assert!(msg.images.is_empty());
        assert_eq!(msg.videos.len(), 1);
        assert_eq!(msg.videos[0].frames.len(), 2);
        assert_eq!(msg.videos[0].timestamps, vec![0.0, 0.5]);
        assert!(
            matches!(msg.media.as_slice(), [GenerateMedia::Video { timestamps, .. }] if timestamps == &vec![0.0, 0.5])
        );
    }

    #[test]
    fn rejects_invalid_video_timestamps_and_fps_at_the_api_boundary() {
        for timestamps in [vec![-1.0, 0.0], vec![2.0, 1.0], vec![f32::NAN, 1.0]] {
            assert!(validate_video_timestamps(&timestamps).is_err());
        }
        assert!(validate_video_timestamps(&[0.0, 0.0, 1.5]).is_ok());

        let invalid_fps = OpenAiMessageContent::Parts(vec![OpenAiContentPart {
            kind: "video_url".to_string(),
            text: None,
            image_url: None,
            video_url: Some(OpenAiVideoUrl {
                url: None,
                frames: vec!["data:image/jpeg;base64,AAA".to_string()],
                timestamps: None,
                fps: Some(f32::INFINITY),
            }),
        }]);
        assert!(invalid_fps.into_parts().is_err());
    }

    #[test]
    fn preserves_mixed_media_order_and_accepts_a_video_file_or_url_source() {
        let request: OpenAiChatRequest = serde_json::from_value(json!({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAA"}},
                {"type": "video_url", "video_url": {"url": "file:///tmp/example.mp4"}},
                {"type": "video_url", "video_url": {
                    "frames": ["data:image/jpeg;base64,BBB"], "timestamps": [3.5]
                }},
                {"type": "text", "text": "compare them"}
            ]}]
        }))
        .unwrap();
        let generated = request
            .into_generate(&SamplingDefaults {
                system_prompt: String::new(),
                ..Default::default()
            })
            .unwrap();
        let message = &generated.messages[0];
        assert_eq!(message.content, "compare them");
        assert!(matches!(message.media.as_slice(), [
            GenerateMedia::Image { url },
            GenerateMedia::VideoSource { url: source },
            GenerateMedia::Video { timestamps, .. },
        ] if url == "data:image/jpeg;base64,AAA"
            && source == "file:///tmp/example.mp4"
            && timestamps == &vec![3.5]));
    }

    #[test]
    fn recognizes_windows_file_paths_as_native_media_sources() {
        assert!(is_media_source_url(r"C:\Users\me\clip.mp4"));
        assert!(is_media_source_url(r"z:\cache\image.jpg"));
        assert!(!is_media_source_url("data:image/jpeg;base64,AAA"));
    }

    #[test]
    fn maps_file_and_http_image_urls_to_bounded_native_sources() {
        let request: OpenAiChatRequest = serde_json::from_value(json!({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "file:///tmp/picture.png"}},
                {"type": "image_url", "image_url": {"url": "https://cdn.example.test/picture.jpg"}}
            ]}]
        }))
        .unwrap();
        let generated = request
            .into_generate(&SamplingDefaults {
                system_prompt: String::new(),
                ..Default::default()
            })
            .unwrap();
        assert!(matches!(generated.messages[0].media.as_slice(), [
            GenerateMedia::ImageSource { url: local },
            GenerateMedia::ImageSource { url: remote },
        ] if local == "file:///tmp/picture.png" && remote == "https://cdn.example.test/picture.jpg"));
    }

    /// When `timestamps` is omitted, they are derived from `fps` (`i / fps`).
    #[test]
    fn video_url_derives_timestamps_from_fps() {
        let request: OpenAiChatRequest = serde_json::from_value(json!({
            "model": "fake",
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {
                    "frames": ["data:image/jpeg;base64,AAA", "data:image/jpeg;base64,BBB",
                               "data:image/jpeg;base64,CCC", "data:image/jpeg;base64,DDD"],
                    "fps": 2.0
                }},
                {"type": "text", "text": "describe"}
            ]}],
            "max_tokens": 8
        }))
        .unwrap();
        let defaults = SamplingDefaults {
            system_prompt: String::new(),
            ..Default::default()
        };
        let generate = request.into_generate(&defaults).unwrap();
        assert_eq!(
            generate.messages[0].videos[0].timestamps,
            vec![0.0, 0.5, 1.0, 1.5]
        );
    }

    /// A `video_url` part with no frames is a 400, and a timestamp/frame-count mismatch is a 400.
    #[test]
    fn video_url_rejects_empty_and_mismatched() {
        let empty: OpenAiChatRequest = serde_json::from_value(json!({
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"frames": []}}
            ]}]
        }))
        .unwrap();
        assert!(empty.into_generate(&SamplingDefaults::default()).is_err());

        let mismatched: OpenAiChatRequest = serde_json::from_value(json!({
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {
                    "frames": ["data:image/jpeg;base64,AAA"],
                    "timestamps": [0.0, 0.5]
                }}
            ]}]
        }))
        .unwrap();
        assert!(mismatched
            .into_generate(&SamplingDefaults::default())
            .is_err());
    }

    #[test]
    fn applies_sampling_defaults_when_request_omits_them() {
        let request: OpenAiChatRequest = serde_json::from_value(json!({
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();

        let defaults = SamplingDefaults {
            system_prompt: "be terse".to_string(),
            temperature: 0.3,
            top_p: 0.8,
            max_tokens: 64,
            disable_thinking: true,
            ..Default::default()
        };
        let generate = request.into_generate(&defaults).unwrap();
        assert_eq!(generate.messages.len(), 2);
        assert_eq!(generate.messages[0].role, "system");
        assert_eq!(generate.messages[0].content, "be terse");
        assert_eq!(generate.sampling.temperature, Some(0.3));
        assert_eq!(generate.sampling.top_p, Some(0.8));
        assert_eq!(generate.max_new_tokens, 64);
        assert!(matches!(generate.thinking, ThinkingRequest::Auto));
        assert_eq!(generate.disable_thinking, Some(true));
    }

    #[test]
    fn maps_disable_thinking_to_core_thinking_mode() {
        let request: OpenAiChatRequest = serde_json::from_value(json!({
            "messages": [{"role": "user", "content": "hello"}],
            "disable_thinking": true
        }))
        .unwrap();

        let generate = request.into_generate(&SamplingDefaults::default()).unwrap();
        assert!(matches!(generate.thinking, ThinkingRequest::Auto));
        assert_eq!(generate.disable_thinking, Some(true));
    }

    #[test]
    fn empty_auth_token_disables_auth() {
        assert_eq!(normalize_token(Some("  ".to_string())), None);
        assert_eq!(
            normalize_token(Some(" token ".to_string())),
            Some("token".to_string())
        );
    }

    #[test]
    fn constant_time_eq_compares_without_short_circuit() {
        // F-008: equal slices match, and any difference (prefix, suffix, length) rejects without
        // short-circuiting. The function returns false for mismatched lengths and true only on a
        // full byte match.
        assert!(constant_time_eq(b"Bearer secret", b"Bearer secret"));
        assert!(!constant_time_eq(b"Bearer secret", b"Bearer secr3t"));
        assert!(!constant_time_eq(b"Bearer secret", b"Bearer secre"));
        assert!(!constant_time_eq(b"Bearer secret", b"Bearer secrett"));
        assert!(!constant_time_eq(b"", b"a"));
        assert!(constant_time_eq(b"", b""));
    }

    fn http_request(addr: &str, request: String) -> String {
        use std::io::{Read, Write};
        use std::net::TcpStream;

        let mut stream = TcpStream::connect(addr).unwrap();
        stream.write_all(request.as_bytes()).unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        response
    }

    fn http_get(addr: &str, path: &str, token: Option<&str>) -> String {
        let auth = token
            .map(|value| format!("Authorization: Bearer {value}\r\n"))
            .unwrap_or_default();
        http_request(
            addr,
            format!("GET {path} HTTP/1.1\r\nHost: {addr}\r\n{auth}Connection: close\r\n\r\n"),
        )
    }

    fn http_post_json(addr: &str, path: &str, body: Value, token: Option<&str>) -> String {
        let body = body.to_string();
        let auth = token
            .map(|value| format!("Authorization: Bearer {value}\r\n"))
            .unwrap_or_default();
        http_request(
            addr,
            format!(
                "POST {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{auth}Connection: close\r\n\r\n{body}",
                body.len()
            ),
        )
    }

    /// A CORS preflight from a specific `Origin` (so the policy's origin-based branches can be
    /// exercised: the webview origin, the packaged `tauri://localhost`, and a non-granted origin).
    fn http_options_with_origin(addr: &str, path: &str, origin: &str) -> String {
        http_request(
            addr,
            format!(
                "OPTIONS {path} HTTP/1.1\r\nHost: {addr}\r\nOrigin: {origin}\r\nAccess-Control-Request-Method: POST\r\nAccess-Control-Request-Headers: content-type\r\nConnection: close\r\n\r\n"
            ),
        )
    }

    fn response_body(response: &str) -> &str {
        response.split("\r\n\r\n").nth(1).unwrap_or_default()
    }

    #[test]
    fn http_rejects_local_media_before_decoder_access_when_policy_is_off() {
        let server = OpenAiServerHandle::new();
        let status = server
            .start(
                OpenAiServerConfig {
                    port: 0,
                    sampling_defaults: test_sampling_defaults(),
                    ..Default::default()
                },
                loaded_fake_engine(),
            )
            .unwrap();
        let addr = status.bound_addr.unwrap();
        let response = http_post_json(
            &addr,
            "/v1/chat/completions",
            json!({
                "model": "fake-model",
                "messages": [{"role": "user", "content": [{
                    "type": "image_url",
                    "image_url": {"url": "file:///path/that/must/not/be-opened.png"}
                }]}],
                "max_tokens": 8
            }),
            None,
        );
        assert!(response.starts_with("HTTP/1.1 400"));
        assert!(response_body(&response).contains("local media paths are disabled"));
        server.stop().unwrap();
    }

    #[test]
    fn lists_loaded_model() {
        let server = OpenAiServerHandle::new();
        let status = server
            .start(
                OpenAiServerConfig {
                    port: 0,
                    sampling_defaults: test_sampling_defaults(),
                    ..Default::default()
                },
                loaded_fake_engine(),
            )
            .unwrap();
        let addr = status.bound_addr.unwrap();
        let response = http_get(&addr, "/v1/models", None);
        let body: Value = serde_json::from_str(response_body(&response)).unwrap();
        assert_eq!(body["data"][0]["id"], "fake-model");
        server.stop().unwrap();
    }

    #[test]
    fn returns_non_streaming_chat_completion() {
        let server = OpenAiServerHandle::new();
        let status = server
            .start(
                OpenAiServerConfig {
                    port: 0,
                    sampling_defaults: test_sampling_defaults(),
                    ..Default::default()
                },
                loaded_fake_engine(),
            )
            .unwrap();
        let addr = status.bound_addr.unwrap();
        let response = http_post_json(
            &addr,
            "/v1/chat/completions",
            json!({
                "model": "fake-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 8
            }),
            None,
        );
        let body: Value = serde_json::from_str(response_body(&response)).unwrap();
        assert_eq!(body["choices"][0]["message"]["content"], "ok");
        assert_eq!(body["choices"][0]["message"]["reasoning_content"], "reason");
        server.stop().unwrap();
    }

    #[test]
    fn omits_reasoning_when_thinking_disabled() {
        let server = OpenAiServerHandle::new();
        let status = server
            .start(
                OpenAiServerConfig {
                    port: 0,
                    sampling_defaults: test_sampling_defaults(),
                    ..Default::default()
                },
                loaded_fake_engine(),
            )
            .unwrap();
        let addr = status.bound_addr.unwrap();
        let response = http_post_json(
            &addr,
            "/v1/chat/completions",
            json!({
                "model": "fake-model",
                "messages": [{"role": "user", "content": "hello"}],
                "disable_thinking": true,
                "max_tokens": 8
            }),
            None,
        );
        let body: Value = serde_json::from_str(response_body(&response)).unwrap();
        assert_eq!(body["choices"][0]["message"]["content"], "ok");
        assert!(body["choices"][0]["message"]["reasoning_content"].is_null());
        server.stop().unwrap();
    }

    #[test]
    fn streams_chat_completion() {
        let server = OpenAiServerHandle::new();
        let status = server
            .start(
                OpenAiServerConfig {
                    port: 0,
                    sampling_defaults: test_sampling_defaults(),
                    ..Default::default()
                },
                loaded_fake_engine(),
            )
            .unwrap();
        let addr = status.bound_addr.unwrap();
        let response = http_post_json(
            &addr,
            "/v1/chat/completions",
            json!({
                "model": "fake-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": true,
                "max_tokens": 8
            }),
            None,
        );
        // Loopback (allow_lan=false) omits the CORS allow-origin header (code-review F-003).
        assert!(!response
            .to_ascii_lowercase()
            .contains("access-control-allow-origin"));
        assert!(response.contains("data: {\"id\":\"chatcmpl-"));
        assert!(response.contains("\"reasoning_content\":\"reason\""));
        assert!(response.contains("\"content\":\"ok\""));
        assert!(response.contains("data: [DONE]"));
        server.stop().unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dropping_stream_before_first_token_cancels_only_its_request() {
        let (events_tx, mut events_rx) = tokio_mpsc::unbounded_channel();
        *CANCEL_EVENTS
            .get_or_init(|| Mutex::new(None))
            .lock()
            .unwrap() = Some(events_tx);
        let engine = EngineHandle::spawn_with_loader(blocking_loader);
        engine
            .load_model(crate::engine::LoadModelRequest {
                source: "/tmp/blocking-model".to_string(),
                display_name: Some("blocking".to_string()),
                quantize: None,
                projector_source: None,
                cuda_graphs: None,
            })
            .unwrap();
        let request = |prompt| {
            serde_json::from_value(json!({
                "model": "blocking",
                "messages": [{"role": "user", "content": prompt}],
                "stream": true,
                "max_tokens": 8
            }))
            .unwrap()
        };

        let held =
            stream_chat_completion(engine.clone(), request("hold"), &test_sampling_defaults())
                .unwrap()
                .into_response();
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(2), events_rx.recv())
                .await
                .unwrap(),
            Some("started before first token")
        );

        // This response is dropped while its generation is still queued. Its flag must already
        // belong to that request, even though the actor has not registered an in-flight flag.
        let queued =
            stream_chat_completion(engine.clone(), request("queued"), &test_sampling_defaults())
                .unwrap()
                .into_response();
        drop(queued);
        tokio::time::sleep(Duration::from_millis(20)).await;

        drop(held);
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(2), events_rx.recv())
                .await
                .unwrap(),
            Some("cancel observed")
        );

        // Retain a completed response while another generation starts. Dropping that older
        // response must not cancel the newer generation through the engine's shared slot.
        let completed =
            stream_chat_completion(engine.clone(), request("after"), &test_sampling_defaults())
                .unwrap()
                .into_response();
        let status_engine = engine.clone();
        tokio::task::spawn_blocking(move || status_engine.status())
            .await
            .unwrap()
            .unwrap();
        let later =
            stream_chat_completion(engine.clone(), request("hold"), &test_sampling_defaults())
                .unwrap()
                .into_response();
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(2), events_rx.recv())
                .await
                .unwrap(),
            Some("started before first token")
        );
        drop(completed);
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(
            events_rx.try_recv().is_err(),
            "late drop cancelled a newer request"
        );
        drop(later);
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(2), events_rx.recv())
                .await
                .unwrap(),
            Some("cancel observed")
        );

        let next = stream_chat_completion(engine, request("after"), &test_sampling_defaults())
            .unwrap()
            .into_response();
        let body = tokio::time::timeout(
            Duration::from_secs(2),
            axum::body::to_bytes(next.into_body(), usize::MAX),
        )
        .await
        .unwrap()
        .unwrap();
        assert!(String::from_utf8_lossy(&body).contains("data: [DONE]"));
        assert!(
            events_rx.try_recv().is_err(),
            "queued request reached the provider"
        );
    }

    #[test]
    fn streams_terminal_native_telemetry_over_http() {
        let server = OpenAiServerHandle::new();
        let status = server
            .start(
                OpenAiServerConfig {
                    port: 0,
                    sampling_defaults: test_sampling_defaults(),
                    ..Default::default()
                },
                loaded_telemetry_engine(),
            )
            .unwrap();
        let addr = status.bound_addr.unwrap();
        let response = http_post_json(
            &addr,
            "/v1/chat/completions",
            json!({
                "model": "fake-telemetry",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": true,
                "mtp": {"mode": "enabled", "draft_tokens": 2},
                "max_tokens": 8
            }),
            None,
        );
        assert!(response.contains(
            "\"chatworks_mtp\":{\"proposed_tokens\":4,\"accepted_tokens\":3,\"target_forwards\":2}"
        ));
        assert!(response.contains("\"chatworks_timings\":{\"prefill_ms\":12,\"decode_ms\":34}"));
        // The decode path reaches API clients on the terminal chunk (sc-24139).
        assert!(response.contains("\"chatworks_decode\":{\"path\":\"mtp\",\"proposer\":\"mtp\""));
        assert!(response.contains("\"fallback_reason\":\"deltanet_state_unstable\""));
        assert!(response.contains("data: [DONE]"));
        server.stop().unwrap();
    }

    #[test]
    fn accepts_vision_sized_json_bodies() {
        let server = OpenAiServerHandle::new();
        let status = server
            .start(
                OpenAiServerConfig {
                    port: 0,
                    sampling_defaults: test_sampling_defaults(),
                    ..Default::default()
                },
                loaded_fake_engine(),
            )
            .unwrap();
        let addr = status.bound_addr.unwrap();
        let response = http_post_json(
            &addr,
            "/v1/chat/completions",
            json!({
                "model": "fake-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 8,
                "vision_payload_padding": "x".repeat(3 * 1024 * 1024)
            }),
            None,
        );
        assert!(response.starts_with("HTTP/1.1 200 OK"), "{response}");
        server.stop().unwrap();
    }

    #[test]
    fn loopback_denies_non_webview_origin() {
        // On loopback (allow_lan=false, the default) a random browser origin gets NO CORS grant,
        // tightening the old static `*` that let any page drive the loopback model (F-003). Uses a
        // non-webview origin so it is not confused with the always-allowed webview origin.
        let server = OpenAiServerHandle::new();
        let status = server
            .start(
                OpenAiServerConfig {
                    port: 0,
                    sampling_defaults: test_sampling_defaults(),
                    ..Default::default()
                },
                loaded_fake_engine(),
            )
            .unwrap();
        let addr = status.bound_addr.unwrap();
        let response =
            http_options_with_origin(&addr, "/v1/chat/completions", "http://evil.example");
        assert!(response.starts_with("HTTP/1.1 204 No Content"));
        assert!(
            !response
                .to_ascii_lowercase()
                .contains("access-control-allow-origin"),
            "loopback must not grant a non-webview origin: {response}"
        );
        server.stop().unwrap();
    }

    #[test]
    fn loopback_grants_app_webview_origin() {
        // The app's own webview always gets a CORS grant even on loopback (allow_lan=false): the
        // in-app chat is a cross-origin browser fetch, so without this the preflight fails and every
        // send dies. This is the regression the first F-003 attempt introduced (PR #30 review).
        let server = OpenAiServerHandle::new();
        let status = server
            .start(
                OpenAiServerConfig {
                    port: 0,
                    sampling_defaults: test_sampling_defaults(),
                    ..Default::default()
                },
                loaded_fake_engine(),
            )
            .unwrap();
        let addr = status.bound_addr.unwrap();
        // The Vite dev webview origin.
        let response =
            http_options_with_origin(&addr, "/v1/chat/completions", "http://127.0.0.1:5173");
        assert!(response.starts_with("HTTP/1.1 204 No Content"));
        assert!(response.contains("access-control-allow-origin: http://127.0.0.1:5173"));
        assert!(response.contains("access-control-allow-methods: GET, POST, OPTIONS"));
        // The packaged macOS webview origin.
        let response = http_options_with_origin(&addr, "/v1/chat/completions", "tauri://localhost");
        assert!(response.contains("access-control-allow-origin: tauri://localhost"));
        // The packaged Windows WebView2 origin needs both a preflight grant and the actual
        // response grant; otherwise its JSON fetch fails before reaching the chat handler.
        let windows_origin = "http://tauri.localhost";
        let response = http_options_with_origin(&addr, "/v1/chat/completions", windows_origin);
        assert!(response.starts_with("HTTP/1.1 204 No Content"));
        assert!(response.contains("access-control-allow-origin: http://tauri.localhost"));
        assert!(response.contains("access-control-allow-methods: GET, POST, OPTIONS"));
        assert!(response.contains("access-control-allow-headers: authorization, content-type"));
        let body = json!({
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 8
        })
        .to_string();
        let actual = http_request(
            &addr,
            format!(
                "POST /v1/chat/completions HTTP/1.1\r\nHost: {addr}\r\nOrigin: {windows_origin}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            ),
        );
        assert!(actual.starts_with("HTTP/1.1 200 OK"), "{actual}");
        assert!(actual.contains("access-control-allow-origin: http://tauri.localhost"));
        let actual_body: Value = serde_json::from_str(response_body(&actual)).unwrap();
        assert_eq!(actual_body["choices"][0]["message"]["content"], "ok");
        server.stop().unwrap();
    }

    #[test]
    fn lan_grants_any_origin_with_vary() {
        // When a user opts into LAN serving, the documented OpenAI-compatible surface is opened to
        // any origin (third-party LAN clients), matching the old permissive behavior. We emit `*`
        // and a `Vary: Origin` so a shared cache can't serve one origin's grant to another (F-003).
        let server = OpenAiServerHandle::new();
        let status = server
            .start(
                OpenAiServerConfig {
                    port: 0,
                    allow_lan: true,
                    host: "127.0.0.1".to_string(),
                    sampling_defaults: test_sampling_defaults(),
                    ..Default::default()
                },
                loaded_fake_engine(),
            )
            .unwrap();
        let addr = status.bound_addr.unwrap();
        let response =
            http_options_with_origin(&addr, "/v1/chat/completions", "http://evil.example");
        assert!(response.starts_with("HTTP/1.1 204 No Content"));
        assert!(response.contains("access-control-allow-origin: *"));
        assert!(response.to_ascii_lowercase().contains("vary: origin"));
        assert!(response.contains("access-control-allow-methods: GET, POST, OPTIONS"));
        server.stop().unwrap();
    }

    #[test]
    fn enforces_auth_only_when_token_is_set() {
        let server = OpenAiServerHandle::new();
        let status = server
            .start(
                OpenAiServerConfig {
                    port: 0,
                    auth_token: Some("secret".to_string()),
                    ..Default::default()
                },
                loaded_fake_engine(),
            )
            .unwrap();
        let addr = status.bound_addr.unwrap();
        let denied = http_get(&addr, "/v1/models", None);
        assert!(denied.starts_with("HTTP/1.1 401 Unauthorized"));
        let allowed = http_get(&addr, "/v1/models", Some("secret"));
        assert!(allowed.starts_with("HTTP/1.1 200 OK"));
        server.stop().unwrap();
    }

    #[test]
    fn threads_offered_tools_into_generate_request() {
        let request: OpenAiChatRequest = serde_json::from_value(json!({
            "messages": [{"role": "user", "content": "weather in Paris?"}],
            "tools": [weather_tool()]
        }))
        .unwrap();
        let generate = request.into_generate(&test_sampling_defaults()).unwrap();
        assert_eq!(generate.tools.len(), 1);
        assert_eq!(generate.tools[0].name, "get_weather");
        assert_eq!(generate.tools[0].description, "Get the weather for a city");
        assert_eq!(
            generate.tools[0].parameters["properties"]["location"]["type"],
            "string"
        );
    }

    #[test]
    fn rejects_non_function_tool_type() {
        let request: OpenAiChatRequest = serde_json::from_value(json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "retrieval", "function": {"name": "x"}}]
        }))
        .unwrap();
        let err = request
            .into_generate(&test_sampling_defaults())
            .unwrap_err();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
    }

    #[test]
    fn round_trips_assistant_tool_calls_and_tool_result() {
        let request: OpenAiChatRequest = serde_json::from_value(json!({
            "messages": [
                {"role": "user", "content": "weather in Paris?"},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {
                        "name": "get_weather", "arguments": "{\"location\":\"Paris\"}"
                    }}
                ]},
                {"role": "tool", "tool_call_id": "call_1", "content": "sunny, 24C"}
            ]
        }))
        .unwrap();
        let generate = request.into_generate(&test_sampling_defaults()).unwrap();
        assert_eq!(generate.messages.len(), 3);
        // The assistant turn carries the tool call and no textual content.
        assert_eq!(generate.messages[1].role, "assistant");
        assert_eq!(generate.messages[1].content, "");
        assert_eq!(generate.messages[1].tool_calls.len(), 1);
        assert_eq!(generate.messages[1].tool_calls[0].name, "get_weather");
        assert_eq!(
            generate.messages[1].tool_calls[0].arguments["location"],
            "Paris"
        );
        // The tool result round-trips as a `tool`-role text turn (already mapped to Role::Tool).
        assert_eq!(generate.messages[2].role, "tool");
        assert_eq!(generate.messages[2].content, "sunny, 24C");
    }

    #[test]
    fn rejects_non_object_tool_call_arguments() {
        let request: OpenAiChatRequest = serde_json::from_value(json!({
            "messages": [{"role": "assistant", "tool_calls": [
                {"type": "function", "function": {"name": "get_weather", "arguments": "not json"}}
            ]}]
        }))
        .unwrap();
        let err = request
            .into_generate(&test_sampling_defaults())
            .unwrap_err();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
    }

    #[test]
    fn returns_tool_calls_with_finish_reason() {
        let server = OpenAiServerHandle::new();
        let status = server
            .start(
                OpenAiServerConfig {
                    port: 0,
                    sampling_defaults: test_sampling_defaults(),
                    ..Default::default()
                },
                loaded_tool_engine(),
            )
            .unwrap();
        let addr = status.bound_addr.unwrap();
        let response = http_post_json(
            &addr,
            "/v1/chat/completions",
            json!({
                "model": "fake-tools",
                "messages": [{"role": "user", "content": "weather in Paris?"}],
                "tools": [weather_tool()],
                "max_tokens": 16
            }),
            None,
        );
        let body: Value = serde_json::from_str(response_body(&response)).unwrap();
        assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
        // A pure tool-call turn carries `content: null` (present, OpenAI-style), not "".
        assert!(body["choices"][0]["message"]["content"].is_null());
        let call = &body["choices"][0]["message"]["tool_calls"][0];
        assert_eq!(call["type"], "function");
        assert!(call["id"].as_str().unwrap().starts_with("call_"));
        assert_eq!(call["function"]["name"], "get_weather");
        // OpenAI carries arguments as a JSON-encoded string; it must decode to the call args.
        let args: Value =
            serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["location"], "Paris");
        server.stop().unwrap();
    }

    #[test]
    fn streams_tool_calls_at_finish() {
        let server = OpenAiServerHandle::new();
        let status = server
            .start(
                OpenAiServerConfig {
                    port: 0,
                    sampling_defaults: test_sampling_defaults(),
                    ..Default::default()
                },
                loaded_tool_engine(),
            )
            .unwrap();
        let addr = status.bound_addr.unwrap();
        let response = http_post_json(
            &addr,
            "/v1/chat/completions",
            json!({
                "model": "fake-tools",
                "messages": [{"role": "user", "content": "weather in Paris?"}],
                "tools": [weather_tool()],
                "stream": true,
                "max_tokens": 16
            }),
            None,
        );
        assert!(response.contains("\"finish_reason\":\"tool_calls\""));
        assert!(response.contains("\"name\":\"get_weather\""));
        assert!(response.contains("\"index\":0"));
        assert!(response.contains("Paris"));
        assert!(response.contains("data: [DONE]"));
        server.stop().unwrap();
    }

    #[test]
    fn rejects_tools_on_provider_without_tool_support() {
        let server = OpenAiServerHandle::new();
        let status = server
            .start(
                OpenAiServerConfig {
                    port: 0,
                    sampling_defaults: test_sampling_defaults(),
                    ..Default::default()
                },
                loaded_fake_engine(),
            )
            .unwrap();
        let addr = status.bound_addr.unwrap();
        let response = http_post_json(
            &addr,
            "/v1/chat/completions",
            json!({
                "model": "fake-model",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [weather_tool()],
                "max_tokens": 8
            }),
            None,
        );
        assert!(response.starts_with("HTTP/1.1 400 Bad Request"));
        let body: Value = serde_json::from_str(response_body(&response)).unwrap();
        let message = body["error"]["message"].as_str().unwrap_or_default();
        assert!(
            message.contains("tool"),
            "expected a tool-support error, got: {message}"
        );
        server.stop().unwrap();
    }
}
