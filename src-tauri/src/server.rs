use std::convert::Infallible;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::{mpsc, Mutex};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

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
use crate::engine::{
    EngineHandle, GenerateMessage, GenerateRequest, GenerateResponse, GenerateTool,
    GenerateToolCall, GenerateVideo, LoadedModelStatus, SamplingRequest, StreamChannel,
    StreamPayload, ThinkingRequest, UsagePayload,
};

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
        let bind = validate_config(&config)?;
        let auth_token = normalize_token(config.auth_token.clone());
        let server_config = OpenAiServerConfig {
            auth_token,
            ..config
        };
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
        openai_router(engine, config.auth_token, config.sampling_defaults),
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
        })
        .layer(DefaultBodyLimit::max(OPENAI_JSON_BODY_LIMIT_BYTES))
        .layer(axum::middleware::from_fn(apply_cors_headers))
}

async fn cors_preflight() -> StatusCode {
    StatusCode::NO_CONTENT
}

async fn apply_cors_headers(
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    let mut response = next.run(request).await;
    let headers = response.headers_mut();
    headers.insert(
        axum::http::header::ACCESS_CONTROL_ALLOW_ORIGIN,
        HeaderValue::from_static("*"),
    );
    headers.insert(
        axum::http::header::ACCESS_CONTROL_ALLOW_METHODS,
        HeaderValue::from_static("GET, POST, OPTIONS"),
    );
    headers.insert(
        axum::http::header::ACCESS_CONTROL_ALLOW_HEADERS,
        HeaderValue::from_static("authorization, content-type"),
    );
    response
}

#[derive(Clone)]
struct ApiState {
    engine: EngineHandle,
    auth_token: Option<String>,
    sampling_defaults: SamplingDefaults,
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
    if request.stream {
        let stream = stream_chat_completion(state.engine, request, &state.sampling_defaults)?;
        Ok(stream.into_response())
    } else {
        let model = request.model_name();
        let generate_request = request.into_generate(&state.sampling_defaults)?;
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

    tokio::task::spawn_blocking(move || {
        let result = engine.generate(generate_request, |payload| {
            if let StreamPayload::Token { text, channel, .. } = payload {
                let chunk = match channel {
                    StreamChannel::Content => {
                        OpenAiChatChunk::token(id.clone(), created, model.clone(), text)
                    }
                    StreamChannel::Thinking => {
                        OpenAiChatChunk::reasoning(id.clone(), created, model.clone(), text)
                    }
                };
                let _ = tx.blocking_send(Ok(sse_json(&chunk)));
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
                );
                let _ = tx.blocking_send(Ok(sse_json(&finish)));
                let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
            }
            Err(error) => {
                let _ = tx.blocking_send(Ok(sse_json(&OpenAiErrorBody::server(error))));
                let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
            }
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
    let expected = format!("Bearer {token}");
    match headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
    {
        Some(actual) if actual == expected => Ok(()),
        _ => Err(ApiError::auth("missing or invalid bearer token")),
    }
}

fn validate_config(config: &OpenAiServerConfig) -> ServerResult<SocketAddr> {
    let host = config
        .host
        .parse::<IpAddr>()
        .map_err(|_| format!("invalid bind host '{}'", config.host))?;
    if is_unspecified(host) && !config.allow_lan {
        return Err("binding to 0.0.0.0 requires allow_lan=true".to_string());
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
    max_tokens: Option<u32>,
    #[serde(default)]
    max_completion_tokens: Option<u32>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    stop: Option<StopValue>,
    #[serde(default)]
    disable_thinking: Option<bool>,
    /// Tools / functions offered to the model, in the OpenAI function-tool shape
    /// (`{"type":"function","function":{"name","description","parameters"}}`). Threaded to the
    /// provider, which rejects them with a 400 if it does not advertise tool support.
    #[serde(default)]
    tools: Option<Vec<OpenAiTool>>,
}

impl OpenAiChatRequest {
    fn model_name(&self) -> String {
        self.model
            .clone()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| "chatworks".to_string())
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
                tool_calls: Vec::new(),
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
        let disable_thinking = self.disable_thinking.unwrap_or(defaults.disable_thinking);
        Ok(GenerateRequest {
            messages,
            sampling: SamplingRequest {
                temperature: Some(self.temperature.unwrap_or(defaults.temperature)),
                top_p: Some(self.top_p.unwrap_or(defaults.top_p)),
                top_k: None,
                repetition_penalty: None,
                repetition_context: None,
            },
            max_new_tokens: self
                .max_completion_tokens
                .or(self.max_tokens)
                .unwrap_or(defaults.max_tokens),
            seed: self.seed,
            stop: self.stop.map(StopValue::into_vec).unwrap_or_default(),
            thinking: if disable_thinking {
                ThinkingRequest::Disabled
            } else {
                ThinkingRequest::Auto
            },
            tools,
        })
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
}

impl OpenAiChatMessage {
    fn into_generate(self) -> Result<GenerateMessage, ApiError> {
        let (content, images, videos) = match self.content {
            Some(content) => content.into_parts()?,
            None => (String::new(), Vec::new(), Vec::new()),
        };
        Ok(GenerateMessage {
            role: self.role,
            content,
            images,
            videos,
            tool_calls: self
                .tool_calls
                .into_iter()
                .map(OpenAiToolCall::into_generate)
                .collect::<Result<Vec<_>, _>>()?,
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
    /// Split OpenAI content into concatenated text, the ordered image-URL attachments, and the
    /// ordered video attachments. A plain string is text with no images/videos (the text path stays
    /// byte-identical).
    ///
    /// **Video representation (sc-8081).** There is no standard OpenAI `image_url` analog for video.
    /// We accept a **`video_url` content part carrying pre-sampled frames** plus optional per-frame
    /// timestamps:
    /// ```json
    /// { "type": "video_url",
    ///   "video_url": {
    ///     "frames": ["data:image/jpeg;base64,…", "data:image/jpeg;base64,…"],
    ///     "timestamps": [0.0, 0.5],   // optional; seconds, one per frame
    ///     "fps": 2.0                  // optional; used to derive timestamps when absent
    ///   } }
    /// ```
    /// The host (the ChatWorks frontend) samples frames client-side, so v1 needs **no heavy
    /// server-side video-file decoder** (arbitrary `.mp4` decode is a tracked follow-up). Each frame
    /// is an image data URL decoded exactly like an `image_url`. If `timestamps` is omitted it is
    /// derived from `fps` (`i / fps`) or, lacking both, frame index seconds (`i`, i.e. 1 fps) — the
    /// engine forwards these straight into `VideoRef`, which drives Text–Timestamp Alignment.
    #[allow(clippy::type_complexity)]
    fn into_parts(self) -> Result<(String, Vec<String>, Vec<GenerateVideo>), ApiError> {
        match self {
            Self::Text(value) => Ok((value, Vec::new(), Vec::new())),
            Self::Parts(parts) => {
                let mut text = String::new();
                let mut images = Vec::new();
                let mut videos = Vec::new();
                for part in parts {
                    match part.kind.as_str() {
                        "text" => text.push_str(&part.text.unwrap_or_default()),
                        "image_url" => {
                            let url = part.image_url.map(|image| image.url).ok_or_else(|| {
                                ApiError::bad_request("image_url part is missing its url")
                            })?;
                            images.push(url);
                        }
                        "video_url" => {
                            let video = part.video_url.ok_or_else(|| {
                                ApiError::bad_request("video_url part is missing its video_url")
                            })?;
                            if video.frames.is_empty() {
                                return Err(ApiError::bad_request(
                                    "video_url part must carry at least one frame",
                                ));
                            }
                            // Derive timestamps when absent: explicit > fps-derived > 1-fps index.
                            let n = video.frames.len();
                            let timestamps = match video.timestamps {
                                Some(ts) if ts.len() == n => ts,
                                Some(ts) => {
                                    return Err(ApiError::bad_request(format!(
                                        "video_url timestamps length {} != frame count {n}",
                                        ts.len()
                                    )))
                                }
                                None => {
                                    let fps = video.fps.filter(|f| *f > 0.0).unwrap_or(1.0);
                                    (0..n).map(|i| i as f32 / fps).collect()
                                }
                            };
                            videos.push(GenerateVideo { frames: video.frames, timestamps });
                        }
                        other => {
                            return Err(ApiError::bad_request(format!(
                                "unsupported content part type '{other}'"
                            )))
                        }
                    }
                }
                Ok((text, images, videos))
            }
        }
    }
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
    /// Sampled frames, in temporal order, each a `data:image/…;base64,…` URL (or bare base64).
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
}

impl OpenAiChatResponse {
    fn from_generate(model: String, response: GenerateResponse) -> Self {
        let GenerateResponse {
            text,
            thinking,
            tool_calls,
            usage,
            finish_reason,
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
        }
    }
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
        }
    }

    fn finish(
        id: String,
        created: u64,
        model: String,
        finish_reason: String,
        usage: Option<OpenAiUsage>,
        tool_calls: Vec<OpenAiToolCallDelta>,
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
    format!("call_{}_{index}", timestamp_nanos())
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
    format!("chatcmpl-{}", timestamp_nanos())
}

fn created_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

fn timestamp_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_llm::{
        Channel, Content, FinishReason, LoadSpec, Role, TextLlm, TextLlmCapabilities,
        TextLlmDescriptor, TextLlmOutput, TextLlmRequest, ThinkingMode, Usage,
    };
    use serde_json::{json, Value};

    struct FakeProvider {
        descriptor: TextLlmDescriptor,
    }

    impl TextLlm for FakeProvider {
        fn descriptor(&self) -> &TextLlmDescriptor {
            &self.descriptor
        }

        fn validate(&self, req: &TextLlmRequest) -> core_llm::Result<()> {
            self.descriptor
                .capabilities
                .validate_request(&self.descriptor.id, req)
        }

        fn generate(
            &self,
            req: &TextLlmRequest,
            on_event: &mut dyn FnMut(core_llm::StreamEvent),
        ) -> core_llm::Result<TextLlmOutput> {
            self.validate(req)?;
            assert_eq!(req.messages.len(), 1);
            assert_eq!(req.messages[0].role, Role::User);
            assert_eq!(
                req.messages[0].content,
                vec![Content::Text("hello".to_string())]
            );
            let thinking = if req.thinking == ThinkingMode::Disabled {
                None
            } else {
                on_event(core_llm::StreamEvent::Token {
                    id: 9,
                    text: "reason".to_string(),
                    index: 0,
                    channel: Channel::Thinking,
                });
                Some("reason".to_string())
            };
            on_event(core_llm::StreamEvent::Token {
                id: 1,
                text: "ok".to_string(),
                index: 1,
                channel: Channel::Content,
            });
            let usage = Usage {
                prompt_tokens: 2,
                generated_tokens: 1,
            };
            on_event(core_llm::StreamEvent::Done {
                finish_reason: FinishReason::Stop,
                usage,
            });
            Ok(TextLlmOutput {
                text: "ok".to_string(),
                thinking,
                tool_calls: Vec::new(),
                usage,
                finish_reason: Some(FinishReason::Stop),
            })
        }
    }

    fn fake_loader(_: &LoadSpec) -> core_llm::Result<Box<dyn TextLlm>> {
        Ok(Box::new(FakeProvider {
            descriptor: TextLlmDescriptor {
                id: "fake".to_string(),
                family: "test".to_string(),
                backend: "unit".to_string(),
                capabilities: TextLlmCapabilities {
                    supports_system_prompt: true,
                    supports_thinking: true,
                    max_new_tokens: 8,
                    ..Default::default()
                },
            },
        }))
    }

    fn loaded_fake_engine() -> EngineHandle {
        let engine = EngineHandle::spawn_with_loader(fake_loader);
        engine
            .load_model(crate::engine::LoadModelRequest {
                source: "/tmp/fake-model".to_string(),
                display_name: Some("fake-model".to_string()),
                quantize: None,
            })
            .unwrap();
        engine
    }

    /// A tool-capable provider that echoes the offered tools back as a single `get_weather(Paris)`
    /// call, so the OpenAI tool_calls + finish_reason path can be exercised without real weights.
    struct FakeToolProvider {
        descriptor: TextLlmDescriptor,
    }

    impl TextLlm for FakeToolProvider {
        fn descriptor(&self) -> &TextLlmDescriptor {
            &self.descriptor
        }

        fn validate(&self, req: &TextLlmRequest) -> core_llm::Result<()> {
            self.descriptor
                .capabilities
                .validate_request(&self.descriptor.id, req)
        }

        fn generate(
            &self,
            req: &TextLlmRequest,
            on_event: &mut dyn FnMut(core_llm::StreamEvent),
        ) -> core_llm::Result<TextLlmOutput> {
            self.validate(req)?;
            // The tools must have been threaded through to the core request.
            assert_eq!(req.tools.len(), 1);
            assert_eq!(req.tools[0].name, "get_weather");
            let usage = Usage {
                prompt_tokens: 3,
                generated_tokens: 4,
            };
            on_event(core_llm::StreamEvent::Done {
                finish_reason: FinishReason::Stop,
                usage,
            });
            let mut arguments = serde_json::Map::new();
            arguments.insert("location".to_string(), json!("Paris"));
            Ok(TextLlmOutput {
                text: String::new(),
                thinking: None,
                tool_calls: vec![core_llm::ToolCall::new("get_weather", arguments)],
                usage,
                finish_reason: Some(FinishReason::Stop),
            })
        }
    }

    fn fake_tool_loader(_: &LoadSpec) -> core_llm::Result<Box<dyn TextLlm>> {
        Ok(Box::new(FakeToolProvider {
            descriptor: TextLlmDescriptor {
                id: "fake-tools".to_string(),
                family: "test".to_string(),
                backend: "unit".to_string(),
                capabilities: TextLlmCapabilities {
                    supports_system_prompt: true,
                    supports_tools: true,
                    max_new_tokens: 64,
                    ..Default::default()
                },
            },
        }))
    }

    fn loaded_tool_engine() -> EngineHandle {
        let engine = EngineHandle::spawn_with_loader(fake_tool_loader);
        engine
            .load_model(crate::engine::LoadModelRequest {
                source: "/tmp/fake-tools".to_string(),
                display_name: Some("fake-tools".to_string()),
                quantize: None,
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
        let defaults = SamplingDefaults { system_prompt: String::new(), ..Default::default() };
        let generate = request.into_generate(&defaults).unwrap();
        let msg = &generate.messages[0];
        assert_eq!(msg.content, "what happens");
        assert!(msg.images.is_empty());
        assert_eq!(msg.videos.len(), 1);
        assert_eq!(msg.videos[0].frames.len(), 2);
        assert_eq!(msg.videos[0].timestamps, vec![0.0, 0.5]);
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
        let defaults = SamplingDefaults { system_prompt: String::new(), ..Default::default() };
        let generate = request.into_generate(&defaults).unwrap();
        assert_eq!(generate.messages[0].videos[0].timestamps, vec![0.0, 0.5, 1.0, 1.5]);
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
        assert!(mismatched.into_generate(&SamplingDefaults::default()).is_err());
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
        };
        let generate = request.into_generate(&defaults).unwrap();
        assert_eq!(generate.messages.len(), 2);
        assert_eq!(generate.messages[0].role, "system");
        assert_eq!(generate.messages[0].content, "be terse");
        assert_eq!(generate.sampling.temperature, Some(0.3));
        assert_eq!(generate.sampling.top_p, Some(0.8));
        assert_eq!(generate.max_new_tokens, 64);
        assert!(matches!(generate.thinking, ThinkingRequest::Disabled));
    }

    #[test]
    fn maps_disable_thinking_to_core_thinking_mode() {
        let request: OpenAiChatRequest = serde_json::from_value(json!({
            "messages": [{"role": "user", "content": "hello"}],
            "disable_thinking": true
        }))
        .unwrap();

        let generate = request.into_generate(&SamplingDefaults::default()).unwrap();
        assert!(matches!(generate.thinking, ThinkingRequest::Disabled));
    }

    #[test]
    fn empty_auth_token_disables_auth() {
        assert_eq!(normalize_token(Some("  ".to_string())), None);
        assert_eq!(
            normalize_token(Some(" token ".to_string())),
            Some("token".to_string())
        );
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

    fn http_options(addr: &str, path: &str) -> String {
        http_request(
            addr,
            format!(
                "OPTIONS {path} HTTP/1.1\r\nHost: {addr}\r\nOrigin: http://127.0.0.1:5173\r\nAccess-Control-Request-Method: POST\r\nAccess-Control-Request-Headers: content-type\r\nConnection: close\r\n\r\n"
            ),
        )
    }

    fn response_body(response: &str) -> &str {
        response.split("\r\n\r\n").nth(1).unwrap_or_default()
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
        assert!(response.contains("access-control-allow-origin: *"));
        assert!(response.contains("data: {\"id\":\"chatcmpl-"));
        assert!(response.contains("\"reasoning_content\":\"reason\""));
        assert!(response.contains("\"content\":\"ok\""));
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
    fn accepts_cors_preflight() {
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
        let response = http_options(&addr, "/v1/chat/completions");
        assert!(response.starts_with("HTTP/1.1 204 No Content"));
        assert!(response.contains("access-control-allow-origin: *"));
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
