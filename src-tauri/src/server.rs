use std::convert::Infallible;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::{mpsc, Mutex};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::runtime::Runtime;
use tokio::sync::{mpsc as tokio_mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;

use crate::engine::{
    EngineHandle, GenerateMessage, GenerateRequest, GenerateResponse, LoadedModelStatus,
    SamplingRequest, StreamPayload, UsagePayload,
};

pub const DEFAULT_OPENAI_HOST: &str = "127.0.0.1";
pub const DEFAULT_OPENAI_PORT: u16 = 8000;

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
}

impl Default for OpenAiServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            allow_lan: false,
            auth_token: None,
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
    axum::serve(listener, openai_router(engine, config.auth_token))
        .with_graceful_shutdown(async {
            let _ = shutdown_rx.await;
        })
        .await
        .map_err(|error| error.to_string())
}

fn openai_router(engine: EngineHandle, auth_token: Option<String>) -> Router {
    Router::new()
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(ApiState { engine, auth_token })
}

#[derive(Clone)]
struct ApiState {
    engine: EngineHandle,
    auth_token: Option<String>,
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
        let stream = stream_chat_completion(state.engine, request)?;
        Ok(stream.into_response())
    } else {
        let model = request.model_name();
        let generate_request = request.into_generate()?;
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
) -> Result<impl IntoResponse, ApiError> {
    let model = request.model_name();
    let id = completion_id();
    let created = created_timestamp();
    let generate_request = request.into_generate()?;
    let (tx, rx) = tokio_mpsc::channel::<Result<Event, Infallible>>(32);

    tokio::task::spawn_blocking(move || {
        let result = engine.generate(generate_request, |payload| {
            if let StreamPayload::Token { text, .. } = payload {
                let chunk = OpenAiChatChunk::token(id.clone(), created, model.clone(), text);
                let _ = tx.blocking_send(Ok(sse_json(&chunk)));
            }
        });

        match result {
            Ok(response) => {
                let finish = OpenAiChatChunk::finish(
                    id.clone(),
                    created,
                    model.clone(),
                    response.finish_reason,
                    Some(OpenAiUsage::from(response.usage)),
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
}

impl OpenAiChatRequest {
    fn model_name(&self) -> String {
        self.model
            .clone()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| "chatworks".to_string())
    }

    fn into_generate(self) -> Result<GenerateRequest, ApiError> {
        if self.messages.is_empty() {
            return Err(ApiError::bad_request("messages must not be empty"));
        }
        Ok(GenerateRequest {
            messages: self
                .messages
                .into_iter()
                .map(OpenAiChatMessage::into_generate)
                .collect::<Result<Vec<_>, _>>()?,
            sampling: SamplingRequest {
                temperature: self.temperature,
                top_p: self.top_p,
                top_k: None,
                repetition_penalty: None,
                repetition_context: None,
            },
            max_new_tokens: self
                .max_completion_tokens
                .or(self.max_tokens)
                .unwrap_or(512),
            seed: self.seed,
            stop: self.stop.map(StopValue::into_vec).unwrap_or_default(),
        })
    }
}

#[derive(Deserialize)]
struct OpenAiChatMessage {
    role: String,
    content: OpenAiMessageContent,
}

impl OpenAiChatMessage {
    fn into_generate(self) -> Result<GenerateMessage, ApiError> {
        Ok(GenerateMessage {
            role: self.role,
            content: self.content.into_text()?,
        })
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum OpenAiMessageContent {
    Text(String),
    Parts(Vec<OpenAiContentPart>),
}

impl OpenAiMessageContent {
    fn into_text(self) -> Result<String, ApiError> {
        match self {
            Self::Text(value) => Ok(value),
            Self::Parts(parts) => parts
                .into_iter()
                .map(OpenAiContentPart::into_text)
                .collect::<Result<Vec<_>, _>>()
                .map(|values| values.join("")),
        }
    }
}

#[derive(Deserialize)]
struct OpenAiContentPart {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: Option<String>,
}

impl OpenAiContentPart {
    fn into_text(self) -> Result<String, ApiError> {
        if self.kind == "text" {
            Ok(self.text.unwrap_or_default())
        } else {
            Err(ApiError::bad_request(format!(
                "unsupported content part type '{}'",
                self.kind
            )))
        }
    }
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
        Self {
            id: completion_id(),
            object: "chat.completion",
            created: created_timestamp(),
            model,
            choices: vec![OpenAiChatChoice {
                index: 0,
                message: Some(OpenAiResponseMessage {
                    role: "assistant",
                    content: response.text,
                }),
                delta: None,
                finish_reason: Some(response.finish_reason),
            }],
            usage: OpenAiUsage::from(response.usage),
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
    ) -> Self {
        Self {
            id,
            object: "chat.completion.chunk",
            created,
            model,
            choices: vec![OpenAiChatChoice {
                index: 0,
                message: None,
                delta: Some(OpenAiDelta { content: None }),
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
    content: String,
}

#[derive(Serialize)]
struct OpenAiDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
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
        Content, FinishReason, LoadSpec, Role, TextLlm, TextLlmCapabilities, TextLlmDescriptor,
        TextLlmOutput, TextLlmRequest, Usage,
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
            on_event(core_llm::StreamEvent::Token {
                id: 1,
                text: "ok".to_string(),
                index: 0,
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

        let generate = request.into_generate().unwrap();
        assert_eq!(generate.messages.len(), 1);
        assert_eq!(generate.messages[0].role, "user");
        assert_eq!(generate.messages[0].content, "hello");
        assert_eq!(generate.sampling.temperature, Some(0.2));
        assert_eq!(generate.sampling.top_p, Some(0.9));
        assert_eq!(generate.max_new_tokens, 7);
        assert_eq!(generate.seed, Some(42));
        assert_eq!(generate.stop, vec!["END"]);
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
        assert_eq!(body["usage"]["completion_tokens"], 1);
        server.stop().unwrap();
    }

    #[test]
    fn streams_chat_completion() {
        let server = OpenAiServerHandle::new();
        let status = server
            .start(
                OpenAiServerConfig {
                    port: 0,
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
        assert!(response.contains("data: {\"id\":\"chatcmpl-"));
        assert!(response.contains("\"content\":\"ok\""));
        assert!(response.contains("data: [DONE]"));
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
}
