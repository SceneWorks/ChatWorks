use std::path::Path;
use std::sync::mpsc;
use std::thread;

use core_llm::{
    load_for_model, CancelFlag, Content, FinishReason, LoadSpec, Message, Quantize, Role, Sampling,
    StreamEvent, TextLlm, TextLlmCapabilities, TextLlmDescriptor, TextLlmRequest, Usage,
};
use serde::{Deserialize, Serialize};

pub type EngineResult<T> = Result<T, String>;

type Loader = fn(&LoadSpec) -> core_llm::Result<Box<dyn TextLlm>>;

#[derive(Clone)]
pub struct EngineHandle {
    tx: mpsc::Sender<EngineCommand>,
}

impl EngineHandle {
    pub fn spawn() -> Self {
        Self::spawn_with_loader(load_for_model)
    }

    pub(crate) fn spawn_with_loader(loader: Loader) -> Self {
        let (tx, rx) = mpsc::channel();
        thread::Builder::new()
            .name("chatworks-engine".to_string())
            .spawn(move || EngineActor::new(loader, rx).run())
            .expect("failed to start ChatWorks engine thread");
        Self { tx }
    }

    pub fn load_model(&self, request: LoadModelRequest) -> EngineResult<EngineStatus> {
        let (reply_tx, reply_rx) = mpsc::channel();
        self.send(EngineCommand::Load { request, reply_tx })?;
        recv_reply(reply_rx)
    }

    pub fn unload_model(&self) -> EngineResult<EngineStatus> {
        let (reply_tx, reply_rx) = mpsc::channel();
        self.send(EngineCommand::Unload { reply_tx })?;
        recv_reply(reply_rx)
    }

    pub fn status(&self) -> EngineResult<EngineStatus> {
        let (reply_tx, reply_rx) = mpsc::channel();
        self.send(EngineCommand::Status { reply_tx })?;
        recv_reply(reply_rx)
    }

    pub fn generate(
        &self,
        request: GenerateRequest,
        mut on_event: impl FnMut(StreamPayload),
    ) -> EngineResult<GenerateResponse> {
        let (reply_tx, reply_rx) = mpsc::channel();
        let (event_tx, event_rx) = mpsc::channel();
        self.send(EngineCommand::Generate {
            request,
            event_tx,
            reply_tx,
        })?;
        while let Ok(event) = event_rx.recv() {
            on_event(event);
        }
        recv_reply(reply_rx)
    }

    fn send(&self, command: EngineCommand) -> EngineResult<()> {
        self.tx
            .send(command)
            .map_err(|_| "engine thread is not running".to_string())
    }
}

fn recv_reply<T>(rx: mpsc::Receiver<EngineResult<T>>) -> EngineResult<T> {
    rx.recv()
        .map_err(|_| "engine thread stopped before replying".to_string())?
}

enum EngineCommand {
    Load {
        request: LoadModelRequest,
        reply_tx: mpsc::Sender<EngineResult<EngineStatus>>,
    },
    Unload {
        reply_tx: mpsc::Sender<EngineResult<EngineStatus>>,
    },
    Status {
        reply_tx: mpsc::Sender<EngineResult<EngineStatus>>,
    },
    Generate {
        request: GenerateRequest,
        event_tx: mpsc::Sender<StreamPayload>,
        reply_tx: mpsc::Sender<EngineResult<GenerateResponse>>,
    },
}

struct EngineActor {
    loader: Loader,
    rx: mpsc::Receiver<EngineCommand>,
    loaded: Option<LoadedModel>,
}

impl EngineActor {
    fn new(loader: Loader, rx: mpsc::Receiver<EngineCommand>) -> Self {
        Self {
            loader,
            rx,
            loaded: None,
        }
    }

    fn run(mut self) {
        while let Ok(command) = self.rx.recv() {
            match command {
                EngineCommand::Load { request, reply_tx } => {
                    let _ = reply_tx.send(self.load(request));
                }
                EngineCommand::Unload { reply_tx } => {
                    self.loaded = None;
                    let _ = reply_tx.send(Ok(self.status()));
                }
                EngineCommand::Status { reply_tx } => {
                    let _ = reply_tx.send(Ok(self.status()));
                }
                EngineCommand::Generate {
                    request,
                    event_tx,
                    reply_tx,
                } => {
                    let result = self.generate(request, event_tx);
                    let _ = reply_tx.send(result);
                }
            }
        }
    }

    fn load(&mut self, request: LoadModelRequest) -> EngineResult<EngineStatus> {
        if request.source.trim().is_empty() {
            return Err("model source is required".to_string());
        }
        let spec = LoadSpec {
            source: request.source.clone(),
            quantize: request.quantize.map(Into::into),
        };
        let provider = (self.loader)(&spec).map_err(|error| error.to_string())?;
        let descriptor = provider.descriptor().clone();
        self.loaded = Some(LoadedModel {
            source: request.source,
            display_name: request.display_name,
            quantize: request.quantize,
            provider,
            descriptor,
        });
        Ok(self.status())
    }

    fn generate(
        &mut self,
        request: GenerateRequest,
        event_tx: mpsc::Sender<StreamPayload>,
    ) -> EngineResult<GenerateResponse> {
        let loaded = self
            .loaded
            .as_ref()
            .ok_or_else(|| "no model loaded".to_string())?;
        let core_request = request.into_core()?;
        let output = loaded
            .provider
            .generate(&core_request, &mut |event| {
                let _ = event_tx.send(StreamPayload::from(event));
            })
            .map_err(|error| error.to_string())?;
        Ok(GenerateResponse {
            text: output.text,
            usage: UsagePayload::from(output.usage),
            finish_reason: output
                .finish_reason
                .map(finish_reason_name)
                .unwrap_or("unknown")
                .to_string(),
        })
    }

    fn status(&self) -> EngineStatus {
        EngineStatus {
            loaded: self.loaded.as_ref().map(LoadedModel::status),
            providers: core_llm::textllms()
                .map(|registration| ProviderSummary::from((registration.descriptor)()))
                .collect(),
        }
    }
}

struct LoadedModel {
    source: String,
    display_name: Option<String>,
    quantize: Option<QuantizeRequest>,
    provider: Box<dyn TextLlm>,
    descriptor: TextLlmDescriptor,
}

impl LoadedModel {
    fn status(&self) -> LoadedModelStatus {
        LoadedModelStatus {
            source: self.source.clone(),
            name: self
                .display_name
                .clone()
                .unwrap_or_else(|| model_name(&self.source)),
            quantize: self.quantize,
            provider: ProviderSummary::from(self.descriptor.clone()),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LoadModelRequest {
    pub source: String,
    #[serde(default)]
    pub display_name: Option<String>,
    #[serde(default)]
    pub quantize: Option<QuantizeRequest>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum QuantizeRequest {
    Q4,
    Q8,
}

impl From<QuantizeRequest> for Quantize {
    fn from(value: QuantizeRequest) -> Self {
        match value {
            QuantizeRequest::Q4 => Quantize::Q4,
            QuantizeRequest::Q8 => Quantize::Q8,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GenerateRequest {
    pub messages: Vec<GenerateMessage>,
    #[serde(default)]
    pub sampling: SamplingRequest,
    #[serde(default = "default_max_new_tokens")]
    pub max_new_tokens: u32,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stop: Vec<String>,
}

impl GenerateRequest {
    fn into_core(self) -> EngineResult<TextLlmRequest> {
        if self.messages.is_empty() {
            return Err("messages must not be empty".to_string());
        }
        Ok(TextLlmRequest {
            messages: self
                .messages
                .into_iter()
                .map(GenerateMessage::into_core)
                .collect::<EngineResult<Vec<_>>>()?,
            sampling: self.sampling.into_core(),
            max_new_tokens: self.max_new_tokens,
            seed: self.seed,
            constraint: None,
            stop: self.stop,
            cancel: CancelFlag::new(),
        })
    }
}

fn default_max_new_tokens() -> u32 {
    512
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GenerateMessage {
    pub role: String,
    pub content: String,
}

impl GenerateMessage {
    fn into_core(self) -> EngineResult<Message> {
        Ok(Message {
            role: role_from_str(&self.role)?,
            content: vec![Content::Text(self.content)],
        })
    }
}

fn role_from_str(role: &str) -> EngineResult<Role> {
    match role {
        "system" | "developer" => Ok(Role::System),
        "user" => Ok(Role::User),
        "assistant" => Ok(Role::Assistant),
        "tool" => Ok(Role::Tool),
        other => Err(format!("unsupported message role '{other}'")),
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct SamplingRequest {
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub repetition_context: Option<usize>,
}

impl SamplingRequest {
    fn into_core(self) -> Sampling {
        let mut sampling = Sampling::default();
        if let Some(value) = self.temperature {
            sampling.temperature = value;
        }
        if let Some(value) = self.top_p {
            sampling.top_p = value;
        }
        if let Some(value) = self.top_k {
            sampling.top_k = value;
        }
        if let Some(value) = self.repetition_penalty {
            sampling.repetition_penalty = value;
        }
        if let Some(value) = self.repetition_context {
            sampling.repetition_context = value;
        }
        sampling
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct EngineStatus {
    pub loaded: Option<LoadedModelStatus>,
    pub providers: Vec<ProviderSummary>,
}

#[derive(Clone, Debug, Serialize)]
pub struct LoadedModelStatus {
    pub source: String,
    pub name: String,
    pub quantize: Option<QuantizeRequest>,
    pub provider: ProviderSummary,
}

#[derive(Clone, Debug, Serialize)]
pub struct ProviderSummary {
    pub id: String,
    pub family: String,
    pub backend: String,
    pub capabilities: CapabilitySummary,
}

impl From<TextLlmDescriptor> for ProviderSummary {
    fn from(value: TextLlmDescriptor) -> Self {
        Self {
            id: value.id,
            family: value.family,
            backend: value.backend,
            capabilities: CapabilitySummary::from(value.capabilities),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct CapabilitySummary {
    pub max_context_tokens: usize,
    pub max_new_tokens: u32,
    pub supports_system_prompt: bool,
    pub supports_vision: bool,
    pub supported_constraints: Vec<String>,
}

impl From<TextLlmCapabilities> for CapabilitySummary {
    fn from(value: TextLlmCapabilities) -> Self {
        Self {
            max_context_tokens: value.max_context_tokens,
            max_new_tokens: value.max_new_tokens,
            supports_system_prompt: value.supports_system_prompt,
            supports_vision: value.supports_vision,
            supported_constraints: value
                .supported_constraints
                .into_iter()
                .map(|constraint| format!("{constraint:?}"))
                .collect(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct GenerateResponse {
    pub text: String,
    pub usage: UsagePayload,
    pub finish_reason: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamPayload {
    Token {
        id: u32,
        text: String,
        index: usize,
    },
    Done {
        finish_reason: String,
        usage: UsagePayload,
    },
}

impl From<StreamEvent> for StreamPayload {
    fn from(value: StreamEvent) -> Self {
        match value {
            StreamEvent::Token { id, text, index } => Self::Token { id, text, index },
            StreamEvent::Done {
                finish_reason,
                usage,
            } => Self::Done {
                finish_reason: finish_reason_name(finish_reason).to_string(),
                usage: UsagePayload::from(usage),
            },
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct UsagePayload {
    pub prompt_tokens: u32,
    pub generated_tokens: u32,
    pub total_tokens: u32,
}

impl From<Usage> for UsagePayload {
    fn from(value: Usage) -> Self {
        Self {
            prompt_tokens: value.prompt_tokens,
            generated_tokens: value.generated_tokens,
            total_tokens: value.total_tokens(),
        }
    }
}

fn finish_reason_name(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        FinishReason::Cancelled => "cancelled",
        FinishReason::ContentFilter => "content_filter",
    }
}

fn model_name(source: &str) -> String {
    Path::new(source)
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or(source)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_llm::{TextLlmOutput, Usage};

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
            on_event: &mut dyn FnMut(StreamEvent),
        ) -> core_llm::Result<TextLlmOutput> {
            self.validate(req)?;
            on_event(StreamEvent::Token {
                id: 1,
                text: "ok".to_string(),
                index: 0,
            });
            let usage = Usage {
                prompt_tokens: 2,
                generated_tokens: 1,
            };
            on_event(StreamEvent::Done {
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

    fn fake_loader(spec: &LoadSpec) -> core_llm::Result<Box<dyn TextLlm>> {
        if spec.source == "bad" {
            return Err(core_llm::Error::Load("bad model".to_string()));
        }
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

    #[test]
    fn load_generate_and_unload_round_trip() {
        let engine = EngineHandle::spawn_with_loader(fake_loader);
        let status = engine
            .load_model(LoadModelRequest {
                source: "/tmp/fake-model".to_string(),
                display_name: None,
                quantize: None,
            })
            .unwrap();
        assert_eq!(status.loaded.unwrap().name, "fake-model");

        let mut events = Vec::new();
        let output = engine
            .generate(
                GenerateRequest {
                    messages: vec![GenerateMessage {
                        role: "user".to_string(),
                        content: "hello".to_string(),
                    }],
                    sampling: SamplingRequest::default(),
                    max_new_tokens: 8,
                    seed: None,
                    stop: Vec::new(),
                },
                |event| events.push(event),
            )
            .unwrap();
        assert_eq!(output.text, "ok");
        assert_eq!(events.len(), 2);

        let status = engine.unload_model().unwrap();
        assert!(status.loaded.is_none());
    }

    #[test]
    fn generate_requires_loaded_model() {
        let engine = EngineHandle::spawn_with_loader(fake_loader);
        let result = engine.generate(
            GenerateRequest {
                messages: vec![GenerateMessage {
                    role: "user".to_string(),
                    content: "hello".to_string(),
                }],
                sampling: SamplingRequest::default(),
                max_new_tokens: 8,
                seed: None,
                stop: Vec::new(),
            },
            |_| {},
        );
        assert_eq!(result.unwrap_err(), "no model loaded");
    }
}
