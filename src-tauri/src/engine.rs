use std::path::Path;
use std::sync::mpsc;
use std::thread;

use core_llm::{
    load_for_model, CancelFlag, Channel, Content, FinishReason, ImageRef, KvCacheQuant,
    KvCacheQuantMethod, LoadSpec, Message, Quantize, Role, Sampling, StreamEvent, TextLlm,
    TextLlmCapabilities, TextLlmDescriptor, TextLlmRequest, ThinkingMode, ToolCall, ToolSpec, Usage,
    VideoRef,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

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
            // KV-cache quantization (sc-8533) — DISTINCT from the weight quant above. `None` (the
            // default) ⇒ a dense cache; a provider/model that doesn't support it surfaces a clean
            // error from `load` (no crash, no silent fallback).
            kv_cache_quant: request.kv_cache_quant.map(Into::into),
        };
        let provider = (self.loader)(&spec).map_err(|error| error.to_string())?;
        let descriptor = provider.descriptor().clone();
        self.loaded = Some(LoadedModel {
            source: request.source,
            display_name: request.display_name,
            quantize: request.quantize,
            kv_cache_quant: request.kv_cache_quant,
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
            thinking: output.thinking,
            tool_calls: output
                .tool_calls
                .into_iter()
                .map(GenerateToolCall::from)
                .collect(),
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
    kv_cache_quant: Option<KvCacheQuantRequest>,
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
            kv_cache_quant: self.kv_cache_quant,
            provider: ProviderSummary::from(self.descriptor.clone()),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LoadModelRequest {
    pub source: String,
    #[serde(default)]
    pub display_name: Option<String>,
    /// Load-time **weight** quantization (Q4/Q8). DISTINCT from [`kv_cache_quant`](Self::kv_cache_quant).
    #[serde(default)]
    pub quantize: Option<QuantizeRequest>,
    /// **KV-cache** quantization (sc-8533) — runtime compression of the per-step key/value cache,
    /// entirely separate from the weight [`quantize`](Self::quantize) above. `None`/absent (the
    /// default) ⇒ a dense cache; an unsupported backend/model surfaces a clean load error.
    #[serde(default)]
    pub kv_cache_quant: Option<KvCacheQuantRequest>,
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

/// KV-cache quantization config (sc-8533): a compression method + bit-width, mirroring
/// [`core_llm::KvCacheQuant`]. Kept separate from [`QuantizeRequest`] (weight quant) end to end.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct KvCacheQuantRequest {
    pub method: KvCacheQuantMethodRequest,
    pub bits: u8,
}

/// The KV-cache compression method (sc-8533), mirroring [`core_llm::KvCacheQuantMethod`].
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum KvCacheQuantMethodRequest {
    Rvq,
}

impl From<KvCacheQuantMethodRequest> for KvCacheQuantMethod {
    fn from(value: KvCacheQuantMethodRequest) -> Self {
        match value {
            KvCacheQuantMethodRequest::Rvq => KvCacheQuantMethod::Rvq,
        }
    }
}

impl From<KvCacheQuantRequest> for KvCacheQuant {
    fn from(value: KvCacheQuantRequest) -> Self {
        KvCacheQuant {
            method: value.method.into(),
            bits: value.bits,
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
    #[serde(default)]
    pub thinking: ThinkingRequest,
    /// Tools / functions offered to the model. Rendered into the prompt by the chat template and used
    /// to type-coerce the model's parsed tool calls. Honored only by providers advertising
    /// `supports_tools`; a non-empty `tools` on a provider without it is rejected by the provider's
    /// `validate` (surfaced as a 400). Empty ⇒ no tool section, behavior unchanged.
    #[serde(default)]
    pub tools: Vec<GenerateTool>,
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
            thinking: self.thinking.into(),
            tools: self
                .tools
                .into_iter()
                .map(GenerateTool::into_core)
                .collect(),
            stop: self.stop,
            cancel: CancelFlag::new(),
        })
    }
}

/// A function offered to the model, mirroring [`core_llm::ToolSpec`] (the OpenAI function-tool shape).
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GenerateTool {
    /// The function name the model calls.
    pub name: String,
    /// What the function does and when to use it.
    #[serde(default)]
    pub description: String,
    /// JSON-Schema for the call arguments (typically an `{"type":"object","properties":{…}}` object);
    /// rendered into the prompt verbatim and used to type-coerce parsed arguments.
    #[serde(default = "default_tool_parameters")]
    pub parameters: Value,
}

impl GenerateTool {
    fn into_core(self) -> ToolSpec {
        ToolSpec::new(self.name, self.description, self.parameters)
    }
}

/// A no-argument function's default schema (an empty object), matching the `transformers` convention.
fn default_tool_parameters() -> Value {
    serde_json::json!({"type": "object", "properties": {}})
}

/// A tool / function call: an assistant turn's call (multi-turn input) and the model's parsed output,
/// mirroring [`core_llm::ToolCall`].
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GenerateToolCall {
    /// The called function's name.
    pub name: String,
    /// The call arguments as an ordered name→value map (insertion order preserved).
    #[serde(default)]
    pub arguments: Map<String, Value>,
}

impl GenerateToolCall {
    fn into_core(self) -> ToolCall {
        ToolCall::new(self.name, self.arguments)
    }
}

impl From<ToolCall> for GenerateToolCall {
    fn from(value: ToolCall) -> Self {
        Self {
            name: value.name,
            arguments: value.arguments,
        }
    }
}

fn default_max_new_tokens() -> u32 {
    512
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingRequest {
    #[default]
    Auto,
    Enabled,
    Disabled,
}

impl From<ThinkingRequest> for ThinkingMode {
    fn from(value: ThinkingRequest) -> Self {
        match value {
            ThinkingRequest::Auto => ThinkingMode::Auto,
            ThinkingRequest::Enabled => ThinkingMode::Enabled,
            ThinkingRequest::Disabled => ThinkingMode::Disabled,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GenerateMessage {
    pub role: String,
    pub content: String,
    /// Image attachments for a vision model, as `data:<mime>;base64,<data>` URLs (or raw base64).
    /// Decoded to RGB8 and placed *before* the text block, matching the Qwen-VL convention.
    #[serde(default)]
    pub images: Vec<String>,
    /// Video attachments for a video-capable model (sc-8081): pre-sampled frames + per-frame
    /// timestamps. Decoded to RGB8 frames and placed *before* the text block (after images), so the
    /// vision providers see visuals before text, matching the Qwen3-VL convention.
    #[serde(default)]
    pub videos: Vec<GenerateVideo>,
    /// An assistant turn's tool / function calls, re-rendered by the chat template to continue a
    /// multi-step tool exchange (paired with the following `tool`-role result turn). Empty for
    /// non-tool turns.
    #[serde(default)]
    pub tool_calls: Vec<GenerateToolCall>,
}

/// A sampled video attachment: ordered frame image data URLs + per-frame timestamps (seconds). The
/// host (frontend / API caller) samples the frames; this carries them straight to [`VideoRef`].
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GenerateVideo {
    /// Sampled frames in temporal order, each a `data:image/…;base64,…` URL (or bare base64).
    pub frames: Vec<String>,
    /// Per-frame timestamp in seconds (one per frame), driving Text–Timestamp Alignment.
    pub timestamps: Vec<f32>,
}

impl GenerateVideo {
    fn into_core(self) -> EngineResult<VideoRef> {
        let frames = self
            .frames
            .iter()
            .map(|f| decode_image(f))
            .collect::<EngineResult<Vec<ImageRef>>>()?;
        VideoRef::new(frames, self.timestamps)
    }
}

impl GenerateMessage {
    fn into_core(self) -> EngineResult<Message> {
        // Visuals first (images then videos), then text — the order the chat templates / vision
        // providers expect (Qwen3-VL: visual placeholder before the question text).
        let mut content = Vec::with_capacity(self.images.len() + self.videos.len() + 1);
        for image in &self.images {
            content.push(Content::Image(decode_image(image)?));
        }
        for video in self.videos {
            content.push(Content::Video(video.into_core()?));
        }
        if !self.content.is_empty() || content.is_empty() {
            content.push(Content::Text(self.content));
        }
        Ok(Message {
            role: role_from_str(&self.role)?,
            content,
            thinking: None,
            tool_calls: self
                .tool_calls
                .into_iter()
                .map(GenerateToolCall::into_core)
                .collect(),
        })
    }
}

/// Decode an image attachment (`data:<mime>;base64,<data>` URL or bare base64) to an RGB8 [`ImageRef`].
fn decode_image(data: &str) -> EngineResult<ImageRef> {
    use base64::Engine as _;
    // Strip the optional `data:<mime>;base64,` prefix.
    let b64 = data.rsplit_once(',').map(|(_, rest)| rest).unwrap_or(data).trim();
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .map_err(|error| format!("invalid base64 image attachment: {error}"))?;
    let rgb = image::load_from_memory(&bytes)
        .map_err(|error| format!("could not decode image attachment: {error}"))?
        .to_rgb8();
    let (width, height) = rgb.dimensions();
    ImageRef::new(width, height, rgb.into_raw())
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
    /// The KV-cache quantization the loaded model is running with (sc-8533), or `None` for a dense
    /// cache. Surfaced so the UI can reflect the active state (and restore "dense" when toggled off).
    pub kv_cache_quant: Option<KvCacheQuantRequest>,
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
    /// Whether the loaded model accepts video input (sc-8081) — surfaced so the UI can enable the
    /// video attach affordance.
    pub supports_video: bool,
    pub supports_thinking: bool,
    pub supports_tools: bool,
    /// Whether the loaded model can run a quantized KV cache (sc-8533) — surfaced so the UI only
    /// offers the KV-cache-quant toggle where the active backend/model honors it.
    pub supports_kv_cache_quant: bool,
    pub supported_constraints: Vec<String>,
}

impl From<TextLlmCapabilities> for CapabilitySummary {
    fn from(value: TextLlmCapabilities) -> Self {
        Self {
            max_context_tokens: value.max_context_tokens,
            max_new_tokens: value.max_new_tokens,
            supports_system_prompt: value.supports_system_prompt,
            supports_vision: value.supports_vision,
            supports_video: value.supports_video,
            supports_thinking: value.supports_thinking,
            supports_tools: value.supports_tools,
            supports_kv_cache_quant: value.supports_kv_cache_quant,
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
    pub thinking: Option<String>,
    /// Tool / function calls the model emitted (empty if none, or if the request offered no tools).
    pub tool_calls: Vec<GenerateToolCall>,
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
        channel: StreamChannel,
    },
    Done {
        finish_reason: String,
        usage: UsagePayload,
    },
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamChannel {
    Content,
    Thinking,
}

impl From<Channel> for StreamChannel {
    fn from(value: Channel) -> Self {
        match value {
            Channel::Content => Self::Content,
            Channel::Thinking => Self::Thinking,
        }
    }
}

impl From<StreamEvent> for StreamPayload {
    fn from(value: StreamEvent) -> Self {
        match value {
            StreamEvent::Token {
                id,
                text,
                index,
                channel,
            } => Self::Token {
                id,
                text,
                index,
                channel: channel.into(),
            },
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

    /// A `LoadModelRequest` with no `kv_cache_quant` field in the JSON deserializes with `None`
    /// (backward compatible with callers/persisted requests written before sc-8533), and weight
    /// quant stays independent.
    #[test]
    fn load_request_kv_cache_quant_absent_is_none() {
        let req: LoadModelRequest =
            serde_json::from_str(r#"{"source":"/snap","quantize":"q4"}"#).unwrap();
        assert!(req.kv_cache_quant.is_none());
        assert!(matches!(req.quantize, Some(QuantizeRequest::Q4)));
    }

    /// An explicit RVQ `kv_cache_quant` round-trips through serde and converts to the core-llm
    /// contract type with the method + bit-width preserved, distinct from the weight quant.
    #[test]
    fn load_request_kv_cache_quant_round_trip_and_into_core() {
        let json = r#"{"source":"/snap","kv_cache_quant":{"method":"rvq","bits":4}}"#;
        let req: LoadModelRequest = serde_json::from_str(json).unwrap();
        let kv = req.kv_cache_quant.expect("kv_cache_quant present");
        assert_eq!(kv.method, KvCacheQuantMethodRequest::Rvq);
        assert_eq!(kv.bits, 4);
        assert!(req.quantize.is_none()); // weight quant independent / absent

        // Re-serialize → deserialize is stable.
        let again: KvCacheQuantRequest =
            serde_json::from_str(&serde_json::to_string(&kv).unwrap()).unwrap();
        assert_eq!(again, kv);

        // Converts to the core-llm contract type faithfully.
        let core: KvCacheQuant = kv.into();
        assert_eq!(core.method, KvCacheQuantMethod::Rvq);
        assert_eq!(core.bits, 4);
    }

    /// The capability summary surfaces `supports_kv_cache_quant` so the UI can gate the toggle.
    #[test]
    fn capability_summary_carries_kv_cache_quant_support() {
        let caps = TextLlmCapabilities {
            supports_kv_cache_quant: true,
            ..Default::default()
        };
        let summary = CapabilitySummary::from(caps);
        assert!(summary.supports_kv_cache_quant);
        assert!(!CapabilitySummary::from(TextLlmCapabilities::default()).supports_kv_cache_quant);
    }

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
                channel: Channel::Content,
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
                thinking: None,
                tool_calls: Vec::new(),
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
                kv_cache_quant: None,
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
                        images: Vec::new(),
                        videos: Vec::new(),
                        tool_calls: Vec::new(),
                    }],
                    sampling: SamplingRequest::default(),
                    max_new_tokens: 8,
                    seed: None,
                    stop: Vec::new(),
                    thinking: ThinkingRequest::Auto,
                    tools: Vec::new(),
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
                    images: Vec::new(),
                    videos: Vec::new(),
                    tool_calls: Vec::new(),
                }],
                sampling: SamplingRequest::default(),
                max_new_tokens: 8,
                seed: None,
                stop: Vec::new(),
                thinking: ThinkingRequest::Auto,
                tools: Vec::new(),
            },
            |_| {},
        );
        assert_eq!(result.unwrap_err(), "no model loaded");
    }
}
