use std::ffi::OsString;
use std::fs;
use std::io::{Read, Write};
use std::net::{IpAddr, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::{Duration, Instant};

use crate::core_llm::{
    CancelFlag, Channel, Constraint, Content, FinishReason, GenerationTimings, ImageRef, LoadSpec,
    Message, MtpCapabilities, MtpMode, MtpStats, Quantize, ReasoningEffort, Role, Sampling,
    StreamEvent, TextLlm, TextLlmCapabilities, TextLlmDescriptor, TextLlmRequest, ThinkingMode,
    ToolCall, ToolSpec, Usage, VideoRef,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub type EngineResult<T> = Result<T, String>;

type Loader = fn(&LoadSpec) -> crate::core_llm::Result<Box<dyn TextLlm>>;

/// The in-flight generation's cancel flag, shared between the engine thread and the handle so a
/// `cancel()` call can trip it without going through the actor's serial command loop (which is
/// blocked while a generation runs). `None` when no generation is in flight (code-review F-004).
///
/// The single (unkeyed) slot is safe because the [`EngineActor`] runs one command at a time on a
/// single thread: at most ONE generation is ever in flight, so `cancel()` can only ever target it.
/// This holds even though the handle is shared between the Tauri UI and the HTTP server — the actor
/// serializes their `Generate` commands. Two caveats a future change must respect:
/// (1) If the actor ever runs generations concurrently (e.g. a worker pool), the slot must be keyed
///     by request id so cancel targets the right generation (PR #30 review, R7).
/// (2) A `cancel()` issued in the window after a `Generate` command is sent but before the actor
///     installs the flag returns `false` (the flag isn't installed yet); the caller may retry. This
///     window is bounded by the actor's dequeue time and is not a correctness bug — the generation
///     simply isn't cancellable until it starts.
type CancelSlot = Arc<Mutex<Option<CancelFlag>>>;

#[derive(Clone)]
pub struct EngineHandle {
    tx: mpsc::Sender<EngineCommand>,
    cancel: CancelSlot,
}

impl EngineHandle {
    pub fn spawn() -> Self {
        Self::spawn_with_loader(crate::inference_runtime::load_for_model)
    }

    pub(crate) fn spawn_with_loader(loader: Loader) -> Self {
        let (tx, rx) = mpsc::channel();
        let cancel: CancelSlot = Arc::new(Mutex::new(None));
        let actor_cancel = Arc::clone(&cancel);
        thread::Builder::new()
            .name("chatworks-engine".to_string())
            .spawn(move || EngineActor::new(loader, rx, actor_cancel).run())
            .expect("failed to start ChatWorks engine thread");
        Self { tx, cancel }
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

    /// Request cancellation of the in-flight generation, if any. The provider is handed the
    /// [`CancelFlag`] at generation start; tripping it asks the provider to stop promptly (the
    /// provider returns a partial output marked `FinishReason::Cancelled`). Returns `true` if a
    /// generation was in flight and its flag was tripped (code-review F-004).
    pub fn cancel(&self) -> bool {
        if let Ok(slot) = self.cancel.lock() {
            if let Some(flag) = slot.as_ref() {
                flag.cancel();
                return true;
            }
        }
        false
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
    cancel: CancelSlot,
}

impl EngineActor {
    fn new(loader: Loader, rx: mpsc::Receiver<EngineCommand>, cancel: CancelSlot) -> Self {
        Self {
            loader,
            rx,
            loaded: None,
            cancel,
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
            projector_source: request.projector_source.clone(),
            quantize: request.quantize.map(Into::into),
        };
        let provider = (self.loader)(&spec).map_err(|error| error.to_string())?;
        let descriptor = provider.descriptor().clone();
        self.loaded = Some(LoadedModel {
            source: request.source,
            display_name: request.display_name,
            quantize: request.quantize,
            projector_source: request.projector_source,
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
        // Register the cancel flag before media staging. A direct video URL can spend time
        // downloading or sampling frames before a provider emits its first token, and it must be
        // cancellable through the same lifecycle as generation.
        let cancel = CancelFlag::new();
        if let Ok(mut slot) = self.cancel.lock() {
            *slot = Some(cancel.clone());
        }
        let result = request.into_core(cancel).and_then(|core_request| {
            loaded
                .provider
                .generate(&core_request, &mut |event| {
                    let _ = event_tx.send(StreamPayload::from(event));
                })
                .map_err(|error| error.to_string())
        });
        // Always clear the in-flight flag, whether media preparation, generation, or cancellation
        // ended the request — a stale flag must never cancel a later generation.
        if let Ok(mut slot) = self.cancel.lock() {
            *slot = None;
        }
        let output = result?;
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
            mtp: output.mtp.map(MtpStatsPayload::from),
            timings: output.timings.map(GenerationTimingsPayload::from),
        })
    }

    fn status(&self) -> EngineStatus {
        EngineStatus {
            loaded: self.loaded.as_ref().map(LoadedModel::status),
            providers: crate::inference_runtime::textllms()
                .map(|registration| ProviderSummary::from((registration.descriptor)()))
                .collect(),
        }
    }
}

struct LoadedModel {
    source: String,
    display_name: Option<String>,
    quantize: Option<QuantizeRequest>,
    projector_source: Option<String>,
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
            projector_source: self.projector_source.clone(),
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
    /// An explicit multimodal projector paired with a GGUF language model. Providers validate the
    /// artifact; an omitted value deliberately keeps a packed GGUF text-only.
    #[serde(default)]
    pub projector_source: Option<String>,
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
    #[serde(default)]
    pub thinking: ThinkingRequest,
    /// Explicit `enable_thinking` template kwarg. Kept separate from the legacy
    /// `disable_thinking` wire flag so contradictory requests fail at the boundary.
    #[serde(default)]
    pub enable_thinking: Option<bool>,
    /// Backwards-compatible no-think flag used by existing ChatWorks clients.
    #[serde(default)]
    pub disable_thinking: Option<bool>,
    /// Qwen3.8's typed reasoning budget; rejected downstream unless the loaded provider
    /// explicitly advertises this Qwen-specific control.
    #[serde(default)]
    pub reasoning_effort: Option<ReasoningEffortRequest>,
    #[serde(default)]
    pub preserve_thinking: Option<bool>,
    #[serde(default)]
    pub mtp: MtpRequest,
    #[serde(default)]
    pub constraint: Option<ConstraintRequest>,
    /// Tools / functions offered to the model. Rendered into the prompt by the chat template and used
    /// to type-coerce the model's parsed tool calls. Honored only by providers advertising
    /// `supports_tools`; a non-empty `tools` on a provider without it is rejected by the provider's
    /// `validate` (surfaced as a 400). Empty ⇒ no tool section, behavior unchanged.
    #[serde(default)]
    pub tools: Vec<GenerateTool>,
}

impl GenerateRequest {
    fn into_core(self, cancel: CancelFlag) -> EngineResult<TextLlmRequest> {
        if self.messages.is_empty() {
            return Err("messages must not be empty".to_string());
        }
        Ok(TextLlmRequest {
            messages: self
                .messages
                .into_iter()
                .map(|message| message.into_core(&cancel))
                .collect::<EngineResult<Vec<_>>>()?,
            sampling: self.sampling.into_core(),
            // `max_new_tokens` is intentionally not clamped against the loaded provider's
            // `max_new_tokens` here: the provider's `validate` rejects an over-limit value (the
            // FakeProvider test exercises exactly that), so the clamp is deliberately downstream of
            // the engine boundary (code-review F-014).
            max_new_tokens: self.max_new_tokens,
            seed: self.seed,
            constraint: self.constraint.map(ConstraintRequest::into_core),
            thinking: resolve_thinking(self.thinking, self.enable_thinking, self.disable_thinking)?,
            reasoning_effort: self.reasoning_effort.map(ReasoningEffortRequest::into_core),
            preserve_thinking: self.preserve_thinking,
            mtp: self.mtp.into_core()?,
            tools: self
                .tools
                .into_iter()
                .map(GenerateTool::into_core)
                .collect(),
            stop: self.stop,
            cancel,
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

fn resolve_thinking(
    thinking: ThinkingRequest,
    enable_thinking: Option<bool>,
    disable_thinking: Option<bool>,
) -> EngineResult<ThinkingMode> {
    if enable_thinking == Some(true) && disable_thinking == Some(true) {
        return Err("enable_thinking=true conflicts with disable_thinking=true".to_string());
    }
    if enable_thinking.is_some() && !matches!(thinking, ThinkingRequest::Auto) {
        return Err("enable_thinking cannot be combined with thinking".to_string());
    }
    if disable_thinking.is_some() && !matches!(thinking, ThinkingRequest::Auto) {
        return Err("disable_thinking cannot be combined with thinking".to_string());
    }
    if let Some(enabled) = enable_thinking {
        return Ok(if enabled {
            ThinkingMode::Enabled
        } else {
            ThinkingMode::Disabled
        });
    }
    if disable_thinking.unwrap_or(false) {
        return Ok(ThinkingMode::Disabled);
    }
    Ok(thinking.into())
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffortRequest {
    Low,
    Medium,
    Xhigh,
}
impl ReasoningEffortRequest {
    fn into_core(self) -> ReasoningEffort {
        match self {
            Self::Low => ReasoningEffort::Low,
            Self::Medium => ReasoningEffort::Medium,
            Self::Xhigh => ReasoningEffort::XHigh,
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(tag = "mode", rename_all = "lowercase")]
pub enum MtpRequest {
    #[default]
    Off,
    Auto,
    Enabled {
        draft_tokens: u32,
    },
}
impl MtpRequest {
    fn into_core(self) -> EngineResult<MtpMode> {
        match self {
            Self::Off => Ok(MtpMode::Off),
            Self::Auto => Ok(MtpMode::Auto),
            Self::Enabled { draft_tokens: 0 } => {
                Err("mtp.draft_tokens must be at least 1".to_string())
            }
            Self::Enabled { draft_tokens } => Ok(MtpMode::Enabled { draft_tokens }),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ConstraintRequest {
    Json,
}
impl ConstraintRequest {
    fn into_core(self) -> Constraint {
        match self {
            Self::Json => Constraint::Json,
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
    /// Ordered visual inputs. New requests use this to retain image/video ordering; `images` and
    /// `videos` above remain readable for persisted conversations written before sc-23941.
    #[serde(default)]
    pub media: Vec<GenerateMedia>,
    /// An assistant turn's tool / function calls, re-rendered by the chat template to continue a
    /// multi-step tool exchange (paired with the following `tool`-role result turn). Empty for
    /// non-tool turns.
    #[serde(default)]
    pub tool_calls: Vec<GenerateToolCall>,
    /// Preserved assistant reasoning, accepted from OpenAI's `reasoning_content` history field.
    #[serde(default, alias = "reasoning_content")]
    pub thinking: Option<String>,
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

/// A visual attachment in its original user/API order. Video frames are already decoded/sampled by
/// the host and retain their timestamp order through to [`VideoRef`].
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GenerateMedia {
    Image {
        url: String,
    },
    /// A local image file or HTTP(S) URL, staged with the same public-network and size rules as
    /// video sources before decoding into RGB8.
    ImageSource {
        url: String,
    },
    Video {
        frames: Vec<String>,
        timestamps: Vec<f32>,
    },
    /// A file path, `file://` URI, or HTTPS URL. It is staged under strict byte/network bounds and
    /// decoded into the same timestamped frame representation as pre-sampled video.
    VideoSource {
        url: String,
    },
}

impl GenerateMedia {
    fn into_core(self, cancel: &CancelFlag) -> EngineResult<Content> {
        match self {
            Self::Image { url } => Ok(Content::Image(decode_image(&url)?)),
            Self::ImageSource { url } => Ok(Content::Image(decode_image_source(&url, cancel)?)),
            Self::Video { frames, timestamps } => Ok(Content::Video(
                GenerateVideo { frames, timestamps }.into_core()?,
            )),
            Self::VideoSource { url } => Ok(Content::Video(decode_video_source(&url, cancel)?)),
        }
    }
}

impl GenerateVideo {
    fn into_core(self) -> EngineResult<VideoRef> {
        // Per-video frame cap (F-002). This is deliberately NOT cumulative across the request: the
        // frontend re-sends full history each turn, so a cumulative cap would brick a conversation
        // once enough video turns accumulate (PR #30 review). Per-video bounds a single oversized
        // clip without making history toxic.
        if self.frames.len() > MAX_FRAMES_PER_VIDEO {
            return Err(format!(
                "video attachment has {} frames, exceeding the {MAX_FRAMES_PER_VIDEO} per-video limit",
                self.frames.len()
            ));
        }
        let frames = self
            .frames
            .iter()
            .map(|f| decode_image(f))
            .collect::<EngineResult<Vec<ImageRef>>>()?;
        VideoRef::new(frames, self.timestamps)
    }
}

impl GenerateMessage {
    fn into_core(self, cancel: &CancelFlag) -> EngineResult<Message> {
        // Ordered `media` preserves the composer/API's image-video sequence. Legacy persisted
        // messages have no `media`, so retain their historical images-then-videos behavior.
        let mut content =
            Vec::with_capacity(self.media.len() + self.images.len() + self.videos.len() + 1);
        if self.media.is_empty() {
            for image in &self.images {
                content.push(Content::Image(decode_image(image)?));
            }
            for video in self.videos {
                content.push(Content::Video(video.into_core()?));
            }
        } else {
            for media in self.media {
                content.push(media.into_core(cancel)?);
            }
        }
        if !self.content.is_empty() || content.is_empty() {
            content.push(Content::Text(self.content));
        }
        Ok(Message {
            role: role_from_str(&self.role)?,
            content,
            thinking: self.thinking,
            tool_calls: self
                .tool_calls
                .into_iter()
                .map(GenerateToolCall::into_core)
                .collect(),
        })
    }
}

/// Maximum decoded pixel budget per image attachment (width × height). A 64 MiB JSON body can
/// carry a base64 payload that decodes to a multi-gigabyte RGB buffer (a decompression bomb); this
/// cap rejects oversized images at the decode boundary so both the IPC and HTTP paths share the
/// guard (code-review F-002). 3_318_240 px ≈ 1830×1830, comfortably above the frontend's 1536 px
/// self-limit (`src/media/image.js`) while bounding a single image to ~10 MB of RGB8.
const MAX_IMAGE_PIXELS: u64 = 3_318_240;

/// Per-axis decompression-bomb ceiling. The `image` crate's strict limits are per-axis (not a
/// pixel product), so this is set generously — high enough that no legitimate image near the
/// [`MAX_IMAGE_PIXELS`] budget is rejected (e.g. a 2560×800 panorama, 2.05 Mpx, is well under budget
/// but exceeds a naive `floor(sqrt(budget))` per-axis cap of 1821 px — PR #30 review). The real
/// budget is enforced post-decode on the width×height product below.
const MAX_IMAGE_AXIS: u32 = 4096;

/// Maximum frames per single video attachment, bounding the worst case where every frame is itself a
/// dimension-capped image (code-review F-002). The frontend samples up to ~8 frames per clip
/// (`VIDEO_ATTACHMENT_MAX_FRAMES` in `src/media/video.js`); this is a per-video safety ceiling, NOT
/// cumulative across the re-sent transcript — a cumulative cap would brick conversations once enough
/// video turns accumulate, since the frontend re-sends full history each turn (PR #30 review).
const MAX_FRAMES_PER_VIDEO: usize = 64;
const MAX_IMAGE_SOURCE_BYTES: u64 = 32 * 1024 * 1024;
const MAX_VIDEO_SOURCE_BYTES: u64 = 256 * 1024 * 1024;
const MAX_VIDEO_SOURCE_SECONDS: f64 = 10.0 * 60.0;
const VIDEO_SOURCE_FRAMES: usize = 8;
const VIDEO_SOURCE_MAX_AXIS: u32 = 768;

fn decode_video_source(source: &str, cancel: &CancelFlag) -> EngineResult<VideoRef> {
    let staged = stage_video_source(source, cancel)?;
    let result = decode_staged_video(&staged.path, cancel);
    if let Some(path) = staged.cleanup {
        let _ = fs::remove_file(path);
    }
    result
}

fn decode_image_source(source: &str, cancel: &CancelFlag) -> EngineResult<ImageRef> {
    let staged = stage_media_source(source, MAX_IMAGE_SOURCE_BYTES, "image", cancel)?;
    let result = fs::read(&staged.path)
        .map_err(|error| format!("could not read staged image: {error}"))
        .and_then(|bytes| decode_image_bytes(&bytes));
    if let Some(path) = staged.cleanup {
        let _ = fs::remove_file(path);
    }
    result
}

struct StagedVideo {
    path: PathBuf,
    cleanup: Option<PathBuf>,
}

fn stage_video_source(source: &str, cancel: &CancelFlag) -> EngineResult<StagedVideo> {
    stage_media_source(source, MAX_VIDEO_SOURCE_BYTES, "video", cancel)
}

fn stage_media_source(
    source: &str,
    max_bytes: u64,
    label: &str,
    cancel: &CancelFlag,
) -> EngineResult<StagedVideo> {
    let value = source.trim();
    if value.is_empty() {
        return Err("video_url.url must not be empty".to_string());
    }
    if let Some(raw) = value.strip_prefix("file://") {
        let path = PathBuf::from(raw);
        validate_media_file(&path, max_bytes, label)?;
        return Ok(StagedVideo {
            path,
            cleanup: None,
        });
    }
    if value.starts_with("http://") || value.starts_with("https://") {
        return download_media_url(value, max_bytes, label, cancel);
    }
    let path = PathBuf::from(value);
    validate_media_file(&path, max_bytes, label)?;
    Ok(StagedVideo {
        path,
        cleanup: None,
    })
}

fn validate_media_file(path: &Path, max_bytes: u64, label: &str) -> EngineResult<()> {
    let metadata =
        fs::metadata(path).map_err(|error| format!("could not read {label} file: {error}"))?;
    if !metadata.is_file() {
        return Err(format!("{label} source must be a regular file"));
    }
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase);
    let allowed = match label {
        "image" => matches!(extension.as_deref(), Some("jpg" | "jpeg" | "png" | "webp")),
        "video" => matches!(
            extension.as_deref(),
            Some("avi" | "m4v" | "mkv" | "mov" | "mp4" | "mpeg" | "mpg" | "ts" | "webm")
        ),
        _ => false,
    };
    if !allowed {
        return Err(format!(
            "{label} file must use a supported media filename extension"
        ));
    }
    if metadata.len() > max_bytes {
        return Err(format!(
            "{label} file exceeds the {} MiB limit",
            max_bytes / 1024 / 1024
        ));
    }
    Ok(())
}

fn download_media_url(
    value: &str,
    max_bytes: u64,
    label: &str,
    cancel: &CancelFlag,
) -> EngineResult<StagedVideo> {
    let url =
        reqwest::Url::parse(value).map_err(|error| format!("invalid {label} URL: {error}"))?;
    if url.scheme() != "https" && url.scheme() != "http" {
        return Err(format!("{label} URL must use http or https"));
    }
    let host = url
        .host_str()
        .ok_or_else(|| format!("{label} URL is missing a host"))?;
    let port = url.port_or_known_default().unwrap_or(443);
    let addresses = (host, port)
        .to_socket_addrs()
        .map_err(|error| format!("could not resolve {label} URL host: {error}"))?
        .collect::<Vec<_>>();
    if addresses.is_empty() {
        return Err(format!("could not resolve {label} URL host"));
    }
    if addresses
        .iter()
        .any(|address| disallowed_video_host(address.ip()))
    {
        return Err(format!(
            "{label} URL resolves to a loopback, private, or reserved address"
        ));
    }
    // Pin the connection to the addresses just checked. Without this resolver override, a hostile
    // hostname could answer public DNS here and a private address when reqwest resolves it again.
    let mut builder = reqwest::blocking::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(30));
    for address in addresses {
        builder = builder.resolve(host, address);
    }
    let client = builder
        .build()
        .map_err(|error| format!("could not prepare {label} download: {error}"))?;
    let mut response = client
        .get(url)
        .send()
        .map_err(|error| format!("could not download {label} URL: {error}"))?;
    if !response.status().is_success() {
        return Err(format!("{label} URL returned HTTP {}", response.status()));
    }
    if response
        .content_length()
        .is_some_and(|size| size > max_bytes)
    {
        return Err(format!(
            "{label} URL exceeds the {} MiB limit",
            max_bytes / 1024 / 1024
        ));
    }
    let path = std::env::temp_dir().join(format!(
        "chatworks-{label}-{}-{}.bin",
        std::process::id(),
        crate::fsutil::now_nanos()
    ));
    let mut file =
        fs::File::create(&path).map_err(|error| format!("could not stage {label} URL: {error}"))?;
    let mut downloaded = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        if cancel.is_cancelled() {
            let _ = fs::remove_file(&path);
            return Err("request cancelled while downloading media".to_string());
        }
        let read = match response.read(&mut buffer) {
            Ok(read) => read,
            Err(error) => {
                let _ = fs::remove_file(&path);
                return Err(format!("could not read {label} URL: {error}"));
            }
        };
        if read == 0 {
            break;
        }
        downloaded = downloaded.saturating_add(read as u64);
        if downloaded > max_bytes {
            let _ = fs::remove_file(&path);
            return Err(format!(
                "{label} URL exceeds the {} MiB limit",
                max_bytes / 1024 / 1024
            ));
        }
        if let Err(error) = file.write_all(&buffer[..read]) {
            let _ = fs::remove_file(&path);
            return Err(format!("could not stage {label} URL: {error}"));
        }
    }
    Ok(StagedVideo {
        path: path.clone(),
        cleanup: Some(path),
    })
}

fn disallowed_video_host(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => {
            let [a, b, c, d] = ip.octets();
            a == 0
                || a == 10
                || a == 127
                || a >= 224
                || (a == 100 && (64..=127).contains(&b))
                || (a == 169 && b == 254)
                || (a == 172 && (16..=31).contains(&b))
                || (a == 192 && b == 0 && (c == 0 || c == 2))
                || (a == 192 && b == 168)
                || (a == 198 && (b == 18 || b == 19))
                || (a == 198 && b == 51 && c == 100)
                || (a == 203 && b == 0 && c == 113)
                || (a == 255 && b == 255 && c == 255 && d == 255)
        }
        IpAddr::V6(ip) => {
            let segments = ip.segments();
            if segments[..5] == [0, 0, 0, 0, 0] && segments[5] == 0xffff {
                let v4 = std::net::Ipv4Addr::new(
                    (segments[6] >> 8) as u8,
                    segments[6] as u8,
                    (segments[7] >> 8) as u8,
                    segments[7] as u8,
                );
                return disallowed_video_host(IpAddr::V4(v4));
            }
            ip.is_loopback()
                || ip.is_unspecified()
                || ip.is_multicast()
                || (segments[0] & 0xfe00) == 0xfc00
                || (segments[0] & 0xffc0) == 0xfe80
        }
    }
}

fn configured_media_binary(name: &str) -> EngineResult<OsString> {
    // Development and test runs do not have a Tauri application bundle. Their explicit
    // override/PATH route is deliberately compiled out of release binaries, which always use the
    // checksum-verified sidecar bundled by `scripts/provision-ffmpeg-sidecars.sh`.
    #[cfg(debug_assertions)]
    {
        let variable = if name == "ffmpeg" {
            "CHATWORKS_FFMPEG"
        } else {
            "CHATWORKS_FFPROBE"
        };
        Ok(std::env::var_os(variable).unwrap_or_else(|| OsString::from(name)))
    }

    #[cfg(not(debug_assertions))]
    {
        let extension = if cfg!(target_os = "windows") {
            ".exe"
        } else {
            ""
        };
        let binary = format!("{name}{extension}");
        let executable = std::env::current_exe()
            .map_err(|error| format!("could not locate the bundled {name} sidecar: {error}"))?;
        let parent = executable.parent().ok_or_else(|| {
            format!("could not locate the bundled {name} sidecar beside the application")
        })?;
        // Tauri's `externalBin` places sidecars alongside the app executable. Resources is kept
        // as a compatibility candidate for older platform bundle layouts.
        let candidates = [
            parent.join(&binary),
            parent.join("..").join("Resources").join(&binary),
        ];
        candidates
            .into_iter()
            .find(|candidate| candidate.is_file())
            .map(|candidate| candidate.into_os_string())
            .ok_or_else(|| {
                format!(
                    "the bundled {name} media component is missing; reinstall this ChatWorks release"
                )
            })
    }
}

const MEDIA_COMMAND_TIMEOUT: Duration = Duration::from_secs(30);

fn run_media_command(
    command: &mut Command,
    cancel: &CancelFlag,
    operation: &str,
) -> EngineResult<(std::process::ExitStatus, Vec<u8>)> {
    if cancel.is_cancelled() {
        return Err("request cancelled before media decoding".to_string());
    }
    command.stdout(Stdio::piped()).stderr(Stdio::null());
    let mut child = command
        .spawn()
        .map_err(|error| format!("FFmpeg could not {operation} the video: {error}"))?;
    let deadline = Instant::now() + MEDIA_COMMAND_TIMEOUT;
    loop {
        if cancel.is_cancelled() {
            let _ = child.kill();
            let _ = child.wait();
            return Err("request cancelled while decoding media".to_string());
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!(
                "FFmpeg timed out while attempting to {operation} the video"
            ));
        }
        if let Some(status) = child
            .try_wait()
            .map_err(|error| format!("could not wait for FFmpeg: {error}"))?
        {
            let mut stdout = Vec::new();
            if let Some(mut pipe) = child.stdout.take() {
                pipe.read_to_end(&mut stdout)
                    .map_err(|error| format!("could not read FFmpeg output: {error}"))?;
            }
            return Ok((status, stdout));
        }
        thread::sleep(Duration::from_millis(25));
    }
}

fn decode_staged_video(path: &Path, cancel: &CancelFlag) -> EngineResult<VideoRef> {
    let ffprobe = configured_media_binary("ffprobe")?;
    let mut probe_command = Command::new(&ffprobe);
    probe_command
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
        ])
        .arg(path);
    let (probe_status, probe_stdout) = run_media_command(&mut probe_command, cancel, "inspect")?;
    if !probe_status.success() {
        return Err("FFmpeg could not inspect the video file or URL".to_string());
    }
    let duration = String::from_utf8_lossy(&probe_stdout)
        .trim()
        .parse::<f64>()
        .map_err(|_| "FFmpeg did not report a finite video duration".to_string())?;
    if !duration.is_finite() || duration <= 0.0 || duration > MAX_VIDEO_SOURCE_SECONDS {
        return Err(format!(
            "video duration must be between 0 and {} seconds",
            MAX_VIDEO_SOURCE_SECONDS as u64
        ));
    }
    let count = VIDEO_SOURCE_FRAMES.min(MAX_FRAMES_PER_VIDEO);
    let timestamps = (0..count)
        .map(|index| ((index + 1) as f64 * duration / (count + 1) as f64) as f32)
        .collect::<Vec<_>>();
    let ffmpeg = configured_media_binary("ffmpeg")?;
    let mut frames = Vec::with_capacity(count);
    for (index, timestamp) in timestamps.iter().enumerate() {
        let output = std::env::temp_dir().join(format!(
            "chatworks-video-frame-{}-{}-{index}.jpg",
            std::process::id(),
            crate::fsutil::now_nanos()
        ));
        let mut command = Command::new(&ffmpeg);
        command
            .args(["-v", "error", "-ss", &format!("{timestamp:.3}"), "-i"])
            .arg(path)
            .args([
                "-frames:v",
                "1",
                "-vf",
                &format!("scale={VIDEO_SOURCE_MAX_AXIS}:{VIDEO_SOURCE_MAX_AXIS}:force_original_aspect_ratio=decrease"),
                "-q:v",
                "4",
                "-y",
            ])
            .arg(&output);
        let (status, _) = match run_media_command(&mut command, cancel, "sample") {
            Ok(result) => result,
            Err(error) => {
                let _ = fs::remove_file(&output);
                return Err(error);
            }
        };
        if !status.success() {
            let _ = fs::remove_file(&output);
            return Err("FFmpeg could not sample a frame from the video file or URL".to_string());
        }
        let bytes = fs::read(&output);
        let _ = fs::remove_file(&output);
        let bytes =
            bytes.map_err(|error| format!("could not read sampled video frame: {error}"))?;
        use base64::Engine as _;
        frames.push(format!(
            "data:image/jpeg;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(bytes)
        ));
    }
    GenerateVideo { frames, timestamps }.into_core()
}

/// Decode an image attachment (`data:<mime>;base64,<data>` URL or bare base64) to an RGB8
/// [`ImageRef`], rejecting images whose decoded dimensions exceed [`MAX_IMAGE_PIXELS`]
/// (code-review F-002). The per-axis strict limit is only a bomb guard (see [`MAX_IMAGE_AXIS`]); the
/// real pixel budget is enforced post-decode on the width×height product, so legitimate
/// non-square images under budget are accepted.
fn decode_image(data: &str) -> EngineResult<ImageRef> {
    use base64::Engine as _;
    // Strip the optional `data:<mime>;base64,` prefix.
    let b64 = data
        .rsplit_once(',')
        .map(|(_, rest)| rest)
        .unwrap_or(data)
        .trim();
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .map_err(|error| format!("invalid base64 image attachment: {error}"))?;
    decode_image_bytes(&bytes)
}

fn decode_image_bytes(bytes: &[u8]) -> EngineResult<ImageRef> {
    use image::GenericImageView;
    let mut reader = image::ImageReader::new(std::io::Cursor::new(&bytes));
    reader.set_format(
        image::guess_format(bytes)
            .map_err(|error| format!("could not determine image attachment format: {error}"))?,
    );
    // Bomb guard: reject absurd per-axis dimensions before decoding the full buffer. Set generously
    // (MAX_IMAGE_AXIS); the real budget is the product check after decode.
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_IMAGE_AXIS);
    limits.max_image_height = Some(MAX_IMAGE_AXIS);
    reader.limits(limits);
    let img = reader
        .decode()
        .map_err(|error| format!("could not decode image attachment: {error}"))?;
    let (width, height) = img.dimensions();
    let pixels = width as u64 * height as u64;
    if pixels > MAX_IMAGE_PIXELS {
        return Err(format!(
            "image attachment is too large: {width}x{height} ({pixels} px) exceeds the {MAX_IMAGE_PIXELS} px limit"
        ));
    }
    // `into_rgb8()` moves the decoded buffer instead of cloning it (`.to_rgb8()` copies).
    let rgb = img.into_rgb8();
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
    pub presence_penalty: Option<f32>,
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
        if let Some(value) = self.presence_penalty {
            sampling.presence_penalty = value;
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub projector_source: Option<String>,
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
    pub supports_reasoning_effort: bool,
    /// The effort levels this specific loaded template exposes as distinct UI choices. The request
    /// parser may retain compatibility aliases outside this list.
    pub reasoning_efforts: Vec<String>,
    pub model_sampling_defaults: Option<ModelSamplingDefaultsPayload>,
    pub supports_preserve_thinking: bool,
    pub supports_tools: bool,
    pub mtp: Option<MtpCapabilitiesPayload>,
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
            supports_reasoning_effort: value.supports_reasoning_effort,
            reasoning_efforts: value
                .reasoning_efforts
                .into_iter()
                .map(|effort| effort.as_str().to_string())
                .collect(),
            model_sampling_defaults: value
                .model_sampling_defaults
                .map(ModelSamplingDefaultsPayload::from),
            supports_preserve_thinking: value.supports_preserve_thinking,
            supports_tools: value.supports_tools,
            mtp: value.mtp.map(MtpCapabilitiesPayload::from),
            supported_constraints: value
                .supported_constraints
                .into_iter()
                .map(|constraint| format!("{constraint:?}"))
                .collect(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct MtpCapabilitiesPayload {
    pub max_draft_tokens: u32,
    pub recommended_draft_tokens: u32,
}
impl From<MtpCapabilities> for MtpCapabilitiesPayload {
    fn from(value: MtpCapabilities) -> Self {
        Self {
            max_draft_tokens: value.max_draft_tokens,
            recommended_draft_tokens: value.recommended_draft_tokens,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct SamplingDefaultsPayload {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub presence_penalty: f32,
    pub repetition_penalty: f32,
    pub repetition_context: usize,
}

impl From<Sampling> for SamplingDefaultsPayload {
    fn from(value: Sampling) -> Self {
        Self {
            temperature: value.temperature,
            top_p: value.top_p,
            top_k: value.top_k,
            presence_penalty: value.presence_penalty,
            repetition_penalty: value.repetition_penalty,
            repetition_context: value.repetition_context,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelSamplingDefaultsPayload {
    pub thinking: SamplingDefaultsPayload,
    pub non_thinking: SamplingDefaultsPayload,
}

impl From<crate::core_llm::ModelSamplingDefaults> for ModelSamplingDefaultsPayload {
    fn from(value: crate::core_llm::ModelSamplingDefaults) -> Self {
        Self {
            thinking: SamplingDefaultsPayload::from(value.thinking),
            non_thinking: SamplingDefaultsPayload::from(value.non_thinking),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct MtpStatsPayload {
    pub proposed_tokens: u32,
    pub accepted_tokens: u32,
    pub target_forwards: u32,
}
impl From<MtpStats> for MtpStatsPayload {
    fn from(value: MtpStats) -> Self {
        Self {
            proposed_tokens: value.proposed_tokens,
            accepted_tokens: value.accepted_tokens,
            target_forwards: value.target_forwards,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct GenerationTimingsPayload {
    pub prefill_ms: u128,
    pub decode_ms: u128,
}
impl From<GenerationTimings> for GenerationTimingsPayload {
    fn from(value: GenerationTimings) -> Self {
        Self {
            prefill_ms: value.prefill.as_millis(),
            decode_ms: value.decode.as_millis(),
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mtp: Option<MtpStatsPayload>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timings: Option<GenerationTimingsPayload>,
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
    // The shared weightless fakes live in `test_support` so the engine and server tests can't drift
    // apart (code-review F-012).
    use crate::test_support::fake_loader;

    #[test]
    fn load_generate_and_unload_round_trip() {
        let engine = EngineHandle::spawn_with_loader(fake_loader);
        let status = engine
            .load_model(LoadModelRequest {
                source: "/tmp/fake-model".to_string(),
                display_name: None,
                quantize: None,
                projector_source: None,
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
                        media: Vec::new(),
                        tool_calls: Vec::new(),
                        thinking: None,
                    }],
                    sampling: SamplingRequest::default(),
                    max_new_tokens: 8,
                    seed: None,
                    stop: Vec::new(),
                    thinking: ThinkingRequest::Auto,
                    enable_thinking: None,
                    disable_thinking: None,
                    reasoning_effort: None,
                    preserve_thinking: None,
                    mtp: MtpRequest::Off,
                    constraint: None,
                    tools: Vec::new(),
                },
                |event| events.push(event),
            )
            .unwrap();
        assert_eq!(output.text, "ok");
        // The shared FakeProvider emits a reasoning token + a content token + Done (3 events) when
        // thinking is not disabled (code-review F-012 unifies this with the server fake).
        assert_eq!(events.len(), 3);
        assert_eq!(output.thinking.as_deref(), Some("reason"));

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
                    media: Vec::new(),
                    tool_calls: Vec::new(),
                    thinking: None,
                }],
                sampling: SamplingRequest::default(),
                max_new_tokens: 8,
                seed: None,
                stop: Vec::new(),
                thinking: ThinkingRequest::Auto,
                enable_thinking: None,
                disable_thinking: None,
                reasoning_effort: None,
                preserve_thinking: None,
                mtp: MtpRequest::Off,
                constraint: None,
                tools: Vec::new(),
            },
            |_| {},
        );
        assert_eq!(result.unwrap_err(), "no model loaded");
    }

    /// A single video with `MAX_FRAMES_PER_VIDEO + 1` frames is rejected (F-002). The cap is
    /// per-video, NOT cumulative across the re-sent transcript — a cumulative cap would brick a
    /// conversation once enough video turns accumulate (PR #30 review). Frames are bare base64;
    /// decode never runs because the cap fires first.
    #[test]
    fn rejects_video_exceeding_per_video_frame_cap() {
        let frames: Vec<String> = (0..=MAX_FRAMES_PER_VIDEO)
            .map(|i| format!("data:image/png;base64,FRAME{i}"))
            .collect();
        let timestamps: Vec<f32> = (0..frames.len()).map(|i| i as f32).collect();
        let request = GenerateRequest {
            messages: vec![GenerateMessage {
                role: "user".to_string(),
                content: "describe".to_string(),
                images: Vec::new(),
                videos: vec![GenerateVideo { frames, timestamps }],
                media: Vec::new(),
                tool_calls: Vec::new(),
                thinking: None,
            }],
            sampling: SamplingRequest::default(),
            max_new_tokens: 8,
            seed: None,
            stop: Vec::new(),
            thinking: ThinkingRequest::Auto,
            enable_thinking: None,
            disable_thinking: None,
            reasoning_effort: None,
            preserve_thinking: None,
            mtp: MtpRequest::Off,
            constraint: None,
            tools: Vec::new(),
        };
        let engine = EngineHandle::spawn_with_loader(fake_loader);
        engine
            .load_model(LoadModelRequest {
                source: "/tmp/fake-model".to_string(),
                display_name: None,
                quantize: None,
                projector_source: None,
            })
            .unwrap();
        let err = engine.generate(request, |_| {}).unwrap_err();
        assert!(
            err.contains("frames") && err.contains("per-video limit"),
            "expected a per-video frame-cap error, got: {err}"
        );
    }

    #[test]
    fn qwen_controls_and_conflicts_are_typed_at_request_boundary() {
        assert!(matches!(
            resolve_thinking(ThinkingRequest::Auto, Some(true), Some(true)),
            Err(message) if message.contains("conflicts")
        ));
        assert!(matches!(
            MtpRequest::Enabled { draft_tokens: 0 }.into_core(),
            Err(message) if message.contains("draft_tokens")
        ));
        assert!(matches!(
            MtpRequest::Enabled { draft_tokens: 3 }.into_core(),
            Ok(MtpMode::Enabled { draft_tokens: 3 })
        ));
    }

    #[test]
    fn cancelled_media_command_never_spawns_a_decoder() {
        let cancel = CancelFlag::new();
        cancel.cancel();
        let mut command = Command::new("chatworks-no-such-media-command");
        let err = run_media_command(&mut command, &cancel, "inspect").unwrap_err();
        assert!(err.contains("cancelled before media decoding"));
    }

    #[test]
    fn rejects_private_network_video_urls_before_download() {
        assert!(disallowed_video_host("127.0.0.1".parse().unwrap()));
        assert!(disallowed_video_host("10.0.0.1".parse().unwrap()));
        assert!(disallowed_video_host("100.64.0.1".parse().unwrap()));
        assert!(disallowed_video_host("198.18.0.1".parse().unwrap()));
        assert!(disallowed_video_host("::1".parse().unwrap()));
        assert!(disallowed_video_host("::ffff:127.0.0.1".parse().unwrap()));
        assert!(!disallowed_video_host("8.8.8.8".parse().unwrap()));
        assert!(
            download_media_url(
                "http://127.0.0.1/example.mp4",
                MAX_VIDEO_SOURCE_BYTES,
                "video",
                &CancelFlag::new()
            )
            .is_err()
        );
    }

    /// cancel() returns false when nothing is in flight and true once a generation's flag is
    /// installed. Because FakeProvider.generate runs synchronously to completion, we can't observe
    /// a mid-stream cancel end-to-end here, but we can confirm the handle exposes the cancel path
    /// and the flag is cleared after generation finishes (F-004).
    #[test]
    fn cancel_returns_false_when_idle() {
        let engine = EngineHandle::spawn_with_loader(fake_loader);
        // No generation in flight: cancel is a no-op.
        assert!(!engine.cancel());
        engine
            .load_model(LoadModelRequest {
                source: "/tmp/fake-model".to_string(),
                display_name: None,
                quantize: None,
                projector_source: None,
            })
            .unwrap();
        engine
            .generate(
                GenerateRequest {
                    messages: vec![GenerateMessage {
                        role: "user".to_string(),
                        content: "hello".to_string(),
                        images: Vec::new(),
                        videos: Vec::new(),
                        media: Vec::new(),
                        tool_calls: Vec::new(),
                        thinking: None,
                    }],
                    sampling: SamplingRequest::default(),
                    max_new_tokens: 8,
                    seed: None,
                    stop: Vec::new(),
                    thinking: ThinkingRequest::Auto,
                    enable_thinking: None,
                    disable_thinking: None,
                    reasoning_effort: None,
                    preserve_thinking: None,
                    mtp: MtpRequest::Off,
                    constraint: None,
                    tools: Vec::new(),
                },
                |_| {},
            )
            .unwrap();
        // Generation finished: the flag is cleared, so cancel is again a no-op.
        assert!(!engine.cancel());
    }
}
