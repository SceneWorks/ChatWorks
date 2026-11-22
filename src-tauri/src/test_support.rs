//! Shared test-only scaffolding for the engine + server unit tests (code-review F-012).
//!
//! Both `engine::tests` and `server::tests` independently defined a `FakeProvider` implementing
//! [`core_llm::TextLlm`], and the two copies had already drifted (descriptor capabilities, token
//! ids). This module hosts one canonical fake the two `#[cfg(test)]` blocks import, so they stay
//! honest. The fake emits a reasoning token + a content token (unless thinking is disabled), then a
//! `Stop` finish — enough to exercise the streaming, reasoning, and finish-reason paths without real
//! weights. A tool-call variant is provided for the server's tool-calling tests.

#![cfg(test)]

use crate::core_llm::{
    AdaptiveCostSummary, AdaptiveEventKind, AdaptiveSpeculativeEvent, AdaptiveSpeculativeReport,
    Channel, CudaGraphsReport, DecodeReport, DemotionBasis, DraftReport, FinishReason,
    GenerationTimings, LoadReport, LoadSpec, MonitorDecision, MtpCapabilities, MtpStats,
    PathReport, PrefixCheckpointCapabilities, PrefixCheckpointMode, PrefixCheckpointPolicy,
    PrefixCheckpointReport, PrefixDocumentSpan, ProjectionReport, ProposerCapabilities,
    ProposerKind, Quantize, Speculative, SpeculativeProposer, StreamEvent, TextLlm,
    TextLlmCapabilities, TextLlmDescriptor, TextLlmOutput, TextLlmRequest, ThinkingMode, Usage,
    MAX_CHECKPOINTS_PER_REQUEST,
};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

/// A weightless `TextLlm` that streams a reasoning token then a content token and finishes `Stop`.
/// The reasoning token is emitted only when the request's thinking mode is not `Disabled`.
pub struct FakeProvider {
    pub descriptor: TextLlmDescriptor,
    pub emit_telemetry: bool,
    /// The load report this fake returns (`None` = a provider that does not report one).
    pub load_report: Option<LoadReport>,
}

/// The decode report the telemetry fake emits: native MTP under the load's CUDA-graph switch,
/// with the graph runner falling back for a named reason, and NVFP4 projections served by the
/// decode GEMV after a cuBLASLt prefill (sc-24139): one prefill and two verify steps of two
/// drafts each, three of the four accepted (a mean accepted length of 1.5, sc-24445).
pub fn fake_decode_report() -> DecodeReport {
    DecodeReport {
        path: "mtp".to_string(),
        proposer: ProposerKind::Mtp,
        draft_tokens: Some(2),
        sampler: "device".to_string(),
        kv_cache: "static".to_string(),
        attention: "gqa".to_string(),
        cuda_graphs: CudaGraphsReport {
            enabled: true,
            path: "eager".to_string(),
            replayed: 0,
            eager: 3,
            captured: 0,
            fallback_reason: Some("deltanet_state_unstable".to_string()),
        },
        graph_path: "eager".to_string(),
        nvfp4_projections: PathReport {
            path: "mixed".to_string(),
            reason: Some("rows".to_string()),
        },
        fused_primitives: PathReport {
            path: "fused".to_string(),
            reason: None,
        },
        target_forwards: 3,
        prefill_forwards: 1,
        proposed_tokens: 4,
        accepted_tokens: 3,
        verify_steps: 2,
        replay_forwards: 0,
        prefix_hit_tokens: 0,
        prefix_cache: PathReport {
            path: "miss".to_string(),
            reason: None,
        },
        prefix_checkpoints: Some(PrefixCheckpointReport {
            policy: "auto".to_string(),
            reused_tokens: 0,
            reread_tokens: 2,
            matched_kind: None,
            checkpoints_captured: 1,
            captured_bytes: 2048,
            restore_time: Duration::from_micros(250),
            capture_time: Duration::from_micros(500),
            resident_bytes: 2048,
            protected_bytes: 0,
            evicted_entries: 0,
            fallbacks: Vec::new(),
        }),
        adaptive_speculative: Some(AdaptiveSpeculativeReport {
            adaptive: true,
            candidates: vec![
                ProposerKind::Mtp,
                ProposerKind::PromptLookup,
                ProposerKind::None,
            ],
            initial_proposer: ProposerKind::Mtp,
            initial_depth: 2,
            final_proposer: ProposerKind::Mtp,
            final_depth: 2,
            depth_changes: 0,
            proposer_switches: 0,
            events: vec![AdaptiveSpeculativeEvent {
                token_offset: 0,
                verify_step: 0,
                kind: AdaptiveEventKind::Selected,
                from_proposer: ProposerKind::Mtp,
                from_depth: 2,
                to_proposer: ProposerKind::Mtp,
                to_depth: 2,
                reason: "auto selected `mtp` at depth 2".to_string(),
            }],
            plain_reason: None,
            cost: Some(AdaptiveCostSummary {
                proposal_ns: 1_500_000,
                verify_ns: 6_000_000,
                rollback_ns: 0,
                sync_ns: 500_000,
                total_ns: 8_000_000,
                host_syncs: 2,
                committed_tokens: 4,
            }),
        }),
        fallbacks: Vec::new(),
        ..DecodeReport::default()
    }
}

/// What a recording loader saw of one `LoadSpec`: the weight format and the CUDA-graph switch.
pub type RecordedLoad = (Option<Quantize>, Option<bool>);

/// Every `LoadSpec` a recording loader saw, keyed by source path, so a test can assert what
/// reached the runtime's load request without sharing state with a concurrently running test.
pub fn recorded_load_specs() -> &'static Mutex<HashMap<String, RecordedLoad>> {
    static SPECS: OnceLock<Mutex<HashMap<String, RecordedLoad>>> = OnceLock::new();
    SPECS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// A telemetry fake that records the `LoadSpec` it was handed (see [`recorded_load_specs`]) and
/// reports a load report naming the requested format.
pub fn recording_loader(spec: &LoadSpec) -> crate::core_llm::Result<Box<dyn TextLlm>> {
    recorded_load_specs()
        .lock()
        .unwrap()
        .insert(spec.source.clone(), (spec.quantize, spec.cuda_graphs));
    let mut provider = telemetry_provider();
    provider.load_report = Some(LoadReport {
        requested: spec.quantize,
        projections: vec![ProjectionReport {
            kind: if spec.quantize == Some(Quantize::Nvfp4) {
                "nvfp4".to_string()
            } else {
                "dense".to_string()
            },
            count: 448,
            params: 1_000,
            resident_bytes: 562,
        }],
        // Settled like the Candle runtime: the request, else its default (off).
        cuda_graphs: Some(spec.cuda_graphs.unwrap_or(false)),
        // Like a Qwen3.5/3.8 decoder on the Candle runtime: with the switch on, the step cannot
        // be captured (its `graph_support` refusal), and a companion head is refused because the
        // fake already carries its own MTP predictor — both in the runtime's own wording.
        fallbacks: [
            (spec.cuda_graphs == Some(true))
                .then(|| "cuda_graphs: positions_host_scalar".to_string()),
            spec.mtp_head_source.as_ref().map(|head| {
                format!(
                    "mtp_head: the target already carries its own MTP predictor (`{head}`); \
                     the model loaded without a companion head"
                )
            }),
        ]
        .into_iter()
        .flatten()
        .collect(),
        // The requested budget settles unchanged (no headroom clamp in the fake).
        prefix_cache_bytes: spec.prefix_cache_bytes,
        // A named draft is resident unless its source names a refusal.
        draft: spec.draft_source.as_ref().map(|source| {
            if source.contains("refused") {
                DraftReport::refused(source, "draft vocabulary is not the target's")
            } else {
                DraftReport::resident(source)
            }
        }),
    });
    Ok(Box::new(provider))
}

impl TextLlm for FakeProvider {
    fn descriptor(&self) -> &TextLlmDescriptor {
        &self.descriptor
    }

    fn load_report(&self) -> Option<LoadReport> {
        self.load_report.clone()
    }

    fn validate(&self, req: &TextLlmRequest) -> crate::core_llm::Result<()> {
        self.descriptor
            .capabilities
            .validate_request(&self.descriptor.id, req)
    }

    fn generate(
        &self,
        req: &TextLlmRequest,
        on_event: &mut dyn FnMut(StreamEvent),
    ) -> crate::core_llm::Result<TextLlmOutput> {
        self.validate(req)?;
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, crate::core_llm::Role::User);
        assert_eq!(
            req.messages[0].content,
            vec![crate::core_llm::Content::Text("hello".to_string())]
        );
        let thinking = if req.thinking == ThinkingMode::Disabled {
            None
        } else {
            on_event(StreamEvent::Token {
                id: 9,
                text: "reason".to_string(),
                index: 0,
                channel: Channel::Thinking,
            });
            Some("reason".to_string())
        };
        on_event(StreamEvent::Token {
            id: 1,
            text: "ok".to_string(),
            index: 1,
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
            thinking,
            tool_calls: Vec::new(),
            usage,
            mtp: self.emit_telemetry.then_some(MtpStats {
                proposed_tokens: 4,
                accepted_tokens: 3,
                target_forwards: 2,
            }),
            timings: self.emit_telemetry.then_some(GenerationTimings {
                prefill: Duration::from_millis(12),
                decode: Duration::from_millis(34),
            }),
            decode: self.emit_telemetry.then(fake_decode_report),
            finish_reason: Some(FinishReason::Stop),
            kv_cache: None,
        })
    }
}

/// A loader that builds a [`FakeProvider`] whose descriptor advertises system prompts + thinking and
/// caps `max_new_tokens` at 8 (so over-limit requests are rejected by `validate`).
pub fn fake_loader(_: &LoadSpec) -> crate::core_llm::Result<Box<dyn TextLlm>> {
    Ok(Box::new(FakeProvider {
        descriptor: thinking_descriptor("fake", 8),
        emit_telemetry: false,
        load_report: None,
    }))
}

/// A weightless MTP-capable fake that reports deterministic native output evidence. It validates
/// the full HTTP/SSE telemetry path without loading model weights.
pub fn fake_telemetry_loader(_: &LoadSpec) -> crate::core_llm::Result<Box<dyn TextLlm>> {
    Ok(Box::new(telemetry_provider()))
}

fn telemetry_provider() -> FakeProvider {
    let mut descriptor = thinking_descriptor("fake-telemetry", 8);
    descriptor.capabilities.mtp = Some(MtpCapabilities {
        max_draft_tokens: 4,
        recommended_draft_tokens: 2,
    });
    descriptor.capabilities.prefix_checkpoints = Some(PrefixCheckpointCapabilities {
        max_checkpoints_per_request: MAX_CHECKPOINTS_PER_REQUEST as u32,
        supports_document_span: true,
        supports_hybrid_boundaries: true,
    });
    FakeProvider {
        descriptor,
        emit_telemetry: true,
        load_report: None,
    }
}

/// Build a descriptor for a thinking-capable fake with the given id + `max_new_tokens`.
pub fn thinking_descriptor(id: &str, max_new_tokens: u32) -> TextLlmDescriptor {
    TextLlmDescriptor {
        id: id.to_string(),
        family: "test".to_string(),
        backend: "unit".to_string(),
        capabilities: TextLlmCapabilities {
            supports_system_prompt: true,
            supports_thinking: true,
            max_new_tokens,
            ..Default::default()
        },
    }
}

/// A weightless tool-capable provider that echoes the offered tools back as a single
/// `get_weather(Paris)` call, so the OpenAI `tool_calls` + `finish_reason` path can be exercised
/// without real weights.
pub struct FakeToolProvider {
    pub descriptor: TextLlmDescriptor,
}

impl TextLlm for FakeToolProvider {
    fn descriptor(&self) -> &TextLlmDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TextLlmRequest) -> crate::core_llm::Result<()> {
        self.descriptor
            .capabilities
            .validate_request(&self.descriptor.id, req)
    }

    fn generate(
        &self,
        req: &TextLlmRequest,
        on_event: &mut dyn FnMut(StreamEvent),
    ) -> crate::core_llm::Result<TextLlmOutput> {
        self.validate(req)?;
        // The tools must have been threaded through to the core request.
        assert_eq!(req.tools.len(), 1);
        assert_eq!(req.tools[0].name, "get_weather");
        let usage = Usage {
            prompt_tokens: 3,
            generated_tokens: 4,
        };
        on_event(StreamEvent::Done {
            finish_reason: FinishReason::Stop,
            usage,
        });
        let mut arguments = serde_json::Map::new();
        arguments.insert("location".to_string(), serde_json::json!("Paris"));
        Ok(TextLlmOutput {
            text: String::new(),
            thinking: None,
            tool_calls: vec![crate::core_llm::ToolCall::new("get_weather", arguments)],
            usage,
            mtp: None,
            timings: None,
            decode: None,
            finish_reason: Some(FinishReason::Stop),
            kv_cache: None,
        })
    }
}

/// A loader that builds a [`FakeToolProvider`] advertising tool support.
pub fn fake_tool_loader(_: &LoadSpec) -> crate::core_llm::Result<Box<dyn TextLlm>> {
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

/// What a [`resident_loader`] has seen, keyed by namespace (the source's first path segment) so
/// concurrently running tests never see each other's providers.
#[derive(Default)]
pub struct ResidentLedger {
    /// Providers alive per namespace.
    pub alive: HashMap<String, usize>,
    /// Every load the loader was asked for: the source, and how many providers of its namespace
    /// were alive when that load started.
    pub loads: Vec<(String, usize)>,
    /// Namespaces whose next load is refused (then cleared).
    pub fail_next: std::collections::HashSet<String>,
}

impl ResidentLedger {
    /// The providers alive at the start of each load of `namespace`, in order.
    pub fn alive_at_load(&self, namespace: &str) -> Vec<usize> {
        self.loads
            .iter()
            .filter(|(source, _)| resident_namespace(source) == namespace)
            .map(|(_, alive)| *alive)
            .collect()
    }
}

pub fn resident_ledger() -> &'static Mutex<ResidentLedger> {
    static LEDGER: OnceLock<Mutex<ResidentLedger>> = OnceLock::new();
    LEDGER.get_or_init(|| Mutex::new(ResidentLedger::default()))
}

fn resident_namespace(source: &str) -> String {
    source
        .trim_start_matches('/')
        .split('/')
        .next()
        .unwrap_or_default()
        .to_string()
}

/// A [`FakeProvider`] whose lifetime the [`resident_ledger`] tracks: it stands in for a model
/// holding device memory until it is dropped.
pub struct ResidentProvider {
    inner: FakeProvider,
    namespace: String,
}

impl Drop for ResidentProvider {
    fn drop(&mut self) {
        if let Ok(mut ledger) = resident_ledger().lock() {
            if let Some(alive) = ledger.alive.get_mut(&self.namespace) {
                *alive -= 1;
            }
        }
    }
}

impl TextLlm for ResidentProvider {
    fn descriptor(&self) -> &TextLlmDescriptor {
        self.inner.descriptor()
    }

    fn validate(&self, req: &TextLlmRequest) -> crate::core_llm::Result<()> {
        self.inner.validate(req)
    }

    fn generate(
        &self,
        req: &TextLlmRequest,
        on_event: &mut dyn FnMut(StreamEvent),
    ) -> crate::core_llm::Result<TextLlmOutput> {
        self.inner.generate(req, on_event)
    }
}

/// A loader of [`ResidentProvider`]s. A namespace in `fail_next` has its next load refused; a
/// `one-fits` namespace is refused the way the runtime's load admission refuses a model that does
/// not fit (`core_llm::admit_request_memory`) while another of its providers is resident.
pub fn resident_loader(spec: &LoadSpec) -> crate::core_llm::Result<Box<dyn TextLlm>> {
    let namespace = resident_namespace(&spec.source);
    let mut ledger = resident_ledger().lock().unwrap();
    let alive = ledger.alive.get(&namespace).copied().unwrap_or(0);
    ledger.loads.push((spec.source.clone(), alive));
    if ledger.fail_next.remove(&namespace) {
        return Err(crate::core_llm::Error::Load(
            "fixture refuses this load".to_string(),
        ));
    }
    if namespace == "one-fits" && alive > 0 {
        return Err(crate::core_llm::Error::InvalidRequest(
            "request requires an estimated 2 bytes of native workspace but only 1 bytes are \
             available; reduce prompt/media length or max_new_tokens"
                .to_string(),
        ));
    }
    *ledger.alive.entry(namespace.clone()).or_default() += 1;
    Ok(Box::new(ResidentProvider {
        inner: FakeProvider {
            descriptor: thinking_descriptor("fake-resident", 8),
            emit_telemetry: false,
            load_report: None,
        },
        namespace,
    }))
}

/// Block drafts resident in a [`block_draft_loader`] provider, keyed by the target's source: one
/// while the provider holding it lives, zero once it is dropped (SW-7199 AC2).
pub fn resident_block_drafts() -> &'static Mutex<HashMap<String, usize>> {
    static RESIDENT: OnceLock<Mutex<HashMap<String, usize>>> = OnceLock::new();
    RESIDENT.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The resident block-draft count a [`block_draft_loader`] saw for a target each time it was
/// asked to load it: a reload must find the previous draft already released.
pub fn block_drafts_resident_at_load() -> &'static Mutex<HashMap<String, Vec<usize>>> {
    static SEEN: OnceLock<Mutex<HashMap<String, Vec<usize>>>> = OnceLock::new();
    SEEN.get_or_init(|| Mutex::new(HashMap::new()))
}

/// A block draft's residency, released when the provider holding it drops.
struct ResidentBlockDraft {
    target: String,
}

impl ResidentBlockDraft {
    fn admit(target: &str) -> Self {
        *resident_block_drafts()
            .lock()
            .unwrap()
            .entry(target.to_string())
            .or_default() += 1;
        Self {
            target: target.to_string(),
        }
    }
}

impl Drop for ResidentBlockDraft {
    fn drop(&mut self) {
        if let Some(count) = resident_block_drafts()
            .lock()
            .unwrap()
            .get_mut(&self.target)
        {
            *count -= 1;
        }
    }
}

/// A telemetry fake with a block draft beside it, like the MLX runtime (SW-F360): a draft source
/// containing `missing` is refused with the runtime's unreadable-config reason, a precision other
/// than 8 or 4 bits is refused by name, and a resident draft advertises `block_draft` and reports
/// its source, format and precision on every block-draft decode.
pub struct BlockDraftFake {
    inner: FakeProvider,
    draft: Option<(crate::core_llm::DecodeDraft, ResidentBlockDraft)>,
}

pub fn block_draft_loader(spec: &LoadSpec) -> crate::core_llm::Result<Box<dyn TextLlm>> {
    use crate::core_llm::{
        BlockDraftPrecision, BlockDraftRefusal, DecodeDraft, ProposerCapabilities,
        SpeculativeProposer,
    };
    let resident = resident_block_drafts()
        .lock()
        .unwrap()
        .get(&spec.source)
        .copied()
        .unwrap_or(0);
    block_drafts_resident_at_load()
        .lock()
        .unwrap()
        .entry(spec.source.clone())
        .or_default()
        .push(resident);
    let mut inner = telemetry_provider();
    let mut load_report = LoadReport::default();
    let mut draft = None;
    if let Some(source) = spec.draft_source.as_deref() {
        let outcome = if source.contains("missing") {
            Err(BlockDraftRefusal::Format(format!(
                "read {source}/config.json: not found"
            )))
        } else {
            BlockDraftPrecision::from_request(spec.draft_quantize)
        };
        match outcome {
            Ok(precision) => {
                inner
                    .descriptor
                    .capabilities
                    .speculative
                    .push(ProposerCapabilities {
                        proposer: SpeculativeProposer::BlockDraft,
                        max_depth: 8,
                        recommended_depth: 4,
                    });
                load_report.record_draft(DraftReport::resident(source));
                draft = Some((
                    DecodeDraft {
                        source: source.to_string(),
                        format: "dspine".to_string(),
                        precision: precision.label().to_string(),
                    },
                    ResidentBlockDraft::admit(&spec.source),
                ));
            }
            Err(refusal) => {
                load_report.record_draft(DraftReport::refused(source, refusal.to_string()))
            }
        }
    }
    inner.load_report = Some(load_report);
    Ok(Box::new(BlockDraftFake { inner, draft }))
}

impl TextLlm for BlockDraftFake {
    fn descriptor(&self) -> &TextLlmDescriptor {
        self.inner.descriptor()
    }

    fn load_report(&self) -> Option<LoadReport> {
        self.inner.load_report()
    }

    fn validate(&self, req: &TextLlmRequest) -> crate::core_llm::Result<()> {
        self.inner.validate(req)
    }

    fn generate(
        &self,
        req: &TextLlmRequest,
        on_event: &mut dyn FnMut(StreamEvent),
    ) -> crate::core_llm::Result<TextLlmOutput> {
        let mut output = self.inner.generate(req, on_event)?;
        if let (
            crate::core_llm::Speculative::Proposer {
                proposer: crate::core_llm::SpeculativeProposer::BlockDraft,
                depth,
            },
            Some((draft, _)),
        ) = (req.speculative_mode(), &self.draft)
        {
            // Two verify passes of `depth` drafts, five accepted in all.
            output.decode = Some(DecodeReport {
                path: "block_draft".to_string(),
                proposer: ProposerKind::BlockDraft,
                draft_tokens: Some(depth),
                draft: Some(draft.clone()),
                proposed_tokens: u64::from(depth) * 2,
                accepted_tokens: 5,
                verify_steps: 2,
                target_forwards: 3,
                ..fake_decode_report()
            });
        }
        Ok(output)
    }
}

/// Recorded request metadata from [`prefix_checkpoint_loader`], keyed by model source.
#[derive(Clone, Debug)]
pub struct RecordedCheckpointRequest {
    pub messages: Vec<(String, String)>,
    pub prefix_checkpoints: Option<PrefixCheckpointPolicy>,
}

pub fn recorded_checkpoint_requests(
) -> &'static Mutex<HashMap<String, Vec<RecordedCheckpointRequest>>> {
    static REQUESTS: OnceLock<Mutex<HashMap<String, Vec<RecordedCheckpointRequest>>>> =
        OnceLock::new();
    REQUESTS.get_or_init(|| Mutex::new(HashMap::new()))
}

#[derive(Clone)]
struct StoredFakeCheckpoint {
    key: Vec<(String, String)>,
    tokens: u64,
    kind: &'static str,
    protected: bool,
    bytes: u64,
}

#[derive(Default)]
struct CheckpointFakeStore {
    entries: Vec<StoredFakeCheckpoint>,
}

/// Multi-turn weightless fake that advertises `prefix_checkpoints`, records every request's
/// messages and policy, and simulates message/document checkpoint hits vs ordinary `off` hits.
pub struct PrefixCheckpointFake {
    source: String,
    descriptor: TextLlmDescriptor,
    store: Mutex<CheckpointFakeStore>,
}

pub fn prefix_checkpoint_loader(spec: &LoadSpec) -> crate::core_llm::Result<Box<dyn TextLlm>> {
    let mut descriptor = thinking_descriptor("fake-prefix-checkpoints", 64);
    descriptor.capabilities.prefix_checkpoints = Some(PrefixCheckpointCapabilities {
        max_checkpoints_per_request: MAX_CHECKPOINTS_PER_REQUEST as u32,
        supports_document_span: true,
        supports_hybrid_boundaries: true,
    });
    Ok(Box::new(PrefixCheckpointFake {
        source: spec.source.clone(),
        descriptor,
        store: Mutex::new(CheckpointFakeStore::default()),
    }))
}

impl TextLlm for PrefixCheckpointFake {
    fn descriptor(&self) -> &TextLlmDescriptor {
        &self.descriptor
    }

    fn load_report(&self) -> Option<LoadReport> {
        Some(LoadReport {
            prefix_cache_bytes: Some(64 << 20),
            ..LoadReport::default()
        })
    }

    fn validate(&self, req: &TextLlmRequest) -> crate::core_llm::Result<()> {
        self.descriptor
            .capabilities
            .validate_request(&self.descriptor.id, req)
    }

    fn generate(
        &self,
        req: &TextLlmRequest,
        on_event: &mut dyn FnMut(StreamEvent),
    ) -> crate::core_llm::Result<TextLlmOutput> {
        self.validate(req)?;
        let message_pairs: Vec<(String, String)> = req
            .messages
            .iter()
            .map(|m| (format!("{:?}", m.role), m.text_content()))
            .collect();
        recorded_checkpoint_requests()
            .lock()
            .unwrap()
            .entry(self.source.clone())
            .or_default()
            .push(RecordedCheckpointRequest {
                messages: message_pairs.clone(),
                prefix_checkpoints: req.prefix_checkpoints,
            });

        let policy = req.prefix_checkpoints_or(PrefixCheckpointMode::Off);
        let msg_tokens = |pair: &(String, String)| -> u64 {
            (pair.1.split_whitespace().count().max(1) as u64) + 2
        };
        let total_prompt_tokens: u64 = message_pairs.iter().map(msg_tokens).sum::<u64>() + 1;

        let mut store = self.store.lock().unwrap();
        // Match longest stored checkpoint that is a strict prefix of message_pairs.
        let best = store
            .entries
            .iter()
            .filter(|entry| {
                entry.key.len() < message_pairs.len()
                    && message_pairs.starts_with(entry.key.as_slice())
            })
            .max_by_key(|entry| (entry.tokens, entry.protected))
            .cloned();

        let (reused_tokens, matched_kind, restore_time) = match best {
            Some(hit) => (
                hit.tokens,
                Some(hit.kind.to_string()),
                Duration::from_micros(300),
            ),
            None => (0, None, Duration::ZERO),
        };
        let reread_tokens = total_prompt_tokens.saturating_sub(reused_tokens);

        let mut captured = 0u32;
        let mut captured_bytes = 0u64;
        let mut push_entry = |len: usize, kind: &'static str, protected: bool| {
            if len == 0 || len > message_pairs.len() {
                return;
            }
            let key = message_pairs[..len].to_vec();
            let tokens: u64 = key.iter().map(msg_tokens).sum();
            let bytes = tokens * 256;
            if let Some(existing) = store.entries.iter_mut().find(|e| e.key == key) {
                if protected && !existing.protected {
                    existing.protected = true;
                    existing.kind = kind;
                }
                return;
            }
            store.entries.push(StoredFakeCheckpoint {
                key,
                tokens,
                kind,
                protected,
                bytes,
            });
            captured += 1;
            captured_bytes += bytes;
        };

        if policy.is_active() {
            if let Some(PrefixDocumentSpan::Messages { count }) = policy.document {
                push_entry(count, "document", true);
            }
            if message_pairs.len() >= 2 {
                push_entry(1, "shared_root", true);
                push_entry(message_pairs.len() - 1, "message_boundary", false);
            }
            push_entry(message_pairs.len(), "conversation", false);
        } else {
            push_entry(message_pairs.len(), "conversation", false);
        }

        let resident_bytes: u64 = store.entries.iter().map(|e| e.bytes).sum();
        let protected_bytes: u64 = store
            .entries
            .iter()
            .filter(|e| e.protected)
            .map(|e| e.bytes)
            .sum();

        on_event(StreamEvent::Token {
            id: 1,
            text: "ok".to_string(),
            index: 0,
            channel: Channel::Content,
        });
        let usage = Usage {
            prompt_tokens: total_prompt_tokens as u32,
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
            mtp: None,
            timings: None,
            decode: Some(DecodeReport {
                prefix_hit_tokens: reused_tokens,
                prefix_cache: PathReport {
                    path: if reused_tokens > 0 { "hit" } else { "miss" }.to_string(),
                    reason: None,
                },
                prefix_checkpoints: Some(PrefixCheckpointReport {
                    policy: policy.label().to_string(),
                    reused_tokens,
                    reread_tokens,
                    matched_kind,
                    checkpoints_captured: captured,
                    captured_bytes,
                    restore_time,
                    capture_time: if captured > 0 {
                        Duration::from_micros(400)
                    } else {
                        Duration::ZERO
                    },
                    resident_bytes,
                    protected_bytes,
                    evicted_entries: 0,
                    fallbacks: Vec::new(),
                }),
                ..fake_decode_report()
            }),
            finish_reason: Some(FinishReason::Stop),
            kv_cache: None,
        })
    }
}

/// Weightless fake exercising SW-F357 adaptive speculative telemetry and explicit control
/// preservation across desktop IPC and OpenAI HTTP (streaming + non-streaming).
pub struct AdaptiveSpeculativeFake {
    descriptor: TextLlmDescriptor,
}

pub fn adaptive_speculative_loader(_: &LoadSpec) -> crate::core_llm::Result<Box<dyn TextLlm>> {
    let mut descriptor = thinking_descriptor("fake-adaptive-speculative", 64);
    descriptor.capabilities.speculative = vec![
        ProposerCapabilities {
            proposer: SpeculativeProposer::Mtp,
            max_depth: 7,
            recommended_depth: 3,
        },
        ProposerCapabilities {
            proposer: SpeculativeProposer::PromptLookup,
            max_depth: 7,
            recommended_depth: 4,
        },
    ];
    Ok(Box::new(AdaptiveSpeculativeFake { descriptor }))
}

impl TextLlm for AdaptiveSpeculativeFake {
    fn descriptor(&self) -> &TextLlmDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TextLlmRequest) -> crate::core_llm::Result<()> {
        self.descriptor
            .capabilities
            .validate_request(&self.descriptor.id, req)
    }

    fn generate(
        &self,
        req: &TextLlmRequest,
        on_event: &mut dyn FnMut(StreamEvent),
    ) -> crate::core_llm::Result<TextLlmOutput> {
        self.validate(req)?;
        let prompt = req
            .messages
            .last()
            .map(crate::core_llm::Message::text_content)
            .unwrap_or_default();
        let decode = match req.speculative_mode() {
            Speculative::Off => DecodeReport {
                path: "reference".to_string(),
                proposer: ProposerKind::None,
                draft_tokens: None,
                proposed_tokens: 0,
                accepted_tokens: 0,
                verify_steps: 4,
                target_forwards: 5,
                adaptive_speculative: Some(AdaptiveSpeculativeReport::explicit(
                    ProposerKind::None,
                    0,
                    None,
                    None,
                )),
                speculative_monitor: None,
                speculative_demoted_at: None,
                fallbacks: Vec::new(),
                ..fake_decode_report()
            },
            Speculative::Proposer { proposer: _, depth } => {
                let resolved = crate::core_llm::resolve_speculative(
                    req.speculative_mode(),
                    &self.descriptor.capabilities,
                );
                let effective_proposer = resolved.plan.proposer();
                let effective_depth = resolved.plan.depth().unwrap_or(0);
                let fallbacks: Vec<String> = resolved.fallback.into_iter().collect();
                DecodeReport {
                    path: if effective_proposer == ProposerKind::None {
                        "reference".to_string()
                    } else {
                        effective_proposer.label().to_string()
                    },
                    proposer: effective_proposer,
                    draft_tokens: (effective_depth > 0).then_some(effective_depth),
                    proposed_tokens: u64::from(effective_depth) * 2,
                    accepted_tokens: u64::from(effective_depth),
                    verify_steps: 2,
                    target_forwards: 3,
                    adaptive_speculative: Some(AdaptiveSpeculativeReport::explicit(
                        effective_proposer,
                        if effective_proposer == ProposerKind::None {
                            0
                        } else {
                            depth.min(effective_depth)
                        },
                        fallbacks.first().cloned(),
                        None,
                    )),
                    speculative_monitor: None,
                    speculative_demoted_at: None,
                    fallbacks,
                    ..fake_decode_report()
                }
            }
            Speculative::Auto => match prompt.as_str() {
                "adaptive-switch" => DecodeReport {
                    path: "prompt_lookup".to_string(),
                    proposer: ProposerKind::PromptLookup,
                    draft_tokens: Some(4),
                    proposed_tokens: 36,
                    accepted_tokens: 24,
                    verify_steps: 10,
                    target_forwards: 11,
                    speculative_demoted_at: None,
                    speculative_monitor: Some(MonitorDecision {
                        window: 1,
                        verifies: 10,
                        accepted: 24,
                        timed_steps: 8,
                        timed_tokens: 32,
                        timed_ns: 64_000_000,
                        plain_step_ns: Some(4_000_000),
                        basis: DemotionBasis::Measured,
                        demoted: false,
                    }),
                    adaptive_speculative: Some(AdaptiveSpeculativeReport {
                        adaptive: true,
                        candidates: vec![
                            ProposerKind::Mtp,
                            ProposerKind::PromptLookup,
                            ProposerKind::None,
                        ],
                        initial_proposer: ProposerKind::Mtp,
                        initial_depth: 3,
                        final_proposer: ProposerKind::PromptLookup,
                        final_depth: 4,
                        depth_changes: 1,
                        proposer_switches: 1,
                        events: vec![
                            AdaptiveSpeculativeEvent {
                                token_offset: 0,
                                verify_step: 0,
                                kind: AdaptiveEventKind::Selected,
                                from_proposer: ProposerKind::Mtp,
                                from_depth: 3,
                                to_proposer: ProposerKind::Mtp,
                                to_depth: 3,
                                reason: "auto selected `mtp` at depth 3".to_string(),
                            },
                            AdaptiveSpeculativeEvent {
                                token_offset: 16,
                                verify_step: 5,
                                kind: AdaptiveEventKind::DepthIncreased,
                                from_proposer: ProposerKind::Mtp,
                                from_depth: 3,
                                to_proposer: ProposerKind::Mtp,
                                to_depth: 4,
                                reason: "acceptance 0.85 >= 0.75".to_string(),
                            },
                            AdaptiveSpeculativeEvent {
                                token_offset: 32,
                                verify_step: 9,
                                kind: AdaptiveEventKind::ProposerSwitched,
                                from_proposer: ProposerKind::Mtp,
                                from_depth: 4,
                                to_proposer: ProposerKind::PromptLookup,
                                to_depth: 4,
                                reason: "probe `prompt_lookup` beat incumbent by 22%".to_string(),
                            },
                        ],
                        plain_reason: None,
                        cost: Some(AdaptiveCostSummary {
                            proposal_ns: 4_000_000,
                            verify_ns: 52_000_000,
                            rollback_ns: 2_000_000,
                            sync_ns: 6_000_000,
                            total_ns: 64_000_000,
                            host_syncs: 10,
                            committed_tokens: 34,
                        }),
                    }),
                    fallbacks: Vec::new(),
                    ..fake_decode_report()
                },
                "adaptive-fallback" => DecodeReport {
                    path: "prompt_lookup".to_string(),
                    proposer: ProposerKind::PromptLookup,
                    draft_tokens: Some(4),
                    proposed_tokens: 16,
                    accepted_tokens: 1,
                    verify_steps: 8,
                    target_forwards: 9,
                    speculative_demoted_at: Some(24),
                    speculative_monitor: Some(MonitorDecision {
                        window: 1,
                        verifies: 8,
                        accepted: 1,
                        timed_steps: 8,
                        timed_tokens: 9,
                        timed_ns: 100_000_000,
                        plain_step_ns: Some(8_000_000),
                        basis: DemotionBasis::Measured,
                        demoted: true,
                    }),
                    adaptive_speculative: Some(AdaptiveSpeculativeReport {
                        adaptive: true,
                        candidates: vec![ProposerKind::PromptLookup, ProposerKind::None],
                        initial_proposer: ProposerKind::PromptLookup,
                        initial_depth: 4,
                        final_proposer: ProposerKind::None,
                        final_depth: 0,
                        depth_changes: 0,
                        proposer_switches: 1,
                        events: vec![
                            AdaptiveSpeculativeEvent {
                                token_offset: 0,
                                verify_step: 0,
                                kind: AdaptiveEventKind::Selected,
                                from_proposer: ProposerKind::PromptLookup,
                                from_depth: 4,
                                to_proposer: ProposerKind::PromptLookup,
                                to_depth: 4,
                                reason: "auto selected `prompt_lookup` at depth 4".to_string(),
                            },
                            AdaptiveSpeculativeEvent {
                                token_offset: 24,
                                verify_step: 8,
                                kind: AdaptiveEventKind::FallbackToPlain,
                                from_proposer: ProposerKind::PromptLookup,
                                from_depth: 4,
                                to_proposer: ProposerKind::None,
                                to_depth: 0,
                                reason: "measured gain 0.72 below break-even".to_string(),
                            },
                        ],
                        plain_reason: Some("measured gain 0.72 below break-even".to_string()),
                        cost: Some(AdaptiveCostSummary {
                            proposal_ns: 500_000,
                            verify_ns: 95_000_000,
                            rollback_ns: 1_500_000,
                            sync_ns: 3_000_000,
                            total_ns: 100_000_000,
                            host_syncs: 8,
                            committed_tokens: 9,
                        }),
                    }),
                    fallbacks: Vec::new(),
                    ..fake_decode_report()
                },
                "adaptive-plain" => {
                    let reason = "speculative: auto found no proposer this model can run \
                                  (it advertises none; decoded without a proposer)"
                        .to_string();
                    DecodeReport {
                        path: "reference".to_string(),
                        proposer: ProposerKind::None,
                        draft_tokens: None,
                        proposed_tokens: 0,
                        accepted_tokens: 0,
                        verify_steps: 4,
                        target_forwards: 5,
                        speculative_demoted_at: None,
                        speculative_monitor: None,
                        adaptive_speculative: Some(AdaptiveSpeculativeReport::auto_stayed_plain(
                            reason.clone(),
                            vec![ProposerKind::None],
                        )),
                        fallbacks: vec![reason],
                        ..fake_decode_report()
                    }
                }
                _ => fake_decode_report(),
            },
        };
        on_event(StreamEvent::Token {
            id: 1,
            text: "ok".to_string(),
            index: 0,
            channel: Channel::Content,
        });
        let usage = Usage {
            prompt_tokens: 4,
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
            mtp: None,
            timings: None,
            decode: Some(decode),
            finish_reason: Some(FinishReason::Stop),
            kv_cache: None,
        })
    }
}
