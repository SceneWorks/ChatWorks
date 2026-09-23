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
    Channel, CudaGraphsReport, DecodeReport, FinishReason, GenerationTimings, LoadReport, LoadSpec,
    MtpCapabilities, MtpStats, PathReport, ProjectionReport, ProposerKind, Quantize, StreamEvent,
    TextLlm, TextLlmCapabilities, TextLlmDescriptor, TextLlmOutput, TextLlmRequest, ThinkingMode,
    Usage,
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
/// decode GEMV after a cuBLASLt prefill (sc-24139).
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
        nvfp4_projections: PathReport {
            path: "mixed".to_string(),
            reason: Some("rows".to_string()),
        },
        fused_primitives: PathReport {
            path: "fused".to_string(),
            reason: None,
        },
        target_forwards: 2,
        proposed_tokens: 4,
        accepted_tokens: 3,
        replay_forwards: 0,
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
