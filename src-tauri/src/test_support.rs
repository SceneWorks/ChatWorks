//! Shared test-only scaffolding for the engine + server unit tests (code-review F-012).
//!
//! Both `engine::tests` and `server::tests` independently defined a `FakeProvider` implementing
//! [`core_llm::TextLlm`], and the two copies had already drifted (descriptor capabilities, token
//! ids). This module hosts one canonical fake the two `#[cfg(test)]` blocks import, so they stay
//! honest. The fake emits a reasoning token + a content token (unless thinking is disabled), then a
//! `Stop` finish — enough to exercise the streaming, reasoning, and finish-reason paths without real
//! weights. A tool-call variant is provided for the server's tool-calling tests.

#![cfg(test)]

use core_llm::{
    Channel, FinishReason, LoadSpec, StreamEvent, TextLlm, TextLlmCapabilities,
    TextLlmDescriptor, TextLlmOutput, TextLlmRequest, ThinkingMode, Usage,
};

/// A weightless `TextLlm` that streams a reasoning token then a content token and finishes `Stop`.
/// The reasoning token is emitted only when the request's thinking mode is not `Disabled`.
pub struct FakeProvider {
    pub descriptor: TextLlmDescriptor,
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
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, core_llm::Role::User);
        assert_eq!(
            req.messages[0].content,
            vec![core_llm::Content::Text("hello".to_string())]
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
            finish_reason: Some(FinishReason::Stop),
        })
    }
}

/// A loader that builds a [`FakeProvider`] whose descriptor advertises system prompts + thinking and
/// caps `max_new_tokens` at 8 (so over-limit requests are rejected by `validate`).
pub fn fake_loader(_: &LoadSpec) -> core_llm::Result<Box<dyn TextLlm>> {
    Ok(Box::new(FakeProvider {
        descriptor: thinking_descriptor("fake", 8),
    }))
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
            tool_calls: vec![core_llm::ToolCall::new("get_weather", arguments)],
            usage,
            finish_reason: Some(FinishReason::Stop),
        })
    }
}

/// A loader that builds a [`FakeToolProvider`] advertising tool support.
pub fn fake_tool_loader(_: &LoadSpec) -> core_llm::Result<Box<dyn TextLlm>> {
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
