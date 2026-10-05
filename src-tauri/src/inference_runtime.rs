//! ChatWorks' explicit, process-wide inference composition.

use std::sync::OnceLock;

use crate::core_llm::{
    BackendCapabilities, DecodeBackend, FeatureSupport, LoadSpec, Speculative, TextLlm,
    TextLlmRegistration, TextLlmRegistry,
};

#[cfg(all(
    not(all(target_os = "macos", target_arch = "aarch64")),
    feature = "cpu",
    not(feature = "cuda")
))]
use runtime_cpu as platform_runtime;
#[cfg(all(not(target_os = "macos"), feature = "cuda", not(feature = "cpu")))]
use runtime_cuda as platform_runtime;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use runtime_macos as platform_runtime;

fn catalog() -> &'static platform_runtime::RuntimeCatalog {
    static CATALOG: OnceLock<platform_runtime::RuntimeCatalog> = OnceLock::new();
    CATALOG.get_or_init(|| {
        platform_runtime::catalog().unwrap_or_else(|error| {
            panic!("the compile-time inference bundle must form a valid runtime catalog: {error}")
        })
    })
}

pub(crate) fn text() -> &'static TextLlmRegistry {
    catalog().text()
}

pub(crate) fn load_for_model(spec: &LoadSpec) -> crate::core_llm::Result<Box<dyn TextLlm>> {
    text().load_for_model(spec)
}

pub(crate) fn textllms() -> impl ExactSizeIterator<Item = &'static TextLlmRegistration> {
    text().registrations()
}

/// What the linked runtime can serve on this host before any load (sc-24139): the load device,
/// its CUDA compute capability, and whether NVFP4 weights and CUDA graphs are available, each with
/// the runtime's own refusal reason when not. The runtime probes once per process (on CUDA it
/// opens the load device the way a load does); the answer is cached here too.
pub(crate) fn backend_capabilities() -> &'static BackendCapabilities {
    static CAPABILITIES: OnceLock<BackendCapabilities> = OnceLock::new();
    CAPABILITIES.get_or_init(platform_runtime::text_backend_capabilities)
}

/// Whether an NVFP4 load of the snapshot at `spec.source` can pass every gate the linked runtime's
/// load runs before it reads a weight (sc-24139): the runtime answers for the provider it would
/// load the snapshot with, from that provider's own load gates, then the device gate. Reads only
/// the snapshot's `config.json`; asked before a snapshot is registered (or its weights downloaded)
/// as NVFP4. ChatWorks keeps no model-family rule of its own.
pub(crate) fn nvfp4_support(spec: &LoadSpec) -> FeatureSupport {
    platform_runtime::text_nvfp4_support(spec)
}

/// The runtime's decode-defaults row (epic sc-24432 E5, `core_llm::DecodeBackend`) for the backend
/// and device a load lands on here, read off the runtime's own capability report: MLX, else the
/// Candle row for the load device (`cuda:*`, `metal:*`, else the CPU — a CUDA build with no CUDA
/// device loads on the CPU, and takes the CPU row).
pub(crate) fn decode_backend() -> DecodeBackend {
    decode_backend_for(backend_capabilities())
}

fn decode_backend_for(capabilities: &BackendCapabilities) -> DecodeBackend {
    if capabilities.backend == "mlx" {
        DecodeBackend::Mlx
    } else if capabilities.device.starts_with("cuda") {
        DecodeBackend::CandleCuda
    } else if capabilities.device.starts_with("metal") {
        DecodeBackend::CandleMetal
    } else {
        DecodeBackend::CandleCpu
    }
}

/// The speculative option a request that names none runs under on this host (epic sc-24432 E5):
/// the runtime's per-backend defaults table (`core_llm::speculative_default`, sc-24446) for
/// [`decode_backend`]. The one place ChatWorks reads the default; it keeps no copy of its own.
pub(crate) fn speculative_default() -> Speculative {
    crate::core_llm::speculative_default(decode_backend())
}

/// Whether a load that leaves `LoadSpec::cuda_graphs` unset captures CUDA graphs on this host
/// (sc-24446): the runtime's defaults-table row for [`decode_backend`], where the runtime offers
/// the switch at all — a build or stream that can never capture (`flash-attn`, the legacy stream)
/// reports it unavailable and runs eagerly. A process `CANDLE_LLM_CUDA_GRAPHS` override still wins
/// at load; the load report says what ran.
pub(crate) fn cuda_graphs_default() -> bool {
    cuda_graphs_default_for(backend_capabilities())
}

fn cuda_graphs_default_for(capabilities: &BackendCapabilities) -> bool {
    capabilities.cuda_graphs.supported && decode_backend_for(capabilities).defaults().cuda_graphs
}

pub(crate) const fn execution_backend() -> &'static str {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        "mlx"
    }
    #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
    {
        "candle-cuda"
    }
    #[cfg(all(
        not(all(target_os = "macos", target_arch = "aarch64")),
        not(feature = "cuda")
    ))]
    {
        "candle-cpu"
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn composition_is_explicit_without_loading_weights() {
        let ids: Vec<String> = super::textllms()
            .map(|registration| (registration.descriptor)().id)
            .collect();

        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        assert_eq!(super::execution_backend(), "mlx");
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        assert_eq!(
            ids,
            [
                "mlx-llama",
                "mlx-joycaption",
                "mlx-starvector-1b",
                "mlx-starvector-8b"
            ]
        );

        #[cfg(all(
            not(all(target_os = "macos", target_arch = "aarch64")),
            not(feature = "cuda")
        ))]
        assert_eq!(
            ids,
            ["candle-llama", "candle-llava", "candle-starvector-1b"]
        );
        // The CUDA bundle composes the StarVector-8B provider as well (`cuda_text_registry`).
        #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
        assert_eq!(
            ids,
            [
                "candle-llama",
                "candle-llava",
                "candle-starvector-1b",
                "candle-starvector-8b"
            ]
        );
        #[cfg(all(
            not(all(target_os = "macos", target_arch = "aarch64")),
            feature = "cpu"
        ))]
        assert_eq!(super::execution_backend(), "candle-cpu");
        #[cfg(all(not(target_os = "macos"), feature = "cuda", not(feature = "cpu")))]
        assert_eq!(super::execution_backend(), "candle-cuda");
    }

    /// AC2 (sc-24139): the runtime's capability report is the one source the weight-format and
    /// CUDA-graph controls are gated on. Off the CUDA build both are unavailable with a reason.
    #[test]
    fn backend_capabilities_come_from_the_linked_runtime() {
        let caps = super::backend_capabilities();
        assert_eq!(caps.backend, super::execution_backend());
        // The CUDA build reports a CUDA load device — the switch offered, the compute capability
        // reported, NVFP4 following the sm_120 floor with the gate's reason — or, with no device
        // to open, refuses both features with a reason. Never a silent pass: `REQUIRE_CUDA=1` (a
        // GPU lane) makes a missing device a failure.
        #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
        if caps.device.starts_with("cuda:") {
            assert!(caps.cuda_graphs.supported, "{:?}", caps.cuda_graphs);
            let (major, _) = caps.compute_capability.expect("a CUDA device reports it");
            assert_eq!(caps.nvfp4.supported, major >= 12, "{:?}", caps.nvfp4);
            if !caps.nvfp4.supported {
                assert!(caps.nvfp4.reason.as_deref().unwrap().contains("sm_120"));
            }
        } else {
            assert!(
                std::env::var("REQUIRE_CUDA").as_deref() != Ok("1"),
                "REQUIRE_CUDA=1 but the CUDA build found no CUDA device: {}",
                caps.device
            );
            for feature in [&caps.nvfp4, &caps.cuda_graphs] {
                assert!(!feature.supported, "{feature:?}");
                assert!(
                    feature.reason.as_deref().is_some_and(|r| !r.is_empty()),
                    "{feature:?}"
                );
            }
        }
        #[cfg(not(all(not(target_os = "macos"), feature = "cuda")))]
        {
            assert!(!caps.nvfp4.supported);
            assert!(caps.nvfp4.reason.as_deref().unwrap().starts_with("nvfp4: "));
            assert!(!caps.cuda_graphs.supported);
            assert!(caps.cuda_graphs.reason.is_some());
        }
    }

    /// sc-24446: the defaults row follows the runtime's backend and load device, and the CUDA-graph
    /// default is that row's value only where the runtime offers the switch.
    #[test]
    fn decode_defaults_follow_the_runtime_backend_and_device() {
        use crate::core_llm::{BackendCapabilities, DecodeBackend, FeatureSupport};
        let caps = |backend: &str, device: &str, graphs: bool| BackendCapabilities {
            backend: backend.to_string(),
            device: device.to_string(),
            cuda_graphs: if graphs {
                FeatureSupport::available()
            } else {
                FeatureSupport::unavailable("cuda_graphs: test")
            },
            ..BackendCapabilities::default()
        };
        for (backend, device, graphs, row) in [
            ("mlx", "metal", false, DecodeBackend::Mlx),
            ("candle-cuda", "cuda:0", true, DecodeBackend::CandleCuda),
            ("candle-cuda", "cpu", false, DecodeBackend::CandleCpu),
            ("candle-metal", "metal:0", false, DecodeBackend::CandleMetal),
            ("candle-cpu", "cpu", false, DecodeBackend::CandleCpu),
        ] {
            let caps = caps(backend, device, graphs);
            assert_eq!(super::decode_backend_for(&caps), row, "{backend} {device}");
            assert_eq!(
                super::cuda_graphs_default_for(&caps),
                graphs && row.defaults().cuda_graphs,
                "{backend} {device}"
            );
        }
        // A CUDA device whose switch the runtime refuses (flash-attn, legacy stream) runs eagerly.
        assert!(!super::cuda_graphs_default_for(&caps(
            "candle-cuda",
            "cuda:0",
            false
        )));
        // The host's answers are the table's, for the row the linked runtime reports.
        assert_eq!(
            super::speculative_default(),
            crate::core_llm::speculative_default(super::decode_backend())
        );
        assert_eq!(
            super::cuda_graphs_default(),
            super::cuda_graphs_default_for(super::backend_capabilities())
        );
    }
}
