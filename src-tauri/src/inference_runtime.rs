//! ChatWorks' explicit, process-wide inference composition.

use std::sync::OnceLock;

use crate::core_llm::{LoadSpec, TextLlm, TextLlmRegistration, TextLlmRegistry};

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

        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        assert_eq!(
            ids,
            ["candle-llama", "candle-llava", "candle-starvector-1b"]
        );
        #[cfg(all(
            not(all(target_os = "macos", target_arch = "aarch64")),
            feature = "cpu"
        ))]
        assert_eq!(super::execution_backend(), "candle-cpu");
        #[cfg(all(not(target_os = "macos"), feature = "cuda", not(feature = "cpu")))]
        assert_eq!(super::execution_backend(), "candle-cuda");
    }
}
