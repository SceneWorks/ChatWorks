//! ChatWorks' explicit, process-wide inference composition.

use std::sync::OnceLock;

use crate::core_llm::{LoadSpec, TextLlm, TextLlmRegistration, TextLlmRegistry};

#[cfg(all(not(target_os = "macos"), feature = "cpu", not(feature = "cuda")))]
use runtime_cpu as platform_runtime;
#[cfg(all(not(target_os = "macos"), feature = "cuda", not(feature = "cpu")))]
use runtime_cuda as platform_runtime;
#[cfg(target_os = "macos")]
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
    #[cfg(target_os = "macos")]
    {
        "mlx"
    }
    #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
    {
        "candle-cuda"
    }
    #[cfg(all(not(target_os = "macos"), not(feature = "cuda")))]
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

        #[cfg(target_os = "macos")]
        assert_eq!(
            ids,
            ["mlx-llama", "mlx-joycaption", "mlx-starvector-1b", "mlx-starvector-8b"]
        );

        #[cfg(not(target_os = "macos"))]
        assert_eq!(ids, ["candle-llama", "candle-llava"]);
    }
}
