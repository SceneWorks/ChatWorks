#[cfg(all(
    not(all(target_os = "macos", target_arch = "aarch64")),
    feature = "cpu",
    feature = "cuda"
))]
compile_error!("ChatWorks CPU and CUDA inference profiles are mutually exclusive");
#[cfg(all(target_os = "macos", not(target_arch = "aarch64"), feature = "cuda"))]
compile_error!("Candle CUDA is unavailable on macOS; Intel Macs require the `cpu` feature");
#[cfg(all(
    not(all(target_os = "macos", target_arch = "aarch64")),
    not(any(feature = "cpu", feature = "cuda"))
))]
compile_error!(
    "ChatWorks must enable either the `cpu` or `cuda` feature outside Apple Silicon macOS"
);

// One platform bundle is the product's inference composition root. Re-exporting its neutral
// contract preserves ChatWorks' public type paths without introducing a separately pinned source.
#[cfg(all(
    not(all(target_os = "macos", target_arch = "aarch64")),
    feature = "cpu",
    not(feature = "cuda")
))]
pub use runtime_cpu::core_llm;
#[cfg(all(not(target_os = "macos"), feature = "cuda", not(feature = "cpu")))]
pub use runtime_cuda::core_llm;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub use runtime_macos::core_llm;

pub mod app_settings;
pub mod conversations;
pub mod engine;
pub mod fsutil;
mod inference_runtime;
pub mod mlx_metallib;
pub mod model_registry;
pub mod profile;
pub mod server;
pub mod tools;

#[cfg(test)]
pub mod test_support;
