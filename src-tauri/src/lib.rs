#[cfg(all(not(target_os = "macos"), feature = "cpu", feature = "cuda"))]
compile_error!("ChatWorks CPU and CUDA inference profiles are mutually exclusive");
#[cfg(all(not(target_os = "macos"), not(any(feature = "cpu", feature = "cuda"))))]
compile_error!("a non-macOS ChatWorks build must enable either the `cpu` or `cuda` feature");

// One platform bundle is the product's inference composition root. Re-exporting its neutral
// contract preserves ChatWorks' public type paths without introducing a separately pinned source.
#[cfg(all(not(target_os = "macos"), feature = "cpu", not(feature = "cuda")))]
pub use runtime_cpu::core_llm;
#[cfg(all(not(target_os = "macos"), feature = "cuda", not(feature = "cpu")))]
pub use runtime_cuda::core_llm;
#[cfg(target_os = "macos")]
pub use runtime_macos::core_llm;

pub mod app_settings;
pub mod conversations;
pub mod engine;
pub mod fsutil;
mod inference_runtime;
pub mod model_registry;
pub mod profile;
pub mod server;
pub mod tools;

#[cfg(test)]
pub mod test_support;
