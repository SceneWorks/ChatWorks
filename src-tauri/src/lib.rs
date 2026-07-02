// Link the platform's inference backend so its `core-llm` provider registers itself
// (registration is a link-time `inventory::submit!`, pulled in only when the crate is linked).
// macOS uses the Apple MLX backend; every other platform uses the cross-platform Candle backend.
#[cfg(target_os = "macos")]
use mlx_llm as _;

#[cfg(not(target_os = "macos"))]
use candle_llm as _;

pub mod app_settings;
pub mod conversations;
pub mod engine;
pub mod fsutil;
pub mod model_registry;
pub mod server;
pub mod tools;

#[cfg(test)]
pub mod test_support;
