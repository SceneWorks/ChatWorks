# Running ChatWorks on Windows

ChatWorks runs on Windows as well as macOS. The only platform-specific piece is the
inference backend, which is selected automatically at build time:

| Platform        | Backend     | Provider id     | Default compute |
| --------------- | ----------- | --------------- | --------------- |
| macOS (Apple)   | MLX         | `mlx-llama`     | Apple Metal     |
| Windows / Linux | Candle      | `candle-llama`  | CPU             |

Both backends implement the same [`core-llm`](https://github.com/SceneWorks/core-llm)
contract, so the engine, OpenAI-compatible server, model import, and UI are identical
across platforms. The selection lives in `src-tauri/Cargo.toml` (target-specific
dependencies) and `src-tauri/src/lib.rs` (which backend crate gets linked).

## Prerequisites

- **Rust** (stable, MSVC toolchain): `rustup default stable-x86_64-pc-windows-msvc`
- **Visual Studio C++ Build Tools** (the MSVC linker that the Rust MSVC toolchain needs)
- **Node.js** >= 20
- **WebView2 runtime** — preinstalled on Windows 11; on older Windows install the
  Microsoft Edge WebView2 Evergreen runtime
- *(optional, GPU)* **NVIDIA CUDA toolkit** — only if building the Candle `cuda` feature

## Run (development)

```powershell
npm install
npm run tauri:dev
```

## Build an installer

```powershell
npm run tauri:build
```

This produces MSI and NSIS installers under
`src-tauri/target/release/bundle/`.

## GPU acceleration (optional)

The default Windows build runs Candle on the **CPU**, which works everywhere but is slow
for larger models. To use an NVIDIA GPU, edit the `candle-llm` dependency in
`src-tauri/Cargo.toml`:

```toml
[target.'cfg(not(target_os = "macos"))'.dependencies]
candle-llm = { git = "https://github.com/SceneWorks/candle-llm", branch = "main", features = ["cuda"] }
```

Then rebuild. This requires the CUDA toolkit to be installed. (`features = ["flash-attn"]`
additionally enables fused FlashAttention-2 kernels.)

## Model cache

ChatWorks discovers already-downloaded HuggingFace models from the standard cache
locations, which on Windows is `%USERPROFILE%\.cache\huggingface\hub` (plus `HF_HOME`
or `HUGGINGFACE_HUB_CACHE` if you set them). Models imported through the app are stored
under the app data directory (`%APPDATA%\net.trefry.chatworks`).
