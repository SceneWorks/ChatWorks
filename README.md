# ChatWorks

A SceneWorks-styled desktop app for serving local LLMs. ChatWorks is a [Tauri](https://tauri.app/)
application: a Rust backend that loads models and runs inference, fronted by an
OpenAI-compatible HTTP server and a web chat UI. The inference backend is selected per-platform
at build time — Apple **MLX** on macOS, cross-platform **Candle** on Windows/Linux — through one
immutable [`SceneWorks/inference`](https://github.com/SceneWorks/inference) runtime release. The
current cutover pin is `runtime-2026.07.2`; the bundle re-exports the neutral `core-llm` contract
and explicitly lists every available provider.

- Running on Windows/Linux (Candle): see [WINDOWS.md](WINDOWS.md).
- Sending video over the OpenAI-compatible API: see [docs/VIDEO_API.md](docs/VIDEO_API.md).
- Release packaging provisions pinned FFmpeg/ffprobe sidecars with `npm run provision:media`; see [the FFmpeg notice](third_party/ffmpeg/NOTICE.md).

## Package validation

`.github/workflows/package-validation.yml` builds native packages on standard GitHub-hosted
runners for Apple Silicon macOS (`aarch64-apple-darwin`), Intel macOS
(`x86_64-apple-darwin`), x64 Linux (`x86_64-unknown-linux-gnu`), arm64 Linux
(`aarch64-unknown-linux-gnu`), and x64 Windows (`x86_64-pc-windows-msvc`). The jobs build the
checksum-pinned FFmpeg 9.0 source on each native runner, build an unsigned `.app`, `.deb`,
or NSIS package, then inspect that package for both FFmpeg and ffprobe.

All Git-sourced runtime dependencies are public, so package validation requires no repository
credential or custom Actions secret. The workflow's built-in token retains read-only contents
permission; packaging does not publish artifacts, create a release, or require production signing
credentials.

## Quick start (development)

The Rust backend resolves its runtime from the public
[`SceneWorks/inference`](https://github.com/SceneWorks/inference) repository, so a normal Git client
can fetch it without extra credentials.

```sh
npm install
npm run tauri:dev
```

ChatWorks discovers already-downloaded Hugging Face models from the standard cache
(`~/.cache/huggingface/hub`, plus `HF_HOME` / `HUGGINGFACE_HUB_CACHE` if set) and can import
models through the app.

## Supported models

Model support is detected from a snapshot's `config.json` at load time (see
`src-tauri/src/model_registry.rs`), then served by the platform's inference provider
(`mlx-llama` on macOS, `candle-llama` elsewhere). Vision-capable checkpoints are recognized by
their `model_type` plus a `vision_config`.

| Model | Family (`model_type`) | Platform / backend | Modalities | Tool calling |
| ----- | --------------------- | ------------------ | ---------- | ------------ |
| **Qwen3-VL-8B-Instruct** | `qwen3_vl` | macOS / Apple Silicon (MLX) | Text, image, multi-image, **video** | Yes |
| Qwen3.6 (e.g. 27B) | `qwen3_5` | macOS / Apple Silicon (MLX) | Text, image, multi-image | Yes |
| Text-only Qwen / LLaMA-family checkpoints | various | macOS (MLX) · Windows/Linux (Candle) | Text | Model-dependent |

### Qwen3-VL-8B-Instruct

Qwen3-VL-8B-Instruct (`Qwen/Qwen3-VL-8B-Instruct`) is served end-to-end on **macOS / Apple
Silicon** via the MLX (`mlx-llama`) provider. It is a full vision-language model and is exercised
by the real-weights tests under `src-tauri/tests/qwen3vl_*.rs` (gated on `MLX_LLM_QWEN3VL_MODEL`).

- **Platform:** macOS / Apple Silicon only (MLX / Apple Metal). On Windows/Linux the Candle
  backend does not yet serve this checkpoint — see Limitations.
- **Modality scope:**
  - Single image and **multi-image** (image ordering is preserved across content parts).
  - **Video**, sent as pre-sampled frames with optional per-frame timestamps for Qwen3-VL's
    **Text–Timestamp Alignment** (temporal questions). See [docs/VIDEO_API.md](docs/VIDEO_API.md).
    The client can send pre-sampled frames or a bounded `video_url.url` file/URL source; see
    [the video API](docs/VIDEO_API.md).
  - 32-language OCR, spatial / 2D grounding, and long context (the checkpoint advertises a
    262 K-token window).
  - **Tool calling**, including in the same turn as an image (the model emits parseable
    `<tool_call>` blocks surfaced as OpenAI `tool_calls`).
- **Quantization:** load dense (bf16) or quantize on the fly to **q4** / **q8**
  (`LoadModelRequest { quantize: Some(Q4 | Q8) }`). The current quantized configurations are
  **mixed-precision** — the language decoder is quantized while the ViT vision tower stays dense
  (fully quantizing the ViT tower is tracked in sc-8118).

#### Validated behavior (macOS / Apple Silicon, sc-8083)

Close-out QA was run against the real `Qwen/Qwen3-VL-8B-Instruct` snapshot (rev `0c351dd0`)
through the ChatWorks OpenAI server on an Apple M-series machine. All dimensions passed at
dense, q4, and q8:

| Dimension | Result |
| --------- | ------ |
| Single-image grounding | PASS — "solid red circle" for a red circle on white |
| Multi-image (ordering) | PASS — `first=red, second=blue` |
| OCR (32-language claim) | PASS — English, Chinese (你好世界), Russian (Привет мир), Arabic (مرحبا بالعالم), and digits all transcribed correctly |
| Spatial / 2D grounding | PASS — top-left RED / bottom-right YELLOW on a 4-quadrant image |
| Long context | PASS — needle (passphrase) recovered from a ~10 K-token prompt |

Indicative performance (M-series, single stream; sustained text decode, wall-clock incl. small
prefill; peak resident memory via `/usr/bin/time -l`):

| Config | Sustained decode | Peak RSS (load + serve) | Minimum RAM |
| ------ | ---------------- | ----------------------- | ----------- |
| Dense (bf16) | ~31 tok/s | ~16.8 GiB | 24 GB+ recommended |
| q8 (mixed) | ~54 tok/s | ~4.6 GiB | 16 GB |
| q4 (mixed) | ~80 tok/s | ~4.9 GiB | 16 GB |

Numbers are indicative, not a benchmark: decode rate falls with longer context (a ~10 K-token
prefill drops effective throughput sharply), and peak RSS includes the transient quantize/convert
working set and the KV cache. On-disk weights are ~16 GB (bf16). Quantized peak RSS is dominated
by the dequantize-from-bf16 working set at load, so q4 and q8 land in the same ~4.6–4.9 GiB band.

## Limitations and deferred scope

- **Cross-platform (Candle) Qwen3-VL** — Qwen3-VL is macOS/MLX-only today. Bringing the VLM to the
  Candle backend (Windows/Linux) is tracked under epic **sc-8084**.
- **Fully-quantized ViT tower** — q4/q8 quantize the language decoder but keep the ViT vision tower
  dense (mixed precision). Quantizing the vision tower is tracked in **sc-8118**.

## Documentation

CODEGRAPH.md is auto-generated by the CodeGraph tool and must not be hand-edited; it regenerates
from the codebase. This README (plus WINDOWS.md and docs/) is the hand-authored documentation home.
