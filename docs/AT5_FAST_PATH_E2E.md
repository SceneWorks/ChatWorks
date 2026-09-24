# AT5: Qwen3.8-27B fast-path end-to-end check

Epic sc-24128 (fast decode on Blackwell), acceptance test 5, story sc-24140. The check runs ChatWorks
end to end on the pinned inference runtime **with the shipped defaults**: Qwen3.8-27B streams,
calls a tool, is cancelled mid-stream, and is reloaded and unloaded. An automated harness covers
the engine and the OpenAI server. A short manual checklist covers the Tauri window, which the
harness cannot drive.

The terminal story runs both **once on the final pin** (the released runtime tag) and keeps the
sealed record.

## The harness: `src-tauri/tests/qwen38_fast_path_e2e.rs`

The test is `#[ignore]`d, so only an explicit run starts it. It needs:

- a Candle CUDA build (`--no-default-features --features cuda`);
- one GPU pinned by PCI bus order;
- a Qwen3.8-27B snapshot;
- an output path **outside the repository**.

The Qwen3-8B snapshot is optional; without it, step 6 is skipped.

```powershell
# PowerShell with the MSVC 14.44 vcvars loaded (Windows), or any shell on Linux.
# Check `nvidia-smi -i 0` first: the GPU must be idle, and only one 27B is loaded at a time.
$env:CUDA_DEVICE_ORDER = "PCI_BUS_ID"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:CHATWORKS_QWEN38_SNAPSHOT = "E:\huggingface\hub\models--Qwen--Qwen3.8-27B\snapshots\<rev>"
$env:CHATWORKS_QWEN3_8B_SNAPSHOT = "E:\huggingface\hub\models--Qwen--Qwen3-8B\snapshots\<rev>"
$env:CHATWORKS_E2E_OUTPUT = "C:\evidence\sc-24128-at5-<date>.json"
cd src-tauri
cargo test --no-default-features --features cuda --test qwen38_fast_path_e2e -- --ignored --nocapture
```

On an RTX Pro 6000 the whole run takes about two minutes. It prints one `[PASS]` or `[FAIL]` line per
check. If any check failed, the test fails at the end.

### What it checks

| Step | What runs | Checks |
|---|---|---|
| 0 | Engine start | The runtime is `candle-cuda` and the load device is `sm_120`. |
| 1 | Load Qwen3.8-27B through `model_registry::app_load_request`, the same function the Models screen uses, under `AppSettings::default()` | `mtp_mode = auto` with 3 draft tokens and CUDA graphs off. The load request is dense and sends `cuda_graphs: Some(false)`. The settled switch is `Some(false)`. The load report is dense-only. The provider has an MTP head and tool calling. |
| 2 | Streaming `/v1/chat/completions` request with no `mtp` field | More than one SSE delta and `[DONE]`. The answer names Paris. `chatworks_decode`: `proposer = mtp`, `draft_tokens = 3`, `kv_cache = static`, CUDA graphs disabled (`path = none`). `chatworks_mtp.accepted_tokens > 0`. |
| 3 | Tool call, non-streamed and streamed | `finish_reason = tool_calls`. One `get_weather` call whose arguments are JSON matching the schema. Decoded on MTP. |
| 4 | Three cancels: (a) the HTTP client drops the SSE stream; (b) a `CancelFlag` tripped from the token callback (`generate_with_cancel`); (c) `EngineHandle::cancel`, the path of the Stop button's `stop_generation` | Each stops well short of `max_tokens` on the MTP path with `finish_reason = cancelled`. Nothing is left in flight. A greedy request afterwards is token-identical to the same request on the fresh load. |
| 5 | **Reload** in place (the served source loaded again), then **Unload**, then serve again | The reload releases the resident copy first (`load_transition.reason = reload`) and keeps one copy on the device. After the unload, device memory is within 1 GiB of the pre-load level. After each load, the greedy request is again token-identical. |
| 6 | Optional (`CHATWORKS_QWEN3_8B_SNAPSHOT`): the step-2 request on Qwen3-8B, which has no MTP head | `proposer = none`, `kv_cache = static`, `sampler = device` under the shipped temperature 0.7 and top-p 0.9. Runtime cross-check (CUDA build): the same request through the pinned runtime's llama provider reports `logits_to_host == 0`. |
| 7 | Seal the record | Written once to `CHATWORKS_E2E_OUTPUT` and never overwritten. |

The record at `CHATWORKS_E2E_OUTPUT` has this shape:

```json
{
  "seal": { "sha256": "<sha256 of the compact JSON of `record`>", "over": "..." },
  "record": {
    "outcome": "complete | aborted | incomplete",
    "meta": {
      "hardware_label": "RTX Pro 6000 / sm_120",
      "chatworks": { "commit": "...", "branch": "...", "dirty": false },
      "inference_pin": { "package": "runtime-cuda", "source": "git+...", "commit": "..." },
      "pinned_gpu": { "name": "...", "pci.bus_id": "...", "uuid": "...", "driver_version": "..." },
      "shipped_settings": { "sampling": {}, "runtime": {} },
      "backend_capabilities": {}
    },
    "summary": { "checks": 0, "passed": 0, "failed": 0, "failed_checks": [] },
    "checks": [ { "step": "2", "check": "...", "pass": true, "observed": {} } ],
    "observations": { "step_2": { "stream": {} } }
  }
}
```

The file is created with `create_new` and then made read-only. To verify it, hash the compact
serialization of `record` with its keys sorted, as `serde_json` writes it, and compare the result
with `seal.sha256`. For the record run, the ChatWorks tree must be clean (`meta.chatworks.dirty`
must be `false`).

### Expected results at earlier pins

- Pins before SceneWorks/inference#1036 (for example `a400b2e83`) fail step 6's runtime
  cross-check. At those pins the engine's no-draft step drew on the host under temperature + top-p,
  so `logits_to_host > 0`. The product label still read `device` because that draw was not
  recorded. #1036 fixed both.
- Step 2 reports `sampler = host:speculative_distribution` on Qwen3.8's MTP path under the shipped
  temperature. The stochastic acceptance test reads distributions on the host. This is recorded,
  but it is not a check.

## Manual checklist: the Tauri app on the final pin

Use a fresh profile (`CHATWORKS_PROFILE_DIR`, see the README) and the packaged CUDA build. Tick
each item once.

**Settings**
- [ ] Speculative decoding (Multi-token prediction) shows **Automatic**. CUDA graphs
  (experimental) is off, and the speculative notice does not appear.

**Models → serve Qwen3.8-27B (Dense)**
- [ ] The **Served model decode path** panel shows:
  - Backend: `candle-cuda · cuda:0 · sm_120`
  - Weights: `checkpoint encoding`, with `Resident projections: dense × 505`
  - CUDA-graph switch: `off`
- [ ] Before any chat, the Last-generation rows read "not measured yet".

**Chat**
- [ ] A prompt streams token by token.
- [ ] After the reply, the Last-generation rows show:
  - Decode implementation: `mtp`
  - Proposer: `mtp · 3 drafts`, with an "Accepted N of M drafts" detail
  - CUDA graphs: `off`
  - NVFP4 projections: `none (no NVFP4 weights)`
  - KV cache: `static · gqa attention`
- [ ] "What time is it? Use a tool." opens the approval panel for `get_current_time`. Approving it
  returns an answer that uses the result.
- [ ] **Stop** during a long answer ends the stream at once. The partial turn is kept, and a
  follow-up prompt answers normally.

**Reload and unload**
- [ ] Turn CUDA graphs on in Settings. The served model offers **Reload**. Reload succeeds, and the
  notice says the served copy was unloaded first. Turn graphs off and Reload again.
- [ ] **Unload model** returns device memory (Task Manager or `nvidia-smi`) to the pre-load level.
  Serve the model again and one prompt answers.

**Optional: NVFP4**
- [ ] Only if the NVFP4 variant is registered: its row reads `… · NVFP4 load (lossy)`. When
  served, the chat header carries the `NVFP4 · lossy` badge, and the Weights row reads
  `NVFP4 (lossy)`.
