# Video input over the OpenAI-compatible API (sc-8081)

ChatWorks' local OpenAI-compatible server accepts **video input** for video-capable models
(Qwen3-VL). The OpenAI Chat Completions API has no standard content-part for video (only
`image_url`), so ChatWorks defines a concrete, justified shape.

## The `video_url` content part

A video is sent as a content part of type `video_url` whose object carries an ordered list of
**already-sampled frames** (image data URLs) plus optional per-frame **timestamps**:

```jsonc
{
  "role": "user",
  "content": [
    {
      "type": "video_url",
      "video_url": {
        "frames": [
          "data:image/jpeg;base64,…",   // frame 0
          "data:image/jpeg;base64,…",   // frame 1
          "data:image/jpeg;base64,…"    // …in temporal order
        ],
        "timestamps": [0.0, 0.5, 1.0],  // optional; seconds, one per frame
        "fps": 2.0                      // optional; used to derive timestamps when absent
      }
    },
    { "type": "text", "text": "What happens over the course of this video?" }
  ]
}
```

- `frames` (**required**): the sampled frames in temporal order, each an image data URL
  (`data:image/…;base64,…`) or bare base64. Decoded to RGB8 exactly like an `image_url` part.
- `timestamps` (optional): finite, non-negative, monotonically nondecreasing wall-clock seconds,
  one per frame. Drives **Text–Timestamp
  Alignment** — the model is told `<{t:.1f} seconds>` before each frame, which is what lets it answer
  temporal questions ("what is shown first / at the end / when does X happen").
- `fps` (optional): finite, positive sampling rate. When `timestamps` is omitted, timestamps are derived as `i / fps`;
  lacking both, they default to the frame index in seconds (1 fps).

Validation: at least one frame is required; invalid `fps`, timestamp count, value, or temporal order
returns a 400 before generation.

### File and URL sources

Clients can instead pass one local file, `file://` URI, or public HTTP(S) URL. ChatWorks stages the source,
samples eight timestamped JPEG frames, and sends them through the same temporal path:

```json
{
  "type": "video_url",
  "video_url": { "url": "file:///Users/me/Movies/example.mp4" }
}
```

HTTP API access to local paths is disabled by default. It can be enabled in Settings only while
bearer authentication is enabled; every such request must authenticate. Desktop file-picker
attachments use trusted Tauri IPC and remain available independently of this network policy.

Remote URLs must resolve to a public address; loopback, private, link-local, and reserved destinations are
rejected. Redirects are not followed, downloads are capped at 256 MiB, clips at ten minutes, and
each sampled frame is constrained to a 768-pixel longest axis. Release bundles include pinned FFmpeg/ffprobe sidecars built by
[`scripts/provision-ffmpeg-sidecars.sh`](../scripts/provision-ffmpeg-sidecars.sh); they never
download at runtime. The build recipe verifies the upstream source checksum and ships the LGPL
notice in [`third_party/ffmpeg`](../third_party/ffmpeg). Development builds may set
`CHATWORKS_FFMPEG` and `CHATWORKS_FFPROBE` to local binaries.

## Why this shape

1. **It mirrors the existing `image_url` plumbing.** Each frame is decoded by the same `decode_image`
   path; the part lives next to `image_url` in the same `content` array, preserving the
   visuals-before-text ordering vision providers expect.
2. **It carries timestamps explicitly**, which is the data Qwen3-VL's Text–Timestamp Alignment needs.
   The provider folds `temporal_patch_size` frames per emitted vision frame and renders
   `<{t} seconds>` tags from these timestamps — the same values `Qwen3VLProcessor.replace_video_token`
   computes.
3. **It degrades gracefully.** Timestamps can be omitted (derived from `fps` or frame index), so a
   minimal caller can send just `frames`.

## The ChatWorks frontend

The frontend's "Video" attach button samples up to 8 evenly-spaced frames from a chosen local video
client-side, downscales them, and sends them as a `video_url` part with derived timestamps. Public
image/video URLs instead pass through native bounded staging and decoding, so the remote server does
not need WebView CORS headers and ChatWorks does not widen its CSP. The button is shown only when the
loaded model advertises `supports_video`.
