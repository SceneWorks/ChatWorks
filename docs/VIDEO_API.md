# Video input over the OpenAI-compatible API (sc-8081)

ChatWorks' local OpenAI-compatible server accepts **video input** for video-capable models
(Qwen3-VL). The OpenAI Chat Completions API has no standard content-part for video (only
`image_url`), so ChatWorks defines a concrete, justified shape.

## The decision: a `video_url` content part carrying pre-sampled frames

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
- `timestamps` (optional): per-frame wall-clock seconds, one per frame. Drives **Text–Timestamp
  Alignment** — the model is told `<{t:.1f} seconds>` before each frame, which is what lets it answer
  temporal questions ("what is shown first / at the end / when does X happen").
- `fps` (optional): sampling rate. When `timestamps` is omitted, timestamps are derived as `i / fps`;
  lacking both, they default to the frame index in seconds (1 fps).

Validation: at least one frame is required; if `timestamps` is present it must have exactly one entry
per frame (otherwise the request is a 400).

## Why this shape

1. **No heavy server-side video decoder for v1.** Decoding arbitrary `.mp4`/`.webm`/`.mov` containers
   server-side would pull in a large native dependency (FFmpeg/libav). By accepting *pre-sampled
   frames*, ChatWorks needs no such dependency — the host that already has a decoder (a browser via
   `<video>`+canvas, or any client) samples frames and sends them. The ChatWorks frontend does exactly
   this client-side.
2. **It mirrors the existing `image_url` plumbing.** Each frame is decoded by the same `decode_image`
   path; the part lives next to `image_url` in the same `content` array, preserving the
   visuals-before-text ordering vision providers expect.
3. **It carries timestamps explicitly**, which is the data Qwen3-VL's Text–Timestamp Alignment needs.
   The provider folds `temporal_patch_size` frames per emitted vision frame and renders
   `<{t} seconds>` tags from these timestamps — the same values `Qwen3VLProcessor.replace_video_token`
   computes.
4. **It degrades gracefully.** Timestamps can be omitted (derived from `fps` or frame index), so a
   minimal caller can send just `frames`.

## The ChatWorks frontend

The frontend's "Video" attach button samples up to 8 evenly-spaced frames from the chosen video file
client-side (`<video>` element + canvas, no native decoder), downscales them, and sends them as a
`video_url` part with derived timestamps. The button is shown only when the loaded model advertises
`supports_video`.

## Follow-ups (deferred, tracked)

- **Server-side arbitrary-video-file decode.** Accepting a single `video_url.url` pointing at a real
  video file (and sampling frames in the backend) requires a heavy decode dependency (FFmpeg/libav)
  and a frame-sampling policy. Deferred; the frames-based representation above covers v1 without it.
  When added, it would be an additive variant of the same `video_url` part (a `url` alongside
  `frames`), so this shape does not need to change.
