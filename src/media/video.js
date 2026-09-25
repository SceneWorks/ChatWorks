import { canvasToBlob, readBlobAsDataUrl } from "./image.js";

// Video frame sampling (sc-8081): the frontend samples a small number of evenly-spaced frames from
// an attached video client-side (no native decoder needed) and sends them as a `video_url` part with
// per-frame timestamps. Frames are downscaled like images. Keeping the count small bounds prompt size
// and keeps inference responsive.
export const VIDEO_ATTACHMENT_MAX_FRAMES = 8;
export const VIDEO_FRAME_MAX_DIMENSION = 768;
export const VIDEO_FRAME_QUALITY = 0.7;

// `seeked` marks the media clock change, not presentation of the decoded frame. In particular,
// WebKit can still paint black for the first seek until the frame reaches the compositor.
function waitForPresentedFrame(video, previousTime, binEnd, signal) {
  signal?.throwIfAborted();
  let frameId;
  let timeout;
  let resume;
  let duplicateTimeout;
  let onAbort;
  const ready = new Promise((resolve, reject) => {
    onAbort = () => reject(new DOMException("Media preparation cancelled", "AbortError"));
    signal?.addEventListener("abort", onAbort, { once: true });
    const wake = () => {
      if (!video.paused) return;
      try { Promise.resolve(video.play()).catch(reject); } catch (error) { reject(error); }
    };
    const request = () => {
      frameId = video.requestVideoFrameCallback((_, metadata) => {
        frameId = undefined;
        const time = metadata?.mediaTime;
        if (!Number.isFinite(time) || time < 0) {
          reject(new Error("Video decoder presented a frame outside the sampled time range."));
        } else if (time > binEnd + 0.001) {
          // Playback after a duplicate may reach the next real frame, outside this bin.
          // Leave that frame for a later seek; do not fail a low-frame-rate attachment.
          if (duplicateTimeout) resolve(null);
          else reject(new Error("Video decoder presented a frame outside the sampled time range."));
        } else if (previousTime !== null && time < previousTime - 0.001) {
          // A compositor callback from the previous seek is not the frame sought now.
          request();
        } else if (previousTime !== null && time <= previousTime + 0.001) {
          // Low-frame-rate clips can legitimately present the same PTS at several midpoints.
          // Give the decoder a short chance to advance, then omit the duplicate frame.
          if (!duplicateTimeout) {
            wake();
            duplicateTimeout = setTimeout(() => resolve(null), 500);
          }
          request();
        } else {
          resolve(time);
        }
      });
    };
    request();
    // A paused WebView can wait indefinitely for a new compositor frame after a seek.
    resume = setTimeout(wake, 250);
    timeout = setTimeout(() => reject(new Error("Video frame was not ready after seeking.")), 5000);
  });
  return ready.finally(() => {
    clearTimeout(timeout);
    clearTimeout(resume);
    clearTimeout(duplicateTimeout);
    video.pause();
    if (frameId !== undefined) video.cancelVideoFrameCallback(frameId);
    signal?.removeEventListener("abort", onAbort);
  });
}

/// Sample up to `VIDEO_ATTACHMENT_MAX_FRAMES` evenly-spaced frames from a video file or HTTPS URL,
/// using a hidden `<video>` element + canvas (no native decoder). Returns `{ frames, timestamps, fps
/// }` where `frames` are downscaled JPEG data URLs in temporal order and `timestamps` are the
/// wall-clock seconds of each sampled frame — exactly the `video_url` shape the local server expects
/// (sc-8081). `fps` is the *sampled* rate (frames per second over the captured span), forwarded so
/// the server can derive timestamps if needed.
export async function sampleVideoAttachment(source, signal) {
  const remote = typeof source === "string";
  const url = remote ? source : URL.createObjectURL(source);
  const name = remote ? source : source.name || "video attachment";
  const video = document.createElement("video");
  video.preload = "auto";
  video.muted = true;
  video.playsInline = true;
  if (remote) video.crossOrigin = "anonymous";
  video.src = url;

  let rejectPending;
  const onAbort = () => {
    video.pause();
    video.removeAttribute("src");
    video.load();
    rejectPending?.(new DOMException("Media preparation cancelled", "AbortError"));
  };
  signal?.addEventListener("abort", onAbort, { once: true });
  const ready = new Promise((resolve, reject) => {
    rejectPending = reject;
    video.onloadedmetadata = () => resolve();
    video.onerror = () => reject(new Error(remote
      ? `Could not decode ${name}. Remote video URLs must allow CORS for frame sampling.`
      : `Could not load or decode local video ${name}. Check that its format is supported by this WebView.`));
  });

  try {
    signal?.throwIfAborted();
    await ready;
    if (typeof video.requestVideoFrameCallback !== "function") {
      throw new Error("Video frame capture is unavailable in this WebView (requestVideoFrameCallback is required).");
    }
    const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 0;
    const count = Math.max(1, Math.min(VIDEO_ATTACHMENT_MAX_FRAMES, duration > 0 ? VIDEO_ATTACHMENT_MAX_FRAMES : 1));
    // Even sampling across the duration (midpoints of `count` equal segments) so frames span the clip.
    const times = duration > 0
      ? Array.from({ length: count }, (_, i) => ((i + 0.5) / count) * duration)
      : [0];

    const vw = video.videoWidth || 1;
    const vh = video.videoHeight || 1;
    const scale = Math.min(1, VIDEO_FRAME_MAX_DIMENSION / Math.max(vw, vh));
    const width = Math.max(1, Math.round(vw * scale));
    const height = Math.max(1, Math.round(vh * scale));
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("Could not prepare video frame canvas.");

    const seekTo = (t) =>
      new Promise((resolve, reject) => {
        signal?.throwIfAborted();
        rejectPending = reject;
        const onSeeked = () => {
          video.removeEventListener("seeked", onSeeked);
          resolve();
        };
        video.addEventListener("seeked", onSeeked);
        video.onerror = () => reject(new Error("Could not seek video for frame sampling."));
        // Clamp to just inside the duration to avoid a seek past the end never firing `seeked`.
        video.currentTime = Math.min(t, Math.max(0, (duration || t) - 0.01));
      });

    const frames = [];
    const timestamps = [];
    for (const [index, t] of times.entries()) {
      signal?.throwIfAborted();
      await seekTo(t);
      const mediaTime = await waitForPresentedFrame(
        video, timestamps.length ? timestamps[timestamps.length - 1] : null,
        duration > 0 ? duration * (index + 1) / count : 0,
        signal,
      );
      if (mediaTime === null) continue;
      signal?.throwIfAborted();
      context.drawImage(video, 0, 0, width, height);
      const blob = await canvasToBlob(canvas, "image/jpeg", VIDEO_FRAME_QUALITY);
      frames.push(await readBlobAsDataUrl(blob));
      timestamps.push(mediaTime);
    }
    const span = timestamps.length > 1 ? timestamps[timestamps.length - 1] - timestamps[0] : 0;
    const fps = span > 0 ? (timestamps.length - 1) / span : 1;
    return { frames, timestamps, fps };
  } finally {
    signal?.removeEventListener("abort", onAbort);
    video.pause();
    video.removeAttribute("src");
    video.load();
    if (!remote) URL.revokeObjectURL(url);
  }
}
