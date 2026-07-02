import { canvasToBlob, readBlobAsDataUrl } from "./image";

// Video frame sampling (sc-8081): the frontend samples a small number of evenly-spaced frames from
// an attached video client-side (no native decoder needed) and sends them as a `video_url` part with
// per-frame timestamps. Frames are downscaled like images. Keeping the count small bounds prompt size
// and keeps inference responsive.
export const VIDEO_ATTACHMENT_MAX_FRAMES = 8;
export const VIDEO_FRAME_MAX_DIMENSION = 768;
export const VIDEO_FRAME_QUALITY = 0.7;

/// Sample up to `VIDEO_ATTACHMENT_MAX_FRAMES` evenly-spaced frames from a video file, client-side,
/// using a hidden `<video>` element + canvas (no native decoder). Returns `{ frames, timestamps, fps
/// }` where `frames` are downscaled JPEG data URLs in temporal order and `timestamps` are the
/// wall-clock seconds of each sampled frame — exactly the `video_url` shape the local server expects
/// (sc-8081). `fps` is the *sampled* rate (frames per second over the captured span), forwarded so
/// the server can derive timestamps if needed.
export async function sampleVideoAttachment(file) {
  const url = URL.createObjectURL(file);
  const video = document.createElement("video");
  video.preload = "auto";
  video.muted = true;
  video.playsInline = true;
  video.src = url;

  const ready = new Promise((resolve, reject) => {
    video.onloadedmetadata = () => resolve();
    video.onerror = () => reject(new Error(`Could not decode ${file.name || "video attachment"}.`));
  });

  try {
    await ready;
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
    for (const t of times) {
      await seekTo(t);
      context.drawImage(video, 0, 0, width, height);
      const blob = await canvasToBlob(canvas, "image/jpeg", VIDEO_FRAME_QUALITY);
      frames.push(await readBlobAsDataUrl(blob));
      timestamps.push(Number(video.currentTime.toFixed(3)));
    }
    const span = timestamps.length > 1 ? timestamps[timestamps.length - 1] - timestamps[0] : 0;
    const fps = span > 0 ? (timestamps.length - 1) / span : 1;
    return { frames, timestamps, fps };
  } finally {
    URL.revokeObjectURL(url);
  }
}
