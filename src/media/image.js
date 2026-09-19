export const IMAGE_ATTACHMENT_MAX_DIMENSION = 1536;
export const IMAGE_ATTACHMENT_MAX_BYTES = 8 * 1024 * 1024;
export const IMAGE_ATTACHMENT_QUALITY_STEPS = [0.86, 0.76, 0.66, 0.56];

export function parseNumber(value) {
  if (value === "") return undefined;
  const number = Number(value);
  return Number.isFinite(number) ? number : undefined;
}

export function canvasToBlob(canvas, type, quality) {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (blob) {
          resolve(blob);
        } else {
          reject(new Error("Could not encode image attachment."));
        }
      },
      type,
      quality,
    );
  });
}

export function readBlobAsDataUrl(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("Could not read image attachment."));
    reader.readAsDataURL(blob);
  });
}

export async function loadDrawableImage(source, signal) {
  signal?.throwIfAborted();
  if (!signal && typeof source !== "string" && typeof createImageBitmap === "function") {
    try {
      const bitmap = await createImageBitmap(source, { imageOrientation: "from-image" });
      return {
        source: bitmap,
        width: bitmap.width,
        height: bitmap.height,
        close: () => bitmap.close?.(),
      };
    } catch {
      // Fall through to the HTMLImageElement decoder for formats createImageBitmap cannot open.
    }
  }

  return new Promise((resolve, reject) => {
    const remote = typeof source === "string";
    const url = remote ? source : URL.createObjectURL(source);
    const name = remote ? source : source.name || "image attachment";
    const image = new Image();
    // Canvas must remain origin-clean because the normalized attachment is sent as a data URL.
    // A remote server without CORS permission fails here rather than silently sending unusable data.
    if (remote) image.crossOrigin = "anonymous";
    const abort = () => {
      image.src = "";
      if (!remote) URL.revokeObjectURL(url);
      reject(new DOMException("Media preparation cancelled", "AbortError"));
    };
    signal?.addEventListener("abort", abort, { once: true });
    image.onload = () => {
      signal?.removeEventListener("abort", abort);
      resolve({
        source: image,
        width: image.naturalWidth,
        height: image.naturalHeight,
        close: () => {
          if (!remote) URL.revokeObjectURL(url);
        },
      });
    };
    image.onerror = () => {
      signal?.removeEventListener("abort", abort);
      if (!remote) URL.revokeObjectURL(url);
      reject(new Error(`Could not decode ${name}. Remote image URLs must allow CORS for attachment processing.`));
    };
    image.src = url;
  });
}

export async function normalizeImageAttachment(source, signal) {
  const drawable = await loadDrawableImage(source, signal);
  try {
    const scale = Math.min(1, IMAGE_ATTACHMENT_MAX_DIMENSION / Math.max(drawable.width, drawable.height));
    const width = Math.max(1, Math.round(drawable.width * scale));
    const height = Math.max(1, Math.round(drawable.height * scale));
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("Could not prepare image attachment.");
    context.fillStyle = "#fff";
    context.fillRect(0, 0, width, height);
    context.drawImage(drawable.source, 0, 0, width, height);

    for (const quality of IMAGE_ATTACHMENT_QUALITY_STEPS) {
      signal?.throwIfAborted();
      const blob = await canvasToBlob(canvas, "image/jpeg", quality);
      if (blob.size <= IMAGE_ATTACHMENT_MAX_BYTES) return readBlobAsDataUrl(blob);
    }
  } finally {
    drawable.close?.();
  }

  const name = typeof source === "string" ? source : source.name || "Image attachment";
  throw new Error(`${name} is too large after compression.`);
}
