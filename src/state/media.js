export async function prepareRemoteMedia(invoke, source, kind, signal) {
  const id = await invoke("begin_media_preparation");
  const cancel = () => invoke("cancel_media_preparation", { id }).catch(() => {});
  signal?.addEventListener("abort", cancel, { once: true });
  try {
    if (signal?.aborted) {
      await cancel();
      throw new DOMException("Media preparation cancelled", "AbortError");
    }
    return await invoke("prepare_remote_media", { id, source, kind });
  } finally {
    signal?.removeEventListener("abort", cancel);
  }
}
