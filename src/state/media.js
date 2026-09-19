export function prepareRemoteMedia(invoke, source, kind) {
  return invoke("prepare_remote_media", { source, kind });
}
