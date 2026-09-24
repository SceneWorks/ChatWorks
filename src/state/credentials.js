export async function loadAppCredentialState(invoke) {
  const settings = await invoke("load_app_settings");
  if (!settings.server.authEnabled) return { settings, token: null, error: null };
  try {
    const token = await invoke("api_auth_token");
    return { settings, token, error: token ? null : "API auth token is unavailable" };
  } catch (cause) {
    return { settings, token: null, error: String(cause) };
  }
}

/// The settings the UI starts from: the saved ones, or, when they cannot be read, this build's
/// defaults as the backend reports them (`default_app_settings`). Never UI constants: the
/// speculative-decoding default differs by build (sc-24140 feature-end review).
export async function loadAppSettingsOrDefaults(invoke) {
  try {
    return await loadAppCredentialState(invoke);
  } catch (cause) {
    const error = `Could not load settings: ${String(cause)}`;
    try {
      return { settings: await invoke("default_app_settings"), token: null, error };
    } catch {
      return { settings: null, token: null, error };
    }
  }
}

export async function saveAppCredentialState(invoke, settings, token) {
  const [savedSettings, status, savedToken] = await invoke("save_app_settings", {
    settings,
    apiAuthToken: token,
  });
  return { settings: savedSettings, status, token: savedToken };
}

export function checkHfCredentialStatus(invoke) {
  return invoke("hf_token_status");
}
