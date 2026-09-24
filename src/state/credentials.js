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
