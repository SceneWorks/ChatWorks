use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager};

use crate::fsutil::write_json_atomic;
use crate::server::{DEFAULT_OPENAI_HOST, DEFAULT_OPENAI_PORT};

const API_AUTH_KEYCHAIN_SERVICE: &str = "net.trefry.chatworks.openai";
const API_AUTH_KEYCHAIN_USER: &str = "api-auth-token";

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AppSettings {
    #[serde(default)]
    pub server: ServerSettings,
    #[serde(default)]
    pub sampling: SamplingDefaults,
}

impl AppSettings {
    pub fn normalized(mut self) -> Result<Self, String> {
        self.server.host = self.server.host.trim().to_string();
        if self.server.host.is_empty() {
            return Err("bind host is required".to_string());
        }
        if self.server.port == 0 {
            return Err("port must be between 1 and 65535".to_string());
        }
        self.sampling.system_prompt = self.sampling.system_prompt.trim().to_string();
        if !(0.0..=2.0).contains(&self.sampling.temperature) {
            return Err("temperature must be between 0 and 2".to_string());
        }
        if !(0.0..=1.0).contains(&self.sampling.top_p) {
            return Err("top_p must be between 0 and 1".to_string());
        }
        if self.sampling.max_tokens == 0 {
            return Err("max tokens must be at least 1".to_string());
        }
        Ok(self)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ServerSettings {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default)]
    pub allow_lan: bool,
    #[serde(default)]
    pub auth_enabled: bool,
}

impl Default for ServerSettings {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            allow_lan: false,
            auth_enabled: false,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SamplingDefaults {
    #[serde(default = "default_system_prompt")]
    pub system_prompt: String,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_disable_thinking")]
    pub disable_thinking: bool,
}

impl Default for SamplingDefaults {
    fn default() -> Self {
        Self {
            system_prompt: default_system_prompt(),
            temperature: default_temperature(),
            top_p: default_top_p(),
            max_tokens: default_max_tokens(),
            disable_thinking: default_disable_thinking(),
        }
    }
}

pub fn load_app_settings(app: &AppHandle) -> Result<AppSettings, String> {
    let path = settings_path(app)?;
    if !path.exists() {
        return Ok(AppSettings::default());
    }
    let body = fs::read_to_string(path).map_err(|error| error.to_string())?;
    serde_json::from_str::<AppSettings>(&body)
        .map_err(|error| error.to_string())?
        .normalized()
}

pub fn save_app_settings(app: &AppHandle, settings: &AppSettings) -> Result<(), String> {
    let path = settings_path(app)?;
    write_settings(&path, settings)
}

pub fn api_auth_token_present() -> bool {
    read_api_auth_token().ok().flatten().is_some()
}

pub fn read_api_auth_token() -> Result<Option<String>, keyring::Error> {
    let entry = keyring::Entry::new(API_AUTH_KEYCHAIN_SERVICE, API_AUTH_KEYCHAIN_USER)?;
    match entry.get_password() {
        Ok(token) if token.trim().is_empty() => Ok(None),
        Ok(token) => Ok(Some(token)),
        Err(keyring::Error::NoEntry) => Ok(None),
        Err(error) => Err(error),
    }
}

pub fn save_api_auth_token(token: &str) -> Result<(), String> {
    let token = token.trim();
    if token.is_empty() {
        return Err("API auth token is required".to_string());
    }
    let entry = keyring::Entry::new(API_AUTH_KEYCHAIN_SERVICE, API_AUTH_KEYCHAIN_USER)
        .map_err(|error| error.to_string())?;
    entry.set_password(token).map_err(|error| error.to_string())
}

pub fn clear_api_auth_token() -> Result<(), String> {
    let entry = keyring::Entry::new(API_AUTH_KEYCHAIN_SERVICE, API_AUTH_KEYCHAIN_USER)
        .map_err(|error| error.to_string())?;
    match entry.delete_credential() {
        Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
        Err(error) => Err(error.to_string()),
    }
}

fn settings_path(app: &AppHandle) -> Result<PathBuf, String> {
    app.path()
        .app_data_dir()
        .map(|path| path.join("settings.json"))
        .map_err(|error| error.to_string())
}

fn write_settings(path: &std::path::Path, settings: &AppSettings) -> Result<(), String> {
    // Delegates to the shared atomic-write helper (code-review F-010). For `settings.json` the
    // previous local temp name (`with_extension("json.tmp")` → `settings.json.tmp`) matches the
    // shared helper's appended `.tmp` (`settings.json.tmp`), so the on-disk temp name is unchanged.
    write_json_atomic(path, settings)
}

fn default_host() -> String {
    DEFAULT_OPENAI_HOST.to_string()
}

fn default_port() -> u16 {
    DEFAULT_OPENAI_PORT
}

fn default_system_prompt() -> String {
    "You are a helpful local assistant.".to_string()
}

fn default_temperature() -> f32 {
    0.7
}

fn default_top_p() -> f32 {
    0.9
}

fn default_max_tokens() -> u32 {
    512
}

fn default_disable_thinking() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_and_validates_settings() {
        let settings = AppSettings {
            server: ServerSettings {
                host: " 127.0.0.1 ".to_string(),
                ..Default::default()
            },
            sampling: SamplingDefaults {
                system_prompt: " hello ".to_string(),
                ..Default::default()
            },
        }
        .normalized()
        .unwrap();

        assert_eq!(settings.server.host, "127.0.0.1");
        assert_eq!(settings.sampling.system_prompt, "hello");
    }

    #[test]
    fn rejects_invalid_sampling_defaults() {
        assert!(AppSettings {
            sampling: SamplingDefaults {
                temperature: 3.0,
                ..Default::default()
            },
            ..Default::default()
        }
        .normalized()
        .is_err());
    }
}
