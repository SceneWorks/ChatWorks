use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use tauri::AppHandle;

use crate::fsutil::write_json_atomic;
use crate::server::{DEFAULT_OPENAI_HOST, DEFAULT_OPENAI_PORT};

const API_AUTH_KEYCHAIN_SERVICE: &str = "net.trefry.chatworks.openai";
const API_AUTH_KEYCHAIN_USER: &str = "api-auth-token";

/// The settings schema this build writes (sc-24139). Every save stamps it; a file without one was
/// written before this marker existed.
///
/// **Migration rule for the speculative-decoding default.** Before sc-24139 the MTP default was
/// `off` everywhere and every save serialized `mtpMode`, so an `"mtpMode": "off"` in a pre-marker
/// file cannot be told apart from "the user chose off". Pre-marker (version 0) files therefore
/// keep what they have: a present `mtpMode` (including `off`) is honored as written, and only a
/// **missing** `mtpMode` takes the new platform default ([`default_mtp_mode`]: `auto` on the Candle
/// CUDA build, `off` on MLX and Candle CPU). Files at version 1 or later were written under the new
/// default, so a saved `off` there is an explicit choice by construction.
///
/// Because every pre-marker save wrote `mtpMode`, most upgrading users keep `off` under that rule.
/// Where the build's default is not `off` (Candle CUDA), such a carried-over `off` is flagged
/// ([`NoticeSettings::speculative_off_carried_over`]) so the UI can offer — once, dismissibly — to
/// turn on Auto, without ever changing the saved choice itself.
pub const CURRENT_SETTINGS_VERSION: u32 = 1;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AppSettings {
    /// Schema marker ([`CURRENT_SETTINGS_VERSION`]); `0` for a file written before it existed.
    #[serde(default)]
    pub settings_version: u32,
    #[serde(default)]
    pub server: ServerSettings,
    #[serde(default)]
    pub sampling: SamplingDefaults,
    #[serde(default)]
    pub runtime: RuntimeSettings,
    #[serde(default)]
    pub notices: NoticeSettings,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            settings_version: CURRENT_SETTINGS_VERSION,
            server: ServerSettings::default(),
            sampling: SamplingDefaults::default(),
            runtime: RuntimeSettings::default(),
            notices: NoticeSettings::default(),
        }
    }
}

/// One-time UI notices and their persisted dismissals (sc-24139).
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct NoticeSettings {
    /// Speculative decoding is `off` only because a pre-marker settings file carried `off` over on
    /// a build whose default is `auto` (see [`CURRENT_SETTINGS_VERSION`]). Set by the migration;
    /// cleared as soon as the mode is anything but `off`, so a later explicit `off` never
    /// re-raises the notice.
    #[serde(default)]
    pub speculative_off_carried_over: bool,
    /// The user dismissed the "speculative decoding is available" notice.
    #[serde(default)]
    pub speculative_notice_dismissed: bool,
}

/// Load-time runtime options (sc-24139). These change how a model is **loaded**, so a change
/// applies the next time a model is loaded, not to the one already served.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeSettings {
    /// Capture decode steps as CUDA graphs (Candle CUDA only). Off by default: no measured win
    /// yet, and Qwen3.5/3.8 steps currently fall back eagerly with a named reason. Sent as the
    /// runtime's `LoadSpec::cuda_graphs` only where the runtime reports the switch as supported.
    #[serde(default)]
    pub cuda_graphs: bool,
}

impl AppSettings {
    /// Parse a stored settings file, applying the schema migration (see
    /// [`CURRENT_SETTINGS_VERSION`]) and validation. The result carries the current marker, so the
    /// next save records it.
    pub fn from_stored_json(body: &str) -> Result<Self, String> {
        Self::from_stored_json_for(body, crate::inference_runtime::execution_backend())
    }

    /// [`from_stored_json`](Self::from_stored_json) for an execution backend label, whose
    /// speculative default ([`default_mtp_mode_for`]) decides whether a pre-marker `off` was
    /// carried over rather than chosen under the current default.
    pub fn from_stored_json_for(body: &str, execution_backend: &str) -> Result<Self, String> {
        let mut settings =
            serde_json::from_str::<AppSettings>(body).map_err(|error| error.to_string())?;
        if settings.settings_version == 0
            && settings.sampling.mtp_mode == "off"
            && default_mtp_mode_for(execution_backend) != "off"
        {
            settings.notices.speculative_off_carried_over = true;
        }
        settings.normalized()
    }

    pub fn normalized(mut self) -> Result<Self, String> {
        // A pre-marker file's values are kept as written (validated below); stamping the marker
        // records on the next save that this file now carries explicit choices.
        self.settings_version = CURRENT_SETTINGS_VERSION;
        if self.sampling.mtp_mode != "off" {
            self.notices.speculative_off_carried_over = false;
        }
        self.server.host = self.server.host.trim().to_string();
        if self.server.host.is_empty() {
            return Err("bind host is required".to_string());
        }
        if self.server.port == 0 {
            return Err("port must be between 1 and 65535".to_string());
        }
        if self.server.allow_local_files && !self.server.auth_enabled {
            return Err(
                "local media access for API clients requires bearer authentication".to_string(),
            );
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
        if !matches!(self.sampling.mtp_mode.as_str(), "off" | "auto" | "enabled") {
            return Err("mtp mode must be off, auto, or enabled".to_string());
        }
        if !matches!(
            self.sampling.reasoning_effort.as_deref(),
            None | Some("low" | "medium" | "xhigh")
        ) {
            return Err("reasoning effort must be low, medium, or xhigh".to_string());
        }
        if self.sampling.mtp_draft_tokens == 0 {
            return Err("MTP draft tokens must be at least 1".to_string());
        }
        if let Some(penalty) = self.sampling.repetition_penalty {
            if penalty <= 0.0 || !penalty.is_finite() {
                return Err("repetition penalty must be finite and greater than 0".to_string());
            }
        }
        if let Some(penalty) = self.sampling.presence_penalty {
            if !penalty.is_finite() {
                return Err("presence penalty must be finite".to_string());
            }
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
    /// Permit authenticated OpenAI API callers to reference local media paths. Desktop file
    /// attachments use trusted Tauri IPC and do not depend on this network-facing policy.
    #[serde(default)]
    pub allow_local_files: bool,
}

impl Default for ServerSettings {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            allow_lan: false,
            auth_enabled: false,
            allow_local_files: false,
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
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    pub preserve_thinking: Option<bool>,
    #[serde(default = "default_mtp_mode")]
    pub mtp_mode: String,
    #[serde(default = "default_mtp_draft_tokens")]
    pub mtp_draft_tokens: u32,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub repetition_context: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
}

impl Default for SamplingDefaults {
    fn default() -> Self {
        Self {
            system_prompt: default_system_prompt(),
            temperature: default_temperature(),
            top_p: default_top_p(),
            max_tokens: default_max_tokens(),
            disable_thinking: default_disable_thinking(),
            reasoning_effort: None,
            preserve_thinking: None,
            mtp_mode: default_mtp_mode(),
            mtp_draft_tokens: default_mtp_draft_tokens(),
            top_k: None,
            presence_penalty: None,
            repetition_penalty: None,
            repetition_context: None,
            seed: None,
        }
    }
}

pub fn load_app_settings(app: &AppHandle) -> Result<AppSettings, String> {
    let path = settings_path(app)?;
    if !path.exists() {
        return Ok(AppSettings::default());
    }
    let body = fs::read_to_string(path).map_err(|error| error.to_string())?;
    AppSettings::from_stored_json(&body)
}

pub fn save_app_settings(app: &AppHandle, settings: &AppSettings) -> Result<(), String> {
    let path = settings_path(app)?;
    write_settings(&path, settings)
}

pub fn api_auth_token_present() -> bool {
    read_api_auth_token().ok().flatten().is_some()
}

pub fn read_api_auth_token() -> Result<Option<String>, keyring::Error> {
    let entry = crate::profile::credential(API_AUTH_KEYCHAIN_SERVICE, API_AUTH_KEYCHAIN_USER)?;
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
    let entry = crate::profile::credential(API_AUTH_KEYCHAIN_SERVICE, API_AUTH_KEYCHAIN_USER)
        .map_err(|error| error.to_string())?;
    entry.set_password(token).map_err(|error| error.to_string())
}

pub fn clear_api_auth_token() -> Result<(), String> {
    let entry = crate::profile::credential(API_AUTH_KEYCHAIN_SERVICE, API_AUTH_KEYCHAIN_USER)
        .map_err(|error| error.to_string())?;
    match entry.delete_credential() {
        Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
        Err(error) => Err(error.to_string()),
    }
}

fn settings_path(app: &AppHandle) -> Result<PathBuf, String> {
    crate::profile::data_dir(app).map(|path| path.join("settings.json"))
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
/// The speculative-decoding (MTP) default for fresh settings or a missing value (sc-24139): `auto`
/// on the Candle CUDA build (the runtime resolves it to the checkpoint's MTP head where one exists,
/// and to ordinary decode, reported as `proposer=none`, where none does) and `off` on MLX and
/// Candle CPU.
pub fn default_mtp_mode() -> String {
    default_mtp_mode_for(crate::inference_runtime::execution_backend()).to_string()
}

/// [`default_mtp_mode`] for an execution backend label (`candle-cuda`, `candle-cpu`, `mlx`).
pub fn default_mtp_mode_for(execution_backend: &str) -> &'static str {
    if execution_backend == "candle-cuda" {
        "auto"
    } else {
        "off"
    }
}
fn default_mtp_draft_tokens() -> u32 {
    3
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
            ..Default::default()
        }
        .normalized()
        .unwrap();

        assert_eq!(settings.server.host, "127.0.0.1");
        assert_eq!(settings.sampling.system_prompt, "hello");
    }

    #[test]
    fn generation_defaults_round_trip_with_platform_mtp_defaults() {
        let platform = default_mtp_mode_for(crate::inference_runtime::execution_backend());
        let defaults = SamplingDefaults::default();
        assert_eq!(defaults.mtp_mode, platform);
        assert_eq!(defaults.mtp_draft_tokens, 3);
        assert!(defaults.reasoning_effort.is_none());
        let decoded: SamplingDefaults = serde_json::from_str("{}").unwrap();
        assert_eq!(decoded.mtp_mode, platform);
        assert_eq!(decoded.mtp_draft_tokens, 3);
    }

    /// AC1 (sc-24139): speculative decoding defaults to `auto` on the Candle CUDA build and `off`
    /// on MLX and Candle CPU.
    #[test]
    fn speculative_default_is_auto_on_cuda_and_off_elsewhere() {
        assert_eq!(default_mtp_mode_for("candle-cuda"), "auto");
        assert_eq!(default_mtp_mode_for("candle-cpu"), "off");
        assert_eq!(default_mtp_mode_for("mlx"), "off");
        #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
        assert_eq!(default_mtp_mode(), "auto");
        #[cfg(not(all(not(target_os = "macos"), feature = "cuda")))]
        assert_eq!(default_mtp_mode(), "off");
    }

    /// AC1: fresh settings (no file) take the platform default and the current schema marker, and
    /// a save records both.
    #[test]
    fn fresh_settings_take_the_platform_default_and_record_the_marker() {
        let fresh = AppSettings::default().normalized().unwrap();
        assert_eq!(fresh.settings_version, CURRENT_SETTINGS_VERSION);
        assert_eq!(fresh.sampling.mtp_mode, default_mtp_mode());
        assert!(!fresh.runtime.cuda_graphs, "CUDA graphs ship off");
        let saved = serde_json::to_value(&fresh).unwrap();
        assert_eq!(saved["settingsVersion"], CURRENT_SETTINGS_VERSION);
        assert_eq!(saved["sampling"]["mtpMode"], default_mtp_mode());
        assert_eq!(saved["runtime"]["cudaGraphs"], false);
    }

    /// AC1: an upgrade never flips a saved choice. A pre-marker file that serialized
    /// `"mtpMode": "off"` keeps `off` (it cannot be told apart from an explicit choice), whatever
    /// the build's new default.
    #[test]
    fn a_saved_off_survives_the_upgrade() {
        let legacy = r#"{"server":{},"sampling":{"mtpMode":"off","mtpDraftTokens":3}}"#;
        let migrated = AppSettings::from_stored_json(legacy).unwrap();
        assert_eq!(migrated.sampling.mtp_mode, "off");
        assert_eq!(migrated.settings_version, CURRENT_SETTINGS_VERSION);
        // Every other saved value is kept too.
        let legacy_enabled = r#"{"sampling":{"mtpMode":"enabled","mtpDraftTokens":5}}"#;
        let migrated = AppSettings::from_stored_json(legacy_enabled).unwrap();
        assert_eq!(migrated.sampling.mtp_mode, "enabled");
        assert_eq!(migrated.sampling.mtp_draft_tokens, 5);
        // A current-schema file's `off` is an explicit choice and stays.
        let current = r#"{"settingsVersion":1,"sampling":{"mtpMode":"off"}}"#;
        assert_eq!(
            AppSettings::from_stored_json(current)
                .unwrap()
                .sampling
                .mtp_mode,
            "off"
        );
    }

    /// AC1: only a MISSING value takes the new default. A file written before MTP existed has no
    /// `mtpMode`, so it gets `auto` on CUDA and `off` elsewhere.
    #[test]
    fn a_missing_mtp_mode_takes_the_platform_default() {
        let pre_mtp = r#"{"server":{"port":8000},"sampling":{"temperature":0.5}}"#;
        let migrated = AppSettings::from_stored_json(pre_mtp).unwrap();
        assert_eq!(migrated.sampling.mtp_mode, default_mtp_mode());
        assert_eq!(migrated.sampling.temperature, 0.5);
        assert!(!migrated.runtime.cuda_graphs);
        let empty = AppSettings::from_stored_json("{}").unwrap();
        assert_eq!(empty.sampling.mtp_mode, default_mtp_mode());
    }

    /// sc-24139 feature-end review: a pre-marker `off` kept by the migration on a build whose
    /// default is `auto` is flagged as carried over (the UI then offers Auto, once) — while the
    /// saved `off` itself is untouched. Nothing is flagged where `off` is the default, for a
    /// current-schema `off`, or for any other saved mode; the flag and a dismissal persist through
    /// a save, and moving off `off` clears the flag for good.
    #[test]
    fn a_carried_over_off_is_flagged_for_the_speculative_notice() {
        let legacy = r#"{"server":{},"sampling":{"mtpMode":"off","mtpDraftTokens":3}}"#;
        let migrated = AppSettings::from_stored_json_for(legacy, "candle-cuda").unwrap();
        assert_eq!(
            migrated.sampling.mtp_mode, "off",
            "the saved choice survives"
        );
        assert!(migrated.notices.speculative_off_carried_over);
        assert!(!migrated.notices.speculative_notice_dismissed);

        // Not carried over: where `off` is the default, a current-schema `off`, another mode.
        for (body, backend) in [
            (legacy, "candle-cpu"),
            (legacy, "mlx"),
            (
                r#"{"settingsVersion":1,"sampling":{"mtpMode":"off"}}"#,
                "candle-cuda",
            ),
            (r#"{"sampling":{"mtpMode":"enabled"}}"#, "candle-cuda"),
            // A missing mode takes this build's default, which is never a carried-over `off`.
            (
                r#"{"sampling":{}}"#,
                crate::inference_runtime::execution_backend(),
            ),
        ] {
            let settings = AppSettings::from_stored_json_for(body, backend).unwrap();
            assert!(
                !settings.notices.speculative_off_carried_over,
                "{body} on {backend}"
            );
        }

        // The flag and a dismissal survive a save (the file is at the current schema from then on).
        let mut dismissed = migrated.clone();
        dismissed.notices.speculative_notice_dismissed = true;
        let saved = serde_json::to_string(&dismissed.normalized().unwrap()).unwrap();
        let reloaded = AppSettings::from_stored_json_for(&saved, "candle-cuda").unwrap();
        assert!(reloaded.notices.speculative_off_carried_over);
        assert!(reloaded.notices.speculative_notice_dismissed);

        // Turning Auto on clears the flag, so a later explicit `off` never re-raises the notice.
        let mut auto = migrated;
        auto.sampling.mtp_mode = "auto".to_string();
        let auto = auto.normalized().unwrap();
        assert!(!auto.notices.speculative_off_carried_over);
        let mut off_again = auto;
        off_again.sampling.mtp_mode = "off".to_string();
        assert!(
            !off_again
                .normalized()
                .unwrap()
                .notices
                .speculative_off_carried_over
        );
    }

    #[test]
    fn the_cuda_graph_toggle_round_trips_through_the_settings_file() {
        let on = r#"{"settingsVersion":1,"runtime":{"cudaGraphs":true}}"#;
        let settings = AppSettings::from_stored_json(on).unwrap();
        assert!(settings.runtime.cuda_graphs);
        let saved = serde_json::to_string(&settings).unwrap();
        assert!(
            AppSettings::from_stored_json(&saved)
                .unwrap()
                .runtime
                .cuda_graphs
        );
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
