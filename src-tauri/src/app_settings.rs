use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use tauri::AppHandle;

use crate::core_llm::{MtpMode, Speculative};
use crate::fsutil::write_json_atomic;
use crate::server::{DEFAULT_OPENAI_HOST, DEFAULT_OPENAI_PORT};

const API_AUTH_KEYCHAIN_SERVICE: &str = "net.trefry.chatworks.openai";
const API_AUTH_KEYCHAIN_USER: &str = "api-auth-token";

/// The settings schema this build writes. Every save stamps it; a file without one was written
/// before this marker existed (version 0).
///
/// * **Version 2 (sc-24445)** stores speculative decoding as the runtime's proposer-agnostic option
///   ([`SamplingDefaults::speculative`]: `"off" | "auto" | {"proposer", "depth"}`, or `null` for
///   "the runtime's default"). An older file's `mtpMode` / `mtpDraftTokens` map onto it through the
///   runtime's own legacy mapping (`From<MtpMode> for Speculative`): `off` -> `off`, `auto` ->
///   `auto`, `enabled` + N -> `{proposer: mtp, depth: N}` ([`legacy_speculative`]). A file with no
///   speculative value at all follows the runtime's default ([`runtime_speculative_default`]),
///   never a ChatWorks-side per-backend copy (epic sc-24432 E5).
/// * **Version 1 (sc-24139)** marked files written after the speculative default stopped being `off`
///   everywhere. Every pre-version-2 save serialized `mtpMode` — including an MLX version-1 `off`
///   that was almost always the untouched default — so a pre-version-2 `"mtpMode": "off"` cannot be
///   told apart from "the user chose off": it is honoured as written. Where the runtime's default is
///   not `off`, such a carried-over `off` is flagged
///   ([`NoticeSettings::speculative_off_carried_over`]) so the UI can offer — once, dismissibly —
///   to turn on Auto, without ever changing the saved choice itself.
pub const CURRENT_SETTINGS_VERSION: u32 = 2;

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
    /// Speculative decoding is `off` only because a pre-version-2 settings file carried `off` over
    /// where the runtime's default is not `off` (see [`CURRENT_SETTINGS_VERSION`]). Set by the
    /// migration; cleared as soon as the option is anything but `off`, so a later explicit `off`
    /// never re-raises the notice.
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
        Self::from_stored_json_for(body, runtime_speculative_default())
    }

    /// [`from_stored_json`](Self::from_stored_json) against a runtime speculative default, which
    /// decides whether a pre-version-2 `off` was carried over rather than chosen under that default.
    pub fn from_stored_json_for(body: &str, runtime_default: Speculative) -> Result<Self, String> {
        let mut settings =
            serde_json::from_str::<AppSettings>(body).map_err(|error| error.to_string())?;
        if settings.settings_version < 2
            && settings.sampling.legacy_mtp_mode.as_deref() == Some("off")
            && runtime_default != Speculative::Off
        {
            settings.notices.speculative_off_carried_over = true;
        }
        settings.normalized()
    }

    pub fn normalized(mut self) -> Result<Self, String> {
        // A pre-marker file's values are kept as written (validated below); stamping the marker
        // records on the next save that this file now carries explicit choices.
        self.settings_version = CURRENT_SETTINGS_VERSION;
        self.sampling.migrate_legacy_speculative()?;
        if self.sampling.speculative != Some(Speculative::Off) {
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
        validate_speculative(self.sampling.speculative)?;
        if !matches!(
            self.sampling.reasoning_effort.as_deref(),
            None | Some("low" | "medium" | "xhigh")
        ) {
            return Err("reasoning effort must be low, medium, or xhigh".to_string());
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
    /// The saved speculative-decoding option (sc-24445), the runtime's own proposer-agnostic
    /// type: `off`, `auto`, or `{proposer, depth}`. `None` (`null`) follows the runtime's default
    /// ([`runtime_speculative_default`]). A depth above what the loaded model advertises is
    /// clamped by the runtime, which names the clamp in the decode report.
    #[serde(default)]
    pub speculative: Option<Speculative>,
    /// The pre-sc-24445 `mtpMode` (`off | auto | enabled`), read only to migrate it onto
    /// [`speculative`](Self::speculative); never written back.
    #[serde(default, rename = "mtpMode", skip_serializing)]
    pub(crate) legacy_mtp_mode: Option<String>,
    /// The pre-sc-24445 `mtpDraftTokens`, read only with [`legacy_mtp_mode`](Self::legacy_mtp_mode).
    #[serde(default, rename = "mtpDraftTokens", skip_serializing)]
    pub(crate) legacy_mtp_draft_tokens: Option<u32>,
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
            speculative: None,
            legacy_mtp_mode: None,
            legacy_mtp_draft_tokens: None,
            top_k: None,
            presence_penalty: None,
            repetition_penalty: None,
            repetition_context: None,
            seed: None,
        }
    }
}

impl SamplingDefaults {
    /// Map a pre-sc-24445 `mtpMode` / `mtpDraftTokens` onto [`speculative`](Self::speculative)
    /// when the file carries no speculative option of its own, then drop the legacy fields.
    pub(crate) fn migrate_legacy_speculative(&mut self) -> Result<(), String> {
        let legacy = legacy_speculative(
            self.legacy_mtp_mode.take().as_deref(),
            self.legacy_mtp_draft_tokens.take(),
        )?;
        if self.speculative.is_none() {
            self.speculative = legacy;
        }
        Ok(())
    }
}

/// The runtime's mapping of a pre-sc-24445 saved `mtpMode` (`off | auto | enabled`, with
/// `mtpDraftTokens` drafts, `3` when absent) onto the proposer-agnostic option: the runtime's own
/// `From<MtpMode> for Speculative`, so `auto` means the runtime's `auto` (MTP where the model has a
/// head, else prompt lookup). `None` when no legacy mode was saved.
pub(crate) fn legacy_speculative(
    mode: Option<&str>,
    draft_tokens: Option<u32>,
) -> Result<Option<Speculative>, String> {
    let mode = match mode {
        None => return Ok(None),
        Some("off") => MtpMode::Off,
        Some("auto") => MtpMode::Auto,
        Some("enabled") => match draft_tokens.unwrap_or(LEGACY_MTP_DRAFT_TOKENS) {
            0 => return Err("MTP draft tokens must be at least 1".to_string()),
            draft_tokens => MtpMode::Enabled { draft_tokens },
        },
        Some(other) => {
            return Err(format!(
                "mtp mode must be off, auto, or enabled (found `{other}`)"
            ))
        }
    };
    Ok(Some(Speculative::from(mode)))
}

/// The draft count a legacy `"mtpMode": "enabled"` without `mtpDraftTokens` ran with.
const LEGACY_MTP_DRAFT_TOKENS: u32 = 3;

/// A saved speculative option must name at least one draft per verify step; the upper bound is the
/// loaded model's, which the runtime clamps to.
pub(crate) fn validate_speculative(speculative: Option<Speculative>) -> Result<(), String> {
    match speculative {
        Some(Speculative::Proposer { depth: 0, .. }) => {
            Err("speculative depth must be at least 1".to_string())
        }
        _ => Ok(()),
    }
}

/// The speculative option a request that names none runs under, as the linked runtime resolves it
/// (epic sc-24432 E5): its own resolution of an unset request option
/// (`TextLlmRequest::speculative_mode`). ChatWorks keeps no per-backend copy; the runtime's
/// per-backend defaults table (sc-24446) answers here once it exists.
pub fn runtime_speculative_default() -> Speculative {
    crate::inference_runtime::speculative_default()
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

pub fn resolve_api_auth_token<E: std::fmt::Display>(
    auth_enabled: bool,
    read: impl FnOnce() -> Result<Option<String>, E>,
) -> Result<Option<String>, String> {
    if !auth_enabled {
        return Ok(None);
    }
    read()
        .map_err(|error| format!("could not read API auth token: {error}"))?
        .filter(|token| !token.trim().is_empty())
        .ok_or_else(|| "API auth token must be saved before enabling auth".to_string())
        .map(Some)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enabled_auth_requires_a_readable_token_for_every_local_file_policy() {
        for allow_local_files in [false, true] {
            let mut settings = AppSettings::default();
            settings.server.auth_enabled = true;
            settings.server.allow_local_files = allow_local_files;
            let settings = settings.normalized().unwrap();
            assert!(
                resolve_api_auth_token(settings.server.auth_enabled, || Ok::<_, &str>(None))
                    .is_err()
            );
            assert!(
                resolve_api_auth_token(settings.server.auth_enabled, || Err::<Option<String>, _>(
                    "denied"
                ))
                .is_err()
            );
            assert_eq!(
                resolve_api_auth_token(settings.server.auth_enabled, || Ok::<_, &str>(Some(
                    "secret".into()
                )))
                .unwrap(),
                Some("secret".into())
            );
        }
    }

    #[test]
    fn disabled_auth_never_reads_credentials() {
        assert_eq!(
            resolve_api_auth_token(false, || -> Result<Option<String>, &str> {
                panic!("credential read")
            })
            .unwrap(),
            None
        );
    }

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

    use crate::core_llm::SpeculativeProposer;

    fn mtp(depth: u32) -> Option<Speculative> {
        Some(Speculative::proposer(SpeculativeProposer::Mtp, depth))
    }

    /// sc-24445 AC1: every saved `mtpMode` maps onto the runtime's proposer-agnostic option —
    /// `off` -> `off`, `auto` -> `auto`, `enabled` + N -> `{proposer: mtp, depth: N}` (3 drafts when
    /// the file never saved a count) — for every schema version that could have written one, and
    /// a save writes only the new option.
    #[test]
    fn saved_mtp_modes_map_onto_the_speculative_option() {
        for version in ["", r#""settingsVersion":0,"#, r#""settingsVersion":1,"#] {
            for (sampling, expected) in [
                (
                    r#"{"mtpMode":"off","mtpDraftTokens":3}"#,
                    Some(Speculative::Off),
                ),
                (
                    r#"{"mtpMode":"auto","mtpDraftTokens":3}"#,
                    Some(Speculative::Auto),
                ),
                (r#"{"mtpMode":"enabled","mtpDraftTokens":5}"#, mtp(5)),
                (r#"{"mtpMode":"enabled"}"#, mtp(3)),
                (r#"{"mtpDraftTokens":5}"#, None),
            ] {
                let body = format!(r#"{{{version}"sampling":{sampling}}}"#);
                let migrated = AppSettings::from_stored_json(&body).unwrap();
                assert_eq!(migrated.sampling.speculative, expected, "{body}");
                assert_eq!(migrated.settings_version, CURRENT_SETTINGS_VERSION);
                let saved = serde_json::to_value(&migrated).unwrap();
                assert!(saved["sampling"].get("mtpMode").is_none(), "{saved}");
                assert!(saved["sampling"].get("mtpDraftTokens").is_none(), "{saved}");
                let reloaded = AppSettings::from_stored_json(&saved.to_string()).unwrap();
                assert_eq!(
                    reloaded.sampling.speculative, expected,
                    "{body} after a save"
                );
            }
        }
        // A legacy value the old schema itself refused is still refused, naming the field.
        for bad in [
            r#"{"sampling":{"mtpMode":"always"}}"#,
            r#"{"sampling":{"mtpMode":"enabled","mtpDraftTokens":0}}"#,
        ] {
            assert!(AppSettings::from_stored_json(bad).is_err(), "{bad}");
        }
    }

    /// The new option round-trips in its wire form, and an explicit option wins over a stray
    /// legacy field.
    #[test]
    fn the_speculative_option_round_trips_in_its_wire_form() {
        for (wire, expected) in [
            (r#""off""#, Some(Speculative::Off)),
            (r#""auto""#, Some(Speculative::Auto)),
            (
                r#"{"proposer":"prompt_lookup","depth":4}"#,
                Some(Speculative::proposer(SpeculativeProposer::PromptLookup, 4)),
            ),
            (
                r#"{"proposer":"draft_model","depth":2}"#,
                Some(Speculative::proposer(SpeculativeProposer::DraftModel, 2)),
            ),
            ("null", None),
        ] {
            let body = format!(r#"{{"settingsVersion":2,"sampling":{{"speculative":{wire}}}}}"#);
            let settings = AppSettings::from_stored_json(&body).unwrap();
            assert_eq!(settings.sampling.speculative, expected, "{body}");
            let saved = serde_json::to_value(&settings).unwrap();
            assert_eq!(
                saved["sampling"]["speculative"],
                serde_json::from_str::<serde_json::Value>(wire).unwrap()
            );
        }
        let both = r#"{"sampling":{"speculative":"off","mtpMode":"auto"}}"#;
        assert_eq!(
            AppSettings::from_stored_json(both)
                .unwrap()
                .sampling
                .speculative,
            Some(Speculative::Off)
        );
        let zero = r#"{"sampling":{"speculative":{"proposer":"mtp","depth":0}}}"#;
        assert!(AppSettings::from_stored_json(zero).is_err());
    }

    /// Epic sc-24432 E5: no per-backend default lives in ChatWorks. Fresh settings, and a file
    /// that never saved a speculative value, follow the runtime's default (`None`), which is the
    /// runtime's own resolution of an unset request option.
    #[test]
    fn an_unsaved_speculative_option_follows_the_runtime_default() {
        let fresh = AppSettings::default().normalized().unwrap();
        assert_eq!(fresh.settings_version, CURRENT_SETTINGS_VERSION);
        assert_eq!(fresh.sampling.speculative, None);
        assert!(!fresh.runtime.cuda_graphs, "CUDA graphs ship off");
        let saved = serde_json::to_value(&fresh).unwrap();
        assert_eq!(saved["settingsVersion"], CURRENT_SETTINGS_VERSION);
        assert!(saved["sampling"]["speculative"].is_null());
        assert_eq!(saved["runtime"]["cudaGraphs"], false);
        for body in [
            "{}",
            r#"{"server":{"port":8000},"sampling":{"temperature":0.5}}"#,
        ] {
            let migrated = AppSettings::from_stored_json(body).unwrap();
            assert_eq!(migrated.sampling.speculative, None, "{body}");
        }
        assert_eq!(
            runtime_speculative_default(),
            crate::core_llm::TextLlmRequest::default().speculative_mode()
        );
    }

    /// A pre-version-2 `off` (version 0 or 1) kept by the migration where the runtime's default is
    /// not `off` is flagged as carried over (the UI then offers Auto, once) — while the saved `off`
    /// itself is untouched. Nothing is flagged where the runtime's default is `off`, for a version-2
    /// `off`, or for any other saved mode; the flag and a dismissal persist through a save, and
    /// moving off `off` clears the flag for good.
    #[test]
    fn a_carried_over_off_is_flagged_for_the_speculative_notice() {
        let legacy = r#"{"server":{},"sampling":{"mtpMode":"off","mtpDraftTokens":3}}"#;
        let migrated = AppSettings::from_stored_json_for(legacy, Speculative::Auto).unwrap();
        assert_eq!(
            migrated.sampling.speculative,
            Some(Speculative::Off),
            "the saved choice survives"
        );
        assert!(migrated.notices.speculative_off_carried_over);
        assert!(!migrated.notices.speculative_notice_dismissed);

        // An MLX version-1 `off` is almost always the untouched default: flagged too.
        let v1 = r#"{"settingsVersion":1,"sampling":{"mtpMode":"off"}}"#;
        let v1 = AppSettings::from_stored_json_for(v1, Speculative::Auto).unwrap();
        assert_eq!(v1.sampling.speculative, Some(Speculative::Off));
        assert!(v1.notices.speculative_off_carried_over);

        for (body, runtime_default) in [
            (legacy, Speculative::Off),
            (
                r#"{"settingsVersion":2,"sampling":{"speculative":"off"}}"#,
                Speculative::Auto,
            ),
            (r#"{"sampling":{"mtpMode":"enabled"}}"#, Speculative::Auto),
            (r#"{"sampling":{}}"#, Speculative::Auto),
        ] {
            let settings = AppSettings::from_stored_json_for(body, runtime_default).unwrap();
            assert!(
                !settings.notices.speculative_off_carried_over,
                "{body} under {runtime_default:?}"
            );
        }

        let mut dismissed = migrated.clone();
        dismissed.notices.speculative_notice_dismissed = true;
        let saved = serde_json::to_string(&dismissed.normalized().unwrap()).unwrap();
        let reloaded = AppSettings::from_stored_json_for(&saved, Speculative::Auto).unwrap();
        assert!(reloaded.notices.speculative_off_carried_over);
        assert!(reloaded.notices.speculative_notice_dismissed);
        assert_eq!(reloaded.sampling.speculative, Some(Speculative::Off));

        let mut auto = migrated;
        auto.sampling.speculative = Some(Speculative::Auto);
        let auto = auto.normalized().unwrap();
        assert!(!auto.notices.speculative_off_carried_over);
        let mut off_again = auto;
        off_again.sampling.speculative = Some(Speculative::Off);
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
