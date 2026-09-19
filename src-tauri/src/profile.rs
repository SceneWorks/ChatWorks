//! Opt-in acceptance isolation. Ordinary launches retain their existing files and credentials.
use std::ffi::OsString;
use std::path::PathBuf;

use sha2::{Digest, Sha256};
use tauri::{AppHandle, Manager};

fn resolve_root(value: Option<OsString>) -> Result<Option<PathBuf>, String> {
    let Some(value) = value else { return Ok(None) };
    let path = PathBuf::from(value);
    if !path.is_absolute() {
        return Err("CHATWORKS_PROFILE_DIR must be an absolute existing directory".into());
    }
    let root = path
        .canonicalize()
        .map_err(|error| format!("invalid CHATWORKS_PROFILE_DIR: {error}"))?;
    if !root.is_dir() {
        return Err("CHATWORKS_PROFILE_DIR must name a directory".into());
    }
    Ok(Some(root))
}

pub fn root() -> Result<Option<PathBuf>, String> {
    resolve_root(std::env::var_os("CHATWORKS_PROFILE_DIR"))
}

fn digest(root: &std::path::Path) -> [u8; 32] {
    Sha256::digest(root.as_os_str().as_encoded_bytes()).into()
}

fn select_data_dir(
    profile: Option<PathBuf>,
    default: impl FnOnce() -> Result<PathBuf, String>,
) -> Result<PathBuf, String> {
    match profile {
        Some(root) => Ok(root),
        None => default(),
    }
}

pub fn data_dir(app: &AppHandle) -> Result<PathBuf, String> {
    select_data_dir(root()?, || {
        app.path().app_data_dir().map_err(|error| error.to_string())
    })
}

fn scoped_service(base: &str, root: Option<&std::path::Path>) -> String {
    match root {
        None => base.to_owned(),
        Some(root) => format!(
            "{base}.profile.{:x}",
            Sha256::digest(root.as_os_str().as_encoded_bytes())
        ),
    }
}

pub fn credential(base: &str, user: &str) -> Result<keyring::Entry, keyring::Error> {
    let profile = root()
        .map_err(|error| keyring::Error::PlatformFailure(Box::new(std::io::Error::other(error))))?;
    // One selected service only: a missing profile credential never falls back to normal Keychain.
    keyring::Entry::new(&scoped_service(base, profile.as_deref()), user)
}

pub fn isolate_webviews(config: &mut tauri::Config) -> Result<(), String> {
    isolate_config(config, root()?);
    Ok(())
}

fn isolate_config(config: &mut tauri::Config, profile: Option<PathBuf>) {
    if let Some(root) = profile {
        let bytes = digest(&root);
        let mut identifier = [0; 16];
        identifier.copy_from_slice(&bytes[..16]);
        for window in &mut config.app.windows {
            window.data_directory = Some(root.join("webview"));
            // WKWebView ignores data_directory; its persistent store is selected by UUID instead.
            window.data_store_identifier = Some(identifier);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_profile_preserves_paths_and_keyring_service() {
        assert_eq!(resolve_root(None).unwrap(), None);
        assert_eq!(
            select_data_dir(None, || Ok(PathBuf::from("normal"))).unwrap(),
            PathBuf::from("normal")
        );
        assert_eq!(scoped_service("service", None), "service");
    }

    #[test]
    fn explicit_profile_never_uses_default_paths_or_keyring_namespace() {
        let dir = crate::fsutil::TempDir::new("profile-isolation");
        let root = resolve_root(Some(dir.as_os_str().to_owned()))
            .unwrap()
            .unwrap();
        assert_eq!(
            select_data_dir(Some(root.clone()), || panic!("default path fallback")).unwrap(),
            root
        );
        let service = scoped_service("service", Some(&root));
        assert!(service.starts_with("service.profile."));
        assert_eq!(service, scoped_service("service", Some(&root)));
        assert_ne!(
            service,
            scoped_service("service", Some(&root.join("other")))
        );
        assert_ne!(service, "service");
        assert!(!service.contains(root.to_str().unwrap()));
    }

    #[test]
    fn invalid_profile_errors_instead_of_falling_back() {
        assert!(resolve_root(Some("relative".into())).is_err());
        assert!(resolve_root(Some("".into())).is_err());
        let dir = crate::fsutil::TempDir::new("profile-invalid");
        assert!(resolve_root(Some(dir.join("missing").into_os_string())).is_err());
        let file = dir.join("file");
        std::fs::write(&file, "x").unwrap();
        assert!(resolve_root(Some(file.into_os_string())).is_err());
    }

    #[test]
    fn webview_storage_is_profile_specific_and_default_configuration_is_unchanged() {
        let mut config = tauri::Config::default();
        config.app.windows.push(Default::default());
        let original = serde_json::to_value(&config).unwrap();
        isolate_config(&mut config, None);
        assert_eq!(serde_json::to_value(&config).unwrap(), original);
        let root = crate::fsutil::TempDir::new("profile-webview");
        let first = root.join("first");
        isolate_config(&mut config, Some(first.clone()));
        assert_eq!(
            config.app.windows[0].data_directory,
            Some(first.join("webview"))
        );
        let first_id = config.app.windows[0].data_store_identifier.unwrap();
        isolate_config(&mut config, Some(first));
        assert_eq!(config.app.windows[0].data_store_identifier, Some(first_id));
        isolate_config(&mut config, Some(root.join("second")));
        assert_ne!(config.app.windows[0].data_store_identifier, Some(first_id));
    }

    #[cfg(unix)]
    #[test]
    fn aliases_share_canonical_namespace() {
        let dir = crate::fsutil::TempDir::new("profile-alias");
        let actual = dir.join("actual");
        std::fs::create_dir(&actual).unwrap();
        let alias = dir.join("alias");
        std::os::unix::fs::symlink(&actual, &alias).unwrap();
        let first = resolve_root(Some(actual.into_os_string()))
            .unwrap()
            .unwrap();
        let second = resolve_root(Some(alias.into_os_string())).unwrap().unwrap();
        assert_eq!(
            scoped_service("service", Some(&first)),
            scoped_service("service", Some(&second))
        );
    }
}
