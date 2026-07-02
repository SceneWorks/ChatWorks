//! Shared filesystem + time helpers.
//!
//! Centralizes the small atomic-write + epoch-timestamp utilities that were previously copy-pasted
//! across [`crate::app_settings`], [`crate::conversations`], [`crate::model_registry`], and
//! [`crate::server`] (code-review F-010 / F-011). Keeping them here makes the atomic-write
//! invariant explicit and testable, and gives every module one `now_secs()`.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

/// Suffix appended to an in-progress atomic write before rename.
const TMP_SUFFIX: &str = "tmp";

/// Atomically write `value` serialized as pretty JSON to `path` via a temp sibling + rename, so a
/// crash mid-write never leaves a half-written file. The parent directory is created on demand so
/// callers do not need a separate `create_dir_all`.
///
/// The temp sibling is `path` with an extra `.tmp` appended (e.g. `foo.json` → `foo.json.tmp`),
/// preserving any existing extension so the final file and its temp never collide on a rename.
pub fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let tmp = with_appended_suffix(path, TMP_SUFFIX);
    fs::write(
        &tmp,
        serde_json::to_string_pretty(value).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    fs::rename(tmp, path).map_err(|error| error.to_string())
}

/// Builds a sibling path with an extra `.{suffix}` appended (e.g. `foo.json` + `tmp` -> `foo.json.tmp`).
fn with_appended_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".");
    name.push(suffix);
    PathBuf::from(name)
}

/// The current Unix epoch time in seconds (0 if the system clock is before the epoch).
pub fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

/// The current Unix epoch time in nanoseconds (0 if the system clock is before the epoch).
pub fn now_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0)
}

/// A scoped temp directory that removes itself on drop, so a panicking test never leaks a temp dir
/// under `std::env::temp_dir()` (code-review F-015). Each is namespaced with the process id plus the
/// supplied label so parallel runs don't collide. `Deref`s to [`Path`] and deref-coerces via
/// [`AsRef<Path>`], so it drops into test code written against a `&Path` / `PathBuf`.
#[cfg(test)]
pub struct TempDir {
    path: PathBuf,
}

#[cfg(test)]
impl TempDir {
    /// Create (and clear) a fresh namespaced temp directory.
    pub fn new(label: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("chatworks-{label}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Self { path }
    }

    /// The directory path.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
impl std::ops::Deref for TempDir {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
impl AsRef<Path> for TempDir {
    fn as_ref(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_json_atomic_creates_parent_and_round_trips() {
        let dir = TempDir::new("atomic-round-trip");
        let nested = dir.path().join("nested").join("settings.json");
        let value = serde_json::json!({"a": 1, "b": [2, 3]});
        write_json_atomic(&nested, &value).unwrap();
        let read: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&nested).unwrap()).unwrap();
        assert_eq!(read, value);
        // The temp sibling must be gone after the rename.
        assert!(!with_appended_suffix(&nested, TMP_SUFFIX).exists());
    }

    #[test]
    fn with_appended_suffix_preserves_extension() {
        let path = Path::new("dir/manifest.json");
        assert_eq!(
            with_appended_suffix(path, TMP_SUFFIX),
            PathBuf::from("dir/manifest.json.tmp")
        );
    }

    #[test]
    fn now_secs_is_monotonic_nonzero() {
        let a = now_secs();
        let b = now_secs();
        assert!(b >= a);
        // A real wall clock is far past the epoch; guard the degenerate clock case only.
        if a == 0 {
            assert_eq!(b, 0);
        }
    }
}
