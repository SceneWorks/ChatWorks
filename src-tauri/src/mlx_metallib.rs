use std::ffi::OsStr;
use std::path::{Path, PathBuf};

/// An installed MLX app must use its own Metal kernels, even when a user cache or an
/// inherited PMETAL_METALLIB_PATH points at a different build.
pub fn bundled_metallib(executable: &Path) -> Result<Option<PathBuf>, String> {
    let Some(macos) = executable.parent() else {
        return Ok(None);
    };
    let Some(contents) = macos.parent() else {
        return Ok(None);
    };
    let Some(app) = contents.parent() else {
        return Ok(None);
    };
    if macos.file_name() != Some(OsStr::new("MacOS"))
        || contents.file_name() != Some(OsStr::new("Contents"))
        || app.extension() != Some(OsStr::new("app"))
    {
        return Ok(None);
    }
    let library = contents.join("Resources/mlx.metallib");
    if !library.is_file() {
        return Err(format!(
            "installed ChatWorks MLX package is missing {}",
            library.display()
        ));
    }
    Ok(Some(library))
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub fn configure_before_app_start() -> Result<(), String> {
    let executable = std::env::current_exe().map_err(|error| error.to_string())?;
    if let Some(library) = bundled_metallib(&executable)? {
        // Called as the first operation in main, before Tauri or the inference worker starts.
        // MLX's explicit override precedes its compiled path and the shared user cache.
        std::env::set_var("PMETAL_METALLIB_PATH", library);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn installed_app_requires_its_colocated_mlx_library() {
        let root = crate::fsutil::TempDir::new("mlx-metallib");
        let contents = root.path().join("ChatWorks.app/Contents");
        let executable = contents.join("MacOS/chatworks");
        fs::create_dir_all(executable.parent().unwrap()).unwrap();
        fs::write(&executable, b"app").unwrap();
        assert!(bundled_metallib(&executable)
            .unwrap_err()
            .contains("missing"));
        fs::write(contents.join("MacOS/mlx.metallib"), b"wrong location").unwrap();
        assert!(bundled_metallib(&executable)
            .unwrap_err()
            .contains("missing"));
        let resource = contents.join("Resources/mlx.metallib");
        fs::create_dir_all(resource.parent().unwrap()).unwrap();
        fs::write(&resource, b"bundled").unwrap();
        assert_eq!(bundled_metallib(&executable).unwrap(), Some(resource));
        assert_eq!(
            bundled_metallib(&root.path().join("target/release/chatworks")).unwrap(),
            None
        );
    }
}
