#!/usr/bin/env bash
# Prove that a completed Tauri package contains both native media sidecars.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="${MEDIA_SIDECAR_TARGET:-$(rustc -vV | sed -n 's/^host: //p')}"
target_dir="${CARGO_TARGET_DIR:-$root/target}"
profile="${TAURI_BUILD_PROFILE:-release}"

if command -v cygpath >/dev/null 2>&1 && [[ "$target_dir" =~ ^[A-Za-z]:\\ ]]; then
  target_dir="$(cygpath -u "$target_dir")"
fi

bundle_dir="$target_dir/$profile/bundle"
if [[ ! -d "$bundle_dir" ]]; then
  echo "Tauri bundle directory does not exist: $bundle_dir" >&2
  exit 1
fi

require_listing_entry() {
  local listing="$1"
  local name="$2"
  if ! grep -Eiq "(^|[/\\])${name}$" <<<"$listing"; then
    echo "Packaged application is missing $name." >&2
    exit 1
  fi
}

case "$target" in
  aarch64-apple-darwin|x86_64-apple-darwin)
    package="$(find "$bundle_dir/macos" -maxdepth 1 -type d -name '*.app' -print -quit)"
    if [[ -z "$package" ]]; then
      echo "No macOS .app was produced under $bundle_dir/macos." >&2
      exit 1
    fi
    for name in ffmpeg ffprobe; do
      binary="$package/Contents/MacOS/$name"
      if [[ ! -f "$binary" || ! -x "$binary" ]]; then
        echo "macOS package is missing executable Contents/MacOS/$name." >&2
        exit 1
      fi
    done
    ;;
  x86_64-unknown-linux-gnu|aarch64-unknown-linux-gnu)
    package="$(find "$bundle_dir/deb" -maxdepth 1 -type f -name '*.deb' -print -quit)"
    if [[ -z "$package" ]]; then
      echo "No Linux .deb was produced under $bundle_dir/deb." >&2
      exit 1
    fi
    listing="$(dpkg-deb --contents "$package" | awk '{print $NF}')"
    require_listing_entry "$listing" ffmpeg
    require_listing_entry "$listing" ffprobe
    ;;
  x86_64-pc-windows-msvc)
    package="$(find "$bundle_dir/nsis" -maxdepth 1 -type f -name '*-setup.exe' -print -quit)"
    if [[ -z "$package" ]]; then
      echo "No Windows NSIS installer was produced under $bundle_dir/nsis." >&2
      exit 1
    fi
    if ! command -v 7z >/dev/null 2>&1; then
      echo "7z is required to inspect the Windows NSIS package." >&2
      exit 1
    fi
    listing="$(7z l -ba "$package")"
    require_listing_entry "$listing" 'ffmpeg\.exe'
    require_listing_entry "$listing" 'ffprobe\.exe'
    ;;
  *)
    echo "No package verification recipe for target: $target" >&2
    exit 1
    ;;
esac

printf 'Verified packaged FFmpeg sidecars for %s in %s.\n' "$target" "$package"
