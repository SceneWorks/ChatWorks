#!/usr/bin/env bash
# Build the pinned, LGPL-only FFmpeg/ffprobe sidecars consumed by the Tauri bundle.
# This is an explicit release-build input: it never runs in the installed application.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
version="9.0"
archive="ffmpeg-${version}.tar.xz"
url="https://ffmpeg.org/releases/${archive}"
sha256="7f607a00dd0d28a729d5a4811205812eef01cf6ef6155025febb6f36a9062d52"
target="${MEDIA_SIDECAR_TARGET:-$(rustc -vV | sed -n 's/^host: //p')}"
host="$(rustc -vV | sed -n 's/^host: //p')"
cache_dir="${CHATWORKS_FFMPEG_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/chatworks/ffmpeg}"
work_dir="${CHATWORKS_FFMPEG_WORKDIR:-$cache_dir/build-$target}"
out_dir="$root/src-tauri/binaries"

case "$target" in
  aarch64-apple-darwin)
    configure_target=(--target-os=darwin --arch=aarch64)
    binary_suffix=""
    ;;
  x86_64-apple-darwin)
    configure_target=(--target-os=darwin --arch=x86_64)
    binary_suffix=""
    ;;
  x86_64-unknown-linux-gnu)
    configure_target=(--target-os=linux --arch=x86_64)
    binary_suffix=""
    ;;
  aarch64-unknown-linux-gnu)
    configure_target=(--target-os=linux --arch=aarch64)
    binary_suffix=""
    ;;
  x86_64-pc-windows-msvc)
    configure_target=(--target-os=win32 --arch=x86_64 --toolchain=msvc)
    binary_suffix=".exe"
    ;;
  *)
    echo "No reviewed FFmpeg sidecar build recipe for target: $target" >&2
    exit 1
    ;;
esac

if [[ "${1:-}" == "--print-output-paths" ]]; then
  printf '%s\n' "$out_dir/ffmpeg-$target$binary_suffix" "$out_dir/ffprobe-$target$binary_suffix"
  exit 0
fi

if [[ "$target" != "$host" ]]; then
  echo "Refusing to cross-build FFmpeg: target $target differs from native host $host." >&2
  echo "Run this provisioning step on the matching release runner." >&2
  exit 1
fi

mkdir -p "$cache_dir" "$work_dir" "$out_dir"
archive_path="$cache_dir/$archive"
if [[ ! -f "$archive_path" ]]; then
  curl --fail --location --proto '=https' --tlsv1.2 --silent --show-error \
    --output "$archive_path.part" "$url"
  mv "$archive_path.part" "$archive_path"
fi
if command -v shasum >/dev/null 2>&1; then
  actual_sha="$(shasum -a 256 "$archive_path" | awk '{print $1}')"
elif command -v sha256sum >/dev/null 2>&1; then
  actual_sha="$(sha256sum "$archive_path" | awk '{print $1}')"
else
  echo "Need shasum or sha256sum to verify the pinned FFmpeg source." >&2
  exit 1
fi
if [[ "$actual_sha" != "$sha256" ]]; then
  echo "FFmpeg source checksum mismatch: expected $sha256, got $actual_sha" >&2
  exit 1
fi

source_dir="$work_dir/ffmpeg-$version"
if [[ ! -f "$source_dir/config.mak" ]]; then
  rm -rf "$source_dir"
  tar -xJf "$archive_path" -C "$work_dir"
  (
    cd "$source_dir"
    ./configure "${configure_target[@]}" \
      --disable-autodetect --disable-doc --disable-debug --disable-network --disable-x86asm \
      --disable-shared --enable-static --enable-small --disable-everything \
      --enable-ffmpeg --enable-ffprobe \
      --enable-avcodec --enable-avformat --enable-avutil --enable-swscale \
      --enable-protocol=file,pipe \
      --enable-demuxer=avi,matroska,mov,mpegts \
      --enable-decoder=av1,h264,hevc,mjpeg,mpeg4,vp8,vp9 \
      --enable-parser=av1,h264,hevc,mpeg4video,vp8,vp9 \
      --enable-filter=scale --enable-encoder=mjpeg --enable-muxer=image2 \
      --disable-programs --enable-ffmpeg --enable-ffprobe
  )
fi
make -C "$source_dir" -j"${CHATWORKS_FFMPEG_JOBS:-4}" "ffmpeg$binary_suffix" "ffprobe$binary_suffix"
cp "$source_dir/ffmpeg$binary_suffix" "$out_dir/ffmpeg-$target$binary_suffix"
cp "$source_dir/ffprobe$binary_suffix" "$out_dir/ffprobe-$target$binary_suffix"
chmod 0755 "$out_dir/ffmpeg-$target$binary_suffix" "$out_dir/ffprobe-$target$binary_suffix"
printf 'Provisioned FFmpeg %s sidecars for %s.\n' "$version" "$target"
