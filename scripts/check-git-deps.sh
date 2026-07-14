#!/usr/bin/env bash
# Supply-chain gate for the canonical SceneWorks inference release.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
lock="$root/Cargo.lock"
manifest="$root/src-tauri/Cargo.toml"
pinfile="$root/scripts/git-deps-pinned.csv"

if [[ ! -f "$lock" || ! -f "$manifest" || ! -f "$pinfile" ]]; then
  echo "FAIL: expected Cargo.lock, src-tauri/Cargo.toml, and scripts/git-deps-pinned.csv." >&2
  exit 1
fi

expected_tag="$(awk -F, '$1 == "SceneWorks/inference" { print $2 }' "$pinfile")"
expected_sha="$(awk -F, '$1 == "SceneWorks/inference" { print $3 }' "$pinfile")"
if [[ ! "$expected_tag" =~ ^runtime-[0-9]{4}\.[0-9]{2}\.[0-9]+([.-][0-9A-Za-z.-]+)?$ ]]; then
  echo "FAIL: scripts/git-deps-pinned.csv must record one runtime release tag." >&2
  exit 1
fi
if [[ ! "$expected_sha" =~ ^[0-9a-f]{40}$ ]]; then
  echo "FAIL: scripts/git-deps-pinned.csv must record the release tag's full commit SHA." >&2
  exit 1
fi

manifest_tags="$(
  sed -nE '/git = "https:\/\/github\.com\/SceneWorks\/inference"/s/.*tag = "([^"]+)".*/\1/p' "$manifest" \
    | sort -u
)"
if [[ "$manifest_tags" != "$expected_tag" ]]; then
  echo "FAIL: every runtime bundle must use the recorded immutable inference release tag." >&2
  printf '       expected tag: %s\n' "$expected_tag" >&2
  printf '       manifest tags: %s\n' "${manifest_tags:-"(none)"}" >&2
  exit 1
fi

resolved_tags="$(
  sed -nE '/^source = "git\+https:\/\/github\.com\/SceneWorks\/inference\?tag=/s/.*\?tag=([^#]+)#[0-9a-f]{40}"/\1/p' "$lock" \
    | sort -u
)"
resolved_revs="$(
  sed -nE '/^source = "git\+https:\/\/github\.com\/SceneWorks\/inference\?tag=/s/.*#([0-9a-f]{40})"/\1/p' "$lock" \
    | sort -u
)"
if [[ "$resolved_tags" != "$expected_tag" || "$resolved_revs" != "$expected_sha" ]]; then
  echo "FAIL: Cargo.lock must resolve every inference package to the recorded tag and commit." >&2
  printf '       expected %s at %s\n' "$expected_tag" "$expected_sha" >&2
  printf '       resolved tags: %s\n' "${resolved_tags:-"(none)"}" >&2
  printf '       resolved revisions: %s\n' "${resolved_revs:-"(none)"}" >&2
  exit 1
fi

legacy_pattern='github\.com/(SceneWorks/(core-llm|mlx-llm|candle-llm)|michaeltrefry/(mlx-gen|candle-gen))'
if grep -Eq "$legacy_pattern" "$manifest" "$lock"; then
  echo "FAIL: an individual legacy inference repository remains in the manifest or lockfile." >&2
  grep -En "$legacy_pattern" "$manifest" "$lock" >&2
  exit 1
fi

package_count="$(grep -c '^source = "git+https://github.com/SceneWorks/inference?tag=' "$lock")"
if [[ "$package_count" -eq 0 ]]; then
  echo "FAIL: Cargo.lock contains no packages from SceneWorks/inference." >&2
  exit 1
fi

printf 'Supply-chain gate OK: %s inference packages use %s at %s.\n' \
  "$package_count" "$expected_tag" "${expected_sha:0:12}"
