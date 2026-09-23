#!/usr/bin/env bash
# Supply-chain gate for the canonical SceneWorks inference pin.
#
# scripts/git-deps-pinned.csv records exactly one pin: `SceneWorks/inference,<ref>,<sha>`.
#   * A released runtime (`<ref>` = `runtime-YYYY.MM.N[-rc.N]`): every bundle uses `tag = "<ref>"`
#     and Cargo.lock resolves `?tag=<ref>#<sha>`. This is the only pin `main` accepts.
#   * A pre-release feature-train pin (`<ref>` = `rev`): every bundle uses `rev = "<sha>"` and
#     Cargo.lock resolves `?rev=<sha>#<sha>`. A feature branch tracks the matching inference feature
#     branch this way until its terminal story replaces the pin with the released runtime tag, so a
#     `rev` pin fails on any pull request into `main` and on `main` itself.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
lock="$root/Cargo.lock"
manifest="$root/src-tauri/Cargo.toml"
pinfile="$root/scripts/git-deps-pinned.csv"

if [[ ! -f "$lock" || ! -f "$manifest" || ! -f "$pinfile" ]]; then
  echo "FAIL: expected Cargo.lock, src-tauri/Cargo.toml, and scripts/git-deps-pinned.csv." >&2
  exit 1
fi

expected_ref="$(awk -F, '$1 == "SceneWorks/inference" { print $2 }' "$pinfile")"
expected_sha="$(awk -F, '$1 == "SceneWorks/inference" { print $3 }' "$pinfile")"
if [[ ! "$expected_sha" =~ ^[0-9a-f]{40}$ ]]; then
  echo "FAIL: scripts/git-deps-pinned.csv must record the pinned full commit SHA." >&2
  exit 1
fi
if [[ "$expected_ref" =~ ^runtime-[0-9]{4}\.[0-9]{2}\.[0-9]+([.-][0-9A-Za-z.-]+)?$ ]]; then
  pin_kind="tag"
  manifest_expected="tag=$expected_ref"
  lock_expected="tag=$expected_ref#$expected_sha"
elif [[ "$expected_ref" == "rev" ]]; then
  pin_kind="rev"
  manifest_expected="rev=$expected_sha"
  lock_expected="rev=$expected_sha#$expected_sha"
  # GitHub sets GITHUB_BASE_REF on pull requests and GITHUB_REF on pushes.
  if [[ "${GITHUB_BASE_REF:-}" == "main" || ( -z "${GITHUB_BASE_REF:-}" && "${GITHUB_REF:-}" == "refs/heads/main" ) ]]; then
    echo "FAIL: a pre-release inference rev pin must not reach main; pin the released runtime tag." >&2
    exit 1
  fi
else
  echo "FAIL: scripts/git-deps-pinned.csv must record one runtime release tag or a pre-release 'rev' pin." >&2
  exit 1
fi

inference_git_line='git = "https://github.com/SceneWorks/inference"'
manifest_lines="$(grep -c "$inference_git_line" "$manifest" || true)"
manifest_refs="$(
  grep "$inference_git_line" "$manifest" \
    | sed -nE 's/.*(tag|rev|branch) = "([^"]+)".*/\1=\2/p' \
    | sort -u
)"
manifest_pinned="$(grep "$inference_git_line" "$manifest" | grep -cE '(tag|rev|branch) = "' || true)"
if [[ "$manifest_lines" -eq 0 || "$manifest_pinned" != "$manifest_lines" || "$manifest_refs" != "$manifest_expected" ]]; then
  echo "FAIL: every runtime bundle must use the recorded inference pin." >&2
  printf '       expected: %s\n' "$manifest_expected" >&2
  printf '       manifest: %s\n' "${manifest_refs:-"(none)"}" >&2
  exit 1
fi

lock_sources="$(grep -c '^source = "git+https://github.com/SceneWorks/inference' "$lock" || true)"
lock_refs="$(
  sed -nE 's/^source = "git\+https:\/\/github\.com\/SceneWorks\/inference\?(tag|rev)=([^#"]+)#([0-9a-f]{40})"$/\1=\2#\3/p' "$lock" \
    | sort -u
)"
lock_matched="$(grep -cE '^source = "git\+https://github\.com/SceneWorks/inference\?(tag|rev)=[^#"]+#[0-9a-f]{40}"$' "$lock" || true)"
if [[ "$lock_sources" -eq 0 ]]; then
  echo "FAIL: Cargo.lock contains no packages from SceneWorks/inference." >&2
  exit 1
fi
if [[ "$lock_matched" != "$lock_sources" || "$lock_refs" != "$lock_expected" ]]; then
  echo "FAIL: Cargo.lock must resolve every inference package to the recorded pin and commit." >&2
  printf '       expected: %s\n' "$lock_expected" >&2
  printf '       resolved: %s\n' "${lock_refs:-"(none)"}" >&2
  exit 1
fi

legacy_pattern='github\.com/(SceneWorks/(core-llm|mlx-llm|candle-llm)|michaeltrefry/(mlx-gen|candle-gen))'
if grep -Eq "$legacy_pattern" "$manifest" "$lock"; then
  echo "FAIL: an individual legacy inference repository remains in the manifest or lockfile." >&2
  grep -En "$legacy_pattern" "$manifest" "$lock" >&2
  exit 1
fi

if [[ "$pin_kind" == "tag" ]]; then
  printf 'Supply-chain gate OK: %s inference packages use %s at %s.\n' \
    "$lock_sources" "$expected_ref" "${expected_sha:0:12}"
else
  printf 'Supply-chain gate OK: %s inference packages use pre-release rev %s (feature train; not accepted on main).\n' \
    "$lock_sources" "${expected_sha:0:12}"
fi
