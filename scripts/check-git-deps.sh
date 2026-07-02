#!/usr/bin/env bash
# Supply-chain gate for the core inference git dependencies (code-review F-001, epic 9318).
#
# The three inference crates (core-llm, mlx-llm, candle-llm) are git deps. Cargo keys a git source by
# its (url, ref-spec), so for the link-time provider registry to UNIFY on a single core-llm, every
# consumer must use the SAME ref-spec form. The upstream backends (mlx-llm, candle-llm) currently pin
# core-llm with `branch = "main"`, so this crate's direct core-llm dep MUST also use `branch = "main"`
# — a unilateral switch to `rev` here splits the graph into two core-llm sources and breaks the build
# (verified). Moving everyone to `rev` is a cross-repo coordination tracked separately.
#
# Until that lands, this gate enforces the trust model: the resolved commit for each core dep is
# pinned in Cargo.lock (the second layer), and this script fails if a core dep is NOT pinned in the
# lock to a concrete commit, so a lockfile refresh can't silently float a core dep to a new HEAD
# without showing up here. Run locally and in CI.
#
# Usage: scripts/check-git-deps.sh
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
lock="$root/Cargo.lock"
manifest="$root/src-tauri/Cargo.toml"

if [[ ! -f "$lock" ]]; then
  echo "error: $lock not found (run from repo root or after a build)" >&2
  exit 1
fi

# The core inference deps whose resolved commits must be lock-pinned.
core_deps=("core-llm" "mlx-llm" "candle-llm")

status=0
for dep in "${core_deps[@]}"; do
  # Each lock entry looks like:
  #   name = "core-llm"
  #   version = "0.0.0"
  #   source = "git+https://...?branch=main#<40-hex-sha>"
  block="$(awk -v dep="$dep" '
    /^name = "/ {
      name=$0; sub(/^name = "/,"",name); sub(/"$/,"",name);
      if (name==dep) found=1; else found=0
    }
    found && /^source = / { print; found=0 }
  ' "$lock")"

  if [[ -z "$block" ]]; then
    echo "FAIL: $dep is not present in Cargo.lock — the lockfile is stale." >&2
    status=1
    continue
  fi

  # Extract the resolved commit (the #<sha> at the end of the source line). BSD sed (macOS) and GNU
  # sed both handle the basic regex below; avoid `\?` etc.
  commit="$(printf '%s' "$block" | sed -nE 's/.*#([0-9a-f]{40}).*/\1/p')"
  if [[ -z "$commit" ]]; then
    echo "FAIL: $dep has no resolved commit pinned in Cargo.lock source line:" >&2
    printf '       %s\n' "$block" >&2
    status=1
    continue
  fi

  # Confirm the manifest still references this dep (guards against a rename leaving a stale gate).
  if ! grep -q "^[[:space:]]*${dep} = { git =" "$manifest"; then
    echo "WARN: $dep is lock-pinned but not declared in $manifest (gate may be stale)." >&2
  fi

  # Report which ref-spec form is used (branch/rev/tag), for visibility.
  ref="fixed"
  if printf '%s' "$block" | grep -q 'branch='; then ref="branch"; fi
  if printf '%s' "$block" | grep -q 'rev='; then ref="rev"; fi
  if printf '%s' "$block" | grep -q 'tag='; then ref="tag"; fi
  printf 'OK:   %-12s %-6s %s\n' "$dep" "$ref" "${commit:0:12}"
done

if [[ $status -ne 0 ]]; then
  echo >&2
  echo "Supply-chain gate FAILED: a core inference dep is not lock-pinned to a concrete commit." >&2
  echo "Re-run \`cargo update -p <dep>\` intentionally and review the new commit before committing." >&2
  exit "$status"
fi

echo "Supply-chain gate OK: all core inference deps are lock-pinned to concrete commits."
