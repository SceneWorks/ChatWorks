#!/usr/bin/env bash
# Supply-chain gate for the core inference git dependencies (code-review F-001, epic 9318).
#
# Fails unless every core inference dep's RESOLVED commit in Cargo.lock exactly matches the expected
# SHA recorded in scripts/git-deps-pinned.csv. This is the meaningful gate: a `cargo update -p
# core-llm` that floats to a new upstream HEAD changes the lock's resolved commit and fails this
# check, so a lockfile refresh becomes a deliberate, reviewed bump (update the SHA in the pin file in
# the same PR). The previous version of this script only checked that a commit *existed* on the
# source line — a property every valid lockfile always has, so it was vacuous (PR #30 review, R5).
#
# Run locally and in CI (`.github/workflows/supply-chain.yml`). Usage: scripts/check-git-deps.sh
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
lock="$root/Cargo.lock"
pinfile="$root/scripts/git-deps-pinned.csv"

if [[ ! -f "$lock" ]]; then
  echo "error: $lock not found (run from repo root or after a build)" >&2
  exit 1
fi
if [[ ! -f "$pinfile" ]]; then
  echo "error: pin file $pinfile not found" >&2
  exit 1
fi

status=0

# Read each non-comment, non-empty pin line as `<crate>,<sha>`.
while IFS=, read -r crate expected_sha; do
  # Skip comment/blank lines.
  case "$crate" in "" | \#*) continue ;; esac

  # Extract the resolved source line from Cargo.lock for this crate.
  block="$(awk -v crate="$crate" '
    /^name = "/ {
      name=$0; sub(/^name = "/,"",name); sub(/"$/,"",name);
      if (name==crate) found=1; else found=0
    }
    found && /^source = / { print; found=0 }
  ' "$lock")"

  if [[ -z "$block" ]]; then
    echo "FAIL: $crate is not present in Cargo.lock — the lockfile is stale." >&2
    status=1
    continue
  fi

  # The resolved commit is the #<sha> at the end of the source line.
  resolved_sha="$(printf '%s' "$block" | sed -nE 's/.*#([0-9a-f]{40}).*/\1/p')"
  if [[ -z "$resolved_sha" ]]; then
    echo "FAIL: $crate has no resolved commit in its Cargo.lock source line:" >&2
    printf '       %s\n' "$block" >&2
    status=1
    continue
  fi

  if [[ "$resolved_sha" != "$expected_sha" ]]; then
    echo "FAIL: $crate resolved commit drift." >&2
    printf '       expected  %s (from %s)\n' "$expected_sha" "scripts/git-deps-pinned.csv" >&2
    printf '       resolved  %s (from Cargo.lock)\n' "$resolved_sha" >&2
    echo "       A lockfile refresh floated $crate to a new HEAD. To accept this intentionally," >&2
    echo "       verify the new commit and update the SHA in scripts/git-deps-pinned.csv." >&2
    status=1
    continue
  fi

  printf 'OK:   %-12s %s\n' "$crate" "${resolved_sha:0:12}"
done < "$pinfile"

if [[ $status -ne 0 ]]; then
  echo >&2
  echo "Supply-chain gate FAILED: a core inference dep does not match its pinned commit." >&2
  exit "$status"
fi

echo "Supply-chain gate OK: all core inference deps match their pinned commits."
