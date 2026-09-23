import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const verifier = fileURLToPath(new URL('../scripts/verify-tauri-package.sh', import.meta.url));
const bashVerifier = process.platform === 'win32'
  ? `/${verifier[0].toLowerCase()}${verifier.slice(2).replaceAll('\\', '/')}`
  : verifier;

function verifyWindowsListing(listing) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'chatworks-package-verifier-'));
  try {
    const bin = path.join(root, 'bin');
    const nsis = path.join(root, 'release', 'bundle', 'nsis');
    fs.mkdirSync(bin);
    fs.mkdirSync(nsis, { recursive: true });
    fs.writeFileSync(path.join(nsis, 'ChatWorks-setup.exe'), 'fixture');
    fs.writeFileSync(path.join(bin, '7z'), '#!/usr/bin/env bash\nprintf "%s" "$FAKE_7Z_LISTING"\n', { mode: 0o755 });
    return spawnSync('bash', [bashVerifier], {
      encoding: 'utf8',
      env: {
        ...process.env,
        PATH: `${bin}${path.delimiter}${process.env.PATH ?? ''}`,
        CARGO_TARGET_DIR: root,
        MEDIA_SIDECAR_TARGET: 'x86_64-pc-windows-msvc',
        CHATWORKS_PACKAGE_BACKEND: 'cpu',
        FAKE_7Z_LISTING: listing,
      },
    });
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
}

const row = (name) => `2026-09-23 05:19:00 ....A 123456 78901 ${name}\r\n`;

test('Windows NSIS listing accepts bare and nested sidecar names with CRLF', () => {
  const result = verifyWindowsListing(row('ffmpeg.exe') + row('$INSTDIR\\ffprobe.EXE'));
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /Verified packaged FFmpeg sidecars/);
});

test('Windows NSIS listing rejects missing and wrong-suffix sidecars', () => {
  for (const [listing, missing] of [
    [row('ffprobe.exe'), 'ffmpeg.exe'],
    [row('notffmpeg.exe') + row('ffprobe.exe'), 'ffmpeg.exe'],
    [row('ffmpeg.exe.bak') + row('ffprobe.exe'), 'ffmpeg.exe'],
    [row('ffmpeg.exe'), 'ffprobe.exe'],
    [row('ffmpeg.exe') + row('ffprobe.exe.manifest'), 'ffprobe.exe'],
  ]) {
    const result = verifyWindowsListing(listing);
    assert.equal(result.status, 1, `listing: ${listing}\n${result.stderr}`);
    assert.match(result.stderr, new RegExp(`Packaged application is missing ${missing.replaceAll('.', '[.]')}`));
  }
});
