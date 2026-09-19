import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import crypto from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { collect, stageCuda, cudaLibraries, verifyCudaExtracted } from '../scripts/package-evidence.mjs';

function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'chatworks-package-test-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  for (const file of ['Cargo.lock', 'src-tauri/Cargo.toml', 'scripts/git-deps-pinned.csv', 'scripts/CUDA-NOTICE.txt']) {
    fs.mkdirSync(path.dirname(path.join(root, file)), { recursive: true });
    fs.writeFileSync(path.join(root, file), file);
  }
  return root;
}

test('retained installer hashes bind exact bytes, source and backend; ambiguous packages fail', t => {
  const root = fixture(t), build = path.join(root, 'build'), out = path.join(root, 'evidence');
  const packages = path.join(build, 'release/bundle/nsis'); fs.mkdirSync(packages, { recursive: true });
  fs.writeFileSync(path.join(packages, 'ChatWorks-setup.exe'), 'installer bytes');
  const result = collect(root, build, out, 'x86_64-pc-windows-msvc', 'cpu', 'a'.repeat(40));
  assert.equal(result.backend, 'cpu'); assert.equal(result.installed_app_acceptance, false);
  assert.equal(result.package.sha256, crypto.createHash('sha256').update('installer bytes').digest('hex'));
  assert.match(result.package.file, /-cpu-/);
  assert.ok(fs.existsSync(path.join(out, 'Cargo.lock')));
  fs.writeFileSync(path.join(packages, 'stale-setup.exe'), 'stale');
  assert.throws(() => collect(root, build, out, 'x86_64-pc-windows-msvc', 'cpu', 'a'.repeat(40)), /exactly one/);
});

test('CUDA staging and installer verification reject missing, corrupted and misplaced runtime DLLs', t => {
  const root = fixture(t), toolkit = path.join(root, 'toolkit'), stage = path.join(root, 'stage');
  fs.mkdirSync(path.join(toolkit, 'bin'), { recursive: true });
  fs.writeFileSync(path.join(toolkit, 'version.json'), JSON.stringify({ cuda: { version: '12.9.1' } }));
  fs.writeFileSync(path.join(toolkit, 'EULA.txt'), 'toolkit license fixture');
  for (const name of cudaLibraries) fs.writeFileSync(path.join(toolkit, 'bin', name), name);
  const config = JSON.parse(fs.readFileSync(stageCuda(root, toolkit, stage), 'utf8'));
  assert.deepEqual(Object.values(config.bundle.resources).filter(name => name.endsWith('.dll')), cudaLibraries);
  assert.ok(!Object.values(config.bundle.resources).includes('nvcuda.dll'));
  const receipt = JSON.parse(fs.readFileSync(path.join(stage, 'cuda-runtime.json'), 'utf8'));
  fs.writeFileSync(path.join(stage, 'chatworks.exe'), 'app');
  verifyCudaExtracted(stage, receipt);
  fs.writeFileSync(path.join(stage, cudaLibraries[0]), 'corrupt');
  assert.throws(() => verifyCudaExtracted(stage, receipt), /runtime mismatch/);
  fs.copyFileSync(path.join(toolkit, 'bin', cudaLibraries[0]), path.join(stage, cudaLibraries[0]));
  fs.mkdirSync(path.join(stage, 'nested'));
  fs.renameSync(path.join(stage, cudaLibraries[0]), path.join(stage, 'nested', cudaLibraries[0]));
  assert.throws(() => verifyCudaExtracted(stage, receipt), /runtime mismatch/);
  fs.writeFileSync(path.join(toolkit, 'version.json'), JSON.stringify({ cuda: { version: '13.0.0' } }));
  assert.throws(() => stageCuda(root, toolkit, stage), /12.9/);
});

test('macOS evidence archives preserve app permissions and symlinks', { skip: process.platform === 'win32' }, t => {
  const root = fixture(t), build = path.join(root, 'build'), out = path.join(root, 'evidence');
  const app = path.join(build, 'release/bundle/macos/ChatWorks.app/Contents/MacOS');
  fs.mkdirSync(app, { recursive: true }); fs.writeFileSync(path.join(app, 'chatworks'), 'binary', { mode: 0o755 });
  fs.symlinkSync('chatworks', path.join(app, 'alias'));
  const receipt = collect(root, build, out, 'aarch64-apple-darwin', 'mlx', 'b'.repeat(40));
  assert.match(receipt.package.file, /\.app\.tar\.gz$/);
  assert.ok(receipt.package.bytes > 0);
  const extracted = path.join(root, 'extracted'); fs.mkdirSync(extracted);
  execFileSync('tar', ['-xzf', path.join(out, receipt.package.file), '-C', extracted]);
  const binary = path.join(extracted, 'ChatWorks.app/Contents/MacOS/chatworks');
  assert.equal(fs.statSync(binary).mode & 0o777, 0o755);
  assert.equal(fs.readlinkSync(path.join(path.dirname(binary), 'alias')), 'chatworks');
});
