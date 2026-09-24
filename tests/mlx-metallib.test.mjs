import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { execFileSync } from 'node:child_process';
import { collect, stageMlxMetallib, verifyMlxMetallib } from '../scripts/package-evidence.mjs';

const revision = 'd5a7fc018d713a37091e1cd102873eab355a00c6';

function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'chatworks-mlx-package-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  for (const [relative, body] of Object.entries({
    'Cargo.lock': `[[package]]\nname = "pmetal-mlx-sys"\nsource = "git+https://github.com/michaeltrefry/mlx-rs?rev=${revision}#${revision}"\n`,
    'src-tauri/Cargo.toml': 'pinned app',
    'scripts/git-deps-pinned.csv': 'pinned inference',
  })) {
    const file = path.join(root, relative);
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, body);
  }
  execFileSync('git', ['init', '-q', root]);
  execFileSync('git', ['-C', root, 'add', '.']);
  execFileSync('git', ['-C', root, '-c', 'user.name=Package Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture']);
  const sha = execFileSync('git', ['-C', root, 'rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();
  const build = path.join(root, 'build');
  const app = path.join(build, 'release/bundle/macos/ChatWorks.app');
  const binary = path.join(app, 'Contents/MacOS/chatworks');
  const output = path.join(build, 'release/build/pmetal-mlx-sys-aabbccdd/output');
  const out = path.dirname(output);
  const source = path.join(out, 'out/build/lib/mlx.metallib');
  const compiled = path.join(out, 'out/build/_deps/mlx-build/mlx/backend/metal/kernels/mlx.metallib');
  const manifest = path.join(path.dirname(source), 'pmetal-mlx-prebuilt.txt');
  fs.mkdirSync(path.dirname(binary), { recursive: true });
  fs.mkdirSync(path.dirname(source), { recursive: true });
  fs.mkdirSync(path.dirname(compiled), { recursive: true });
  fs.writeFileSync(binary, `Mach-O fixture: ${path.dirname(compiled)}//mlx.metallib`);
  fs.writeFileSync(source, 'same build Metal kernels');
  fs.writeFileSync(compiled, 'same build Metal kernels');
  fs.writeFileSync(output, `cargo:metallib=${source}\n`);
  fs.writeFileSync(manifest, `schema=1\ntarget=aarch64-apple-darwin\ndeployment_target=26.2\nbuild_type=Release\nfeatures=accelerate,metal\nfingerprint=${'a'.repeat(64)}\n`);
  return { root, build, app, source, compiled, manifest, binary, sha };
}

test('MLX package binds the exact executable-linked build and records the bundled hash', { skip: process.platform === 'win32' }, t => {
  const f = fixture(t);
  const staged = stageMlxMetallib(f.root, f.build);
  assert.equal(staged.build_manifest.deployment_target, '26.2');
  assert.equal(staged.mlx_rs_revision, revision);
  assert.deepEqual(verifyMlxMetallib(f.app, f.build, f.root), staged);
  const receipt = collect(f.root, f.build, path.join(f.root, 'evidence'), 'aarch64-apple-darwin', 'mlx', f.sha);
  assert.deepEqual(receipt.mlx_metallib, staged);
  assert.ok(fs.existsSync(path.join(f.root, 'evidence/mlx-metallib-provenance.json')));
  const resource = path.join(f.app, 'Contents/Resources/mlx.metallib');
  fs.writeFileSync(resource, 'wrong Metal kernels');
  assert.throws(() => verifyMlxMetallib(f.app, f.build, f.root), /bytes or SHA-256/);
  fs.rmSync(resource);
  assert.throws(() => verifyMlxMetallib(f.app, f.build, f.root), /exactly one regular/);
  const wrongLocation = path.join(f.app, 'Contents/MacOS/mlx.metallib');
  fs.writeFileSync(wrongLocation, 'same build Metal kernels');
  assert.throws(() => verifyMlxMetallib(f.app, f.build, f.root), /exactly one regular/);
  fs.rmSync(wrongLocation);
  fs.symlinkSync(f.source, resource);
  assert.throws(() => verifyMlxMetallib(f.app, f.build, f.root), /exactly one regular/);
});

test('MLX staging rejects stale, nonmatching, or wrong-target build outputs', { skip: process.platform === 'win32' }, t => {
  const f = fixture(t);
  fs.writeFileSync(f.binary, 'app without embedded MLX build path');
  assert.throws(() => stageMlxMetallib(f.root, f.build), /found 0/);
  fs.writeFileSync(f.binary, `Mach-O fixture: ${path.dirname(f.compiled)}//mlx.metallib`);
  fs.writeFileSync(f.source, 'different kernels');
  assert.throws(() => stageMlxMetallib(f.root, f.build), /differs/);
  fs.writeFileSync(f.source, 'same build Metal kernels');
  fs.writeFileSync(f.manifest, fs.readFileSync(f.manifest, 'utf8').replace('deployment_target=26.2', 'deployment_target=14.0'));
  assert.throws(() => stageMlxMetallib(f.root, f.build), /release contract/);
});
