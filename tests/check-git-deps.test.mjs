// The inference pin gate (sc-24139): a released runtime tag anywhere, a pre-release `rev` pin only
// off main, and every bundle and lockfile entry on the one recorded pin.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const gate = fileURLToPath(new URL('../scripts/check-git-deps.sh', import.meta.url));
const SHA = 'dbda066a079173380e544b690cfa455dee9b77b4';
const OLD = '431b4e19c4651ad2e56d1e3dacf999678eac23a2';
const GIT = 'https://github.com/SceneWorks/inference';

function manifest(refs) {
  return refs.map((ref, index) => `runtime-${index} = { git = "${GIT}", ${ref}, default-features = false }\n`).join('');
}

function lock(sources) {
  return sources.map((source, index) => `[[package]]\nname = "p${index}"\nversion = "0.0.0"\nsource = "git+${GIT}?${source}"\n\n`).join('');
}

function run({ pin, refs, sources, env = {} }) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'chatworks-pin-gate-'));
  try {
    fs.mkdirSync(path.join(root, 'scripts'));
    fs.mkdirSync(path.join(root, 'src-tauri'));
    fs.copyFileSync(gate, path.join(root, 'scripts', 'check-git-deps.sh'));
    fs.writeFileSync(path.join(root, 'scripts', 'git-deps-pinned.csv'), `# pin\n${pin}\n`);
    fs.writeFileSync(path.join(root, 'src-tauri', 'Cargo.toml'), manifest(refs));
    fs.writeFileSync(path.join(root, 'Cargo.lock'), lock(sources));
    const script = path.join(root, 'scripts', 'check-git-deps.sh');
    const bashScript = process.platform === 'win32'
      ? `/${script[0].toLowerCase()}${script.slice(2).replaceAll('\\', '/')}`
      : script;
    const clean = { ...process.env };
    delete clean.GITHUB_BASE_REF;
    delete clean.GITHUB_REF;
    return spawnSync('bash', [bashScript], { encoding: 'utf8', env: { ...clean, ...env } });
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
}

const revPin = { pin: `SceneWorks/inference,rev,${SHA}`, refs: Array(3).fill(`rev = "${SHA}"`), sources: Array(4).fill(`rev=${SHA}#${SHA}`) };
const tagPin = {
  pin: `SceneWorks/inference,runtime-2026.09.0,${OLD}`,
  refs: Array(3).fill('tag = "runtime-2026.09.0"'),
  sources: Array(4).fill(`tag=runtime-2026.09.0#${OLD}`),
};

test('a released tag pin passes everywhere, including main', () => {
  assert.equal(run(tagPin).status, 0);
  assert.equal(run({ ...tagPin, env: { GITHUB_BASE_REF: 'main' } }).status, 0);
});

test('a pre-release rev pin passes on the feature train and fails into or on main', () => {
  const feature = run({ ...revPin, env: { GITHUB_BASE_REF: 'feature/sc-24128-fast-decode-blackwell' } });
  assert.equal(feature.status, 0, feature.stderr);
  assert.match(feature.stdout, /pre-release rev dbda066a0791/);
  const pr = run({ ...revPin, env: { GITHUB_BASE_REF: 'main' } });
  assert.equal(pr.status, 1);
  assert.match(pr.stderr, /must not reach main/);
  assert.equal(run({ ...revPin, env: { GITHUB_REF: 'refs/heads/main' } }).status, 1);
});

test('every bundle and every locked package must use the one recorded pin', () => {
  const mixedManifest = run({ ...revPin, refs: [`rev = "${SHA}"`, 'tag = "runtime-2026.09.0"', `rev = "${SHA}"`] });
  assert.equal(mixedManifest.status, 1);
  const unpinned = run({ ...revPin, refs: [`rev = "${SHA}"`, 'branch = "main"', `rev = "${SHA}"`] });
  assert.equal(unpinned.status, 1);
  const staleLock = run({ ...revPin, sources: [`rev=${SHA}#${SHA}`, `tag=runtime-2026.09.0#${OLD}`] });
  assert.equal(staleLock.status, 1);
  assert.match(staleLock.stderr, /Cargo\.lock must resolve/);
  const wrongCommit = run({ ...revPin, sources: Array(4).fill(`rev=${OLD}#${OLD}`) });
  assert.equal(wrongCommit.status, 1);
  assert.equal(run({ ...revPin, pin: `SceneWorks/inference,feature-branch,${SHA}` }).status, 1);
});
