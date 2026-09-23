import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

const workflow = fs.readFileSync(new URL('../.github/workflows/package-validation.yml', import.meta.url), 'utf8');
function validateSelfHostedShells(source) {
  const job = source.split('  windows-cuda-package:')[1];
  assert.ok(job, 'CUDA packaging job exists');
  assert.match(job, /defaults:\n      run:\n        shell: powershell/);
  assert.doesNotMatch(job, /shell: pwsh/);
  const bootstrap = job.indexOf('echo C:\\Program Files\\Git\\bin>>"%GITHUB_PATH%"');
  assert.ok(bootstrap >= 0 && bootstrap < job.indexOf('uses: dtolnay/rust-toolchain'), 'Git Bash must precede the Rust action');
  assert.match(job, /if not exist "C:\\Program Files\\Git\\bin\\bash.exe" exit \/b 1/);
}

test('self-hosted Windows uses installed PowerShell and bootstraps Git Bash before Rust', () => {
  validateSelfHostedShells(workflow);
  const hosted = workflow.split('  windows-cuda-package:')[0];
  assert.match(hosted, /runner: windows-2025/);
  assert.match(hosted, /shell: pwsh/);
});

test('six package lanes label Intel macOS CPU and Apple Silicon MLX', () => {
  const hosted = workflow.split('  windows-cuda-package:')[0];
  assert.equal((hosted.match(/^          - runner:/gm) ?? []).length, 5);
  assert.match(hosted, /runner: macos-26\n            target: aarch64-apple-darwin\n            bundle: app\n            backend: mlx/);
  assert.match(hosted, /runner: macos-26-intel\n            target: x86_64-apple-darwin\n            bundle: app\n            backend: cpu/);
  assert.match(workflow, /  windows-cuda-package:\n/);
});

test('workflow guards reject missing defaults, pwsh and missing Git Bash bootstrap', () => {
  for (const [before, after] of [
    ['defaults:\n      run:\n        shell: powershell', 'defaults:\n      run:\n        shell: pwsh'],
    ['name: Select isolated CUDA build directory\n        shell: powershell', 'name: Select isolated CUDA build directory\n        shell: pwsh'],
    ['echo C:\\Program Files\\Git\\bin>>"%GITHUB_PATH%"', 'echo bootstrap omitted'],
  ]) {
    assert.ok(workflow.includes(before));
    assert.throws(() => validateSelfHostedShells(workflow.replace(before, after)));
  }
});
