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
  assert.match(job, /name: Configure the Windows compiler\n        uses: ilammy\/msvc-dev-cmd@v1\n        with:\n          arch: x64\n          vsversion: '2022'/);
  assert.match(job, /call "%VCVARS%"\n          if errorlevel 1 exit \/b 1\n          if \/i not "%VisualStudioVersion%"=="17\.0"/);
}

function validateCudaSevenZip(source) {
  const job = source.split('  windows-cuda-package:')[1];
  const selection = job?.split('      - name: Select 7-Zip for CUDA package verification\n')[1]
    ?.split('      - name: Configure the Windows compiler\n')[0];
  assert.ok(selection, '7-Zip selection must precede CUDA packaging');
  for (const required of [
    "'/c/Program Files/7-Zip/7z.exe'",
    "'/c/Program Files (x86)/7-Zip/7z.exe'",
    'pacman --noconfirm --needed -S mingw-w64-x86_64-7zip',
    'seven_zip=/mingw64/bin/7z.exe',
    '$(cygpath -w "$(dirname "$seven_zip")")',
    '$(cygpath -u "$GITHUB_PATH")',
  ]) assert.ok(selection.includes(required), `missing 7-Zip setup: ${required}`);
  const verification = job.split('      - name: Verify packaged FFmpeg and CUDA runtime\n')[1]
    ?.split('      - name: Preserve exact CUDA package and source identity\n')[0];
  assert.ok(verification?.includes("execFileSync('7z', ['i']"), 'native Node must resolve 7z from the shared PATH');
  assert.ok(verification.includes('bash scripts/verify-tauri-package.sh'), 'shell verifier must use the same PATH');
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

test('CUDA job gives shell and native Node a verified 7-Zip executable', () => {
  validateCudaSevenZip(workflow);
  for (const [before, after] of [
    ['pacman --noconfirm --needed -S mingw-w64-x86_64-7zip', 'echo installer omitted'],
    ['$(cygpath -u "$GITHUB_PATH")', 'omitted-path-file'],
    ["execFileSync('7z', ['i']", "execFileSync('missing-7z', ['i']"],
  ]) {
    assert.ok(workflow.includes(before));
    assert.throws(() => validateCudaSevenZip(workflow.replace(before, after)));
  }
});

test('workflow guards reject missing defaults, pwsh and missing Git Bash bootstrap', () => {
  for (const [before, after] of [
    ['defaults:\n      run:\n        shell: powershell', 'defaults:\n      run:\n        shell: pwsh'],
    ['name: Select isolated CUDA build directory\n        shell: powershell', 'name: Select isolated CUDA build directory\n        shell: pwsh'],
    ['echo C:\\Program Files\\Git\\bin>>"%GITHUB_PATH%"', 'echo bootstrap omitted'],
    ["vsversion: '2022'", "vsversion: '2026'"],
    ['if /i not "%VisualStudioVersion%"=="17.0"', 'rem compiler guard omitted'],
  ]) {
    assert.ok(workflow.includes(before));
    assert.throws(() => validateSelfHostedShells(workflow.replace(before, after)));
  }
});
