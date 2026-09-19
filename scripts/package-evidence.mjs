import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import os from 'node:os';
import { execFileSync } from 'node:child_process';
import { pathToFileURL } from 'node:url';

export const cudaLibraries = ['cudart64_12.dll', 'cublas64_12.dll', 'cublasLt64_12.dll',
  'curand64_10.dll', 'nvrtc64_120_0.dll', 'nvrtc-builtins64_129.dll'];
const hash = file => {
  const digest = crypto.createHash('sha256');
  const fd = fs.openSync(file, 'r');
  const buffer = Buffer.alloc(1024 * 1024);
  try {
    let length;
    while ((length = fs.readSync(fd, buffer, 0, buffer.length, null)) > 0) digest.update(buffer.subarray(0, length));
  } finally { fs.closeSync(fd); }
  return digest.digest('hex');
};
const row = file => ({ file: path.basename(file), bytes: fs.statSync(file).size, sha256: hash(file) });

export function stageCuda(root, toolkit, output) {
  const version = JSON.parse(fs.readFileSync(path.join(toolkit, 'version.json'), 'utf8'));
  if (!version.cuda?.version?.startsWith('12.9.')) throw new Error('CUDA Toolkit 12.9 is required');
  fs.mkdirSync(output, { recursive: true });
  for (const name of cudaLibraries) {
    const source = path.join(toolkit, 'bin', name);
    if (!fs.statSync(source).isFile()) throw new Error(`Missing CUDA runtime ${name}`);
    fs.copyFileSync(source, path.join(output, name));
  }
  const license = ['EULA.txt', 'doc/EULA.txt'].map(relative => path.join(toolkit, relative)).find(file => fs.existsSync(file));
  if (!license) throw new Error('CUDA Toolkit EULA.txt is required for redistribution');
  fs.copyFileSync(license, path.join(output, 'CUDA-EULA.txt'));
  fs.copyFileSync(path.join(root, 'scripts/CUDA-NOTICE.txt'), path.join(output, 'CUDA-NOTICE.txt'));
  fs.writeFileSync(path.join(output, 'cuda-runtime.json'), JSON.stringify({ toolkit: version.cuda.version,
    files: cudaLibraries.map(name => row(path.join(output, name))),
    license: row(path.join(output, 'CUDA-EULA.txt')) }, null, 2));
  const resources = Object.fromEntries([...cudaLibraries, 'CUDA-NOTICE.txt', 'CUDA-EULA.txt', 'cuda-runtime.json']
    .map(name => [path.join(output, name).replaceAll('\\', '/'), name]));
  const config = path.join(output, 'tauri.cuda.json');
  fs.writeFileSync(config, JSON.stringify({ bundle: { resources } }, null, 2));
  return config;
}

export function verifyCudaExtracted(directory, receipt) {
  const entries = fs.readdirSync(directory, { recursive: true, withFileTypes: true });
  const files = entries.filter(entry => entry.isFile()).map(entry => path.join(entry.parentPath, entry.name));
  const apps = files.filter(file => path.basename(file).toLowerCase() === 'chatworks.exe');
  if (apps.length !== 1) throw new Error('CUDA installer must contain exactly one chatworks.exe');
  const installRoot = path.dirname(apps[0]);
  if (receipt.files.length !== cudaLibraries.length) throw new Error('CUDA receipt is incomplete');
  for (const name of cudaLibraries) {
    const expected = receipt.files.find(item => item.file === name);
    const file = path.join(installRoot, name);
    if (!expected || !fs.existsSync(file) || row(file).sha256 !== expected.sha256) {
      throw new Error(`CUDA installer runtime mismatch: ${name}`);
    }
  }
  if (!fs.existsSync(path.join(installRoot, 'CUDA-NOTICE.txt'))) throw new Error('Missing CUDA notice');
  if (!receipt.license || hash(path.join(installRoot, 'CUDA-EULA.txt')) !== receipt.license.sha256) {
    throw new Error('CUDA installer license mismatch');
  }
}

export function verifyCudaPackage(installer, stage) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'chatworks-cuda-package-'));
  try {
    execFileSync('7z', ['x', '-y', `-o${directory}`, installer], { stdio: 'pipe' });
    const receipt = JSON.parse(fs.readFileSync(path.join(stage, 'cuda-runtime.json'), 'utf8'));
    verifyCudaExtracted(directory, receipt);
  } finally { fs.rmSync(directory, { recursive: true, force: true }); }
}

export function collect(root, targetDir, output, target, backend, sha, cudaStage) {
  if (!/^[a-f0-9]{40}$/.test(sha)) throw new Error('Exact source SHA required');
  if (!['cpu', 'cuda', 'mlx'].includes(backend)) throw new Error('Explicit backend required');
  fs.mkdirSync(output, { recursive: true });
  const kind = target.includes('apple') ? 'macos' : target.includes('windows') ? 'nsis' : 'deb';
  const bundle = path.join(targetDir, 'release/bundle', kind);
  const suffix = { macos: '.app', nsis: '-setup.exe', deb: '.deb' }[kind];
  const files = fs.readdirSync(bundle).filter(name => name.endsWith(suffix));
  if (files.length !== 1) throw new Error(`Expected exactly one ${kind} package; found ${files.length}`);
  const source = path.join(bundle, files[0]);
  const destination = path.join(output, `chatworks-${target}-${backend}-${files[0]}${kind === 'macos' ? '.tar.gz' : ''}`);
  if (kind === 'macos') execFileSync('tar', ['-czf', destination, '-C', bundle, files[0]]);
  else fs.copyFileSync(source, destination);
  const pins = {};
  for (const relative of ['Cargo.lock', 'src-tauri/Cargo.toml', 'scripts/git-deps-pinned.csv']) {
    const file = path.join(root, relative);
    pins[relative] = hash(file);
    fs.copyFileSync(file, path.join(output, path.basename(relative)));
  }
  const receipt = { schema_version: 1, source_sha: sha, target, backend, signed: false,
    installed_app_acceptance: false, package: row(destination), pin_files_sha256: pins };
  if (backend === 'cuda') {
    if (!cudaStage) throw new Error('CUDA runtime evidence required');
    const runtime = path.join(cudaStage, 'cuda-runtime.json');
    receipt.cuda_runtime = JSON.parse(fs.readFileSync(runtime, 'utf8'));
    fs.copyFileSync(runtime, path.join(output, 'cuda-runtime.json'));
  }
  fs.writeFileSync(path.join(output, 'package-evidence.json'), JSON.stringify(receipt, null, 2));
  fs.writeFileSync(path.join(output, 'SHA256SUMS'), `${receipt.package.sha256}  ${receipt.package.file}\n`);
  return receipt;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const [command, ...args] = process.argv.slice(2);
  if (command === 'stage-cuda') stageCuda(...args);
  else if (command === 'collect') collect(...args);
  else if (command === 'verify-cuda') verifyCudaPackage(...args);
  else throw new Error('Expected stage-cuda or collect');
}
