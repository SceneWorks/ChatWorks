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
const mlxResource = 'Contents/Resources/mlx.metallib';
const mlxProvenanceFile = 'mlx-metallib-provenance.json';

function macosApp(targetDir) {
  const bundle = path.join(targetDir, 'release/bundle/macos');
  const apps = fs.readdirSync(bundle).filter(name => name.endsWith('.app'));
  if (apps.length !== 1) throw new Error(`Expected exactly one macOS app; found ${apps.length}`);
  return path.join(bundle, apps[0]);
}

function mlxRevision(root) {
  const lock = fs.readFileSync(path.join(root, 'Cargo.lock'), 'utf8');
  const block = lock.split('[[package]]').find(section => /\bname = "pmetal-mlx-sys"\n/.test(section));
  const source = block?.match(/^source = "([^"]+)"$/m)?.[1];
  const match = source?.match(/^git\+https:\/\/github\.com\/michaeltrefry\/mlx-rs\?rev=([a-f0-9]{40})#([a-f0-9]{40})$/);
  if (!match || match[1] !== match[2]) throw new Error('Cargo.lock lacks an exact pinned pmetal-mlx-sys revision');
  return match[1];
}

function mlxManifest(file) {
  const values = Object.fromEntries(fs.readFileSync(file, 'utf8').split(/\r?\n/)
    .filter(line => line.includes('=')).map(line => {
      const split = line.indexOf('=');
      return [line.slice(0, split), line.slice(split + 1)];
    }));
  if (values.schema !== '1' || values.target !== 'aarch64-apple-darwin'
      || values.deployment_target !== '26.2' || values.build_type !== 'Release'
      || values.features !== 'accelerate,metal' || !/^[a-f0-9]{64}$/.test(values.fingerprint ?? '')) {
    throw new Error('MLX build manifest does not match the Apple Silicon release contract');
  }
  return values;
}

function linkedMlxBuild(root, targetDir, app) {
  const executable = path.join(app, 'Contents/MacOS/chatworks');
  const binary = fs.readFileSync(executable);
  const buildDir = path.join(targetDir, 'release/build');
  const matches = [];
  for (const name of fs.readdirSync(buildDir).filter(name => /^pmetal-mlx-sys-[a-f0-9]+$/.test(name))) {
    const output = path.join(buildDir, name, 'output');
    if (!fs.existsSync(output)) continue;
    const sources = [...fs.readFileSync(output, 'utf8').matchAll(/^cargo:metallib=(.+)$/gm)];
    if (sources.length !== 1) continue;
    const source = sources[0][1].trim();
    const expected = path.join(buildDir, name, 'out/build/lib/mlx.metallib');
    if (source !== expected || !fs.existsSync(source) || !fs.statSync(source).isFile()) continue;
    const kernels = path.join(buildDir, name, 'out/build/_deps/mlx-build/mlx/backend/metal/kernels');
    if (![`${kernels}/mlx.metallib`, `${kernels}//mlx.metallib`]
      .some(value => binary.includes(Buffer.from(value)))) continue;
    matches.push({ output, source, compiled: path.join(kernels, 'mlx.metallib') });
  }
  if (matches.length !== 1) throw new Error(`Expected one linked MLX metallib build; found ${matches.length}`);
  const { output, source, compiled } = matches[0];
  if (!fs.existsSync(compiled) || hash(source) !== hash(compiled)) {
    throw new Error('MLX metadata library differs from the executable-linked Metal kernels');
  }
  const manifestFile = path.join(path.dirname(source), 'pmetal-mlx-prebuilt.txt');
  const manifest = mlxManifest(manifestFile);
  return { output, source, manifestFile, manifest, revision: mlxRevision(root) };
}

export function stageMlxMetallib(root, targetDir) {
  const app = macosApp(targetDir);
  const { output, source, manifestFile, manifest, revision } = linkedMlxBuild(root, targetDir, app);
  const destination = path.join(app, mlxResource);
  fs.mkdirSync(path.dirname(destination), { recursive: true });
  fs.copyFileSync(source, destination);
  const provenance = {
    schema_version: 1,
    mlx_rs_revision: revision,
    build_script_output: path.relative(targetDir, output),
    build_manifest: manifest,
    build_manifest_sha256: hash(manifestFile),
    metallib: { ...row(destination), file: mlxResource },
  };
  fs.writeFileSync(path.join(targetDir, 'release', mlxProvenanceFile), JSON.stringify(provenance, null, 2));
  return provenance;
}

export function verifyMlxMetallib(app, targetDir, root) {
  if (!root) throw new Error('MLX package verification requires the pinned source root');
  const provenance = JSON.parse(fs.readFileSync(path.join(targetDir, 'release', mlxProvenanceFile), 'utf8'));
  const resource = path.join(app, mlxResource);
  const matches = fs.readdirSync(app, { recursive: true, withFileTypes: true })
    .filter(entry => entry.name === 'mlx.metallib');
  if (matches.length !== 1 || !fs.existsSync(resource) || !fs.lstatSync(resource).isFile()) {
    throw new Error('macOS MLX package must contain exactly one regular Contents/Resources/mlx.metallib');
  }
  const actual = { ...row(resource), file: mlxResource };
  if (JSON.stringify(actual) !== JSON.stringify(provenance.metallib)) {
    throw new Error('packaged MLX metallib bytes or SHA-256 differ from build provenance');
  }
  const linked = linkedMlxBuild(root, targetDir, app);
  if (provenance.schema_version !== 1 || provenance.mlx_rs_revision !== linked.revision
      || provenance.build_script_output !== path.relative(targetDir, linked.output)
      || provenance.build_manifest_sha256 !== hash(linked.manifestFile)
      || JSON.stringify(provenance.build_manifest) !== JSON.stringify(linked.manifest)
      || actual.sha256 !== hash(linked.source)) {
    throw new Error('packaged MLX metallib provenance differs from the executable-linked build');
  }
  return provenance;
}

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
    license: row(path.join(output, 'CUDA-EULA.txt')),
    notice: row(path.join(output, 'CUDA-NOTICE.txt')) }, null, 2));
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
  if (!receipt.notice || hash(path.join(installRoot, 'CUDA-NOTICE.txt')) !== receipt.notice.sha256) {
    throw new Error('CUDA installer notice mismatch');
  }
  const packagedReceipt = JSON.parse(fs.readFileSync(path.join(installRoot, 'cuda-runtime.json'), 'utf8'));
  if (JSON.stringify(packagedReceipt) !== JSON.stringify(receipt)) throw new Error('CUDA installer receipt mismatch');
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
  const git = args => execFileSync('git', ['-C', root, ...args], { encoding: 'utf8' }).trim();
  if (git(['rev-parse', 'HEAD']) !== sha) throw new Error('Source SHA differs from checked-out HEAD');
  if (git(['status', '--porcelain', '--untracked-files=no'])) throw new Error('Tracked source differs from committed identity');
  for (const relative of ['Cargo.lock', 'src-tauri/Cargo.toml', 'scripts/git-deps-pinned.csv']) {
    const committed = execFileSync('git', ['-C', root, 'show', `${sha}:${relative}`]);
    if (!committed.equals(fs.readFileSync(path.join(root, relative)))) throw new Error(`Uncommitted pin file: ${relative}`);
  }
  if (!['cpu', 'cuda', 'mlx'].includes(backend)) throw new Error('Explicit backend required');
  const expected = {
    'aarch64-apple-darwin': ['mlx'],
    'x86_64-apple-darwin': ['cpu'],
    'x86_64-unknown-linux-gnu': ['cpu'],
    'aarch64-unknown-linux-gnu': ['cpu'],
    'x86_64-pc-windows-msvc': ['cpu', 'cuda'],
  }[target];
  if (!expected?.includes(backend)) throw new Error(`Backend ${backend} is not valid for ${target}`);
  fs.mkdirSync(output, { recursive: true });
  const kind = target.includes('apple') ? 'macos' : target.includes('windows') ? 'nsis' : 'deb';
  const bundle = path.join(targetDir, 'release/bundle', kind);
  const suffix = { macos: '.app', nsis: '-setup.exe', deb: '.deb' }[kind];
  const files = fs.readdirSync(bundle).filter(name => name.endsWith(suffix));
  if (files.length !== 1) throw new Error(`Expected exactly one ${kind} package; found ${files.length}`);
  const source = path.join(bundle, files[0]);
  const mlxMetallib = backend === 'mlx' ? verifyMlxMetallib(source, targetDir, root) : null;
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
  if (mlxMetallib) {
    receipt.mlx_metallib = mlxMetallib;
    fs.copyFileSync(path.join(targetDir, 'release', mlxProvenanceFile), path.join(output, mlxProvenanceFile));
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
  else if (command === 'stage-mlx') stageMlxMetallib(...args);
  else if (command === 'verify-mlx') verifyMlxMetallib(...args);
  else throw new Error('Expected stage-cuda, verify-cuda, stage-mlx, verify-mlx, or collect');
}
