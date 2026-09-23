# FFmpeg sidecar notice

ChatWorks bundles `ffmpeg` and `ffprobe` built by
[`scripts/provision-ffmpeg-sidecars.sh`](../../scripts/provision-ffmpeg-sidecars.sh).

- Upstream source: <https://ffmpeg.org/releases/ffmpeg-9.0.tar.xz>
- SHA-256: `7f607a00dd0d28a729d5a4811205812eef01cf6ef6155025febb6f36a9062d52`
- License: GNU Lesser General Public License, version 2.1 or later
- Included programs: `ffmpeg`, `ffprobe`

The provisioner disables GPL components and external libraries, and enables only the container
and video codecs needed to inspect and sample local/API video sources. The complete LGPL text is
in [`COPYING.LGPLv2.1`](COPYING.LGPLv2.1). The source and build recipe remain available so the
sidecars can be replaced or relinked under the LGPL.

Provisioning is performed on the target release runner: macOS needs Xcode command-line tools,
Linux needs a C compiler and GNU make, and Windows needs Visual Studio Build Tools plus GNU make
from MSYS2/Git Bash. The recipe disables x86 assembly and external libraries, so it does not need
Nasm or a package-manager FFmpeg. A target mismatch is rejected rather than cross-compiling an
unreviewed sidecar.
