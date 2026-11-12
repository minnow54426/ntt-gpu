#!/usr/bin/env python3
"""Install pinned NVIDIA compiler/runtime redistributables locally; no driver."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import tarfile
import tempfile
import urllib.request

BASE = 'https://developer.download.nvidia.com/compute/cuda/redist/'
ROOT = Path(__file__).resolve().parents[1]


def install(destination, cache):
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise RuntimeError('This lock file supports Linux x86_64 only; use your platform toolkit')
    lock = json.loads((ROOT / 'scripts/cuda-toolkit-lock.json').read_text())
    marker = destination / 'redist-record.json'
    if destination.exists():
        if marker.is_file() and json.loads(marker.read_text()) == lock and (destination / 'bin/nvcc').is_file():
            print(f'Using pinned toolkit: {destination}', flush=True)
            return
        raise RuntimeError(f'Destination already exists without the matching complete lock: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='cuda-install-', dir=destination.parent) as scratch:
        stage = Path(scratch) / 'toolkit'
        stage.mkdir()
        for name, component in lock['components'].items():
            archive = cache / Path(component['relative_path']).name
            if not archive.exists():
                print(f'Downloading {name}', flush=True)
                partial = archive.with_suffix(archive.suffix + '.partial')
                try:
                    with urllib.request.urlopen(BASE + component['relative_path'], timeout=60) as response, partial.open('wb') as output:
                        shutil.copyfileobj(response, output)
                    partial.replace(archive)
                finally:
                    partial.unlink(missing_ok=True)
            with archive.open('rb') as source:
                digest = hashlib.file_digest(source, 'sha256').hexdigest()
            if digest != component['sha256']:
                raise RuntimeError(f'Checksum mismatch: {archive}; remove this cache file before retrying')
            unpack = Path(scratch) / name
            with tarfile.open(archive) as package:
                package.extractall(unpack, filter='data')
            top = list(unpack.iterdir())
            if len(top) != 1 or not top[0].is_dir():
                raise RuntimeError(f'Unexpected archive layout: {archive}')
            for item in top[0].iterdir():
                if item.is_dir():
                    shutil.copytree(item, stage / item.name, dirs_exist_ok=True, symlinks=True)
                else:
                    notices = stage / 'component-notices' / name
                    notices.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, notices / item.name)
        if not (stage / 'lib64').exists():
            (stage / 'lib64').symlink_to('lib', target_is_directory=True)
        (stage / 'redist-record.json').write_text(json.dumps(lock, indent=2) + '\n')
        stage.replace(destination)
    print(f'Installed compiler-only toolkit: {destination}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=ROOT / 'build/toolchains/cuda-13.0.2')
    parser.add_argument('--cache', type=Path, default=ROOT / 'build/toolchains/downloads')
    args = parser.parse_args()
    install(args.destination.resolve(), args.cache.resolve())
