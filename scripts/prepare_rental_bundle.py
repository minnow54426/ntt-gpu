#!/usr/bin/env python3
"""Freeze source and CPU-prepared inputs; verify/restore/plan without GPU work."""
import argparse
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ('rtx-4080-super', 'rtx-4080', 'rtx-5090')
CELLS = tuple(product((4096, 65536), (3, 40), (30, 60)))
ROTATION_HEADER = b'NTT CKKS rotation benchmark v1\n'


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def run(command, cwd=None):
    return subprocess.run(list(map(str, command)), cwd=cwd, check=True,
                          capture_output=True, text=True).stdout.strip()


def regular(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('Require a regular file: ' + str(path))
    return path


def source_revision(repo):
    if run(['git', 'status', '--porcelain', '--untracked-files=all'], repo):
        raise ValueError('Commit source changes before freezing the rental bundle')
    return run(['git', 'rev-parse', 'HEAD'], repo)


def input_hashes(directory):
    result = {}
    for filename, suffix in (('vectors.json', '.txt'), ('conversion-references.json', '.conversion')):
        manifest = json.loads(regular(directory / filename).read_text())
        names = {f'benchmark-vector-n{n}-k{k}-b{b}{suffix}' for n, k, b in CELLS}
        if not isinstance(manifest, dict) or set(manifest) != names:
            raise ValueError(filename + ' must contain exactly the eight rental workloads')
        for name, checksum in manifest.items():
            if not isinstance(checksum, str) or not re.fullmatch('[0-9a-f]{64}', checksum):
                raise ValueError('Invalid input checksum: ' + name)
            if digest(regular(directory / name)) != checksum:
                raise ValueError('Missing or modified prepared input: ' + name)
            result[name] = checksum
        result[filename] = digest(directory / filename)
    return result


def plan_command(repo, bundle, target, build, output, cuda_root, openfhe_prefix):
    return [sys.executable, repo / 'scripts/gpu_rental_batch.py', '--mode', 'plan',
            '--target', target, '--prepared-vectors', bundle / 'prepared',
            '--conversion-references', bundle / 'prepared',
            '--rotation-fixture', bundle / 'rotation-benchmark.bin',
            '--extended-probes', '--defer-analysis', '--build-dir', build,
            '--output-dir', output, '--cuda-root', cuda_root,
            '--openfhe-prefix', openfhe_prefix]


def reviewed_plan(command, repo):
    plan = json.loads(run(command, repo))
    if plan.get('status') != 'PLANNED' or plan.get('rentals_created') is not False:
        raise ValueError('Unexpected rental plan scope')
    native_command = list(map(str, command))
    native_command[native_command.index('--mode') + 1] = 'gpu'
    plan['gpu_command_for_later'] = native_command
    plan['gpu_execution'] = 'NOT_RUN'
    return plan


def verify_bundle(directory):
    directory = directory.resolve()
    manifest = json.loads(regular(directory / 'manifest.json').read_text())
    if (manifest.get('schema') != 1 or manifest.get('status') != 'OFFLINE_HANDOFF_READY'
            or manifest.get('gpu_execution') != 'NOT_RUN' or manifest.get('target') not in TARGETS
            or not re.fullmatch('[0-9a-f]{40}', manifest.get('source_commit', ''))):
        raise ValueError('Invalid handoff manifest or execution scope')
    hashes = manifest.get('files')
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError('Missing handoff inventory')
    for name, checksum in hashes.items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or path.as_posix() != name:
            raise ValueError('Invalid inventory path: ' + name)
        if not isinstance(checksum, str) or not re.fullmatch('[0-9a-f]{64}', checksum):
            raise ValueError('Invalid inventory checksum: ' + name)
    entries = list(directory.rglob('*'))
    if any(p.is_symlink() or not (p.is_file() or p.is_dir()) for p in entries):
        raise ValueError('Handoff contains a link or special file')
    actual = {p.relative_to(directory).as_posix() for p in entries if p.is_file()}
    if actual != set(hashes) | {'manifest.json', 'SHA256SUMS'}:
        raise ValueError('Handoff file inventory changed')
    for name, checksum in hashes.items():
        if digest(regular(directory / name)) != checksum:
            raise ValueError('Handoff file changed: ' + name)
    expected_sums = ''.join(digest(directory / name) + '  ' + name + '\n'
                            for name in sorted(set(hashes) | {'manifest.json'}))
    if regular(directory / 'SHA256SUMS').read_text() != expected_sums:
        raise ValueError('Handoff checksums or manifest changed')
    heads = run(['git', 'bundle', 'list-heads', directory / 'source.bundle'])
    if manifest['source_commit'] + ' HEAD' not in heads.splitlines():
        raise ValueError('Source bundle does not match the manifest revision')
    input_hashes(directory / 'prepared')
    with (directory / 'rotation-benchmark.bin').open('rb') as stream:
        if stream.read(len(ROTATION_HEADER)) != ROTATION_HEADER:
            raise ValueError('Invalid rotation fixture header')
    return manifest


def create_bundle(repo, prepared, rotation, output, target, verification_dir=None):
    repo, prepared = repo.resolve(), prepared.resolve()
    if target not in TARGETS:
        raise ValueError('Unsupported GPU target')
    output = output.absolute()
    if output.resolve() != output:
        raise ValueError('Handoff path must not traverse a symlink')
    for path in (output, output.with_suffix('.tar.gz'), output.with_suffix('.tar.gz.sha256'),
                 output.with_suffix('.tar.gz.verify.json')):
        if path.exists() or path.is_symlink():
            raise ValueError('Use fresh paths; refusing to overwrite ' + str(path))
    if output.is_relative_to(repo):
        ignored = subprocess.run(['git', 'check-ignore', '--quiet', str(output) + '/'], cwd=repo)
        if ignored.returncode:
            raise ValueError('In-repository handoffs must use a Git-ignored directory')
    revision = source_revision(repo)
    inputs = input_hashes(prepared)
    regular(rotation)
    with rotation.open('rb') as stream:
        if stream.read(len(ROTATION_HEADER)) != ROTATION_HEADER:
            raise ValueError('Invalid rotation fixture header')
    rotation_hash = digest(rotation)
    verification = {}
    if verification_dir is not None:
        inventory = list(verification_dir.rglob('*'))
        if any(p.is_symlink() or not (p.is_file() or p.is_dir()) for p in inventory):
            raise ValueError('Verification records must contain regular files')
        verification = {p.relative_to(verification_dir): digest(p) for p in inventory if p.is_file()}
        if not verification:
            raise ValueError('Verification record directory is empty')
    # Read the helper from the commit, so the portable verifier is pinned too.
    helper = subprocess.run(['git', 'show', revision + ':scripts/prepare_rental_bundle.py'],
                            cwd=repo, check=True, capture_output=True).stdout
    output.mkdir(parents=True)
    (output / 'prepared').mkdir()
    for name, checksum in inputs.items():
        destination = output / 'prepared' / name
        shutil.copyfile(prepared / name, destination)
        if digest(destination) != checksum:
            raise ValueError('Prepared input changed during copy: ' + name)
    shutil.copyfile(rotation, output / 'rotation-benchmark.bin')
    if digest(output / 'rotation-benchmark.bin') != rotation_hash:
        raise ValueError('Rotation fixture changed during copy')
    for relative, checksum in verification.items():
        destination = output / 'verification' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(verification_dir / relative, destination)
        if digest(destination) != checksum:
            raise ValueError('Verification record changed during copy: ' + str(relative))
    run(['git', 'bundle', 'create', output / 'source.bundle', 'HEAD'], repo)
    run(['git', 'bundle', 'verify', output / 'source.bundle'], repo)
    (output / 'handoff.py').write_bytes(helper)
    command = plan_command(repo, output, target, output / 'future-build', output / 'future-results',
                           Path('/usr/local/cuda'), Path('/path/to/pinned-openfhe'))
    plan = reviewed_plan(command, repo)
    (output / 'creation-plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    (output / 'README.txt').write_text(
        'OFFLINE HANDOFF: no rental, GPU tests, timings or profiles have run.\n'
        'Source commit: ' + revision + '\nTarget: ' + target + '\n\n'
        'On the receiving machine, verify the external archive checksum before extraction.\n'
        'Then, from this extracted directory:\n'
        '  python3 handoff.py verify .\n'
        '  python3 handoff.py restore . /absolute/path/to/ntt-gpu\n'
        '  python3 handoff.py plan . /absolute/path/to/ntt-gpu\n\n'
        'The plan command validates inputs and prints commands; it runs no GPU work.\n'
        'creation-plan.json contains creation-host paths; regenerate the plan after restore.\n'
        'Build CUDA and pinned OpenFHE for the rental image before the paid session when possible.\n'
        'Do not reuse binaries under a different commit, path, architecture or library fingerprint.\n'
        'Run the small real CUDA/NVTX and SM/L2/DRAM access probes before the long GPU batch.\n'
        'Follow docs/next-gpu-rental.md from the restored checkout. Include --rotation-fixture,\n'
        '--extended-probes and --defer-analysis in the actual native command.\n'
        'Export and verify the result archive before stopping provider billing.\n')
    if source_revision(repo) != revision:
        raise ValueError('Source revision changed while creating the handoff')
    files = {p.relative_to(output).as_posix(): digest(p) for p in sorted(output.rglob('*')) if p.is_file()}
    manifest = dict(schema=1, status='OFFLINE_HANDOFF_READY', gpu_execution='NOT_RUN',
                    created_utc=datetime.now(timezone.utc).isoformat(), source_commit=revision,
                    target=target, files=files)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    from gpu_rental_batch import archive as archive_files
    archive = archive_files(output)
    verify_bundle(output)
    return dict(status='OFFLINE_HANDOFF_READY', source_commit=revision,
                archive=str(archive), sha256=digest(archive), files=len(files) + 2,
                gpu_execution='NOT_RUN')


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    actions = cli.add_subparsers(dest='action', required=True)
    create = actions.add_parser('create')
    create.add_argument('--repo', type=Path, default=ROOT)
    create.add_argument('--prepared-vectors', type=Path, required=True)
    create.add_argument('--rotation-fixture', type=Path, required=True)
    create.add_argument('--output-dir', type=Path, required=True)
    create.add_argument('--verification-dir', type=Path,
                        help='Retain existing offline logs with their original revision and scope')
    create.add_argument('--target', choices=TARGETS, default='rtx-4080-super')
    for name in ('verify', 'restore', 'plan'):
        parser = actions.add_parser(name)
        parser.add_argument('bundle', type=Path)
        if name != 'verify':
            parser.add_argument('checkout', type=Path)
        if name == 'plan':
            parser.add_argument('--cuda-root', type=Path, default=Path('/usr/local/cuda'))
            parser.add_argument('--openfhe-prefix', type=Path)
            parser.add_argument('--build-dir', type=Path)
            parser.add_argument('--output-dir', type=Path)
    args = cli.parse_args(argv)
    try:
        if args.action == 'create':
            result = create_bundle(args.repo, args.prepared_vectors, args.rotation_fixture,
                                   args.output_dir, args.target, args.verification_dir)
        else:
            manifest = verify_bundle(args.bundle)
            result = dict(status='VERIFIED', source_commit=manifest['source_commit'], gpu_execution='NOT_RUN')
            if args.action == 'restore':
                if args.checkout.exists() or args.checkout.is_symlink():
                    raise ValueError('Restore requires a new checkout path')
                run(['git', 'clone', args.bundle.resolve() / 'source.bundle', args.checkout])
                run(['git', 'checkout', '--detach', manifest['source_commit']], args.checkout)
                result['checkout'] = str(args.checkout.resolve())
            if args.action == 'plan':
                checkout = args.checkout.resolve()
                if source_revision(checkout) != manifest['source_commit']:
                    raise ValueError('Checkout differs from the frozen handoff revision')
                command = plan_command(checkout, args.bundle.resolve(), manifest['target'],
                    args.build_dir or checkout / 'build/rental-next',
                    args.output_dir or checkout / 'bench-results-rental-next', args.cuda_root,
                    args.openfhe_prefix or checkout / 'build/openfhe-compat/install')
                result = reviewed_plan(command, checkout)
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            print(error.stderr, file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
