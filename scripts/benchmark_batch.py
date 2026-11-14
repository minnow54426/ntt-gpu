"""Bounded, restartable measurement jobs; never substitutes for native gates."""
import hashlib
import json
import os
from pathlib import Path
import sys

from benchmark_report import validate_jsonl

DIMENSIONS = (4096, 8192, 16384, 32768, 65536)
PRIME_BITS = (30, 50, 55, 60)
TOWER_COUNTS = (1, 3, 8, 16, 24, 32, 40)
MAX_TOWERS = 40
SCOPES = {'public': ('ntt_cuda_cli', 'bench-jsonl', 'host_wall_public_call'),
          'events': ('ntt_cuda_cli', 'events-jsonl', 'cuda_event_resident_and_pinned_copy'),
          'openfhe': ('ntt_openfhe_bench', None, 'host_wall_openfhe_native')}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def collect_batch(session, hardware_identity):
    """Rerun gates on every attempt; reuse only hash-checked complete jobs.

    A fresh process runs each measurement, so one failed job cannot leave a
    poisoned CUDA context in the next job. Any failure makes the batch fail.
    """
    args, output = session.args, session.output
    build = args.build_dir.resolve()
    prefix = args.openfhe_prefix.resolve()
    libraries = {str(path.relative_to(prefix)): digest(path)
                 for path in prefix.rglob('*OPENFHE*') if path.is_file()
                 and (path.suffix in ('.a', '.so', '.dylib') or '.so.' in path.name)}
    if not libraries:
        raise RuntimeError('No OpenFHE libraries found under the requested prefix')
    context = {'source_commit': session.report['source_commit'],
               'source_dirty': session.report['source_dirty'],
               'hardware_identity': hardware_identity.strip(),
               'selected_device': session.report['selected_device'],
               'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
               'architectures': args.architectures,
               'warmups': args.benchmark_warmups, 'samples': args.benchmark_samples,
               'dimensions': list(args.benchmark_dimensions),
               'tower_counts': list(args.benchmark_towers), 'prime_bits': list(PRIME_BITS),
               'openfhe_libraries': libraries,
               'binaries': {name: digest(build / name) for name in ('ntt_cuda_cli', 'ntt_openfhe_bench')}}
    manifest_path = output / 'benchmark-manifest.json'
    manifest = {'context': context, 'jobs': {}}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if context['source_dirty'] or manifest['context'] != context:
            raise RuntimeError('Cannot resume benchmarks: source, hardware, binaries or matrix changed; use a fresh output directory')
    else:
        atomic_json(manifest_path, manifest)
    session.report['benchmark_manifest'] = manifest_path.name
    session.report['benchmark_summaries'] = {}
    errors = []
    for dimension in args.benchmark_dimensions:
        for towers in args.benchmark_towers:
            for bits in PRIME_BITS:
                cell = f'n{dimension}-k{towers}-b{bits}'
                vector = output / f'benchmark-vector-{cell}.txt'
                # Regenerate from source even on resume; q/roots must still
                # match every retained scope for this cell.
                try:
                    session.run(f'parameters-{cell}', [sys.executable,
                        session.root / 'tests/oracle/gen_vectors.py', '--zero-coefficients', vector,
                        str(dimension), str(towers), '1', str(bits)])
                    tokens = [word for line in vector.read_text().splitlines()
                              if not line.startswith('#') for word in line.split()]
                    parameters = {'moduli': [int(tokens[2 + 2 * i]) for i in range(towers)],
                                  'roots': [int(tokens[3 + 2 * i]) for i in range(towers)]}
                except Exception as error:
                    errors.append(f'{cell} parameters: {error}')
                    continue
                for scope, (binary, subcommand, timing) in SCOPES.items():
                    name = f'benchmark-{cell}-{scope}'
                    job = manifest['jobs'].get(name, {})
                    try:
                        reused = False
                        if job.get('status') == 'PASS':
                            path = output / job['log']
                            # Only collector-generated basenames are accepted.
                            if Path(job['log']).name != job['log'] or digest(path) != job['sha256']:
                                raise ValueError('retained log hash/path mismatch')
                            raw = path.read_text()
                            reused = True
                        else:
                            command = [build / binary]
                            if subcommand:
                                command.append(subcommand)
                            raw = session.run(name, command + [vector,
                                str(args.benchmark_warmups), str(args.benchmark_samples)])
                            path = output / session.report['stages'][-1]['log']
                        summary = validate_jsonl(raw, expected_bits=bits, expected_n=dimension,
                                                 expected_towers=towers, expected_timing=timing)
                        workload = summary['workload']
                        if workload.get('simulation'):
                            raise ValueError('synthetic benchmark cannot count as native evidence')
                        if (workload['samples'] != args.benchmark_samples or workload['warmups'] != args.benchmark_warmups
                                or workload.get('seed') != 7492057
                                or any(workload[key] != values for key, values in parameters.items())):
                            raise ValueError('benchmark ignored requested inputs/counts')
                        manifest['jobs'][name] = {'status': 'PASS', 'log': path.name,
                            'sha256': digest(path),
                            'attempt': job['attempt'] if reused else session.report['attempt']}
                        summary['reused'] = reused
                        summary['origin_attempt'] = manifest['jobs'][name]['attempt']
                        session.report['benchmark_summaries'][name] = summary
                    except Exception as error:
                        errors.append(f'{name}: {error}')
                        manifest['jobs'][name] = {'status': 'FAIL', 'error': str(error)}
                    atomic_json(manifest_path, manifest)
                    session.save()
    if errors:
        session.report['benchmark_errors'] = errors
        raise RuntimeError(f'{len(errors)} benchmark jobs failed; completed jobs retained for --resume')
    session.report['benchmark'] = 'PASS_30_50_55_60_BIT_PUBLIC_EVENTS_OPENFHE'


def collect_profiles(session):
    """Small selected workload only; profiler timings are never benchmarks."""
    args, build = session.args, session.args.build_dir.resolve()
    vector = session.output / 'profile-vector-n65536-k3-b60.txt'
    session.run('profile-parameters', [sys.executable, session.root / 'tests/oracle/gen_vectors.py', '--zero-coefficients',
                                      vector, '65536', '3', '1', '60'])
    session.run('nsys-version', ['nsys', '--version'])
    session.run('ncu-version', ['ncu', '--version'])
    # Unique prefix on every attempt; never force-overwrite prior profiler data.
    prefix = session.output / f'profile-attempt-{session.report["attempt"]}'
    session.run('nsys', ['nsys', 'profile', '--trace=cuda', '--sample=none',
        '--cpuctxsw=none', '--output', str(prefix) + '-timeline',
        build / 'ntt_cuda_cli', 'bench-jsonl', vector, '1', '2'])
    session.run('ncu', ['ncu', '--set', 'basic', '--launch-count', '8',
        '--export', str(prefix) + '-counters', build / 'ntt_cuda_cli',
        'events-jsonl', vector, '1', '2'])
    session.report['profiling'] = 'SELECTED_N65536_K3_B60_CAPTURED'
