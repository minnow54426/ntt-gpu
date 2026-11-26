#!/usr/bin/env python3
"""Plan or run one bounded rental batch, retaining and archiving all evidence."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
from statistics import median
import subprocess
import sys
import tarfile
import time

import gpu_session
from analyze_gpu_batch import PUBLIC_REPEAT_BITS, REPEAT_ORDER as ORDER, export_analysis
from benchmark_batch import SCOPES, atomic_json, digest, select_conversion_reference, vector_name
from benchmark_report import validate_jsonl

ROOT = Path(__file__).resolve().parents[1]
GPU_TARGETS = {
    'rtx-5090': ('RTX 5090', '12.0', '120'),
    'rtx-4080': ('RTX 4080', '8.9', '89'),
    'rtx-4080-super': ('RTX 4080 SUPER', '8.9', '89'),
}
ROTATION_WARMUPS, ROTATION_PAIRS = 2, 10
ROTATION_FIXTURE_HEADER = b'NTT CKKS rotation benchmark v1\n'


def rotation_fixture_digest(path):
    with path.open('rb') as stream:
        if stream.read(len(ROTATION_FIXTURE_HEADER)) != ROTATION_FIXTURE_HEADER:
            raise ValueError('Invalid CKKS rotation fixture header')
    return digest(path)


def summarize_rotation_pairs(raw, warmups, pairs, backend='cuda'):
    records = [json.loads(line) for line in raw.splitlines()]
    if len(records) != 2 * pairs:
        raise ValueError('Missing or extra paired rotation samples')
    summaries = {}
    for group, active_q in enumerate((5, 3)):
        samples = records[group * pairs:(group + 1) * pairs]
        speedups = []
        for index, row in enumerate(samples):
            order = 'hoisted_first' if (warmups + index) % 2 else 'individual_first'
            expected = dict(schema=1, backend=backend, timing='public_call_wall',
                            ring_dimension=65536, active_q=active_q, shifts=[1, -1, 2, -2],
                            pair=index + 1, order=order)
            if not isinstance(row, dict) or any(row.get(key) != value for key, value in expected.items()):
                raise ValueError('Rotation sample workload, backend or pair order differs')
            individual, hoisted = row.get('individual_ns'), row.get('hoisted_ns')
            if any(type(value) is not int or value <= 0 for value in (individual, hoisted)):
                raise ValueError('Rotation timings must be positive integer nanoseconds')
            speedups.append(individual / hoisted)
        summaries[str(active_q)] = dict(
            median_individual_ns=median(row['individual_ns'] for row in samples),
            median_hoisted_ns=median(row['hoisted_ns'] for row in samples),
            median_paired_speedup=median(speedups))
    return summaries


def make_plan(args):
    output, build = args.output_dir.resolve(), args.build_dir.resolve()
    model, capability, architecture = GPU_TARGETS[args.target]
    collector = ['--mode', 'offline' if args.mode == 'offline' else 'gpu',
                 '--compare-ntt', '--cuda-root', str(args.cuda_root.resolve()),
                 '--openfhe-prefix', str(args.openfhe_prefix.resolve()),
                 '--build-dir', str(build), '--output-dir', str(output / 'paired'),
                 '--architectures', architecture, '--jobs', str(args.jobs),
                 '--command-timeout', str(args.command_timeout)]
    if args.mode != 'offline':
        collector += ['--benchmark', '--benchmark-scope', 'gpu-conversion',
                      '--benchmark-dimensions', '4096', '65536',
                      '--benchmark-towers', '3', '40', '--benchmark-prime-bits', '30', '60',
                      '--benchmark-warmups', '2', '--benchmark-samples', '10']
        for option, directory in (('--prepared-vectors', args.prepared_vectors),
                                  ('--conversion-references', args.conversion_references)):
            if directory is not None:
                collector.extend([option, str(directory.resolve())])
    tasks = [model + ' identity and clean-source preflight',
             'Clock/temperature logging',
             'Both schedules: native exactness, 28 sanitizer commands and 32 measurement jobs each',
             'Resident repeat: legacy, group, group, legacy; 5 warmups and 50 samples']
    if args.extended_probes:
        tasks.append('Ordered public-call repeats for 30/60-bit 40-tower workloads')
    if args.rotation_fixture is not None:
        tasks.append('Paired CKKS rotation timing with a local OpenFHE fixture')
    tasks.append('Independent nsys and ncu captures for resident NTT and conversion')
    if args.extended_probes:
        tasks.append('40-tower conversion CUDA/NVTX timeline and SQLite export')
    tasks.append('Defer paired analysis to the local machine' if args.defer_analysis
                 else 'Validate raw records; export operation CSV and JSON summaries')
    tasks.append('Stop telemetry; archive records and verify every archived file')
    return dict(mode=args.mode, status='PLANNED', budget_minutes=args.budget_minutes,
                extended_probes=args.extended_probes,
                rotation_fixture=str(args.rotation_fixture.resolve()) if args.rotation_fixture else None,
                target=dict(model=model, compute_cap=capability, cuda_architecture=architecture),
                output_dir=str(output), build_dir=str(build),
                collector_command=[sys.executable, str(ROOT / 'scripts/gpu_session.py'), *collector],
                collector_arguments=collector,
                tasks=tasks,
                offline_scope=f'Compile both sm_{architecture} schedules and run host tests only; no GPU, timings or profiles',
                timing_scopes='Device events, public calls, resident NTT and resident ModUp/ModDown kept separate',
                profiling_scope='Separate NVTX executables label setup, warmups, samples and verification; counters select sample ranges only',
                budget_scope='Preflight, compilation, checks, measurements and profiles; evidence packaging continues after cutoff',
                rentals_created=False)


def checked_paths(args):
    output, build = args.output_dir.resolve(), args.build_dir.resolve()
    if output == build or output in build.parents or build in output.parents:
        raise ValueError('Build and record roots must be disjoint')
    for path in (output, build, output.with_suffix('.tar.gz'),
                 output.with_suffix('.tar.gz.sha256'), output.with_suffix('.tar.gz.verify.json')):
        if path.exists() or path.is_symlink():
            raise ValueError('Use fresh paths; refusing to overwrite ' + str(path))
    # A record created in a tracked directory would dirty the measured checkout.
    if args.mode == 'gpu':
        for path, directory in ((output, True), (build, True),
                                (output.with_suffix('.tar.gz'), False),
                                (output.with_suffix('.tar.gz.sha256'), False),
                                (output.with_suffix('.tar.gz.verify.json'), False)):
            if path.is_relative_to(ROOT):
                result = subprocess.run(['git', 'check-ignore', '--quiet', str(path) + (os.sep if directory else '')], cwd=ROOT)
                if result.returncode:
                    raise ValueError('In-repository outputs must be ignored by Git: ' + str(path))


class Batch:
    def __init__(self, args):
        self.args = args
        self.output = args.output_dir.resolve()
        self.build = args.build_dir.resolve()
        self.deadline = time.monotonic() + args.budget_minutes * 60
        self.report = make_plan(args)
        self.report.update(status='RUNNING', started_utc=datetime.now(timezone.utc).isoformat(),
                           gpu_execution='NOT_RUN', stages=[])
        self.clock = None
        self.clock_log = None

    def save(self):
        atomic_json(self.output / 'batch.json', self.report)

    def run(self, name, command, optional=False, limit=None):
        command = list(map(str, command))
        stage = dict(name=name, command=command, log=name + '.log', status='RUNNING', optional=optional)
        self.report['stages'].append(stage)
        self.save()
        try:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, 0)
            with (self.output / stage['log']).open('w') as stream:
                result = gpu_session.run_process(command, cwd=ROOT, stdout=stream,
                    stderr=subprocess.STDOUT, timeout=min(remaining, limit or self.args.command_timeout))
            stage.update(exit_code=result.returncode, status='PASS' if result.returncode == 0 else 'FAIL')
            if result.returncode:
                raise RuntimeError(name + ' exited ' + str(result.returncode))
            return (self.output / stage['log']).read_text()
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            stage.update(status='FAIL', error=str(error))
            if not optional:
                raise
            return None
        except KeyboardInterrupt:
            stage['status'] = 'INTERRUPTED'
            raise
        finally:
            self.save()

    def preflight(self):
        commit = self.run('source', ['git', 'rev-parse', 'HEAD']).strip()
        if self.run('source-state', ['git', 'status', '--porcelain']).strip():
            raise ValueError('Native measurements require a clean checkout')
        self.report['source_commit'] = commit
        text = self.run('gpu-identity', ['nvidia-smi',
            '--query-gpu=index,uuid,name,compute_cap,driver_version', '--format=csv,noheader'])
        devices = list(csv.reader(text.splitlines(), skipinitialspace=True))
        selected = [row for row in devices if len(row) == 5 and row[0].strip() == str(self.args.device_index)]
        if len(selected) != 1 or len(selected[0]) != 5:
            raise ValueError('Selected physical GPU index was not found')
        index, uuid, name, capability, driver = selected[0]
        expected_model, expected_capability, _ = GPU_TARGETS[self.args.target]
        model = name.strip().removeprefix('NVIDIA ').removeprefix('GeForce ')
        if model != expected_model or capability.strip() != expected_capability:
            raise ValueError(f'This batch requires {expected_model} / compute {expected_capability}; '
                             f'found {name} / compute {capability}')
        self.report['device'] = dict(index=index, uuid=uuid, name=name, compute_cap=capability, driver=driver)
        self.run('sanitizer-version', ['compute-sanitizer', '--version'])
        return uuid.strip()

    def start_clock(self):
        try:
            self.clock_log = (self.output / 'clocks.csv').open('w')
            self.clock = subprocess.Popen(['nvidia-smi', '-i', str(self.args.device_index),
                '--query-gpu=timestamp,index,temperature.gpu,clocks.sm,clocks.mem,power.draw',
                '--format=csv', '-l', '2'], stdout=self.clock_log, stderr=subprocess.STDOUT,
                start_new_session=True)
            self.report['telemetry'] = 'RUNNING'
        except OSError as error:
            self.report['telemetry'] = 'UNAVAILABLE: ' + str(error)
        self.save()

    def stop_clock(self):
        if self.clock is not None:
            exited = self.clock.poll()
            if exited is None:
                try:
                    os.killpg(self.clock.pid, signal.SIGTERM)
                    self.clock.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(self.clock.pid, signal.SIGKILL)
                    self.clock.wait()
                except ProcessLookupError:
                    pass
                self.report['telemetry'] = 'CAPTURED'
            else:
                self.report['telemetry'] = 'FAILED: exit ' + str(exited)
        if self.clock_log is not None:
            self.clock_log.close()
        if self.report.get('telemetry') == 'CAPTURED' and len((self.output / 'clocks.csv').read_text().splitlines()) < 2:
            self.report['telemetry'] = 'FAILED: no telemetry samples'

    def repeat(self, scope='chain', towers=3, bits=60):
        optional = scope == 'gpu-public'
        stem = f'public-repeat-b{bits}' if optional else 'controlled-repeat'
        log_prefix = stem if optional else 'repeat'
        binary_name, subcommand, timing = SCOPES[scope]
        record = dict(status='RUNNING', runs=[])
        path = self.output / (stem + '.json')
        atomic_json(path, record)
        try:
            for index, schedule in enumerate(ORDER, 1):
                binary = self.build / schedule / binary_name
                vector = self.output / 'paired' / schedule / vector_name(65536, towers, bits)
                log = f'{log_prefix}-{index}-{schedule}'
                command = [binary, *([subcommand] if subcommand else []), vector, '5', '50']
                raw = self.run(log, command, optional=True) if optional else self.run(log, command)
                if raw is None:
                    return
                summary = validate_jsonl(raw, expected_n=65536, expected_towers=towers,
                    expected_bits=bits, expected_timing=timing)
                record['runs'].append(dict(order=index, schedule=schedule, raw_log=log + '.log',
                    raw_sha256=digest(self.output / (log + '.log')), binary_sha256=digest(binary), summary=summary))
                atomic_json(path, record)
            record['status'] = 'PASS'
        except (OSError, ValueError) as error:
            if not optional:
                raise
            self.report['stages'].append(dict(name=stem + '-validation', optional=True,
                                              status='FAIL', error=str(error)))
            self.save()
        finally:
            if record['status'] == 'RUNNING':
                record['status'] = 'FAIL'
            atomic_json(path, record)

    def benchmark_rotations(self, fixture, checksum):
        binary = self.build / 'group_twiddle_ct/ntt_resident_cuda_hybrid'
        name = 'ckks-rotation-pairs'
        raw = self.run(name, [binary, '--rotation-benchmark', fixture, ROTATION_WARMUPS, ROTATION_PAIRS])
        if rotation_fixture_digest(fixture) != checksum:
            raise ValueError('CKKS rotation fixture changed during the rental')
        self.report['rotation_benchmark'] = dict(
            warmups=ROTATION_WARMUPS, pairs=ROTATION_PAIRS, fixture_sha256=checksum,
            binary_sha256=digest(binary), raw_log=name + '.log',
            raw_sha256=digest(self.output / (name + '.log')),
            summary=summarize_rotation_pairs(raw, ROTATION_WARMUPS, ROTATION_PAIRS))
        self.save()

    def profiles(self):
        paired = self.output / 'paired/group_twiddle_ct'
        for profiler in ('nsys', 'ncu'):
            self.run(profiler + '-version', [profiler, '--version'], optional=True, limit=30)
        self.report['profile_binaries'] = {}
        captures = [('ntt', 'ntt_resident_profile', 3, ('nsys', 'ncu')),
                    ('conversion', 'ntt_resident_conversion_profile', 3, ('nsys', 'ncu'))]
        if self.args.extended_probes:
            captures.append(('conversion-k40', 'ntt_resident_conversion_profile', 40, ('nsys',)))
        for scope, binary, towers, profilers in captures:
            target = self.build / 'group_twiddle_ct' / binary
            self.report['profile_binaries'][binary] = digest(target) if target.is_file() else None
            vector = paired / vector_name(65536, towers, 60)
            workload = [target, vector, '1', '2']
            if scope.startswith('conversion') and self.args.conversion_references is not None:
                try:
                    reference, checksum = select_conversion_reference(self.args.conversion_references, 65536, towers, 60)
                    manifest = json.loads((paired / 'benchmark-manifest.json').read_text())
                    if checksum != manifest['context']['conversion_reference_sha256'].get(reference.name):
                        raise ValueError('profile conversion reference differs from the measured batch')
                except (OSError, ValueError, KeyError) as error:
                    self.report['stages'].append(dict(name='profile-' + scope + '-reference', status='FAIL',
                                                      optional=True, error=str(error)))
                    self.save()
                    continue
                key = 'profile_' + scope.replace('-', '_') + '_reference_sha256'
                self.report[key] = checksum
                workload.append(reference)
            for profiler in profilers:
                name = profiler + '-' + scope
                if profiler == 'nsys':
                    command = ['nsys', 'profile', '--trace=cuda,nvtx', '--sample=none', '--cpuctxsw=none',
                               '--output', self.output / name]
                else:
                    command = ['ncu', '--nvtx', '--nvtx-include', 'ntt-gpu@ntt.sample/',
                               '--set', 'basic', '--launch-count', '8', '--export', self.output / name]
                captured = self.run(name, [*command, *workload], optional=True, limit=180)
                artifact = self.output / (name + ('.nsys-rep' if profiler == 'nsys' else '.ncu-rep'))
                if captured is None:
                    continue
                if (self.report['profile_binaries'][binary] is None or
                        not artifact.is_file() or artifact.stat().st_size == 0):
                    self.report['stages'][-1].update(status='FAIL', error='Missing profiling executable or nonempty capture')
                    self.save()
                    continue
                if scope == 'conversion-k40':
                    database = self.output / (name + '.sqlite')
                    exported = self.run(name + '-sqlite', ['nsys', 'export', '--type', 'sqlite',
                        '--output', database, artifact], optional=True, limit=60)
                    if exported is not None and (not database.is_file() or database.stat().st_size == 0):
                        self.report['stages'][-1].update(status='FAIL', error='Missing SQLite export')
                        self.save()

    def analyze(self):
        export_analysis(self.output, self.output / 'analysis')
        self.report['analysis'] = 'VALIDATED'

    def execute(self):
        self.output.mkdir(parents=True, exist_ok=False)
        self.save()
        previous_device = os.environ.get('CUDA_VISIBLE_DEVICES')
        result = 1
        try:
            collector = self.report['collector_arguments']
            if self.args.prepared_vectors is not None or self.args.conversion_references is not None:
                gpu_session.validate_prepared_inputs(gpu_session.parser().parse_args(collector))
            fixture = self.args.rotation_fixture
            fixture_checksum = rotation_fixture_digest(fixture) if fixture is not None else None
            if self.args.mode == 'gpu':
                os.environ['CUDA_VISIBLE_DEVICES'] = self.preflight()
                self.start_clock()
            if self.args.mode == 'gpu':
                self.report['gpu_execution'] = 'PENDING_COLLECTOR_RESULT'
            try:
                result = gpu_session.main(collector, deadline=self.deadline)
            except SystemExit as error:
                raise ValueError('Paired collector rejected its arguments') from error
            if result == 130:
                raise KeyboardInterrupt
            if result:
                raise RuntimeError('Paired collector failed; subsequent GPU work skipped')
            if self.args.mode == 'gpu':
                self.report['gpu_execution'] = 'EXACT_TESTS_AND_SANITIZERS_PASS'
                self.repeat()
                if self.args.extended_probes:
                    for bits in PUBLIC_REPEAT_BITS:
                        self.repeat('gpu-public', 40, bits)
                if fixture is not None:
                    self.benchmark_rotations(fixture.resolve(), fixture_checksum)
                self.profiles()
                if self.args.defer_analysis:
                    self.report['analysis'] = 'DEFERRED_TO_LOCAL'
                else:
                    self.analyze()
                self.report['status'] = 'GPU_BATCH_PASSED'
            else:
                self.report['status'] = 'OFFLINE_READY'
            result = 0
        except KeyboardInterrupt:
            self.report.update(status='INTERRUPTED', error='Interrupted; partial evidence retained')
            result = 130
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            self.report.update(status='FAIL', error=str(error))
            print(error, file=sys.stderr)
            result = 1
        finally:
            if self.report['gpu_execution'] == 'PENDING_COLLECTOR_RESULT':
                self.report['gpu_execution'] = 'INCOMPLETE; inspect paired child reports'
            if previous_device is None:
                os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = previous_device
            try:
                self.stop_clock()
            finally:
                self.report['finished_utc'] = datetime.now(timezone.utc).isoformat()
                if self.report['status'] == 'GPU_BATCH_PASSED' and (
                        self.report.get('telemetry') != 'CAPTURED' or
                        any(stage['status'] != 'PASS' for stage in self.report['stages'] if stage['optional'])):
                    self.report['status'] = 'GPU_BATCH_PASSED_WITH_LIMITATIONS'
                self.save()
                archive(self.output)
        print('Batch:', self.output / 'batch.json', '(' + self.report['status'] + ')')
        return result


def archive(output):
    """Package regular files, then verify archived bytes against their source hashes."""
    target = output.with_suffix('.tar.gz')
    for path in (output / 'SHA256SUMS', target, target.with_suffix(target.suffix + '.sha256'),
                 target.with_suffix(target.suffix + '.verify.json')):
        if path.exists() or path.is_symlink():
            raise FileExistsError('Refusing to overwrite ' + str(path))
    inventory = list(output.rglob('*'))
    paths = sorted(path for path in inventory if not path.is_dir())
    if any(path.is_symlink() for path in inventory) or any(not path.is_file() for path in paths):
        raise ValueError('Evidence must contain regular files, not links or special files')
    sums = output / 'SHA256SUMS'
    sums.write_text(''.join(digest(path) + '  ' + path.relative_to(output).as_posix() + '\n'
                            for path in paths))
    paths.append(sums)
    expected = {output.name + '/' + path.relative_to(output).as_posix(): digest(path) for path in paths}
    with target.open('xb') as stream, tarfile.open(fileobj=stream, mode='w:gz') as package:
        for path in paths:
            package.add(path, arcname=output.name + '/' + path.relative_to(output).as_posix(), recursive=False)
    with tarfile.open(target, 'r:gz') as package:
        members = package.getmembers()
        if len(members) != len(expected) or {member.name for member in members} != set(expected):
            raise ValueError('Archive inventory mismatch')
        for member in members:
            if not member.isfile():
                raise ValueError('Non-file archive member')
            with package.extractfile(member) as stream:
                checksum = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    checksum.update(chunk)
                actual = checksum.hexdigest()
            if actual != expected[member.name]:
                raise ValueError('Archive content mismatch: ' + member.name)
    checksum = digest(target)
    target.with_suffix(target.suffix + '.sha256').write_text(checksum + '  ' + target.name + '\n')
    atomic_json(target.with_suffix(target.suffix + '.verify.json'),
                dict(status='VERIFIED', sha256=checksum, files=len(expected), archive=target.name))
    print('Verified archive:', target)
    return target


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--mode', choices=('plan', 'offline', 'gpu'), default='plan')
    cli.add_argument('--target', choices=GPU_TARGETS, default='rtx-5090',
                     help='Expected GPU model and matching CUDA compile architecture')
    cli.add_argument('--cuda-root', type=Path, default=Path('/usr/local/cuda'))
    cli.add_argument('--openfhe-prefix', type=Path, default=ROOT / 'build/openfhe-compat/install')
    cli.add_argument('--prepared-vectors', type=Path, help='Directory containing CPU-prepared vectors and vectors.json')
    cli.add_argument('--conversion-references', type=Path,
                     help='Directory containing CPU-prepared conversion arrays and their manifest')
    cli.add_argument('--rotation-fixture', type=Path,
                     help='Locally prepared ciphertexts, rotation evaluation keys and OpenFHE references')
    cli.add_argument('--extended-probes', action='store_true',
                     help='Add 40-tower public-call repeats and an Nsight Systems SQLite export')
    cli.add_argument('--defer-analysis', action='store_true',
                     help='Archive raw evidence and run analyze_gpu_batch.py on a local machine')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    cli.add_argument('--output-dir', type=Path, default=ROOT / ('bench-results-rental-' + stamp))
    cli.add_argument('--build-dir', type=Path, default=ROOT / 'build' / ('rental-' + stamp))
    cli.add_argument('--budget-minutes', type=float, default=120)
    cli.add_argument('--device-index', type=int, default=0, help='Physical nvidia-smi index; selected by UUID for CUDA')
    cli.add_argument('--jobs', type=int, default=2)
    cli.add_argument('--command-timeout', type=int, default=1800)
    args = cli.parse_args(argv)
    if not 0 < args.budget_minutes <= 1440 or args.jobs < 1 or args.device_index < 0 or not 1 <= args.command_timeout <= 7200:
        cli.error('Require budget in (0,1440] minutes, positive jobs, nonnegative device and timeout 1..7200')
    prepared = args.prepared_vectors is not None or args.conversion_references is not None
    if prepared and args.mode == 'offline':
        cli.error('Prepared vectors and references are used by the GPU batch, not offline compilation')
    if args.rotation_fixture is not None and args.mode == 'offline':
        cli.error('--rotation-fixture requires a GPU batch or its plan')
    if args.extended_probes and args.mode == 'offline':
        cli.error('--extended-probes requires a GPU batch or its plan')
    if args.defer_analysis and args.mode == 'offline':
        cli.error('--defer-analysis requires a GPU batch or its plan')
    try:
        checked_paths(args)
        if args.mode == 'plan':
            plan = make_plan(args)
            if prepared:
                gpu_session.validate_prepared_inputs(gpu_session.parser().parse_args(plan['collector_arguments']))
            if args.rotation_fixture is not None:
                plan['rotation_fixture_sha256'] = rotation_fixture_digest(args.rotation_fixture)
            print(json.dumps(plan, indent=2))
            return 0
        def interrupted(signum, frame):
            raise KeyboardInterrupt
        prior_handler = signal.signal(signal.SIGTERM, interrupted)
        try:
            return Batch(args).execute()
        finally:
            signal.signal(signal.SIGTERM, prior_handler)
    except (OSError, ValueError, RuntimeError) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
