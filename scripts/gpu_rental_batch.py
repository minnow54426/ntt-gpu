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
import subprocess
import sys
import tarfile
import time

import gpu_session
from analyze_gpu_batch import analyze, normalized_context, summarize, write_csv
from benchmark_batch import atomic_json, digest
from benchmark_report import validate_jsonl

ROOT = Path(__file__).resolve().parents[1]
ORDER = ('legacy_dif', 'group_twiddle_ct', 'group_twiddle_ct', 'legacy_dif')


def make_plan(args):
    output, build = args.output_dir.resolve(), args.build_dir.resolve()
    collector = ['--mode', 'offline' if args.mode == 'offline' else 'gpu',
                 '--compare-ntt', '--cuda-root', str(args.cuda_root.resolve()),
                 '--openfhe-prefix', str(args.openfhe_prefix.resolve()),
                 '--build-dir', str(build), '--output-dir', str(output / 'paired'),
                 '--architectures', '120', '--jobs', str(args.jobs),
                 '--command-timeout', str(args.command_timeout)]
    if args.mode != 'offline':
        collector += ['--benchmark', '--benchmark-scope', 'gpu-conversion',
                      '--benchmark-dimensions', '4096', '65536',
                      '--benchmark-towers', '3', '40', '--benchmark-prime-bits', '30', '60',
                      '--benchmark-warmups', '2', '--benchmark-samples', '10']
    return dict(mode=args.mode, status='PLANNED', budget_minutes=args.budget_minutes,
                output_dir=str(output), build_dir=str(build),
                collector_command=[sys.executable, str(ROOT / 'scripts/gpu_session.py'), *collector],
                collector_arguments=collector,
                tasks=['RTX 5090 identity and clean-source preflight',
                       'Clock/temperature logging',
                       'Both schedules: native exactness, 24 sanitizer commands and 32 measurement jobs each',
                       'Resident repeat: legacy, group, group, legacy; 5 warmups and 50 samples',
                       'Independent nsys and ncu captures for resident NTT and conversion',
                       'Validate raw records; export operation CSV and JSON summaries',
                       'Stop telemetry; archive records and verify every archived file'],
                offline_scope='Compile both sm_120 schedules and run host tests only; no GPU, timings or profiles',
                timing_scopes='Device events, public calls, resident NTT and resident ModUp/ModDown kept separate',
                profiling_scope='Whole-process captures include setup and verification; profiler timings are not benchmark samples',
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
        if 'RTX 5090' not in name or capability.strip() != '12.0':
            raise ValueError('This batch requires RTX 5090 / compute 12.0; found ' + name)
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

    def repeat(self):
        record = dict(status='RUNNING', runs=[])
        path = self.output / 'controlled-repeat.json'
        atomic_json(path, record)
        try:
            for index, schedule in enumerate(ORDER, 1):
                binary = self.build / schedule / 'ntt_resident_bench'
                vector = self.output / 'paired' / schedule / 'benchmark-vector-n65536-k3-b60.txt'
                log = f'repeat-{index}-{schedule}'
                raw = self.run(log, [binary, vector, '5', '50'])
                summary = validate_jsonl(raw, expected_n=65536, expected_towers=3,
                    expected_bits=60, expected_timing='host_wall_resident_chain')
                record['runs'].append(dict(order=index, schedule=schedule, raw_log=log + '.log',
                    raw_sha256=digest(self.output / (log + '.log')), binary_sha256=digest(binary), summary=summary))
                atomic_json(path, record)
            record['status'] = 'PASS'
        except BaseException:
            record['status'] = 'FAIL'
            raise
        finally:
            atomic_json(path, record)

    def profiles(self):
        vector = self.output / 'paired/group_twiddle_ct/benchmark-vector-n65536-k3-b60.txt'
        for profiler in ('nsys', 'ncu'):
            self.run(profiler + '-version', [profiler, '--version'], optional=True, limit=30)
        for scope, binary in [('ntt', 'ntt_resident_bench'), ('conversion', 'ntt_resident_conversion_bench')]:
            target = self.build / 'group_twiddle_ct' / binary
            for profiler in ('nsys', 'ncu'):
                name = profiler + '-' + scope
                if profiler == 'nsys':
                    command = ['nsys', 'profile', '--trace=cuda', '--sample=none', '--cpuctxsw=none',
                               '--output', self.output / name]
                else:
                    command = ['ncu', '--set', 'basic', '--launch-count', '8', '--export', self.output / name]
                captured = self.run(name, [*command, target, vector, '1', '2'], optional=True, limit=180)
                artifact = self.output / (name + ('.nsys-rep' if profiler == 'nsys' else '.ncu-rep'))
                if captured is not None and (not artifact.is_file() or artifact.stat().st_size == 0):
                    self.report['stages'][-1].update(status='FAIL', error='Profiler returned without a nonempty capture')
                    self.save()

    def analyze(self):
        contexts, rows, repeats = analyze(self.output)
        destination = self.output / 'analysis'
        destination.mkdir()
        write_csv(destination / 'paired-operations.csv', rows)
        write_csv(destination / 'resident-repeats.csv', repeats)
        atomic_json(destination / 'summary.json', dict(context=normalized_context(contexts['group_twiddle_ct']),
            paired_operation_rows=len(rows), repeat_operation_rows=len(repeats), operations=summarize(rows)))
        self.report['analysis'] = 'VALIDATED'

    def execute(self):
        self.output.mkdir(parents=True, exist_ok=False)
        self.save()
        previous_device = os.environ.get('CUDA_VISIBLE_DEVICES')
        result = 1
        try:
            if self.args.mode == 'gpu':
                os.environ['CUDA_VISIBLE_DEVICES'] = self.preflight()
                self.start_clock()
            plan = make_plan(self.args)
            if self.args.mode == 'gpu':
                self.report['gpu_execution'] = 'PENDING_COLLECTOR_RESULT'
            result = gpu_session.main(plan['collector_arguments'], deadline=self.deadline)
            if result == 130:
                raise KeyboardInterrupt
            if result:
                raise RuntimeError('Paired collector failed; subsequent GPU work skipped')
            if self.args.mode == 'gpu':
                self.report['gpu_execution'] = 'EXACT_TESTS_AND_SANITIZERS_PASS'
                self.repeat()
                self.profiles()
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
    cli.add_argument('--cuda-root', type=Path, default=Path('/usr/local/cuda'))
    cli.add_argument('--openfhe-prefix', type=Path, default=ROOT / 'build/openfhe-compat/install')
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
    try:
        checked_paths(args)
        if args.mode == 'plan':
            print(json.dumps(make_plan(args), indent=2))
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
