#!/usr/bin/env python3
"""Collect CPU benchmarks, compile CUDA offline, or record native GPU checks."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
from benchmark_batch import (DIMENSIONS, MAX_TOWERS, TOWER_COUNTS, SCOPE_GROUPS,
                             benchmark_scope, collect_batch, collect_profiles)

ROOT = Path(__file__).resolve().parents[1]
GPU_TESTS = {'cuda_oracle_crosscheck', 'cuda_matches_ref', 'cuda_verify', 'cuda_openfhe_exact', 'cuda_setup_fault_injection', 'cuda_benchmark_matrix_exact'}
OPENFHE_TESTS = {'openfhe_exact_compat', 'openfhe_representation_contract', 'openfhe_hybrid_graph', 'openfhe_benchmark_protocol'}
HOST_TESTS = {'cuda_host_validation', 'setup_ownership_host', 'measurement_host'}


def tool(name):
    local = ROOT / '.venv/bin' / name
    return str(local) if local.is_file() else name


def run_process(command, *, timeout, **options):
    """Bound command trees on Linux/macOS, including compiler/profiler children."""
    with subprocess.Popen(command, start_new_session=(os.name == 'posix'), **options) as process:
        try:
            return subprocess.CompletedProcess(command, process.wait(timeout=timeout))
        except BaseException:
            if os.name == 'posix':
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.wait()
            raise


class Session:
    def __init__(self, args):
        self.args = args
        self.root = ROOT
        self.output = args.output_dir.resolve()
        previous_attempt = 0
        if getattr(args, 'resume', False):
            previous = json.loads((self.output / 'report.json').read_text())
            if previous['mode'] != args.mode or args.mode not in ('gpu', 'cpu'):
                raise ValueError('Resume requires the same CPU/GPU session mode')
            if previous.get('benchmark_scope', 'all') != benchmark_scope(args):
                raise ValueError('Cannot change benchmark scope on resume; use a fresh output directory')
            previous_attempt = previous.get('attempt', 1)
            history = self.output / ('report-history-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
            with history.open('x') as archive:
                json.dump(previous, archive, indent=2)
        else:
            self.output.mkdir(parents=True, exist_ok=False)
        self.report = {'mode': args.mode, 'status': 'RUNNING', 'started_utc': datetime.now(timezone.utc).isoformat(),
                       'platform': platform.platform(), 'build_dir': str(args.build_dir.resolve()),
                       'attempt': previous_attempt + 1,
                       'stages': [], 'gpu_execution': 'NOT_RUN', 'cpu_execution': 'NOT_RUN', 'sanitizers': 'NOT_RUN',
                       'profiling': 'NOT_RUN' if getattr(args, 'profile', False) else 'NOT_REQUESTED',
                       'benchmark': 'NOT_RUN',
                       'benchmark_scope': benchmark_scope(args)}
        self.save()

    def save(self):
        temporary = self.output / 'report.json.tmp'
        temporary.write_text(json.dumps(self.report, indent=2) + '\n')
        temporary.replace(self.output / 'report.json')

    def run(self, name, command):
        command = [str(x) for x in command]
        print(f'[{name}] {" ".join(command)}', flush=True)
        log = self.output / (name + '.log')
        retry = 1
        while log.exists():
            retry += 1
            log = self.output / f'{name}-retry-{retry}.log'
        stage = {'name': name, 'command': command, 'log': log.name, 'status': 'RUNNING'}
        self.report['stages'].append(stage)
        self.save()
        try:
            with log.open('w') as output:
                result = run_process(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                     timeout=getattr(self.args, 'command_timeout', 1800))
            stage.update(status='PASS' if result.returncode == 0 else 'FAIL', exit_code=result.returncode)
        except (OSError, subprocess.TimeoutExpired) as error:
            stage.update(status='FAIL', error=str(error))
            self.save()
            raise RuntimeError(f'{name}: {error}') from error
        except KeyboardInterrupt:
            stage.update(status='INTERRUPTED')
            self.save()
            raise
        self.save()
        if result.returncode:
            raise RuntimeError(f'{name} failed ({result.returncode}); see {log}')
        return log.read_text(errors='replace')

    def record_cpu_identity(self):
        """Fingerprint stable host fields; keep variable clock data out of resume keys."""
        identity = {'machine': platform.machine(), 'node': platform.node()}
        if platform.system() == 'Linux':
            raw = json.loads(self.run('cpu-info', ['env', 'LC_ALL=C', 'lscpu', '--json']))
            fields = {row['field'].rstrip(':'): row['data'] for row in raw['lscpu']}
            for required in ('Architecture', 'Model name', 'CPU(s)'):
                if not fields.get(required):
                    raise RuntimeError('CPU identity missing ' + required)
            stable = ('Architecture', 'Model name', 'CPU(s)', 'Vendor ID',
                      'CPU family', 'Model', 'Stepping', 'Thread(s) per core',
                      'Core(s) per socket', 'Socket(s)', 'Hypervisor vendor')
            identity.update({key: fields[key] for key in stable if key in fields})
        elif platform.system() == 'Darwin':
            keys = ('hw.model', 'machdep.cpu.brand_string', 'hw.ncpu', 'hw.memsize')
            values = self.run('cpu-info', ['sysctl', '-n', *keys]).strip().splitlines()
            if len(values) != len(keys) or not all(values):
                raise RuntimeError('Incomplete macOS CPU identity')
            identity.update(zip(keys, values))
        else:
            raise RuntimeError('CPU identity collection supports Linux and macOS')
        self.report['cpu_identity'] = identity
        self.save()
        return json.dumps(identity, sort_keys=True)

    def execute(self):
        args = self.args
        try:
            self.report['source_commit'] = self.run('source', ['git', 'rev-parse', 'HEAD']).strip()
            self.report['source_dirty'] = bool(self.run('source-state', ['git', 'status', '--porcelain']).strip())
            if args.mode in ('gpu', 'cpu') and self.report['source_dirty'] and not args.allow_dirty:
                raise RuntimeError('Measurement records require a clean checkout; --allow-dirty marks an exploratory run')
            self.run('cmake-version', [args.cmake, '--version'])
            self.run('compiler-version', ['c++', '--version'])
            if args.mode != 'cpu':
                nvcc = args.cuda_root.resolve() / 'bin/nvcc'
                self.run('nvcc-version', [nvcc, '--version'])
            if args.mode in ('cpu', 'gpu'):
                cpu_identity = self.record_cpu_identity()
            if args.mode == 'gpu':
                self.run('hardware', ['nvidia-smi', '--query-gpu=name,driver_version,memory.total,pstate,power.limit,clocks.sm,clocks.mem', '--format=csv'])
                hardware_identity = self.run('hardware-identity', ['nvidia-smi', '--query-gpu=uuid,pci.bus_id,name,driver_version', '--format=csv'])
                if not args.skip_sanitizers:
                    self.run('sanitizer-version', ['compute-sanitizer', '--version'])
            build = args.build_dir.resolve()
            configure = [args.cmake, '-S', ROOT, '-B', build, '-DCMAKE_BUILD_TYPE=Release',
                         '-DNTT_ENABLE_OPENFHE_COMPAT=ON',
                         f'-DCMAKE_PREFIX_PATH={args.openfhe_prefix.resolve()}']
            if args.mode == 'cpu':
                configure += ['-DNTT_REQUIRE_CUDA=OFF', '-DNTT_ENABLE_CUDA=OFF']
            else:
                configure += ['-DNTT_REQUIRE_CUDA=ON', '-DNTT_ENABLE_CUDA=ON',
                    f'-DCMAKE_CUDA_COMPILER={nvcc}', f'-DCMAKE_CUDA_ARCHITECTURES={args.architectures}']
            self.run('configure', configure)
            cache = build / 'CMakeCache.txt'
            if cache.is_file():
                keys = {'CMAKE_CXX_COMPILER', 'CMAKE_CUDA_COMPILER', 'CMAKE_CUDA_HOST_COMPILER',
                        'CMAKE_CUDA_ARCHITECTURES', 'CMAKE_BUILD_TYPE', 'OpenFHE_DIR',
                        'CMAKE_CXX_FLAGS', 'CMAKE_CXX_FLAGS_RELEASE', 'CMAKE_CUDA_FLAGS', 'CMAKE_CUDA_FLAGS_RELEASE',
                        'NTT_REQUIRE_CUDA', 'NTT_ENABLE_CUDA', 'NTT_ENABLE_OPENFHE_COMPAT'}
                self.report['configured_cache'] = {}
                for line in cache.read_text().splitlines():
                    if '=' in line and ':' in line.split('=', 1)[0]:
                        name = line.split(':', 1)[0]
                        if name in keys:
                            self.report['configured_cache'][name] = line.split('=', 1)[1]
            self.run('build', [args.cmake, '--build', build, '-j', str(args.jobs)])
            registered = json.loads(self.run('registered-tests', [args.ctest, '--test-dir', build, '--show-only=json-v1']))
            tests = {test['name']: test for test in registered['tests']}
            required = OPENFHE_TESTS | HOST_TESTS
            if args.mode != 'cpu':
                required |= GPU_TESTS | {'cuda_constructor_preflight'}
            missing = required - tests.keys()
            if missing:
                raise RuntimeError('Required tests missing for ' + args.mode + ': ' + ', '.join(sorted(missing)))
            for name in GPU_TESTS if args.mode != 'cpu' else ():
                labels = next((p['value'] for p in tests[name]['properties'] if p['name'] == 'LABELS'), [])
                if 'gpu' not in labels:
                    raise RuntimeError(f'{name} lacks its mandatory GPU label')
            if args.mode == 'cpu':
                if GPU_TESTS & tests.keys() or any('gpu' in prop['value'] for test in tests.values()
                        for prop in test.get('properties', []) if prop['name'] == 'LABELS'):
                    raise RuntimeError('CPU mode requires a CUDA-disabled build without GPU tests')
                self.report['registered_gpu_tests'] = []
                self.report['cpu_execution'] = 'RUNNING'
                self.run('host-tests', [args.ctest, '--test-dir', build, '-LE', 'gpu', '--output-on-failure'])
                self.report['cpu_execution'] = 'EXACT_TESTS_PASS'
                if args.benchmark:
                    self.report['benchmark'] = 'RUNNING'
                    collect_batch(self, cpu_identity)
                self.report['status'] = 'CPU_CHECKS_PASSED'
            elif args.mode == 'offline':
                self.report['registered_gpu_tests'] = sorted(GPU_TESTS)
                if args.architectures == 'native':
                    raise RuntimeError('Offline compilation requires explicit architecture numbers, not native')
                self.run('host-tests', [args.ctest, '--test-dir', build, '-LE', 'gpu', '--output-on-failure'])
                self.report['status'] = 'OFFLINE_READY'
            else:
                self.report['registered_gpu_tests'] = sorted(GPU_TESTS)
                self.report['selected_device'] = self.run('device-info', [build / 'ntt_cuda_cli', 'info']).strip()
                self.report['gpu_execution'] = 'RUNNING'
                exact = [args.ctest, '--test-dir', build, '--output-on-failure']
                if benchmark_scope(args) == 'gpu':
                    # Direct CUDA/OpenFHE exactness remains mandatory. The separate
                    # CPU-only timing-protocol sweep belongs on the CPU host.
                    exact += ['-E', '^openfhe_benchmark_protocol$']
                    self.report['excluded_tests'] = ['openfhe_benchmark_protocol']
                self.run('exact-tests', exact)
                self.report['gpu_execution'] = 'EXACT_TESTS_PASS'
                if not args.skip_sanitizers:
                    self.report['sanitizers'] = 'RUNNING'
                    for sanitizer in ['memcheck', 'racecheck', 'initcheck', 'synccheck']:
                        self.run('sanitizer-' + sanitizer, ['compute-sanitizer', '--tool', sanitizer,
                            '--error-exitcode', '1', build / 'ntt_cuda_openfhe_tests', '--smoke'])
                        self.run('sanitizer-matrix-' + sanitizer, ['compute-sanitizer', '--tool', sanitizer,
                            '--error-exitcode', '1', build / 'ntt_cuda_openfhe_tests', '--matrix-smoke'])
                    self.report['sanitizers'] = 'SMOKE_MATRIX_PASS'
                else:
                    self.report['sanitizers'] = 'EXPLICITLY_SKIPPED'
                if args.benchmark:
                    self.report['benchmark'] = 'RUNNING'
                    self.report['benchmark_summaries'] = {}
                    collect_batch(self, hardware_identity)
                if args.profile:
                    self.report['profiling'] = 'RUNNING'
                    collect_profiles(self)
                self.report['status'] = 'GPU_CHECKS_PASSED'
                self.report['remaining_gates'] = ['Actual driver OOM/transfer errors and failed-context cleanup beyond pre-call injection',
                    'Native verification of event/CPU measurement and profiler data; per-kernel counters beyond the bounded capture',
                    'Full production parameter/key matrix and optimized runtime integration']
            return 0
        except KeyboardInterrupt:
            for field in ['gpu_execution', 'cpu_execution', 'sanitizers', 'benchmark', 'profiling']:
                if self.report[field] == 'RUNNING':
                    self.report[field] = 'INTERRUPTED'
            self.report.update(status='INTERRUPTED', error='Interrupted by user; partial evidence retained')
            return 130
        except Exception as error:
            for field in ['gpu_execution', 'cpu_execution', 'sanitizers', 'benchmark', 'profiling']:
                if self.report[field] == 'RUNNING':
                    self.report[field] = 'FAIL'
            self.report.update(status='FAIL', error=str(error))
            print(error, file=sys.stderr)
            return 1
        finally:
            self.report['finished_utc'] = datetime.now(timezone.utc).isoformat()
            self.save()
            print(f'Record: {self.output / "report.json"} ({self.report["status"]})', flush=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--mode', choices=['offline', 'gpu', 'cpu'], required=True)
    result.add_argument('--cuda-root', type=Path, help='Required for offline/gpu modes; omit in CPU mode')
    result.add_argument('--openfhe-prefix', type=Path, required=True)
    result.add_argument('--build-dir', type=Path)
    result.add_argument('--output-dir', type=Path)
    result.add_argument('--architectures', help='Default: 89;120 offline, native on a GPU')
    result.add_argument('--jobs', type=int, default=2)
    result.add_argument('--cmake', default=tool('cmake'))
    result.add_argument('--ctest', default=tool('ctest'))
    result.add_argument('--benchmark', action='store_true')
    result.add_argument('--benchmark-scope', choices=SCOPE_GROUPS,
                        help='Default: all in GPU mode, cpu in CPU mode; gpu selects public/events only')
    result.add_argument('--benchmark-warmups', type=int, default=5)
    result.add_argument('--benchmark-samples', type=int, default=100)
    result.add_argument('--benchmark-dimensions', type=int, nargs='+', choices=DIMENSIONS, default=list(DIMENSIONS))
    result.add_argument('--benchmark-towers', type=int, nargs='+', choices=range(1, MAX_TOWERS + 1),
                        default=list(TOWER_COUNTS), help='RNS prime count (1..40); default: 1 3 8 16 24 32 40')
    result.add_argument('--profile', action='store_true', help='Bounded nsys/ncu capture; requires both tools and permissions')
    result.add_argument('--resume', action='store_true', help='Rerun CPU/GPU gates; reuse only validated matching benchmark jobs')
    result.add_argument('--command-timeout', type=int, default=1800, help='Seconds per command (1..7200)')
    result.add_argument('--skip-sanitizers', action='store_true')
    result.add_argument('--allow-dirty', action='store_true')
    return result


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        benchmark_scope(args)
    except ValueError as error:
        cli.error(str(error))
    if args.mode != 'cpu' and args.cuda_root is None:
        cli.error('--cuda-root is required for offline/gpu modes')
    if args.mode == 'cpu' and (args.cuda_root or args.architectures or args.profile or args.skip_sanitizers):
        cli.error('CPU mode cannot use CUDA paths/architectures, profiling or sanitizer flags')
    if args.benchmark_scope is not None and not args.benchmark:
        cli.error('--benchmark-scope requires --benchmark')
    if args.jobs < 1:
        cli.error('--jobs must be positive')
    if not 1 <= args.benchmark_warmups <= 10000 or not 2 <= args.benchmark_samples <= 100000:
        cli.error('benchmark requires 1..10000 warmups and 2..100000 samples')
    if not 1 <= args.command_timeout <= 7200:
        cli.error('--command-timeout must be 1..7200 seconds')
    if len(set(args.benchmark_dimensions)) != len(args.benchmark_dimensions) or len(set(args.benchmark_towers)) != len(args.benchmark_towers):
        cli.error('benchmark matrix axes must not contain duplicates')
    if args.resume and (args.output_dir is None or not args.benchmark or args.allow_dirty):
        cli.error('--resume requires --output-dir, --benchmark and a clean checkout')
    if args.mode == 'offline' and (args.benchmark or args.profile or args.resume or args.skip_sanitizers or args.architectures == 'native'):
        cli.error('Offline mode cannot execute a benchmark or use GPU/native execution flags')
    if args.mode != 'cpu':
        args.architectures = args.architectures or ('89;120' if args.mode == 'offline' else 'native')
    args.build_dir = args.build_dir or ROOT / {'offline': 'build/cuda-offline13',
                                             'gpu': 'build/native-gpu', 'cpu': 'build/native-cpu'}[args.mode]
    args.output_dir = args.output_dir or ROOT / ('bench-results-' + args.mode + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    return Session(args).execute()


if __name__ == '__main__':
    sys.exit(main())
