#!/usr/bin/env python3
"""Collect CPU benchmarks, compile CUDA offline, or record native GPU checks."""
import argparse
from datetime import datetime, timezone
from itertools import product
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
from benchmark_batch import (DIMENSIONS, MAX_TOWERS, PRIME_BITS, TOWER_COUNTS, SCOPE_GROUPS,
                             atomic_json, benchmark_scope, check_vector, collect_batch, collect_profiles, digest,
                             select_conversion_reference, stage_vector, vector_hashes)

ROOT = Path(__file__).resolve().parents[1]
OFFLINE_FIXTURES = ('vector_n1024.txt', 'vector_n65536.txt')
GPU_TESTS = {'cuda_oracle_crosscheck', 'cuda_matches_ref', 'cuda_verify', 'cuda_openfhe_exact', 'cuda_setup_fault_injection', 'cuda_benchmark_matrix_exact', 'resident_cuda_exact', 'cuda_basis_exact', 'cuda_shoup_exact', 'cuda_barrett_exact', 'resident_cuda_hybrid_exact'}
OPENFHE_TESTS = {'openfhe_exact_compat', 'openfhe_representation_contract', 'openfhe_hybrid_graph', 'openfhe_basis_conversion', 'basis_openfhe_exact', 'openfhe_benchmark_protocol', 'resident_openfhe_host'}
HOST_TESTS = {'cuda_host_validation', 'setup_ownership_host', 'measurement_host', 'resident_runtime_host', 'resident_benchmark_host', 'resident_conversion_benchmark_host', 'basis_conversion_bigint', 'benchmark_profile_ranges', 'resident_profile_host', 'resident_conversion_profile_host', 'barrett_transform_check'}
NTT_SCHEDULES = {'group_twiddle_ct': 'OFF', 'legacy_dif': 'ON'}


def offline_build_hashes(build, prefix):
    executables = sorted(path for path in build.glob('ntt_*')
                         if path.is_file() and os.access(path, os.X_OK))
    libraries = sorted(path for path in prefix.rglob('*OPENFHE*')
                       if path.is_file() and path.suffix in ('.a', '.so', '.dylib'))
    if not executables or not libraries:
        raise ValueError('offline build is missing executables or OpenFHE libraries')
    return {
        'build': {str(path.relative_to(build)): digest(path)
                  for path in [build / name for name in ('CMakeCache.txt', 'CTestTestfile.cmake', *OFFLINE_FIXTURES)] + executables},
        'openfhe': {str(path.relative_to(prefix)): digest(path) for path in libraries},
    }


def verify_offline_build(report_path, build, prefix, commit, schedule, architectures):
    report = json.loads(report_path.read_text())
    if (report.get('status') != 'OFFLINE_READY' or report.get('mode') != 'offline'
            or report.get('source_dirty') is not False or report.get('source_commit') != commit
            or report.get('build_dir') != str(build) or report.get('source_dir') != str(ROOT)
            or report.get('openfhe_prefix') != str(prefix) or report.get('transform_schedule') != schedule
            or report.get('configured_cache', {}).get('CMAKE_CUDA_ARCHITECTURES') != architectures
            or not any(s.get('name') == 'host-tests' and s.get('status') == 'PASS'
                       for s in report.get('stages', []))):
        raise ValueError('offline record does not match the clean checkout, paths, architecture or schedule')
    hashes = offline_build_hashes(build, prefix)
    if hashes != report.get('offline_build_sha256'):
        raise ValueError('offline build files or OpenFHE libraries changed')
    return hashes


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
        schedule = getattr(args, 'ntt_schedule', None)
        if getattr(args, 'resume', False):
            previous = json.loads((self.output / 'report.json').read_text())
            if previous['mode'] != args.mode or args.mode not in ('gpu', 'cpu'):
                raise ValueError('Resume requires the same CPU/GPU session mode')
            if previous.get('benchmark_scope', 'all') != benchmark_scope(args):
                raise ValueError('Cannot change benchmark scope on resume; use a fresh output directory')
            if previous.get('transform_schedule') != schedule:
                raise ValueError('Cannot change NTT schedule on resume; use a fresh output directory')
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
        if schedule:
            self.report['transform_schedule'] = schedule
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
            timeout = getattr(self.args, 'command_timeout', 1800)
            deadline = getattr(self.args, 'deadline_monotonic', None)
            if deadline is not None:
                timeout = min(timeout, deadline - time.monotonic())
                if timeout <= 0:
                    raise subprocess.TimeoutExpired(command, 0)
            with log.open('w') as output:
                result = run_process(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                     timeout=timeout)
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
            schedule = self.report.get('transform_schedule')
            legacy_option = NTT_SCHEDULES[schedule or 'group_twiddle_ct']
            configure = [args.cmake, '-S', ROOT, '-B', build, '-DCMAKE_BUILD_TYPE=Release',
                         '-DNTT_ENABLE_OPENFHE_COMPAT=ON',
                         f'-DNTT_CUDA_LEGACY_BUTTERFLY={legacy_option}',
                         f'-DCMAKE_PREFIX_PATH={args.openfhe_prefix.resolve()}']
            if args.mode == 'cpu':
                configure += ['-DNTT_REQUIRE_CUDA=OFF', '-DNTT_ENABLE_CUDA=OFF']
            else:
                configure += ['-DNTT_REQUIRE_CUDA=ON', '-DNTT_ENABLE_CUDA=ON',
                    f'-DCMAKE_CUDA_COMPILER={nvcc}', f'-DCMAKE_CUDA_ARCHITECTURES={args.architectures}']
            offline_records = getattr(args, 'offline_records', None)
            if offline_records is not None:
                report_path = offline_records / schedule / 'report.json'
                self.report['reused_offline_build'] = {
                    'report_sha256': digest(report_path),
                    'files': verify_offline_build(report_path, build, args.openfhe_prefix.resolve(),
                                                  self.report['source_commit'], schedule, args.architectures)}
            else:
                self.run('configure', configure)
            cache = build / 'CMakeCache.txt'
            if cache.is_file():
                keys = {'CMAKE_CXX_COMPILER', 'CMAKE_CUDA_COMPILER', 'CMAKE_CUDA_HOST_COMPILER',
                        'CMAKE_CUDA_ARCHITECTURES', 'CMAKE_BUILD_TYPE', 'OpenFHE_DIR',
                        'CMAKE_CXX_FLAGS', 'CMAKE_CXX_FLAGS_RELEASE', 'CMAKE_CUDA_FLAGS', 'CMAKE_CUDA_FLAGS_RELEASE',
                        'NTT_REQUIRE_CUDA', 'NTT_ENABLE_CUDA', 'NTT_ENABLE_OPENFHE_COMPAT',
                        'NTT_CUDA_LEGACY_BUTTERFLY'}
                self.report['configured_cache'] = {}
                for line in cache.read_text().splitlines():
                    if '=' in line and ':' in line.split('=', 1)[0]:
                        name = line.split(':', 1)[0]
                        if name in keys:
                            self.report['configured_cache'][name] = line.split('=', 1)[1]
            if schedule and self.report.get('configured_cache', {}).get(
                    'NTT_CUDA_LEGACY_BUTTERFLY') != legacy_option:
                raise RuntimeError('CMake cache does not confirm requested NTT schedule')
            if offline_records is None:
                self.run('build', [args.cmake, '--build', build, '-j', str(args.jobs)])
            registered = json.loads(self.run('registered-tests', [args.ctest, '--test-dir', build, '--show-only=json-v1']))
            tests = {test['name']: test for test in registered['tests']}
            required = OPENFHE_TESTS | HOST_TESTS
            if args.mode != 'cpu':
                required |= GPU_TESTS | {'cuda_constructor_preflight', 'resident_cuda_preflight'}
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
                if schedule and not self.report['source_dirty'] and cache.is_file():
                    prefix = args.openfhe_prefix.resolve()
                    self.report.update(source_dir=str(ROOT), openfhe_prefix=str(prefix),
                                       offline_build_sha256=offline_build_hashes(build, prefix))
                self.report['status'] = 'OFFLINE_READY'
            else:
                self.report['registered_gpu_tests'] = sorted(GPU_TESTS)
                self.report['selected_device'] = self.run('device-info', [build / 'ntt_cuda_cli', 'info']).strip()
                if schedule and f'transform schedule: {schedule}' not in self.report['selected_device'].splitlines():
                    raise RuntimeError('CUDA executable does not report requested NTT schedule')
                self.report['gpu_execution'] = 'RUNNING'
                exact = [args.ctest, '--test-dir', build, '--output-on-failure']
                if benchmark_scope(args) in ('gpu', 'gpu-chain', 'gpu-conversion'):
                    # Native tests still run their OpenFHE references and CTest fixtures.
                    exact += ['-L', 'gpu']
                    if offline_records is not None:
                        exact += ['-FS', '^(VECTORS|VECTORS_N64K)$']
                    self.report['exact_test_label'] = 'gpu'
                self.run('exact-tests', exact)
                self.report['gpu_execution'] = 'EXACT_TESTS_PASS'
                if not args.skip_sanitizers:
                    self.report['sanitizers'] = 'RUNNING'
                    conversion_smoke = None
                    conversion_reference = []
                    if benchmark_scope(args) == 'gpu-conversion':
                        conversion_smoke = self.output / 'conversion-smoke-n4096-k40-b60.txt'
                        stage_vector(self, 'conversion-smoke-parameters', conversion_smoke, 4096, 40, 60)
                        if getattr(args, 'conversion_references', None) is not None:
                            reference, checksum = select_conversion_reference(args.conversion_references, 4096, 40, 60)
                            conversion_reference = [reference]
                            self.report['conversion_smoke_reference_sha256'] = checksum
                    for sanitizer in ['memcheck', 'racecheck', 'initcheck', 'synccheck']:
                        self.run('sanitizer-' + sanitizer, ['compute-sanitizer', '--tool', sanitizer,
                            '--error-exitcode', '1', build / 'ntt_cuda_openfhe_tests', '--smoke'])
                        self.run('sanitizer-matrix-' + sanitizer, ['compute-sanitizer', '--tool', sanitizer,
                            '--error-exitcode', '1', build / 'ntt_cuda_openfhe_tests', '--matrix-smoke'])
                        self.run('sanitizer-resident-' + sanitizer, ['compute-sanitizer', '--tool', sanitizer,
                            '--error-exitcode', '1', build / 'ntt_resident_cuda_tests', '--smoke'])
                        self.run('sanitizer-basis-' + sanitizer, ['compute-sanitizer', '--tool', sanitizer,
                            '--error-exitcode', '1', build / 'ntt_basis_cuda_tests', '--smoke'])
                        self.run('sanitizer-shoup-' + sanitizer, ['compute-sanitizer', '--tool', sanitizer,
                            '--error-exitcode', '1', build / 'ntt_shoup_cuda_tests'])
                        self.run('sanitizer-barrett-' + sanitizer, ['compute-sanitizer', '--tool', sanitizer,
                            '--error-exitcode', '1', build / 'ntt_barrett_cuda_tests'])
                        if conversion_smoke:
                            self.run('sanitizer-conversion-benchmark-' + sanitizer,
                                ['compute-sanitizer', '--tool', sanitizer, '--error-exitcode', '1',
                                 build / 'ntt_resident_conversion_bench', conversion_smoke, '1', '2',
                                 *conversion_reference])
                    self.report['sanitizers'] = 'SMOKE_MATRIX_PASS'
                    self.report['sanitizer_scopes'] = ['baseline_smoke', 'matrix_smoke', 'resident_smoke', 'basis_smoke', 'shoup_exact', 'barrett_exact']
                    if conversion_smoke:
                        self.report['sanitizer_scopes'].append('conversion_benchmark_smoke')
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


def compare_ntt(args):
    """Run the standard gates twice and pair only matching validated workloads."""
    output, build = args.output_dir.resolve(), args.build_dir.resolve()
    if output == build or output in build.parents or build in output.parents:
        raise ValueError('Comparison build and record roots must be disjoint')
    for schedule in NTT_SCHEDULES:
        child = build / schedule
        if getattr(args, 'offline_records', None) is not None:
            if child.is_symlink() or not child.is_dir():
                raise ValueError('Prepared build directory is missing: ' + str(child))
        elif child.is_symlink() or child.exists():
            raise ValueError('Comparison requires fresh build directories: ' + str(child))
    output.mkdir(parents=True, exist_ok=False)
    index = output / 'comparison.json'
    record = {'mode': args.mode, 'status': 'RUNNING', 'runs': {}}
    atomic_json(index, record)
    sessions = []
    try:
        for schedule in NTT_SCHEDULES:
            child_args = argparse.Namespace(**vars(args))
            child_args.ntt_schedule = schedule
            child_args.build_dir, child_args.output_dir = build / schedule, output / schedule
            session = Session(child_args)
            result = session.execute()
            sessions.append(session)
            record['runs'][schedule] = {'status': session.report['status'],
                                        'report': f'{schedule}/report.json'}
            atomic_json(index, record)
            if result == 130:
                raise KeyboardInterrupt
        expected = 'OFFLINE_READY' if args.mode == 'offline' else 'GPU_CHECKS_PASSED'
        if any(session.report['status'] != expected for session in sessions):
            raise RuntimeError('One or more child sessions failed; records retained')
        if len({(s.report['source_commit'], s.report['source_dirty']) for s in sessions}) != 1:
            raise RuntimeError('Source changed between builds')
        if args.mode == 'gpu':
            left, right = sessions
            def context(session):
                data = json.loads((session.output / 'benchmark-manifest.json').read_text())['context']
                binaries = data.pop('binaries')
                data.pop('transform_schedule')
                data['selected_device'] = [line for line in data['selected_device'].splitlines()
                                           if not line.startswith('transform schedule: ')]
                return data, binaries
            left_context, left_binaries = context(left)
            right_context, right_binaries = context(right)
            if (left_context != right_context or left_binaries.keys() != right_binaries.keys()
                    or not left_binaries
                    or any(value == right_binaries[name] for name, value in left_binaries.items())):
                raise RuntimeError('Comparison requires the same environment and distinct binaries')
            def workloads(session):
                return {name: {key: value for key, value in data['workload'].items()
                               if key != 'transform_schedule'}
                        for name, data in session.report['benchmark_summaries'].items()}
            matched = workloads(left)
            if (not matched or len(matched) != left.report['benchmark_expected_jobs']
                    or len(matched) != right.report['benchmark_expected_jobs']
                    or matched != workloads(right)):
                raise RuntimeError('Compared benchmark workloads do not match')
            record['matched_jobs'] = sorted(matched)
        record['status'] = 'OFFLINE_READY' if args.mode == 'offline' else 'MATCHED_GPU_RUNS'
        return 0
    except KeyboardInterrupt:
        record['status'] = 'INTERRUPTED'
        return 130
    except Exception as error:
        record.update(status='FAIL', error=str(error))
        print(error, file=sys.stderr)
        return 1
    finally:
        atomic_json(index, record)
        print(f'Comparison: {index} ({record["status"]})', flush=True)


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
    result.add_argument('--prepared-vectors', type=Path, help='Use a CPU-prepared vector directory')
    result.add_argument('--conversion-references', type=Path,
                        help='Use CPU-prepared OpenFHE arrays for conversion benchmarking')
    result.add_argument('--offline-records', type=Path,
                        help='Reuse verified builds from an offline paired run in GPU comparison mode')
    result.add_argument('--ntt-schedule', choices=NTT_SCHEDULES, help='Select and verify a single CUDA schedule')
    result.add_argument('--compare-ntt', action='store_true', help='Build and record both CUDA schedules separately')
    result.add_argument('--benchmark-prime-bits', type=int, nargs='+', choices=PRIME_BITS, default=list(PRIME_BITS))
    result.add_argument('--benchmark-scope', choices=SCOPE_GROUPS,
                        help='Default: all in GPU mode, cpu in CPU mode; gpu selects public/events; gpu-chain adds the NTT chain; gpu-conversion adds ModUp/ModDown')
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


def validate_prepared_inputs(args):
    if args.conversion_references is not None:
        if not (args.mode == 'gpu' and args.benchmark and args.benchmark_scope == 'gpu-conversion'):
            raise ValueError('--conversion-references requires GPU conversion benchmarking')
        cells = set(product(args.benchmark_dimensions, args.benchmark_towers, args.benchmark_prime_bits))
        if not args.skip_sanitizers:
            cells.add((4096, 40, 60))
        for cell in sorted(cells):
            select_conversion_reference(args.conversion_references, *cell)
    if args.prepared_vectors is not None:
        if not (args.benchmark or args.profile):
            raise ValueError('--prepared-vectors requires --benchmark or --profile')
        hashes = vector_hashes(args.prepared_vectors)
        cells = set()
        if args.benchmark:
            cells.update(product(args.benchmark_dimensions, args.benchmark_towers, args.benchmark_prime_bits))
        if args.benchmark_scope == 'gpu-conversion' and not args.skip_sanitizers:
            cells.add((4096, 40, 60))
        if args.profile:
            cells.add((65536, 3, 60))
        for n, k, bits in sorted(cells):
            check_vector(args.prepared_vectors, hashes, n, k, bits)


def main(argv=None, *, deadline=None):
    cli = parser()
    args = cli.parse_args(argv)
    args.deadline_monotonic = deadline
    if args.compare_ntt:
        if (args.mode == 'cpu' or args.ntt_schedule or args.resume or args.allow_dirty
                or args.profile or args.skip_sanitizers):
            cli.error('--compare-ntt requires fresh offline/GPU records; no schedule override, dirty runs, profiling or skipped sanitizers')
        if args.mode == 'gpu':
            if not args.benchmark or args.benchmark_scope not in (None, 'gpu', 'gpu-chain', 'gpu-conversion'):
                cli.error('GPU comparison requires --benchmark and GPU-only scope')
            args.benchmark_scope = args.benchmark_scope or 'gpu'
    if args.offline_records is not None:
        if args.mode != 'gpu' or not args.compare_ntt or args.architectures in (None, 'native'):
            cli.error('--offline-records requires GPU comparison with an explicit architecture')
        if not args.offline_records.is_dir():
            cli.error('--offline-records directory does not exist')
    if args.ntt_schedule and args.mode == 'cpu':
        cli.error('--ntt-schedule requires offline or GPU mode')
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
    if args.benchmark_scope in ('gpu-chain', 'gpu-conversion') and args.benchmark_warmups + args.benchmark_samples > 200:
        cli.error('resident chain allows at most 200 warmups and samples combined')
    if not 1 <= args.command_timeout <= 7200:
        cli.error('--command-timeout must be 1..7200 seconds')
    if any(len(set(axis)) != len(axis) for axis in
           (args.benchmark_dimensions, args.benchmark_towers, args.benchmark_prime_bits)):
        cli.error('benchmark matrix axes must not contain duplicates')
    try:
        validate_prepared_inputs(args)
    except (OSError, ValueError, TypeError) as error:
        cli.error(str(error))
    if args.resume and (args.output_dir is None or not args.benchmark or args.allow_dirty):
        cli.error('--resume requires --output-dir, --benchmark and a clean checkout')
    if args.mode == 'offline' and (args.benchmark or args.profile or args.resume or args.skip_sanitizers or args.architectures == 'native'):
        cli.error('Offline mode cannot execute a benchmark or use GPU/native execution flags')
    if args.mode != 'cpu':
        args.architectures = args.architectures or ('89;120' if args.mode == 'offline' else 'native')
    args.build_dir = args.build_dir or ROOT / {'offline': 'build/cuda-offline13',
                                             'gpu': 'build/native-gpu', 'cpu': 'build/native-cpu'}[args.mode]
    args.output_dir = args.output_dir or ROOT / ('bench-results-' + args.mode + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    if args.compare_ntt:
        try:
            return compare_ntt(args)
        except (ValueError, OSError) as error:
            cli.error(str(error))
    return Session(args).execute()


if __name__ == '__main__':
    sys.exit(main())
