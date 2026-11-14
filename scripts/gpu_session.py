#!/usr/bin/env python3
"""Compile/test without a GPU, or collect an explicit native GPU session record."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
from benchmark_report import validate_jsonl

ROOT = Path(__file__).resolve().parents[1]
GPU_TESTS = {'cuda_oracle_crosscheck', 'cuda_matches_ref', 'cuda_verify', 'cuda_openfhe_exact', 'cuda_setup_fault_injection'}
OPENFHE_TESTS = {'openfhe_exact_compat', 'openfhe_representation_contract', 'openfhe_hybrid_graph'}


def tool(name):
    local = ROOT / '.venv/bin' / name
    return str(local) if local.is_file() else name


class Session:
    def __init__(self, args):
        self.args = args
        self.output = args.output_dir.resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.report = {'mode': args.mode, 'status': 'RUNNING', 'started_utc': datetime.now(timezone.utc).isoformat(),
                       'platform': platform.platform(), 'build_dir': str(args.build_dir.resolve()),
                       'stages': [], 'gpu_execution': 'NOT_RUN', 'sanitizers': 'NOT_RUN', 'benchmark': 'NOT_RUN'}
        self.save()

    def save(self):
        temporary = self.output / 'report.json.tmp'
        temporary.write_text(json.dumps(self.report, indent=2) + '\n')
        temporary.replace(self.output / 'report.json')

    def run(self, name, command):
        command = [str(x) for x in command]
        print(f'[{name}] {" ".join(command)}', flush=True)
        log = self.output / (name + '.log')
        stage = {'name': name, 'command': command, 'log': log.name, 'status': 'RUNNING'}
        self.report['stages'].append(stage)
        self.save()
        try:
            with log.open('w') as output:
                result = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
            stage.update(status='PASS' if result.returncode == 0 else 'FAIL', exit_code=result.returncode)
        except OSError as error:
            stage.update(status='FAIL', error=str(error))
            self.save()
            raise RuntimeError(f'{name}: {error}') from error
        self.save()
        if result.returncode:
            raise RuntimeError(f'{name} failed ({result.returncode}); see {log}')
        return log.read_text(errors='replace')

    def execute(self):
        args = self.args
        try:
            self.report['source_commit'] = self.run('source', ['git', 'rev-parse', 'HEAD']).strip()
            self.report['source_dirty'] = bool(self.run('source-state', ['git', 'status', '--porcelain']).strip())
            if args.mode == 'gpu' and self.report['source_dirty'] and not args.allow_dirty:
                raise RuntimeError('Native records require a clean checkout; --allow-dirty marks an exploratory run')
            self.run('cmake-version', [args.cmake, '--version'])
            self.run('compiler-version', ['c++', '--version'])
            nvcc = args.cuda_root.resolve() / 'bin/nvcc'
            self.run('nvcc-version', [nvcc, '--version'])
            if args.mode == 'gpu':
                self.run('hardware', ['nvidia-smi', '--query-gpu=name,driver_version,memory.total,pstate,power.limit,clocks.sm,clocks.mem', '--format=csv'])
                if not args.skip_sanitizers:
                    self.run('sanitizer-version', ['compute-sanitizer', '--version'])
            build = args.build_dir.resolve()
            self.run('configure', [args.cmake, '-S', ROOT, '-B', build,
                '-DCMAKE_BUILD_TYPE=Release', '-DNTT_REQUIRE_CUDA=ON', '-DNTT_ENABLE_CUDA=ON',
                f'-DCMAKE_CUDA_COMPILER={nvcc}', f'-DCMAKE_CUDA_ARCHITECTURES={args.architectures}',
                '-DNTT_ENABLE_OPENFHE_COMPAT=ON', f'-DCMAKE_PREFIX_PATH={args.openfhe_prefix.resolve()}'])
            cache = build / 'CMakeCache.txt'
            if cache.is_file():
                keys = {'CMAKE_CXX_COMPILER', 'CMAKE_CUDA_COMPILER', 'CMAKE_CUDA_HOST_COMPILER',
                        'CMAKE_CUDA_ARCHITECTURES', 'CMAKE_BUILD_TYPE', 'OpenFHE_DIR',
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
            missing = (GPU_TESTS | OPENFHE_TESTS | {'cuda_constructor_preflight', 'cuda_host_validation',
                        'setup_ownership_host', 'measurement_host'}) - tests.keys()
            if missing:
                raise RuntimeError('Required tests missing (CPU-only/incomplete builds are rejected): ' + ', '.join(sorted(missing)))
            for name in GPU_TESTS:
                labels = next((p['value'] for p in tests[name]['properties'] if p['name'] == 'LABELS'), [])
                if 'gpu' not in labels:
                    raise RuntimeError(f'{name} lacks its mandatory GPU label')
            self.report['registered_gpu_tests'] = sorted(GPU_TESTS)
            if args.mode == 'offline':
                if args.architectures == 'native':
                    raise RuntimeError('Offline compilation requires explicit architecture numbers, not native')
                self.run('host-tests', [args.ctest, '--test-dir', build, '-LE', 'gpu', '--output-on-failure'])
                self.report['status'] = 'OFFLINE_READY'
            else:
                self.run('device-info', [build / 'ntt_cuda_cli', 'info'])
                self.report['gpu_execution'] = 'RUNNING'
                self.run('exact-tests', [args.ctest, '--test-dir', build, '--output-on-failure'])
                self.report['gpu_execution'] = 'EXACT_TESTS_PASS'
                if not args.skip_sanitizers:
                    self.report['sanitizers'] = 'RUNNING'
                    for sanitizer in ['memcheck', 'racecheck', 'initcheck', 'synccheck']:
                        self.run('sanitizer-' + sanitizer, ['compute-sanitizer', '--tool', sanitizer,
                            '--error-exitcode', '1', build / 'ntt_cuda_openfhe_tests', '--smoke'])
                    self.report['sanitizers'] = 'SMOKE_MATRIX_PASS'
                else:
                    self.report['sanitizers'] = 'EXPLICITLY_SKIPPED'
                if args.benchmark:
                    self.report['benchmark'] = 'RUNNING'
                    self.report['benchmark_summaries'] = {}
                    for bits in (50, 55, 60):
                        vector = self.output / f'benchmark-vector-b{bits}.txt'
                        self.run(f'benchmark-parameters-b{bits}', [sys.executable,
                            ROOT / 'tests/oracle/gen_vectors.py', vector, '65536', '3', '1', str(bits)])
                        raw = self.run(f'benchmark-b{bits}', [build / 'ntt_cuda_cli', 'bench-jsonl',
                            vector, str(args.benchmark_warmups), str(args.benchmark_samples)])
                        summary = validate_jsonl(raw, expected_bits=bits)
                        if summary['workload'].get('simulation'):
                            raise RuntimeError('synthetic benchmark cannot count as native evidence')
                        if summary['workload']['samples'] != args.benchmark_samples or summary['workload']['warmups'] != args.benchmark_warmups:
                            raise RuntimeError('benchmark ignored requested repetition counts')
                        self.report['benchmark_summaries'][str(bits)] = summary
                    self.report['benchmark'] = 'PASS_50_55_60_BIT_PUBLIC_CALL_SAMPLES'
                self.report['status'] = 'GPU_CHECKS_PASSED'
                self.report['remaining_gates'] = ['Actual driver OOM/transfer errors and failed-context cleanup beyond pre-call injection',
                    'Kernel-only events, optimized OpenFHE CPU timing and Nsight profiling',
                    'Full production parameter/key matrix and optimized runtime integration']
            return 0
        except Exception as error:
            for field in ['gpu_execution', 'sanitizers', 'benchmark']:
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
    result.add_argument('--mode', choices=['offline', 'gpu'], required=True)
    result.add_argument('--cuda-root', type=Path, required=True)
    result.add_argument('--openfhe-prefix', type=Path, required=True)
    result.add_argument('--build-dir', type=Path)
    result.add_argument('--output-dir', type=Path)
    result.add_argument('--architectures', help='Default: 89;120 offline, native on a GPU')
    result.add_argument('--jobs', type=int, default=2)
    result.add_argument('--cmake', default=tool('cmake'))
    result.add_argument('--ctest', default=tool('ctest'))
    result.add_argument('--benchmark', action='store_true')
    result.add_argument('--benchmark-warmups', type=int, default=5)
    result.add_argument('--benchmark-samples', type=int, default=100)
    result.add_argument('--skip-sanitizers', action='store_true')
    result.add_argument('--allow-dirty', action='store_true')
    return result


def main():
    cli = parser()
    args = cli.parse_args()
    if args.jobs < 1:
        cli.error('--jobs must be positive')
    if not 1 <= args.benchmark_warmups <= 10000 or not 2 <= args.benchmark_samples <= 100000:
        cli.error('benchmark requires 1..10000 warmups and 2..100000 samples')
    if args.mode == 'offline' and (args.benchmark or args.skip_sanitizers or args.architectures == 'native'):
        cli.error('Offline mode cannot execute a benchmark or use GPU/native execution flags')
    args.architectures = args.architectures or ('89;120' if args.mode == 'offline' else 'native')
    args.build_dir = args.build_dir or ROOT / ('build/cuda-offline13' if args.mode == 'offline' else 'build/native-gpu')
    args.output_dir = args.output_dir or ROOT / ('bench-results-' + args.mode + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    return Session(args).execute()


if __name__ == '__main__':
    sys.exit(main())
