"""Failure propagation and coverage classification with mocked subprocesses.

These tests execute the real session/reporting logic, never a simulated GPU
arithmetic implementation or real CUDA runtime.
"""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import sys
import subprocess
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_report import TIMING_OPERATIONS
from benchmark_fixture import conversion_parameters
spec = importlib.util.spec_from_file_location('gpu_session', ROOT / 'scripts/gpu_session.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SessionTests(unittest.TestCase):
    def test_comparison_pairs_only_matching_runs_and_retains_both_reports(self):
        for changed in (None, 'roots', 'device', 'same-binary', 'same-chain', 'failure'):
            with tempfile.TemporaryDirectory() as directory, self.subTest(changed=changed):
                directory = Path(directory)
                args = SimpleNamespace(mode='gpu', output_dir=directory / 'record',
                                       build_dir=directory / 'build')
                seen = []

                class SyntheticSession:
                    def __init__(self, options):
                        self.output = options.output_dir
                        self.output.mkdir()
                        self.options = options
                        seen.append(options)

                    def execute(self):
                        legacy = self.options.ntt_schedule == 'legacy_dif'
                        failed = changed == 'failure' and not legacy
                        self.report = {
                            'source_commit': 'mock', 'source_dirty': False,
                            'status': 'FAIL' if failed else 'GPU_CHECKS_PASSED',
                            'benchmark_expected_jobs': 1,
                            'benchmark_summaries': {'paired-job': {'workload': {
                                'roots': [3 if legacy and changed == 'roots' else 2],
                                'transform_schedule': self.options.ntt_schedule}}},
                        }
                        (self.output / 'report.json').write_text(json.dumps(self.report))
                        identity = 'other device' if legacy and changed == 'device' else 'same device'
                        context = {'hardware_identity': identity,
                                   'selected_device': 'device\ntransform schedule: ' + self.options.ntt_schedule,
                                   'transform_schedule': self.options.ntt_schedule,
                                   'binaries': {
                                       'ntt_cuda_cli': 'A' if changed == 'same-binary' else self.options.ntt_schedule,
                                       'ntt_resident_bench': 'B' if changed == 'same-chain' else self.options.ntt_schedule,
                                   }}
                        (self.output / 'benchmark-manifest.json').write_text(json.dumps({'context': context}))
                        return int(failed)

                with patch.object(module, 'Session', SyntheticSession):
                    self.assertEqual(module.compare_ntt(args), int(changed is not None))
                self.assertEqual([item.ntt_schedule for item in seen], list(module.NTT_SCHEDULES))
                self.assertEqual(len({item.build_dir for item in seen}), 2)
                self.assertEqual(len({item.output_dir for item in seen}), 2)
                record = json.loads((args.output_dir / 'comparison.json').read_text())
                self.assertEqual(set(record['runs']), set(module.NTT_SCHEDULES))
                self.assertEqual(record['status'], 'FAIL' if changed else 'MATCHED_GPU_RUNS')
                self.assertEqual(record.get('matched_jobs'), None if changed else ['paired-job'])
                with self.assertRaises(FileExistsError):
                    module.compare_ntt(args)

    def test_compare_cli_uses_gpu_only_scope(self):
        options = ['--mode', 'gpu', '--cuda-root', '/mock/cuda',
                   '--openfhe-prefix', '/mock/openfhe', '--compare-ntt', '--benchmark',
                   '--benchmark-prime-bits', '30', '60']
        with patch.object(module, 'compare_ntt', return_value=0) as compare:
            self.assertEqual(module.main(options), 0)
            args = compare.call_args.args[0]
            self.assertEqual(args.benchmark_scope, 'gpu')
            self.assertEqual(args.benchmark_prime_bits, [30, 60])
        with patch.object(module, 'compare_ntt', return_value=0) as compare:
            self.assertEqual(module.main(options + ['--benchmark-scope', 'gpu-chain']), 0)
            self.assertEqual(compare.call_args.args[0].benchmark_scope, 'gpu-chain')
        with patch.object(module, 'compare_ntt', return_value=0) as compare:
            self.assertEqual(module.main(options + ['--benchmark-scope', 'gpu-conversion']), 0)
            self.assertEqual(compare.call_args.args[0].benchmark_scope, 'gpu-conversion')
        for scope in ('gpu-chain', 'gpu-conversion'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                module.main(options + ['--benchmark-scope', scope, '--benchmark-samples', '200'])
        for flag in ('--skip-sanitizers', '--profile', '--allow-dirty'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                module.main(options + [flag])

    def test_cli_cpu_needs_no_cuda_and_uses_its_own_build_directory(self):
        with patch.object(module, 'Session') as session:
            session.return_value.execute.return_value = 0
            self.assertEqual(module.main(['--mode', 'cpu', '--openfhe-prefix', '/mock/openfhe', '--benchmark']), 0)
            args = session.call_args.args[0]
            self.assertIsNone(args.cuda_root)
            self.assertIsNone(args.architectures)
            self.assertEqual(module.benchmark_scope(args), 'cpu')
            self.assertEqual(args.build_dir, ROOT / 'build/native-cpu')

    def test_invalid_mode_scope_and_gpu_flag_combinations_fail_before_a_session(self):
        cases = [
            ['--mode', 'gpu'],
            ['--mode', 'offline'],
            ['--mode', 'cpu', '--benchmark', '--benchmark-scope', 'gpu'],
            ['--mode', 'cpu', '--benchmark', '--benchmark-scope', 'all'],
            ['--mode', 'cpu', '--cuda-root', '/unused/cuda'],
            ['--mode', 'cpu', '--profile'],
            ['--mode', 'cpu', '--skip-sanitizers'],
            ['--mode', 'cpu', '--architectures', 'native'],
            ['--mode', 'cpu', '--benchmark-scope', 'cpu'],
            ['--mode', 'gpu', '--cuda-root', '/mock/cuda', '--benchmark', '--benchmark-scope', 'cpu'],
        ]
        for args in cases:
            with self.subTest(args=args), patch.object(module, 'Session') as session, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                module.main(args + ['--openfhe-prefix', '/mock/openfhe'])
            session.assert_not_called()

    def test_resume_scope_or_mode_rejection_preserves_the_original_report(self):
        for mode, scope in [('gpu', 'all'), ('cpu', 'cpu')]:
            with tempfile.TemporaryDirectory() as directory, self.subTest(mode=mode):
                args = SimpleNamespace(output_dir=Path(directory) / 'record', build_dir=Path(directory),
                                       mode='gpu', benchmark_scope='gpu', resume=False)
                first = module.Session(args)
                original = (first.output / 'report.json').read_bytes()
                args.mode, args.benchmark_scope, args.resume = mode, scope, True
                with self.assertRaises(ValueError):
                    module.Session(args)
                self.assertEqual((first.output / 'report.json').read_bytes(), original)
                self.assertFalse(list(first.output.glob('report-history-*')))

    def test_cpu_resume_preserves_attempt_history(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output_dir=Path(directory) / 'record', build_dir=Path(directory),
                                   mode='cpu', benchmark_scope=None, resume=False)
            first = module.Session(args)
            args.resume = True
            second = module.Session(args)
            self.assertEqual(second.report['attempt'], first.report['attempt'] + 1)
            self.assertEqual(second.report['benchmark_scope'], 'cpu')
            self.assertEqual(len(list(first.output.glob('report-history-*'))), 1)

    def test_cpu_identity_is_stable_across_clock_changes_and_supports_macos(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output_dir=Path(directory) / 'record', build_dir=Path(directory), mode='cpu')
            session = module.Session(args)
            fields = [{'field': 'Architecture:', 'data': 'x86_64'},
                      {'field': 'Model name:', 'data': 'Mock CPU'},
                      {'field': 'CPU(s):', 'data': '8'},
                      {'field': 'CPU MHz:', 'data': '1234'}]
            with patch.object(module.platform, 'system', return_value='Linux'), patch.object(session, 'run') as run:
                run.return_value = json.dumps({'lscpu': fields})
                first = session.record_cpu_identity()
                fields[-1]['data'] = '5678'
                run.return_value = json.dumps({'lscpu': fields})
                self.assertEqual(session.record_cpu_identity(), first)
                fields[1]['data'] = 'Other CPU'
                run.return_value = json.dumps({'lscpu': fields})
                self.assertNotEqual(session.record_cpu_identity(), first)
                run.return_value = '{"lscpu": []}'
                with self.assertRaisesRegex(RuntimeError, 'identity missing'):
                    session.record_cpu_identity()
            with patch.object(module.platform, 'system', return_value='Darwin'), patch.object(session, 'run') as run:
                run.return_value = 'MacBookPro18,3\nApple M1 Pro\n8\n17179869184\n'
                identity = json.loads(session.record_cpu_identity())
                self.assertEqual(identity['machdep.cpu.brand_string'], 'Apple M1 Pro')
                self.assertNotIn('lscpu', run.call_args.args[1])
                run.return_value = 'incomplete\n'
                with self.assertRaisesRegex(RuntimeError, 'Incomplete macOS'):
                    session.record_cpu_identity()

    def test_prime_batch_defaults_and_bounds(self):
        required = ['--mode', 'gpu', '--cuda-root', '/mock/cuda', '--openfhe-prefix', '/mock/openfhe']
        self.assertEqual(module.parser().parse_args(required).benchmark_towers, [1, 3, 8, 16, 24, 32, 40])
        self.assertEqual(module.parser().parse_args(required + ['--benchmark-towers', '2', '39', '40']).benchmark_towers,
                         [2, 39, 40])
        for invalid in ['0', '41']:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                module.parser().parse_args(required + ['--benchmark-towers', invalid])

    def test_process_timeout_kills_its_own_group(self):
        process = MagicMock()
        process.pid = 12345
        process.wait.side_effect = [subprocess.TimeoutExpired('probe', 1), 0]
        process.__enter__.return_value = process
        with patch.object(module.subprocess, 'Popen', return_value=process), patch.object(module.os, 'killpg') as kill_group:
            with self.assertRaises(subprocess.TimeoutExpired):
                module.run_process(['probe'], timeout=1)
            if module.os.name == 'posix':
                kill_group.assert_called_once_with(12345, module.signal.SIGKILL)
        self.assertEqual(module.run_process([sys.executable, '-c', 'pass'], timeout=5).returncode, 0)

    def test_resume_preserves_report_and_existing_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'record'
            args = SimpleNamespace(output_dir=output, build_dir=Path(directory), mode='gpu', resume=False)
            with patch.object(module.platform, 'platform', return_value='mock platform'):
                first = module.Session(args)
                (output / 'source.log').write_text('old log')
                args.resume = True
                second = module.Session(args)
                self.assertEqual(second.report['attempt'], 2)
                self.assertEqual(len(list(output.glob('report-history-*.json'))), 1)
                with patch.object(module, 'run_process', return_value=SimpleNamespace(returncode=0)):
                    second.run('source', ['git', 'rev-parse', 'HEAD'])
                self.assertEqual((output / 'source.log').read_text(), 'old log')
                self.assertEqual(second.report['stages'][0]['log'], 'source-retry-2.log')
                self.assertEqual(first.report['attempt'], 1)

    def test_timeout_is_a_failed_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output_dir=Path(directory) / 'new', build_dir=Path(directory), mode='gpu')
            with patch.object(module.platform, 'platform', return_value='mock platform'):
                session = module.Session(args)
                with patch.object(module, 'run_process', side_effect=subprocess.TimeoutExpired('probe', 1)):
                    with self.assertRaises(RuntimeError): session.run('probe', ['probe'])
                self.assertEqual(session.report['stages'][0]['status'], 'FAIL')

    def scenario(self, mode='offline', fail=None, missing_gpu=False, bad_label=False, skip=False,
                 bad_benchmark=False, synthetic=False, scope=None, missing_openfhe=False,
                 unexpected_gpu=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = module.parser().parse_args(['--mode', mode, '--cuda-root', str(root / 'toolkit'),
                '--openfhe-prefix', str(root / 'openfhe'), '--build-dir', str(root / 'build'),
                '--output-dir', str(root / 'record')])
            args.architectures = '89;120' if mode == 'offline' else 'native'
            args.benchmark = mode in ('gpu', 'cpu')
            args.benchmark_scope = scope
            if mode == 'cpu':
                args.cuda_root = args.architectures = None
            args.skip_sanitizers = skip
            # Small matrix here; separate batch tests cover the full defaults.
            args.benchmark_dimensions = [65536]
            args.benchmark_towers = [3]
            (root / 'build').mkdir()
            (root / 'openfhe').mkdir()
            (root / 'openfhe/libOPENFHEcore.a').write_text('mock library')
            for binary in ['ntt_cuda_cli', 'ntt_openfhe_bench', 'ntt_resident_bench', 'ntt_resident_conversion_bench']:
                (root / 'build' / binary).write_text('mock executable bytes')
            commands = []
            names = module.OPENFHE_TESTS | module.HOST_TESTS | {'cuda_constructor_preflight', 'resident_cuda_preflight'}
            if mode == 'cpu':
                names -= {'cuda_constructor_preflight', 'resident_cuda_preflight'}
            if missing_openfhe:
                names -= {'openfhe_exact_compat'}
            if not missing_gpu and mode != 'cpu':
                names |= module.GPU_TESTS
            if unexpected_gpu:
                names |= module.GPU_TESTS
            registered = {'tests': [{'name': name, 'properties': [{'name': 'LABELS', 'value':
                ['gpu'] if name in module.GPU_TESTS and not bad_label else []}]} for name in sorted(names)]}

            def run(command, **kwargs):
                commands.append(command)
                output = kwargs['stdout']
                if '--show-only=json-v1' in command:
                    output.write(json.dumps(registered))
                elif command[0] == 'git' and 'rev-parse' in command:
                    output.write('mock-source-commit\n')
                elif command[0] == 'git':
                    output.write('')
                elif 'lscpu' in command:
                    output.write(json.dumps({'lscpu': [
                        {'field': 'Architecture:', 'data': 'x86_64'},
                        {'field': 'Model name:', 'data': 'Mock CPU'},
                        {'field': 'CPU(s):', 'data': '4'},
                        {'field': 'CPU MHz:', 'data': '1234'}]}))
                elif any(c.endswith('gen_vectors.py') for c in command):
                    vector, dimension, towers, _, bits = command[-5:]
                    dimension, towers, bits = int(dimension), int(towers), int(bits)
                    # Parameters only: benchmark collector does not consume coefficients.
                    Path(vector).write_text(f'{dimension} {towers}\n' + ''.join(
                        f'{(1 << (bits - 1)) + 1 + i * 131072} {i + 2}\n' for i in range(towers)))
                elif ('bench-jsonl' in command or 'gpu-bench-jsonl' in command or
                      'events-jsonl' in command or command[0].endswith(('ntt_openfhe_bench', 'ntt_resident_bench', 'ntt_resident_conversion_bench'))):
                    if bad_benchmark:
                        output.write('not JSONL\n')
                        return SimpleNamespace(returncode=0)
                    bits = int(Path(command[-3]).stem.split('b')[-1])
                    warmups, count = map(int, command[-2:])
                    timing = ('host_wall_gpu_public_call' if 'gpu-bench-jsonl' in command else
                              'host_wall_public_call' if 'bench-jsonl' in command else
                              'cuda_event_resident_and_pinned_copy' if 'events-jsonl' in command else
                              'host_wall_resident_chain' if command[0].endswith('ntt_resident_bench') else
                              'host_wall_resident_conversion' if command[0].endswith('ntt_resident_conversion_bench') else
                              'host_wall_openfhe_native')
                    operations = TIMING_OPERATIONS[timing]
                    metadata = {'type': 'workload', 'schema': 1, 'n': 65536,
                        'timing': timing, 'input_policy': 'fixed_seed_reset_outside_timer', 'seed': 7492057,
                        'setup_included': False, 'towers': 3, 'warmups': warmups, 'samples': count,
                        'moduli': [(1 << (bits-1)) + 1 + i*131072 for i in range(3)],
                        'roots': [2, 3, 4], 'simulation': synthetic}
                    if timing == 'host_wall_resident_conversion':
                        metadata.update(conversion_parameters(65536))
                    output.write(json.dumps(metadata) + '\n')
                    for operation in sorted(operations):
                        for index in range(count):
                            output.write(json.dumps({'type': 'sample', 'operation': operation,
                                'iteration': index, 'milliseconds': 1}) + '\n')
                        output.write(json.dumps({'type': 'summary', 'operation': operation, 'count': count,
                            'mean_ms': 1, 'median_ms': 1, 'p5_ms': 1, 'p95_ms': 1}) + '\n')
                    output.write(json.dumps({'type': 'complete', 'operations': len(operations)}) + '\n')
                else:
                    output.write('mock command output\n')
                return SimpleNamespace(returncode=19 if fail and fail(command) else 0)

            with patch.object(module.platform, 'platform', return_value='mock platform'), patch.object(module.platform, 'system', return_value='Linux'), patch.object(module, 'run_process', side_effect=run), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                session = module.Session(args)
                code = session.execute()
            report = json.loads((root / 'record/report.json').read_text())
            self.assertFalse((root / 'record/report.json.tmp').exists())
            return code, report, commands

    def test_offline_never_executes_gpu_and_reports_compile_only(self):
        code, report, commands = self.scenario()
        self.assertEqual(code, 0)
        self.assertEqual(report['status'], 'OFFLINE_READY')
        self.assertEqual(report['gpu_execution'], 'NOT_RUN')
        self.assertEqual(report['benchmark'], 'NOT_RUN')
        self.assertTrue(any('-LE' in c and 'gpu' in c for c in commands))
        self.assertFalse(any('nvidia-smi' == c[0] or c[0].endswith('ntt_cuda_cli') for c in commands))

    def test_cpu_session_builds_checks_and_benchmarks_without_any_cuda_tools(self):
        code, report, commands = self.scenario(mode='cpu')
        self.assertEqual(code, 0)
        self.assertEqual(report['status'], 'CPU_CHECKS_PASSED')
        self.assertEqual(report['cpu_execution'], 'EXACT_TESTS_PASS')
        self.assertEqual(report['gpu_execution'], 'NOT_RUN')
        self.assertEqual(report['sanitizers'], 'NOT_RUN')
        self.assertEqual(report['benchmark_scopes'], ['openfhe'])
        self.assertEqual(len(report['benchmark_summaries']), 4)
        self.assertFalse(any('nvcc' in str(arg) or 'nvidia-smi' in str(arg) or
            'ntt_cuda_cli' in str(arg) or 'compute-sanitizer' in str(arg) for c in commands for arg in c))
        self.assertTrue(any('-DNTT_ENABLE_CUDA=OFF' in c and '-DNTT_REQUIRE_CUDA=OFF' in c for c in commands))

    def test_cpu_rejects_missing_openfhe_or_unexpected_gpu_tests(self):
        for options in ({'missing_openfhe': True}, {'unexpected_gpu': True}):
            with self.subTest(options=options):
                code, report, commands = self.scenario(mode='cpu', **options)
                self.assertEqual(code, 1)
                self.assertEqual(report['gpu_execution'], 'NOT_RUN')
                self.assertFalse(any('--output-on-failure' in c for c in commands))

    def test_cpu_host_gate_or_measurement_failure_cannot_report_success(self):
        for fail in (lambda c: '--output-on-failure' in c,
                     lambda c: c[0].endswith('ntt_openfhe_bench')):
            code, report, _ = self.scenario(mode='cpu', fail=fail)
            self.assertEqual(code, 1)
            self.assertEqual(report['status'], 'FAIL')
            self.assertEqual(report['gpu_execution'], 'NOT_RUN')
            self.assertIn('FAIL', (report['cpu_execution'], report['benchmark']))

    def test_gpu_only_keeps_native_gates_and_sanitizers_but_omits_cpu_measurements(self):
        code, report, commands = self.scenario(mode='gpu', scope='gpu')
        self.assertEqual(code, 0)
        self.assertEqual(report['gpu_execution'], 'EXACT_TESTS_PASS')
        self.assertEqual(report['sanitizers'], 'SMOKE_MATRIX_PASS')
        self.assertEqual(report['sanitizer_scopes'], ['baseline_smoke', 'matrix_smoke', 'resident_smoke', 'basis_smoke', 'shoup_exact'])
        self.assertEqual(sum(len(c) > 1 and c[-1] == '--smoke' and
                            c[-2].endswith('ntt_resident_cuda_tests') for c in commands), 4)
        for executable in ('ntt_basis_cuda_tests', 'ntt_shoup_cuda_tests'):
            self.assertEqual(sum(c[0] == 'compute-sanitizer' and any(
                argument.endswith(executable) for argument in c) for c in commands), 4)
        self.assertEqual(report['benchmark_scopes'], ['gpu-public', 'events'])
        self.assertEqual(len(report['benchmark_summaries']), 8)
        self.assertFalse(any(c[0].endswith('ntt_openfhe_bench') for c in commands))
        self.assertEqual(sum('gpu-bench-jsonl' in c for c in commands), 4)
        self.assertFalse(any('bench-jsonl' in c for c in commands))
        gate = next(c for c in commands if '--output-on-failure' in c)
        self.assertEqual(gate[-2:], ['-L', 'gpu'])
        self.assertEqual(report['exact_test_label'], 'gpu')
        self.assertEqual(report['registered_gpu_tests'], sorted(module.GPU_TESTS))

    def test_conversion_selection_runs_after_native_gates(self):
        code, report, commands = self.scenario(mode='gpu', scope='gpu-conversion')
        self.assertEqual(code, 0)
        self.assertEqual(report['benchmark_scopes'], ['gpu-public', 'events', 'chain', 'conversion'])
        self.assertEqual(len(report['benchmark_summaries']), 16)
        self.assertIn('conversion_benchmark_smoke', report['sanitizer_scopes'])
        self.assertEqual(sum(stage['name'].startswith('sanitizer-conversion-benchmark-')
                             and stage['status'] == 'PASS' for stage in report['stages']), 4)
        self.assertEqual(sum(command[0].endswith('ntt_resident_conversion_bench') for command in commands), 4)
        gate = next(command for command in commands if '--output-on-failure' in command)
        self.assertEqual(gate[-2:], ['-L', 'gpu'])

    def test_conversion_smoke_failure_prevents_measurement(self):
        code, report, commands = self.scenario(mode='gpu', scope='gpu-conversion',
            fail=lambda command: command[0] == 'compute-sanitizer' and
                                 any(argument.endswith('ntt_resident_conversion_bench') for argument in command))
        self.assertEqual(code, 1)
        self.assertEqual(report['sanitizers'], 'FAIL')
        self.assertFalse(any(command[0].endswith('ntt_resident_conversion_bench') for command in commands))

    def test_cpu_only_test_list_cannot_pass(self):
        code, report, _ = self.scenario(missing_gpu=True)
        self.assertEqual(code, 1)
        self.assertEqual(report['status'], 'FAIL')
        self.assertIn('missing', report['error'])

    def test_gpu_labels_are_required_for_offline_exclusion(self):
        code, report, _ = self.scenario(bad_label=True)
        self.assertEqual(code, 1)
        self.assertIn('label', report['error'])

    def test_configure_failure_stops_before_build(self):
        code, report, commands = self.scenario(fail=lambda c: '-S' in c)
        self.assertEqual(code, 1)
        self.assertEqual(report['status'], 'FAIL')
        self.assertFalse(any('--build' in c for c in commands))

    def test_test_failure_stops_before_benchmark(self):
        code, report, commands = self.scenario(mode='gpu', fail=lambda c: '--output-on-failure' in c)
        self.assertEqual(code, 1)
        self.assertEqual(report['gpu_execution'], 'FAIL')
        self.assertFalse(any('bench-jsonl' in c for c in commands))

    def test_candidate_sanitizer_failure_stops_before_timing(self):
        for executable in ('ntt_basis_cuda_tests', 'ntt_shoup_cuda_tests'):
            code, report, commands = self.scenario(mode='gpu', fail=lambda command:
                command[0] == 'compute-sanitizer' and
                any(argument.endswith(executable) for argument in command))
            self.assertEqual(code, 1)
            self.assertEqual(report['sanitizers'], 'FAIL')
            self.assertFalse(any('bench-jsonl' in command for command in commands))

    def test_benchmark_failure_cannot_return_success(self):
        code, report, _ = self.scenario(mode='gpu', fail=lambda c: 'bench-jsonl' in c)
        self.assertEqual(code, 1)
        self.assertEqual(report['status'], 'FAIL')
        self.assertEqual(report['benchmark'], 'FAIL')
        self.assertEqual(report['gpu_execution'], 'EXACT_TESTS_PASS')

    def test_malformed_benchmark_cannot_return_success(self):
        code, report, _ = self.scenario(mode='gpu', bad_benchmark=True)
        self.assertEqual(code, 1)
        self.assertEqual(report['benchmark'], 'FAIL')

    def test_raw_benchmarks_cover_all_widths(self):
        code, report, commands = self.scenario(mode='gpu')
        self.assertEqual(code, 0)
        self.assertEqual(len(report['benchmark_summaries']), 12)
        for bits in [30, 50, 55, 60]:
            self.assertIn(f'benchmark-n65536-k3-b{bits}-public', report['benchmark_summaries'])
        self.assertEqual(sum('bench-jsonl' in command for command in commands), 4)
        self.assertEqual(sum('events-jsonl' in command for command in commands), 4)

    def test_synthetic_benchmark_cannot_count_as_native(self):
        code, report, _ = self.scenario(mode='gpu', synthetic=True)
        self.assertEqual(code, 1)
        self.assertEqual(report['benchmark'], 'FAIL')

    def test_device_probe_failure_stops_runtime_tests(self):
        code, report, commands = self.scenario(mode='gpu', fail=lambda c: c[-1] == 'info')
        self.assertEqual(code, 1)
        self.assertEqual(report['gpu_execution'], 'NOT_RUN')
        self.assertFalse(any('--output-on-failure' in c for c in commands))

    def test_sanitizer_failure_is_preserved(self):
        code, report, commands = self.scenario(mode='gpu', fail=lambda c: 'racecheck' in c)
        self.assertEqual(code, 1)
        self.assertEqual(report['sanitizers'], 'FAIL')
        self.assertFalse(any('bench-jsonl' in c for c in commands))

    def test_skipped_sanitizers_are_explicit(self):
        code, report, commands = self.scenario(mode='gpu', skip=True)
        self.assertEqual(code, 0)
        self.assertEqual(report['sanitizers'], 'EXPLICITLY_SKIPPED')
        self.assertIn('remaining_gates', report)
        self.assertFalse(any(c[0] == 'compute-sanitizer' for c in commands))

    def test_missing_executable_records_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output_dir=Path(directory) / 'new', build_dir=Path(directory), mode='offline')
            with patch.object(module.platform, 'platform', return_value='mock platform'), patch.object(module, 'run_process', side_effect=FileNotFoundError('missing program')), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                session = module.Session(args)
                self.assertEqual(session.execute(), 1)
            self.assertEqual(session.report['status'], 'FAIL')
            self.assertEqual(session.report['stages'][0]['status'], 'FAIL')

    def test_existing_record_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output_dir=Path(directory))
            with self.assertRaises(FileExistsError):
                module.Session(args)


if __name__ == '__main__':
    unittest.main()
