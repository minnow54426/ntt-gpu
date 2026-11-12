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
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('gpu_session', ROOT / 'scripts/gpu_session.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SessionTests(unittest.TestCase):
    def scenario(self, mode='offline', fail=None, missing_gpu=False, bad_label=False, skip=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = module.parser().parse_args(['--mode', mode, '--cuda-root', str(root / 'toolkit'),
                '--openfhe-prefix', str(root / 'openfhe'), '--build-dir', str(root / 'build'),
                '--output-dir', str(root / 'record')])
            args.architectures = '89;120' if mode == 'offline' else 'native'
            args.benchmark = mode == 'gpu'
            args.skip_sanitizers = skip
            commands = []
            names = module.OPENFHE_TESTS | {'cuda_constructor_preflight', 'cuda_host_validation'}
            if not missing_gpu:
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
                else:
                    output.write('mock command output\n')
                return SimpleNamespace(returncode=19 if fail and fail(command) else 0)

            with patch.object(module.platform, 'platform', return_value='mock platform'), patch.object(module.subprocess, 'run', side_effect=run), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
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
        self.assertFalse(any('bench' in c for c in commands))

    def test_benchmark_failure_cannot_return_success(self):
        code, report, _ = self.scenario(mode='gpu', fail=lambda c: 'bench' in c)
        self.assertEqual(code, 1)
        self.assertEqual(report['status'], 'FAIL')
        self.assertEqual(report['benchmark'], 'FAIL')
        self.assertEqual(report['gpu_execution'], 'EXACT_TESTS_PASS')

    def test_device_probe_failure_stops_runtime_tests(self):
        code, report, commands = self.scenario(mode='gpu', fail=lambda c: c[-1] == 'info')
        self.assertEqual(code, 1)
        self.assertEqual(report['gpu_execution'], 'NOT_RUN')
        self.assertFalse(any('--output-on-failure' in c for c in commands))

    def test_sanitizer_failure_is_preserved(self):
        code, report, commands = self.scenario(mode='gpu', fail=lambda c: 'racecheck' in c)
        self.assertEqual(code, 1)
        self.assertEqual(report['sanitizers'], 'FAIL')
        self.assertFalse(any('bench' in c for c in commands))

    def test_skipped_sanitizers_are_explicit(self):
        code, report, commands = self.scenario(mode='gpu', skip=True)
        self.assertEqual(code, 0)
        self.assertEqual(report['sanitizers'], 'EXPLICITLY_SKIPPED')
        self.assertIn('remaining_gates', report)
        self.assertFalse(any(c[0] == 'compute-sanitizer' for c in commands))

    def test_missing_executable_records_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output_dir=Path(directory) / 'new', build_dir=Path(directory), mode='offline')
            with patch.object(module.platform, 'platform', return_value='mock platform'), patch.object(module.subprocess, 'run', side_effect=FileNotFoundError('missing program')), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
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
