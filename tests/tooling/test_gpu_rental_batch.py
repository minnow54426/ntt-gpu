"""Batch orchestration tests use synthetic subprocesses, never native GPU evidence."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import gpu_rental_batch as batch
import gpu_session


def options(root, mode='gpu'):
    return type('Options', (), dict(mode=mode, cuda_root=Path('/mock/cuda'),
        openfhe_prefix=Path('/mock/openfhe'), output_dir=root / 'records',
        build_dir=root / 'build', budget_minutes=5, jobs=2, command_timeout=120,
        device_index=1))()


class BatchTests(unittest.TestCase):
    def test_plan_needs_no_gpu_and_schedules_all_scopes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with contextlib.redirect_stdout(io.StringIO()) as stream, patch.object(batch, 'Batch') as execute:
                self.assertEqual(batch.main(['--output-dir', str(root / 'out'),
                                              '--build-dir', str(root / 'build')]), 0)
            execute.assert_not_called()
            self.assertEqual(list(root.iterdir()), [])
            plan = json.loads(stream.getvalue())
            self.assertEqual(plan['status'], 'PLANNED')
            self.assertFalse(plan['rentals_created'])
            args = gpu_session.parser().parse_args(plan['collector_arguments'])
            self.assertEqual(args.benchmark_scope, 'gpu-conversion')
            self.assertTrue(args.compare_ntt)
            self.assertFalse(args.skip_sanitizers)
            self.assertEqual(args.benchmark_dimensions, [4096, 65536])
            self.assertEqual(args.benchmark_towers, [3, 40])
            self.assertEqual(args.benchmark_prime_bits, [30, 60])

    def test_offline_runs_only_collector_and_archives_not_run_gpu_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = options(Path(temporary), 'offline')
            with patch.object(gpu_session, 'main', return_value=0) as collect, \
                    patch.object(batch.Batch, 'preflight') as native, patch.object(batch.Batch, 'profiles') as profile:
                self.assertEqual(batch.Batch(args).execute(), 0)
            native.assert_not_called(); profile.assert_not_called()
            self.assertIn('--mode', collect.call_args.args[0])
            self.assertIn('offline', collect.call_args.args[0])
            self.assertNotIn('--benchmark', collect.call_args.args[0])
            self.assertIsNotNone(collect.call_args.kwargs['deadline'])
            report = json.loads((args.output_dir / 'batch.json').read_text())
            self.assertEqual(report['status'], 'OFFLINE_READY')
            self.assertEqual(report['gpu_execution'], 'NOT_RUN')
            verified = json.loads(args.output_dir.with_suffix('.tar.gz.verify.json').read_text())
            self.assertEqual(verified['status'], 'VERIFIED')

    def test_failed_or_interrupted_collector_skips_later_work_and_keeps_records(self):
        for result, status in [(1, 'FAIL'), (130, 'INTERRUPTED')]:
            with self.subTest(result=result), tempfile.TemporaryDirectory() as temporary:
                args = options(Path(temporary))
                with patch.object(batch.Batch, 'preflight', return_value='GPU-mock'), \
                        patch.object(batch.Batch, 'start_clock'), \
                        patch.object(gpu_session, 'main', return_value=result), \
                        patch.object(batch.Batch, 'repeat') as repeat, \
                        patch.object(batch.Batch, 'profiles') as profile, \
                        patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'prior'}):
                    self.assertEqual(batch.Batch(args).execute(), result)
                    self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], 'prior')
                repeat.assert_not_called(); profile.assert_not_called()
                report = json.loads((args.output_dir / 'batch.json').read_text())
                self.assertEqual(report['status'], status)
                self.assertIn('INCOMPLETE', report['gpu_execution'])
                self.assertTrue(args.output_dir.with_suffix('.tar.gz.sha256').exists())

    def test_native_preflight_selects_physical_gpu_and_rejects_other_hardware(self):
        for identity, accepted in [
                ('0, GPU-A, NVIDIA RTX 4080 SUPER, 8.9, 595\n1, GPU-B, NVIDIA GeForce RTX 5090, 12.0, 595', True),
                ('1, GPU-A, NVIDIA RTX 4080 SUPER, 8.9, 595', False),
                ('0, GPU-B, NVIDIA GeForce RTX 5090, 12.0, 595', False)]:
            with self.subTest(identity=identity), tempfile.TemporaryDirectory() as temporary:
                current = batch.Batch(options(Path(temporary)))
                def run(name, command):
                    return {'source': 'abc\n', 'source-state': '', 'gpu-identity': identity,
                            'sanitizer-version': 'version'}[name]
                with patch.object(current, 'run', side_effect=run):
                    if accepted:
                        self.assertEqual(current.preflight(), 'GPU-B')
                    else:
                        with self.assertRaises(ValueError):current.preflight()

    def test_profiles_are_independent_when_counter_capture_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            current = batch.Batch(options(Path(temporary)))
            current.output.mkdir()
            def process(command, **kwargs):
                if command[0] == 'ncu':raise FileNotFoundError('no counters')
                kwargs['stdout'].write('mock profile\n')
                if '--output' in command:
                    Path(command[command.index('--output') + 1] + '.nsys-rep').write_text('mock capture')
                return subprocess.CompletedProcess(command, 0)
            with patch.object(gpu_session, 'run_process', side_effect=process):
                current.profiles()
            stages = current.report['stages']
            self.assertEqual([stage['status'] for stage in stages], ['PASS', 'FAIL', 'PASS', 'FAIL', 'PASS', 'FAIL'])
            self.assertTrue(all(stage['optional'] for stage in stages))

    def test_gpu_success_reports_profile_limits_and_restores_device(self):
        for limited in (False, True):
            with self.subTest(limited=limited), tempfile.TemporaryDirectory() as temporary:
                args = options(Path(temporary));current = batch.Batch(args)
                def profiles():
                    current.report['stages'].append(dict(name='ncu', optional=True,
                                                         status='FAIL' if limited else 'PASS'))
                def stop():current.report['telemetry'] = 'CAPTURED'
                with patch.object(current, 'preflight', return_value='GPU-test'), \
                        patch.object(current, 'start_clock'), patch.object(current, 'stop_clock', side_effect=stop), \
                        patch.object(gpu_session, 'main', return_value=0), \
                        patch.object(current, 'repeat') as repeat, patch.object(current, 'profiles', side_effect=profiles), \
                        patch.object(current, 'analyze') as analyze, patch.dict(os.environ, {}, clear=True):
                    self.assertEqual(current.execute(), 0)
                    self.assertNotIn('CUDA_VISIBLE_DEVICES', os.environ)
                repeat.assert_called_once();analyze.assert_called_once()
                self.assertEqual(current.report['status'], 'GPU_BATCH_PASSED_WITH_LIMITATIONS' if limited else 'GPU_BATCH_PASSED')

    def test_missing_profile_artifacts_are_not_reported_as_captured(self):
        with tempfile.TemporaryDirectory() as temporary:
            current = batch.Batch(options(Path(temporary)));current.output.mkdir()
            with patch.object(gpu_session, 'run_process', return_value=subprocess.CompletedProcess([], 0)):
                current.profiles()
            self.assertEqual([stage['status'] for stage in current.report['stages']], ['PASS', 'PASS'] + ['FAIL'] * 4)

    def test_shared_deadline_stops_commands_before_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            current = batch.Batch(options(Path(temporary)))
            current.output.mkdir();current.deadline = -1
            with patch.object(gpu_session, 'run_process') as launch:
                with self.assertRaises(subprocess.TimeoutExpired):current.run('expired', ['unused'])
                current.run('optional-expired', ['unused'], optional=True)
            launch.assert_not_called()
            self.assertEqual(current.report['stages'][0]['status'], 'FAIL')
            args = type('Args', (), dict(mode='offline', output_dir=Path(temporary) / 'child',
                build_dir=Path(temporary) / 'compiled', deadline_monotonic=-1))()
            session = gpu_session.Session(args)
            with patch.object(gpu_session, 'run_process') as launch:
                with self.assertRaises(RuntimeError):session.run('expired', ['unused'])
            launch.assert_not_called()
            self.assertEqual(session.report['stages'][0]['status'], 'FAIL')

    def test_archive_verifies_contents_and_refuses_overwrite_or_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);output = root / 'records';output.mkdir()
            (output / 'nested').mkdir();(output / 'nested/log.txt').write_text('partial evidence')
            target = batch.archive(output)
            self.assertEqual(batch.digest(target), target.with_suffix('.gz.sha256').read_text().split()[0])
            with tarfile.open(target) as package:
                self.assertEqual(package.extractfile('records/nested/log.txt').read(), b'partial evidence')
            before = target.read_bytes()
            with self.assertRaises(FileExistsError):batch.archive(output)
            self.assertEqual(target.read_bytes(), before)
            second = root / 'linked';second.mkdir();(second / 'external').symlink_to(output, target_is_directory=True)
            with self.assertRaises(ValueError):batch.archive(second)

    def test_path_validation_preserves_existing_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);args = options(root)
            for path in [args.output_dir, args.build_dir, args.output_dir.with_suffix('.tar.gz')]:
                path.write_text('keep')
                with self.assertRaises(ValueError):batch.checked_paths(args)
                self.assertEqual(path.read_text(), 'keep');path.unlink()
            args.build_dir = args.output_dir / 'build'
            with self.assertRaises(ValueError):batch.checked_paths(args)

    def test_git_ignore_checks_treat_fresh_output_paths_as_directories(self):
        args = options(ROOT)
        args.output_dir = ROOT / 'bench-results-never-created-batch-test'
        args.build_dir = ROOT / 'build/never-created-batch-test'
        batch.checked_paths(args)
        self.assertFalse(args.output_dir.exists())

    def test_archive_siblings_must_also_be_ignored_inside_the_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(['git', 'init', '--quiet', root], check=True)
            (root / '.gitignore').write_text('records/\nbuild/\n')
            with patch.object(batch, 'ROOT', root), self.assertRaisesRegex(ValueError, 'ignored'):
                batch.checked_paths(options(root))

    def test_repeat_keeps_order_and_usable_raw_evidence(self):
        from benchmark_report import TIMING_OPERATIONS
        from test_analyze_gpu_batch import fixture
        with tempfile.TemporaryDirectory() as temporary:
            current = batch.Batch(options(Path(temporary)));current.output.mkdir()
            fixture(current.output, dimension=65536, bits=60, scope='gpu-chain')
            for schedule in batch.ORDER[:2]:
                directory = current.build / schedule;directory.mkdir(parents=True)
                (directory / 'ntt_resident_bench').write_text(schedule)
            def raw(name, command):
                schedule = command[0].parent.name
                rows = [dict(type='workload', schema=1, n=65536, towers=3,
                    moduli=[(1 << 59) + 1 + i * 131072 for i in range(3)], roots=[2, 3, 4],
                    timing='host_wall_resident_chain', input_policy='fixed_seed_reset_outside_timer',
                    seed=7492057, setup_included=False, warmups=5, samples=50, transform_schedule=schedule)]
                operations = TIMING_OPERATIONS['host_wall_resident_chain']
                for operation in operations:
                    rows.extend(dict(type='sample', operation=operation, iteration=i, milliseconds=1) for i in range(50))
                    rows.append(dict(type='summary', operation=operation, count=50,
                                     mean_ms=1, median_ms=1, p5_ms=1, p95_ms=1))
                rows.append(dict(type='complete', operations=len(operations)))
                text = '\n'.join(map(json.dumps, rows))
                (current.output / (name + '.log')).write_text(text)
                return text
            with patch.object(current, 'run', side_effect=raw):current.repeat()
            record = json.loads((current.output / 'controlled-repeat.json').read_text())
            self.assertEqual(record['status'], 'PASS')
            self.assertEqual([row['schedule'] for row in record['runs']], list(batch.ORDER))
            self.assertEqual([row['order'] for row in record['runs']], [1, 2, 3, 4])
            self.assertTrue(all(row['raw_sha256'] == batch.digest(current.output / row['raw_log']) for row in record['runs']))
            current.analyze()
            self.assertEqual(current.report['analysis'], 'VALIDATED')
            summary = json.loads((current.output / 'analysis/summary.json').read_text())
            self.assertEqual(summary['repeat_operation_rows'], 12)


if __name__ == '__main__':
    unittest.main()
