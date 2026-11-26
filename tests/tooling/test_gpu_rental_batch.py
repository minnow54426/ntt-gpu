"""Batch orchestration tests use synthetic subprocesses, never native GPU evidence."""
import contextlib
import io
from itertools import product
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
from benchmark_batch import vector_name


def options(root, mode='gpu', target='rtx-5090'):
    return type('Options', (), dict(mode=mode, cuda_root=Path('/mock/cuda'),
        openfhe_prefix=Path('/mock/openfhe'), output_dir=root / 'records',
        build_dir=root / 'build', budget_minutes=5, jobs=2, command_timeout=120,
        device_index=1, target=target, prepared_vectors=None, conversion_references=None, extended_probes=False,
        defer_analysis=False, rotation_fixture=None))()


def profile_binaries(current):
    directory = current.build / 'group_twiddle_ct'
    directory.mkdir(parents=True)
    for name in ('ntt_resident_profile', 'ntt_resident_conversion_profile'):
        (directory / name).write_text('synthetic profiling executable ' + name)


class BatchTests(unittest.TestCase):
    def test_rotation_pairs_validate_native_backend_and_full_pair_order(self):
        samples = [
            dict(schema=1, backend='cuda', timing='public_call_wall',
                 ring_dimension=65536, active_q=towers, shifts=[1, -1, 2, -2],
                 pair=pair, order='individual_first' if pair == 1 else 'hoisted_first',
                 individual_ns=200, hoisted_ns=100)
            for towers, pair in product((5, 3), (1, 2))
        ]
        def jsonl(records):
            return '\n'.join(json.dumps(record) for record in records)

        summary = batch.summarize_rotation_pairs(jsonl(samples), 0, 2)
        self.assertEqual(summary['5']['median_paired_speedup'], 2)
        self.assertEqual(summary['3']['median_hoisted_ns'], 100)
        for change in ({'backend': 'host_simulation'}, {'order': 'hoisted_first'}, {'hoisted_ns': 0}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                batch.summarize_rotation_pairs(jsonl([samples[0] | change, *samples[1:]]), 0, 2)
        with self.assertRaises(ValueError):
            batch.summarize_rotation_pairs(jsonl(samples[:-1]), 0, 2)

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
            self.assertEqual(args.architectures, '120')
            self.assertEqual(args.benchmark_scope, 'gpu-conversion')
            self.assertTrue(args.compare_ntt)
            self.assertFalse(args.skip_sanitizers)
            self.assertEqual(args.benchmark_dimensions, [4096, 65536])
            self.assertEqual(args.benchmark_towers, [3, 40])
            self.assertEqual(args.benchmark_prime_bits, [30, 60])

    def test_ada_plans_compile_for_sm_89(self):
        for target, model in [('rtx-4080', 'RTX 4080'), ('rtx-4080-super', 'RTX 4080 SUPER')]:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                plan = batch.make_plan(options(Path(temporary), 'offline', target))
                self.assertEqual(plan['target'], dict(model=model, compute_cap='8.9', cuda_architecture='89'))
                collector = gpu_session.parser().parse_args(plan['collector_arguments'])
                self.assertEqual(collector.architectures, '89')

    def test_prepared_batch_validates_inputs_before_device_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = options(Path(temporary), target='rtx-4080-super')
            prepared = Path(temporary) / 'prepared'
            prepared.mkdir()
            args.prepared_vectors = args.conversion_references = prepared
            vector_hashes, reference_hashes = {}, {}
            for cell in product((4096, 65536), (3, 40), (30, 60)):
                vector = prepared / vector_name(*cell)
                reference = vector.with_suffix('.conversion')
                vector.write_text('fixture vector')
                reference.write_bytes(b'fixture reference')
                vector_hashes[vector.name] = batch.digest(vector)
                reference_hashes[reference.name] = batch.digest(reference)
            batch.atomic_json(prepared / 'vectors.json', vector_hashes)
            batch.atomic_json(prepared / 'conversion-references.json', reference_hashes)
            current = batch.Batch(args)
            collector = gpu_session.parser().parse_args(current.report['collector_arguments'])
            self.assertEqual(collector.prepared_vectors, prepared)
            self.assertEqual(collector.conversion_references, prepared)
            gpu_session.validate_prepared_inputs(collector)

            reference.write_bytes(b'changed reference')
            with patch.object(current, 'preflight') as preflight, patch.object(gpu_session, 'main') as collect:
                self.assertEqual(current.execute(), 1)
            preflight.assert_not_called()
            collect.assert_not_called()
            self.assertIn('modified conversion reference', current.report['error'])
            self.assertEqual(current.report['gpu_execution'], 'NOT_RUN')

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
        for target, identity, expected_uuid in [
                ('rtx-5090', '0, GPU-A, NVIDIA RTX 4080 SUPER, 8.9, 595\n1, GPU-B, NVIDIA GeForce RTX 5090, 12.0, 595', 'GPU-B'),
                ('rtx-5090', '1, GPU-A, NVIDIA RTX 4080 SUPER, 8.9, 595', None),
                ('rtx-5090', '0, GPU-B, NVIDIA GeForce RTX 5090, 12.0, 595', None),
                ('rtx-4080-super', '1, GPU-A, NVIDIA GeForce RTX 4080 SUPER, 8.9, 595', 'GPU-A'),
                ('rtx-4080-super', '1, GPU-A, NVIDIA GeForce RTX 4080 SUPER, 12.0, 595', None),
                ('rtx-4080', '1, GPU-A, NVIDIA GeForce RTX 4080 SUPER, 8.9, 595', None),
                ('rtx-4080', '1, GPU-A, NVIDIA GeForce RTX 4080, 8.9, 595', 'GPU-A')]:
            with self.subTest(target=target, identity=identity), tempfile.TemporaryDirectory() as temporary:
                current = batch.Batch(options(Path(temporary), target=target))
                def run(name, command):
                    return {'source': 'abc\n', 'source-state': '', 'gpu-identity': identity,
                            'sanitizer-version': 'version'}[name]
                with patch.object(current, 'run', side_effect=run):
                    if expected_uuid:
                        self.assertEqual(current.preflight(), expected_uuid)
                    else:
                        with self.assertRaises(ValueError):current.preflight()

    def test_profiles_are_independent_when_counter_capture_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            current = batch.Batch(options(Path(temporary)))
            current.output.mkdir()
            profile_binaries(current)
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
            self.assertTrue(all(len(value) == 64 for value in current.report['profile_binaries'].values()))
            counters = [stage for stage in stages if stage['name'] == 'ncu-conversion'][0]['command']
            self.assertIn('ntt-gpu@ntt.sample/', counters)
            self.assertIn(str(current.build / 'group_twiddle_ct/ntt_resident_conversion_profile'), counters)

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

    def test_extended_batch_can_defer_analysis_until_after_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = options(Path(temporary))
            args.extended_probes = args.defer_analysis = True
            current = batch.Batch(args)
            with patch.object(current, 'preflight', return_value='GPU-test'), \
                    patch.object(current, 'start_clock'), patch.object(current, 'stop_clock'), \
                    patch.object(gpu_session, 'main', return_value=0), \
                    patch.object(current, 'repeat') as repeat, patch.object(current, 'profiles') as profiles, \
                    patch.object(current, 'analyze') as analyze:
                self.assertEqual(current.execute(), 0)
            self.assertEqual([call.args for call in repeat.call_args_list],
                             [(), ('gpu-public', 40, 30), ('gpu-public', 40, 60)])
            profiles.assert_called_once()
            analyze.assert_not_called()
            self.assertEqual(current.report['analysis'], 'DEFERRED_TO_LOCAL')
            self.assertTrue(args.output_dir.with_suffix('.tar.gz.sha256').exists())

    def test_missing_profile_artifacts_are_not_reported_as_captured(self):
        with tempfile.TemporaryDirectory() as temporary:
            current = batch.Batch(options(Path(temporary)));current.output.mkdir()
            with patch.object(gpu_session, 'run_process', return_value=subprocess.CompletedProcess([], 0)):
                current.profiles()
            self.assertEqual([stage['status'] for stage in current.report['stages']], ['PASS', 'PASS'] + ['FAIL'] * 4)

    def test_conversion_profile_reuses_the_measured_reference(self):
        for matches in (True, False):
            with self.subTest(matches=matches), tempfile.TemporaryDirectory() as temporary:
                args = options(Path(temporary))
                args.conversion_references = Path(temporary) / 'prepared'
                args.conversion_references.mkdir()
                reference = args.conversion_references / Path(vector_name(65536, 3, 60)).with_suffix('.conversion')
                reference.write_bytes(b'fixture reference')
                checksum = batch.digest(reference)
                batch.atomic_json(args.conversion_references / 'conversion-references.json', {reference.name: checksum})
                current = batch.Batch(args)
                paired = current.output / 'paired/group_twiddle_ct'
                paired.mkdir(parents=True)
                batch.atomic_json(paired / 'benchmark-manifest.json', {'context': {
                    'conversion_reference_sha256': {reference.name: checksum if matches else 'different'}
                }})
                with patch.object(current, 'run', return_value=None) as run:
                    current.profiles()
                commands = {call.args[0]: call.args[1] for call in run.call_args_list}
                self.assertNotIn(reference, commands['nsys-ntt'])
                if matches:
                    self.assertEqual(commands['nsys-conversion'][-1], reference)
                    self.assertEqual(commands['ncu-conversion'][-1], reference)
                    self.assertEqual(current.report['profile_conversion_reference_sha256'], checksum)
                else:
                    self.assertNotIn('nsys-conversion', commands)
                    failure = current.report['stages'][-1]
                    self.assertTrue(failure['optional'])
                    self.assertIn('differs from the measured batch', failure['error'])

    def test_extended_capture_exports_large_conversion_to_sqlite(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = options(Path(temporary))
            args.extended_probes = True
            current = batch.Batch(args)
            current.output.mkdir()
            profile_binaries(current)
            def process(command, **kwargs):
                if '--output' in command:
                    output = Path(command[command.index('--output') + 1])
                    if command[1] == 'profile':
                        output.with_suffix('.nsys-rep').write_text('mock capture')
                    if command[1] == 'export':
                        output.write_text('mock sqlite')
                return subprocess.CompletedProcess(command, 0)
            with patch.object(gpu_session, 'run_process', side_effect=process):
                current.profiles()
            stages = {stage['name']: stage for stage in current.report['stages']}
            large = stages['nsys-conversion-k40']['command']
            self.assertTrue(large[-3].endswith('benchmark-vector-n65536-k40-b60.txt'))
            self.assertNotIn('ncu-conversion-k40', stages)
            export = stages['nsys-conversion-k40-sqlite']
            self.assertEqual(export['status'], 'PASS')
            self.assertIn('sqlite', export['command'])
            self.assertTrue(export['command'][-1].endswith('nsys-conversion-k40.nsys-rep'))


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

    def test_public_repeats_are_validated_against_the_paired_workload(self):
        from benchmark_report import TIMING_OPERATIONS
        from test_analyze_gpu_batch import fixture
        for bits in batch.PUBLIC_REPEAT_BITS:
            with self.subTest(bits=bits), tempfile.TemporaryDirectory() as temporary:
                current = batch.Batch(options(Path(temporary)))
                current.output.mkdir()
                fixture(current.output, dimension=65536, bits=bits, towers=40)
                paired = current.output / 'paired'
                for schedule in batch.ORDER[:2]:
                    directory = current.build / schedule
                    directory.mkdir(parents=True)
                    (directory / 'ntt_cuda_cli').write_text(schedule)
                scope = 'host_wall_gpu_public_call'
                def raw(name, command, optional):
                    self.assertTrue(optional)
                    binary, subcommand, vector, warmups, samples = command
                    self.assertEqual(subcommand, 'gpu-bench-jsonl')
                    self.assertEqual((warmups, samples), ('5', '50'))
                    self.assertEqual(vector.name, vector_name(65536, 40, bits))
                    schedule = binary.parent.name
                    report = json.loads((paired / schedule / 'report.json').read_text())
                    job = f'benchmark-n65536-k40-b{bits}-gpu-public'
                    records = [dict(report['benchmark_summaries'][job]['workload'],
                                    type='workload', warmups=5, samples=50)]
                    for operation in TIMING_OPERATIONS[scope]:
                        records.extend(dict(type='sample', operation=operation, iteration=i,
                                            milliseconds=1) for i in range(50))
                        records.append(dict(type='summary', operation=operation, count=50,
                                            mean_ms=1, median_ms=1, p5_ms=1, p95_ms=1))
                    records.append(dict(type='complete', operations=len(TIMING_OPERATIONS[scope])))
                    data = '\n'.join(map(json.dumps, records))
                    (current.output / (name + '.log')).write_text(data)
                    return data
                with patch.object(current, 'run', side_effect=raw):
                    current.repeat('gpu-public', 40, bits)
                current.analyze()
                summary = json.loads((current.output / 'analysis/summary.json').read_text())
                self.assertEqual(summary['public_repeat_operation_rows'], 4 * len(TIMING_OPERATIONS[scope]))
                self.assertTrue((current.output / 'analysis/public-repeats.csv').is_file())
                log = current.output / f'public-repeat-b{bits}-1-legacy_dif.log'
                log.write_text('changed')
                with self.assertRaisesRegex(ValueError, 'repeat identity/hash mismatch'):
                    batch.export_analysis(current.output, current.output / 'reanalyzed')


if __name__ == '__main__':
    unittest.main()
