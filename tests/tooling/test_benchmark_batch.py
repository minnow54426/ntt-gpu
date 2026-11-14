"""Synthetic batch-control evidence only; never launches CUDA/profilers."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_batch import collect_batch, collect_profiles, DIMENSIONS, PRIME_BITS, TOWER_COUNTS
from benchmark_report import TIMING_OPERATIONS


class FakeSession:
    def __init__(self, directory, dimensions=(65536,), towers=(3,)):
        self.root, self.output = ROOT, Path(directory)
        build = self.output / 'build'
        build.mkdir()
        for binary in ('ntt_cuda_cli', 'ntt_openfhe_bench'):
            (build / binary).write_text('synthetic executable identity')
        prefix = self.output / 'openfhe'
        prefix.mkdir()
        (prefix / 'libOPENFHEcore.a').write_text('mock library identity')
        self.args = SimpleNamespace(build_dir=build, openfhe_prefix=prefix, architectures='native',
            benchmark_warmups=1, benchmark_samples=2, benchmark_dimensions=list(dimensions),
            benchmark_towers=list(towers))
        self.report = {'source_commit': 'mock-source', 'source_dirty': False,
                       'selected_device': 'mock device', 'stages': [], 'attempt': 1}
        self.commands = []
        self.fail = set()
        self.wrong_parameters = False

    def save(self):
        pass

    def run(self, name, command):
        command = list(map(str, command))
        self.commands.append((name, command))
        if name in self.fail:
            raise RuntimeError('injected subprocess failure')
        path = self.output / f'{name}-{len(self.commands)}.log'
        if 'parameters' in name:
            vector, dimension, towers, _, bits = command[-5:]
            dimension, towers, bits = int(dimension), int(towers), int(bits)
            Path(vector).write_text(f'{dimension} {towers}\n' + ''.join(
                f'{(1 << (bits - 1)) + 1 + i * 2 * dimension} {i + 2}\n' for i in range(towers)))
            raw = 'mock parameters generated'
        elif name.startswith('benchmark-'):
            tokens = Path(command[-3]).read_text().split()
            dimension, towers = map(int, tokens[:2])
            timing = ('host_wall_public_call' if name.endswith('-public') else
                      'cuda_event_resident_and_pinned_copy' if name.endswith('-events') else
                      'host_wall_openfhe_native')
            operations = TIMING_OPERATIONS[timing]
            rows = [{'type': 'workload', 'schema': 1, 'n': dimension, 'towers': towers,
                'moduli': [int(tokens[2 + 2 * i]) for i in range(towers)],
                'roots': [int(tokens[3 + 2 * i]) for i in range(towers)],
                'timing': timing, 'input_policy': 'fixed_seed_reset_outside_timer',
                'seed': 7492057, 'setup_included': False, 'warmups': 1, 'samples': 2}]
            if self.wrong_parameters:
                rows[0]['roots'][0] += 1
            for operation in operations:
                for iteration in range(2):
                    rows.append({'type': 'sample', 'operation': operation,
                                 'iteration': iteration, 'milliseconds': 1})
                rows.append({'type': 'summary', 'operation': operation, 'count': 2,
                             'mean_ms': 1, 'median_ms': 1, 'p5_ms': 1, 'p95_ms': 1})
            rows.append({'type': 'complete', 'operations': len(operations)})
            raw = '\n'.join(map(json.dumps, rows))
        else:
            raw = 'mock profiler command'
        path.write_text(raw)
        self.report['stages'].append({'name': name, 'log': path.name})
        return raw


class BatchTests(unittest.TestCase):
    def test_small_parameter_files_retain_40_exact_primes(self):
        with tempfile.TemporaryDirectory() as directory:
            for bits in PRIME_BITS:
                vector = Path(directory) / f'parameters-b{bits}.txt'
                subprocess.run([sys.executable, str(ROOT / 'tests/oracle/gen_vectors.py'),
                    '--zero-coefficients', str(vector), '65536', '40', '1', str(bits)],
                    check=True, stdout=subprocess.DEVNULL, timeout=30)
                tokens = [word for line in vector.read_text().splitlines()
                          if not line.startswith('#') for word in line.split()]
                self.assertEqual(tokens[:2], ['65536', '40'])
                moduli = [int(tokens[2 + 2 * index]) for index in range(40)]
                self.assertEqual(len(set(moduli)), 40)
                for index, modulus in enumerate(moduli):
                    self.assertEqual(modulus.bit_length(), bits)
                    self.assertEqual(modulus % 131072, 1)
                    self.assertEqual(pow(int(tokens[3 + 2 * index]), 65536, modulus), modulus - 1)
                self.assertEqual(tokens[82], 'COEFFS')
                self.assertEqual(tokens[83:], ['0'] * 65536)
                self.assertLess(vector.stat().st_size, 150000)

    def test_default_matrix_contains_30_bit_and_all_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory, DIMENSIONS, TOWER_COUNTS)
            collect_batch(session, 'mock GPU UUID and driver')
            self.assertEqual(PRIME_BITS, (30, 50, 55, 60))
            self.assertEqual(TOWER_COUNTS, (1, 3, 8, 16, 24, 32, 40))
            self.assertEqual(len(session.report['benchmark_summaries']), 420)
            for dimension in DIMENSIONS:
                self.assertIn(f'benchmark-n{dimension}-k3-b30-events', session.report['benchmark_summaries'])
                self.assertIn(f'benchmark-n{dimension}-k40-b30-events', session.report['benchmark_summaries'])

    def test_partial_failure_keeps_other_jobs_and_resume_reuses_only_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            failed = 'benchmark-n65536-k3-b30-public'
            session.fail.add(failed)
            with self.assertRaises(RuntimeError): collect_batch(session, 'GPU')
            self.assertEqual(len(session.report['benchmark_summaries']), 11)
            session.fail.clear()
            session.commands.clear()
            collect_batch(session, 'GPU')
            self.assertEqual(sum(name.startswith('benchmark-') for name, _ in session.commands), 1)
            self.assertEqual(sum(row['reused'] for row in session.report['benchmark_summaries'].values()), 11)

    def test_resume_rejects_source_device_binary_or_options_changes(self):
        for change in ('source', 'dirty', 'hardware', 'binary', 'library', 'samples', 'selected_device'):
            with tempfile.TemporaryDirectory() as directory, self.subTest(change=change):
                session = FakeSession(directory)
                collect_batch(session, 'GPU')
                if change == 'source': session.report['source_commit'] = 'new-source'
                if change == 'dirty': session.report['source_dirty'] = True
                if change == 'binary': (session.args.build_dir / 'ntt_cuda_cli').write_text('changed binary')
                if change == 'library': (session.args.openfhe_prefix / 'libOPENFHEcore.a').write_text('changed library')
                if change == 'samples': session.args.benchmark_samples = 3
                if change == 'selected_device': session.report['selected_device'] = 'new device'
                with self.assertRaises(RuntimeError):
                    collect_batch(session, 'other GPU' if change == 'hardware' else 'GPU')

    def test_tampered_log_cannot_count_as_completed_job(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            collect_batch(session, 'GPU')
            manifest = json.loads((session.output / 'benchmark-manifest.json').read_text())
            job = next(iter(manifest['jobs'].values()))
            (session.output / job['log']).write_text('tampered')
            with self.assertRaises(RuntimeError): collect_batch(session, 'GPU')
            self.assertEqual(len(session.report['benchmark_summaries']), 11)

    def test_same_width_different_roots_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            session.wrong_parameters = True
            with self.assertRaises(RuntimeError): collect_batch(session, 'GPU')
            self.assertFalse(session.report['benchmark_summaries'])

    def test_selected_profiler_commands_are_bounded_and_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(directory)
            collect_profiles(session)
            commands = dict(session.commands)
            self.assertIn('--launch-count', commands['ncu'])
            self.assertIn('8', commands['ncu'])
            self.assertEqual(commands['nsys'][-2:], ['1', '2'])
            self.assertTrue(any('profile-attempt-1' in arg for arg in commands['nsys']))
            self.assertFalse(any('force-overwrite' in arg for command in commands.values() for arg in command))


if __name__ == '__main__':
    unittest.main()
