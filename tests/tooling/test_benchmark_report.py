"""Synthetic JSONL validation; these fixtures are NOT GPU measurements."""
import copy
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from benchmark_report import OPERATIONS, TIMING_OPERATIONS, validate_jsonl


def fixture():
    rows = [{'type': 'workload', 'schema': 1, 'n': 65536, 'towers': 1,
             'samples': 4, 'warmups': 2, 'moduli': [562949954338817], 'roots': [2],
             'timing': 'host_wall_public_call', 'input_policy': 'fixed_seed_reset_outside_timer',
             'setup_included': False}]
    for name in sorted(OPERATIONS):
        for index, value in enumerate([1, 2, 3, 4]):
            rows.append({'type': 'sample', 'operation': name,
                         'iteration': index, 'milliseconds': value})
        rows.append({'type': 'summary', 'operation': name, 'count': 4,
                     'mean_ms': 2.5, 'median_ms': 2.5, 'p5_ms': 1.15, 'p95_ms': 3.85})
    rows.append({'type': 'complete', 'operations': 6})
    return rows


def validate(rows, **kwargs):
    return validate_jsonl('\n'.join(json.dumps(row) for row in rows), **kwargs)


class BenchmarkReportTests(unittest.TestCase):
    def test_gpu_only_scope_rejects_cpu_timing(self):
        rows = fixture()
        rows[0]['timing'] = 'host_wall_gpu_public_call'
        rows[-1]['operations'] = 5
        with self.assertRaises(ValueError):
            validate(rows)
        rows = [row for row in rows if row.get('operation') != 'cpu_reference_forward']
        result = validate(rows)
        self.assertEqual(len(result['summaries']), 5)
        self.assertNotIn('cpu_reference_forward', result['summaries'])

    def test_each_scope_and_supported_dimension(self):
        for timing, operations in TIMING_OPERATIONS.items():
            for dimension in [4096, 8192, 16384, 32768, 65536]:
                rows = fixture()
                rows[0].update(n=dimension, timing=timing)
                template = rows[1:6]
                rows = rows[:1]
                for operation in operations:
                    for original in template:
                        row = dict(original, operation=operation)
                        rows.append(row)
                rows.append({'type': 'complete', 'operations': len(operations)})
                with self.subTest(timing=timing, dimension=dimension):
                    validate(rows, expected_n=dimension, expected_towers=1, expected_timing=timing)
                    with self.assertRaises(ValueError):
                        validate(rows, expected_towers=3)
                    with self.assertRaises(ValueError):
                        validate(rows, expected_timing='wrong-scope')

    def test_30_bit_parameters_and_invalid_dimensions(self):
        rows = fixture()
        rows[0]['moduli'] = [536903681]
        validate(rows, expected_bits=30)
        for dimension in [False, 0, 4097, 131072]:
            rows[0]['n'] = dimension
            with self.subTest(n=dimension), self.assertRaises(ValueError): validate(rows)

    def test_recomputes_linear_percentiles(self):
        result = validate(fixture(), expected_bits=50)
        self.assertEqual(result['summaries']['gpu_inverse']['median_ms'], 2.5)

    def test_truncated_record_rejected(self):
        with self.assertRaises(ValueError): validate(fixture()[:-1])

    def test_nonfinite_negative_and_boolean_samples_rejected(self):
        for value in [float('nan'), float('inf'), -1, True]:
            rows = fixture()
            rows[1]['milliseconds'] = value
            with self.subTest(value=value), self.assertRaises(ValueError): validate(rows)

    def test_duplicate_sample_rejected(self):
        rows = fixture()
        rows.insert(2, copy.deepcopy(rows[1]))
        with self.assertRaises(ValueError): validate(rows)

    def test_wrong_summary_and_missing_operation_rejected(self):
        rows = fixture()
        rows[5]['mean_ms'] = 9
        with self.assertRaises(ValueError): validate(rows)
        rows = [row for row in fixture() if row.get('operation') != 'gpu_inverse']
        with self.assertRaises(ValueError): validate(rows)

    def test_wrong_scope_width_and_count_rejected(self):
        for key, value in [('timing', 'kernel_only'), ('samples', 0), ('warmups', False),
                           ('input_policy', 'mutated_input'), ('roots', [])]:
            rows = fixture()
            rows[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): validate(rows)
        with self.assertRaises(ValueError): validate(fixture(), expected_bits=60)


if __name__ == '__main__': unittest.main()
