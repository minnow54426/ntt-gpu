"""Reject damaged or incomparable evidence before calculating performance ratios."""
import json
from pathlib import Path
import tempfile
import unittest

from test_benchmark_batch import FakeSession
from benchmark_batch import collect_batch, digest
from benchmark_report import validate_jsonl
from analyze_gpu_batch import analyze


def fixture(root, dimension=4096, bits=30, scope='gpu'):
    paired = root / "paired"
    paired.mkdir()
    matched = []
    for schedule, elapsed in (("group_twiddle_ct", 1), ("legacy_dif", 2)):
        directory = paired / schedule
        directory.mkdir()
        session = FakeSession(directory, dimensions=(dimension,), scope=scope)
        session.args.ntt_schedule = session.schedule_label = schedule
        session.args.benchmark_prime_bits = [bits]
        for binary in session.args.build_dir.iterdir():
            binary.write_text(schedule)
        collect_batch(session, "one mock GPU")
        session.report.update(status="GPU_CHECKS_PASSED", gpu_execution="EXACT_TESTS_PASS",
                              sanitizers="SMOKE_MATRIX_PASS", transform_schedule=schedule)
        session.report["stages"] = [{"name": "exact-tests", "status": "PASS"}] + [
            {"name": "sanitizer-" + scope + tool, "status": "PASS"}
            for scope in ("", "matrix-", "resident-", "basis-", "shoup-")
            for tool in ("memcheck", "racecheck", "initcheck", "synccheck")]
        manifest_path = directory / "benchmark-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for name, job in manifest["jobs"].items():
            path = directory / job["log"]
            records = [json.loads(line) for line in path.read_text().splitlines()]
            for record in records:
                for key in ("milliseconds", "median_ms", "mean_ms", "p5_ms", "p95_ms"):
                    if key in record:
                        record[key] = elapsed
            path.write_text("\n".join(map(json.dumps, records)))
            job["sha256"] = digest(path)
            session.report["benchmark_summaries"][name] = validate_jsonl(path.read_text())
        manifest_path.write_text(json.dumps(manifest))
        (directory / "report.json").write_text(json.dumps(session.report))
        matched = sorted(manifest["jobs"])
    (paired / "comparison.json").write_text(json.dumps(
        {"status": "MATCHED_GPU_RUNS", "matched_jobs": matched}))


class AnalysisTests(unittest.TestCase):
    def test_new_barrett_records_require_all_four_sanitizer_stages(self):
        for complete in (False, True):
            with self.subTest(complete=complete), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture(root)
                for schedule in ('group_twiddle_ct', 'legacy_dif'):
                    path = root / 'paired' / schedule / 'report.json'
                    report = json.loads(path.read_text())
                    report['registered_gpu_tests'] = ['cuda_barrett_exact']
                    report['sanitizer_scopes'] = ['barrett_exact']
                    for tool in ('memcheck', 'racecheck', 'initcheck', 'synccheck')[:4 if complete else 3]:
                        report['stages'].append(dict(name='sanitizer-barrett-' + tool, status='PASS'))
                    path.write_text(json.dumps(report))
                if complete:
                    self.assertTrue(analyze(root)[1])
                else:
                    with self.assertRaisesRegex(ValueError, 'incomplete'):
                        analyze(root)

    def test_known_ratio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            _, rows, repeats = analyze(root)
            self.assertEqual(len(rows), 11)
            self.assertTrue(all(row["legacy_over_group"] == 2 for row in rows))
            self.assertEqual(repeats, [])

    def test_rejects_corruption_and_mismatched_provenance(self):
        for mutation in ("hash", "matrix", "hardware", "binary", "sanitizer", "summary", "path"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture(root)
                child = root / "paired/legacy_dif"
                manifest_path = child / "benchmark-manifest.json"
                report_path = child / "report.json"
                manifest = json.loads(manifest_path.read_text())
                report = json.loads(report_path.read_text())
                name = next(iter(manifest["jobs"]))
                if mutation == "hash":
                    (child / manifest["jobs"][name]["log"]).write_text("modified")
                elif mutation == "matrix":
                    manifest["jobs"].pop(name)
                elif mutation == "hardware":
                    manifest["context"]["hardware_identity"] = "other GPU"
                elif mutation == "binary":
                    original = json.loads((root / "paired/group_twiddle_ct/benchmark-manifest.json").read_text())
                    manifest["context"]["binaries"] = original["context"]["binaries"]
                elif mutation == "sanitizer":
                    report["stages"].pop()
                elif mutation == "summary":
                    operation = next(iter(report["benchmark_summaries"][name]["summaries"]))
                    report["benchmark_summaries"][name]["summaries"][operation]["median_ms"] += 1
                elif mutation == "path":
                    manifest["jobs"][name]["log"] = "../outside.log"
                manifest_path.write_text(json.dumps(manifest))
                report_path.write_text(json.dumps(report))
                with self.assertRaises(ValueError):
                    analyze(root)


if __name__ == "__main__":
    unittest.main()
