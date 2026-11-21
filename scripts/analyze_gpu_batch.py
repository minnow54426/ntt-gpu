#!/usr/bin/env python3
"""Revalidate an exported paired GPU batch and write operation-level CSV."""
import argparse
import csv
from itertools import product
import json
import math
from pathlib import Path
import statistics

from benchmark_batch import SCOPES, digest
from benchmark_report import validate_jsonl

SCHEDULES = ("group_twiddle_ct", "legacy_dif")
SANITIZER_SCOPES = ("", "matrix-", "resident-", "basis-", "shoup-")


def record_file(directory, name):
    path = directory / name
    if path.resolve().parent != directory.resolve() or path.is_symlink():
        raise ValueError("record path must be a file directly inside its directory")
    return path


def normalized_context(context):
    result = dict(context)
    result.pop("binaries")
    result.pop("transform_schedule")
    result["selected_device"] = "\n".join(
        line for line in result["selected_device"].splitlines()
        if not line.startswith("transform schedule: "))
    return result


def load_run(directory, schedule):
    report = json.loads((directory / "report.json").read_text())
    manifest = json.loads((directory / "benchmark-manifest.json").read_text())
    context = manifest["context"]
    if (report["status"] != "GPU_CHECKS_PASSED" or
            report["gpu_execution"] != "EXACT_TESTS_PASS" or
            report["sanitizers"] != "SMOKE_MATRIX_PASS" or
            context["mode"] != "gpu" or context["source_dirty"] or
            context["transform_schedule"] != schedule or
            report["transform_schedule"] != schedule or
            report["source_commit"] != context["source_commit"]):
        raise ValueError("missing clean native exactness/sanitizer evidence")
    stages = {stage["name"]: stage for stage in report["stages"]}
    required = {"exact-tests"}
    required.update("sanitizer-" + scope + tool for scope in SANITIZER_SCOPES
                    for tool in ("memcheck", "racecheck", "initcheck", "synccheck"))
    if "conversion" in context["scopes"]:
        required.update("sanitizer-conversion-benchmark-" + tool
                        for tool in ("memcheck", "racecheck", "initcheck", "synccheck"))
    if not required.issubset(stages) or any(stages[name]["status"] != "PASS" for name in required):
        raise ValueError("incomplete exactness/sanitizer stages")
    for axis in ("dimensions", "tower_counts", "prime_bits", "scopes"):
        if not context[axis] or len(set(context[axis])) != len(context[axis]):
            raise ValueError("empty or duplicate workload axis")
    cells = product(context["dimensions"], context["tower_counts"], context["prime_bits"], context["scopes"])
    expected = {f"benchmark-n{n}-k{k}-b{bits}-{scope}": (n, k, bits, scope)
                for n, k, bits, scope in cells}
    if (manifest["jobs"].keys() != expected.keys() or
            report["benchmark_summaries"].keys() != expected.keys()):
        raise ValueError("incomplete or unexpected workload matrix")
    parsed = {}
    for name, (dimension, towers, bits, scope) in expected.items():
        job = manifest["jobs"][name]
        path = record_file(directory, job["log"])
        if job["status"] != "PASS" or digest(path) != job["sha256"]:
            raise ValueError("measurement log hash/status mismatch")
        result = validate_jsonl(path.read_text(), expected_n=dimension, expected_towers=towers,
                                expected_bits=bits, expected_timing=SCOPES[scope][2])
        workload = result["workload"]
        if (workload.get("simulation") or workload.get("transform_schedule") != schedule or
                workload["samples"] != context["samples"] or
                workload["warmups"] != context["warmups"] or workload.get("seed") != 7492057):
            raise ValueError("wrong native workload identity")
        saved = report["benchmark_summaries"][name]
        if result["workload"] != saved["workload"] or result["summaries"] != saved["summaries"]:
            raise ValueError("report disagrees with raw samples")
        parsed[name] = result
    return context, parsed


def analyze(records):
    comparison = json.loads((records / "paired/comparison.json").read_text())
    if comparison["status"] != "MATCHED_GPU_RUNS":
        raise ValueError("paired collector did not complete successfully")
    contexts, runs = {}, {}
    for schedule in SCHEDULES:
        contexts[schedule], runs[schedule] = load_run(records / "paired" / schedule, schedule)
    left, right = (contexts[schedule] for schedule in SCHEDULES)
    if normalized_context(left) != normalized_context(right):
        raise ValueError("paired environment or settings differ")
    if (left["binaries"].keys() != right["binaries"].keys() or
            any(value == right["binaries"][name] for name, value in left["binaries"].items())):
        raise ValueError("paired executable identities must be distinct")
    if sorted(comparison["matched_jobs"]) != sorted(runs[SCHEDULES[0]]):
        raise ValueError("paired index does not match raw workload matrix")
    rows = []
    for name, current in runs[SCHEDULES[0]].items():
        legacy = runs[SCHEDULES[1]][name]
        identity = lambda result: {key: value for key, value in result["workload"].items()
                                   if key != "transform_schedule"}
        if identity(current) != identity(legacy):
            raise ValueError("paired inputs or timing scopes differ")
        workload = current["workload"]
        for operation, group_stats in sorted(current["summaries"].items()):
            legacy_stats = legacy["summaries"][operation]
            median = group_stats["median_ms"]
            if median <= 0 or legacy_stats["median_ms"] <= 0:
                raise ValueError("positive medians required for ratios")
            row = dict(job=name, n=workload["n"], towers=workload["towers"],
                       prime_bits=workload["moduli"][0].bit_length(), timing=workload["timing"],
                       operation=operation, samples=workload["samples"],
                       legacy_over_group=legacy_stats["median_ms"] / median)
            for schedule, stats in (("group", group_stats), ("legacy", legacy_stats)):
                row.update({schedule + "_" + key: stats[key]
                            for key in ("median_ms", "p5_ms", "p95_ms")})
            rows.append(row)
    repeats = []
    repeat_path = records / "controlled-repeat.json"
    if repeat_path.exists():
        repeat = json.loads(repeat_path.read_text())
        order = ["legacy_dif", "group_twiddle_ct", "group_twiddle_ct", "legacy_dif"]
        if repeat["status"] != "PASS" or [run["schedule"] for run in repeat["runs"]] != order:
            raise ValueError("wrong controlled-repeat order/status")
        reference = runs["group_twiddle_ct"]["benchmark-n65536-k3-b60-chain"]["workload"]
        for index, run in enumerate(repeat["runs"], 1):
            path = record_file(records, run["raw_log"])
            schedule = run["schedule"]
            if (run["order"] != index or digest(path) != run["raw_sha256"] or
                    run["binary_sha256"] != contexts[schedule]["binaries"]["ntt_resident_bench"]):
                raise ValueError("repeat identity/hash mismatch")
            result = validate_jsonl(path.read_text(), expected_n=65536, expected_towers=3,
                                    expected_bits=60, expected_timing="host_wall_resident_chain")
            expected = dict(reference, warmups=5, samples=50, transform_schedule=schedule)
            if result["workload"] != expected or result != run["summary"]:
                raise ValueError("repeat input or summary mismatch")
            for operation, stats in result["summaries"].items():
                repeats.append(dict(order=index, schedule=schedule, operation=operation, **stats))
    return contexts, rows, repeats


def summarize(rows):
    result = []
    groups = sorted({(row["timing"], row["operation"]) for row in rows})
    for timing, operation in groups:
        ratios = [row["legacy_over_group"] for row in rows
                  if (row["timing"], row["operation"]) == (timing, operation)]
        result.append(dict(timing=timing, operation=operation, workloads=len(ratios),
                           median_ratio=statistics.median(ratios),
                           geometric_mean_ratio=math.exp(statistics.mean(map(math.log, ratios))),
                           minimum_ratio=min(ratios), maximum_ratio=max(ratios),
                           group_faster=sum(value > 1 for value in ratios)))
    return result


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    contexts, rows, repeats = analyze(args.records)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "paired-operations.csv", rows)
    write_csv(args.output_dir / "resident-repeats.csv", repeats)
    result = dict(context=normalized_context(contexts[SCHEDULES[0]]),
                  paired_operation_rows=len(rows), repeat_operation_rows=len(repeats),
                  operations=summarize(rows))
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["operations"], indent=2))


if __name__ == "__main__":
    main()
