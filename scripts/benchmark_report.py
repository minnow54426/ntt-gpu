"""Validate raw public-call JSONL; no GPU inference from synthetic input files."""
import argparse
import json
import math
from pathlib import Path
import statistics

OPERATIONS = {"cpu_reference_forward", "gpu_forward", "gpu_inverse",
              "gpu_pointwise", "gpu_unfused_multiply", "gpu_fused_multiply"}


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def validate_jsonl(content, expected_bits=None):
    records = [json.loads(line) for line in content.splitlines() if line.strip()]
    if len(records) < 2 or any(not isinstance(row, dict) for row in records):
        raise ValueError("missing benchmark records")
    metadata = records[0]
    if metadata.get("type") != "workload" or metadata.get("schema") != 1:
        raise ValueError("missing workload/schema")
    if metadata.get("n") != 65536 or metadata.get("timing") != "host_wall_public_call":
        raise ValueError("wrong workload/timing scope")
    if metadata.get("input_policy") != "fixed_seed_reset_outside_timer" or metadata.get("setup_included") is not False:
        raise ValueError("wrong input/setup policy")
    count, warmups, towers = (metadata.get(name) for name in ("samples", "warmups", "towers"))
    if any(type(value) is not int for value in (count, warmups, towers)) or not (
            2 <= count <= 100000 and 1 <= warmups <= 10000 and 1 <= towers <= 65535):
        raise ValueError("invalid sample/warmup/tower counts")
    for name in ("moduli", "roots"):
        values = metadata.get(name)
        if not isinstance(values, list) or len(values) != towers or any(type(x) is not int or x <= 0 for x in values):
            raise ValueError("invalid parameter list")
    if expected_bits is not None and any(q.bit_length() != expected_bits for q in metadata["moduli"]):
        raise ValueError("wrong modulus bit width")
    if records[-1] != {"type": "complete", "operations": 6}:
        raise ValueError("missing completion marker")
    samples = {name: [] for name in OPERATIONS}
    summaries = {}
    for row in records[1:-1]:
        name = row.get("operation")
        if name not in OPERATIONS:
            raise ValueError("unexpected operation")
        if row.get("type") == "sample":
            value = row.get("milliseconds")
            if name in summaries or type(row.get("iteration")) is not int or row["iteration"] != len(samples[name]):
                raise ValueError("duplicate/out-of-order sample")
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("nonfinite/negative sample")
            samples[name].append(value)
        elif row.get("type") == "summary":
            if name in summaries or len(samples[name]) != count or row.get("count") != count:
                raise ValueError("incomplete/duplicate summary")
            expected = {"mean_ms": statistics.mean(samples[name]),
                        "median_ms": statistics.median(samples[name]),
                        "p5_ms": percentile(samples[name], .05),
                        "p95_ms": percentile(samples[name], .95)}
            for key, want in expected.items():
                got = row.get(key)
                if type(got) not in (float, int) or not math.isfinite(got) or not math.isclose(got, want, rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError("summary disagrees with raw samples")
            summaries[name] = expected
        else:
            raise ValueError("unexpected record type")
    if summaries.keys() != OPERATIONS:
        raise ValueError("missing operations")
    return {"workload": metadata, "summaries": summaries}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_jsonl(args.jsonl.read_text()), indent=2))
