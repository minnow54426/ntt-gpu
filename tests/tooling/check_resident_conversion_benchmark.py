"""Verify the host driver's conversion timing protocol and admission limits."""
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from benchmark_report import validate_jsonl

program, vector = sys.argv[1:]
result = subprocess.run([program, vector, "1", "2"], check=True, capture_output=True, text=True)
parsed = validate_jsonl(result.stdout, expected_n=64, expected_towers=3,
                        expected_timing="host_wall_resident_conversion")
workload = parsed["workload"]
assert workload.get("simulation") is True and workload["samples"] == 2
records = [json.loads(line) for line in result.stdout.splitlines()]
records[0]["p_roots"][0] = 1
try:
    validate_jsonl("\n".join(map(json.dumps, records)))
except ValueError:
    pass
else:
    raise AssertionError("invalid special root was accepted")
bad = subprocess.run([program, vector, "2", "200"], capture_output=True, text=True)
assert bad.returncode != 0 and not bad.stdout
