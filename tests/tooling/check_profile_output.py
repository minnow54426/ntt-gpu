"""Run actual NVTX-instrumented CPU simulation; forbid benchmark ingestion."""
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from benchmark_report import validate_jsonl
program, vector = sys.argv[1:]
result = subprocess.run([program, vector, '1', '2'], check=True, capture_output=True, text=True)
records = [json.loads(line) for line in result.stdout.splitlines()]
assert records[0]['profiled'] is True and records[0]['simulation'] is True
try:
    validate_jsonl(result.stdout)
except ValueError as error:
    assert 'profile-instrumented' in str(error)
else:
    raise AssertionError('profiled timings were accepted as benchmark measurements')
# Check the normal JSON protocol independently after testing the rejection.
records[0].pop('profiled')
validate_jsonl('\n'.join(map(json.dumps, records)), expected_n=64, expected_towers=3)
