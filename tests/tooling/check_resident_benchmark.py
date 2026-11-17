"""Exercise the resident measurement protocol with the deferred host driver."""
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from benchmark_report import validate_jsonl

program, vector = sys.argv[1:]
result = subprocess.run([program, vector, '1', '2'], check=True, capture_output=True, text=True)
workload = validate_jsonl(result.stdout, expected_n=64, expected_towers=3,
                          expected_timing='host_wall_resident_chain')['workload']
if workload.get('simulation') is not True or workload['samples'] != 2:
    raise SystemExit('host workload must identify itself as a simulation with two samples')
