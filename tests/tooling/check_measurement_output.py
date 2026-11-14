"""Cross-language serialization gate with synthetic, explicitly marked data."""
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from benchmark_report import validate_jsonl

raw = subprocess.check_output([sys.argv[1], '--emit'], text=True)
result = validate_jsonl(raw, expected_bits=50)
if result['workload'].get('simulation') is not True:
    raise AssertionError('synthetic output must be explicitly marked')
print('C++/Python benchmark JSONL interop passed; no GPU measurements')
