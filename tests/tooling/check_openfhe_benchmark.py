"""Real CPU/OpenFHE protocol and exact-result checks; no GPU measurements."""
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests/oracle'))
from benchmark_batch import DIMENSIONS, PRIME_BITS, TOWER_COUNTS
from benchmark_report import validate_jsonl
from ntt_oracle import find_primes, find_psi

with tempfile.TemporaryDirectory() as directory:
    for dimension in DIMENSIONS:
        for bits in PRIME_BITS:
            for towers in TOWER_COUNTS:
                moduli = find_primes(dimension, towers, bits)
                roots = [find_psi(modulus, dimension) for modulus in moduli]
                vector = Path(directory) / 'parameters.txt'
                vector.write_text(f'{dimension} {towers}\n' + ''.join(
                    f'{q} {root}\n' for q, root in zip(moduli, roots)) + 'COEFFS\n' + '0\n' * dimension)
                raw = subprocess.check_output([sys.argv[1], str(vector), '1', '2'], text=True, timeout=120)
                result = validate_jsonl(raw, expected_n=dimension, expected_bits=bits,
                    expected_towers=towers, expected_timing='host_wall_openfhe_native')
                if result['workload']['moduli'] != moduli or result['workload']['roots'] != roots:
                    raise AssertionError('OpenFHE benchmark did not preserve parameters')
print(f'{len(DIMENSIONS) * len(PRIME_BITS) * len(TOWER_COUNTS)} CPU/OpenFHE workload protocol checks passed; no GPU execution')
