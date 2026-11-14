"""Generate a vector file consumed by ntt_cli, unit tests, and test_vectors.py.

Usage: gen_vectors.py <out-path> [N=65536] [num_primes=3] [seed=1] [bits=46]
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ntt_oracle import find_primes, find_psi


def main() -> int:
    out = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 65536
    num_primes = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    bits = int(sys.argv[5]) if len(sys.argv) > 5 else 46
    if n < 2 or n & (n - 1) or n > 65536 or not 1 <= num_primes <= 65535 or not 2 <= bits <= 62:
        raise ValueError("invalid dimension, prime count or width")

    primes = find_primes(n, num_primes, bits)
    psis = [find_psi(q, n) for q in primes]

    q_total = 1
    for q in primes:
        q_total *= q

    rng = random.Random(seed)
    coeffs = [rng.randrange(q_total) for _ in range(n)]

    with open(out, "w", encoding="utf-8") as f:
        f.write(
            "# ntt-gpu vector v1: N, primes+psis, "
            "decimal coefficients in [0, Q)\n"
        )
        f.write(f"{n} {num_primes}\n")
        for q, psi in zip(primes, psis):
            f.write(f"{q} {psi}\n")
        f.write("COEFFS\n")
        for c in coeffs:
            f.write(f"{c}\n")

    prime_hex_values = [f"{q:x}" for q in primes]
    print(
        f"wrote {out}: N={n}, primes={prime_hex_values}, "
        f"Q~2^{q_total.bit_length()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
