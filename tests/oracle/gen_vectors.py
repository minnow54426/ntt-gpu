"""Generate a vector file consumed by ntt_cli, unit tests, and test_vectors.py.

Usage: gen_vectors.py <out-path> [N=65536] [num_primes=3] [seed=1] [bits=46]
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ntt_oracle import find_primes, find_psi


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out')
    parser.add_argument('n', type=int, nargs='?', default=65536)
    parser.add_argument('num_primes', type=int, nargs='?', default=3)
    parser.add_argument('seed', type=int, nargs='?', default=1)
    parser.add_argument('bits', type=int, nargs='?', default=46)
    parser.add_argument('--zero-coefficients', action='store_true',
                        help='Small benchmark parameter file; zero placeholders, not a correctness fixture')
    args = parser.parse_args()
    out, n, num_primes, seed, bits = args.out, args.n, args.num_primes, args.seed, args.bits
    if n < 2 or n & (n - 1) or n > 65536 or not 1 <= num_primes <= 65535 or not 2 <= bits <= 62:
        raise ValueError("invalid dimension, prime count or width")

    primes = find_primes(n, num_primes, bits)
    psis = [find_psi(q, n) for q in primes]

    q_total = 1
    for q in primes:
        q_total *= q

    rng = random.Random(seed)
    coeffs = [0] * n if args.zero_coefficients else [rng.randrange(q_total) for _ in range(n)]

    with open(out, "w", encoding="utf-8") as f:
        f.write(
            "# ntt-gpu vector v1: N, primes+psis, "
            "decimal coefficients in [0, Q)\n"
        )
        if args.zero_coefficients:
            f.write('# coefficient policy: zero placeholders for benchmark parameters only\n')
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
