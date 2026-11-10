"""Check compiled host primitives against Python arbitrary-precision arithmetic.

No shared C++ reduction expression is used to calculate expected answers.
Invalid preconditions that invoke undefined behavior (notably q=0) are NOT run.
"""

import random
import subprocess
import sys

from ntt_oracle import find_primes


def main():
    rng = random.Random(0xD6)
    queries, expected = [], []

    def check(operation, first, second, modulus, answer):
        queries.append(f"{operation} {first} {second} {modulus}\n")
        expected.append(str(answer))

    # add/sub retain their advertised q < 2^63 contract. mul/pow additionally
    # exercise full-width products and arbitrary (not necessarily prime) q.
    moduli = [1, 2, 17, 1000000007]
    for bits in [32, 50, 55, 60, 63, 64]:
        moduli.extend([(1 << bits) - 1, (1 << bits) - 25])
    for modulus in moduli:
        edges = {value % modulus for value in
                 [0, 1, 2, modulus // 2, modulus - 2, modulus - 1,
                  (1 << 32) - 1, 1 << 32, (1 << 32) + 1]}
        pairs = [(a, b) for a in sorted(edges) for b in sorted(edges)]
        pairs += [(rng.randrange(modulus), rng.randrange(modulus))
                  for _ in range(256)]
        for first, second in pairs:
            if modulus < 1 << 63:
                check("add", first, second, modulus, (first + second) % modulus)
                check("sub", first, second, modulus, (first - second) % modulus)
            check("mul", first, second, modulus, first * second % modulus)
        for first in sorted(edges) + [(1 << 64) - 1]:
            for exponent in [0, 1, 2, 63, 64, (1 << 64) - 1]:
                check("pow", first, exponent, modulus,
                      pow(first, exponent, modulus))
    # Parameter search supplies compatible 50/55/60-bit primes. Expected
    # inverses use Python's modular-inverse algorithm, not Fermat/pow(q-2).
    primes = [2, 17, 1000000007]
    primes += [find_primes(65536, 1, bits)[0] for bits in [50, 55, 60]]
    for modulus in primes:
        for first in [1, modulus - 1, modulus + 1, (1 << 64) - 1]:
            if first % modulus:
                check("inv", first, 0, modulus, pow(first, -1, modulus))
        for first in [0, modulus]:
            check("inv", first, 0, modulus, "invalid")

    result = subprocess.run([sys.argv[1]], input="".join(queries), text=True,
                            capture_output=True, check=True)
    actual = result.stdout.splitlines()
    if len(actual) != len(expected):
        raise AssertionError(f"expected {len(expected)} results, got {len(actual)}")
    for query, got, want in zip(queries, actual, expected):
        if got != want:
            raise AssertionError(f"{query.strip()}: got {got}, want {want}")
    print(f"ARITHMETIC BIGINT CHECK PASSED: {len(expected)} exact results")


if __name__ == "__main__":
    main()
