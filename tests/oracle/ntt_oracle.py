"""Big-integer oracle for the negacyclic NTT. Stdlib only.

Ground truth is computed with Python's arbitrary-precision integers, naive
O(N^2) evaluation, and zero shared code with the C++ implementation.

Convention (must match include/ntt/ntt.hpp exactly):
    psi = primitive 2N-th root of unity mod q   (psi^N == -1)
    forward:  A_j = sum_i a_i * psi^((2j+1)*i) mod q
"""

import random

# Deterministic for n < 3.3e24 with these bases (well beyond our 60-bit primes).
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in _MR_BASES:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _MR_BASES:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def find_primes(n: int, count: int, bits: int = 46) -> list:
    """`count` distinct primes q with (2n) | (q-1) and 2^(bits-1) <= q < 2^bits."""
    step = 2 * n
    q = ((1 << (bits - 1)) // step + 1) * step + 1
    primes = []
    while len(primes) < count:
        if q >= (1 << bits):
            raise RuntimeError("prime search exhausted the bit range")
        if is_prime(q):
            primes.append(q)
        q += step
    return primes


def find_psi(q: int, n: int) -> int:
    """Element of order exactly 2n mod q: zeta with zeta^n == -1."""
    rng = random.Random(q ^ n)
    while True:
        g = rng.randrange(2, q - 1)
        zeta = pow(g, (q - 1) // (2 * n), q)
        if pow(zeta, n, q) == q - 1:
            return zeta


def negacyclic_ntt_naive(a: list, q: int, psi: int) -> list:
    """Definition of the transform: A_j = sum_i a_i * psi^((2j+1)*i) mod q."""
    n = len(a)
    table = [1] * (2 * n)  # psi^k mod q for k in [0, 2n)
    for k in range(1, 2 * n):
        table[k] = table[k - 1] * psi % q
    out = []
    for j in range(n):
        acc = 0
        base = (2 * j + 1) % (2 * n)
        for i in range(n):
            acc += a[i] * table[(base * i) % (2 * n)]
        out.append(acc % q)
    return out


def crt_reconstruct(limb_values: list, primes: list) -> int:
    """CRT over arbitrary-precision integers: unique x in [0, Q) matching all limbs.

    limb_values: one *scalar* residue per prime, not per-limb arrays.
    """
    for v in limb_values:
        if not isinstance(v, int):
            raise TypeError(f"crt_reconstruct expects scalars, got {type(v).__name__}")
    q_total = 1
    for q in primes:
        q_total *= q
    x = 0
    for v, q in zip(limb_values, primes):
        m = q_total // q
        x += v * m * pow(m % q, q - 2, q)  # Fermat inverse (q prime)
    return x % q_total


def read_vector_file(path: str):
    n = None
    q, psi, coeffs = [], [], []
    with open(path) as f:
        tokens = [t for line in f if not line.startswith("#") for t in line.split()]
    pos = 0

    def nxt():
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        return tok

    n = int(nxt())
    num_primes = int(nxt())
    for _ in range(num_primes):
        q.append(int(nxt()))
        psi.append(int(nxt()))
    assert nxt() == "COEFFS"
    for _ in range(n):
        coeffs.append(int(nxt()))
    assert pos == len(tokens), "trailing tokens in vector file"
    return n, q, psi, coeffs
