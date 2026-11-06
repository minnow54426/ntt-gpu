"""Cross-language check: the C++ implementation vs. the big-int oracle.

For every limb (prime):
  1. per-limb forward outputs must equal the naive oracle transform exactly;
  2. CRT reconstruction of the limbs must equal the oracle applied per-limb
     (demonstrates RNS composition end to end);
plus the CLI's roundtrip-identity self-check must exit 0.

Usage: test_vectors.py <ntt_cli-path> <vector-file>
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ntt_oracle import crt_reconstruct, negacyclic_ntt_naive, read_vector_file


def run_cli(cli: str, vec: str, cmd: str) -> str:
    r = subprocess.run([cli, vec, cmd], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ntt_cli {cmd} failed: {r.stderr}")
    return r.stdout


def main() -> int:
    cli, vec = sys.argv[1], sys.argv[2]
    n, primes, psis, coeffs = read_vector_file(vec)

    # Parse "LIMB k" blocks from the CLI transform output.
    limbs = []
    cur = None
    expect_index = False
    for tok in run_cli(cli, vec, "transform").split():
        if tok == "LIMB":
            if cur is not None:
                limbs.append(cur)
            cur = []
            expect_index = True
            continue
        if expect_index:
            expect_index = False
            continue
        if cur is None:
            raise RuntimeError("malformed CLI output")
        cur.append(int(tok))
    if cur is not None:
        limbs.append(cur)

    if len(limbs) != len(primes) or any(len(b) != n for b in limbs):
        block_lengths = [len(block) for block in limbs]
        raise RuntimeError(
            f"expected {len(primes)} blocks of {n}, got {block_lengths}"
        )
    limb_outputs = limbs

    oracle_limbs = []
    for k, (q, psi) in enumerate(zip(primes, psis)):
        a = [c % q for c in coeffs]
        want = negacyclic_ntt_naive(a, q, psi)
        # The reference forward is DIF: it emits the values in BIT-REVERSED
        # evaluation order (see ntt.hpp) so that forward -> multiply ->
        # inverse needs no bit-reversal pass. Apply the same permutation to
        # the definitional oracle output before comparing.
        lg = n.bit_length() - 1
        want = [want[int(bin(j)[2:].zfill(lg)[::-1], 2)] for j in range(n)]
        got = limb_outputs[k]
        if got != want:
            for j in (x for x in range(n) if got[x] != want[x]):
                print(f"MISMATCH limb {k} coeff {j}: got {got[j]}, want {want[j]}")
                break
            print("CROSS-CHECK FAILED")
            return 1
        oracle_limbs.append(want)

    # RNS composition: CRT the limbs position-wise and compare full-precision.
    for j in range(n):
        got_crt = crt_reconstruct(
            [limb_outputs[k][j] for k in range(len(primes))], primes
        )
        want_crt = crt_reconstruct(
            [oracle_limbs[k][j] for k in range(len(primes))], primes
        )
        if got_crt != want_crt:
            print(f"CRT RECONSTRUCTION MISMATCH at coeff {j}")
            return 1

    if "IDENTITY OK" not in run_cli(cli, vec, "identity"):
        print("IDENTITY CHECK FAILED")
        return 1

    print(
        f"CROSS-CHECK PASSED: {len(primes)} limbs x N={n}, "
        "CRT composition verified"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
