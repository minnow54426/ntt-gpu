# Modular arithmetic: bounds and exactness gates

D6 connects the butterfly formulas to the integer operations they actually
execute. The current target is **N=65536, with 50–60-bit prime towers stored
in uint64_t**. This is a correctness contract, not a GPU performance result.

Read [the buffer/dataflow map](cuda-buffer-dataflow.md) first for where the
arrays live. The implementation is in [mod_arith.hpp](../include/ntt/mod_arith.hpp),
[ntt.hpp](../include/ntt/ntt.hpp), and [dev_core.hpp](../include/ntt/dev_core.hpp).

## 1. What callers must supply

Canonical means an integer in \([0,q)\). None of the hot arithmetic functions
validates this range at runtime.

| Operation | Required input | Output |
|---|---|---|
| `add_mod(a,b,q)` | Canonical a,b; \(0<q<2^{63}\) | Canonical \((a+b)\bmod q\) |
| `sub_mod(a,b,q)` | Canonical a,b; \(0<q<2^{63}\) | Canonical \((a-b)\bmod q\) |
| `mul_mod(a,b,q)` | Canonical a,b; \(0<q<2^{64}\) | Canonical \(ab\bmod q\) |
| `pow_mod(a,e,q)` | Any uint64_t a,e; q>0 | Canonical \(a^e\bmod q\); q=1 returns 0 |
| `inv_mod(x,q)` | Prime q≥2; x not divisible by q | Canonical inverse via \(x^{q-2}\) |

Add/sub retain their documented conservative bound; this review does not
expand the public contract. Multiplication's wider bound does **not** make a
full transform with q≥2^63 part of the supported contract.

For a transform, also require a power-of-two N≥2, prime q with \(2N\mid(q-1)\),
and the same primitive 2N-th root ψ as the comparison implementation. At the
target N, \(2N=131072\) and q>N. Forward consumes canonical coefficients in
natural order; inverse consumes canonical evaluations in bit-reversed order.
There is no Montgomery representation or lazy residue format in this code.

The constructor checks size shape, divisibility and \(\psi^N=q-1\), but **does not
test primality or the advertised modulus-width limit**. Public array methods
check length, not residue bounds. Trusted parameter selection and canonical
input are caller obligations today, not rejection guarantees.

## 2. Why intermediates fit

For canonical a,b:

- Addition: \(0\le a+b\le 2q-2\). One subtraction is sufficient. The code forms
  the sum in 128 bits; after subtraction it is below q and fits uint64_t.
- Subtraction: if a≥b, use a-b. Otherwise \(0\le a+(q-b)<q\); the selected
  branch therefore cannot overflow uint64_t. Do not compute unsigned a-b
  first and assume wraparound implements reduction modulo q.
- Multiplication: \(0\le ab\le(q-1)^2<2^{128}\). The cast happens **before**
  multiplication, preserving the full product; remainder produces a value
  below q. Multiplying in uint64_t first would lose the high half.
- At q<2^60, the product is below 2^120 and a sum below 2^61. Storage in
  uint64_t is safe, but a uint64_t product is still insufficient.
- Exponentiation reduces a before its first multiply. Each multiplication
  preserves canonical range. A uint64_t exponent takes at most 64 loop
  iterations. Even `e=0` computes `1 % q`, so q=0 is never legal.

These are bounds for the current exact-remainder code. Barrett, Shoup,
Montgomery, 32-bit limbs, or lazy reduction will need their own proofs,
including quotient error, carries, temporary width, and final correction.

## 3. The invariant through every butterfly

Forward and inverse use these canonical modular operations:

\[
\begin{aligned}
\text{Forward: }&t=cv\bmod q,\quad u'=u+t\bmod q,\quad v'=(u-t)w\bmod q,\\
\text{Inverse: }&t=yw^{-1}\bmod q,\quad u'=x+t\bmod q,\quad v'=(x-t)c^{-1}\bmod q.
\end{aligned}
\]

Tables contain canonical powers.
Every operation above returns canonical values, so induction over all 16
stages preserves \([0,q)\). Inverse's final multiplication by N-inverse does too.
Pointwise multiplication has the same invariant.

There is **no stage where values are intentionally in \([0,2q)\) or \([0,4q)\)**.
Feeding lazy OpenFHE/internal buffers directly into this interface without
an explicit representation contract would violate its assumptions.

### What arithmetic work is actually counted

N=65536 has \(32768\times16=524288\) butterflies per direction and per tower.
The CPU reference uses three modular multiplications per butterfly: two
for data and one to advance the twiddle. Shared device-stage helpers use
two, reading precomputed twiddles instead. Thus the loop bodies perform
1,572,864 and 1,048,576 multiplications respectively; inverse adds 65,536
normalization multiplies. Setup powers/table generation are additional work.

The old arithmetic-header comment counted the CPU loop incorrectly; D6 fixes
that comment without changing the algorithms. `__uint128_t` remainder is
the correctness baseline. Its CUDA lowering and cost must be checked in a
native build/profile, not inferred as one hardware division instruction.

## 4. Executable adversarial coverage

Two new gates run in the normal CPU CTest suite:

| Gate | Inputs and comparison | What it establishes |
|---|---|---|
| `arithmetic_bigint_check` | 14,956 compiled results versus Python arbitrary-precision addition, subtraction, multiplication, power and modular inverse | Exact host primitive answers, not a copy of the same C++ reduction expression |
| `arithmetic_transform_check` | N=65536 at generated 50-, 55-, 60-bit primes; six patterns in both coefficient/evaluation roundtrips | Canonical forward/inverse results and two-sided roundtrip identity |
| Same transform gate | \(x^{N-1}\cdot x\) must produce q-1 at index 0 and zero elsewhere | Closed-form negacyclic wrap sign, not merely roundtrip consistency |

Primitive inputs include 0, 1, 2, q/2, q-2, q-1, values around 2^32, and
256 seeded random pairs per modulus. Moduli include small, even, composite,
50/55/60-bit boundaries, just below 2^63, and full-width uint64_t boundaries.
The latter use mul/pow only, respecting the add/sub contract. Exponents
include 0, 1, 2, 63, 64 and UINT64_MAX; pow also receives an unreduced base.
Inverse checks small and generated 50/55/60-bit primes against Python's
modular-inverse algorithm, and confirms the host exception for zero/multiples
of q. Parameter generation is reused; this is not an independent primality
certification test.

The six transform patterns are zero, all q-1, alternating zero/q-1, a final
coefficient impulse, ramp and deterministic pseudorandom. A roundtrip alone
cannot prove spectral ordering or OpenFHE compatibility; use the other gates.

```sh
cmake -S . -B build
cmake --build build -j 8
ctest --test-dir build --output-on-failure
ctest --test-dir build -R 'arithmetic_' -V
```

Sources: [C++ probe and full-size tests](../tests/arithmetic_contract_tests.cpp),
[independent primitive oracle](../tests/oracle/test_arithmetic.py).

## 5. Exactness ladder: do not substitute one gate for another

| Existing gate | Scope | Limit |
|---|---|---|
| `unit_tests` | Small arithmetic and NTT product against schoolbook | Schoolbook shares arithmetic primitives |
| `oracle_crosscheck` | N=1024 forward arrays against direct Python NTT sum, plus CRT and identity | Not a full-size GPU test |
| Column/inverse/tile mapping checks | Exhaustive target-size index/twiddle probes of shared CPU/CUDA helpers | CPU execution; no CUDA barriers/races tested |
| `device_schedule_check` | Exact simulator/reference arrays, including N=65536 fixtures | Current fixtures use 46-bit primes |
| `openfhe_exact_compat` (opt-in) | N=65536 NativePoly forward/inverse arrays, including independent evaluation input; OpenFHE selects parameters via FirstPrime(50) and LastPrime(59) | Pinned OpenFHE 1.6.1 CPU comparison, not a CUDA/OpenFHE pipeline |
| Native CUDA oracle/reference gates (when CUDA is available) | Actual kernel outputs versus reference/oracle | Must be run on named hardware; not verified on this Mac |

The OpenFHE gate includes zero, all q-1, ramp, random coefficients and random
evaluation arrays. Same q **and root** are essential: another valid root may
give a different evaluation permutation. The [D7 contract](openfhe-representation-contract.md) freezes the representation
and tower boundary. CKKS approximate decryption is not a replacement
for these exact per-index comparisons.

## 6. Deferred adversarial checks and unsafe inputs

D6's review is complete; these are explicit later implementation gates:

- **P2/P3, real GPU:** repeat the 50/55/60-bit patterns through actual kernels,
  compare each forward/inverse/pointwise result and canonical range, then run
  compute-sanitizer race/memory checks and test launch failures. CPU success
  does not establish nvcc support, device reduction correctness, or timing.
- **API hardening:** D9/D10 define the contracts. Later
  [offline preparation](offline-gpu-preparation.md) adds baseline rejection of
  noncanonical residues, invalid roots, composite moduli, unsafe dimensions
  and vector lengths, with host negative tests. Runtime format/provenance tags,
  device failure paths and the production resident API remain later gates;
  current APIs do not reject every possible misuse.
- **Parameter helpers:** `find_primes` currently assumes safe shifts and a
  nonzero, nonoverflowing `2N`. Invalid bit counts can cause undefined shifts;
  `find_psi` assumes a compatible prime q>3 and has no attempt limit. These
  helpers are not hardened untrusted-input parsers.
- **Raw primitive misuse:** q=0 causes invalid remainder/division operations;
  it is not a test input to execute. Composite q violates Fermat inverse's
  premise even if x is invertible: for example q=15,x=2 returns 2, not 8.
  Host zero-input inverse throws; the device branch returns 0 and is not a
  supported inverse/error-reporting API. Device kernels currently never invert.
- **Future reduction optimization:** exercise maximum correction counts,
  carry/borrow transitions, high-word products and every accepted lazy range
  against the independent bigint gate before allowing an optimized dispatch.

These gaps do not block the D6 source-review deliverable. They do block claims
of a hardened production API or GPU/OpenFHE integration. The
[D8 Hybrid graph](openfhe-hybrid-operation-graph.md) and
[D9 buffer/stream/cache contract](device-backend-contract.md) are complete.
[D10's ADR](architecture-decision.md) closes P1's design gate; native hardening
and implementation are tracked in the [backlog](implementation-backlog.md).
