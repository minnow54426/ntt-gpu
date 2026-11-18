<div id="modular-arithmetic-bounds-and-exactness-gates" aria-hidden="true"></div>

# Keep every modular operation exact

For a prime q, a **canonical residue** is an integer from 0 to q−1. For example,
−1 is stored as 16 modulo 17. Every butterfly in this implementation starts
and ends with canonical values.

The main target is N=65536 coefficients with 50–60-bit primes. Each stored
value uses `uint64_t`, an unsigned 64-bit integer. One 60-bit value fits,
but the product of two can need 120 bits. This page
explains the temporary widths, input assumptions and independent checks.
Read the [buffer map](cuda-buffer-dataflow.md) for where arrays live.
Sources: [mod_arith.hpp](../include/ntt/mod_arith.hpp),
[ntt.hpp](../include/ntt/ntt.hpp) and [dev_core.hpp](../include/ntt/dev_core.hpp).

## 1. What callers must supply

The notation \([0,q)\) includes 0 and excludes q: it means integers
0, 1, …, q−1. These small arithmetic functions run inside the main loops,
so they assume their inputs are in range instead of checking every call.

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
The code stores ordinary residues. It does not use Montgomery encoding
(an alternative arithmetic representation) or lazy reduction (temporarily
allowing values above q−1).

The raw CPU `NegacyclicNtt` constructor checks size, divisibility and
ψᴺ=q−1, but assumes primality and the documented width. Its array methods
check length, not residue bounds. The [CUDA wrapper](cuda-buffer-dataflow.md)
and resident runtime add their own validation; do not assume the raw hot
arithmetic functions reject invalid inputs.

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

There are two things to preserve: **what each value means** and **how it is
stored**. A stage must compute the correct partial polynomial evaluations,
and every stored word must remain in \([0,q)\). The negacyclic merge equations
explain both. All equations in this section use arithmetic modulo q.

### Split a negacyclic polynomial into two smaller problems

Start with \(a(x)\) modulo \(x^N+1\). Set \(d=N/2\) and split its coefficient
array into low and high halves:

\[
\begin{aligned}
a(x)&=U(x)+x^dV(x),\\
U(x)&=\sum_{i=0}^{d-1}a_i x^i,\\
V(x)&=\sum_{i=0}^{d-1}a_{i+d}x^i.
\end{aligned}
\]

Because \(\psi^N=-1\), choose \(\alpha=\psi^d\), so \(\alpha^2=-1\).
The original modulus factors as

\[
x^N+1=(x^d-\alpha)(x^d+\alpha).
\]

Inside the first smaller ring, \(x^d=\alpha\); inside the second,
\(x^d=-\alpha\). Substitution gives the **negacyclic merge**:

\[
\begin{aligned}
a(x)\bmod(x^d-\alpha)&=U(x)+\alpha V(x),\\
a(x)\bmod(x^d+\alpha)&=U(x)-\alpha V(x).
\end{aligned}
\]

For each coefficient pair \(u=a_i\), \(v=a_{i+d}\), the two output words
are therefore \(u+\alpha v\) and \(u-\alpha v\). One multiplication is shared
by the plus and minus branches: this is a butterfly. The minus sign comes
from the two factors of the negacyclic modulus, not from an extra array operation.

### Repeat the same merge through all 16 stages

At a later stage, a block of 2d coefficients represents a remainder modulo
\(x^{2d}-\beta\). Choose the appropriate \(\alpha\) with
\(\alpha^2=\beta\). The same identity applies:

\[
x^{2d}-\beta=(x^d-\alpha)(x^d+\alpha).
\]

The block splits into its two child remainders \(U+\alpha V\) and
\(U-\alpha V\). This is the **meaning invariant**: each block is still the
original polynomial reduced modulo its current factor. At d=1, each factor
is linear, \(x-z\), and its remainder is exactly \(a(z)\).

For the default [shared CPU/CUDA stage helper](../include/ntt/transform_stage.hpp),
let \(m=N/(2d)\) be the number of groups, \(g\in[0,m)\) the group index,
and \(\operatorname{rev}_L\) reverse L bits, where \(L=\log_2N\). Its group twiddle is

\[
\alpha_{m,g}=\psi^{\operatorname{rev}_L(m+g)}\bmod q.
\]

One twiddle applies to every coefficient pair in that group. As d runs from
N/2 down to 1, these choices select the successive factors above. At the end,
array position k holds

\[
a\!\left(\psi^{2\operatorname{rev}_L(k)+1}\right)\bmod q.
\]

Thus the transform produces evaluations at the roots of \(x^N+1\), in
bit-reversed order. It keeps N words throughout; the intermediate words are
partial remainders, not final evaluations yet. See the
[exact evaluation formulas](openfhe-representation-contract.md#3-exact-evaluation-and-inverse-formulas)
for the direct sum that these butterflies compute.

### Keep the stored values canonical, and reverse the merge

The default forward butterfly implements those child remainders with
canonical operations:

\[
\begin{aligned}
t&=\operatorname{mul\_mod}(\alpha,v,q),\\
p&=\operatorname{add\_mod}(u,t,q),\\
r&=\operatorname{sub\_mod}(u,t,q).
\end{aligned}
\]

The inputs u, v and table entry α are in \([0,q)\). Multiplication reduces
the full-width product before storing t; addition and subtraction reduce
before storing p and r. Consequently **each output is again in \([0,q)\)**,
even though an unreduced sum could reach 2q−2 or a difference could be negative.
This proves the range invariant by induction over all 16 stages.

To reconstruct the parent, add and subtract its two child remainders:

\[
\begin{aligned}
p+r&\equiv2u\pmod q,\\
(p-r)\alpha^{-1}&\equiv2v\pmod q.
\end{aligned}
\]

The inverse helper forms a canonical difference before multiplying:

\[
\begin{aligned}
\delta&=\operatorname{sub\_mod}(p,r,q),\\
\widetilde u&=\operatorname{add\_mod}(p,r,q),\\
\widetilde v&=\operatorname{mul\_mod}(\delta,\alpha^{-1},q).
\end{aligned}
\]

Here \(\widetilde u\) and \(\widetilde v\) represent 2u and 2v modulo q.
The helper postpones division by 2: every reversed stage contributes a factor 2,
so 16 stages contribute \(2^{16}=N\). One final multiplication by
\(N^{-1}\bmod q\) recovers the original coefficients. All these stored
intermediates are canonical too. Pointwise multiplication preserves the same
range; its evaluation products correspond to polynomial multiplication modulo
\(x^N+1\).

### A complete four-word example

For illustration, use N=4, q=17 and ψ=2. Start with canonical coefficients
`[1, 15, 3, 15]`, representing \(1-2x+3x^2-2x^3\) modulo 17.

```text
Forward, default group-twiddle schedule:
  d=2, alpha=4:       [1, 15, 3, 15] -> [13, 7, 6, 6]
  d=1, alphas=2, 8:  [13, 7, 6, 6] -> [10, 16, 3, 9]

Inverse, without the final scale yet:
  d=1, inverses=9,15: [10, 16, 3, 9] -> [9, 14, 12, 12]
  d=2, inverse=13:    [9, 14, 12, 12] -> [4, 9, 12, 9]
  multiply by 4^-1=13 modulo 17:      -> [1, 15, 3, 15]
```

For example, the first forward pair is
\(t=4\times3\bmod17=12\), \(p=(1+12)\bmod17=13\),
\(r=(1-12)\bmod17=6\). Every displayed word stays between 0 and 16.
The final values evaluate the polynomial at `[2, 15, 8, 9]`, the odd powers
of ψ in bit-reversed order. N=4 only makes the arithmetic visible; the project
workload remains N=65536.

<details markdown="1">
<summary>How the CPU reference folds the same negacyclic twist into its butterflies</summary>

The [CPU reference](../include/ntt/ntt.hpp) and optional legacy CUDA build use
a different intermediate layout. Write \(b_i=\psi^i a_i\) and
\(\omega=\psi^2\). The negacyclic evaluation sum becomes a cyclic sum:

\[
a(\psi^{2j+1})=\sum_i a_i\psi^{(2j+1)i}
=\sum_i b_i\omega^{ji}.
\]

The reference carries the factors \(\psi^i\) implicitly rather than storing
a separate twisted array. For a pair at local offset i and distance d,
write the implicit inputs as \(\psi^iu\) and \(\psi^{i+d}v\).
With \(c=\psi^d\) and \(w=\psi^{iN/d}\), a cyclic DIF split factors as

\[
\begin{aligned}
\psi^iu+\psi^{i+d}v&=\psi^i(u+cv),\\
(\psi^iu-\psi^{i+d}v)w&=\psi^i(u-cv)w.
\end{aligned}
\]

After leaving the common \(\psi^i\) implicit, the stored pair is
\(p=(u+cv)\bmod q\), \(r=(u-cv)w\bmod q\).
The reverse butterfly computes \(t=rw^{-1}\bmod q\), then
\((p+t)\bmod q\) and \((p-t)c^{-1}\bmod q\), giving 2u and 2v.
It also postpones normalization until the final N-inverse multiply.

For the example above, the reference's first-stage array is `[13, 7, 6, 7]`,
while the default helper's is `[13, 7, 6, 6]`. Their final evaluations are
identical. Do not require intermediate arrays from these two schedules to
match. Each schedule preserves its own partial-transform meaning and the
same canonical range.

</details>

There is **no stage where values are intentionally in \([0,2q)\) or \([0,4q)\)**.
The upstream [`native forward NTT`](https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/core/include/math/hal/intnat/transformnat-impl.h#L337-L507) uses lazy intermediate arithmetic. Feeding lazy OpenFHE/internal buffers directly into this interface without
an explicit representation contract would violate its assumptions.

### What arithmetic work is actually counted

N=65536 has \(32768\times16=524288\) butterflies per direction and per tower.
The CPU reference uses three modular multiplications per butterfly: two
for data and one to advance the twiddle. The optional legacy device schedule
uses two, reading precomputed twiddles instead. The default group-twiddle
helper uses one, as in the merge equations above. Thus their loop bodies
perform 1,572,864, 1,048,576 and 524,288 multiplications respectively;
inverse adds 65,536 normalization multiplies per tower in each schedule.
Setup powers/table generation are additional work.

The old arithmetic-header comment counted the CPU loop incorrectly; D6 fixes
that comment without changing the algorithms. `__uint128_t` remainder is
the correctness baseline. The CUDA compiler may translate this remainder expression into many
instructions. Inspect the generated GPU code and profile it; one source
expression does not imply one hardware division instruction.

## 4. Executable adversarial coverage

Two required checks, called gates, run in the ordinary CPU CTest suite.
They use difficult boundary inputs as well as normal examples:

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

The six transform inputs are all zero, all q−1, alternating zero/q−1,
one nonzero final coefficient, increasing values, and repeatable pseudorandom
values. A forward/inverse round trip may hide matching mistakes in the two
directions. Independent evaluation checks are also needed to establish the
required order and OpenFHE compatibility.

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
| Native CUDA oracle/reference gates (when CUDA is available) | Actual kernel outputs versus reference/oracle | Run on named hardware; see the dated Ada report for exercised cases |

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
  does not establish device reduction correctness or timing for a new variant.
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
