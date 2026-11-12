# OpenFHE Hybrid key switching: the exact operation graph

D8 traces **OpenFHE 1.6.1**, commit
`b59e1f291965f14fbee5172cd83a20f5b856204f`, with **N=65536** and the
64-bit NativeInteger backend. It fixes the sequence of representation changes,
CRT representatives, key addressing, accumulation, and reduction that a future
backend must preserve. Read the [D7 representation contract](openfhe-representation-contract.md)
for the NTT's exact array order.

The executable evidence is [openfhe_hybrid_tests.cpp](../tests/openfhe_hybrid_tests.cpp).
It calls the actual OpenFHE PKE library and compares complete arrays with an
independent unreduced-CRT formula and scalar arithmetic. These are CPU results.

## 1. Read the complete path

The ordinary path starts at `KeySwitchCore(c,key)`. Its output is two
**contributions** in the active Q basis, ordered **B first, A second**.
The ciphertext wrapper decides how to combine them with existing components.

<ol class="site-plan">
<li><strong>Split the active Q towers</strong><span>Contiguous groups of up to α towers form the digits. Keep their original evaluation arrays.</span></li>
<li><strong>Convert each digit to coefficients</strong><span>Inverse NTT only the towers in that digit.</span></li>
<li><strong>Extend to its complement</strong><span>Approximate CRT produces coefficients in every other active Q tower and every P tower.</span></li>
<li><strong>Return the complement to evaluations</strong><span>Forward NTT those new towers, then stitch them around the saved original digit towers.</span></li>
<li><strong>Multiply and accumulate in QlP</strong><span>All digits contribute to two evaluation arrays, using the B and A key vectors.</span></li>
<li><strong>Reduce each accumulated result</strong><span>Inverse NTT its P towers, approximate-switch P to active Q, forward NTT the correction, subtract, and multiply by the inverse of P.</span></li>
<li><strong>Update the ciphertext</strong><span>Add the first contribution to c0; add or replace c1 according to the caller.</span></li>
</ol>

The implementation has no general coefficient-domain round trip of all QlP
towers. It preserves the original digit evaluations during extension and
leaves Q evaluations intact during ModDown.

## 2. Names, levels, and tower order

Let M be the full context's Q-tower count, L the current ciphertext's active
Q-tower count, K the P-tower count, and D the configured large-digit count.
The active basis is an ordered prefix of Q:

\[
Q_\ell=\prod_{i=0}^{L-1}q_i,\qquad
P=\prod_{j=0}^{K-1}p_j,\qquad
\alpha=\left\lceil\frac{M}{D}\right\rceil,\qquad
\beta_\ell=\left\lceil\frac{L}{\alpha}\right\rceil.
\]

The source also caps the active digit count at the configured partition count.
For valid active prefixes this does not alter the formula above. **α belongs
to the full context; do not recompute it from L.** The last active digit may
be shorter than α, including when only part of the first digit remains.

| Object | Tower order | Representation |
|---|---|---|
| Input component | q₀ … qₗ₋₁ | Evaluation, bit-reversed within each tower |
| Digit d's source | q at indices dα … min((d+1)α,L)−1 | Evaluation, then coefficient |
| Digit d's complement | Other active Q towers in order, then p₀ … pₖ₋₁ | Coefficient, then evaluation |
| Assembled digit | Active Q prefix, then all P | Evaluation |
| Stored key pair for digit d | Full Q, then all P | Evaluation |
| Each MAC output | Active Q prefix, then all P | Evaluation |
| Each ModDown output | Active Q prefix | Evaluation |

Here L is a **tower count**, not the ciphertext's public `GetLevel()` value.
Derive it from the actual element parameters; the [D9 proposal](device-backend-contract.md) keeps these metadata
consistent. Every tower retains its exact OpenFHE modulus and root.

## 3. Why approximate CRT must keep the unreduced sum

For a digit's actual source basis \(B=\prod_i b_i\), define
\(\widehat B_i=B/b_i\). At coefficient index r, the implementation computes:

\[
t_i=[x_i\widehat B_i^{-1}]_{b_i},\qquad
S=\sum_i t_i\widehat B_i,\qquad
\widetilde x_m=[S]_m.
\]

Brackets mean the canonical residue in the indicated modulus. The final
modulus m is a complement tower. **There is no reduction of S modulo B**
before reducing it modulo m. In general:

\[
S=x+\kappa B,\qquad 0\le x<B,\qquad
[S]_m\ne[x]_m.
\]

`ApproxSwitchCRTBasis` implements this sum using the precomputed hat inverses,
hat residues, and modular multiplication/accumulation. Our test independently
forms the full unreduced sum with arbitrary-precision `BigInteger` arithmetic;
it does not read those conversion tables to compute its expected answer.

The digit is converted to coefficient format **before** this operation. Its
complement is then transformed using each destination tower's own root. The
digit's original evaluation towers are copied directly into their slots.
Applying the CRT formula to evaluation coordinates would change the result.

This contract uses **WITH_REDUCED_NOISE=OFF**. The centered-conversion branch
selected by that build option is different and is outside this fixture matrix.

## 4. A reduced level changes key addresses

A key keeps all M Q towers followed by K P towers. A ciphertext digit contains
only L Q towers followed by those same K P towers. For active-array index i:

\[
\operatorname{keyIndex}(i)=
\begin{cases}
i,&i<L,\\
i+(M-L),&i\ge L.
\end{cases}
\]

For the fixture **M=5, α=3, K=3**, reducing to L=4 gives:

| Array | Tower positions |
|---|---|
| Active digit | q₀, q₁, q₂, q₃, p₀, p₁, p₂ |
| Full key indices used | 0, 1, 2, 3, 5, 6, 7 |
| Full key index skipped | 4, the dropped q₄ |

Index 4 in the active digit is p₀, while index 4 in the key is q₄.
Using the same flat index for both silently reads the wrong key tower when
unchecked multiplication is used. The diagnostic key gives every digit,
tower, and evaluation position a distinct deterministic value to expose this.

<div data-hybrid-explorer>
<label for="hybrid-level"><strong>Explore the active level</strong></label>
<p>Choose the number of remaining Q towers in the executable fixture.</p>
<select id="hybrid-level" aria-label="Active Q tower count"><option value="5">5 Q towers</option><option value="4">4 Q towers</option><option value="3">3 Q towers</option><option value="2">2 Q towers</option><option value="1">1 Q tower</option></select>
<div data-hybrid-layout aria-live="polite"></div>
</div>
<script defer src="assets/hybrid-graph.js"></script>

## 5. Two MACs, then two ModDowns

For digit evaluation array d and key arrays A and B, each active tower m and
bit-reversed evaluation coordinate k compute:

\[
Z_{0,m,k}=\left[\sum_d D_{d,m,k}B_{d,\operatorname{keyIndex}(m),k}\right]_m,
\]
\[
Z_{1,m,k}=\left[\sum_d D_{d,m,k}A_{d,\operatorname{keyIndex}(m),k}\right]_m.
\]

`EvalFastKeySwitchCoreExt` returns these two QlP arrays. The multiplication is
pointwise; there is no NTT inside the MAC loop. All products and sums remain
canonical residues. The evaluation key already supplies the gadget scaling:
in private-to-private generation, its old-secret term is P times the old
secret on that digit's Q towers, and zero on its other towers. Do not add an
extra global twist or a guessed CRT/gadget multiplier to the digits.

For each accumulated array Z, extract and inverse-transform its P towers:

\[
u_j=\operatorname{INTT}_{p_j}(Z_{p_j}),\qquad
\widehat P_j=P/p_j,
\]
\[
R_i=\left[\sum_j [u_j\widehat P_j^{-1}]_{p_j}\widehat P_j\right]_{q_i},
\]
\[
Y_{q_i}=\left[(Z_{q_i}-\operatorname{NTT}_{q_i}(R_i))
[P^{-1}]_{q_i}\right]_{q_i}.
\]

This is the **exact implementation order** for the pinned CKKS path. Its
noise scale is 1, so the source passes t=0 and skips the BGV-specific t factors.
There is no floating-point rounding operation in these steps. Calling this
“division by P” does not authorize replacing its approximate representative
with a canonical or centered CRT representative, or with a generic rounding rule.

**Accumulate before ModDown.** Moving ModDown inside the digit sum changes
carry/correction terms. A simple scalar illustration with q=97 and P=17 is
\(\operatorname{Down}(16)=0\), \(\operatorname{Down}(1)=0\), but
\(\operatorname{Down}(17)=1\). This is a modular-arithmetic example, not a
small polynomial fixture. The full-size test also explicitly checks that the
wrong per-digit reduction order produces different arrays.

## 6. Which wrapper calls which graph?

| Entry point | Result and caller obligation |
|---|---|
| `EvalKeySwitchPrecomputeCore` | β active digit arrays in QlP; reusable for hoisted operations with compatible parameters |
| `EvalFastKeySwitchCoreExt` | Two QlP MAC arrays; reduction is deferred |
| `EvalFastKeySwitchCore` | Extended MAC followed by ModDown of both arrays |
| `KeySwitchCore` | Precompute followed by the fast core; two Q contributions |
| `KeySwitchInPlace` with two components | Add B contribution to c₀; **replace** c₁ with the A contribution |
| `KeySwitchInPlace` with three components | Add B to c₀ and A to c₁; discard c₂ |
| Relinearization | Switch each component from index 2 using the corresponding evaluation key, accumulate into c₀/c₁, then resize to two |

For the in-place wrapper:

\[
c'_0=c_0+Y_0,\qquad
c'_1=\begin{cases}Y_1,&\text{two-component input},\\c_1+Y_1,&\text{three-component input}.\end{cases}
\]

D8 executes the two- and three-component wrapper cases. The source trace also
records the general relinearization loop; higher ciphertext degrees and their
full key families remain a later integration test.

### Delayed QP operations are distinct from digit extension

`KeySwitchExt(ciphertext,addFirst)` multiplies the retained Q evaluations by P
and appends **zero P towers**. It does not approximate-lift the original
ciphertext into P. If `addFirst=false`, its entire first component is zero.
`KeySwitchDown` reduces both accumulated QP components; `KeySwitchDownFirstElement`
reduces only the first. The test checks both addFirst settings and delayed
addition of a QP MAC result before reduction.

The pinned CKKS `EvalFastRotationExt` has another ordering requirement: it
computes the extended MAC, optionally adds P times c₀ on the Q towers, then
applies the automorphism to **both QP components**. Reduction is deferred.
The automorphism map and full rotation API are source-traced here; their exact
end-to-end arrays remain P6/P7 work. Do not move the permutation across the
selected rotation key without a separate equivalence proof.

## 7. Full-size fixture and evidence

The deterministic fixture uses FIXEDMANUAL CKKS with multiplicative depth 4,
first modulus size 60, scaling modulus size 50, two large digits, and the fixed
ring dimension. It produces **five Q towers and three P towers**.

| Tower | Modulus | Primitive root |
|---|---:|---:|
| q₀ | 1152921504606584833 | 18043022392882 |
| q₁ | 1125899902124033 | 15995550378 |
| q₂ | 1125899911168001 | 21670117260 |
| q₃ | 1125899903827969 | 938640682 |
| q₄ | 1125899908022273 | 6511788946 |
| p₀ | 1152921504598720513 | 800790938143 |
| p₁ | 1152921504597016577 | 17749908910371 |
| p₂ | 1152921504595968001 | 11469071954203 |

The configuration requests 50 scaling bits, but q₂ and q₄ are 51-bit primes
just above the power-of-two boundary; q₁ and q₃ are 50-bit. All P towers and
q₀ are 60-bit. Use the exact table rather than inferring widths from the knob.

These exact parameters are asserted in the executable. Pattern generation and
diagnostic key generation use fixed integer formulas in that source file.
The diagnostic key is **not a cryptographic evaluation key**. A separate pair
of actual generated secret keys supplies a real key-switching key; its fixed
arrays are shared by all comparisons in that run. Random key generation is
not compared across separate runs.

| Cases | Coverage |
|---|---|
| L=5,4,3,1 × four input patterns | Full level, partial last digit, exact digit boundary, partial first digit |
| Zero, all q−1, monomial x¹⁷, deterministic dense residues | Zero path, bounds, closed-form coefficient structure, tower-distinct data |
| Actual generated key at L=5 and L=4 | Same exact graph with real cryptographic key arrays |
| Intermediate states | Digit INTT, complement CRT and NTT, digit assembly, both MACs, P INTT, correction CRT and NTT, final scale |
| Composition and wrappers | Fast/full core equality, input immutability, two/three-component update, delayed QP path |
| Negative alternatives | Canonical-lift substitution, missing P-key offset, per-digit ModDown |

Observed on Linux x86_64 with GCC 14.2, Release, NativeInteger 64, and OpenMP
off: **18 cases and 117,571,584 residue comparisons passed**. All **15 CTests**
in the OpenFHE-enabled build passed. The deterministic reduced-level fixture
found 524,288 residues that distinguish the wrong per-digit reduction order.

Every comparison checks all 65536 entries of every involved tower, plus
modulus/root order and outer/child format. Intermediate conversion states are
observed by calling the same public primitive separately; the tests do not
instrument or modify OpenFHE's implementation. Full-core equality checks the
composition against the independent formulas.

Run the opt-in gate after [installing pinned OpenFHE](../scripts/openfhe_setup.sh):

```sh
cmake -S . -B build/openfhe-compat/ntt-build \
  -DCMAKE_BUILD_TYPE=Release -DNTT_ENABLE_OPENFHE_COMPAT=ON \
  -DCMAKE_PREFIX_PATH="$PWD/build/openfhe-compat/install"
cmake --build build/openfhe-compat/ntt-build -j
ctest --test-dir build/openfhe-compat/ntt-build \
  -R 'openfhe_' --output-on-failure
```

The Hybrid gate links OpenFHE PKE in addition to core. The standalone build
still has no OpenFHE dependency. The setup script already builds all modules.

## 8. Work counts and the next boundary

For one core call, counting transforms per tower rather than per launch:

\[
\begin{aligned}
\text{Digit INTTs}&=L,\\
\text{Complement NTTs}&=\beta_\ell(L+K)-L,\\
\text{ModDown INTTs}&=2K,\\
\text{ModDown NTTs}&=2L.
\end{aligned}
\]

At M=L=5, K=3, α=3, β=2, this is **11 inverse and 21 forward NTTs**.
At L=4 it is 10 inverse and 18 forward. These are logical counts from the
source graph, not GPU launch counts or measured performance.

With 8-byte residues and N=65536, the assembled digits request
\(8N\beta_\ell(L+K)\) bytes and the two QP accumulators request
\(16N(L+K)\) bytes. Each is **8 MiB** at full fixture level. Keys, temporary
coefficient buffers, parameters, allocator overhead, and simultaneous lifetime
choices are additional. [D9 specifies the proposed lifetimes](device-backend-contract.md#7-budget-the-hybrid-lifetimes);
an implementation and allocation measurements are still needed before claiming
a peak-memory bound.

D8 freezes this source graph and the exercised array boundary.
[D10's ADR](architecture-decision.md) accepts it together with the
[D9 device/buffer/stream/cache design](device-backend-contract.md), closing P1's
architecture gate while leaving runtime implementation and native validation open.
P4/P6 must broaden base-conversion and cryptographic fixtures across the
production parameter/level/key matrix. This is not a GPU implementation,
performance result, or complete CKKS encryption/decryption integration gate.

## Pinned source references

- [Hybrid key generation and ordinary/delayed core][hybrid].
- [ApproxSwitchCRTBasis and ApproxModDown][basis].
- [Partition construction and level-specific conversion tables][params].
- [Relinearization and ciphertext component accumulation][relinearize].
- [CKKS extended fast rotation ordering][rotation].

[hybrid]: https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/pke/lib/keyswitch/keyswitch-hybrid.cpp#L205
[basis]: https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/core/include/lattice/hal/default/dcrtpoly-impl.h#L938
[params]: https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/pke/lib/schemerns/rns-cryptoparameters.cpp#L98
[relinearize]: https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/pke/lib/schemebase/base-leveledshe.cpp#L313
[rotation]: https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/pke/lib/scheme/ckksrns/ckksrns-leveledshe.cpp#L595
