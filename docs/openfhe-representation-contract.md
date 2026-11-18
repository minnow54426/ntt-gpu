<div id="openfhe-representation-contract-d7" aria-hidden="true"></div>

# Match OpenFHE's polynomial representation

A GPU result must match both the values and their positions in OpenFHE's
arrays. This page defines that boundary for OpenFHE 1.6.1, NativeInteger 64,
at N=65536. The pinned commit is `b59e1f291965f14fbee5172cd83a20f5b856204f`.

A `NativePoly` stores one polynomial modulo one prime. A `DCRTPoly` stores
several such arrays, called towers, with different primes. **Format** says
whether an array contains coefficients or NTT evaluations. Preserve each
tower's prime, root, format and position when moving it to the GPU.

[The executable checks](../tests/openfhe_representation_tests.cpp) use the real
OpenFHE core library on CPU. Read [the Hybrid graph](openfhe-hybrid-operation-graph.md)
after this contract. OpenFHE's [public aliases][aliases] resolve to `PolyImpl`
and `DCRTPolyImpl` in the source. Linked excerpts are © 2022 OpenFHE, under
[BSD 2-Clause](site-assets/licenses/OpenFHE-BSD-2-Clause.txt).

## 1. The boundary in one table

| Property | Required contract |
|---|---|
| Ring | \(\mathbb Z_q[x]/(x^N+1)\), \(N=65536\); cyclotomic order \(m=2N=131072\) |
| Coefficient index i | Coefficient of \(x^i\), in natural order |
| Evaluation memory index k | Evaluation at \(\psi^{2\operatorname{rev}_{16}(k)+1}\), with 16-bit index reversal |
| Tower modulus q | The exact modulus in that tower's OpenFHE parameters |
| Root ψ | The exact primitive 2N-th root in those same parameters |
| Residue representation | Unsigned canonical values in \([0,q)\); no external twist or Montgomery conversion |
| Forward | Natural coefficients → bit-reversed evaluations, in place |
| Inverse | Bit-reversed evaluations → natural coefficients; N-inverse normalization included |
| GPU buffer order | `values[limb*N + index]`, preserving the supplied tower list |

These evaluation indices are **not CKKS's decoded complex-slot indices**.
Encoding/decoding uses another embedding and is outside this NTT contract.

## 2. What each format means

OpenFHE's [Format enum][format] names `COEFFICIENT` and `EVALUATION`.
The label alone does not express index ordering. For the power-of-two ring,
[`NativePoly::SwitchFormat`][poly-switch] calls
[`ForwardTransformToBitReverseInPlace`][forward-ntt] or [`InverseTransformFromBitReverseInPlace`][inverse-ntt].
The NTT's evaluation array is therefore bit-reversed even though its public
format is simply `EVALUATION`.

<details markdown="1">
<summary>Read the format-to-transform dispatch</summary>

[OpenFHE · `poly-impl.h:467–473`](https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/core/include/lattice/hal/default/poly-impl.h#L467-L473)

```cpp
if (m_format != Format::COEFFICIENT) {
    m_format = Format::COEFFICIENT;
    ChineseRemainderTransformFTT<VecType>().InverseTransformFromBitReverseInPlace(ru, co, &(*m_values));
    return;
}
m_format = Format::EVALUATION;
ChineseRemainderTransformFTT<VecType>().ForwardTransformToBitReverseInPlace(ru, co, &(*m_values));
```

The two branches call different transform entry points; changing the format is an operation on the array.

</details>


Our reference and CUDA entry points do not store an OpenFHE format flag.
The caller must know the current domain before invoking `forward()` or
`inverse()`. Calling forward twice is not an idempotent format assignment.

No explicit input twist or output permutation is needed at this boundary.
The negacyclic twist is already folded into the butterfly schedule. Ordinary
bit reversal changes an **index**, not byte order or the bits of a residue.

## 3. Exact evaluation and inverse formulas

Work with **one tower**: one polynomial modulo one prime q. The input array
\([a_0,a_1,\ldots,a_{N-1}]\) stores the coefficients of
\(a(x)=a_0+a_1x+\cdots+a_{N-1}x^{N-1}\).
Forward describes this same polynomial by its values at N special points;
inverse recovers its coefficients. Repeat the operation independently for
other towers.

### Read the symbols

| Symbol | Meaning |
|---|---|
| \(N\) | Power-of-two transform length: the number of coefficients and evaluation points; 65536 in the project |
| \(q\) | This tower's prime modulus; stored values lie in \([0,q)\) |
| \(\psi\) | A primitive 2N-th root: \(\psi^{2N}=1\) and \(\psi^N=-1\pmod q\) |
| \(i,\ a_i\) | Coefficient index and coefficient of \(x^i\), with \(0\le i<N\) |
| \(k,\ y_k\) | Memory index and stored evaluation value, with \(0\le k<N\) |
| \(j\) | Natural evaluation-point index; it can differ from memory index k |
| \(\operatorname{rev}_b(k)\) | Reverse exactly b binary digits of k, where \(b=\log_2N\); keep leading zeros |
| \(r_k\) | The evaluation point assigned to memory position k |
| \(N^{-1}\) | The modular inverse of N: \(N\cdot N^{-1}\equiv1\pmod q\) |

### Choose the point stored at position k

For N=65536, an index has 16 bits. First find
\(j=\operatorname{rev}_{16}(k)\), then choose the odd root power:

\[
r_k=\psi^{2j+1}=\psi^{2\operatorname{rev}_{16}(k)+1}\pmod q.
\]

**Why odd powers?** This project multiplies modulo \(x^N+1\), so the evaluation
points must satisfy \(r_k^N=-1\). Odd powers have that property because
\((\psi^{2j+1})^N=(-1)^{2j+1}=-1\).

**Why bit reversal?** OpenFHE's NTT stores these evaluations in bit-reversed
point order. It rearranges the addresses of the results, not their numeric
bits. The first memory positions therefore mean:

| Memory k | Natural point index j = rev₁₆(k) | Root exponent 2j+1 |
|---:|---:|---:|
| 0 | 0 | 1 |
| 1 | 32768 | 65537 |
| 2 | 16384 | 32769 |
| 3 | 49152 | 98305 |
| 4 | 8192 | 16385 |

In particular, \(r_0=\psi\), while
\(r_1=\psi^{N+1}=\psi^N\psi=-\psi\pmod q\).
Thus memory positions 0 and 1 contain \(a(\psi)\) and \(a(-\psi)\).
A negative value is stored as its canonical residue; for example, −2 is 15
modulo 17. This ordering concerns NTT evaluations, not decoded CKKS slots.

### Forward: plug the point into the polynomial

\[
\begin{aligned}
y_k&=\mathrm{forward}(a)[k]\\
&=a(r_k)\\
&=\sum_{i=0}^{N-1}a_i r_k^i\pmod q.
\end{aligned}
\]

For **one fixed k**, the sum means
\(a_0+a_1r_k+a_2r_k^2+\cdots+a_{N-1}r_k^{N-1}\).
Multiply each coefficient by the corresponding power, add the terms, and
reduce modulo q. That gives one output value yₖ. Repeating for all k gives
the complete evaluation array.

The equation defines the required answer. Real implementations use
[butterflies](butterfly-explained.html) to share work rather than calculate
N separate N-term sums directly.

### Inverse: collect contributions back into each coefficient

\[
\begin{aligned}
a_i&=\mathrm{inverse}(y)[i]\\
&=N^{-1}\sum_{k=0}^{N-1}y_k r_k^{-i}\pmod q.
\end{aligned}
\]

Now hold **coefficient index i** fixed and sum over **all evaluation positions
k**. Every yₖ contributes with weight \(r_k^{-i}\). Add those contributions
and multiply by N-inverse to obtain aᵢ.

Two details matter:

- \(r_k^{-i}=(r_k^{-1})^i\). A negative exponent means a modular inverse,
  not a negative array index or an ordinary floating-point reciprocal.
- N-inverse removes the factor N accumulated by the inverse sum. It is an
  integer modulo q, not decimal division by N. N is invertible because the
  supported prime satisfies \(2N\mid(q-1)\).

Keep each yₖ paired with its original rₖ. The inverse consumes the forward's
bit-reversed evaluation order directly and returns coefficients in natural
order; do not sort the values or add another bit-reversal pass.

### A complete four-point example

Use **N=4, q=17, ψ=2**, so \(\psi^4=16\equiv-1\pmod{17}\).
This is a small arithmetic example; the project workload remains N=65536.
The index now has **two bits**, so use rev₂ instead of rev₁₆.

Start with \(a(x)=1+2x+3x^2+4x^3\), stored as `[1, 2, 3, 4]`.
Here k is the memory position, j is the natural point index, and the two-bit
labels make the reversal visible:

| \(k\) (bits) | \(j\) (bits) | \(r_k\) | \(y_k\) |
|---|---|---:|---:|
| 0 (`00`) | 0 (`00`) | 2 | 15 |
| 1 (`01`) | 2 (`10`) | 15 | 11 |
| 2 (`10`) | 1 (`01`) | 8 | 13 |
| 3 (`11`) | 3 (`11`) | 9 | 16 |

For k=0, substitute r₀=2:

\[
\begin{aligned}
y_0&=1+2\cdot2+3\cdot2^2+4\cdot2^3\\
&=49\equiv15\pmod{17}.
\end{aligned}
\]

The stored forward array is **`[15, 11, 13, 16]`**. Evaluating in natural j
order would give `[15, 13, 11, 16]`: the same point values at different
addresses. Exact OpenFHE compatibility requires the stored order.

For the inverse, \(4^{-1}=13\pmod{17}\), since \(4\cdot13=52\equiv1\).
The inverse points are `[9, 8, 15, 2]`; for example, \(2\cdot9\equiv1\pmod{17}\).

For i=0, every weight is 1. The evaluations sum to 55, which is 4 modulo 17;
scaling gives \(a_0=13\cdot4\bmod17=1\).
For i=1, call the weighted sum S₁:

\[
\begin{aligned}
S_1&=15\cdot9+11\cdot8\\
&\quad+13\cdot15+16\cdot2\\
&=450\equiv8\pmod{17},\\
a_1&=13S_1\bmod17=2.
\end{aligned}
\]

All four coefficient calculations are:

| Coefficient i | Weighted sum, reduced mod 17 | Multiply by 13, mod 17 |
|---:|---:|---:|
| 0 | 4 | 1 |
| 1 | 8 | 2 |
| 2 | 12 | 3 |
| 3 | 16 | 4 |

The inverse returns **`[1, 2, 3, 4]`**: the original coefficients.

<details markdown="1">
<summary>Why does the inverse recover exactly the original coefficients?</summary>

Let h be a second coefficient index while i stays fixed. Substitute the
forward definition into the inverse and collect the contribution from aₕ:

\[
\mathrm{inverse}(y)[i]
=\sum_{h=0}^{N-1}a_h
\left(N^{-1}\sum_{k=0}^{N-1}r_k^{h-i}\right)\pmod q.
\]

The inner sum is N when h=i, because every term is 1. Otherwise it is zero.
To see why, write \(\omega=\psi^2\), a primitive N-th root. The points rₖ
visit all \(\psi\omega^j\), only in a permuted order, so:

\[
\sum_{k=0}^{N-1}r_k^{h-i}
=\psi^{h-i}\sum_{j=0}^{N-1}(\omega^{h-i})^j.
\]

For h≠i, set \(z=\omega^{h-i}\). Then z≠1 but \(z^N=1\). Multiplying the
geometric sum by 1−z cancels its middle terms:

\[
(1-z)(1+z+\cdots+z^{N-1})=1-z^N=0\pmod q.
\]

Because q is prime and 1−z is nonzero, it has a modular inverse, so the sum
itself must be zero. For h=i, every term is 1 and the sum is N. The final
N-inverse therefore keeps aᵢ and cancels every other coefficient's
contribution. Bit reversal does not change this sum, because it visits each
point exactly once.

</details>

### Simple probes that catch ordering and scaling errors

Returning to N=65536, the test suite uses polynomials with easy expected
answers. For example, evaluating a(x)=x returns the points themselves, which
reveals their stored order. Setting just one evaluation value to 1 isolates
one term of the inverse formula.

| Probe | Expected array |
|---|---|
| \(a(x)=1\) | Every evaluation is 1 |
| \(a(x)=x\) | Slot k is \(r_k\) |
| \(a(x)=x^{17}\) | Slot k is \(r_k^{17}\) |
| \(a(x)=x^{N-1}\) | Slot k is \(r_k^{N-1}\) |
| Evaluation array with 1 only at slot s | Coefficient i is \(N^{-1}r_s^{-i}\) |

These closed forms distinguish order and scaling errors that a shared
forward/inverse round trip could conceal. The D7 gate checks all N outputs,
not just the first few slots. See the upstream [inverse normalization loop][normalization]
for the scaling implementation, and [our CPU reference](../include/ntt/ntt.hpp)
for the matching transform convention.

## 4. Changing format versus changing its label

The shared [SetFormat implementation][set-format] calls [`SwitchFormat`][poly-switch] only
when the requested format differs from the current one.

<details markdown="1">
<summary>Read SetFormat’s conditional conversion</summary>

[OpenFHE · `ilelement.h:474–478`](https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/core/include/lattice/ilelement.h#L474-L478)

```cpp
inline void SetFormat(const Format format, uint32_t thread_limit = 0) {
    if (this->GetFormat() != format) {
        this->SwitchFormat(thread_limit);
    }
}
```

The condition is why a same-format request preserves values. Compare the label-only [`DCRTPoly::OverrideFormat`][dcrt-override] before importing GPU arrays.

</details>


| API | Effect |
|---|---|
| [`SetFormat(current_format)`][set-format] | No conversion; values remain unchanged |
| [`SetFormat(other_format)`][set-format] | Runs the forward or inverse transform |
| [`SwitchFormat()`][poly-switch] | Always toggles format and transforms values |
| [`NativePoly::SetValues(values, format)`][set-values] | Assigns the values and label; does not transform them |
| [`NativePoly::OverrideFormat(format)`][poly-override] | Changes only the label |
| [`DCRTPoly::OverrideFormat(format)`][dcrt-override] | Changes only the outer label; does not retag or transform its NativePoly children |

The last two methods are not substitutes for conversion. A DCRT object with
outer `EVALUATION` and coefficient-format children is inconsistent. Keep the
outer and child formats aligned when importing GPU results.

To import an already-transformed array, create an evaluation-format NativePoly
for each tower and assign its values with the evaluation label. Inverting it
again during import would undo the GPU computation. D7 tests this import path
using a separate DCRT object.

## 5. Roots, prime widths, and the transform cache

A valid primitive root is not necessarily the **same** root. Replacing ψ by
\(\psi^3\) still gives a primitive 2N-th root but changes the evaluation addresses.
Do not regenerate roots using this repository's independent parameter finder
when comparing or importing an OpenFHE context. Read each tower's root.

The pinned [RootOfUnity implementation][root] chooses the smallest primitive
root by enumerating primitive powers. The D7 fixtures use this function with
[`LastPrime(bits, 2N)`][last-prime] for exact **50-, 55-, and 60-bit** primes. The test
independently enumerates odd root powers and checks the minimum.

[`FirstPrime(bits, m)`][first-prime] searches above \(2^{\mathrm{bits}}\); its argument is not a guarantee
of a prime with exactly that many bits. `LastPrime` searches below that bound
and checks the resulting bit length. This is why the D7 matrix uses LastPrime.

**Cache constraint:** the pinned [FTT table cache][tables] looks up by modulus
and checks table length; it does not compare the requested root on a cache hit.
Use a consistent root for each modulus/ring-size pair. The gate tests the
normal OpenFHE-generated root choice. Arbitrary custom-root contexts sharing
a modulus are not covered by this result. Do not call a global cache reset
inside an integration adapter to work around this constraint.

<details markdown="1">
<summary>Read the FTT cache-hit condition</summary>

[OpenFHE · `transformnat-impl.h:720–726`](https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/core/include/math/hal/intnat/transformnat-impl.h#L720-L726)

```cpp
const uint32_t ringDim = (cycloOrder >> 1);
{
    std::shared_lock<std::shared_mutex> rlk(TablesMutex());
    auto it = m_tablesByModulus.find(modulus);
    if (it != m_tablesByModulus.end() && it->second->rootReverse.GetLength() == ringDim)
        return it->second;
}
```

The hit checks the modulus and table length. No root comparison appears in this condition, which motivates retaining exact imported roots.

</details>


Default CUDA uses the same bit-reversed group indexing for root and inverse-root
powers, but owns its tables and uses the repository's modular reduction. The
legacy build uses natural powers \([\psi^0,\ldots,\psi^N]\). Do not pass OpenFHE's
internal tables or preconditioned multiply constants to CUDA. Share the exact
parameters and polynomial arrays; construct backend-specific tables separately.

## 6. DCRT tower order and flat GPU packing

[DCRTPoly][dcrt-switch] owns a vector of NativePoly objects. Each tower has its
own modulus, root, format, and value allocation. The
[ILDCRTParams constructor][params] retains the provided modulus/root order and
forms the composite modulus Q as their product.

The polynomial's composite Q can exceed 64 bits. It is not a native tower
modulus and must not be truncated into the CUDA modulus array.

For N=65536, each row below is copied from its own NativePoly allocation:

| OpenFHE tower | Parameters | Flat GPU element range | Byte offset |
|---|---|---|---:|
| 0 | \((q_0,\psi_0)\) | 0…65535 | 0 |
| 1 | \((q_1,\psi_1)\) | 65536…131071 | 524288 |
| 2 | \((q_2,\psi_2)\) | 131072…196607 | 1048576 |

\[
\mathrm{address}(\ell,i)=65536\ell+i,\qquad
Q=q_0q_1q_2.
\]

Read the active tower list from the object, not a hard-coded sorted list or
an assumed CKKS level. The D7 test deliberately uses bit widths in the order
`55, 50, 60` so sorting or swapping towers cannot silently pass.

Packing coefficient data requires \(a_i\bmod q_\ell\) at each address. For an
integer coefficient -1, this is \(q_\ell-1\) in every tower. The test starts from
a BigInteger polynomial containing \(Q-1\) and \(2^{64}+17\), imports it as a DCRTPoly,
and checks all residues and coefficient-domain CRT reconstruction. No
interpolation is needed to run the per-tower NTT itself.

[`CRTInterpolate()`][interpolate] requires coefficient format and returns a BigInteger Poly
with the placeholder root 1. That root is not a valid primitive 2N-th root;
do not use the interpolated object's placeholder to configure an NTT backend.

OpenFHE's separate tower allocations are not one contiguous host buffer.
Copy values tower by tower into the flat buffer; do not reinterpret-cast the
DCRT object or assume its NativeInteger objects have the GPU's scalar ABI.
The [D5 dataflow map](cuda-buffer-dataflow.md) describes the current host-vector
API and the ownership/stream design subsequently specified in
[D9](device-backend-contract.md) and accepted by [D10](architecture-decision.md).

## 7. Pointwise products and active levels

With matching q, ψ, format, and slot order, pointwise evaluation multiplication
followed by inverse gives the negacyclic ring product. D7 multiplies a full
three-tower DCRT polynomial by x and checks the independent coefficient rule:

\[
\mathrm{out}_i=
\begin{cases}
-\mathrm{in}_{N-1}\pmod q,&i=0,\\
\mathrm{in}_{i-1},&1\le i<N.
\end{cases}
\]

Both the wrap term and large-input coefficient residues are checked.

[`DropLastElement()`][drop] removes the last tower and its parameter. It retains the
remaining ordered prefix without scaling or transforming values. It is not
CKKS rescaling, ModUp, or ModDown. D7 checks this prefix behavior, including
that dropping a copied object's last tower does not change the original's
parameter list. The [D8 graph](openfhe-hybrid-operation-graph.md) records the actual Hybrid basis and reduction operations.

## 8. Adapter checklist and remaining scope

Before handing an object to the backend:

1. Check N and cyclotomic order, then obtain the actual ordered active towers.
2. Retain each tower's q and ψ; validate the supported modulus/root contract.
3. Establish the domain and ordering, with consistent outer/child formats.
4. Pack canonical values at `limb*N+index`, without a twist or permutation.
5. Run the explicitly requested operation. Inverse includes normalization.
6. Import results with matching parameters and truthful format labels.

This freezes the power-of-two NTT and DCRT array boundary. It does not freeze
Hybrid digit order, Q/P concatenation, rounding, key layout, device ownership,
stream/event lifetime, or CPU fallback. The [D8 graph](openfhe-hybrid-operation-graph.md)
now specifies Hybrid digit order, Q/P layout, and the pinned CKKS reduction
sequence. [D9's device ownership/API design](device-backend-contract.md) is now
accepted by [D10's ADR](architecture-decision.md), closing P1's architecture gate.
Native runtime implementation and its exactness/failure tests remain later work.

## 9. Run the checks

After installing the pinned OpenFHE release with
[`scripts/openfhe_setup.sh`](../scripts/openfhe_setup.sh):

```sh
cmake -S . -B build/openfhe-compat/ntt-build \
  -DCMAKE_BUILD_TYPE=Release -DNTT_ENABLE_OPENFHE_COMPAT=ON \
  -DCMAKE_PREFIX_PATH="$PWD/build/openfhe-compat/install"
cmake --build build/openfhe-compat/ntt-build -j
ctest --test-dir build/openfhe-compat/ntt-build \
  -R 'openfhe_exact_compat|openfhe_representation_contract' --output-on-failure
```

The original gate retains random and boundary coefficient cases plus an
independent inverse input. The new representation gate adds full-size monomial
and inverse-basis formulas, format API behavior, root selection, three-tower
packing/import, BigInteger coefficients, pointwise multiplication, and tower
dropping. It links only OpenFHE core. See the
[daily entry](journal/2026-11-11.md) for observed build and publication results.

## Pinned source references

All links below identify the exact OpenFHE commit inspected for this contract.
The linked entry points lead to the statements used above.

- [Format enum][format] and [SetFormat][set-format].
- [NativePoly SwitchFormat][poly-switch] and [SetValues][set-values].
- [FTT dispatch and cached tables][tables].
- [RootOfUnity][root], [FirstPrime][first-prime], and [LastPrime][last-prime].
- [DCRT format conversion][dcrt-switch] and [tower parameters][params].
- [DCRT interpolation][interpolate] and [dropping a tower][drop].

[format]: https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/core/include/utils/inttypes.h#L65
[set-format]: https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/core/include/lattice/ilelement.h#L474-L478
[poly-switch]: https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/core/include/lattice/hal/default/poly-impl.h#L454-L474
[set-values]: https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/core/include/lattice/hal/default/poly-impl.h#L226-L243
[tables]: https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/core/include/math/hal/intnat/transformnat-impl.h#L718-L730
[root]: https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/core/include/math/nbtheory-impl.h#L190-L239
[dcrt-switch]: https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/core/include/lattice/hal/default/dcrtpoly-impl.h#L1956-L1969
[params]: https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/core/include/lattice/hal/default/ildcrtparams.h#L135-L149
[interpolate]: https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/core/include/lattice/hal/default/dcrtpoly-impl.h#L817-L847
[drop]: https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/core/include/lattice/hal/default/dcrtpoly-impl.h#L709-L719

[aliases]: https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/core/include/lattice/hal/lat-backend.h#L59-L60

[poly-override]: https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/core/include/lattice/hal/default/poly.h#L296-L298

[dcrt-override]: https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/core/include/lattice/hal/default/dcrtpoly.h#L875-L877

[towers]: https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/core/include/lattice/hal/default/dcrtpoly.h#L879-L889

[forward-ntt]: https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/core/include/math/hal/intnat/transformnat-impl.h#L337-L507

[inverse-ntt]: https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/core/include/math/hal/intnat/transformnat-impl.h#L542-L637

[normalization]: https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/core/include/math/hal/intnat/transformnat-impl.h#L609-L637

[first-prime]: https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/core/include/math/nbtheory-impl.h#L336-L354

[last-prime]: https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/core/include/math/nbtheory-impl.h#L357-L380
