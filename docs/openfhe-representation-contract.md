# OpenFHE representation contract — D7

This freezes the NTT/DCRT boundary for **OpenFHE 1.6.1**, commit
`b59e1f291965f14fbee5172cd83a20f5b856204f`, at **N=65536**. This contract uses the **64-bit NativeInteger backend**. It covers
coefficient/evaluation format, exact array order, per-tower parameters, and
packing into the current CUDA interface. The [D8 Hybrid operation graph](openfhe-hybrid-operation-graph.md) builds on this boundary.

The executable evidence is [openfhe_representation_tests.cpp](../tests/openfhe_representation_tests.cpp).
It uses the real OpenFHE core library on CPU. No native CUDA result is claimed.

Code excerpts © 2022 OpenFHE, reproduced under the [BSD 2-Clause license](site-assets/licenses/OpenFHE-BSD-2-Clause.txt). Click a code symbol or an excerpt’s file/line link to open the corresponding GitHub source.

The public [NativePoly/DCRTPoly aliases][aliases] resolve to `PolyImpl` and
`DCRTPolyImpl`; those are the class names you will see in the implementation.

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

Let `rev16(k)` reverse the low 16 bits of k and define
\(r_k=\psi^{2\operatorname{rev}_{16}(k)+1}\pmod q\). Then:

\[
\mathrm{forward}(a)[k]=\sum_{i=0}^{N-1}a_i r_k^i\pmod q,
\qquad
\mathrm{inverse}(y)[i]=N^{-1}\sum_{k=0}^{N-1}y_k r_k^{-i}\pmod q.
\]

The first natural evaluation indices in memory are
`0, 32768, 16384, 49152, 8192, ...`. Thus the corresponding root exponents
begin `1, 65537, 32769, 98305, 16385, ...`. In particular, slot 0 contains
\(a(\psi)\) and slot 1 contains \(a(-\psi)\).

| Probe | Expected array |
|---|---|
| \(a(x)=1\) | Every evaluation is 1 |
| \(a(x)=x\) | Slot k is \(r_k\) |
| \(a(x)=x^{17}\) | Slot k is \(r_k^{17}\) |
| \(a(x)=x^{N-1}\) | Slot k is \(r_k^{N-1}\) |
| Evaluation array with 1 only at slot s | Coefficient i is \(N^{-1}r_s^{-i}\) |

These closed forms distinguish order and scaling errors that a shared
forward/inverse round trip could conceal. The D7 gate checks all N outputs,
not just the first few slots. The upstream [inverse normalization loop][normalization] shows where the remaining N-inverse factors are applied.

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


OpenFHE's bit-reversed power tables and preconditioned multiplication constants
are internal implementation data. They are not interchangeable with this
repository's \([\psi^0,\ldots,\psi^N]\) table. Share parameter values and polynomial
arrays; construct backend-specific twiddle tables separately.

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
