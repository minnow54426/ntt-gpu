# Prepare basis conversion before the next GPU rental

We can prepare conversion arithmetic and reference answers on the M1 Pro.
The November 19 run checked the CUDA implementation on compute 8.9.
The RTX 5090 hardware check remains in the next combined session.

## What the converter computes

A polynomial has N coefficients. For each source prime q_i, a tower holds
N residues. Towers occupy consecutive sections of the array. A conversion
changes the primes under which these coefficients are represented.

Let Q be the product of all active source primes, and let p_j be one target
prime. The coefficient at position k is converted with

\[
\begin{aligned}
\widehat Q_i &= Q/q_i, \\
u_i(k) &= a_i(k)\widehat Q_i^{-1}\bmod q_i.
\end{aligned}
\]
\[
\begin{aligned}
b_j(k) &= \Bigl(\sum_i u_i(k) \\
&\qquad\cdot(\widehat Q_i\bmod p_j)\Bigr)\bmod p_j.
\end{aligned}
\]

Here a_i(k) is the input residue under q_i. The inverse of the product of the
other source primes scales that residue. Multiplying by the reduced product
and summing gives the target residue b_j(k).

The sum follows OpenFHE's approximate CRT convention. Reducing the complete
sum modulo Q first can change b_j(k). Independent tests include inputs that
detect this change.

The plan snapshots the source and target primes in their given order.
Shortening the active source prefix or a Hybrid digit requires a new plan.
Its constants depend on that complete source product.

## Shoup multiplication

The source scaling uses a precomputed constant w. Store

\[
\begin{aligned}
w' &= \left\lfloor\frac{w2^{64}}q\right\rfloor, \\
h &= \left\lfloor\frac{xw'}{2^{64}}\right\rfloor, \\
r &= xw-hq.
\end{aligned}
\]

x is the input residue, w is the inverse above, and q is its source prime.
For canonical x and w with q below 2^63, h underestimates the exact quotient
by at most one. Therefore r lies between zero and 2q, and one conditional
subtraction produces a residue below q.

The host computes the high product with a 128-bit integer. CUDA uses
`__umul64hi`. The per-coefficient Shoup operation uses multiplication and
subtraction. The general target-product operation currently retains the
reference 128-bit remainder calculation. Performance selection follows
native measurements.

## Local checks and the rental batch

The independent Python oracle computes the unreduced sums with arbitrary-size
integers. It covers 30/50/55/60/63-bit primes, source counts through 40, both
directions, boundary values, and rejected input shapes or parameters.

The OpenFHE comparison covers N=4,096 through 65,536, 30/50/55/60-bit primes,
and source counts 1, 3 and 40. It checks every coefficient in both directions
using zero, q−1, monomial and deterministic dense inputs. Source towers are
deliberately unsorted. The Hybrid fixtures also compare the reusable converter
against their independent CRT answers, including shortened digits and the
P-to-Q correction used by ModDown.

```sh
ctest --test-dir build/openfhe-compat/ntt-build \
  -R 'basis|arithmetic_bigint|openfhe_hybrid' --output-on-failure
```

The prepared native tests are `cuda_shoup_exact` and `cuda_basis_exact`.
The collector requires both and runs their sanitizer checks before timing.
The CUDA probe assigns one target coefficient to each thread and loops over
the source primes. Its OpenFHE fixture supplies exact expected arrays.

The later [Barrett candidate](modular-arithmetic-contract.md#barrett-candidate-for-changing-operands)
has a separate `cuda_barrett_exact` probe and four sanitizer checks per build.
Its CPU and compiler checks do not extend the earlier native evidence.

On November 19, both schedules compiled for sm_120, and the native probes
passed on a server reporting RTX 4080 SUPER / compute 8.9. The conversion
test compared 107,675,648 residues against pinned OpenFHE in each build.
All four sanitizer tools passed the selected scopes. The
[RTX 5090 session](next-gpu-rental.md) remains the primary-device check. Native validation of the new resident ModUp/ModDown composition,
GPU Hybrid key switching, and tuned arithmetic remain later deliverables.

## Keep ModUp and ModDown in the resident chain

The resident prototype now connects the existing NTT and basis-conversion
operations without downloading their intermediate polynomial arrays. A session
uses one active Q prefix followed by all its P towers. Create a separate
session when that active prefix changes.

- `switch_basis(input, target_layout)` takes coefficients and computes the
  unreduced approximate CRT sum under a disjoint destination basis. Both
  bases must be contiguous selections from the session's parameter tables.
- `mod_up(input)` accepts Q coefficients or Q evaluations. It preserves the
  original Q values in evaluation format, computes the P coefficients from
  Q coefficients, transforms P, and packs Q followed by P. Its output is
  always in evaluation format, matching OpenFHE.
- `mod_down(input)` accepts QP evaluations. It inverse-transforms only P,
  converts P coefficients to the Q correction, transforms that correction,
  subtracts it from the original Q evaluations, and multiplies by P-inverse.
  The output contains Q evaluations.

The [public API](../include/ntt/resident.hpp),
[queue implementation](../src/resident.cpp), and
[shared arithmetic helpers](../include/ntt/resident_conversion.hpp) show the
actual implementation. Compare OpenFHE's pinned
[`ApproxModUp`](https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/core/include/lattice/hal/default/dcrtpoly-impl.h#L987-L1017)
and [`ApproxModDown`](https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/core/include/lattice/hal/default/dcrtpoly-impl.h#L1020-L1055).
This path follows CKKS with reduced-noise mode OFF; it does not implement
the BGV correction factors or a complete Hybrid key switch.

After importing and validating the exact Q/P parameters, the call sequence is:

```cpp
// qp_layout describes active Q followed by P, in coefficient format.
auto session = ntt::resident::make_cuda_session(qp_layout);
auto q_layout = qp_layout;
q_layout.towers = ntt::proposal::q_prefix(*q_layout.parameters,
                                        q_layout.active_q);
auto q = session.upload(q_layout, coefficient_values);
auto qp = session.mod_up(q);
auto reduced = session.mod_down(qp);
auto result = session.download(reduced);
auto evaluation_values = result.wait_values();
session.close();
```

Keep `result` alive while reading `evaluation_values`. ModDown changes scale
using the modular inverse of P. Compare the composed result with OpenFHE's
corresponding operations; do not assume it recovers the original input.

### Temporary tables, memory and errors

Inputs remain unchanged, and each result has a new allocation. Each basis
switch builds and uploads a small parameter table; ModDown's final correction
also uploads its moduli and P-inverse values. These copies contain parameters,
not intermediate polynomial arrays. `Stats::parameter_uploads` counts them
separately from polynomial uploads/downloads and device copies. An upload →
ModUp → ModDown chain has one polynomial upload, three parameter uploads and
one final polynomial download, excluding the session's fixed setup tables.

The runtime reserves device memory for those parameter tables and CPU memory
for their staging. These budgets count the requested table/staging payloads;
temporary CPU plan objects and allocator overhead are separate. Queued
operations retain the tables and staging through completion. If
submission fails and the stream cannot be proved stopped, the existing
quarantine keeps these resources alive too. Per-operation budgets apply;
a composed call can fail after earlier operations were queued. The original
input remains unchanged, and there is no automatic CPU retry.

This prototype builds tables per operation. Reusable conversion-plan caches,
allocation pools and performance selection remain later work. The new
composition needs its own native CUDA and sanitizer run; the November 19
standalone conversion results do not validate later code.

### What has been checked locally

All 29 CPU/OpenFHE CTests pass after adding the conversion chain. The resident
control suite passes 450 checks, including injected failures and parameter-table
memory limits, and passes AddressSanitizer, UndefinedBehaviorSanitizer and
leak detection. The shared fixture checks both ModUp input formats and an
independent ModDown input against the actual OpenFHE public operations.
Cases include all five active prefixes of a five-Q/three-P context and a
40-Q/three-P dense case, preserving the given prime order.

Both default and legacy CUDA builds compile the extended fixture for sm_89
and sm_120 with the stock CUDA 13 compiler. Driver-free constructor/factory
preflight tests pass. No execution or timing of the new conversion kernels
is claimed here. The next native `resident_cuda_exact` run and its sanitizer
smoke will execute those prepared cases on the selected GPU.
