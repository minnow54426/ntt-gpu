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

On November 19, both schedules compiled for sm_120, and the native probes
passed on a server reporting RTX 4080 SUPER / compute 8.9. The conversion
test compared 107,675,648 residues against pinned OpenFHE in each build.
All four sanitizer tools passed the selected scopes. The
[RTX 5090 session](next-gpu-rental.md) remains the primary-device check. Full resident ModUp/ModDown composition,
GPU Hybrid key switching, and tuned arithmetic remain later deliverables.
