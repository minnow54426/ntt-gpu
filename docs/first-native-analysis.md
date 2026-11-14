# First native run: analysis and next experiments

**2026-11-14 · offline analysis · source 19cc285 · N=65536**

The supplied Ada machine passed the prepared correctness tests. This makes the
current CUDA code a verified baseline **for the executed cases**, not a
production backend or a measured RTX 5090 implementation. Prepare better
measurement and failure-path tooling offline before paying for another session.

The user reports the rental is stopped. This analysis uses only exported logs
and pinned source; no reconnection, new GPU execution or implementation changes.
See the [session record](journal/2026-11-14.md) for environment, commands,
archive checksum and local report paths.

## 1. Exactness: what the comparison count means

The archive checksum matches. Its report contains 17 successful stages, a clean
checkout, native compilation and `GPU_CHECKS_PASSED`. The complete log confirms
23/23 CTests passed; **four execute GPU work**, not 23 separate GPU tests.

The strongest native check is
[cuda_openfhe_tests.cpp](../tests/cuda_openfhe_tests.cpp):

| Scope | Verified from saved output |
|---|---|
| Basis groups | Nine: unsorted 55/50/60-bit basis, then Q and QlP at L=5,4,3,1 |
| Patterns per group | Seven: zero, one, q−1, alternating bounds, x^17, x^(N−1), dense seeded residues |
| Pattern/basis combinations | 63 |
| Sum of tower counts across groups | 41 |
| Residue positions across patterns | `65536 × 41 × 7 = 18,808,832` |
| Equality comparisons | Nine checks per position: **169,279,488** |

The nine checks are CPU/OpenFHE forward, GPU forward, roundtrip, CPU/OpenFHE
independent inverse, GPU independent inverse, CPU/OpenFHE square, GPU
pointwise/inverse, GPU complete square and host-source immutability. The total
is **not 169 million independent random cases**.

Arbitrary evaluation input tests inverse independently of forward. OpenFHE
comparison checks actual ordering/root/scaling, not merely mutual invertibility.
The separate `cuda_verify` also uses distinct left/right operands against the
CPU reference; the direct OpenFHE product matrix primarily tests squares.

No mismatch was observed. This supports real native correctness for this
schedule/arithmetic on this machine, not every prime, key or concurrent use.
GPU operations over D8's Q/P moduli test NTT/products on those bases; they do
**not** implement the Hybrid graph. That graph's gate here executes on CPU.

### Sanitizer coverage

Each of memcheck, racecheck, initcheck and synccheck ran zero/dense patterns at
N=65536 with three towers: two pattern/basis rows and 3,538,944 comparisons per
tool. Saved summaries report zero errors; racecheck also reports zero warnings
and hazards. This is kernel smoke coverage, not all 63 combinations, allocation
failure, failed-context cleanup or asynchronous ownership verification.

## 2. Latency: useful baseline, limited comparison

One session, three synthetic **46-bit** towers, ten timed repetitions per
operation. The code preserves means, not individual samples.

| Operation | Printed mean | Timing scope |
|---|---:|---|
| CPU reference forward | 47.4496 ms | Correctness-first scalar code, serial tower loop |
| GPU forward | 0.786293 ms | Public host-vector call |
| GPU inverse | 0.806584 ms | Public host-vector call |
| GPU complete multiply | 1.49319 ms | Two forwards, pointwise, inverse and public-call costs |

Source: [run_bench and synchronous wrapper](../src/cuda/ntt_kernels.cu).
Device allocations and table construction/upload precede timing. Validation,
device guards, launches, operand transfers and blocking download are included.

`47.4496 / 0.786293 ≈ 60.35`: the recorded public GPU forward is about 60×
faster than **this scalar reference in this experiment**. This is not an
OpenFHE, Hybrid or inference speedup. No optimized OpenFHE CPU timing or
application timing exists here. CPU twiddle generation also differs from the
device's precomputed tables.

CTest's 14.96-second native-gate duration includes many operations, CPU/OpenFHE
expected values and comparisons. It is not kernel latency. The 36.64-second
suite duration is not inference time either.

### Benchmark caveats found in source

| Finding | Consequence | Proposed correction |
|---|---|---|
| Only means saved | Cannot recover median/p5/p95, variance or outliers offline | Save each sample and workload/timing metadata |
| Inverse lacks dedicated warm-up and input reset | It follows warm forward calls, then repeatedly inverts its previous result | Prepare fixed evaluation data; warm inverse and reset outside timing |
| CPU forward mutates its previous output; timed GPU forward resets | Repetition input/cache policies differ | Use matched fixed-input/reset protocols |
| Multiply receives left operand by value | Host copy/allocation and discarded-result destruction are inside call timing | Keep honest public-call timing; add prepared-buffer/device timing separately |
| No event/per-kernel measurements | Copies, launch gaps, arithmetic and memory stalls cannot be separated | Add resident CUDA-event and explicit transfer/call boundaries |
| Performance only at 46 bits/K=3/B=1 | 50/55/60-bit correctness is not performance evidence there | Time actual OpenFHE parameters, levels and batches |
| Hardware snapshot captured idle state | P8/210 MHz does not establish clocks during kernels | Record active clocks, power/thermal state and repeated runs |

There are two warm-up forwards and multiplies, but no explicit inverse warm-up.
Inverse does follow forward work, so this is not a claim that the device was
cold. Forward/inverse means differ by about 2.58%; without samples, do not call
this statistically meaningful or attribute it to inverse's extra launch.

## 3. Fused multiplication and transfer accounting

One operand has `S=8×65536×3=1,572,864 B` (1.5 MiB). From the
[D5 transfer map](cuda-buffer-dataflow.md):

| Pipeline | Operand transfer payload | Kernel launches |
|---|---:|---:|
| Public forward(A), forward(B), pointwise, inverse | 9S = 13.5 MiB | 8 |
| Public multiply(A,B) | 3S = 4.5 MiB | 8 |

Fusion removes six operand copies, not six kernels. It also reduces public
validation/guard paths, although multiply has the host-copy cost above. Thus
1.49319 ms for multiply is plausible even though two separately timed forward
means sum to 1.572586 ms. It is not evidence that the individual timings fail.

Two forward means plus inverse total 2.37917 ms **without** the unmeasured
standalone pointwise call. Their ratio to multiply is about 1.59, but this is
**not a measured fusion speedup**: it combines unmatched measurements and omits
a stage. Benchmark the complete unfused and fused chains under one protocol.

Architecturally, preserve device-resident chains when adding basis conversion
and Hybrid operations. Avoid unnecessary host roundtrips; measure residency
benefits explicitly rather than projecting this scalar ratio onto inference.

## 4. Bottleneck hypotheses, not a measured winner

At N=65536 the code fuses eight far stages into the column pass and eight near
stages into the row pass. There are 524,288 butterflies per tower.

| Source-derived property | Candidate issue | Missing evidence |
|---|---|---|
| Two full-width modular multiplies per device butterfly | Reduction instruction cost | PTX/SASS, ptxas resources, per-kernel timing and a proven alternative |
| Adjacent column-gather/scatter threads access coefficients 2,048 B apart | Inefficient memory transactions | Sectors/requested byte, L2/DRAM counters, equivalent coalesced variant |
| A block loads 256 values but has 128 butterfly pairs per stage | Some launched threads do not execute butterflies | Register/shared-memory/active-warp data and measured staging variants; not an occupancy percentage |
| Public calls transfer and synchronously download | Host/copy/launch overhead | Separate copy events and a resident-chain measurement |

For three towers, source-level modular-multiply call counts, excluding setup:
forward 3,145,728; inverse 3,342,336 including normalization; complete product
9,830,400 including pointwise and normalization. These are not hardware
instruction counts.

Coefficient-array global reads/writes requested by the source total 4S for
forward, 6S for inverse and 17S for multiplication. They omit twiddle/constants,
shared memory and host/device copies. These are not measured DRAM traffic or
bandwidth. Do not divide them by full public-call latency to infer utilization.

The source calls integer remainder slow, but this session contains no
instruction profile or alternative-reduction comparison. Arithmetic, strided
memory, transfers and launch overhead cannot yet be ranked. Change one major
factor at a time after profiling; otherwise a speedup cannot be attributed.

## 5. Hardware and reproducibility limits

Driver/runtime identify compute capability 8.9, name RTX 4080 SUPER, and
33,796,456,448 device bytes / 32,760 MiB. This is an observed rental identity,
not independent confirmation of retail board specifications. Ask the provider
before normalizing results against a supposedly identical card. Neither the
RTX 5090 primary nor RTX 4090 control has been measured here.

The snapshot reports a 320 W limit and idle clocks, not a continuous active
trace. Driver API 13.2 does not mean the compiler is 13.2: saved nvcc is 13.0.88
and runtime API is 13.0. Preserve this distinction in comparison tables.

Dependency downloads failed twice before a clean pinned source bundle resolved
setup. Prepare compatible source/dependency caches before the next rental.
Do not copy macOS binaries onto Linux. The successful collector ran for about
72 seconds **after setup**; this excludes installation, dependency compilation
and troubleshooting and is not total rental duration or cost.

## 6. Next experiments: prepare offline first

No implementation changes are included in this analysis. Proposed order:

1. **Measurement harness:** raw samples and actual q/root metadata, fixed-input
   reset/warm-up policy, matched OpenFHE CPU baseline, setup/public-call/resident
   timing scopes. Test statistics/report validation without a GPU.
2. **Failure-path harness:** deterministic injection at six allocations and
   four constant uploads. Host doubles can test bookkeeping; real device
   cleanup and context failures still need native execution. B02 stays open.
3. **Broader cases:** direct OpenFHE products with distinct operands, missing
   L=2 coverage, exact 50/55/60-bit performance workloads, opt-in full sanitizer
   coverage. Keep the current passing baseline for regression comparison.
4. **Profiling preflight:** verify provider hardware identity, Nsight tools and
   counter permissions before booking. Prepare bounded commands/output paths.
5. **Next GPU session:** obtain primary-device evidence, repeat baseline,
   execute fault tests, matched timing and per-pass profiling; export before
   stopping. Do not pay for a repeat of this limited row alone.
6. **Then optimize one factor:** choose reduction, coalesced staging or
   residency/launch work from evidence, and retain an exact same-device A/B test.

This is partial B04/D15 analysis, not a completed bottleneck/performance gate.
Primary-device evidence, failure verification, production coverage and later
GPU base conversion/Hybrid/inference remain open.
