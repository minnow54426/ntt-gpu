<div id="ckks-gpu-to-openfhe-integration-plan" aria-hidden="true"></div>

# Roadmap to OpenFHE integration

Build in this order: **NTT → base conversion → Hybrid key switching → OpenFHE**.
The Number Theoretic Transform (NTT) helps multiply polynomials by turning
that task into separate products at special evaluation points. Base
conversion changes which primes represent each coefficient. Hybrid key
switching combines these operations with evaluation-key arrays to change a
ciphertext’s key representation without decrypting it.

The estimate is **95 working days plus 15 reserve days**. D labels a working-day
milestone and P labels a phase; neither is a calendar date or a continuous
rental period. Preparation may happen early, but a phase closes only after its
acceptance checks pass.

Target OpenFHE 1.6.1 and N=65536. Use the RTX 4080 SUPER for development and
retain the RTX 5090 as a hardware validation target. Baseline
benchmarks also span N=4096–65536 and 30/50/55/60-bit primes in `uint64_t`.
Last reviewed: 2026-11-24 (Asia/Shanghai).

## Live tracker

| Field | Current value |
|---|---|
| Working day | **D11** |
| Active phase | **P2 — first RTX GPU baseline (Ada diagnostic passed)** |
| Active task | Use the RTX 4080 SUPER for routine native checks and tuning while RTX 5090 access is limited |
| Next gate | P2: compare CUDA and OpenFHE answers on real hardware, then record the GPU and baseline measurements (D11–D15) |
| Overall state | In progress; the temporary Ada GPU was tested, while primary-device and other required P2 checks remain unfinished |

The [November 24 Ada analysis](paired-ntt-native-analysis.md#what-changed-in-the-november-24-ada-results)
records native conversion replay, paired timings and scoped Nsight Systems
captures. Group twiddles consistently improved NTT device-event timings.
Public-call timing varied by workload, and the rental blocked performance
counters. P2/D11 remains open.

See the [next GPU rental queue](next-gpu-rental.md) for the offline work and
the order of the bundled native checks.

Hardware: RTX 4080 SUPER working baseline, with the RTX 5090 retained for
target validation. Keep benchmark results tied to the exact card and repeat
comparisons on the 5090 when available. See the
[GPU comparison](gpu-selection-survey.md).

Read the tracker as follows: `DONE` means its acceptance checks passed;
`IN PROGRESS` means some work or evidence is missing; `NOT STARTED` means
implementation has not begun; `BLOCKED` means a dependency prevents progress.
An exit gate is the set of checks required to finish a phase. An oracle is
a separate reference calculation used to check answers.

| Phase | Working days | Status | Exit artifact |
|---|---:|---|---|
| P0 — pre-plan correctness baseline | Before D1 | DONE | CPU oracle, schedule simulator, CUDA source, exact OpenFHE NTT gate |
| P1 — contracts and architecture | D1–D10 | DONE | [Accepted ADR](architecture-decision.md), frozen operation graph and [implementation backlog](implementation-backlog.md) |
| P2 — first RTX GPU baseline | D11–D15 | IN PROGRESS | Native Ada diagnostic passed; RTX 5090 evidence, failure injection and full profiling pending |
| P3 — production GPU NTT | D16–D40 | NOT STARTED | Tuned, device-resident NTT dispatch |
| P4 — exact base-conversion model | D41–D49 | NOT STARTED | OpenFHE-exact CPU Q↔QP oracle and fixtures |
| P5 — GPU base conversion | D50–D61 | NOT STARTED | Exact and tuned ModUp/ModDown kernels |
| P6 — Hybrid key-switch contract | D62–D68 | NOT STARTED | Frozen decomposition/QP/MAC/ModDown graph |
| P7 — GPU Hybrid key switching | D69–D85 | NOT STARTED | Exact relinearization and rotation backend |
| P8 — OpenFHE integration | D86–D95 | NOT STARTED | Integrated backend with fallback and full tests |
| P9 — risk reserve | D96–D110 | NOT STARTED | Used only for integration and correctness risks |

## Definition of exact compatibility

Use the same parameters, input arrays, ciphertext components and evaluation
keys in both programs. The GPU must then produce exactly the same values
in every prime tower as the selected OpenFHE version. CKKS decrypts to
approximate numbers, so similar decrypted answers are a useful second check
but cannot prove exact array compatibility. Generate a key once and use
those same key arrays for both paths; independently generated random keys
are different inputs. The concrete upstream path is
[`NativePoly::SwitchFormat`](https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/core/include/lattice/hal/default/poly-impl.h#L454-L474) for transforms and
[`KeySwitchCore`](https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/pke/lib/keyswitch/keyswitch-hybrid.cpp#L295-L298) → [`EvalFastKeySwitchCoreExt`](https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/pke/lib/keyswitch/keyswitch-hybrid.cpp#L389-L421) → [`ApproxModDown`](https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/core/include/lattice/hal/default/dcrtpoly-impl.h#L1020-L1055) for Hybrid contributions.

Every production kernel must pass, in order:

1. arithmetic primitive versus the `__uint128_t` oracle;
2. operation versus this repository's CPU reference;
3. operation versus pinned OpenFHE 1.6.1;
4. full pipeline versus OpenFHE using fixed inputs and keys;
5. benchmark/profiler gate on named hardware.

## P0 — pre-plan baseline (DONE)

- [x] Negacyclic CPU NTT and Python big-integer oracle.
- [x] Shared CPU/CUDA stage bodies and portable schedule simulator.
- [x] Correctness-first two-pass CUDA source.
- [x] Exact public OpenFHE 1.6.1 transform compatibility for all target `N`.
- [x] RTX 4090/5090 build targets and rented-GPU setup script.
- [x] Benchmark and tuning methodology.
- [ ] Primary-device native acceptance; tracked separately in P2.

## P1 — contract and architecture freeze (D1–D10)

**Status: DONE.** [ADR 0001](architecture-decision.md) accepts the contracts
and implementation backlog; native runtime gates remain in later phases.

| Day | Work | Required output |
|---:|---|---|
| D1 | Read forward column kernel and launch geometry | Index and memory-access map for `N=65536` |
| D2 | Derive forward column butterfly indices | Worked `p,g,j,u,v,r,d` example and twiddle proof |
| D3 | Read inverse column kernel and DIT arithmetic | Forward/inverse correspondence |
| D4 | Read contiguous tile kernels | Proof that every global stage appears exactly once |
| D5 | Read constructor, buffers, and public API | Host/device dataflow and allocation map |
| D6 | Read modular arithmetic and exact test gates | Arithmetic bounds and adversarial-test list |
| D7 | Trace OpenFHE NTT/DCRT representation | Natural/bit-reversed and coefficient/evaluation contract |
| D8 | Trace OpenFHE Hybrid implementation | Exact `Q -> QP -> MAC -> Q` operation graph |
| D9 | Design buffer views and backend seam | Device buffer/stream/cache API proposal |
| D10 | Architecture review | Accepted ADR and implementation backlog |

Completed evidence is documented in the linked lessons and dated journal:

- D1–D4: [block/thread mapping and stage proofs](ntt-65536-work-distribution.html),
  with exhaustive CPU probes of shared CUDA stage helpers.
- D5–D6: [buffer/copy map](cuda-buffer-dataflow.md) and
  [integer bounds](modular-arithmetic-contract.md), including bigint tests.
- D7: [OpenFHE roots, formats and tower packing](openfhe-representation-contract.md).
- D8: [Hybrid graph](openfhe-hybrid-operation-graph.md), with 18 CPU cases and
  117,571,584 residue comparisons.
- D9–D10: [checked backend metadata](device-backend-contract.md), the
  [accepted ADR](architecture-decision.md) and [implementation tasks](implementation-backlog.md).

These complete P1's design work. They do not replace native CUDA lifetime,
failure, exactness or performance checks. Historical test counts and intermediate
tracker states remain in the [journal](journal/2026-11-12.md).

P1 exit gate:

- [x] NTT/DCRT (D7) and Hybrid (D8) representation, ordering, and active-tower transitions are explicit.
- [x] The exact pinned CKKS approximate-CRT and reduction order is identified (D8); broader production fixtures remain P4/P6 work.
- [x] Buffer views record ring size, Q/P towers, level, representation,
  ordering, device, stream, and event ownership (D9 proposal and host checks;
  runtime implementation pending).
- [x] The first integration seam and CPU fallback behavior are frozen by
  [ADR 0001](architecture-decision.md#3-accept-the-first-openfhe-seam-and-fallback-policy).
- [x] Architecture review accepted and implementation work has dependencies,
  observable acceptance criteria and explicit remaining runtime gates.

## P2 — first real GPU baseline (D11–D15)

The [November 14 temporary Ada batch](batch40-native-analysis.md) passed 29
native CTests, eight sanitizer runs and 280 GPU timing jobs. A separate M1 Pro
baseline covered 140 CPU jobs. The reported GPU was RTX 4080 SUPER/compute 8.9,
not the primary RTX 5090 or planned RTX 4090 control. Hardware counters were
blocked by the provider.

Later [offline preparation](offline-gpu-preparation.md) and the
[resident-chain prototype](resident-operation-chain.md) pass host tests and
sm_89/sm_120 compilation. The [November 19 batch](journal/2026-11-19.md#native-gpu-batch-completed)
then passed the new resident, Shoup and conversion checks on reported compute
8.9. Both schedules passed their native gates and 40 sanitizer stages across
the pair. All 48 paired measurement jobs, four resident repeat runs and a CUDA
timeline were exported and verified locally. Primary-device acceptance, actual
driver/context failure handling and hardware-counter profiling remain open.

The November 24 run exercised the resident ModUp/ModDown composition and
the prepared OpenFHE reference replay on the RTX 4080 SUPER. Next, run new
native checks and controlled timing comparisons on the available Ada card.
Repeat the hardware gates when an RTX 5090 is available. Use the
[split collector](offline-test-tooling.md#collector-and-validation) for GPU-only
measurements and collect the CPU baseline locally.

| Day | Work | Required output |
|---:|---|---|
| D11 | Provision the selected GPU and CUDA toolkit | Reproducible environment record |
| D12 | Build native architecture, resolve baseline safety/build blockers and run smoke tests | Native build, constructor/failure cleanup and validated smoke tests (B02) |
| D13 | Run exact gates over all `N` and modulus widths | Correctness report |
| D14 | Profile with CUDA events and Nsight | Kernel, L2, DRAM, occupancy, and register data |
| D15 | Analyze and publish baseline | Raw samples, summary table, first bottleneck decision |

[B01–B04](implementation-backlog.md#2-p2-establish-the-native-baseline) define
the acceptance criteria. Native ownership/failure tests and exact arrays must
pass before accepting timings. GPU-event timings and a timeline show how long work takes and when it runs.
We still need counters showing how much memory is fetched, how the L2 cache
and GPU memory are used, and how many warps can remain active (occupancy).

Minimum cases: zero, one, `q-1`, alternating bounds, deterministic random,
forward, inverse, round-trip, evaluation-domain inverse, and polynomial
multiplication. Record 50-, 55-, and 59/60-bit OpenFHE-compatible primes.

P2 exit gate:

- [ ] `OpenFHE == CPU reference == CUDA` for every target size.
- [ ] Kernel-only and transfer-inclusive timings are separated.
- [ ] Actual sectors/requested byte and L2/DRAM traffic are recorded for the
  strided pass.

## P3 — production GPU NTT (D16–D40)

### D16–D25 — arithmetic and device API

| Days | Work |
|---:|---|
| D16–D17 | Introduce an arithmetic-policy interface while retaining the oracle path |
| D18–D20 | Implement Shoup multiplication and parallel twiddle quotient tables |
| D21 | Add adversarial Shoup tests at maximum supported modulus width |
| D22–D23 | Implement and test Barrett variable multiplication |
| D24 | Add device-buffer, asynchronous stream, and event API |
| D25 | Run complete exact regression and benchmark |

### D26–D40 — layout, radix, and selection

| Days | Work |
|---:|---|
| D26–D29 | Prototype coalesced transpose/Four-Step replacement for the column pass |
| D30–D32 | Implement radix-4 register staging |
| D33–D35 | Implement radix-8 register staging and padded shared memory |
| D36 | Compare warp shuffle and shared-memory exchange for small stages |
| D37 | Add compile-time specialization for N=65536 |
| D38 | Run latency/throughput autotuning over real tower and batch counts |
| D39 | Freeze the per-device/per-size dispatch table |
| D40 | Production NTT gate and documentation update |

P3 exit gate:

- [ ] Exact transform for all target sizes and moduli.
- [ ] No optimized kernel silently uses device-side integer remainder.
- [ ] Device-resident forward, inverse, and pointwise operations compose
  without host synchronization.
- [ ] A selected latency and throughput configuration exists for every target
  size on the primary GPU.

## P4 — exact base-conversion model (D41–D49)

| Day | Work |
|---:|---|
| D41 | Locate and trace OpenFHE ModUp (`Q -> QP`) |
| D42 | Locate and trace OpenFHE ApproxModDown (`QP -> Q`) |
| D43 | Record precomputed constants, reductions, rounding, and table layouts |
| D44 | Extract fixed OpenFHE intermediate fixtures |
| D45 | Implement standalone CPU `Q -> QP` oracle |
| D46 | Implement standalone CPU `QP -> Q` oracle |
| D47 | Test every target level, tower count, and adversarial coefficient |
| D48 | Design base-conversion buffer and batched APIs |
| D49 | Review and freeze the exact contract |

P4 exit gate: the standalone CPU base-conversion arrays equal pinned OpenFHE
at every intermediate boundary, not only after a complete key switch.

## P5 — GPU base conversion (D50–D61)

| Days | Work |
|---:|---|
| D50–D52 | Implement batched GPU Q-to-P basis extension |
| D53 | Exact Q-to-P tests and profiler pass |
| D54–D56 | Implement GPU QP-to-Q reduction |
| D57 | Exact QP-to-Q tests and profiler pass |
| D58 | Add dynamic CKKS level and tower-count support |
| D59 | Tune coefficient tile, output-tower mapping, accumulation, and reduction |
| D60 | Run the complete size/tower/batch benchmark matrix |
| D61 | Production base-conversion gate |

P5 exit gate:

- [ ] Exact Q-to-QP and QP-to-Q arrays.
- [ ] Dynamic active Q/P tower counts.
- [ ] NTT and base conversion share device buffers without conversion copies.
- [ ] Selected latency and throughput configurations are recorded.

## P6 — Hybrid key-switch contract (D62–D68)

| Day | Work |
|---:|---|
| D62 | Trace evaluation-key generation and storage |
| D63 | Trace digit decomposition and transform placement |
| D64 | Trace ModUp and evaluation-key multiplication |
| D65 | Trace ModDown and final ciphertext folding |
| D66 | Extract fixed intermediate fixtures from OpenFHE |
| D67 | Design GPU evaluation-key layout, cache, and lifetime rules |
| D68 | Freeze the full Hybrid operation graph |

P6 exit gate: all decomposition, basis, ordering, scaling, rounding, and final
accumulation boundaries have exact OpenFHE fixtures.

## P7 — GPU Hybrid key switching (D69–D85)

| Days | Work |
|---:|---|
| D69–D71 | Implement device-resident digit decomposition |
| D72–D73 | Connect inverse NTT and Q-to-QP ModUp |
| D74–D76 | Implement evaluation-key multiply-accumulate in QP |
| D77–D78 | Connect QP-to-Q ModDown |
| D79 | Implement final `c0/c1` accumulation |
| D80 | Validate relinearization |
| D81 | Validate automorphism/rotation |
| D82 | Exercise all target CKKS levels and digit counts |
| D83 | Run exact intermediate and final-array tests |
| D84 | Profile key cache, L2 behavior, memory footprint, and launch gaps |
| D85 | Production Hybrid key-switch gate |

P7 exit gate:

- [ ] Fixed inputs and evaluation keys produce exact OpenFHE tower arrays.
- [ ] Relinearization and rotations pass exact and decryption tests.
- [ ] Evaluation keys and constants upload once and remain cached.
- [ ] Full key-switch latency and batched throughput are reported.

## P8 — OpenFHE integration (D86–D95)

| Day | Work |
|---:|---|
| D86 | Implement the OpenFHE GPU backend adapter |
| D87 | Add CMake CUDA discovery and opt-in configuration |
| D88 | Map `DCRTPoly` towers to versioned GPU buffer views |
| D89 | Add parameter, twiddle, basis-table, and scratch caches |
| D90 | Add evaluation-key cache |
| D91 | Dispatch NTT, base conversion, and Hybrid key switching |
| D92 | Implement explicit unsupported-case CPU fallback |
| D93 | Finish stream/event lifetime, synchronization, and error handling |
| D94 | Run OpenFHE CKKS examples and relevant upstream tests |
| D95 | Integration release gate, documentation, and benchmark report |

Dispatch policy:

```text
invalid input or inconsistent format    -> error
valid same-format request               -> validated no-op
available + supported + verified        -> CUDA backend
unavailable/unsupported/unverified      -> CPU only if allowed before submission
busy/budget/allocation failure          -> visible resource error
submission attempted, then any failure  -> visible error; no automatic CPU retry
```

The opt-in adapter defaults to `RequireDevice`. See
[ADR 0001](architecture-decision.md) for the accepted transaction, ownership,
cache-budget and admission rules. Production integration remains a P8 gate.

P8 exit gate:

- [ ] OpenFHE CKKS runs with Hybrid key switching through the GPU backend.
- [ ] Original CPU tests and fallback remain green.
- [ ] NTT, base conversion, relinearization, and rotation meet exact gates.
- [ ] Constants and evaluation keys remain resident across operations.
- [ ] All results identify hardware, clocks/power mode, driver, toolkit,
  compiler, commit, parameters, timing boundary, and correctness status.

## P9 — risk reserve (D96–D110)

Use only for work that blocks a preceding exit gate:

- OpenFHE rounding/reduction-order differences;
- natural/bit-reversed ordering mistakes;
- CKKS level/tower corner cases;
- register spilling, shared-memory pressure, or poor occupancy;
- evaluation-key memory pressure and cache eviction;
- CUDA stream/event lifetime defects;
- primary-GPU compiler or architecture problems;
- cross-GPU dispatch regressions.

Do not schedule new features in the reserve. Running an encrypted inference model begins only after P8’s checks pass.
That needs a separate plan for model operations, scale changes, rotations,
execution order, numerical accuracy and the time for a complete request.

## Progress log

<details markdown="1">
<summary>Dated milestones and their original evidence</summary>


| Date | Day | Change | Evidence/commit |
|---|---:|---|---|
| 2026-11-06 | D1 | Created tracked integration plan; began line-by-line CUDA forward-column review; completed primary-source GPU selection survey | dbf06cd |
| 2026-11-07 | D1 | Added the N=8 VS Code debug example and native LLDB DAP setup; clarified CPU butterfly variable names; expanded the direct-sum and negacyclic derivations with locally bundled math rendering. CPU learning prerequisites are covered; the D1 CUDA index/memory map remains the next deliverable. | Debug example and CTest gates; 57 formulas verified in the browser |
| 2026-11-08 | D1 | Organized the documentation into an HTML engineering notebook with a daily journal, shared navigation, search, and a Pages workflow. Historical entries cite commits and are marked reconstructed. Implementation gates remain unchanged. | [Daily entry](journal/2026-11-08.md); documentation build and regression checks |
| 2026-11-09 | D1 → D2 | Completed both endpoint index/memory maps (4096 and 65536), with an interactive size comparison and portable schedule regression at N=65536. Next: column butterfly indices and twiddle proof. P1 exit gate remains open. | [Work-distribution map](ntt-65536-work-distribution.html); [daily entry](journal/2026-11-09.md); CPU schedule and explorer checks |
| 2026-11-09 | D2 → D3 | Completed the column-pair/twiddle proof and six worked examples; exhaustive CPU execution of the real stage helper checked 270,336 pairs using 540,672 probes. Next: inverse-column correspondence. P1 exit gate remains open. | [Column proof](ntt-65536-work-distribution.html#column-proof); `column_mapping_check`; `device_schedule_check` |
| 2026-11-10 | D3 → D4 | Completed inverse-column correspondence, factor-2/factor-N checks and normalization, plus inverse-mode explorer and worked examples. Next: contiguous kernels and exact stage coverage. P1 exit gate remains open. | [Inverse proof](ntt-65536-work-distribution.html#inverse-proof); `inverse_column_check`; [daily entry](journal/2026-11-10.md) |
| 2026-11-10 | D4 → D5 | Completed row-kernel pair/twiddle equivalence and exact stage coverage. Added exhaustive CPU execution of both row-stage helpers plus all-size stage-list and endpoint checks. Next: constructor/buffer/API dataflow. P1 exit gate remains open. | [Row proof](ntt-65536-work-distribution.html#tile-proof); `tile_mapping_check`; all 11 CTests passed |
| 2026-11-10 | D5 → D6 | Completed constructor/allocation/transfer/API map and six static source-contract checks. Recorded constructor exception cleanup and device/stream ownership gaps. Next: modular arithmetic bounds and adversarial tests. P1 exit gate remains open. | [D5 map](cuda-buffer-dataflow.md); `tests/docs/test_cuda_dataflow.py`; [daily entry](journal/2026-11-10.md) |
| 2026-11-10 | D6 → D7 | Completed arithmetic bounds, canonical-range proof and adversarial-test inventory; added independent bigint and full-size 50/55/60-bit CPU gates. Next: OpenFHE representation/tower contract. P1 remains open. | [D6 contract](modular-arithmetic-contract.md); `arithmetic_bigint_check`; `arithmetic_transform_check`; [daily entry](journal/2026-11-10.md) |
| 2026-11-11 | D7 → D8 | Frozen the pinned OpenFHE NTT/DCRT array contract; full-size formula, format API, root, and three-tower checks pass. P1 remains open. | [D7 contract](openfhe-representation-contract.md); [daily entry](journal/2026-11-11.md); `openfhe_representation_contract` |
| 2026-11-11 | D8 → D9 | Completed the pinned Hybrid graph and full-size intermediate-array gate; P1 remained open for ownership and architecture. | [D8 graph](openfhe-hybrid-operation-graph.md); [daily entry](journal/2026-11-11.md#d8-freeze-the-pinned-hybrid-operation-graph) |
| 2026-11-12 | D9 → D10 | Completed checked metadata and the backend API proposal; runtime implementation remained pending. | [D9 contract](device-backend-contract.md); 78 host checks and 16 CPU/OpenFHE CTests |
| 2026-11-12 | D10 → D11 | Accepted ADR 0001 and the implementation backlog; P1 is DONE. P2 awaits native GPU provisioning. | [ADR](architecture-decision.md); [backlog](implementation-backlog.md); [daily review](journal/2026-11-12.md#d10-accept-architecture-and-close-p1) |
| 2026-11-12 | D11 preparation | Completed stock CUDA sm_89/sm_120 compilation, baseline ownership/preflight fixes, native fixture compilation and tested failure-aware session collection. P2 preparation is in progress; D11 hardware acceptance and native gates remain open. | [Preparation guide](offline-gpu-preparation.md); 19 host/OpenFHE CTests; [daily entry](journal/2026-11-12.md#offline-preparation-before-gpu-rental) |
| 2026-11-14 | D11 diagnostic | Supplied RTX 4080 SUPER reports compute 8.9; clean native run passed 23 CTests, 169279488 CUDA-gate comparisons and four sanitizer smoke tools. Exported reports and initial 46-bit latency sample. Primary RTX 5090 and remaining P2 gates stay open. | [Native session](journal/2026-11-14.md); local `report.json`: `GPU_CHECKS_PASSED` |

</details>
