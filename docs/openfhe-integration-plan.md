# CKKS GPU-to-OpenFHE integration plan

**Target:** OpenFHE 1.6.1 CKKS with Hybrid key switching  
**Primary arithmetic:** 50–60-bit RNS primes stored in `uint64_t`, with
`uint32_t` indexing/control paths where safe  
**Ring dimension:** `65536`<br>
**Estimate:** 95 planned working days + 15 risk-reserve days  
**Last updated:** 2026-11-12 (Asia/Shanghai)

Follow the live tracker, then open the phase you need from the section sidebar.
The sequence is NTT → base conversion → Hybrid key switching → OpenFHE.
Update this tracker only when a deliverable passes its gate.

## Live tracker

| Field | Current value |
|---|---|
| Working day | **D11** |
| Active phase | **P2 — first RTX GPU baseline (ready to start)** |
| Active task | Provision and record the named GPU environment; P1 and D1–D10 are complete |
| Next gate | P2: native CUDA/OpenFHE exactness and named-hardware baseline (D11–D15) |
| Overall state | In progress; P2 awaits GPU provisioning |

Hardware: RTX 5090 primary, RTX 4090 control. Use larger-memory hardware only
when measurements justify it; see the [GPU comparison](gpu-selection-survey.md).

Status vocabulary: `DONE`, `IN PROGRESS`, `NOT STARTED`, `BLOCKED`.

| Phase | Working days | Status | Exit artifact |
|---|---:|---|---|
| P0 — pre-plan correctness baseline | Before D1 | DONE | CPU oracle, schedule simulator, CUDA source, exact OpenFHE NTT gate |
| P1 — contracts and architecture | D1–D10 | DONE | [Accepted ADR](architecture-decision.md), frozen operation graph and [implementation backlog](implementation-backlog.md) |
| P2 — first RTX GPU baseline | D11–D15 | NOT STARTED | Exact CUDA results and named-hardware baseline |
| P3 — production GPU NTT | D16–D40 | NOT STARTED | Tuned, device-resident NTT dispatch |
| P4 — exact base-conversion model | D41–D49 | NOT STARTED | OpenFHE-exact CPU Q↔QP oracle and fixtures |
| P5 — GPU base conversion | D50–D61 | NOT STARTED | Exact and tuned ModUp/ModDown kernels |
| P6 — Hybrid key-switch contract | D62–D68 | NOT STARTED | Frozen decomposition/QP/MAC/ModDown graph |
| P7 — GPU Hybrid key switching | D69–D85 | NOT STARTED | Exact relinearization and rotation backend |
| P8 — OpenFHE integration | D86–D95 | NOT STARTED | Integrated backend with fallback and full tests |
| P9 — risk reserve | D96–D110 | NOT STARTED | Used only for integration and correctness risks |

## Definition of exact compatibility

Given the same parameters, input arrays, ciphertext components, and evaluation
keys, the GPU path must produce the same per-tower arrays as the pinned OpenFHE
implementation. Approximate CKKS decryption agreement is a secondary test, not
the compatibility criterion. Key generation randomness is outside this rule;
the fixed generated key is part of the test input.

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
- [ ] Execution of CUDA gates on real hardware; this belongs to P2.

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

D1 complete: the [interactive work-distribution map](ntt-65536-work-distribution.html)
covers N=65536, gather/scatter and shared-memory indices, launch geometry,
active thread ranges, and requested memory-access patterns. This is a source
analysis and CPU-schedule check, not evidence of real-GPU correctness or speed.

D2 complete: the same page's [column proof](ntt-65536-work-distribution.html#column-proof)
derives CPU group/offset equivalence, both global exponents, coverage, bounds,
and three worked examples. `column_mapping_check` runs the production stage
body with two tagged-table probes for every column butterfly at N=65536;
the separate schedule gate checks real-root arrays. Real-GPU validation is
still a P2 deliverable.

D3 complete: the [inverse proof](ntt-65536-work-distribution.html#inverse-proof)
derives the factor of 2 per butterfly and final N-inverse normalization,
documents row→column→scale ordering, and provides worked examples.
`inverse_column_check` probes every inverse column pair and inactive thread
at N=65536, then checks real-root stage composition and full normalization
for two 46-bit primes. These are CPU-side checks of shared helpers,
not actual GPU execution.

D4 complete: the [row-kernel and coverage proof](ntt-65536-work-distribution.html#tile-proof)
derives tile-local/global group equivalence and global twiddle exponents,
proves the far/near stage partition and legal block-local stage fusion, and
provides worked examples. `tile_mapping_check` probes every near-stage pair
in both directions at N=65536 and checks stage/endpoint coverage
for N=65536 plus N=2,64,256,512. Hardware races and CUDA
launch behavior remain P2 work.

D5 complete: [CUDA buffers and dataflow](cuda-buffer-dataflow.md) maps the
constructor, all six device allocations, normal destruction, per-method
copies, launch composition, representation ownership and missing integration
contracts. Static source checks pin this map and byte-count arithmetic; they
are not CUDA-runtime tests. Constructor exception cleanup is an open gap:
the raw `Impl*` can leak host state and partially allocated device buffers.
RAII/failure-path verification and device/stream ownership must be addressed
before the production API is accepted; no cleanup fix is claimed here.

D6 complete: [Arithmetic bounds and tests](modular-arithmetic-contract.md)
records canonical-range induction, integer-width bounds, parameter preconditions,
the exactness ladder and deferred adversarial cases. Two CPU gates compare
14,956 primitive results against Python big integers and exercise six patterns
in both transform directions plus a closed-form negacyclic product at N=65536
with 50/55/60-bit primes. The multiplication-count comment is corrected;
arithmetic algorithms are unchanged. Native GPU and invalid-input hardening
remain explicit later gates, not evidence claimed by this review.

D7 complete: the [OpenFHE representation contract](openfhe-representation-contract.md)
traces pinned NativePoly/DCRTPoly format conversion, exact odd-root evaluation
indices, minimum-root selection and its cache constraint, canonical residues,
ordered tower packing/import, and active-prefix behavior. A new opt-in core-only
gate checks full-size formulas and three-tower operations with 50/55/60-bit
primes against OpenFHE 1.6.1. The OpenFHE-enabled Release suite passes 14 tests
on this Linux CPU environment. No CUDA execution or Hybrid rounding graph is
claimed by that D7 result. D8 and D10's later architecture acceptance are documented below.

D8 complete: the [Hybrid operation graph](openfhe-hybrid-operation-graph.md)
traces digit splitting, partial partitions, coefficient/evaluation transitions,
noncanonical approximate CRT, full-key P offsets at reduced levels, B/A MAC
ordering, exact CKKS ModDown, wrapper updates, and delayed QP operations.
`openfhe_hybrid_graph` checks 18 full-size cases and 117,571,584 residue
comparisons against independent unreduced-CRT and scalar formulas, including
real generated keys and negative alternatives. The opt-in Release suite now
has 15 CPU tests. Rotation automorphisms and higher-degree relinearization are
source-traced but remain later end-to-end gates; production matrix coverage,
GPU execution and performance are not claimed.

D9 complete: the [device backend contract](device-backend-contract.md) proposes
immutable parameter snapshots, checked whole-tower views, device/session and
generation identities, explicit stream/event dependencies, pinned resource
lifetimes, cache identities, and an opt-in transactional DCRT format adapter.
The portable metadata prototype is checked by `backend_contract_check` and a
bridge to actual D8 OpenFHE output/key parameters. The opt-in Release suite
has 16 CPU tests. Abstract interfaces compile without a CUDA dependency;
CUDA resource management, event execution, caches and the adapter are not
implemented. D10's acceptance is recorded below.

D10 complete: [ADR 0001](architecture-decision.md) accepts the D7–D9 behavioral
contracts and the [dependency-ordered implementation backlog](implementation-backlog.md).
It freezes the explicit DCRT adapter, validated no-op and preflight-only fallback,
immutable outputs, registry/lease ownership, partial-submission quarantine,
separate configurable memory budgets and fail-fast admission. The review accepts
P1's architecture; it does not promote the abstract API to a runtime backend.
The existing CUDA constructor gap, native build, exactness, profiling and all
later implementation gates remain open backlog items. D11 is next; P2 has not
started because a native GPU environment has not been provisioned here.

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

| Day | Work | Required output |
|---:|---|---|
| D11 | Provision the selected GPU and CUDA toolkit | Reproducible environment record |
| D12 | Build native architecture, resolve baseline safety/build blockers and run smoke tests | Native build, constructor/failure cleanup and validated smoke tests (B02) |
| D13 | Run exact gates over all `N` and modulus widths | Correctness report |
| D14 | Profile with CUDA events and Nsight | Kernel, L2, DRAM, occupancy, and register data |
| D15 | Analyze and publish baseline | Raw samples, summary table, first bottleneck decision |

Follow [backlog B01–B04](implementation-backlog.md#2-p2-establish-the-native-baseline).
Constructor RAII, device ownership and input-validation blockers must be fixed
and fault-tested before the native baseline is accepted. The setup helper's
default CPU/46-bit checks and benchmark verdict do not substitute for the
OpenFHE-enabled native matrix or validated timing collection.

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

Do not schedule new features in the reserve. Cipher inference begins only after
P8 is green; it is a separate plan covering model operators, rescaling,
rotations, graph scheduling, accuracy, and end-to-end latency.

## Progress log

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
