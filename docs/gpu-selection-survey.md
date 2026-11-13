# Choosing a GPU for CKKS

Hardware assessment recorded on 2026-11-06. The RTX 5090 choice is provisional;
production selection requires the benchmark gate below. Specifications and
historical prices are reference values, not current quotes or kernel timings.

## Decision

| Role | GPU | Reason |
|---|---|---|
| Primary development | RTX 5090 | 32 GB, 1.792 TB/s bandwidth, Blackwell `sm_120` |
| Compatibility control | RTX 4090 | Ada `sm_89`; useful when owned or rented |
| More capacity / enterprise support | RTX PRO 6000 Blackwell | 96 GB ECC; evaluate when the working set exceeds 32 GB |
| Later production comparison | H200 / B200 | Larger HBM and bandwidth; benchmark the complete workload |

Use the 5090 for development. Select a production GPU only after measuring
resident memory and end-to-end Hybrid key switching.

## Why FP8 and FP16 marketing does not answer our question

These kernels perform exact modular `u32`/`u64` arithmetic and move RNS data.
Tensor TOPS do not measure that workload. NVIDIA's
[Tensor Core type table](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#features-and-technical-specifications)
does not list arbitrary `u32`/`u64` modular operations.

Relevant primitives include [`mul.hi`, `mul.lo`, and `mul.wide`](https://docs.nvidia.com/cuda/parallel-thread-execution/#integer-arithmetic-instructions-mul)
and [`__umul64hi`](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html).
The [native instruction table](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#throughput-of-native-arithmetic-instructions)
does not publish a directly comparable 64-bit multiply throughput. Measure
Shoup/Barrett instruction chains rather than deriving an “INT64 TOPS” ranking.

## Official specification comparison

Bandwidth is a peak specification. INT32 is published non-Tensor throughput,
not modular multiplication performance. “—” means no comparable value was
published in the cited product material.

| GPU | Memory | Peak bandwidth | L2 | CC | INT32 TOPS | Power |
|---|---|---|---|---|---|---|
| RTX 5090 | 32 GB GDDR7 | 1.792 TB/s | 96 MB | 12.0 | 104.8 | 575 W |
| RTX 4090 | 24 GB GDDR6X | 1.008 TB/s | 72 MB | 8.9 | 41.3 | 450 W |
| RTX PRO 6000 Blackwell | 96 GB ECC GDDR7 | 1.792 TB/s | 128 MB | 12.0 | 126 | 600 W |
| L40S | 48 GB ECC GDDR6 | 864 GB/s | Not stated | 8.9 | — | 350 W |
| A100 80 GB SXM | 80 GB HBM2e | 2.039 TB/s | 40 MB | 8.0 | 19.5 | 400 W standard |
| H100 SXM | 80 GB HBM3 | 3.35 TB/s | 50 MB | 9.0 | 30 | Up to 700 W |
| H200 SXM | 141 GB HBM3e | 4.8 TB/s | 50 MB | 9.0 | — | Up to 700 W |
| B200 SXM | 180 GB HBM3e | Up to 8 TB/s | 126 MB | 10.0 | — | Up to 1,000 W |

CC 12.0/10.0 compilation requires CUDA 12.8+. RTX 5090 memory correction is
not the same enterprise reliability/support offering as RTX PRO or data-center
ECC. L40S L2 is not stated on its product page; the full AD102/L40 figure
must not be silently substituted.

### Primary sources for the table

<details markdown="1">
<summary>Open NVIDIA specification sources and qualifications</summary>

- NVIDIA's [CUDA GPU compute-capability list](https://developer.nvidia.com/cuda/gpus) maps RTX 5090 and RTX PRO 6000 Blackwell to CC 12.0, B200 to CC 10.0, H100/H200 to CC 9.0, RTX 4090/L40S to CC 8.9, and A100 to CC 8.0.
- NVIDIA's [CUDA architecture/toolkit matrix](https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html) lists first toolkit support and ongoing support; CUDA 12.8 added `sm_100`, `sm_101`, and `sm_120` compiler support.
- The [RTX Blackwell architecture whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf) provides the RTX 5090's 170 SMs, 21,760 CUDA cores, 32 GB GDDR7, 1.792 TB/s, 96 MB L2, 575 W, 104.8 INT32 TOPS, and built-in GDDR7 single-bit correction. It also provides RTX 4090 comparison figures.
- The [RTX PRO Blackwell architecture whitepaper](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/pdf/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1_1.pdf) provides 96 GB ECC GDDR7, 1.792 TB/s, 128 MB L2, 600 W, and 126 INT32 TOPS. The shorter [product datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/rtx-pro-6000-blackwell-workstation-edition/workstation-blackwell-rtx-pro-6000-workstation-edition-nvidia-us-3519208-web.pdf) confirms the main board specifications.
- The official [L40S product page](https://www.nvidia.com/en-us/data-center/l40s/) specifies 48 GB GDDR6 with ECC, 864 GB/s, 18,176 CUDA cores, and 350 W. The [Ada architecture whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf) documents the 96 MB L2 full-AD102/L40 configuration; the current L40S page does not separately state L2.
- The [A100 product page](https://www.nvidia.com/en-us/data-center/a100/) specifies 80 GB HBM2e, 2,039 GB/s for SXM, and 400 W. NVIDIA's [Ampere architecture analysis](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/) documents 40 MB L2 and SECDED protection of HBM, L2, L1, and register files.
- The [H100 datasheet](https://resources.nvidia.com/en-us-gpu/h100-datasheet-24306) specifies 80 GB, 3.35 TB/s, and up to 700 W. NVIDIA's [Hopper architecture analysis](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) documents 50 MB L2 and its architecture comparison.
- The [H200 datasheet](https://resources.nvidia.com/en-us-gpu-resources/hpc-datasheet-sc23) specifies 141 GB HBM3e, 4.8 TB/s, and up to 700 W.
- NVIDIA's [HGX component table](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html) specifies H100/H200/B200 capacity and bandwidth and a configurable limit up to 1 kW per GPU. The [Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/) documents B200/GB200's 180 GB memory support and 126 MB L2.
- NVIDIA's [GPU memory-error management guide](https://docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/latest/595/overview.html) documents HBM ECC correction and error-management behavior for Ampere, Hopper, and Blackwell data-center families.

</details>

## Cost and availability snapshot

Prices in the original assessment were regional, time-sensitive reference
points. Compare the complete workstation or VM cost and measured work per hour.

<details markdown="1">
<summary>Read the historical pricing evidence</summary>

Prices are volatile, regional, and not directly comparable between a bare card and a complete cloud VM. The following are official reference points, not a purchasing quote:

- NVIDIA lists the RTX 5090 from **$1,999**. NVIDIA launched the RTX 4090 from **$1,599**, but its current official product page no longer provides a meaningful live purchase price. A used or already-owned 4090 can still be economical; a new project should not assume launch-price availability.
- NVIDIA Marketplace listed a partner RTX PRO 6000 Blackwell Workstation Edition at **$14,079.99** when this report was checked. That is about seven times the RTX 5090 list price for three times the VRAM, professional ECC positioning, MIG, enterprise support, and about 20% more published non-Tensor INT32 peak—not seven times the kernel performance. See the [official NVIDIA Marketplace listing](https://marketplace.nvidia.com/en-us/enterprise/laptops-workstations/pny-rtx-pro-6000-blackwell-workstation-edition/).
- Google Cloud lists an RTX PRO 6000 96 GB virtual-workstation GPU at **$1.09565 per GPU-hour** in the first displayed region, before the VM, storage, network, and any licensing charges. The page states that price and availability vary by region. See [Google Cloud GPU pricing](https://cloud.google.com/products/compute/gpus-pricing?hl=en).
- AWS Capacity Blocks list examples of about **$5.97 per H100-hour** and **$14.21–$14.80 per B200-hour**, sold as eight-GPU blocks in the shown regions. These are suitable for bounded cross-GPU experiments, not evidence that either card is cost-effective for daily development. See [AWS Capacity Blocks pricing](https://aws.amazon.com/ec2/capacityblocks/pricing/).

**Inference:** local RTX 5090 ownership is the best development economics when the card remains well utilized. A cloud RTX PRO 6000 is an attractive way to validate the 96 GB path without buying a workstation card. H200/B200 rentals should be delayed until optimized kernels and a fixed benchmark harness exist, otherwise expensive time will mostly measure unfinished code.

</details>

## Interpretation for this project

| Comparison | Published difference | What still needs measurement |
|---|---|---|
| 5090 versus 4090 | 1.78× bandwidth, 33% more VRAM/L2, about 2.54× INT32 peak | Actual NTT and key-switch speedup |
| PRO 6000 versus 5090 | Same bandwidth, 3× VRAM, 33% more L2, about 20% more INT32 peak | Benefit from capacity, reliability, and batching |
| H200/B200 versus RTX | Larger HBM and higher bandwidth | Modular arithmetic, cache behavior, and end-to-end cost |

At N=65536, a 40-tower polynomial uses `65536 × 40 × 8 = 20 MiB`.
Evaluation keys, QP temporaries, twiddles, and concurrent ciphertexts can consume
much more. Record peak memory with all keys resident before choosing capacity.

## Recommended acquisition strategy

For dated provider rates, billing rules and the first paid session, read the
[GPU rental market survey](gpu-rental-market-survey.md), checked November 13,
2026. Keep those rental observations separate from this hardware assessment's
historical purchase prices.

1. Develop and tune on RTX 5090; check workstation power and cooling.
2. Retain or rent RTX 4090 for cross-architecture correctness and timing.
3. Evaluate RTX PRO 6000 if measured residency exceeds 28 GB or enterprise
   reliability/support is required.
4. Rent H200/B200 after the optimized kernels and fixed harness exist.

## Mandatory cross-GPU benchmark gate

Run the same commit, parameters, and timing boundaries. Record toolkit, clocks,
power policy, median, p95, and peak memory.

| Layer | Required cases |
|---|---|
| Arithmetic | u32/u64 add/subtract, high/low products, Shoup, Barrett, reference remainder; registers and spills |
| Memory / NTT | Coalesced copy, strided gather, L2/DRAM cases, forward/inverse at N=65536, realistic tower counts |
| Hybrid | ModUp, ModDown, decomposition, key MAC, relinearization, rotation; one request and batches |
| Correctness | Exact pinned-OpenFHE arrays at every intermediate boundary: [`NTT`](https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/core/include/lattice/hal/default/poly-impl.h#L454-L474) and [`Hybrid core`](https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/pke/lib/keyswitch/keyswitch-hybrid.cpp#L295-L298) |

See the [benchmark plan](benchmark-tuning-plan.md) for counters and result
records. Keep kernel-only and transfer-inclusive timing separate.

### Selection rule

Keep RTX 5090 as primary if the complete resident set fits and its end-to-end
latency is within 15% of the fastest tested card. If only capacity blocks it,
compare RTX PRO 6000 with H200 rental and deployment costs. Use profiling to
separate memory-layout limits from modular-multiply limits before redesigning
arithmetic or selecting hardware.

## Bottom line

RTX 5090 is the development choice. Production hardware remains an open,
measured decision; FP8/FP16 Tensor figures do not settle it.
