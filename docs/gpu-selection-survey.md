# GPU Selection Survey for CKKS NTT, Base Conversion, and Hybrid Key Switching

Last updated: 2026-11-06  
Status: initial hardware decision; validate with the benchmark gate below before purchase or long-term rental

## Decision

**Use the GeForce RTX 5090 as the primary development and optimization target.** It is the strongest practical single-workstation choice for this project: Blackwell `sm_120`, 32 GB of GDDR7, 1.792 TB/s peak memory bandwidth, 96 MB of L2, and 104.8 peak non-Tensor INT32 TOPS. NVIDIA [lists a US starting price of $1,999](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/), although actual market prices vary. These characteristics are relevant to NTT/RNS work even though the card is also marketed heavily for FP4/FP8 AI.

**Use an RTX PRO 6000 Blackwell when 32 GB is proven insufficient or production reliability is required.** It has the same 1.792 TB/s memory bandwidth but 96 GB of GDDR7 with ECC, 128 MB of L2, 24,064 CUDA cores, and 126 peak non-Tensor INT32 TOPS. The tradeoff is a much higher price and a 600 W board power limit.

**Do not choose H100/H200/B200 solely from AI Tensor-Core numbers.** Their HBM capacity and bandwidth are excellent, but NVIDIA does not publish a directly comparable native INT64 modular-multiply throughput. They may win a memory-bound, large-batch production workload, but that is an inference that must be established by our kernels and profiler data.

This is a provisional decision, not a claim that the RTX 5090 is universally fastest. The final selection must pass the cross-GPU benchmark gate in this report.

## Why FP8 and FP16 marketing does not answer our question

Our hot operations are expected to be:

- `u32`/`u64` modular add and subtract;
- 32-by-32-to-64 and 64-by-64 product-high/product-low operations;
- Shoup or Barrett reduction;
- permutations, gathers, transposes, and RNS-tower traffic;
- shared-memory and register movement;
- base-conversion accumulation across Q and P towers.

NVIDIA's [Tensor Core input-type table](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#features-and-technical-specifications) lists FP64, TF32, BF16, FP16, FP8, FP6, FP4, INT8, and INT4. It does **not** list arbitrary `u32` or `u64` modular arithmetic. Therefore, FP8/FP16 Tensor TOPS are not a direct performance metric for our kernels.

NVIDIA's [PTX integer instruction documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#integer-arithmetic-instructions-mul) supports `mul.hi`, `mul.lo`, and `mul.wide`. The full-width `.wide` form exists only for 16- and 32-bit inputs. CUDA also exposes [`__umul64hi`](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html), which returns the high half of a 64-by-64-bit product. These are the relevant building blocks for Shoup/Barrett arithmetic.

The [CUDA Best Practices throughput table](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#throughput-of-native-arithmetic-instructions) gives `mad.lo.s32` a theoretical 64 results per cycle per SM for the listed architectures, with a footnote of 32 for extended-precision operations. It publishes a row for 64-bit integer addition, but **no row for 64-bit integer multiplication**. Consequently, any claim such as “GPU X has Y INT64 TOPS” would be unsupported by NVIDIA's public specifications.

NVIDIA's RTX Blackwell whitepaper does state that the RTX Blackwell SM doubles integer-math throughput per clock compared with Ada and publishes peak non-Tensor INT32 totals. This is useful evidence for the RTX 5090, but it still does not replace measurement of our multi-instruction modular multiplication.

## Official specification comparison

`INT32` below means the published non-Tensor peak when NVIDIA provides one. It must not be read as INT64 or modular-multiplication throughput. `Not published` is intentional; this report does not invent missing numbers.

| GPU | Architecture / CC | Memory | Peak bandwidth | Official L2 | ECC / reliability statement | Power | Published non-Tensor INT32 | CUDA support |
|---|---|---:|---:|---:|---|---:|---:|---|
| GeForce RTX 5090 | RTX Blackwell GB202 / 12.0 | 32 GB GDDR7 | 1,792 GB/s | 96 MB | GDDR7 built-in single-bit correction is always enabled; not the same enterprise product/support proposition as RTX PRO/data-center cards | 575 W TGP | 104.8 TOPS | CUDA 12.8+; compile `sm_120` |
| GeForce RTX 4090 | Ada AD102 / 8.9 | 24 GB GDDR6X | 1,008 GB/s | 72 MB | No product-level ECC feature published | 450 W TGP | 41.3 TOPS | CUDA 11.8+; compile `sm_89` |
| RTX PRO 6000 Blackwell Workstation | RTX Blackwell GB202 / 12.0 | 96 GB GDDR7 with ECC | 1,792 GB/s | 128 MB | ECC, enterprise drivers/support, MIG | 600 W board power | 126.0 TOPS | CUDA 12.8+; compile `sm_120` |
| L40S | Ada / 8.9 | 48 GB GDDR6 with ECC | 864 GB/s | Not stated on the current L40S page; 96 MB is documented for L40/full AD102 | ECC; designed for continuous data-center operation | 350 W | Not published for L40S | CUDA 11.8+; compile `sm_89` |
| A100 80 GB SXM | Ampere GA100 / 8.0 | 80 GB HBM2e | 2,039 GB/s | 40 MB | HBM, L2, L1, and register-file ECC documented | 400 W standard; selected systems up to 500 W | 19.5 TOPS in NVIDIA's architecture comparison | CUDA 11.0+; compile `sm_80` |
| H100 SXM | Hopper GH100 / 9.0 | 80 GB HBM3 | 3.35 TB/s | 50 MB | Data-center HBM ECC/error-containment family | Up to 700 W | 30 TOPS in NVIDIA's architecture comparison | CUDA 11.8+/12.x; compile `sm_90` |
| H200 SXM | Hopper GH100 / 9.0 | 141 GB HBM3e | 4.8 TB/s | 50 MB Hopper L2 | Data-center HBM ECC/error-containment family | Up to 700 W | Not published in current product sheet | CUDA 11.8+/12.x; compile `sm_90` |
| B200 SXM | Blackwell GB100 / 10.0 | 180 GB HBM3e | Up to 8 TB/s | 126 MB | Data-center HBM ECC/error-containment family | Configurable up to 1,000 W in HGX | Not published for native INT32/INT64 modular arithmetic | CUDA 12.8+; compile `sm_100` |

### Primary sources for the table

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

## Cost and availability snapshot

Prices are volatile, regional, and not directly comparable between a bare card and a complete cloud VM. The following are official reference points, not a purchasing quote:

- NVIDIA lists the RTX 5090 from **$1,999**. NVIDIA launched the RTX 4090 from **$1,599**, but its current official product page no longer provides a meaningful live purchase price. A used or already-owned 4090 can still be economical; a new project should not assume launch-price availability.
- NVIDIA Marketplace listed a partner RTX PRO 6000 Blackwell Workstation Edition at **$14,079.99** when this report was checked. That is about seven times the RTX 5090 list price for three times the VRAM, professional ECC positioning, MIG, enterprise support, and about 20% more published non-Tensor INT32 peak—not seven times the kernel performance. See the [official NVIDIA Marketplace listing](https://marketplace.nvidia.com/en-us/enterprise/laptops-workstations/pny-rtx-pro-6000-blackwell-workstation-edition/).
- Google Cloud lists an RTX PRO 6000 96 GB virtual-workstation GPU at **$1.09565 per GPU-hour** in the first displayed region, before the VM, storage, network, and any licensing charges. The page states that price and availability vary by region. See [Google Cloud GPU pricing](https://cloud.google.com/products/compute/gpus-pricing?hl=en).
- AWS Capacity Blocks list examples of about **$5.97 per H100-hour** and **$14.21–$14.80 per B200-hour**, sold as eight-GPU blocks in the shown regions. These are suitable for bounded cross-GPU experiments, not evidence that either card is cost-effective for daily development. See [AWS Capacity Blocks pricing](https://aws.amazon.com/ec2/capacityblocks/pricing/).

**Inference:** local RTX 5090 ownership is the best development economics when the card remains well utilized. A cloud RTX PRO 6000 is an attractive way to validate the 96 GB path without buying a workstation card. H200/B200 rentals should be delayed until optimized kernels and a fixed benchmark harness exist, otherwise expensive time will mostly measure unfinished code.

## Interpretation for this project

### 1. RTX 5090 versus RTX 4090

**Fact:** RTX 5090 has 1.78 times the peak memory bandwidth, 33% more memory capacity, 33% more L2, and about 2.54 times the published peak non-Tensor INT32 rate.

**Inference:** it should be a materially better target for NTT, basis conversion, and key-switch kernels than RTX 4090. The actual speedup will be smaller or larger depending on whether the kernel is constrained by memory coalescing, extended-precision multiply chains, register pressure, or synchronization.

The RTX 4090 remains useful as an Ada compatibility target and may be attractive if already owned. It should not be purchased as the project's primary new card without a major price advantage.

### 2. RTX 5090 versus RTX PRO 6000 Blackwell

**Fact:** both have 1.792 TB/s peak memory bandwidth and CC 12.0. RTX PRO 6000 has 3 times the memory, 33% more L2, about 20% more published INT32 throughput, ECC positioning, and enterprise features.

**Inference:** for a kernel whose working set fits in 32 GB, the RTX PRO 6000 is unlikely to offer a performance increase proportional to its price. It becomes the better choice when evaluation-key caching, batching, or multiple contexts exceeds 32 GB, or when long-running production reliability/support matters more than acquisition cost.

### 3. RTX 5090 versus A100/H100/H200/B200

**Fact:** the data-center GPUs offer larger ECC HBM and much more bandwidth: about 2.0 TB/s for A100, 3.35 TB/s for H100, 4.8 TB/s for H200, and up to 8 TB/s for B200. H100/H200/B200 also allow larger shared-memory allocations than CC 12.x RTX Blackwell: NVIDIA's [Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/#occupancy) states 228 KB physical capacity and 227 KB maximum per block for CC 10.0, versus 128 KB physical capacity and 99 KB maximum per block for CC 12.0.

**Fact:** NVIDIA does not provide a current product-table value for native INT64 multiply throughput on any of them. Their headline FP8/FP16/INT8 Tensor performance does not measure our workload.

**Inference:** H200 or B200 may be best for a large-batch, memory-bandwidth-bound production service, especially when evaluation keys exceed 32/96 GB or multiple key sets must stay resident. Conversely, RTX 5090 can outperform older data-center GPUs in compute-bound `u32` work because its published non-Tensor INT32 peak is much higher. Only our modular-arithmetic microbenchmarks and end-to-end Hybrid key switching can settle the balance.

### 4. Is 32 GB enough?

For a single RNS polynomial, storage is approximately:

```text
bytes = ring_dimension * active_towers * 8
```

At `N = 65,536`, one polynomial with 40 towers is about 20 MiB. Ciphertext components alone therefore fit easily. Hybrid evaluation keys, QP temporaries, cached twiddle tables, concurrent ciphertexts, and model rotations can dominate the footprint.

**Inference:** 32 GB is likely sufficient for development and many single-request CKKS inference cases, but it must not be assumed sufficient for the final application. We should add peak device-memory measurement to the OpenFHE integration tests before deciding whether production needs 48, 80, 96, 141, or 180 GB.

## Recommended acquisition strategy

1. **Primary local card:** RTX 5090, if the machine can support its power and cooling requirements.
2. **Compatibility card:** retain or rent an RTX 4090/L40S to ensure `sm_89` remains correct and to quantify Blackwell-specific gains.
3. **Production-capacity candidate:** RTX PRO 6000 Blackwell if the measured resident set exceeds 28 GB or enterprise ECC/support is required.
4. **Cloud benchmark candidates:** one H200 and one B200 run after the optimized NTT and base-conversion kernels exist. A100/H100 are useful baselines but should not be assumed optimal for integer-heavy FHE.
5. **Do not commit to a data-center platform** until the same binaries, parameter set, and timing boundaries have been compared.

## Mandatory cross-GPU benchmark gate

Run the same commit, CUDA version where possible, and fixed clocks/power policy. Record both median and p95 over enough iterations to remove startup effects.

### Arithmetic microbenchmarks

- `u32` add/subtract and `mul.hi`/`mul.wide`;
- `u64` add/subtract;
- 64-by-64 low/high product using `__umul64hi`;
- Shoup multiplication for fixed twiddles;
- Barrett multiplication for variable operands;
- the current `__uint128_t % q` reference path;
- register count, spills, achieved occupancy, and instructions per result.

### Memory and NTT tests

- coalesced copy and strided gather bandwidth;
- L2-resident and DRAM-resident cases;
- shared-memory butterfly throughput;
- NTT forward/inverse for `N = 4,096, 8,192, 16,384, 32,768, 65,536`;
- active RNS tower counts representative of early and late CKKS levels;
- kernel-only latency and end-to-end latency separately.

### Base conversion and Hybrid key switching

- Q to QP and QP to Q conversion by level;
- digit decomposition;
- evaluation-key multiply-accumulate;
- relinearization and rotation;
- one request and increasing batch sizes;
- peak device memory with all required keys resident;
- exact equality against the pinned OpenFHE reference at every intermediate boundary.

### Selection rule

- If RTX 5090 fits the complete resident set and is within 15% of the fastest tested card end to end, keep it as the primary platform.
- If 32 GB is the only blocker, compare RTX PRO 6000 against H200 rental cost and deployment constraints.
- If H200/B200 gains track memory bandwidth, prioritize memory-layout and batching work before changing the arithmetic design.
- If gains track modular-multiply throughput, specialize arithmetic and dispatch by compute capability.
- Publish measured results; do not rank cards using FP8/FP16 Tensor TOPS.

## Bottom line

Your teammate is correct that much of RTX 5090 marketing emphasizes FP4/FP8/FP16 AI. That does **not** make the card unsuitable for FHE. RTX 5090 also has a very large general-purpose CUDA engine, a published 104.8 non-Tensor INT32 TOPS peak, high GDDR7 bandwidth, and a large L2 cache. For this project's current development stage, it is the best-balanced target.

The caveat is `u64`: NVIDIA does not publish enough information to declare any card the winner from specifications. We should choose RTX 5090 now, keep the arithmetic layer portable, and require the benchmark gate before selecting the production GPU.
