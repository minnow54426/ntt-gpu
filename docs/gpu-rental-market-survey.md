# GPU rental market: where to run this project

**Checked November 13, 2026 · single-GPU CUDA development · N=65536**

Start with a short, on-demand **RTX 5090** session. Compare RunPod Community
Cloud with verified Vast.ai offers, then choose the machine that supports our
toolchain and profiling work at the lowest complete session cost. Keep an RTX
4090 as a compatibility control. H100/H200/B200 rentals become useful when a
measured workload justifies their capacity or performance.

This is a primary-source market survey, with a read-only marketplace sample.
No machine was booked or tested. Prices are observations, not reserved capacity;
the exact configuration and checkout total decide what a session will cost.

## What we are renting for

Our first paid session verifies the environment and runs the prepared native
CUDA checks. The [offline preparation](offline-gpu-preparation.md) is complete;
the [D11 hardware gate](openfhe-integration-plan.md) remains open. Learning,
documentation, host testing and most build preparation can continue on a CPU.

| Requirement | Project choice |
|---|---|
| Primary GPU | One full RTX 5090, 32 GB, `sm_120` |
| Control GPU | One full RTX 4090, 24 GB, `sm_89` |
| Workload | Exact `u64` modular arithmetic and N=65536 transforms; later OpenFHE Hybrid operations |
| Environment | Linux x86_64, SSH, C++20 compiler, compatible driver and CUDA development toolkit |
| Profiling | Compute Sanitizer; Nsight Compute performance-counter access when profiling is scheduled |
| Practical build allowance | About 8 vCPUs, 16–32 GB host RAM and 40–80 GB disk as a starting recommendation |

The CPU/RAM/disk allowance is our planning estimate, not a measured minimum.
More host resources can save paid compilation time. A runtime-only CUDA image
may lack `nvcc`; choose a development image or install the toolkit. RTX 5090
compilation needs CUDA 12.8 or newer, and the driver must support the toolkit
actually used: see NVIDIA's
[architecture/toolkit matrix](https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html).
Our offline build used CUDA 13.0; that does not establish a rental's compatibility.

Use the [hardware assessment](gpu-selection-survey.md) for GPU specifications.
Tensor FLOPS, image-generation throughput and LLM tokens per second do not rank
our modular kernels. A MIG slice or a different regional GPU variant should
not silently replace the named full-GPU baseline.

## How the market is divided

| Rental model | What matters for us |
|---|---|
| Host marketplace, such as Vast.ai | Variable price, location, CPU allocation, driver and host reliability; inspect each offer |
| GPU containers, such as RunPod Pods and AutoDL containers | Convenient environments; host-controlled driver and profiling permissions still matter |
| GPU virtual machines, such as TensorDock and Jarvislabs VM | More operating-system control; inspect passthrough, resources and counter permissions |
| Data-center GPU clouds, such as Lambda | Useful for later HBM/capacity comparisons; select the actual instance size |
| Serverless endpoints and inference APIs | A different product; exclude their rates from an interactive CUDA development comparison |

Choose **on-demand** for the first debugging session. Interruptible/spot capacity
can make sense later for repeatable jobs that save results and tolerate restart.
Monthly reservations are premature while our GPU use is intermittent. Renting
also avoids buying a workstation before we know sustained utilization.

## Published price snapshot

All dollar prices below are **USD per hour**. RunPod values are the public Pods
rate card with its hourly toggle selected. TensorDock's figure is an advertised
starting price. Jarvislabs values use the one-GPU selector. These are different
bundles, so compare CPU/RAM, disk and final billing before choosing.

<div class="site-table-scroll" role="region" aria-label="Published provider rates and qualifications" tabindex="0" markdown="1">
<div style="min-width:760px" markdown="1">

| Provider / product | GPU | Published rate | Evidence and qualification |
|---|---|---|---|
| RunPod Community Cloud | RTX 5090 · 32 GB | **$0.69** | [Pods pricing](https://www.runpod.io/pricing), Community tab; 9 vCPUs / 35 GB host RAM shown |
| RunPod Secure Cloud | RTX 5090 · 32 GB | **$1.39** | Same page, Secure tab; compare the available region and deployment total |
| RunPod Community Cloud | RTX 4090 · 24 GB | **$0.34** | Same page, Community tab; 6 vCPUs / 41 GB host RAM shown |
| RunPod Secure Cloud | RTX 4090 · 24 GB | **$0.89** | Same page, Secure tab |
| TensorDock on-demand VM | RTX 4090 · 24 GB | **From $0.37** | [Model page](https://www.tensordock.com/gpu-4090.html); CPU/RAM allocation affects the price; current full VM quote unverified |
| Jarvislabs on-demand | RTX PRO 6000 Blackwell · 96 GB | **$2.19** | [Pricing](https://jarvislabs.ai/pricing), 1 GPU; 28 vCPUs / 160 GB host RAM shown |
| Jarvislabs on-demand | H100 SXM · 80 GB | **$3.49** | Same page, 1 GPU; later data-center comparison candidate |

</div>
</div>

The RunPod page says it was updated November 12. Both cloud tabs were inspected
in a browser; the default Secure tab alone does not reveal Community pricing.
Neither these public cards nor the following sample guarantees stock.

### Vast.ai: observed offers, not a fixed tariff

At **01:16 UTC on November 13**, a public, read-only offer search returned 64
records. The query requested one RTX 5090 or RTX 4090, verified/rentable hosts,
on-demand access, reported reliability at least 0.99, CUDA capability at least
13.0, at least 16,000 in the API's host-RAM field, 80 GB available disk and one
direct port. Returned records were checked against those filters.

<div class="site-table-scroll" role="region" aria-label="Vast sample compute prices" tabindex="0" markdown="1">
<div style="min-width:620px" markdown="1">

| GPU | Offers in this response | Compute-only range | Sample median |
|---|---|---|---|
| RTX 5090 | 33 | **$0.4533–$1.0533/h** | **$0.72/h** |
| RTX 4090 | 31 | **$0.40–$0.80/h** | **$0.5333/h** |

</div>
</div>

These are the response's `dph_base` values, rounded to four decimals, not
market-wide minima or an exhaustive regional inventory. The endpoint reported
`truncated: false`, but the figures describe only this response. The cheapest
5090 record was offer **50556739** in Sichuan, China; the cheapest 4090 was
**58010958** in Türkiye. Location suitability depends on your connection.

The [saved sample and query](data/gpu-rental-vast-2026-11-13.json) preserve the
observed rates and relevant host metadata. Public IP addresses are omitted.
`dph_total` includes the endpoint's default disk allowance, **not our proposed
80 GB build disk**, so it is not used as an all-in quote. Available disk capacity
is not included storage. Transfer, selected storage and tax can increase cost.

[Vast's pricing documentation](https://docs.vast.ai/guides/instances/pricing)
explains host-set pricing and separate compute, storage and bandwidth charges.
Host verification and a reliability score do not establish Nsight permission or
guarantee uptime. Requery with your region and full resources before renting.

### AutoDL and Lambda: why no current rate is quoted

**AutoDL** is worth checking if its network locations and CNY payment options
suit you. Its [market](https://www.autodl.com/market/list) was not verified to
a specific current offer in this survey. Compare the exact GPU name, CPU/RAM,
disk and driver; treat 4090D/5090D labels as distinct configurations rather than
substituting them into a 4090/5090 result. Keep CNY and USD quotes separate until
you have a dated exchange rate and payment fees.

**Lambda** offers GPU instances, but its
[public table](https://lambda.com/instances) explicitly takes effect
**November 16, 2026 at 00:00 UTC**, three days after this survey. It directs users
to their account for the current price. The table also defaults to eight-GPU
instances while quoting **per GPU-hour**: multiply by the actual GPU count.
Future rates are not used in today's cost comparison.

## Provider fit and tradeoffs

**RunPod — convenient first candidate.** Community's 5090 card is an affordable
starting point with a documented environment workflow. Secure costs more;
compare its specific location and service offering against your needs. Confirm
the selected Pod's resources and profiling support before paying.

**Vast.ai — price-sensitive alternative.** The observed 5090 compute floor was
lower than RunPod's public Community card, while the sample median was higher.
Select an individual host, not just a GPU label. A low floor does not establish
the cheapest usable session after disk, transfer and setup time.

**TensorDock — 4090 VM candidate.** Its model page describes on-demand KVM VMs
with root access and GPU passthrough, and explicitly allows 1–8 GPUs. The page
also retains old Q1 2024 wording; its $0.37 floor is not a verified live cart.
Get a whole-VM quote with sufficient build resources.

**AutoDL — conditional regional candidate.** Consider it if access and payment
are convenient. Verify a specific market offer and profiler permissions; this
survey establishes neither a current hourly GPU price nor Nsight availability.

**Jarvislabs — later capacity comparison.** Its 96 GB RTX PRO card is a concrete
option if our measured working set outgrows 32 GB. Its
[VM product](https://jarvislabs.ai/products/vm) documents SSH and root access;
confirm that the selected product and region match the quoted configuration.

**Lambda — later data-center comparison.** Useful when the experiment requires
HBM or a named data-center GPU. Obtain the current one-GPU or multi-GPU quote;
do not pay for eight devices to run a single-device correctness check.

AWS, Azure and Google Cloud also belong in an enterprise procurement comparison.
Their region, quota, instance size and attached-resource pricing need a separate
matched quote. This shortlist targets our first small CUDA session, rather than
claiming complete coverage or a global cheapest provider.

## The cost beyond GPU-hours

Use one currency and the **whole instance** rate:

\[
C_{\mathrm{session}} = r_{\mathrm{instance}}t_{\mathrm{billable}}
  + C_{\mathrm{disk}} + C_{\mathrm{transfer}} + C_{\mathrm{tax/fees}}.
\]

Billable time includes environment setup and rebuilding dependencies, not only
the seconds spent inside a kernel. Add separately charged CPU/RAM to the
instance rate. An account top-up is prepaid balance, not automatically the cost
of this experiment. Check region, tax, minimum credit and billing granularity
on the final quote.

| Billing issue | What the official sources establish |
|---|---|
| RunPod Pods | [Per-second billing](https://docs.runpod.io/pods/pricing); no ingress/egress fee; one hour's configuration credit needed to deploy on-demand |
| RunPod retained volume | $0.10/GB/month running and $0.20/GB/month stopped; terminating the Pod removes its attached volume disk |
| RunPod network volume | Separately retained storage: $0.07/GB/month below 1 TB; remove it separately when finished |
| Vast.ai | [Compute, disk and bandwidth](https://docs.vast.ai/guides/instances/pricing) are separately priced; [stopping](https://docs.vast.ai/api-reference/hello-world) halts compute but keeps disk charges |
| TensorDock | [Stopping and unreserving the GPU](https://www.tensordock.com/gpu-4090.html) changes an on-demand server to storage-only pricing; verify reserved resources |
| AutoDL | [On-demand billing](https://www.autodl.com/docs/price/) is per second with a CNY 0.01 minimum; [paid disk expansion](https://www.autodl.com/docs/local_disk/) can remain billable while powered off |
| Jarvislabs | [Paused instances](https://jarvislabs.ai/termsandservice) retain storage charges; deleting compute does not necessarily remove separately created storage |

Export reports before deleting resources. Check the billing/resource view after
cleanup, including independent volumes, snapshots and reserved IPs where used.

### Small-session budget examples

The following is **our arithmetic**, using the observed public rates and exactly
four or ten billable hours. It excludes all storage, transfer, tax and separately
billed resources; it is not a quotation or a promise that the work finishes in
that time.

<div class="site-table-scroll" role="region" aria-label="Illustrative compute budgets" tabindex="0" markdown="1">
<div style="min-width:560px" markdown="1">

| Configuration | 4 hours of compute | 10 hours of compute |
|---|---|---|
| RunPod Community RTX 5090 | **$2.76** | **$6.90** |
| RunPod Secure RTX 5090 | **$5.56** | **$13.90** |
| RunPod Community RTX 4090 | **$1.36** | **$3.40** |
| Vast sample's lowest RTX 5090 compute rate | **About $1.81** | **About $4.53** |
| Jarvislabs RTX PRO 6000, 1 GPU | **$8.76** | **$21.90** |

</div>
</div>

For a first attempt, a **$10–$15 spending allowance** is a reasonable planning
choice for one short 5090 session plus modest extras. It is neither a provider
minimum deposit nor an upper bound: a long build, retained disks, transfer or a
different quote can exceed it. Record actual costs and choose the next allowance
from observed session time.

At the RunPod Community card rates, 5090 compute costs about **2.03×** the 4090.
The 5090 must finish a fixed job over 2.03× faster to win on compute cost alone.
That speedup is unmeasured; development value and extra VRAM are separate
reasons to choose it. Rank later providers by dollars per verified workload,
including transfers, rather than dollars per advertised GPU-hour.

## Profiling access can decide the provider

A machine can run CUDA correctly and still reject Nsight Compute's hardware
counters with `ERR_NVGPUCTRPERM`. NVIDIA's
[permission guide](https://developer.nvidia.com/ERR_NVGPUCTRPERM) explains the
host/driver permissions and container capabilities involved. SSH access or a
root shell inside a container does not itself establish those permissions.

**Counter access is unverified for every specific offer in this survey.** Ask
whether Nsight Compute hardware-counter collection is permitted on the chosen
GPU and product, and whether the provider enables the required host permissions.
A full VM is worth considering when container restrictions block the experiment,
but it also needs verification. A successful `nvidia-smi` is not that verification.

Treat correctness, sanitizer coverage and profiling as separate observed results.
If a host cannot provide counters, it may still serve the correctness session;
schedule profiling on a host that can. Do not report the profiling gate complete
from timing-only runs.

## Recommended first rental

1. **Finish preparation on CPU.** Use the pinned source/dependencies and the
   [offline handoff](offline-gpu-preparation.md). Check image compatibility before
   copying binaries; rebuilding OpenFHE can consume paid time.
2. **Compare two live carts.** Select one full on-demand RTX 5090 on RunPod
   Community and a suitable verified Vast.ai host. Match region, driver,
   CPU/RAM, an explicit disk allocation and the complete hourly total.
3. **Resolve profiling access.** Obtain a clear answer for the chosen product
   before a counter-dependent session. Confirm SSH and development tools too.
4. **Run the prepared collection path.** Follow the handoff's
   [`gpu_session.py`](../scripts/gpu_session.py) command with the actual CUDA and
   OpenFHE paths. Capture environment, native exactness, sanitizer and benchmark
   reports. The runner is not the complete later profiling/failure-test suite.
5. **Export evidence and end billing.** Save logs and cost records, then stop or
   delete according to the provider's resource rules. Perform analysis offline.

The next evidence we need is one successful, reproducible native GPU session.
Keep D11 open until that evidence exists. Revisit 96 GB cards or H100/H200/B200
after measuring residency and complete-workload performance.

## Sources and freshness

Pricing pages, linked billing documentation and NVIDIA's permission guide were
checked on November 13, 2026. The Vast JSON is a timestamped observation, with
the exact read-only query and selected public fields. RunPod's two cloud tabs
were read in Chromium. AutoDL's billing rules were available through official
indexed documentation; a current GPU offer was not established. No checkout,
paid instance, network-speed measurement or provider support confirmation was
performed. Recheck price, inventory and permissions before spending.
