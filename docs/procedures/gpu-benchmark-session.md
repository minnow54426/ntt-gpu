# Prepare and run a rented GPU session

## Batch the prepared tasks

At the next rental, first complete the
[AutoDL profiling-access TODO](../next-gpu-rental.md#todo-verify-autodl-profiling-access-at-the-next-rental).
Test a small CUDA/NVTX trace and real SM/L2/DRAM counter capture before the
long batch; save both successes and permission failures in the session record.

Preview `python3 scripts/gpu_rental_batch.py --mode plan` without a GPU.
Use the [single-batch commands](../next-gpu-rental.md#run-the-queue-as-one-batch)
for offline compilation or a native RTX 5090 run. The native batch combines
both schedules, resident conversion, sanitizers, ordered repeats, independent
profiles and a verified evidence archive within one command budget.

## Before releasing the GPU

1. Run CPU tests and prepare workload vectors on the Mac before renting.
   Follow the [GPU rental runbook](../next-gpu-rental.md) for the current
   matrix and commands.
2. On the rental, record the actual GPU and tool versions. Run the native
   OpenFHE comparisons and sanitizer checks before timing. Collect a
   separate CUDA timeline even if hardware counters are unavailable.
3. Copy the results archive and its SHA-256 file to the Mac. Verify the
   checksum, extract the records, and confirm the required jobs completed
   before releasing the GPU.

## Analyze on the Mac

1. [Run the paired analyzer](../paired-ntt-native-analysis.md#reproduce-the-analysis-on-the-m1-pro)
   on the extracted records. Require matching inputs and passing native checks.
2. Compare medians and p95 by workload and timing boundary. The legacy
   median divided by the group-twiddle median is the reported speedup.
   Check both run orders for a representative case if host or thermal drift
   could explain the result.
3. Use operation ranges when attributing timeline costs. Whole-trace totals
   can include setup and verification. Record the result and profiler limits
   in the journal.

The November 22 batch exercised the native run, archive handoff, and local
analysis. Offline conversion-reference replay still needs a native GPU run.
[November 22 journal](../journal/2026-11-22.md)
