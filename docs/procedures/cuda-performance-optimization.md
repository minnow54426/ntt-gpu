# Optimize a CUDA operation

## Establish the baseline

1. Pick the operation the caller needs and state the timing boundaries. For
   NTT, record CUDA event time for the device operation and wall time for the
   resident interface and public call. For ModUp/ModDown, compare resident
   input, the host-facing call, and an explicit host roundtrip for the
   intermediate QP result.
2. Record the source revision, GPU, CUDA build, dimensions, primes, roots,
   input seed, warmups and sample count. Reuse the same workload for both
   variants. Keep profiling and latency runs separate.
3. Pass native array comparisons against pinned OpenFHE and the applicable
   sanitizer checks before collecting timing samples. Preserve tower and
   coefficient ordering and compare the full output.
4. Capture a CUDA timeline after timing. Use it to inspect launches, gaps,
   allocations and copies. When hardware counters are available, profile
   the affected kernels by name to examine memory traffic, occupancy or
   instruction behavior. A profiler permission failure belongs in the record.

## Choose and test one change

1. Write down one hypothesis and the evidence that would distinguish it.
   The survey proposes Shoup multiplication for fixed twiddles and coalesced
   loads for the strided NTT pass. Test these in separate comparisons.
2. Keep the dispatch, data layout and transfer policy fixed except where the
   experiment changes them. Run the same correctness and sanitizer checks on
   the candidate.
3. [Analyze the paired results](gpu-benchmark-session.md#analyze-on-the-mac)
   and check whether the gain survives at the operation and caller boundaries.
4. Record the decision with the raw evidence and the workloads it supports.
   Change the default after the relevant exactness and performance checks pass.

## Where this ran

The November 19 Ada comparison shows why the boundaries matter. Group
twiddles improved median forward NTT device-event time by about 1.52× across
the eight measured workloads. The corresponding public-call ratio was 1.01×.
[Paired Ada analysis](../paired-ntt-native-analysis.md)

Shoup twiddles, coalesced strided loads, and allocation reuse still need
paired comparisons. [CUDA optimization survey](../cuda-kernel-optimization-survey.md)
