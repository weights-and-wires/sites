---
id: "roofline"
title: "Roofline: An Insightful Visual Performance Model for Multicore Architectures"
authors: "Williams et al."
year: 2009
venue: "Communications of the ACM 52(4)"
arxiv: null
url: "https://doi.org/10.1145/1498765.1498785"
track: gpu-systems
tier: core
est_hours: 3
status: full
summary: "The Roofline model bounds a kernel's performance by the smaller of two limits: the machine's peak compute rate and its memory bandwidth multiplied by the kernel's operational intensity, which is FLOPs per byte of DRAM traffic. Plotted on log-log axes, the two limits form a roof whose corner, the ridge point, separates memory-bound kernels from compute-bound ones. Extra ceilings below the roof name the optimizations that lift a kernel, and cache optimizations that cut traffic move it right."
related_course: "gpu: performance model"
prereqs: []
background: [arithmetic-intensity, memory-bandwidth, flops]
course_links:
  - title: "The roofline and latency hiding"
    url: "https://gpu.ifkash.dev/c/04-roofline-and-latency"
  - title: "The roofline"
    url: "https://qwen.ifkash.dev/c/10-roofline"
self_check:
  - q: 'A machine has a peak of 100 GFLOP/s and 10 GB/s of memory bandwidth. What is the attainable performance of a kernel with an intensity of 4 FLOPs per byte, and is it memory-bound or compute-bound?'
    a: 'The bandwidth roof is $10 \times 4 = 40$ GFLOP/s, which is below the 100 GFLOP/s peak, so the attainable performance is 40 GFLOP/s and the kernel is memory-bound. The ridge point is $100 / 10 = 10$ FLOPs per byte, and the kernel sits to its left.'
    see: arithmetic-intensity
  - q: 'The paper counts only the bytes that reach DRAM after the cache hierarchy has filtered the accesses. Why does that choice matter when you compare two versions of the same kernel, one of them tiled for the cache?'
    a: 'Both versions do the same FLOPs, but the tiled version sends fewer bytes to DRAM, so its intensity is higher and the kernel moves right on the plot. If you counted bytes between the processor and the cache instead, both versions would have the same intensity and the model could not show the benefit of tiling.'
    see: memory-bandwidth
  - q: 'A kernel with an intensity of 0.25 FLOPs per byte runs at 4.2 GFLOP/s. How much DRAM bandwidth does it use?'
    a: 'Bandwidth is performance divided by intensity: $4.2 / 0.25 = 16.8$ GB/s. This is the check the paper makes in Table 4 for SpMV on the Opteron X4. A kernel on the slanted part of the roof is close to saturating the memory system.'
    see: flops
---

## The problem

A processor's data sheet lists one number for compute, its peak FLOP/s. Programmers read that number as what the machine can do. For many kernels it is far from true. In 2008, when the paper was written, a dual-socket AMD Opteron X2 had a peak of 17.6 GFLOP/s and 15 GB/s of DRAM bandwidth. Take a sparse matrix-vector multiply (SpMV) that does about 0.25 FLOPs per byte of DRAM traffic. It can't run faster than $15 \times 0.25 = 3.75$ GFLOP/s on that machine, however well the code is written. That is 21% of the peak (computed by `scripts/roofline_example.py`). The paper notes that conventional SpMV code often runs below 10% of peak on a single processor (Section 6.3.1).

Two questions follow, and the peak number answers neither. Is this kernel limited by compute or by memory? And what should a programmer try next? Earlier tools gave detailed predictions that were hard to use, or a table of counters that needed an expert to read. The authors argue in Section 1 that a model is useful if it is insightful, even when it is imperfect. They propose a bound-and-bottleneck model that fits on one graph.

A note on sources. This page follows the UCB technical report version of the paper (EECS-2008-134), which is the version submitted to CACM. Section and figure numbers below are from that report, and the CACM print may number sections differently. The paper has no numbered equations, so the page cites the unnumbered formula in Section 3.

## Key ideas

The page uses the paper's terms: operational intensity, peak floating-point performance, and peak memory bandwidth. The paper writes its formula in words. The page uses these symbols, which are not the paper's: $P_{\text{peak}}$ for peak performance in FLOP/s, $B$ for bandwidth in bytes/s, $I$ for operational intensity in FLOPs per byte, and $I^{*}$ for the ridge point. Every number comes from a script in `scripts/`.

### The roof: min(peak, bandwidth x intensity)

**Intuition.** A kernel needs both compute time and data-movement time, and these overlap. The slower one decides the run time. A fast processor with a slow memory system spends most of its time waiting for data.

**Math.** Section 3 defines operational intensity as operations per byte of DRAM traffic and states the model:

$$
\text{Attainable GFLOP/s} = \min\bigl(\text{Peak Floating-Point Performance},\; \text{Peak Memory Bandwidth} \times \text{Operational Intensity}\bigr)
$$

In the page's symbols:

$$
P_{\text{attain}}(I) = \min\bigl(P_{\text{peak}},\; B \cdot I\bigr)
$$

On a log-log plot with intensity on the x-axis and performance on the y-axis, $P_{\text{peak}}$ is a horizontal line. $B \cdot I$ is a line at 45 degrees, because bytes per second equals (FLOP/s) divided by (FLOPs/byte). The paper points out that you draw these two lines once per machine, not once per kernel. A kernel is a vertical line at its intensity, and its performance must lie on that line, at or below the roof.

**Tiny example, computed by `scripts/roofline_example.py`.** The Opteron X2 has $P_{\text{peak}} = 17.6$ GFLOP/s and $B = 15$ GB/s (Section 3, Figure 1a).

| Intensity $I$ | $B \cdot I$ | Attainable | Regime |
|---|---|---|---|
| 0.25 | 3.75 | 3.75 GFLOP/s | memory-bound |
| 0.5 | 7.5 | 7.50 GFLOP/s | memory-bound |
| 1 | 15 | 15.00 GFLOP/s | memory-bound |
| 2 | 30 | 17.60 GFLOP/s | compute-bound |
| 4 | 60 | 17.60 GFLOP/s | compute-bound |

The paper's own check matches: in Figure 1a, a kernel with an intensity of 2 is compute-bound and one with an intensity of 1 is memory-bound. Exercise 1 asks you to implement this function and the regime test.

### Operational intensity counts DRAM bytes

**Intuition.** The x-axis only means something if bytes are counted in a place that a program can change. The paper counts bytes that reach main memory, after the caches have absorbed what they can. A kernel that reuses data from cache has a higher intensity than one that streams everything from DRAM, even when both do the same arithmetic.

**Math.** Section 3 defines the quantity and says why it differs from two older terms. "Arithmetic intensity" and "machine balance" count traffic between the processor and the cache. Operational intensity counts traffic between the caches and DRAM, so cache optimizations show up in the model. The paper also uses "operational" instead of "arithmetic" so that the model can cover kernels that do no arithmetic (Section 7). The course pages and the concept list on this site say "arithmetic intensity". For this paper the two differ in where the bytes are counted, and the page uses the paper's term. In symbols:

$$
I = \frac{\text{FLOPs}}{\text{bytes moved between the caches and DRAM}}
$$

The paper counts the stencil kernel this way. Each point update does 8 FLOPs and, on a write-allocate machine, moves 24 bytes of compulsory traffic, so $I = 8/24 \approx 0.33$ (Section 6.3.3). Section 7 applies the same counting to the FFT transpose, which does no floating-point work at all. It counts one exchange, a 16-byte read plus a 16-byte write, as the unit of work, so $I = 1/32$.

**Tiny example, computed by `scripts/kernel_intensities.py`.** The paper's kernels are scientific codes. The same counting works for the three kernels at the heart of a Transformer. The page's counting rules are explicit, and they are the page's, not the paper's: only compulsory traffic, 2 FLOPs per multiply-add, 5 FLOPs per softmax element, and 8 FLOPs per LayerNorm element. Other conventions change the constants and keep the ordering.

| Kernel (FP32, 4 bytes per element) | FLOPs | Bytes | $I$ (FLOPs/byte) |
|---|---|---|---|
| Matmul, $4096^3$ | $1.374 \times 10^{11}$ | $2.013 \times 10^{8}$ | 682.67 |
| Matmul, $512^3$ | $2.684 \times 10^{8}$ | $3.146 \times 10^{6}$ | 85.33 |
| Softmax, $4096 \times 4096$ | $8.389 \times 10^{7}$ | $1.342 \times 10^{8}$ | 0.625 |
| LayerNorm, $4096 \times 4096$ | $1.342 \times 10^{8}$ | $1.343 \times 10^{8}$ | 1.000 |

A square matmul has $I = n/6$ in FP32: $2n^3$ FLOPs over $12n^2$ bytes. Intensity grows with the problem size, as the paper says for dense matrix and FFT problems (Section 5). Softmax and LayerNorm do a fixed amount of work per element, so no size makes them compute-heavy. Exercise 2 asks you to implement these counts.

### The ridge point

**Intuition.** The corner of the roof says how much work a kernel must do per byte before the machine's compute becomes the limit. A ridge point far to the right means only very dense kernels can reach peak. A ridge point near zero means almost anything can.

**Math.** The ridge point is where the two lines meet. Setting $B \cdot I^{*} = P_{\text{peak}}$ gives

$$
I^{*} = \frac{P_{\text{peak}}}{B}
$$

Section 3 describes the ridge point as the smallest operational intensity at which a kernel can reach peak performance. Section 6.3.5 links it to programming effort. The machine with the lowest ridge point, the Sun T2+, was the easiest to program. The one with the highest, the Intel Xeon, had the lowest unoptimized performance.

**Tiny example, computed by `scripts/roofline_example.py` and `scripts/gpu_rooflines.py`.**

- The X2 ridge is $17.6 / 15 = 1.17$ FLOPs per byte. The paper's text rounds it to 1.0 (Section 3).
- The Opteron X4 has more than four times the peak (73.6 GFLOP/s, a ratio of 4.18) on the same sockets. With 16.6 GB/s from Table 1, its ridge is 4.43, which the paper reports as 4.4. A kernel at an intensity of 1 gains only 16.6 / 15 = 1.11 times on the X4, and one at 4 gains 3.77 times (66.4 versus 17.6 GFLOP/s).

The same idea carries over to GPUs, with numbers that are not from the paper. They come from NVIDIA's data sheets for the A100 80 GB SXM (312 TFLOP/s dense BF16, 2,039 GB/s) and the H100 SXM (989 TFLOP/s dense BF16, 3,350 GB/s). The ridge points are 153.0 and 295.2 FLOPs per byte. The H100 raised peak compute 3.17 times but bandwidth only 1.64 times, so its ridge moved right by 1.93 times. A BF16 $512^3$ matmul at $I = 170.7$ is compute-bound on the A100 and memory-bound on the H100, where it reaches 57.8% of peak. Softmax at $I = 1.25$ reaches 0.8% of the A100's peak and 0.4% of the H100's, which is why fused kernels keep intermediate values on chip.

Exercise 3 places a list of kernels on a roofline and derives the run time from it.

### Ceilings: which optimization comes next

**Intuition.** The roof is the best case. Real code usually sits below it, and each missing optimization is a lower ceiling. To rise through a ceiling, you apply the optimization it names.

**Math.** Section 4 adds ceilings to the roofline for the Opteron X2. Two compute ceilings (Figure 2a) are 8.8 GFLOP/s without a balanced mix of multiplies and adds, and 2.2 GFLOP/s if ILP and SIMD are also missing. Three bandwidth ceilings (Figure 2b) are 11 GB/s without software prefetching, 4.8 GB/s without memory affinity as well, and 2.7 GB/s with only unit-stride access. Each ceiling is a lower roof with its own peak or bandwidth, and the same formula applies to it. Ceilings are ordered from the bottom, and the paper says that to break through one you must already have broken through all the ones below it (Section 4). The gap between a ceiling and the next one up is the potential reward for the optimization.

**Tiny example, computed by `scripts/roofline_example.py`.** At $I = 0.5$ on the X2:

| Ceiling | Bandwidth | Attainable |
|---|---|---|
| Roofline | 15 GB/s | 7.50 GFLOP/s |
| No software prefetching | 11 GB/s | 5.50 GFLOP/s |
| No memory affinity | 4.8 GB/s | 2.40 GFLOP/s |
| Unit stride only | 2.7 GB/s | 1.35 GFLOP/s |

Software prefetching is worth 2 GFLOP/s here, and memory affinity is worth 3.1. At $I = 4$, the bandwidth ceilings don't matter, because all of them are above the compute ceiling. The 2.2 GFLOP/s SIMD-and-ILP ceiling caps the kernel, and the next step is computational.

Table 4 shows the model in use. SpMV on the Opteron X4 reaches 4.2 GFLOP/s at an intensity of 0.25, which is 16.8 GB/s of traffic. That is 95% of the 17.6 GB/s Stream ceiling the paper lists for that row. The kernel is at the memory roof. A change to the arithmetic can't help it.

### Move right: raise operational intensity

**Intuition.** Ceilings say how to climb. The other direction is sideways. A kernel can move right if it sends fewer bytes to DRAM, and that also shifts it to a more favorable region of the plot.

**Math.** Section 5 ties the 3Cs cache model to the roofline. Compulsory misses set the minimum traffic and so the highest possible intensity. Conflict and capacity misses lower it. Padding arrays to cut conflict misses and using no-allocate stores both raise $I$. The paper's advice is to improve operational intensity before other optimizations. Section 6.3.2 gives a number: no-allocate stores raise LBMHD from 0.70 to 1.07.

Tiling a matmul is the standard example, though the paper doesn't use it. Suppose an $n \times n$ matmul computes $T \times T$ tiles of $C$ and each tile stays on chip. DRAM traffic is $b(2n^3/T + n^2)$ bytes for $b$ bytes per element. The intensity is

$$
I(T) = \frac{2n^3}{b\,(2n^3/T + n^2)}
$$

This formula is the page's model and assumes the tile fits on chip. It is not from the paper.

**Tiny example, computed by `scripts/kernel_intensities.py`.** For $n = 4096$ and BF16 ($b = 2$), $I$ is 0.50 at $T = 1$, 7.98 at $T = 16$, 31.75 at $T = 64$, 63.02 at $T = 128$, and 1365.33 at $T = n$. Tiling by 128 raises the intensity 126 times, with no change in the FLOP count. For the H100's ridge of 295.2, the smallest tile that reaches it at $n = 8192$ is 613, computed by the last lines of `scripts/kernel_intensities.py`. That tile is too big for a real kernel, which is why real GPU kernels also lean on registers and on tensor-core formats with smaller $b$.

<!-- exercises -->

## Pitfalls

- **Counting bytes in the wrong place.** If you count loads and stores between the processor and the L1 cache, a kernel that fits in cache looks as intensive as one that doesn't. Count what crosses the DRAM boundary. A profiler counter for DRAM bytes gives the real value. A count from the source code gives the compulsory minimum.
- **Mixing units.** The model's three quantities need consistent units. If peak is in TFLOP/s and bandwidth is in GB/s, then $I^{*}$ comes out 1,000 times too small. Exercise 3 uses base units for this reason.
- **Using the wrong peak.** The ridge point depends on which compute roof applies. On an H100 the BF16 tensor-core peak gives a ridge of 295.2 FLOPs per byte, and the FP32 CUDA-core peak gives 20.0. A kernel that is compute-bound against one roof can be memory-bound against the other.
- **Using the spec bandwidth.** The paper uses measured sustainable bandwidth, not pin bandwidth. It measures the steady-state bandwidth potential with microbenchmarks (Section 3). A data sheet number overstates the roof.
- **Reading the roof as a prediction.** It is an upper bound. A kernel can sit well below it for reasons the model doesn't capture. The paper says the model need not be perfect, and that is why the ceilings exist (Section 4).
- **Treating the intensity as fixed.** It changes with problem size and with optimization. Matmul and FFT intensities grow with size, and cache blocking raises it (Section 5). The x-position of a kernel is a property of one implementation.
- **Assuming bigger caches always help.** The paper's fallacies list says doubling the cache won't raise intensity for kernels that already reach compulsory traffic (Section 7).
- **Forgetting latency.** The model has no latency term. Low-bandwidth ceilings, such as those without software prefetching, stand in for the failure to hide it (Section 7). A kernel with too little parallelism can sit below the bandwidth roof, and the roofline alone won't say why.

## Further reading

- Williams, Waterman, and Patterson, [Roofline: An Insightful Visual Performance Model for Floating-Point Programs and Multicore Architectures](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html), UCB/EECS-2008-134. This is the longer technical report that this page follows. It includes Appendix A on how to measure peak performance, bandwidth, and operational intensity.
- [Roofline model on Wikipedia](https://en.wikipedia.org/wiki/Roofline_model) for a short summary of the model.
