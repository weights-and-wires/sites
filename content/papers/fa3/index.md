---
id: "fa3"
title: "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision"
authors: "Shah et al."
year: 2024
venue: "NeurIPS 2024"
arxiv: "2407.08608"
url: "https://arxiv.org/abs/2407.08608"
track: gpu-systems
tier: extended
est_hours: 5
status: full
summary: "FlashAttention-2 reaches only 35% of an H100's peak because its loop is synchronous: matrix multiplies and softmax take turns, and it uses no FP8. FlashAttention-3 splits the work between a producer warp that loads data and consumer warpgroups that compute, and it overlaps one block's softmax with another block's matrix multiplies. It also runs the forward pass in FP8, using one scale per block and a random orthogonal rotation to limit the rounding error from outliers."
related_course: "gpu: Hopper kernels"
prereqs:
  - id: fa2
    reason: "FlashAttention-3 keeps the FlashAttention-2 loop structure and its unscaled output update, so you need the online softmax of that paper to read Algorithms 1 and 2."
  - id: fp8
    reason: "The FP8 forward pass uses the E4M3 format, so you need that paper's grid, rounding, and scaling to follow the error analysis in Section 3.3."
background: [hopper-tma-and-wgmma, warp-specialization]
course_links:
  - title: "FlashAttention"
    url: "https://gpu.ifkash.dev/c/16-flash-attention"
  - title: "Hopper GEMM"
    url: "https://gpu.ifkash.dev/c/26-hopper-gemm"
  - title: "Pipelining and asynchronous copies"
    url: "https://gpu.ifkash.dev/c/12-pipelining-and-async-copies"
  - title: "FP8, MXFP4, and NVFP4"
    url: "https://gpu.ifkash.dev/c/33-fp8-and-fp4"
self_check:
  - q: 'A query row has the state $m = 2$, $\ell = 1.1353$, $\tilde O = (1, 0.1353)$ after block 1. Block 2 has the scores $[3, 1]$ and the values $(1, 1)$ and $(2, 0)$. What are the rescale factor, the new $\ell$, and the new $\tilde O$?'
    a: 'The maximum grows from 2 to 3, so the factor is $e^{2-3} = 0.3679$. The new weights are $e^{[3,1]-3} = [1, 0.1353]$. Then $\ell = 0.3679 \cdot 1.1353 + 1 + 0.1353 = 1.5530$ and $\tilde O = 0.3679 \cdot (1, 0.1353) + 1 \cdot (1, 1) + 0.1353 \cdot (2, 0) = (1.6386, 1.0498)$. Dividing by $\ell$ gives $O = (1.0551, 0.6760)$.'
    see: fa2
  - q: 'A producer warpgroup fills a circular buffer with $s = 2$ stages and loads $T_c = 6$ key and value blocks. Which slot does block $j$ use, and for which $j$ does the producer have to wait?'
    a: 'Block $j$ uses slot $j \bmod s$, so the slots alternate 0, 1, 0, 1, 0, 1. The producer waits only for $j \ge s$, that is $j = 2, \dots, 5$, because the first $s$ iterations find empty slots (Section 3.1, Algorithm 1, lines 6 to 10).'
    see: warp-specialization
  - q: 'You quantize the row $[1, 2, 3, 4]$ to E4M3 with one scale per block. What is the scale, and what does the 3 become after you dequantize?'
    a: 'The scale is $\mathrm{amax} / 448 = 4 / 448$. The 3 maps to 336, which is a tie between 320 and 352, and ties go to the even mantissa, 320. Dequantized, it is $320 \cdot 4 / 448 = 2.8571$. The other three entries survive exactly.'
    see: fp8
---

## The problem

[FlashAttention-2](/fa2/) made attention much faster on the A100, and it left most of the H100 unused. The paper reports that FlashAttention-2 reaches 35% utilization on the H100, while an optimized matrix multiplication (GEMM) reaches 80-90% (Section 1). Attention does the same kind of arithmetic as a GEMM, so the shortfall comes from the algorithm's schedule, not from the work.

The paper names two reasons.

- **The loop is synchronous.** FlashAttention-2 computes $S = QK^\top$, then the softmax, then $PV$, and each step waits for the previous one. The H100 has separate hardware units for matrix multiplication (Tensor Cores), memory copies (the Tensor Memory Accelerator, TMA), and everything else, and a synchronous loop keeps most of them idle at any moment (Section 1).
- **It uses no FP8.** The H100's FP8 Tensor Cores run at twice the FP16 rate per streaming multiprocessor (SM), but plain FP8 attention has a large rounding error when the activations contain outliers (Sections 1 and 3.3).

Here is a concrete failing case. The H100 SXM5 has 989 TFLOPs/s of FP16 matrix multiplication and only 3.9 TFLOPs/s of special functions such as the exponential that softmax needs (Section 3.1). For head dimension 128, each exponential comes with $4 \cdot 128 = 512$ matrix-multiply FLOPs. The matrix units are about 254 times faster, so the exponentials alone take about half as long as the matrix multiplies. If the two phases take turns, the Tensor Cores are idle for a third of the time before you count any other overhead. `scripts/pingpong_schedule.py` prints these numbers and the toy schedule that follows from them.

The paper's answer has three parts: a producer-consumer split with warp specialization, an overlap of softmax with GEMMs, and FP8 with block quantization and incoherent processing (Section 1). It reports up to 740 TFLOPs/s in FP16, which is 75% of the H100's peak, and close to 1.2 PFLOPs/s in FP8 (Abstract).

## Key ideas

The paper's notation follows FlashAttention-2. $Q_i \in \mathbb{R}^{B_r \times d}$ is a block of queries, $K_j, V_j \in \mathbb{R}^{B_c \times d}$ are blocks of keys and values, $T_c = \lceil N / B_c \rceil$ is the number of key blocks, and $m_i$ and $\ell_i$ are the running row maximum and row sum. $\tilde P$ is the unnormalized softmax weights, and $L_i = m_i + \log \ell_i$ is the logsumexp saved for the backward pass. The paper omits the softmax scale in its algorithms; the exercises take it as `tau`. The paper's displays have no equation numbers, so this page cites section numbers and algorithm lines.

### The loop that the kernel runs

**Intuition.** Each consumer iteration multiplies a block of queries by one block of keys, updates a running softmax, and multiplies the weights by one block of values. The softmax never sees a full row, so it keeps a running maximum and rescales what it has already accumulated when the maximum grows. This is the FlashAttention-2 forward pass, now the body of a warpgroup's mainloop (Section 2.3).

**Math.** The consumer mainloop of Algorithm 1 (Section 3.1, lines 17 to 21) runs for each key block $j$:

$$
S_i^{(j)} = Q_i K_j^\top, \qquad m_i = \max\!\big(m_i^{\text{old}}, \operatorname{rowmax}(S_i^{(j)})\big), \qquad \tilde P_i^{(j)} = \exp\!\big(S_i^{(j)} - m_i\big)
$$

$$
\ell_i = e^{m_i^{\text{old}} - m_i}\, \ell_i + \operatorname{rowsum}\!\big(\tilde P_i^{(j)}\big), \qquad O_i = \operatorname{diag}\!\big(e^{m_i^{\text{old}} - m_i}\big)\, O_i + \tilde P_i^{(j)} V_j
$$

After the last block, line 24 computes $O_i = \operatorname{diag}(\ell_i)^{-1} O_i$ and $L_i = m_i + \log \ell_i$. The paper prints the rescale factor in line 21 with an inverse, $\operatorname{diag}(\exp(m_i^{\text{old}} - m_i))^{-1}$. The exponent is at most 0, so the factor is at most 1 and must shrink the old output; this page treats the inverse as a typo, and exercise 1 checks the result against naive attention.

**Tiny example, computed by `scripts/online_softmax_example.py`.** Take one query row with the scores $[2, 0, 3, 1]$, the values $(1,0), (0,1), (1,1), (2,0)$, and two key blocks of two:

| Block | $m$ | $\ell$ | rescale factor | $\tilde O$ |
|---|---|---|---|---|
| 1 | 2 | 1.1353 | 0 (first block) | (1.0000, 0.1353) |
| 2 | 3 | 1.5530 | 0.3679 | (1.6386, 1.0498) |

Check block 2 by hand. The maximum grows from 2 to 3, so the old state shrinks by $e^{-1} = 0.3679$, and the new weights are $e^{[3,1]-3} = [1, 0.1353]$. Then $\ell = 0.3679 \cdot 1.1353 + 1.1353 = 1.5530$. Dividing gives $O = (1.0551, 0.6760)$ and $L = 3 + \log 1.5530 = 3.4402$, the same values as the naive softmax. The script also shows that the tiled result matches naive attention to about $10^{-16}$ for block sizes from 1 to 128, and that it stays finite for scores of about 3000, where the naive softmax without the running maximum overflows.

Exercise 1 asks you to implement this loop.

### Producer and consumer warps on a circular buffer

**Intuition.** On Hopper, one thread issues a TMA copy and a warpgroup issues an asynchronous matrix multiply (WGMMA). A thread block no longer needs every thread to do both jobs. Warp specialization gives the block two roles: a *producer* warpgroup that only issues loads, and *consumer* warpgroups that only compute. A circular buffer in shared memory (SMEM) with $s$ stages connects them, so the producer loads block $j + 1$ while the consumers work on block $j$ (Section 3.1).

**Math.** Algorithm 1 (Section 3.1) gives the protocol. The producer loads $Q_i$ once. Then for each $j$ it waits until stage $j \bmod s$ is released, issues the loads of $K_j$ and $V_j$ into that stage, and signals the consumers (lines 6 to 10). The consumers wait for the data, compute, and release the stage (lines 15 to 22). The paper uses `setmaxnreg` to move registers from the producer, which needs few, to the consumers, which hold the accumulators (Sections 2.2 and 3.1, lines 3 and 12).

**Tiny example, computed by `scripts/circular_buffer.py`.** Take $s = 2$ stages, $T_c = 6$ blocks, a load time of 1 unit, and a consumer step of 3 units. The slots go 0, 1, 0, 1, 0, 1. The producer waits only for $j = 2, \dots, 5$, because the first $s$ iterations find empty slots. The consumer is busy for 18 of 19 units, and its only idle time is the first load. This is a toy timing, not a measurement. It shows why the loads disappear from the critical path when the loads are faster than the compute.

The [Hopper GEMM](https://gpu.ifkash.dev/c/26-hopper-gemm) lesson builds the same producer-consumer pair for a GEMM, and the [pipelining](https://gpu.ifkash.dev/c/12-pipelining-and-async-copies) lesson explains the barriers.

### Ping-pong scheduling hides the softmax

**Intuition.** The exponential runs on a different unit from the Tensor Core, so the two can work at the same time, provided they work on different data. Use two consumer warpgroups. While warpgroup 1 runs its softmax, warpgroup 2 runs its GEMMs, and then they swap, which gives the name "pingpong" (Section 3.1).

**Math.** The paper forces the order with `bar.sync` barriers: the GEMMs of warpgroup 1 (the $PV$ product of one iteration and the $QK^\top$ product of the next) issue before those of warpgroup 2. Then the softmax of warpgroup 1 overlaps the GEMMs of warpgroup 2. The cost model has two parts. Exponentials take $256 / 512 = 50\%$ of the matrix-multiply time (512 times fewer operations, 256 times lower throughput) for head dimension 128 in FP16, and the ratio doubles in FP8 because the matrix rate doubles and the exponential rate does not (Section 3.1). The paper adds that the schedule is "not as clean as depicted" in practice, and it reports an improvement from 570 to 620-640 TFLOPs/s for FP16 forward with head dimension 128 and sequence length 8192.

**Tiny example, computed by `scripts/pingpong_schedule.py`.** The toy model has a Tensor Core queue and a special-function queue, each running one task at a time in issue order. A task waits for its dependencies. Each of $n = 16$ blocks has a GEMM of duration $g$ and a softmax of duration $s$. The table gives the Tensor Core utilization, which is its busy time divided by the makespan:

| Setting | Serial (Algorithm 1) | 2-stage (Algorithm 2) | Ping-pong |
|---|---|---|---|
| FP16-like: $g = 2$, $s = 1$ | 66.7% | 100.0% | 98.5% |
| FP8-like: $g = 1$, $s = 1$ | 50.0% | 94.1% | 97.0% |

The serial schedule has the closed form $g / (g + s)$. Ping-pong has the closed form $2ng / (2ng + s)$ when $g \ge s$, because only the last softmax is exposed: $64 / 65 = 98.5\%$ and $32 / 33 = 97.0\%$. These numbers are an upper bound from a model that ignores the epilogue, the loads, and the barriers. The paper's 75% utilization is a measurement that includes all of them.

Exercise 2 asks you to write the simulator and the ping-pong schedule.

### Two-stage pipelining inside one warpgroup

**Intuition.** Even one warpgroup can overlap softmax with GEMM. The softmax of block $j + 1$ needs only $S^{(j+1)}$, and the product $\tilde P^{(j)} V_j$ needs only block $j$'s weights. So issue the next $QK^\top$ first, then the $PV$ product of the current block, both without waiting, and run the softmax of the next block while the $PV$ product runs (Section 3.2).

**Math.** Algorithm 2 (Section 3.2) keeps $S_{\text{cur}}$ and $S_{\text{next}}$ in registers. In the mainloop it issues $S_{\text{next}} = Q_i K_j^\top$ (line 9) and $O_i = O_i + \tilde P_{\text{cur}} V_{j-1}$ (line 11), both with commit-but-do-not-wait. It then waits for the first product, computes $m_i$, $\tilde P_{\text{next}}$ and $\ell_i$ (lines 12 and 13), waits for the second, and rescales $O_i$ (line 14). The cost is registers: the extra $S_{\text{next}}$ takes $B_r \times B_c \times \text{sizeof(float)}$ bytes per thread block (Section 3.2). A 3-stage variant overlaps more and costs more registers (Appendix B.3).

**Tiny example, computed by `scripts/pingpong_schedule.py`.** For $B_r = B_c = 128$, the extra tile is $128 \cdot 128 \cdot 4 = 65{,}536$ bytes, or 64 KiB, which competes with larger tiles for the register file. In the toy model, the 2-stage schedule reaches 100% in the FP16-like case. In the paper's ablation (Section 4.2, Table 2), removing the GEMM-softmax pipelining lowers the speed from 661 to 582 TFLOPs/s (a factor of 1.136), and removing warp specialization lowers it to 570 (a factor of 1.160).

### FP8: layout, block quantization, and incoherent processing

**Intuition.** FP8 doubles the Tensor Core rate, and it costs accuracy for two reasons. First, E4M3 has 3 mantissa bits, so every value carries a relative error of a few percent. Second, LLM activations have outliers that are much larger than the rest, and a single scale for a tensor makes the rest of the values small and coarse (Section 3.3). The paper adds a layout problem: FP8 WGMMA accepts only $k$-major operands, which forces an in-kernel transpose of $V$ and a register permutation of the accumulator (Section 3.3). That part is a kernel detail, and the exercises skip it.

**Math.** The paper uses two techniques to reduce the error (Section 3.3):

1. **Block quantization** keeps one scale per block of $B_r \times d$ or $B_c \times d$ values of $Q$, $K$, and $V$, instead of one per tensor. The attention algorithm already works block by block, so it can fold each block's scale into $S$ at no extra cost.
2. **Incoherent processing** multiplies $Q$ and $K$ by a random orthogonal matrix $M$ before quantizing. Because $MM^\top = I$, $(QM)(KM)^\top = QK^\top$, so the attention output does not change before rounding. Each entry of $QM$ is a random sum of entries of $Q$, so an outlier spreads over the row. The paper takes $M$ as a product of random $\pm 1$ diagonal matrices and a Hadamard matrix, which costs $O(d \log d)$ per row instead of $O(d^2)$.

Table 3 reports the RMSE against an FP64 reference on inputs with $N(0,1) + N(0,100) \cdot \text{Bernoulli}(0.001)$ entries (Section 4.3). The baseline FP8 attention has an RMSE of $2.4 \times 10^{-2}$ and FlashAttention-3 in FP8 has $9.1 \times 10^{-3}$, which is the paper's 2.6 times lower error: $2.4 \times 10^{-2} / 9.1 \times 10^{-3} = 2.64$. The ablation rows say that removing block quantization gives $9.3 \times 10^{-3}$ and that removing incoherent processing gives $2.4 \times 10^{-2}$, the same as the baseline. So in the paper's measurement, the rotation carries almost all of the gain.

**Tiny example, computed by `scripts/fp8_incoherent.py`.** Take the block $[1, 2, 3, 4]$. The scale is $4 / 448$, the 3 maps to 336, and the tie between 320 and 352 rounds to 320, so the dequantized block is $[1, 2, 2.8571, 4]$. Then the script measures the relative RMSE of $S = QK^\top$ on a toy with $N = 512$, $d = 64$, and blocks of 64 rows. The quantizer is a NumPy E4M3 rounding that matches the [FP8 paper](/fp8/)'s format:

| Inputs | Per tensor | Per block | Rotation only | Both | Per tensor / both |
|---|---|---|---|---|---|
| Paper-like isolated outliers (0.1% of entries, sd 10) | 3.63e-2 | 3.66e-2 | 3.74e-2 | 3.72e-2 | 0.98x |
| 2 outlier channels, 30x | 3.71e-2 | 3.86e-2 | 7.11e-3 | 7.02e-3 | 5.29x |
| 2 outlier channels, 100x | 3.72e-2 | 3.49e-2 | 9.15e-3 | 9.14e-3 | 4.07x |

This toy does not reproduce the paper's 2.6 times, and the reason is useful. E4M3 is a floating-point format, so each entry's error is proportional to its own size. The error of one score is dominated by the products where both $q$ and $k$ are large, which happens when a whole channel is an outlier in both. Rotation spreads that channel over all 64 coordinates and removes the large products. Isolated outliers rarely line up in $q$ and $k$, so the rotation has nothing to remove. Real LLM outliers are channel-structured (the paper cites references 20 and 54), so the channel rows are the closer analogue, but they are still a toy and not the paper's distribution. Block quantization also changes little here, which agrees with the paper's ablation. It matters when the range between blocks exceeds what E4M3's exponent covers, which is the case in Exercise 3's test.

Exercises 3 and 4 ask you to write the block quantizer and the rotation.

### What the paper reports

On an H100 SXM5, FlashAttention-3 in FP16 is 1.5-2.0 times faster than FlashAttention-2 in the forward pass and 1.5-1.75 times faster in the backward pass. It reaches up to 740 TFLOPs/s, which is 75% of the theoretical maximum (Sections 1 and 4.1). In FP8 it reaches close to 1.2 PFLOPs/s. For sequences of 1k and longer, the FP16 kernel is faster than cuDNN. The FP8 kernel is ahead of it for head dimension 64 and, for large sequence lengths, at par or behind for head dimensions 128 and 256 (Section 1, footnote 2; Section 4.1). The paper counts the forward-pass FLOPs as $4 \cdot \text{seqlen}^2 \cdot \text{head dimension} \cdot \text{number of heads}$ and halves that with a causal mask. FP16 FlashAttention-3 has the same numerical error as FlashAttention-2, and both are 1.7 times lower than standard attention, because the softmax intermediates stay in FP32 (Table 3, Section 4.3).

<!-- exercises -->

## Pitfalls

- **Reading the inverse in line 21 literally.** Algorithm 1 prints $\operatorname{diag}(\exp(m^{\text{old}} - m))^{-1}$. The factor must be $\exp(m^{\text{old}} - m)$, at most 1, so that the old output shrinks when the maximum grows. Exercise 1's tests use a maximum that grows across blocks to catch the literal reading.
- **Rescaling $\ell$ but not $O$, or the reverse.** Both carry the same factor. The tests fail when either is missing.
- **Expecting ping-pong without the order barrier.** The overlap exists only because the Tensor Core queue alternates between warpgroups. Exercise 2 builds the alternating order and the per-warpgroup dependencies. A schedule with the wrong order or the wrong dependencies gives a different, and often better-looking, makespan, so the tests check the structure as well as the totals.
- **Treating the toy utilization as a prediction.** The model reaches 98.5% because it ignores loads, barriers, the epilogue, and registers. The paper measures 75% on the real kernel and says that its schedule is not as clean as the figure (Section 3.1).
- **Forgetting the register bill of the 2-stage pipeline.** The extra $S_{\text{next}}$ tile is $B_r \times B_c$ floats, and it competes with the larger tiles that a high-utilization kernel also wants (Section 3.2). The speedup in Table 2 is the net effect.
- **Expecting rotation to help every input.** In the toy, rotation cuts the error about 4-5 times on outlier channels and by nothing on isolated Gaussian outliers. Exercise 4 tests both cases, because the benefit depends on the structure of the outliers.
- **Rotating only one of $Q$ and $K$.** $(QM)K^\top \ne QK^\top$. Both operands need the same $M$, and the tests catch a one-sided rotation.
- **Comparing the toy's error to Table 3.** The toy measures the relative error of the scores in NumPy rounding. Table 3 measures the RMSE of the attention output against FP64 on a different input distribution. Only the direction of the effect carries over.
- **Forgetting that FP8 WGMMA needs $k$-major operands.** In the real kernel this forces a transpose of $V$ and a permutation of the accumulator (Section 3.3). The NumPy exercises hide this entirely.
- **Reading "up to" as typical.** The 740 TFLOPs/s and 1.2 PFLOPs/s figures are peaks over the paper's benchmark settings, not averages (Section 4.1).

## Further reading

- [flash-attention](https://github.com/Dao-AILab/flash-attention) is the official code. The Hopper kernels are in the `hopper` directory.
- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://tridao.me/blog/2024/flash3/) is Tri Dao's post about the paper.
- [FlashAttention-3 on the PyTorch blog](https://pytorch.org/blog/flashattention-3/) is a shorter summary of the same results.
- [CUTLASS tutorial: WGMMA on Hopper](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/) explains the asynchronous matrix multiply that the paper builds on.
- [GPU MODE lecture 36](https://github.com/gpu-mode/lectures/tree/main/lecture_036) covers CUTLASS and FlashAttention-3.
- [FlashAttention-2](/fa2/) and [FP8 Formats for Deep Learning](/fp8/) are the prerequisite pages on this site, and [Online normalizer calculation for softmax](/online-softmax/) derives the running maximum.
