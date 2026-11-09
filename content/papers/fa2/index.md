---
id: "fa2"
title: "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
authors: "Dao et al."
year: 2023
venue: "ICLR 2024"
arxiv: "2307.08691"
url: "https://arxiv.org/abs/2307.08691"
track: gpu-systems
tier: core
est_hours: 4
status: full
summary: "FlashAttention computes exact attention without writing the score matrix to GPU memory, but it reaches only a quarter to a half of the hardware's peak matrix-multiply rate. FlashAttention-2 removes most of the shortfall in three steps: it rearranges the online softmax so less work runs outside the tensor cores, it launches one thread block per block of queries so that long sequences with small batches fill the GPU, and it splits the work inside a thread block by query rows so warps stop exchanging partial results through shared memory. The output is the same as before, and the paper reports about twice the speed of FlashAttention on an A100."
related_course: "gpu: attention kernels"
prereqs:
  - id: flashattention
    reason: "FlashAttention-2 keeps the tiled attention with an online softmax and changes how its loops, statistics, and per-block work are organized, so you need the Q, K, V block notation and Algorithm 1 of that paper."
background: [warps, occupancy, non-matmul-flops]
course_links:
  - title: "FlashAttention"
    url: "https://gpu.ifkash.dev/c/16-flash-attention"
  - title: "FlashAttention"
    url: "https://qwen.ifkash.dev/c/14-flash-attention"
self_check:
  - q: 'A query row has the scores $[1, 0, 2, 1]$ against four value rows $(1, 2)$, $(3, 4)$, $(5, 6)$, $(7, 8)$, split into two blocks of two. After block 1 the FlashAttention-2 state is $m = 1$, $\ell = 1.3679$, and the unscaled output is $\tilde O = (2.1036, 3.4715)$. Run block 2 and give the final $O$ and the logsumexp $L$.'
    a: 'The block maximum is 2, so $m^{(2)} = 2$ and the rescale factor is $e^{1-2} = 0.3679$. The new weights are $e^{[2,1]-2} = [1, 0.3679]$. Then $\ell^{(2)} = 0.3679 \cdot 1.3679 + 1.3679 = 1.8711$ and $\tilde O^{(2)} = 0.3679 \cdot (2.1036, 3.4715) + 1 \cdot (5, 6) + 0.3679 \cdot (7, 8) = (8.3490, 10.2201)$. One division at the end gives $O = \tilde O^{(2)} / \ell^{(2)} = (4.4621, 5.4621)$, and $L = m + \log \ell = 2 + 0.6265 = 2.6265$.'
    see: flashattention
  - q: 'A batch holds 16,384 tokens as one sequence ($N = 16384$) with 16 heads, on an A100 with 108 SMs. How many thread blocks does FlashAttention launch for the forward pass, and how many does FlashAttention-2 launch with $B_r = 128$? Assume one block per SM at a time.'
    a: 'FlashAttention launches one block per (batch, head) pair, so $1 \cdot 16 = 16$ blocks. That fills 16 of 108 SMs, or 14.8% of them. FlashAttention-2 also launches one per row block, $\lceil 16384 / 128 \rceil = 128$ per head, so $16 \cdot 128 = 2048$ blocks. That takes 19 waves and keeps 99.8% of SM time busy, because the last wave has 104 of 108 SMs busy.'
    see: occupancy
  - q: 'On an A100, tensor cores run 312 TFLOPs/s of FP16 matmul and the FP32 units run 19.5 TFLOPs/s. A tile of attention has $F$ matmul FLOPs and $n$ non-matmul operations. Why can a small $n$ still limit the kernel?'
    a: 'Each non-matmul operation costs about $312 / 19.5 = 16$ times as much as a matmul FLOP, so the time share of the non-matmul work is $16n / (16n + F)$, not $n / (n + F)$. In the counting model of this page, at $N = 4096$ and $d = 64$ the non-matmul work is 30.9% of the time in FlashAttention and 24.1% in FlashAttention-2, even though it is under 3% of the operations.'
    see: non-matmul-flops
---

## The problem

[FlashAttention](/flashattention/) fixed the memory traffic of attention: the score matrix never goes to HBM, and the result is exact. It did not make attention as fast as the matrix multiplications inside it. The paper measures FlashAttention at 30-50% of the A100's theoretical FLOPs/s in the forward pass and 25-35% in the backward pass. An optimized matrix multiply (GEMM) reaches 80-90% (Section 1). Attention does the same kind of arithmetic as a GEMM, so about half of the peak is being lost somewhere.

Profiling led the paper to blame "suboptimal work partitioning between different thread blocks and warps" (Section 1). It names three causes and gives one fix for each:

1. **Work that does not run on tensor cores.** The online softmax rescales the output on every block. An A100 does 312 TFLOPs/s of FP16 or BF16 matmul but only 19.5 TFLOPs/s of FP32 non-matmul work, so each non-matmul FLOP "is 16x more expensive than a matmul FLOP" (Section 3.1).
2. **Idle multiprocessors.** FlashAttention uses one thread block per (batch, head). Long sequences usually come with small batches or few heads, so a launch can have fewer blocks than the A100's 108 streaming multiprocessors (SMs) (Section 3.2).
3. **Warps that talk to each other.** Inside a thread block, FlashAttention splits $K$ and $V$ across four warps, and the warps must write partial results to shared memory and add them (Section 3.3).

Here is a concrete failing case. Take a 16,384-token sequence with 16 heads, which is the shape of the paper's long-sequence benchmark (Section 4.1). FlashAttention launches $1 \cdot 16 = 16$ thread blocks, so at most 16 of the 108 SMs have work. No amount of tuning inside the kernel helps the other 92.

**Notation.** This page uses the paper's, which follows [FlashAttention](/flashattention/): $N$ is the sequence length, $d$ is the head dimension, $B_r$ and $B_c$ are the row and column block sizes, $T_r = \lceil N / B_r \rceil$ and $T_c = \lceil N / B_c \rceil$ are the block counts, and $m_i^{(j)}$ and $\ell_i^{(j)}$ are the running maximum and sum for row block $i$ after column block $j$. The paper omits the softmax scale (Section 2.2, footnote 2). The exercises here take it as `tau`, as the FlashAttention page does. Two differences from the paper: the page writes $\tilde O_i^{(j)}$ for the unscaled output that Algorithm 1 calls $\mathbf{O}_i^{(j)}$, to keep it apart from the final $O_i$, and the page writes the rescale factor as multiplication by $e^{m^{(j-1)} - m^{(j)}}$. The paper prints that factor as $\operatorname{diag}(e^{m^{(j-1)} - m^{(j)}})^{-1}$ in Algorithm 1, line 10, and in the two-block derivation of Section 3.1.1. With that exponent, the inverse reads as a typo, because the old output must shrink when the maximum grows.

The paper's displays have no equation numbers, so this page cites them by section and by algorithm line.

## Key ideas

### Fewer non-matmul FLOPs

**Intuition.** FlashAttention's update keeps the output $O_i$ normalized after every block. That costs a multiplication by the old $\ell$, a rescale, and a division by the new $\ell$ for each block, all on a $B_r \times d$ matrix. FlashAttention-2 lets the output stay unnormalized, rescales it by one factor per block, and divides once at the end.

**Math.** The paper makes two changes (Section 3.1.1). First, it keeps an "un-scaled" output $\tilde O$ and divides by $\ell$ only after the last block. Second, it saves only the logsumexp $L = m + \log \ell$ for the backward pass, not both $m$ and $\ell$. For one row block, Algorithm 1 (lines 9-13) runs, for $j = 1, \dots, T_c$:

$$
S_i^{(j)} = Q_i K_j^\top, \qquad m_i^{(j)} = \max\!\left(m_i^{(j-1)}, \operatorname{rowmax}(S_i^{(j)})\right), \qquad \tilde P_i^{(j)} = \exp\!\left(S_i^{(j)} - m_i^{(j)}\right)
$$

$$
\ell_i^{(j)} = e^{m_i^{(j-1)} - m_i^{(j)}} \ell_i^{(j-1)} + \operatorname{rowsum}\!\left(\tilde P_i^{(j)}\right), \qquad
\tilde O_i^{(j)} = \operatorname{diag}\!\left(e^{m_i^{(j-1)} - m_i^{(j)}}\right) \tilde O_i^{(j-1)} + \tilde P_i^{(j)} V_j
$$

and after the last block (lines 12-13)

$$
O_i = \operatorname{diag}\!\left(\ell_i^{(T_c)}\right)^{-1} \tilde O_i^{(T_c)}, \qquad L_i = m_i^{(T_c)} + \log \ell_i^{(T_c)}.
$$

The inner loop touches the output with one scaling per block, and the division happens $T_r$ times in total, not $T_r T_c$ times. The paper states that the result is unchanged, with $O(N^2 d)$ FLOPs and $O(N)$ extra memory for $L$ (Section 3.1.1, "Correctness, runtime, and memory requirement").

**Tiny example, computed by `scripts/fa2_trace.py`.** Use the four-key example from the FlashAttention page: $q = (1, 0)$, scores $[1, 0, 2, 1]$, value rows $(1, 2), (3, 4), (5, 6), (7, 8)$, and $B_c = 2$. The script prints the state after each block:

| After block | $m$ | $\ell$ | $\tilde O$ (unscaled) | $\tilde O / \ell$ |
|---|---|---|---|---|
| 1 | 1 | 1.3679 | `[2.1036, 3.4715]` | `[1.5379, 2.5379]` |
| 2 | 2 | 1.8711 | `[8.3490, 10.2201]` | `[4.4621, 5.4621]` |

The last column matches the FlashAttention page, which divided at every block. FlashAttention-2 computes it only once. Check block 2 by hand: the factor is $e^{1-2} = 0.3679$, so $\tilde O^{(2)} = 0.3679 \cdot (2.1036, 3.4715) + (5, 6) + 0.3679 \cdot (7, 8) = (8.3490, 10.2201)$. The final $L = 2 + \log 1.8711 = 2.6265$, which equals the log of the sum of $e^{\text{score}}$ over all four scores. The script also confirms that the Q-outer loop and the K-and-V-outer loop agree to about $10^{-16}$ on random data.

**Counting the saving, computed by `scripts/nonmatmul_count.py`.** The paper does not count operations, so the page uses a model, which Exercise 2 states in full: each scalar operation costs 1, the shared $B_r \times B_c$ work (`rowmax`, subtract, `exp`, `rowsum`) costs $4 B_r B_c$, and the matmul accumulation is not counted. For $B_r = 128$, $B_c = 64$, and $d = 64$, one tile costs

| Version | Non-matmul operations per tile |
|---|---|
| FlashAttention | $4B_rB_c + 9B_r + 3B_rd = 58{,}496$ |
| FlashAttention-2 | $4B_rB_c + 5B_r + B_rd = 41{,}600$ |

while the two matmuls cost $4 B_r B_c d = 2{,}097{,}152$ FLOPs. The non-matmul work is under 3% of the operations. Weighted by the paper's factor of 16, it is a larger share of the time. At $N = 4096$ and $d = 64$ the model gives 30.9% of the time for FlashAttention and 24.1% for FlashAttention-2. At $d = 128$ it gives 24.1% and 16.0%. The model counts a `exp` as one operation, and a real GPU runs it more slowly, so these shares are an illustration of the effect and not a profile.

**Loop order and HBM traffic, computed by `scripts/fa2_trace.py`.** Algorithm 1 also swaps the loops: the outer loop runs over row blocks of $Q$ and the inner loop over blocks of $K$ and $V$. In the page's element count, for $N = 4096$, $d = 64$, and $B_r = B_c = 128$, FlashAttention moves 26,214,400 elements and FlashAttention-2 moves 17,305,600, a ratio of 1.51. FlashAttention-2 reads $Q$ once and writes $O$ and $L$ once. It re-reads $K$ and $V$ $T_r$ times. The paper does not claim this count. Its claim is about utilization, so treat the traffic as a side effect and not as the source of the speedup.

Exercise 1 asks you to write this forward pass, and Exercise 2 asks you to count the operations.

**Causal masking.** The block structure gives two more savings (Section 3.1.1, "Causal masking"). A tile whose column indices are all greater than its row indices is entirely masked, so the kernel skips it. A tile whose row indices are all at least its column indices needs no mask. With square blocks, only one tile per row block crosses the diagonal and needs the mask. The paper reports about 1.7-1.8 times the speed of the same attention without a mask. `scripts/causal_tiles.py` counts the tiles: for $N = 1024$ and $B_r = B_c = 128$ there are 64 tiles, of which 28 are skipped, 8 need the mask, and 28 are unmasked. Computing 36 of 64 tiles bounds the speedup at $64/36 = 1.78$, which is the paper's range. The bound approaches 2 as $N$ grows: 1.98 at $N = 16384$. With $B_r = 128$ and $B_c = 64$, each row block has two masked tiles, since the square-block case is the one the paper describes. Exercise 4 asks you to classify the tiles.

### Parallelize over the sequence length

**Intuition.** Each row block of the output depends only on its own queries and on all of $K$ and $V$. Two row blocks never need each other's results, so they can run on different SMs without communicating. FlashAttention left that parallelism unused.

**Math.** FlashAttention launches $\text{batch} \cdot \text{heads}$ thread blocks, one per head, and each runs on one SM (Section 3.2). The paper says that this is efficient when the number is large, "say $\geq 80$", because then it uses nearly all of the A100's 108 SMs. FlashAttention-2 adds a grid dimension (Section 3.2):

$$
\text{forward blocks} = \text{batch} \cdot \text{heads} \cdot T_r, \qquad \text{backward blocks} = \text{batch} \cdot \text{heads} \cdot T_c
$$

In the forward pass, the loop over row blocks is "embarrassingly parallel", so each thread block takes one row block of $Q$ and loops over all of $K$ and $V$. That is why Algorithm 1 puts the row-block loop outside. The paper credits Phil Tillet's Triton implementation with first suggesting the loop swap and the sequence-length parallelism (Section 3.2).

The backward pass is the other way around (Section 3.2, Figure 2). In Algorithm 2, different column blocks share only the update of $dQ_i$ (line 15): load $dQ_i$ from HBM, add $dS_i^{(j)} K_j$, and write it back. So each thread block takes one column block $j$, computes its own $dK_j$ and $dV_j$, and uses atomic adds to update $dQ$. The paper describes this in prose and does not give a proof.

**Tiny example, computed by `scripts/grid_occupancy.py`.** The script uses the paper's benchmark shape (Section 4.1): 16,384 tokens in total, so $\text{batch} = 16384 / N$, with 32 heads ($d = 64$) or 16 heads ($d = 128$). It assumes $B_r = 128$ and one block per SM at a time, which is a simplification, because a real SM can hold several blocks. The fraction is the busy share of SM-time over all waves:

| $N$ | batch | heads | FlashAttention blocks | utilization | FlashAttention-2 blocks | waves | utilization |
|---|---|---|---|---|---|---|---|
| 512 | 32 | 32 | 1024 | 94.8% | 4096 | 38 | 99.8% |
| 2048 | 8 | 32 | 256 | 79.0% | 4096 | 38 | 99.8% |
| 16384 | 1 | 32 | 32 | 29.6% | 4096 | 38 | 99.8% |
| 2048 | 8 | 16 | 128 | 59.3% | 2048 | 19 | 99.8% |
| 16384 | 1 | 16 | 16 | 14.8% | 2048 | 19 | 99.8% |

Check the last row by hand: $1 \cdot 16 = 16$ blocks over 108 SMs is $16/108 = 14.8\%$, and $16 \cdot 128 = 2048$ blocks take $\lceil 2048/108 \rceil = 19$ waves, with $2048 / (19 \cdot 108) = 99.8\%$. Short sequences already have many blocks, so FlashAttention-2 helps them less. The paper's claim is for the long-sequence regime (Section 3.2).

Exercise 3 asks you to compute these grids.

### Split the work between warps by query rows

**Intuition.** Inside a thread block, 4 or 8 warps share one $B_r \times B_c$ tile (Section 3.3). The question is which dimension of the tile to cut. If you cut the key dimension, every warp ends up with a partial sum for the same output rows, and someone has to add them. If you cut the query dimension, every warp owns different output rows, and there is nothing to add.

**Math.** FlashAttention uses a "split-K" scheme (Section 3.3, Figure 3a): it splits $K$ and $V$ across the 4 warps and keeps $Q$ visible to all of them. Each warp computes a slice of $QK^\top$ and multiplies it by its slice of $V$, which gives a partial $B_r \times d$ output per warp. The warps then write the partials to shared memory, synchronize, and add them. FlashAttention-2 splits $Q$ across the warps and keeps $K$ and $V$ visible to all of them (Figure 3b). Each warp computes a slice of $QK^\top$ and multiplies it by the whole $V_j$ to get its own rows of the output. "There is no need for communication between warps."

In matrix terms, split-K computes

$$
O = \sum_{w=1}^{W} P_{:,\,c_w}\, V_{c_w,\,:}
$$

where $c_w$ is the set of columns that warp $w$ owns, and split-Q computes $O_{r_w,\,:} = P_{r_w,\,:}\, V$ for the rows $r_w$ that warp $w$ owns. Both give the same matrix. They differ in what crosses warps.

The paper uses the same idea for the backward pass but notes that it "still requires some synchronization" because of the dependencies among $Q$, $K$, $V$, $O$, $dO$, $dQ$, $dK$, and $dV$ (Section 3.3). It also says to choose block sizes from $\{64, 128\} \times \{64, 128\}$ by head dimension and shared-memory size, because larger blocks cut shared-memory traffic but need more registers (Section 3.3, "Tuning block sizes").

**Tiny example, computed by `scripts/warp_partition.py`.** Take $B_r = 128$, $B_c = 64$, $d = 64$, and $W = 4$ warps. The script runs both layouts on random data and confirms that both equal $PV$. In split-K, each of the 4 warps holds a $128 \times 64$ partial output, which is 8,192 elements. The page's model is that each partial is written once and read once, so the block writes 32,768 elements to shared memory and reads 32,768 back. In split-Q, no element crosses between warps for the product $PV$. The count excludes the softmax statistics, which split-K must also share across warps. The model gives the direction of the saving. The paper measures the effect only as part of the end-to-end speedup (Section 4).

### Check the backward pass with column blocks

Algorithm 2 recomputes $P$ from $Q$, $K$, and the stored $L$ (line 11), and uses $D = \operatorname{rowsum}(dO \circ O)$ (line 4) for the softmax gradient. `scripts/backward_check.py` runs it with the column blocks in a shuffled order, as thread blocks with atomic adds would. For $N = 12$, $d = 4$, $B_r = 4$, and $B_c = 3$, the order was 3, 1, 0, 2. The gradients agree with the standard formulas of Section 2.2 to about $10^{-16}$ for $dQ$, $dK$, and $dV$. That shows why the column blocks need no ordering: floating-point addition is not associative, but the differences are at rounding level. Algorithm 2 prints $D \in \mathbb{R}^d$ on line 4, which is a typo for $\mathbb{R}^N$, one value per row.

### What the paper reports

On an A100 80GB, FlashAttention-2 is 1.7-3.0 times faster than FlashAttention, 1.3-2.5 times faster than the Triton implementation, and 3-10 times faster than a standard PyTorch implementation, with up to 230 TFLOPs/s, which is 73% of the theoretical maximum (Section 4). Without a causal mask, head dimension 128, and $N = 16$k, the forward pass reaches 223 TFLOPs/s (Figure 5b). End to end, GPT-style training reaches up to 225 TFLOPs/s per A100, which is 72% model FLOPs utilization, and is up to 1.3 times faster than FlashAttention and 2.8 times faster than a baseline without it (Section 4.2). The paper's FLOP count for these figures is $4 \cdot \text{seqlen}^2 \cdot \text{head dim} \cdot \text{heads}$ for the forward pass, halved with a causal mask, and 2.5 times that for the backward pass, because the backward pass does 5 matmuls and recomputation (Section 4.1).

<!-- exercises -->

## Pitfalls

- **Dividing inside the loop.** The output $\tilde O$ is unscaled, and its rows do not sum to a weighted average until the final division. Dividing at every block gives the right answer and gives back FlashAttention's cost. Exercise 1's tests catch a missing division, and the per-tile cost in Exercise 2 shows what the extra division costs.
- **Rescaling $\ell$ but not $\tilde O$, or the reverse.** Both carry the factor $e^{m_{\text{old}} - m_{\text{new}}}$ when the maximum grows. Test with scores that increase from block to block, and again with the blocks reversed.
- **Reading the inverse in the paper's rescale literally.** Algorithm 1 prints $\operatorname{diag}(\cdot)^{-1}$ on the factor $e^{m^{(j-1)} - m^{(j)}}$. Taking that literally multiplies the old output by a number at least 1 when the maximum grows, which is wrong.
- **Storing $m$ and $\ell$ for the backward pass.** FlashAttention-2 stores only $L = m + \log \ell$. Then $P = \exp(S - L)$ in Algorithm 2, line 11, already includes the normalization. Dividing by $\ell$ again, or using $m$ alone, breaks the gradients.
- **Expecting a different result.** FlashAttention-2 changes the schedule, not the math. It computes the same $\operatorname{softmax}(QK^\top)V$ as the first paper and as the [Transformer](/transformer/) definition, up to rounding.
- **Expecting a speedup at every shape.** The parallelism change matters when $\text{batch} \cdot \text{heads}$ is small, which usually means long sequences. With 1024 or more blocks already, the table above shows 94.8% utilization before the change.
- **Treating the counts as measurements.** The grid, warp, and operation counts on this page are models with stated simplifications: one block per SM, one cost per scalar operation, and each partial written once and read once. They show which way each change moves the cost. Only the paper's benchmarks measure the speedup.
- **Skipping the mask where it is needed.** With square blocks, one tile per row block crosses the diagonal and needs the mask. With $B_r = 128$ and $B_c = 64$, that is two tiles. Applying the mask to only one of them leaks future keys.
- **Mixing up the backward grid.** The backward pass assigns one thread block per column block, not per row block, because each block owns $dK_j$ and $dV_j$. They share only $dQ$, and update it with atomic adds (Section 3.2).
- **Assuming the H100 numbers need new code.** The paper reports up to 335 TFLOPs/s on an H100 with the same implementation and no new hardware instructions, and expects another 1.5-2 times from using them (Section 4.1). The [FlashAttention-3](/fa3/) page covers that follow-up.

## Further reading

- [flash-attention](https://github.com/Dao-AILab/flash-attention) is the official code release of FlashAttention and FlashAttention-2.
- [Fused Attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) is the Triton tutorial that the paper credits for the loop swap and the sequence-length parallelism (Section 3.2).
- [FlashAttention](/flashattention/) is the prerequisite page on this site, with the IO analysis that this paper builds on.
- [Online normalizer calculation for softmax](/online-softmax/) derives the running-maximum update that both versions use.
- [FlashAttention-3](/fa3/) is the follow-up on this site.
