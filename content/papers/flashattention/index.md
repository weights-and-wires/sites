---
id: "flashattention"
title: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
authors: "Dao et al."
year: 2022
venue: "NeurIPS 2022"
arxiv: "2205.14135"
url: "https://arxiv.org/abs/2205.14135"
track: gpu-systems
tier: core
est_hours: 6
status: full
summary: "Standard attention is slow because it writes the full score matrix and the full probability matrix to GPU high-bandwidth memory and reads them back, and softmax is memory-bound. FlashAttention splits the queries, keys, and values into blocks that fit in on-chip SRAM, runs both matrix multiplications and the softmax on each block with running softmax statistics, and never writes the full matrix to HBM. For the backward pass it stores only the output and two vectors and recomputes the blocks on chip, which adds FLOPs but removes far more HBM traffic, so the result is exact and faster."
related_course: "gpu: attention kernels"
prereqs:
  - id: transformer
    reason: "The algorithm computes the softmax(QK^T)V of scaled dot-product attention exactly, and it needs the S, P, and O matrices defined there."
  - id: roofline
    reason: "The paper's argument is that attention is memory-bound, which you read off the roofline as an operational intensity left of the ridge point."
  - id: online-softmax
    reason: "Algorithm 1 merges per-block softmax statistics with the running-maximum and running-sum update, and applies the same rescale to the partial output."
background: [gpu-memory-hierarchy, sram-vs-hbm, tiling, recomputation]
course_links:
  - title: "FlashAttention"
    url: "https://gpu.ifkash.dev/c/16-flash-attention"
  - title: "FlashAttention"
    url: "https://qwen.ifkash.dev/c/14-flash-attention"
self_check:
  - q: 'A row of scores is split into the blocks $[0, 0]$ and $[1, 3]$. Block 1 has $(m, \ell) = (0, 2)$. Block 2 has $(m, \ell) = (3, 1.1353)$. Merge them, and say what the merged $\ell$ equals.'
    a: 'The merged maximum is $\max(0, 3) = 3$. The merged normalizer is $\ell = e^{0-3} \cdot 2 + e^{3-3} \cdot 1.1353 = 0.0996 + 1.1353 = 1.2349$. That equals $\sum_j e^{x_j - 3}$ over the whole row $[0, 0, 1, 3]$, which is the point of the merge: no block needs the others.'
    see: online-softmax
  - q: 'Standard attention with $N = 1024$ and $d = 64$ moves 4,456,448 elements to and from HBM and does about $4N^2 d$ FLOPs. In fp16, what is its operational intensity, and is it left or right of the ridge point of a GPU with 312 TFLOP/s and 1.5 TB/s?'
    a: 'The FLOPs are $4 \cdot 1024^2 \cdot 64 \approx 2.68 \times 10^8$, and the bytes are $2 \times 4{,}456{,}448 \approx 8.9 \times 10^6$. The intensity is about 30 FLOPs per byte. The ridge point is $312 / 1.5 = 208$ FLOPs per byte, so attention is far left of it, which means memory-bound.'
    see: roofline
  - q: 'The backward pass of attention needs $P$, an $N \times N$ matrix. What does FlashAttention store in the forward pass in place of it, and why is the backward pass still faster despite the extra FLOPs?'
    a: 'It stores the output $O$ and the softmax statistics $m$ and $\ell$, which hold $O(N)$ numbers beyond $O$. The backward pass rebuilds each block of $P$ on chip as $\exp(S_{ij} - m_i) / \ell_i$ from blocks of $Q$ and $K$. The extra matrix multiplication costs FLOPs, but reading an $N \times N$ matrix from HBM costs more time than recomputing it.'
    see: recomputation
---

## The problem

Attention on a long sequence is slow, and the usual explanation is the FLOP count: the score matrix has $N^2$ entries. That explanation is incomplete. The paper's central claim is that wall-clock time follows the number of reads and writes between GPU memory levels, which it calls IO (Section 1).

Take the standard implementation (Section 2.2). With $Q, K, V \in \mathbb{R}^{N \times d}$ it computes

$$
S = QK^\top \in \mathbb{R}^{N \times N}, \qquad P = \operatorname{softmax}(S) \in \mathbb{R}^{N \times N}, \qquad O = PV \in \mathbb{R}^{N \times d}
$$

and it materializes $S$ and $P$ in HBM, the GPU's large and slow memory. Take GPT-2, with $N = 1024$ and $d = 64$. Each head holds a 1024 x 1024 matrix, twice (once for $S$, once for $P$), and the softmax between the two matrix multiplications does little arithmetic per byte it moves. The two matrix multiplications are fast, and the memory round trips around them dominate the time.

Earlier work on approximate attention (sparse, low-rank, and combinations) cut FLOPs to near-linear in $N$, but many of those methods showed no wall-clock speedup against standard attention, because they ignored memory access (Section 1). FlashAttention keeps attention exact and attacks the memory traffic.

**Notation.** This page uses the paper's. $N$ is the sequence length, $d$ is the head dimension, and $M$ is the SRAM size in elements. Blocks have $B_r$ rows of $Q$ and $B_c$ rows of $K$ and $V$, and there are $T_r = \lceil N / B_r \rceil$ and $T_c = \lceil N / B_c \rceil$ of them. The [Transformer page](/transformer/) writes $n$ and $d_k$ instead, and divides the scores by $\sqrt{d_k}$. Section 3 of this paper writes $S = QK^\top$ with no scale. Algorithm 4 adds a scale $\tau$ and the exercises here use it as `tau`.

## Key ideas

### Attention is limited by HBM traffic

**Intuition.** A GPU has a small, fast on-chip SRAM and a large, slow HBM. Each kernel reads its inputs from HBM, computes, and writes its outputs back. For an operation that does little arithmetic per byte, such as softmax, the reads and writes set the time (Section 2.1). The [roofline model](/roofline/) states the same fact: an operation to the left of the ridge point is limited by bandwidth, not by the peak FLOP rate.

**Math.** The paper's Algorithm 0 (Section 2.2) has three steps, each of which goes through HBM:

1. Load $Q$ and $K$ and write $S$.
2. Read $S$, compute $P$, and write $P$.
3. Load $P$ and $V$ and write $O$.

Counting elements, step 1 moves $2Nd + N^2$, step 2 moves $2N^2$, and step 3 moves $N^2 + 2Nd$. The total is $4Nd + 4N^2$, which is $\Theta(Nd + N^2)$ as in Theorem 2. The extra memory is $O(N^2)$ for $S$ and $P$.

**Tiny example, computed by `scripts/io_count.py`.** For $N = 1024$ and $d = 64$, the count is $4 \cdot 65{,}536 + 4 \cdot 1{,}048{,}576 = 4{,}456{,}448$ elements. Check the sum by hand: $262{,}144 + 4{,}194{,}304$. In fp16 that is 8.9 MB per head per forward pass, and $S$ and $P$ together occupy 4.0 MiB per head.

The two matrix multiplications do $4N^2d \approx 2.68 \times 10^8$ FLOPs. Divide by the 8.9 MB and the operational intensity is 30.1 FLOPs per byte. The paper's A100 has an HBM bandwidth of 1.5 to 2.0 TB/s (Section 2.1). With 1.5 TB/s and an fp16 tensor-core peak of 312 TFLOP/s, the ridge point is 208 FLOPs per byte. The page takes the 312 TFLOP/s figure from NVIDIA's A100 datasheet, not from the paper. Attention at 30 FLOPs per byte sits far left of the ridge.

The tiled algorithm below changes this count, and Exercise 2 asks you to write it.

### Tile Q, K, and V and keep running softmax statistics

**Intuition.** The softmax of a row needs every score in that row, which looks like it forces the full $N \times N$ matrix into memory. It doesn't. A softmax over a row can be built from softmaxes over pieces of the row, if each piece also reports its maximum and its sum. This is the running-maximum update from [Online normalizer calculation for softmax](/online-softmax/). FlashAttention adds one more step: it applies the same rescaling to the partial output $O$, so the matrix multiplication with $V$ happens block by block as well.

**Math.** Section 3.1 defines, for a vector $x \in \mathbb{R}^B$,

$$
m(x) = \max_i x_i, \qquad f(x) = \left[e^{x_1 - m(x)}, \dots, e^{x_B - m(x)}\right], \qquad \ell(x) = \sum_i f(x)_i, \qquad \operatorname{softmax}(x) = \frac{f(x)}{\ell(x)}
$$

For the concatenation $x = [x^{(1)}, x^{(2)}]$, the paper gives (Section 3.1; these displays have no equation numbers)

$$
m(x) = \max\!\left(m(x^{(1)}), m(x^{(2)})\right), \qquad
\ell(x) = e^{m(x^{(1)}) - m(x)}\,\ell(x^{(1)}) + e^{m(x^{(2)}) - m(x)}\,\ell(x^{(2)})
$$

Algorithm 1 applies this to a row block of the attention matrix. The loop (lines 5-15) takes blocks $K_j, V_j$ in the outer loop and blocks $Q_i$ in the inner loop. For each pair, it computes on chip:

- $S_{ij} = Q_i K_j^\top \in \mathbb{R}^{B_r \times B_c}$ (line 9).
- $\tilde m_{ij} = \operatorname{rowmax}(S_{ij})$, $\tilde P_{ij} = \exp(S_{ij} - \tilde m_{ij})$, and $\tilde\ell_{ij} = \operatorname{rowsum}(\tilde P_{ij})$ (line 10).
- $m_i^{\text{new}} = \max(m_i, \tilde m_{ij})$ and $\ell_i^{\text{new}} = e^{m_i - m_i^{\text{new}}}\ell_i + e^{\tilde m_{ij} - m_i^{\text{new}}}\tilde\ell_{ij}$ (line 11).
- The output update (line 12):

$$
O_i \leftarrow \operatorname{diag}(\ell_i^{\text{new}})^{-1}\left(\operatorname{diag}(\ell_i)\, e^{m_i - m_i^{\text{new}}} O_i + e^{\tilde m_{ij} - m_i^{\text{new}}} \tilde P_{ij} V_j\right)
$$

The old $O_i$ is already normalized by the old $\ell_i$. The update multiplies it back by $\ell_i$, rescales to the new maximum, adds the new block's unnormalized contribution, and divides by the new $\ell_i^{\text{new}}$. After the last block, $O_i$ is the exact attention output. Theorem 1 states that Algorithm 1 returns $\operatorname{softmax}(QK^\top)V$ with $O(N^2 d)$ FLOPs and $O(N)$ extra memory beyond the inputs and output, and Appendix C proves it.

Tiling also makes **kernel fusion** possible (Section 3.1). The matrix multiply, softmax, and second matrix multiply run in one CUDA kernel, so the intermediate blocks never leave SRAM.

**Tiny example, computed by `scripts/tile_trace.py`.** Take $d = 2$, $B_c = 2$, one query row $q = [1, 0]$, and four keys and values

$$
K = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 2 & 0 \\ 1 & 1 \end{bmatrix}, \qquad
V = \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \\ 7 & 8 \end{bmatrix}.
$$

The scores are $[1, 0, 2, 1]$. The script runs Algorithm 1 over the two blocks of keys and prints the state of row 0 after each block:

| After block | Scores seen | $m$ | $\ell$ | $O$ (row 0) |
|---|---|---|---|---|
| 1 | $[1, 0]$ | 1 | 1.3679 | `[1.5379, 2.5379]` |
| 2 | $[2, 1]$ | 2 | 1.8711 | `[4.4621, 5.4621]` |

Check block 2 by hand. The maximum grows from 1 to 2, so the old state is scaled by $e^{1-2} = 0.3679$. Then $\ell = 0.3679 \cdot 1.3679 + 1.3679 = 1.8711$. The new block has $\tilde P = [1, 0.3679]$, so $\tilde P V_2 = [5 + 0.3679 \cdot 7,\ 6 + 0.3679 \cdot 8] = [7.5752, 8.9430]$. The old unnormalized sum is $0.3679 \cdot 1.3679 \cdot [1.5379, 2.5379] = [0.7739, 1.2771]$. Their total, divided by 1.8711, is $[4.4621, 5.4621]$. Naive attention gives the same row, and the largest difference over all rows is about $10^{-15}$.

The script ends with large scores in float32: $q = [1000, 0]$ against four keys gives scores $[1000, 900, 0, 800]$. Softmax without the max shift returns `NaN` because $e^{1000}$ overflows. The tiled run returns $[1, 0]$ with $m = 1000$ and $\ell = 1$.

Exercise 1 asks you to write this forward pass.

### Fewer HBM accesses: the IO analysis

**Intuition.** Tiling reads $K$ and $V$ once. It re-reads $Q$ and $O$ once per block of $K$, so the cost is the number of K/V blocks times the size of $Q$ and $O$. A bigger SRAM means bigger blocks and fewer passes.

**Math.** Algorithm 1, line 1 sets the block sizes from the SRAM size:

$$
B_c = \left\lceil \frac{M}{4d} \right\rceil, \qquad B_r = \min\!\left(\left\lceil \frac{M}{4d} \right\rceil, d\right)
$$

Theorem 2 states that, for $d \le M \le Nd$, standard attention needs $\Theta(Nd + N^2)$ HBM accesses and FlashAttention needs $\Theta(N^2 d^2 M^{-1})$. The proof counts $T_c = \lceil N / B_c \rceil$ passes over $Q$ and $O$, and each pass moves $\Theta(Nd)$ elements, so $\Theta(Nd\, T_c)$ with $T_c = \Theta(Nd / M)$ (proof of Theorem 2, Appendix C). Proposition 3 adds a lower bound: no exact attention algorithm achieves $o(N^2 d^2 M^{-1})$ accesses for all $M$ in $[d, Nd]$, so the tiled algorithm is asymptotically optimal over SRAM sizes.

The ratio of the two counts is about $4N^2 / (12 N^2 d^2 / M) = M / (3 d^2)$ for large $N$, so the saving depends on $d$ and $M$ and not on $N$. For $d = 64$ and $M$ of about 100 KB, the paper says $d^2$ is many times smaller than $M$ and reports up to 9 times fewer accesses (Section 1, Figure 2).

**Tiny example, computed by `scripts/io_count.py`.** The script counts elements line by line. Per outer iteration of Algorithm 1 it counts the loads of $Q$, $O$, $\ell$, and $m$ and the writes of $O$, $\ell$, and $m$. It counts $K$ and $V$ once. With $N = 1024$, $d = 64$, and $M = 65{,}536$ elements, the blocks are $B_c = 256$ and $B_r = 64$, so $T_c = 4$ and $T_r = 16$. The count is $2Nd + T_c(3Nd + 4N) = 131{,}072 + 4 \cdot 200{,}704 = 933{,}888$ elements, against 4,456,448 for standard attention. That is 4.77 times fewer. The asymptotic ratio is $M / (3d^2) = 5.33$, and the ratio against $N$ approaches it:

| $N$ | Standard | FlashAttention | Ratio |
|---|---|---|---|
| 512 | 1,179,648 | 266,240 | 4.43 |
| 1,024 | 4,456,448 | 933,888 | 4.77 |
| 4,096 | 68,157,440 | 13,369,344 | 5.10 |
| 8,192 | 270,532,608 | 52,428,800 | 5.16 |

Block size matters, as Figure 2 (middle) shows on an A100. At $N = 1024$, going from $B_c = 64$ to 128, 256, and 512 gives 3,342,336, 1,736,704, 933,888, and 532,480 accesses. The paper measures the runtime falling with the access count until $B_c$ passes 256, where other costs take over (Section 3.2).

The count does not by itself make attention compute-bound. At fp16 and $B_c = 256$, the operational intensity rises from 30.1 to 143.7 FLOPs per byte, still left of the 208 ridge point in the example above. The paper's evidence is in wall-clock time. Figure 2 (left), measured on GPT-2 medium with $N = 1024$ on an A100, reports for standard attention and FlashAttention (forward plus backward): 66.6 and 75.2 GFLOPs, 40.3 and 4.4 GB of HBM reads and writes, and 41.7 and 7.3 ms. Those are the paper's measurements, copied by the script and not computed. The HBM ratio is 9.2 and the runtime ratio is 5.7.

Exercise 2 asks you to write these counts.

### Recompute the attention matrix in the backward pass

**Intuition.** The backward pass of attention needs $P$. Storing the full $P$ after the forward pass would bring back the $O(N^2)$ memory. Instead, FlashAttention saves only $O$ and the statistics $m$ and $\ell$. Given those, any block of $P$ can be rebuilt from blocks of $Q$ and $K$ in SRAM. The paper describes this as a form of selective gradient checkpointing (Section 3.1). Ordinary checkpointing trades speed for memory. Here the recomputation also saves time, because it replaces an HBM read of a large matrix.

**Math.** Appendix B.1 writes attention in terms of the normalizer $L_i = \sum_j e^{q_i^\top k_j}$ (Equation 1) and $o_i = \sum_j \frac{e^{q_i^\top k_j}}{L_i} v_j$ (Equation 2). The paper calls $q_i$, $k_j$, and $v_j$ "columns" there, but they are the rows of $Q$, $K$, and $V$ as defined in Section 2.2. Appendix B.2 derives the gradients for a scalar loss $\phi$ with $dO = \partial \phi / \partial O$:

$$
dv_j = \sum_i P_{ij}\, do_i \qquad \text{(Equation 3)}
$$

$$
D_i = P_{i:}^\top dP_{i:} = do_i^\top o_i \qquad \text{(Equation 4)}
$$

$$
dq_i = \sum_j P_{ij}\left(do_i^\top v_j - D_i\right) k_j \qquad \text{(Equation 5)}
$$

$$
dk_j = \sum_i P_{ij}\left(do_i^\top v_j - D_i\right) q_i \qquad \text{(Equation 6)}
$$

Along the way, $dP_{ij} = do_i^\top v_j$ and $dS_{ij} = P_{ij}(dP_{ij} - D_i)$, which comes from the softmax Jacobian $\operatorname{diag}(P_{i:}) - P_{i:}P_{i:}^\top$. The shortcut in Equation 4 is the key step for tiling. The sum $\sum_j P_{ij}\, dP_{ij}$ runs over a whole row, which might not fit in SRAM, but $do_i^\top o_i$ is a dot product of two length-$d$ vectors (Appendix B.4).

Algorithm 4 (Appendix B.4) is the blocked version. In its inner loop, it recomputes $S_{ij}$ (line 11) and $P_{ij} = \operatorname{diag}(\ell_i)^{-1}\exp(S_{ij} - m_i)$ (line 13) from the saved final statistics, then computes $D_i = \operatorname{rowsum}(dO_i \circ O_i)$ (line 19), $dS_{ij} = P_{ij} \circ (dP_{ij} - D_i)$ (line 20), and accumulates $dQ_i$ (line 21) and $dK_j$ (line 22). It also accumulates $dV_j$ (line 16). The dropout mask is regenerated from a saved random-number-generator state, so it takes no $O(N^2)$ storage either. Theorem 5 gives the backward pass the same IO complexity as the forward pass.

**Tiny example, computed by `scripts/backward_check.py`.** One query row $q = [1, 0]$ with keys $K = [[1, 0], [0, 1]]$, values $V = [[1, 2], [3, 4]]$, and $do = [1, 0]$. The scores are $[1, 0]$ and $P = [0.7311, 0.2689]$. The output is $o = [1.5379, 2.5379]$ and $dP = [do \cdot v_1,\ do \cdot v_2] = [1, 3]$. Equation 4 gives $D = P \cdot dP = 0.7311 + 0.8067 = 1.5379$, which equals $do \cdot o = 1.5379$. Then $dS = P \circ (dP - D) = [0.7311 \cdot (-0.5379),\ 0.2689 \cdot 1.4621] = [-0.3932, 0.3932]$. The entries sum to 0, as they must for a softmax.

The script then runs Algorithm 4 with $B_r = 3$ and $B_c = 4$ on a random problem with $N = 10$ and $d = 4$, using only $Q, K, V, O, dO, m, \ell$. The gradients $dQ$, $dK$, and $dV$ differ from naive attention's by at most $6 \times 10^{-16}$. $dQ$ also matches a finite-difference gradient to $6 \times 10^{-10}$, so the reference isn't circular. The forward pass stores $10 \cdot 4 + 2 \cdot 10 = 60$ numbers ($O$, $\ell$, $m$), where the attention matrix would take 100.

Exercise 3 asks you to write the per-block backward step.

### What this page leaves out

The paper also covers block-sparse FlashAttention (Section 3.3, Proposition 4), the benchmarks in Section 4, dropout and masking inside the kernel (Appendix B), and the comparison with Rabe and Staats (Appendix B.5). That comparison says both methods tile and recompute, but FlashAttention updates the output after each block, so it stores one copy of $O$, and it recomputes only the attention matrix in the backward pass. Rabe and Staats target total memory, while FlashAttention targets memory accesses and is faster as a result. The page also doesn't cover the CUDA implementation. See the [FlashAttention-2](/fa2/) page for the follow-up.

<!-- exercises -->

## Pitfalls

- **Confusing fewer FLOPs with fewer HBM accesses.** FlashAttention does more FLOPs than standard attention (75.2 against 66.6 GFLOPs in Figure 2, left) and is faster. Count bytes that cross the HBM boundary, not operations.
- **Reading $m$ and $\ell$ as per-block values.** After the update, $m_i$ and $\ell_i$ describe every key seen so far in row $i$. Only $\tilde m_{ij}$ and $\tilde\ell_{ij}$ are block-local. Using the block-local values to rebuild $P$ in the backward pass gives block-local softmax weights, which is wrong (Exercise 3).
- **Forgetting to rescale the old output.** When a later block has a larger maximum, the old $O_i$ and $\ell_i$ both need the factor $e^{m_i - m_i^{\text{new}}}$. A run on random data with small scores can pass by luck. Test with scores that increase from block to block.
- **Dropping the last partial block.** With $N$ not divisible by $B_r$ or $B_c$, use `range(0, N, B)` slices and let the last block be short. Integer division silently drops rows.
- **A $-\infty$ row.** The state starts at $m = -\infty$. That works on the first block because $e^{-\infty}$ is 0, but a row that is fully masked has $m^{\text{new}} = -\infty$, and $-\infty - (-\infty)$ is `NaN`. Production kernels guard this case. The exercises here have no mask.
- **Counting the loop order backward.** In Algorithm 1 the outer loop runs over $K$ and $V$ blocks and the inner loop over $Q$ blocks, so $Q$ and $O$ are re-read $T_c$ times. Swapping the loops changes which matrices are re-read. [FlashAttention-2](/fa2/) reorders the loops.
- **Mixing up scale conventions.** Section 3 omits the $1/\sqrt{d}$ scale and Algorithm 4 includes a $\tau$. The paper doesn't say how $\tau$ is set in Algorithm 1. A reference that compares your output with a Transformer must apply $\tau = 1/\sqrt{d}$ in both.
- **Treating the speedups as one number.** The 7.6 times in Figure 1 (right) is for the attention computation on GPT-2. End-to-end, the paper reports 3 times on GPT-2 (sequence length 1K) and 15% on BERT-large (sequence length 512) against the MLPerf 1.1 record (Section 1).
- **Expecting an exact count from the page's numbers.** The element counts ignore the one-time initialization of $O$, $\ell$, and $m$ (line 2), cache effects, and the real kernel's tile shapes. They show the scaling, not a profiler reading.

## Further reading

- [FlashAttention](https://github.com/Dao-AILab/flash-attention) is the official code release. The paper's footnote points to the earlier HazyResearch repository.
- [Self-attention Does Not Need O(n^2) Memory](https://arxiv.org/abs/2112.05682), by Rabe and Staats, is the tiled, recomputing algorithm that the paper compares against in Appendix B.5.
- [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html), by Horace He, explains compute-bound, memory-bound, and overhead-bound operations in the terms of Section 2.1.
- [Online normalizer calculation for softmax](/online-softmax/) and [Roofline](/roofline/) are the two prerequisite pages on this site.
- [FlashAttention-2](/fa2/) is the follow-up on this site.
