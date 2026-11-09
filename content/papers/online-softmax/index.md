---
id: "online-softmax"
title: "Online normalizer calculation for softmax"
authors: "Milakov et al."
year: 2018
venue: "arXiv preprint"
arxiv: "1805.02867"
url: "https://arxiv.org/abs/1805.02867"
track: gpu-systems
tier: core
est_hours: 2
status: full
summary: "Safe softmax reads its input three times: once for the maximum, once for the normalizer, and once for the outputs. This paper computes the maximum and the normalizer in a single pass by keeping a running maximum and a running sum, and rescaling the sum by an exponential factor whenever the maximum grows. That cuts memory accesses from 4 to 3 per element, the same update merges partial results from separate blocks, and it lets softmax fuse with a following top-k."
related_course: "gpu: reductions"
prereqs: []
background: [softmax-numerics, running-max]
course_links:
  - title: "Softmax and normalization"
    url: "https://gpu.ifkash.dev/c/15-softmax-and-normalization"
  - title: "FlashAttention"
    url: "https://gpu.ifkash.dev/c/16-flash-attention"
self_check:
  - q: 'Why does an implementation compute $\operatorname{softmax}([1000, 1001])$ as $e^{x_i - 1001} / \sum_j e^{x_j - 1001}$, and what are the two outputs?'
    a: 'In float32, $e^{1000}$ overflows to infinity, so the naive formula returns `NaN`. Subtracting the maximum multiplies the numerator and denominator by the same constant, so the outputs are unchanged: $[0.2689, 0.7311]$. Every exponent is at most 0.'
    see: softmax-numerics
  - q: 'A running maximum has seen $[2, 1]$ and holds $m = 2$ and a sum $d = 1 + e^{-1} = 1.3679$ measured against that maximum. The next value is 4. What are $m$ and $d$ afterward?'
    a: 'The maximum becomes $m = 4$. The old sum was measured against 2, so it gets multiplied by $e^{2 - 4} = 0.1353$, and the new element adds $e^{4 - 4} = 1$. The result is $d = 1.3679 \times 0.1353 + 1 = 1.1851$.'
    see: running-max
  - q: 'Safe softmax makes three passes over a vector of $V$ elements. How many memory accesses per element is that, and why does one pass over the data for the maximum and the sum matter on a GPU?'
    a: 'Three loads and one store: 4 accesses per element. Softmax does very little arithmetic per byte, so the memory bandwidth limits its speed, and removing one load speeds it up by up to the ratio 4/3.'
    see: softmax-numerics
---

## The problem

Softmax turns a vector of scores into probabilities (Section 2, Equation 1):

$$
y_i = \frac{e^{x_i}}{\sum_{j=1}^{V} e^{x_j}}
$$

Here $x, y \in \mathbb{R}^V$, and $V$ is the vector length. This page keeps the paper's notation: $x_j$ for the inputs, $m_j$ for the running maximum, $d_j$ for the running normalizer, and $\oplus$ for the merge operator. The code in the exercises writes `xs` for the input and `combine` for $\oplus$.

The direct implementation, Algorithm 1 ("naive softmax"), has two problems.

**It overflows.** In float32, $e^{x}$ is infinite for $x \geq 89$. Take $x = [100, 101, 102]$. Every exponential is infinite, the sum is infinite, and every output is $\infty / \infty = \text{NaN}$. The same happens at the other end: for $x = [-200, -201, -202]$, every $e^{x}$ underflows to 0, and every output is $0/0$. The script `scripts/overflow_example.py` prints `[nan nan nan]` for both inputs.

**The fix costs a pass.** Equation 2 subtracts the maximum $m_V = \max_k x_k$ from every score, which changes no output and keeps every exponent at most 0:

$$
y_i = \frac{e^{x_i - m_V}}{\sum_{j=1}^{V} e^{x_j - m_V}}
$$

The paper calls this "safe softmax", and Section 2 lists the frameworks that use it (TensorFlow, PyTorch, MXNet, CNTK, and Chainer, at the versions it names). Algorithm 2 needs three loops over the input: one for $m_V$, one for the normalizer $d_V = \sum_j e^{x_j - m_V}$, and one for the outputs $y_i$. That is three loads and one store, or 4 memory accesses per element. Algorithm 1 needs 3.

Softmax has a very low ratio of arithmetic to memory traffic (Section 5), so the memory system, not the arithmetic, limits its speed. The question is whether the safe version can drop a pass. The paper's answer is yes, by computing $m_V$ and $d_V$ together. Section 3 reports that this reduces the accesses from 4 to 3, and that the benchmarks on a Tesla V100 show up to 1.3x speedup (Section 5.1).

`scripts/access_counts.py` counts the accesses by wrapping the input and output arrays in a counter. It prints 3 for naive softmax, 4 for safe softmax, and 3 for online softmax.

## Key ideas

### Keep a running maximum and a running sum

**Intuition.** Suppose you want $\sum_j e^{x_j - m}$ but don't know $m$ until you have seen every element. Keep a guess $m_j$, the largest value so far, and a sum $d_j$ measured against that guess. When a new element arrives and the guess is still the largest, add its exponential. When it isn't, the sum was measured against the wrong reference.

**Math.** Algorithm 3 starts from $m_0 = -\infty$ and $d_0 = 0$ and runs, for $j = 1, \dots, V$ (lines 3-6):

$$
m_j = \max(m_{j-1}, x_j)
$$

$$
d_j = d_{j-1} \, e^{m_{j-1} - m_j} + e^{x_j - m_j}
$$

The second line is line 5 of the algorithm. The update of $m_j$ comes first, because $d_j$ uses both $m_{j-1}$ and $m_j$. The paper says the inspiration was the online algorithm for variance (Welford, reference [18]).

**Tiny example, computed by `scripts/trace_example.py`.** Take $x = [2, 1, 4, 3]$. The trace is as follows:

| $j$ | $x_j$ | $m_j$ | Rescale factor $e^{m_{j-1} - m_j}$ | $d_j$ |
|---|---|---|---|---|
| 1 | 2 | 2 | 0 (from $e^{-\infty}$) | 1.0000 |
| 2 | 1 | 2 | 1.0000 | 1.3679 |
| 3 | 4 | 4 | 0.1353 | 1.1851 |
| 4 | 3 | 4 | 1.0000 | 1.5530 |

Check step 3 by hand: the old sum 1.3679 was measured against 2, so it becomes $1.3679 \times e^{-2} = 0.1851$, and the new element adds $e^{4-4} = 1$. The total is 1.1851. The final pair $(m_4, d_4) = (4, 1.5530)$ equals what the two-loop safe version computes, and the outputs are $y = [0.0871, 0.0321, 0.6439, 0.2369]$.

Theorem 1 (Section 3) proves by induction that lines 1-6 of Algorithm 3 produce exactly $m_V = \max_k x_k$ and $d_V = \sum_{j=1}^{V} e^{x_j - m_V}$. The proof's inductive step is the identity $\left(\sum_{j<S} e^{x_j - m_{S-1}}\right) e^{m_{S-1} - m_S} = \sum_{j<S} e^{x_j - m_S}$. Rescaling the old sum by one factor changes the reference point for every term at once.

The paper also states that the algorithm is safe: $m_j$ stays between the smallest and largest running values, and $1 \le d_j \le j$ for every $j$. The script checks the bound on 200 random vectors.

Exercise 1 asks you to write this loop.

### Online softmax: three accesses, not four

**Intuition.** The loop above produces both $m_V$ and $d_V$ in one read of the input. The outputs still need $m_V$ and $d_V$, so a second pass remains. Safe softmax needs three loops, and this version needs two.

**Math.** Algorithm 3 finishes with a loop that computes $y_i = e^{x_i - m_V} / d_V$ (lines 7-9). The first loop loads each element once, and the second loads it once and stores one output. The paper counts 3 accesses per element, against 4 for Algorithm 2, which is a 1.33x reduction (Section 3).

The extra work is "negligible additional cost of two operations per vector element" in the paper's words: one more exponential and one multiply per element in the first loop. Because softmax is bound by memory bandwidth, the paper expects the saved access to pay for them. Section 5.1 reports that it does, on a Tesla V100 in float32. The speedup over safe softmax is about 1.3x for 4000 vectors of length $V \geq 4000$ (Figure 1), and about 1.15x for 10 vectors with $V \geq 1000$ (Figure 2). Below about $V = 1000$ all three algorithms perform alike.

One more claim from the Results section: if you currently use naive softmax, switching to online softmax costs little or nothing and improves accuracy, because it is safe.

### Merge partial results with an associative operator

**Intuition.** A sequential loop uses one thread. To use a whole GPU, split the vector into blocks, summarize each block on its own, and merge the summaries. Each summary is a pair $[m, d]$, and merging two of them is the same rescale as one step of the loop.

**Math.** Section 3.1 generalizes the update to a binary operation (Equation 3 and Equation 4). Equation 3 writes the result as a fold over the pairs $[x_i, 1]$:

$$
\begin{bmatrix} m_V \\ d_V \end{bmatrix}
= \begin{bmatrix} x_1 \\ 1 \end{bmatrix}
\oplus \begin{bmatrix} x_2 \\ 1 \end{bmatrix}
\oplus \dots \oplus
\begin{bmatrix} x_V \\ 1 \end{bmatrix}
$$

and Equation 4 defines $\oplus : \mathbb{R}^2 \times \mathbb{R}^2 \to \mathbb{R}^2$:

$$
\begin{bmatrix} m_i \\ d_i \end{bmatrix}
\oplus
\begin{bmatrix} m_j \\ d_j \end{bmatrix}
=
\begin{bmatrix}
\max(m_i, m_j) \\
d_i \, e^{m_i - \max(m_i, m_j)} + d_j \, e^{m_j - \max(m_i, m_j)}
\end{bmatrix}
$$

A single element $x_i$ is the summary $[x_i, 1]$, because its maximum is $x_i$ and $e^{x_i - x_i} = 1$. Folding from left to right gives the loop of Algorithm 3. The paper states that $\oplus$ is associative, which allows any tree of merges, and commutative, which allows any order. It omits both proofs "for brevity". The page's script checks them numerically but doesn't prove them.

**Tiny example, computed by `scripts/merge_example.py`.** Split $x = [2, 1, 4, 3]$ into the blocks $[2, 1]$ and $[4, 3]$. Each block summary is computed independently:

| Block | $m$ | $d$ |
|---|---|---|
| $[2, 1]$ | 2 | 1.3679 |
| $[4, 3]$ | 4 | 1.3679 |

The merge takes $m = \max(2, 4) = 4$ and $d = 1.3679 \, e^{2-4} + 1.3679 \, e^{4-4} = 0.1851 + 1.3679 = 1.5530$. That equals the normalizer of the whole vector. The script also draws random states and finds the largest relative gap between the two groupings of three summaries is about $10^{-16}$, which is floating-point rounding. It reduces 10 values in blocks of 3, 3, 3, and 1, left to right and as a tree, and gets the same $d$.

Exercise 2 asks you to write $\oplus$ and a blocked softmax.

### Fuse softmax with top-k

**Intuition.** Beam search in an autoregressive model computes a softmax over the vocabulary and then keeps only the $K$ most likely tokens. Nobody needs the other $V - K$ probabilities. If the normalizer is available online, the selection can run in the same loop, and the full output vector is never written.

**Math.** The paper defines top-$K$ (Section 4, Equation 5) as the pair $(v, z)$ with $v_i = y_{z_i}$ and $v_i \ge y_j$ for every selected $i$ and every $j \notin z$. Here $y \in \mathbb{R}^V$, $z \in \mathbb{Z}^K$, and $v \in \mathbb{R}^K$.

Algorithm 4 extends Algorithm 3. Besides $m$ and $d$, it keeps a sorted array $u \in \mathbb{R}^{K+1}$ of the largest inputs so far and an array $p$ of their indices. For each $x_j$ it does the following:

1. Updates $m_j$ and $d_j$ as before (lines 6-7).
2. Writes $x_j$ and $j$ into slot $K+1$ (lines 8-9).
3. Moves the new entry up while the entry above it is strictly smaller (lines 10-15). The first $K$ entries are already sorted, so one insertion pass is enough.

At the end it outputs $v_i = e^{u_i - m_V} / d_V$ and $z_i = p_i$ for $i = 1, \dots, K$ (lines 17-20). The probabilities need the global $m_V$ and $d_V$, which include every element, not only the kept ones.

**Access counts, computed by `scripts/access_counts.py`.** Top-$K$ loads each element at least once. The paper counts the options as follows (Section 4):

| Pipeline | Accesses per element |
|---|---|
| Safe softmax, then top-$K$ | $4 + 1 = 5$ |
| Online softmax, then top-$K$ | $3 + 1 = 4$ |
| Algorithm 4, fused | 1 |

On the Tesla V100 with $K = 5$, Section 5.2 reports a speedup of up to 5x over unfused safe softmax and top-$K$. The paper splits that into 2.5x from fusion and 2x from online softmax. The gain shrinks as $K$ grows, because the insertion step starts to dominate: 3.5x for $K = 10$, 2x for $K = 15$, and 1.4x for $K = 30$.

Exercise 3 asks you to write Algorithm 4.

### What this paper does not cover

Section 7 suggests fusing the softmax with the preceding layer to avoid a round trip through memory, and calls that "more challenging". The paper doesn't describe attention. The later [FlashAttention](/flashattention/) work applies the same rescaling idea to attention: when the running maximum grows, it also rescales a running weighted sum of value vectors. That step is not in this paper.

<!-- exercises -->

## Pitfalls

- **Updating $d$ before $m$.** The rescale factor $e^{m_{j-1} - m_j}$ needs the new maximum. Computing $d_j$ with the old maximum and updating $m_j$ afterward runs without an error and gives a wrong sum.
- **Forgetting the rescale.** Adding $e^{x_j - m_j}$ to the old $d_{j-1}$ without the factor is correct only while the maximum never changes. On sorted-descending input it passes every check. On ascending input it fails, so test both orders.
- **Starting the running maximum at 0.** The initial value must be $-\infty$ (Algorithm 3, line 1). With all-negative data, a start at 0 never updates, and the sum is measured against a maximum that isn't in the data.
- **The empty summary.** $[-\infty, 0]$ acts as an identity for $\oplus$ when the other summary is finite. Merging two of them computes $-\infty - (-\infty)$, which is `NaN`. A row that is entirely $-\infty$, for example a fully masked attention row, needs a special case. The paper doesn't cover it.
- **Calling it one pass.** Online softmax computes $m_V$ and $d_V$ in one pass. The outputs still need a second pass over $x$, so the total is 3 accesses, not 2. Only the top-$K$ fusion, which discards the outputs, reaches 1.
- **Reading the speedups as universal.** The numbers in Section 5 are for float32 on one GPU generation (Tesla V100, CUDA 9.1), with a benchmark in CUDA C. Section 7 says the authors didn't test CPUs. The 1.33x figure is the reduction in accesses, and 1.3x is the best speedup the paper measures.
- **Ignoring the bound on $d$.** Section 3 notes that $1 \le d_j \le j$, so a float32 $d$ is safe for up to about $1.7 \times 10^{37}$ elements, and a longer vector needs a 64-bit $d$. Real vectors are far shorter, but the bound explains why the algorithm is safe.
- **Dropping the partial last block.** A blocked implementation that drops or pads the partial last block gives a normalizer that is silently wrong. Exercise 2 tests a leftover block that holds the maximum.
- **Breaking ties in top-$K$ the other way.** Algorithm 4 swaps only when $u_k < u_{k+1}$, so equal values keep their input order. Swapping on $\le$ returns the later index first.

## Further reading

- [NVIDIA/online-softmax](https://github.com/NVIDIA/online-softmax) is the authors' benchmark code, linked in Section 5 of the paper.
- Welford, "Note on a method for calculating corrected sums of squares and products" (Technometrics, 1962) is the online variance algorithm that inspired the update. It is reference [18] in the paper.
- The [FlashAttention](/flashattention/) page on this site shows the same running-maximum rescale applied to attention, with a running weighted sum of values.
- The course lesson [Softmax and normalization](https://gpu.ifkash.dev/c/15-softmax-and-normalization) covers the safe and online versions as GPU reductions.
