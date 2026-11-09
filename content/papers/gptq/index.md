---
id: "gptq"
title: "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"
authors: "Frantar et al."
year: 2022
venue: "ICLR 2023"
arxiv: "2210.17323"
url: "https://arxiv.org/abs/2210.17323"
track: inference
tier: core
est_hours: 5
status: full
summary: "GPTQ quantizes the weights of a pre-trained language model to 3 or 4 bits in one shot, without retraining, and it handles models of 175 billion parameters in about four GPU hours. It treats each linear layer as a least-squares problem: it rounds the weight columns one at a time and, after each rounding, shifts the not-yet-rounded weights to cancel the output error, using the inverse of the layer's Hessian. Processing columns in one fixed order, in lazy blocks, with a Cholesky factor of the inverse Hessian makes this second-order method fast and numerically stable enough for the largest models."
related_course: "inference: quantization"
prereqs:
  - id: llm-int8
    reason: "GPTQ uses the same per-row min-max grid as the round-to-nearest weight quantizer of LLM.int8(), treats that quantizer as its main baseline, and keeps the activations in 16-bit, so you need the absmax and vector-wise ideas first."
background: [hessian, least-squares, layer-wise-quantization]
course_links:
  - title: "Quantization on Ampere"
    url: "https://qwen.ifkash.dev/c/18-quantization"
  - title: "Quantized kernels"
    url: "https://gpu.ifkash.dev/c/18-quantized-kernels"
self_check:
  - q: 'For one row $w$ of a layer, the squared error in the output on inputs $X$ is $\lVert wX - \hat{w}X \rVert^2$. What is the Hessian of this error with respect to $\hat{w}$, and why does the paper need only one Hessian for the whole layer?'
    a: 'The Hessian is $H = 2XX^\top$. It depends on the layer inputs $X$ and not on any weight, and all rows of $W$ see the same inputs. So every row has the same $H$, and one inverse serves the whole layer (Section 4, Step 1).'
    see: hessian
  - q: 'In the page example, rounding $w_0 = 0.37$ to $0.25$ moves $w_1$ from $-0.82$ to $-0.7817$. In which direction does the correction push $w_1$, and why, given that $H_{01} = 2 > 0$?'
    a: 'It pushes $w_1$ up, by $+0.0383$. The positive off-diagonal entry says that inputs 0 and 1 move together. Rounding $w_0$ down by $0.12$ lowers the output, so $w_1$ rises to make up some of the difference, which is what a least-squares fit to the full-precision output does.'
    see: least-squares
  - q: 'With the same 3-bit grid for both, GPTQ gives a larger mean weight error $\lvert W - Q \rvert$ than round-to-nearest on the page example (0.1918 against 0.1541) but a smaller layer error. How can both be true?'
    a: 'The layer objective (Equation 1) measures the error of the output $WX$, not of the weights. Rounding to nearest makes each weight as close as it can be to the original, but the errors add up in the output. GPTQ lets some weights move away from their nearest grid point so that their errors cancel across the inputs, which cuts the layer error to 0.00518 from 0.03016. This is the point of layer-wise quantization.'
    see: layer-wise-quantization
---

## The problem

A 175-billion-parameter language model stores its weights in 326 GB in 16-bit floating point, which is more than any single GPU holds, so inference needs a multi-GPU setup (Section 1). Quantizing the weights to 3 or 4 bits would shrink the model by about 4 to 5 times. The obstacle is that the methods which preserve accuracy at low bit widths either retrain the model, which is extremely expensive at this size, or do not scale (Section 1). The accurate post-training methods of the time were too complex to scale to billions of parameters (Section 1). The second group of methods, the ones used at scale, are variants of round-to-nearest (RTN), including the weight quantizer of LLM.int8() (Section 1 and Section 5, Baselines). RTN works at 8 bits and fails below that.

The failure is easy to see on one layer. The following results come from `scripts/rtn_vs_gptq.py`. Take a random linear layer with 16 output rows and 32 input columns, quantize it to $b$ bits with RTN on a per-row min-max grid, and measure the error of the layer's output on 256 inputs, relative to the output's own size. The inputs are correlated: they come from 4 shared directions plus a little noise, as real activations do. The numbers are means over 20 random layers:

| Bits | RTN output error | GPTQ output error | GPTQ / RTN |
|---|---|---|---|
| 4 | 0.00758 | 0.00092 | 0.121 |
| 3 | 0.03212 | 0.00393 | 0.122 |
| 2 | 0.18922 | 0.02346 | 0.124 |

Both methods use the same grid and the same number of bits. GPTQ's weights differ from RTN's only in which grid point each weight rounds to, and that is enough to cut the output error by a factor of about 8 here. On OPT-175B the same effect appears at full scale: with 3-bit weights, RTN reaches a WikiText2 perplexity of 7.3e3, which is a collapse, and GPTQ reaches 8.68 against 8.34 for the 16-bit model (Table 3).

## Key ideas

This page uses the paper's notation (Section 3, Section 4). A linear layer has weights $W \in \mathbb{R}^{d_{row} \times d_{col}}$, and $X \in \mathbb{R}^{d_{col} \times m}$ holds the layer's inputs for $m$ calibration samples, one sample per column. This is the transpose of the convention in the LLM.int8() page, where $X$ has one row per token. The quantized weights are $\hat{W}$ in Equation 1 and $Q$ in Algorithm 1, and $\mathrm{quant}(w)$ rounds a weight to the nearest point of the quantization grid. The matrix $H$ is the layer Hessian, and $F$ is the set of weights that are still in full precision. One deviation: Algorithm 1 reuses the symbol $H^{-1}$ for the Cholesky factor of the inverse Hessian. This page calls that factor $U$ and keeps $H^{-1}$ for the inverse itself. Every number comes from a script in `scripts/`; run one from the paper folder to reproduce it.

### The layer-wise reconstruction objective

**Intuition.** Don't ask each weight to be close to its original value. Ask the layer to produce nearly the same outputs on real inputs. The weights can then trade errors with one another, which RTN never allows.

**Math.** Let $W_\ell$ be the weights of a linear layer $\ell$, and let $X_\ell$ be its input for a small set of $m$ data points. The goal is a matrix $\hat{W}$ on the quantization grid that minimizes the squared error against the full-precision output (Section 3, Equation 1):

$$
\operatorname*{argmin}_{\hat{W}} \lVert WX - \hat{W}X \rVert_2^2
$$

The grid for each row is fixed before the process starts, and individual weights may move freely among its points (Section 3). The squared norm is a sum over the rows of $W$, so the problem splits into $d_{row}$ independent problems. For one row $w$ and a change $\delta = \hat{w} - w$, the error is $\delta X X^\top \delta^\top$, a quadratic with Hessian

$$
H = 2XX^\top \in \mathbb{R}^{d_{col} \times d_{col}}.
$$

The paper writes this as $H_F = 2X_F X_F^\top$, restricted to the set $F$ of weights that are still free (Section 3). The matrix $H$ depends only on the layer inputs, never on the weights, and it is the same for all rows. The calibration data in the paper are 128 random segments of 2048 tokens from C4, which means GPTQ sees no task-specific data (Section 5, Setup). Section 5 also describes a refinement: the inputs of each layer come from the partially quantized model, not from the full-precision one.

**Tiny example, computed by `scripts/rtn_vs_gptq.py`.** For the layer with seed 0 at 3 bits, the relative layer error is 0.03016 for RTN and 0.00518 for GPTQ. The mean absolute weight error runs the other way, 0.1541 for RTN and 0.1918 for GPTQ. RTN wins on weights and loses on outputs, so the weights are the wrong thing to measure. Exercise 1 builds the RTN baseline and its grid.

### Optimal update after rounding one weight

**Intuition.** Round one weight and look at the error it leaves. The remaining weights can absorb part of that error, because the inputs they multiply overlap with the input of the rounded weight. The best correction is the least-squares solution, and it has a closed form.

**Math.** This idea comes from Optimal Brain Quantization (OBQ), which the paper summarizes in Section 3. OBQ picks the weight $w_q$ to quantize next and the update $\delta_F$ of all the remaining weights as follows (Section 3, Equation 2):

$$
w_q = \operatorname*{argmin}_{w_q} \frac{(\mathrm{quant}(w_q) - w_q)^2}{[H_F^{-1}]_{qq}}, \qquad
\delta_F = -\frac{w_q - \mathrm{quant}(w_q)}{[H_F^{-1}]_{qq}} \cdot (H_F^{-1})_{:,q}
$$

After the step, the row $q$ and column $q$ must leave the problem. OBQ removes them from the inverse directly, with one step of Gaussian elimination (Section 3, Equation 3):

$$
H_{-q}^{-1} = \left( H^{-1} - \frac{1}{[H^{-1}]_{qq}} H^{-1}_{:,q} H^{-1}_{q,:} \right)_{-p}
$$

Here the subscript $-p$ means that the row and column of the removed weight are dropped. The update $\delta_F$ keeps $w_q$ on its grid value, because the entry of $\delta_F$ at position $q$ equals $\mathrm{quant}(w_q) - w_q$. Among all ways to fix that one weight, it is the one with the smallest increase in the error. The following consequence is this page's, not the paper's. The error of Equation 1 grows by $\tfrac{1}{2}\,\delta H \delta^\top = \tfrac{1}{2}(\mathrm{quant}(w_q) - w_q)^2 / [H_F^{-1}]_{qq}$, which is half of the quantity that Equation 2 ranks the candidate weights by. The ranking is the same.

**Tiny example, computed by `scripts/update_example.py`.** Take one row $w = [0.37, -0.82, 0.55, 0.14]$ and the Hessian

$$
H = \begin{bmatrix} 4 & 2 & 1 & 0 \\ 2 & 6 & 2 & 1 \\ 1 & 2 & 6 & 2 \\ 0 & 1 & 2 & 5 \end{bmatrix}.
$$

Quantize $w_0 = 0.37$ on a grid with step $0.25$, which gives $0.25$, an error of $0.12$. The first column of $H^{-1}$ is $[0.3060, -0.0976, -0.0288, 0.0310]$, so $[H^{-1}]_{00} = 0.3060$, and the update is

$$
\delta = -\frac{0.12}{0.3060} \cdot [0.3060, -0.0976, -0.0288, 0.0310]^\top = [-0.1200, 0.0383, 0.0113, -0.0122]^\top.
$$

Check one entry by hand: $-0.12 / 0.3060 \cdot (-0.0976) = 0.0383$. The new row is $[0.25, -0.7817, 0.5613, 0.1278]$. The script also solves the constrained least-squares problem directly, with $w_0$ fixed at $0.25$, and the two answers differ by $3 \times 10^{-18}$. The error of Equation 1 grows by 0.02353 with the update and by 0.02880 if only $w_0$ is rounded and nothing else moves. Exercise 2 asks you to write this update.

### Arbitrary order: one inverse for every row

**Intuition.** OBQ picks the next weight greedily, and that makes every row follow its own order, so each row needs its own sequence of inverses. The paper finds that the greedy order helps little on large layers, because the weights it postpones are quantized late, when few free weights remain to compensate for them. A fixed order for all rows costs little accuracy and shares the work.

**Math.** Section 4, Step 1 says that for large, heavily parametrized layers any fixed order does well. If all rows quantize their weights in the same order, the set $F$ and the matrix $H_F^{-1}$ are the same for all rows, because $H_F$ depends only on $X_F$. The Equation 3 update then runs $d_{col}$ times, once per column, and not $d_{row} \cdot d_{col}$ times, once per weight. The cost falls from $O(d_{row} \cdot d_{col}^3)$ to $O(\max\{d_{row} \cdot d_{col}^2, d_{col}^3\})$, which is a factor of $\min\{d_{row}, d_{col}\}$.

**Tiny example, computed by `scripts/update_counts.py`.** A $16 \times 32$ layer needs 512 inverse updates with a per-weight order and 32 with a shared order, a ratio of 16. A $4096 \times 4096$ layer needs 16,777,216 against 4,096, a ratio of 4,096. The accuracy cost is small, as the table in the Problem section shows: the quantizer in that table is the shared-order one.

### Lazy batch updates

**Intuition.** The direct algorithm touches a large matrix for every column, with only a few operations per entry. A GPU spends its time waiting for memory and not computing. The rounding of column $i$ depends only on the updates that earlier columns made to column $i$. So the updates to columns further right can wait, and be applied together.

**Math.** Process $B$ columns at a time (the paper uses $B = 128$). Inside a block, update only the $B$ columns of the block and the matching $B \times B$ block of $H^{-1}$. After the block, update everything to its right at once, using the multi-weight versions of Equations 2 and 3, with $Q$ a set of indices (Section 4, Step 2, Equations 4 and 5):

$$
\delta_F = -(w_Q - \mathrm{quant}(w_Q))\,([H_F^{-1}]_{QQ})^{-1}\,(H_F^{-1})_{:,Q}, \qquad
H_{-Q}^{-1} = \left( H^{-1} - H^{-1}_{:,Q}\,([H^{-1}]_{QQ})^{-1}\,H^{-1}_{Q,:} \right)_{-Q}
$$

The paper notes that this does not reduce the theoretical amount of compute, that it removes the memory-throughput bottleneck, and that it gives an order-of-magnitude speedup for very large models (Section 4, Step 2). Because it only reschedules the same updates, it must not change the answer. Algorithm 1 writes the block update as $W_{:,(i+B):} \leftarrow W_{:,(i+B):} - E \cdot H^{-1}_{i:(i+B),(i+B):}$, with $H^{-1}$ standing for the Cholesky factor.

**Tiny example, computed by `scripts/lazy_blocks.py`.** Quantize a $16 \times 32$ layer to 3 bits with block sizes 1, 4, 8, 16, and 32. The largest difference between any blocked result and the unblocked one is 0, and the layer loss is 2242.6091 for all of them. Exercise 4 asks you to write the block loop.

### Cholesky reformulation and dampening

**Intuition.** Repeated applications of Equation 3 accumulate rounding errors in $H^{-1}$. For models beyond a few billion parameters, the matrix $H_F^{-1}$ becomes indefinite in at least a few layers, and the algorithm then pushes the other weights in wrong directions (Section 4, Step 3). The fix is to stop updating the inverse. The algorithm needs only one row of each successive inverse, and a Cholesky decomposition delivers all those rows at once.

**Math.** Only row $q$ of $H^{-1}_{F_q}$ is needed when column $q$ is quantized, starting at the diagonal, where $F_q$ is the set of weights still unquantized at that moment. The row removal of Equation 3 for a symmetric $H^{-1}$ is essentially a Cholesky decomposition, except that the Cholesky factor divides row $q$ by $([H^{-1}_{F_q}]_{qq})^{1/2}$ (Section 4, Step 3). Let $U$ be the upper triangular matrix with $U^\top U = H^{-1}$, which is the transpose of the lower Cholesky factor of $H^{-1}$. Then, for each $j$,

$$
U_{jj}^2 = [H^{-1}_{F_j}]_{jj}, \qquad U_{j,\,j:} = \frac{(H^{-1}_{F_j})_{j,\,j:}}{\sqrt{[H^{-1}_{F_j}]_{jj}}}.
$$

This is the paper's remark, written with $U$. Dividing the error by $U_{jj}$ and multiplying by row $j$ of $U$ gives the same step as Equation 2, since $(w - \mathrm{quant}(w))/U_{jj} \cdot U_{j,:}$ equals $(w - \mathrm{quant}(w))/[H^{-1}_{F_j}]_{jj} \cdot (H^{-1}_{F_j})_{j,:}$. Dampening adds a small constant $\lambda$ to the diagonal of $H$, and the paper always uses 1% of the average diagonal value (Section 4, Step 3). The paper reports that dampening alone is enough for smaller models, and that the Cholesky form is what makes the largest models work. The paper also reports that a well-optimized Cholesky kernel gives a further speedup.

**Tiny example, computed by `scripts/cholesky_example.py`.** With the $4 \times 4$ Hessian of the previous example,

$$
U = \begin{bmatrix} 0.5532 & -0.1764 & -0.0521 & 0.0561 \\ 0 & 0.4341 & -0.1336 & -0.0334 \\ 0 & 0 & 0.4385 & -0.1754 \\ 0 & 0 & 0 & 0.4472 \end{bmatrix}.
$$

The script removes one column after another with Equation 3. The diagonal values $[H^{-1}_{F_j}]_{jj}$ are $0.3060, 0.1884, 0.1923, 0.2000$. They match $U_{jj}^2$ (for $j = 0$, $0.5532^2 = 0.3060$), and each row divided by the square root of its diagonal equals the row of $U$ to $10^{-16}$. The first step of the previous example, $-(0.12 / 0.5532)\,U_{0,:}$, gives the same $\delta$ as before.

### The full algorithm

**Intuition.** Put the pieces together. Fix the grid for each row from the original weights. Compute $U$ once for the layer. Sweep the columns in order, in blocks. For each column, round, scale the error by $U_{jj}$, and push it onto the remaining columns of the block with row $j$ of $U$. After each block, push the block's errors onto everything to its right.

**Math.** Algorithm 1 in Section 4 takes the inverse Hessian $H^{-1} = (2XX^\top + \lambda I)^{-1}$ and block size $B$, and computes the Cholesky form first. In this page's notation, with $U$ the upper Cholesky factor:

$$
\begin{aligned}
&\textbf{for } i = 0, B, 2B, \ldots \textbf{ do} \\
&\quad \textbf{for } j = i, \ldots, i + B - 1 \textbf{ do} \\
&\quad\quad Q_{:,j} \leftarrow \mathrm{quant}(W_{:,j}) \\
&\quad\quad E_{:,j-i} \leftarrow (W_{:,j} - Q_{:,j}) / U_{jj} \\
&\quad\quad W_{:,j:(i+B)} \leftarrow W_{:,j:(i+B)} - E_{:,j-i} \cdot U_{j,\,j:(i+B)} \\
&\quad \textbf{end for} \\
&\quad W_{:,(i+B):} \leftarrow W_{:,(i+B):} - E \cdot U_{i:(i+B),\,(i+B):} \\
&\textbf{end for}
\end{aligned}
$$

The algorithm in this form needs $O(\max\{d_{row} \cdot d_{col}^2, d_{col}^3\})$ operations per layer. The paper loads one transformer block, with 6 linear layers, into GPU memory at a time, accumulates the layer Hessians, quantizes, and then runs the block's inputs through the quantized block to make the inputs of the next one (Section 5, Setup).

**Tiny example, computed by `scripts/rtn_vs_gptq.py`.** The Problem section's table is this algorithm, with $B$ equal to the number of columns, on the grid of Exercise 1. Exercise 3 asks you to write the single-block version, and Exercise 4 adds the blocks.

### What the result shows, and what it doesn't

The paper's Table 3 gives WikiText2 perplexity for the OPT family. The rows below show a selection of sizes:

| OPT model | 1.3B | 6.7B | 30B | 66B | 175B |
|---|---|---|---|---|---|
| 16-bit | 14.63 | 10.86 | 9.56 | 9.34 | 8.34 |
| RTN, 4-bit | 48.17 | 12.10 | 10.98 | 110 | 10.54 |
| GPTQ, 4-bit | 15.47 | 11.39 | 9.63 | 9.55 | 8.37 |
| RTN, 3-bit | 1.3e4 | 5.8e3 | 1.6e3 | 6.1e3 | 7.3e3 |
| GPTQ, 3-bit | 20.97 | 14.86 | 10.27 | 14.16 | 8.68 |

At 4 bits, GPTQ loses 0.03 perplexity on OPT-175B, where RTN loses 2.2 points and ends up worse than the 10 times smaller 13B model. The paper notes that larger models generally are easier to quantize, with OPT-66B as an exception that it links to a large share of dead units in its early layers (Section 5, footnote 2). Quantizing the 175B models takes about four GPU hours on one A100 (Table 2: 4.2 hours for OPT-175B and 3.8 hours for BLOOM-176B). A 3-bit OPT-175B takes about 63 GB, which fits a single 80 GB A100 together with about 9 GB of key-value cache, and the paper's custom kernels give an average per-token latency of 71 ms against 230 ms for 16-bit on five A100s (Table 6, a 3.24 times speedup). On A6000 GPUs, the paper reports 130 ms against 589 ms, a 4.53 times speedup.

Grouping the weights gives further accuracy. With 2-bit weights, a group size of 128 costs about 2.2 bits per weight on average and raises the OPT-175B perplexity from 8.34 to 9.58, and a group size of 32 raises it to 8.94 (Table 7).

The limits are stated in Section 1 and Section 6. GPTQ gets its speedup from reduced memory movement and not from fewer computations: the multiplications still run in higher precision, because mainstream hardware lacks mixed-precision operands such as FP16 by INT4. The study covers generative tasks, and it doesn't quantize activations. The embeddings and the output layer stay in FP16 (Section 5, Practical Speedups).

<!-- exercises -->

## Pitfalls

- **Measuring the wrong error.** GPTQ increases the error of the weights while it decreases the error of the layer output (the Problem section and the third self-check). Judge a quantizer by $\lVert WX - \hat{W}X \rVert^2$ on held-out inputs or by the perplexity of the model, not by $\lVert W - \hat{W} \rVert$.
- **Mixing up the layout of $X$.** In the paper $X$ is $d_{col} \times m$, so $H = 2XX^\top$ is $d_{col} \times d_{col}$. A library that stores inputs as $m \times d_{col}$ needs $X^\top X$ instead. The wrong product has the wrong size or, for square cases, the wrong meaning.
- **Quantizing from the original weights.** Each column must be rounded from the current weights, after the updates of all earlier columns. Rounding the original weights and then adding the updates gives a result that is not on the grid.
- **Moving the grid.** The grid is fixed before the sweep (Section 3). Recomputing the min and max from the updated weights shifts the grid under the algorithm, and the error scaled by $U_{jj}$ no longer matches the grid that the rounding used. Grouping is the exception: the paper computes group parameters during the sweep, from the current weights, when the sweep reaches the group (Section 5).
- **Using the lower Cholesky factor.** The algorithm needs the upper factor $U$ with $H^{-1} = U^\top U$, because row $j$ of $U$ is zero to the left of the diagonal and so only touches the columns not yet quantized. A row of the lower factor is nonzero to the left of the diagonal and would change columns that are already quantized.
- **Factoring $H$ instead of $H^{-1}$.** The Cholesky factor in Algorithm 1 is of the inverse Hessian. Factoring $H$ itself gives a plausible-looking matrix that has a different meaning.
- **Skipping dampening.** A column of $X$ that is all zero, or two perfectly correlated columns, makes $H$ singular. The constant $\lambda$ (1% of the average diagonal in the paper) keeps the inverse defined, and the paper reports that large models need the Cholesky form in addition.
- **Forgetting the block update.** The in-block updates only reach columns inside the block. Without the final update of the columns to the right of the block, the later blocks never see the earlier errors, and the result no longer matches the unblocked algorithm (Exercise 4).
- **Treating the speedup as a compute saving.** The paper's speedups come from reading fewer bytes of weights during generation. It says the method does not reduce the computation (Section 6).
- **Reading the greedy order as essential.** The paper finds that the greedy order of OBQ is not needed for large layers (Section 4, Step 1). On small layers the difference can be larger, and the paper doesn't claim otherwise.

## Further reading

- [IST-DASLab/gptq](https://github.com/IST-DASLab/gptq), the paper's official implementation, which includes the quantization code, perplexity evaluation, and the 3-bit CUDA kernel.
- [Optimal Brain Compression](https://arxiv.org/abs/2208.11580), the framework behind OBQ, which the paper builds on (Section 3). It is the source of Equations 2 and 3.
- [LLM.int8()](/llm-int8/), the round-to-nearest weight quantizer that GPTQ uses as its baseline, with outlier handling for activations.
- [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) and [Hugging Face Optimum](https://huggingface.co/docs/optimum/index), which package GPTQ for use with popular model libraries.
