---
id: "precision-scaling"
title: "Scaling Laws for Precision"
authors: "Kumar et al."
year: 2024
venue: "arXiv preprint"
arxiv: "2411.04330"
url: "https://arxiv.org/abs/2411.04330"
track: scaling-efficiency
tier: extended
est_hours: 4
status: full
summary: "The paper extends the Chinchilla loss law with numerical precision. It treats training in fewer bits as a smaller model with an effective parameter count, and it adds a separate penalty for quantizing a model after training that grows with the ratio of training tokens to parameters. The fits predict that a fixed compute budget is best spent at about 7 to 8 bits, that going much below 4 bits needs far larger models, and that extra pretraining data can eventually hurt a model that you quantize afterward."
related_course: "ml: scaling laws"
prereqs:
  - id: chinchilla
    reason: "The paper keeps Chinchilla's loss $E + A/N^{\\alpha} + B/D^{\\beta}$ and its compute accounting, and it changes only what $N$ means and what a unit of compute costs."
  - id: gptq
    reason: "The post-training half of the paper quantizes trained models with GPTQ, so you need to know what that quantizer does to a trained weight matrix and why fewer bits add error."
  - id: fp8
    reason: "The low-precision training half quantizes weights, activations, and the attention KV cache, and the paper's floating-point runs use formats such as FP4 and FP8 that this paper defines."
background: [power-laws, flops-estimate-6nd, curve-fitting, floating-point-formats, quantization]
course_links:
  - title: "Scaling laws and the compute budget"
    url: "https://ml.ifkash.dev/lessons/073-scaling-laws"
  - title: "FP8, MXFP4, and NVFP4"
    url: "https://gpu.ifkash.dev/c/33-fp8-and-fp4"
self_check:
  - q: 'A model has $N = 10^9$ parameters and the weights use 4 bits. The paper models the weights as $N_{\text{eff}} = N(1 - e^{-P_w/\gamma_w})$ with $\gamma_w = 2.6745$. How many effective parameters does the model have, and what happens at 8 bits?'
    a: 'At 4 bits, $1 - e^{-4/2.6745} = 0.7759$, so $N_{\text{eff}} \approx 7.76 \times 10^8$. At 8 bits the factor is 0.9498. The first bits matter most, and the curve saturates (Section 4.1).'
    see: chinchilla
  - q: 'Chinchilla spends a budget as $C = 6ND$. The paper writes $C = \tfrac{6}{16} N D P$. What does the second form say about a model trained at 8 bits, and why does it help to lower $P$?'
    a: 'The second form reduces to $6ND$ at $P = 16$. At 8 bits the same budget buys twice as many $N \cdot D$ products, so you can train a larger model or use more tokens. Lowering $P$ trades loss from the smaller $N_{\text{eff}}$ against the extra size and data (Section 4.3).'
    see: flops-estimate-6nd
  - q: 'Why does the paper say that more pretraining data can hurt a model that you quantize after training?'
    a: 'The loss added by post-training quantization grows as $D^{\gamma_D}/N^{\gamma_N}$ with $\gamma_D \approx 0.51$ and $\gamma_N \approx 0.34$ (Equation 2), roughly as a power of $D/N$. The benefit of data falls as $D^{-\beta}$. Past a critical $D$, the added quantization loss grows faster than the pretraining loss falls (Section 3.1, Finding 1).'
    see: gptq
---

## The problem

Chinchilla treats every parameter as a full-precision number and every FLOP as equal. Neither assumption holds in practice. Teams train in BF16, deploy in 4 or 8 bits, and ask whether going lower is free.

Consider two failing cases. First, take a model that you train for longer to improve it, and then quantize with GPTQ. The paper's measurements show that the loss added by quantization grows with the number of training tokens (Section 3.1, Finding 1). A model that saw many tokens per parameter degrades more than one that saw few. At some point, the extra tokens lower the loss before quantization and raise it after.

Second, take a team that wants to train in 4 bits to save compute. A Chinchilla-style plan has no term for the precision, so it can't say how much larger the 4-bit model must be to match an 8-bit one.

The paper adds precision to the loss law. It fits the result on more than 465 pretraining runs of OLMo-style Transformer++ models with 30M to 220M non-embedding parameters, trained on 1.5B to 26B tokens of Dolma, with weights, activations, and KV cache each quantized to 3 to 12 bits (Section 2.3, Section 4). It then checks the compute-optimal result on floating-point runs up to 1.6B parameters (Section 4.3.2).

## Key ideas

The sections that follow use the paper's notation. $N$ is the number of non-embedding parameters and $D$ is the number of training tokens. $P_w$, $P_a$, and $P_{kv}$ are the bit widths of the weights, the activations, and the KV cache during training. $P_{\text{post}}$ is the bit width that post-training quantization uses. Every number below comes from a script in `scripts/`. Run a script from the paper folder to reproduce the output.

The paper ties the exponents, $\alpha = \beta$ (Appendix K), and calls its constants unlikely to transfer to other architectures or datasets. The constants in Appendix K, Table 2 are what this page uses. Read the numbers as a demonstration of the shape of the laws.

### Fewer bits make a smaller model

**Intuition.** A model trained with 4-bit weights isn't a worse learner with the same size. The paper models it as a model with fewer parameters. The first few bits recover most of the capacity, and later bits add little.

**Math.** Section 4.1 fixes $D$ and sweeps $N$ against the weight precision $P_w$. It finds that the best-fitting effective parameter count is

$$
N_{\text{eff}}(N, P_w) = N\left(1 - e^{-P_w/\gamma_w}\right),
$$

where $\gamma_w$ is a fitted sensitivity constant. Substituting $N_{\text{eff}}$ for $N$ in the Chinchilla loss gives (Section 4.1, Equation 3)

$$
L(N, D, P_w) = A\left[N\left(1 - e^{-P_w/\gamma_w}\right)\right]^{-\alpha} + B D^{-\beta} + E.
$$

Appendix M shows how to read $N_{\text{eff}}$ from a measured loss. The $B$, $D$, and $E$ terms cancel in the loss gap between a run at precision $P_w$ and a full-precision run, which leaves $A[N_{\text{eff}}^{-\alpha} - N^{-\alpha}]$. Solving for $f = N_{\text{eff}}/N$ gives Equation 30. Exercise 2 implements it.

**Tiny example, computed by `scripts/n_eff.py`.** With $\gamma_w = 2.6745$ from Table 2, the factor $1 - e^{-P_w/\gamma_w}$ is 0.5266 at 2 bits, 0.6743 at 3 bits, 0.7759 at 4 bits, 0.8939 at 6 bits, 0.9498 at 8 bits, and 0.9975 at 16 bits. A 1B-parameter model with 4-bit weights behaves like a 776M-parameter model. Going from 8 to 16 bits adds 0.0477, and going from 2 to 4 bits adds 0.2493. Exercise 1 implements the factor.

### Weights, activations, and KV cache multiply

**Intuition.** The model has three places where rounding adds noise. The paper finds that the three effects act independently, so the factors multiply.

**Math.** Section 4.2 quantizes all three (Finding 2, Equation 4):

$$
N_{\text{eff}}(P) = N\left(1 - e^{-P_w/\gamma_w}\right)\left(1 - e^{-P_a/\gamma_a}\right)\left(1 - e^{-P_{kv}/\gamma_{kv}}\right).
$$

When all three share one precision $P$, the paper simplifies this to $N(1 - e^{-P/\bar\gamma})^3$, where $\bar\gamma$ is the mean of the three constants (Section 4.2). Table 2 lists $\gamma_w = 2.6745$, $\gamma_i = 2.2102$, and $\gamma_{kv} = 0.9578$. This page reads $\gamma_i$ as the activation constant $\gamma_a$ of the main text, which is an assumption about the label. Appendix K adds that the fits include an offset: a term $e^{-x/\gamma_x}$ is computed as $e^{-x/\gamma_x + n}$, with $n_w = 0.3037$, $n_i = 1.4072$, and $n_{kv} = 2.4185$. This page reads the offset as a shift of the exponent, and the next sections check that reading against the paper's results. Appendix L adds that runs with $P_a \le 3$ or $P_{kv} \le 3$ often diverge.

**Tiny example, computed by `scripts/n_eff.py`.** The mean is $\bar\gamma = 1.9475$, and the shared-precision form gives $N_{\text{eff}}/N$ of 0.4851 at 3 bits, 0.6625 at 4 bits, 0.8685 at 6 bits, and 0.9515 at 8 bits. The order of the precisions matters because the constants differ. For $(P_w, P_a, P_{kv})$ equal to $(8, 4, 8)$, $(4, 8, 8)$, and $(8, 8, 4)$, the ratio $N_{\text{eff}}/N$ is 0.7941, 0.7549, and 0.9101. Four-bit weights cost the most, and 4-bit KV cache costs the least, which is the order of the constants. With the Table 2 offsets and all three at 4 bits, the ratio is 0.1910. The activation factor at 3 bits is $-0.0511$ with the offset, so the form stops being valid there, which matches the divergence that Appendix L reports.

### A smaller model needs more parameters

**Intuition.** If low precision cuts the effective size, a low-precision model reaches the same loss only if you make it bigger. The loss law tells you how much bigger.

**Math.** With uniform precision, the loss is (Section 4.3, Equation 5)

$$
L(N, D, P) = A\left[N\left(1 - e^{-P/\bar\gamma}\right)^3\right]^{-\alpha} + B D^{-\beta} + E.
$$

Section 4.3.1 concludes that if you must train in low precision, you increase parameters before data (Equation 6): the optimal $N$ grows and the optimal $D$ shrinks, compared with Chinchilla, as $P$ falls.

**Tiny example, computed by `scripts/loss_precision.py`.** The script uses the Table 2 constants with offsets and an $N = 220$M model on $D = 26$B tokens. The loss is 3.3596 at 5 bits, 3.2824 at 6 bits, 3.2271 at 8 bits, and 3.1976 at 16 bits. The data term is 0.1218 in all four rows, so the whole difference comes from the size term. A 5-bit model needs $N_{\text{eff}}(8)/N_{\text{eff}}(5) = 1.938$ times the parameters of an 8-bit model to reach the same loss. With $N = 100$M at 8 bits and $N = 193.8$M at 5 bits, both reach 3.3903. Exercise 1 implements this loss.

### The compute-optimal precision

**Intuition.** Lower precision shrinks the effective model, and it makes each step cheaper. You can spend the savings on more parameters and more tokens. At some bit width the two effects balance.

**Math.** The paper costs a run as $C = \tfrac{6}{16} N D P$, which equals Chinchilla's $6ND$ at $P = 16$ (Section 4.3). It minimizes Equation 5 under that constraint over $N$, $D$, and $P$ (Section 4.3.2). The optimum satisfies a condition on $P$ alone (Equation 7), so it doesn't depend on the compute budget $C$ (Finding 3).

This page derives the same result for the tied case $\alpha = \beta$. Set $c = 16C/6$, so that $D = c/(NP)$. Then the loss minus $E$ is $a N^{-\alpha} + b N^{\alpha}$, with $a = A f(P)^{-3\alpha}$ and $b = B (P/c)^{\alpha}$. Its minimum over $N$ is $2\sqrt{ab}$. The only factor that depends on $P$ is $P^{\alpha} f(P)^{-3\alpha}$, so the optimal precision minimizes $P / f(P)^3$, with $f(P) = 1 - e^{-P/\bar\gamma}$. Setting its derivative to zero with $x = P/\bar\gamma$ gives

$$
e^{x} - 1 = 3x, \qquad P^{*} = x^{*}\,\bar\gamma .
$$

The condition has no $A$, $B$, $E$, $\alpha$, or $C$ in it. The paper reports $P^{*}$ at about 7 to 8 bits for integer quantization, and it says that 16 bits wastes bits while going below 4 bits needs a disproportionately larger model (Finding 3). Section 4.3.3 adds that when $N$ is fixed in advance, the optimal precision grows with $\log C$.

**Tiny example, computed by `scripts/optimal_precision.py`.** The root of $e^x - 1 = 3x$ is $x^* = 1.9038$. With $\bar\gamma = 1.9475$ and no offsets, $P^* = 3.708$ bits, and a brute-force search over $N$ and $P$ in steps of half a bit picks 3.5 bits at $C = 10^{19}$, $10^{21}$, and $10^{23}$. That is lower than the paper's 7 to 8 bits. The offsets explain the gap. Minimizing $P/(f_w f_a f_{kv})$ with the Table 2 offsets gives $P^* = 7.002$ bits, and the brute-force search over $N$ and $P$ also picks 7.0 at all three budgets. This matches the paper's range and supports the reading of the offset as a shift of the exponent. At $C = 10^{21}$ with offsets, the optimal model has $N^* = 8.33 \times 10^9$ at 5 bits, $5.35 \times 10^9$ at 7 bits, $4.73 \times 10^9$ at 8 bits, and $3.05 \times 10^9$ at 16 bits, with losses 2.9205, 2.9125, 2.9133, and 2.9333. Lower precision favors a larger model. The loss is flat near the optimum, and 16 bits costs 0.0208 more than 7 bits. Exercise 3 implements $P^*$ and the search.

### Post-training quantization degrades overtrained models

**Intuition.** A model trained on many tokens packs more information into each weight, so rounding the weights hurts it more.

**Math.** The paper trains in BF16 and quantizes with GPTQ afterward (Section 3). It defines $\delta_{\text{PTQ}}$ as the loss increase relative to the end of pretraining and fits (Section 3.1, Equation 2)

$$
\delta_{\text{PTQ}}(N, D, P_{\text{post}}) = C_T\,\frac{D^{\gamma_D}}{N^{\gamma_N}}\,e^{-P_{\text{post}}/\gamma_{\text{post}}}.
$$

Because $\gamma_D$ and $\gamma_N$ are close, the term behaves like a power law in $D/N$. The degradation grows exponentially as $P_{\text{post}}$ falls. Total loss after quantization adds $\delta_{\text{PTQ}}$ to the training-time law, with the training precision and $N_{\text{eff}}$ entering $\delta_{\text{PTQ}}$ (Section 5, Equations 9 and 11). The paper reports $R^2 = 0.90$ for this unified fit on more than 1000 points (Section 5, Finding 4).

Finding 1 follows. Pretraining loss falls as $B D^{-\beta}$ while $\delta_{\text{PTQ}}$ rises as $D^{\gamma_D}$, so the sum has a minimum. Setting its derivative to zero gives the critical dataset size (Appendix E.1, Equations 12 and 13)

$$
D_{\text{crit}} = \left(\frac{\beta B N^{\gamma_N} e^{P_{\text{post}}/\gamma_{\text{post}}}}{\gamma_D C_T}\right)^{1/(\gamma_D + \beta)}.
$$

Table 2 doesn't label $\gamma_{\text{post}}$. This page takes the unlabeled row $\gamma = 0.5907$, which follows $C_T$, $\gamma_D$, and $\gamma_N$, to be $\gamma_{\text{post}}$. With $C_T = 0.0598$, $\gamma_D = 0.5068$, and $\gamma_N = 0.3439$, the resulting $D_{\text{crit}}$ is consistent with the paper's statement that the effect is negligible at 8 bits and significant near 5 bits for heavily overtrained models (Section 3.1).

**Tiny example, computed by `scripts/ptq.py`.** At 4 bits, $\delta_{\text{PTQ}}$ is 0.0082 for $N = 30$M and $D = 1.5$B, and 0.0349 for $N = 30$M and $D = 26$B. A 220M model on the same 26B tokens has 0.0176. Tenfold more tokens multiplies $\delta_{\text{PTQ}}$ by $10^{0.5068} = 3.212$, tenfold more parameters multiplies it by $10^{-0.3439} = 0.453$, and one more bit multiplies it by 0.1840. For $N = 30$M, $D_{\text{crit}}$ is $8.86 \times 10^{10}$ tokens at 4 bits, $4.79 \times 10^{11}$ at 5 bits, and $7.56 \times 10^{13}$ at 8 bits. Those correspond to $D_{\text{crit}}/N$ of 2952, $1.60 \times 10^4$, and $2.52 \times 10^6$. The script checks the closed form by confirming that the slope of $B D^{-\beta} + \delta_{\text{PTQ}}$ at $D_{\text{crit}}$ is zero. Exercise 4 implements $\delta_{\text{PTQ}}$ and $D_{\text{crit}}$.

<!-- exercises -->

## Pitfalls

- **Dropping the offsets.** The main text writes the saturating exponential without offsets, and Appendix K says the fitted values include them. With Table 2 and no offsets, the optimal precision is 3.7 bits. With the offsets, it is 7.0 bits. A reader who copies only the main-text equation gets a different headline result from the same constants.
- **Treating the constants as universal.** Appendix K says that fitted constants depend heavily on the architecture and the dataset. Use the shapes: a saturating factor in $P$, a multiplicative composition, and a power law in $D/N$ for the PTQ term.
- **Reading $P^*$ as a recommendation for your cluster.** The paper's cost model assumes compute scales linearly with the bit width. Section 6 notes that halving the precision usually gives less than a 2x speedup in practice. It also measures loss only, uses a single architecture, and simulates quantization.
- **Confusing the two optima.** With $N$, $D$, and $P$ all free, $P^*$ doesn't depend on compute (Finding 3). With $N$ fixed in advance, as in a model family trained on similar data, $P^*$ grows with $\log C$ (Section 4.3.3). Don't mix the two statements.
- **Mixing training precision and post-training precision.** $P_w$, $P_a$, and $P_{kv}$ are bit widths during training. $P_{\text{post}}$ is the bit width after training. The paper's PTQ term contains both, and a model that trained at low precision is more robust to later quantization (Section 5, Equation 9).
- **Extrapolating a fit past its data.** The experiments reach $D/N$ of about $10^3$ for 30M to 220M models, and the paper's predictions extrapolate beyond that. Runs with activations or KV cache at 3 bits or fewer often diverge, so the paper doesn't fit them (Appendix L).
- **Assuming integer and floating-point bits behave alike.** In floating point, exponent and mantissa bits play different roles. Appendix N fits the same form to floating-point runs with a different $\gamma_w$ and finds that standard bit allocations still follow it.
- **Reading numbers off this page as the paper's.** The $\bar\gamma$ values, the loss tables, and the $D_{\text{crit}}$ values on this page are evaluations of the paper's formulas with the Table 2 constants. The paper doesn't print them.

## Further reading

- [Scaling Laws for Precision](https://arxiv.org/abs/2411.04330), by Kumar et al., is the paper. Appendix E has the derivations and Appendix K has the fitted constants.
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556), by Hoffmann et al., is the law that this paper extends. See [the Chinchilla page](/chinchilla/) on this site.
- [GPTQ](https://arxiv.org/abs/2210.17323), by Frantar et al., is the post-training quantizer the paper uses. See [the GPTQ page](/gptq/).
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433), by Micikevicius et al., defines the 8-bit formats. See [the FP8 page](/fp8/).
- [Scaling laws and the compute budget](https://ml.ifkash.dev/lessons/073-scaling-laws) is the course lesson on the underlying fits.
- [FP8, MXFP4, and NVFP4](https://gpu.ifkash.dev/c/33-fp8-and-fp4) is the course chapter on the low-precision formats.
