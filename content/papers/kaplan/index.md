---
id: "kaplan"
title: "Scaling Laws for Neural Language Models"
authors: "Kaplan et al."
year: 2020
venue: "arXiv preprint"
arxiv: "2001.08361"
url: "https://arxiv.org/abs/2001.08361"
track: scaling-efficiency
tier: core
est_hours: 3
status: full
summary: "The paper trains many decoder-only Transformers on WebText2 and finds that test loss falls as a smooth power law in the number of non-embedding parameters, the number of training tokens, and the compute budget, across many orders of magnitude, while depth, width, and head count matter very little. One combined equation in parameters and data predicts how much a model overfits, and a second equation in parameters and training steps predicts the compute-optimal split between model size and training time. For a fixed budget, the fits say to spend most of the extra compute on a larger model and stop well before convergence, a recommendation that later work revised."
related_course: "ml: scaling laws"
prereqs:
  - id: gpt2
    reason: "Every model in the paper is a GPT-2-style decoder-only Transformer trained on WebText2 with the same BPE vocabulary, and the loss it measures is the next-token cross-entropy that GPT-2 defines."
background: [power-laws, log-log-regression]
course_links:
  - title: "Scaling laws and the compute budget"
    url: "https://ml.ifkash.dev/lessons/055-scaling-laws"
self_check:
  - q: 'A language model has a mean next-token cross-entropy of $3.0$ nats on a test set. What perplexity is that, and which unit does the Kaplan paper report its loss in?'
    a: 'The perplexity is $e^{3.0} \approx 20.1$. The paper reports the cross-entropy loss in nats, averaged over the tokens of a 1024-token context (Section 1.3 and Section 2).'
    see: gpt2
  - q: 'A loss follows $L(X) = (X_c / X)^{\alpha}$. What happens to $L$ when you double $X$, and what does the curve look like on log-log axes?'
    a: 'Doubling $X$ multiplies $L$ by $2^{-\alpha}$, whatever the starting $X$ is. On log-log axes the curve is a straight line with slope $-\alpha$, because $\ln L = \alpha \ln X_c - \alpha \ln X$.'
    see: power-laws
  - q: 'A small study measures a test loss of $4.0$ at $10^6$ parameters and $3.4$ at $10^7$ parameters. Estimate the exponent $\alpha$ of a power law through the two points.'
    a: 'The slope on log-log axes is $-\alpha = \log_{10}(3.4 / 4.0) / \log_{10}(10^7 / 10^6) = -0.0706$, so $\alpha \approx 0.071$. A least-squares line through many points in $(\ln X, \ln L)$ does the same job and averages out noise.'
    see: log-log-regression
---

## The problem

By 2019, people trained language models by guessing. Should the next run use a wider model, a deeper one, or more data? A team with a fixed GPU budget had no way to answer without trying each choice. A run that cost a week told you little about a run that would cost a year.

Here is a concrete case. You have a budget of 1 PF-day, which is $8.64 \times 10^{19}$ floating-point operations. You can train a 100-million-parameter model for many steps, or a 1-billion-parameter model for few steps. Both spend the same compute. Which reaches a lower loss? Before this paper, the answer came from habit, because nobody had a formula that connected loss to model size, data, and compute.

The paper measures that connection. It trains Transformers from about $10^3$ to $1.5 \times 10^9$ non-embedding parameters on WebText2 (Section 2.3), on datasets from $2.2 \times 10^7$ to $2.3 \times 10^{10}$ tokens (Section 3), and fits simple equations to the test loss. The fits hold over many orders of magnitude, so a small run can predict a large one. The practical consequence is in Section 6: for a fixed budget, train a large model and stop early.

## Key ideas

The sections that follow use the paper's notation. $L$ is the cross-entropy test loss in nats. $N$ is the number of non-embedding parameters, $D$ is the dataset size in tokens, $C$ is the training compute, $B$ is the batch size in tokens, and $S$ is the number of parameter updates (Section 1.3). The symbols $\alpha_X$ are power-law exponents, and $X_c$ are the scale constants. The paper's equation labels use the form $(1.1)$, so this page writes "Equation 1.1". Every number comes from a script in `scripts/`. Run a script from the paper folder to reproduce the output.

### Loss is a power law in the number of parameters

**Intuition.** Make a model bigger and its loss falls, but by a smaller amount each time. Plot loss against parameters on log-log axes and the points fall on a straight line, so each doubling of $N$ removes the same percentage of loss.

**Math.** For a model trained to convergence on enough data, the paper fits (Section 1.2, Equation 1.1; Section 3.2, Equation 3.1)

$$
L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \qquad \alpha_N \approx 0.076,\quad N_c \approx 8.8 \times 10^{13}.
$$

Two details matter. First, $N$ counts only non-embedding parameters. With embeddings included, the depth of the model shifts the curve (Figure 6, left), and with embeddings excluded the curves for different depths collapse onto one line (Figure 6, right). Second, $N_c$ depends on the tokenizer, so it has no fundamental meaning (Section 1.2). The exponent is the part to compare across studies.

The paper also finds that the shape barely matters. With $N$ held fixed, the loss moves by only a few percent as $n_\text{layer}$, $n_\text{heads}$, and $d_\text{ff}$ change (Section 3.1, Figure 5). A $(6, 4288)$ model comes within 3% of a $(48, 1600)$ one.

**Tiny example, computed by `scripts/loss_vs_n.py`.** Take the constants above:

| $N$ | $L(N)$ |
|---|---|
| $10^6$ | 4.0159 |
| $10^7$ | 3.3712 |
| $10^8$ | 2.8300 |
| $10^9$ | 2.3756 |
| $10^{10}$ | 1.9943 |

Doubling $N$ multiplies the loss by $2^{-0.076} = 0.9487$, which matches the 0.95 in Section 1.2. A tenfold increase multiplies it by $10^{-0.076} = 0.8395$. To cut the loss by 10%, you need $0.9^{-1/0.076} = 4.00$ times as many parameters. To halve the loss, you need about $9.1 \times 10^3$ times as many. The curve rewards scale, but each step costs more than the last.

### Fitting a power law in log space

**Intuition.** A power law is a straight line after you take logs of both axes, so you can fit its two constants with an ordinary line fit. Fit the small runs, then read off the prediction for a run you haven't paid for.

**Math.** Write $\ln L = \alpha \ln X_c - \alpha \ln X$. A least-squares line through $(\ln X_i, \ln L_i)$ has slope $m$ and intercept $b$, and then

$$
\alpha = -m, \qquad X_c = \exp(b / \alpha).
$$

For a new point $x$, the one-sigma uncertainty of a single new measurement, in log units, is

$$
\sigma(x)^2 = s^2\left(1 + \frac{1}{n} + \frac{(\ln x - \overline{\ln X})^2}{S_{xx}}\right),
$$

where $s^2$ is the residual variance with $n - 2$ degrees of freedom and $S_{xx} = \sum (\ln X_i - \overline{\ln X})^2$. This is standard regression, not a formula from the paper. The paper doesn't give error bars for its extrapolations. It tests them by training the larger runs.

**Tiny example, computed by `scripts/fit_power_law.py`.** The script draws losses from Equation 1.1 at 11 sizes from $10^4$ to $10^9$, with 1% noise, and fits only the 8 smallest ($10^4$ through $10^{7.5}$). The fit returns $\hat\alpha_N = 0.0748$ against the true $0.076$, with residual scatter $s = 0.0053$ in log units. Then it predicts the three models it held out:

| $N$ | observed | predicted | one-sigma, log units |
|---|---|---|---|
| $10^8$ | 2.8101 | 2.8574 | 0.0067 |
| $3.16 \times 10^8$ | 2.5603 | 2.6216 | 0.0072 |
| $10^9$ | 2.3609 | 2.4052 | 0.0078 |

The fit extrapolates $10^{1.5}$ times past its largest training point, and the observed losses land within 2.4 standard deviations of the line. A fit with 8 points underestimates the noise: the true scatter is 0.01 and the fit reports 0.0053. Treat error bars from a few runs as a lower bound.

A common wrong fit uses $\ln L$ against $N$ and not $\ln N$. On the same data it predicts a loss of about 0 at $10^9$ parameters, against the observed 2.36. Exercise 1 asks you to write the correct fit and its error bar.

### Compute is about 6 N per token

**Intuition.** Every parameter does one multiply-add per token in the forward pass, and the backward pass costs about twice as much, so training costs about six floating-point operations per parameter per token.

**Math.** For a standard Transformer with $d_\text{attn} = d_\text{ff}/4 = d_\text{model}$ (Section 2.1, Equation 2.1):

$$
N \approx 2\, d_\text{model}\, n_\text{layer}\,(2\, d_\text{attn} + d_\text{ff}) = 12\, n_\text{layer}\, d_\text{model}^2.
$$

A forward pass costs $C_\text{forward} \approx 2N + 2\, n_\text{layer}\, n_\text{ctx}\, d_\text{attn}$ (Equation 2.2). The second term is small when $d_\text{model} \gg n_\text{ctx}/12$, so the paper drops it and defines the training compute as $C \approx 6NBS$ (Section 2.1 and Section 1.3). One PF-day is $10^{15} \times 24 \times 3600 = 8.64 \times 10^{19}$ operations.

**Tiny example, computed by `scripts/param_compute.py`.** GPT-2 XL has $n_\text{layer} = 48$ and $d_\text{model} = 1600$, so $N = 12 \times 48 \times 1600^2 = 1.4746 \times 10^9$. With the paper's default run (Section 2.2: $2.5 \times 10^5$ steps of 512 sequences of 1024 tokens), $B = 524{,}288$ tokens and $D = BS = 1.3107 \times 10^{11}$ tokens. Then

$$
C = 6NBS = 1.1596 \times 10^{21}\ \text{FLOPs} = 13.42\ \text{PF-days}.
$$

The context term is 5.3% of $2N$ for this shape, which is why dropping it is safe. The embedding matrices hold $(50257 + 1024) \times 1600 = 8.2 \times 10^7$ parameters, which is 5.3% of the model. For GPT-2 small ($12$ layers, $d_\text{model} = 768$), $N = 8.49 \times 10^7$ and the embeddings hold $3.94 \times 10^7$, which is 31.7% of all parameters. That difference is why the choice to exclude embeddings changes small-model curves the most.

### Overfitting depends on one ratio

**Intuition.** If the dataset is much larger than the model needs, loss depends only on $N$. If the model is much larger than the data supports, loss depends only on $D$. One equation joins the two regimes.

**Math.** The paper proposes (Section 1.2, Equation 1.5; Section 4.1, Equation 4.1)

$$
L(N, D) = \left[\left(\frac{N_c}{N}\right)^{\alpha_N / \alpha_D} + \frac{D_c}{D}\right]^{\alpha_D}.
$$

Section 4.1 gives three principles behind this form. A change of tokenizer must only rescale the loss. $D \to \infty$ must give $L(N)$, and $N \to \infty$ must give $L(D)$. The loss must have a series in $1/D$ with integer powers. The paper calls the third principle "more speculative" and justifies the equation mainly by its fit.

Define the overfitting penalty as $\delta L = L(N, D)/L(N, \infty) - 1$ (Equation 4.2). Equation 4.3 shows that $\delta L$ depends on $N$ and $D$ only through $N^{\alpha_N/\alpha_D}/D$. If the penalty must stay under the seed-to-seed noise of about 0.02, the paper derives (Section 4.2, Equation 4.4)

$$
D \gtrsim \left(5 \times 10^3\right) N^{0.74}.
$$

The data needed grows slower than linearly in $N$. Section 1.1 puts it as follows: an 8x larger model needs only about 5x more data.

A note on constants. The fit in Section 4.2 (Table 2) uses $\alpha_N = 0.076$, $\alpha_D = 0.103$, $N_c = 6.4 \times 10^{13}$, and $D_c = 1.8 \times 10^{13}$. The single-variable fit of Section 1.2 gives $\alpha_D = 0.095$ and $D_c = 5.4 \times 10^{13}$. The paper says the values differ slightly because Table 2 fits the full surface and the other fits only one axis (Section 4.2). This page uses Table 2 whenever it evaluates Equation 1.5.

**Tiny example, computed by `scripts/overfitting_example.py`.** Fix $N = 10^9$. With unlimited data, $L(N) = 2.3188$. Then:

| $D$ (tokens) | $L(N, D)$ | $\delta L$ |
|---|---|---|
| $10^8$ | 3.4846 | 0.5028 |
| $10^9$ | 2.7943 | 0.2051 |
| $10^{10}$ | 2.4197 | 0.0435 |
| $2.2 \times 10^{10}$ | 2.3693 | 0.0218 |

Equation 4.4 asks for $2.285 \times 10^{10}$ tokens at $N = 10^9$, and the table shows $\delta L \approx 0.022$ at $2.2 \times 10^{10}$, which is right at the 0.02 threshold. For an 8x larger model, $8^{0.74} = 4.659$, so the data requirement grows 4.66x. Exercise 3 implements the equation and its two limits.

### Spend compute on model size, not on training to convergence

**Intuition.** Large models learn more from each token. So with a fixed budget, a big model trained briefly beats a small model trained to convergence. Most of a larger budget should go to a larger model, a smaller share to a larger batch, and almost none to more serial steps.

**Math.** Training at a very small batch minimizes compute, and at a very large batch minimizes steps. The paper uses the critical batch size $B_\text{crit}(L)$ (Section 5.1, Equations 5.1 to 5.3), which depends only on the loss:

$$
B_\text{crit}(L) = \frac{B_*}{L^{1/\alpha_B}}, \qquad B_* \approx 2 \times 10^8\ \text{tokens},\quad \alpha_B \approx 0.21
$$

(Equation 1.4; Table 5 gives $B_* = 2.1 \times 10^8$, which this page uses.) A run at batch size $B$ is corrected to the minimum steps and minimum compute with (Equations 5.4 and 5.5)

$$
S_\text{min} = \frac{S}{1 + B_\text{crit}/B}, \qquad C_\text{min} = \frac{C}{1 + B/B_\text{crit}}.
$$

Written in terms of $C_\text{min}$, the loss and the optimal allocations are also power laws (Equation 1.3, Equation 6.1, Equation 6.2, and Table 6):

$$
L(C_\text{min}) = \left(\frac{C_c^\text{min}}{C_\text{min}}\right)^{\alpha_C^\text{min}}, \qquad
N_\text{opt} \propto C_\text{min}^{0.73}, \quad
B \propto C_\text{min}^{0.24}, \quad
S_\text{min} \propto C_\text{min}^{0.03},
$$

with $\alpha_C^\text{min} \approx 0.050$ and $C_c^\text{min} \approx 3.1 \times 10^8$ PF-days. The three exponents sum to 1, which is $C = 6NBS$ in disguise. Equation 1.8 derives $\alpha_C^\text{min} = 1/(1/\alpha_S + 1/\alpha_B + 1/\alpha_N)$ from the pieces. With $\alpha_S = 0.76$, $\alpha_B = 0.21$, and $\alpha_N = 0.076$, the script gives 0.0520. The paper prints $\approx 0.054$ in Equation 6.4 and $\approx 0.052$ in Equation B.7, so treat the third digit as soft.

**Tiny example, computed by `scripts/compute_optimal.py`.** Table 6 constants give

| $C_\text{min}$ (PF-days) | $N_\text{opt}$ | $D_\text{opt}$ (tokens) | $S_\text{min}$ | $L(C_\text{min})$ |
|---|---|---|---|---|
| $10^{-3}$ | $8.39 \times 10^6$ | $3.10 \times 10^9$ | 4389 | 3.7546 |
| $1$ | $1.30 \times 10^9$ | $2.00 \times 10^{10}$ | 5400 | 2.6581 |
| $10^2$ | $3.75 \times 10^{10}$ | $6.93 \times 10^{10}$ | 6200 | 2.1114 |

Ten times the compute multiplies $N_\text{opt}$ by $10^{0.73} = 5.370$ and $D_\text{opt}$ by $10^{0.27} = 1.862$, and multiplies the loss by $10^{-0.05} = 0.8913$. A billion-fold compute increase multiplies the model size by $3.7 \times 10^6$, the batch size by 144.5, and the serial steps by only 1.86. Those match Figure 3, which shows more than a million-fold model growth, about 100x batch growth, and under 10x step growth.

For the batch correction, take a target loss of 3.0. Then $B_\text{crit} = 1.1226 \times 10^6$ tokens. A run at $B = 2^{19} = 524{,}288$ tokens for $10^5$ steps would need only $S_\text{min} = 31{,}836$ steps at a very large batch, and it spends $C_\text{min} = 0.6816\, C$. Training at exactly $B_\text{crit}$ costs $2S_\text{min}$ steps and $2C_\text{min}$ compute (Section 5.1). Exercise 4 implements these formulas.

### Where the extrapolation must fail

**Intuition.** Loss can't fall below the entropy of text, so the power laws must bend. The paper finds a place where two of its own laws disagree and argues that this marks the limit.

**Math.** Overfitting control asks for $D \propto N^{0.74} \propto C_\text{min}^{0.54}$ (Equation 6.6). One epoch of compute-optimal training uses only (Equation 6.7)

$$
D(C_\text{min}) = \frac{2 C_\text{min}}{6 N(C_\text{min})} \approx \left(4 \times 10^{10}\ \text{tokens}\right)\left(\frac{C_\text{min}}{\text{PF-day}}\right)^{0.26}.
$$

Data grows with an exponent of 0.26 when the model needs an exponent of 0.54, so a compute-optimal run eventually can't avoid overfitting, even with no repeated data. When data limits the loss, $L(D) \propto D^{-0.095}$ gives $L \propto C_\text{min}^{-0.03}$, which falls more slowly than $L(C_\text{min}) \propto C_\text{min}^{-0.050}$. The two curves cross at (Section 6.3, Equation 6.8)

$$
C^* \sim 10^4\ \text{PF-days}, \quad N^* \sim 10^{12}\ \text{parameters}, \quad D^* \sim 10^{12}\ \text{tokens}, \quad L^* \sim 1.7\ \text{nats/token}.
$$

The paper says these values are "highly uncertain", by an order of magnitude either way, and it only conjectures that $L^*$ estimates the entropy of text.

**Tiny example, computed by `scripts/compute_optimal.py`.** At $C_\text{min} = 10^4$ PF-days, one epoch gives $4.386 \times 10^{11}$ tokens. Equation 4.4 asks for $5 \times 10^3\, N^{0.74} = 4.02 \times 10^{12}$ tokens at the matching $N_\text{opt}$. The need exceeds the supply by about 9x, and the gap widens with compute.

**What changed later.** The paper's recommendation is that data should grow slowly and the model should grow fast. Later work re-ran the experiment and reached a different split. Hoffmann et al. (2022) found that model size and data should grow in equal proportion (see [Chinchilla](/chinchilla/)). This page doesn't reproduce that result. It is outside the Kaplan paper, so read Hoffmann et al. for the evidence.

<!-- exercises -->

## Pitfalls

- **Including the embeddings in $N$.** The paper excludes them (Section 1.3, Section 3.2). With embeddings in, the loss depends on depth and the trend blurs (Figure 6). Use $12\, n_\text{layer}\, d_\text{model}^2$ to compare with the paper.
- **Comparing the scale constants across tokenizers.** $N_c$, $D_c$, and $C_c$ depend on the vocabulary and tokenization, so they have no fundamental meaning (Section 1.2). Loss in nats per token isn't comparable between a 50k-token BPE and a 32k-token or byte-level vocabulary.
- **Mixing up $C$ and $C_\text{min}$.** The fits that predict well use $C_\text{min}$, which corrects for the batch size (footnote 3, Section 1.2; Equation 5.5). The raw fit to $C$ has a different exponent, $0.057$ against $0.050$ (Table 5). Footnote 3 says to use $C_\text{min}$ for predictions.
- **Fitting in the wrong space.** Fit $\ln L$ against $\ln N$. A fit of $\ln L$ against $N$ is an exponential and extrapolates to a loss near zero.
- **Forgetting the $n - 2$ and the "1 +" in the error bar.** The first understates the noise at small $n$. The second drops the scatter of one new measurement, so the interval is too narrow for a single training run.
- **Trusting error bars from a handful of runs.** The example above shows a fit that reports 0.0053 for scatter that is 0.01. A fit over many orders of magnitude can still miss the next one.
- **Reading $\alpha_D$ as one number.** Section 1.2 gives $0.095$, and Table 2 gives $0.103$ for the joint fit. Use the table that matches the equation you are evaluating.
- **Treating an exponent as a promise.** The paper says the laws must break down, and gives Equation 6.8 as the estimate of where. It also warns that smooth loss can hide qualitative changes in what a model can do (Section 8).
- **Reading "stop short of convergence" as "use little data".** Compute-efficient training uses fewer steps and a larger model, and the batch size grows with compute. Equation 4.4 still demands more data as the model grows.

## Further reading

- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361), by Kaplan et al., is the paper. The appendices derive the compute-efficient frontier (Appendix B) and adds further analyses (Appendix D).
- [An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162), by McCandlish et al., defines the critical batch size and the gradient noise scale that Section 5.1 builds on.
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556), by Hoffmann et al., re-measures the optimal split between parameters and tokens. Read it alongside Section 6.
- [Deep Learning Scaling is Predictable, Empirically](https://arxiv.org/abs/1712.00409), by Hestness et al., is the earlier study of power-law learning curves that the paper cites in Section 7.
- [Scaling laws and the compute budget](https://ml.ifkash.dev/lessons/055-scaling-laws) is the course lesson that walks through the same fits.
