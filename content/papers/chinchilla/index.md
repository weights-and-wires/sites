---
id: "chinchilla"
title: "Training Compute-Optimal Large Language Models"
authors: "Hoffmann et al."
year: 2022
venue: "NeurIPS 2022"
arxiv: "2203.15556"
url: "https://arxiv.org/abs/2203.15556"
track: scaling-efficiency
tier: core
est_hours: 3
status: full
summary: "The paper asks how to split a fixed training compute budget between model size and training tokens, and answers with three estimation methods, run on more than 400 models, that all say the two should grow in equal proportion with compute. That contradicts the earlier Kaplan recommendation, which grew the model much faster than the data, and implies that models such as Gopher were far too large for the data they saw. The authors test the claim by training Chinchilla, a 70B-parameter model on 1.4 trillion tokens, which beats the 280B-parameter Gopher on nearly every evaluation at about the same compute and with a quarter of the inference cost."
related_course: "ml: scaling laws"
prereqs:
  - id: kaplan
    reason: "Chinchilla keeps Kaplan's setup of loss as a smooth power law in parameters and tokens with training compute of about $6ND$, and it asks the same question about the best split of a budget, but it reaches a different answer."
background: [flops-estimate-6nd, curve-fitting]
course_links:
  - title: "Scaling laws and the compute budget"
    url: "https://ml.ifkash.dev/lessons/055-scaling-laws"
  - title: "The training step, and what a run costs"
    url: "https://ml.ifkash.dev/lessons/053-training-loop-and-cost"
self_check:
  - q: 'A run has a budget of $C = 5.88 \times 10^{23}$ FLOPs and a model with $N = 7 \times 10^{10}$ parameters. How many tokens $D$ does it see if it spends the whole budget, using $C \approx 6ND$?'
    a: '$D = C / (6N) = 5.88 \times 10^{23} / (4.2 \times 10^{11}) = 1.4 \times 10^{12}$ tokens, which is 20 tokens per parameter. The 6 is 2 FLOPs per parameter per token in the forward pass plus 4 in the backward pass.'
    see: flops-estimate-6nd
  - q: 'You train six models of different sizes at one fixed budget $C$ and record each final loss. Which function do you fit to find the best model size, and how do you read the answer off the fit?'
    a: 'Fit a quadratic $c_2 x^2 + c_1 x + c_0$ to the losses against $x = \ln N$, by least squares. The vertex at $x^* = -c_1 / (2 c_2)$ gives the best size $N^* = e^{x^*}$ (Section 3.2). A parabola in $N$ itself fits a wide range of sizes badly.'
    see: curve-fitting
  - q: 'Kaplan et al. have $N_{opt} \propto C^{0.73}$ and $D_{opt} \propto C^{0.27}$. By what factor does each of $N$ and $D$ grow when the budget grows tenfold? Compare with exponents of $0.5$ and $0.5$.'
    a: 'With Kaplan, $N$ grows by $10^{0.73} = 5.37$ and $D$ by $10^{0.27} = 1.86$. With $0.5$ and $0.5$, both grow by $10^{0.5} = 3.16$. The two exponents in each pair add to 1 because $C = 6ND$.'
    see: kaplan
---

## The problem

By early 2022, the largest language models had grown quickly in size and hardly at all in data. GPT-3 has 175B parameters and saw 300B tokens. Gopher has 280B parameters and saw 300B tokens. Megatron-Turing NLG has 530B parameters and saw 270B tokens (Table 1). The practice followed [Kaplan et al.](/kaplan/), who found that for a larger budget you should grow the model about 5.5 times and the data only about 1.8 times for each tenfold increase in compute (Section 1).

Here is the failing case. Gopher used 280B parameters and 300B tokens, so each parameter saw about 1.07 tokens. Almost every Transformer in the field had been trained with roughly 300B tokens, whatever its size. If that were the right split, a team with Gopher's budget could not do better by making the model smaller. The paper's claim is that it could, and by a lot: a model four times smaller, trained on four times more tokens, reaches a lower loss.

The authors state the question as an optimization (Section 1, Equation 1). Given a budget $C$, choose the parameter count $N$ and token count $D$ that minimize the final loss $L(N, D)$, subject to $\mathrm{FLOPs}(N, D) = C$. They answer it by training over 400 models, from under 70M to over 16B parameters, on 5B to over 400B tokens (Section 1).

## Key ideas

The sections that follow use the paper's notation. $N$ is the number of parameters, and $D$ is the number of training tokens. $C$ is the compute in FLOPs, and $L$ is the final training loss. The exponents $a$ and $b$ are defined by $N_{opt} \propto C^a$ and $D_{opt} \propto C^b$ (Section 3.1). One difference from [Kaplan](/kaplan/) matters: Kaplan's $N$ excludes embedding parameters, but this paper counts them in $N$ and in the FLOPs (Appendix F). Every number below comes from a script in `scripts/`. Run a script from the paper folder to reproduce the output.

### The budget is a constraint, so the question is a trade-off

**Intuition.** A fixed budget buys either many parameters, each seen by few tokens, or few parameters, each seen by many tokens. Both options cost the same, and they reach different losses.

**Math.** The paper minimizes $L$ under the constraint $\mathrm{FLOPs}(N, D) = C$ (Section 1, Equation 1). For the analysis it uses the approximation $\mathrm{FLOPs}(N, D) \approx 6ND$ that Kaplan et al. introduced (Section 3.3). The accurate count in Appendix F adds the embeddings, the attention logits, and the softmax. Table A4 shows the ratio of the accurate count to $6ND$ for six model sizes, between 0.99 and 1.10, and the paper says the difference doesn't affect the analysis.

**Tiny example, computed by `scripts/gopher_vs_chinchilla.py`.** Gopher has $6ND = 6 \times 280\text{B} \times 300\text{B} = 5.04 \times 10^{23}$ FLOPs. Chinchilla has $6 \times 70\text{B} \times 1.4\text{T} = 5.88 \times 10^{23}$ FLOPs. The two differ by 17% under $6ND$. The paper counts the budgets as equal because the accurate count in Appendix F gives about $5.76 \times 10^{23}$ for Gopher (Appendix F and Table 3). The same model also has a tokens-per-parameter ratio of $1.4\text{T} / 70\text{B} = 20$ against Gopher's 1.07.

### Why Kaplan et al. got a different split

**Intuition.** If you cut a training run short, you see a loss that is higher than the loss of a run designed to be that short. A study built from cut-short runs makes small-data runs look worse than they are.

**Math.** The paper names two differences (Section 2). First, Kaplan et al. used one fixed token count and one learning-rate schedule for every model. With a cosine schedule that decays over 130B tokens, a loss measured at 50B tokens is an overestimate of the loss that a run with a 50B-token schedule would reach. Using those intermediate losses "underestimates the effectiveness of training models on less data" and pushes toward bigger models (Section 2). Second, most of their runs had under 100M parameters, while most of the runs here have more than 500M, and the loss frontier curves slightly in log-log space (Section 2 and Appendix E).

The fix is to match the schedule to the horizon. Each training run here decays the learning rate by a factor of 10 over the number of tokens that the run is meant to see, and the lowest-loss points all fall in the last 15% of training (Section 3.1, footnote 4; Appendix B). In a head-to-head test at $10^{21}$ FLOPs, the model Kaplan's fit predicts (4.74B parameters) loses to the model that this paper's Approach 1 predicts (2.80B parameters) (Appendix D.4, Figure A4).

**Tiny example, computed by `scripts/gopher_vs_chinchilla.py`.** Under Kaplan's exponents, a tenfold budget multiplies $N$ by $10^{0.73} = 5.37$ and $D$ by $10^{0.27} = 1.86$. The paper rounds these to 5.5 and 1.8 (Section 1). Under equal scaling, both grow by $10^{0.5} = 3.16$. After a thousandfold increase in compute, the tokens per parameter under Kaplan's rule shrink by a factor of $1000^{0.27 - 0.73} = 0.042$ (computed by `scripts/table3_exponent.py`), which is the gap that Chinchilla removes.

### Three ways to estimate the frontier

**Intuition.** One method can be wrong in ways that you can't see. The paper uses three that fail differently, and if they agree, the answer is more credible.

**Math.** All three fit power laws $N_{opt} \propto C^a$ and $D_{opt} \propto C^b$ (Section 3). They are the following:

1. *Approach 1 (Section 3.1).* Train each of several fixed model sizes (70M to over 10B) with four different schedule lengths. Smooth each loss curve, and at 1500 log-spaced FLOP values pick the run with the lowest loss. Then fit power laws to the chosen $(C, N)$ and $(C, D)$ pairs.
1. *Approach 2, IsoFLOP profiles (Section 3.2).* Fix nine budgets from $6 \times 10^{18}$ to $3 \times 10^{21}$ FLOPs. At each one, train models of many sizes, with the token count set by the budget, and fit a parabola to the final loss against model size. The vertex is the optimal size at that budget (Figure 3).
1. *Approach 3, parametric loss (Section 3.3).* Fit a formula for $L(N, D)$ to all the final losses, then solve for the frontier. The next two sections cover this one.

The paper reports the following exponents (Table 2). Bootstrap 10th and 90th percentiles are in parentheses.

| Approach | $a$ ($N_{opt} \propto C^a$) | $b$ ($D_{opt} \propto C^b$) |
|---|---|---|
| 1. Minimum over training curves | 0.50 (0.488, 0.502) | 0.50 (0.501, 0.512) |
| 2. IsoFLOP profiles | 0.49 (0.462, 0.534) | 0.51 (0.483, 0.529) |
| 3. Parametric modelling of the loss | 0.46 (0.454, 0.455) | 0.54 (0.542, 0.543) |
| Kaplan et al. (2020) | 0.73 | 0.27 |

**Tiny example, computed by `scripts/table3_exponent.py`.** Fit a line to $(\ln C, \ln N)$ for the nine rows of Table 3 (Approach 1, from 400M to 10T parameters). The slope is 0.4981, and the slope for tokens is 0.5019, which sum to 1 because $C = 6ND$. The same fit on Table A3 (Approach 2) gives 0.4899 and 0.5100. The tokens per parameter in Table 3 range from 20.0 to 22.4 across 400M to 10T parameters, which is where the rule of about 20 tokens per parameter comes from. Exercise 4 asks you to write this fit.

### A parametric loss: three terms

**Intuition.** The loss has a floor that no model can pass, a penalty because the model is finite, and a penalty because the training is finite. Each penalty shrinks as a power law in its own variable.

**Math.** Approach 3 proposes (Section 3.3, Equation 2)

$$
\hat L(N, D) \triangleq E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}.
$$

$E$ is the loss of an ideal generative process, which the paper says should correspond to the entropy of natural text. The second term is the penalty for a finite model, and the third is the penalty for a finite number of optimization steps on a finite sample. Appendix D.2 derives the form from a risk decomposition (Equation 9). The fit minimizes a Huber loss with $\delta = 10^{-3}$ on the log of the loss (Section 3.3, Equation 3; Appendix D.2, Equation 11) and uses L-BFGS from a grid of starting points. Appendix D.2 reports the fitted values (Equation 10):

$$
\hat L(N, D) = 1.69 + \frac{406.4}{N^{0.34}} + \frac{410.7}{D^{0.28}}.
$$

So $E = 1.69$, $A = 406.4$, $B = 410.7$, $\alpha = 0.34$, and $\beta = 0.28$. The notation here is the paper's. Kaplan's Equation 1.5 uses $N_c$ and $D_c$ and a different shape, so the constants are not comparable.

**Tiny example, computed by `scripts/parametric_loss.py`.** Evaluate Equation 10 at three pairs:

| $N$ | $D$ | $A/N^\alpha$ | $B/D^\beta$ | $\hat L$ |
|---|---|---|---|---|
| $2.8 \times 10^{11}$ (Gopher) | $3 \times 10^{11}$ | 0.0521 | 0.2511 | 1.9933 |
| $7 \times 10^{10}$ (Chinchilla) | $1.4 \times 10^{12}$ | 0.0835 | 0.1632 | 1.9366 |
| $10^9$ | $2 \times 10^{10}$ | 0.3540 | 0.5361 | 2.5800 |

For Gopher, the data term is almost five times the size term, so the loss is limited by the number of tokens. The paper doesn't report these $\hat L$ values. They are this page's evaluation of Equation 10, and the 0.057 gap between the first two rows is a prediction of the fit and not a measured result. Exercise 1 implements $\hat L$.

### The efficient frontier in closed form

**Intuition.** Raise the model size by a little and lower the token count to keep $6ND$ fixed. The size term falls and the data term rises. The best split is the point where the two effects cancel.

**Math.** Substitute $D = C / (6N)$ into $\hat L$ and set the derivative with respect to $N$ to zero. The result is a power law in $C$ (Section 3.3, Equation 4):

$$
N_{opt}(C) = G\left(\frac{C}{6}\right)^{a}, \quad D_{opt}(C) = G^{-1}\left(\frac{C}{6}\right)^{b}, \quad G = \left(\frac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha + \beta}}, \quad a = \frac{\beta}{\alpha + \beta}, \quad b = \frac{\alpha}{\alpha + \beta}.
$$

The exponents depend only on $\alpha$ and $\beta$. The size exponent $a$ gets the data exponent $\beta$ in its numerator, so a model that learns data-efficiency slowly (small $\beta$) should put more of the budget into data.

**Tiny example, computed by `scripts/parametric_loss.py`.** With the constants of Equation 10, $a = 0.28 / 0.62 = 0.4516$, $b = 0.5484$, and $G = 1.3447$. The paper's Table 2 prints 0.46 and 0.54 for this approach, which is a rounding of the same numbers within a few hundredths. The script then checks the closed form against a brute-force search over $N$:

| $C$ | $N_{opt}$ | $D_{opt}$ | $D/N$ | $\hat L$ |
|---|---|---|---|---|
| $10^{19}$ | $2.28 \times 10^8$ | $7.31 \times 10^9$ | 32.1 | 2.9857 |
| $10^{21}$ | $1.82 \times 10^9$ | $9.14 \times 10^{10}$ | 50.1 | 2.3289 |
| $5.76 \times 10^{23}$ | $3.22 \times 10^{10}$ | $2.98 \times 10^{12}$ | 92.7 | 1.9307 |

The brute-force search finds the same $N$ at all three budgets. At the Gopher budget, Equation 10 gives a 32B-parameter model on 3.0T tokens. The paper states that Approach 3 projects 40B parameters for that budget (Section 3.3, Figure 4). The difference isn't explained in the paper, so treat Equation 10 as a rounded summary of the fit and don't treat it as an exact source of the frontier. More important for the "20 tokens per parameter" rule: the ratios in this table, 32 to 93, are far above 20. The rule comes from Approaches 1 and 2 (Table 3 and Table A3), and Approach 3 predicts smaller models and more tokens per parameter at large budgets (Section 3.4). Exercise 2 implements Equation 4.

### IsoFLOP profiles and the valley

**Intuition.** At a fixed budget, a model that is too small has seen plenty of data but can't use it, and a model that is too large has capacity but hasn't seen enough tokens. The loss has a valley between them.

**Math.** For one budget $C$, vary $N$ and set $D = C / (6N)$. The final loss traces a U-shaped curve against $\log N$ (Section 3.2, Figure 3 left). The paper fits a parabola to each curve and takes the vertex as the optimal size. It then fits a power law to the vertices across the nine budgets to get $a = 0.49$ and $b = 0.51$ (Table 2). Figure 4 (right) shows the same slices for the parametric model.

**Tiny example, computed by `scripts/isoflop.py`.** Take $C = 10^{21}$ and six model sizes, doubling from 0.4B to 12.8B parameters. The losses come from Equation 10, not from training runs:

| $N$ | $D$ | $D/N$ | $\hat L$ |
|---|---|---|---|
| $4 \times 10^8$ | $4.17 \times 10^{11}$ | 1041.7 | 2.4024 |
| $8 \times 10^8$ | $2.08 \times 10^{11}$ | 260.4 | 2.3500 |
| $1.6 \times 10^9$ | $1.04 \times 10^{11}$ | 65.1 | 2.3294 |
| $3.2 \times 10^9$ | $5.21 \times 10^{10}$ | 16.3 | 2.3384 |
| $6.4 \times 10^9$ | $2.60 \times 10^{10}$ | 4.1 | 2.3762 |
| $1.28 \times 10^{10}$ | $1.30 \times 10^{10}$ | 1.0 | 2.4433 |

A parabola through the six points in $\ln N$ has its vertex at $N = 1.862 \times 10^9$, against the exact minimum $1.824 \times 10^9$, a 2.1% error. The fitted minimum loss is 2.3288 against the true 2.3289. The curve is flat near the bottom. A model 4x too small (0.4B) costs 0.073 in loss at this budget, which is why a few percent of error in $N_{opt}$ is fine. Exercise 3 asks you to write the parabola fit, and Exercise 4 fits the power law over the vertices.

### Chinchilla against Gopher

**Intuition.** A fit is a prediction, and the test is to train the model it recommends. The authors do that at Gopher's budget.

**Math.** The three approaches put the optimal size for Gopher's budget between 40B and 70B parameters (Section 4). The authors train a model at the larger end: Chinchilla, with 70B parameters and 1.4T tokens (Section 4, Table 1). It has 80 layers, 64 heads, key/value size 128, and $d_\text{model} = 8192$ (Table 4). It uses the same architecture and FLOP budget as Gopher, and the same MassiveText dataset, with four changes (Section 4.1): a slightly different mix of data subsets, AdamW in place of Adam (see [AdamW](/adamw/)), a SentencePiece tokenizer without NFKC normalization, and a float32 copy of the weights in the optimizer state. The batch size is 1.5M tokens, doubled to 3M midway through training.

The headline results are the following:

- On MMLU, Chinchilla reaches 67.6% average 5-shot accuracy against 60.0% for Gopher and 43.9% for GPT-3 (Section 4.2.2, Table 6). The abstract rounds this to 67.5%, and the table says 67.6%.
- On Wikitext103, Chinchilla's perplexity is 7.16 against 7.75 for Gopher (Section 4.2.1). The paper warns that train and test overlap may inflate language-modeling numbers for a model trained on 4x more data, and it puts more weight on MMLU and other tasks.
- On The Pile, Chinchilla beats Gopher on every subset (Section 4.2.1, Figure 5).

**Tiny example, computed by `scripts/gopher_vs_chinchilla.py`.** Chinchilla has a quarter of Gopher's parameters, so a forward pass costs $2N$ FLOPs per token, which is 4.0x less. The paper makes the same point about fine-tuning and inference (Section 4).

<!-- exercises -->

## Pitfalls

- **Treating "20 tokens per parameter" as a law.** It is the ratio in Table 3, which comes from Approach 1, and it ranges from 20.0 to 22.4 across the table. Approach 2 gives 19.2 to 29.2 over the same rows (Table A3), and Approach 3 gives much larger ratios at large budgets (Table A3). The equal-scaling claim is about exponents. The constant in front is less certain.
- **Reading Equation 10 as the frontier.** Evaluating Equation 4 with the printed constants gives a 32B model for Gopher's budget and 93 tokens per parameter, while the paper's own Approach 3 figure shows 40B. Use the constants to see the shape, and use Table 3 for numbers to quote.
- **Using a fixed learning-rate schedule.** The central methodological point is that the cosine cycle should end near the token count that the run is meant to see. Losses read from the middle of a longer schedule are overestimates, and they make short runs look worse (Section 2; Appendix B).
- **Fitting the IsoFLOP parabola in $N$.** The valley is symmetric in $\log N$, and sizes in the paper span a factor of 100 or more. A parabola in $N$ puts the vertex in the wrong place.
- **Mixing parameter conventions.** This paper counts embedding parameters in $N$ and in the FLOPs (Appendix F). [Kaplan](/kaplan/) excludes embeddings, so $N$ values and fitted constants don't transfer between the two papers without a correction.
- **Using $6ND$ for everything.** Table A4 shows that the accurate FLOP count is between 0.99x and 1.10x of $6ND$ for the sizes listed. That is the reason the paper's budget for Gopher ($5.76 \times 10^{23}$) differs from $6ND$ ($5.04 \times 10^{23}$).
- **Taking the optimum as the target for deployment.** The paper's objective is the lowest loss for a fixed training budget. It does not include the cost of serving the model. A model that is smaller than the frontier and trained longer can cost more to train for the same loss, and cost less to serve (Section 4 notes the benefit of a smaller model for inference).
- **Extrapolating without the caveats.** The authors list their own limits (Section 5): only two large runs (Chinchilla and Gopher) test the prediction, the fit assumes a power law although Appendix E shows curvature that may mean even smaller models are optimal at large budgets, and all runs use less than one epoch of data.

## Further reading

- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556), by Hoffmann et al., is the paper. Appendix D gives the fitting details, and Appendix E shows the curvature in the frontier.
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361), by Kaplan et al., is the study that this paper revises. See [the Kaplan page](/kaplan/) on this site.
- [Chinchilla Scaling: A replication attempt](https://arxiv.org/abs/2404.10102), by Besiroglu et al., re-fits the parametric loss from the paper's data and reports that the published Approach 3 constants don't reproduce well. Read it with the Equation 10 caution above.
- [Scaling Language Models: Methods, Analysis & Insights from Training Gopher](https://arxiv.org/abs/2112.11446), by Rae et al., describes the baseline model that Chinchilla is compared against.
- [Scaling laws and the compute budget](https://ml.ifkash.dev/lessons/055-scaling-laws) is the course lesson that walks through the same fits.
