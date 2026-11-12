---
id: "spec-decoding"
title: "Fast Inference from Transformers via Speculative Decoding"
authors: "Leviathan et al."
year: 2022
venue: "ICML 2023"
arxiv: "2211.17192"
url: "https://arxiv.org/abs/2211.17192"
track: inference
tier: core
est_hours: 4
status: full
summary: "Speculative decoding makes a large autoregressive model generate faster without changing its output distribution. A small draft model proposes several tokens, and the large target model scores all of them in one parallel pass. A rejection-sampling rule keeps each draft token with probability min(1, p/q) and resamples from a corrected distribution after the first rejection, so every run of the target model yields at least one token and usually several."
related_course: "inference: speculative decoding"
prereqs:
  - id: transformer
    reason: "Verification works because a causal Transformer decoder returns the next-token distribution at every position of a sequence in one forward pass."
background: [rejection-sampling, autoregressive-decoding]
course_links:
  - title: "Speculative decoding"
    url: "https://qwen.ifkash.dev/c/20-speculative-decoding"
  - title: "Decoding and sampling: cache, temperature, speculation"
    url: "https://ml.ifkash.dev/lessons/075-decoding-and-sampling"
self_check:
  - q: 'Decoding $K$ tokens with a Transformer takes how many serial runs of the model, and why can''t the runs overlap?'
    a: 'It takes $K$ serial runs. Token $t$ is sampled from a distribution that depends on token $t-1$, so run $t$ can''t start until run $t-1$ has produced its token. Each run is also dominated by reading the weights and the KV cache, not by arithmetic, which is the slack speculative decoding uses.'
    see: autoregressive-decoding
  - q: 'The draft model has $q = [0.2, 0.3, 0.5]$ and the target has $p = [0.5, 0.4, 0.1]$. The draft proposes token 2. What is the chance the target accepts it, and what is the chance that a random proposal is accepted?'
    a: 'Token 2 is accepted with probability $\min(1, p/q) = 0.1 / 0.5 = 0.2$. A random proposal is accepted with probability $\sum_x \min(p(x), q(x)) = 0.2 + 0.3 + 0.1 = 0.6$. Accepting only these tokens is not enough on its own, because it under-samples tokens where $p > q$. The rejection branch has to resample from $\operatorname{norm}(\max(0, p - q))$ to fix that.'
    see: rejection-sampling
  - q: 'Why can the target model score all $\gamma + 1$ positions in one forward pass, with no extra cost for the guesses?'
    a: 'A decoder with a causal mask computes the next-token distribution at every input position in parallel. This is the same property that lets it train on a whole sequence at once. Position $i$ sees only tokens up to $i$, so feeding the prefix plus $\gamma$ draft tokens returns exactly the distributions that $\gamma + 1$ separate runs would return.'
    see: transformer
---

## The problem

A large Transformer decodes one token per forward pass. Generating $K$ tokens takes $K$ serial runs, because each token is an input to the next run (Section 1). Suppose a 11-billion-parameter model produces a 100-token translation. That is 100 passes, and in each pass the hardware streams every weight and the whole KV cache from memory to compute one row of logits.

That run is bound by memory bandwidth, not by arithmetic. The accelerator has far more compute than one token needs, and it sits idle. The paper makes this observation in Section 1 and builds on it: if the extra compute is free, work that is speculative and sometimes wasted costs nothing.

Earlier speedups, such as distillation, quantization, and adaptive computation, usually change the architecture or the training, and they change the output distribution. This paper asks for something stricter: a faster sampler whose output has exactly the distribution of the original model, with no retraining.

The main result is Table 2. On T5-XXL (11B parameters) with a T5-small (77M) draft, speculative decoding is 3.4x faster for English-to-German translation with argmax sampling, and 2.6x faster with temperature 1, with identical output distributions.

## Key ideas

The sections that follow use the paper's notation. $M_p$ is the target model with next-token distribution $p(x_t \mid x_{<t})$, and $M_q$ is the draft (approximation) model with distribution $q(x_t \mid x_{<t})$. The number of guesses per run is $\gamma$, the acceptance rate is $\beta$ for one position and $\alpha = E(\beta)$ on average, and $c$ is the cost of one draft run relative to one target run. The paper writes $p(x)$ and $q(x)$ and drops the prefix when it is clear (Section 2.1, footnote 1). This page does the same. Every number comes from a script in `scripts/`; run any of them from the paper folder.

One difference from the paper: Algorithm 1 indexes positions from 1. The exercises count from 0, so the paper's $p_{n+1}$ is `p_dists[n]` in code.

### Draft cheaply, verify in one pass

**Intuition.** A hard language-modeling task contains easy steps. After "New York" the next token is probably "City", and a small model gets that right. So let the small model guess a few tokens ahead, and let the large model check all the guesses at once. Checking is cheap because a Transformer decoder scores every position of a sequence in a single parallel pass.

**Math.** Section 2.1 describes one run of the algorithm, "SpeculativeDecodingStep" (Algorithm 1):

1. Run $M_q$ autoregressively for $\gamma$ steps to get guesses $x_1, \dots, x_\gamma$, where $x_i \sim q_i(x)$.
1. Run $M_p$ once, in parallel, on the prefix extended by each guess. This gives $p_1(x), \dots, p_{\gamma+1}(x)$.
1. Keep a prefix of the guesses (the next idea says which), and add one token from $M_p$.

Each run of $M_p$ yields at least one token, so the number of serial target runs can never exceed that of plain decoding (Section 2.1). It yields up to $\gamma + 1$ tokens when every guess is accepted, because $p_{\gamma+1}$ is already computed for free.

The method also needs both models to use the same sampling rule. Section 2.2 notes that argmax, top-$k$, nucleus sampling, and temperature all reduce to sampling from an adjusted distribution. You apply the adjustment to the logits of both models first, and then $p$ and $q$ in the algorithm are the adjusted distributions.

### Speculative sampling

**Intuition.** Take one position. The draft proposes a token $x \sim q$. If the target likes the token at least as much as the draft does, keep it. If the target likes it less, keep it only some of the time. When you reject, don't sample from $p$ directly. The accepted samples already cover most of $p$, and the rejection branch has to supply only the part of $p$ that the draft under-proposes.

**Math.** Section 2.3 states the rule. Sample $x \sim q(x)$. Keep it if $q(x) \le p(x)$. Otherwise reject it with probability $1 - p(x)/q(x)$ and sample again from the adjusted distribution

$$
p'(x) = \operatorname{norm}\bigl(\max(0,\; p(x) - q(x))\bigr).
$$

Equivalently, accept with probability $\min(1, p(x)/q(x))$. Algorithm 1 implements this with a uniform draw $r_i \sim U(0, 1)$ and rejects guess $i$ when $r_i > p_i(x)/q_i(x)$.

The acceptance rate follows directly (Theorem 3.5):

$$
\beta = E_{x \sim q}\min\!\left(1, \frac{p(x)}{q(x)}\right) = \sum_x \min\bigl(p(x), q(x)\bigr).
$$

Appendix A.1 shows that the normalizing constant of $p'$ is $1 - \beta$, because $\sum_x \bigl(p(x) - \min(p(x), q(x))\bigr) = 1 - \beta$.

**Tiny example, computed by `scripts/accept_reject_example.py`.** Take a three-token vocabulary with $p = [0.5, 0.4, 0.1]$ and $q = [0.2, 0.3, 0.5]$.

| Step | Result |
|---|---|
| Acceptance probability $\min(1, p/q)$ | `[1, 1, 0.2]` |
| $\beta = \sum_x \min(p, q)$ | $0.2 + 0.3 + 0.1 = 0.6$ |
| $\max(0, p - q)$ | `[0.3, 0.1, 0]`, which sums to $0.4 = 1 - \beta$ |
| $p' = \operatorname{norm}(\max(0, p - q))$ | `[0.75, 0.25, 0]` |

Token 2 is the draft's favorite ($q = 0.5$), but the target gives it only 0.1. It is accepted 20% of the time.

### Why the output follows the target

**Intuition.** A token can reach the output in two ways: the draft proposed it and it was accepted, or the draft's proposal was rejected and the residual produced it. The accepted route contributes $\min(p, q)$ to each token. The residual route contributes exactly the missing mass, $p - \min(p, q)$. The two parts add up to $p$.

**Math.** This is Appendix A.1. For any token $x'$:

$$
P(x = x') = \underbrace{q(x')\min\!\left(1, \frac{p(x')}{q(x')}\right)}_{\text{accepted}} + \underbrace{(1 - \beta)\,p'(x')}_{\text{rejected}} = \min(p(x'), q(x')) + p(x') - \min(p(x'), q(x')) = p(x').
$$

The proof never uses any property of $q$, so the guarantee holds for any draft model (Section 3.6), including a bad one. A bad draft costs speed, not correctness.

**Tiny example, computed by `scripts/accept_reject_example.py`.** With the distributions above:

| Token | P(accepted, $x$) | P(rejected, $x$) | Total | $p$ |
|---|---|---|---|---|
| 0 | 0.2 | $0.4 \times 0.75 = 0.3$ | 0.5 | 0.5 |
| 1 | 0.3 | $0.4 \times 0.25 = 0.1$ | 0.4 | 0.4 |
| 2 | 0.1 | $0.4 \times 0 = 0$ | 0.1 | 0.1 |

The script also draws 100,000 seeded samples. The frequencies are `[0.5003, 0.4002, 0.0995]`, the empirical acceptance rate is 0.6008, and the chi-square statistic against $p$ is 0.26 with 2 degrees of freedom. The 0.001 critical value is 13.82, so there is no evidence against the claim. Exercise 1 asks you to implement this step and run the same test.

Appendix A.2 compares this rule with classic rejection sampling, which accepts with probability $p(x)/(M q(x))$ for $M = \max_x p(x)/q(x)$. Its acceptance rate is $1/M$, and `scripts/rejection_vs_speculative.py` computes $M = 2.5$ for this example, so $1/M = 0.4$, against $\beta = 0.6$. Speculative sampling also never loops: on rejection it takes one draw from $p'$.

### Verifying a whole draft

**Intuition.** Verification goes left to right. Once guess $i$ is rejected, guesses after it were drafted from a prefix that is no longer the real prefix, so they are discarded. The token that replaces guess $i$ comes from the residual. If every guess passes, the target's last distribution $p_{\gamma+1}$ supplies a bonus token.

**Math.** Algorithm 1 finds the number of accepted guesses

$$
n = \min\bigl(\{\, i - 1 \mid 1 \le i \le \gamma,\; r_i > p_i(x)/q_i(x) \,\} \cup \{\gamma\}\bigr),
$$

with $p'(x) = p_{n+1}(x)$ when $n = \gamma$, and $p'(x) = \operatorname{norm}(\max(0, p_{n+1}(x) - q_{n+1}(x)))$ when $n < \gamma$. It then samples $t \sim p'$ and returns the prefix plus $[x_1, \dots, x_n, t]$.

**Tiny example.** Let $\gamma = 3$ and say the three draws give $r = (0.1, 0.9, 0.0)$, with ratios $p_i/q_i = (1.0, 0.25, 0.2)$ for the three guesses. Guess 1 passes ($0.1 \le 1.0$). Guess 2 fails ($0.9 > 0.25$), so $n = 1$. Guess 3 would pass on its own ($0.0 \le 0.2$), but it is discarded. The output has $n + 1 = 2$ tokens: guess 1 and one token drawn from $\operatorname{norm}(\max(0, p_2 - q_2))$. Exercise 2 builds this step and tests it against these cases.

### Expected tokens per run

**Intuition.** If every guess is accepted with the same probability $\alpha$, the number of accepted guesses is a geometric random variable, cut off at $\gamma$. A longer draft adds a term $\alpha^k$ that shrinks quickly, so extra guesses give diminishing returns.

**Math.** Section 3.1 defines the acceptance rate $\beta_{x_{<t}}$ for a prefix (Definition 3.1). It then assumes the $\beta$s are i.i.d. with mean $\alpha = E(\beta)$. This is a simplification, and the paper says so. Under it, a single run of Algorithm 1 produces a capped geometric number of tokens, with success probability $1 - \alpha$ and cap $\gamma + 1$, and the expected count is (Section 3.1, Equation 1)

$$
E(\#\text{generated tokens}) = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha} = \sum_{k=0}^{\gamma} \alpha^k.
$$

Corollary 3.6 gives $\alpha = E(\min(p, q)) = 1 - E(D_{LK}(p, q))$, where $D_{LK}$ is the paper's symmetric divergence (Definition 3.2). It equals the total variation distance. When $\gamma \to \infty$ the count approaches $1/(1 - \alpha)$.

**Tiny example, computed by `scripts/tokens_per_iteration.py`.** Take $\alpha = 0.8$ and $\gamma = 5$. Then $\alpha^6 = 0.262144$ and

$$
\frac{1 - 0.262144}{0.2} = 3.6893.
$$

The probabilities of $n = 0, 1, \dots, 5$ accepted guesses are `[0.2, 0.16, 0.128, 0.1024, 0.0819, 0.3277]`, which sum to 1, and their mean of $n + 1$ is 3.6893. A seeded simulation of 200,000 runs gives a mean of 3.691. Five guesses never yield more than the limit $1/(1 - \alpha) = 5$.

### Walltime speedup and the best gamma

**Intuition.** Guessing costs time, because the draft runs serially. Each run of speculative decoding costs one target run plus $\gamma$ draft runs. The speedup is tokens per run divided by the cost per run. A larger $\gamma$ gives more tokens per run, but a higher cost, so there is an optimum.

**Math.** Let $c$ be the ratio of one draft run's time to one target run's time (Definition 3.7). One run of Algorithm 1 costs $T(c\gamma + 1)$, where $T$ is the time of one target run, and it produces Equation 1 tokens on average. The expected walltime improvement is (Theorem 3.8)

$$
\frac{1 - \alpha^{\gamma+1}}{(1 - \alpha)(\gamma c + 1)}.
$$

This assumes that $\gamma + 1$ target evaluations in parallel take the same time as one (Section 3.3), which needs spare compute. Corollary 3.9 says that if $\alpha > c$, some $\gamma$ gives a speedup, at least $(1 + \alpha)/(1 + c)$, which is the $\gamma = 1$ value. Section 3.5 notes that the best integer $\gamma$ is easy to find numerically (Figure 3).

The compute side has a cost. The expected factor of increase in arithmetic operations is $\frac{(1 - \alpha)(\gamma \hat{c} + \gamma + 1)}{1 - \alpha^{\gamma+1}}$ (Theorem 3.11), where $\hat{c}$ is the draft-to-target ratio of operations per token. Rejected guesses are wasted work. Memory reads for the target's weights and KV cache shrink instead, by the Equation 1 factor (Section 3.4).

**Tiny example, computed by `scripts/speedup_example.py`.** With $\alpha = 0.8$, $\gamma = 5$, and $c = 0.05$, a run costs $5 \times 0.05 + 1 = 1.25$ target runs and yields 3.6893 tokens. The speedup is $3.6893 / 1.25 = 2.95$. If drafting were free ($c = 0$), it would be 3.69. The script reproduces the paper's Table 1, where $c = \hat{c} = 0$:

| $\alpha$ | $\gamma$ | Operations | Speed |
|---|---|---|---|
| 0.6 | 2 | 1.53x | 1.96x |
| 0.8 | 5 | 1.63x | 3.69x |
| 0.9 | 10 | 1.60x | 6.86x |

The script prints all six rows, and they match the paper's. It also evaluates Theorem 3.8 for the 12 rows of the paper's Table 4. Every computed value is within 0.1 of the paper's "expected" column. Three rows round differently, because the paper's $\alpha$ and $c$ are themselves rounded. For example, $\alpha = 0.71$, $c = 0.11$, $\gamma = 3$ gives 1.93, where the paper lists 2.0.

The best $\gamma$ grows with $\alpha$ and shrinks with $c$. For $\alpha = 0.8$ the script gives the best $\gamma$ as 14, 11, 8, 6 for $c = 0.01, 0.02, 0.05, 0.1$, with speedups of 4.23, 3.82, 3.09, 2.47. At $\alpha = 0.3$ and $c = 0.2$, one guess gives 1.083, and a second makes things worse (0.993). Exercise 3 asks you to build these functions.

## Implement it

<!-- exercises -->

## Pitfalls

- **Dropping the residual.** If you reject and then resample from $p$, the output is $\min(p, q) + (1 - \beta)p$, not $p$. With the example above it is `[0.4, 0.46, 0.14]`. That is the common first bug, and only a distribution test catches it.
- **Resampling from $q$ or from $p - q$ without clipping.** The correction is $\operatorname{norm}(\max(0, p - q))$. Negative entries are not probabilities.
- **Off-by-one in the verification loop.** The residual for a rejection at guess $n + 1$ uses $p_{n+1}$ and $q_{n+1}$, the distributions at the rejected position. The bonus token exists only when all $\gamma$ guesses pass.
- **Forgetting that $\alpha$ is an average.** Equation 1 and Theorem 3.8 assume the $\beta$s are i.i.d. Real acceptance rates vary with the prefix. The paper says the theory matches the measurements "with some variance" (Section 4.1, Appendix A.3), and Table 4 shows the gaps, for example 2.5x expected against 1.7x measured for T5-large.
- **Comparing models with different sampling rules.** $p$ and $q$ must be the adjusted distributions (temperature, top-$k$, nucleus) of both models, with the same standardization. Otherwise the proof doesn't apply to the real sampler (Section 2.2).
- **Picking a draft that is too slow.** In Table 2, T5-large as the draft has the highest $\alpha$ but the lowest speedup (1.7x), because $c$ is 0.11. The paper says a draft about two orders of magnitude smaller usually works best (Section 3.6).
- **Expecting a win without spare compute.** Theorem 3.8 assumes $\gamma + 1$ parallel target evaluations cost the same as one. The paper tested batch size 1 on a TPU. At large batch sizes the accelerator is already busy, and the arithmetic cost in Theorem 3.11 starts to matter. The paper doesn't measure this regime, so treat the batched case as untested.
- **Rolling back state.** After a rejection, the draft's and the target's caches must drop the entries for discarded guesses. The paper's pseudocode leaves cache handling to the implementation.
- **Expecting a change in the output.** Every run of Algorithm 1 gives a sample from $p$, but the specific tokens differ from plain sampling with the same random seed, because the random draws are consumed in a different order. The guarantee is about the distribution. With argmax sampling the output is the same text.

## Further reading

- [Leviathan, Kalman, and Matias, "Fast Inference from Transformers via Speculative Decoding" (arXiv 2211.17192)](https://arxiv.org/abs/2211.17192). Read Appendix A.1 for the correctness proof and Appendix A.5 for the lenience variant, which trades exactness for a higher $\alpha$.
- [Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling" (arXiv 2302.01318)](https://arxiv.org/abs/2302.01318). An independent, concurrent paper with the same accept/reject rule.
- [Speculative decoding lesson](https://qwen.ifkash.dev/c/20-speculative-decoding) and [Decoding and sampling lesson](https://ml.ifkash.dev/lessons/075-decoding-and-sampling) in the course.
