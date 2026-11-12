---
id: "llada"
title: "Large Language Diffusion Models"
authors: "Nie et al."
year: 2025
venue: "arXiv preprint"
arxiv: "2502.09992"
url: "https://arxiv.org/abs/2502.09992"
track: generative
tier: extended
est_hours: 5
status: full
summary: "LLaDA is a language model that generates text by diffusion instead of left-to-right prediction. The forward process hides each token with a probability t that is drawn at random for every training sequence, and a Transformer with no causal mask learns to recover the hidden tokens under a cross-entropy loss weighted by 1/t, which bounds the negative log-likelihood. To generate, the model starts from an all-mask response, predicts every position at once, keeps the predictions it is most confident about, and masks the rest again, repeating until nothing is masked. Trained at 8 billion parameters on 2.3 trillion tokens, it is competitive with LLaMA3 8B on the benchmarks the paper reports."
related_course: "ml: diffusion language models"
prereqs:
  - id: ddpm
    reason: "LLaDA reuses the diffusion recipe of a fixed forward corruption process, a learned reverse process, and a training loss that bounds the likelihood, with token masking in place of Gaussian noise."
  - id: bert
    reason: "The mask predictor is a bidirectional Transformer trained to fill in masked tokens, and LLaDA replaces BERT's fixed 15 percent masking ratio with a ratio drawn uniformly at random."
  - id: transformer
    reason: "The network is a Transformer whose attention has no causal mask, so every position reads every other position in both directions."
background: [masked-language-modeling, cross-entropy, autoregressive-language-modeling, elbo, expectation]
course_links:
  - title: "Diffusion language models"
    url: "https://ml.ifkash.dev/lessons/087-diffusion-language-models"
  - title: "Diffusion models"
    url: "https://ml.ifkash.dev/lessons/086-diffusion-models"
self_check:
  - q: 'A sequence has $L = 8$ tokens and the forward process draws $t = 0.25$. How many tokens does it mask on average, what weight does Equation 3 give the loss, and how likely is a draw that masks nothing?'
    a: 'Each token is masked with probability $t$, so the mean is $L t = 2$ tokens. The loss weight is $1/t = 4$. The chance that no token is masked is $(1 - t)^L = 0.75^8 \approx 0.1001$. A draw with no masks contributes zero loss, which is part of why the paper averages over many values of $t$.'
    see: masked-language-modeling
  - q: 'In a four-token sequence, positions 2 and 4 are masked at $t = 0.5$. The model gives the true tokens probability $0.5$ and $0.25$ there. What is the loss of Equation 3, and what does Algorithm 1 report?'
    a: 'The sum of negative log-probabilities over the masked positions is $\ln 2 + \ln 4 = 2.0794$. Equation 3 multiplies it by $1/t = 2$ and gives $4.1589$. Algorithm 1 also divides by the sequence length $L = 4$, which gives $1.0397$.'
    see: cross-entropy
  - q: 'DDPM adds Gaussian noise to an image. Why can a language model not add a "little noise" to a token, and what plays that role in LLaDA?'
    a: 'A token is a discrete symbol, so there is no nearby value that is slightly wrong. The forward process instead replaces tokens by a special mask symbol, one at a time and independently, and the replacement ratio $t$ plays the role of the noise level. At $t = 0$ the text is clean, and at $t = 1$ every token is hidden, just as the final DDPM latent is pure noise.'
    see: ddpm
---

## The problem

Nearly every large language model in 2025 predicts text one token at a time, from left to right. The paper calls this the autoregressive formulation, and asks whether it is the only way to reach the abilities of a large language model, such as scaling, in-context learning, and instruction following (Section 1). The paper argues that those abilities come from generative modeling principles, and not from the left-to-right factorization.

Left-to-right generation has concrete costs:

- **Generation is sequential.** The model commits to each token before it has produced the ones after it, and can't revise an earlier token.
- **The context is one-sided.** Token $i$ is predicted from tokens $1, \dots, i-1$ only. The paper tests this with a reversal task, where the model must complete a Chinese poem line from the line that follows it. GPT-4o scores 82.7 on the forward direction and 34.3 on the reversal (Table 4).

You can't fix this by reusing BERT, because BERT's masking ratio is fixed. A model trained at one ratio is not a generative model: it has no principled way to start from nothing and produce a sentence. Image diffusion models solved the same problem for pixels, but their forward process adds Gaussian noise, and a token has no meaningful "slightly noisy" version.

The paper keeps the diffusion recipe and swaps the noise for masking. It calls the result a masked diffusion model, and trains it at scale as LLaDA (Large Language Diffusion with mAsking). The 8B model reaches 65.9 on MMLU against 65.4 for LLaMA3 8B, and 70.3 on GSM8K against 48.7 (Table 1). On the reversal task it scores 45.6, above GPT-4o (Table 4).

## Key ideas

The sections that follow use the paper's notation. A sequence is $x_0 = (x_0^1, \dots, x_0^L)$ of length $L$, and $M$ is the mask token. The mask ratio is $t \in (0, 1]$, the noisy sequence is $x_t$, and the network is the mask predictor $p_\theta(\cdot \mid x_t)$ (Section 2.1). Every number comes from a script in `scripts/`. Run any script from the paper folder to reproduce the output. The toy examples use short sequences and a made-up vocabulary so that you can check them by hand. They aren't results from the paper.

### Masking as the forward process

**Intuition.** Take a sentence and hide each word with probability $t$, deciding independently for every word. At $t = 0$ you see the whole sentence. At $t = 1$ you see nothing. In between, you see a random subset of the words. This plays the part of DDPM's noise schedule, but it is a continuous knob that turns a discrete sequence into blanks.

**Math.** For $t \in (0, 1)$, the forward process masks each token independently (Section 2.1, with the full definition in Appendix A.1):

$$
q(x_t^i \mid x_0^i) = \begin{cases} 1 - t & \text{if } x_t^i = x_0^i, \\ t & \text{if } x_t^i = M. \end{cases}
$$

The number of masked tokens is therefore binomial with $L$ trials and success probability $t$. The paper draws $t$ uniformly from $[0, 1]$ for each training sequence, so the mask ratio varies from sequence to sequence, and a full-mask sequence at $t = 1$ is the starting point for sampling.

**Tiny example, computed by `scripts/forward_masking.py`.** Take $L = 8$ and the sequence $(5, 3, 9, 1, 7, 2, 8, 4)$ with seed 0 and $t = 0.5$. The script masks positions 2, 3, and 4 and prints $x_t = (5, M, M, M, 7, 2, 8, 4)$. Over 200,000 sequences the average number of masks is:

| $t$ | Expected masks $L t$ | Empirical mean | $P(\text{no mask}) = (1-t)^8$ |
|---|---|---|---|
| 0.25 | 2.00 | 1.9997 | 0.1001 |
| 0.50 | 4.00 | 3.9984 | 0.0039 |
| 0.75 | 6.00 | 6.0034 | 0.0000 |
| 1.00 | 8.00 | 8.0000 | 0.0000 |

Exercise 1 asks you to implement this process.

### The masked cross-entropy loss and its bound

**Intuition.** Train the model to fill in the blanks. For each masked position, ask for the probability that the model gives the original token, and take the negative log. The unmasked positions are visible input, so they cost nothing. The only subtlety is the weight: a sequence with few masks has few terms in its sum, so the loss divides by $t$ to compensate. Then every value of $t$ contributes on the same scale.

**Math.** The pre-training objective is (Section 2.1, Equation 3):

$$
\mathcal{L}(\theta) = -\mathbb{E}_{t,\, x_0,\, x_t}\!\left[\frac{1}{t} \sum_{i=1}^{L} \mathbf{1}[x_t^i = M]\, \log p_\theta(x_0^i \mid x_t)\right]
$$

The indicator restricts the sum to masked positions, and $p_\theta$ predicts all masked tokens in one forward pass. The paper proves that this loss is an upper bound on the negative log-likelihood of the model (Equation 4, derived in Appendix A.1):

$$
-\mathbb{E}_{x_0 \sim p_{\text{data}}}\left[\log p_\theta(x_0)\right] \le \mathcal{L}(\theta)
$$

That bound is what separates LLaDA from BERT and from MaskGIT. BERT uses a fixed ratio and has no such bound. The paper says MaskGIT drops the $1/t$ weight, which makes its objective a heuristic (Section 2.1). Algorithm 1, the pre-training loop, also divides by $L$, so the number it reports is Equation 3 per token. This page keeps Equation 3 as written.

The network is a Transformer with no causal mask, so each prediction reads the whole partially masked sequence in both directions (Section 2.2). The paper sets it up like LLaMA, with RMSNorm, SwiGLU, and RoPE, and uses ordinary multi-head attention.

**Tiny example, computed by `scripts/masked_loss.py`.** Take $L = 4$, with positions 2 and 4 masked at $t = 0.5$. The model gives the true tokens probability $0.5$ and $0.25$. Then:

| Quantity | Value |
|---|---|
| $-\log 0.5 - \log 0.25$ | 2.0794 |
| Equation 3, multiplied by $1/t = 2$ | 4.1589 |
| Algorithm 1, divided by $L = 4$ | 1.0397 |

The value 4.1589 equals $6 \ln 2$. As a check on the weight, consider a model that is uniform over $V = 16$ tokens. Its negative log-likelihood for $L = 8$ is exactly $L \ln V = 22.1807$ for every sequence. With the $1/t$ weight, the expected loss equals that for every $t$, because the mean number of masks is $L t$ and the weight cancels the $t$. Over 400,000 draws:

| $t$ | Mean loss without $1/t$ | Mean loss with $1/t$ |
|---|---|---|
| 0.1 | 2.2166 | 22.1659 |
| 0.5 | 11.0950 | 22.1901 |
| 0.9 | 19.9651 | 22.1835 |

Without the weight, the loss grows with $t$ and depends on which ratio you draw. Exercise 2 asks you to implement Equation 3 and to run this check.

### Conditioning on a prompt

**Intuition.** To follow instructions, the model must see the prompt and produce a response. Keep the prompt visible at every noise level and mask only the response. The loss then trains the response distribution given the prompt.

**Math.** For a prompt $p_0$ and a response $r_0$ of length $L'$, the supervised fine-tuning loss is (Section 2.3, Equation 5):

$$
-\mathbb{E}_{t,\, p_0,\, r_0,\, r_t}\!\left[\frac{1}{t} \sum_{i=1}^{L'} \mathbf{1}[r_t^i = M]\, \log p_\theta(r_0^i \mid p_0, r_t)\right]
$$

Pre-training is the special case with an empty prompt. The paper pads short responses with end-of-sequence tokens to a common length and counts those tokens as part of the response, so the model learns to predict where the text ends (Section 2.3).

**Tiny example, computed by `scripts/forward_masking.py`.** With a 3-token prompt and a 5-token response at $t = 0.4$, the prompt is never masked. The response has $5 \times 0.4 = 2.00$ masks on average, and the simulation gives 2.0001.

### The sampler and remasking

**Intuition.** Generation runs the forward process backward. Start with a response that is entirely $M$ ($t = 1$). At each step, let the model predict every masked position at once. Then keep some of those predictions and mask the others again, so that the sequence has fewer masks at the next step. After $N$ steps nothing is masked. The open question is which predictions to keep.

**Math.** The reverse process moves from $t = 1$ to $t = 0$ in $N$ uniformly spaced steps (Section 2.4, Appendix A.3). At a step from $t$ to $s < t$, the model predicts all masked tokens, and a fraction $s/t$ of the predicted tokens returns to $M$, so that the expected mask ratio at $s$ is $s$. The paper describes two ways to choose which tokens to remask:

- **Random remasking** (Algorithm 4) chooses the tokens to remask at random. This matches the reverse process of the model.
- **Low-confidence remasking** (Algorithm 5) scores each prediction with the probability the model assigns to it, and remasks the least confident ones. The paper follows MaskGIT here and says it is inspired by the way annealing sharpens sampling in autoregressive models (Section 2.4).

Both algorithms use greedy decoding for the predicted tokens. After the step to time $s$, a sequence of length $L$ has $n_{un} = \lfloor L(1 - s) \rfloor$ unmasked tokens (Algorithm 5). For $s = 1 - k/N$ after $k$ steps, this is $\lfloor L k / N \rfloor$, which you can compute in integer arithmetic. Already unmasked tokens stay fixed, and the exercises give them a confidence of infinity. The paper also describes semi-autoregressive block sampling, which splits the response into blocks and fills them from left to right. It needs no retraining (Section 2.4).

**Tiny example, computed by `scripts/sampler.py`.** The script uses a toy deterministic model with a table of confidences. It is not a trained network. The target is $(2, 0, 3, 1, 2, 0)$. The model puts probability $c$ on the right token at each position and $1 - c$ on one wrong token. The value $c$ starts at 0.9 for positions 1, 3, and 5 and at 0.4 for positions 2, 4, and 6, and it grows by 0.15 for each unmasked neighbor, up to 0.95. A position with $c < 0.5$ is predicted wrongly, so the hard positions are only right once their neighbors are visible.

With $L = 6$ and $N = 3$, the script prints $n_{un} = 2, 4, 6$ and this trace of low-confidence remasking ($M$ is the mask):

| Step | Sequence after the step |
|---|---|
| 1 | $(2, M, 3, M, M, M)$ |
| 2 | $(2, 0, 3, M, 2, M)$ |
| 3 | $(2, 0, 3, 1, 2, 0)$ |

At step 1, the three easy positions all have confidence 0.9, and the tie goes to the lower indices 1 and 3. The hard positions are still masked when their neighbors become visible, so they come out right. Random remasking averages a token accuracy of 0.7538 over 2000 seeds, and only 14.60 percent of runs are fully correct. A single step ($N = 1$) predicts all positions from the all-mask input and gets 0.5 token accuracy. Exercise 3 asks you to implement one step of each strategy, and Exercise 4 asks you to build the full sampler.

<!-- exercises -->

## Pitfalls

- **Dropping the $1/t$ weight.** The loss still trains, so nothing fails loudly. But it no longer bounds the likelihood, and the model weights low-$t$ sequences differently from the paper. This is the objective the paper contrasts with MaskGIT (Section 2.1).
- **Weighting by $t$ instead of $1/t$.** The weight must grow as fewer tokens are masked. The uniform-model check in the exercise fails for this bug.
- **Summing over every position.** The indicator in Equation 3 selects masked positions. Visible tokens are inputs and carry no loss, and a mean over all positions divides by the wrong count.
- **Comparing losses across normalizations.** Equation 3 is a sum, and Algorithm 1 divides by $L$. A logged value can differ by a factor of $L$ between codebases.
- **Large variance at small $t$.** A draw with $t$ near 0 usually masks few or no tokens, and when it masks one, the weight is large. Training averages over many draws, so the noise from small $t$ averages out across a batch. For evaluation, the paper reports a lower-variance estimator that masks exactly $l$ tokens, with $l$ uniform on $\{1, \dots, L\}$, and weights by $L/l$ (Section 2.4, Equation 6, Algorithm 3).
- **Treating the bound as the likelihood.** Equation 4 is an upper bound on the negative log-likelihood, so a perplexity from it is an upper bound too.
- **Float rounding in the step count.** $1 - 2/3$ in floating point is slightly above $1/3$, and similar roundings can drop or add a token at a step. Use $\lfloor L k / N \rfloor$ with integers.
- **Remasking tokens you already committed.** In the sampler on this page, a token that is visible stays visible, with confidence infinity. If a variant lets low-confidence tokens return to $M$, the number of unmasked tokens no longer follows $n_{un}$. The paper's text and its pseudocode for Algorithm 5 describe the remasking count in different terms, so read the paper and the official code together before you port it.
- **Reading the toy table as a result.** The toy model in `scripts/sampler.py` is built to make confidence informative. It shows how the algorithm works, and it doesn't measure how much low-confidence remasking helps a trained model.
- **Padding and early stopping.** The model pads short SFT responses with end-of-sequence tokens, so low-confidence remasking can favor those tokens and finish early. The paper attributes the gain from block sampling on GSM8K (78.6 against 69.4) and Math (42.2 against 31.9) to this effect (Section 3.2, Appendix B.4).
- **Setting the generation length.** The response length is a fixed hyperparameter, and the sampler discards tokens after the first end-of-sequence token. The paper reports that results are insensitive to it (Appendix B.5).

## Further reading

- [The official LLaDA code and weights](https://github.com/ML-GSAI/LLaDA) are linked from the paper and contain the sampler that produced the reported results.
- [Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524), by Sahoo et al., and [Simplified and Generalized Masked Diffusion for Discrete Data](https://arxiv.org/abs/2406.04329), by Shi et al., derive the masked diffusion objective that LLaDA scales up.
- [MaskGIT: Masked Generative Image Transformer](https://arxiv.org/abs/2202.04200), by Chang et al., introduced the confidence-based decoding that Algorithm 5 follows.
- [Structured Denoising Diffusion Models in Discrete State-Spaces](https://arxiv.org/abs/2107.03006), by Austin et al., defines the absorbing-state diffusion that masking is a case of.
- [Denoising Diffusion Probabilistic Models](/ddpm/) and [BERT](/bert/) are the two papers this page combines.
