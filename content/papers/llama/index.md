---
id: "llama"
title: "LLaMA: Open and Efficient Foundation Language Models"
authors: "Touvron et al."
year: 2023
venue: "arXiv preprint"
arxiv: "2302.13971"
url: "https://arxiv.org/abs/2302.13971"
track: nlp-llm
tier: core
est_hours: 4
status: full
summary: "LLaMA is a family of decoder-only language models from 7B to 65B parameters, trained only on public data, whose architecture became the default recipe for open models: RMSNorm applied before each sub-layer, a SwiGLU feed-forward network, and rotary position embeddings in every layer. The authors train each model on far more tokens than the compute-optimal count from the Chinchilla paper, because a smaller model that sees more data costs less to serve. The result is that the 13B model beats the 175B GPT-3 on most benchmarks and the 65B model is competitive with Chinchilla-70B and PaLM-540B."
related_course: "ml: LLM architecture"
prereqs:
  - id: gpt3
    reason: "LLaMA keeps the GPT-3 decoder-only design, including the pre-normalization of sub-layer inputs that Section 2.2 credits to GPT-3, and its headline claim is a comparison with the 175B model."
  - id: rope
    reason: "LLaMA drops absolute position embeddings and rotates the query and key vectors in every layer, so you need the rotation and the relative-offset property from RoFormer."
  - id: layernorm
    reason: "RMSNorm is LayerNorm with the mean subtraction and the bias removed, so you need to know what LayerNorm computes and why it works per example."
  - id: chinchilla
    reason: "The paper's training choice is defined against Chinchilla's compute-optimal split of parameters and tokens, and its argument is that the split ignores the cost of serving the model."
background: [rmsnorm, swiglu]
course_links:
  - title: "The transformer block: SwiGLU, residuals, and layer norm"
    url: "https://ml.ifkash.dev/lessons/050-the-transformer-block"
  - title: "RMSNorm and the residual stream"
    url: "https://qwen.ifkash.dev/c/04-rmsnorm-and-residuals"
self_check:
  - q: 'Normalize $x = [1, 2, 3, 4]$ with RMSNorm (gain 1, no $\epsilon$). What is the result, and how does it differ from LayerNorm on the same vector?'
    a: 'The mean square is $30/4 = 7.5$, so the RMS is $2.7386$ and the result is $[0.3651, 0.7303, 1.0954, 1.4606]$. LayerNorm first subtracts the mean 2.5 and divides by the standard deviation 1.1180, which gives $[-1.3416, -0.4472, 0.4472, 1.3416]$. RMSNorm keeps the mean and divides by the root mean square.'
    see: rmsnorm
  - q: 'A feed-forward network with ReLU, width $d$ in and out, and hidden width $4d$ has $8d^2$ weights. How wide must a SwiGLU hidden layer be to match that count, and why?'
    a: 'SwiGLU has three weight matrices ($W_1$, $V$, $W_2$) and not two, so each of them can have only $2/3$ of the width: $3 \cdot d \cdot h = 8d^2$ gives $h = \tfrac{2}{3} \cdot 4d = 8d/3$. This is the "$\tfrac{2}{3}4d$" in Section 2.2.'
    see: swiglu
  - q: 'The Chinchilla rule of thumb is about 20 tokens per parameter. LLaMA-7B sees 1.0T tokens. Roughly how many tokens per parameter is that, and what does the paper trade for the extra training cost?'
    a: 'About 150 tokens per parameter, roughly 7 times the Chinchilla count. The extra training compute buys a smaller model that is cheaper at inference, which the Chinchilla objective ignores (Section 1).'
    see: chinchilla
---

## The problem

By early 2023, the strongest language models were large and closed. GPT-3 has 175B parameters, Gopher 280B, and PaLM 540B. Each of them trained on data that the public can't inspect, such as "Books - 2TB" or "Social media conversations" (Section 1). Two problems follow.

The first is cost at serving time. The Chinchilla paper showed that, for a fixed training budget, a smaller model trained on more tokens beats a larger one trained on fewer. But that objective counts only training compute. A model that serves billions of requests pays its inference cost on every one of them. Section 1 makes the point with an example: Hoffmann et al. recommend training a 10B model on 200B tokens, but the authors find that a 7B model keeps improving after 1T tokens. Weights alone show the gap. At 2 bytes per parameter, GPT-3 needs 350 GB, which is at least five 80 GB GPUs before you store any activations. LLaMA-13B needs 26 GB and fits on one 32 GB GPU. The paper says the 13B model "runs on a single V100 GPU" (Section 3.2). `scripts/training_budget.py` computes these numbers.

The second is access. Of the open models the paper lists in Section 1 (OPT, GPT-NeoX, BLOOM, GLM), none is competitive with PaLM-62B or Chinchilla. The paper asks whether a model trained only on public data can reach that level, and whether a recipe tuned for inference cost, not training cost, gets there with a model that fits on a single GPU. Section 3 reports that it can. LLaMA-13B outperforms GPT-3 on most benchmarks despite being 10 times smaller, and LLaMA-65B beats Chinchilla-70B on all reported common sense benchmarks except BoolQ (Table 3).

## Key ideas

This page uses $d$ for the model width, which the paper calls "dimension" in Table 2. It uses $h$ for the feed-forward hidden width, $H$ for the number of heads, $N$ for the parameter count, and $D$ for the number of training tokens. The LLaMA paper is short and prints no equations for its three architecture changes. It cites the papers that define them, so the formulas below come from those papers. The page says so wherever that is the case.

### Pre-norm with RMSNorm

**Intuition.** A transformer block has two sub-layers, attention and the feed-forward network. The original Transformer normalizes after each sub-layer, on the sum of the input and the output. LLaMA normalizes the input of each sub-layer and leaves the residual stream alone. The paper's reason is training stability (Section 2.2). The normalizer is RMSNorm, which is cheaper than LayerNorm: it skips the mean subtraction and the bias.

**Math.** Section 2.2 says "we normalize the input of each transformer sub-layer, instead of normalizing the output" and uses RMSNorm from Zhang and Sennrich (2019). The paper doesn't print the formula. In that paper's definition, for a vector $x \in \mathbb{R}^d$ and a learned gain $g \in \mathbb{R}^d$:

$$
\operatorname{RMSNorm}(x)_i = \frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^{d} x_j^2 + \epsilon}}\; g_i
$$

Here $\epsilon$ is a small constant such as $10^{-6}$ that keeps the division finite on a zero vector. Compare this with LayerNorm from [the layer normalization paper](/layernorm/), which computes $(x_i - \mu)/\sqrt{\sigma^2 + \epsilon}$ with a gain and a bias. RMSNorm drops $\mu$ and the bias. It keeps the part of LayerNorm that makes the output independent of the input's scale, and it gives up independence from the input's mean.

With pre-norm, a block computes

$$
x \leftarrow x + \operatorname{Attention}(\operatorname{RMSNorm}(x)), \qquad x \leftarrow x + \operatorname{FFN}(\operatorname{RMSNorm}(x)).
$$

The residual stream $x$ passes through each block unnormalized, and each sub-layer sees a unit-scale input.

**Tiny example, computed by `scripts/rmsnorm_example.py`.** Take $x = [1, 2, 3, 4]$. The mean square is 7.5 and the RMS is 2.7386, so

| Input | Output |
|---|---|
| RMSNorm of $x$ | `[0.3651, 0.7303, 1.0954, 1.4606]` |
| RMSNorm of $10x$ | `[0.3651, 0.7303, 1.0954, 1.4606]` (the same) |
| LayerNorm of $x$ | `[-1.3416, -0.4472, 0.4472, 1.3416]` |
| RMSNorm of $x + 10$ | `[0.8765, 0.9562, 1.0359, 1.1155]` (changed) |
| LayerNorm of $x + 10$ | `[-1.3416, -0.4472, 0.4472, 1.3416]` (the same) |

RMSNorm is scale invariant but not shift invariant. For a vector with zero mean, such as $[-3, -1, 1, 3]$, the two normalizers give identical outputs. The script confirms it.

Pre-norm has one more consequence. The sub-layer's input is the same whether the stream is $x$ or $100x$, so its output has the same size in both cases. In the script, a stand-in update of size 1.0 (the norm of the normalized input) is 0.3651 times the size of the stream $x$, but only 0.0037 times the size of the stream $100x$. As the stream grows, the later updates change it by a smaller fraction. That is also the reason for stability: the stream gets no rescaling, so the gradient has a direct identity path through every block.

The saving in parameters is small. For LLaMA-7B ($d = 4096$, 32 layers), the model has $2 \cdot 32 + 1 = 65$ norm layers if you count one final norm before the output layer. The final norm comes from the released code, not from the paper text. RMSNorm needs $65 \cdot 4096 = 266{,}240$ gains. A LayerNorm with a bias would need 532,480. The real saving is the removed mean and its reduction, not the parameters.

Exercise 1 asks you to implement `rmsnorm`.

### SwiGLU feed-forward

**Intuition.** A gated linear unit multiplies two projections of the input elementwise. One projection passes through a nonlinearity and acts as a gate. The other carries the content. SwiGLU uses SiLU, also called swish, as the gate nonlinearity. The gate decides, per hidden unit and per token, how much of the content to pass on.

**Math.** Section 2.2 says "We replace the ReLU non-linearity by the SwiGLU activation function, introduced by Shazeer (2020)". It doesn't print the formula either. In Shazeer's form, with $x \in \mathbb{R}^d$, $W_1, V \in \mathbb{R}^{d \times h}$, and $W_2 \in \mathbb{R}^{h \times d}$:

$$
\operatorname{FFN}_{\text{SwiGLU}}(x) = \big(\operatorname{SiLU}(x W_1) \odot (x V)\big)\, W_2, \qquad \operatorname{SiLU}(a) = a\,\sigma(a) = \frac{a}{1 + e^{-a}}
$$

where $\odot$ is the elementwise product. The Transformer's feed-forward network (Section 3.3 of [Attention Is All You Need](/transformer/), Equation 2) is $\max(0, xW_1 + b_1)W_2 + b_2$. It has two matrices. SwiGLU has three and no biases.

Three matrices at the old width would cost 50% more weights. To keep the cost the same, the paper shrinks the hidden width: "We use a dimension of $\tfrac{2}{3}4d$ instead of $4d$ as in PaLM" (Section 2.2). The ReLU network has $2 \cdot d \cdot 4d = 8d^2$ weights, and the SwiGLU network has $3 \cdot d \cdot h$, so $h = \tfrac{8}{3}d$ gives the same count.

**Tiny example, computed by `scripts/swiglu_example.py`.** Let $d = 2$, $h = 3$, $x = [1, 2]$, and

$$
W_1 = \begin{bmatrix} 1 & 0 & -1 \\ 0 & 1 & 1 \end{bmatrix}, \quad
V = \begin{bmatrix} 1 & 1 & 0 \\ 0 & 1 & 1 \end{bmatrix}, \quad
W_2 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}.
$$

The steps are:

| Step | Result |
|---|---|
| $xW_1$ | `[1, 2, 1]` |
| $xV$ | `[1, 3, 2]` |
| $\operatorname{SiLU}(xW_1)$ | `[0.7311, 1.7616, 0.7311]` |
| Gated product with $xV$ | `[0.7311, 5.2848, 1.4621]` |
| Multiply by $W_2$ | `[2.1932, 6.7469]` |

Put the SiLU on the $V$ branch instead and the output is `[2.4927, 7.4770]`, so the two roles aren't interchangeable. Unlike ReLU, SiLU is smooth and dips slightly below zero: $\operatorname{SiLU}(-1) = -0.2689$, and $\operatorname{SiLU}(-10) = -0.00045$.

For $d = 4096$, the script counts the weights:

| Network | Hidden width | Weights |
|---|---|---|
| ReLU or GELU MLP | 16,384 | 134,217,728 |
| SwiGLU at the same width | 16,384 | 201,326,592 (1.5 times) |
| SwiGLU at $\tfrac{2}{3}4d$ | 10,922 | 134,209,536 |
| SwiGLU at 11,008 | 11,008 | 135,266,304 (1.0078 times) |

The last row needs a note. $\tfrac{2}{3} \cdot 4 \cdot 4096 = 10922.67$ isn't a multiple of anything useful. The released code rounds the width up to a multiple of 256, which gives 11,008, and the paper doesn't mention this. Section 2.2's "$\tfrac{2}{3}4d$" is the rule, and the rounded widths for the other sizes are 13,824, 17,920, and 22,016. The next section shows that those widths reproduce the parameter counts in Table 2.

Exercise 2 asks you to implement `silu`, `swiglu_ffn`, and `ffn_hidden`.

### Rotary embeddings in every layer

**Intuition.** The model has no learned or sinusoidal position vector added to the input. Instead, each layer rotates the query and key vectors by an angle proportional to the token's position. The attention score then depends on how far apart two tokens are, and not on where they are.

**Math.** Section 2.2 says: "We remove the absolute positional embeddings, and instead, add rotary positional embeddings (RoPE), introduced by Su et al. (2021), at each layer of the network." The full derivation is on [the RoFormer page](/rope/). The summary is that pair $j$ of a vector at position $m$ turns by $m\theta_j$, with $\theta_j = 10000^{-2j/d_{\text{head}}}$, and the score between a query at position $m$ and a key at position $n$ is a function of $n - m$ only (RoFormer, Equation 16). Three details matter for LLaMA:

- The rotation applies to the queries and keys of each head, after the projections $W_q$ and $W_k$. It doesn't apply to the values.
- It repeats in every layer, because the rotation acts inside attention. A model with RoPE has no position input to its first layer.
- It adds no parameters.

**Tiny example, computed by `scripts/rope_example.py`.** With $d_{\text{head}} = 4$, the angles are $\theta = [1, 0.01]$. The vector $[1, 2, 3, 4]$ at position 1 becomes `[-1.1426, 1.9221, 2.9599, 4.0298]` when you rotate adjacent pairs, which is RoFormer's convention. For random 8-dimensional $q$ and $k$ (seed 0), the score $q_m \cdot k_n$ is $-1.467987$ at offsets 0 from positions 0, 5, and 40, and $-1.763129$ at offset $m - n = 2$ from $(3, 1)$, $(10, 8)$, and $(50, 48)$.

A pairing convention matters for loading weights. Pairing coordinate $j$ with coordinate $j + d/2$ also gives the relative property, with different numbers: the same vector at position 1 becomes `[-1.9841, 1.9599, 2.4624, 4.0198]`, and the score at $(3, 1)$ becomes $-3.619648$. Exercise 3 uses adjacent pairs. See Pitfalls.

Exercise 3 asks you to implement `apply_rope`.

### The block, and what the parameters add up to

**Intuition.** Put the three changes together and you have a decoder block that is otherwise the standard one. Table 2 lists the model sizes. A parameter count that you derive from the three changes can be checked against it.

**Math.** One block has, with $H$ heads of width $d/H$ and the weights in order, $W_q, W_k, W_v, W_o \in \mathbb{R}^{d \times d}$ for attention, $W_1, V \in \mathbb{R}^{d \times h}$ and $W_2 \in \mathbb{R}^{h \times d}$ for the feed-forward network, and two gains of size $d$:

$$
\text{params per block} = 4d^2 + 3dh + 2d
$$

The full model adds an input embedding and an output matrix, each of size $v \times d$ for a vocabulary of $v$ tokens, and one final gain of size $d$. The paper says it tokenizes with SentencePiece BPE (Section 2.1) but doesn't give $v$. The count below uses $v = 32{,}000$ from the released tokenizer, no biases, and untied embeddings, all of which come from the released code and not from the paper.

**Tiny example, computed by `scripts/param_count.py`.** With $h$ rounded up to a multiple of 256, the formula gives:

| Model | $d$ | Layers | $h$ | Computed | Table 2 |
|---|---|---|---|---|---|
| LLaMA-7B | 4096 | 32 | 11,008 | 6.74B | 6.7B |
| LLaMA-13B | 5120 | 40 | 13,824 | 13.02B | 13.0B |
| LLaMA-33B | 6656 | 60 | 17,920 | 32.53B | 32.5B |
| LLaMA-65B | 8192 | 80 | 22,016 | 65.29B | 65.2B |

The 65B count is 0.1% above Table 2's 65.2B, so one of the assumptions is slightly off for that size, or the paper truncates. The other three agree after rounding. Every model has a head width of 128 (4096/32, 5120/40, 6656/52, 8192/64). In LLaMA-7B, the feed-forward network holds 64.2% of the weights, attention 31.9%, and the two embedding matrices 3.9%.

Exercise 4 asks you to assemble the block from RMSNorm, SwiGLU, and RoPE, with a causal mask.

### Train smaller models on more tokens

**Intuition.** Chinchilla chooses the model size that minimizes loss for a fixed training budget. LLaMA chooses the model size that minimizes cost at a target quality, when the model will be queried many times. Serving a token costs about $2N$ FLOPs, so each parameter you remove saves compute on every token you ever generate. The extra training pays for itself once enough tokens are served.

**Math.** With the usual estimates, training costs $C_{\text{train}} \approx 6ND$ FLOPs and serving costs $C_{\text{serve}} \approx 2N D_{\text{served}}$ FLOPs. Neither estimate is in the LLaMA paper; both come from the scaling-law literature, as in [Chinchilla](/chinchilla/). The paper's own statements are the model and token counts in Table 2 and the argument in Section 1.

**Tiny example, computed by `scripts/training_budget.py`.** The tokens per parameter in Table 2 are:

| Model | Parameters | Tokens | Tokens per parameter | Multiple of 20 per parameter |
|---|---|---|---|---|
| LLaMA-7B | 6.7B | 1.0T | 149 | 7.5 |
| LLaMA-13B | 13.0B | 1.0T | 77 | 3.8 |
| LLaMA-33B | 32.5B | 1.4T | 43 | 2.2 |
| LLaMA-65B | 65.2B | 1.4T | 21.5 | 1.1 |

The 65B model sits close to the Chinchilla ratio. The smaller models go far past it. Training the 65B model costs $6 \cdot 65.2 \times 10^9 \cdot 1.4 \times 10^{12} = 5.48 \times 10^{23}$ FLOPs.

The paper's own throughput figure checks out. Section 2.4 reports about 380 tokens per second per GPU on 2048 A100 GPUs, and 1.4T tokens at that rate take 20.8 days, which matches "approximately 21 days". The same pace gives 1,023,392 GPU-hours against Table 15's 1,022,362. Section 6 turns that into energy: $1{,}022{,}362 \times 400\,\text{W} \times 1.1 = 449.8$ MWh, and at 0.385 kg CO2eq per kWh, 173 tCO2eq, as in Table 15.

Now the trade-off with an assumed pair of models. Suppose, hypothetically, that a 10B model on 200B tokens and a 7B model on 1T tokens reach the same quality. The first costs $1.2 \times 10^{22}$ FLOPs to train and the second $4.2 \times 10^{22}$, so the smaller model costs $3.0 \times 10^{22}$ more. Serving saves $2 \cdot (10 - 7) \times 10^9 = 6 \times 10^9$ FLOPs per token, so the break-even is $5 \times 10^{12}$ served tokens. Past that point, the smaller model is cheaper in total. The paper doesn't claim equal quality for these two models and the numbers are an illustration of the argument, not a result.

The training recipe is standard. Section 2.3 uses AdamW with $\beta_1 = 0.9$ and $\beta_2 = 0.95$, a cosine schedule that ends at 10% of the peak rate, weight decay 0.1, gradient clipping at 1.0, and 2,000 warmup steps. The peak rate is $3.0 \times 10^{-4}$ for the 7B and 13B models and $1.5 \times 10^{-4}$ for the 33B and 65B models, with a batch of 4M tokens (Table 2). At 4M tokens per step, 1.4T tokens is about 350,000 optimizer steps. The $\beta_2 = 0.95$ differs from the 0.999 default of [Adam](/adam/), and a lower value reacts faster to changes in the gradient scale. The paper doesn't give a reason.

<!-- exercises -->

## Pitfalls

- **Reading RMSNorm as LayerNorm with a different name.** RMSNorm doesn't subtract the mean, so adding a constant to every feature changes its output, while it doesn't change LayerNorm's. The tiny example shows the difference. If a model has large mean activations, the two aren't interchangeable. Exercise 1 tests for it.
- **Putting the epsilon in the wrong place.** The $\epsilon$ goes inside the square root. Adding it after the root gives a different result when $\epsilon$ isn't tiny, and exercise 1 catches it.
- **Giving SwiGLU the old hidden width.** Using $4d$ with three matrices raises the feed-forward weights by 50%, and the model is no longer the one in Table 2. The width is $\tfrac{2}{3}4d$, rounded up to a multiple of 256 in the released code. Exercise 2 tests the widths for all four sizes.
- **Putting the gate on the wrong branch.** SiLU applies to one of the two projections. Swapping them gives a different function with the same shapes, and nothing crashes. Check against the checkpoint's naming: in the released code, `w1` is the gate, `w3` is the value branch, and `w2` is the output projection. Those names come from the code, not the paper.
- **Mixing RoPE pairing conventions.** The RoFormer paper and the original code rotate adjacent pairs. Some ports, including the Hugging Face one, pair coordinate $j$ with $j + d/2$ and permute the weights of $W_q$ and $W_k$ at conversion time to compensate. Loading weights with the wrong convention gives wrong output without an error. Exercise 3 follows adjacent pairs.
- **Rotating the values, or rotating before the projections.** RoPE applies to $q$ and $k$ after $W_q$ and $W_k$. Exercise 4 has tests that fail for both mistakes.
- **Normalizing the residual stream.** In pre-norm, only the input of each sub-layer is normalized. If you also normalize the sum, you have a post-norm block. Exercise 4 checks that the update doesn't depend on the scale of the stream.
- **Reading "competitive" as "better".** On Table 9 (MMLU, five-shot), LLaMA-65B averages 63.4, below Chinchilla-70B at 67.5 and PaLM-540B at 69.3. Section 3.6 offers a potential explanation: LLaMA used 177 GB of books and academic papers, while the other models saw up to 2 TB of books. After a small amount of instruction finetuning, LLaMA-I reaches 68.9 (Section 4, Table 10). In Table 3, the 65B model trails PaLM-540B on BoolQ and WinoGrande. The strong claim is the one about size: 13B beats GPT-3 175B on most tasks.
- **Calling every LLaMA model Chinchilla-optimal.** Only the 65B model is near 20 tokens per parameter. The 7B model trained on about 150, deliberately. The paper's argument is about inference cost, and it doesn't claim that these token counts minimize training loss for the budget.
- **Treating "open" as "freely licensed".** The paper's claim is that the training data is public (Section 2.1) and that the models are released "to the research community". It doesn't describe the license terms of the weights.
- **Trusting parameter counts without the assumptions.** The count in this page needs a 32,000-token vocabulary, untied embeddings, no biases, and the 256 rounding. None of them is in the paper text. If you change any of them, the totals move by 1% or more.

## Further reading

- [The LLaMA code release](https://github.com/facebookresearch/llama), linked from the paper, has the reference model code, including the norm, the feed-forward network, and the rotary embedding.
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467), by Zhang and Sennrich, defines RMSNorm.
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202), by Shazeer, defines SwiGLU and the other gated variants.
- [Training Compute-Optimal Large Language Models](/chinchilla/) sets the token counts that LLaMA goes past.
- [RoFormer](/rope/) derives the rotary embedding in full.
- [Language Models are Few-Shot Learners](/gpt3/) is the decoder-only baseline whose size LLaMA-13B matches on benchmarks.
