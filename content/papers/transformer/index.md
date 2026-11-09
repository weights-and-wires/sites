---
id: "transformer"
title: "Attention Is All You Need"
authors: "Vaswani et al."
year: 2017
venue: "NeurIPS 2017"
arxiv: "1706.03762"
url: "https://arxiv.org/abs/1706.03762"
track: architectures
tier: core
est_hours: 6
status: full
summary: "The Transformer replaces recurrence with self-attention: every position reads every other position through one matrix multiplication, so training parallelizes across the sequence. Attention is a softmax-weighted average of value vectors, with weights from scaled query-key dot products, and several heads run in parallel. Residual connections, layer normalization, and a position-wise feed-forward network wrap each attention layer, and sinusoidal positional encodings supply word order."
related_course: "ml: transformers"
prereqs:
  - id: bahdanau
    reason: "The Transformer keeps the idea of attending over a source sequence and replaces the additive alignment score with a scaled dot product."
  - id: layernorm
    reason: "Every sub-layer ends in layer normalization, which normalizes each position across its features and not across the batch."
  - id: resnet
    reason: "Every sub-layer sits inside a residual connection, which lets a 6-layer stack train and keeps a direct path for gradients."
background: [matrix-multiplication, softmax-numerics, residual-connections, positional-information]
course_links:
  - title: "Scaled attention, many heads, and the causal mask"
    url: "https://ml.ifkash.dev/lessons/047-scaled-attention-and-heads"
  - title: "The transformer block: SwiGLU, residuals, and layer norm"
    url: "https://ml.ifkash.dev/lessons/050-the-transformer-block"
  - title: "Position encodings: sinusoidal clocks and rotary embeddings"
    url: "https://ml.ifkash.dev/lessons/049-position-encodings"
self_check:
  - q: 'Compute $\operatorname{softmax}([2, 0])$ by hand. Why does an implementation subtract the row maximum before it calls `exp`?'
    a: 'The result is $[0.8808, 0.1192]$, because $e^2 / (e^2 + 1) \approx 0.8808$. Subtracting the maximum changes no output, because the constant cancels in the ratio, and it keeps `exp` from overflowing on scores such as 1000.'
    see: softmax-numerics
  - q: 'In `LayerNorm(x + Sublayer(x))`, over which axis does LayerNorm compute its mean and variance for a tensor of shape `(batch, seq, d_model)`?'
    a: 'Over the last axis, `d_model`, separately for every position of every example. Batch normalization would average over the batch axis instead, which makes the output depend on the other examples in the batch.'
    see: layernorm
  - q: 'In Bahdanau attention, a decoder state scores each encoder state with a small MLP. What does the Transformer use in place of the MLP, and what stays the same?'
    a: 'The score becomes a dot product $q \cdot k$, divided by $\sqrt{d_k}$. What stays the same: the scores go through a softmax, and the output is the weighted sum of value vectors.'
    see: bahdanau
---

## The problem

Before 2017, the strongest translation models were recurrent encoder-decoders with attention (see [Sequence to sequence learning](/seq2seq/) and [Bahdanau attention](/bahdanau/)). A recurrent network reads a sentence one token at a time. Step $t$ needs the hidden state from step $t-1$.

That design has two costs. Consider the sentence "The animal didn't cross the street because it was too tired," padded to 50 tokens.

- **Training can't parallelize across positions.** All 50 steps run in order, so a GPU sits mostly idle on each step. Longer sequences make this worse, because memory limits the batch size.
- **Distant tokens connect through a long path.** For "it" to read "animal", the signal passes through every hidden state in between. Table 1 of the paper lists the maximum path length between two positions as $O(n)$ for a recurrent layer and $O(1)$ for self-attention.

The paper asks whether attention alone, with no recurrence and no convolution, can do the sequence modeling. Section 6.1 reports that it can. The big model reaches 28.4 BLEU on WMT 2014 English-to-German after 3.5 days on 8 P100 GPUs. Table 2 lists $3.3 \times 10^{18}$ training FLOPs for the base model, against $9.6 \times 10^{18}$ for ConvS2S and $2.3 \times 10^{19}$ for GNMT + RL.

## Key ideas

The sections that follow use the paper's notation: queries $Q$, keys $K$, values $V$, key width $d_k$, value width $d_v$, model width $d_{\text{model}}$, and $h$ heads. Every number comes from a script in `scripts/`. Run any script from the paper folder to reproduce the output.

### Scaled dot-product attention

**Intuition.** Each position asks a question (the query). Every position advertises what it holds (the key) and carries content (the value). The answer to a query is a blend of the values, weighted by how well each key matches the query.

**Math.** The paper packs the queries into a matrix $Q \in \mathbb{R}^{n \times d_k}$, the keys into $K \in \mathbb{R}^{m \times d_k}$, and the values into $V \in \mathbb{R}^{m \times d_v}$ (Section 3.2.1, Equation 1):

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V
$$

The softmax runs over the keys, which is the last axis of the $n \times m$ score matrix. Each row of weights sums to 1.

**Tiny example, computed by `scripts/attention_example.py`.** Take $d_k = 2$, with

$$
Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad
K = \begin{bmatrix} 1 & 0 \\ 1 & 1 \end{bmatrix}, \quad
V = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}.
$$

The steps and results are as follows:

| Step | Result |
|---|---|
| $QK^\top$ | `[[1, 1], [0, 1]]` |
| Divide by $\sqrt{2} \approx 1.4142$ | `[[0.7071, 0.7071], [0, 0.7071]]` |
| Softmax of each row | `[[0.5, 0.5], [0.3302, 0.6698]]` |
| Multiply by $V$ | `[[2.0, 3.0], [2.3395, 3.3395]]` |

Check row 2 by hand: $0.3302 \cdot [1, 2] + 0.6698 \cdot [3, 4] = [2.3395, 3.3395]$. Row 1 has two equal scores, so its output is the plain average of the two value rows.

Exercise 1 asks you to implement this function with a stable softmax and an optional mask.

### Why divide by sqrt(d_k)

**Intuition.** A dot product of two long vectors tends to be large, and a softmax with large inputs is nearly one-hot. A nearly one-hot softmax has almost no gradient, so the model stops learning which keys to attend to.

**Math.** Footnote 4 in Section 3.2.1 makes the argument. Assume the components of $q$ and $k$ are independent random variables with mean 0 and variance 1. Then $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ has mean 0 and variance $d_k$. Dividing by $\sqrt{d_k}$ returns the variance to 1.

The softmax gradient explains why the size matters. For two scores with gap $s$, the larger weight is $p = \sigma(s)$, and its derivative with respect to the gap is $p(1-p)$. That derivative approaches 0 as $p$ approaches 1.

**Tiny example, computed by `scripts/scaling_variance.py`.** With $d_k = 64$ and 100,000 random pairs, the sample variance of $q \cdot k$ is 63.98, close to the theoretical 64. After scaling, it is 1.00. Compare two score gaps:

| Score gap | Softmax | Gradient $p(1-p)$ |
|---|---|---|
| 8 (unscaled; the standard deviation of $q \cdot k$ is $\sqrt{64} = 8$) | `[0.99966, 0.00034]` | 0.00034 |
| 1 (scaled) | `[0.7311, 0.2689]` | 0.19661 |

The script also shows a seeded row of 10 scores with $d_k = 64$. Without scaling, the largest weight is 0.9858. With scaling, it is 0.2839.

### Multi-head attention

**Intuition.** One attention pattern per layer is limiting, because a weighted average blurs different relations together. Several smaller heads can each look for a different relation at the same time, in a different learned subspace.

**Math.** Section 3.2.2 projects the queries, keys, and values $h$ times with learned matrices, runs attention on each projection, concatenates the results, and projects again:

$$
\operatorname{MultiHead}(Q, K, V) = \operatorname{Concat}(\operatorname{head}_1, \dots, \operatorname{head}_h)\,W^O
$$

$$
\operatorname{head}_i = \operatorname{Attention}(QW_i^Q,\; KW_i^K,\; VW_i^V)
$$

The projections have shapes $W_i^Q, W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$, and $W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$. The base model uses $h = 8$ and $d_k = d_v = d_{\text{model}}/h = 64$.

In self-attention, $Q$, $K$, and $V$ all come from the same input $X \in \mathbb{R}^{n \times d_{\text{model}}}$, so the head is $\operatorname{Attention}(XW_i^Q, XW_i^K, XW_i^V)$. Implementations stack the $h$ matrices $W_i^Q$ side by side into one $d_{\text{model}} \times d_{\text{model}}$ matrix. One matrix product then computes all heads, and a reshape splits the result into heads. The paper doesn't spell out this stacking. It is the standard way to implement the formulas.

**Tiny example, computed by `scripts/multihead_example.py`.** Parameter count first. With $d_{\text{model}} = 512$ and $h = 8$, one head needs $3 \cdot 512 \cdot 64 = 98{,}304$ projection weights. Eight heads need 786,432, and $W^O$ adds 262,144. The total is 1,048,576, which equals $4 \cdot 512^2$, the cost of one full-width head. The paper makes the same point: the reduced width per head keeps the total cost similar to single-head attention (Section 3.2.2). The formulas have no biases, so the count excludes them.

Now the reshape. Let $n = 3$, $d_{\text{model}} = 4$, $h = 2$, and let $X$ hold the numbers 0 to 11 row by row:

- Row 0 is `[0, 1, 2, 3]`, row 1 is `[4, 5, 6, 7]`, and row 2 is `[8, 9, 10, 11]`.
- `X.reshape(3, 2, 2).transpose(1, 0, 2)` has shape `(2, 3, 2)`.
- Head 0 sees columns 0 and 1 of every row: `[[0, 1], [4, 5], [8, 9]]`.
- Head 1 sees columns 2 and 3: `[[2, 3], [6, 7], [10, 11]]`.
- `X.reshape(2, 3, 2)` without the transpose gives head 0 the rows `[[0, 1], [2, 3], [4, 5]]`, which mixes tokens.

Exercise 4 asks you to implement multi-head attention and test it for causality and permutation equivariance.

### Positional encoding

**Intuition.** Attention treats its input as a set. If you shuffle the tokens, the outputs shuffle the same way, and no output records where a token was. The model needs position information in its input.

**Math.** Section 3.5 adds a fixed vector to each input embedding. It uses sines and cosines of different frequencies. The paper doesn't number these two formulas:

$$
PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \qquad
PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

Even dimensions hold sines and odd dimensions hold cosines. The wavelengths form a geometric progression from $2\pi$ to $10000 \cdot 2\pi$. The paper chose this form because, for any fixed offset $k$, $PE_{pos+k}$ is a linear function of $PE_{pos}$. Each (sine, cosine) pair rotates by the angle $k / 10000^{2i/d_{\text{model}}}$ when the position moves by $k$.

**Tiny example, computed by `scripts/positional_encoding_example.py`.** With $d_{\text{model}} = 4$:

| Position | $PE$ |
|---|---|
| 0 | `[0, 1, 0, 1]` |
| 1 | `[0.8415, 0.5403, 0.0100, 1.0000]` |
| 2 | `[0.9093, -0.4161, 0.0200, 0.9998]` |

Position 1 is $[\sin 1, \cos 1, \sin 0.01, \cos 0.01]$. The first pair has wavelength $2\pi \approx 6.28$, and the second has wavelength $628.3$. For the rotation claim, rotate $[\sin 1, \cos 1]$ by $k = 1$ radian. The result is $[0.9093, -0.4161]$, which is $[\sin 2, \cos 2]$, the first pair of position 2.

The paper also tried learned positional embeddings and found nearly identical results (Section 3.5, Table 3 row E). Exercise 5 uses learned positions. Exercise 3 asks you to implement the sinusoidal version.

### Masking

**Intuition.** During training, the decoder sees the whole target sentence at once. Position $i$ must not read positions after $i$, or it could copy the answer. A mask hides the future.

**Math.** Section 3.2.3 implements the mask inside scaled dot-product attention. It sets every score for an illegal connection to $-\infty$ before the softmax, so the weight is exactly 0. For an $n \times n$ score matrix $S$, the masked scores are

$$
\tilde{S}_{ij} =
\begin{cases}
S_{ij} & j \le i \\
-\infty & j > i
\end{cases}
$$

Code uses a large negative number such as $-10^9$ in place of $-\infty$. The weight is then 0 to within floating-point precision, and a row that is fully masked yields a uniform distribution instead of `NaN`. The decoder's output embeddings are also offset by one position, so the prediction for position $i$ depends only on known outputs at positions less than $i$ (Section 3.1).

**Tiny example, computed by `scripts/attention_example.py`.** Apply the lower-triangular mask to the earlier example. The masked scores are `[[0.7071, -1e9], [0, 0.7071]]`, and the weights are `[[1, 0], [0.3302, 0.6698]]`. Row 1 attends only to itself, so its output is $V$'s first row, `[1, 2]`. Row 2 is unchanged, `[2.3395, 3.3395]`, because it already saw both positions.

Now the common bug. If you mask with 0 instead of $-10^9$, a masked score of 0 still counts. The weights become `[[0.6698, 0.3302], [0.3302, 0.6698]]`, and position 1 leaks 33 percent of its attention to the future.

Exercise 2 asks you to build the mask.

### The block: residuals, layer normalization, and feed-forward

**Intuition.** Attention mixes information across positions. The feed-forward network then transforms each position on its own. A residual connection around each step gives gradients a direct path, and layer normalization keeps the scale of each position stable.

**Math.** The encoder is a stack of $N = 6$ identical layers, and each layer has two sub-layers: multi-head self-attention and a position-wise feed-forward network. The output of each sub-layer is (Section 3.1)

$$
\operatorname{LayerNorm}(x + \operatorname{Sublayer}(x))
$$

The feed-forward network applies two linear maps with a ReLU between them, separately at each position (Section 3.3, Equation 2):

$$
\operatorname{FFN}(x) = \max(0,\, xW_1 + b_1)W_2 + b_2
$$

The base model has $d_{\text{model}} = 512$ and inner width $d_{ff} = 2048$. The decoder adds a third sub-layer that attends over the encoder output, and its self-attention uses the causal mask (Section 3.2.3). The paper applies dropout with $P_{drop} = 0.1$ to each sub-layer's output before the residual addition, and also to the sums of embeddings and positional encodings. It trains with label smoothing $\epsilon_{ls} = 0.1$ (Section 5.4).

The paper's order, normalize after the residual addition, is called *post-LN*. Most later models use *pre-LN*, $x + \operatorname{Sublayer}(\operatorname{LayerNorm}(x))$. Xiong et al. (see Further reading) show that post-LN needs a warmup phase to train stably, and that pre-LN trains without one.

**Tiny example, computed by `scripts/block_example.py`.**

- **LayerNorm.** For $x = [1, 2, 3, 4]$, the mean is 2.5 and the variance is 1.25. Each entry becomes $(x_i - 2.5)/\sqrt{1.25}$, which gives `[-1.3416, -0.4472, 0.4472, 1.3416]`. The input $[10, 20, 30, 40]$ gives the same output, because the scale divides out. The script sets the learned gain to 1, the bias to 0, and $\epsilon$ to 0, to keep the arithmetic exact.
- **FFN.** Take $d_{\text{model}} = 2$ and $d_{ff} = 3$, with $W_1 = \begin{bmatrix} 1 & -1 & 0 \\ 0 & 1 & 1 \end{bmatrix}$, $b_1 = [0, 0, -1]$, $W_2 = \begin{bmatrix} 1 & 0 \\ 2 & 1 \\ 0 & 3 \end{bmatrix}$, and $b_2 = [0.5, 0]$. For $x = [1, 2]$, the hidden vector is `[1, 1, 1]`, ReLU leaves it unchanged, and the output is `[3.5, 4.0]`. Check the first entry by hand: $1 \cdot 1 + 1 \cdot 2 + 0 \cdot 1 + 0.5 = 3.5$.
- **FFN size.** The base model's two matrices hold $2 \cdot 512 \cdot 2048 = 2{,}097{,}152$ weights. With biases, the count is 2,099,712.
- **Post-LN versus pre-LN.** Let the residual stream be $x = [100, 0, 0, 0]$, and use $\operatorname{Sublayer}(v) = 2v$ as a stand-in. Post-LN returns `[1.7321, -0.5774, -0.5774, -0.5774]`, so the residual stream is rescaled at every layer. Pre-LN returns `[103.46, -1.1547, -1.1547, -1.1547]`, so the original 100 still travels along the identity path.

Exercise 5 builds one pre-LN block and trains it.

### The learning-rate schedule

**Intuition.** Early in training, the weights are random and the gradients are unreliable, so a large step can do damage. The schedule starts with tiny steps, ramps up linearly, and then decays.

**Math.** Section 5.3 trains with Adam ($\beta_1 = 0.9$, $\beta_2 = 0.98$, $\epsilon = 10^{-9}$) and varies the learning rate by (Equation 3)

$$
lrate = d_{\text{model}}^{-0.5} \cdot \min\!\left(step^{-0.5},\; step \cdot warmup^{-1.5}\right), \qquad warmup = 4000
$$

The rate grows linearly for the first $warmup$ steps and then falls with the inverse square root of the step number. The two terms in the minimum are equal at $step = warmup$.

**Tiny example, computed by `scripts/lr_schedule.py`.** With $d_{\text{model}} = 512$:

| Step | Learning rate |
|---|---|
| 1 | $1.747 \times 10^{-7}$ |
| 1,000 | $1.747 \times 10^{-4}$ |
| 4,000 (peak) | $6.988 \times 10^{-4}$ |
| 8,000 | $4.941 \times 10^{-4}$ |
| 100,000 | $1.398 \times 10^{-4}$ |

Check the peak by hand: $512^{-0.5} = 0.04419$ and $4000^{-0.5} = 0.01581$, and their product is $6.988 \times 10^{-4}$.

<!-- exercises -->

## Pitfalls

- **Softmax over the wrong axis.** The weights must sum to 1 over the keys, which is the last axis of the score matrix. A softmax over queries still returns numbers that look plausible.
- **Scaling by the wrong width.** Divide by $\sqrt{d_k}$ for the per-head key width. Dividing by $\sqrt{d_{\text{model}}}$ in a multi-head layer, or by $\sqrt{d_v}$, changes the weights.
- **Masking with 0.** A zero score still gets weight. Mask with a large negative number, and test with scores that are negative enough to expose the leak.
- **Using `-inf` on a fully masked row.** The softmax of a row of $-\infty$ is `NaN`. A large finite number gives a uniform row instead.
- **Reshaping without a transpose.** `(n, d_model)` must become `(h, n, d_k)` through `reshape(n, h, d_k)` and then a transpose. `reshape(h, n, d_k)` runs without an error and silently mixes tokens across heads.
- **Interleaving the positional encoding wrongly.** The paper puts sines in the even dimensions and cosines in the odd ones. Concatenating a block of sines and a block of cosines is a different encoding. The model can still learn with it, but you can't compare your values with the paper's.
- **Mixing up post-LN and pre-LN.** The paper is post-LN. Most code you read is pre-LN. A post-LN stack at depth needs the warmup in Equation 3. Copying the paper's learning rate onto a pre-LN model, or the reverse, is a common source of unexplained divergence.
- **Forgetting the embedding details.** Section 3.4 multiplies the embedding weights by $\sqrt{d_{\text{model}}}$ and shares one weight matrix between the input embeddings and the pre-softmax linear layer. The block diagrams don't show either step.
- **Expecting attention to know word order.** The Exercise 4 layer has no positional information, so shuffling the input rows shuffles the output rows the same way. That property is the reason Section 3.5 exists.
- **Treating the validation curve in Exercise 5 as smooth.** With weight decay 1.0 and only 28 validation pairs, accuracy can jump for a few hundred steps and recover. The exercise records the best value for that reason.

## Further reading

- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/), by Alexander Rush and collaborators, is a line-by-line PyTorch implementation of the paper.
- [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY), by Andrej Karpathy, is a video that builds a decoder-only Transformer.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/), by Jay Alammar, is a visual walkthrough.
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745), by Xiong et al., explains why pre-LN trains without warmup and post-LN doesn't.
- [tensor2tensor](https://github.com/tensorflow/tensor2tensor) is the official code release from the authors' team.
- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html), by Elhage et al., describes the residual-stream view of the block.
