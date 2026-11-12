---
id: "switch"
title: "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"
authors: "Fedus et al."
year: 2021
venue: "JMLR 2022"
arxiv: "2101.03961"
url: "https://arxiv.org/abs/2101.03961"
track: scaling-efficiency
tier: extended
est_hours: 4
status: full
summary: "The Switch Transformer replaces the Transformer's feed-forward layer with many expert copies and sends each token to exactly one of them, so parameter count grows with the number of experts while the compute per token stays almost constant. A small auxiliary loss keeps the router from sending all tokens to a few experts, and a capacity factor caps how many tokens each expert accepts, with overflow tokens skipping the layer through the residual connection. The paper reports pre-training speedups of up to 7 times over matched dense T5 models and scales the design to a model with 1.6 trillion parameters."
related_course: "ml: mixture of experts"
prereqs:
  - id: transformer
    reason: "A Switch layer replaces the Transformer's position-wise feed-forward sub-layer, and the attention, residual connections, and layer normalization around it stay as they were."
background: [top-k-routing, load-balancing]
course_links:
  - title: "Mixtures of experts: routing, the router's gradient, and balance"
    url: "https://ml.ifkash.dev/lessons/082-mixtures-of-experts"
  - title: "Mixture-of-experts and sparse kernels"
    url: "https://gpu.ifkash.dev/c/20-moe-and-sparse-kernels"
self_check:
  - q: 'Which sub-layer of a Transformer block does a Switch layer replace, and what happens to a token that the layer drops because its expert is full?'
    a: 'It replaces the position-wise feed-forward network. The attention sub-layer stays. A dropped token skips the layer, and the residual connection carries its representation, unchanged, to the next layer, so the token is not zeroed.'
    see: transformer
  - q: 'A router gives one token the probabilities $[0.6, 0.3, 0.1]$. What does top-2 routing output, what does top-1 routing output, and why does the router still receive a gradient under top-1 routing?'
    a: 'Top-2 outputs $0.6\,E_1(x) + 0.3\,E_2(x)$, and top-1 outputs $0.6\,E_1(x)$. The argmax has no gradient, but the output is multiplied by the gate value $p_1(x)$, which is a differentiable function of the router logits.'
    see: top-k-routing
  - q: 'With $N = 4$ experts and $\alpha = 0.01$, what does the auxiliary loss equal when every token goes to expert 0 with probability 1, and what does it equal under perfectly uniform routing?'
    a: 'In the first case $f = P = [1, 0, 0, 0]$, so $\sum_i f_i P_i = 1$ and the loss is $\alpha N = 0.04$. Under uniform routing $\sum_i f_i P_i = 1/N$, so the loss is $\alpha = 0.01$.'
    see: load-balancing
---

## The problem

A dense Transformer, as in [Attention is all you need](/transformer/), uses every parameter for every token. Adding parameters adds floating-point operations (FLOPs) in the same proportion. Scaling a model from 223 million to 11 billion parameters (the T5-Base and T5-XXL rows of Table 9) multiplies the compute per token as well.

The paper tests a different axis (Section 2). Keep the FLOPs per token fixed and grow the parameter count only. A mixture-of-experts (MoE) layer does this. It holds $N$ copies of the feed-forward network, called experts, and a router that picks which experts see each token. Shazeer et al. (2017) built such a layer with top-$k$ routing, and GShard (Lepikhin et al., 2020) used $k = 2$. Three problems limited its adoption (Section 1): complexity, communication cost, and training instability.

Consider what goes wrong in a naive MoE layer with 8 experts and a router that starts out slightly uneven.

- **Compute grows with $k$.** With top-2 routing, every token runs through two experts. The expert FLOPs per token double. `scripts/params_flops.py` shows that at $N = 128$, a top-2 layer costs 2.02 times a dense layer of the same width, while a top-1 layer costs 1.02 times.
- **The router can collapse.** An expert that gets more tokens trains faster, so the router favors it more. Most tokens end up at one or two experts, and the others never learn.
- **Hardware needs fixed shapes.** On TPUs, every tensor has a size set at compile time. An expert that receives 40% of a batch can't simply use a bigger buffer.

The Switch Transformer answers each one: route to a single expert, add an auxiliary loss for balance, and give each expert a fixed capacity with a rule for overflow. Table 1 compares it with a top-2 MoE that has the same number of experts, at a capacity factor of 1.0. The Switch layer reaches a negative log perplexity of $-1.561$ at 1000 examples per second, against $-1.572$ at 860 examples per second for the MoE.

## Key ideas

This page uses the paper's notation: $N$ experts $E_1, \dots, E_N$, a batch $\mathcal{B}$ of $T$ tokens, a router matrix $W_r$, logits $h(x) = W_r \cdot x$, and gate probabilities $p_i(x)$. The paper treats $x$ as a column vector. The code and exercises store tokens as rows, so the logits are `x @ Wr` with `Wr` of shape `(d_model, N)`. The numbers below come from the scripts in `scripts/`. Run any of them from the paper folder.

### Top-1 routing

**Intuition.** A standard feed-forward layer applies one network to every token. A Switch layer owns $N$ such networks and applies exactly one to each token, chosen by a small linear router. Which network a token gets depends on the token itself, so the model holds $N$ times the parameters but runs one network per token.

**Math.** The router produces logits $h(x) = W_r \cdot x$ and normalizes them with a softmax over the experts (Section 2.1, Equation 1):

$$p_i(x) = \frac{e^{h(x)_i}}{\sum_{j=1}^{N} e^{h(x)_j}}.$$

Shazeer et al. (2017) select a set $\mathcal{T}$ of the top-$k$ indices and mix their outputs (Section 2.1, Equation 2):

$$y = \sum_{i \in \mathcal{T}} p_i(x)\, E_i(x).$$

They conjectured that $k > 1$ was needed so that the router gets a useful gradient. The Switch layer uses $k = 1$, so $\mathcal{T}$ holds one index $i = \arg\max_j p_j(x)$ and

$$y = p_i(x)\, E_i(x).$$

The paper lists three benefits (Section 2.1): the router computation shrinks, each expert's batch (its capacity) can be at least halved, and the implementation and communication get simpler. The router is lightweight. It costs $O(d_{\text{model}} \times N)$ per token (Section 3).

**Numeric example.** Six tokens and three experts, with logits written down in `scripts/router_example.py`. Token 0 has $h = [2, 0, 0]$. The softmax gives

$$p(x_0) = \left[\frac{e^2}{e^2 + 2},\ \frac{1}{e^2 + 2},\ \frac{1}{e^2 + 2}\right] = [0.7870,\ 0.1065,\ 0.1065].$$

The script prints the chosen experts $[0, 0, 0, 0, 1, 2]$ for the six tokens and the gate values $[0.787, 0.6285, 0.5065, 0.6652, 0.787, 0.787]$. Token 0 outputs $0.787 \cdot E_0(x_0)$. Notice that the gate is not 1. The output is scaled by the probability, which is how the router receives a gradient.

Parameters and compute follow different curves. `scripts/params_flops.py` uses the T5-Base shape from Table 9 ($d_{\text{model}} = 768$, $d_{ff} = 2048$, a gated feed-forward layer with three weight matrices). One feed-forward layer has 4,718,592 parameters. With $N = 128$ experts the layer holds 604,078,080 parameters, 128.02 times as many, yet the FLOPs per token grow only 1.0208 times, and the router accounts for 2.04% of the layer's FLOPs. Table 9 shows the same effect for the full model: Switch-Base has 7B parameters against 0.2B for T5-Base, with the same 124B FLOPs per sequence.

### Load-balancing loss

**Intuition.** Left alone, the router drifts toward a few experts. The auxiliary loss penalizes a router that sends many tokens to experts that already get a large share of the probability mass.

**Math.** For a batch $\mathcal{B}$ with $T$ tokens, define the fraction of tokens dispatched to expert $i$ (Section 2.2, Equation 5) and the mean router probability for expert $i$ (Equation 6):

$$f_i = \frac{1}{T} \sum_{x \in \mathcal{B}} \mathbb{1}\{\arg\max p(x) = i\}, \qquad P_i = \frac{1}{T} \sum_{x \in \mathcal{B}} p_i(x).$$

The loss is the scaled dot product of the two vectors (Equation 4):

$$\text{loss} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i.$$

The loss is added to the training objective for every Switch layer. The vector $f$ comes from an argmax, so it has no gradient. The vector $P$ is differentiable, and all the gradient from this loss reaches the router through $P$. Under uniform routing $f_i = P_i = 1/N$, so $\sum_i f_i P_i = 1/N$ and the factor $N$ keeps the loss at $\alpha$ for any number of experts. The paper uses $\alpha = 10^{-2}$ after a sweep from $10^{-1}$ to $10^{-5}$ in powers of 10 (Section 2.2). The pseudo code in Appendix F (Figure 14) computes the same quantity as a mean over experts times $N^2$.

**Numeric example.** For the six tokens above, `scripts/router_example.py` prints

$$f = [0.6667,\ 0.1667,\ 0.1667], \qquad P = [0.4667,\ 0.2770,\ 0.2562].$$

Then $\sum_i f_i P_i = 0.4$, and the loss is $0.01 \cdot 3 \cdot 0.4 = 0.012001$ (the script prints the unrounded value). Uniform routing gives $0.01$, and sending everything to one expert gives $\alpha N = 0.03$. The loss sits between the two. Expert 0 gets 66.7% of the tokens but only 46.7% of the probability mass. This gap shows why $f$ and $P$ are different vectors, and why the loss needs both.

### Capacity factor and token dropping

**Intuition.** A static-shape accelerator allocates a buffer of fixed size for each expert. If a router sends more tokens than the buffer holds, the extra tokens are dropped. A capacity factor above 1 makes the buffer bigger than a perfectly even split, at the price of padding.

**Math.** The expert capacity is the number of tokens per expert (Section 2.2, Equation 3):

$$\text{expert capacity} = \left(\frac{\text{tokens per batch}}{\text{number of experts}}\right) \times \text{capacity factor}.$$

When too many tokens pick one expert, the layer skips the excess ones, and "the token representation is passed directly to the next layer through the residual connection" (Section 2.2). The paper's pseudo code assigns slots in token order with a cumulative sum (Appendix F, Figure 15), so earlier tokens win. The paper doesn't say how to round a non-integer capacity. This page and the exercises round up.

A larger capacity factor wastes compute and memory on padding, and a smaller one drops more tokens (Figure 3). With the auxiliary loss on, the paper reports that fewer than 1% of tokens are typically dropped (Section 2.2). Table 1 shows that Switch-Base is better at low capacity factors of 1.0 and 1.25.

**Numeric example.** With $T = 6$ and $N = 3$, a capacity factor of 1.0 gives a capacity of 2. Four tokens chose expert 0, so tokens 2 and 3 overflow, and a third of the batch is dropped. At a capacity factor of 1.5 the capacity is 3, and only token 3 is dropped (`scripts/router_example.py`).

Even a perfectly balanced router overflows sometimes, because random batches are never exactly even. `scripts/capacity_drops.py` routes 1024 tokens to 8 experts at random, averaged over 200 batches:

| Capacity factor | Capacity | Dropped, uniform router | Dropped, router with 30% on expert 0 |
|---|---|---|---|
| 1.0 | 128 | 3.35% | 17.53% |
| 1.25 | 160 | 0.01% | 14.32% |
| 1.5 | 192 | 0.00% | 11.21% |
| 2.0 | 256 | 0.00% | 5.03% |

A uniform router needs little headroom, and a skewed one needs a lot. That is why the balancing loss and the capacity factor work together.

### The router learns through the gate

**Intuition.** Choosing an expert is a discrete step with no gradient. The router still learns because the chosen expert's output is multiplied by the gate $p_i(x)$. A loss that wants a bigger output from that expert pushes the gate up, and one that wants a smaller output pushes it down.

**Math.** For one token routed to expert $i$, let $g = p_i(x)$ and $o = E_i(x)$, so $y = g\,o$. Given the loss gradient $\partial L / \partial y$, the gradient to the gate is $\partial L / \partial g = \sum_k (\partial L / \partial y_k)\, o_k$. Only the chosen entry of $p$ receives it. It reaches the logits through the softmax Jacobian, $\partial p_i / \partial h_j = p_i(\delta_{ij} - p_j)$, and then reaches $W_r$ through $h = W_r \cdot x$. The paper notes that "the gate value $p_i(x)$ in Equation 2 permits differentiability of the router" for both MoE and Switch routing (Section 2.1). Exercise 4 has you write this backward pass and check it against finite differences.

**Numeric example.** `scripts/toy_moe.py` builds a toy mixture task: four clusters of 2-D inputs, each needing its own linear map. One linear map fitted by least squares leaves a mean squared error of 1.9065 on targets with variance 1.9378, so it explains almost nothing. A four-expert Switch layer trained with plain gradient descent for 2000 steps reaches 0.0133. Each cluster ends up at its own expert, with 106, 103, 98, and 93 tokens per expert. This is the same specialization effect the paper describes at scale, shrunk to a size you can run in a second.

<!-- exercises -->

## Pitfalls

- **Treating $f$ as differentiable.** The loss has a gradient only through $P$. If you compute $f$ from the soft probabilities, you change the loss, and the page's example drops from $0.012001$ to a different value.
- **Forgetting the gate multiplication.** A top-1 router without $p_i(x)$ on the output gets no gradient at all, so it never learns. The Switch output is $p_i(x)\,E_i(x)$, not $E_i(x)$.
- **Zeroing dropped tokens.** A dropped token still reaches the next layer through the residual connection. The layer contributes nothing, and the input goes through unchanged (Section 2.2).
- **Per-batch versus per-core quantities.** The paper's pseudo code computes the balancing statistics and the capacity separately for each core's tokens (Appendix F, Figures 14 and 15), so "the batch" in Equations 3 to 6 is the group of tokens on one core. The exercises treat the whole input as one group.
- **Assuming experts sit in every layer.** Table 1 and Table 9 show experts at every other feed-forward layer for Switch-Base and Switch-Large. Switch-C, the 1.6-trillion-parameter model, uses experts in every layer.
- **Reading total parameters as active parameters.** All $N$ experts must sit in memory, even though a token uses one. The paper spreads the experts across devices (Section 2) and accepts extra communication, so a speedup on a step basis doesn't always carry over to wall-clock time (Section 3.2).
- **Training in bfloat16 without care.** The paper reports that a Switch-Base model trained fully in bfloat16 diverged, at $-3.780$, while casting only the router to float32 gave $-1.716$ at nearly the same speed (Section 2.4, Table 2). Scaling the default initialization down by 10 also cut the standard deviation of quality across seeds from 0.68 to 0.01 (Table 3).
- **Expecting rerouting to help.** The paper tried No-Token-Left-Behind, which sends an overflowing token to its next-best expert. It found no empirical benefit (Appendix B). It also used multiplicative input jitter in the router for exploration (Appendix C).

## Further reading

- The paper's official code is in the T5X repository at [github.com/google-research/t5x](https://github.com/google-research/t5x). Appendix F of the paper has the Mesh TensorFlow pseudo code for the router and the loss.
- Shazeer et al. (2017), [Outrageously large neural networks: the sparsely-gated mixture-of-experts layer](https://arxiv.org/abs/1701.06538), is the top-$k$ layer that this paper simplifies.
- Lepikhin et al. (2020), [GShard](https://arxiv.org/abs/2006.16668), introduces top-2 routing with expert capacity at scale.
