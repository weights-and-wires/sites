---
id: "mixtral"
title: "Mixtral of Experts"
authors: "Jiang et al."
year: 2024
venue: "arXiv preprint"
arxiv: "2401.04088"
url: "https://arxiv.org/abs/2401.04088"
track: scaling-efficiency
tier: core
est_hours: 3
status: full
summary: "Mixtral 8x7B is a decoder-only model with the Mistral 7B architecture, except that every feed-forward sub-layer is replaced by 8 SwiGLU experts and a router that sends each token to 2 of them. Each token therefore touches only about 13B of the model's 47B parameters, so the compute per token is close to a 13B dense model while the quality matches or beats Llama 2 70B on most benchmarks. A routing analysis finds that experts follow syntax more than topic, and that consecutive tokens often reuse the same experts."
related_course: "inference: MoE serving"
prereqs:
  - id: switch
    reason: "Mixtral's router is the Switch router with two changes, top-2 selection in place of top-1 and a softmax over only the selected logits, so you need the router matrix, the gate that carries the gradient, and the idea of parameters that grow faster than compute."
  - id: llama
    reason: "Mixtral is the Llama-style decoder, with RMSNorm before each sub-layer, rotary positions, and SwiGLU, and each expert is exactly Llama's SwiGLU feed-forward network."
background: [top-2-routing, active-vs-total-parameters]
course_links:
  - title: "Mixtures of experts: routing, the router's gradient, and balance"
    url: "https://ml.ifkash.dev/lessons/082-mixtures-of-experts"
  - title: "Mixture-of-experts and sparse kernels"
    url: "https://gpu.ifkash.dev/c/20-moe-and-sparse-kernels"
self_check:
  - q: 'A router gives one token the logits $[2.0, 0.5, 1.5, 0.0]$. Which two experts does Mixtral select, what are their gate weights, and how do those compare with the full softmax values of the same two experts?'
    a: 'Experts 0 and 2 are selected, with logits 2.0 and 1.5. The softmax over just those two logits gives $[0.6225, 0.3775]$, which sums to 1. The full softmax over all four logits gives these experts about $0.509$ and $0.309$, which sum to less than 1, because the unselected experts keep part of the mass.'
    see: top-2-routing
  - q: 'Suppose a Mixtral-style layer used $K = 1$ with the gate $\operatorname{Softmax}(\operatorname{TopK}(x W_g))$. What would the gate value be, and why would the router stop learning? What changes with $K = 2$?'
    a: 'The softmax runs over one logit, so the gate is exactly 1 for every input, and the logits get no gradient through it. With $K = 2$ the gate of the first expert is $\sigma(\ell_1 - \ell_2)$, which depends on the logit gap, so the output carries a gradient to both logits. Switch avoids the problem differently, by taking the softmax over all experts before it picks one.'
    see: switch
  - q: 'In Mixtral, which parameters are counted in the 47B but not in the 13B, and which are counted in both?'
    a: 'The 47B counts all 8 experts in every layer, and the 13B counts only 2. Attention, the routers, the norms, and the two embedding matrices are in both counts. The weights must still fit in memory for all 47B parameters, while the compute per token follows the 13B.'
    see: active-vs-total-parameters
---

## The problem

Before this paper, the way to a stronger model was a bigger dense model. [LLaMA](/llama/) showed that smaller models trained on more tokens are competitive with much larger ones, but every token still pays for every weight. The Llama 2 70B model needs about 138 GFLOPs of matrix multiplication per token, and `scripts/param_count.py` computes that figure. That cost sets the price of serving the model.

[Switch Transformers](/switch/) showed a different trade: add parameters without adding compute per token. Its experiments used encoder-decoder T5 models, and it routed each token to one expert and relied on an auxiliary loss, a capacity factor, and token dropping to keep the layer trainable and balanced. What was missing was a strong open model that used the idea in a standard decoder-only language model and was good enough to compare against the best dense models.

Mixtral 8x7B is that model (Section 1). It has the Mistral 7B architecture, with every feed-forward block replaced by 8 experts and a router that picks 2 of them per token. The results are in Table 2:

| Model | Active parameters | MMLU | HumanEval | GSM8K |
|---|---|---|---|---|
| Llama 2 70B | 70B | 69.9% | 29.3% | 69.6% |
| Mixtral 8x7B | 13B | 70.6% | 40.2% | 74.4% |

Mixtral matches or exceeds Llama 2 70B on most of the benchmarks the paper reports while using 5 times fewer active parameters (Section 3). The paper is careful about what this does and doesn't mean. Active parameters track the compute per token. They don't track memory, because all 47B parameters must be loaded (Section 3, "Size and Efficiency"). The page returns to that point in the second idea.

## Key ideas

The page uses the paper's notation: $n$ experts $E_0, \dots, E_{n-1}$, a gating network $G$ with weight matrix $W_g$, and $K$ experts per token (Section 2.1). The paper treats $x$ as a vector and writes $x \cdot W_g$. The code and exercises store tokens as rows, so the logits are `x @ Wg` with `Wg` of shape `(dim, n)`. The paper's equations in Section 2.1 aren't numbered, so the page cites them by section. Every number comes from a script in `scripts/`. Run any script from the paper folder to reproduce the output.

### Softmax over the top-K logits

**Intuition.** A mixture-of-experts layer is a weighted sum of expert outputs. If most weights are zero, you can skip those experts. The router must therefore produce a sparse weight vector that still sums to 1 and that still depends smoothly on the router's logits.

**Math.** The layer output for an input $x$ is (Section 2.1)

$$
y = \sum_{i=0}^{n-1} G(x)_i \cdot E_i(x).
$$

The paper takes the gate to be "the softmax over the Top-K logits of a linear layer":

$$
G(x) := \operatorname{Softmax}(\operatorname{TopK}(x \cdot W_g)),
$$

where $(\operatorname{TopK}(\ell))_i := \ell_i$ if $\ell_i$ is among the $K$ largest coordinates of $\ell \in \mathbb{R}^n$, and $-\infty$ otherwise. Because $e^{-\infty} = 0$, the softmax gives the unselected experts a weight of exactly 0, and the $K$ selected weights sum to 1. Mixtral sets $K = 2$ and $n = 8$ (Table 1).

The order of the two operations matters. Compare this gate with the Switch router of [Switch Transformers](/switch/), which takes the softmax over all experts first and then keeps the top entries, as in $p_i(x) = e^{h(x)_i} / \sum_j e^{h(x)_j}$ (Switch, Section 2.1, Equation 1). Taking the top two probabilities from that softmax gives weights that sum to less than 1. Applying the softmax to the selected logits only is the same as renormalizing those two probabilities:

$$
G(x)_i = \frac{p_i(x)}{p_a(x) + p_b(x)} \quad \text{for the selected experts } a, b.
$$

For $K = 2$ the weights also have a closed form. If the selected logits are $\ell_a \ge \ell_b$, then $G(x)_a = \sigma(\ell_a - \ell_b)$ and $G(x)_b = 1 - G(x)_a$, where $\sigma$ is the logistic sigmoid. Only the gap between the two selected logits matters.

**Tiny example, computed by `scripts/top2_example.py`.** One token has the 8 router logits $[2.0, 0.5, 1.0, -1.0, 0.0, 1.5, -0.5, 0.2]$. The two largest are experts 0 (logit 2.0) and 5 (logit 1.5). The numbers are:

| Quantity | Value |
|---|---|
| Full softmax, experts 0 and 5 | $0.3802$ and $0.2306$ (mass $0.6108$) |
| $\operatorname{Softmax}(\operatorname{Top2})$, experts 0 and 5 | $0.6225$ and $0.3775$ (sum 1) |
| $\sigma(2.0 - 1.5) = \sigma(0.5)$ | $0.6225$ |

The script checks that renormalizing the full-softmax values gives the same $0.6225$ and $0.3775$. Now give the experts the scalar outputs $E_i(x) = 10(i+1)$, so expert 0 returns 10 and expert 5 returns 60. The layer output is $0.6225 \cdot 10 + 0.3775 \cdot 60 = 28.877$. Without the renormalization, the output would be $0.3802 \cdot 10 + 0.2306 \cdot 60 = 17.6392$, which is too small, because the weights sum to only 0.6108.

Exercise 2 asks you to implement this gate for any $K$.

### Experts are SwiGLU blocks in every layer

**Intuition.** The layer replaces one feed-forward network with $n$ of them. Everything else in the block stays the same, so the model keeps the Llama design that you know.

**Math.** Section 2 says Mixtral uses the same modifications as Mistral 7B, with a fully dense context of 32k tokens and the feed-forward blocks replaced by mixture-of-experts layers. The paper doesn't restate the block, so the details below come from your prerequisites and from Table 1. Each block applies RMSNorm, attention, a residual connection, RMSNorm, the MoE layer, and a residual connection, as in [LLaMA](/llama/). Each expert is the SwiGLU network of that page (Section 2.1):

$$
y = \sum_{i=0}^{n-1} \operatorname{Softmax}(\operatorname{Top2}(x \cdot W_g))_i \cdot \operatorname{SwiGLU}_i(x).
$$

Each expert has its own three matrices of shape $4096 \times 14336$ (Table 1: `dim` 4096, `hidden_dim` 14336), and every one of the 32 layers has 8 of them. The paper contrasts this with GShard, which replaces every other feed-forward block and uses a more elaborate rule for the second expert. Attention is shared by all tokens, so it has no experts. Table 1 gives 32 query heads and 8 key-value heads, each of width 128. The paper doesn't comment on the 8 key-value heads in the text. The count below assumes that the query and output projections have the full width and the key and value projections have 8 heads.

The paper doesn't describe an auxiliary load-balancing loss, a capacity factor, or token dropping. It says only that expert parallelism makes load balancing "essential" (Section 2.1). If you need the loss that [Switch Transformers](/switch/) uses, that page has it, but this paper's training recipe isn't given.

**Tiny example, computed by `scripts/param_count.py`.** The weights of one block add up as follows:

| Part | Parameters |
|---|---|
| Attention ($W_q, W_k, W_v, W_o$) | 41,943,040 |
| One expert (3 matrices of $4096 \times 14336$) | 176,160,768 |
| Router $W_g$ ($4096 \times 8$) | 32,768 |

Exercise 1 asks you to turn this table into a function of the Table 1 configuration.

### Active versus total parameters

**Intuition.** A token uses 2 of the 8 experts in each layer, so it touches a quarter of the expert weights, plus all the shared weights. The total parameter count measures what you store, and the active count measures what a token computes with.

**Math.** Section 2.1 draws the distinction. The total (or sparse) parameter count grows with $n$. The active parameter count grows with $K$, up to $n$. Let $L$ be the number of layers, $P_{\text{attn}}$ the attention weights per layer, $P_E$ the weights of one expert, $P_g = d \cdot n$ the router, and $P_{\text{emb}}$ the input embedding plus output matrix. Ignoring the small norm gains:

$$
P_{\text{total}} = L\,(P_{\text{attn}} + n P_E + P_g) + P_{\text{emb}}, \qquad
P_{\text{active}} = L\,(P_{\text{attn}} + K P_E + P_g) + P_{\text{emb}}.
$$

The matrix multiplications of a forward pass cost about 2 FLOPs per weight that a token touches, so the FLOPs per token are about $2 P_{\text{active}}$. This estimate leaves out the attention-score term, which depends on the context length. The paper doesn't give this formula. It says only that the active count "is directly proportional to the inference compute cost" (Section 3).

**Tiny example, computed by `scripts/param_count.py`.** The counts assume untied embeddings, no biases, and two RMSNorm gains per layer plus one final gain, which come from the released code and not from the paper text:

| Quantity | Computed | Paper |
|---|---|---|
| Total parameters | 46,702,792,704 (46.70B) | 47B (Abstract, Section 3) |
| Active parameters | 12,879,925,248 (12.88B) | 13B |
| Active / total | 0.2758 | |
| Share of the total in expert weights | 0.9656 | |
| Same shape with one expert | 7,241,732,096 (7.24B) | |
| Llama 2 70B | 68.98B | 70B |
| Llama 2 70B / Mixtral active | 5.36 | "5x" |

The paper's rounded numbers match. Mixtral's forward pass costs about 25.8 GFLOPs per token, and Llama 2 70B's costs 138.0 GFLOPs. A dense model with the shape of one expert (the Mistral 7B shape) would cost 14.5 GFLOPs, so top-2 routing costs about 1.8 times that, and the model stores 6.4 times as many parameters. The Llama 2 70B row uses the published shape (dimension 8192, 80 layers, 8 key-value heads, hidden width 28,672), which isn't in the Mixtral paper.

Memory goes the other way. In bfloat16 at 2 bytes per weight, Mixtral needs 93.4 GB for the weights, and Llama 2 70B needs 138.0 GB. The paper's wording is that the memory cost is "proportional to its sparse parameter count, 47B, which is still smaller than Llama 2 70B" (Section 3). Section 3 also writes "47B tokens" where it means parameters. Treat that as a typo.

Exercise 1 asks you to implement this count.

### Why a batch touches more than two experts

**Intuition.** One token reads 2 experts per layer. A batch of tokens reads the union of all their choices, and the union grows quickly. This is why the paper says the model suits batched workloads, and why the saving in compute doesn't turn into a saving in weight reads at large batch sizes.

**Math.** Section 3 states the effect in words: the layer adds routing overhead and more memory loads when a device runs more than one expert, and it is suited to workloads where one can reach a good arithmetic intensity. A simple model quantifies the first part. Suppose each token picks its pair of experts uniformly at random, independently of the others. A given expert is unused by one token with probability $1 - K/n$, so for a batch of $B$ tokens

$$
\mathbb{E}[\text{experts touched}] = n\left(1 - \left(1 - \frac{K}{n}\right)^{B}\right).
$$

Real routing isn't uniform or independent (see the next idea), so this is a rough model, not a measurement.

**Tiny example, computed by `scripts/experts_touched.py`.** With $n = 8$ and $K = 2$:

| Batch size $B$ | Experts touched per layer | Expert weights read, relative to one token |
|---|---|---|
| 1 | 2.000 | 1.00 |
| 2 | 3.500 | 1.75 |
| 4 | 5.469 | 2.73 |
| 8 | 7.199 | 3.60 |
| 32 | 7.999 | 4.00 |

The simulation in the script agrees with the formula. By $B = 8$ a layer reads 3.6 times the expert weights that a single token reads. At a batch of 32, every expert is read, so each weight read serves 8 tokens on average and the layer is back to a dense layer's arithmetic intensity, with 4 times the weights of the active set.

### What the router learns

**Intuition.** One might expect experts to specialize by subject: one for mathematics, one for biology. Section 5 tests that and finds something else.

**Math.** The analysis measures the proportion of tokens each expert receives on different subsets of The Pile, at layers 0, 15, and 31 (Section 5, Figure 7). The paper finds no obvious pattern by topic. The distributions for ArXiv (LaTeX), PubMed abstracts (biology), and PhilPapers (philosophy) are very similar at every layer. Only DM Mathematics differs a little, which the paper attributes to its synthetic nature. The paper concludes that "the router does exhibit some structured syntactic behavior". Figure 8 shows that tokens such as `self` in Python and `Question` in English often go to the same expert, and that indentation tokens in code go to the same experts, especially at the first and last layers.

The second finding is positional locality. Table 5 reports the proportion of consecutive tokens that get the same expert, either as the first choice or as first or second choice. For a random assignment over 8 experts, the paper gives the baselines $1/8 = 12.5\%$ and $1 - \frac{6}{8}\cdot\frac{5}{7} \approx 46\%$. The first number is the chance that the next token's first choice equals this one's. The second is the chance that two random pairs out of 8 experts share at least one expert. The paper doesn't spell out its "first or second choice" definition, and the set-overlap reading above is the one that reproduces the 46% baseline.

**Tiny example, computed by `scripts/routing_locality.py`.** The script prints the exact baselines $0.1250$ and $0.4643$, and a simulation of 200,000 random tokens gives $0.1250$ and $0.4652$. Now compare the paper's measurements in Table 5 for first choice: 14.0%, 27.9%, and 22.7% for ArXiv at layers 0, 15, and 31. At layer 0 the router is close to random. At layers 15 and 31 it repeats an expert about twice as often as chance. For the first or second choice, ArXiv is 46.5%, 62.3%, and 52.9% at the same layers.

The paper draws two consequences (Section 5). Locality can cause over-subscription of some experts under expert parallelism, and it can be used for caching, as in the offloading work it cites. These are conclusions from one dataset family and not a general property of the model.

Exercise 4 asks you to compute these repetition rates from routing decisions.

### Putting the layer together

Exercise 3 puts the gate and the experts together in a sparse layer. For each expert, gather the tokens that chose it, run the expert once on those rows, and scatter the weighted result back. The tests count the rows that each expert receives and require $T \cdot K$ rows in total.

<!-- exercises -->

## Pitfalls

- **Taking the softmax before the top-2.** The paper's order is top-2 first, then softmax over the two logits (Section 2.1). If you take the softmax over all experts and keep two entries, the weights sum to less than 1 and the layer output shrinks. In the page's example the sum is 0.6108. Exercises 2 and 3 catch it.
- **Reading "8x7B" as 56B.** The experts share attention, the embeddings, and the router, so the total is 46.70B in the page's count and 47B in the paper, not 8 times 7B. Only the feed-forward weights are multiplied by 8.
- **Treating 13B as the memory footprint.** Active parameters give the compute per token. All 47B must be resident, and a batch of tokens touches most experts (Section 3). The paper claims faster inference at low batch sizes and higher throughput at large batch sizes (Section 1), and it adds that routing overhead and extra memory loads make the model better suited to batched workloads (Section 3).
- **Counting only query heads for attention.** Table 1 lists 32 heads and 8 key-value heads. If you count key and value projections at the full 32 heads, the total moves from 46.70B to about 47.51B, and exercise 1 fails.
- **Assuming the paper trains with a balancing loss.** It doesn't describe one, and it doesn't describe capacity factors or token dropping. The details of the training recipe aren't in the paper. Don't copy the Switch recipe and call it Mixtral's.
- **Reading specialization as topic specialization.** Section 5 reports the opposite for this model: the assignment follows syntax more than domain. The finding comes from a small analysis of The Pile validation subsets at three layers, and the paper itself calls it a small analysis.
- **Comparing the baselines with the wrong reference.** For consecutive tokens, the random baselines are 12.5% for the first choice and about 46% for first or second choice, not 12.5% and 25%. Table 5 at layer 0 is close to them, and the higher layers are well above.
- **Mixing up the benchmark tables.** Table 2 is the paper's own re-evaluation of the Llama models, and Table 3 shows other settings (for example HellaSwag 10-shot, and GSM-8K 5-shot) for Llama 2 70B and GPT-3.5. The numbers for the same benchmark differ between the two tables, so don't mix them. The paper also says it re-ran every benchmark with its own pipeline (Section 3).
- **Treating the instruct model as the base model.** Mixtral 8x7B Instruct is trained with supervised fine-tuning and Direct Preference Optimization (Section 4). The 8.30 MT-Bench score belongs to the instruct model, and the base-model benchmarks above are separate.

## Further reading

- The paper's reference code is in [github.com/mistralai/mistral-src](https://github.com/mistralai/mistral-src), linked from the abstract.
- The paper says it integrated Megablocks into vLLM for inference. [MegaBlocks](https://arxiv.org/abs/2211.15841) casts the expert computation as block-sparse matrix multiplication, which handles a variable number of tokens per expert.
- [A Review of Sparse Expert Models in Deep Learning](https://arxiv.org/abs/2209.01667), by Fedus, Dean, and Zoph, is the overview that the paper recommends for a more in-depth treatment (Section 2.1).
- [Mistral 7B](https://arxiv.org/abs/2310.06825) defines the architecture that Mixtral builds on.
- [Switch Transformers](/switch/) is the router, balancing loss, and capacity design that this page builds on.
- [LLaMA](/llama/) defines the RMSNorm, SwiGLU, and rotary-embedding block that Mixtral uses.
