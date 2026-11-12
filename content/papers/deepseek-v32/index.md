---
id: "deepseek-v32"
title: "DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models"
authors: "DeepSeek-AI"
year: 2025
venue: "arXiv preprint"
arxiv: "2512.02556"
url: "https://arxiv.org/abs/2512.02556"
track: inference
tier: extended
est_hours: 4
status: full
summary: "DeepSeek-V3.2 adds DeepSeek Sparse Attention (DSA) to DeepSeek-V3.1-Terminus through continued pre-training. A small lightning indexer scores every earlier token for each query, and the main attention then reads only the top 2048 key-value entries, which cuts the core attention cost from quadratic to linear in the sequence length. The report also describes a scaled reinforcement learning stage and an agentic data pipeline, which this page covers briefly."
related_course: "inference: KV economy and attention variants"
prereqs:
  - id: deepseek-v2
    reason: "DSA is instantiated on Multi-head Latent Attention, so you need MLA's latent key-value entry and its MQA-style use at decode time to see what the indexer selects."
  - id: deepseek-v3
    reason: "DeepSeek-V3.2 continues from a checkpoint of the DeepSeek-V3 line (DeepSeek-V3.1-Terminus) and reuses its post-training recipe, so the base model is the baseline for every comparison."
  - id: fp8
    reason: "The paper says the indexer can run in FP8, so you need the idea of a low-precision matrix multiply to read why a cheap scoring pass is worth adding."
background: [kv-cache, softmax, kl-divergence, dot-products]
course_links:
  - title: "KV economy: MLA, windows, sinks, and eviction"
    url: "https://qwen.ifkash.dev/c/07a-kv-economy-attention-variants"
self_check:
  - q: 'A sequence has $L = 6$ tokens and DSA uses $k = 3$. Count the query-key pairs that dense causal attention scores and that DSA main attention scores. Does the indexer save any pairs?'
    a: 'Dense causal attention scores $6 \cdot 7 / 2 = 21$ pairs. Query $t$ (counting from 0) keeps $\min(t + 1, 3)$ keys, so DSA scores $1 + 2 + 3 + 3 + 3 + 3 = 15$ pairs. The indexer still scores all 21 pairs, because it ranks every earlier token (Section 2.3). Only the main attention shrinks.'
    see: kv-cache
  - q: 'Why does the indexer need its own loss, and what does the paper train it to match?'
    a: 'The paper doesn''t give its reason, but a hard top-k selection passes no gradient to the scores, so this page reads the separate loss as the way to train the indexer. The paper trains the indexer with a KL loss to the main attention distribution, summed over heads and L1-normalized (Equations 3 and 4). It also detaches the indexer input, so the indexer learns only from this loss and the main model learns only from the language-modeling loss (Section 2.1.1).'
    see: kl-divergence
  - q: 'Equation 3 writes $D_{KL}(p_{t,:} \,\|\, \mathrm{softmax}(I_{t,:}))$ with the attention target first. What happens to a term when $p$ is large for a key but $\mathrm{softmax}(I)$ is near 0 for it?'
    a: 'The term $p \log(p / q)$ grows without bound as $q \to 0$ with $p > 0$. The loss punishes the indexer hardest for ranking a key low when the main attention relies on it, which is the miss that DSA must avoid.'
    see: kl-divergence
---

## The problem

Decoding with a long context is expensive for two reasons. The KV cache grows with the sequence, and each new token attends to every earlier one. The [DeepSeek-V2](/deepseek-v2/) page shows how Multi-head Latent Attention (MLA) shrinks the cache. MLA does not change how many tokens a query reads. Core attention still costs $O(L^2)$ over a sequence of length $L$.

The introduction names this as the first of three weaknesses of open models. It says reliance on vanilla attention "severely constrains efficiency for long sequences" and makes both deployment and post-training hard (Section 1). A query at position 100,000 reads 100,000 entries.

A fixed pattern, such as a sliding window, can't adapt to the content of the query. DSA lets the model learn which entries to read. It adds a cheap scorer that ranks the earlier tokens, then runs the full attention on only the best ones (Section 2.1).

The report has two other parts. A scaled reinforcement learning stage spends a post-training budget of more than 10 percent of the pre-training cost (Section 1). An agentic pipeline synthesizes over 1,800 environments and 85,000 prompts for tool-use training (Section 1). The abstract says the high-compute variant, DeepSeek-V3.2-Speciale, reaches gold-medal performance on the 2025 IMO and IOI. This page focuses on DSA (Section 2); the last part of the key ideas covers the rest briefly.

## Key ideas

The notation follows the paper. A query token has hidden state $h_t \in \mathbb{R}^d$, and $h_s$ is an earlier token with $s \le t$. The paper doesn't state the number of indexer heads $H^I$ or the indexer width $d^I$, so the examples on this page use small toy values. They are not the model's settings.

### Score every earlier token with a lightning indexer

**Intuition.** The indexer is a small multi-head attention-like scorer. It gives each query-key pair one number, and it doesn't produce values or a softmax. A high number means that the main attention probably cares about that key.

**Math.** Section 2.1 (Equation 1) defines the index score:

$$
I_{t,s} = \sum_{j=1}^{H^I} w^I_{t,j} \cdot \mathrm{ReLU}\!\left(q^I_{t,j} \cdot k^I_s\right)
$$

Here $q^I_{t,j} \in \mathbb{R}^{d^I}$ and the scalar $w^I_{t,j}$ come from the query token $h_t$, and $k^I_s \in \mathbb{R}^{d^I}$ comes from the earlier token $h_s$. The paper chooses ReLU for throughput, and notes that the indexer has few heads and can run in FP8.

Two features of the formula matter. Each head clips negative dot products to 0 before the weighting, so a head can only vote for a key and never against it. The weights $w^I_{t,j}$ depend on the query, so the query decides how much each head counts. Nothing constrains the weights to be positive.

**Tiny example, computed by `scripts/indexer_example.py`.** Take $H^I = 2$ and $d^I = 2$. For query 3 and key 2, the two head queries are $(1, 0)$ and $(1, 1)$, the key is $(1, 1)$, and the weights are $(0.5, 0.5)$. The dot products are 1 and 2, both positive, so $I_{3,2} = 0.5 \cdot 1 + 0.5 \cdot 2 = 1.5$. For key 3 the dots are $-1$ and $0$, ReLU clips both to 0, and $I_{3,3} = 0$.

Exercise 1 asks you to compute the whole $L \times L$ score array with a causal mask.

### Keep only the top-k entries

**Intuition.** For each query, rank the earlier tokens by $I_{t,s}$ and keep the best $k$. The main attention then sees a short list, not the whole prefix.

**Math.** Section 2.1 (Equation 2) computes the attention output $u_t$ from the query token and only the selected key-value entries:

$$
u_t = \mathrm{Attn}\!\left(h_t, \{\, c_s \mid I_{t,s} \in \operatorname{Top-k}(I_{t,:}) \,\}\right)
$$

The entries $c_s$ are MLA latent vectors. Section 2.1.1 sets $k = 2048$ for the sparse training stage. A query with fewer than $k$ visible keys selects all of them, so DSA matches dense attention for short prefixes and for any $L \le k$.

**Tiny example, computed by `scripts/indexer_example.py`.** With $k = 3$ on a six-token sequence, query 2 sees three keys and keeps all three. Query 3 sees four keys and keeps $\{0, 1, 2\}$, dropping key 3 with score 0. Query 4 keeps $\{1, 2, 3\}$, and query 5 keeps $\{2, 4, 5\}$.

Exercise 2 asks you to write this selection, including the short-prefix case.

### Attend over the selection in MQA mode

**Intuition.** The selected keys differ for every query, but a fast kernel wants many queries to read the same key-value entry. MLA has a mode that gives exactly this.

**Math.** Section 2.1 builds DSA on MLA, because the model continues from DeepSeek-V3.1-Terminus, which uses MLA. For kernel efficiency, each key-value entry must be shared across multiple queries, so the paper implements DSA in the MQA mode of MLA. In that mode, each latent vector, the key-value entry of MLA, is shared across all query heads of the query token (Section 2.1, Figure 2, Appendix A). Appendix A notes that DeepSeek-V3.1-Terminus uses the MHA mode for training and prefilling and the MQA mode for decoding. The softmax then runs only over the selected set. The exercise on this idea uses an ordinary scaled dot product with one shared key and value per token, which stands in for the latent entry. It doesn't model MLA's projections.

**Tiny example, computed by `scripts/sparse_attention_example.py`.** Query 3 has logits $(0, 1, 2, 0)$ over four keys. Dense attention gives weights $(0.0826, 0.2245, 0.6103, 0.0826)$. Selecting keys $\{0, 1, 2\}$ renormalizes to $(0.09, 0.2447, 0.6652)$. The dropped key had weight 0.0826, and the output changes by at most 0.1999 in one coordinate. When $k$ covers all four keys, the output equals the dense one. When $k = 1$, the output is the value of the one selected key.

Exercise 3 asks you to implement this attention and check these limits.

### Train the indexer by aligning it with dense attention

**Intuition.** The paper trains the indexer with its own loss, which pulls the indexer toward the model's dense attention, in two stages. The page's reading is that a hard top-k selection passes no gradient to the scores, so the language-modeling loss can't train the indexer; the paper doesn't state this reason.

**Math.** Section 2.1.1 starts from DeepSeek-V3.1-Terminus, whose context length is already 128K, and uses the same data as that model's 128K extension for both stages.

- **Dense warm-up.** Attention stays dense, and every parameter except the indexer is frozen. The target is the main attention scores summed over heads and L1-normalized along the sequence, $p_{t,:} \in \mathbb{R}^t$. The loss is a KL divergence (Equation 3):

  $$
  \mathcal{L}^I = \sum_t D_{KL}\!\left(p_{t,:} \,\middle\|\, \mathrm{Softmax}(I_{t,:})\right)
  $$

  The learning rate is $10^{-3}$. The indexer trains for 1000 steps of 16 sequences of 128K tokens each.
- **Sparse training.** The top-k selection turns on, and all parameters train. The indexer loss now covers only the selected set $\mathcal{S}_t$ (Equation 4):

  $$
  \mathcal{L}^I = \sum_t D_{KL}\!\left(p_{t,\mathcal{S}_t} \,\middle\|\, \mathrm{Softmax}(I_{t,\mathcal{S}_t})\right)
  $$

  The indexer input is detached from the computation graph. The indexer learns only from $\mathcal{L}^I$, and the main model learns only from the language-modeling loss. The learning rate is $7.3 \times 10^{-6}$, with $k = 2048$ and 15,000 steps of 480 sequences of 128K tokens each.

The paper doesn't say whether Equation 4 renormalizes $p$ and the softmax over $\mathcal{S}_t$. This page and exercise 4 renormalize both, which is one reading of the equation.

**Tiny example, computed by `scripts/kl_warmup_example.py`.** Query 3 sees four keys. Two heads give the weights $(0.1, 0.2, 0.6, 0.1)$ and $(0.3, 0.1, 0.4, 0.2)$, which sum to $(0.4, 0.3, 1.0, 0.3)$ and normalize to $p = (0.2, 0.15, 0.5, 0.15)$. Index scores $(1, 0.5, 1.5, 0)$ give $\mathrm{softmax}(I) = (0.276, 0.1674, 0.4551, 0.1015)$, so the dense loss is 0.0247. If $I = \log p$, the loss is 0. With $\mathcal{S}_3 = \{0, 1, 2\}$, the selected set holds 0.85 of the target mass, and the renormalized loss is 0.0157.

`scripts/cost_counts.py` also checks the paper's token totals: $1000 \cdot 16 \cdot 131072 = 2,097,152,000$ tokens (the paper's 2.1B) for the warm-up, and $15000 \cdot 480 \cdot 131072 = 943,718,400,000$ (943.7B) for the sparse stage, taking 128K as 131,072. The sparse stage uses 450 times the warm-up's tokens.

Exercise 4 asks you to write both losses.

### Count the savings, and note what the paper leaves out

**Intuition.** DSA shrinks the expensive part but adds a new cost. The indexer still scores every pair.

**Math.** Section 2.3 states that DSA reduces the core attention complexity from $O(L^2)$ to $O(Lk)$ with $k \ll L$. The indexer is still $O(L^2)$, but it needs far less computation than MLA in DeepSeek-V3.1-Terminus. Figure 3 plots token cost against position in the sequence for prefilling and decoding, measured on H800 GPUs at a rental price of 2 USD per GPU hour. For short-sequence prefilling, the implementation uses a masked MHA mode to simulate DSA. The paper gives Figure 3 as a chart, so this page quotes no cost figures from it.

**Tiny example, computed by `scripts/cost_counts.py`.** With $k = 2048$, the table counts causal query-key pairs. The indexer column equals the dense one.

| $L$ | Dense pairs | DSA main pairs | Dense / DSA main |
|---|---|---|---|
| 2,048 | 2,098,176 | 2,098,176 | 1.00 |
| 8,192 | 33,558,528 | 14,681,088 | 2.29 |
| 32,768 | 536,887,296 | 65,012,736 | 8.26 |
| 131,072 | 8,590,000,128 | 266,339,328 | 32.25 |

At $L = k$ there is no saving, and the gain grows with $L$. These are pair counts. The paper doesn't give $H^I$ or $d^I$, so you can't turn them into FLOPs or dollars from the text.

**The rest of the report.** After continued pre-training, post-training (Section 3) uses the same sparse attention. It trains specialist models for each domain, distills them into one checkpoint, and then runs one mixed reinforcement learning stage with GRPO that merges reasoning, agent, and alignment training. The paper's own parity check against DeepSeek-V3.1-Terminus finds no substantial degradation on short- and long-context tasks (Section 2.2).

<!-- exercises -->

## Pitfalls

- **Reading DSA as a fixed sparse pattern.** The selection is content-based and differs for each query. A window or a block pattern needs no indexer.
- **Assuming the indexer is free.** It is still $O(L^2)$ (Section 2.3). Only the main attention becomes $O(Lk)$.
- **Expecting a saving for short sequences.** With $k = 2048$, any query with at most 2048 visible keys keeps them all. The paper handles short prefilling with a masked MHA mode that simulates DSA.
- **Applying ReLU after the head sum.** Equation 1 clips each head's dot product before the weight. A negative weight times a positive ReLU output is allowed, and the order changes the score.
- **Selecting future keys.** The score is defined for $s \le t$. Mask the future before the top-k, or the selection can pick keys that the query can't see.
- **Forgetting the short-prefix case.** For $t + 1 < k$, select all $t + 1$ keys. A fixed-size `argsort` slice picks masked entries.
- **Sending the language-modeling gradient into the indexer.** The paper detaches the indexer input, and the indexer trains only on $\mathcal{L}^I$ (Section 2.1.1).
- **Swapping the KL arguments.** Equation 3 puts the attention target $p$ first. The reversed divergence weights the errors differently.
- **Inventing the indexer's size.** The paper doesn't state $H^I$, $d^I$, or the FP8 details beyond "can be implemented in FP8". The toy values here aren't the model's.
- **Quoting costs from Figure 3 as if they were in the text.** The figure reports measured costs on H800 clusters. This page reproduces no numbers from it.

## Further reading

- The paper points to an [open-source inference implementation](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/tree/main/inference) of DeepSeek-V3.2-Exp, which specifies the architecture unambiguously (Section 2.1).
- Appendix A (Figure 7) draws the MHA and MQA modes of MLA and the transformation between them. Read it with the [DeepSeek-V2](/deepseek-v2/) page.
- Section 2.2 reports the parity evaluation against DeepSeek-V3.1-Terminus, including the independent long-context results (AA-LCR and Fiction.liveBench).
- Section 3.1 lists the stability changes to GRPO for scaled reinforcement learning, if you want the post-training side of the report.
