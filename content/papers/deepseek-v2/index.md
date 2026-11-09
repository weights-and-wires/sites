---
id: "deepseek-v2"
title: "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"
authors: "DeepSeek-AI et al."
year: 2024
venue: "arXiv preprint"
arxiv: "2405.04434"
url: "https://arxiv.org/abs/2405.04434"
track: architectures
tier: core
est_hours: 5
status: full
summary: "DeepSeek-V2 is a 236B-parameter mixture-of-experts language model that activates 21B parameters per token and supports a 128K context. Its Multi-head Latent Attention (MLA) caches one small latent vector per token instead of per-head keys and values, and a separate rotary key keeps position information compatible with the compression. Its feed-forward layers use DeepSeekMoE with device-limited routing, which together cut the KV cache by more than 90 percent and training cost by 42.5 percent against the dense DeepSeek 67B."
related_course: "inference: KV cache compression"
prereqs:
  - id: gqa
    reason: "MLA is compared with GQA and MHA by KV cache size per token, and GQA's group count is the yardstick for how small the MLA cache is."
  - id: deepseekmoe
    reason: "DeepSeek-V2 uses the DeepSeekMoE layer, with shared and fine-grained routed experts, for every feed-forward layer after the first."
  - id: rope
    reason: "MLA can't apply rotary embeddings to the compressed keys, so the paper adds a separate rotary key and query that carry the position."
background: [low-rank-projection, kv-cache]
course_links:
  - title: "The KV cache"
    url: "https://qwen.ifkash.dev/c/09-the-kv-cache"
  - title: "Grouped-query attention"
    url: "https://qwen.ifkash.dev/c/07-grouped-query-attention"
self_check:
  - q: 'A model has $n_h = 4$ heads of width $d_h = 8$ and 1 layer. How many elements does multi-head attention cache per token? How many does an MLA layer cache with $d_c = 8$ and $d_h^R = 4$?'
    a: 'MHA caches $2 n_h d_h = 64$ elements, a key and a value for each head. MLA caches $d_c + d_h^R = 12$ elements: one latent plus one shared rotary key (Section 2.1.3). The factor of 2 disappears because keys and values come from the same latent.'
    see: kv-cache
  - q: 'A matrix $W^{UK} \in \mathbb{R}^{16384 \times 512}$ expands a latent into keys. What is the largest rank of the matrix $W^{UK} W^{DKV}$ that maps $h_t$ to the keys, and why does that make the projection low-rank?'
    a: 'At most 512, because the product passes through a 512-wide bottleneck. A full key projection could have rank up to 5120 for $d = 5120$. All heads read the same 512-dimensional latent, so the keys of all heads live in one shared 512-dimensional subspace.'
    see: low-rank-projection
  - q: 'In RoPE, the score of a query at position $m$ and a key at position $n$ is $q^\top R_{n-m} k$ for a rotation matrix $R$. Why does that stop you from folding $W^{UK}$ into the query projection once the keys are rotated?'
    a: 'The score becomes $q^\top R\, W^{UK} c$, and $R$ depends on the relative position $n - m$. Matrix products don''t commute, so you can''t move $W^{UK}$ next to $W^Q$ to make one fixed matrix. You would need a new matrix for every offset.'
    see: rope
---

## The problem

During generation a decoder keeps the keys and values of every earlier token, the KV cache (see the [GQA](/gqa/) page for the full story). Standard multi-head attention (MHA) caches $2 n_h d_h l$ elements per token (Section 2.1.1), where $n_h$ is the head count, $d_h$ the head width, and $l$ the layer count. Section 2.1.1 calls this cache "a large bottleneck that limits the maximum batch size and sequence length."

Take the shape that Section 3.1.2 gives DeepSeek-V2: $n_h = 128$, $d_h = 128$, and $l = 60$. MHA would cache 1,966,080 elements per token. At 16 bits per element and a 128K-token context, that is 480 GiB for one sequence. Both numbers come from `scripts/kv_cache_sizes.py`.

[GQA](/gqa/) and multi-query attention shrink the cache by sharing key-value heads, but Appendix D.1 shows the price. In 7B dense models trained on 1.33T tokens, MHA scores 45.2 on MMLU, GQA with 8 groups scores 41.2, and MQA scores 37.9 (Table 8). The paper asks for a cache as small as the one for GQA with very few groups, without the quality loss.

Its answer has two parts (Section 2). Multi-head Latent Attention (MLA) shrinks the cache. The feed-forward layers use [DeepSeekMoE](/deepseekmoe/) to train a large model at a lower cost. The abstract summarizes the result against the dense DeepSeek 67B: 42.5 percent less training cost, a 93.3 percent smaller KV cache, and 5.76 times the maximum generation throughput. The model has 236B total parameters, activates 21B per token, and trains on 8.1T tokens.

## Key ideas

The notation follows the paper. Row $t$ of the input is the vector $h_t \in \mathbb{R}^d$. The paper writes column vectors, so $W h_t$ is `H @ W.T` in the code of the exercises. The paper sometimes leaves out the head index on the up-projection matrices. This page writes $W^{UK}_i$ for the block of rows that belongs to head $i$.

### Cache one latent vector instead of keys and values

**Intuition.** MHA stores $n_h$ keys and $n_h$ values per token, but they are all linear functions of the same $h_t$. MLA squeezes $h_t$ through a narrow bottleneck first and stores only the narrow vector. A later step expands it back into keys and values.

**Math.** Section 2.1.2 defines the low-rank joint compression (Equations 9-11):

$$
c_t^{KV} = W^{DKV} h_t, \qquad k_t^C = W^{UK} c_t^{KV}, \qquad v_t^C = W^{UV} c_t^{KV}
$$

The latent $c_t^{KV} \in \mathbb{R}^{d_c}$ has $d_c \ll d_h n_h$. The down-projection is $W^{DKV} \in \mathbb{R}^{d_c \times d}$, and the up-projections are $W^{UK}, W^{UV} \in \mathbb{R}^{d_h n_h \times d_c}$. The key and value of head $i$ are slices of $k_t^C$ and $v_t^C$, as in Equations 4-6. Only $c_t^{KV}$ is cached, so the cache holds $d_c\,l$ elements per token. The paper also compresses the queries (Equations 12-13) with width $d_c'$. That saves activation memory in training and doesn't change the cache (Section 2.1.2).

**Tiny example, computed by `scripts/kv_cache_sizes.py`.** DeepSeek-V2 sets $d_c = 512$ (Section 3.1.2), which is $4 d_h$. The latent path holds 512 numbers per token and layer, while MHA holds $2 \cdot 128 \cdot 128 = 32{,}768$. The matrices $W^{DKV}$ and $W^{UK}$ together have 11,010,048 weights per layer. A full-rank key projection from $d = 5120$ to 16,384 outputs has 83,886,080. The product $W^{UK} W^{DKV}$ has rank at most 512, so every head reads keys from the same 512-dimensional subspace.

### Fold the up-projections into the query and output

**Intuition.** If the cache holds latents and every step has to expand them into keys, the saving would cost compute at every step. Matrix multiplication is associative, so the expansion can move to the other side of the dot product.

**Math.** Section 2.1.2 states that "$W^{UK}$ can be absorbed into $W^Q$, and $W^{UV}$ can be absorbed into $W^O$." Write the query of head $i$ as $q_{t,i}$ and the key of head $j$ as $k^C_{j,i} = W^{UK}_i c_j^{KV}$. Then

$$
q_{t,i}^\top k^C_{j,i} = q_{t,i}^\top W^{UK}_i c_j^{KV} = \big({W^{UK}_i}^\top q_{t,i}\big)^\top c_j^{KV}.
$$

The right side scores the cached latent directly. The value side works the same way. The weighted sum $\sum_j a_{t,j} v^C_{j,i} = W^{UV}_i \sum_j a_{t,j} c_j^{KV}$ takes the sum over latents first, and the multiplication by $W^{UV}_i$ can then merge with the output projection $W^O$ (Equation 8). Appendix C says the same, and it writes the query-side matrix as $W^{UQ}$, because the paper's training-time queries go through the compressed $c_t^Q$ (Equations 12-13). Absorption then merges ${W^{UK}_i}^\top W^{UQ}_i$. This page drops the query compression in the exercises to keep the shapes small.

**Tiny example, computed by `scripts/absorb_example.py`.** Take one head with $d_h = d_c = 2$, and

$$
W^{UK} = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}, \quad q = \begin{bmatrix} 2 \\ 1 \end{bmatrix}, \quad c = \begin{bmatrix} 1 \\ 3 \end{bmatrix}.
$$

The expanded key is $W^{UK} c = [4, 3]$ and the score is $q \cdot k = 8 + 3 = 11$. The absorbed query is ${W^{UK}}^\top q = [2, 3]$, and its dot product with $c$ is $2 + 9 = 11$. A random check with $n_h = 2$, $d_h = 4$, $d_c = 3$, and 6 tokens finds a maximum difference of $2.7 \times 10^{-15}$ between the two causal attention outputs. The naive version holds 16 numbers per token, and the latent holds 3.

Exercise 2 asks you to write both versions and test that they agree.

### Decouple the rotary part

**Intuition.** RoPE rotates a key by an angle that depends on its position. If the rotation sits between the up-projection and the query, the absorbed matrix would change with the distance between the two tokens. The fix is to keep position out of the compressed path altogether and add a small separate path that carries it.

**Math.** Section 2.1.3 explains the conflict. If RoPE is applied to $k_t^C$, "a RoPE matrix related to the currently generating token will lie between $W^Q$ and $W^{UK}$" and the keys of the whole prefix must be recomputed. The decoupled strategy adds multi-head queries $q^R_{t,i} \in \mathbb{R}^{d_h^R}$ and one key $k_t^R \in \mathbb{R}^{d_h^R}$ shared by all heads (Equations 14-17):

$$
q_t^R = \mathrm{RoPE}(W^{QR} c_t^Q), \qquad k_t^R = \mathrm{RoPE}(W^{KR} h_t),
$$

$$
q_{t,i} = [q^C_{t,i};\, q^R_{t,i}], \qquad k_{t,i} = [k^C_{t,i};\, k^R_t].
$$

The attention output is (Equation 18)

$$
o_{t,i} = \sum_{j=1}^{t} \operatorname{Softmax}_j\!\left(\frac{q_{t,i}^\top k_{j,i}}{\sqrt{d_h + d_h^R}}\right) v^C_{j,i}.
$$

The score splits into a content term $q^{C\top}_{t,i} k^C_{j,i}$ with no position, and a rotary term that depends only on $t - j$. The rotary key must be cached too, so the cache is $(d_c + d_h^R)\,l$ elements per token (Section 2.1.3). The values come only from the latent, with no rotary part. The paper's Appendix C (Equations 37-47) lists the complete forward pass.

**Tiny example, computed by `scripts/decoupled_rope_example.py`.** Take $d_h = d_h^R = 2$, $q^C = [1, 2]$, $k^C = [3, 1]$, so the content score is 5. Let $q^R = k^R = [1, 0]$ before rotation. With the first rotary pair at frequency 1, the rotary term for positions $(t, j)$ is $\cos(t - j)$. For $(1, 0)$, $(5, 4)$, and $(9, 8)$ it is 0.5403 each time, and the scaled score is $(5 + 0.5403)/\sqrt{4} = 2.7702$. For $(3, 0)$ it is $\cos 3 = -0.9900$ and the scaled score is 2.0050. The same offset gives the same score, as in [RoPE](/rope/).

The script also shows what goes wrong without decoupling. For $W^{UK}$ above, the matrix ${W^{UK}}^\top R(\Delta)$ that would sit in the absorbed score is $[[1, 0], [1, 1]]$ at $\Delta = 0$, $[[0.5403, -0.8415], [1.3818, -0.3012]]$ at $\Delta = 1$, and a different matrix at every other offset. (The sign of the angle depends on the convention.)

Exercise 3 asks you to compute the decoupled scores.

### Compare cache sizes

**Intuition.** MLA's cache has no factor of 2 and no head count in it, so a few hundred numbers per layer replace thousands.

**Math.** Table 1 (Section 2.1.4) lists the cache per token: MHA caches $2 n_h d_h l$, GQA caches $2 n_g d_h l$, MQA caches $2 d_h l$, and MLA caches $(d_c + d_h^R)\,l$. DeepSeek-V2 sets $d_c = 4 d_h$ and $d_h^R = d_h/2$, so MLA caches $\tfrac{9}{2} d_h l$ elements, "equal to GQA with only 2.25 groups". Solving $2 n_g d_h = (d_c + d_h^R)$ gives $n_g = 2.25$.

**Tiny example, computed by `scripts/kv_cache_sizes.py`.**

| Mechanism | Elements per token, $l = 60$ |
|---|---|
| MHA, $n_h = 128$ | 1,966,080 |
| GQA, $n_g = 8$ | 122,880 |
| GQA, $n_g = 2$ | 30,720 |
| MQA | 15,360 |
| MLA, $d_c = 512$, $d_h^R = 64$ | 34,560 |

MLA is 56.89 times smaller than this MHA, a saving of 98.24 percent, and its 128K-token cache in 16-bit precision is 8.438 GiB against 480 GiB. The 34,560 matches the value that Table 9 lists for the MLA large MoE model (34.6K); it falls between GQA with 2 groups and GQA with 8. The abstract's 93.3 percent compares against DeepSeek 67B, whose attention configuration isn't in this paper, so you can't reproduce it from the table.

Exercise 1 asks you to write these formulas.

### Put DeepSeekMoE in the feed-forward layers

**Intuition.** The attention change saves inference memory. The feed-forward change saves training compute, because each token passes through a few small experts and not through one big layer.

**Math.** Section 2.2.1 applies DeepSeekMoE (Equations 20-22), which is the layer from the [DeepSeekMoE](/deepseekmoe/) page. The output is the residual input, plus $N_s$ shared experts that every token uses, plus $N_r$ routed experts weighted by gates $g_{i,t}$. A gate is the softmax affinity $s_{i,t} = \operatorname{Softmax}_i(u_t^\top e_i)$ if it is among the top $K_r$, and 0 otherwise. This page's notation matches the paper's. Section 3.1.2 sets $N_s = 2$, $N_r = 160$, $K_r = 6$, expert hidden width 1536, and uses MoE in every layer except the first.

DeepSeek-V2 adds two things to the layer from the earlier paper. Device-limited routing (Section 2.2.2) first picks the $M$ devices that hold the highest-affinity experts for a token and then takes the top $K_r$ among experts on those devices, which bounds communication; the paper finds $M \ge 3$ works about as well as unrestricted routing. The balance loss now has three levels (Section 2.2.3): experts (Equations 23-25), devices (Equations 26-28), and communication (Equations 29-31). Section 2.2.4 adds token dropping at training time.

**Tiny example, computed by `scripts/kv_cache_sizes.py`.** One layer has $2 + 160 = 162$ experts, and each token uses $2 + 6 = 8$ of them.

<!-- exercises -->

## Pitfalls

- **Treating MLA as a cheaper way to compute attention.** MLA saves cache memory, not scores. After absorption the score is a dot product of width $d_c = 512$ per head, against $d_h = 128$ in the expanded form. The paper doesn't discuss this trade, and the cost depends on your kernel and hardware.
- **Counting values in the MLA cache.** The cache has $c_t^{KV}$ and $k_t^R$. It has no separate values, because $v_t^C = W^{UV} c_t^{KV}$ (Section 2.1.2). Counting a factor of 2, as for MHA, gives twice the paper's number.
- **Forgetting the rotary key.** A cache of only $d_c\,l$ elements is 30,720 per token here, not 34,560. Section 2.1.3 requires caching $k_t^R$.
- **Rotating the content key.** Applying RoPE to $k^C_t$ is the exact conflict that Section 2.1.3 describes. The rotary part of Equations 14-15 is a separate, narrow vector.
- **Using the wrong scale.** Equation 18 divides by $\sqrt{d_h + d_h^R}$, the width of the concatenated vector. In the absorbed form the dot product has width $d_c$, but the divisor stays the same.
- **Sharing the rotary query across heads.** The rotary key $k^R_t$ is shared and has no head index. The rotary queries $q^R_{t,i}$ are per head, from $W^{QR} \in \mathbb{R}^{d_h^R n_h \times d_c'}$.
- **Reading Table 1 as bytes.** The paper counts elements "regardless of the storage precision."
- **Reading the 93.3 percent as a saving against MHA.** The abstract compares with DeepSeek 67B, a different model. The MHA comparison with this shape is 98.24 percent here, computed by `scripts/kv_cache_sizes.py`.

## Further reading

- The official repository is [deepseek-ai/DeepSeek-V2](https://github.com/deepseek-ai/DeepSeek-V2), where the paper points for the model checkpoints (abstract).
- Appendix D.2 compares MLA and MHA on small and large MoE models: Table 9 lists 15.6K against 110.6K and 34.6K against 860.2K cache elements per token, with MLA ahead on most benchmarks.
- Appendix C (Equations 37-47) is the full MLA forward pass. Read it with the page open next to it.
