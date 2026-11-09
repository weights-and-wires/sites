---
id: "gqa"
title: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
authors: "Ainslie et al."
year: 2023
venue: "EMNLP 2023"
arxiv: "2305.13245"
url: "https://arxiv.org/abs/2305.13245"
track: architectures
tier: core
est_hours: 3
status: full
summary: "Autoregressive decoding is slow mostly because each step loads the weights and every cached key and value from memory. Grouped-query attention splits the query heads into G groups and gives each group one shared key head and one shared value head, which shrinks the KV cache by a factor of H/G and sits between multi-head attention (G = H) and multi-query attention (G = 1). The paper converts an existing multi-head checkpoint by mean-pooling the key and value heads in each group, then trains for 5% of the original steps, and the result gets close to multi-head quality at close to multi-query speed."
related_course: "inference: KV cache"
prereqs:
  - id: transformer
    reason: "GQA edits the multi-head attention layer from that paper, so you need its per-head projections $W_i^Q$, $W_i^K$, $W_i^V$ and the concatenate-then-$W^O$ step."
background: [kv-cache, memory-bandwidth]
course_links:
  - title: "Grouped-query attention"
    url: "https://qwen.ifkash.dev/c/07-grouped-query-attention"
  - title: "The KV cache"
    url: "https://qwen.ifkash.dev/c/09-the-kv-cache"
self_check:
  - q: 'A layer has $H = 32$ query heads of width $d_k = 128$. How many key and value numbers does it store per token under MHA, GQA-8, and MQA?'
    a: 'MHA stores $2 \cdot 32 \cdot 128 = 8192$ numbers per layer, GQA-8 stores $2 \cdot 8 \cdot 128 = 2048$, and MQA stores $2 \cdot 1 \cdot 128 = 256$. The factor 2 counts one key and one value per cached head, and the cache shrinks by $H/G$.'
    see: kv-cache
  - q: 'During generation, a decoding step computes attention for one new token. Why is the step limited by memory bandwidth and not by arithmetic?'
    a: 'Each step multiplies a few vectors by large matrices and by the whole cache, so it does little arithmetic per byte it loads. It must read all the weights and all cached keys and values to produce one token, which is why shrinking the cache speeds it up.'
    see: memory-bandwidth
  - q: 'In multi-head attention, what are the shapes of $W_i^Q$, $W_i^K$, and $W_i^V$ for one head, and which of them does GQA share across heads?'
    a: 'Each has shape $d_{\text{model}} \times d_k$. GQA keeps one $W_i^Q$ per query head and shares $W^K$ and $W^V$ across the heads of a group, so $W^O$ and the number of query heads stay the same.'
    see: transformer
---

## The problem

The [Transformer](/transformer/) trains fast because attention over a whole sequence is one matrix product. Generating text is different. A decoder emits one token at a time, and each step attends to every earlier token. To avoid recomputing the keys and values of earlier tokens, the decoder stores them in the KV cache and reads them back at every step.

Section 1 of the paper names the cost: autoregressive decoder inference is a severe bottleneck because of "the memory bandwidth overhead from loading decoder weights and all attention keys and values at every decoding step". The cache grows with the number of layers, the number of key-value heads, the head width, and the sequence length, and every sequence in a batch needs its own copy.

Take a 7B-scale model with 32 layers, 32 heads of width 128, and fp16 values. These are assumptions for illustration and aren't numbers from the paper. `scripts/kv_cache_example.py` computes the following:

- One token adds $2 \cdot 32 \cdot 32 \cdot 128 \cdot 2 = 524{,}288$ bytes, which is 512 KiB.
- A 4096-token sequence therefore holds 2 GiB of cache.
- The fp16 weights take about 12.55 GiB. On an 80 GiB accelerator that leaves room for 33 such sequences at once.

Multi-query attention (MQA), proposed by Shazeer (2019), keeps every query head but uses one key head and one value head. That shrinks the cache by a factor of $H$ and relieves the bandwidth pressure. The paper lists its costs in Section 1: MQA "can lead to quality degradation and training instability", and training a separate model just for inference may not be feasible. Many public models, such as T5 and LLaMA, were trained with multi-head attention (MHA) and have no MQA version.

The paper makes two contributions. It shows that an MHA checkpoint converts to MQA with a small amount of extra training (Section 2.1). It also proposes grouped-query attention (GQA), which keeps more than one key-value head and gets most of the quality back (Section 2.2).

## Key ideas

This page uses the paper's names: $H$ for the number of query heads and $G$ for the number of groups, which is also the number of key-value heads. The paper doesn't write out any equations, so the formulas below are mine, built from the Transformer page's notation ($d_k$ for head width, $d_{\text{model}}$ for model width, $W_i^Q$, $W_i^K$, $W_i^V$, $W^O$). The Transformer paper calls the number of heads $h$. Where the two papers' letters could clash, this page uses $H$.

### Share one key-value head per group of query heads

**Intuition.** In multi-head attention, every query head owns a private key head and a private value head, and all of them go in the cache. GQA keeps the $H$ query heads and shrinks the number of cached heads to $G$. Each group of $H/G$ neighboring query heads reads the same keys and values and asks them different questions.

**Math.** Section 2.2 says GQA "divides query heads into $G$ groups, each of which shares a single key head and value head", and writes GQA-$G$ for $G$ groups. Number the query heads $i = 0, \dots, H-1$, and let $g(i) = \lfloor i / (H/G) \rfloor$ be the group of head $i$. In the notation of the Transformer page, a GQA head is

$$
\operatorname{head}_i = \operatorname{Attention}\!\left(XW_i^Q,\; XW_{g(i)}^K,\; XW_{g(i)}^V\right),
$$

and the layer output is still $\operatorname{Concat}(\operatorname{head}_0, \dots, \operatorname{head}_{H-1})\,W^O$. There are $H$ matrices $W_i^Q$ but only $G$ matrices $W_g^K$ and $G$ matrices $W_g^V$. The paper doesn't say which query heads share. I assume neighboring heads do, which is what `np.repeat` produces. A different assignment, such as $i \bmod G$, is a different model.

Two special cases follow from the definition (Section 2.2):

- GQA-1 has one group and one key-value head. It is MQA.
- GQA-$H$ has $H$ groups, one per query head. It is MHA.

**Tiny example, computed by `scripts/gqa_example.py`.** Take $H = 4$ and $d_k = 2$. Then $g(i)$ maps the four query heads as follows:

| $G$ | Query head to group | Key-value heads cached | Cache reduction $H/G$ |
|---|---|---|---|
| 4 (MHA) | `[0, 1, 2, 3]` | 4 | 1x |
| 2 (GQA-2) | `[0, 0, 1, 1]` | 2 | 2x |
| 1 (MQA) | `[0, 0, 0, 0]` | 1 | 4x |

Per token and per layer, the cache holds $2 \cdot G \cdot d_k$ numbers: 16, 8, and 4 for $G = 4, 2, 1$. The script also checks that the $G = H$ case matches a plain per-head multi-head computation, to numerical precision.

For the 7B-scale configuration, `scripts/kv_cache_example.py` gives these cache sizes. The reduction is $H/G$ with $H = 32$, so GQA-8 gives 4x here, not 8x:

| Variant | Bytes per token | Per 4096-token sequence | Sequences that fit in 80 GiB after the weights |
|---|---|---|---|
| MHA, $G = 32$ | 524,288 (512 KiB) | 2 GiB | 33 |
| GQA-8 | 131,072 (128 KiB) | 0.5 GiB | 134 |
| MQA, $G = 1$ | 16,384 (16 KiB) | 0.0625 GiB | 1079 |

The $W^K$ and $W^V$ matrices shrink too. At $d_{\text{model}} = 4096$, one layer's $W^K$ and $W^V$ hold 33,554,432 weights under MHA, 8,388,608 under GQA-8, and 1,048,576 under MQA.

Exercise 1 asks you to compute the cache size and the batch that fits. Exercise 2 asks you to implement the attention layer.

### Why a middle setting helps

**Intuition.** MQA cuts the cache by $H$, and $H$ grows with model size. A bigger model has more heads, so MQA becomes a more aggressive cut exactly where models get large. GQA lets you pick $G$ and hold the proportional saving steady as the model grows.

**Math.** Section 2.2 argues three points:

- Going from MHA to MQA reduces $H$ key-value heads to one, which shrinks the cache and the data loaded by a factor of $H$. GQA-$G$ gives a factor of $H/G$ for any $G$ that divides $H$. Larger models "generally scale the number of heads", so MQA cuts more of the capacity as models grow. GQA "lets us keep the same proportional decrease in bandwidth and capacity as model size increases".
- Larger models suffer relatively less from attention's memory traffic, because the cache scales with model dimension while FLOPs and parameters scale with its square.
- Standard sharding for large models replicates the single MQA key-value head across the model partitions (Pope et al., 2022). GQA removes that waste.

Section 2.2 also notes that GQA doesn't apply to the encoder's self-attention. Encoder representations are computed in parallel, so memory bandwidth isn't the main bottleneck there. The paper applies GQA to the decoder self-attention and cross-attention layers (Section 3.1).

**Tiny example, computed by `scripts/table1_ratios.py`.** Table 1 reports time per sample, `Tinfer`, for T5 models on summarization, translation, and question answering. Using the paper's numbers:

| Model | Tinfer | Average score | Speedup over MHA-XXL | Score gap to MHA-XXL |
|---|---|---|---|---|
| MHA-Large | 0.37 | 46.0 | 4.08x | -1.2 |
| MHA-XXL | 1.51 | 47.2 | 1.00x | 0.0 |
| MQA-XXL | 0.24 | 46.6 | 6.29x | -0.6 |
| GQA-8-XXL | 0.28 | 47.1 | 5.39x | -0.1 |

GQA-8-XXL takes 1.17 times as long as MQA-XXL and loses 0.1 average points against MHA-XXL, while MQA-XXL loses 0.6. The MQA and GQA models are 5% uptrained from public T5.1.1 XXL checkpoints (Section 3.1). A larger MQA model already beats MHA-Large on both quality and speed (Section 3.2), and GQA-8 closes most of the remaining gap to MHA-XXL.

Section 3.3 and Figure 6 study the number of groups on T5-XXL. Going from 1 group to 8 adds a modest inference cost, and the cost rises as the group count approaches MHA. The authors chose 8 groups "as a favorable middle ground".

### Uptrain a checkpoint by mean pooling

**Intuition.** You can't just delete key-value heads from a trained model, because the remaining heads were trained alongside the others. The paper averages the heads in each group into one head, which keeps information from all of them, and then continues pre-training so the model can adapt to the new structure.

**Math.** Section 2.1 describes two steps: convert the checkpoint, then pre-train further for a proportion $\alpha$ of the original training steps, using the original recipe and data. For conversion, "the projection matrices for key and value heads are mean pooled into single projection matrices". For GQA, Section 2.2 pools "all the original heads within that group". In the notation above, with $G$ groups of $r = H/G$ heads each:

$$
\bar W_g^K = \frac{1}{r} \sum_{i \,:\, g(i) = g} W_i^K, \qquad
\bar W_g^V = \frac{1}{r} \sum_{i \,:\, g(i) = g} W_i^V.
$$

The paper gives these two averages in words and has no equation numbers for them. Only the key and value projections change. The query heads and $W^O$ are copied unchanged.

Section 3.3 compares conversion methods on T5-Large uptrained to MQA at $\alpha = 0.05$ (Figure 4). Mean pooling works best, then selecting a single head, then random initialization. The paper's explanation is that the order matches how much of the pre-trained model each method preserves. Figure 5 shows two further results for T5-XXL. GQA already performs reasonably right after conversion, but MQA needs uptraining to be useful. Both gain from 5% uptraining, with diminishing returns by 10%. For $\alpha = 0.05$, training took about 600 TPUv3 chip-days (Section 3.1).

**Tiny example, computed by `scripts/mean_pool_example.py`.** Take $d_{\text{model}} = 2$ and $H = 4$ heads of width $d_k = 1$, with

$$
W^K = \begin{bmatrix} 1 & 3 & 5 & 7 \\ 2 & 4 & 6 & 8 \end{bmatrix},
$$

where each column is one head. Pooling to $G = 2$ averages columns $(0, 1)$ and $(2, 3)$, which gives

$$
\bar W^K = \begin{bmatrix} 2 & 6 \\ 3 & 7 \end{bmatrix}.
$$

Pooling to $G = 1$ averages all four columns to $[4, 5]^\top$. For the input $x = [1, 1]$, the four original keys are $3, 7, 11, 15$, and the two pooled keys are $5$ and $13$, which are the averages of $(3, 7)$ and $(11, 15)$.

The conversion changes the function unless the heads already agree. With random weights ($d_{\text{model}} = 8$, $H = 4$, $d_k = 2$, $G = 2$, seed 0), the largest output difference between the original MHA layer and the converted GQA-2 layer is 0.4292. If the two heads in each group are already identical, the difference is 0 (printed as `0.0e+00`). The first number is why uptraining exists.

Exercise 3 asks you to implement the pooling.

<!-- exercises -->

## Pitfalls

- **Counting query heads in the cache.** The cache size depends on the number of key-value heads $G$. A 32-head model with $G = 8$ caches 8 heads, not 32. Computing the cache from the query-head count overstates GQA's memory by $H/G$.
- **Forgetting the factor of 2.** The cache stores a key and a value for every token, every layer, and every cached head.
- **Assuming "8x" is universal.** The saving is $H/G$. For $H = 32$ and $G = 8$ it is 4x. The paper's GQA-8 gives a larger factor only because its models have more heads.
- **Grouping interleaved heads.** Query heads $\{0, 1\}$ share a group, and $\{2, 3\}$ share the next. `np.tile` gives $\{0, 2\}$ and $\{1, 3\}$ instead. Both run without an error. They produce different models, so a converted checkpoint must use the same assignment its loader uses. The paper doesn't spell out the assignment.
- **Taking a sum, not a mean.** Summing the heads in a group multiplies the keys by $H/G$ and sharpens every softmax. Section 2.1 says mean pool.
- **Mean pooling the attention outputs.** The paper pools the projection matrices $W^K$ and $W^V$, not the per-head outputs. Pooling matrices gives one shared key and value, which is what the cache stores.
- **Expecting a converted checkpoint to match.** The converted model is not equal to the original. Figure 5 shows MQA performing poorly before uptraining. Plan for the 5% training run.
- **Applying GQA to the encoder.** The paper leaves encoder self-attention as MHA (Section 2.2) and applies GQA to decoder self-attention and cross-attention (Section 3.1).
- **Reading Table 1's timing as a general speedup.** `Tinfer` is the time per sample per TPUv4 chip on specific T5 models and summarization-style input and output lengths (Section 3.1). Your hardware, batch size, and context length change the ratios.

## Further reading

- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150), by Noam Shazeer, introduces multi-query attention and explains the bandwidth argument that GQA builds on.
- [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102), by Pope et al., analyzes how MQA and the KV cache affect serving cost and sharding.
- [Flaxformer](https://github.com/google/flaxformer) is the codebase that the paper's experiments use (Section 3.1).
