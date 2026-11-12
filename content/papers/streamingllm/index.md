---
id: "streamingllm"
title: "Efficient Streaming Language Models with Attention Sinks"
authors: "Xiao et al."
year: 2023
venue: "ICLR 2024"
arxiv: "2309.17453"
url: "https://arxiv.org/abs/2309.17453"
track: inference
tier: extended
est_hours: 3
status: full
summary: "A chat server that never stops needs a KV cache that stops growing, but keeping only the most recent tokens makes perplexity explode the moment the first tokens are evicted. The paper explains this with softmax: a query with nothing relevant to attend to still has to spread weight that sums to 1, models learn to park it on the first few tokens, and those tokens act as attention sinks whose removal changes every later attention output. StreamingLLM keeps the first four tokens plus a rolling window of recent tokens and numbers positions inside the cache, which lets models trained on short texts run stably on streams of 4 million tokens, and a learnable sink token added in pre-training makes one sink enough."
related_course: "inference: KV economy"
prereqs:
  - id: transformer
    reason: "StreamingLLM changes which keys and values a decoder layer can attend to, so you need the scaled dot-product attention and causal masking of that paper."
  - id: rope
    reason: "The paper caches keys before the rotary transform and rotates them by their place in the cache, so you need to know how rotary embeddings turn positions into relative offsets."
  - id: pagedattention
    reason: "The paper's problem is a KV cache that grows with every token, and that paper shows what the cache costs and why serving systems care."
background: [kv-cache, softmax, positional-information, softmax-numerics]
course_links:
  - title: "KV economy: MLA, windows, sinks, and eviction"
    url: "https://qwen.ifkash.dev/c/07a-kv-economy-attention-variants"
  - title: "The KV cache"
    url: "https://qwen.ifkash.dev/c/09-the-kv-cache"
self_check:
  - q: 'A cache keeps 4 sink tokens and the 3 most recent tokens. The model has generated tokens 0 to 8 and is about to decode token 9. Which tokens are cached, which positions does the model use, and which tokens are gone?'
    a: 'The cache holds tokens 0, 1, 2, 3, 6, 7, and 8. Tokens 4 and 5 are evicted. Including token 9, the rotary positions are 0 to 7, not the original 0, 1, 2, 3, 6, 7, 8, 9 (Section 3.2, Figure 4).'
    see: rope
  - q: 'A softmax over 8 logits has one logit of 6 and seven logits near 0. About how much weight does the large logit get, and what happens to each other weight if you delete that token and renormalize?'
    a: 'The large logit gets about 98% of the weight, since $e^{6} \approx 403$ against about 7 for the rest. Deleting it removes the 98%, so each remaining weight grows by about $1 / (1 - 0.98) \approx 54$ times. Attention outputs that were nearly zero become ordinary averages of the remaining values.'
    see: softmax
  - q: 'Why does a cache of the most recent 1024 tokens use a constant amount of memory, while the full cache of a 4-million-token stream does not?'
    a: 'The window cache holds at most `sinks + window` tokens whatever the text length, so its size is fixed. A full cache stores a key and a value in every layer for every token, so it grows linearly with the stream.'
    see: kv-cache
---

## The problem

A chat assistant or a coding agent can run for days. Two things stop a language model from serving such a stream (Section 1). The first is memory. The decoder stores the keys and values of every earlier token, the *KV cache*, and the cache grows with every token. The second is length. A model trained on texts of length $L$ doesn't produce good text far beyond $L$.

The obvious fix is *window attention* (Figure 1b). The cache holds only the most recent $L$ tokens, and each new token evicts the oldest one. Memory is constant, and nothing in the method looks wrong. It fails abruptly anyway: perplexity is fine until the text grows past the cache size, and then it spikes (Section 3.1, Figure 3).

**A concrete failure, from the paper.** Table 1 evaluates Llama-2-13B on the first book of PG-19, a 65K-token text, with a cache of 1024 tokens. The notation $x + y$ means $x$ initial tokens plus the $y$ most recent tokens. Window attention, written $0 + 1024$, reaches a perplexity of 5158.07. Keeping the first four tokens and the 1020 most recent, $4 + 1020$, reaches 5.40. The two caches differ by four tokens, and perplexity differs by a factor of about 955. A perplexity of 5158 means the model assigns almost no probability to the correct next token.

Two cheaper-looking alternatives also fail. Recomputing the window from scratch for every new token (the *sliding window with re-computation* baseline) gives good perplexity, but each step reprocesses the whole window, so latency grows quadratically with the cache size (Section 4.5, Figure 10). Dense attention over the whole stream needs a cache that grows with $T$ and runs on texts longer than the model was trained on (Figure 1a).

## Key ideas

The paper uses three pieces of notation on this page. $L$ is the pre-training length and the window size, $T$ is the index of the token being predicted with $T \gg L$ (Figure 1), and $N$ is the number of tokens in a softmax sum. A cache configuration $x + y$ has $x$ sink tokens and $y$ window tokens.

### Attention sinks: softmax has to put its weight somewhere

**Intuition.** Attention weights are a softmax, so they sum to 1 whether or not any earlier token is relevant. A query that has nothing to look up still has to distribute weight. Figure 2 shows what Llama-2-7B does about it: apart from the first two layers, which attend locally, nearly every layer and head puts a large share of its attention on the first token (Section 3.1). The first token is visible to every later token during training, so it is the one place that every query can safely park its weight. The paper calls such tokens *attention sinks*.

**Math.** The paper writes the attention softmax as Equation 1,

$$
\mathrm{SoftMax}(x)_i = \frac{e^{x_i}}{e^{x_1} + \sum_{j=2}^{N} e^{x_j}}, \qquad x_1 \gg x_j \text{ for } j \in \{2, \dots, N\},
$$

where $x_j$ is the attention logit for token $j$ and $x_1$ is the logit of the initial token (Section 3.1). The large $x_1$ dominates the denominator, so the other tokens each get a small weight. Evicting token 1 removes $e^{x_1}$ from the denominator. Every remaining weight grows by the same factor, and the sum of the weights is still 1, so the layer now attends to tokens that it was ignoring on purpose. The model never saw this distribution in training.

The paper checks whether the meaning of the first tokens matters. It mostly doesn't, and the absolute position does (Section 3.1). Replacing the first four tokens with four line-break tokens gives 5.60, close to the 5.40 of the real first four (Table 1, four line-break tokens plus 1020 recent tokens).

**Tiny example, computed by `scripts/sink_softmax.py`.** A toy query has eight logits: the sink has 6.0 and the other seven are within 0.3 of zero, so nothing is relevant. The page picked these numbers, and they aren't from the paper. The sink gets 0.9815 of the weight, and the largest weight on any other token is 0.0033. The sink's value vector is almost zero, so the output has a norm of 0.0102, which is about "no information". Evict the sink and renormalize, and each remaining weight grows by about 54 times. The weights are now between 0.107 and 0.177, and the output norm is 0.715. The two outputs are 0.706 apart, which is large against the 0.0102 that the layer used to produce. The sink was working as a no-op.

Exercise 3 asks you to write this computation.

### Sink tokens plus a rolling window

**Intuition.** If the first tokens are the sink, keep them. StreamingLLM caches the keys and values of a few initial tokens and, next to them, a rolling window of the most recent tokens. Everything in between is evicted (Section 3.2, Figure 4). Memory is constant, because the cache holds $x + y$ tokens at every step, and no fine-tuning is needed, because the model sees the sinks it was trained with.

**Math.** After $n$ tokens, with $s$ sinks and a window of $w$, the cache holds the index set

$$
C_n = \{0, \dots, s - 1\} \cup \{n - w, \dots, n - 1\} \quad \text{when } n > s + w,
$$

and every index otherwise. The paper's notation for this is $x + y$ with $x = s$ and $y = w$. The index sets are this page's notation, not an equation in the paper. The paper keeps four sink tokens by default, and Table 2 is the evidence. On the 400K-token concatenated PG-19 text, with a total cache of 4096 for Llama-2-7B and 2048 for the others, one sink isn't enough, and four is about as good as eight:

| Model | Cache | $0 + y$ | $1 + y$ | $2 + y$ | $4 + y$ | $8 + y$ |
|---|---|---|---|---|---|---|
| Llama-2-7B | 4096 | 3359.95 | 11.88 | 10.51 | 9.59 | 9.54 |
| Falcon-7B | 2048 | 17.90 | 12.12 | 12.12 | 12.12 | 12.12 |
| MPT-7B | 2048 | 460.29 | 14.99 | 15.00 | 14.99 | 14.98 |
| Pythia-12B | 2048 | 21.62 | 11.95 | 12.09 | 12.09 | 12.02 |

(Table 2, perplexity.) The paper attributes the need for several sinks to the lack of a dedicated sink token in pre-training (Section 3.3). A bigger window doesn't reliably help either. For Llama-2-7B, $4 + 2044$ gives 9.08 and $4 + 4092$ gives 9.59, and for MPT-7B perplexity rises from 14.12 at $4 + 252$ to 14.99 at $4 + 2044$ (Table 6). The paper tests Llama-2, Falcon, Pythia, and MPT families on streams of up to about 4 million tokens (Section 4.1, Figure 5).

**Tiny example, computed by `scripts/cache_indices.py`.** With 4 sinks and a window of 8, the cache holds all tokens until $n = 12$. At $n = 13$ it holds tokens 0 to 3 and 5 to 12, and token 4 is the first eviction. At $n = 20$ it holds 0 to 3 and 12 to 19, and 8 tokens are evicted. At $n = 100$ it holds 0 to 3 and 92 to 99, and 88 tokens are evicted. The size stays at 12 slots.

Exercise 1 asks you to write this cache.

### Positions come from the cache, not from the text

**Intuition.** The model learned to expect token positions $0, 1, \dots, L - 1$ without gaps. A cache with a hole in the middle breaks that expectation if each token keeps its original position. StreamingLLM numbers tokens by their place *in the cache* (Section 3.2). If tokens 0 to 3 and 6 to 8 are cached and token 9 is decoded, the positions are 0 to 7, as if the text had never had tokens 4 and 5 (Figure 4).

**Math.** With RoPE, the model caches keys *before* the rotary transform. At each decoding step it rotates the cached keys and the new query by their cache positions $0, 1, \dots, m - 1$, where $m$ is the number of rows including the current token (Section 3.2). The score between the query in slot $m - 1$ and the key in slot $j$ then depends only on the slot distance $m - 1 - j$, which is the relative-position property of [RoPE](/rope/). The paper says that ALiBi works the same way, with a contiguous linear bias in place of the jumping one that the original positions would give.

**Tiny example, computed by `scripts/positions_in_cache.py`.** Take the cache of Figure 4 plus token 9. The original positions are $[0, 1, 2, 3, 6, 7, 8, 9]$ and the cache positions are $[0, 1, \dots, 7]$. Token 9 is 9 positions from the sink in the text and 7 in the cache. Its distance to token 6 stays 3. The gap between tokens 3 and 6 shrinks from 3 to 1. Only the distances across the evicted region change, and the distance from the newest tokens to the sink is always at most the cache size. For one rotary pair with angle 1 per position and $q = k = [1, 0]$, the score for the sink is $\cos 7 = 0.7539$ at the cache distance. It is $\cos 9 = -0.9111$ at the original distance, and the model has seen only the first kind during training.

Exercise 2 asks you to write the position assignment and the scores.

### A learnable sink token in pre-training

**Intuition.** The sink is a side effect of softmax, and models waste real tokens on it. If a model is pre-trained with a dedicated placeholder token at the start of every sample, the model can use that one token as its sink, and the cache needs one sink token (Section 3.3).

**Math.** The paper compares three 160M-parameter models trained from scratch for 143,000 steps (Section 3.3): a vanilla model, a model that uses $\mathrm{SoftMax}_1$ (the "zero sink"), and a model with a learnable sink token prepended to every sample. Equation 2 is

$$
\mathrm{SoftMax}_1(x)_i = \frac{e^{x_i}}{1 + \sum_{j=1}^{N} e^{x_j}},
$$

which adds a 1 to the denominator, like a key with logit 0 and a zero value, so the weights can sum to less than 1 (Section 3.3, Equation 2). Table 3 reports perplexity on the first PG-19 sample for each cache configuration:

| Model | $0 + 1024$ | $1 + 1023$ | $2 + 1022$ | $4 + 1020$ |
|---|---|---|---|---|
| Vanilla | 27.87 | 18.49 | 18.05 | 18.05 |
| Zero sink | 29214 | 19.90 | 18.27 | 18.01 |
| Learnable sink | 1235 | 18.01 | 18.01 | 18.02 |

The learnable-sink model reaches 18.01 with one sink token, while the vanilla model needs two initial tokens to reach 18.05. Without any initial token, the learnable-sink model scores 1235 and the zero-sink model 29214, so $\mathrm{SoftMax}_1$ alone doesn't remove the need for initial tokens: it needs four to reach 18.01. Table 4 reports that the sink-token model matches or slightly exceeds the vanilla model on seven zero-shot benchmarks, so the extra token doesn't cost accuracy (Section 3.3). Figure 7 shows its attention concentrated on the sink token.

**Tiny example, computed by `scripts/sink_softmax.py`.** On the toy logits of the first idea without the sink, a plain softmax over the seven remaining logits sums to 1.0 by construction, and $\mathrm{SoftMax}_1$ sums to 0.884, leaving 0.116 for the implicit zero key. With large logits $[6, 5, 4]$, $\mathrm{SoftMax}_1$ sums to 0.9984 and behaves like a softmax.

Exercise 3 includes $\mathrm{SoftMax}_1$.

### What the cache saves, and what it doesn't

**Intuition.** A full cache grows with the text. A sink-plus-window cache is constant. The sliding window with re-computation also uses constant cache memory but pays for it in time, because it re-encodes the window for every new token.

**Math.** One token adds $2 \cdot n_{\text{layers}} \cdot n_{\text{kv heads}} \cdot d_{\text{head}} \cdot b$ bytes, where $b$ is the bytes per value (the factor 2 counts a key and a value). This is this page's arithmetic, not an equation in the paper. For recomputation, one new token with window $L$ means $L$ tokens pass through the model, and attention over them costs on the order of $L^2$. The cached version attends one query to $L$ keys. The paper measures the speedup on an NVIDIA A6000 with Llama-2-7B and 13B, and reports that StreamingLLM's per-token latency grows linearly with the cache size and the recomputation baseline's grows quadratically, with a speedup of up to 22.2 times at similar memory (Section 4.5, Figure 10).

**Tiny example, computed by `scripts/cache_memory.py`.** The page assumes Llama-2-7B geometry: 32 layers, 32 KV heads of width 128, fp16. One token takes 524,288 bytes, which is 512 KiB. A cache of $4 + 2044$ slots takes 1.00 GiB at any text length above 2048. A full cache at 4,096 tokens takes 2.00 GiB, at 20,000 tokens 9.77 GiB, at 400,000 tokens 195.31 GiB, and at 4 million tokens 1953.12 GiB, which is about 1953 times the streaming cache. For the recomputation baseline with $L = 4096$, the script counts 8,390,656 attention pairs per new token against 4,096 for the cached version, a ratio of 2048.5. The paper's measured speedup is up to 22.2 times, much lower than this count, because the attention score count is only part of the work in a step.

Exercise 4 asks you to write the memory counts.

<!-- exercises -->

## Pitfalls

- **StreamingLLM doesn't extend the context.** The model can read and write indefinitely, but it only sees what is in the cache. The paper says it doesn't add long-term memory (Appendix A). On StreamEval, accuracy falls to 0% once the distance between a question and the answer's source exceeds the cache size (Appendix C, Table 7).
- **Evicted tokens are gone.** On LongBench, the $4 + 3496$ configuration does worse than a plain truncation baseline, for example 11.6 against 18.7 on NarrativeQA (Appendix D, Table 8). The paper recommends the method for short-term-memory tasks such as daily dialogue and short-document question answering, not for long-document question answering or summarization.
- **A bigger window isn't a better window.** Table 6 shows perplexity that rises with the window for MPT-7B and doesn't improve past a point for Llama-2. Pick the window from the memory budget and the task, not from the hope of lower perplexity.
- **The sinks must be the first tokens of the stream, not any four tokens.** The mechanism depends on the first tokens being visible to every later token during training. Table 1 shows that other content in those slots works (line breaks give 5.60), but the slots must hold the initial positions. Evicting them is the failure.
- **Original positions break the cache.** Using the original indices for the rotary transform gives the model a gap it never saw in training, and the sink sits at the wrong distance (exercise 2). Cache keys before the rotary transform so you can rotate them again at every step.
- **Re-rotating costs time.** The cache stores un-rotated keys, so every step rotates all cached keys again. This adds work to every step, and the paper doesn't report its cost separately.
- **A cache that drops a row but keeps its softmax weight is wrong.** Evicting a token means removing its logit from the denominator, not zeroing its weight after the softmax (exercise 3).
- **The page's toy numbers are illustrative.** The sink logit of 6.0 and the near-zero values in `scripts/sink_softmax.py` are this page's choices. The paper's evidence is the perplexity tables and attention maps (Figure 2, Table 1).
- **The learnable sink needs pre-training.** The paper trains small 160M-parameter models from scratch. The one-sink result comes from those models, so treat it as a pre-training result.

## Further reading

- The paper: [arxiv.org/abs/2309.17453](https://arxiv.org/abs/2309.17453). Figure 2 (attention maps), Figure 4 (the cache), and Tables 1 to 3 carry the argument.
- The authors' code: [github.com/mit-han-lab/streaming-llm](https://github.com/mit-han-lab/streaming-llm).
- [PagedAttention](/pagedattention/), which manages the cache that StreamingLLM bounds, and [RoPE](/rope/), whose relative-position property makes cache-relative positions work.
- [GQA](/gqa/), another way to make the cache smaller: it shrinks the cost of each token where StreamingLLM limits how many tokens count.
