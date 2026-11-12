---
id: "sarathi"
title: "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve"
authors: "Agrawal et al."
year: 2024
venue: "OSDI 2024"
arxiv: "2403.02310"
url: "https://arxiv.org/abs/2403.02310"
track: inference
tier: extended
est_hours: 4
status: full
summary: "A prompt's prefill runs in one long, compute-bound iteration, and a serving system that runs it eagerly makes every ongoing decode wait, which users see as a multi-second generation stall. Sarathi-Serve caps each iteration at a token budget, always packs the running decodes first, and fills the rest of the budget with chunks of waiting prompts, so no decode stalls. The budget trades latency against prefill efficiency, and the paper reports 2.6 to 5.6 times higher serving capacity than vLLM under tail-latency limits."
related_course: "inference: scheduling"
prereqs:
  - id: pagedattention
    reason: "Sarathi-Serve is built on vLLM, so it reuses PagedAttention's block-table KV cache and extends the attention kernel to read earlier chunks' blocks."
  - id: roofline
    reason: "The whole design rests on the rule that an operation takes the larger of its math time and its memory time, which shows that a decode iteration has idle compute."
background: [prefill-vs-decode, stall-free-scheduling]
course_links:
  - title: "Continuous batching"
    url: "https://qwen.ifkash.dev/c/16-continuous-batching"
  - title: "Scheduling policy, routing, and the cluster"
    url: "https://qwen.ifkash.dev/c/16b-scheduling-policy-and-routing"
  - title: "A model of step time and goodput"
    url: "https://qwen.ifkash.dev/c/17a-performance-model"
  - title: "Benchmarks that mean something"
    url: "https://qwen.ifkash.dev/c/17-benchmarks"
self_check:
  - q: 'On a GPU with a ridge point of 153 FLOP/byte, a decode iteration for a 7B-parameter fp16 model runs 32 tokens through the weights. Is it compute-bound or memory-bound, and what happens to its time if you add 100 prefill tokens to the batch?'
    a: 'The intensity is about 32 FLOP/byte, far below 153, so the iteration is memory-bound and takes the time to read the weights. Adding 100 tokens gives 132 tokens, which is still below the ridge, so the time stays the same. That slack is the room that chunked prefill uses.'
    see: roofline
  - q: 'A prompt of 4096 tokens arrives while 32 requests are decoding. Why do those 32 users see a pause if the engine runs the whole prompt in one iteration, even though the engine also handles decodes?'
    a: 'The prefill iteration processes 4096 tokens and takes far longer than a 32-token decode iteration, and the decodes do not emit a token until it ends. The gap between two of their tokens grows by the whole prefill time, which is a generation stall.'
    see: prefill-vs-decode
  - q: 'Why does a stall-free scheduler admit decode tokens before prefill chunks, instead of splitting the budget evenly?'
    a: 'Decodes are the latency-sensitive work: each one is a user waiting for the next token. Packing them first means a decode never waits behind a prefill chunk, and the prefill chunks use only the budget that the decodes leave.'
    see: stall-free-scheduling
---

## The problem

Every request has two phases. *Prefill* processes the whole prompt in parallel and emits the first token. *Decode* then emits one token per iteration (Section 2.2). Prefill iterations are long and saturate the GPU's math units. Decode iterations are short and leave them mostly idle, because each request contributes one token (abstract, Section 3.1).

Batching helps decode a lot and prefill very little: batch size lifts decode throughput almost linearly, while prefill throughput saturates with a single request (Section 3.1, Figure 3, Takeaway-1). So a system wants large decode batches. The scheduler that builds them admits waiting prompts as early as it can. That is *prefill-prioritizing* scheduling, and vLLM and Orca use it (Section 2.5). It raises throughput and breaks latency, because a prefill iteration runs for as long as the prompt needs, and every running decode waits for it. The paper calls the pause a *generation stall* (Section 3.2). Figure 1a shows stalls of several seconds for Yi-34B on two A100 GPUs. The opposite policy, *decode-prioritizing* request-level batching as in FasterTransformer, never stalls a decode but wastes slots until the longest request in the batch finishes (Section 2.5, Algorithm 1). Lowering the maximum batch size, the earlier advice from [Orca](/orca/), does not fix it in the paper's tests: vLLM's capacity under a tight SLO barely changes across maximum batch sizes of 32, 64, and 128 (Section 5.2).

**A concrete failure, computed by `scripts/stall_free_sim.py`.** Take four requests that are each decoding with 12 tokens to go, and a 1000-token prompt that arrives after 3 iterations. Use a toy cost model: an iteration costs 10 ms up to 100 tokens and grows linearly after that. A decode-only iteration costs 10 ms. A prefill-first scheduler runs the prompt alone:

| Scheduler | Max time between tokens | TTFT of the new prompt |
|---|---|---|
| Prefill-first | 110 ms | 100 ms |
| Stall-free, budget 200 | 20 ms | 110 ms |

The four users wait 110 ms between two tokens, which is 11 times their normal 10 ms. The 1000-token prompt pays 10 ms more for its first token under the stall-free scheduler.

The paper adds a second failure for large models. With pipeline parallelism, micro-batches that mix a long prefill and a decode take very different times, so some pipeline stage waits on another (Section 3.3, Figure 8). For Falcon-180B, the paper measures one 4k-token prompt at about 1150 ms and a 32-request decode iteration at about 200 ms, so alternating them wastes about 950 ms (Section 3.3).

## Key ideas

The paper has no numbered equations. It gives its model in prose in Section 3.1 and its scheduler as Algorithm 3 in Section 4.2, so this page cites those. The notation is the paper's: $\tau$ is the token budget, $n_t$ is the number of tokens already in the batch, and $T_{math}$ and $T_{mem}$ are the math time and the memory time. The numeric examples use a model that this page defines, not the paper's measurements. They treat the linear layers only, use fp16 weights with $P = 7.24 \times 10^9$ parameters (Mistral-7B, which the paper evaluates), and use A100 80GB datasheet values: 312 TFLOP/s and 2.039 TB/s. The datasheet values and the parameter count are not from the paper.

### Decode leaves compute idle

**Intuition.** A decode iteration reads every weight to do one multiply-add per request. The read takes a fixed time, however few tokens the batch holds. While the read runs, the math units wait. You can put more tokens into the same iteration, and until the math time catches up with the read time, the iteration doesn't take longer.

**Math.** The paper approximates the time of an operation as

$$
T = \max(T_{math},\, T_{mem})
$$

(Section 3.1). The operation is *memory-bound* when $T_{math} < T_{mem}$. *Arithmetic intensity* is math operations per byte fetched. The best point is where the intensity equals the device's FLOPs-to-bandwidth ratio (Section 3.1). For a pass of $n$ tokens through the linear layers, $2Pn$ FLOPs run over $Pb$ weight bytes, where $b$ is bytes per parameter, so the intensity is $2n/b$ and does not depend on $P$. The ridge sits at $n^* = (\text{peak}/\text{bandwidth}) \cdot b / 2$ tokens. This page derives it; it is not a formula in the paper. Figure 5 plots the same trend for LLaMA2-70B on four A100s, and Figure 6 shows the time staying flat up to a threshold and growing linearly after it (Takeaway-2).

**Tiny example, computed by `scripts/roofline_cost.py`.**

| Tokens in the iteration | Intensity (FLOP/byte) | Time | Bound |
|---|---|---|---|
| 1 | 1 | 7.10 ms | memory |
| 32 | 32 | 7.10 ms | memory |
| 153 | 153 | 7.10 ms | memory |
| 512 | 512 | 23.76 ms | compute |
| 4096 | 4096 | 190.10 ms | compute |

The ridge is 153 FLOP/byte, so the model predicts that up to 153 tokens cost the 7.10 ms weight read. The paper notes that theory gives about 200 tokens on an A100 and that it observes about 500 to 600 for higher tensor-parallel degrees because of fixed overheads (Section 3.1, footnote 2). A decode batch of 32 plus a hybrid budget of 153 tokens costs 7.10 ms, the same as 32 decodes alone, and a budget of 512 costs 23.76 ms, which is 3.35 times the floor. Decode throughput rises from 141 tokens/s at batch 1 to 4506 tokens/s at batch 32, while prefill saturates at 21,547 tokens/s once you pass the ridge. That is the shape of Figure 3 in the paper.

Exercise 1 asks you to write this cost model.

### Chunked prefill

**Intuition.** A long prompt is too big to add to a decode batch in one piece, but you can split it. Run a few hundred of its tokens in each iteration, next to the decodes. A chunk of about 500 tokens still saturates the GPU, so you lose little, and an iteration stays short.

**Math.** Chunked prefill computes a prompt over several iterations, each with a subset of its tokens (Section 4.1). The sizes come from Algorithm 3 (lines 11 and 15), which asks for the largest chunk that fits the leftover budget. The paper gives two reasons that this works. A prefill of modest length already saturates the compute (Figure 4 shows it around 512 tokens). And real prompts have thousands of tokens: the median is 1730 for openchat_sharegpt4 and 7059 for arxiv_summarization (Table 2). The price is a repeated read. The attention for each chunk reads the KV cache of all earlier chunks of the same prompt, so with $N$ chunks the first chunk's KV is loaded $N-1$ times, the second $N-2$ times, and so on (Section 4.3). The computation doesn't change, and the paper finds that attention stays compute-bound even at small chunk sizes (Section 4.3).

**Tiny example, computed by `scripts/chunked_prefill.py`.** A 4096-token prompt with budget 512 splits into 8 chunks of 512, and the chunks re-read $512 \cdot (0 + 1 + \dots + 7) = 14336$ earlier KV tokens. With budget 1000 it splits into 5 chunks (the last holds 96 tokens), with 10,000 tokens re-read. At 131,072 KV bytes per token, which assumes Mistral-7B's shape (32 layers, 8 KV heads, head size 128, fp16), the 512-token case re-reads 1.88 GB. That costs 0.92 ms at the A100's bandwidth, against a 190.1 ms prefill. The weight read is the larger overhead when a chunk falls below the ridge:

| Chunk size | Chunks | Prefill time | Overhead vs. unchunked |
|---|---|---|---|
| 64 | 64 | 454.5 ms | 2.39x |
| 128 | 32 | 227.2 ms | 1.20x |
| 256 | 16 | 190.1 ms | 1.00x |
| 512 | 8 | 190.1 ms | 1.00x |

Each chunk pays the 7.10 ms weight read, so a chunk below 153 tokens wastes the difference. The model predicts no overhead from 256 tokens up, because it has no kernel-launch cost and no tile effects. The paper measures a moderate overhead of at most about 25% for a 512-token chunk on Yi-34B and almost none at 2048 (Section 5.4.1, Figure 14). Read the model's 1.00x as a floor on the overhead.

Exercise 2 asks you to split a prompt and count the re-reads.

### Stall-free batching

**Intuition.** Fill each iteration to a fixed token count. Put the decodes in first, because they are the users waiting. Put prompt chunks in the space that remains. Then each iteration costs about the same, and a decode never waits behind a long prefill.

**Math.** Algorithm 3 (Section 4.2) works as follows. Compute $\tau$ from the TBT service-level objective (SLO), and start each iteration with $n_t = 0$.

1. For each running request whose prefill is complete, add one token: $n_t \leftarrow n_t + 1$ (lines 6 to 8).
1. For each running request with an unfinished prefill, add the next chunk $c$, where `get_next_chunk_size` fits $c$ into $\tau - n_t$ (lines 9 to 12).
1. While a new request can be allocated and $n_t < \tau$, add its chunk $c$ if $c > 0$, and stop otherwise (lines 13 to 20).
1. Run the hybrid batch, drop finished requests, and reset $n_t = 0$ (lines 22 to 24).

The paper stresses the order: running decodes first, then a partial prefill, and only then new requests (Section 4.2). Decodes always enter the batch. If they alone exceed $\tau$, the chunks are 0.

**Tiny example, computed by `scripts/stall_free_sim.py`.** The failure case from earlier, with 4 decodes, a 1000-token prompt, and $\tau = 200$. Each iteration holds 4 decode tokens and 196 prompt tokens, so it costs 20 ms. The prompt needs 5 such iterations (980 tokens) and a sixth with the last 20 tokens, which costs 10 ms. The decodes emit a token in every iteration, with at most 20 ms between tokens. The prompt's first token arrives 110 ms after it started waiting. Varying the budget shows the dial:

| Budget $\tau$ | Max time between tokens | TTFT |
|---|---|---|
| 100 | 10 ms | 110 ms |
| 200 | 20 ms | 110 ms |
| 1000 | 100 ms | 110 ms |
| Prefill-first | 110 ms | 100 ms |

A budget of 1000 is one big iteration with the decodes inside, close to Orca's hybrid batch. The paper's Figure 9 shows that naive hybrid batching raises TBT up to 28.3 times over a decode-only batch, while chunking bounds it (Section 4.2).

The same script scales the example to the roofline model: 32 decodes, a 4096-token prompt, and a 7.10 ms decode iteration.

| Scheduler | Max time between tokens | TTFT |
|---|---|---|
| Prefill-first | 197.20 ms | 190.10 ms |
| Stall-free, budget 153 | 7.10 ms | 241.45 ms |
| Stall-free, budget 512 | 23.76 ms | 203.46 ms |
| Stall-free, budget 2048 | 95.05 ms | 197.20 ms |

A smaller budget means a lower worst-case TBT and a higher TTFT for the new prompt. Table 4 in the paper shows the same direction: chunked prefills alone raise the median TTFT, and hybrid batching alone leaves a high P99 TBT, while the combination lowers both (Section 5.4.2).

Exercises 3 and 4 ask you to write the batch builder and the simulator.

### Choosing the token budget

**Intuition.** The budget is a latency dial. A smaller budget lowers the worst iteration time and costs prefill efficiency. A larger one is the opposite. Pick the largest budget whose iteration time still meets your TBT target.

**Math.** Section 4.3 names the competing factors. A smaller $\tau$ lowers TBT but adds chunking overhead from lower utilization and repeated KV reads. The paper recommends one-time profiling of batches with different token counts, then taking the largest count that meets the TBT SLO. Two further effects matter. *Tile quantization* means the matmul does extra work when a dimension isn't a multiple of the tile size, and the paper reports that a chunk of 257 tokens can cost 32% more prefill time than 256 (Section 4.3). With pipeline parallelism, larger chunks raise the variation between micro-batches and add bubbles, so the paper lowers one budget for that reason (Section 5.1). The paper uses Vidur, a profiler and simulator, to choose a budget (Section 4.3).

**Tiny example, computed by `scripts/roofline_cost.py`.** The paper defines the strict SLO as 5 times the time of a decode iteration without prefill interference, and the relaxed one as 25 times (Section 5.1). Under the roofline model, a 5 times limit is 35.51 ms, and the largest budget under it is 765 tokens. The paper's strict-SLO budget is 512 and its relaxed budget is 2048 (1536 for LLaMA2-70B), which fall in the same range (Section 5.1). The model's 765 is an estimate. It ignores attention, the fixed overheads, and the observed 500 to 600 token knee.

Exercise 1 includes the `token_budget` function.

### What the evaluation shows

The paper compares Sarathi-Serve with vLLM and Orca on four models (Table 1): Mistral-7B on one A100, Yi-34B on two A100s, LLaMA2-70B on eight A40s, and Falcon-180B on eight A100s over two nodes. The traces use request lengths from openchat_sharegpt4 and arxiv_summarization with Poisson arrivals (Section 5). *Capacity* is the maximum queries per second that meets a P99 TBT target, with a limit of 2 seconds on the median scheduling delay (Sections 2.4 and 5.1). The headline gains are as follows:

- **Capacity.** Up to 3.7 times vLLM's capacity for Yi-34B under the strict SLO on openchat_sharegpt4, and 2.6 times for Mistral-7B (abstract). With pipeline parallelism, the paper reports up to 4.3 times vLLM's capacity for LLaMA2-70B and up to 5.6 times for Falcon-180B (Section 5.1, Figure 11, abstract).
- **Why vLLM stops scaling.** Its capacity stays nearly the same for maximum batch sizes of 32, 64, and 128 under a tight SLO, because generation stalls cap it (Section 5.2, Figure 12).
- **The budget is a dial.** With a budget of 512, Sarathi-Serve reaches 3.5 times vLLM's capacity at a 100 ms SLO on Mistral-7B. With 2048, it reaches 1.65 times vLLM's capacity at a 1 s SLO on Yi-34B (Section 5.2).
- **Pipelines.** Uniform hybrid batches cut the bubbles. On Falcon-180B, Sarathi-Serve gets 3.6 times vLLM's hybrid-parallel capacity under the strict SLO, and tensor parallelism across nodes more than doubles median TBT (Section 5.3, Figure 13).

<!-- exercises -->

## Pitfalls

- **The speedups are capacity, not latency.** A figure such as 3.7 times is the load a system sustains under a P99 TBT limit, on specific models and traces. It is not the ratio of latencies at one load (Section 5.1).
- **The budget isn't free.** The paper's own ablation shows chunked prefills alone raise the median TTFT, for example from 0.53 s with hybrid batching alone to 1.04 s on openchat_sharegpt4 (Section 5.4.2, Table 4). You trade first-token latency for steadier tokens.
- **Chunking doesn't change the arithmetic.** The attention FLOPs are the same, but the KV reads repeat and each iteration re-reads all weights. A chunk below the ridge wastes the difference, so a budget of 64 on this page's model costs 2.39 times the unchunked prefill.
- **Decodes can exceed the budget.** Algorithm 3 packs every running decode first, so with a very large decode batch, $n_t$ can pass $\tau$ and the prefill chunks are 0 (exercise 3). Size the maximum batch with the budget in mind.
- **Don't chunk to a size the GPU dislikes.** The paper reports that 257 tokens can run 32% slower than 256 because of tile quantization (Section 4.3). Round budgets to a multiple of the tile size, and profile before you trust a number.
- **Stall-free is not disaggregation.** Sarathi-Serve keeps prefill and decode on the same GPUs and removes the interference by scheduling. Splitting them onto separate replicas removes it by placement, at the cost of moving the KV cache (Section 6). [DistServe](/distserve/) takes that route.
- **The simulators here are simplified.** They use a one-line roofline, charge nothing for attention, and have one arriving prompt. They show the scheduling effect and not the paper's absolute numbers. Compare their direction with Table 4 and Figure 9, not their values.

## Further reading

- The paper's PDF: [arxiv.org/abs/2403.02310](https://arxiv.org/abs/2403.02310). Section 3 is the motivation, and Algorithm 3 in Section 4.2 is short.
- The official code, built on vLLM: [github.com/microsoft/sarathi-serve](https://github.com/microsoft/sarathi-serve).
- [Efficient memory management for LLM serving with PagedAttention](/pagedattention/), whose block table the chunked-prefill kernel reads.
- [Orca](/orca/), which introduced the iteration-level scheduling and hybrid batches that Sarathi-Serve builds on.
- [DistServe](/distserve/), which removes prefill and decode interference by putting them on separate GPUs.
