---
id: "pagedattention"
title: "Efficient Memory Management for Large Language Model Serving with PagedAttention"
authors: "Kwon et al."
year: 2023
venue: "SOSP 2023"
arxiv: "2309.06180"
url: "https://arxiv.org/abs/2309.06180"
track: inference
tier: core
est_hours: 5
status: full
summary: "A serving system can batch only as many requests as their key-value (KV) caches fit in GPU memory, and systems that store each request's cache in one contiguous chunk sized for the maximum length waste most of that memory. PagedAttention splits each request's cache into fixed-size blocks that can sit anywhere in memory, and a per-request block table maps logical blocks to physical ones, as an operating system maps virtual pages to frames. Because every block has the same size, waste stays inside each request's last block, and reference-counted blocks with copy-on-write let parallel samples, beam candidates, and shared prefixes use the same memory, which lets the vLLM system serve 2 to 4 times the throughput at the same latency."
related_course: "inference: vLLM"
prereqs:
  - id: orca
    reason: "PagedAttention keeps Orca's iteration-level scheduling and replaces its per-request reservation of the maximum number of K/V slots, which is the waste this paper measures."
background: [kv-cache, virtual-memory-and-paging, fragmentation]
course_links:
  - title: "Paged attention"
    url: "https://qwen.ifkash.dev/c/15-paged-attention"
  - title: "The KV cache"
    url: "https://qwen.ifkash.dev/c/09-the-kv-cache"
self_check:
  - q: 'A model has 40 layers and hidden size 5120, and stores keys and values in fp16. How many bytes of KV cache does one token need?'
    a: 'Keys and values are two vectors of 5120 numbers per layer, and fp16 takes 2 bytes, so a token needs $2 \times 5120 \times 40 \times 2 = 819{,}200$ bytes, which is 800 KiB. A 2048-token request needs about 1.56 GiB.'
    see: kv-cache
  - q: 'With a block size of 16 tokens, how many blocks does a 33-token sequence hold, and how many slots are wasted?'
    a: 'It holds $\lceil 33/16 \rceil = 3$ blocks, which is 48 slots, so 15 slots are wasted. Waste is always less than one block, and it sits in the last block.'
    see: fragmentation
  - q: 'Two samples share a 7-token prompt stored in blocks of 4 tokens. The first sample generates its first token. What happens to the prompt blocks?'
    a: 'The full block 0 stays shared. Block 1 has 3 tokens and a free slot, and both samples reference it, so the first sample gets a copy of block 1 and writes there; the second sample then owns the original alone and writes in place. This is copy-on-write, as when an OS forks a process.'
    see: virtual-memory-and-paging
---

## The problem

A decoder-only model generates one token per step, and each step reads the keys and values of every earlier token. Systems cache those vectors, the *KV cache*, instead of recomputing them (Section 2.2). The cache is large and it changes size. For OPT-13B, one token needs $2 \times 5120 \times 40 \times 2 = 819{,}200$ bytes, which the paper rounds to 800 KB, so a 2048-token request needs about 1.6 GB (Section 3, "Large KV cache"). On the 40 GB A100 of Figure 1, the 26 GB of weights leave 12 GB for the cache, which is 15.7K token slots (Table 1). Because decoding is memory-bound, throughput grows with the batch size, and the batch size is capped by how many caches fit (Section 1, Section 3).

The systems that came before, FasterTransformer and [Orca](/orca/), store one request's cache as one contiguous tensor, because most framework operators require it. The length of a request is not known in advance, so they reserve a chunk for the maximum length (Section 3.1). That wastes memory in three ways: slots that will hold a future token but sit idle until then (*reserved*), slots that the request never uses (*internal fragmentation*), and gaps between chunks (*external fragmentation*). The paper measures that only 20.4% to 38.2% of the cache memory holds token states in the baselines (Section 1, Figure 2).

**A concrete failure, computed by `scripts/fragmentation_example.py`.** Figure 3 shows two requests. Request A has a maximum length of 2048 tokens, a 7-token prompt, and one generated token, so it uses 8 slots and reserves 2040 more. Request B has a maximum of 512, a 3-token prompt, and one generated token, so it uses 4 slots and reserves 508 more. With 15,728 slots (12 GiB at 800 KiB per token) and a reservation of 2048 slots each, at most $\lfloor 15728 / 2048 \rfloor = 7$ requests run at once, however short they are. The 7 matches the Orca (Max) bar in Figure 13a, which averages 7.00 batched requests on ShareGPT.

Sharing has no home in this layout either. Parallel sampling and beam search produce several sequences that share a prompt or a long prefix, but each sequence sits in its own contiguous chunk, so the shared part is stored again for every sequence (Section 1, Section 3).

## Key ideas

The paper numbers four equations. Equation 1 is the autoregressive factorization, Equation 2 the query, key, and value projections, Equation 3 attention, and Equation 4 attention over blocks. The text of Section 2.1 says "Eq. 4" where Equation 3 is meant, so this page cites by the equation's content. The notation is the paper's: $B$ is the block size, $K_j$ and $V_j$ are the key and value blocks, $A_{ij}$ is the attention weights on block $j$, and $d$ is the head dimension. The paper writes the sequence position as $i$ and this page writes $n$ tokens for a single query at the last position.

### Fixed-size blocks and a block table

**Intuition.** Treat the KV cache like virtual memory. The paper says to "think of blocks as pages, tokens as bytes, and requests as processes" (Section 1). A request sees *logical* blocks, filled from left to right. The GPU holds *physical* blocks, all the same size. A *block table* per request maps one to the other, so neighboring logical blocks can sit anywhere. The system allocates a physical block only when the previous logical block is full, so it never reserves for the maximum length (Section 4.2, Section 4.3).

**Math.** PagedAttention splits the keys and values of a sequence into blocks of $B$ tokens: $K_j = (k_{(j-1)B+1}, \dots, k_{jB})$ and $V_j$ likewise (Section 4.1). Ordinary attention for one query (Section 2.1, Equations 2 and 3) is

$$
a_{ij} = \frac{\exp(q_i^\top k_j / \sqrt{d})}{\sum_{t=1}^{i} \exp(q_i^\top k_t / \sqrt{d})}, \qquad o_i = \sum_{j=1}^{i} a_{ij} v_j .
$$

PagedAttention computes the same result block by block (Section 4.1, Equation 4):

$$
A_{ij} = \frac{\exp(q_i^\top K_j / \sqrt{d})}{\sum_{t=1}^{\lceil i/B \rceil} \exp(q_i^\top K_t \mathbf{1} / \sqrt{d})}, \qquad o_i = \sum_{j=1}^{\lceil i/B \rceil} V_j A_{ij}^\top ,
$$

where $A_{ij} = (a_{i,(j-1)B+1}, \dots, a_{i,jB})$ is the row vector of weights on block $j$ and $\mathbf{1}$ is a column of ones, which sums the exponentials within a block. The denominator runs over all blocks, so the softmax is global. The block table only changes where the kernel reads $K_j$ and $V_j$ from, so the output is identical to dense attention and the model's accuracy is unchanged (Section 1).

**Tiny example, computed by `scripts/block_table_example.py`.** Figure 6 uses $B = 4$ and a 7-token prompt, with free physical blocks handed out in the order 7, 1, 3 to match the figure:

| Step | Block table (logical to physical, tokens filled) |
|---|---|
| After the prompt | 0 to 7 (4), 1 to 1 (3) |
| First decoding step | 0 to 7 (4), 1 to 1 (4) |
| Second decoding step | 0 to 7 (4), 1 to 1 (4), 2 to 3 (1) |

The first step fills the one reserved slot of block 1. At the second step the last block is full, so the manager takes a new physical block (3).

**Numeric check, computed by `scripts/paged_attention_example.py`.** The script stores a 10-token sequence with $d = 8$ and $B = 4$ in physical blocks 4, 0, and 5, so the last block holds 2 tokens. Equation 4 over those blocks differs from dense attention by at most $2.2 \times 10^{-16}$. Two common bugs show up clearly. A softmax taken inside each block, not across blocks, is off by 1.356 in the largest output entry. Attending to the two unfilled zero slots is off by 0.066.

**Cost.** The paper's fused kernels read blocks through the table and run 20% to 26% slower than FasterTransformer's attention kernel, and that only affects the attention operator (Section 7.1, Figure 18a). The block size trades kernel efficiency against waste. A small block reads inefficiently, and a large one wastes memory and reduces sharing (Section 7.2). The default is 16.

Exercise 1 asks you to write the allocator and the block table. Exercise 4 asks you to write Equation 4.

### Near-zero fragmentation

**Intuition.** Blocks are allocated on demand and are all the same size, so two kinds of waste go away. External fragmentation disappears because any free block fits any request. Reservation and most internal fragmentation disappear because a request holds blocks only for tokens it has already produced, and all its waste is inside the last block (Section 1, Section 4.3).

**Math.** A sequence of $n$ tokens holds $\lceil n/B \rceil$ blocks and wastes $B\lceil n/B \rceil - n$ slots, which is between 0 and $B - 1$. This is my arithmetic from the paper's argument in Section 4.3, not an equation in the paper. If the number of tokens in the last block is spread evenly, the mean waste is about $(B-1)/2$ per sequence. Contiguous allocation with maximum length $L_{\max}$ wastes $L_{\max} - n$ instead.

**Tiny example, computed by `scripts/fragmentation_example.py`.** Take Figure 3 again with $B = 16$. Request A uses 8 slots and holds 1 block, so it wastes 8 slots instead of 2040. Request B uses 4 slots and wastes 12 instead of 508.

**At scale.** The same script draws 20,000 requests with total lengths from a clipped lognormal distribution (median 300, mean 434.9, range 8 to 2048, seed 0). That distribution is the script's own, so only the shape carries over to the paper. It then admits requests in arrival order into 15,728 slots until one doesn't fit:

| Layout | Requests at once |
|---|---|
| Reserve 2048 slots each | 7 |
| Reserve the next power of two (like Orca Pow2) | 31 |
| Reserve the exact length (oracle) | 42 |
| Paged, $B = 16$ | 41 |
| Paged, $B = 256$ | 34 |

The mean internal waste per request is 7.43 slots for $B = 16$, close to $(B-1)/2 = 7.5$, against 1613.1 slots for the maximum reservation. Paging gets within one request of an oracle that knows every length in advance, and a block of 256 gives back much of that gain, which is the block-size trade-off of Section 7.2. The paper's measured result is that 96.3% of vLLM's cache memory holds token states (Figure 2), and that it batches 30.42 requests against Orca (Max)'s 7.00 on ShareGPT (Figure 13a).

Exercise 2 asks you to write these calculations.

### Copy-on-write sharing

**Intuition.** Once blocks are mapped through a table, two sequences can map a logical block to the same physical block, as two processes map the same page after a fork. A reference count on each physical block says how many tables point at it. A shared block is read-only in effect: the first sequence that must write into it copies it first (Section 4.4).

**Math.** The rule is this. When a sequence writes a token into its last block:

1. If the block is full, allocate a new block with reference count 1. No copy is needed.
1. If the block has a free slot and its count is 1, write in place.
1. If the block has a free slot and its count is greater than 1, allocate a new block, copy the contents, lower the old block's count by 1, and write into the new block.

The paper builds three decoding modes from three methods, `fork`, `append`, and `free` (Section 5.2). A block returns to the free list when its count reaches 0 (Section 4.4, beam search).

**Tiny example, computed by `scripts/block_table_example.py`.** Figure 8 shows parallel sampling with two samples and a 7-token prompt in blocks 7 and 1 ($B = 4$). After the fork, both blocks have count 2. When sample A1 writes its first token, block 1 has a free slot and count 2, so vLLM allocates block 3, copies block 1 into it, and lowers block 1 to count 1. When A2 writes, block 1 has count 1, so A2 writes in place. The result is three physical blocks, not four, and one block copy:

| Sample | Block table | Reference counts |
|---|---|---|
| A1 | 7, 3 | block 7: 2 |
| A2 | 7, 1 | block 1: 1, block 3: 1 |

**At scale, computed by `scripts/sharing_example.py`.** The script counts blocks with $B = 16$ and the page's own sizes, using the saving metric of Section 6.3: blocks saved divided by blocks without sharing.

| Prompt $P$ | Output $T$ per sample | Samples | Blocks without sharing | Blocks with sharing | Saving |
|---|---|---|---|---|---|
| 100 | 50 | 4 | 40 | 22 | 45.0% |
| 1000 | 50 | 4 | 264 | 78 | 70.5% |
| 1000 | 500 | 4 | 376 | 190 | 49.5% |

A long prompt and a short output save the most, and the saving falls as outputs grow, because the generated tokens are never shared. The paper measures 6.1% to 9.8% saving for parallel sampling on the Alpaca trace and 16.2% to 30.5% on ShareGPT (Section 6.3, Figure 15). These are lower than the table because real prompts are short against the outputs (Figure 11).

*Shared prefix.* A provider can keep the KV blocks of a common system prompt and map them into every request that uses it, with the last, partial block marked copy-on-write (Section 4.4). For the paper's 341-token five-shot prefix and $B = 16$, 21 blocks are full and shared and 5 tokens sit in a partial block. With 100 requests, sharing saves 2079 of 2200 prefix blocks. The paper reports 1.67 times the throughput of Orca (Oracle) with a one-shot prefix of 80 tokens, and 3.58 times with the five-shot prefix (Section 6.4, Figure 16).

*Beam search.* The sharing pattern changes at every step, like a process tree created by repeated forks. In the example of Figure 9 with $k = 4$, after one step the four candidates share blocks 0, 1, and 3, two share block 6, and two share block 7. If each candidate then has one new block, which is my reading of blocks 9 to 12 in the figure, the four tables hold 20 logical blocks over 9 physical blocks. The paper measures 37.6% to 55.2% saving for beam widths 2 to 6 on Alpaca and 44.3% to 66.3% on ShareGPT (Section 6.3, Figure 15). A copy is needed only when a candidate writes into a shared block, and then it copies one block. Earlier systems copied a large part of a candidate's cache when the beam reordered (Section 4.4).

Exercise 3 asks you to add `fork` and copy-on-write.

### Scheduling, preemption, and what the evaluation shows

**Block exhaustion.** Because the output length is unknown, the pool of physical blocks can run out. vLLM serves requests first come, first served and preempts the latest arrivals first. It evicts all the blocks of a sequence or none, because the whole cache is needed to run it, and it treats the sequences of one request as a group (Section 4.5). It recovers an evicted sequence in one of two ways: *swapping* the blocks to CPU memory, or *recomputation*, which feeds the prompt and the tokens generated so far through one prompt-phase iteration (Section 4.5). The swap space never needs to exceed the GPU's block pool. Swapping is expensive with small blocks, because it issues many small PCIe transfers, and recomputation costs the same for any block size, though its overhead is never above 20% of swapping's latency (Section 7.3, Figure 19).

**Distributed execution.** One KV cache manager in the central scheduler serves all workers under Megatron-LM style tensor parallelism. Each worker gets the same physical block IDs, but stores only the cache for its share of the attention heads (Section 4.6).

**Results.** The paper serves OPT-13B, 66B, and 175B and LLaMA-13B on A100 GPUs, with workloads synthesized from ShareGPT and Alpaca and Poisson arrivals, and compares against FasterTransformer and three Orca variants, which differ in how much they over-reserve: Max, Pow2, and Oracle (Section 6.1). The headline result is 2 to 4 times the throughput of the state of the art at similar latency (Abstract, Section 1). With basic sampling on ShareGPT, vLLM sustains 1.7 to 2.7 times the request rate of Orca (Oracle) and 2.7 to 8 times that of Orca (Max) (Section 6.2, Figure 12). The gain over Orca (Oracle) grows from 1.3 times with basic sampling to 2.3 times with beam width 6, on OPT-13B and Alpaca (Section 6.3). For OPT-175B on Alpaca the gain is small, because the large memory and the short sequences make the workload compute-bound (Section 6.2, Figure 12f).

<!-- exercises -->

## Pitfalls

- **Paging does not speed up one request.** It enlarges the batch that fits in memory, and the paper's gains come from that (Section 1, Section 6.2). The kernel itself is 20% to 26% slower than a contiguous one (Section 7.1). When the workload is compute-bound, as with OPT-175B on Alpaca, the gain nearly disappears (Section 6.2).
- **The paper's comparison with Orca uses the authors' reimplementation.** Orca is not public, so the authors implemented three variants that assume a buddy allocator (Section 6.1). The results show the memory effect of each reservation policy, not the original system's behavior.
- **The workloads are synthetic.** Lengths come from two datasets, and arrivals are a Poisson process (Section 6.1). The speedups depend on the length distribution, as the Alpaca and ShareGPT results differ.
- **Only the last block is ever copied or partly filled.** A block that is full never changes, so it can be shared safely. Copying a full block, or checking only the reference count and not whether the block has a free slot, adds copies the paper's design doesn't need (exercise 3).
- **A larger block is not free.** Bigger blocks read faster but waste $(B-1)/2$ slots per sequence on average and share less. The paper picks 16 after measuring it, and on short sequences like Alpaca, large blocks hurt (Section 7.2, exercise 2).
- **Do not attend to unfilled slots.** The last block holds garbage after position $n$. A kernel that reads the whole block or normalizes per block gives wrong output (exercise 4).
- **Freeing a shared block early corrupts another sequence.** `free` must lower the count and return the block only at 0. Likewise a fork that forgets to take references frees a block that another table still uses (exercises 1 and 3).
- **The idea applies where memory is the limit.** The paper says paging doesn't help when shapes are static, as in training, or when a model is compute-bound (Section 8).
- **The simulations on this page are simplified.** They ignore time, scheduling, and the cost of copies. They show the memory effect, not vLLM's throughput.

## Further reading

- The paper: [arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180). Section 4 is the core, and Figures 6 and 8 are worth redrawing by hand.
- The vLLM source code: [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm), linked from the abstract.
- The vLLM announcement post: [blog.vllm.ai/2023/06/20/vllm.html](https://blog.vllm.ai/2023/06/20/vllm.html).
- [Orca](/orca/), whose iteration-level scheduling vLLM builds on and whose maximum-length reservation this paper replaces.
- [Sarathi](/sarathi/), which chunks prefill so long prompts don't stall decoding in a paged engine, and [SGLang](/sglang/), which automates KV-cache reuse across requests.
