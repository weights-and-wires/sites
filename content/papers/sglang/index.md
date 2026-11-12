---
id: "sglang"
title: "SGLang: Efficient Execution of Structured Language Model Programs"
authors: "Zheng et al."
year: 2023
venue: "NeurIPS 2024"
arxiv: "2312.07104"
url: "https://arxiv.org/abs/2312.07104"
track: inference
tier: extended
est_hours: 5
status: full
summary: "Real language model programs send many calls that start with the same text, such as a system prompt, few-shot examples, or an earlier chat turn, and an engine that frees each request's key-value cache when the request ends recomputes that text every time. SGLang keeps the caches of finished requests in a radix tree over token ids, finds the longest cached prefix of each new request, and evicts least recently used leaves when memory runs short. It also schedules waiting requests by how much of their prefix is cached, and it decodes structured output faster by jumping over the characters that a compressed finite-state machine already forces, which gives up to 6.4 times higher throughput in the paper's tests."
related_course: "inference: prefix caching"
prereqs:
  - id: pagedattention
    reason: "RadixAttention stores its cached keys and values in the paged, non-contiguous layout from PagedAttention and adds an index over requests on top of it."
background: [radix-tree, prefix-caching]
course_links:
  - title: "Prefix caching and radix trees"
    url: "https://qwen.ifkash.dev/c/15a-prefix-caching-and-radix-trees"
  - title: "Constrained decoding"
    url: "https://qwen.ifkash.dev/c/11a-constrained-decoding"
  - title: "Scheduling policy, routing, and the cluster"
    url: "https://qwen.ifkash.dev/c/16b-scheduling-policy-and-routing"
  - title: "Paged attention"
    url: "https://qwen.ifkash.dev/c/15-paged-attention"
self_check:
  - q: 'A radix tree stores the token sequence $[1, 2, 3, 4]$ as one edge. You insert $[1, 2, 5]$. Which nodes does the tree hold afterwards, and how many tokens did the insertion add?'
    a: 'The edge splits after token 2. The tree holds a shared node $(1, 2)$ with two children, $(3, 4)$ and $(5)$, so it has 3 nodes and 5 stored tokens. The insertion added 1 token, because $[1, 2]$ was already cached.'
    see: radix-tree
  - q: 'The cache is full and holds a shared node $(1, 2)$ with two leaves $(3, 4)$ and $(5, 6)$, and a request that uses $(5, 6)$ is still running. Which nodes can the cache evict, and in what order?'
    a: 'Only leaves with a reference count of 0 can go, so the leaf $(3, 4)$ goes first. The running request holds a reference on $(5, 6)$ and on its ancestor $(1, 2)$, so neither can go. Once the request finishes, $(5, 6)$ goes, and then $(1, 2)$ becomes a leaf and can go too.'
    see: prefix-caching
  - q: 'PagedAttention already shares blocks between sequences through block tables. What does RadixAttention add that a block table does not provide?'
    a: 'A block table maps the logical blocks of one sequence to physical blocks, and it only shares what the system was told to share, such as the samples of one request. RadixAttention adds an index over all cached token sequences, so a new request finds the longest prefix that any earlier request left behind, including requests that already finished.'
    see: pagedattention
---

## The problem

A language model *program* is code that calls a model several times and passes text between the calls. The paper gives examples such as few-shot prompting, self-consistency, ReAct agents, tree-of-thought, and multi-turn chat (Section 1). Two properties make these programs expensive on a naive engine. Many calls start with the same tokens, and many calls must produce output in a fixed format such as JSON.

An engine like [vLLM](/pagedattention/) frees the key-value (KV) cache of a request when the request ends. The next call that begins with the same system prompt then runs prefill on that prompt again. Prefill is compute-bound, so repeated prefill wastes the most expensive resource.

**A concrete failure, computed by `scripts/radix_example.py`.** Four requests arrive. Two chats share a 3-token system prompt and the first chat sends a second turn that repeats its first turn. A few-shot query shares nothing with them. The requests hold 21 tokens in total, and only 13 of those tokens are distinct. A system that keeps no cache computes 21 tokens of KV. A system that keeps the caches of earlier requests and finds them by prefix computes 13. The gap grows with the length of the shared text.

A flat cache is not enough. A system that stores one cache per prompt text only hits when a new prompt equals an old one, so the second chat above, which shares 3 tokens with the first and then differs, gets no hit. A hand-written rule such as "cache the system prompt" covers one known shape and misses few-shot examples, chat history, and the branches of a tree-of-thought search (Section 3, Figure 3). The paper wants the engine to find the shared prefix automatically and to keep that prefix across requests.

The second problem is structured output. To force a regex, systems mask the invalid tokens at each step, so each decoding step runs the whole model to produce a token that the format already determined. A constant such as `{"summary": ` spans several tokens, and each one costs a forward pass (Section 4).

The paper answers with a language and a runtime (Section 1, Figure 1). The frontend is an embedded Python language with primitives such as `gen` (with a `regex` option), `select`, `fork`, and `join` (Section 2). The runtime has three ideas. RadixAttention reuses the KV cache, a compressed finite state machine speeds up constrained decoding, and an API speculative execution step reduces the cost of calls to hosted models (Sections 3 to 5). This page covers the first two in depth. The frontend and the API speculation are context only.

## Key ideas

The paper gives few equations. The notation on this page is its own. A request is a sequence of token ids $x = (x_1, \dots, x_n)$. The *cache hit rate* is the fraction of prompt tokens that are already cached (Section 6.2):

$$
\text{hit rate} = \frac{\text{cached prompt tokens}}{\text{total prompt tokens}}.
$$

### RadixAttention: a radix tree over cached prefixes

**Intuition.** Keep the KV cache of finished requests instead of freeing it, and index it so that a new request can find the longest prefix it shares with anything in the cache. A *trie* does that with one node per token. A *radix tree* is its space-efficient form, in which an edge carries a whole sequence of tokens and a node exists only where sequences branch or end (Section 3). The tree maps token sequences to KV cache tensors, which sit in a paged, non-contiguous layout. The paper uses pages of one token each, so a node's tensors are as long as its edge (Section 3).

**Math.** Let $S$ be the set of sequences the tree stores, and let $P(S)$ be the set of their non-empty prefixes. Four facts define the data structure:

1. The tree holds exactly $|P(S)|$ tokens, one per distinct prefix, because shared prefixes are stored once. In the page's notation, tokens stored $= |P(S)|$.
1. A lookup for $x$ returns the largest $k$ such that $(x_1, \dots, x_k)$ is in $P(S) \cup \{()\}$. The match can end in the middle of an edge.
1. An insertion of $x$ splits an edge at the first token where $x$ diverges from it, or where $x$ ends inside it. The shared part becomes a node of its own and the old remainder becomes its child.
1. Only the tokens beyond the match are new. Prefill runs only on those, and the first $k$ tokens reuse cached keys and values.

The paper shows these steps in the nine-step trace of Figure 3, with a system prompt shared by two chat sessions, then a few-shot query, then self-consistency sampling. In that figure a node is split so that two chat sessions share a system prompt, and the root is split when a new query shares no prefix with any node.

**Tiny example, computed by `scripts/radix_example.py`.** Tokens are words. After the four requests above, the tree is:

```text
You are helpful
  Hi there
    Bye now
  What is KV
Q1 A1 Q2
```

The second insertion, `You are helpful What is KV`, matches 3 tokens, so 3 are new, and it splits the first edge into `You are helpful` and `Hi there`. The third insertion, the second turn of chat 1, matches 5 tokens and adds 2. The tree holds 13 tokens in 5 nodes. Lookups return the following lengths:

| Request | Match |
|---|---|
| `You are helpful Hi there Bye` | 6 of 6 |
| `You are` (ends inside an edge) | 2 of 2 |
| `Q1 A1 Q9` | 2 of 3 |
| `Zebra` | 0 of 1 |

A smaller case: insert `a b c d` (1 node), then `a b c e` (3 nodes, 1 new token), then `a b` (0 new tokens, and an extra split so that `a b` is a node of its own, 4 nodes in all).

**Notes.** The tree is the index. The physical memory is still paged, and the paper says the tensors sit in a paged layout (Section 3), so the prerequisite for this page, [PagedAttention](/pagedattention/), supplies the memory. A cache hit saves prefill compute and prefill memory. It does not shorten decoding.

Exercise 1 asks you to write `insert` and `match` with edge splitting.

### Least recently used eviction with reference counts

**Intuition.** The cache shares one memory pool with the running requests, so it has no fixed size. When the pool runs short, the engine must remove cached tokens. The paper evicts the *least recently used leaf* first. Evicting leaves keeps the tree valid, because a node's KV tensors depend on every token above it, so a node can only be used together with its ancestors. A common ancestor stays until all its children are gone, and then it becomes a leaf and is a candidate too (Section 3).

A running request must not lose its tokens. The paper gives each node a *reference counter* that counts the running requests using it, and a node can be evicted only when its counter is 0 (Section 3).

**Rules.** One eviction round, in this page's words:

1. Collect the nodes that are not the root, have no children, and have a reference count of 0.
1. Remove the one with the oldest last-access time and add its edge length to the tokens freed.
1. Repeat until enough tokens are freed or no node qualifies.

Two details keep the bookkeeping correct. Every node on the path of a request gets a new access time when the request uses it. When an edge splits, the new parent node copies the reference count and the access time of the child it came from, because every request that used the old node also uses the new one.

**Tiny example, computed by `scripts/eviction_example.py`.** Insert $X = (1,2,3,4)$, $Y = (1,2,5,6)$, and $Z = (9,9,9)$ in that order. The cache holds a shared node $(1,2)$ with leaves $(3,4)$ at time 1 and $(5,6)$ at time 2, and a leaf $(9,9,9)$ at time 3. Total size is 9 tokens. A running request holds $Y$, which gives $(1,2)$ and $(5,6)$ a reference count of 1.

- `evict(3)` removes $(3,4)$, the oldest evictable leaf, for 2 tokens. That is not enough, so it removes $(9,9,9)$ for 3 more and frees 5 tokens in total. The locked leaf $(5,6)$ and its parent stay, and the cache holds 4 tokens.
- After the request finishes and releases $Y$, `evict(3)` removes $(5,6)$, frees 2 tokens, sees that $(1,2)$ is now a leaf, removes it too, and frees 4 tokens in total. The cache is empty.

A request frees whole nodes, so an eviction can free more than you ask for. The paper reports that managing the radix tree is cheap: 0.2 seconds of a 74.3-second run, below 0.3% (Section 6.3).

Exercise 2 asks you to write `evict`.

### Cache-aware scheduling

**Intuition.** What you keep depends on what you serve next. If requests that share a prefix are served back to back, the shared prefix stays hot, and if requests from different prefixes interleave, each evicts the other's prefix before it is reused. The paper therefore reorders the waiting queue by the length of the cached prefix each request would match, and runs the longest first (Section 3, Algorithm 1).

**Algorithm 1, in this page's paraphrase.**

1. Take all waiting requests and match each against the tree.
1. Sort them by matched prefix length, longest first.
1. Admit requests while they fit in the available space, which is the free memory plus the evictable memory, and increment the reference counters of their nodes.
1. Merge them into the running batch, allocating memory and evicting if needed.
1. Run the batch. When requests finish, decrement the counters and insert their tokens into the tree.

**Math.** Theorem 3.1 states that for a batch of requests, visiting the radix tree of the requests in depth-first search (DFS) order achieves the optimal cache hit rate, provided the cache is at least as large as the longest request. The paper says that longest-shared-prefix-first order is equivalent to DFS order for one batch. When requests arrive online, new arrivals disrupt the order, but the schedule still approximates DFS on the newly added part of the tree. The paper notes that this greedy rule can starve requests with short matches, and it lists combining it with fair scheduling as future work (Section 3).

**Tiny example, computed by `scripts/scheduling_example.py`.** Three shared prompts of 6 tokens each receive three requests each, with 2 extra tokens per request, and they arrive interleaved: prompt 0, 1, 2, 0, 1, 2, and so on. The 9 requests hold 72 tokens, and their union tree holds 36, so at most 36 tokens can hit. The cache holds a budget of tokens.

| Capacity (tokens) | FCFS hit rate | Longest prefix first |
|---|---|---|
| 10 | 0 of 72 = 0.000 | 36 of 72 = 0.500 |
| 14 | 0 of 72 = 0.000 | 36 of 72 = 0.500 |
| 20 | 0 of 72 = 0.000 | 36 of 72 = 0.500 |
| 24 | 36 of 72 = 0.500 | 36 of 72 = 0.500 |
| 1000 | 36 of 72 = 0.500 | 36 of 72 = 0.500 |

First-come-first-served evicts each prompt before the next request with that prompt arrives, so it never hits while the cache is small. Longest-prefix-first serves the requests in the order 0, 3, 6, 1, 4, 7, 2, 5, 8 at capacity 20, which keeps one prompt hot at a time. With room for everything, the order does not matter, because the total hits equal the size of the repeated part.

**At scale, same script.** The script draws 400 random batches of 12 requests, each built from one of four short shared prefixes plus a short random tail, with a cache of the longest request plus 0 to 11 tokens (seed 0). Longest-prefix-first hits exactly as many tokens as DFS (sorted) order in 400 of 400 batches. It beats first-come-first-served in 389 batches and loses in none, and the mean hit rate is 0.469 against 0.321. These are the page's own random traces and show the shape of the effect, not the paper's numbers.

The paper's measurements show the same effect at a larger scale. Across its benchmarks the cache hit rate is 50% to 99%, and cache-aware scheduling reaches about 96% of the optimal hit rate on average (Section 6.2). Its ablation, which turns off the tree structure, the cache, or the scheduling, finds that each part is needed for the best result (Section 6.3, Figure 8c).

Exercise 3 asks you to write the scheduler and measure the hit rate.

### A compressed finite-state machine for constrained decoding

**Intuition.** A regex or a JSON schema compiles to a finite-state machine (FSM) over characters. Systems that support constrained decoding walk the FSM during decoding and mask the tokens that would leave it, one token per step. Many states have only one way forward. In `{"summary": "`, after `{` the next character must be `"`, then `s`, and so on, and the model has no real choice. The paper merges these forced steps so the engine can append many characters in one step (Section 4).

**Definitions (Section 4).** A *singular transition edge* is an edge whose source state has exactly one successor and that accepts only one character or string. A *compressed edge* is a chain of consecutive singular edges, merged into one edge whose text is the concatenation of the chain. The construction converts the regex to a character-level FSM, then merges singular edges into their predecessors until nothing more can be merged. The runtime then *jumps forward*: when the FSM reaches a compressed edge, it appends the whole text and decodes the next choice, in one forward pass instead of one per token.

This page uses one extra rule for accepting states. A state that can end the output is a real choice even when it has a single outgoing edge, because the model can also emit the end token. Exercise 4 enforces that rule.

**Tiny example, computed by `scripts/fsm_example.py`.** The pattern is `{"name": "`, then `alice` or `bob`, then `", "age": `, then one digit from 1 to 3, then `}`. Its FSM has 30 states, and only 2 of them offer a choice. The forced chains are `{"name": "` (10 characters), `lice", "age": ` after `a`, `ob", "age": ` after `b`, and the final `}`. Decoding `{"name": "alice", "age": 1}` (27 characters) takes 2 steps with jump-forward and 27 without. Decoding `{"name": "bob", "age": 3}` (25 characters) takes 2 steps with jump-forward and 25 without.

The count is optimistic because the script treats a character as a token. Real tokens cover several characters, so the unconstrained cost is lower than one step per character, and the savings are smaller. The paper reports that the compressed FSM raises the throughput of JSON decoding by 1.6 times, and that without reusing the preprocessing across requests the throughput is 2.4 times lower (Section 6.3).

**Retokenization.** Appending forced text as separate tokens can produce a token sequence that the model never saw in training. The script's toy greedy vocabulary shows it. If the model emitted `{"` and `na`, and the forced rest is `me":`, then tokenizing the forced text alone gives `{"`, `na`, `me`, `":`. Tokenizing the whole text gives `{"`, `name`, `":`. The paper's fix is to retokenize the previous text plus the forced text with the model's own tokenizer (Section 4, Appendix B.2).

Exercise 4 asks you to write `jump_forward` and a decoder that counts steps.

### The frontend, hints, and what the evaluation shows

**Co-design.** The runtime gets hints from the program. When a program calls `fork`, the frontend first sends the shared prefix so that the runtime inserts it into the tree, then sends the remaining prompts. The paper's ablation shows that dropping these hints hurts performance (Section 3, Section 6.3).

**Setup.** The paper evaluates Llama-2 (7B and 70B), Mixtral-8x7B, LLaVA-v1.5-7B, LLaVA-NeXT-34B, and GPT-3.5 on workloads that include 5-shot MMLU, 20-shot HellaSwag, ReAct and generative agents, tree-of-thought, skeleton-of-thought, an LLM judge, JSON decoding, multi-turn chat, and a DSPy retrieval pipeline. The baselines are Guidance v0.1.8, vLLM v0.2.5, and LMQL v0.7.3. The main hardware is AWS EC2 G5 instances with A10G 24 GB GPUs, with some runs on A100 80 GB (Section 6.1).

**Results.** The abstract and Section 6.2 report up to 6.4 times higher throughput and up to 3.7 times lower latency than the baselines on open-weight models. On the multimodal models, LLaVA-v1.5-7B rises from 0.18 to 1.15 images per second, and LLaVA-NeXT-34B from 0.02 to 0.10 frames per second (Table 2). In the Chatbot Arena deployment, the cache hit rate was 52.4% for LLaVA-NeXT-34B and 74.1% for Vicuna-33B, and the first-token latency of Vicuna-33B fell 1.7 times (Section 6.2).

<!-- exercises -->

## Pitfalls

- **A hit saves prefill, not decoding.** Reused keys and values replace compute on the prefix only. The output tokens are still generated one at a time, so the speedups in the paper are largest for workloads with long shared prompts and short outputs, such as few-shot benchmarks (Section 6.2).
- **The hit rate depends on the workload.** The paper's 50% to 99% range comes from specific benchmarks and from production traffic that was 52.4% and 74.1% (Section 6.2). Requests with unique prompts see no gain, and the tree costs a little memory and compute for no benefit.
- **Match on token ids, not text.** Two strings that look alike can tokenize differently, and a different token at the boundary breaks the match. A prefix cache keyed on text can return KV for a different token sequence.
- **Split a node, and copy its reference count.** A new parent node must inherit the child's reference count and access time. If it starts at 0, the first `unlock` drives a count negative, and the node can never be evicted (exercise 2).
- **Evict leaves, not interior nodes.** Removing an interior node drops its whole subtree, or leaves children without the tokens they depend on. A node is a candidate only after its children are gone (exercise 2).
- **Lock a request before you evict for it.** The tokens a request matched, and the new tokens it inserts, must be protected until it finishes. An engine that evicts first can free the prefix it was about to use.
- **Greedy cache-aware scheduling can starve requests.** A request with no matching prefix waits behind every request that matches. The paper names this and leaves fair scheduling as future work (Section 3). Production systems add aging or fairness on top (see the course chapter on scheduling policy).
- **Longest-prefix-first equals DFS only for one batch with a large enough cache.** Theorem 3.1 needs a cache at least as big as the longest request, and online arrivals only approximate the DFS order (Section 3). The exercise tests only the single-batch case.
- **One-token pages are the paper's choice, not a law.** vLLM names blocks of several tokens by a hash chain instead, and only full blocks can hit. The tree on this page stores whole token sequences, so a hit can end mid-edge (see the course chapter on prefix caching).
- **A forced edge is not forced if the state can end.** An accepting state with one outgoing edge still gives the model a choice, the end token (exercise 4).
- **Jump-forward needs retokenization.** Appending forced text token by token can give a sequence the model never saw. Retokenize the text so far, and mind that the character-level count on this page overstates the saving (Section 4, Appendix B.2).
- **The baselines are the authors' choice of versions.** The comparisons use Guidance v0.1.8, vLLM v0.2.5, and LMQL v0.7.3 on A10G GPUs for most runs. Later releases of those systems, and later versions of vLLM that added automatic prefix caching, change the gaps (Section 6.1).
- **The simulations on this page are simplified.** They serve one request at a time, ignore time and batching, and treat a character as a token. They show the memory and scheduling effect, not SGLang's throughput.

## Further reading

- The paper: [arxiv.org/abs/2312.07104](https://arxiv.org/abs/2312.07104). Figure 3 shows the radix tree evolving over nine steps, and Section 4 covers the compressed FSM.
- The SGLang source code: [github.com/sgl-project/sglang](https://github.com/sgl-project/sglang).
- The SGLang announcement post: [lmsys.org/blog/2024-01-17-sglang](https://lmsys.org/blog/2024-01-17-sglang/).
- [PagedAttention](/pagedattention/), whose paged KV memory the radix tree indexes, and [Sarathi](/sarathi/), which schedules prefill and decode together.
