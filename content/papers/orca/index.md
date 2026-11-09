---
id: "orca"
title: "Orca: A Distributed Serving System for Transformer-Based Generative Models"
authors: "Yu et al."
year: 2022
venue: "OSDI 2022"
arxiv: null
url: "https://www.usenix.org/conference/osdi22/presentation/yu"
track: inference
tier: core
est_hours: 4
status: full
summary: "A generative model runs once per output token, so a request needs many iterations, and serving systems that fix a batch until every request in it finishes waste slots and delay early finishers. Orca lets the scheduler choose the batch before every iteration, so a finished request returns at once and a new one joins after a single iteration. Because requests then sit at different token positions, Orca batches the token-wise operations over a flattened tensor and runs attention separately for each request."
related_course: "inference: continuous batching"
prereqs:
  - id: transformer
    reason: "Orca schedules around the layer structure of a decoder-only Transformer and its attention, which is the one operation that can't batch across requests of different lengths."
background: [batching, autoregressive-decoding, scheduling]
course_links:
  - title: "Continuous batching"
    url: "https://qwen.ifkash.dev/c/16-continuous-batching"
self_check:
  - q: 'A decoder generates 3 tokens for request A and 100 tokens for request B, and both start in the same batch. How many model iterations does the batch run, and why?'
    a: 'It runs 100 iterations, because each iteration produces one token per request and the batch lasts as long as its longest request. After iteration 3, request A produces nothing useful.'
    see: autoregressive-decoding
  - q: 'Why does batching 8 requests make a decode iteration cheaper per request, even though each request still emits one token?'
    a: 'The GPU reads each weight matrix from memory once per iteration and reuses it for all 8 tokens. Weight traffic per request falls to one eighth, and decode is limited by memory traffic.'
    see: batching
  - q: 'In causal self-attention, why can query position $i$ of request A not use the same key matrix as query position $i$ of request B?'
    a: 'Each request has its own tokens, so it has its own keys and values. Position $i$ attends only to positions $j \le i$ of its own sequence, which is why attention needs a notion of which request a token belongs to.'
    see: transformer
---

## The problem

A classifier such as BERT answers a request in one forward pass. A generative model such as GPT-3 produces one token per forward pass and feeds it back in, so a request needs as many passes as it has output tokens (Section 1, Section 2). The paper calls one pass through all the layers an *iteration*. The first iteration, the *initiation phase*, reads the whole prompt and emits the first token. Each later iteration, an *increment phase*, reads one token and emits the next (Section 2).

Serving systems of the time, such as Triton with FasterTransformer, schedule at the granularity of a *request* (Section 2, Figure 2). The scheduler forms a batch, hands it to the engine, and gets results back only when every request in the batch is done. That causes two failures (Section 3, challenge C1):

- **Early finishers wait.** A request that needs 2 tokens stays in the batch, running wasted computation, until the request that needs 6 finishes (Figure 3).
- **Late arrivals wait.** A request that arrives one iteration after the batch starts waits for the whole batch to finish.

**A concrete failure, computed by `scripts/static_vs_continuous.py`.** Take four requests with arrival times `[0, 0, 1, 2]` and lengths `[2, 6, 2, 2]` iterations, and a maximum batch size of 2. The simulators count time in iterations and give every iteration the same cost. Request-level scheduling runs requests 0 and 1 as one batch for 6 iterations, then requests 2 and 3 for 2 more:

| Scheduler | Finish times | Mean latency | Iterations | Slot utilization |
|---|---|---|---|---|
| Request-level | `[6, 6, 8, 8]` | 6.25 | 8 | 0.750 |
| Iteration-level | `[2, 6, 4, 6]` | 3.75 | 6 | 1.000 |

Request 0 needs 2 iterations and returns at time 6. Slot utilization is the useful slot-iterations divided by `max_bs` times the iterations run.

## Key ideas

The paper has no numbered equations. It describes its techniques in prose, in Figure 4, Figure 5, and Algorithm 1, so this page cites those. The notation is the paper's: $H$ is the hidden size, $L$ is the number of tokens processed together, $B$ is the batch size, and the paper writes `max_bs`, `n_slots`, and `max_tokens` in its scheduler. Time here is measured in iterations, which is a simplification the paper doesn't make: a real iteration with more tokens takes longer.

### Iteration-level scheduling

**Intuition.** Make the scheduler's unit of work one iteration, not one request. Before each iteration it chooses which requests run. A finished request leaves and a waiting request joins, so a slot is idle for no more than one iteration.

**Math.** The paper gives the loop in Section 3 (S1) and Figure 4. Repeat forever: (1) select requests to run next, (2) invoke the engine for one iteration on them, (3) receive the output tokens and append each to its request. The scheduler sees every return, so it detects a finished request and answers the client at once. The policy is *iteration-level first-come-first-served* (Section 4.2). For any two requests $x_i$, $x_j$ in the pool, if $x_i$ arrived earlier, then $x_i$ has run at least as many iterations as $x_j$. A later request can still return first if it needs fewer iterations.

**Tiny example, computed by `scripts/static_vs_continuous.py`.** The same four requests as above, with `max_bs = 2`:

| Iteration | Runs | Event |
|---|---|---|
| 0, 1 | requests 0, 1 | request 0 has run 2 iterations, so it returns at time 2 |
| 2, 3 | requests 1, 2 | request 2 arrived at time 1 and took the freed slot; it returns at time 4 |
| 4, 5 | requests 1, 3 | request 3 arrived at time 2; both return at time 6 |

Finish times are `[2, 6, 4, 6]`, the same 6 iterations of work, and every slot is busy in every iteration.

**At scale.** The same script runs 200 requests with lengths drawn from $U(1, 128)$, the distribution the paper uses in Section 6.2, and `max_bs = 8`. The random lengths are the script's own, drawn with seed 0, so only the shape of the result carries over to the paper.

| Workload | Scheduler | Iterations | Tokens per iteration | Utilization | Mean latency |
|---|---|---|---|---|---|
| All queued at time 0 | Request-level | 2908 | 4.724 | 0.590 | 1485.8 |
| All queued at time 0 | Iteration-level | 1776 | 7.734 | 0.967 | 857.6 |
| Poisson arrivals, 0.10 per iteration | Request-level | 3006 | 4.570 | 0.571 | 482.6 |
| Poisson arrivals, 0.10 per iteration | Iteration-level | 2430 | 5.653 | 0.707 | 74.4 |

In the second workload the iteration-level scheduler is limited by arrivals, not by slots, so its utilization is below 1. Its mean latency still drops by a factor of 6.5. In the first workload, the same script prints the useful slots over the first 20 iterations: `[8, 8, 8, 7, 7, 7, 6, ...]` for the request-level scheduler, which decays as short requests finish, and a constant `[8, 8, ..., 8]` for the iteration-level scheduler.

Exercises 1 and 2 ask you to write both simulators.

### Selective batching

**Intuition.** Iteration-level scheduling creates a problem. The batch now holds requests at different positions: some in the initiation phase with different prompt lengths, some in the increment phase with different context lengths (Section 3, challenge C2). Their tensors have different shapes, so you can't stack them into one $[B, L, H]$ tensor. The paper's observation is that most operations don't care.

**Math.** Section 3 (S2) splits the Transformer layer's operations into two kinds:

- *Token-wise operations* (QKV Linear, Attn Out Linear, LayerNorm, Add, GeLU, and the MLP) treat each token independently. They accept a flattened $[\sum L, H]$ tensor with no batch dimension.
- *Attention* needs to know which tokens belong to the same request, because a token attends only to the tokens of its own sequence. The engine inserts a **Split** before Attention, runs it for each request, and **Merges** the outputs back into $[\sum L, H]$ (Figure 5).

Attention in the increment phase reads the request's earlier keys and values from the *Attention K/V manager*, which keeps them per request until the scheduler says the request is done (Section 3, S2).

**Tiny example, computed by `scripts/selective_batching_example.py`.** Figure 5 batches four requests with 1, 1, 2, and 3 new tokens. Take $H = 4$:

| Item | Shape |
|---|---|
| Flattened input | $(7, 4)$ |
| QKV Linear output | $(7, 12)$, that is $[7, 3H]$ |
| Per-request QKV after Split | $(1, 12)$, $(1, 12)$, $(2, 12)$, $(3, 12)$ |
| Padded $[B, L, H]$ alternative | $(4, 3, 4)$, which is 12 token rows against 7 |

Padding to the longest request would waste 5 of 12 rows. Selective batching wastes none.

**Why the split costs little.** The paper's argument is that Attention has no model parameters, so batching it saves no weight reads (Section 1, Section 6.1). The same script puts numbers on that claim for one GPT-3 175B layer ($H = 12288$, fp16). That is my arithmetic, not a figure from the paper. The layer has $12H^2$ weights, which is 3.62 GB. The GPU reads them once per iteration whatever the batch size, so weight traffic per request falls from 3624 MB at batch size 1 to 113 MB at batch size 32. A request's K/V at context 512 is 25.2 MB per layer, and each request reads only its own. Batching the weights pays off and batching Attention doesn't, so Orca splits only Attention. The paper also reports that its engine's batch times are close to FasterTransformer's when all requests in the batch are identical (Section 6.1, Figure 9a), and it fuses the per-request attention kernels into one launch (Section 5).

Exercise 4 asks you to write the Split and Merge around a provided attention helper.

### Scheduling with K/V reservation

**Intuition.** Free slots in a batch are not the only limit. Each request's keys and values stay in GPU memory until it finishes. If the scheduler admits too many requests, they can fill memory and then none of them can run another iteration, which is a deadlock (Section 4.2).

**Math.** Algorithm 1 (Section 4.2) avoids this by reserving the worst case when a request first runs. A *slot* holds the key and value of one token. The request attribute `max_tokens` bounds the tokens the request can have, so reserving `max_tokens` slots guarantees room until it finishes. The `Select` function (lines 17 to 28) works as follows:

1. Drop requests in the `RUNNING` state, which are already inside the engine (line 19).
1. Sort the rest by arrival time (line 20).
1. For each request, stop if the batch holds `max_bs` requests (line 22).
1. For a request in `INITIATION`, set `new_n_rsrv` to `n_rsrv + req.max_tokens`. If that exceeds `n_slots`, **stop** (lines 23 to 25). Otherwise commit the reservation.
1. Add the request to the batch (line 27).

A finished request subtracts its `max_tokens` from `n_rsrv` (line 14). The paper notes that the operator can set `n_slots` to the largest value that fits in memory, with no experiments, while `max_bs` needs tuning against a latency budget (Section 4.2).

**Tiny example, computed by `scripts/select_example.py`.** Let `n_slots = 100`, `max_bs = 4`, and request A (`max_tokens` 40) already running with its 40 slots reserved. B (50), C (30), and D (10) are new, in that arrival order:

| Request | Reservation | Total | Result |
|---|---|---|---|
| B | 40 + 50 | 90 | admitted |
| C | 90 + 30 | 120 | exceeds 100, so `Select` stops |
| D | 90 + 10 | 100 | would fit, but isn't considered |

D waits behind C. That preserves the FCFS property from the previous idea: skipping ahead to a smaller request could starve a large one. Note that 100 would be an exact fit and the check allows it, because the condition is `>` and not `>=`.

Exercise 3 asks you to implement `Select`.

### What the evaluation shows

Orca is also distributed. It combines intra-layer and inter-layer model parallelism (Section 4.1) and keeps control messages off the NCCL channel that carries tensors (Section 4.1). Because the scheduler doesn't need one batch to finish before it injects the next, it keeps `n_workers` batches in flight and every pipeline stage busy (Section 4.2, Figure 8). FasterTransformer splits a batch into microbatches to pipeline, which shrinks each batch.

Section 6 runs GPT-3-style models of 13B to 341B parameters on A100 GPUs (Table 1). In the end-to-end test (Section 6.2, Figure 10), the paper's headline comparison fixes a median normalized latency of 190 ms on the 175B model. At that latency FasterTransformer reaches 0.185 requests per second and Orca reaches 6.81, which the paper reports as a 36.9x speedup. Dividing the rounded values gives about 36.8; the paper's figure comes from unrounded measurements. The paper's engine-only test shows Orca up to 47% faster than FasterTransformer on the 175B model without any scheduling effect, which it credits to the control-data plane separation (Section 6.1, Figure 9c).

<!-- exercises -->

## Pitfalls

- **The 36.9x number is not a general speedup.** It compares Orca to FasterTransformer behind a custom request-level scheduler, at one latency, on a synthetic trace (Section 6.2). The model never emits an end-of-sequence token: every request generates exactly `max_gen_tokens`, because the authors had no checkpoint or real text (Section 6). Real workloads with early stops stress the scheduler differently.
- **"Continuous batching" is not just a bigger batch.** In the `Poisson` row above, the gain comes from admitting and returning requests between iterations. With `max_bs = 1` the paper notes that Orca degenerates into a pipeline that does no batching (Section 6.2).
- **A larger max batch size is not always better.** The paper says Orca's throughput rises with `max_bs` in its tests, but there's no guarantee for other hardware, models, or workloads, and `max_bs` trades latency for throughput (Section 4.2, Section 6.2).
- **`max_tokens` reservation wastes memory.** Reserving the worst case for each request is safe but leaves slots unused when a request stops early. This is the problem [PagedAttention](/pagedattention/) addresses by allocating K/V memory in blocks on demand.
- **Long prompts stall decoding.** An initiation iteration processes the whole prompt, so one long prompt makes every other request in the batch wait for that iteration. [Sarathi](/sarathi/) splits prompts into chunks for this reason.
- **Skipping versus stopping in `Select`.** Using `continue` where Algorithm 1 uses `break` lets a small request overtake a large one and breaks iteration-level FCFS (exercise 3).
- **Attention over the flat tensor leaks.** If you run causal attention on the whole $[\sum L, H]$ tensor, a token in request B attends to request A's tokens. Everything except the attention call may use the flat layout (exercise 4).
- **The simulators here are simplified.** They charge one time unit per iteration, ignore prompt length, and ignore memory. They show the scheduling effect, not Orca's absolute numbers.

## Further reading

- The paper's PDF: [usenix.org/system/files/osdi22-yu.pdf](https://www.usenix.org/system/files/osdi22-yu.pdf). Section 3 is the core, and Algorithm 1 in Section 4.2 is short.
- The OSDI 2022 presentation page, with the paper and slides: [usenix.org/conference/osdi22/presentation/yu](https://www.usenix.org/conference/osdi22/presentation/yu).
- [Efficient memory management for LLM serving with PagedAttention](/pagedattention/), which keeps iteration-level scheduling and replaces the `max_tokens` reservation.
- [Sarathi](/sarathi/), which chunks prefill so long prompts don't stall decoding.
