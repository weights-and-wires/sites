---
id: "distserve"
title: "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving"
authors: "Zhong et al."
year: 2024
venue: "OSDI 2024"
arxiv: "2401.09670"
url: "https://arxiv.org/abs/2401.09670"
track: inference
tier: extended
est_hours: 5
status: full
summary: "Serving systems that run the prompt phase and the token-by-token phase on the same GPUs make each phase slow the other, so a service must over-provision GPUs to meet both a first-token limit and a per-token limit. DistServe runs the two phases on separate GPU groups, picks a parallelism plan and an instance count for each phase by simulation, and places the groups so the KV cache moves over fast links. The paper reports up to 7.4 times more requests or 12.6 times tighter latency limits than the baselines while at least 90 percent of requests still meet their limits."
related_course: "inference: serving architecture"
prereqs:
  - id: sarathi
    reason: "DistServe argues that chunked prefill with piggybacked decodes, as in Sarathi-Serve, reduces prefill-decode interference but cannot remove it, so you need that scheduler's trade-off in mind to follow the case for separate GPUs."
background: [goodput, slos]
course_links:
  - title: "Disaggregated prefill and decode"
    url: "https://qwen.ifkash.dev/c/17b-disaggregated-serving"
  - title: "A model of step time and goodput"
    url: "https://qwen.ifkash.dev/c/17a-performance-model"
  - title: "Benchmarks that mean something"
    url: "https://qwen.ifkash.dev/c/17-benchmarks"
  - title: "Continuous batching"
    url: "https://qwen.ifkash.dev/c/16-continuous-batching"
self_check:
  - q: 'Sarathi-Serve keeps decodes smooth by adding chunks of a prompt to each decode iteration. Why does DistServe say that this still cannot remove the interference between the two phases?'
    a: 'Every decode in a hybrid batch still waits for the prefill chunk that shares its iteration, so its time per output token rises. A chunk far below the size that saturates the GPU lengthens the prefill, and each later chunk re-reads the KV cache of the earlier chunks, which costs $O(N^2)$ chunk loads for $N$ chunks (Section 2.3).'
    see: sarathi
  - q: 'A service runs 2 prefill GPUs and 1 decoding GPU and sustains 10 requests per second with 90% of requests inside their limits. What is the per-GPU goodput, and what must "90%" count?'
    a: 'The per-GPU goodput is 10 / 3, about 3.3 requests per second. The 90% counts requests that meet both the TTFT and the TPOT limit, which is the SLO attainment (Section 1).'
    see: goodput
  - q: 'A system has a P90 TTFT within its limit and a P90 TPOT within its limit. Can fewer than 90% of requests meet both limits?'
    a: 'Yes. The requests that miss the TTFT limit and the ones that miss the TPOT limit can be different requests, so the share that meets both can be lower than either share. Goodput counts a request only when it meets both limits.'
    see: slos
---

## The problem

Every request has two phases. *Prefill* processes the whole prompt in parallel and emits the first token. *Decoding* then emits one token per step until the end (Section 1). A service measures them with two latencies: *time to first token* (TTFT) for prefill, and *time per output token* (TPOT) for decoding, which is the average gap between the later tokens of one request. The overall latency of a request is its TTFT plus its TPOT times the number of generated tokens (Section 1, footnote 1). Different applications weigh the two differently. A chatbot needs a fast first token and a TPOT only as quick as a person reads. A summarizer needs a fast TPOT and tolerates a slow first token (Section 1).

Serving systems of the time, such as vLLM, colocate both phases on each GPU and batch prefill steps with decoding steps across requests to maximize throughput (Section 1, Section 2.2). Under latency limits that is the wrong objective. A prefill step takes much longer than a decoding step, so a decode that shares a batch with a prefill waits for the prefill and its TPOT rises. A decode step in the batch also slows the prefill and raises its TTFT (Section 1, Section 2.3, Figure 2). Scheduling the phases in separate iterations does not help: the decodes then queue behind running prefills, and favoring one phase hurts the other. The paper also names a second cost. Colocation forces one resource and parallelism plan on both phases, although prefill is compute-bound and wants intra-op parallelism to cut its time, and decoding is bound by memory bandwidth and wants a large batch (Section 2.3).

The paper measures this on a 13B model with 512-token prompts and 64 output tokens on one A100 80GB. With the 90 percent target, the best colocated rate is about 1.6 requests per second. A GPU that runs only prefill sustains 5.6 requests per second, and a GPU that runs only decoding sustains 10. Two prefill GPUs and one decoding GPU would serve 10 requests per second on three GPUs, which is 3.3 per GPU and 2.1 times the colocated figure (Section 1, Figure 1). Throughput misses this, because it counts every token, including the tokens of requests that arrived too late to matter. The metric the paper maximizes is *goodput* (Section 1).

**A concrete failure, computed by `scripts/goodput_sim.py`.** The script runs a toy model, not the paper's simulator. Every request has a 0.10 s prefill and 64 output tokens. A decode iteration takes 0.03 s for any batch of up to 32 requests. The limits are a TTFT of 0.40 s and a TPOT of 0.05 s. One colocated GPU runs prefill first: when a request waits, the GPU runs its prefill, and the running decodes stall. The table shows the share of 2000 Poisson requests that meet each limit.

| Rate (requests/s) | Meet TTFT | Meet TPOT | Meet both |
|---|---|---|---|
| 2 | 1.000 | 0.980 | 0.980 |
| 3 | 0.998 | 0.865 | 0.864 |
| 4 | 0.988 | 0.535 | 0.526 |
| 5 | 0.931 | 0.234 | 0.228 |
| 6 | 0.557 | 0.049 | 0.045 |

The first token arrives on time up to rate 5, and the decodes miss the TPOT limit long before that, because each arriving prompt stalls them. The colocated goodput at a 90 percent target is 2.83 requests per second. Two prefill GPUs and one decoding GPU sustain 17.19 requests per second on three GPUs, which is 5.73 per GPU and 2.02 times the colocated figure. That ratio belongs to this toy and its limits. It is not the paper's result.

## Key ideas

The paper's notation is $R$ for the Poisson arrival rate, $D$ for the execution time of one prefill, and $K$ for the speedup of intra-op parallelism. The paper numbers three equations, all in Section 3.1, and gives two placement algorithms in Section 4. The numeric examples use the toy model above and OPT-66B's published shape, not the paper's measurements. The toy model has no queueing in the decode phase beyond a limit of 32 slots per GPU, and no attention cost.

### Goodput and SLO attainment

**Intuition.** *SLO attainment* is the share of requests that meet their latency limits. *Goodput* is the highest request rate at which the attainment stays at or above a goal, divided by the number of GPUs you provision. If a request is late, the work you spent on it is wasted from the user's point of view, so goodput counts only the requests that were on time.

**Math.** The paper defines per-GPU goodput as the maximum request rate that meets the SLO attainment goal, say 90 percent, for each GPU provisioned (Section 1). A request meets the goal only when it meets the TTFT limit and the TPOT limit (Section 2). The simulator that the placement algorithms use has no closed form for the attainment, because real traces have varied lengths and irregular arrivals. So the paper finds the maximum rate by binary search over simulation trials (Section 4.1). The search relies on attainment falling as the rate rises.

**Tiny example, computed by `scripts/goodput_sim.py`.** Four requests have TTFT of 0.1, 0.5, 0.2, and 0.2 s and TPOT of 0.03, 0.03, 0.06, and 0.04 s. With limits of 0.40 s and 0.05 s, the second request misses on TTFT and the third misses on TPOT, so the attainment is 0.50. For the colocated GPU, the search starts from the range 0.05 to 40 requests per second. Its midpoints 20.025 and 10.037 give an attainment of 0.000 and the third, 5.044, gives 0.222. All three fail the 90 percent target, so the upper end drops. At 2.547 the attainment is 0.942, so the lower end rises. After 24 rounds the answer is 2.83 requests per second.

Exercise 3 asks you to write the attainment and the bisection.

### Disaggregation removes the interference

**Intuition.** Give the prompt phase its own GPUs and the token phase its own GPUs. A decode step then never waits behind a prefill, and a prefill never waits behind decodes. The price is that the KV cache of each prompt must travel from a prefill GPU to a decoding GPU.

**Math.** The paper uses the word *instance* for a unit of resources that holds one complete copy of the weights, which can span many GPUs. A *prefill instance* runs the prefill of a request and produces the first token. It then sends the KV cache and the token to a *decoding instance*, which produces all later tokens (Section 2.3, "Opportunities"). Each phase then has its own weights, batch size, and parallelism plan. Because decoding has low GPU utilization, the system can attach several prefill instances to each decoding instance, which builds larger decoding batches (Section 2.3, Section 3.2).

**Tiny example, computed by `scripts/goodput_sim.py`.** In the toy model the 2 prefill plus 1 decoding split keeps both limits up to rate 17 (attainment 0.922) and falls off at rate 18 (0.120), where the 32 decode slots saturate. At rate 12, the attainment is 0.997. Other splits show the trade-off: 1 prefill plus 1 decoding sustains 6.82 requests per second on 2 GPUs (3.41 per GPU), 3 prefill plus 1 decoding sustains 17.26 on 4 GPUs (4.32 per GPU), and 3 prefill plus 2 decoding sustains 28.37 on 5 GPUs (5.67 per GPU). The colocated GPU manages 2.83.

**How this differs from Sarathi-Serve.** [Sarathi-Serve](/sarathi/) keeps both phases on one GPU and tunes the token budget of each iteration. DistServe argues that this trades TTFT for TPOT and cannot remove the interference. A budget much below the saturation point slows the prefill, and a budget near it leaves little room for decodes. Chunking also loads the earlier chunks' KV cache again for every chunk, which is $N + (N-1) + \dots + 1 = O(N^2)$ chunk loads for $N$ equal chunks, against $O(N)$ without chunking (Section 2.3). Disaggregation removes the interference by placement, and its cost is the KV transfer. The toy baseline on this page is prefill-first, as in vLLM, so it overstates the gain over a chunked baseline. The paper's own comparison with chunked prefill is DeepSpeed-MII in Section 6.2.

### The prefill instance as a queue

**Intuition.** After disaggregation, a prefill instance has no decodes to compete with. If every prompt has the same length, one request saturates the GPU, so the instance serves one request at a time, first come, first served. That is an M/D/1 queue: Markovian (Poisson) arrivals, Deterministic service, one server. Queueing theory gives the average wait in closed form, which lets you reason about parallelism before you simulate.

**Math.** With Poisson arrival rate $R$ and a fixed execution time $D$ per request, and $RD < 1$, the average TTFT is

$$
\text{Avg\_TTFT} = D + \frac{R D^2}{2(1 - RD)}
$$

(Section 3.1, Equation 1). The first term is the execution time and the second is the queueing delay. For 2-way inter-op parallelism, the request-level latency stays about $D$ and the slowest of the two stages takes $D_m \approx D/2$, because the paper assumes negligible communication between layers. Equation 1 with $D_m$ in the queue term gives

$$
\text{Avg\_TTFT}_{\text{inter}} = D + \frac{R D^2}{4(2 - RD)}
$$

(Section 3.1, Equation 2). For 2-way intra-op parallelism, the paper introduces a speedup coefficient $K$ with $1 < K < 2$, which reflects the communication cost of intra-op parallelism. The execution time becomes $D/K$, and

$$
\text{Avg\_TTFT}_{\text{intra}} = \frac{D}{K} + \frac{R D^2}{2K(K - RD)}
$$

(Section 3.1, Equation 3). The paper's conclusion is that intra-op parallelism wins at low rates, where the execution time dominates, and inter-op wins at high rates, where the queueing delay dominates. A tighter TTFT limit favors intra-op, and a smaller $K$ weakens it (Figure 4). The paper also notes that replicating the instance across $N$ copies replaces $R$ with $R/N$ in Equation 1, at the cost of more copies of the weights (Section 3.2).

**Tiny example, computed by `scripts/md1_ttft.py`.** The script works in units of $D$. First it checks Equation 1 against a queue simulation of 200,000 requests: at $RD = 0.5$ the formula gives 1.5000 and the simulation 1.4968, and at $RD = 0.2$ they give 1.1250 and 1.1248. It also checks that Equation 2 follows from Equation 1 with $D_m = D/2$: at $RD = 1$ both give 1.2500. The comparison of the three plans:

| $RD$ | One GPU | Inter-op | Intra-op, $K = 1.6$ | Intra-op, $K = 1.9$ |
|---|---|---|---|---|
| 0.1 | 1.056 | 1.013 | 0.646 | 0.541 |
| 0.5 | 1.500 | 1.083 | 0.767 | 0.620 |
| 0.9 | 5.500 | 1.205 | 1.027 | 0.763 |
| 1.1 | unstable | 1.306 | 1.312 | 0.888 |
| 1.5 | unstable | 1.750 | 5.312 | 1.513 |

For $K = 1.6$ the two plans cross at $RD = 1.095$, and for $K = 1.9$ at $RD = 1.626$. The values for $K$ of 1.5, 1.7, and 1.8 are 0.936, 1.257, and 1.429. The paper's Figure 4(b) shows the same effect for $K$ from 1.5 to 1.9.

Exercise 1 asks you to write the three equations and the choice between them.

### The decoding instance

**Intuition.** One decode step reads all the weights to emit one token per request, so a single request barely uses the GPU. Batching is the fix, and it is hard to do on a colocated GPU, because a larger decode batch means more prefill work interleaved at the same rate. A dedicated decoding instance can hold a batch as large as its memory allows.

**Math.** Section 3.2 gives rules, not equations. Decoding is bandwidth-bound at small batch sizes, and throughput rises with the batch size (Figure 3b). After disaggregation, the batch size is limited by the memory for the KV caches of all active requests. PagedAttention and grouped-query attention let the batch grow until the phase is nearly compute-bound. For the parallelism plan, intra-op parallelism lowers the latency with diminishing returns, and inter-op parallelism scales the throughput almost linearly (Figure 5). So when the TPOT limit is tight, use intra-op parallelism first, and then use inter-op parallelism for throughput. If the model fits on one GPU, replication is a third option.

**Tiny example, computed by `scripts/placement.py`.** In the toy model one prefill GPU sustains 6.82 requests per second of TTFT-compliant traffic, and one decoding GPU with 32 slots sustains 17.26 requests per second of TPOT-compliant traffic. The decoding GPU is 2.5 times faster, so a balanced pool has about 2.5 prefill GPUs for each decoding GPU. That is the paper's reason to attach several prefill instances to one decoding instance.

### The cost of moving the KV cache

**Intuition.** Disaggregation adds a copy. The KV cache of a request holds a key and a value vector per layer and per token, and it grows with the prompt. The copy has to be cheap enough to hide, and that depends on the link.

**Math.** The paper gives one worked case: the KV cache of a 512-token request on OPT-66B is about 1.13 GB, so 10 requests per second need 11.3 GB per second, or about 90 Gbit/s to make the overhead invisible (Section 3.3, "Communication overhead"). Clusters with InfiniBand (800 Gbit/s in the paper's example) meet that. When the cross-node bandwidth is limited, the paper uses the NVLink inside a node, where the peak between A100 GPUs is 600 GB/s (Section 3.3). This puts a constraint on the placement, which Algorithm 2 handles (Section 4.2). The paper adds that models with grouped-query or multi-query attention have a smaller cache and cost less to move (Section 6.1), and that the relative cost of the copy shrinks for long prompts, because the prefill grows faster than the cache (Section 7).

**Tiny example, computed by `scripts/kv_transfer.py`.** OPT-66B has 64 layers and a hidden size of 9216 (the OPT model's published shape, not a number from this paper). The cache needs $2 \times 64 \times 9216 \times 2 = 2{,}359{,}296$ bytes per token in fp16, so 512 tokens need 1,207,959,552 bytes. The script prints that as 1.125 GiB, and the paper's 1.13 GB and 90 Gbit/s are the same figures in binary units: 10 requests per second is 11.25 GiB/s, or 90.0 Gibit/s. In decimal units it is 1.208 GB and 96.6 Gbit/s. The transfer time of one request:

| Link | Time |
|---|---|
| NVLink, 600 GB/s | 2.01 ms |
| NVLink, 300 GB/s (one direction) | 4.03 ms |
| InfiniBand, 400 Gbit/s | 24.16 ms |
| Ethernet, 25 Gbit/s (the paper's cross-node link) | 386.55 ms |

On the paper's testbed, the KV transfer takes less than 0.1 percent of the total latency for OPT-175B on ShareGPT, and more than 95 percent of requests wait less than 30 ms for it, because the placement keeps the transfer on NVLink (Section 6.3, Figure 10). In the toy model, adding the 25 Gbit/s transfer time to every request lowers the 2 prefill plus 1 decoding goodput from 17.19 to 16.90 requests per second, and the 2.0 ms NVLink time leaves it at 17.19. A grouped-query model with 8 key-value heads in place of 72 would cut the cache by a factor of 8 (this is the script's own example).

Exercise 2 asks you to write these functions.

### Searching for a placement

**Intuition.** The best number of GPUs and the best parallelism plan for each phase depend on the model, the workload, and the limits. The paper treats them as a search. For each phase, find the plan with the best per-GPU goodput by simulation, then replicate each phase to cover the traffic.

**Math.** A placement is the parallelism of the prefill and decoding instances, the number of each, and where they sit on the cluster (Section 4). *Algorithm 1* (Section 4.1) is for clusters with fast cross-node links. For every intra-op degree and every inter-op degree that fits in memory, it simulates the prefill goodput with `simu_prefill` and the decoding goodput with `simu_decode`, and keeps the best configuration of each per GPU. Then it replicates them: $n = \lceil R / \text{goodput}_p \rceil$ prefill instances and $m = \lceil R / \text{goodput}_d \rceil$ decoding instances for a traffic rate $R$. Its complexity is $O(NM^2)$ for $N$ nodes per instance and $M$ GPUs per node, and the search takes under 1.3 minutes in the largest setting (Section 4.1). The simulator fits a distribution to the request history, resamples a trace, and uses an analytical latency model of the prefill and decoding steps (Appendix A). Its attainment differs from the real system by less than 2 percent (Section 6.4, Table 2).

*Algorithm 2* (Section 4.2) is for clusters with slow cross-node links. The KV cache moves only between the same layers of the two instances, so the algorithm splits each instance into segments by inter-op stage, and puts the prefill segment and the decoding segment of one stage on the same node. The transfer then runs over NVLink. It enumerates the parallelism that fits in one node and simulates each joint configuration. In the ablation, Algorithm 1's placement beats Algorithm 2's, because it is free of the same-node constraint. The paper runs that comparison in simulation, because vLLM has no inter-op parallelism and the testbed has no fast cross-node link (Section 6.4, Figure 11).

**Tiny example, computed by `scripts/placement.py`.** This page's toy has one GPU per instance, so the search covers only the instance counts. It measures one prefill GPU against the TTFT limit alone (6.82 requests per second) and one decoding GPU against the TPOT limit alone (17.26). The system rate of a split is the smaller of the two pools, $\min(n_p g_p, n_d g_d)$. The brute force over a budget of 8 GPUs returns 6 prefill and 2 decoding, with 34.52 requests per second. The best split by budget:

| GPU budget | Best split | Rate (requests/s) | Per GPU |
|---|---|---|---|
| 2 | 1 prefill + 1 decoding | 6.82 | 3.41 |
| 3 | 2 prefill + 1 decoding | 13.64 | 4.55 |
| 4 | 3 prefill + 1 decoding | 17.26 | 4.32 |
| 5 | 3 prefill + 2 decoding | 20.47 | 4.09 |
| 8 | 6 prefill + 2 decoding | 34.52 | 4.32 |

The full simulation of 6 prefill plus 2 decoding gives 35.40 requests per second, a little above the rule. The same gap shows for 2 prefill plus 1 decoding: the rule says 13.64 and the simulation 17.19. The rule is conservative because the toy sends requests to the prefill GPUs in round-robin order, and the stream that each GPU sees is more regular than Poisson. Eight colocated GPUs give 22.66 requests per second, so the best split is 1.52 times better in this toy.

Exercise 4 asks you to write the brute force and the replica count.

### Online scheduling

The runtime is simple. A central controller sends each request to the prefill instance with the shortest queue, then to the least loaded decoding instance, in first-come-first-served order (Section 4.3). Three refinements matter. To reduce pipeline bubbles from non-uniform prompts, the scheduler batches prefills to a total length close to $L_m$, the shortest length that saturates the GPU, so every micro-batch takes about the same time. To cope with bursts, decoding instances *pull* the KV cache from prefill instances, which use their own memory as a queue, instead of having it pushed into a decoding instance that may be full. And a workload profiler triggers a new placement run when the length or rate pattern shifts (Section 4.3). The system does not implement preemption or fault tolerance. A fault in one decoding instance that serves several prefill instances can propagate (Section 4.3).

### What the evaluation shows

The testbed has 4 nodes with 8 A100 80GB GPUs each, linked by NVLink, and a 25 Gbit/s link between nodes. The paper uses OPT-13B, OPT-66B, and OPT-175B in fp16, with Poisson arrivals over three workloads (Section 6.1, Table 1):

| Application | Model | TTFT limit | TPOT limit | Dataset |
|---|---|---|---|---|
| Chatbot | OPT-13B | 0.25 s | 0.1 s | ShareGPT |
| Chatbot | OPT-66B | 2.5 s | 0.15 s | ShareGPT |
| Chatbot | OPT-175B | 4.0 s | 0.2 s | ShareGPT |
| Code completion | OPT-66B | 0.125 s | 0.2 s | HumanEval |
| Summarization | OPT-66B | 15 s | 0.15 s | LongBench |

The baselines are vLLM and DeepSpeed-MII, which does chunked prefill. The results at the 90 percent target are as follows (Section 6.2, Figures 8 and 9):

- **Chatbot.** DistServe sustains 2.0 to 4.6 times the rate of vLLM and 1.6 to 7.4 times the rate of DeepSpeed-MII. It meets a limit that is 1.8 to 3.2 times tighter than vLLM's and 1.7 to 1.8 times tighter than DeepSpeed-MII's.
- **Code completion.** 5.7 times the rate and a 1.4 times tighter limit than vLLM, and 1.6 times the rate and a 1.4 times tighter limit than DeepSpeed-MII.
- **Summarization.** 4.3 times the rate and a 12.6 times tighter limit than vLLM, and 1.8 times the rate and a 2.6 times tighter limit than DeepSpeed-MII.
- **At a 99 percent target,** the gains over vLLM are 3 to 8 times in rate and 1.24 to 6.67 times in limit (Appendix C).
- **Latency breakdown.** The KV transfer is under 0.1 percent of the total latency for OPT-175B (Section 6.3, Figure 10).
- **Ablation.** On OPT-66B with ShareGPT, the best parallelism for vLLM is its default (intra-op 4), so searching for a better plan does not help a colocated system (Section 6.4).

The placements that the search chose are in Appendix B, Table 3. For OPT-13B the prefill instance uses tensor parallelism 2 and the decoding instance uses 1. For OPT-66B, prefill uses 4 and decoding uses tensor 2 with pipeline 2. For OPT-175B, prefill uses tensor 3 with pipeline 3, and decoding uses tensor 4 with pipeline 3. The phases get different plans, which is the second claim of the paper.

<!-- exercises -->

## Pitfalls

- **The headline is the best case.** The 7.4 times is the top of a range measured against DeepSpeed-MII on the chatbot workload, and the 12.6 times is the tightest limit measured against vLLM on summarization (Section 6.2). The two numbers come from different baselines and different metrics. The ratio against vLLM on chatbot is 2.0 to 4.6 times.
- **Goodput is per GPU and per request.** The 3.3 requests per second in Figure 1 is the 10 requests per second divided by 3 GPUs. A system with more GPUs gets a higher total rate, so compare the rate per GPU. And a request counts only when it meets both limits, not when the averages do.
- **P90 is not attainment.** The share of requests meeting a P90 TTFT limit and the share meeting a P90 TPOT limit can overlap little, so the share that meets both can be well under 90 percent.
- **The two phases hold two copies of the weights.** Each instance manages its own complete copy (Section 2.3). On a single GPU, or on a small cluster, the design space of DistServe shrinks and a colocated engine can be simpler and as good (Section 7).
- **It is a goodput design, not a throughput design.** For offline work with no latency limits, the paper says that chunked prefill that fills every batch can be better (Section 7).
- **Placement depends on the link.** At 25 Gbit/s the 512-token OPT-66B transfer takes 386.55 ms in this page's calculation, which is why Algorithm 2 keeps same-stage segments on one node. If you copy the idea to a cluster with a slow network and ignore the placement, the copy becomes the bottleneck.
- **The evaluation uses multi-head attention on purpose.** OPT has the largest KV cache among the common designs, which puts the most pressure on the transfer. Models with grouped-query attention move less (Section 6.1).
- **The queueing model has assumptions.** Equation 1 needs a fixed execution time and no batching, which holds for equal-length prompts only. Equation 2 assumes negligible communication between pipeline stages. With variable prompt lengths, inter-op parallelism has pipeline bubbles that the model does not show (Section 3.3).
- **The toy model is not the paper's simulator.** It uses fixed lengths, a decode step that costs the same at any batch size up to 32, and a prefill-first colocated baseline. It shows the direction of the effect. The paper's simulator uses the latency model in Appendix A and agrees with the real system within 2 percent.
- **Disaggregation adds failure modes.** A fault in one decoding instance can propagate to the prefill instances that feed it, and the paper leaves preemption and fault tolerance as future work (Section 4.3).

## Further reading

- The paper's PDF: [arxiv.org/abs/2401.09670](https://arxiv.org/abs/2401.09670). Section 3 holds the queueing analysis, and Algorithms 1 and 2 are in Section 4.
- The official code: [github.com/LLMServe/DistServe](https://github.com/LLMServe/DistServe).
- The authors' post: [Throughput is Not All You Need: Maximizing Goodput in LLM Serving using Prefill-Decode Disaggregation](https://haoailab.com/blogs/distserve/) (Hao AI Lab, March 2024).
- [Sarathi-Serve](/sarathi/), the colocated alternative that DistServe compares itself against.
- [Efficient memory management for LLM serving with PagedAttention](/pagedattention/), which the decoding instance uses to grow its batch.
- Splitwise and TetriInfer, concurrent work that the paper cites for the same disaggregation idea (Section 8).
