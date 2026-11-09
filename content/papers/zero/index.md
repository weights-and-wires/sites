---
id: "zero"
title: "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
authors: "Rajbhandari et al."
year: 2019
venue: "SC 2020"
arxiv: "1910.02054"
url: "https://arxiv.org/abs/1910.02054"
track: gpu-systems
tier: core
est_hours: 4
status: full
summary: "ZeRO removes the redundancy in data-parallel training: instead of every GPU holding a full copy of the optimizer state, gradients, and parameters, each GPU holds one slice of them. With mixed-precision Adam a model needs 16 bytes per parameter, and the three stages shard the 12 bytes of optimizer state, then the 2 bytes of gradients, then the 2 bytes of parameters across the data-parallel workers. The first two stages move the same amount of data as ordinary data parallelism, and the third moves 1.5 times as much."
related_course: "gpu: distributed training"
prereqs:
  - id: megatron
    reason: "ZeRO is positioned against tensor model parallelism: Megatron-LM splits each layer across GPUs and pays for it in communication, while ZeRO keeps data parallelism and splits only the stored states."
  - id: adam
    reason: "Adam keeps a momentum and a variance tensor for every parameter, and those two tensors, plus the master weights, are the memory that ZeRO shards first."
  - id: mixed-precision
    reason: "Mixed-precision training keeps fp16 weights and gradients for compute and an fp32 master copy for the update, which is where the 16 bytes per parameter come from."
background: [data-parallelism, optimizer-state-memory]
course_links:
  - title: "Data parallel and ZeRO-1"
    url: "https://gpu.ifkash.dev/c/25-ddp-and-zero"
self_check:
  - q: 'A 1.5B-parameter GPT-2 has 3 GB of fp16 weights. How much memory do the model states need when you train it with mixed-precision Adam, and why is it so much more than 3 GB?'
    a: 'The model states need $2\Psi + 2\Psi + K\Psi = 16\Psi$ bytes with $K = 12$, which is 24 GB for $\Psi = 1.5 \times 10^9$. The fp16 weights are only $2\Psi$. The fp16 gradients add another $2\Psi$, and the optimizer adds an fp32 copy of the weights, a momentum, and a variance at 4 bytes each (Section 3.1).'
    see: optimizer-state-memory
  - q: 'The forward and backward passes run in fp16. Why does the mixed-precision optimizer still keep an fp32 copy of the parameters, and which share of the 16 bytes per parameter is it?'
    a: 'The update $\alpha \hat m / (\sqrt{\hat v} + \epsilon)$ is often much smaller than a weight, and an fp16 weight would round the change away. The fp32 copy accumulates the updates. It costs 4 of the 16 bytes per parameter, so it is one third of the 12-byte optimizer state (Section 3.1).'
    see: mixed-precision
  - q: 'A 1B-parameter model trains on $N_d = 4$ data-parallel GPUs. How many gigabytes of model states does each GPU hold with plain data parallelism, and with gradient and optimizer-state sharding ($P_{os+g}$)?'
    a: 'Plain data parallelism holds $16\Psi = 16$ GB per GPU. With $P_{os+g}$ each GPU holds $2\Psi + 14\Psi / N_d = 2 + 3.5 = 5.5$ GB (Section 5.2). Every GPU still holds the full fp16 parameters.'
    see: data-parallelism
---

## The problem

Suppose you want to train a 1.5 billion parameter GPT-2 on 32 GB GPUs. The weights in fp16 take 3 GB, so it looks like it fits with room to spare. It doesn't. Training needs more than the weights, and with the standard recipe of data parallelism, every GPU holds the whole bill (Section 3).

With mixed-precision Adam, the model states for $\Psi$ parameters take $2\Psi$ bytes for the fp16 parameters, $2\Psi$ for the fp16 gradients, and $K\Psi$ for the optimizer, where $K = 12$: an fp32 copy of the parameters, a momentum, and a variance, at 4 bytes each (Section 3.1). That is $16\Psi$ bytes. For $\Psi = 1.5 \times 10^9$, it is 24 GB, computed by `scripts/memory_example.py`. Activations and buffers come on top, and the paper reports that this model cannot be trained on a single 32 GB GPU with TensorFlow or PyTorch (Section 3).

Adding GPUs with data parallelism doesn't help. Each of the $N_d$ workers keeps an identical copy of all 16 bytes per parameter, so memory per GPU doesn't fall as $N_d$ grows. The paper puts the limit for pure data parallelism at less than 1.5 billion parameters on a 32 GB V100 (Section 5.4).

The other tool of the time was model parallelism, covered in [Megatron-LM](/megatron/). It splits the layers and does cut memory, but it shrinks each matrix multiplication and adds communication on every layer. The paper's observation (Section 4.1) is that data parallelism scales better and model parallelism saves more memory. The goal is to get both. ZeRO, the Zero Redundancy Optimizer, keeps data parallelism and removes the redundant copies. The authors say this scales beyond 1 trillion parameters on 1024 32 GB GPUs (Section 1, Section 5.4).

## Key ideas

This page uses the paper's notation: $\Psi$ for the parameter count, $K$ for the optimizer-state bytes per parameter, and $N_d$ for the data-parallel degree. The paper names the stages $P_{os}$ (optimizer states), $P_{os+g}$ (plus gradients), and $P_{os+g+p}$ (plus parameters). The community calls them ZeRO-1, ZeRO-2, and ZeRO-3, and this page does too. The paper has no numbered equations. Its formulas are in Figure 1 and in the text of Sections 3 and 5, so the citations below name those places. One GB is $10^9$ bytes, as in the paper. Every number comes from a script in `scripts/`. Run any script from the paper folder to reproduce the output.

The paper has two parts. ZeRO-DP shards the model states and is the subject of this page. ZeRO-R trims the residual memory (Section 4.2 and Section 6): partitioned activation checkpointing ($P_a$), constant-size buffers ($C_B$), and memory defragmentation ($M_D$). The exercises cover ZeRO-DP only.

### Optimizer state dominates memory

**Intuition.** Of the 16 bytes per parameter, only 2 are the weights the model computes with. Another 2 are the gradients. The other 12 are the optimizer's bookkeeping, and the fp32 update math needs them only once per step, at the very end.

**Math.** The paper counts the model states as follows (Section 3.1):

$$
\underbrace{2\Psi}_{\text{fp16 params}} + \underbrace{2\Psi}_{\text{fp16 grads}} + \underbrace{K\Psi}_{\text{optimizer}} = 16\Psi \quad (K = 12)
$$

The $K\Psi$ term splits into $4\Psi$ for the fp32 parameter copy, $4\Psi$ for the momentum $m$, and $4\Psi$ for the variance $v$. See [Adam](/adam/) for $m$ and $v$ and [mixed-precision training](/mixed-precision/) for the fp32 copy. The paper's claim that the optimizer states "usually consume the most memory" is the reason the first ZeRO stage targets them.

**Tiny example, computed by `scripts/memory_example.py`.** Take a 7B-parameter model:

| Bytes per parameter | Per GPU for $\Psi = 7 \times 10^9$ |
|---|---|
| 2 (fp16 params) | 14 GB |
| 2 (fp16 grads) | 14 GB |
| 12 (optimizer) | 84 GB |
| 16 (total) | 112 GB |

Even on an 80 GB GPU, the 84 GB of optimizer state alone doesn't fit.

Exercise 1 asks you to implement these formulas for every stage.

### Stages 1, 2, and 3 shard progressively more

**Intuition.** Data-parallel workers all hold the same states, and one worker needs only a slice of them at a time. Split the states into $N_d$ slices and let worker $i$ own slice $i$. Do it in three steps: first the optimizer state, which only the update touches, then the gradients, then the parameters.

**Math.** Each stage shards one more term. The per-GPU memory is (Section 5, Figure 1):

| Stage | Paper name | Per-GPU memory | Limit for large $N_d$ |
|---|---|---|---|
| 0 | data parallelism | $2\Psi + 2\Psi + K\Psi$ | $16\Psi$ |
| 1 | $P_{os}$ | $2\Psi + 2\Psi + \dfrac{K\Psi}{N_d}$ | $4\Psi$ (4x less) |
| 2 | $P_{os+g}$ | $2\Psi + \dfrac{(2+K)\Psi}{N_d}$ | $2\Psi$ (8x less) |
| 3 | $P_{os+g+p}$ | $\dfrac{(2+2+K)\Psi}{N_d}$ | $16\Psi / N_d$ |

Stage 1 gives "4x" and stage 2 gives "8x" in the limit (Section 5.1, Section 5.2). In stage 3, memory falls linearly with $N_d$ (Section 5.3).

**Tiny example, computed by `scripts/memory_example.py`.** The paper's Figure 1 uses $\Psi = 7.5 \times 10^9$ and $N_d = 64$, and this page reproduces it:

| Stage | Per GPU |
|---|---|
| Data parallelism | 120.00 GB |
| ZeRO-1 | 31.41 GB |
| ZeRO-2 | 16.64 GB |
| ZeRO-3 | 1.88 GB |

These match the paper's 120, 31.4, 16.6, and 1.88 GB (Figure 1, Table 1). Check the ZeRO-1 value by hand: $4 \cdot 7.5 + 12 \cdot 7.5 / 64 = 30 + 1.406 = 31.41$.

The same function answers "how large a model fits". With $N_d = 64$ and 32 GB for model states, the script gives 7.6B parameters for ZeRO-1, 14.4B for ZeRO-2, and 128.0B for ZeRO-3, which matches the paper's 7.5B, 14B, and 128B (Section 5.4). The same script reports that plain data parallelism can hold at most 2.0B parameters of model states. The paper says "less than 1.5 Billion" (Section 5.4), because activations and buffers need space too, and this page's count ignores them. For 1T parameters on $N_d = 1024$ with ZeRO-3, the script gives 15.6 GB per GPU (Table 1).

Exercise 1 asks you to write `model_state_bytes` and `max_params`.

### Stage 1 in practice: shard the Adam state

**Intuition.** In stage 1, every worker still computes gradients for the full model and ends up with the same averaged gradient. What changes is who runs the optimizer. Worker $i$ runs Adam only on the $i$th slice of the parameters, so it needs $m$ and $v$ only for that slice. Then the workers swap slices so that all of them hold the full, updated parameters.

**Math.** The steps for one training step are as follows (Section 5.1):

1. Each worker computes gradients on its own micro-batch.
1. The gradients are averaged across workers.
1. Worker $i$ updates the $i$th slice of the parameters with Adam, using its $1/N_d$ share of the optimizer state.
1. An all-gather over the workers gives every worker all updated parameters.

Adam acts on each parameter independently, so splitting the parameter vector into slices changes no result. A sharded run is not an approximation of data-parallel Adam. It computes the same numbers.

**Tiny example, computed by `scripts/stage1_sim.py`.** The script runs one step on $\Psi = 10$ parameters and $N_d = 4$ workers. The slices have sizes 3, 3, 2, 2. A worker in plain data parallelism holds 20 Adam floats ($m$ and $v$ for 10 parameters). The ZeRO-1 workers hold 6, 6, 4, and 4. The updated parameters match plain data-parallel Adam, with a maximum absolute difference of 0.0. The first three updated values are `[0.2257, -0.2321, 0.7404]` on both paths.

Exercise 3 asks you to write this step.

### Same communication as data parallelism for stages 1 and 2

**Intuition.** Sharding states sounds like it must cost more communication. It doesn't, because data parallelism already does the work. An all-reduce is a reduce-scatter followed by an all-gather, and ZeRO only needs those two halves at different times.

**Math.** The paper counts the elements each process sends and receives. A reduce-scatter or an all-gather moves $\Psi$ elements, so the all-reduce of data parallelism moves $2\Psi$ (Section 7.1). The three ZeRO stages compare as follows:

| Stage | Communication per step | Volume |
|---|---|---|
| 0 | all-reduce of gradients | $2\Psi$ |
| 1 and 2 | reduce-scatter of gradients, then all-gather of updated parameters | $\Psi + \Psi = 2\Psi$ (Section 7.2.1) |
| 3 | reduce-scatter of gradients, plus an all-gather of parameters in the forward pass and another in the backward pass | $3\Psi$, which is $1.5\times$ (Section 7.2.2) |

The paper schedules the stage-3 parameter gathers as broadcasts, one slice at a time. It discards each slice after use, so the memory stays low (Section 5.3, Section 7.2.2). In stage 2 the gradient of each layer is reduced onto the worker that owns it, and then freed, rather than being all-reduced and kept (Section 5.2).

**Tiny example, computed by `scripts/comm_example.py`.** The script prints the volumes `2, 2, 2, 3` for stages 0 to 3, which gives `1.0, 1.0, 1.0, 1.5` times the baseline. It also runs a reduce-scatter and an all-gather on 4 workers with 8-element gradients, and checks that the result equals the plain sum of the gradients. Worker 0 keeps `[0, 9]` after the reduce-scatter, and worker 3 keeps `[6, -11]`.

Exercise 2 asks you to implement the collectives and the volume table.

<!-- exercises -->

## Pitfalls

- **Reading ZeRO as a replacement for data parallelism.** ZeRO-DP is data parallelism with the states sharded. Every worker still sees a different micro-batch and still runs the full forward and backward pass. The compute per GPU doesn't shrink.
- **Forgetting that stage 1 and 2 keep the full fp16 parameters on every GPU.** Only stage 3 shards the parameters, which is why the limit for stage 2 is $2\Psi$ and not $0$.
- **Treating the memory formulas as the whole footprint.** They cover model states only. Activations, temporary buffers, and fragmentation come on top (Section 3.2), and that is what ZeRO-R addresses.
- **Mixing up the byte counts.** The $K = 12$ bytes are for mixed-precision Adam. Plain fp32 Adam, or another optimizer, has a different $K$, and the stage-by-stage formulas keep $K$ as a parameter for that reason.
- **Checking an Adam shard implementation with only one step.** Adam's first step is almost insensitive to the scale of the gradient, because the update is a ratio of $\hat m$ to $\sqrt{\hat v}$. A bug such as summing the worker gradients where you should average them can pass a one-step test. Compare several steps against a reference, as the tests in Exercise 3 do.
- **Mismatched slice order.** In a sharded optimizer, the shard that a worker updates, the gradient slice it receives, and the optimizer state it holds must all use the same slice. An off-by-one-slice bug converges badly and doesn't crash.

## Further reading

- The paper: [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054). Section 5 and Table 1 hold the memory analysis, and Section 7 holds the communication analysis.
- [DeepSpeed](https://github.com/microsoft/DeepSpeed), the authors' implementation of ZeRO.
- [PyTorch FSDP documentation](https://pytorch.org/docs/stable/fsdp.html), which shards parameters, gradients, and optimizer state in the spirit of stage 3.
- [Megatron-LM](/megatron/) for the model-parallel alternative, and [Adam](/adam/) and [mixed-precision training](/mixed-precision/) for the optimizer and precision background.
