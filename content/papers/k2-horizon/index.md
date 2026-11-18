---
id: "k2-horizon"
title: "K2-Horizon: A Frontier Fleet of Open Source Language Models"
authors: "K2 Team (MBZUAI Institute of Foundation Models)"
year: 2026
venue: "arXiv preprint"
arxiv: "2611.21003"
url: "https://arxiv.org/abs/2611.21003"
track: architectures
tier: extended
est_hours: 4
status: full
summary: "K2-Horizon is a family of six fully open language models, from a 0.9B dense model to a 375B-parameter mixture-of-experts model that activates 23B parameters per token, released with data, intermediate checkpoints, logs, and training code. Its 36B-A4B model adds Mixture-of-Value Attention, which routes each token to a few value experts without growing the KV cache, and a softplus attention gate that starts as the identity and can amplify. The report also gives two hyperparameter rules that hold across the family and a robust detector for the loss spikes that kept appearing under a constant learning rate."
related_course: "ml: frontier training recipe"
prereqs:
  - id: deepseekmoe
    reason: "K2-Horizon's feed-forward layers use DeepSeekMoE-style fine-grained routed experts plus a shared expert, and MoVA applies the same routing idea to the value projection."
  - id: gqa
    reason: "Every K2-Horizon model uses grouped-query attention with 8 KV heads, and MoVA's main claim is that it keeps the KV cache at exactly the GQA size."
  - id: adamw
    reason: "The report sets weight decay through the AdamW timescale, which only makes sense once you know how decoupled weight decay shrinks the weights at each step."
background: [rmsnorm, swiglu, top-k-routing, active-vs-total-parameters, kv-cache, weight-decay]
course_links:
  - title: "Mixtures of experts: routing, the router's gradient, and balance"
    url: "https://ml.ifkash.dev/lessons/082-mixtures-of-experts"
  - title: "Normalisation layers: batch norm, layer norm, and their gradients"
    url: "https://ml.ifkash.dev/lessons/040-normalisation-layers"
  - title: "Scaling a learning rate with width, and the Muon update"
    url: "https://ml.ifkash.dev/lessons/074-optimiser-scaling-rules"
  - title: "Pretraining data: dedup, filtering, and mixing"
    url: "https://ml.ifkash.dev/lessons/072-pretraining-data"
self_check:
  - q: 'A MoVA layer routes each token to 4 of 64 value experts. Why doesn''t its KV cache grow by a factor of 4 compared with GQA?'
    a: 'The router mixes the 4 expert outputs into one value vector per KV head before attention runs (Section 2.2, Equation 6). Only that mixed value is cached, so the cache holds the same keys and values per token as GQA. The attention computation itself is unchanged, which is also why MoVA works with FlashAttention.'
    see: kv-cache
  - q: 'Why does the paper pick $\beta = \log 2$ for the softplus gate, and what can this gate do that a sigmoid gate can''t?'
    a: 'With $\beta = \log 2$, $\mathrm{softplus}(0; \beta) = \log 2 / \log 2 = 1$, so a gate with zero pre-activation leaves the attention output unchanged and the model starts out like ungated attention (Section 2.3, Equation 9). A sigmoid gate is bounded in $(0, 1)$, so it halves the output at zero and can only attenuate. Softplus is unbounded above, so it can also amplify.'
    see: swiglu
  - q: 'The 7B model uses a peak learning rate of $3 \times 10^{-4}$ with 6.946B active parameters. What does the $N^{-1/2}$ rule predict for the 32B model with 32.213B, and what did the paper use?'
    a: 'The rule gives $3 \times 10^{-4} \cdot \sqrt{6.946 / 32.213} = 1.39 \times 10^{-4}$ (Section 3.1, Equation 11; computed by `scripts/hyperparams.py`). The paper used $1.5 \times 10^{-4}$ (Table 2), which is within 8% of the prediction.'
    see: weight-decay
---

## The problem

Most open-weight models ship weights and a short report. The data, the intermediate checkpoints, and the decisions that produced the model stay private, so you can't rerun or study the run. K2-Horizon, the successor to K2-V2 in the LLM360 family, releases six models with their pre-training data, intermediate checkpoints, training logs, the xLLM training framework, and the RL360 reinforcement learning stack (Section 1). The models range from a 0.9B dense model to a 375B-total, 23B-active mixture of experts (MoE) (Table 1).

Four of the models (3.7B, 7B, 32B, and 36B-A4B) train on the same 22T-token corpus with the same recipe. That gives a controlled comparison that is rare in public work: a dense 32B model against a sparse model with about one-seventh of its active parameters.

The sparse model has to close a gap. Standard MoE puts experts only in the feed-forward network (FFN). The attention block stays dense, so a model with 4.7B active parameters has small attention projections. The report adds capacity to attention in two ways that don't change the KV cache: Mixture-of-Value Attention (MoVA) and a softplus output gate (Sections 2.2 and 2.3).

Training at this scale also breaks in ways that a short run doesn't. The flagship run logged 83 accepted and 184 erased spike episodes in phase 1 alone (Table 5), and the authors had to tune learning rates for six model sizes without a full sweep at each size (Section 3.1).

This page covers four parts of the report: the architecture changes (Sections 2.1 to 2.3), the hyperparameter rules (Section 3.1), and the loss-spike detector (Section 3.3.2). It skips the data pipelines, post-training, the Uno diffusion adapters, quantization, and safety, and it summarizes the infrastructure in one paragraph.

## Key ideas

The notation follows the paper. Token $t$ has hidden state $x_t \in \mathbb{R}^d$. The paper writes $N$ for active backbone parameters, $T$ for training steps, $\eta$ for the peak learning rate, and $\lambda$ for weight decay. Where this page adds a symbol or a reading of its own, it says so.

The model table gives the context for every idea below (Table 1):

| Model | Type | Total (B) | Active (B) | Layers | Hidden | Norm groups $k$ |
|---|---|---|---|---|---|---|
| 0.9B | Dense | 1.078 | 0.881 | 28 | 1,536 | 1 |
| 3.7B | Dense | 5.058 | 3.775 | 36 | 2,560 | 2 |
| 7B | Dense | 8.999 | 6.946 | 36 | 4,096 | 4 |
| 32B | Dense | 34.779 | 32.213 | 64 | 5,120 | 4 |
| 36B-A4B | MoE and MoVA | 37.445 | 4.665 | 48 | 2,560 | 2 |
| 375B-A23B | MoE | 379.167 | 23.590 | 61 | 6,144 | 1 |

Every model uses GQA with 8 KV heads, RoPE, and SwiGLU. The MoE models use fine-grained routed experts, one shared expert, and dense FFNs in the first three layers. The 375B has 192 routed experts with 8 active per token. The 36B-A4B has 100 routed FFN experts with 8 active, plus 64 value experts with 4 active.

### Grouped RMSNorm bounds each output by the group size

**Intuition.** RMSNorm divides a vector by its root mean square, so the output has an RMS of 1. A single large feature can still take most of that budget. In the extreme, a one-hot vector normalizes to $\sqrt{d}$ in one position. As the model gets wider, that worst case grows. Normalizing smaller groups separately caps it at a value that doesn't depend on the width.

**Math.** RMSNorm computes $\hat{x}_{t,i} = x_{t,i} / v_t$ with $v_t = \sqrt{\tfrac{1}{d}\sum_{i=1}^{d} x_{t,i}^2 + \epsilon}$, with no mean subtraction and no bias (Section 2.1). Since $\sum_i \hat{x}_{t,i}^2 \le d$, each element satisfies $|\hat{x}_{t,i}| \le \sqrt{d}$. Grouped RMSNorm splits the $d$ features into $k$ groups of size $d_g = d / k$ and computes a separate RMS for each group (Equation 3):

$$
\hat{x}_{t,i} = \frac{x_{t,i}}{v_{t,g(i)}}, \qquad v_{t,g} = \sqrt{\frac{1}{d_g} \sum_{j \in g} x_{t,j}^2 + \epsilon},
$$

where $g(i)$ is the group that holds feature $i$. Each element is now within $\pm\sqrt{d_g}$. This page writes $g(i)$; the paper indexes the groups directly. The groups in this page and in the exercise are contiguous slices of the feature axis, which is this page's reading.

The paper uses $k = 2$ or $4$ for the middle four models and $k = 1$ (plain RMSNorm) for the smallest and largest (Table 1). It cites empirical results for the stability benefit, but the sections this page covers show no ablation.

**Tiny example, computed by `scripts/grouped_rmsnorm.py`.** Take $d = 8$ and $x = [8, 0, 0, 0, 1, 1, 1, 1]$. Plain RMSNorm divides by $\sqrt{68/8} = 2.915$ and gives $[2.744, 0, 0, 0, 0.343, 0.343, 0.343, 0.343]$. The large feature takes almost everything. With $k = 2$, the first group has RMS $\sqrt{64/4} = 4$ and the second has RMS 1, so the output is $[2, 0, 0, 0, 1, 1, 1, 1]$. The large feature is capped at $\sqrt{4} = 2$, and the second group keeps its own scale instead of being squashed to 0.343. Exercise 1 asks you to write both norms.

### MoVA routes the value projection, not the attention

**Intuition.** MoE adds parameters to the FFN without adding compute per token. MoVA does the same for the value projection. Each token picks a few value experts, and their outputs mix into one value vector. Attention then runs on that vector as usual, so the KV cache and the attention kernel don't change.

**Math.** Standard attention is $Y = f(QK^\top / \tau)\, V$ with $Q, K, V = XW_q, XW_k, XW_v$ (Equation 4). MoVA keeps $Q$ and $K$ and replaces $W_v$ with $E$ linear value experts $V_i(x_t) = x_t W_v^{(i)}$ (Equation 5). A router chooses a set $\mathcal{V}(x_t)$ of $k$ experts for token $t$, and the value is (Equation 6)

$$
v(x_t) = \sum_{i \in \mathcal{V}(x_t)} p_i(x_t)\, \mathrm{SiLU}\big(V_i(x_t)\big).
$$

The SiLU matters. Without it, the mixture is "an input-dependent linear map" whose only nonlinearity is the routing weights (Section 2.2). In the 36B-A4B, the router picks 4 of 64 value experts, and each expert produces values for all 8 KV heads at once (Figure 2). The paper's sections that this page covers don't give the router's normalization. This page and the exercise read $p_i$ as a softmax over the logits of the chosen experts, as in [DeepSeekMoE](/deepseekmoe/).

**What this costs.** The cache holds one key and one mixed value per KV head, which is the same as GQA. With an assumed head dimension of 128, that is $2 \cdot 8 \cdot 128 = 2{,}048$ elements per token per layer, for GQA and MoVA alike (`scripts/mova_example.py`). The extra cost is the router and $k$ value projections per token instead of one.

**Tiny example, computed by `scripts/mova_example.py`.** Take $x = [1, -1]$, four $2 \times 2$ value experts, and router logits $[1, -1, 3, 0.5]$. With $k = 2$, the router picks experts 2 and 0 (0-indexed) with weights $[0.8808, 0.1192]$. Expert 2 gives $xW = [2, 0]$, so SiLU gives $[1.7616, 0]$. Expert 0 gives $[1, -1]$, so SiLU gives $[0.7311, -0.2689]$. The value is $0.8808 \cdot [1.7616, 0] + 0.1192 \cdot [0.7311, -0.2689] = [1.6388, -0.0321]$. Exercise 2 asks you to write the router and Equation 6.

**What the paper measured.** After mid-training, the 36B-A4B with 4.7B active parameters stays close to the dense 32B with 32.2B active (Table 9):

| Benchmark | 32B dense | 36B-A4B |
|---|---|---|
| MMLU | 88.1 | 86.9 |
| MMLU-Pro | 80.3 | 79.5 |
| BBH | 87.6 | 88.9 |
| GPQA Diamond | 83.7 | 81.3 |
| GSM8K | 95.8 | 94.9 |
| AIME 2026 | 92.2 | 92.1 |
| MBPP | 91.2 | 92.8 |
| IFEval | 89.3 | 90.7 |

These are base-model results from the authors' Eval360 harness, not post-trained scores. The comparison bundles MoVA, the softplus gate, and FFN sparsity, so it doesn't isolate MoVA's effect.

### A softplus gate starts as the identity and can amplify

**Intuition.** Gated attention multiplies the attention output element-wise by a learned gate. A sigmoid gate outputs 0.5 at a zero pre-activation and never exceeds 1. At initialization it halves the signal, and at best it can only recover the ungated output. A gate that is 1 at zero and unbounded above starts the model as plain attention and leaves room to amplify.

**Math.** The gate is $\gamma = \varphi(XW_\gamma + b_\gamma)$, and the output is $\gamma \odot Y$ (Equations 7-8). K2-Horizon sets $\varphi$ to a scaled softplus (Equation 9):

$$
\mathrm{softplus}(x; \beta) = \frac{1}{\beta}\log\big(1 + e^{\beta x}\big), \qquad \beta = \log 2.
$$

At $x = 0$, this is $\log 2 / \log 2 = 1$. It is strictly positive, it approaches $x$ for large $x$, and its slope at 0 is $\sigma(0) = 0.5$.

**Tiny example, computed by `scripts/softplus_gate.py`.** The following table compares the two gates:

| $x$ | Sigmoid | Softplus, $\beta = \log 2$ |
|---|---|---|
| $-4$ | 0.0180 | 0.0875 |
| $-1$ | 0.2689 | 0.5850 |
| $0$ | 0.5000 | 1.0000 |
| $1$ | 0.7311 | 1.5850 |
| $4$ | 0.9820 | 4.0875 |

The sections this page covers don't include an ablation of the gate, so read the motivation as an argument, not a measured result. Exercise 3 asks you to write the gate without overflow.

### Two hyperparameter rules hold across six sizes

**Intuition.** Tuning a learning rate and weight decay at every size is expensive. The report fits two rules on its own runs and sets the rest of the family from them.

**Math.** The first rule fixes the AdamW timescale. Decoupled weight decay shrinks the weights by a factor of $1 - \eta\lambda$ per step, so it forgets old updates over about $1/(\eta\lambda)$ steps. The paper divides by the run length to get a fraction of training (Section 3.1, Equation 10):

$$
\tau = \frac{B}{\eta \lambda D} = \frac{1}{\eta \lambda T},
$$

where $B$ is the batch in tokens and $D$ is the token budget. For the 3.7B, 7B, 32B, and 36B-A4B, the paper scales $\tau$ as $1/\sqrt{\text{TPP}}$, where TPP is tokens per active parameter. The second rule scales the peak learning rate as $\eta \propto N^{-1/2}$ (Equation 11). The paper reports $\tau \sqrt{\text{TPP}}$ between 2.6 and 2.9 and $\eta\sqrt{N}$ between $6.8 \times 10^{-4}$ and $8.6 \times 10^{-4}$. The second range only holds with $N$ in billions, so this page uses that unit.

**Check against Table 2, computed by `scripts/hyperparams.py`.** The following table recomputes both products from the published settings:

| Model | $\eta$ | $\lambda$ | $\tau$ | TPP | $\tau\sqrt{\text{TPP}}$ | $\eta\sqrt{N}$ |
|---|---|---|---|---|---|---|
| 0.9B | 8e-4 | 0.10 | 0.0208 | 5,709 | 1.57 | 7.51e-4 |
| 3.7B | 4e-4 | 0.06 | 0.0379 | 5,807 | 2.89 | 7.77e-4 |
| 7B | 3e-4 | 0.06 | 0.0505 | 3,156 | 2.84 | 7.91e-4 |
| 32B | 1.5e-4 | 0.06 | 0.1010 | 680 | 2.63 | 8.51e-4 |
| 36B-A4B | 4e-4 | 0.06 | 0.0379 | 4,699 | 2.60 | 8.64e-4 |
| 375B-A23B | 1.4e-4 | 0.073 | - | - | - | 6.80e-4 |

The four models that the paper names land in 2.60 to 2.89. The 0.9B falls outside at 1.57, which fits its different schedule: it alone uses cosine decay to 10% of the peak, while the others hold the peak learning rate and replace the decay with checkpoint averaging (Section 3.1). The 375B trains in two phases, so this page doesn't compute a single $\tau$ for it.

A worked transfer: from the 7B's $3 \times 10^{-4}$, the rule predicts $3 \times 10^{-4} \cdot \sqrt{6.946 / 32.213} = 1.39 \times 10^{-4}$ for the 32B. The paper used $1.5 \times 10^{-4}$.

**Sparsity and batch size.** Six-layer MoE sweeps with 4 to 128 routed experts gave best learning rates from $3.16 \times 10^{-3}$ to $10^{-2}$ with no trend, so the authors keep one starting learning rate as sparsity grows. They call the sweep noisy. Halving the batch lowered the small-model training loss by 2.7% to 5.1%, so the 375B batch was halved to about 33.5M tokens.

### Spikes persist under a constant learning rate, so detect them robustly

**Intuition.** A loss spike is a step whose loss is far above recent steps. A mean and standard deviation over a recent window are themselves pulled up by any earlier spike in that window, which hides the next one. A median and a median absolute deviation (MAD) ignore a few outliers.

**Math.** Over a rolling window of 1% of the run, take the median $m_t$ and $\mathrm{MAD}_t = \mathrm{median}(|y - m_t|)$ (Section 3.3.2, Equation 12):

$$
z_t = \frac{y_t - m_t}{1.4826 \cdot \mathrm{MAD}_t}.
$$

The constant 1.4826 makes the MAD match the standard deviation for Gaussian noise. A spike episode starts when $z_t > 5$ and ends when $z$ stays below 2 for three steps. This page reads the window as the steps before $t$, excluding $t$.

**Tiny example, computed by `scripts/spike_detector.py`.** On a made-up 20,000-step loss curve with Gaussian noise of 0.01, the window is 200 steps. A spike of 0.30 at step 8000 gives $z = 28.2$ and an episode from step 8000 to 8005. A smaller spike of 0.08 at step 8150 has the first spike in its window. The robust score is 8.0 and flags it. A plain z-score with mean and standard deviation is 3.0, below the threshold of 5, so it misses the spike. Exercise 4 asks you to write the detector.

**What the paper found.** In K2-V2, most spikes happened early. In the 32B K2-Horizon run, only 52% of spike episodes fall in the first 40% of training. The authors attribute this to the constant learning rate. Gradient norm gives little warning: only the last ten steps before a spike carry signal, where each standard deviation of elevated log gradient norm multiplies the spike odds by about 1.9 (phase 1) or 1.8 (phase 2). Most spike episodes in phase 1 (69%) happened on branches that a rollback later erased. The standard response was to resume from the latest checkpoint, with 40 deeper rollbacks, the deepest rewinding 4,217 steps.

**Infrastructure in brief.** The runs use nodes of eight H200 GPUs with one 400 Gb/s InfiniBand adapter per GPU (Section 3.2). Data parallelism is FSDP full-shard. Tensor parallelism and expert parallelism are tied inside one 8-GPU node so that all-to-all traffic stays on NVLink. Context parallelism appears only at long context, reaching 4 at 512k tokens for the 375B. The report uses no pipeline parallelism, because K2-V2 found that pipeline bubbles cost more than the memory they saved.

<!-- exercises -->

## Pitfalls

- **Reading MoVA as more KV heads.** MoVA mixes the expert outputs before attention. The cache holds one value per KV head, as GQA does. Caching each expert's value separately is a different method with a larger cache.
- **Dropping the SiLU in Equation 6.** Without it, a weighted sum of linear experts is one input-dependent linear map. Apply SiLU to each expert's output before you weight it, not to the mixed sum.
- **Calling the 36B-A4B comparison a MoVA ablation.** The sparse model differs from the dense 32B in FFN sparsity, depth, width, MoVA, and the gate. Table 9 shows the combination, not MoVA alone.
- **Using the natural softplus.** With $\beta = 1$, $\mathrm{softplus}(0) = \log 2 \approx 0.693$, so the gate shrinks the output at initialization. The point of $\beta = \log 2$ is a gate of exactly 1 at zero.
- **Computing softplus with `log(1 + exp(x))`.** That overflows for large $x$. Use `np.logaddexp(0, beta * x) / beta`.
- **Grouping the wrong axis.** Grouped RMSNorm splits the feature axis of each token. It doesn't group tokens or heads, and $k$ must divide $d$.
- **Using $N$ in raw parameters for Equation 11.** The range $6.8 \times 10^{-4}$ to $8.6 \times 10^{-4}$ holds with $N$ in billions. In raw counts, the product is about 30,000 times larger.
- **Applying the timescale rule to the 0.9B.** The paper states it for four models with a constant learning rate. The 0.9B uses cosine decay, and its product is 1.57.
- **Using the standard deviation in the spike score.** A spike already in the window inflates the standard deviation and hides the next spike. Use the MAD.
- **Including step $t$ in its own window.** That pulls the median and the MAD toward the spike. Neither the paper nor this page does it, though the effect is small with a large window.
- **Treating the mid-training scores as final.** Tables 7 and 9 report base checkpoints. The post-trained results are in Section 7, which this page doesn't cover.

## Further reading

- The paper's abstract lists the release: weights, intermediate checkpoints, pre-training data, training logs, the xLLM framework, and the RL360 stack. Start from [the arXiv page](https://arxiv.org/abs/2611.21003).
- Section 9 describes Uno, the diffusion adapters that speed up decoding by up to 2.2 times without a separate draft model. Read it after the [LLaDA](/llada/) and [speculative decoding](/spec-decoding/) pages.
- Section 4 describes the synthetic-data pipelines, including the diversity knobs and the WZIP compressor that measures corpus diversity. Read it with the ml course's pretraining data lesson open.
- Section 6 covers the five RL domain experts and their merge into one model.
