---
id: "mamba"
title: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
authors: "Gu et al."
year: 2023
venue: "COLM 2024"
arxiv: "2312.00752"
url: "https://arxiv.org/abs/2312.00752"
track: architectures
tier: core
est_hours: 5
status: full
summary: "Mamba is a sequence model built on a state-space recurrence, where a fixed-size hidden state is updated once per token and no attention is used. Its key change is selection: the step size, input matrix, and output matrix are computed from the current token, so the model can keep or erase what it has stored, which a time-invariant state-space model can't do. Selection rules out the convolution trick that earlier state-space models used for training, so the paper computes the recurrence with a fused, memory-aware parallel scan, which gives linear-time training and a constant-size state at inference."
related_course: "ml: alternatives to attention"
prereqs:
  - id: transformer
    reason: "Mamba is built and measured as a replacement for attention, so you need the Transformer's quadratic training cost and its growing KV cache to see what the paper removes."
  - id: lstm
    reason: "Theorem 1 shows that a selective SSM with one state entry is a gated recurrence, so the LSTM's gate and additive state update are the intuition for the step size Delta."
background: [linear-recurrences, parallel-scan, state-space-models]
course_links:
  - title: "Linear attention and the gated delta rule"
    url: "https://qwen.ifkash.dev/c/06-gated-delta-rule"
  - title: "Scan and histogram"
    url: "https://gpu.ifkash.dev/c/07-scan-and-histogram"
self_check:
  - q: 'The recurrence $h_t = a_t h_{t-1} + b_t$ looks sequential. Which property of the pairs $(a_t, b_t)$ lets a scan compute every $h_t$ in $O(\log L)$ rounds, and why does the order of the two arguments of the operator matter?'
    a: 'Composing two steps $h \mapsto a_1 h + b_1$ and $h \mapsto a_2 h + b_2$ gives another step of the same form, $(a_2 a_1,\; a_2 b_1 + b_2)$, and this operator is associative, so partial results can be combined in any grouping. It is not commutative, because "first step 1, then step 2" differs from the reverse: with $p = (0.5, 1)$ and $q = (0.25, 2)$, composing $p$ then $q$ gives $b = 2.25$ and $q$ then $p$ gives $b = 2.0$.'
    see: parallel-scan
  - q: 'A linear recurrence $h_t = \bar{A} h_{t-1} + \bar{B} x_t$ uses the same $\bar{A}$ and $\bar{B}$ at every step. Why can it be rewritten as a convolution, and why does that stop being possible when $\bar{A}$ and $\bar{B}$ depend on $x_t$?'
    a: 'Unrolling the recurrence gives $y_t = \sum_{k \le t} C \bar{A}^k \bar{B}\, x_{t-k}$, so one fixed kernel $K = (C\bar{B}, C\bar{A}\bar{B}, \dots)$ applies at every position (Equation 3). When $\bar{A}_t$ and $\bar{B}_t$ change with the input, the coefficient of $x_{t-k}$ is a product of different matrices that depends on every token in between, so no single kernel exists and the model must run as a scan.'
    see: linear-recurrences
  - q: 'In an LSTM, the forget gate decides how much of the old cell state to keep. Which quantity plays that role in a selective SSM, and what does a large value of it do to the state?'
    a: 'The step size $\Delta_t$ does, through $\bar{A}_t = \exp(\Delta_t A)$ with $A < 0$. A large $\Delta_t$ drives $\bar{A}_t$ toward 0, so the state resets and the model focuses on the current token. A small $\Delta_t$ keeps $\bar{A}_t$ near 1, so the state persists and the current token is nearly ignored (Section 3.5.2).'
    see: lstm
---

## The problem

By 2023, attention was the dominant sequence layer (see [the Transformer](/transformer/)), and its costs were the main obstacle to very long contexts. Training compares every token with every earlier token, which is quadratic in the sequence length. Generating text needs a key and value for every earlier token in every layer, the KV cache, which grows with each token the model reads. Section 3.1 of the paper puts it this way: attention is effective because it doesn't compress the context at all, and inefficient for the same reason.

Recurrent models sit at the other end. A recurrent network keeps a fixed-size state and updates it once per token, so inference takes constant time and memory per token. Their weakness is how well that state compresses the past. Structured state-space models (SSMs, the S4 family) improved on the classic RNN by making the recurrence linear, which allows training as a convolution. That convolution requires the recurrence to be the same at every step. The paper calls this property linear time invariance (LTI) (Section 2).

An LTI model can't use the content of a token to decide what to remember. Consider the selective copying task of Section 3.1: a sequence has a few data tokens scattered among filler tokens at random positions, and the model must output the data tokens in order. The recurrence applies the same update to a filler token as to a data token, and a fixed convolution kernel can't skip a variable gap. Table 1 of the paper shows the result. With the H3 architecture, an S4 layer scores 57.0% on this task, while swapping in a selective layer (S6) scores 99.7%. With the Mamba block, the same swap goes from 56.4% to 99.8%.

The paper asks whether an SSM can select, with parameters that depend on the input, and still train fast. Section 4.2.1 reports that Mamba is the first attention-free model to match a strong Transformer recipe (Transformer++) in scaling-law experiments on the Pile, from about 125M to 1.3B parameters. Section 4.5 reports 4 to 5 times the inference throughput of a Transformer of similar size.

## Key ideas

The sections that follow use the paper's notation: sequence length $L$, $D$ channels, state size $N$, input $x$, output $y$, state $h$, step size $\Delta$, and matrices $A$, $B$, $C$. A bar marks a discretized matrix, as in the paper: $\bar{A}$, $\bar{B}$. The paper also uses $B$ for the batch size in tensor shapes (Algorithm 2). This page drops the batch axis and writes $B$ only for the input matrix. Every number comes from a script in `scripts/`. Run any script from the paper folder to reproduce the output.

### State-space models: recurrence and convolution

**Intuition.** An SSM reads a one-dimensional signal and carries a small memory, the state $h$. At each step the memory decays a little and absorbs the new input. The output is a linear read-out of the memory. Because every step is linear and identical, you can either run the loop one step at a time (cheap at inference) or unroll it into one long filter and apply it to the whole sequence at once (parallel at training).

**Math.** The continuous system maps $x(t)$ to $y(t)$ through a latent state $h(t) \in \mathbb{R}^N$ (Section 2, Equations 1a and 1b):

$$
h'(t) = A\,h(t) + B\,x(t), \qquad y(t) = C\,h(t).
$$

A discretization rule with step size $\Delta$ turns $(\Delta, A, B)$ into $(\bar{A}, \bar{B})$. The paper uses the zero-order hold (Section 2, Equation 4):

$$
\bar{A} = \exp(\Delta A), \qquad \bar{B} = (\Delta A)^{-1}\bigl(\exp(\Delta A) - I\bigr)\cdot \Delta B.
$$

The discrete model runs as a recurrence (Equations 2a and 2b) or as a global convolution (Equations 3a and 3b):

$$
h_t = \bar{A}\,h_{t-1} + \bar{B}\,x_t, \qquad y_t = C\,h_t, \qquad K = \bigl(C\bar{B},\; C\bar{A}\bar{B},\; \dots,\; C\bar{A}^{k}\bar{B},\; \dots\bigr), \qquad y = x * K.
$$

The paper uses a diagonal $A$ (Section 2, "Structure and Dimensions"), so $A$, $B$, and $C$ are each stored as $N$ numbers per channel. The SSM runs independently on each of the $D$ channels, which makes the total state $DN$ numbers.

**Tiny example, computed by `scripts/ssm_example.py`.** Take one channel with $N = 1$, $A = -1$, $B = C = 1$, and $\Delta = \ln 2 \approx 0.6931$. Then $\bar{A} = e^{-\ln 2} = 0.5$ and $\bar{B} = (0.5 - 1)/(-1) \cdot 1 = 0.5$. Feed in $x = (1, 2, 3, 4)$:

| Step $t$ | $h_t = 0.5\,h_{t-1} + 0.5\,x_t$ | Kernel $K_t = C\bar{A}^t\bar{B}$ | $y_t$ from the convolution |
|---|---|---|---|
| 0 | 0.5 | 0.5 | $0.5 \cdot 1 = 0.5$ |
| 1 | $0.25 + 1 = 1.25$ | 0.25 | $0.5 \cdot 2 + 0.25 \cdot 1 = 1.25$ |
| 2 | $0.625 + 1.5 = 2.125$ | 0.125 | $0.5 \cdot 3 + 0.25 \cdot 2 + 0.125 \cdot 1 = 2.125$ |
| 3 | $1.0625 + 2 = 3.0625$ | 0.0625 | $0.5 \cdot 4 + 0.25 \cdot 3 + 0.125 \cdot 2 + 0.0625 \cdot 1 = 3.0625$ |

Both views give the same $y$. The kernel is the same at every position, and that is what time invariance means. Exercise 1 asks you to implement Equation 4.

### Selection: parameters that depend on the input

**Intuition.** Section 3.1 argues that sequence modeling is compression: a model must squeeze its context into a state. A good compressor keeps what matters and drops the rest, so the update has to depend on the content of the token. Make the step size $\Delta$ and the matrices $B$ and $C$ functions of the current token, and the model can write a token into the state, skip it, or wipe the state.

**Math.** Algorithm 1 (S4) takes $B, C \in \mathbb{R}^{D \times N}$ and $\Delta \in \mathbb{R}^{D}$ as parameters. Algorithm 2 (S6, Section 3.2) computes them from the input $x$ of shape $(B, L, D)$:

$$
B = s_B(x) = \operatorname{Linear}_N(x), \qquad C = s_C(x) = \operatorname{Linear}_N(x), \qquad \Delta = \tau_\Delta\bigl(\text{Parameter} + s_\Delta(x)\bigr),
$$

with $s_\Delta(x) = \operatorname{Broadcast}_D(\operatorname{Linear}_1(x))$ and $\tau_\Delta = \operatorname{softplus}$. The matrix $A$ stays a plain $(D, N)$ parameter. The paper explains that selectivity in $\Delta$ already makes $(\bar{A}, \bar{B})$ selective, because $A$ acts only through $\exp(\Delta A)$ (Section 3.5.2). The recurrence now has a time index on every coefficient:

$$
h_t = \bar{A}_t\,h_{t-1} + \bar{B}_t\,x_t, \qquad y_t = C_t\,h_t.
$$

Because $\bar{A}_t$ and $\bar{B}_t$ change at each step, the kernel in Equation 3 no longer exists, and Algorithm 2 notes that the model can be computed only as a recurrence (a scan).

**The gate hidden in $\Delta$.** Theorem 1 (Section 3.5.1, Equation 5, proof in Appendix C) shows that selection generalizes RNN gating. Set $N = 1$, $A = -1$, $B = 1$, $s_\Delta = \operatorname{Linear}(x)$, and $\tau_\Delta = \operatorname{softplus}$. The ZOH formulas then give $\bar{A}_t = 1 - g_t$ and $\bar{B}_t = g_t$, so

$$
g_t = \sigma\bigl(\operatorname{Linear}(x_t)\bigr), \qquad h_t = (1 - g_t)\,h_{t-1} + g_t\,x_t.
$$

This is the leaky-integrator gate of an LSTM-style cell (see [the LSTM](/lstm/)). The paper reads $\Delta$ as a generalized gate: a large $\Delta$ resets the state and focuses on the current input, and a small $\Delta$ keeps the state and ignores the input (Section 3.5.2).

**Tiny example, computed by `scripts/selection_example.py`.** With $z = \operatorname{Linear}(x_t)$ and $\Delta = \operatorname{softplus}(z)$:

| $z$ | $\Delta$ | $\bar{A}$ | $\bar{B}$ | $\sigma(z)$ |
|---|---|---|---|---|
| $-4$ | 0.0181 | 0.9820 | 0.0180 | 0.0180 |
| $0$ | 0.6931 | 0.5000 | 0.5000 | 0.5000 |
| $4$ | 4.0181 | 0.0180 | 0.9820 | 0.9820 |

The $\bar{B}$ column equals $\sigma(z)$ and $\bar{A} = 1 - \sigma(z)$, as Theorem 1 states. Now run a toy selective-copying stream, $x = (9, 4, 9, 9)$, where 4 is the data token and the 9s are filler. A selective model that sets $z = (-8, 8, -8, -8)$ gets gates $g = (0.0003, 0.9997, 0.0003, 0.0003)$ and the states $h = (0.003, 3.9987, 4.0003, 4.0020)$. It writes the 4 and holds it. A time-invariant model has one gate for every step. With $g = 0.5$ the states are $(4.5, 4.25, 6.625, 7.8125)$, and the filler overwrites the 4.

Exercise 2 asks you to implement the selective recurrence, including the single-step form that inference uses.

### The scan: computing a time-varying recurrence in parallel

**Intuition.** The loop $h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$ looks inherently serial, because step $t$ needs step $t-1$. But each step is an affine map $h \mapsto a h + b$, and composing two affine maps gives another affine map. Composition is associative, so you can combine neighbors in pairs, then pairs of pairs, in a tree, like a prefix sum.

**Math.** Section 3.3.2 states that the selective recurrence can be parallelized with a work-efficient parallel scan (citing Blelloch 1990, Martin and Cundy 2018, and Smith et al. 2023). The paper doesn't write out the scan operator. For one state entry, write each step as the pair $(a_t, b_t) = (\bar{A}_t, \bar{B}_t x_t)$. This pair notation is this page's, not the paper's. The operator is

$$
(a_1, b_1) \bullet (a_2, b_2) = (a_2 a_1,\; a_2 b_1 + b_2),
$$

where the left pair happens first. It is associative but not commutative. The prefix $(a_1, b_1) \bullet \dots \bullet (a_t, b_t)$ has second component $h_t$ when $h_0 = 0$. A log-depth scan finishes in $\lceil \log_2 L \rceil$ rounds. Section 3.3.2 also counts the cost of the two modes: the plain recurrence uses $O(BLDN)$ FLOPs, and the convolution mode uses $O(BLD \log L)$, and the recurrence has the lower constant factor. For long sequences and a small $N$, the recurrence can use fewer FLOPs.

**Tiny example, computed by `scripts/scan_example.py`.** Use the pairs from the first example: $a_t = 0.5$ and $b_t = 0.5\,x_t$, so $b = (0.5, 1, 1.5, 2)$. A Hillis-Steele scan, where every position combines with the position $d$ steps back and $d$ doubles each round, gives:

| Round | Offset $d$ | $a$ | $b$ |
|---|---|---|---|
| start | | `[0.5, 0.5, 0.5, 0.5]` | `[0.5, 1.0, 1.5, 2.0]` |
| 1 | 1 | `[0.5, 0.25, 0.25, 0.25]` | `[0.5, 1.25, 2.0, 2.75]` |
| 2 | 2 | `[0.5, 0.25, 0.125, 0.0625]` | `[0.5, 1.25, 2.125, 3.0625]` |

The final $b$ is the sequential result from the first example. Order matters: for $p = (0.5, 1)$ and $q = (0.25, 2)$, $p \bullet q = (0.125, 2.25)$ and $q \bullet p = (0.125, 2.0)$. For $L = 1024$ the sequential loop takes 1023 dependent steps and this scan takes 10 rounds. It does more total work (9217 combines against 1023), which is why the paper cites a work-efficient scan (Blelloch's) that does $O(L)$ work. Exercise 3 uses the simpler Hillis-Steele form.

### Hardware-aware implementation

**Intuition.** On a GPU, most operations other than matrix multiplication are limited by memory bandwidth, not arithmetic (Section 3.3.2). The selective scan expands every input value into $N$ state values. If a kernel wrote those expanded tensors to GPU memory (HBM) and read them back, the memory traffic would dominate.

**Math.** The naive implementation prepares $\bar{A}, \bar{B}$ of shape $(B, L, D, N)$ in HBM, scans them, and multiplies by $C$. Appendix D replaces it with one fused kernel (Section 3.3.2 gives the overview):

1. Read $(\Delta, A, B, C)$, $O(BLD + DN)$ bytes, from HBM into fast on-chip memory (SRAM).
2. Discretize in SRAM to produce $\bar{A}, \bar{B}$ of shape $(B, L, D, N)$.
3. Run the parallel associative scan in SRAM.
4. Multiply and sum with $C$, then write only the output $y$ of shape $(B, L, D)$ back to HBM.

This cuts memory reads and writes by a factor of $N$. The paper reports that the fused scan is 20 to 40 times faster than a standard PyTorch scan and faster than FlashAttention-2 beyond a sequence length of 2K (Section 4.5). Two more techniques complete it. Sequences too long for SRAM are split into chunks, with the final state of one chunk carried into the next (Appendix D). In the backward pass, the intermediate states aren't stored but recomputed, so the layer's memory matches an optimized Transformer that uses FlashAttention (Section 3.3.2). For the attention side of that comparison, see [FlashAttention](/flashattention/) and the [roofline model](/roofline/).

**Tiny example, computed by `scripts/state_vs_kv.py`.** Take $L = 2048$ and $D = 4096$ inner channels (that is $E \cdot d_{\text{model}}$ with $E = 2$ and $d_{\text{model}} = 2048$) and $N = 16$. The input $x$ has 8,388,608 elements for one sequence. The expanded state has 134,217,728, which is $N = 16$ times more. The fused kernel never writes those 134 million values to HBM.

### The Mamba block

**Intuition.** Earlier SSM models, such as H3, alternate an SSM block with an MLP block. The paper merges the two into one homogeneous block and stacks it (Section 3.4, Figure 3). The block is a gated MLP with an extra path: a short causal convolution followed by the selective SSM on one branch, and a SiLU gate on the other.

**Math.** The block expands the model dimension $D$ by a factor $E$. Most of its parameters are in the linear projections: $2ED^2$ for the input projections and $ED^2$ for the output projection, so $3ED^2$ in total, and the SSM parameters are much smaller (Section 3.4). With $E = 2$, one Mamba block has $6D^2$ parameters. Two blocks make up $12D^2$, which equals the parameters of a Transformer's attention block plus its MLP, so the paper uses two Mamba blocks per Transformer layer when it matches sizes. Other defaults from Section 3.6: real-valued $A$ with the S4D-Real initialization $A_n = -(n+1)$ for state entry $n$, and a $\Delta$ bias initialized to the inverse softplus of a value drawn uniformly from $[0.001, 0.1]$.

The paper doesn't give the width of the causal convolution in the text. The reference code's default is 4, and `scripts/state_vs_kv.py` and Exercise 4 use that value.

### Constant memory per token at inference

**Intuition.** A Transformer decoder stores a key and a value for every earlier token, so its memory grows by a fixed amount per generated token. A Mamba layer stores only its state: $D \cdot N$ numbers for the SSM plus a short convolution buffer, whatever the context length. The paper credits the missing KV cache for the higher inference throughput, because the freed memory allows much larger batches (Section 4.5).

**Math.** For a Transformer with $n$ layers, $H$ heads of width $d_h$, and sequence length $T$, the cache holds $2\,n\,H\,d_h\,T$ numbers. A Mamba model with $n'$ layers, $d_{\text{inner}} = E\,d_{\text{model}}$, state size $N$, and convolution width $w$ holds $n'\,d_{\text{inner}}\,(N + w - 1)$ numbers. The first grows linearly in $T$ and the second doesn't depend on $T$. These formulas are this page's accounting for one sequence, not equations from the paper.

**Tiny example, computed by `scripts/state_vs_kv.py`.** Compare the 1.3B Transformer of Table 12 (24 layers, 32 heads of width 64, fp16) against a Mamba model of the same width. The Mamba sizes are this page's assumptions: 48 layers, $E = 2$, $N = 16$, convolution width 4, and fp16.

| Tokens | KV cache | Mamba state | Ratio |
|---|---|---|---|
| 1,000 | 196,608,000 bytes | 7,471,104 bytes | 26.3 |
| 10,000 | 1,966,080,000 bytes | 7,471,104 bytes | 263.2 |
| 100,000 | 19,660,800,000 bytes | 7,471,104 bytes | 2,631.6 |

At 100,000 tokens the cache is 19.66 GB for one sequence and the Mamba state is 7.47 MB. The two are equal at 38 tokens, and the cache is larger from 39 tokens on. Model weights aren't counted on either side. The state is constant, but the paper's quality claims for long context depend on how well that state compresses, not on its size alone.

Exercise 4 asks you to compute these numbers.

<!-- exercises -->

## Pitfalls

- **Treating $\bar{B}$ as $\Delta B$.** Equation 4 gives $\bar{B}$ through the full zero-order-hold formula. The two agree only for a small $\Delta$. As $\Delta$ grows, $\bar{B}$ saturates at $-B/A$ while $\Delta B$ keeps growing. Implementations of the selective scan often use $\Delta B$ in place of the exact $\bar{B}$ for simplicity, so compare with the formula before you blame a mismatch on a bug. Exercise 1 tests the exact form.
- **Assuming $A = -1$.** In Theorem 1, $\bar{B} = 1 - \bar{A}$ holds only for $A = -1$. For other values of $A$, $\bar{B} = (\bar{A} - 1)/A \cdot B$. A shortcut that passes the gating check can still fail on a general $A$.
- **Reading the output from the old state.** The output is $y_t = C_t h_t$, computed from the state after the update. Using $h_{t-1}$ delays the output by one step and passes any test that only checks shapes.
- **Making the convolution view the default.** Selection removes the convolution. If $\Delta$, $B$, and $C$ vary with the input, there is no fixed kernel. The Exercise 2 test that checks a time-invariant case against Equation 3 is a sanity check on the loop, not a recommended implementation for the selective model.
- **Combining scan pairs in the wrong order.** The scan operator is associative but not commutative. Swapping the arguments gives plausible numbers for small cases (the $b$ component of the example above differs by 0.25) and breaks as soon as the decay varies.
- **Stopping the scan rounds early.** A scan with offsets $1, 2, 4, \dots$ needs rounds until the offset reaches $L$. A loop condition of `d < L // 2` stops one round short: for $L = 4$ it runs a single round, and the last two positions are still partial sums.
- **Giving every channel its own projection of $\Delta$.** The paper projects the input to one value and broadcasts it across the $D$ channels ($\operatorname{Linear}_1$ in Section 3.2), then adds a per-channel bias through the Parameter term. Section 3.6 generalizes the 1 to a small rank $R$. A single $\Delta$ per token means all channels agree on whether to ignore a token, which is what the synthetic tasks need.
- **Counting architectural gating as selection.** Selection controls how much of the *state* carries across time steps. A multiplicative gate in the block, as in an H3 or MLP branch, doesn't act along the sequence axis, so it doesn't count (Section 4.1.1 and Appendix A). Table 1 shows the gap: gated blocks with an S4 layer stay near 57%.
- **Comparing a state to a cache without a unit.** The Mamba state counts numbers per layer, and the KV cache counts numbers per token per layer. Fix the dtype and the batch size before you compare. Training may also keep the state in higher precision than the cache.
- **Expecting the state to remember everything.** Constant memory is a compression limit. The paper's own motivation (Section 3.1) is that efficient models need small states and effective models need states that hold the relevant context, and selection is how it tries to square the two.

## Further reading

- [state-spaces/mamba](https://github.com/state-spaces/mamba) is the official code and checkpoint release from the authors.
- [Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396), by Gu, Goel, and Ré, is the S4 paper that sets up the SSM background in Section 2.
- [Transformers are SSMs](https://arxiv.org/abs/2405.21060), by Dao and Gu, is the follow-up (Mamba-2) that relates selective SSMs to attention.
- [Simplified State Space Layers for Sequence Modeling](https://arxiv.org/abs/2208.04933), by Smith, Warrington, and Linderman, introduces the S5 layer that uses a parallel scan, which Section 3.3.2 cites.
- [Prefix Sums and Their Applications](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf), by Guy Blelloch, is the reference for the work-efficient scan.
- [Parallelizing Linear Recurrent Neural Nets Over Sequence Length](https://arxiv.org/abs/1709.04057), by Martin and Cundy, applies the same scan to linear RNNs.
