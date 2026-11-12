---
id: "rnnt"
title: "Sequence Transduction with Recurrent Neural Networks"
authors: "Graves et al."
year: 2012
venue: "ICML 2012 Workshop"
arxiv: "1211.3711"
url: "https://arxiv.org/abs/1211.3711"
track: speech
tier: extended
est_hours: 5
status: full
summary: "The RNN transducer extends CTC with a second network, the prediction network, that reads the labels emitted so far. A small output layer adds the encoder vector for a frame to the prediction vector for a label prefix and normalizes the sum, which defines a probability for every move on a time-by-label lattice. A forward-backward pass over that lattice sums the probability of all alignments, and the model reaches a 23.2% phoneme error rate on TIMIT."
related_course: "speech: streaming ASR"
prereqs:
  - id: ctc
    reason: "The transducer keeps CTC's idea of an extra null output and a sum over alignments, and this page measures every change against CTC's independence assumption."
background: [joint-networks, lattice]
course_links:
  - title: "Transducers"
    url: "https://speech.ifkash.dev/lessons/034-transducers"
  - title: "Streaming encoders"
    url: "https://speech.ifkash.dev/lessons/036-streaming-encoders"
self_check:
  - q: 'CTC needs at least as many frames as labels. Take the target $(a, b)$ with $T = 3$ frames, and then a three-label target with $T = 1$ frame. How many alignments does each of CTC and the transducer allow in each case?'
    a: 'For $(a, b)$ with $T = 3$, CTC has 5 paths and the transducer has $\binom{T+U-1}{U} = \binom{4}{2} = 6$ alignments. For three labels in one frame, CTC has 0 paths, because it can emit at most one label per frame, and the transducer has exactly 1 alignment: it emits all three labels at frame 1 and then the final null. This is the first point of Section 1: CTC can only handle outputs that are not longer than the input.'
    see: ctc
  - q: 'Every alignment on the lattice has $T + U$ arrows. Why does that make the sum of $\alpha(t, u)\,\beta(t, u)$ over a diagonal $t + u = n$ equal to $\Pr(y \mid x)$ for every $n$ (Equation 19)?'
    a: 'Each arrow, null or label, raises $t + u$ by one, and the final null leaves the grid. So every alignment visits exactly one node on each diagonal from $n = 1$ to $n = T + U$. The product $\alpha(t, u)\,\beta(t, u)$ is the total probability of the alignments through $(t, u)$, so one diagonal adds up every alignment exactly once.'
    see: lattice
  - q: 'The joint $h(k, t, u) = \exp(f_t^k + g_u^k)$ needs a softmax at every node $(t, u)$. The paper says the number of exponentials drops from $O(TU)$ to $O(T + U)$. What does that trick save, and what is still large for $T = 100$, $U = 30$, $K = 40$?'
    a: 'Because $\exp(a + b) = \exp(a)\exp(b)$, you compute $\exp(f_t)$ once per frame and $\exp(g_u)$ once per prefix, and multiply. That saves exponentials, not memory. The lattice still holds one distribution of size $K + 1$ per node: $100 \cdot 31 \cdot 41 = 127{,}100$ values, against $100 \cdot 41 = 4{,}100$ for CTC.'
    see: joint-networks
---

## The problem

CTC can learn speech recognition without frame-level labels, but it has two limits. The transducer paper names both in Section 1. First, CTC defines a distribution over output sequences that are not longer than the input, so it can't handle a task such as text-to-speech, where the output is longer. Second, CTC doesn't model the interdependencies between its outputs.

The second limit is the one that matters for speech. In CTC, the probability of a path is a product of independent per-frame terms, $\prod_t y^t_{\pi_t}$. The network can use context through its hidden state, but the label chosen at one frame never feeds back into the choice at the next. Nothing in the output layer knows that "q" is usually followed by "u".

Here is a case that shows the limit. Two frames carry identical acoustic evidence for the labels $a$ and $b$. A language model would say that $ab$ is a likely sequence and $ba$ is not. CTC can't express that. Its per-frame distribution is $(0.25, 0.375, 0.375)$ for blank, $a$, and $b$, and it gives $p(ab \mid x) = p(ba \mid x) = 0.1406$, because the same distribution applies in both frames (computed by `scripts/ctc_compare.py`). The transducer adds a network that reads the labels emitted so far. In the toy of this page, that network makes $b$ four times as likely after an $a$, and the transducer gives $\Pr(ab \mid x) = 0.0409$ and $\Pr(ba \mid x) = 0.0102$, a ratio of 4.000.

The paper's answer is an RNN transducer. It keeps CTC's null output and its sum over alignments, and it adds two things: a prediction network that sees the previous labels, and the freedom to emit several labels at one frame. The paper then tests it on TIMIT phoneme recognition (Section 3). Its transducer reaches a 23.2% phoneme error rate, against 25.5% for a CTC network. The standalone prediction network misclassifies 72.9% of the next phonemes (Section 3.3, Table 1). The paper notes that 23.2% is among the lowest TIMIT results on record, against a benchmark of 20.5%, and that it is the best the authors know of for a recurrent network. It also calls the gain over CTC slight, and suggests a reason: the TIMIT transcriptions give the prediction network only about 150,000 labels to learn from.

The paper is a short workshop paper with no section on streaming. Its transcription network is bidirectional, so the system as published reads the whole input before it produces output. The reputation of the transducer for streaming comes from later work that swaps in a causal encoder (see [streaming encoders](https://speech.ifkash.dev/lessons/036-streaming-encoders)).

## Key ideas

The sections below use the paper's notation: input $x = (x_1, \dots, x_T)$, output $y = (y_1, \dots, y_U)$ over $K$ labels, the null symbol $\varnothing$, extended space $\bar{\mathcal Y} = \mathcal Y \cup \{\varnothing\}$, transcription vectors $f_t$, prediction vectors $g_u$, the output density $h(k, t, u)$, the probabilities $y(t, u)$ and $\varnothing(t, u)$, and the forward and backward variables $\alpha(t, u)$ and $\beta(t, u)$. Two conventions differ from the paper. The code puts the null at index 0 and the labels at $1, \dots, K$, and it indexes arrays from 0, while the paper starts $t$ at 1. Every number comes from a script in `scripts/`. Run any script from the paper folder to reproduce its output.

### Two networks, one grid

**Intuition.** A transcription network $F$ reads the audio and produces one vector per frame. A prediction network $G$ reads the labels so far and produces one vector per prefix. Every pair (frame, prefix) is a node on a grid. At each node, you combine the two vectors to get a distribution over what happens next: emit the next label, or emit nothing and move to the next frame.

**Math.** The transcription network is a bidirectional RNN, and the paper uses LSTM layers. It outputs $f = (f_1, \dots, f_T)$, and each $f_t$ has $K + 1$ elements, one per symbol of $\bar{\mathcal Y}$ (Section 2.2, Equations 9 to 11). The prediction network sees the output sequence with a null prepended, $\hat y = (\varnothing, y_1, \dots, y_U)$, encoded as one-hot vectors, with the null encoded as all zeros. It outputs $g = (g_0, \dots, g_U)$ (Section 2.1, Equations 2 and 3):

$$
h_u = \mathcal H(W_{ih}\hat y_u + W_{hh}h_{u-1} + b_h), \qquad g_u = W_{ho}h_u + b_o
$$

So $g_u$ depends on $y_1, \dots, y_u$ and on nothing else. The two vectors combine by addition in the log domain (Section 2.3, Equations 12 and 13):

$$
h(k, t, u) = \exp\bigl(f_t^k + g_u^k\bigr), \qquad \Pr(k \mid t, u) = \frac{h(k, t, u)}{\sum_{k' \in \bar{\mathcal Y}} h(k', t, u)}
$$

The paper keeps only two of these probabilities per node (Equations 14 and 15):

$$
y(t, u) \equiv \Pr(y_{u+1} \mid t, u), \qquad \varnothing(t, u) \equiv \Pr(\varnothing \mid t, u)
$$

The joint is a sum followed by a softmax. It has no hidden layer. Later systems put a small feed-forward network on the combined vector; see the [joint network](https://speech.ifkash.dev/lessons/034-transducers) in the course lesson.

**Tiny example, computed by `scripts/lattice_example.py`.** Take $K = 2$ labels ($a$ and $b$), $T = 3$ frames, and the target $y = (a, b)$, so $U = 2$. To keep the arithmetic exact, store $\exp(f_t)$ and $\exp(g_u)$ as small integers. The columns are null, $a$, $b$.

| | null | $a$ | $b$ |
|---|---|---|---|
| $\exp(f_1)$ | 1 | 3 | 1 |
| $\exp(f_2)$ | 1 | 1 | 3 |
| $\exp(f_3)$ | 2 | 1 | 1 |
| $\exp(g_0)$ | 1 | 1 | 1 |
| $\exp(g_1)$ | 1 | 1 | 2 |
| $\exp(g_2)$ | 4 | 1 | 1 |

At node $(t=2, u=1)$, multiply the rows $\exp(f_2) = (1, 1, 3)$ and $\exp(g_1) = (1, 1, 2)$ element by element. You get $h = (1, 1, 6)$, which sums to 8. So $\Pr(\cdot \mid 2, 1) = (0.125, 0.125, 0.75)$. At $u = 1$ the next label is $y_2 = b$, so $\varnothing(2, 1) = 0.125$ and $y(2, 1) = 0.75$. The full tables are as follows.

| $t$ | $\varnothing(t, 0)$ | $\varnothing(t, 1)$ | $\varnothing(t, 2)$ | $y(t, 0)$ | $y(t, 1)$ |
|---|---|---|---|---|---|
| 1 | 0.2000 | 0.1667 | 0.5000 | 0.6000 | 0.3333 |
| 2 | 0.2000 | 0.1250 | 0.5000 | 0.2000 | 0.7500 |
| 3 | 0.5000 | 0.4000 | 0.8000 | 0.2500 | 0.4000 |

All 9 nodes sum to 1 over their $K + 1 = 3$ symbols. Column $u = 2$ has no $y$, because after $y_U$ no label is left to emit.

Exercise 1 asks you to build these two tables from $f$, $g$, and $y$.

### Alignments and the lattice

**Intuition.** An alignment records what the model did at each step: it emitted a label, or it emitted nothing. The sequence $(y_1, \varnothing, \varnothing, y_2, \varnothing, y_3)$ means the same as $(y_1, y_2, y_3)$. As on the CTC page, many alignments collapse to one output. The difference is the shape of the moves. In CTC, every frame emits exactly one symbol, so a label takes up a frame. In the transducer, a label arrow doesn't use up a frame. The model can emit several labels, one after another, at the same frame, and a frame can emit none.

**Math.** The map $\mathcal B : \bar{\mathcal Y}^* \to \mathcal Y^*$ removes the nulls, and the probability of an output sums over every alignment that collapses to it (Section 2, Equation 1):

$$
\Pr(y \in \mathcal Y^* \mid x) = \sum_{a \in \mathcal B^{-1}(y)} \Pr(a \mid x)
$$

The alignments are the paths on the lattice of Figure 1. The path starts at $(1, 0)$ and leaves a node $(t, u)$ in one of two ways. The null arrow has probability $\varnothing(t, u)$ and moves to $(t+1, u)$. The label arrow has probability $y(t, u)$ and moves to $(t, u+1)$. The path ends at $(T, U)$ with a final null. An alignment has $T$ nulls and $U$ labels, so $T + U$ arrows, and the number of alignments is $\binom{T+U-1}{U}$, because the last arrow is fixed. The probability of an alignment is the product of the arrows it takes. Because $\Pr(k \mid t, u)$ is defined for every output length, the same lattice can be drawn for any $y$, and the model defines a distribution over all output sequences (Section 2.3).

**Tiny example, computed by `scripts/ctc_compare.py`.** For the target $(a, b)$ and $K = 2$, the two models count paths as follows.

| $T$ | CTC paths | Transducer alignments |
|---|---|---|
| 1 | 0 | 1 |
| 2 | 1 | 3 |
| 3 | 5 | 6 |
| 4 | 15 | 10 |
| 5 | 35 | 15 |

Two things to notice. With $T = 1$, CTC has no path at all, because two labels need two frames. The transducer emits both labels at frame 1. And the transducer doesn't always have fewer alignments: from $T = 4$ on, CTC has more for this target, but the count of the transducer keeps growing too. The saving that matters comes from the dynamic program. For $T = 100$, $U = 30$, the transducer has $\binom{129}{30} \approx 2.0 \times 10^{29}$ alignments, and the lattice has $T(U+1) = 3{,}100$ nodes. The forward algorithm visits each node once.

At each node, the arrows for the labels other than $y_{u+1}$ lead to other targets' lattices. For this target, only the null arrow and the arrow for $y_{u+1}$ stay on the lattice, and their probabilities add up to less than 1 in general.

### The forward and backward variables

**Intuition.** Summing over $\binom{T+U-1}{U}$ alignments directly is hopeless for real lengths. But alignments share prefixes. If you store, for each node, the total probability of arriving there, you extend those totals one node at a time. This is the same dynamic program as in the CTC paper, on a different graph.

**Math.** The forward variable $\alpha(t, u)$ is the probability of outputting $y_{1:u}$ during frames $f_{1:t}$. A node is reached from the left by a null, or from below by a label (Section 2.4, Equation 16):

$$
\alpha(t, u) = \alpha(t-1, u)\,\varnothing(t-1, u) + \alpha(t, u-1)\,y(t, u-1), \qquad \alpha(1, 0) = 1
$$

The total probability is the forward variable at the last node, times the final null (Equation 17):

$$
\Pr(y \mid x) = \alpha(T, U)\,\varnothing(T, U)
$$

The backward variable $\beta(t, u)$ is the probability of outputting $y_{u+1:U}$ during $f_{t:T}$ (Equation 18):

$$
\beta(t, u) = \beta(t+1, u)\,\varnothing(t, u) + \beta(t, u+1)\,y(t, u), \qquad \beta(T, U) = \varnothing(T, U)
$$

The product $\alpha(t, u)\,\beta(t, u)$ is the probability of emitting the whole sequence with $y_u$ emitted at step $t$. Every alignment crosses each anti-diagonal of the lattice exactly once, so the sum over a diagonal is the same for every $n$ (Section 2.5, Equation 19):

$$
\Pr(y^* \mid x) = \sum_{(t, u) : t + u = n} \alpha(t, u)\,\beta(t, u), \qquad \forall n : 1 \le n \le U + T
$$

**Tiny example, computed by `scripts/forward_example.py`.** Use the tables from the first idea. The forward variables, with rows $t = 1, 2, 3$ and columns $u = 0, 1, 2$, are:

| $t$ | $u=0$ | $u=1$ | $u=2$ |
|---|---|---|---|
| 1 | 1.0000 | 0.6000 | 0.2000 |
| 2 | 0.2000 | 0.1400 | 0.2050 |
| 3 | 0.0400 | 0.0275 | 0.1135 |

Check $\alpha(2, 1)$ by hand. The null arrow from $(1, 1)$ contributes $0.6000 \cdot 0.1667 = 0.1000$, and the label arrow from $(2, 0)$ contributes $0.2000 \cdot 0.2000 = 0.0400$. They sum to 0.1400. Equation 17 gives $\Pr(y \mid x) = 0.1135 \cdot 0.8 = 0.0908$.

Now enumerate all $\binom{4}{2} = 6$ alignments. In the strings below, `e` is a label arrow and `-` is a null arrow, so `ee---` emits $a$ and $b$ at frame 1 and then takes the three nulls.

| Alignment | Probability |
|---|---|
| `ee---` | 0.04000 |
| `e-e--` | 0.03000 |
| `e--e-` | 0.00400 |
| `-ee--` | 0.01200 |
| `-e-e-` | 0.00160 |
| `--ee-` | 0.00320 |

They sum to 0.0908, the same as the forward pass. The backward variables are:

| $t$ | $u=0$ | $u=1$ | $u=2$ |
|---|---|---|---|
| 1 | 0.0908 | 0.1233 | 0.2000 |
| 2 | 0.0840 | 0.3400 | 0.4000 |
| 3 | 0.0800 | 0.3200 | 0.8000 |

The corner is $\beta(3, 2) = \varnothing(3, 2) = 0.8$, and $\beta(1, 0) = 0.0908$ is the total probability. On each of the five diagonals, $\sum \alpha\,\beta$ equals 0.0908, as Equation 19 predicts.

Exercises 2 and 3 ask you to implement the forward recursion with a brute-force check, then the backward variables and the diagonal identity.

### The training gradient

**Intuition.** The loss is the negative log probability of the target. Each arrow's probability enters the total through all the alignments that use it, and the forward and backward variables count those alignments. The rest is the softmax.

**Math.** The loss is $L = -\ln \Pr(y^* \mid x)$ (Section 2.5). Differentiating Equations 16 to 19 gives the gradient with respect to the lattice probabilities (Equation 20):

$$
\frac{\partial L}{\partial \Pr(k \mid t, u)} = -\frac{\alpha(t, u)}{\Pr(y^* \mid x)}
\begin{cases}
\beta(t, u+1) & \text{if } k = y_{u+1} \\
\beta(t+1, u) & \text{if } k = \varnothing \\
0 & \text{otherwise}
\end{cases}
$$

At the last frame, read $\beta(T+1, U)$ as 1 and $\beta(T+1, u)$ as 0 for $u < U$, which agrees with the initial condition $\beta(T, U) = \varnothing(T, U)$. The gradient passes to the transcription and prediction vectors by summing over the lattice (Equations 21 and 22), with the softmax derivative $\partial \Pr(k' \mid t, u) / \partial f_t^k = \Pr(k' \mid t, u)\,[\delta_{kk'} - \Pr(k \mid t, u)]$:

$$
\frac{\partial L}{\partial f_t^k} = \sum_{u=0}^{U} \sum_{k' \in \bar{\mathcal Y}} \frac{\partial L}{\partial \Pr(k' \mid t, u)} \frac{\partial \Pr(k' \mid t, u)}{\partial f_t^k}, \qquad
\frac{\partial L}{\partial g_u^k} = \sum_{t=1}^{T} \sum_{k' \in \bar{\mathcal Y}} \frac{\partial L}{\partial \Pr(k' \mid t, u)} \frac{\partial \Pr(k' \mid t, u)}{\partial g_u^k}
$$

Backpropagation through time then carries the two gradients into the weights of each network separately. The sum in Equation 21 runs over the whole column of the lattice, so a frame vector $f_t$ is shared by every prefix, and each prediction vector $g_u$ is shared by every frame. The paper adds a cost note. A separate softmax at every node is expensive because of the exponentials, and $\exp(a + b) = \exp(a)\exp(b)$ lets you precompute $\exp(f_t)$ and $\exp(g_u)$ once, which cuts the number of exponentials from $O(TU)$ to $O(T + U)$ (Section 2.5).

**Tiny example, computed by `scripts/forward_example.py`.** For the same toy, the gradient with respect to $f$ (rows $t = 1, 2, 3$, columns null, $a$, $b$) is:

| $t$ | null | $a$ | $b$ |
|---|---|---|---|
| 1 | -0.4439 | 0.3577 | 0.0862 |
| 2 | -0.4459 | 0.0656 | 0.3803 |
| 3 | -0.1436 | 0.0930 | 0.0507 |

Each row sums to 0, because adding a constant to all the logits of a frame doesn't change a softmax. The script compares all 18 entries of $\partial L / \partial f$ and $\partial L / \partial g$ with a central finite difference of $-\ln \Pr(y \mid x)$, and the largest difference is $2.1 \times 10^{-10}$. For the toy sizes in this page, the exponential count is small, but for $T = 100$, $U = 30$, $K = 40$ the script counts 127,100 exponentials for a softmax per node and 5,371 when $\exp(f)$ and $\exp(g)$ are computed once.

### The prediction network relaxes CTC's independence

**Intuition.** In CTC the per-frame distribution $y^t$ is fixed before any label is chosen. In the transducer, the distribution at a node depends on which labels have been emitted, because $g_u$ is a function of $y_{1:u}$. Two nodes in the same frame, $(t, u)$ and $(t, u+1)$, get different distributions, even though $f_t$ is the same.

**Math.** The paper states the contrast in Section 1: CTC doesn't model the interdependencies between its outputs, and the transducer models input-output and output-output dependencies jointly. The prediction network works like a next-step-prediction RNN that also has the option of predicting a null (Section 2.1). It reads labels only. The one null it sees is the zero vector at the start, so its state doesn't change when the model advances a frame, and the model attaches the prediction vector $g_u$ to every frame at which the prefix has length $u$.

**Tiny example, computed by `scripts/ctc_compare.py`.** Use $T = 2$ frames with identical evidence, $\exp(f_t) = (2, 3, 3)$ for null, $a$, $b$. For CTC, the per-frame distribution is $(0.25, 0.375, 0.375)$, and both $ab$ and $ba$ have exactly one path, so $p(ab \mid x) = p(ba \mid x) = 0.375^2 = 0.1406$. For the transducer, give the prediction network a hand-made rule: the weights are $(1, 1, 1)$ at the start, and $(1, 1, 4)$ after an $a$, so $b$ is four times as likely after $a$. Then $\Pr(ab \mid x) = 0.0409$ and $\Pr(ba \mid x) = 0.0102$. The ratio is 4.000, the factor in the rule. A trained network learns such preferences from data, which is how the transducer picks up label statistics, much as a language model does.

### Decoding and the beam search

**Intuition.** Training sums over alignments. At test time, you want the most probable output sequence. The paper warns that this is much harder than scoring one sequence (Section 2.6). The prediction vector $g$ depends on all previous outputs, so you can't share work between hypotheses the way an HMM Viterbi pass does.

**Math.** The paper uses a fixed-width beam search through the tree of output sequences. Algorithm 1 keeps a set $B$ of hypotheses with probabilities. For each frame $t$, it merges hypotheses that reach the same sequence, then repeatedly takes the most probable one, $y^*$. Two things happen to $y^*$. Its probability times $\Pr(\varnothing \mid y^*, t)$ goes to $B$, which sends it to the next frame. Each one-label extension $y^* + k$, with probability $\Pr(y^*)\Pr(k \mid y^*, t)$, goes back into the pool for the same frame. The loop stops when $B$ holds $W$ hypotheses more probable than the best one left in the pool. At the end, the algorithm returns the hypothesis in $B$ with the highest length-normalized log probability, $\log \Pr(y) / |y|$. The paper reports that this normalization matters for performance, because without it the search favors short sequences. For LSTMs, you store both the hidden vector and the state vector for each hypothesis, so each extension costs one step of Equation 2 (Section 2.6).

A cheaper decoder takes the argmax at each node. You emit its label and stay at the frame, or you take the null and move on. That is greedy decoding. The paper doesn't use it, and a decoder must cap the number of labels per frame, because nothing in the lattice stops a hypothesis from emitting forever.

**Tiny example, computed by `scripts/decode_example.py`.** Take $\exp(f) = ((4, 4, 4), (6, 2, 1), (4, 4, 2))$, with $T = 3$. The prediction network is a lookup that depends on the last label: the weights are $(5, 4, 1)$ at the start, $(5, 1, 1)$ after $a$, and $(5, 1, 4)$ after $b$. With an empty prefix, the distribution at each frame is as follows.

| $t$ | null | $a$ | $b$ |
|---|---|---|---|
| 1 | 0.5000 | 0.4000 | 0.1000 |
| 2 | 0.7692 | 0.2051 | 0.0256 |
| 3 | 0.5263 | 0.4211 | 0.0526 |

The null wins at every frame, so greedy decoding returns the empty sequence, with $\Pr = 0.2024$. But the sequence $a$ collects probability from every alignment that emits one $a$ at any of the three frames, and it has $\Pr(a \mid x) = 0.3961$. The sequence $a$ is the exact mode. The next sequences are the empty one (0.2024), $aa$ (0.1059), and $ab$ (0.0458). This is the transducer version of the CTC best-path failure. Beam search finds the mode. With $W = 1$, it returns the empty sequence like greedy decoding. With $W = 2$ and $W = 4$, it returns $a$ with the exact probability 0.3961. The sum of $\Pr(y \mid x)$ over all sequences with up to 9 labels is 0.9993, which agrees with the claim that the lattice defines a distribution over all output lengths.

Exercise 4 asks you to implement greedy decoding with a label cap.

<!-- exercises -->

## Pitfalls

- **Confusing the two coordinates.** $t$ runs over frames and $u$ over the output prefix. The lattice has $T$ columns for $t = 1, \dots, T$ and $U + 1$ rows for $u = 0, \dots, U$. The vector $g_u$ exists for $u = 0$ too, because the prediction network reads the null that the paper prepends to the output.
- **Off-by-one in $y(t, u)$.** At node $(t, u)$, the label arrow emits $y_{u+1}$, not $y_u$. In 0-based code, `emit[t, u]` reads label `y[u]`, and `emit` has one fewer column than `null`.
- **Forgetting the final null.** $\Pr(y \mid x) = \alpha(T, U)\,\varnothing(T, U)$. Without the factor, you count alignments that never finish frame $T$. The table in this page has $\alpha(3, 2) = 0.1135$ but $\Pr(y \mid x) = 0.0908$.
- **Counting alignments that end on a label.** Every alignment has exactly $T$ nulls and ends with a null. There are $\binom{T+U-1}{U}$ of them, not $\binom{T+U}{U}$.
- **Reading the joint as a product of probabilities.** Equation 12 adds $f_t^k$ and $g_u^k$ before the exponential, and Equation 13 normalizes the sum. If your two networks already output softmax probabilities, take their logs before you add.
- **Expecting a smaller lattice than CTC.** The transducer lattice has $T(U+1)$ nodes, against $T(2U+1)$ cells in CTC's forward table. But each node holds a distribution over $K + 1$ symbols, and the joint tensor has $T(U+1)(K+1)$ values, against $T(K+1)$ for CTC. For $T = 100$, $U = 30$, $K = 40$, that is 127,100 against 4,100, a factor of $U + 1 = 31$. This memory cost is the usual obstacle in practice. The paper's trick of precomputing $\exp(f_t)$ and $\exp(g_u)$ reduces the exponentials, not this tensor.
- **Reading Algorithm 1 literally.** In the pseudocode, $\Pr(y^*)$ is multiplied by $\Pr(\varnothing \mid y, t)$ before the extensions $\Pr(y^* + k)$ are computed. If you follow that order, every extension is also scaled by the null probability. The script uses the probability from before the null step for the extensions, and it keeps the null mass only for the hypothesis that moves to the next frame. This page also adds the probabilities when two paths reach the same sequence. Check any implementation against enumeration on a toy, as the script does.
- **Dropping the length normalization.** Without the division by $|y|$ in the last line of Algorithm 1, short hypotheses win too often (Section 2.6). The paper doesn't say how to score the empty sequence, where $|y| = 0$. The script divides by $\max(|y|, 1)$.
- **Letting greedy decoding run forever.** Nothing in the lattice limits the labels per frame. Cap them, as in exercise 4, or a decoder that likes one label can loop.
- **Assuming the paper's model streams.** The transcription network in the paper is bidirectional, so each $f_t$ depends on the whole input (Section 2.2). The prediction network and the monotone lattice allow streaming, but you need a causal encoder to get it.
- **Underflow.** Products of arrow probabilities shrink with every step. The paper doesn't discuss numerical issues. The scripts here use plain probabilities on toy sizes. A practical implementation works in log space with a log-sum-exp at every node.
- **Expecting a large gain from the prediction network on a small corpus.** On TIMIT, the paper finds that the transducer's advantage over CTC is slight (23.2% against 25.5%), and the standalone prediction network misclassifies 72.9% of the next labels (Section 3.3). The paper suspects that about 150,000 phoneme labels are too few, and it suggests pretraining the prediction network on target-only data.

## Further reading

- [`torchaudio.functional.rnnt_loss`](https://pytorch.org/audio/stable/generated/torchaudio.functional.rnnt_loss.html) is the PyTorch implementation of the transducer loss. It takes logits of shape (batch, time, target length plus one, symbols), the joint tensor of this page.
- [Streaming End-to-End Speech Recognition for Mobile Devices](https://arxiv.org/abs/1811.06621), by He and others, runs an RNN transducer with a causal encoder on a phone.
- [Sequence Modeling With CTC](https://distill.pub/2017/ctc/), by Awni Hannun, illustrates the CTC lattice that the transducer builds on.
- [CTC](/ctc/), the prerequisite on this site, covers the blank symbol, the collapse map, and the forward-backward algorithm.
- [Supervised Sequence Labelling with Recurrent Neural Networks](https://www.cs.toronto.edu/~graves/preprint.pdf), by Alex Graves, covers RNN sequence labelling, including CTC. It predates the transducer.
