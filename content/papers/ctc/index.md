---
id: "ctc"
title: "Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks"
authors: "Graves et al."
year: 2006
venue: "ICML 2006"
arxiv: null
url: "https://www.cs.toronto.edu/~graves/icml_2006.pdf"
track: speech
tier: core
est_hours: 6
status: full
summary: "CTC trains a recurrent network on label sequences without frame-level alignments. The network emits a softmax over the labels plus a blank at every frame, a many-to-one map removes repeats and blanks, and the probability of a transcript is the sum over every frame-level path that maps to it. A forward-backward dynamic program computes that sum and its gradient in time linear in the sequence length."
related_course: "speech: sequence losses"
prereqs:
  - id: lstm
    reason: "The experiments in Section 5 use a bidirectional LSTM to produce the per-frame outputs, and CTC relies on that recurrent state to carry context from one frame to the next."
background: [dynamic-programming, forward-backward-algorithm, alignment]
course_links:
  - title: "CTC: many paths, one label"
    url: "https://speech.ifkash.dev/lessons/028-ctc-many-paths-one-label"
  - title: "Computing CTC efficiently"
    url: "https://speech.ifkash.dev/lessons/029-ctc-forward-backward"
  - title: "Decoding CTC"
    url: "https://speech.ifkash.dev/lessons/030-decoding-ctc"
self_check:
  - q: 'A CTC network emits $T = 3$ frames over the labels $\{a, b\}$ plus a blank. List the paths that collapse to $ab$. How many are there?'
    a: 'There are five: `-ab`, `a-b`, `aab`, `ab-`, and `abb`, where `-` is the blank. Each one passes through $a$ and then $b$ with at most one repeat or blank between them. `a-b` and `aab` both collapse to $ab$, because repeats merge and blanks drop out.'
    see: alignment
  - q: 'For $T = 50$ frames and a five-label transcript, there are about $2.9 \times 10^{10}$ paths that collapse to it. How many cells does the forward table in Section 4.1 hold, and why can it replace the sum over paths?'
    a: 'The table has $T \times |l''| = 50 \times 11 = 550$ cells, because $l''$ has $2 \cdot 5 + 1 = 11$ positions. Each cell $\alpha_t(s)$ is the total probability of every path prefix that collapses to $l_{1:s}$, and it depends only on three cells from the previous frame. Many paths share prefixes, so the program computes each shared sum once.'
    see: dynamic-programming
  - q: 'Equation 2 assumes the outputs at different times are conditionally independent given the network state. Why does a bidirectional LSTM still produce outputs that use context from the whole utterance?'
    a: 'The independence is conditional on the internal state of the network. The hidden state at frame $t$ already summarizes the inputs before $t$ (and, in a bidirectional network, after $t$), so $y^t$ depends on the whole input $x$. What the assumption removes is feedback from the output layer, so a frame cannot condition on the labels that the network emitted earlier (Section 3.1).'
    see: lstm
---

## The problem

A speech recognizer receives a waveform cut into 10 ms frames and must output a string such as "the sound of". The training data is the audio plus its transcript. It doesn't say which frame belongs to which letter or phoneme. The paper's introduction names the difficulty: standard network objectives are defined separately for each frame, so the training data must be pre-segmented, and the outputs must be post-processed into a label sequence (Section 1).

Before this paper, the best use of recurrent networks for this task was the hybrid HMM-RNN system. An HMM finds the segmentation, and the network classifies each frame. That design inherits the HMM's drawbacks and, in the paper's words, doesn't exploit the full potential of RNNs (Section 1).

Consider why segmentation is the hard part. Suppose an utterance has $T = 100$ frames and three phonemes. A framewise network needs one label for every frame, so someone must choose the two boundaries. There are 4,851 ways to place them (computed by `scripts/collapse_example.py`), and for phonemes that blend into each other, no one boundary is correct. Figure 1 of the paper shows the cost: a framewise network is penalized for putting a phoneme boundary in the wrong place, even when it predicts the right phoneme.

CTC removes the boundaries from the problem. The network emits one distribution per frame, and the loss sums over every way to line the transcript up with the frames. On TIMIT, a 100-block bidirectional LSTM trained this way reaches a label error rate (Section 2.1, Equation 1) of 30.51% with prefix search decoding and 31.47% with best path decoding. A context-dependent HMM gets 35.21%, and the HMM-RNN hybrid with the same network gets 33.84% (Section 5.3, Table 1).

The paper has no arXiv entry, so this page cites the ICML 2006 PDF by section and equation. Equation 1 as printed writes $\operatorname{ED}(h(x))$ and omits the target $z$ as a second argument. The text around it makes the intent clear: the edit distance between the classification and the target.

## Key ideas

The sections below use the paper's notation: input sequence $x$ of length $T$, target labelling $l = (l_1, \dots, l_U)$ over the alphabet $L$ with $U \le T$, extended alphabet $L' = L \cup \{\text{blank}\}$, a path $\pi \in L'^T$, the map $\mathcal{B}$, network outputs $y^t_k$, forward variables $\alpha_t(s)$, and backward variables $\beta_t(s)$. Two conventions differ from the paper. The code and the tables here write the blank as `-` or as index 0, and the code indexes time and positions from 0, while the paper starts at 1. Every number comes from a script in `scripts/`. Run any script from the paper folder to reproduce the output.

### The blank symbol and the collapse map

**Intuition.** The network can't say "nothing here" unless it has a symbol for it. A speech frame between two phonemes, or inside a long one, has no new label to report. CTC adds a blank to the alphabet. A frame that emits the blank says "no new label yet". Merging repeated symbols turns a long phoneme that spans many frames into one label, and the blank lets a transcript contain the same label twice in a row.

**Math.** The softmax layer has $|L| + 1$ units, and $y^t_k$ is the probability of symbol $k$ at time $t$ (Section 3.1). The probability of a whole path is the product over frames (Section 3.1, Equation 2):

$$
p(\pi \mid x) = \prod_{t=1}^{T} y^t_{\pi_t}, \qquad \forall \pi \in L'^T
$$

The map $\mathcal{B}: L'^T \to L^{\le T}$ removes repeated labels first and blanks second. The paper's example is $\mathcal{B}(a{-}ab{-}) = \mathcal{B}({-}aa{-}{-}abb) = aab$. The probability of a labelling sums over every path that maps to it (Section 3.1, Equation 3):

$$
p(l \mid x) = \sum_{\pi \in \mathcal{B}^{-1}(l)} p(\pi \mid x)
$$

**Tiny example, computed by `scripts/collapse_example.py`.** Take $T = 3$ and the symbols blank, $a$, $b$, with this output matrix. Each row is a softmax output, so it sums to 1.

| $t$ | blank | $a$ | $b$ |
|---|---|---|---|
| 1 | 0.4 | 0.5 | 0.1 |
| 2 | 0.3 | 0.4 | 0.3 |
| 3 | 0.5 | 0.2 | 0.3 |

Five of the $3^3 = 27$ paths collapse to $ab$:

| Path | $p(\pi \mid x)$ |
|---|---|
| `-ab` | 0.0480 |
| `a-b` | 0.0450 |
| `aab` | 0.0600 |
| `ab-` | 0.0750 |
| `abb` | 0.0450 |

They sum to $p(ab \mid x) = 0.2730$. Check one by hand: $y^1_{-} \cdot y^2_a \cdot y^3_b = 0.4 \cdot 0.4 \cdot 0.3 = 0.048$.

The repeat rule shows up in $aa$. Only `a-a` collapses to it, because `aaa` and `aa-` merge into a single $a$. So $p(aa \mid x) = 0.5 \cdot 0.3 \cdot 0.2 = 0.0300$. A labelling with adjacent repeats therefore needs extra frames: `cat` needs 3, `aa` needs 3, `hello` needs 6, and `aaa` needs 5. The nine distinct labellings that the 27 paths produce have probabilities that add to 1.0000, which is why Equation 3 defines a proper distribution.

Exercise 1 asks you to implement $\mathcal{B}$ and compute Equation 3 by enumeration.

### Summing over alignments with dynamic programming

**Intuition.** Enumerating paths costs $K^T$. But many paths share prefixes. If you track, for every prefix of the transcript, the total probability of all the path prefixes that produce it, you can extend those totals one frame at a time. That is the same idea as the forward algorithm for HMMs (Section 4.1).

**Math.** Pad the labelling with blanks at both ends and between every pair of labels to get $l'$, of length $2|l| + 1$. The forward variable is the total probability of the prefix $l_{1:s}$ at time $t$ (Section 4.1, Equation 5):

$$
\alpha_t(s) \stackrel{\text{def}}{=} \sum_{\pi \in L'^T:\ \mathcal{B}(\pi_{1:t}) = l_{1:s}} \ \prod_{t'=1}^{t} y^{t'}_{\pi_{t'}}
$$

The initialization is $\alpha_1(1) = y^1_b$, $\alpha_1(2) = y^1_{l_1}$, and $\alpha_1(s) = 0$ for $s > 2$. The recursion is (Section 4.1, Equations 6 and 7):

$$
\alpha_t(s) =
\begin{cases}
\bar\alpha_t(s)\, y^t_{l'_s} & \text{if } l'_s = b \text{ or } l'_{s-2} = l'_s \\
\bigl(\bar\alpha_t(s) + \alpha_{t-1}(s-2)\bigr)\, y^t_{l'_s} & \text{otherwise}
\end{cases}
\qquad
\bar\alpha_t(s) \stackrel{\text{def}}{=} \alpha_{t-1}(s) + \alpha_{t-1}(s-1)
$$

A state can stay, advance by one, or skip a blank to the next label. The skip is forbidden onto a blank, and between two equal labels, because skipping would merge the pair. The probability of the labelling adds the last two states, with and without the final blank (Section 4.1, Equation 8):

$$
p(l \mid x) = \alpha_T(|l'|) + \alpha_T(|l'| - 1)
$$

**Tiny example, computed by `scripts/forward_example.py`.** For $l = ab$, $l' = (-, a, -, b, -)$ has length 5. The forward table has one row per frame:

| $t$ | $s=1$ (`-`) | $s=2$ (`a`) | $s=3$ (`-`) | $s=4$ (`b`) | $s=5$ (`-`) |
|---|---|---|---|---|---|
| 1 | 0.4000 | 0.5000 | 0 | 0 | 0 |
| 2 | 0.1200 | 0.3600 | 0.1500 | 0.1500 | 0 |
| 3 | 0.0600 | 0.0960 | 0.2550 | 0.1980 | 0.0750 |

Equation 8 gives $p(ab \mid x) = 0.0750 + 0.1980 = 0.2730$, the same number that the five-path enumeration produced. Check $\alpha_2(4)$ by hand: it adds $\alpha_1(4) = 0$, $\alpha_1(3) = 0$, and the skip $\alpha_1(2) = 0.5$, then multiplies by $y^2_b = 0.3$, which gives 0.15. For $l = aa$, the skip $\alpha_{t-1}(2) \to \alpha_t(4)$ is blocked, and the table ends with $\alpha_3(4) = 0.0300$, matching $p(aa \mid x)$.

The saving grows quickly. With $|L| = 5$ labels, $T = 50$ frames, and a five-label transcript of distinct labels, the script counts about $2.9 \times 10^{10}$ paths. The forward table holds $50 \times 11 = 550$ cells.

**Rescaling.** The products in $\alpha$ shrink with every frame. Section 4.1 rescales each frame by $C_t = \sum_s \alpha_t(s)$ and states that $\ln p(l \mid x) = \sum_{t=1}^{T} \ln C_t$. That identity needs the rule from the note after Equation 7: set $\alpha_t(s) = 0$ for every state that can no longer finish in the frames that remain ($s < |l'| - 2(T-t) - 1$). Without that, the last frame's sum includes states that don't belong to $l$. The script shows the effect. With $T = 400$, 20 symbols with equal probability, and a five-label transcript, the unscaled $p(l \mid x)$ underflows to exactly 0 in float64, while the rescaled recursion returns $\ln p(l \mid x) = -1153.4704$. On the small example above, the rescaled and unscaled values agree at $-1.2983$.

Exercise 2 asks you to implement the forward recursion and check it against enumeration.

### The backward pass and the training gradient

**Intuition.** The loss is $-\ln p(z \mid x)$. To train the network, you need its derivative with respect to every output at every frame. A frame's output matters through all the paths that pass through it, and the forward and backward variables, multiplied, give exactly that: the probability of all paths that go through a given state at a given time.

**Math.** The backward variable $\beta_t(s)$ is the mirror image of $\alpha$: the total probability of the suffix $l_{s:|l|}$ from time $t$ on (Section 4.1, Equations 9 to 11). Both $\alpha_t(s)$ and $\beta_t(s)$ include the factor $y^t_{l'_s}$, so their product counts it twice. Dividing once gives the probability of all paths through $l'_s$ at time $t$ (Section 4.2). Summing over $s$ gives (Equation 14):

$$
p(l \mid x) = \sum_{s=1}^{|l'|} \frac{\alpha_t(s)\,\beta_t(s)}{y^t_{l'_s}}, \qquad \text{for any } t
$$

The objective is the maximum likelihood error $O^{ML}(S, \mathcal{N}_w) = -\sum_{(x,z) \in S} \ln p(z \mid x)$ (Section 4.2, Equation 12). With $lab(l, k) = \{s : l'_s = k\}$, the derivative of Equation 14 is (Equation 15):

$$
\frac{\partial p(l \mid x)}{\partial y^t_k} = \frac{1}{(y^t_k)^2} \sum_{s \in lab(l,k)} \alpha_t(s)\,\beta_t(s)
$$

Backpropagating through the softmax gives the error signal for the unnormalized outputs $u^t_k$ (Equation 16, with the rescaled variables $\hat\alpha$ and $\hat\beta$):

$$
\frac{\partial O^{ML}}{\partial u^t_k} = y^t_k - \frac{1}{y^t_k Z_t} \sum_{s \in lab(z,k)} \hat\alpha_t(s)\,\hat\beta_t(s), \qquad Z_t \stackrel{\text{def}}{=} \sum_{s=1}^{|l'|} \frac{\hat\alpha_t(s)\,\hat\beta_t(s)}{y^t_{l'_s}}
$$

Without rescaling, $Z_t = p(z \mid x)$ for every $t$, by Equation 14. The first term pushes every output down toward 0, and the second term pushes up the outputs that the transcript uses, in proportion to their share of the probability mass. The standard backpropagation through time carries the signal into the weights (Section 4).

**Tiny example, computed by `scripts/forward_example.py`.** For the matrix above and $l = ab$, the backward table is:

| $t$ | $s=1$ | $s=2$ | $s=3$ | $s=4$ | $s=5$ |
|---|---|---|---|---|---|
| 1 | 0.0480 | 0.2250 | 0.1320 | 0.0390 | 0.0600 |
| 2 | 0 | 0.1200 | 0.0900 | 0.2400 | 0.1500 |
| 3 | 0 | 0 | 0 | 0.3000 | 0.5000 |

At each of the three frames, $\sum_s \alpha_t(s)\beta_t(s) / y^t_{l'_s}$ equals 0.2730, as Equation 14 predicts. The gradient of Equation 16 is below, and its rows sum to 0, because each frame's softmax gradient does.

| $t$ | blank | $a$ | $b$ |
|---|---|---|---|
| 1 | 0.2242 | -0.3242 | 0.1000 |
| 2 | 0.1352 | 0.0044 | -0.1396 |
| 3 | 0.2253 | 0.2000 | -0.4253 |

Frame 1 pushes $a$ up (negative gradient), because the target needs an $a$ early. Frame 3 pushes $b$ up the most. The script checks the table against a central finite difference of $-\ln p(l \mid x)$ through a softmax, and the largest difference is below $10^{-8}$.

Exercise 4 asks you to write the backward pass and the gradient.

### Conditional independence across frames

**Intuition.** The product in Equation 2 treats each frame's output as independent of the others. A frame can't see which label the network chose one frame earlier. The recurrent network can still model context, because its hidden state sees the whole input, but the output layer doesn't feed back.

**Math.** The paper states the assumption right after Equation 2: the outputs at different times are conditionally independent, given the internal state of the network, and the architecture ensures it by allowing no feedback connections from the output layer to itself or to the network (Section 3.1). Section 6 draws the consequence. CTC doesn't explicitly model dependencies between labels, unlike a graphical model whose labels form a $k$th-order Markov chain. It models them implicitly, for example by emitting labels that often occur together as a double spike (Section 6, Figure 1).

**Tiny example.** Independence is what makes Equation 2 a plain product, and that product is what lets the dynamic program keep three numbers per state. If frame $t$ conditioned on the previous output, $\alpha_t(s)$ would need that output as extra state, and the table would grow. The same assumption limits the model. In the example above, the per-frame blank probabilities (0.4, 0.3, 0.5) are fixed by the network before any label is chosen, so no frame can lower its blank probability because an earlier frame already emitted $a$. Later models such as the RNN transducer relax this (see [RNN-T](/rnnt/)).

### Decoding

**Intuition.** Training sums over all alignments, but at test time the network must commit to one transcript. The most probable transcript is not the transcript of the most probable path, because a transcript collects probability from many paths.

**Math.** The classifier should output $h(x) = \arg\max_{l \in L^{\le T}} p(l \mid x)$ (Section 3.2). The paper knows no general tractable algorithm for it and gives two approximations. Best path decoding takes the most active output at every frame, then applies $\mathcal{B}$ (Section 3.2, Equation 4):

$$
h(x) \approx \mathcal{B}(\pi^*), \qquad \pi^* = \arg\max_{\pi} p(\pi \mid x)
$$

It costs one arg max per frame, and it isn't guaranteed to find the most probable labelling. Prefix search decoding extends labelling prefixes and uses a modified forward-backward algorithm to score each extension. It always finds the best labelling given enough time, but the number of prefixes can grow exponentially with $T$. The paper adds a heuristic: split the sequence at frames where the blank probability exceeds a threshold, and search each section separately (Section 3.2, Figure 2). The experiment sets the threshold to 99.99% (Section 5.2).

**Tiny example, computed by `scripts/decode_example.py`.** On the page's matrix, the per-frame winners are $a, a, -$. They collapse to $a$, and the path has probability 0.1000. The labelling $a$ happens to be the best one at 0.3510.

Now a failure. Take $T = 2$, with probability 0.6 for the blank and 0.4 for $a$ in both frames. Best path picks `--`, with probability $0.6 \cdot 0.6 = 0.36$, and decodes to the empty labelling. But $a$ collects three paths, `-a`, `a-`, and `aa`, for $0.24 + 0.24 + 0.16 = 0.64$. The decoder returns a labelling with probability 0.36 when one with probability 0.64 exists. That is the gap that Table 1 measures: prefix search gives 30.51%, and best path gives 31.47%.

Exercise 3 asks you to implement best path decoding and to reproduce this failure.

<!-- exercises -->

## Pitfalls

- **Collapsing in the wrong order.** Merge repeats first, then drop blanks. If you drop blanks first, `a-a` becomes `aa` and then `a`, and you can never emit a doubled letter. The blank between repeated labels is the only thing that keeps them apart.
- **Allowing the skip between equal labels.** In the forward recursion, the skip from $s-2$ to $s$ is legal only when $l'_s$ is not a blank and $l'_s \ne l'_{s-2}$. Dropping the second condition lets `a-a` and `aa` blur together, and $p(aa \mid x)$ comes out too large.
- **Forgetting the final blank in Equation 8.** A path can end on the last label or on the trailing blank. Summing only $\alpha_T(|l'|)$ undercounts every transcript that ends on a label.
- **Underflow.** The unscaled $\alpha$ underflows to 0 on utterances of a few hundred frames. Rescale each frame as in Section 4.1, or work in log space with a log-sum-exp. Remember the note after Equation 7, or $\sum_t \ln C_t$ won't equal $\ln p(l \mid x)$.
- **Counting the factor $y^t_{l'_s}$ twice.** $\alpha_t(s)$ and $\beta_t(s)$ both include it. Equation 14 divides by it once. If you check your backward pass, test Equation 14 at every $t$, because a wrong $\beta$ that is off by a constant looks fine at one frame.
- **Mistaking the gradient with respect to $y$ for the one with respect to $u$.** Equation 15 has $1 / (y^t_k)^2$ and is the gradient for the softmax outputs. Equation 16 is for the pre-softmax $u$ and has a single $1 / y^t_k$ plus the extra term $y^t_k$. Use Equation 16 if your network's last layer is a softmax, and test it with finite differences.
- **Too few frames.** A transcript of $U$ labels with $R$ adjacent repeats needs $T \ge U + R$ frames. With fewer, $p(l \mid x) = 0$ and the loss is infinite. Check that your encoder doesn't downsample the input below that.
- **Reading best path as the best transcript.** Best path decoding can pick a labelling that another labelling beats, as the example above shows. It is a good approximation for a sharply peaked network and a poor one when the probability of a transcript is spread over many paths.
- **Expecting CTC to model a language.** Frames are conditionally independent given the network state, so CTC doesn't learn which label follows which except through the recurrent state. Systems that need a language model add one at decoding time.
- **Expecting alignments.** CTC doesn't segment the input, and its outputs tend to be spikes separated by blanks (Section 6, Figure 1). The spike position is where the network chose to commit, and it isn't a reliable label boundary.

## Further reading

- [Sequence Modeling With CTC](https://distill.pub/2017/ctc/), by Awni Hannun, is an illustrated walkthrough of the loss, the dynamic program, and decoding.
- [Supervised Sequence Labelling with Recurrent Neural Networks](https://www.cs.toronto.edu/~graves/preprint.pdf), by Alex Graves, is the author's thesis, with a longer derivation of CTC in Chapter 7.
- [`torch.nn.CTCLoss`](https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html) is the PyTorch implementation, which works in log space.
- [Deep Speech 2](/ds2/) scales CTC to end-to-end speech recognition, and [RNN-T](/rnnt/) removes the independence assumption.
- [warp-ctc](https://github.com/baidu-research/warp-ctc) is a fast CPU and GPU implementation from Baidu Research.
