---
id: "conformer"
title: "Conformer: Convolution-augmented Transformer for Speech Recognition"
authors: "Gulati et al."
year: 2020
venue: "Interspeech 2020"
arxiv: "2005.08100"
url: "https://arxiv.org/abs/2005.08100"
track: speech
tier: extended
est_hours: 4
status: full
summary: "The Conformer is a speech encoder block that pairs self-attention, which sees the whole utterance, with a depthwise convolution, which sees a short window of neighboring frames. Two half-step feed-forward modules sandwich the attention and convolution modules, and the attention uses relative positions so it copes with utterances of any length. On LibriSpeech, a 118.8M-parameter Conformer reaches 2.1% and 4.3% word error rate on test-clean and test-other without a language model."
related_course: "speech: ASR systems"
prereqs:
  - id: transformer
    reason: "The Conformer block keeps multi-head self-attention from the Transformer and adds a relative positional encoding to it."
  - id: ctc
    reason: "The paper trains a transducer, which inherits the blank symbol and the many-paths-one-label alignment idea from CTC."
background: [depthwise-convolution, relative-position]
course_links:
  - title: "Conformers and SpecAugment"
    url: "https://speech.ifkash.dev/lessons/037-conformers-and-specaugment"
  - title: "Transducers"
    url: "https://speech.ifkash.dev/lessons/034-transducers"
  - title: "Convolution and receptive field"
    url: "https://ml.ifkash.dev/lessons/046-convolution-and-receptive-field"
  - title: "Position encodings"
    url: "https://ml.ifkash.dev/lessons/062-position-encodings"
self_check:
  - q: 'A depthwise convolution with kernel size $k = 32$ acts on $d = 256$ channels. How many weights does it have, and how many would a full convolution from $d$ to $d$ channels with the same kernel have?'
    a: 'The depthwise convolution has $k \cdot d = 8192$ weights, one length-32 filter per channel. A full convolution has $k \cdot d \cdot d = 2{,}097{,}152$ weights, which is $d = 256$ times more. The depthwise version never mixes channels, so the pointwise convolutions in the module do the channel mixing.'
    see: depthwise-convolution
  - q: 'In self-attention with absolute position encodings, a query at frame 7 attends to frame 5, and a query at frame 100 attends to frame 98. Why can the two attention patterns differ, and how does a relative encoding change that?'
    a: 'An absolute encoding adds a different vector to every frame, so the pair (7, 5) and the pair (100, 98) have different inputs even though both have offset 2. A relative encoding scores a pair by the offset $i - j$ only, so both pairs get the same position term, and the model does not depend on the absolute index (Section 2.1).'
    see: relative-position
  - q: 'Self-attention over $T$ frames builds a $T \times T$ score matrix. A 10-second utterance has 1000 frames at a 10 ms stride. How many entries does the matrix have after the 4x subsampling in Figure 1, and why does this matter?'
    a: 'After subsampling to a 40 ms stride, the utterance has about 250 frames, so the matrix has about $250^2 = 62{,}500$ entries instead of $1000^2 = 1{,}000{,}000$, roughly 16 times fewer. Attention cost grows with the square of the length, so subsampling first makes the stack of attention layers affordable.'
    see: transformer
---

## The problem

Speech recognition models need two kinds of context at once. The first is global: the meaning of a word depends on words several seconds away. The second is local: the identity of a phoneme depends on the few frames around it, such as the burst of a consonant or the shape of a formant transition. The two standard building blocks cover one kind each.

- **Self-attention covers global context.** Every frame can attend to every other frame, so one layer reaches across the whole utterance. It has no built-in bias toward neighbors, so it has to learn fine-grained local patterns from data. The paper says Transformers are good at long-range context and less capable at extracting local feature patterns (Section 1).
- **Convolution covers local context.** A kernel learns a shared, position-based filter over a local window. Reaching across a long utterance takes many layers or many parameters, because each layer sees only a few frames. ContextNet adds squeeze-and-excitation to bring in global context, but that only applies a global average over the sequence (Section 1).

A concrete failing case: a Transformer encoder must tell "bat" from "pat" from a 30 ms stretch of voicing onset. The decision depends on a few neighboring frames, and attention has no built-in preference for nearby frames over the other 1000 in the utterance. A convolution with a kernel of a few frames sees that stretch directly. The reverse case also fails: to decide that a sentence ends, a stack of local convolutions needs many layers before any frame sees the other end of the utterance.

The paper's hypothesis is that both kinds of interaction matter for a parameter-efficient model, and that a block should hold both (Section 1). The Conformer block does this. It reaches 2.1% and 4.3% word error rate (WER) on LibriSpeech test-clean and test-other with no language model, and 1.9% and 3.9% with an external one (Abstract, Section 3.3). Its 30.7M-parameter medium model already beats the 139M-parameter Transformer Transducer (Section 3.3, Table 2). Every WER on this page comes from the paper's text and tables. The scripts compute the other numbers.

## Key ideas

The sections below follow the paper's notation: $x_i$ is the input to Conformer block $i$ and $y_i$ is its output. $T$ is the number of frames and $d$ is the encoder dimension. Each frame is a vector of $d$ channels, so a sequence is a $(T, d)$ array. The paper writes FFN, MHSA, and Conv for the three module types. The exercises use the names `x`, `ffn1`, `mhsa`, `conv`, `ffn2`, and `final_ln`. Run any script from the paper folder to reproduce its output.

### The convolution module

**Intuition.** After attention has mixed information across the utterance, the convolution module smooths and sharpens it over a short window. Each channel gets its own filter that slides along time. A pointwise convolution before and after the filter mixes channels, so the module as a whole mixes both time and channels, but in two separate, cheap steps.

**Math.** Section 2.2 and Figure 2 give the module in this order. The page lists the steps with the tensor shapes for $d$ channels and $T$ frames.

1. Layer normalization (shown at the start of Figure 2).
2. A pointwise convolution with expansion factor 2: a kernel of size 1 that maps $d$ channels to $2d$, which is a linear layer applied to each frame.
3. A gated linear unit (GLU). It splits the $2d$ channels into halves $a$ and $b$ and returns $a \odot \sigma(b)$, where $\sigma$ is the sigmoid. The output has $d$ channels again. The paper doesn't say which half gates, and this page follows the common library convention.
4. A single 1-D depthwise convolution along time with kernel size 32 (Table 1 lists 32 for S, M, and L). *Depthwise* means one filter per channel and no mixing between channels.
5. Batch normalization, then Swish: $\mathrm{Swish}(z) = z \cdot \sigma(z)$.
6. A second pointwise convolution, followed by dropout, and the block adds the result to its input (Equation 1).

The depthwise step is where most of the parameter saving comes from. A full convolution from $d$ to $d$ channels with kernel 32 needs $32 d^2$ weights. The depthwise version needs $32 d$.

The paper doesn't state a padding rule. This page and the exercises pad $(k-1)//2$ zeros at the start and the rest at the end, so the output keeps length $T$ for any $k$. For an even $k$ such as 32, symmetric padding isn't possible. Open implementations often use an odd kernel such as 31 for that reason.

**Tiny example, computed by `scripts/conv_module_example.py`.** A depthwise convolution with $k = 3$ and filter $[1, 0, -1]$ on the single channel $[1, 2, 3, 4, 5]$ outputs $[-2, -2, -2, -2, 4]$. Frame $t$ gets $x_{t-1} - x_{t+1}$, a difference of its two neighbors. At the ends, a zero stands in for the missing neighbor: frame 0 gets $0 - 2 = -2$, and frame 4 gets $4 - 0 = 4$. An impulse at frame 3 with filter $[1, 2, 3]$ gives $[0, 0, 3, 2, 1, 0, 0]$, so the filter appears in reverse order around the impulse. The operation is cross-correlation: output frame $t$ multiplies the later taps with the later frames.

The GLU value $a \cdot \sigma(b)$ with $a = 1$ is 0.5000 at $b = 0$, 0.8808 at $b = 2$, and 0.0180 at $b = -4$. The gate opens when $b$ is large and closes when it is very negative. Swish is smooth near zero: Swish(1) is 0.7311, where ReLU gives 1, and Swish(-2) is -0.2384, where ReLU gives 0. Batch normalization of the channel values $[1, 2, 3, 6]$ over time gives $[-1.069, -0.5345, 0, 1.6036]$: mean 3, variance 3.5.

For one module with $d = 256$ and $k = 32$, the script reports 8192 depthwise weights against 2,097,152 for a full convolution, a ratio of 256. On random data with $k = 32$, the vectorized implementation matches a naive triple loop exactly.

Exercises 1 and 2 ask you to implement the depthwise convolution and the module.

### The Macaron-style feed-forward sandwich

**Intuition.** A standard Transformer block applies attention and then one feed-forward network. Macaron-Net splits the feed-forward network into two halves, one before the attention and one after. The Conformer block follows this pattern and puts the convolution module between the two feed-forward modules, right after attention. Each feed-forward module adds only half of its output to the residual stream, so the two together carry the weight of one full feed-forward network.

**Math.** Equation 1 (Section 2.4) gives one block. For block input $x_i$, the output $y_i$ is

$$
\begin{aligned}
\tilde x_i &= x_i + \tfrac{1}{2}\,\mathrm{FFN}(x_i) \\
x'_i &= \tilde x_i + \mathrm{MHSA}(\tilde x_i) \\
x''_i &= x'_i + \mathrm{Conv}(x'_i) \\
y_i &= \mathrm{LayerNorm}\!\left(x''_i + \tfrac{1}{2}\,\mathrm{FFN}(x''_i)\right)
\end{aligned}
$$

Each module uses a pre-norm residual unit: it applies layer normalization to its own input, and it applies dropout to its output before the residual add (Sections 2.1 and 2.3). The feed-forward module has a first linear layer with expansion factor 4, a Swish activation, and a second linear layer back to $d$ (Figure 4). The block ends with one more layer norm. The two feed-forward modules are two separate modules with different weights.

Section 3.4.3 tests the design. In the paper's Table 5, a single feed-forward module or full-step residuals give 4.5 on test-other, against 4.3 for the Macaron design. Table 3 (Section 3.4.1) moves the block step by step toward a Transformer block while keeping the parameter count fixed. On dev-other and test-other, the full model scores 4.4 and 4.3, the row without the convolution block scores 4.8 and 4.9, and the row without the Macaron pair scores 5.1 and 5.0. The text names the convolution the most important feature, and the table shows that the Macaron row is also clearly worse. Section 3.4.2 and Table 4 test the placement of the convolution: before attention, the dev-other WER is 4.5; in parallel with attention, 4.9; after attention, 4.4. A lightweight convolution in place of the depthwise one gives 4.8.

**Tiny example, computed by `scripts/macaron_block.py`.** Take a single scalar frame $x = 1$ and four toy modules: FFN$_1(x) = 2x$, MHSA$(x) = x^2$, Conv$(x) = -x/2$, and FFN$_2(x) = 4x$. The final layer norm is the identity. The steps are:

- $\tilde x = 1 + \tfrac12 \cdot 2 = 2$
- $x' = 2 + 2^2 = 6$
- $x'' = 6 - 6/2 = 3$
- $y = 3 + \tfrac12 \cdot 12 = 9$

With full-step residuals, the same toy modules give $y = 30$. The final layer norm turns the vector $[1, 2, 3, 4]$ into $[-1.3416, -0.4472, 0.4472, 1.3416]$, which has mean 0 and unit variance.

Exercise 3 asks you to wire the block.

### Relative positional attention

**Intuition.** A self-attention layer with absolute positions learns that frame 7 relates to frame 5 and does not transfer that to frames 100 and 98. A relative scheme scores a query and key by their distance, so a pattern such as "attend two frames back" works anywhere in the utterance. Speech utterances vary a lot in length, so this helps the model with lengths not seen in training. Section 2.1 states this motivation: the relative encoding lets the attention module generalize better to different input lengths, and the encoder is more robust to variance in utterance length.

**Math.** The paper integrates the relative sinusoidal positional encoding from Transformer-XL (Section 2.1) and doesn't write out the equation. This page uses the Transformer-XL form for one head, with the notation of the course lesson:

$$
A_{ij} = \frac{(q_i + u)^{\top} k_j + (q_i + v)^{\top} p_{i-j}}{\sqrt{d_k}}
$$

Here $q_i$ and $k_j$ are the query and key vectors of dimension $d_k$, and $u$ and $v$ are learned vectors shared across positions. $p_{r}$ is the encoding of offset $r = i - j$, already projected to key space. Expanding the two products shows four terms: a content-to-content score, a content-dependent position score, and two global biases, one for content and one for position. Transformer-XL uses a different notation, and this page folds the learned projections into $p_r$. The sequence of $T$ frames has $2T - 1$ distinct offsets, from $-(T-1)$ to $T-1$, so one table of $2T - 1$ rows serves every position. In the exercise, row $m$ of that table holds offset $r = T - 1 - m$.

The last row of Table 3 replaces relative attention with vanilla self-attention. It raises test-other WER from 4.3 to 5.6 and dev-other from 4.4 to 5.8, the largest jump in that table.

**Tiny example, computed by `scripts/rel_attention.py`.** Take $T = 2$, $d_k = 1$, queries $[1, 2]$, keys $[3, 4]$, $u = 1$, $v = 2$, and a position table with rows $[10, 20, 30]$ for offsets $1, 0, -1$. The score for $i = 0, j = 0$ has offset 0 and uses the row 20: $(1 + 1) \cdot 3 + (1 + 2) \cdot 20 = 66$. The full matrix is $[[66, 98], [49, 92]]$. In a larger case with $T = 6$ and $d_k = 4$, the vectorized implementation matches the naive double loop to within $9 \times 10^{-16}$. When the queries are identical, the keys are zero, and $u = v = 0$, the score at $(i, j)$ equals the score at $(i + 1, j + 1)$, up to $3.5 \times 10^{-18}$. A sequence of 1000 frames has 1999 offsets but 1,000,000 $(i, j)$ pairs.

Exercise 4 asks you to compute these scores.

### Convolution subsampling and the transducer

**Intuition.** The encoder works on 80-channel filterbank features at a 10 ms stride (Section 3.1). Attention costs grow with the square of the length, so a convolution subsampling layer shortens the sequence before the blocks (Section 2, Figure 1).

**Math.** Figure 1 shows the frame rate going from 10 ms at the input to 40 ms after the subsampling layer, a factor of 4. The paper doesn't give the kernel sizes or strides. The common implementation uses two stride-2 convolutions with kernel 3. For a length-$n$ sequence with no padding, each one produces $\lfloor (n - 3)/2 \rfloor + 1$ frames. This is this page's illustration and not a value from the paper.

The decoder is a single LSTM layer, with dimension 320 for S and 640 for M and L (Table 1), inside a transducer (Section 3.2). A language model, a 3-layer LSTM of width 4096 with word-level perplexity 63.9 on the dev transcripts, is added by shallow fusion with a weight tuned on dev (Section 3.2). Section 3.2 names the model a Conformer Transducer. For the transducer's blank symbol and its joint network, see the [RNN-T](/rnnt/) page.

**Tiny example, computed by `scripts/subsampling.py`.** A 10-second utterance has 1000 frames. Two stride-2 convolutions with kernel 3 give 499 frames and then 249, about 250, which is the 40 ms stride. The attention score matrix shrinks from 1,000,000 entries to 62,001, a factor of 16.1. The 80 filterbank channels become 39 and then 19. A depthwise kernel of 32 frames covers 1280 ms on the 40 ms axis, where the same kernel would cover only 320 ms on the 10 ms axis.

### Model sizes

**Intuition.** The paper picks three sizes by sweeping depth, width, and heads within a parameter limit (Section 3.2). The width $d$ and the layer count set most of the cost.

**Math.** Table 1 lists the settings.

| Setting | Conformer S | Conformer M | Conformer L |
|---|---|---|---|
| Parameters (M) | 10.3 | 30.7 | 118.8 |
| Encoder layers | 16 | 16 | 17 |
| Encoder dimension $d$ | 144 | 256 | 512 |
| Attention heads | 4 | 4 | 8 |
| Convolution kernel size | 32 | 32 | 32 |
| Decoder layers | 1 | 1 | 1 |
| Decoder dimension | 320 | 640 | 640 |

The paper reports these results on LibriSpeech test-clean and test-other (Table 2, WER in percent):

| Model | No LM clean | No LM other | With LM clean | With LM other |
|---|---|---|---|---|
| Conformer S | 2.7 | 6.3 | 2.1 | 5.0 |
| Conformer M | 2.3 | 5.0 | 2.0 | 4.3 |
| Conformer L | 2.1 | 4.3 | 1.9 | 3.9 |

**Tiny example, computed by `scripts/param_count.py`.** Count the weights of one block from the module diagrams. This page's counting rules are: layer norms have a scale and a shift, linear layers have biases, the relative-position projection has no bias, and the attention module has the learned vectors $u$ and $v$. They are this page's assumptions, and the paper doesn't give a formula. For $d = 144$, a feed-forward module has 166,896 weights, the attention module has 104,832, and the convolution module has 67,968, so one block has $2 \cdot 166{,}896 + 104{,}832 + 67{,}968 + 288 = 506{,}880$. Sixteen blocks give 8.11M of the reported 10.3M. For M the encoder blocks come to 25.43M of 30.7M. For L they come to 107.51M of 118.8M. The remainder is 2.19M, 5.27M, and 11.29M, or 21%, 17%, and 10% of the totals. It belongs to the subsampling layer, the LSTM decoder, the embedding, and the joint network, which the paper doesn't itemize, so this page can't check that split. In the convolution module for $d = 144$, the depthwise filters are 4752 of the 67,968 weights, so the pointwise convolutions hold most of the parameters. The paper's gain over the Transformer Transducer is $(4.6 - 3.9)/4.6 = 15.2\%$ relative on test-other with a language model, and the M model has 22.1% of its parameters.

<!-- exercises -->

## Pitfalls

- **Padding an even kernel.** The paper's kernel of 32 is even, so symmetric padding changes the output length. Many implementations use 31 or pad asymmetrically. If you pad by $k // 2$ on the left for an even $k$, every output is shifted by one frame against the rule on this page. Exercise 1 tests the even case.
- **Batch norm statistics over padding.** Batch normalization in training mode averages over every frame, including the padding of short utterances in a batch. The depthwise convolution also reads padded frames at the end of an utterance. Zero the padded frames before the depthwise step and evaluate in evaluation mode. See [Conformers and SpecAugment](https://speech.ifkash.dev/lessons/037-conformers-and-specaugment).
- **Taking the half-step residual out.** Using $x + \mathrm{FFN}(x)$ for both feed-forward modules is the full-step variant. The paper's ablation (Table 5) shows it is worse on test-other, 4.5 against 4.3. The script gives $y = 30$ instead of 9 on the toy example.
- **Putting the convolution first.** Table 4 reports convolution before attention at 4.5 on dev-other against 4.4 after it. The gap is small, so the evidence is a few tenths of a percent, and the paper reports no variance across runs.
- **Treating the ablations as large effects.** Most Table 3 and Table 5 differences are 0.1 to 0.3 absolute WER, and the paper reports no variance across runs and rounds most WERs to one decimal. Read them as directional.
- **Mixing up which half of the GLU gates.** The paper doesn't say. Swapping the halves gives a different function, and trained weights from one convention don't load into the other. Exercise 2 fixes the convention.
- **Assuming the kernel is a tuned optimum.** Table 7 shows kernel 32 is best on dev-other at 4.30 against 4.30 for kernel 7 and 4.31 for kernel 17, and worse at 65, where dev-other is 4.47. The text calls 32 better only on the second decimal of dev WER.
- **Comparing parameter counts across papers.** The reported totals include the decoder and the joint network. The encoder blocks alone are 8.11M, 25.43M, and 107.51M by this page's count.
- **Reading the paper as a CTC model.** The paper trains a transducer, not a CTC model. Open recipes also train Conformer encoders with a CTC or attention loss, but those are different experiments from the paper's.

## Further reading

- [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860), by Dai et al., defines the relative positional encoding that the Conformer uses.
- [Understanding and Improving Transformer From a Multi-Particle Dynamic System Point of View](https://arxiv.org/abs/1906.02762), by Lu et al., introduces Macaron-Net, the source of the half-step feed-forward sandwich.
- [ContextNet: Improving Convolutional Neural Networks for Automatic Speech Recognition with Global Context](https://arxiv.org/abs/2005.03191), by Han et al., is the convolutional model the paper compares against in Table 2.
- [Lingvo](https://github.com/tensorflow/lingvo), the toolkit the paper used (Section 3.2).
- [`torchaudio.models.Conformer`](https://pytorch.org/audio/stable/generated/torchaudio.models.Conformer.html) and [ESPnet](https://github.com/espnet/espnet) are widely used open implementations.
- [Layer normalization](/layernorm/) and [RNN-T](/rnnt/) are covered on this site.
