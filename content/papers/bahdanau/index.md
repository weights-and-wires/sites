---
id: "bahdanau"
title: "Neural Machine Translation by Jointly Learning to Align and Translate"
authors: "Bahdanau et al."
year: 2014
venue: "ICLR 2015"
arxiv: "1409.0473"
url: "https://arxiv.org/abs/1409.0473"
track: architectures
tier: core
est_hours: 3
status: full
summary: "A basic encoder-decoder squeezes a whole source sentence into one fixed-length vector, and translation quality falls as sentences get longer. This paper keeps one annotation vector per source word, produced by a bidirectional RNN, and lets the decoder compute a new context vector at every output step. The context is a softmax-weighted sum of the annotations, with weights from a small neural network that scores each annotation against the decoder state, and because the weights are soft, the whole model trains end to end by backpropagation."
related_course: "ml: attention"
prereqs:
  - id: seq2seq
    reason: "This paper keeps the encoder-decoder with an autoregressive decoder and removes the single fixed vector that the sequence-to-sequence model hands from the encoder to the decoder."
background: [softmax, alignment, weighted-average]
course_links:
  - title: "Attention as a soft lookup, with a dot product or a learned scorer"
    url: "https://ml.ifkash.dev/lessons/056-attention-as-soft-lookup"
  - title: "Attention gradients and the alignment matrix"
    url: "https://ml.ifkash.dev/lessons/057-attention-gradients-and-alignment"
self_check:
  - q: 'Three alignment scores are $e = [\ln 3, 0, 0]$. Compute the weights $\alpha$ by hand, and then the context $c$ for annotations $h_1 = [1, 0]$, $h_2 = [0, 1]$, $h_3 = [1, 1]$.'
    a: 'The softmax is $[3, 1, 1] / 5 = [0.6, 0.2, 0.2]$. The context is $0.6\,[1, 0] + 0.2\,[0, 1] + 0.2\,[1, 1] = [0.8, 0.4]$, a weighted average that lies inside the span of the annotations. `scripts/additive_example.py` prints both.'
    see: softmax
  - q: 'The alignment weights $\alpha_{ij}$ form a $T_y \times T_x$ matrix. Over which index do the weights sum to 1, and what does one row of the matrix tell you?'
    a: 'They sum to 1 over the source index $j$ (Equation 6), so every row sums to 1 and the columns need not. Row $i$ shows how the decoder spread its attention over the source words when it produced target word $i$, which is what Figure 3 plots as a grayscale image.'
    see: alignment
  - q: 'In the sequence-to-sequence model, the decoder sees one vector $c$ for the whole sentence. According to Appendix A.2.2, how do you turn the attention model of this paper back into that fixed-vector encoder-decoder?'
    a: 'Fix every context vector $c_i$ to the last forward state $\overrightarrow{h}_{T_x}$. The decoder then sees the same summary at every step, which is the RNN Encoder-Decoder of Cho et al. that the paper uses as its baseline.'
    see: seq2seq
---

## The problem

In the sequence-to-sequence model (see [Sequence to sequence learning](/seq2seq/)), an encoder RNN reads the source sentence and the decoder starts from one fixed-length vector $c$. Every fact the decoder needs about the source must fit in that vector, whether the sentence has 5 words or 50. The introduction conjectures that this is a bottleneck, and cites Cho et al. (2014b), who showed that the performance of a basic encoder-decoder deteriorates rapidly as the input gets longer (Section 1).

The baseline in this paper is the RNN Encoder-Decoder of Cho et al. (2014a), called RNNencdec. It is close to the model on the seq2seq page, but it is not the same network. The paper trains it and the new model, RNNsearch, on the same WMT '14 English-to-French data, in two variants each. The "-30" variants train on sentences of up to 30 words, and the "-50" variants train on sentences of up to 50 words (Section 4.2).

Table 1 reports BLEU on the test set. On all sentences, RNNencdec-50 scores 17.82 and RNNsearch-50 scores 26.75. On sentences with no unknown words, the scores are 26.71 and 34.16, against 35.63 for the phrase-based system Moses. Figure 2 shows where the gap comes from. The BLEU of RNNencdec falls sharply as sentence length grows, while RNNsearch-50 shows no deterioration even at 50 words or more (Section 5.1). Even RNNsearch-30 beats RNNencdec-50 on Table 1.

A fixed vector forces the encoder to guess in advance what the decoder will need. The paper's alternative is to keep everything the encoder computed and let the decoder look back at it at every step.

## Key ideas

This page follows the paper's notation: source words $x_1, \ldots, x_{T_x}$, annotations $h_j$, decoder states $s_i$, alignment scores $e_{ij}$, weights $\alpha_{ij}$, and context vectors $c_i$. Three differences from the paper are worth knowing.

- The paper's main text (Appendix A.1.2) gives $W_a \in \mathbb{R}^{n \times n}$. Appendix A.2.2 uses $n'$ units in the alignment model, with $v_a \in \mathbb{R}^{n'}$, $W_a \in \mathbb{R}^{n' \times n}$, and $U_a \in \mathbb{R}^{n' \times 2n}$. This page uses $n'$ throughout.
- The math is 1-indexed, as in the paper. The code in the exercises is 0-indexed, and it stores the annotations as the rows of a matrix `H` of shape `(Tx, 2n)`. A score is therefore computed as `H @ Ua.T`.
- Every number comes from a script in `scripts/`. Run any script from the paper folder to reproduce its output.

### Annotations from a bidirectional RNN

**Intuition.** A plain encoder reads left to right, so its state at word $j$ knows only the words before $j$. The decoder wants to attend to a word together with its context on both sides. The paper reads the sentence twice, once in each direction, and keeps both states for every word.

**Math.** The forward RNN $\overrightarrow{f}$ reads $x_1$ to $x_{T_x}$ and produces $\overrightarrow{h}_1, \ldots, \overrightarrow{h}_{T_x}$. The backward RNN $\overleftarrow{f}$ reads $x_{T_x}$ down to $x_1$ and produces $\overleftarrow{h}_1, \ldots, \overleftarrow{h}_{T_x}$, indexed by the word each state sits on. The annotation is the concatenation (Section 3.2, and Equation 7 in Appendix A.2.1):

$$
h_j = \begin{bmatrix} \overrightarrow{h}_j \\ \overleftarrow{h}_j \end{bmatrix} \in \mathbb{R}^{2n}
$$

The two directions share the embedding matrix $E$ but have separate weight matrices (Appendix A.2.1). The paper's units are gated (a GRU-style update). The example below and Exercise 3 use a plain tanh cell, $h_t = \tanh(w x_t + u h_{t-1})$, to keep the arithmetic checkable. The paper's decoder starts from $s_0 = \tanh(W_s \overleftarrow{h}_1)$ (Appendix A.2.2).

**Tiny example, computed by `scripts/bidirectional_example.py`.** Take one hidden unit, $w = 1$, $u = 0.5$, and $x = [1, 0, -1]$, with the same weights in both directions.

| Step | Result |
|---|---|
| Forward states $\overrightarrow{h}_1, \overrightarrow{h}_2, \overrightarrow{h}_3$ | `[0.7616, 0.3634, -0.6741]` |
| Backward states, in the order they are computed ($x_3$ first) | `[-0.7616, -0.3634, 0.6741]` |
| Backward states, re-aligned to positions | `[0.6741, -0.3634, -0.7616]` |
| Annotations $h_j$ | `[[0.7616, 0.6741], [0.3634, -0.3634], [-0.6741, -0.7616]]` |

Check the first forward state by hand: $\tanh(1 \cdot 1 + 0.5 \cdot 0) = \tanh 1 = 0.7616$. The backward state at $x_3 = -1$ is $\tanh(-1) = -0.7616$, and it must land in the last row of the annotation matrix, not the first. The re-alignment is the step that is easy to forget.

Exercise 3 asks you to build the annotations and test that each half sees only its side of the sentence.

### The additive alignment score

**Intuition.** To produce target word $i$, the decoder asks which source words matter right now. It answers with a small neural network that reads the decoder's previous state and one annotation and returns a single number. A high number means "look here".

**Math.** The score is $e_{ij} = a(s_{i-1}, h_j)$ (Section 3.1). The paper parametrizes $a$ as a feedforward network with one hidden layer (Appendix A.1.2):

$$
a(s_{i-1}, h_j) = v_a^\top \tanh\!\left(W_a s_{i-1} + U_a h_j\right)
$$

This is called *additive* attention, because the decoder and annotation terms are added inside the tanh. The score uses $s_{i-1}$, the state *before* the decoder emits $y_i$ (Section 3.1). The weights come from a softmax over the source positions (Section 3.1, Equation 6):

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}
$$

**Tiny example, computed by `scripts/additive_example.py`.** Take $n = 1$, $n' = 1$, $W_a = [1]$, $U_a = [1, -1]$, $v_a = [2]$, and the decoder state $s_{i-1} = [0]$. There are three annotations, $h_1 = [1, 0]$, $h_2 = [0, 1]$, and $h_3 = [1, 1]$.

| Step | Result |
|---|---|
| $W_a s_{i-1} + U_a h_j$ | `[1, -1, 0]` |
| $e_{ij} = 2\tanh(\cdot)$ | `[1.5232, -1.5232, 0]` |
| $\alpha_{ij}$ | `[0.7902, 0.0376, 0.1723]` |

Check $e_{i1}$ by hand: $2\tanh(1) = 2 \cdot 0.7616 = 1.5232$. Now change the state to $s_{i-1} = [1]$. The pre-activations become `[2, 0, 1]`, the scores become `[1.9281, 0, 1.5232]`, and the weights become `[0.5517, 0.0802, 0.3680]`. The same annotations get different weights because the decoder state changed.

**Why the tanh matters.** Remove it, and the score is $v_a^\top W_a s_{i-1} + v_a^\top U_a h_j$. The first term is the same for every $j$, so it cancels in the softmax and the weights no longer depend on the decoder state. `scripts/linear_collapse.py` shows it with random weights ($n = 4$, $n' = 6$, five annotations). With the tanh, three different states give three different weight vectors. Without it, all three give `[0.0001, 0.0000, 0.0010, 0.0000, 0.9990]`. Exercise 1 includes a test for this bug.

### The context vector is an expected annotation

**Intuition.** The decoder can't read five annotations at once, so it needs one vector. A weighted average gives one vector that leans toward the annotations the decoder cares about, and it stays the same size however long the source is.

**Math.** The context for target position $i$ is (Section 3.1, Equation 5):

$$
c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j
$$

The decoder state and the output probability then use it (Section 3.1, Equation 4):

$$
s_i = f(s_{i-1}, y_{i-1}, c_i), \qquad p(y_i \mid y_1, \ldots, y_{i-1}, x) = g(y_{i-1}, s_i, c_i)
$$

Unlike the fixed $c$ in the basic model (Equation 2), $c_i$ is *distinct for each target word*. The paper reads $\alpha_{ij}$ as the probability that $y_i$ is aligned to $x_j$ and $c_i$ as the expected annotation under that distribution (Section 3.1). If you fix $c_i$ to $\overrightarrow{h}_{T_x}$, the model becomes the RNN Encoder-Decoder (Appendix A.2.2).

**Tiny example, computed by `scripts/additive_example.py`.** With the weights above, $c_i = 0.7902\,[1, 0] + 0.0376\,[0, 1] + 0.1723\,[1, 1] = [0.9624, 0.2098]$. A source word that gets almost all the weight pulls $c_i$ almost onto its own annotation.

Exercise 1 asks you to implement one step of this attention. Exercise 2 asks you to compute the whole alignment matrix at once.

### The alignment matrix and the cost of attention

**Intuition.** Attention evaluates the scoring network once for every pair of a target word and a source word. Written as a table, the weights are an alignment, and you can look at it.

**Math.** The alignment model runs $T_x \times T_y$ times per sentence pair (Appendix A.1.2). Appendix A.1.2 notes a saving: $U_a h_j$ does not depend on $i$, so you compute it once per source position and reuse it for every target word. The same holds for $W_a s_{i-1}$ across the source positions. The $T_y \times T_x$ matrix of weights $\alpha_{ij}$ is what Figure 3 plots. In Figure 3, white is weight 1 and black is weight 0.

**Tiny example, computed by `scripts/alignment_cost.py`.** With the paper's sizes, $n = n' = 1000$ (Appendix A.2.3), the alignment model has $1000 \cdot 1000 + 1000 \cdot 2000 + 1000 = 3{,}001{,}000$ parameters. For $T_x = T_y = 50$ there are 2,500 evaluations per sentence pair.

| Strategy | Multiply-adds |
|---|---|
| Recompute $W_a s$ and $U_a h$ for every pair | 7,502,500,000 |
| Compute $W_a s_{i-1}$ once per target word and $U_a h_j$ once per source word | 152,500,000 |

The saving is a factor of about 49. The count covers the matrix products and the dot with $v_a$. It leaves out the tanh and additions, which are small by comparison. Exercise 2 builds the three-dimensional broadcast that uses the same saving.

### Soft alignment is differentiable

**Intuition.** Translation systems before this paper treated the alignment as a hidden variable to infer. Here, the alignment is a softmax output, so the loss gradient can pass through it, and the network that scores positions learns from the translation loss with no alignment labels.

**Math.** The paper makes the point in one sentence: the alignment model "directly computes a soft alignment, which allows the gradient of the cost function to be backpropagated through" (Section 3.1). For one decoder step, let $L$ be the loss and $g = \partial L / \partial c_i$. Then $\partial L / \partial \alpha_{ij} = g^\top h_j$, and the softmax backward rule gives

$$
\frac{\partial L}{\partial e_{ij}} = \alpha_{ij}\left(\frac{\partial L}{\partial \alpha_{ij}} - \sum_{k} \alpha_{ik} \frac{\partial L}{\partial \alpha_{ik}}\right)
$$

The paper does not write this formula. It is the standard softmax Jacobian, applied to Equations 5 and 6. The term inside the parentheses subtracts the weighted mean, so the gradient of each row sums to zero. Adding a constant to a row of scores doesn't change the weights, so the gradient in that direction is zero. The gradient with respect to each annotation through the context is $\alpha_{ij}\, g$.

**Tiny example, computed by `scripts/gradient_example.py`.** Take $e = [1, 0, -1]$, the annotations $h_1 = [1, 0]$, $h_2 = [0, 1]$, $h_3 = [1, 1]$, and $g = [1, 0]$.

| Step | Result |
|---|---|
| $\alpha$ | `[0.6652, 0.2447, 0.0900]` |
| $\partial L / \partial \alpha_j = g^\top h_j$ | `[1, 0, 1]` |
| $\sum_k \alpha_k\, \partial L / \partial \alpha_k$ | 0.7553 |
| $\partial L / \partial e_j$ | `[0.1628, -0.1848, 0.0220]` |

The gradient sums to 0, and finite differences give the same three numbers. Exercise 4 asks you to implement this backward pass for the whole matrix.

**Watching the weights sharpen, computed by `scripts/sharpening.py`.** The script trains only the alignment model on a reversal task. There are five source positions, each with a one-hot annotation, and each decoder step has a one-hot position state. The right answer for step $i$ is source position $T-1-i$. It starts from random weights with a standard deviation of 0.3 and runs 300 steps of gradient descent with the backward rule above.

| Step | Loss | Mean entropy | Mean largest weight |
|---|---|---|---|
| 0 | 1.6204 | 1.5971 | 0.2514 |
| 50 | 0.6477 | 1.2919 | 0.5302 |
| 100 | 0.0548 | 0.2689 | 0.9468 |
| 300 | 0.0068 | 0.0485 | 0.9932 |

The uniform distribution over five positions has entropy $\ln 5 = 1.6094$. By step 300 each row puts 0.99 on the correct position, and the argmax of the rows is `[4, 3, 2, 1, 0]`. Every row sums to 1 at every step. The paper's own weights are less crisp, because real sentences are not one-hot, but Figure 3 shows the same pattern: strong weights along a diagonal, with local reordering. For example, "zone" is aligned with "Area" in Figure 3 (a) (Section 5.2.1).

Soft weights also handle cases that a hard alignment can't. In Figure 3 (d), the French "l'" depends on the word after "the", so the model looks at both "the" and "man" (Section 5.2.1).

<!-- exercises -->

## Pitfalls

- **Scoring with $s_i$ instead of $s_{i-1}$.** The paper scores each annotation against the state before the decoder emits $y_i$ (Section 3.1). Using $s_i$ creates a circular dependency, because $s_i$ itself takes $c_i$ as an input. The Transformer later changed this, because its queries come from the current layer. See [Attention is all you need](/transformer/).
- **Dropping the tanh.** The score then splits into a state term and an annotation term, and the state term cancels in the softmax. The weights become a fixed property of the source sentence, and you can't tell from the loss that attention has stopped working.
- **Softmax over the wrong axis.** The weights sum to 1 over the source positions $j$, one row per target word. A softmax over target steps returns numbers that look plausible and train to something worse.
- **Forgetting to re-align the backward states.** The backward RNN produces its first state at $x_{T_x}$. Stored in the order they are computed, the backward half of $h_j$ belongs to word $T_x - j + 1$. The sentence still translates, only worse, and no shape check catches the mistake.
- **Padding.** The paper trains on minibatches of 80 sentences (Section 4.2) and sorts sentences by length within chunks of 1,600 pairs (Appendix B.2), which reduces padding but doesn't remove it. It doesn't say how the attention treats padded positions. In a padded batch, set the scores of padding positions to a large negative number before the softmax, so that attention never lands on padding.
- **Reading attention maps as explanations.** Figure 3 shows what the decoder weighted, and the paper calls the alignments "linguistically plausible" (Section 5.2.1). A weight does not prove that the model needed that word. It is a useful picture, not a proof of how the model decides.
- **Starting the alignment weights too small.** With near-zero initial weights in $W_a$, $U_a$, and $v_a$, the gradient is a product of small numbers and learning stalls. The script `sharpening.py` stays at entropy 1.6094 for 300 steps when the standard deviation is 0.01. The paper initializes $W_a$ and $U_a$ with a standard deviation of 0.001 and $v_a$ at zero (Appendix B.1). The paper trains for about 5 days with Adadelta (Section 4.2), so a small experiment with a short budget should start from larger weights.
- **Mixing up the baseline.** RNNencdec in the tables is Cho et al.'s encoder-decoder, which has 1000 hidden units in the encoder and decoder (Section 4.2). It is a close relative of the sequence-to-sequence model, not the same network.

## Further reading

- [GroundHog](https://github.com/lisa-groundhog/GroundHog) is the code the paper's authors point to for their implementation (footnote 4 of the paper).
- [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025), by Luong et al., compares dot-product and other scoring functions with the additive form and introduces local attention.
- [On the Properties of Neural Machine Translation: Encoder-Decoder Approaches](https://arxiv.org/abs/1409.1259), by Cho et al., is the study that measured the drop with sentence length.
- [Attention and Augmented Recurrent Neural Networks](https://distill.pub/2016/augmented-rnns/), by Olah and Carter, is a visual account of attention on top of RNNs.
- [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/), by Lilian Weng, surveys attention variants after this paper.
