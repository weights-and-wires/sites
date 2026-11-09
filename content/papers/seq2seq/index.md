---
id: "seq2seq"
title: "Sequence to Sequence Learning with Neural Networks"
authors: "Sutskever et al."
year: 2014
venue: "NeurIPS 2014"
arxiv: "1409.3215"
url: "https://arxiv.org/abs/1409.3215"
track: nlp-llm
tier: extended
est_hours: 3
status: full
summary: "Deep networks need inputs and outputs of a fixed size, which rules out tasks such as translation where the lengths differ and vary. The paper reads the source sentence with one deep LSTM, then runs a second LSTM as a language model that starts from the final state and generates the target one token at a time until it emits an end-of-sentence symbol. Reversing the source sentence shortens the gap between the first source and target words and made training much easier, and the model beat a phrase-based translation system on English to French."
related_course: "ml: sequence models"
prereqs:
  - id: lstm
    reason: "Both the encoder and the decoder are deep LSTMs, and the model relies on the LSTM to carry information across the long gap between a source word and its translation."
background: [cross-entropy, teacher-forcing, beam-search]
course_links:
  - title: "Recurrent cells and backpropagation through time"
    url: "https://ml.ifkash.dev/lessons/041-recurrent-cells-and-bptt"
  - title: "Attention encoder-decoders"
    url: "https://speech.ifkash.dev/lessons/032-attention-encoder-decoders"
self_check:
  - q: 'Without reversal, a source of $n = 25$ tokens is followed by a target of 25 tokens, and source token $j$ translates to target token $j$. What is the lag between each pair, and what are the minimal and average lags after reversing only the source (Section 3.3)?'
    a: 'Without reversal every pair is $n = 25$ positions apart. After reversal the lags are $1, 3, 5, \ldots, 49$, so the minimal lag is 1 and the average is still 25, which is what Section 3.3 says: the average distance is unchanged, but the first source words sit next to the first target words. `scripts/reversal_lags.py` prints these values.'
    see: lstm
  - q: 'The target is `W X Y Z <EOS>` (Figure 1). Which tokens does the decoder read at training time, and what does it read at test time?'
    a: 'With teacher forcing the decoder reads the true previous tokens: `<EOS> W X Y Z`, where the first `<EOS>` is the one that ends the source. At test time the true prefix is not available, so the decoder reads its own previous output, chosen by beam search (Section 3.2).'
    see: teacher-forcing
  - q: 'In the beam search of Section 3.2, what happens to a hypothesis when `<EOS>` is appended, and why does a beam of size 1 sometimes return a less probable translation than a beam of size 2?'
    a: 'The hypothesis leaves the beam and joins the set of complete hypotheses. A beam of size 1 commits to the single best token at each step, so it can''t recover when a lower-ranked first token leads to a much more probable sentence. In the example on this page, beam size 1 returns `a <EOS>` with probability 0.27, and beam size 2 finds `b <EOS>` with probability 0.36.'
    see: beam-search
---

## The problem

Before this paper, deep neural networks worked well only when inputs and targets had a fixed size. The introduction says it plainly: a DNN can only be applied to problems whose inputs and targets can be sensibly encoded with vectors of fixed dimensionality. Translation, speech recognition, and question answering all map a sequence of unknown length to another sequence of unknown length. A plain recurrent network (RNN) maps a sequence to a sequence only when the alignment between input and output is known ahead of time (Section 2).

Section 2 describes the obvious fix and why it was hard. Read the input with one RNN to get a fixed-size vector, then unroll a second RNN from that vector to produce the output. In principle this works, because the vector holds all the relevant information. In practice it creates long-term dependencies, because the first word of the source is many steps away from the first word of the target when the output is produced. The paper cites related work that read a sentence into a vector (Kalchbrenner and Blunsom, Cho et al.) and notes that Cho et al. used it only to rescore hypotheses from a phrase-based system (Section 1).

The failure shows up as a lag. Put the source and the target in one sequence, and the model has to carry the source word $a$ across the whole source and the start of the target before it can use it to produce its translation $\alpha$. The longer the sentence, the longer this minimal lag, and the smaller the gradient signal that connects $\alpha$ to $a$ (see [the LSTM page](/lstm/) for why that signal shrinks). The paper's answer has three parts: an LSTM at both ends, an autoregressive decoder that stops by itself, and a data trick that shortens the lag.

## Key ideas

This page uses the paper's notation. $(x_1, \ldots, x_T)$ is the source sequence, $(y_1, \ldots, y_{T'})$ is the target sequence, $T'$ may differ from $T$, and $v$ is the fixed-size vector that summarizes the source. The paper numbers only Equations 1 and 2. Section 2 also gives an unnumbered RNN recurrence, and Section 3.2 gives an unnumbered training objective. This page cites those by section. The paper doesn't use the terms "teacher forcing" or "cross-entropy"; the page uses them where they name what the paper does. Every number below comes from a script in `scripts/`, and you can run any of them from the paper folder.

### Encode into a vector, decode autoregressively

**Intuition.** One LSTM reads the source one token at a time. Its last state is $v$, a summary of everything it read. A second LSTM starts from $v$ and acts as a language model: at each step it predicts a distribution over the next target token, and the chosen token becomes the input at the next step. It ends by emitting a special `<EOS>` symbol, which is how the model learns to produce outputs of any length.

**Math.** The goal is to estimate $p(y_1, \ldots, y_{T'} \mid x_1, \ldots, x_T)$. The encoder gives $v$, the last hidden state of the LSTM after reading the source, and the decoder is a standard LSTM language model whose initial hidden state is set to $v$ (Section 2, Equation 1):

$$
p(y_1, \ldots, y_{T'} \mid x_1, \ldots, x_T) = \prod_{t=1}^{T'} p(y_t \mid v, y_1, \ldots, y_{t-1})
$$

Each factor is a softmax over the whole target vocabulary. Every sentence, source and target, ends with `<EOS>`, so the model defines a distribution over sequences of every length (Section 2). In Figure 1 the model reads `A B C <EOS>` and then produces `W X Y Z <EOS>`.

The paper's actual models differ from this description in three ways (Section 2). They use two different LSTMs, one for the source and one for the target, which adds parameters at negligible computational cost. They use four layers rather than one, and each extra layer cut perplexity by nearly 10% (Section 3.4). And they reverse the source, which is the third key idea below.

**Tiny example, computed by `scripts/likelihood_example.py`.** Take the target of Figure 1, `W X Y Z <EOS>`, and a decoder that emits the following logits in teacher-forced mode. The script prints the per-step probabilities of the true token:

| Step $t$ | Decoder input | True $y_t$ | $p(y_t \mid \ldots)$ | $\log p$ |
|---|---|---|---|---|
| 1 | `<EOS>` | W | 0.6488 | -0.4327 |
| 2 | W | X | 0.3222 | -1.1326 |
| 3 | X | Y | 0.7784 | -0.2505 |
| 4 | Y | Z | 0.5637 | -0.5732 |
| 5 | Z | `<EOS>` | 0.6488 | -0.4327 |

The product is $0.059512$, and the sum of the logs is $-2.8216$, which matches $\log 0.059512$. A sum is easier to work with than a product of five small numbers, and a real sentence has 20 or more factors.

**Size of the real model, computed by `scripts/param_count.py`.** Section 3.4 gives four layers of 1000 cells each, 1000-dimensional embeddings, and vocabularies of 160,000 (source) and 80,000 (target). A layer has $4 \cdot (1000 \cdot 1000 + 1000 \cdot 1000) = 8$M weights, ignoring biases. So the encoder has 32M and the decoder has 32M, which matches the 32M per LSTM that Section 3.4 reports. The embeddings add 160M and 80M, and the output softmax adds 80M, for 384M in total, matching the paper's count. The state $v$ is $2 \cdot 4 \cdot 1000 = 8000$ numbers (the hidden and cell state of each layer), which matches the 8000 numbers per sentence that Section 3.4 reports.

Exercise 2 asks you to compute this log probability.

### Train by maximum likelihood on the true prefix

**Intuition.** To train, you know the whole target, so you can feed the decoder the true previous tokens and ask it to predict each next one. You don't need to wait for the model to generate anything. This is what the field now calls teacher forcing. The loss for one step is the negative log probability of the true token, which is the cross-entropy between the one-hot target and the decoder's softmax.

**Math.** Section 3.2 states the objective, which is unnumbered. The model is trained by maximizing the average log probability of a correct translation $T$ given the source sentence $S$ over the training set $\mathcal{S}$:

$$
\frac{1}{|\mathcal{S}|} \sum_{(T, S) \in \mathcal{S}} \log p(T \mid S)
$$

The paper writes $T$ for a target sentence and $S$ for a source sentence here. This page writes $y$ and $x$ elsewhere, to match Equation 1. Substituting Equation 1 turns each $\log p(T \mid S)$ into a sum of $T'$ per-token terms. At step $t$ the decoder reads the true token $y_{t-1}$, and the first step reads the `<EOS>` that ended the source. So the decoder inputs are the targets shifted right by one, following the single-LSTM description in Section 2 and Figure 1.

**Tiny example, computed by `scripts/likelihood_example.py`.** The example above has $\log p(y \mid x) = -2.8216$ over 5 tokens, so its per-token cross-entropy is $2.8216 / 5 = 0.5643$ nats and its per-token perplexity is $\exp(0.5643) = 1.7582$. If a second training pair had $\log p = -3.0$, the objective over the two pairs would be $(-2.8216 - 3.0) / 2 = -2.9108$.

Section 3.3 reports test perplexity (5.8 without reversal, 4.7 with) but doesn't define it. The usual per-token definition is $\exp(-\frac{1}{N}\sum \log p)$, and the exercise uses it. The paper's models used plain SGD with a learning rate of 0.7, uniform initialization in $[-0.08, 0.08]$, batches of 128 sentences, and a gradient-norm clip at 5 (Section 3.4). The clip is $g \leftarrow 5g / s$ when $s = \lVert g \rVert_2 > 5$, and the paper says LSTMs can still have exploding gradients.

### Decode with a small beam

**Intuition.** At test time you don't know the true prefix, and you want the most probable translation. Choosing the best token at each step (greedy decoding) can paint you into a corner. The paper keeps a few partial translations alive instead.

**Math.** The goal is the most likely translation (Section 3.2, Equation 2):

$$
\hat{T} = \arg\max_{T}\; p(T \mid S)
$$

The search is a left-to-right beam search with at most $B$ partial hypotheses, where each is a prefix of some translation. At each step every hypothesis in the beam is extended by every word in the vocabulary, and all but the $B$ most likely, by the model's log probability, are discarded. As soon as `<EOS>` is appended to a hypothesis, the hypothesis leaves the beam and joins the set of complete hypotheses (Section 3.2). The paper says the decoder is approximate but simple.

The paper doesn't specify the stopping rule, how to break ties, or whether the `<EOS>` extensions count against the $B$ slots. Exercise 3 reads it as follows: rank all extensions together, keep the top $B$, then move the ones that end in `<EOS>` to the complete set. It also stops when the beam is empty or at a maximum length, and it doesn't normalize by length.

**Tiny example, computed by `scripts/beam_example.py`.** A table-driven toy model has the vocabulary $\{a, b, \texttt{<EOS>}\}$. The first token is $a$ with probability 0.6 or $b$ with probability 0.4. After $a$, `<EOS>` has probability 0.45. After $b$, `<EOS>` has probability 0.90.

| Beam size $B$ | Best complete hypothesis | $\log p$ | $p$ |
|---|---|---|---|
| 1 | `a <EOS>` | -1.3093 | 0.2700 |
| 2 | `b <EOS>` | -1.0217 | 0.3600 |

With $B = 1$ the decoder commits to $a$ at the first step, because $0.6 > 0.4$, and ends with $0.6 \cdot 0.45 = 0.27$. With $B = 2$ the hypothesis $b$ survives the first step, and $0.4 \cdot 0.90 = 0.36$ wins. With $B = 2$ the script returns two complete hypotheses, `b <EOS>` and `a <EOS>`.

Table 1 in the paper shows the same effect at scale. For an ensemble of five reversed LSTMs, beam size 1 gives 33.00 BLEU, beam size 2 gives 34.50, and beam size 12 gives 34.81. The paper concludes that a beam of size 2 provides most of the benefit (Section 3.2).

Exercise 3 asks you to implement this beam search.

### Reverse the source

**Intuition.** Suppose the source is $a, b, c$ and the target is $\alpha, \beta, \gamma$, where $\alpha$ translates $a$, and so on. The model reads `a b c` and then writes `α β γ`. The word $a$ is far from $\alpha$, as far as the length of the source. If you feed the source reversed, `c b a`, then $a$ sits right next to the point where $\alpha$ is produced. The other pairs get further apart, but the first pair is close, and that is the pair the optimizer can use to start learning.

**Math.** Section 2 says the LSTM is asked to map $c, b, a$ to $\alpha, \beta, \gamma$ instead of $a, b, c$ to $\alpha, \beta, \gamma$. Section 3.3 gives the argument in terms of the minimal time lag. Reversing the words in the source leaves the average distance between corresponding words unchanged, but the first few source words become close to the first few target words, so the minimal time lag falls and backpropagation can more easily establish a connection between source and target. The paper says it has no complete explanation, and that the account is a belief. Only the source is reversed, never the target.

**Tiny example, computed by `scripts/reversal_lags.py`.** Let the source and the target each have $n$ tokens, with source token $j$ translating to target token $j$. In the concatenated sequence `source + target`, source token $j$ is at index $j$ and target token $j$ is at index $n + j$ (counting from 0). Without reversal every lag is $n$. With reversal source token $j$ is at index $n - 1 - j$, and the lag is $2j + 1$.

| $n$ | Reversed? | Lags | Mean | Min |
|---|---|---|---|---|
| 4 | no | 4, 4, 4, 4 | 4.0 | 4 |
| 4 | yes | 1, 3, 5, 7 | 4.0 | 1 |
| 5 | no | 5, 5, 5, 5, 5 | 5.0 | 5 |
| 5 | yes | 1, 3, 5, 7, 9 | 5.0 | 1 |
| 25 | no | all 25 | 25.0 | 25 |
| 25 | yes | 1, 3, ..., 49 | 25.0 | 1 |

The mean lag is $n$ both ways. Only the minimum changes, from $n$ to 1. This toy assumes a monotone alignment, which is a simplification. Real translation reorders words, so the effect on a real corpus is a matter of measurement, and the paper measures it. On WMT'14 English to French, reversing the source lowered test perplexity from 5.8 to 4.7 and raised the BLEU of decoded translations from 25.9 to 30.6 (Section 3.3). Table 1 shows 26.17 and 30.59 for a single LSTM at beam size 12.

Exercise 1 asks you to compute these lags.

### The fixed-size bottleneck

**Intuition.** Everything the decoder knows about the source passes through $v$, which has a fixed size no matter how long the source is. Eventually a long enough input has more information than $v$ can hold.

**What the paper found.** The paper doesn't claim the bottleneck bites at the lengths it tested. It reports that the LSTM did not suffer on very long sentences (Section 1), and Figure 3 shows no degradation on sentences with fewer than 35 words and only a minor degradation on the longest ones. It credits reversing the source for this, and calls the result surprising, because the authors had expected the LSTM to fail on long sentences due to its limited memory (Section 5). The 8000-number state of the real model is large. The bottleneck is the concern that Bahdanau et al. raised, and the one that attention addresses (Section 1 cites it). See [Bahdanau et al.](/bahdanau/) for the fix.

**Tiny example, computed by `scripts/bottleneck.py`.** The script uses a linear toy, not an LSTM, so it shows the counting argument and not the paper's result. The encoder is $h_t = A h_{t-1} + E_{:, x_t}$ with a random orthogonal $A$, and $v = h_T$. The decoder is a linear readout fitted by ridge regression that predicts each of the $n$ source tokens from $v$. The state size is $H = 24$, the vocabulary has $V = 4$ tokens, and chance is 0.25. Each position adds $V - 1 = 3$ degrees of freedom, so the state can be exact only while $n (V - 1) \le H$, which is $n \le 8$:

| Source length $n$ | Held-out token accuracy |
|---|---|
| 4 | 1.000 |
| 6 | 1.000 |
| 8 | 0.999 |
| 10 | 0.983 |
| 12 | 0.927 |
| 16 | 0.805 |
| 24 | 0.663 |
| 32 | 0.585 |

Accuracy is near perfect until $n (V - 1)$ reaches $H$, and then it falls. A trained nonlinear LSTM isn't limited in this way, and the paper's results show it handles sentences of 35 words and more with 8000 numbers. The toy only shows the shape of the limit: for any fixed $H$, there is a length past which a single vector can't hold the source.

Exercise 4 asks you to build this experiment.

<!-- exercises -->

## Pitfalls

- **Reversing the target too.** Only the source is reversed. If you reverse both, the lag pattern returns to the original one, and the minimal lag is $n$ again.
- **Expecting reversal to change the average lag.** It doesn't (Section 3.3). The benefit comes from the minimum, not the mean.
- **Treating the reversal as a requirement of the architecture.** It's a data trick, and the paper says it has no complete explanation. The paper also expects a standard RNN to train on reversed sources but didn't verify this (Section 5). With attention, which gives the decoder direct access to every source position, the trick matters much less.
- **Forgetting that the first decoder input is the source's `<EOS>`.** In the single-LSTM description of Section 2 and Figure 1, the model reads `<EOS>` and then predicts the first target token, and the following inputs are the target shifted right by one. The paper has no separate start-of-sequence token, and it doesn't say what its two-LSTM implementation feeds at the first decoder step. Many later implementations add a `<BOS>` token, which is a different convention.
- **Feeding the true prefix at test time.** Teacher forcing is for training. At test time the decoder reads its own predictions. A model that is only ever trained on true prefixes can see inputs at test time that it never saw in training, and the paper doesn't address this.
- **Summing or averaging log probabilities inconsistently.** The training objective averages over pairs (Section 3.2). Within a pair the log probability is a sum over tokens (Equation 1). Averaging inside a sentence changes what the beam search ranks.
- **Expecting beam search to be exact.** Equation 2 is an argmax over all translations, and the beam gives an approximation (Section 3.2). With $B = 1$ it is greedy decoding.
- **Ranking by raw log probability.** A sum of log probabilities favors short hypotheses, because every extra token adds a negative term. The paper describes no length normalization, so a decoder that follows it can prefer short outputs. Exercise 3 inherits this.
- **Misreading Table 1.** The 34.81 BLEU result is an ensemble of five reversed LSTMs with beam size 12, and the paper says it was penalized for out-of-vocabulary words because the target vocabulary has 80,000 words (Section 1). A single reversed LSTM scores 30.59.
- **Reading the toy bottleneck as the paper's result.** The paper's LSTM didn't degrade below 35 words. The linear toy shows why a fixed vector must fail eventually. It doesn't show where an LSTM fails.
- **Forgetting gradient clipping.** The model trains with plain SGD at a learning rate of 0.7. Without the norm clip at 5 (Section 3.4), exploding gradients can wreck a run.

## Further reading

- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473), by Bahdanau, Cho, and Bengio, adds attention to remove the fixed-vector bottleneck. See this site's page on [Bahdanau et al.](/bahdanau/).
- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078), by Cho et al., is the closely related encoder-decoder that introduced the GRU.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) replaces the recurrence with attention. See this site's page on the [Transformer](/transformer/).
- [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850), by Alex Graves, is the LSTM language model that the decoder builds on. The paper uses the LSTM formulation from this work (Section 2).
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/), by Christopher Olah, is an illustrated walkthrough of the cell.
