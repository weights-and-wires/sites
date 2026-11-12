---
id: "ds2"
title: "Deep Speech 2: End-to-End Speech Recognition in English and Mandarin"
authors: "Amodei et al."
year: 2015
venue: "ICML 2016"
arxiv: "1512.02595"
url: "https://arxiv.org/abs/1512.02595"
track: speech
tier: extended
est_hours: 5
status: full
summary: "Deep Speech 2 trains one recurrent network with the CTC loss to map a spectrogram straight to characters, in English and in Mandarin. Sequence-wise batch normalization, a short-utterance-first curriculum called SortaGrad, strided convolutions, and bigram outputs make a deep stack of recurrent layers trainable and cheap enough to train on 11,940 hours of English and 9,400 hours of Mandarin. A forward-only variant with a row convolution layer on top reaches nearly the same accuracy while streaming."
related_course: "speech: ASR systems"
prereqs:
  - id: ctc
    reason: "DS2 trains with the CTC loss and its blank-and-collapse mapping, and every idea on this page assumes that loss."
background: [spectrograms, batch-norm-for-rnns]
course_links:
  - title: "The short-time Fourier transform"
    url: "https://speech.ifkash.dev/lessons/007-the-short-time-fourier-transform"
  - title: "Neural acoustic models"
    url: "https://speech.ifkash.dev/lessons/028-neural-acoustic-models"
  - title: "CTC: many paths, one label"
    url: "https://speech.ifkash.dev/lessons/029-ctc-many-paths-one-label"
  - title: "Language model fusion and hotwords"
    url: "https://speech.ifkash.dev/lessons/032-language-model-fusion"
  - title: "Streaming encoders"
    url: "https://speech.ifkash.dev/lessons/036-streaming-encoders"
  - title: "An ASR training recipe"
    url: "https://speech.ifkash.dev/lessons/048-asr-recipe"
self_check:
  - q: 'CTC needs a blank between two equal neighbors, such as the two $l$ in "hello". How many output frames does "hello" need as five single-letter tokens, and how many as the bigrams `he`, `ll`, `o`?'
    a: 'The letters $h, e, l, l, o$ need $5 + 1 = 6$ frames, because the pair $l, l$ needs a blank between its members. The bigrams have no equal neighbors, so $3$ frames are enough. Fewer tokens is what lets DS2 stride further (Section 3.6).'
    see: ctc
  - q: 'A spectrogram frame is 320 samples of 16 kHz audio and the hop is 160 samples. How many frames does one second of audio give, and what is the spacing between FFT bins?'
    a: 'One second is 16000 samples, so there are $1 + (16000 - 320) / 160 = 99$ frames. The bin spacing is $16000 / 320 = 50$ Hz, so a 1000 Hz tone lands in bin 20.'
    see: spectrograms
  - q: 'Batch normalization computes a mean and variance over a minibatch. For a recurrent layer, which values does DS2 pool in the statistics of one hidden unit, and why does a batch of one utterance still work?'
    a: 'DS2 pools the unit''s value at every time step of every utterance in the minibatch (Section 3.2, Equation 8). A single utterance still gives $T$ values per unit, so the variance is not zero. Statistics over one time step alone would see only one value.'
    see: batch-norm-for-rnns
---

## The problem

A classic speech recognizer chains many hand-built parts: feature extraction, an acoustic model, a pronunciation lexicon, a language model, and speaker adaptation. Each part needs tuning, and a new language needs most of them rebuilt (Section 1). The first Deep Speech paper (DS1) replaced the chain with one recurrent network that reads a spectrogram and writes characters, trained with the [CTC loss](/ctc/). This page calls the next system DS2, as the paper does.

DS1 had a single recurrent layer. The paper argues that a model of that size cannot use thousands of hours of labeled speech, so it adds depth (Section 3). Depth brings four concrete problems.

- **Deep recurrent stacks are hard to optimize.** Table 1 trains four networks with 38 million parameters each, with the depth varied. Depth only pays off if training converges, and Figure 2 shows that the 9-layer network converges slowly without BatchNorm. The 9-layer network reaches 10.83 development WER without BatchNorm and 9.52 with it.
- **Long utterances dominate the early loss.** The CTC loss is a negative log of a sum of products of per-frame probabilities (Equation 9). Each factor is below 1, so the loss grows with the number of frames. With a per-frame probability of 0.9 on one path, the loss is 1.054 at 10 frames, 10.536 at 100 frames, and 105.361 at 1000 frames. A fixed learning rate then takes steps that are too large on long utterances, and early in training the recurrent state can explode (Section 3.3). The script `scripts/sortagrad_example.py` prints these numbers.
- **English cannot be strided as hard as Mandarin.** Cutting the number of time steps saves compute, but CTC needs at least one frame per output symbol. English has 14.1 characters per second in the paper's training data and Mandarin has 3.3 (Section 3.6, footnote 2). The sentence "the cat sat" has 11 characters. It cannot fit into 8 frames when it is spelled letter by letter (`scripts/stride_bigram_example.py`).
- **Bidirectional layers cannot stream.** A bidirectional recurrent layer needs the whole utterance before it can emit the first character (Section 3.7), which rules it out for low-latency products.

The rest of the paper answers these with batch normalization that suits recurrent layers, a curriculum, a better output alphabet, and a new layer. It also adds a large-scale data and compute pipeline. Section 6 reports that DS2 beats human transcribers on three of the four read-speech test sets in Table 13.

## Key ideas

The notation follows Section 3.1. $x$ is an utterance, a sequence of spectrogram frames $x_t$ with $t = 0, \dots, T - 1$. $h^l_t$ is the activation of layer $l$ at time step $t$, with $h^0 = x$. $W^l$ is the input-to-hidden matrix and $U^l$ the recurrent matrix of a layer. The output is a distribution $p(\ell_t \mid x)$ over characters plus the blank. Every number below comes from a script in `scripts/`. Run any script from the paper folder to reproduce its output. Numbers that the paper reports from its own experiments come from `scripts/paper_tables.py`, which copies the raw table values and derives the percentages.

### The architecture and the spectrogram input

**Intuition.** The model is a stack in three parts. Convolutions at the bottom read the spectrogram and shorten the time axis. Bidirectional recurrent layers in the middle carry context in both directions. Fully connected layers at the top feed a softmax over characters, and CTC trains the whole stack end to end. Section 3.1 describes the input as a spectrogram of power-normalized audio.

**Math.** A convolutional layer with context window $c$ computes (Section 3.1, Equation 1)

$$
h^l_{t,i} = f\big(w^l_i \circ h^{l-1}_{t-c:t+c}\big),
$$

where $\circ$ is the element-wise product of the $i$-th filter with the window and $f$ is the clipped ReLU $\sigma(x) = \min\{\max\{x, 0\}, 20\}$. Some layers use a stride of $s$ frames to shorten the sequence. A bidirectional layer sums the forward and backward activations, $h^l = \overrightarrow{h}^l + \overleftarrow{h}^l$ (Equation 2). For the simple recurrence of Equation 3, $\overrightarrow{h}^l_t = f(W^l h^{l-1}_t + \overrightarrow{U}^l \overrightarrow{h}^l_{t-1} + b^l)$, and the paper notes that the input-to-hidden weights $W^l$ are shared by both directions. The output layer is a softmax (Equation 5), and the model trains with CTC (Equation 9).

The text of the paper does not give the window or the hop of the spectrogram. This page assumes 20 ms and 10 ms, a common choice for speech. Section 3.5 runs the first layers as same convolutions, which keep the number of frequency bins and time steps, and Table 4 finds that 2-D convolutions over frequency and time help most on noisy data.

**Tiny example, computed by `scripts/spectrogram_example.py`.** Take one second of a 1000 Hz sine at 16 kHz. A 20 ms window is 320 samples and a 10 ms hop is 160 samples, so the number of frames is $1 + (16000 - 320)/160 = 99$. The real FFT of a 320-point frame has $320/2 + 1 = 161$ bins, spaced $16000/320 = 50$ Hz apart, so 1000 Hz sits in bin 20. The script confirms that every frame peaks in bin 20. A Hann window leaks half of the amplitude into each neighboring bin, which is a power ratio of $0.25$ and a log ratio of $\ln 0.25 = -1.386$. The script prints $-1.377$ for the symmetric window that `np.hanning` builds. With a rectangular window the neighbors are $-33.176$ nats below the peak, which is numerical noise.

For a 2-D layer with a frequency stride of 2, the 161 bins become $\lceil 161/2 \rceil = 81$. Three such layers (the 3-layer 2-D row of Table 4 uses frequency strides of 2, 2, and 2) give 81, 41, and 21 bins, so the later layers see a far shorter vector (`scripts/stride_bigram_example.py`).

Exercise 1 asks you to build this spectrogram.

### Sequence-wise batch normalization

**Intuition.** Batch normalization (BatchNorm) rescales each hidden unit to zero mean and unit variance over a minibatch. In a recurrent layer, each unit takes a value at every time step, so you must choose which values to pool. Normalizing each time step on its own looks natural, but the statistics then come from the batch alone, and the recurrence makes the time steps depend on each other. The paper pools over the minibatch and over all time steps, and it normalizes only the term that feeds the layer from below.

**Math.** BatchNorm in a feed-forward layer replaces $f(Wh + b)$ with $f(B(Wh))$ (Section 3.2), where (Equation 6)

$$
B(x) = \gamma \, \frac{x - \mathrm{E}[x]}{(\mathrm{Var}[x] + \epsilon)^{1/2}} + \beta.
$$

The bias $b$ is dropped because the mean removal cancels it. The paper tries two ways to extend this to a recurrent layer. The first normalizes the whole pre-activation, so the statistics cover a single time step (Equation 7):

$$
\overrightarrow{h}^l_t = f\big(B(W^l h^{l-1}_t + \overrightarrow{U}^l \overrightarrow{h}^l_{t-1})\big).
$$

The paper reports that this gives no optimization benefit. It also tries a running average over earlier time steps, which does not help and complicates backpropagation. The second way is sequence-wise normalization (Equation 8):

$$
\overrightarrow{h}^l_t = f\big(B(W^l h^{l-1}_t) + \overrightarrow{U}^l \overrightarrow{h}^l_{t-1}\big).
$$

For each hidden unit, the mean and variance come from all items in the minibatch and every time step of every sequence. Only the input-to-hidden term is normalized. The term $W^l h^{l-1}_t$ does not depend on the recurrent state, so you can compute it for all time steps in one matrix product and normalize it before the recurrence starts. Table 1 shows a gain that grows with depth, a 12% relative change at the deepest network, and a loss for the shallowest. At deployment, a batch of one utterance does not give good statistics, so DS2 stores running averages of the mean and variance from training and uses them (Section 3.2).

This page writes the statistics over valid positions only. In a batch of utterances with different lengths, the padded frames carry no information, so exercise 2 excludes them. The paper does not discuss padding.

**Tiny example, computed by `scripts/batchnorm_example.py`.** One unit, two utterances with values $[1, 2, 3]$ and $[4, 6]$. Sequence-wise statistics pool all five values: the mean is 3.2, the variance is 2.96, and the standard deviation is 1.7205. The normalized values are $[-1.2787, -0.6975, -0.1162]$ and $[0.465, 1.6275]$. Frame-wise statistics need equal lengths, so cut both utterances to two frames, $[1, 2]$ and $[4, 6]$. The per-step means are $[2.5, 4.0]$ and the variances are $[2.25, 4.0]$, so the output is $[[-1, -1], [1, 1]]$ no matter how large the original values were. With a batch of one utterance $[1, 2, 3]$, frame-wise statistics come from a single value at each step, so the variance is 0 and the output is 0 at every step. Sequence-wise statistics still give $[-1.2247, 0, 1.2247]$.

Table 1 gives 9.52 dev WER with BatchNorm for the 9-layer network, against 10.83 without it, which is $12.1\%$ relative (`scripts/paper_tables.py`). For the 5-layer network with one recurrent layer, BatchNorm moves the dev WER from 13.55 to 14.40, which is $6.3\%$ worse. Exercise 2 asks you to implement both normalizations.

### SortaGrad

**Intuition.** Short utterances give small losses and small gradients, so a network learns to align them first. Long utterances can produce large gradients and unstable recurrent state when the weights are random. SortaGrad starts with the short ones and then switches to a random order.

**Math.** The CTC loss sums over all alignments (Section 3.3, Equation 9):

$$
L(x, y; \theta) = -\log \sum_{\ell \in \mathrm{Align}(x, y)} \prod_{t}^{T} p_{\mathrm{ctc}}(\ell_t \mid x; \theta).
$$

The inner product has $T$ factors that are each below 1, so the loss tends to grow with $T$. The paper uses utterance length as a difficulty heuristic. In the first epoch, it visits minibatches in increasing order of the length of the longest utterance in the minibatch. After the first epoch, it visits the minibatches in random order. The paper suspects that long utterances have larger gradients under a fixed learning rate, and that they are more likely to make the internal state of the recurrent layers explode early.

Table 2 shows the effect on the 9-layer, 7-recurrent-layer model. Without BatchNorm, sorting lowers the dev WER from 11.96 to 10.83, which is $9.4\%$ relative. With BatchNorm, it lowers it from 9.78 to 9.52, which is $2.7\%$. The paper says the two techniques substitute for each other in part, and that both still help together.

**Tiny example, computed by `scripts/sortagrad_example.py`.** Twelve utterances have lengths $[7, 3, 9, 1, 5, 8, 2, 6, 4, 10, 12, 11]$. Sort them and cut minibatches of three, which gives the length groups $[1, 2, 3]$, $[4, 5, 6]$, $[7, 8, 9]$, and $[10, 11, 12]$. Epoch 0 visits the groups in order of their longest utterance, 3, 6, 9, 12. In epoch 1 the same four groups come in a random order, for example 12, 6, 9, 3. The minibatches stay fixed, and only their order changes. The paper does not spell out how it forms the minibatches, so this page assumes buckets of similar length, as in the [ASR recipe lesson](https://speech.ifkash.dev/lessons/048-asr-recipe).

This page has no exercise for SortaGrad because the script is short enough to read.

### Strides and bigram outputs

**Intuition.** A longer stride gives the recurrent layers fewer time steps, which saves compute and memory. CTC then has fewer frames to place the characters in. English has many characters per second, so a large stride runs out of frames. Replace some pairs of characters with a single token, and the transcript gets shorter, so the stride can grow again.

**Math.** Section 3.5 uses same convolutions, so a layer with time stride $s$ turns $T$ steps into $\lceil T / s \rceil$. The paper does not print this formula. It follows from the definition of a same convolution, and this page uses it. CTC can emit a label sequence of length $L$ only if the number of frames is at least $L$ plus the number of adjacent equal labels, because each such pair needs a blank between its members (see [CTC](/ctc/)). In Section 3.6, the paper turns unigram labels into non-overlapping bigrams. The sentence "the cat sat" becomes `th, e, space, ca, t, space, sa, t`. Words with an odd number of letters end in a single letter, and the space is its own token. The output set holds all bigrams that occur in the training set.

Table 5 compares strides 2, 3, and 4 for unigram and bigram outputs on a model with 35 million parameters. With a 5-gram language model, the bigram model keeps its error nearly flat as the stride grows, 9.66, 10.06, 9.93, while the unigram model degrades from 9.52 to 9.65 to 11.92. At stride 4, the bigram model is $16.7\%$ better. At stride 2, it is slightly worse, by $1.5\%$.

**Tiny example, computed by `scripts/stride_bigram_example.py`.** "the cat sat" has 11 unigram tokens, with no equal neighbors, and 8 bigram tokens, so CTC needs at least 11 and 8 frames. The word "hello" needs 6 frames as letters, because of the double $l$, and 3 as `he, ll, o`. For a toy utterance of 24 input frames, stride 3 leaves 8 steps, which fits the bigrams and not the 11 letters. Stride 4 leaves 6 steps, which fits neither. Mandarin does not need this treatment, because it has about 3.3 characters per second (footnote 2 of Section 3.6) and the paper strides it "in the straightforward way".

Every Table 4 row has a total time stride of 2, so 100 input frames become 50 steps in each of them. Exercise 4 asks you to write the bigram split, the output-length formula, and the CTC frame budget.

### Row convolution and unidirectional models

**Intuition.** A unidirectional recurrent layer sees only the past, which is not enough to decide the current character. A bidirectional layer sees everything, which breaks streaming. The paper observes that only a small amount of future context is needed. A row convolution layer on top of the recurrent stack gathers that context at the end, and everything below it can run forward only.

**Math.** With a future context of $\tau$ steps, take the matrix $h_{t:t+\tau} = [h_t, \dots, h_{t+\tau}]$ of size $d \times (\tau + 1)$ and a parameter matrix $W$ of the same size. The layer computes (Section 3.7, Equation 11)

$$
r_{t,i} = \sum_{j=1}^{\tau+1} W_{i,j}\, h_{t+j-1,\,i}, \qquad 1 \le i \le d.
$$

Each feature $i$ has its own filter over the current and next $\tau$ time steps, and features never mix. The paper places the layer above all recurrent layers for two reasons. First, the computation below it can stream with little delay. Second, the paper finds a better Mandarin CER than for its best bidirectional model. The paper does not say what happens near the end of an utterance. This page treats $h$ as zero beyond the last step.

The deployed Mandarin model in Section 7.4 has five forward-only recurrent layers with 2560 hidden units, one row convolution layer with $\tau = 19$, and one fully connected layer with 2560 units. Its CER is 6.10, against 5.81 for the research system, a degradation of $5.0\%$.

**Tiny example, computed by `scripts/row_conv_example.py`.** Take $d = 1$, $\tau = 2$, input $h = [1, 2, 3, 4]$ and weights $W = [1, 10, 100]$. The output is $r = [321, 432, 43, 4]$. For example, $r_0 = 1 \cdot 1 + 10 \cdot 2 + 100 \cdot 3 = 321$, and $r_3 = 1 \cdot 4$, because the next two steps are padding. An impulse at $t = 3$ with $W = [1, 2, 3]$ produces $r = [0, 3, 2, 1, 0, 0]$, so only the outputs at $t = 1, 2, 3$ see it, and the layer looks forward and not back. With $d = 2560$ and $\tau = 19$, the layer has $2560 \cdot 20 = 51{,}200$ weights. A $2560 \times 2560$ fully connected layer has 6,553,600 weights, which is 128 times more.

Exercise 3 asks you to implement row convolution.

### Language model in the beam search

**Intuition.** The network learns some language knowledge from its training transcripts, but those are small next to the text available on the web. DS2 adds an n-gram language model at decoding time. The beam search scores each hypothesis with the CTC network, the language model, and a bonus per word.

**Math.** The decoder looks for the transcription that maximizes (Section 3.8, Equation 12)

$$
Q(y) = \log p_{\mathrm{ctc}}(y \mid x) + \alpha \log p_{\mathrm{lm}}(y) + \beta \cdot \mathrm{word\_count}(y).
$$

The weight $\alpha$ sets how much the language model counts, and $\beta$ rewards more words. The paper tunes both on a development set. For English, the model is a Kneser-Ney smoothed 5-gram model trained with KenLM on Common Crawl text, with a vocabulary of 400,000 words. For Mandarin it is a character-level 5-gram model, and $\beta$ becomes a character insertion term (Section 3.9). Table 6 shows that the language model lowers the English dev WER of the 9-layer model from 14.93 to 9.52, a $36.2\%$ relative gain. For the 5-layer model with one recurrent layer, the gain is $48.2\%$ (27.79 to 14.39). The paper reads the shrinking gain as evidence that deeper networks learn a stronger implicit language model. Mandarin gains less, $27.2\%$ and $23.0\%$, because one Chinese character carries more information than one English letter.

**Tiny example, computed by `scripts/lm_rescoring_example.py`.** The log probabilities in this example are made up for illustration. Hypothesis A, "recognize speech", has $\log p_{\mathrm{ctc}} = -4.0$, $\log p_{\mathrm{lm}} = -6.0$, and 2 words. Hypothesis B, "wreck a nice beach", has $-3.5$, $-14.0$, and 4 words. With $\alpha = 0$ and $\beta = 0$, B wins, $-3.50$ against $-4.00$. With $\alpha = 0.5$ and $\beta = 0$, A wins, $-7.00$ against $-10.50$. A word bonus of $\beta = 2$ flips the result back to B, $-2.50$ against $-3.00$, which shows how $\beta$ counters the language model's pull toward short transcripts.

### Scale: data and compute

**Intuition.** The paper's claim is that an end-to-end model improves with more data, a larger model, and faster training. The architecture changes above are what make that scale practical.

**Math.** There is no new equation, and the results are empirical. The English system trains on 11,940 hours and the Mandarin system on 9,400 hours (Section 1 and Section 5). Section 5.3 finds that WER falls as a power law in the training set size, about 40% relative for each factor of 10 in data. Section 4 attributes the speed to data-parallel synchronous SGD, a custom all-reduce, and a GPU implementation of CTC that is 28.9 times faster than the CPU version on English (Table 8). The abstract reports a 7x speedup over the previous system, and Section 1 says that one model takes 3 to 5 days on 8 or 16 GPUs.

**Tiny example, computed by `scripts/paper_tables.py`.** If each factor of 10 in data multiplies WER by 0.6, a factor of 100 multiplies it by $0.6^2 = 0.36$. The two languages together use 21,340 hours. On the 3,300-utterance Baidu test set, DS2 lowers the WER from 24.01 to 13.59, a $43.4\%$ relative gain over DS1 (Table 12). On read speech, DS2 beats the human WER on WSJ eval'92 (3.60 against 5.03), WSJ eval'93 (4.98 against 8.08), and LibriSpeech test-clean (5.33 against 5.83). It does not on LibriSpeech test-other (13.25 against 12.69).

<!-- exercises -->

## Pitfalls

- **Pooling the wrong values in BatchNorm.** Per-time-step statistics (Equation 7) look natural, and the paper finds them ineffective. They also break down for a batch of one: with a variance of 0, every output is $\beta$. Pool over the batch and all time steps (Equation 8), and normalize only $W h^{l-1}_t$.
- **Letting padding into the statistics.** A batch of utterances with different lengths holds padded frames. If they enter the mean and variance, the statistics depend on how you group the utterances. Mask them out.
- **Using batch statistics at inference time.** The paper deploys with running averages of the mean and variance collected during training, because single utterances give poor statistics (Section 3.2).
- **Applying SortaGrad every epoch.** Section 3.3 sorts only in the first epoch and then returns to random order. Sorting every epoch removes the randomness that stochastic gradient descent relies on.
- **Sorting by the wrong key.** The paper orders minibatches by the length of the longest utterance in each one (Section 3.3). If your minibatches mix lengths, ordering by the shortest utterance gives a different, gentler order.
- **Striding past the CTC limit.** If the number of frames after striding is smaller than the label length plus the repeats, no alignment exists and the loss is infinite. Check `min_ctc_frames` for the longest transcript per frame budget before you raise the stride.
- **Forgetting the blank between repeats.** "hello" needs 6 frames as letters, not 5. Bigrams hide most of these repeats, because `ll` becomes one token.
- **Looking backward in a row convolution.** The layer uses the current step and the next $\tau$ steps. A version that reads earlier steps makes the model causal in the wrong direction and ignores the future context.
- **Mixing features in a row convolution.** Equation 11 has a separate filter for each feature $i$ and never combines features. A full matrix product over features is a different, much larger layer.
- **Comparing WER across tables without checking the set.** The paper reports results on an internal English development set of 2048 utterances (Section 3), a noisy set of 2048 utterances from CHiME (Table 4), an internal Baidu test set of 3,300 utterances (Table 12), and public sets (Table 13). Numbers from different tables are not comparable.
- **Treating the paper's human comparison as proof of human-level speech recognition.** The paper itself finds that DS2 is worse than humans on noisy speech and on accented speech (Tables 14 and 15), and the human numbers come from crowd workers who transcribe the audio.

## Further reading

- [Deep Speech: Scaling up end-to-end speech recognition](https://arxiv.org/abs/1412.5567), by Hannun et al., is DS1, the system that DS2 extends.
- [Batch Normalized Recurrent Neural Networks](https://arxiv.org/abs/1510.01378), by Laurent et al., is the sequence-wise normalization that the paper cites as its starting point.
- [Sequence Modeling with CTC](https://distill.pub/2017/ctc/), by Awni Hannun, explains the loss that DS2 trains with, with figures.
- The [streaming encoders lesson](https://speech.ifkash.dev/lessons/036-streaming-encoders) covers the idea behind Section 3.7, bounded future context, in modern encoders.
- [`warp-ctc`](https://github.com/baidu-research/warp-ctc) is a fast CTC library that the same group released. The paper does not release its training code.
