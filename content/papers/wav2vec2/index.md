---
id: "wav2vec2"
title: "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations"
authors: "Baevski et al."
year: 2020
venue: "NeurIPS 2020"
arxiv: "2006.11477"
url: "https://arxiv.org/abs/2006.11477"
track: speech
tier: core
est_hours: 5
status: full
summary: "wav2vec 2.0 pre-trains a speech model on unlabeled audio. A convolutional encoder turns the waveform into latent frames, spans of those frames are masked, and a Transformer must pick the true quantized latent for each masked frame from a set of distractors. After pre-training, the model is fine-tuned with a CTC loss on labeled data, and ten minutes of transcripts are enough to reach a usable word error rate."
related_course: "speech: self-supervised speech"
prereqs:
  - id: transformer
    reason: "The context network is a standard Transformer encoder, and the paper replaces its fixed positional encodings with a convolutional layer."
  - id: ctc
    reason: "Fine-tuning puts a linear layer on the context network and trains it with the CTC loss, which needs no frame-level alignment."
background: [contrastive-learning, vector-quantization, self-supervised-learning]
course_links:
  - title: "Self-supervised speech"
    url: "https://speech.ifkash.dev/lessons/036-self-supervised-speech"
self_check:
  - q: 'The feature encoder has strides $(5,2,2,2,2,2,2)$. What is its total stride, and why does one second of 16 kHz audio give 49 frames instead of 50?'
    a: 'The total stride is $5 \cdot 2^6 = 320$ samples, or 20 ms. Each unpadded convolution loses $k - s$ samples at the edge, so the length is $\lfloor (n - k)/s \rfloor + 1$ layer by layer, and 16,000 samples end at 49 frames. A single division $16000 / 320 = 50$ ignores that loss (Section 4.2).'
    see: transformer
  - q: 'In Equation 3, the distractors come from other masked time steps of the same utterance. Why not from other utterances, or from unmasked steps of the same one?'
    a: 'The paper samples distractors uniformly from other masked time steps of the same utterance (Section 3.2). Drawn from the same utterance, they share the speaker and the channel, so the model can only win by predicting the content of the sound, not by recognizing who is speaking. Masked steps are used because those are the ones whose targets the model has to predict.'
    see: contrastive-learning
  - q: 'The diversity loss in Equation 4 averages the softmax over a batch before it takes the entropy. What goes wrong if you take the entropy of each step and then average?'
    a: 'A codebook that gives a different confident answer at every step has a uniform average distribution but zero per-step entropy. The loss is meant to push the average use of the $V$ entries toward uniform, so that the model does not collapse to a few codewords. Averaging per-step entropies would reward confident choices, which is the opposite of the goal.'
    see: vector-quantization
---

## The problem

Speech recognition needs transcribed audio, and transcripts are expensive. The paper's introduction notes that current systems need thousands of hours of transcribed speech, and that this is not available for most of the nearly 7,000 languages spoken worldwide (Section 1). Unlabeled audio is far easier to collect. Before this paper, the best use of it was a pipeline: learn discrete units in one step, then train a context model over them in another (Section 1). Another option was self-training, which needs several rounds of labeling, filtering, and retraining.

Consider a concrete setting. You have 10 minutes of transcribed speech, which is 48 recordings with an average length of 12.5 seconds (Section 5.1), and a large pile of audio without text. A supervised [CTC](/ctc/) model trained on those 48 recordings overfits immediately. The question is how to get the pile of audio to teach the model what speech sounds like, before it ever sees a label.

wav2vec 2.0 answers with one training objective over raw audio, followed by a standard fine-tune. With a LARGE model pre-trained on 53.2k hours of LibriVox audio (LV-60k) and fine-tuned on 10 minutes of labels, it reaches 4.8 / 8.2 WER on the clean and other Librispeech test sets with a Transformer language model (Section 5.1, Table 1). With a single hour of labels, the same model reaches 2.9 / 5.8 on the test sets. The previous best method, noisy student, reached 4.2 / 8.6 with 100 hours of labels (Section 5.1, Table 1). With all 960 hours of labels, it reaches 1.8 / 3.3 (Section 5.2, Table 2).

## Key ideas

The notation follows the paper: raw audio $\mathcal{X}$, the feature encoder $f: \mathcal{X} \mapsto \mathcal{Z}$ that produces latent representations $z_1, \dots, z_T$, the context network $g: \mathcal{Z} \mapsto \mathcal{C}$ that produces $c_1, \dots, c_T$, and the quantization module $\mathcal{Z} \mapsto \mathcal{Q}$ that produces targets $q_t$ (Section 2). The paper's symbol for the temperature of the contrastive loss is $\kappa$, and the Gumbel softmax temperature is $\tau$. One difference: the code and tables here index time from 0, while the paper starts at 1. Every number comes from a script in `scripts/`. Run any script from the paper folder to reproduce its output.

### The feature encoder and its downsampling

**Intuition.** A waveform at 16 kHz has 16,000 numbers per second, which is far too many for a Transformer to attend over. The encoder is a stack of strided temporal convolutions that turns the waveform into one latent vector per 20 ms. Each block is a convolution, layer normalization, and a GELU (Section 2). The Transformer then sees one token per frame.

**Math.** The encoder has seven blocks with 512 channels each, strides $(5,2,2,2,2,2,2)$, and kernel widths $(10,3,3,3,3,2,2)$ (Section 4.2). With no padding, a layer with kernel $k$ and stride $s$ maps a length $n$ to $\lfloor (n-k)/s \rfloor + 1$. The total stride is the product of the strides, and the receptive field of one output frame is $1 + \sum_i (k_i - 1) \prod_{j<i} s_j$.

**Tiny example, computed by `scripts/encoder_example.py`.** The total stride is $5 \cdot 2^6 = 320$ samples, which is 20 ms at 16 kHz. The receptive field is $10 + 2\cdot5 + 2\cdot10 + 2\cdot20 + 2\cdot40 + 80 + 160 = 400$ samples, or 25 ms. That matches the paper, which gives a stride of about 20 ms and a receptive field of 400 samples (Section 4.2). The paper reports an output frequency of 49 Hz. The script explains the 49. One second of audio, 16,000 samples, shrinks layer by layer to 3199, 1599, 799, 399, 199, 99, and 49 frames. The naive $16000 / 320 = 50$ is one too many. The BASE crop of 250k samples gives 781 frames, and here the naive division agrees.

Exercise 1 asks you to compute the frame count and the receptive field.

### Masking in the latent space

**Intuition.** BERT hides words and asks the model to recover them. wav2vec 2.0 hides stretches of latent frames. It masks spans, not single frames, because neighbouring 20 ms frames of the same phoneme are nearly identical, and a model could fill in a single missing frame by copying its neighbours.

**Math.** The model samples a proportion $p$ of all time steps without replacement as start indices, then masks the following $M$ consecutive steps from each start. Spans may overlap (Section 3.1). The masked positions of $z$ are replaced with one trained feature vector that is shared by all masked steps, and the inputs to the quantizer are not masked (Section 3.1). Pre-training uses $p = 0.065$ and $M = 10$ (Section 4.2).

**Tiny example, computed by `scripts/masking_example.py`.** The masked fraction is not $pM = 0.65$, because spans overlap. A step stays unmasked only when none of the $M$ starts that could cover it was chosen, so the fraction is about $1 - (1 - p)^M = 0.4894$. The paper says "approximately 49%" and a mean span length of 14.7, or 299 ms (Section 4.2 and Appendix A). The script simulates 2,000 crops of 781 frames and gets a masked fraction of 0.4903 and a mean merged-span length of 14.56 frames, which is 291 ms. The small gap from 14.7 is within what the sampling details (which start positions are allowed near the end of a crop) could produce, and the paper doesn't give those details. For a hand check, take $T = 12$, starts $\{1, 4, 6\}$, and $M = 3$. The spans are $[1,4)$, $[4,7)$, and $[6,9)$. They cover steps 1 to 8, so 8 steps are masked, not 9.

Exercise 2 asks you to implement the masking.

### Quantized targets and the Gumbel softmax

**Intuition.** The model needs targets to predict, and it has no labels. It uses its own latents, but with a twist. The targets are a discretized version of $z$, which makes the task a choice among a finite set of codewords and strips out detail the Transformer shouldn't chase, such as speaker and background (Section 5.4). The codebook is learned jointly with everything else.

**Math.** The quantizer uses product quantization with $G$ codebooks, or groups, each with $V$ entries $e \in \mathbb{R}^{V \times d/G}$. It chooses one entry per group, concatenates the $G$ chosen vectors, and applies a linear map $\mathbb{R}^d \mapsto \mathbb{R}^f$ to get $q$ (Section 2). Choosing an entry is a hard decision, so the paper uses the Gumbel softmax with the straight-through estimator. The encoder output $z$ is mapped to logits $l \in \mathbb{R}^{G \times V}$, and the probability of entry $v$ in group $g$ is (Section 2, Equation 1):

$$
p_{g,v} = \frac{\exp\bigl((l_{g,v} + n_v)/\tau\bigr)}{\sum_{k=1}^{V} \exp\bigl((l_{g,k} + n_k)/\tau\bigr)}, \qquad n = -\log(-\log(u)),\ u \sim \mathcal{U}(0,1)
$$

The paper prints the numerator as $\exp(l_{g,v} + n_v)/\tau$, with the division outside the exponential. The intended form, which is the standard Gumbel softmax, divides inside. In the forward pass, the model takes $i = \arg\max_j p_{g,j}$. In the backward pass, it uses the true gradient of the Gumbel softmax outputs (Section 2). The temperature $\tau$ is annealed from 2 to a minimum of 0.5 for BASE or 0.1 for LARGE, by a factor of 0.999995 per update (Section 4.2).

**Tiny example, computed by `scripts/contrastive_example.py`.** With $G = 2$ groups of $V = 320$ entries, the model can address $320^2 = 102{,}400$ distinct codewords, the theoretical maximum of 102.4k codewords that Section 4.2 gives, while storing only $2 \cdot 320$ vectors.

One design choice matters a great deal. Quantization is used only for the targets. The Transformer input stays continuous. Table 4 of the paper compares the four combinations on the dev sets from a reduced pre-training setup, averaged over three seeds (Section 5.4):

| Context-network input | Target | Average WER |
|---|---|---|
| Continuous | Quantized (the paper's choice) | 7.97 |
| Quantized | Quantized | 12.18 |
| Quantized | Continuous | 11.18 |
| Continuous | Continuous | 8.58 |

Quantizing the input throws away information the context network needs, and continuous targets make the task easier in an unhelpful way. The paper reports that training accuracy of picking the right latent rises from 62% to 78.0% with continuous targets, which it reads as the targets capturing speaker and background artifacts (Section 5.4).

### The contrastive loss and the diversity loss

**Intuition.** For each masked step, the Transformer produces $c_t$. The model must say which of $K + 1$ candidate quantized vectors is the true $q_t$ for that step. The other $K$, the distractors, are the quantized latents of other masked steps from the same utterance (Section 3.2). A good $c_t$ has to carry information about the sound at step $t$, which it can only get from the unmasked context around it.

**Math.** The total loss is $\mathcal{L} = \mathcal{L}_m + \alpha \mathcal{L}_d$ (Section 3.2, Equation 2). The contrastive part is (Section 3.2, Equation 3):

$$
\mathcal{L}_m = -\log \frac{\exp(\mathrm{sim}(c_t, q_t)/\kappa)}{\sum_{\tilde q \sim Q_t} \exp(\mathrm{sim}(c_t, \tilde q)/\kappa)}, \qquad \mathrm{sim}(a, b) = \frac{a^\top b}{\|a\|\,\|b\|}
$$

Here $Q_t$ holds $q_t$ and the $K$ distractors, $\kappa = 0.1$, and $K = 100$ (Section 4.2). This is the same cross-entropy as a softmax classifier, with candidates in place of classes. Because the true target is among the candidates, the loss is at most $\log(K+1)$ when the model cannot tell candidates apart, and the script prints $\log 101 = 4.6151$ for $K = 100$.

A contrastive loss alone could be minimized by a codebook that uses a handful of entries, since distinct masked steps would then often have the same quantized target. The diversity loss pushes against that. It maximizes the entropy of the softmax distribution averaged across a batch, separately for each group. That softmax has no Gumbel noise and no temperature (Section 3.2, Equation 4):

$$
\mathcal{L}_d = \frac{1}{GV} \sum_{g=1}^{G} -H(\bar p_g) = \frac{1}{GV} \sum_{g=1}^{G} \sum_{v=1}^{V} \bar p_{g,v} \log \bar p_{g,v}
$$

The weight is $\alpha = 0.1$ (Section 4.2). A footnote in Section 3.2 says that the implementation maximizes perplexity instead, which the paper calls equivalent.

**Tiny example, computed by `scripts/contrastive_example.py`.** Let $c_t = (2, 1)$, $q_t = (1, 1)$, and two distractors $(1, -1)$ and $(-1, 0.5)$. The cosine similarities are 0.9487, 0.3162, and $-0.6000$. With $\kappa = 1$, the softmax probabilities are 0.5735, 0.3047, and 0.1219, and the loss is $-\ln 0.5735 = 0.5561$. With $\kappa = 0.1$, the logits are 9.49, 3.16, and $-6.00$, the true candidate gets probability 0.9982, and the loss is 0.0018. A small $\kappa$ sharpens the softmax, so the same similarities produce a much smaller loss. Scaling $q_t$ by 10 leaves the cosine at 0.9487, while a raw dot product would grow from 3 to 30. That is why the paper uses the cosine.

For the diversity loss with $G = 1$ and $V = 4$: a uniform average distribution gives $H = 1.3863$ and $\mathcal{L}_d = -0.3466$, which is the minimum, $-\ln V / V$. The skewed average $(0.7, 0.1, 0.1, 0.1)$ gives $-0.2351$, and a nearly collapsed $(0.97, 0.01, 0.01, 0.01)$ gives $-0.0419$. At the paper's $V = 320$, the minimum is $-0.01803$. Take two steps that each pick a different entry with certainty. The per-step entropy is 0, but the average distribution is $(0.5, 0.5)$ with entropy 0.6931. The loss uses the second number.

Exercise 3 asks you to implement the contrastive loss and the distractor sampler. Exercise 4 asks you to implement the diversity loss.

### Fine-tuning with CTC

**Intuition.** After pre-training, the model knows a lot about speech and nothing about characters. Fine-tuning adds a randomly initialized linear layer on top of the context network that maps each frame to the output vocabulary, and trains it with the [CTC](/ctc/) loss (Section 3.3). CTC fits because the labeled data is a transcript without alignment, and the encoder produces one frame per 20 ms, far more frames than characters.

**Math.** For Librispeech the vocabulary is 29 tokens: characters plus a word boundary token (Section 3.3). The loss is the CTC loss of the previous paper, $-\ln p(l \mid x)$ summed over alignments (see Section 3.1, Equation 3 of that paper), applied to the new linear layer's softmax. The feature encoder is not trained during fine-tuning, and for the first 10k updates only the output layer is trained before the Transformer is also updated (Section 4.3). The paper also applies a modified SpecAugment by masking time steps and channels of the encoder output (Sections 3.3 and 4.3). It uses a tri-state learning rate schedule: warm up for 10% of updates, hold for 40%, decay linearly for the rest (Section 4.3).

**Tiny example, computed by `scripts/encoder_example.py`.** CTC needs at least as many frames as labels plus adjacent repeats (see [CTC](/ctc/)). A 12.5-second recording, the average length of the 10-minute subset, is 200,000 samples and gives 624 frames. A transcript of that length has far fewer characters, so the 20 ms stride leaves room for a character-level CTC head.

<!-- exercises -->

## Pitfalls

- **Dividing by the total stride to get the frame count.** Unpadded convolutions lose $k - s$ samples per layer. For 1 second of audio, $16000 // 320 = 50$ but the encoder outputs 49. A mismatch of one frame breaks batching code that builds masks or CTC input lengths from that formula.
- **Reading $p$ as the masked fraction.** The parameter $p = 0.065$ is the probability that a step is a span start. With $M = 10$ and overlaps, about 49% of the steps are masked, not 6.5% and not 65%.
- **Using a dot product instead of the cosine.** Equation 3 normalizes both vectors. With a raw dot product, the model can lower the loss by growing norms, and the temperature $\kappa = 0.1$ no longer has the intended scale.
- **Leaving the positive in the wrong place or out of the denominator.** The candidate set $Q_t$ includes $q_t$. If the denominator sums only over distractors, the loss can go negative and the gradient pushes in the wrong direction.
- **Sampling distractors from the wrong set.** They are drawn from other masked steps of the same utterance. Sampling from all steps would include steps whose targets the model could see in its input, and sampling from the positive itself makes a candidate identical to the target.
- **Applying the Gumbel noise and temperature to the diversity loss.** Equation 4 uses the plain softmax over the logits $l$ and no noise (Section 3.2). Noise would make the diversity term random.
- **Taking the entropy before the average.** Equation 4 averages $p$ over the batch first. The reverse order measures how confident each step is, not how evenly the codebook is used.
- **Quantizing the Transformer input.** Only the targets are quantized. The ablation shows a 4-point WER gap (12.18 against 7.97) when the input is quantized too (Section 5.4, Table 4).
- **Masking the quantizer's input.** The mask applies to the Transformer input only, and the quantizer sees the unmasked latents (Section 3.1). If it saw masked ones, the targets would be the shared mask vector.
- **Training the feature encoder during fine-tuning.** The paper freezes it, and for the first 10k updates it also freezes the Transformer (Section 4.3).
- **Comparing WER across setups without the language model.** The headline results use a Transformer or 4-gram LM at decoding. Table 1 lists the LM per row, and the paper's Appendix C has numbers without one.

## Further reading

- [fairseq wav2vec 2.0 examples](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec) are the official code and models, linked from the paper's first page.
- [Hugging Face Transformers: Wav2Vec2](https://huggingface.co/docs/transformers/model_doc/wav2vec2) documents a widely used PyTorch implementation, including the CTC fine-tuning head.
- [vq-wav2vec](https://arxiv.org/abs/1910.05453) is the two-step predecessor that this paper replaces with joint learning (Section 1 of the paper).
- [Effectiveness of self-supervised pre-training for speech recognition](https://arxiv.org/abs/1911.03912) is the discrete-BERT baseline that appears in Table 1.
- [Categorical reparameterization with Gumbel-softmax](https://arxiv.org/abs/1611.01144) introduces the estimator that Section 2 uses.
- [Libri-light](https://arxiv.org/abs/1912.07875) defines the limited-label splits (10 minutes, 1 hour, 10 hours) used in Section 4.1.
