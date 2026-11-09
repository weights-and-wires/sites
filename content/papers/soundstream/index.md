---
id: "soundstream"
title: "SoundStream: An End-to-End Neural Audio Codec"
authors: "Zeghidour et al."
year: 2021
venue: "IEEE/ACM TASLP 2022"
arxiv: "2107.03312"
url: "https://arxiv.org/abs/2107.03312"
track: speech
tier: extended
est_hours: 4
status: full
summary: "SoundStream compresses audio with a convolutional encoder that turns a 24 kHz waveform into 75 embeddings per second, a residual vector quantizer that replaces each embedding with a sum of code vectors from a cascade of small codebooks, and a convolutional decoder that rebuilds the waveform. Each quantizer stage encodes what the earlier stages missed, so the cost grows linearly with the bit budget where a single codebook would grow exponentially, and training each step with a random number of stages lets one model run at any bitrate from 3 to 18 kbps. The whole system trains end to end with a hinge adversarial loss, a feature-matching loss, and a multi-scale spectral loss."
related_course: "speech: neural codecs"
prereqs:
  - id: gan
    reason: "SoundStream trains its decoder against waveform and STFT discriminators, so you need the generator and discriminator game before the paper's hinge-loss variant of it makes sense."
  - id: wavenet
    reason: "The encoder and decoder are stacks of causal dilated convolutions, the same operation WaveNet uses to reach far back in time without looking at the future."
background: [residual-vector-quantization, adversarial-losses]
course_links:
  - title: "Neural audio codecs"
    url: "https://speech.ifkash.dev/lessons/050-neural-audio-codecs"
  - title: "Residual and scalar quantization"
    url: "https://speech.ifkash.dev/lessons/049-residual-and-scalar-quantization"
self_check:
  - q: 'A codec runs at 6 kbps on 75 embedding frames per second. How many bits does each frame get, how large would one plain codebook be, and how many vectors does a residual quantizer with $N_q = 8$ stages store instead?'
    a: 'Each frame gets $6000 / 75 = 80$ bits. A plain codebook would need $2^{80}$ entries. With $N_q = 8$ stages of $\log_2 N = 10$ bits each, each stage has $N = 1024$ entries, so the quantizer stores $8 \cdot 1024 = 8192$ vectors and can still express $2^{80}$ distinct sums.'
    see: residual-vector-quantization
  - q: 'The discriminator gives a generated frame the logit $D(G(x)) = 1.5$. What are the generator''s adversarial loss and the discriminator''s loss on that frame under the hinge losses of Equations 1 and 2, and what does the first value say about the gradient?'
    a: 'The generator''s term is $\max(0, 1 - 1.5) = 0$, so this frame gives the generator no gradient: the discriminator is already fooled past the margin. The discriminator''s term is $\max(0, 1 + 1.5) = 2.5$, so it still pushes the logit down.'
    see: adversarial-losses
  - q: 'Why does the paper make every convolution causal, and what fixes the architectural latency of the codec?'
    a: 'Causal convolutions pad only the past, so an output never waits for future samples. The latency is then set by the stride product $M = 2 \cdot 4 \cdot 5 \cdot 8 = 320$, because the encoder needs one full frame of 320 samples before it can emit an embedding. At 24 kHz that is 13.3 ms.'
    see: wavenet
---

## The problem

A speech or music call needs a bitstream that is small enough to send and good enough to listen to. Raw 24 kHz audio at 16 bits per sample is 384 kbps (the paper doesn't state a sample width, so the 16 bits are an assumption of this page). Traditional codecs such as Opus and EVS compress it with hand-designed transforms and quantizers. They work well at 12 kbps and above, and they degrade quickly below that. The paper's headline result (Abstract, Figure 1) is that SoundStream at 3 kbps beats Opus at 12 kbps in a listening test, and approaches EVS at 9.6 kbps.

A neural codec has to answer one hard design question: how do you turn a learned embedding into a few bits? Take the paper's own example (Section III-C, "Limitations of Vector Quantization"). The encoder emits 75 frames per second. At 6 kbps, each frame gets $6000 / 75 = 80$ bits. A plain vector quantizer, as in the VQ-VAE, maps each frame to one entry of a codebook. Spending 80 bits per frame needs a codebook of $2^{80}$ vectors, about $1.2 \times 10^{24}$. Nobody can store that, train it, or search it. Computed by `scripts/bitrate_example.py`.

The paper's answer has three parts. A residual vector quantizer reaches the same 80 bits with 8 codebooks of 1024 entries, 8192 vectors in all. A random choice of how many codebooks to use at each training step lets one model serve many bitrates. A mix of adversarial and reconstruction losses keeps the decoded audio sounding natural at low bitrates, where a pure waveform-error loss would blur it.

## Key ideas

The sections below use the paper's notation where it has one: $x \in \mathbb{R}^T$ for the waveform, $\mathrm{enc}(x) \in \mathbb{R}^{S \times D}$ for the embeddings, $N_q$ for the number of quantizers, $N$ for the codebook size, $Q_i$ for the $i$-th quantizer, and $\hat{y}$ for the quantized embedding. Two differences to know about. The paper writes $D$ both for the embedding width and, in Equations 1 to 3, for the discriminator. This page keeps $D$ for the discriminator and writes the embedding width as $D_{\mathrm{emb}}$ where both could appear. Also, the paper uses $n_q$ for the number of quantizers used at one training step and $N_q$ for the maximum, and so does this page. The paper numbers its equations 1 to 7 and its one algorithm as Algorithm 1; its sections use Roman numerals, such as Section III-C. Every number comes from a script in `scripts/`. Run any script from the paper folder to reproduce the output.

### Encoder, decoder, and the frame rate

**Intuition.** The encoder squeezes time. It turns 320 audio samples into one embedding vector, so the quantizer has 75 vectors per second to code instead of 24,000 samples.

**Math.** The encoder is a 1D convolution followed by $B_{\mathrm{enc}}$ blocks (Section III-A, Figure 3). Each block has three residual units with dilations 1, 3, and 9, then a strided convolution that downsamples and doubles the channels. Those dilations are the same device as in [WaveNet](/wavenet/): a small filter reaches far back when the gap between taps grows. All convolutions are causal, so padding is on the past side only. The strides multiply to a temporal resampling ratio $M$. For strides $(2, 4, 5, 8)$,

$$
M = 2 \cdot 4 \cdot 5 \cdot 8 = 320, \qquad S = T / M
$$

so $\mathrm{enc}(x) \in \mathbb{R}^{S \times D_{\mathrm{emb}}}$ (Section III-A). The decoder mirrors the encoder with transposed convolutions and the same strides in reverse (Section III-B). Because the convolutions are causal, the architectural latency is $M$ samples, one frame (Section V-D, "Latency").

**Tiny example, computed by `scripts/bitrate_example.py`.** At $f_s = 24000$ Hz and $M = 320$, one second has $24000 / 320 = 75$ frames, and one frame lasts $320 / 24000 = 13.3$ ms. Doubling the first stride to 4 gives $M = 640$, 37.5 frames per second, and 26.7 ms.

### Residual vector quantization

**Intuition.** Quantize coarsely first. Then quantize the error that is left, with a second codebook that only has to describe small corrections. Then quantize the error of that, and so on. Each stage works on a smaller residual, and the quantized value is the sum of what every stage chose.

**Math.** Algorithm 1 (Section III-C) takes $y = \mathrm{enc}(x)$ and $N_q$ vector quantizers $Q_i$, and does the following for every frame:

$$
\hat{y} \leftarrow 0, \qquad r \leftarrow y
$$

$$
\text{for } i = 1, \dots, N_q: \quad \hat{y} \leftarrow \hat{y} + Q_i(r), \qquad r \leftarrow r - Q_i(r)
$$

Each $Q_i$ returns the entry of its own codebook nearest to its input. The stored bitstream is the $N_q$ indices, and the decoder rebuilds $\hat{y}$ by summing the $N_q$ indexed vectors. After stage $i$, the residual is $r = y - \hat{y}_{1..i}$, where $\hat{y}_{1..i}$ is the sum of the first $i$ quantized values. The bit budget is split evenly: $r_i = r/N_q = \log_2 N$ bits per quantizer, so $N_q = 8$ and $r = 80$ give $N = 2^{80/8} = 1024$ (Section III-C). The paper also trains each codebook with exponential moving average updates (as in VQ-VAE-2), initializes each codebook with k-means on the first batch, and replaces a code vector that no frame has used for several batches by a random frame of the current batch. It tracks usage as a moving average with decay 0.99 and replaces a vector when the statistic drops below 2 (Section III-C). This page doesn't derive those updates.

**Tiny example, computed by `scripts/rvq_example.py`.** Use $D_{\mathrm{emb}} = 2$ and three stages of four vectors each: $C_1 = \{(0,0), (4,0), (0,4), (4,4)\}$, $C_2 = \{(0,0), (-1,1), (1,-1), (1,1)\}$, and $C_3 = 0.25\, C_2$. Quantize $y = (3.2, 0.9)$.

| Stage | Nearest vector | $\hat{y}$ so far | Residual | Residual norm |
|---|---|---|---|---|
| 1 | $(4, 0)$, index 1 | $(4, 0)$ | $(-0.8, 0.9)$ | 1.2042 |
| 2 | $(-1, 1)$, index 1 | $(3, 1)$ | $(0.2, -0.1)$ | 0.2236 |
| 3 | $(0.25, -0.25)$, index 2 | $(3.25, 0.75)$ | $(-0.05, 0.15)$ | 0.1581 |

Twelve stored vectors reach $4^3 = 64$ different sums, which is the effect of the cascade. The stored indices are $[1, 1, 2]$. A decoder that gets only the first two indices rebuilds $(3, 1)$, which is a coarser but valid answer. The most common bug is to quantize $y$ at every stage instead of the residual. Stage 2 would then pick $(1, 1)$, the vector nearest to $y$, and two stages would give $(5, 1)$, an error of 1.8028 where the correct two stages give 0.2236.

Exercise 1 asks you to implement this algorithm.

**A bigger experiment, computed by `scripts/error_vs_nq.py`.** Train eight 16-entry stages by k-means, each on the residual of the one before, on 4000 Gaussian vectors with 8 dimensions. The mean squared error falls from 0.9938 with no quantizer to 0.5966 after stage 1 and 0.0166 after stage 8, and it falls at every stage. Each stage removes about 40 percent of the remaining error, so the error is roughly geometric in the number of stages. That rate is a property of this toy data, not a number from the paper. If you fit every stage on $y$ instead of the residual, the error stalls near 0.25. Exercise 2 asks you to write this experiment.

**Depth against width, from the paper.** For a fixed budget, you can spread the bits over many small codebooks or few large ones. Table II (Section V-D) compares three ways to spend 80 bits per frame at 6 kbps: 8 codebooks of 1024 entries (ViSQOL 4.01), 16 of 32 (3.98), and 80 of 2 (3.92). Fewer, larger codebooks code better and cost more to search. Eighty 1-bit quantizers lose only a little quality, which shows that a very deep residual stack trains without optimization trouble. Memory shows the other side. The three setups store 8192, 512, and 160 vectors. Computed by `scripts/bitrate_example.py`.

### Bitrate scalability and quantizer dropout

**Intuition.** Because a later stage only refines the sum, you can stop after any stage and still have a valid embedding of the same shape. That makes the bitrate a dial. But a model trained always with all stages is never asked to do well with just a few. Training with a random number of stages fixes that.

**Math.** To train for every bitrate at once, the paper modifies Algorithm 1: for each training example it samples $n_q$ uniformly at random in $[1; N_q]$ and uses only $Q_i$ for $i = 1, \dots, n_q$ (Section III-C). The paper calls this a structured dropout applied to the quantization layers. At inference, $n_q$ is chosen from the desired bitrate. The bitrate is

$$
R = \frac{f_s}{M} \cdot n_q \cdot \log_2 N
$$

This formula is not written in the paper, but it follows from the frame rate, from "$N_q \log_2 N$ bits to encode a single frame" (Section V-D), and from the bitrates the paper reports.

**Tiny example, computed by `scripts/bitrate_example.py`.** With $f_s / M = 75$ and $N = 1024$, each quantizer adds $75 \cdot 10 = 750$ bps. So 4 quantizers give 3 kbps, 8 give 6 kbps, and 16 give 12 kbps. The paper says its models cover 3 to 18 kbps (Abstract). That means 24 quantizers at the top, which is this page's arithmetic and not a number the paper states. With $N_q = 8$, each $n_q$ from 1 to 8 has probability $1/8$, so the expected $n_q$ is $4.5$ and the expected training bitrate is 3.375 kbps. Stage 1 is used on every step, and stage 8 on one step in eight.

The paper's Table III repeats the arithmetic for latency at a fixed 6 kbps and $N = 1024$. Strides $(1, 4, 5, 8)$ give $M = 160$, 40 bits per frame, and $N_q = 4$. Strides $(4, 4, 5, 8)$ give $M = 640$, 160 bits per frame, and $N_q = 16$. The script reproduces the $N_q$ column. It also finds frame durations of 6.7 ms and 26.7 ms, where the paper's table prints 7.5 ms and 26 ms for those rows. The $M$ values and the text's 13.3 ms for $M = 320$ agree with the script, so the 7.5 ms entry looks like a typo.

Section V-C reports the effect. A scalable model, trained with quantizer dropout, matches bitrate-specific models at 6 and 12 kbps. At 3 kbps it is only slightly worse. A model trained at 18 kbps without dropout and evaluated with fewer quantizers loses quality as the bitrate drops (Figure 7c).

Exercise 3 asks you to write the bitrate arithmetic and the dropout sampler.

### Adversarial and reconstruction losses

**Intuition.** A reconstruction loss alone averages over the many waveforms that fit a quantized embedding, and the average sounds dull. A discriminator pushes the output toward something that sounds like real audio, and a feature-matching loss gives the generator a stable signal from the discriminator's internal layers. A spectral loss keeps the energy in the right place. The structure is the generator and discriminator game of the [GAN](/gan/) paper. The loss functions are different.

**Math.** Let $G(x) = \mathrm{dec}(Q(\mathrm{enc}(x)))$ and $\hat{x} = G(x)$ (Section III-E). There are $K + 1$ discriminators: $k = 0$ is the STFT discriminator, and $k = 1, \dots, K$ with $K = 3$ are the waveform discriminators at three resolutions (Section III-D). Discriminator $k$ outputs $T_k$ logits along time, because the discriminators are fully convolutional. The discriminator minimizes (Section III-E, Equation 1)

$$
\mathcal{L}_D = \mathbb{E}_x\!\left[\frac{1}{K}\sum_k \frac{1}{T_k}\sum_t \max\bigl(0,\, 1 - D_{k,t}(x)\bigr)\right] + \mathbb{E}_x\!\left[\frac{1}{K}\sum_k \frac{1}{T_k}\sum_t \max\bigl(0,\, 1 + D_{k,t}(G(x))\bigr)\right]
$$

and the generator's adversarial loss is (Equation 2)

$$
\mathcal{L}_G^{\mathrm{adv}} = \mathbb{E}_x\!\left[\frac{1}{K}\sum_{k,t} \frac{1}{T_k}\max\bigl(0,\, 1 - D_{k,t}(G(x))\bigr)\right]
$$

The paper averages over time inside each discriminator, then over discriminators. Two more generator losses push toward fidelity. The feature loss averages the absolute difference between the discriminator's internal layer outputs $D^{(l)}_{k,t}$ for real and generated audio (Equation 3). The multi-scale spectral loss compares 64-bin mel-spectrograms $S^s_t$ at window lengths $s \in \{2^6, \dots, 2^{11}\}$ with hop $s/4$, using an L1 term on the spectrogram and an L2 term on its logarithm, weighted by $\alpha_s = \sqrt{s/2}$ (Equations 4 and 5). The total is (Equation 6)

$$
\mathcal{L}_G = \lambda_{\mathrm{adv}}\mathcal{L}_G^{\mathrm{adv}} + \lambda_{\mathrm{feat}}\mathcal{L}_G^{\mathrm{feat}} + \lambda_{\mathrm{rec}}\mathcal{L}_G^{\mathrm{rec}}
$$

with $\lambda_{\mathrm{adv}} = 1$, $\lambda_{\mathrm{feat}} = 100$, and $\lambda_{\mathrm{rec}} = 1$. Compared with the original GAN objective in the prerequisite paper, which uses logarithms of the discriminator output, the hinge losses clip: once a logit is past the margin of 1, that term gives no gradient. The paper doesn't explain why it chose hinge over other losses. It follows MelGAN and HiFi-GAN, which it cites as the basis for its decoder and loss design (Section II).

**Tiny example, computed by `scripts/losses_example.py`.** One discriminator ($K = 1$) with four logits. Real logits $[1.5, 0.2, 2.0, -0.3]$ give per-logit costs $[0, 0.8, 0, 1.3]$, a mean of 0.525. Fake logits $[-0.5, 0.8, -2.0, 1.2]$ give discriminator costs $\max(0, 1 + D)$ of $[0.5, 1.8, 0, 2.2]$, a mean of 1.125. So $\mathcal{L}_D = 1.65$. The generator's costs $\max(0, 1 - D)$ on the same fake logits are $[1.5, 0.2, 3.0, 0]$, so $\mathcal{L}_G^{\mathrm{adv}} = 1.175$. The last fake logit, 1.2, is past the margin and costs the generator nothing.

For the averaging order, take two discriminators whose fake-term costs average 1.0 over 2 logits and 1.5 over 8 logits. The paper's order gives $(1.0 + 1.5)/2 = 1.25$. One flat mean over all 10 logits gives 1.4, which lets the longer discriminator dominate. The spectral scales are $\alpha_s = 5.66, 8, 11.31, 16, 22.63, 32$ for $s = 64$ to $2048$.

Exercise 4 asks you to write these losses.

### Joint compression and denoising

A conditioning bit lets the same model denoise or not. The paper feeds a two-way one-hot signal through FiLM layers, $\tilde{a}_{n,c} = \gamma_{n,c}\, a_{n,c} + \beta_{n,c}$ (Section III-F, Equation 7), at the bottleneck on either the encoder or the decoder side. Section V-E finds that denoising on the encoder side needs fewer bits, because the quantizer no longer spends them on noise. This page doesn't cover the enhancement results further.

<!-- exercises -->

## Pitfalls

- **Quantizing $y$ at every stage.** Algorithm 1 quantizes the running residual. Quantizing $y$ each time sums unrelated guesses, as the example above shows.
- **Choosing the code by dot product.** The nearest code vector is the one with the smallest distance. The largest dot product picks the longer vector when norms differ. A codebook with unequal norms exposes this at once.
- **Counting the bitrate of the wrong thing.** The bitrate of the codec is the index stream, $N_q \log_2 N$ bits per frame. It doesn't include side information, and it assumes a fixed-length code. The paper notes that entropy coding of the indices could save more, and it estimates the possible savings with a cross-entropy (Figure 7a, Section V-E). Its reported bitrates don't include those savings.
- **Summing strides.** $M$ is a product. A sum of $(2, 4, 5, 8)$ is 19, and a frame rate of $24000 / 19$ has no connection to the model.
- **Mixing up the two kinds of dropout.** Quantizer dropout removes whole codebook stages at the tail, always from stage $n_q + 1$ onward. It doesn't zero random units. A model trained without it still runs at lower $n_q$, but the quality falls off (Section V-C).
- **Codebook collapse.** A codebook entry that no frame selects gets no updates. The paper's k-means initialization and dead-code replacement exist for this reason (Section III-C), and a reimplementation that skips them risks leaving entries unused.
- **Dropping the feature-loss weight.** The paper sets $\lambda_{\mathrm{feat}} = 100$ against 1 for the other two terms (Section III-E). Copy all three weights together, because the paper does not report an ablation of them.
- **Reading ViSQOL as a mean opinion score.** The ablation tables report ViSQOL, an objective metric that the paper found strongly correlated with its listening tests (Section IV-B). The headline comparison against Opus and EVS is from the listening test, not from ViSQOL.

## Further reading

- The SoundStream paper on arXiv: [2107.03312](https://arxiv.org/abs/2107.03312).
- Course lesson, [Residual and scalar quantization](https://speech.ifkash.dev/lessons/049-residual-and-scalar-quantization), for the quantization side in more depth.
- Course lesson, [Neural audio codecs](https://speech.ifkash.dev/lessons/050-neural-audio-codecs), for how later codecs build on this design.
- The VQ-VAE paper (van den Oord et al., 2017), which the paper extends with the residual cascade (Section II).
