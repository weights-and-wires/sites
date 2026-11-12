---
id: "encodec"
title: "High Fidelity Neural Audio Compression"
authors: "Defossez et al."
year: 2022
venue: "TMLR 2023"
arxiv: "2210.13438"
url: "https://arxiv.org/abs/2210.13438"
track: speech
tier: core
est_hours: 4
status: full
summary: "EnCodec is an open neural audio codec: a streaming convolutional and LSTM encoder turns 24 kHz audio into 75 latent steps per second, a residual vector quantizer with up to 32 codebooks of 1024 entries turns each step into integer tokens, and a mirrored decoder rebuilds the waveform, so the bandwidth is a dial from 1.5 to 24 kbps. Training replaces SoundStream's waveform and STFT discriminators with one multi-scale STFT discriminator, and a loss balancer rescales each loss gradient so that a weight means the share of the gradient that loss contributes. A small Transformer over the tokens lets an arithmetic coder save about 25 to 40 percent of the bits, and the tokens themselves are what later models such as VALL-E use as their vocabulary."
related_course: "speech: neural codecs"
prereqs:
  - id: soundstream
    reason: "EnCodec keeps SoundStream's encoder, residual vector quantizer with random bandwidth at training time, decoder, and hinge-loss adversarial training, so you need that design before the paper's changes to the discriminator, the losses, and the training procedure make sense."
background: [codec-bitrate, multi-scale-spectral-loss]
course_links:
  - title: "Neural audio codecs"
    url: "https://speech.ifkash.dev/lessons/060-neural-audio-codecs"
  - title: "Residual and scalar quantization"
    url: "https://speech.ifkash.dev/lessons/059-residual-and-scalar-quantization"
self_check:
  - q: 'EnCodec at 48 kHz uses strides $(2, 4, 5, 8)$ and 1024-entry codebooks. How many codebooks give 3 kbps, and how many tokens per second is that?'
    a: 'The stride product is $M = 320$, so 48 kHz audio gives $48000 / 320 = 150$ latent steps per second. Each codebook adds $150 \cdot 10 = 1500$ bps, so 3 kbps is 2 codebooks, or $150 \cdot 2 = 300$ tokens per second.'
    see: codec-bitrate
  - q: 'A decoded signal $\hat{x}$ equals $-x$ (the waveform with its sign flipped). What is the multi-scale spectral loss of Equation 1, and what does that say about the time-domain term $\ell_t$?'
    a: 'The magnitude spectrogram of $-x$ equals that of $x$, so every $\lVert S_i(x) - S_i(\hat{x}) \rVert$ is zero and the spectral loss is zero. The L1 term $\ell_t = \lVert x - \hat{x} \rVert_1$ is $2 \lVert x \rVert_1$, so it is the term that notices phase and sign errors.'
    see: multi-scale-spectral-loss
  - q: 'In a residual vector quantizer with $N_q = 4$ stages, what does the decoder get from the bitstream for one latent step, and what is the quantized latent it feeds to the decoder network?'
    a: 'It gets 4 integer indices, one per stage. It looks up each index in that stage''s codebook and sums the 4 vectors to form $z_q$. Each stage quantized the residual that the earlier stages left, so the sum approximates the encoder output $z$.'
    see: soundstream
---

## The problem

A codec has to turn audio into a stream of bits that is small enough to send and good enough to hear. The paper's reference signal is 16-bit audio (Figure 3 caption). At 24 kHz mono that is $24000 \cdot 16 = 384$ kbps, and the model targets 1.5 to 24 kbps, which is 16 to 256 times smaller. Traditional codecs such as Opus and EVS are the baselines. In the paper's MUSHRA listening test (Table 1, clean speech), Opus at 6 kbps scores 30.1 out of 100, and EnCodec at the same 6 kbps scores 83.1.

[SoundStream](/soundstream/) showed that a neural encoder, a residual vector quantizer (RVQ), and an adversarially trained decoder can beat those codecs. Three practical problems remained, and they are the ones this paper addresses.

- **Training is fragile.** A codec has a reconstruction loss and several adversarial and feature-matching losses, and their gradients differ in scale by orders of magnitude. The weights $\lambda_i$ that you tuned for one loss scale do not carry to another, and a bad choice lets the discriminator overpower the decoder.
- **The discriminator zoo is heavy.** SoundStream used one STFT discriminator and several waveform discriminators. The paper asks whether one design is enough (Section 4.5.1, Table 2).
- **Fixed-length codes waste bits.** The tokens are not uniformly distributed, so a code of exactly 10 bits per token leaves room for an entropy coder (Section 3.3).

The paper also targets deployment. The 24 kHz model runs in streaming mode with a 13 ms initial latency and processes audio about 10 times faster than real time on one CPU thread (Section 4.6, Table 5). The tokens it produces became the shared vocabulary of later speech language models, for example [VALL-E](/valle/).

## Key ideas

The page uses the paper's notation. The encoder is $E$, the quantizer is $Q$, and the decoder is $G$. The latent is $z$, the quantized latent is $z_q$, and the reconstruction is $\hat{x}$. $N_q$ is the number of codebooks in use, and $\ell_t$, $\ell_f$, $\ell_g$, $\ell_{\mathrm{feat}}$, and $\ell_w$ are the time, frequency, adversarial, feature-matching, and commitment losses (Section 3.4). The paper numbers five equations: the spectral loss is Equation 1, the feature loss is Equation 2, the commitment loss is Equation 3, the generator objective is Equation 4, and the balancer is Equation 5. The discriminator and generator hinge losses are written in the text of Section 3.4 without a number. Section numbers are Arabic (Section 3.4), not Roman as in SoundStream. Every number below comes from a script in `scripts/`, and you can run any of them from the paper folder.

### The streaming encoder-decoder and the token rate

**Intuition.** The encoder shrinks time by a factor of 320, so each latent step stands for 320 audio samples. A streaming codec can output audio as soon as it has received one step of input, and the latency is that one step.

**Math.** The encoder $E$ is a 1D convolution with $C = 32$ channels and kernel size 7, then $B = 4$ convolution blocks, a two-layer LSTM, and a final 1D convolution with kernel size 7 and $D$ output channels (Section 3.1). Each block is one residual unit (two kernel-3 convolutions and a skip connection) followed by a strided convolution whose kernel is twice its stride. The channels double at each downsampling. The strides are $(2, 4, 5, 8)$, so the temporal downsampling factor is

$$
M = 2 \cdot 4 \cdot 5 \cdot 8 = 320.
$$

The paper doesn't name $M$; that symbol comes from SoundStream. The paper states the result instead: 75 latent steps per second at 24 kHz and 150 at 48 kHz (Section 3.1). The decoder mirrors the encoder with transposed convolutions and reversed strides. Compared with SoundStream, the residual units here have no dilations, and an LSTM sits at the bottleneck on both sides. The activation is ELU.

The streamable variant puts all convolution padding before the first time step. For a transposed convolution with stride $s$ it outputs the first $s$ steps and keeps the rest in memory until the next frame arrives. As a result, the model outputs 320 samples (13 ms) as soon as it receives its first 320 samples (Section 3.1). The non-streamable variant splits the padding evenly, cuts the audio into 1 s chunks with 10 ms overlap, and normalizes each chunk. That costs 1 s of initial latency (Section 4.6, Table 5). The streamable model replaces layer normalization over time with weight normalization, because the layer-norm statistics need the whole chunk.

The quantizer is the RVQ of SoundStream, with at most 32 codebooks of 1024 entries, or 16 codebooks at 48 kHz (Section 3.2). Each codebook costs $\log_2 1024 = 10$ bits per step. Given $z$ with shape $[B, D, T]$, the quantizer returns integer indices of shape $[B, N_q, T]$, and the decoder input is the sum of the selected codebook entries (Section 3.2).

**Tiny example, computed by `scripts/tokens_example.py`.** At 24 kHz, one second has $24000 / 320 = 75$ steps, and one step lasts 13.3 ms. Each codebook adds $75 \cdot 10 = 750$ bps and 75 tokens per second.

| $N_q$ | Tokens per second | Bitrate (kbps) | Compression of 16-bit mono |
|---|---|---|---|
| 2 | 150 | 1.5 | 256x |
| 4 | 300 | 3 | 128x |
| 8 | 600 | 6 | 64x |
| 16 | 1200 | 12 | 32x |
| 32 | 2400 | 24 | 16x |

For 48 kHz stereo, one second has 150 steps and 16 codebooks give 24 kbps. The script reproduces the compression column of Table 4: 256 times at 6 kbps and 64 times at 24 kbps. MP3 at 64 kbps is $48000 \cdot 2 \cdot 16 / 64000 = 24$ times, as in the table.

The paper says that the model selects "a number of codebooks as a multiple of 4" at training time, "corresponding to a bandwidth 1.5, 3, 6, 12 or 24 kbps" (Section 3.2). Those two statements disagree. The bandwidth 1.5 kbps is 2 codebooks, which isn't a multiple of 4, and the five bandwidths match $N_q \in \{2, 4, 8, 16, 32\}$. This page uses the list of bandwidths, which Section 3.4 repeats under "Multi-bandwidth training", and treats "multiple of 4" as a loose description. SoundStream instead drew $n_q$ uniformly from $1, \dots, N_q$. Exercise 4 asks you to write this arithmetic.

### RVQ trained with EMA codebooks and a commitment loss

**Intuition.** A codebook entry that no latent is nearest to never gets a gradient, so the codebook can collapse onto a few entries and waste most of its bits. The training recipe fixes this. A moving average pulls each entry toward the latents assigned to it, and any entry that is not used gets replaced by a real latent from the current batch.

**Math.** The paper follows the recipe of Jukebox (Dhariwal et al., 2020) and SoundStream (Section 3.2). Each selected codebook entry is updated by an exponential moving average with decay 0.99. Entries that are not used are replaced by a candidate sampled from the current batch. A straight-through estimator passes the decoder's gradient to the encoder as if quantization were the identity function. A commitment loss, the mean squared error between the quantizer's input and its output with no gradient through the output, pulls the encoder toward its codes. For each residual step $c \in \{1, \dots, C\}$, with $C$ set by the bandwidth target of the batch, $z_c$ is the current residual and $q_c(z_c)$ is the nearest entry of codebook $c$ (Section 3.4, Equation 3):

$$
\ell_w = \sum_{c=1}^{C} \lVert z_c - q_c(z_c) \rVert_2^2.
$$

The paper writes $C$ here and $N_q$ elsewhere; they are the same count. The paper gives no threshold for "not used", and it doesn't write the EMA update as a formula. This page writes the standard version. For codebook entry $k$, let $n_k$ be the number of latents in the batch assigned to it and $m_k$ their sum. The statistics are $N_k \leftarrow \gamma N_k + (1 - \gamma) n_k$ and $M_k \leftarrow \gamma M_k + (1 - \gamma) m_k$ with $\gamma = 0.99$, and the entry is $e_k = M_k / N_k$. An entry with $N_k$ below a threshold is replaced.

**Tiny example, computed by `scripts/losses_example.py`.** Take the toy three-stage quantizer from the [SoundStream](/soundstream/) page with $y = (3.2, 0.9)$. The squared residual norms after each stage are $1.2042^2 = 1.45$, $0.2236^2 = 0.05$, and $0.1581^2 = 0.025$, so $\ell_w = 1.5250$. The first stage contributes 95 percent of the loss.

**A bigger experiment, computed by `scripts/rvq_usage.py`.** The script trains a 4-stage RVQ with 64 entries per stage on 8-dimensional vectors drawn from 10 Gaussian clusters, for 300 steps of 512 vectors, and starts the codebooks far from the data (standard deviation 10 against clusters at about 3). Without replacement, stage 1 ends up with 3 of 64 entries in use and a perplexity of 2.6. The error after stage 1 stays at 34.9 against 63.0 before quantizing. With replacement at threshold 0.5, stage 1 uses 62 of 64 entries at perplexity 57.7, and the error after stage 1 drops to 1.51. After 4 stages, the error is 0.159 with replacement and 6.84 without. The initialization is artificially bad, so the size of the gap is a property of this toy setup. The direction is what the paper's recipe is for. Exercise 1 asks you to write the EMA step with dead-entry replacement and the usage measure.

### Reconstruction in time and frequency

**Intuition.** The model needs two kinds of reconstruction signal. A frequency-domain loss compares spectrograms at several window sizes, so it cares about where the energy is, and it is blind to phase. A time-domain loss compares the waveforms directly and cares about phase.

**Math.** The time term is $\ell_t(x, \hat{x}) = \lVert x - \hat{x} \rVert_1$. The frequency term combines an L1 and an L2 comparison of mel-spectrograms at several scales (Section 3.4, Equation 1):

$$
\ell_f(x, \hat{x}) = \frac{1}{|\alpha| \cdot |s|} \sum_{\alpha_i \in \alpha} \sum_{i \in e} \lVert S_i(x) - S_i(\hat{x}) \rVert_1 + \alpha_i \lVert S_i(x) - S_i(\hat{x}) \rVert_2
$$

$S_i$ is a 64-bin mel-spectrogram of a normalized STFT with window $2^i$ and hop $2^i / 4$, and $e = 5, \dots, 11$ is the set of scales, so the windows run from 32 to 2048 samples. The coefficients $\alpha_i$ balance the L1 and L2 terms, and the paper sets $\alpha_i = 1$, unlike Gritsenko et al. (2020). The paper writes $|s|$ in the denominator but defines $e$ for the scales, so read $|s|$ as the number of scales. Compare SoundStream, which weighted the L2 term by $\sqrt{s/2}$ on log spectrograms.

This page's code and exercise simplify one thing: they use linear magnitude STFTs and skip the mel filterbank and the 64 bins. The norms are sums and Euclidean norms over the whole spectrogram, as written. A mean over bins would change only the scale of the loss. Section 3.4 explains why that scale is not a tuning problem, in the balancer below.

**Tiny example, computed by `scripts/losses_example.py`.** The signal is a 440 Hz sine, 0.25 s at 24 kHz, on 7 scales. If $\hat{x} = g x$, then $S_i(\hat{x}) = g S_i(x)$, so both norms scale by $|1 - g|$ and the loss equals $|1 - g|$ times the loss against silence. The script prints 166.7556 for $g = 0.9$ and 833.7780 for $g = 0.5$, which are 0.1 and 0.5 times the loss against $\hat{x} = 0$. If you squared the L2 term, this relation would break. For $\hat{x} = -x$ the spectral loss is 0, while $\ell_t = 7639.4$. A model trained on the spectral loss alone is free to flip signs and shift phases, and $\ell_t$ is what rules that out. Exercise 3 asks you to write both terms.

### The multi-scale STFT discriminator

**Intuition.** A discriminator that looks at spectrograms at five time-frequency resolutions catches artifacts of different kinds. A short window sees clicks and timing. A long window sees pitch structure and harmonics. One such discriminator replaced SoundStream's waveform and single-STFT discriminators.

**Math.** The MS-STFT discriminator is $K = 5$ identical sub-networks, each on the complex-valued STFT with real and imaginary parts concatenated as channels, at window lengths $[2048, 1024, 512, 256, 128]$ (Section 3.4, Figure 2). Each sub-network is a 2D convolution with 32 channels, three 2D convolutions with dilation 1, 2, and 4 along time and stride 2 along frequency, and a final convolution with kernel $3 \times 3$ that outputs logits. LeakyReLU and weight normalization follow. Figure 2 labels the hop as $h = w / 4$. The text gives the first kernel as $3 \times 8$, and the figure gives $3 \times 9$, so the paper disagrees with itself on this detail.

The adversarial losses are the hinge losses of SoundStream, now averaged over the $K$ discriminators (Section 3.4, unnumbered). The generator's term is $\ell_g(\hat{x}) = \frac{1}{K} \sum_k \max(0, 1 - D_k(\hat{x}))$ and the discriminator's is $\ell_d(x, \hat{x}) = \frac{1}{K} \sum_{k=1}^{K} \max(0, 1 - D_k(x)) + \max(0, 1 + D_k(\hat{x}))$. The paper omits the average over time that SoundStream took inside each discriminator, and states these as per-discriminator values. The generator also gets a relative feature-matching loss over the $L$ layers of each discriminator (Section 3.4, Equation 2):

$$
\ell_{\mathrm{feat}}(x, \hat{x}) = \frac{1}{KL} \sum_{k=1}^{K} \sum_{l=1}^{L} \frac{\lVert D_k^l(x) - D_k^l(\hat{x}) \rVert_1}{\operatorname{mean}\bigl(\lVert D_k^l(x) \rVert_1\bigr)}
$$

The mean in the denominator runs over all dimensions. SoundStream's feature loss had no such division, so a layer with large activations no longer dominates.

Three training details go with it. The discriminator is updated with probability 2/3 at 24 kHz and 1/2 at 48 kHz, because it overpowers the decoder easily. There is a dedicated discriminator per bandwidth, and each batch picks one bandwidth and updates only the matching discriminator. At 48 kHz the windows double and the discriminator trains every two batches (Section 3.4).

**Tiny example, computed by `scripts/msstft_shapes.py` and `scripts/losses_example.py`.** For 1 s of 24 kHz audio without padding, the five windows see 43, 90, 184, 372, and 747 frames of 1025, 513, 257, 129, and 65 bins. Each input holds about 88,000 to 97,000 numbers, and the five views trade time resolution against frequency resolution. The shortest window sees about 10 frames per latent step and the longest sees fewer than 1. For the relative feature loss, take one layer with real activations $[1, -2, 3, 0]$ and generated activations $[1.5, -1, 2, 0.5]$. The mean absolute difference is 0.75, the mean absolute real activation is 1.5, and the relative term is 0.5. Scaling both activations by 10 gives the same 0.5.

**What the ablation shows.** Table 2 compares discriminator setups on SI-SNR, ViSQOL, and MUSHRA. MS-STFT alone scores 6.67, 4.35, and 77.5. SoundStream's setup (MSD plus Mono-STFT) scores 5.99, 4.22, and 62.9, and adding MPD to MS-STFT gives a small MUSHRA gain to 79.0. MPD alone has the highest SI-SNR (7.35) and a MUSHRA of 60.7. So the objective metric SI-SNR and the listening test rank the setups differently, which is the reason the paper runs listening tests.

### The loss balancer

**Intuition.** The gradients of the adversarial and reconstruction losses can differ in size by a factor of a thousand, and they drift during training. A weight $\lambda_i$ on a loss then means nothing by itself. The balancer divides each loss gradient by a running estimate of its own norm, so that $\lambda_i$ becomes the share of the final gradient that loss controls.

**Math.** Take the losses $(\ell_i)_i$ that depend only on the output $\hat{x}$, and let $g_i = \partial \ell_i / \partial \hat{x}$. Let $\langle \lVert g_i \rVert_2 \rangle_\beta$ be the exponential moving average of the gradient norm over the last training batches, and let $R$ be a reference norm. The balancer backpropagates $\sum_i \tilde{g}_i$ instead of $\sum_i \lambda_i g_i$ (Section 3.4, Equation 5), with

$$
\tilde{g}_i = R \, \frac{\lambda_i}{\sum_j \lambda_j} \cdot \frac{g_i}{\langle \lVert g_i \rVert_2 \rangle_\beta}.
$$

If $\sum_i \lambda_i = 1$, each weight is the fraction of the gradient that comes from its loss. The paper uses $R = 1$ and $\beta = 0.999$. The paper's formula writes the average over $g_i$ and says the exponential moving average of $g_i$. The norm is what the formula needs, so this page reads it as the average of $\lVert g_i \rVert_2$. The gradient is with respect to the model output, so the commitment loss, which isn't defined on $\hat{x}$, stays outside the balancer. The paper then trains with $\lambda_t = 0.1$, $\lambda_f = 1$, $\lambda_g = 3$, and $\lambda_{\mathrm{feat}} = 3$ at 24 kHz, and $\lambda_g = \lambda_{\mathrm{feat}} = 4$ at 48 kHz (Section 4.4). The paper doesn't give a value for $\lambda_w$. The total objective is (Section 3.4, Equation 4)

$$
L_G = \lambda_t \ell_t(x, \hat{x}) + \lambda_f \ell_f(x, \hat{x}) + \lambda_g \ell_g(\hat{x}) + \lambda_{\mathrm{feat}} \ell_{\mathrm{feat}}(x, \hat{x}) + \lambda_w \ell_w(w).
$$

**Tiny example, computed by `scripts/balancer_example.py`.** The 24 kHz weights sum to $0.1 + 1 + 3 + 3 = 7.1$, so the four losses get 1.4, 14.1, 42.3, and 42.3 percent of the gradient. Take two gradients with norms 5 (a reconstruction loss) and 0.02 (an adversarial loss) and equal weights. The plain sum has norm 5.016, and 99.6 percent of it comes from the first loss. The balanced pieces are $[0.3, 0.4]$ and $[0, 0.5]$, each with norm 0.5. Multiplying the first loss by 1000 leaves the balanced gradient at $[0.3, 0.9]$. That is why the choice of $\lambda$ no longer depends on the scale of each loss, and why this page could drop the mean over bins above. The moving average is slow. After the norm jumps from 1 to 10, the average is 1.86 after 100 steps and 6.69 after 1000 steps, so for a while a sudden large gradient is only partly normalized. Exercise 2 asks you to write the balancer.

The paper's Table A.4 reports the effect of the balancer on several weight settings. It says only that the balancer "significantly stabilizes the training process" in the main text (Section 4.5.1). This page doesn't reproduce that appendix.

### Entropy coding with a small language model

**Intuition.** Fixed 10-bit codes are wasteful when some tokens are much more likely than others. A model that predicts the next token's distribution lets an arithmetic coder spend fewer bits on the likely ones. The tokens don't change, only the way they are written to the bitstream.

**Math.** The language model is a small Transformer with 5 layers, 8 heads, 200 channels, and a feed-forward width of 800, with a causal receptive field of 3.5 s, trained on 5 s sequences (Section 3.3). At step $t$ its input is the sum of learned embeddings, one table per codebook, of the tokens at step $t - 1$. It has $N_q$ linear output heads, each with 1024 logits. The model therefore treats the codebooks at one step as independent given the past and neglects their mutual information, which makes inference faster. The bitstream is written with a range-based arithmetic coder. The ideal cost of coding with a model whose predicted distribution at step $t$ for codebook $q$ is $p_{q,t}$ is the cross-entropy. When the predicted distribution equals the true one, that cost is the sum of the entropies:

$$
\text{bits per step} = \sum_{q=1}^{N_q} H(p_q), \qquad H(p) = -\sum_k p_k \log_2 p_k.
$$

This formula is this page's, not an equation of the paper. It is a lower bound for a fixed distribution, and the paper measures the real result.

Entropy coding needs the encoder and decoder to compute identical probabilities, and floating-point results differ across hardware and between batch and streaming evaluation by more than $10^{-8}$. The paper rounds the probabilities to a precision of $10^{-6}$, uses a total range width of $2^{24}$, and gives each symbol a minimum range of 2 (Section 3.3). It says that more evaluation contexts would be needed for practical deployment.

**Tiny example, computed by `scripts/tokens_example.py`.** A four-symbol distribution $(0.5, 0.25, 0.125, 0.125)$ has entropy 1.75 bits against a fixed length of 2. For one 1024-entry codebook with half of its probability mass on just 16 entries, the entropy is 7.99 bits of 10, which is 599 bps at 75 steps per second instead of 750. The paper reports the real savings in Table 1: 1.5 kbps becomes 0.9 (40.0 percent saved), 3 becomes 1.9 (36.7), 6 becomes 4.1 (31.7), and 12 becomes 8.9 (25.8). The saving shrinks at higher bandwidth. The paper's explanation is that the small Transformer can't model all the codebooks together (Section 4.5). At 48 kHz stereo, 24 kbps becomes 19.4 kbps, a saving of 19.2 percent (Table 4). The saving costs speed and latency. The real-time factor of the 24 kHz encoder drops from 9.8 to 1.6 with entropy coding, and the decoder waits for the next frame, which adds about 13 ms (Section 4.6). Exercise 4 asks you to compute the entropy bitrate from codebook distributions.

<!-- exercises -->

## Pitfalls

- **Treating the bandwidth as a free parameter.** The model supports a fixed set of bandwidths, 1.5, 3, 6, 12, and 24 kbps at 24 kHz. The number of codebooks is the dial, and it changes the bitrate in steps of 750 bps at 24 kHz and 1500 bps at 48 kHz.
- **Mixing up tokens and bits.** A token is an integer in $[0, 1024)$, and it costs 10 bits only without entropy coding. A language model over tokens, such as VALL-E, sees $75 N_q$ tokens per second at 24 kHz, and the first codebook carries the most information, because later ones refine its residual.
- **Reading the "multiple of 4" sentence literally.** The bandwidth list implies $N_q \in \{2, 4, 8, 16, 32\}$, as the page notes above.
- **Quantizing the latent at every stage.** As in SoundStream, each stage must quantize the residual left by the earlier ones. A stage that quantizes $z$ again sums unrelated guesses.
- **Forgetting the straight-through estimator and the commitment loss.** Quantization has no gradient. Without the straight-through estimator, the encoder gets none from the decoder. Without the commitment loss, the encoder output can drift away from its codes.
- **Squaring the L2 term in the spectral loss.** Equation 1 uses the norm. A squared norm changes the shape of the loss and breaks the scaling property that makes $\ell_f$ behave like a distance.
- **Letting the balancer see the commitment loss.** The balancer takes gradients with respect to $\hat{x}$, and $\ell_w$ has none, so it stays outside (Section 3.4).
- **Reading the discriminator weights as plain multipliers.** With the balancer, $\lambda_g = 3$ doesn't make the adversarial gradient three times the reconstruction gradient. It makes it $3 / 7.1$ of the total, a share, whatever the loss scale.
- **Training the discriminator on every step.** The paper updates it with probability 2/3 (24 kHz) or 1/2 (48 kHz) because it overpowers the decoder.
- **Comparing MUSHRA and SI-SNR as if they agreed.** Table 2 shows they rank discriminator setups differently, and the paper makes its claims on the listening test.
- **Expecting entropy-coded bitrates from the plain model.** The 0.9, 1.9, 4.1, and 8.9 kbps numbers need the language model and the arithmetic coder. The tokens alone cost the full 10 bits each.

## Further reading

- The EnCodec paper on arXiv: [2210.13438](https://arxiv.org/abs/2210.13438). The paper says its code and models are at github.com/facebookresearch/encodec.
- Course lesson, [Neural audio codecs](https://speech.ifkash.dev/lessons/060-neural-audio-codecs), for how EnCodec fits among the neural codecs.
- Course lesson, [Residual and scalar quantization](https://speech.ifkash.dev/lessons/059-residual-and-scalar-quantization), for the quantizer in more depth.
- The [SoundStream](/soundstream/) page on this site, for the design EnCodec starts from.
- The VALL-E page, [/valle/](/valle/), for how the tokens are used downstream.
