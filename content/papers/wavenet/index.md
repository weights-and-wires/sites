---
id: "wavenet"
title: "WaveNet: A Generative Model for Raw Audio"
authors: "van den Oord et al."
year: 2016
venue: "arXiv preprint"
arxiv: "1609.03499"
url: "https://arxiv.org/abs/1609.03499"
track: speech
tier: extended
est_hours: 4
status: full
summary: "WaveNet models raw audio one sample at a time, with each sample predicted from all earlier samples by a stack of causal convolutions whose dilations double from layer to layer, so the receptive field grows exponentially with depth. Each sample is companded with mu-law and quantized to 256 levels, so the model outputs a softmax over 256 classes and trains with cross-entropy. Gated activation units, residual connections, and skip connections shape each layer, and conditioning on speaker or text features lets one network speak in many voices."
related_course: "speech: audio generation"
prereqs: []
background: [dilated-causal-convolution, mu-law, autoregressive-models]
course_links:
  - title: "WaveNet"
    url: "https://speech.ifkash.dev/lessons/054-wavenet"
self_check:
  - q: 'A causal stack has filter length 2 and dilations $1, 2, 4, 8$. How many input samples does one output depend on, and why does the paper double the dilation at each layer?'
    a: 'The receptive field is $1 + (2-1)(1+2+4+8) = 16$. Doubling the dilation makes each layer reach twice as far back as the one before, so the receptive field grows exponentially with depth. With no dilation, it grows only linearly, as $\#\text{layers} + k - 1$.'
    see: dilated-causal-convolution
  - q: 'Why does WaveNet apply mu-law companding before quantizing to 256 levels, and what does it gain over 8-bit linear quantization?'
    a: 'A softmax over the 65,536 values of 16-bit audio is costly, and 256 evenly spaced levels leave quiet samples with a coarse grid. Mu-law spends more levels near zero, where speech spends most of its time. With $\mu = 255$, one bin is about 0.00017 wide near silence, against 0.00784 for linear 8-bit.'
    see: mu-law
  - q: 'Training predicts all $T$ samples in one forward pass, but generation takes $T$ passes. What makes both true?'
    a: 'Equation 1 factors the joint probability into conditionals $p(x_t \mid x_1, \dots, x_{t-1})$. In training, the true history is known at every position, and causal convolutions let the network score all positions at once. At generation time, each sample must be drawn and fed back before the next prediction, so 1 second of 16 kHz audio takes 16,000 sequential steps.'
    see: autoregressive-models
---

## The problem

Before WaveNet, text-to-speech systems did not model the waveform directly. Concatenative systems stitched together recorded snippets of speech. Parametric systems predicted acoustic features, and a vocoder turned the features into audio. Both sounded less natural than a human voice. Section 3.2 of the paper reports mean opinion scores (MOS) on a 5-point naturalness scale for North American English: 3.67 for an LSTM-RNN parametric system, 3.86 for an HMM-driven concatenative system, and 4.46 for natural speech stored as 8-bit mu-law.

The paper asks whether a neural network can model the waveform itself (Section 1). That is hard, because audio has very high temporal resolution. The paper says at least 16,000 samples per second, so one second of speech is a sequence of 16,000 values. A model must then capture structure at two scales:

- **Local structure.** The shape of a single pitch period spans tens of samples.
- **Long-range structure.** A phoneme spans thousands of samples, and the intonation of a sentence spans far more.

A plain convolution that sees 3 samples is far too short. To see 1,000 samples, an undilated stack needs about 1,000 layers or filters 1,000 samples long. An RNN covers any distance in principle, but it must run step by step during training. The paper's answer (Section 2) is an autoregressive model built from dilated causal convolutions. Training stays parallel, and the receptive field grows exponentially.

WaveNet reports that listeners rate it as more natural than the best parametric and concatenative systems in both English and Mandarin. Table 1 gives 4.21 for WaveNet conditioned on linguistic features and log fundamental frequency (L+F), against 3.86 for the best baseline in English. In Mandarin, WaveNet scores 4.08 against 3.79. The paper says the gap to natural speech fell from 0.69 to 0.34 points in US English, a 51% reduction.

## Key ideas

The sections that follow use the paper's notation: a waveform $\mathbf{x} = \{x_1, \dots, x_T\}$ and conditioning input $\mathbf{h}$. One notation difference to know about: the paper uses $k$ as the **layer index** in the gated unit (Equation 2, $W_{f,k}$). The receptive-field arithmetic below uses $k$ as the **filter length**, and $\ell$ for the layer index. Every number comes from a script in `scripts/`. Run any script from the paper folder to reproduce the output.

### The autoregressive factorization and the softmax output

**Intuition.** Predict each audio sample from all the samples before it. A model that does this well assigns high probability to realistic waveforms, and you can generate audio by sampling one value at a time and feeding it back in.

**Math.** The joint probability of a waveform is a product of conditionals (Section 2, Equation 1):

$$
p(\mathbf{x}) = \prod_{t=1}^{T} p(x_t \mid x_1, \dots, x_{t-1})
$$

With an input $\mathbf{h}$, such as a speaker identity or text features, the conditionals also see $\mathbf{h}$ (Section 2.5, Equation 3):

$$
p(\mathbf{x} \mid \mathbf{h}) = \prod_{t=1}^{T} p(x_t \mid x_1, \dots, x_{t-1}, \mathbf{h})
$$

The model outputs a categorical distribution over the next value with a softmax layer, and it maximizes the log-likelihood (Section 2). Section 2.2 argues that a softmax works better than a mixture density network even when the data is implicitly continuous, because a categorical distribution makes no assumption about its shape. The log-likelihood is tractable, so you can also compute validation loss and detect over-fitting.

At training time the network predicts every conditional in one pass, because the whole true waveform is known. At generation time, "the predictions are sequential: after each sample is predicted, it is fed back into the network" (Section 2.1).

**Tiny example, computed by `scripts/gated_example.py`.** Take three samples over three levels, $x = (2, 0, 1)$. Suppose the model assigns $p(x_1 = 2) = 0.2$, $p(x_2 = 0 \mid x_1 = 2) = 0.1$, and $p(x_3 = 1 \mid x_1 = 2, x_2 = 0) = 0.3$. Then $p(\mathbf{x}) = 0.2 \cdot 0.1 \cdot 0.3 = 0.006$ and $\log p(\mathbf{x}) = -5.116$. Generating 1 second of 16 kHz audio takes 16,000 sequential steps, which is why generation is slow even though training is fast.

### Dilated causal convolutions

**Intuition.** A causal convolution looks only at the present and past samples, so the output at time $t$ can't leak information from the future. A dilated convolution skips inputs with a fixed gap, so a small filter reaches far back. Doubling the gap at each layer doubles the reach.

**Math.** A causal convolution guarantees that the prediction $p(x_{t+1} \mid x_1, \dots, x_t)$ at time $t$ does not depend on $x_{t+1}, x_{t+2}, \dots, x_T$ (Section 2.1, Figure 2). For 1-D audio the paper implements this by "shifting the output of a normal convolution by a few timesteps." With a filter $w$ of length $k$ and dilation $d$, the output is

$$
y_t = \sum_{j=0}^{k-1} w_j \, x_{t-(k-1-j)d}
$$

with $x_i = 0$ for $i < 0$. The paper doesn't write this formula. It is a notation choice for this page, with $w_{k-1}$ on the current sample. Dilation $d = 1$ is the standard convolution (Section 2.1).

Each layer with dilation $d_\ell$ adds $(k-1)d_\ell$ samples of past context, so a stack of $L$ layers has receptive field

$$
R = 1 + (k-1)\sum_{\ell=1}^{L} d_\ell
$$

The paper doesn't write this formula either, but it gives the cases. Figure 2 has four undilated layers with $k = 2$ and a receptive field of 5 ($= \#\text{layers} + k - 1$). Figure 3 uses dilations $1, 2, 4, 8$. The paper doubles the dilation up to a limit and then repeats, for example $1, 2, 4, \dots, 512$, three times. It says each $1, 2, \dots, 512$ block has a receptive field of 1024 and works as a more efficient and discriminative counterpart of a $1 \times 1024$ convolution.

**Tiny example, computed by `scripts/receptive_field.py`.** The script computes $R$ by the formula and by pushing an impulse through an all-ones stack and finding the last output it reaches. Both agree:

| Stack ($k = 2$) | Layers | $R$ | Duration at 16 kHz |
|---|---|---|---|
| Undilated, Figure 2 | 4 | 5 | 0.3 ms |
| Dilations 1, 2, 4, 8, Figure 3 | 4 | 16 | 1.0 ms |
| One block, 1 to 512 | 10 | 1024 | 64.0 ms |
| Three blocks | 30 | 3070 | 191.9 ms |

Ten layers with $k = 2$ reach 1,024 samples. An undilated stack would need 1,023 layers for the same reach. The three-block stack has $R = 1 + 3 \cdot 1023 = 3070$, not $3 \cdot 1024$, because each block adds 1023 extra samples of context. With $k = 3$, one block gives $R = 2047$.

The paper reports receptive fields of 240 ms for text-to-speech and about 300 ms for the multi-speaker experiment (Section 3.1, Section 3.2). It doesn't list the dilation stacks, so this page doesn't try to reproduce those figures.

Exercise 1 asks you to write the causal dilated convolution. Exercise 2 asks you to compute the receptive field two ways.

### mu-law quantization to 256 classes

**Intuition.** Raw audio is stored as 16-bit integers, so a softmax over every possible value needs 65,536 outputs per timestep. Speech has most of its samples near zero, so spacing 256 levels evenly wastes them on loud values. Mu-law companding stretches the quiet range first, and then a uniform 256-level grid gives quiet sounds fine resolution.

**Math.** Section 2.2 applies a mu-law companding transformation (the paper cites ITU-T, 1988) and then quantizes to 256 values:

$$
f(x_t) = \operatorname{sign}(x_t)\,\frac{\ln(1 + \mu |x_t|)}{\ln(1 + \mu)}, \qquad -1 < x_t < 1, \quad \mu = 255
$$

The inverse is $x = \operatorname{sign}(f)\,\big((1+\mu)^{|f|} - 1\big)/\mu$. The paper doesn't give the inverse or the bin indexing. This page maps $f \in [-1, 1]$ to the integer $q = \lfloor (f+1)\mu/2 + 0.5 \rfloor \in \{0, \dots, 255\}$, which is the common convention in WaveNet code. Decoding maps $q$ to $f = 2q/\mu - 1$ and then applies the inverse.

**Tiny example, computed by `scripts/mu_law_example.py`.** Encode $x = 0.1$:

- $f = \ln(26.5)/\ln(256) = 0.5910$.
- $q = \lfloor 1.5910 \cdot 255 / 2 + 0.5 \rfloor = \lfloor 203.35 \rfloor = 203$.
- Decoding gives $f = 2 \cdot 203/255 - 1 = 0.5922$ and $x = 0.10067$, so the error is 0.00067.

| $x$ | $f(x)$ | $q$ | Decoded | Absolute error |
|---|---|---|---|---|
| 0.001 | 0.0410 | 133 | 0.00106 | 0.00006 |
| 0.01 | 0.2285 | 157 | 0.01023 | 0.00023 |
| 0.1 | 0.5910 | 203 | 0.10067 | 0.00067 |
| 0.5 | 0.8757 | 239 | 0.49668 | 0.00332 |
| 0.9 | 0.9811 | 253 | 0.91637 | 0.01637 |

The error grows with amplitude, which is the point. Linear 8-bit quantization over $[-1, 1]$ has bins 0.00784 wide everywhere and a worst-case error of 0.00392. For mu-law with $\mu = 255$, one bin is 0.00017 wide at $x = 0$ and 0.04366 wide at $x = 1$. For $|x| < 0.01$, the worst-case round-trip error is 0.00030, and over the full range it is 0.02159. The endpoints $-1, 0, 1$ encode to $0, 128, 255$.

Exercise 3 asks you to implement the encoder and decoder.

### Gated activation units

**Intuition.** Each layer computes two things from the same input: a candidate value (the filter) and a gate that decides how much of it passes. The product lets a layer switch features on and off by context, which a plain ReLU can't do.

**Math.** Section 2.3 uses the activation unit from gated PixelCNN (Equation 2):

$$
\mathbf{z} = \tanh(W_{f,k} * \mathbf{x}) \odot \sigma(W_{g,k} * \mathbf{x})
$$

where $*$ is a convolution (here a causal dilated one), $\odot$ is element-wise multiplication, $\sigma$ is the sigmoid, $k$ is the layer index (see the notation note above), and $f$ and $g$ label the filter and the gate. The two convolutions have separate weights. The paper reports that this non-linearity "worked significantly better" than ReLU for audio.

With global conditioning (Section 2.5), a learned projection of $\mathbf{h}$ is added inside each branch: $\tanh(W_{f,k} * \mathbf{x} + V_{f,k}^{\top}\mathbf{h}) \odot \sigma(W_{g,k} * \mathbf{x} + V_{g,k}^{\top}\mathbf{h})$. The vector $V^{\top}\mathbf{h}$ is broadcast over time. For local conditioning, the second series is first upsampled to the audio rate by a transposed convolution and enters through a $1 \times 1$ convolution.

Section 2.4 and Figure 4 wrap this unit in a residual block. A $1 \times 1$ convolution maps $\mathbf{z}$ back, a residual connection adds it to the block's input, and a second $1 \times 1$ convolution feeds a skip connection. The skip outputs from all blocks are summed and pass through ReLU, a $1 \times 1$ convolution, ReLU, another $1 \times 1$ convolution, and the softmax. The paper says both connection types speed up convergence and enable much deeper models.

**Tiny example, computed by `scripts/gated_example.py`.** Take $x = [1, 2, 3, 4]$ with one channel, filter weights $[1.0, 0.5]$ with 0.5 on the current sample, and gate weights $[0.5, -1.0]$ with $-1.0$ on the current sample. At $t = 2$ (the second sample):

- The filter convolution is $1.0 \cdot 1 + 0.5 \cdot 2 = 2.0$, and $\tanh(2.0) = 0.9640$.
- The gate convolution is $0.5 \cdot 1 - 1.0 \cdot 2 = -1.5$, and $\sigma(-1.5) = 0.1824$.
- The output is $0.9640 \cdot 0.1824 = 0.1759$.

| $t$ | Filter conv | Gate conv | $\tanh$ | $\sigma$ | $z$ |
|---|---|---|---|---|---|
| 1 | 0.5 | -1.0 | 0.4621 | 0.2689 | 0.1243 |
| 2 | 2.0 | -1.5 | 0.9640 | 0.1824 | 0.1759 |
| 3 | 3.5 | -2.0 | 0.9982 | 0.1192 | 0.1190 |
| 4 | 5.0 | -2.5 | 0.9999 | 0.0759 | 0.0759 |

The filter grows with $t$, but the gate closes and the output falls. A ReLU on the same filter would return $0.5, 2.0, 3.5, 5.0$, with no way to close.

Exercise 4 asks you to implement the unit.

<!-- exercises -->

## Pitfalls

- **Padding on the wrong side.** A causal convolution pads on the left only. Padding on both sides, or using a "same" convolution, lets output $t$ read $x_{t+1}$. The training loss then falls to near zero, and generation fails because the future is not available.
- **Reversing the filter taps.** Frameworks differ on whether a convolution flips its filter. When you port weights, check which tap lands on the current sample.
- **Counting nonzero outputs as the receptive field.** With dilation, an impulse reaches a sparse set of outputs. The receptive field is the distance to the farthest one, plus 1. Exercise 2 tests this.
- **Adding block receptive fields.** Stacking three $1, 2, \dots, 512$ blocks gives 3070 samples, not 3072. Each layer adds $(k-1)d$, and the initial $+1$ counts once.
- **Reading $k$ in Equation 2 as the filter length.** In the paper's gated unit, $k$ is the layer index.
- **Rounding mu-law by truncation.** `floor` without the $+0.5$ biases every decoded value toward $-1$. Indexing 256 levels with $\mu + 1$ in place of $\mu$ shifts every code.
- **Applying mu-law to audio outside $[-1, 1]$.** The formula holds for $-1 < x < 1$. Normalize 16-bit integers by 32768 first, and clip.
- **Using the same weights for filter and gate.** Equation 2 has two weight tensors, $W_f$ and $W_g$. Sharing one makes the unit a function of a single projection and removes the point of the gate.
- **Expecting fast sampling.** Section 2.1 says generation is sequential. A direct implementation makes 16,000 network calls per second of audio, and each call recomputes overlapping history unless you cache layer activations. The paper doesn't cover that optimization.
- **Expecting long-range coherence.** Section 3.1 says the lack of long-range coherence is partly due to the receptive field of about 300 ms, which covers only the last 2 to 3 phonemes. Section 3.3 says music samples shift in genre, instrument, and volume from second to second even with a receptive field of several seconds.

## Further reading

- [WaveNet: A generative model for raw audio](https://deepmind.google/discover/blog/wavenet-a-generative-model-for-raw-audio/) is the DeepMind blog post that accompanies the paper, with audio samples.
- [Conditional Image Generation with PixelCNN Decoders](https://arxiv.org/abs/1606.05328), by van den Oord et al., introduces the gated activation unit that WaveNet reuses.
- [Pixel Recurrent Neural Networks](https://arxiv.org/abs/1601.06759), by van den Oord et al., introduces the masked convolution and the autoregressive framing that the paper builds on.
- [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122), by Yu and Koltun, is the dilated-convolution paper that Section 2.1 cites.
- [Fast WaveNet Generation Algorithm](https://arxiv.org/abs/1611.09482), by Paine et al., caches activations to remove the repeated computation in generation.
