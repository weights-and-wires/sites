---
id: "f5tts"
title: "F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching"
authors: "Chen et al."
year: 2024
venue: "arXiv preprint"
arxiv: "2410.06885"
url: "https://arxiv.org/abs/2410.06885"
track: speech
tier: extended
est_hours: 5
status: full
summary: "F5-TTS generates a mel spectrogram with a flow-matching Diffusion Transformer and needs no duration model, no phoneme alignment, and no codec tokens. It pads the character sequence with filler tokens to the length of the mel spectrogram, masks part of a real recording during training, and learns to fill in the masked frames from the text and the unmasked audio. At inference it uses Sway Sampling, a fixed warp of the flow steps that spends more of the step budget early, so 16 steps already give a real-time factor of 0.15."
related_course: "speech: flow-matching TTS"
prereqs:
  - id: flow-matching
    reason: "F5-TTS trains with the optimal-transport conditional flow-matching loss and samples by solving the learned ODE, so you need that regression target and the noise-to-data time convention."
  - id: dit
    reason: "The F5-TTS backbone is a DiT with adaLN-zero blocks that takes mel frames instead of latent image patches, with the flow step entering through adaLN."
background: [infilling, mel-spectrograms]
course_links:
  - title: "Flow matching for speech"
    url: "https://speech.ifkash.dev/lessons/068-flow-matching-for-speech"
  - title: "Flow matching"
    url: "https://speech.ifkash.dev/lessons/066-flow-matching"
  - title: "Straighter paths and better solvers"
    url: "https://speech.ifkash.dev/lessons/067-straighter-paths-and-better-solvers"
  - title: "Classifier-free guidance"
    url: "https://speech.ifkash.dev/lessons/064-classifier-free-guidance"
self_check:
  - q: 'F5-TTS uses the optimal-transport path with $\sigma_1 = 0$. A frame has $x_0 = -1$ (noise) and $x_1 = 3$ (data). What are $x_t$ at $t = 0.5$ and the regression target?'
    a: 'The point is $x_t = (1 - 0.5)(-1) + 0.5 \cdot 3 = 1$, and the target is $x_1 - x_0 = 4$ at every $t$ (Section 2.1, Equation 3). The velocity is constant along a straight line from noise to data.'
    see: flow-matching
  - q: 'A random 80% of the frames of a 500-frame mel spectrogram is masked for the infilling task. How many frames does the model see as audio context, and which frames does the training signal come from?'
    a: 'The model sees $500 - 400 = 100$ unmasked frames as audio context, together with the full text. The model reconstructs the 400 masked frames $m \odot x_1$ (Section 3.1), and the loss on this page counts only those frames.'
    see: infilling
  - q: 'The audio is 24 kHz and the mel hop length is 256. How many frames is a 4-second clip, and how many filler tokens pad a 50-character transcript to that length?'
    a: 'The frame rate is $24000 / 256 = 93.75$ frames per second, so 4 seconds are 375 frames. The padded text has $375 - 50 = 325$ filler tokens $\langle F \rangle$ (Section 3.1, Equation 5; Section 4).'
    see: mel-spectrograms
---

## The problem

A text-to-speech model has to decide how long each part of the text lasts. Many non-autoregressive systems answer with a separate duration predictor or a forced alignment between phonemes and audio. That machinery has to be built, trained, and kept in sync with the acoustic model. E2 TTS (Eskimez et al., 2024) removes it: the model pads the characters with filler tokens to the length of the mel spectrogram and learns the alignment on its own (Section 3). The price is robustness.

Here is a failing example. The paper reproduces E2 TTS and trains small models on 945 hours of Mandarin (Section 5.1). The E2 TTS model fails on about 7% of the test samples with a word error rate (WER) above 50%, and re-ranking can't fix those failures. Its WER on Seed-TTS test-zh is 9.63, against 4.17 for the F5-TTS design at the same size (Section 5.1). Dropping the audio prompt removes the failures, so the paper reads this as evidence that E2 TTS entangles text and audio features: the text is a short signal padded with fillers and concatenated to a much richer speech signal, and the model has to learn from that mismatch alone (Section 3.2). E2 TTS also converges slowly.

F5-TTS keeps the filler padding and the infilling objective. It changes the backbone to a DiT with a ConvNeXt V2 stack that refines the text before it joins the audio, and it changes inference with Sway Sampling (Section 3.2). With 16 flow steps the paper reports a real-time factor (RTF) of 0.15 and a WER of 2.53 on LibriSpeech-PC test-clean (Table 1).

## Key ideas

The paper's notation is the page's notation. A mel spectrogram is $x_1 \in \mathbb{R}^{F \times N}$ with $F = 100$ mel channels and $N$ frames. Noise is $x_0 \sim \mathcal{N}(0, I)$, and the flow step $t \in [0, 1]$ runs from noise at $t = 0$ to data at $t = 1$, the same convention as the [flow matching page](/flow-matching/). The binary temporal mask is $m \in \{0, 1\}^{F \times N}$, and the padded text is $z$. The exercises call the mask `mask` and give it one entry per frame. Every number in a worked example comes from a script in `scripts/`; run it from the paper folder to reproduce the output.

### Infilling: learn to fill in masked mel frames

**Intuition.** Take a real utterance and hide some of its frames. Ask the model to reconstruct the hidden frames from the visible ones and from the full transcript. At inference the visible part is the reference recording and the hidden part is the speech you want, so zero-shot voice cloning is the same task the model trained on. This is the text-guided speech infilling task (Section 3).

**Math.** The model gets three inputs (Section 3.1). The first is the noisy speech $(1 - t)\,x_0 + t\,x_1$. The second is the masked speech $(1 - m) \odot x_1$, which keeps the context frames and zeroes the rest. The third is the text $z$ of the next idea. The paper writes the target as the learned distribution $P(m \odot x_1 \mid (1 - m) \odot x_1, z)$. The loss is the OT conditional flow-matching loss (Section 2.1, Equation 3):

$$
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t,\, q(x_1),\, p(x_0)} \big\| v_t\big((1 - t)\,x_0 + t\,x_1\big) - (x_1 - x_0) \big\|^2
$$

The paper's boundary conditions are $\sigma_0 = 1$ and $\sigma_1 = 0$ (Section 2.1), so there is no $\sigma_{\min}$ term and the target is exactly $x_1 - x_0$. That is the case $\sigma_{\min} = 0$ of the [flow matching page](/flow-matching/), where the target is $x_1 - (1 - \sigma_{\min})\,x_0$. The paper writes Equation 3 over the whole $x_1$ and says the model learns to reconstruct $m \odot x_1$ (Section 3.1). It does not write a masked loss. This page and Exercise 1 count the squared error on the masked frames only, which is what "reconstruct $m \odot x_1$" asks for. Counting the context frames as well would train the model to copy what it already sees.

For training, a random 70% to 100% of the mel frames is masked (Section 4). The paper doesn't say whether the masked frames form one span. The model is a function of $x_t$, $t$, the masked speech, and $z$.

**Tiny example, computed by `scripts/masked_target.py`.** Use six frames with one mel channel, $x_1 = (1, 2, 3, 4, 5, 6)$, noise $x_0 = (0.5, -1, 0, 1, -0.5, 2)$, $t = 0.5$, and a mask that hides frames 3 to 5 (indices 2, 3, 4):

| frame | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| mask $m$ | 0 | 0 | 1 | 1 | 1 | 0 |
| noisy $x_t$ | 0.75 | 0.50 | 1.50 | 2.50 | 2.25 | 4.00 |
| masked speech | 1 | 2 | 0 | 0 | 0 | 6 |
| target $x_1 - x_0$ | 0.5 | 3 | 3 | 3 | 5.5 | 4 |

Check frame 2 by hand: $x_t = 0.5 \cdot 0 + 0.5 \cdot 3 = 1.5$ and the target is $3 - 0 = 3$. A model that outputs zero everywhere has a masked loss of $(3^2 + 3^2 + 5.5^2)/3 = 16.0833$. A model that is exact on the masked frames and wrong by 3 on every context frame has a masked loss of 0 and a full-sequence loss of 4.5. Exercise 1 asks you to build these inputs and this loss.

### Text padded with filler tokens, and the duration estimate

**Intuition.** The model can only align text and audio if both sequences have the same length, so the text gets padded. The padding has no meaning. The model has to decide for itself which frames each character covers. This is what removes the duration model and the explicit alignment: the paper notes that, unlike Voicebox's phoneme-level forced alignment, no rigid text boundary is introduced (Section 3.2).

**Math.** The raw text is split into characters (alphabets and symbols for English, full pinyin for Chinese) and padded with a filler token $\langle F \rangle$ to the mel length (Section 3.1, Equation 5):

$$
z = (c_1, c_2, \ldots, c_M,\ \underbrace{\langle F \rangle, \ldots, \langle F \rangle}_{(N - M)\text{ times}})
$$

At inference you must pick $N$. One option is a separate duration predictor. The paper instead estimates the duration from the ratio of the number of characters in the generated text $y_{gen}$ and the reference transcript $y_{ref}$ (Section 3.1). The model then sees the concatenated reference and generated characters, padded to the total length, together with the reference mel $x_{ref}$ (Equation 6). After sampling, the reference part of the output is discarded and a vocoder (Vocos) turns the rest into audio (Sections 3.1 and 4). The paper doesn't give a rounding rule. This page floors the generated length, and Exercise 2 does the same. It assumes the characters always fit in the mel length.

The audio is 24 kHz with a hop length of 256 and 100 log mel channels (Section 4), which gives $24000 / 256 = 93.75$ frames per second.

**Tiny example, computed by `scripts/filler_and_duration.py`.** Padding "hello" to 12 frames gives 5 characters and 7 fillers. A 4-second clip is 375 frames, so a 50-character transcript has 325 fillers. For the duration, take a reference of 320 frames (3.41 s) with 40 characters, and generate speech for 60 characters. The reference takes $320 / 40 = 8$ frames per character, so the new text takes $8 \cdot 60 = 480$ frames, and $N = 320 + 480 = 800$ frames, which is 8.53 s. With the script's longer sentences, a 50-character reference and a 46-character new text, the generated length is $\lfloor 320 \cdot 46 / 50 \rfloor = 294$ frames and the total is 614. Exercise 2 asks you to write both functions.

The estimate treats every character as equally long. A sentence with a long pause, or one with many short words, breaks this assumption, See the pitfalls.

### The F5-TTS backbone: DiT with a text refinement stack

**Intuition.** E2 TTS runs a flat U-Net-style Transformer with long skip connections on the concatenation of padded text and audio. The paper finds that a plain DiT can't learn the alignment from padded characters alone, so it adds a small ConvNeXt V2 stack that processes the text *before* it meets the audio. The text gets "individual modeling space" to prepare itself, and the in-context learning that follows is easier (Section 3.2).

**Math.** The paper builds on the [DiT](/dit/) with adaLN-zero blocks (Section 3.2). The character sequence $z$ gets an absolute sinusoidal position embedding, then goes through ConvNeXt V2 blocks. The result is concatenated along the feature dimension with the noisy speech and the masked speech. The flow step $t$ is embedded sinusoidally and enters through adaLN-zero, not as an extra token as in Voicebox. The model uses RoPE for self-attention and a convolutional position embedding as in Voicebox. It drops the U-Net skip connections of E2 TTS and Voicebox.

The base model has 22 DiT layers, 16 heads, and 1024/2048 embedding and feed-forward sizes, plus 4 ConvNeXt V2 layers with 512/1024 sizes. That adds up to 335.8M parameters (Section 4). The reproduced E2 TTS has 333.2M parameters, 24 layers, and 1024/4096 sizes. The training data is Emilia, which leaves about 95K hours of English and Chinese after filtering. Training runs 1.2M updates with a batch of 307,200 audio frames (0.91 hours), for over a week on 8 A100 80G GPUs, with AdamW at a peak learning rate of 7.5e-5 (Section 4).

This idea has no script-backed number of its own. The sizes above are the paper's.

The paper's ablations (Section 5.1, Figure 2) say what each piece buys. A pure adaLN DiT without the text stack fails to learn the alignment from padded characters. An MMDiT variant learns fast and then collapses into repeated utterances. Adding the same ConvNeXt branch for the audio costs 1.61 WER and gains 0.01 in similarity.

### Classifier-free guidance and Euler sampling

**Intuition.** Inference solves the ODE $\mathrm{d}x/\mathrm{d}t = v_t$ from noise at $t = 0$ to a mel spectrogram at $t = 1$. Classifier-free guidance (CFG) runs the model twice per step, with and without the conditions, and extrapolates from the unconditional velocity to the conditional one. The extrapolation sharpens the output toward the condition.

**Math.** Training drops the conditions at random (Section 4). First the masked speech is dropped with probability 0.3. Then the masked speech and the text together are dropped with probability 0.2. The paper suggests this two-stage scheme helps the model learn the text alignment. Inference uses (Section 2.2, Equation 4):

$$
v_{t,\text{CFG}} = v_t(\psi_t(x_0), c) + \alpha\,\big(v_t(\psi_t(x_0), c) - v_t(\psi_t(x_0))\big)
$$

The paper writes $\alpha$ for the strength, and this page and Exercise 4 write $w$ so that it doesn't clash with other symbols. The default is 2 (Section 5). The paper notes that CFG doubles the inference time. It uses the Euler solver for F5-TTS and the midpoint solver for E2 TTS, with EMA weights (Section 4). The RTF is measured on 10 seconds of speech on an NVIDIA RTX 3090.

**Tiny example, computed by `scripts/toy_cfg.py`.** No trained network is needed to see the effect, because a two-cluster toy has an exact flow. The data given the condition $c$ is $\mathcal{N}(\mu_c, 0.5^2 I)$ in two dimensions, with $\mu_0 = (-2, 0)$ and $\mu_1 = (+2, 0)$. For this data the OT path gives the closed-form field

$$
u_c(x, t) = \mu_c + \frac{t S^2 - (1 - t)}{(1 - t)^2 + t^2 S^2}\,(x - t\mu_c), \qquad S = 0.5,
$$

and the unconditional field is the mixture of the two, weighted by the posterior of each component at $(x, t)$. At $t = 0$ and $x = (0.5, 0)$ the conditional fields are $(1.5, 0)$ for $c = 1$ and $(-2.5, 0)$ for $c = 0$. The two components are equally likely at $t = 0$, so the unconditional field is their average, $(-0.5, 0)$. With $w = 2$ the guided velocity is $1.5 + 2 \cdot (1.5 + 0.5) = 5.5$. Sampling 4000 noise points with 16 Euler steps and condition $c = 1$ gives:

| $w$ | mean of $x$ | std of $x$ | samples on the wrong side |
|---|---|---|---|
| 0 | 1.999 | 0.460 | 0 |
| 1 | 2.341 | 0.324 | 0 |
| 2 | 2.556 | 0.285 | 0 |
| 4 | 2.873 | 0.252 | 0 |

The true data mean is 2 and its std is 0.5. Guidance moves the cluster away from the other class and shrinks its spread: fidelity to the condition goes up and diversity goes down. The Euler error against a 2000-step reference is $0.2409$, $0.1279$, $0.0682$, $0.0353$, and $0.0179$ at 2, 4, 8, 16, and 32 steps, so the error halves when the steps double. Exercise 4 asks you to write the sampler.

### Sway Sampling

**Intuition.** Training draws $t$ uniformly, but inference doesn't have to space its steps uniformly. The paper's view is that the early steps (small $t$) sketch the contours of the speech from noise, and the later steps add fine detail, so the alignment between text and speech is fixed by the first few results (Section 3.2). Spend more of the step budget there. This changes the step schedule only. It needs no retraining and applies to other CFM models (Section 3.2).

**Math.** Sample $u \sim \mathcal{U}[0, 1]$, then warp it (Section 3.2, Equation 7):

$$
f_{\text{sway}}(u; s) = u + s\,\Big(\cos\big(\tfrac{\pi}{2}\,u\big) - 1 + u\Big)
$$

The function is monotonic for $s \in [-1, \tfrac{2}{\pi - 2}]$, and $s = 0$ gives uniform steps. The paper's default is $s = -1$ (Section 5). The condition for monotonicity comes from the derivative $1 + s\,(1 - \tfrac{\pi}{2}\sin(\tfrac{\pi}{2}u))$, which is $1 + s$ at $u = 0$ and $1 + s\,(1 - \tfrac{\pi}{2})$ at $u = 1$. For $s = -1$ the function is $1 - \cos(\tfrac{\pi}{2}u)$. At inference you use the warped values as the step boundaries, so the Euler update of the previous idea takes the interval widths from $t_{i+1} - t_i$. A negative $s$ makes the steps small near $t = 0$ and large near $t = 1$ ("sway to the left").

**Tiny example, computed by `scripts/sway_sampling.py`.** For eight steps, $u = 0, 0.125, \ldots, 1$ maps with $s = -1$ to

| $u$ | 0 | 0.125 | 0.25 | 0.375 | 0.5 | 0.625 | 0.75 | 0.875 | 1 |
|---|---|---|---|---|---|---|---|---|---|
| $t$ | 0 | 0.0192 | 0.0761 | 0.1685 | 0.2929 | 0.4444 | 0.6173 | 0.8049 | 1 |

The midpoint is $1 - \cos(\pi/4) = 0.2929$. The step sizes grow from $0.0192$ to $0.1951$. With 16 steps, $s = -1$ puts 11 of the 16 steps below $t = 0.5$, against 8 for uniform steps and 7 for $s = +0.5$. The first step is $0.0048$ wide and the last is $0.0980$. The script also checks the range: the smallest slope is $s + 1 = 0$ at $u = 0$ for $s = -1$, and it is about zero at $u = 1$ for $s = 2/(\pi - 2) = 1.7519$. Outside the range ($s = -1.2$ or $2$) the function is no longer monotonic.

The paper supports the choice with a "leak and override" experiment (Section 5.2). It starts sampling from a noise input that leaks the audio prompt and asks the model for different text. With Sway Sampling the model overrides the leaked content, and with uniform steps it speaks the leaked prompt text. That shows the early steps decide the content. Exercise 3 asks you to implement Equation 7.

The toy of the previous idea does not reproduce the benefit. With $w = 0$ and 4 steps the mean absolute error against the reference is $0.1279$ for uniform steps and $0.1804$ for $s = -1$, and with 8 steps it is $0.0682$ against $0.0962$. A Gaussian cluster has no alignment to settle early, so its error is smallest with even steps. The warp pays off only where the early steps carry the structure, as the paper argues for text-to-speech.

### Results

All numbers in this idea are the paper's. The test sets are LibriSpeech-PC test-clean (1127 samples) and Seed-TTS test-en and test-zh. WER comes from Whisper-large-v3 for English and Paraformer-zh for Chinese, and SIM-o is the speaker similarity measured with a WavLM-large model (Section 4).

| Model (steps) | LibriSpeech-PC WER | SIM-o | RTF |
|---|---|---|---|
| Ground truth | 2.23 | 0.69 | none |
| E2 TTS (32) | 2.95 | 0.69 | 0.68 |
| F5-TTS (16) | 2.53 | 0.66 | 0.15 |
| F5-TTS (32) | 2.42 | 0.66 | 0.31 |

F5-TTS (32) also reaches WER 1.83 on Seed-TTS test-en and 1.56 on test-zh, against 2.19 and 1.97 for E2 TTS (Table 2). E2 TTS keeps a higher speaker similarity on test-en (0.71 against 0.67). The paper notes that Seed-TTS DiT, trained on several million hours, is better still (Section 5). Treat the comparisons with care: the F5-TTS and E2 TTS rows come from models the authors trained on the same data.

<!-- exercises -->

## Pitfalls

- **Masking the loss in the wrong place.** The inputs include the unmasked frames as the condition. If the loss also counts them, the model learns to copy its own input. Exercise 1's test sets a large error on the context frames and expects a loss of 0.
- **Treating the paper's equations as a masked loss.** Equation 3 has no mask. The masked reconstruction comes from Section 3.1's description, and the paper doesn't say how the loss is normalized. This page divides by the number of masked entries. Dividing by all entries only rescales the loss by the masked fraction, which changes the effective weighting between utterances with different mask sizes.
- **Mixing the time directions.** F5-TTS has noise at $t = 0$ and data at $t = 1$, like the flow matching paper and unlike diffusion papers. Sway Sampling with $s < 0$ spends the extra steps near noise. Reverse $t$ and you spend them near the data.
- **Using the wrong range for $s$.** The paper's range is $[-1, 2/(\pi - 2)]$, which is about $[-1, 1.7519]$. The expression $2/\pi - 1$ is about $-0.36$, and a bound written that way rejects valid positive values such as $s = 0.5$. Outside the paper's range the warp stops being monotonic and the Euler step widths go negative.
- **Applying Sway Sampling to anything.** The toy shows the warp is not free: on a Gaussian it increased the integration error. The paper's evidence is for text-to-speech models with an alignment to settle, and for models trained with uniform $t$.
- **Trusting the duration estimate.** The estimate sets one speaking rate for the whole utterance from the reference, so a slow reference makes slow output. It also sets the final length, and the model can't change it. If the estimate is too short for the text, the speech runs fast or cuts off. The text still has to fit: $M \le N$.
- **Forgetting the doubled cost of CFG.** Each Euler step calls the network twice. The RTF of 0.15 at 16 steps includes that cost (Section 5, footnote to Equation 4), so a quoted number of function evaluations (NFE) is steps, not forward passes.
- **Comparing against the wrong E2 TTS.** The E2 TTS in the paper's tables is the authors' reproduction (333.2M parameters), trained on Emilia, not the original model (Section 4).
- **Comparing WER across test sets.** The paper builds and releases its own LibriSpeech-PC subset because earlier work used unreleased prompt lists (Section 4). A WER on another subset isn't comparable.

## Further reading

- [F5-TTS on arXiv](https://arxiv.org/abs/2410.06885) has the paper, with the appendices that cover ODE solvers, the vocoder, and the data-scale study.
- [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS) is the official code and the released checkpoints.
- [E2 TTS](https://arxiv.org/abs/2406.18009), by Eskimez et al., is the filler-token design that F5-TTS fixes.
- [Voicebox](https://arxiv.org/abs/2306.15687), by Le et al., introduced the text-guided speech infilling objective with flow matching.
- [Flow Matching for Generative Modeling](/flow-matching/) and [Scalable Diffusion Models with Transformers](/dit/) on this site cover the loss and the backbone.
- [VALL-E](/valle/) is the autoregressive codec-token counterpart that F5-TTS compares against.
