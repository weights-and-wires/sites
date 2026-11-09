---
id: "valle"
title: "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers"
authors: "Wang et al."
year: 2023
venue: "arXiv preprint"
arxiv: "2301.02111"
url: "https://arxiv.org/abs/2301.02111"
track: speech
tier: core
est_hours: 5
status: full
summary: "VALL-E treats text to speech as language modeling over the discrete codes of a neural audio codec, so a phoneme sequence and a 3-second recording of an unseen speaker go in as a prompt and codec tokens come out. An autoregressive decoder predicts the first codebook of the residual quantizer, which sets the length, the content, and most of the voice, and a non-autoregressive decoder predicts the other seven codebooks in seven parallel passes. Training on 60,000 hours of audiobook speech makes the prompt work like in-context learning: the model copies the speaker's voice and recording conditions with no fine-tuning."
related_course: "speech: TTS"
prereqs:
  - id: encodec
    reason: "VALL-E uses EnCodec's 8-codebook residual quantizer at 75 frames per second as both its tokenizer and its waveform decoder, so its output is exactly that code grid."
  - id: audiolm
    reason: "AudioLM showed that a language model over codec tokens can continue speech in the prompt's voice, and VALL-E keeps that idea but replaces the semantic tokens with phonemes and the later stages with a non-autoregressive model."
background: [in-context-prompting, autoregressive-and-non-autoregressive-decoding]
course_links:
  - title: "Codec language models"
    url: "https://speech.ifkash.dev/lessons/051-codec-language-models"
self_check:
  - q: 'EnCodec at 6 kbps turns a 10-second, 24 kHz clip into a grid of codes. What are its dimensions, and why does VALL-E treat the first column differently from the other seven?'
    a: 'The grid is $750 \times 8$, because $24000 / 320 = 75$ frames per second and there are 8 codebooks. Residual quantization makes column 1 carry the coarse content, including most of the speaker identity, and each later column encodes the residual left by the earlier ones. VALL-E generates column 1 with an autoregressive model and columns 2 to 8 with a non-autoregressive model (Section 4.2).'
    see: encodec
  - q: 'AudioLM conditions on semantic tokens from w2v-BERT. What does VALL-E use in their place, and what does the paper say that buys?'
    a: 'VALL-E uses phonemes from a text transcription, aligned with a hybrid DNN-HMM ASR model for training (Section 5.1). The paper credits the better alignment with the input text for its lower word error rate than GSLM and AudioLM in Table 2 (Section 5.2). Phonemes also make the model a text-to-speech system, so you can synthesize text the speaker never said.'
    see: audiolm
  - q: 'Why does VALL-E use an autoregressive model for codebook 1 and a non-autoregressive model for codebooks 2 to 8?'
    a: 'The number of frames $T$ is not known in advance, and it depends on how fast the speaker talks. An autoregressive model finds $T$ by emitting `<EOS>`. Once $T$ is fixed, each later codebook is a column of $T$ slots that one parallel pass can fill, so seven passes replace $7T$ steps (Section 4.2).'
    see: autoregressive-and-non-autoregressive-decoding
---

## The problem

Before VALL-E, a typical text to speech system was a cascade. An acoustic model mapped phonemes to a mel spectrogram, and a vocoder mapped the spectrogram to a waveform (Section 1). These systems train on clean studio recordings, and Table 1 of the paper lists 600 hours or fewer as typical. The text adds that existing systems train on dozens of hours of single-speaker data or hundreds of hours of multi-speaker data (Section 1).

That design breaks on a new voice. Suppose a user gives the system a 3-second clip of a speaker it has never heard and asks it to read a new sentence. The model must reproduce the speaker's timbre, speaking rate, and even the room echo, but a few hundred training speakers can't cover that variety. The earlier fixes each cost something. Speaker adaptation needs fine-tuning on the new voice, and speaker encoders need hand-designed features. Both degrade on unseen speakers (Section 1).

The paper makes a different bet, borrowed from text language models: use a simple objective, and scale the data until the model generalizes. Large corpora of found speech are noisy, which suits a language model over discrete tokens better than regression onto a spectrogram. The paper trains on LibriLight, 60,000 hours from about 7,000 speakers, with transcriptions from an ASR model (Section 5.1). On LibriSpeech test-clean with a 3-second prompt, the word error rate is 5.9 against 7.7 for the YourTTS baseline, and the speaker similarity score is 0.580 against 0.337 (Section 5.2, Table 2).

## Key ideas

This page follows the paper's notation: a phoneme sequence $\mathbf{x}$, an acoustic code matrix $\mathbf{C} \in \mathbb{Z}^{T \times 8}$ with row $\mathbf{c}_{t,:}$ for frame $t$ and column $\mathbf{c}_{:,j}$ for codebook $j$, and an acoustic prompt matrix $\tilde{\mathbf{C}} \in \mathbb{Z}^{T' \times 8}$ from the enrolled recording. A tilde marks the prompt. Codebooks are numbered from 1, and the paper's frame index $t$ starts at 0. The paper writes both $x$ and $X$ for the phoneme sequence in Equation 3. This page writes $\mathbf{x}$ throughout.

### TTS as conditional codec language modeling

**Intuition.** A neural codec turns audio into a grid of integers, and a decoder turns the grid back into audio. After that step, "say this sentence in this voice" is a sequence problem: continue a token sequence whose beginning is a voice sample and whose condition is a text.

**Math.** The pre-trained codec encodes an utterance $\mathbf{y}$ as $\operatorname{Encodec}(\mathbf{y}) = \mathbf{C}^{T \times 8}$ and decodes with $\operatorname{Decodec}(\mathbf{C}) \approx \hat{\mathbf{y}}$ (Section 4.1). The model maximizes $p(\mathbf{C} \mid \mathbf{x}, \tilde{\mathbf{C}})$. The paper uses EnCodec at 24 kHz with a 320-fold downsampling, which gives 75 frames per second, and eight quantizers with 1,024 entries each (Section 3). That setting is 6 kbps.

**Tiny example, computed by `scripts/codec_tokens.py`.** A 10-second waveform becomes $750 \times 8 = 6{,}000$ codes, the same $750 \times 8$ grid that the paper gives for this clip (Section 3). Each code takes 10 bits, so the bitrate is $75 \cdot 8 \cdot 10 = 6{,}000$ bit/s. A 3-second prompt is 225 frames and 1,800 codes. At 16 quantizers, the paper's 12 kbps setting, the bitrate is 12,000 bit/s.

The model doesn't output a waveform, and that is the point. The codec decoder already exists, so VALL-E needs no vocoder training (Section 3).

### AR for the first codebook, NAR for the rest

**Intuition.** Residual quantization is hierarchical. The first codebook carries the coarse information, and each later codebook refines the residual left by the ones before it (Section 4.2). Because column 1 decides how long the utterance is, it needs a model that can say "stop". Once the length is known, the other seven columns are fixed-size slots that don't need left-to-right generation.

**Math.** The autoregressive (AR) model is a decoder-only transformer over the first codebook, conditioned on the phonemes and the prompt's first column (Section 4.2, Equation 1):

$$
p(\mathbf{c}_{:,1} \mid \mathbf{x}, \tilde{\mathbf{C}}_{:,1}; \theta_{AR}) = \prod_{t=0}^{T} p(c_{t,1} \mid \mathbf{c}_{<t,1}, \tilde{\mathbf{c}}_{:,1}, \mathbf{x}; \theta_{AR})
$$

The non-autoregressive (NAR) model predicts the columns $j = 2, \dots, 8$ one column at a time, given all earlier columns (Equation 2):

$$
p(\mathbf{C}_{:,2:8} \mid \mathbf{x}, \tilde{\mathbf{C}}; \theta_{NAR}) = \prod_{j=2}^{8} p(\mathbf{c}_{:,j} \mid \mathbf{C}_{:,<j}, \mathbf{x}, \tilde{\mathbf{C}}; \theta_{NAR})
$$

The two stages combine into one factorization (Equation 3):

$$
p(\mathbf{C} \mid \mathbf{x}, \tilde{\mathbf{C}}; \theta) = p(\mathbf{c}_{:,1} \mid \tilde{\mathbf{C}}_{:,1}, \mathbf{x}; \theta_{AR}) \prod_{j=2}^{8} p(\mathbf{c}_{:,j} \mid \mathbf{c}_{:,<j}, \mathbf{x}, \tilde{\mathbf{C}}; \theta_{NAR})
$$

The product in Equation 2 is over codebooks, not over frames. Inside one factor, all $T$ frames are predicted together. The paper says this reduces the time complexity from $O(T)$ to $O(1)$ for those stages (Section 4.2).

**Tiny example, computed by `scripts/codec_tokens.py`.** For the 10-second clip, $T = 750$. VALL-E needs 750 AR steps and 7 NAR passes, which is 757 forward passes. A model that generated all eight columns one token at a time would need $750 \cdot 8 = 6{,}000$ steps. The ratio is $7.93\times$.

Exercise 4 builds this two-stage loop around a stand-in model.

### The AR sequence layout and attention masks

**Intuition.** The AR model reads text first and then audio. Every audio token can look at the whole sentence it must pronounce, and at the audio that came before it, but not at the audio that comes after.

**Math.** The input is the concatenation of $\mathbf{x}$ and $\mathbf{c}_{:,1}$, with an `<EOS>` token appended to each (Section 4.2.1). The paper computes the sinusoidal position embeddings separately for the prompt part and the audio part. Each audio token $c_{t,1}$ attends to $(\mathbf{x}, \mathbf{c}_{\le t,1})$ (Section 4.2.1, Figure 3, left). The NAR model instead lets every token attend to all input tokens (Section 4.2.2, Figure 3, right).

There is no special prompt token in the AR model. The paper explains that the concatenation of $\tilde{\mathbf{c}}_{:,1}$ and $\mathbf{c}_{:,1}$ is a single sequence, and that training is plain causal language modeling, so any prefix $\mathbf{c}_{<t,1}$ acts as a prompt for the rest (Section 4.2.1). At inference, only the target is predicted, and the prompt prefix is supplied.

The paper's text doesn't say whether phoneme positions see later phonemes. Exercise 1 accepts either choice and fixes only the audio rows, which the paper specifies. The table below uses the prefix-language-model choice, where every phoneme sees every phoneme.

**Tiny example, computed by `scripts/mask_example.py`.** With three text positions (two phonemes and `<EOS>`) and three audio positions, the AR mask is a $6 \times 6$ matrix. A 1 means the query may attend to the key:

| Query | Keys it sees (text, then audio) |
|---|---|
| text 0 to 2 | `[1 1 1 0 0 0]` |
| audio 0 | `[1 1 1 1 0 0]` |
| audio 1 | `[1 1 1 1 1 0]` |
| audio 2 | `[1 1 1 1 1 1]` |

Audio position 0 sees 4 keys (3 text and itself), and audio position 2 sees all 6. The NAR mask for the same sequence has all 36 entries set to 1.

Exercise 1 asks you to build both masks.

### NAR inputs: summed codebook embeddings and the stage in AdaLN

**Intuition.** The NAR model predicts the codes of one codebook at a time, and it has to know which one. It receives the codes of all earlier codebooks of the target, summed into one vector per frame, together with the full acoustic prompt, so it knows whose voice to refine.

**Math.** Training samples a stage $i \in [2, 8]$ at random. The model embeds codebooks $1$ to $i-1$ of the target with a separate table $W_a^j$ for each codebook and sums them (Section 4.2.2, Equations 4 and 5):

$$
\mathbf{e}_{c_{t,j}} = W_a^j \odot c_{t,j}, \qquad \mathbf{e}_{c_t} = \sum_{j=1}^{i-1} \mathbf{e}_{c_{t,j}}
$$

Here $\odot$ is index selection, a table lookup. The acoustic prompt $\tilde{\mathbf{C}}$ is embedded the same way, except that it sums all eight codebooks: $\mathbf{e}_{\tilde{c}_t} = \sum_{j=1}^{8} \mathbf{e}_{\tilde{c}_{t,j}}$. The transformer input is the concatenation $(\mathbf{e}_{\mathbf{x}}, \mathbf{e}_{\tilde{\mathbf{c}}}, \mathbf{e}_{\mathbf{c}_{:,<i}})$.

The stage index reaches the network through adaptive layer normalization (Section 4.2.2). The paper doesn't number this formula:

$$
\operatorname{AdaLN}(\mathbf{h}, i) = a_i \operatorname{LayerNorm}(\mathbf{h}) + b_i
$$

Here $a_i$ and $b_i$ come from a linear projection of the stage embedding. The prediction layer for codebook $j$ shares its weights with the acoustic embedding of codebook $j+1$ (Section 4.2.2).

**Tiny example, computed by `scripts/nar_input_example.py`.** Use 3 codebooks instead of 8, and embeddings with $W^j[v] = [10j + v,\ v]$. Take one prompt frame $[1, 0, 2]$ and two target frames $[3, 1, 0]$ and $[2, 0, 1]$, and predict codebook $i = 3$.

- The prompt embedding sums all three codebooks: $[11, 1] + [20, 0] + [32, 2] = [63, 3]$.
- Target frame 0 sums codebooks 1 and 2: $[13, 3] + [21, 1] = [34, 4]$.
- Target frame 1 sums codebooks 1 and 2: $[12, 2] + [20, 0] = [32, 2]$. Its third code is never used.

At stage $i = 2$, the target embeddings are the codebook 1 embeddings alone: $[13, 3]$ and $[12, 2]$.

Exercise 2 asks you to build this input and the AdaLN operator.

### The acoustic prompt as in-context learning

**Intuition.** A text language model told "translate to French: cheese ->" infers the task from its context, with no gradient step. VALL-E's context holds a voice. Tokens from a 3-second clip sit in front of the continuation, and the model has seen enough speakers during training to infer the voice from them.

**Math.** At inference, the transcription of the enrolled clip is prepended to the target text as the phoneme prompt, and the first-codebook tokens $\tilde{\mathbf{c}}_{:,1}$ of the clip are the acoustic prefix (Section 4.3). The NAR model gets the whole matrix $\tilde{\mathbf{C}}$. The AR model is decoded by sampling, because the paper observes that beam search may trap the model in an infinite loop. The NAR model is decoded greedily (Section 4.3). A variant, VALL-E-continual, uses the first 3 seconds of the true utterance and its full transcription as the prompts, and asks the model to continue (Section 4.3).

The ablations show which prompt does what (Section 5.2):

| Ablation | WER | Speaker score |
|---|---|---|
| NAR with no prompt (Table 4) | 19.6 | 0.518 |
| NAR with a phoneme prompt (Table 4) | 3.0 | 0.541 |
| NAR with both prompts (Table 4) | 2.8 | 0.732 |
| AR with the acoustic prompt (Table 5) | 5.9 | 0.585 |
| AR without the acoustic prompt (Table 5) | 5.9 | 0.236 |

The phoneme prompt mainly fixes the content, and the acoustic prompt fixes the speaker. Both stages benefit from the acoustic prompt.

**Tiny example, computed by `scripts/icl_example.py`.** The paper's model learns this behavior implicitly. A toy that makes the mechanism visible is a mixture over $K = 3$ speakers, each a Markov chain over 6 tokens. Speaker $k$ steps from token $v$ to $(v + k + 1) \bmod 6$ with probability 0.7, and to each other token with probability 0.06. The prior is uniform, and the prompt comes from speaker 1: $0, 2, 4, 0, \dots$. This toy is not VALL-E's computation.

| Prompt length | Posterior over speakers |
|---|---|
| 1 | `[0.3333, 0.3333, 0.3333]` |
| 2 | `[0.0732, 0.8537, 0.0732]` |
| 3 | `[0.0072, 0.9855, 0.0072]` |
| 5 | `[0.0001, 0.9999, 0.0001]` |

Check the second row by hand. The single transition $0 \to 2$ has likelihood $0.06$, $0.7$, and $0.06$ under the three speakers. The sum is $0.82$, so speaker 1 gets $0.7 / 0.82 = 0.8537$. With the prompt $[0, 2]$, the next-token distribution is `[0.06, 0.06, 0.06, 0.1068, 0.6063, 0.1068]`, and the greedy continuation is $[0, 2, 4, 0, 2, 4, 0]$. With the one-token prompt $[0]$, the model has no evidence about the speaker, and the next-token mass is spread over the three speakers' favorites.

Exercise 3 asks you to build the posterior, the mixture, and the continuation.

### Scale

**Intuition.** The prompt only works because the model has seen thousands of voices. The paper's central claim is that data scale, not a new architecture, makes zero-shot synthesis possible (Section 1, contributions).

**Math.** Both the AR and NAR models are 12-layer transformers with 16 heads, embedding dimension 1,024, feed-forward dimension 4,096, and dropout 0.1. Training crops each utterance to a random 10 to 20 seconds, and the NAR acoustic prompt is a random 3-second segment of the same utterance. The models train on 16 V100 GPUs for 800,000 steps with AdamW, a peak learning rate of $5 \times 10^{-4}$ reached after 32,000 warm-up updates, and then a linear decay (Section 5.1).

**Tiny example, computed by `scripts/codec_tokens.py`.** At 75 frames per second, 60,000 hours is $60000 \cdot 3600 \cdot 75 = 1.62 \times 10^{10}$ frames, or $1.296 \times 10^{11}$ codes over 8 codebooks. Table 1 lists 600 hours or fewer for current systems, so the corpus is about 100 times larger than that figure.

<!-- exercises -->

## Pitfalls

- **Treating the AR and NAR models as one network.** They share an architecture, but they are two models with separate weights, and the NAR model has eight separate acoustic embedding tables (Section 4.2.2). The NAR model is called seven times per utterance, once for each of the codebooks 2 to 8.
- **Letting the NAR input see the stage it predicts.** The input holds codebooks $1$ to $i-1$ of the target. Summing codebooks $1$ to $i$ gives the model the answer during training. The prompt, in contrast, sums all eight codebooks.
- **Forgetting the text of the prompt.** The phoneme prompt is the transcription of the enrolled clip followed by the target text (Section 4.3). Passing the prompt's audio without its words leaves the model unable to line the prefix tokens up with a transcript.
- **Beam search for the AR model.** The paper says that beam search may lead the model into an infinite loop, and it decodes by sampling (Section 4.3). A run that never emits `<EOS>` needs a hard cap on the length.
- **Reading the first codebook as "just content".** Column 1 carries the coarse acoustic information, including much of the speaker's identity (Section 4.2). The paper's ablation drops the speaker score from 0.585 to 0.236 when the AR model loses its acoustic prompt (Table 5), even though the WER stays at 5.9.
- **Expecting the NAR model to fix timing.** The NAR model fills columns of a length that the AR model already chose. Missing, repeated, or garbled words come from the autoregressive alignment, and the paper lists that as a limitation (Section 6).
- **Expecting more than the codec can give.** The codec decoder produces the audio, so the output quality has a ceiling set by the 6 kbps token grid.
- **Comparing speaker scores across tables without care.** The paper reports 0.580 for VALL-E in Table 2 and 0.585 in Table 5. The text doesn't explain the difference.
- **Ignoring misuse.** The paper notes that a system that keeps speaker identity can be used for spoofing voice identification or impersonation, and it suggests a detection model (Section 6).

## Further reading

- [The paper's demo page](https://aka.ms/valle) has audio samples, including the emotion and acoustic-environment examples from Section 5.4.
- [EnCodec](https://github.com/facebookresearch/encodec) is the official code for the tokenizer and decoder that VALL-E uses.
- [AudioLM](/audiolm/) is the earlier codec language model, and [EnCodec](/encodec/) is the tokenizer. Read the two together with this page.
- [VALL-E 2](https://arxiv.org/abs/2406.05370), by Chen et al., is the follow-up that addresses the repetition and robustness limitation in Section 6 of this paper.
- [The microsoft/unilm repository](https://github.com/microsoft/unilm) is the code location that the paper's title page names.
