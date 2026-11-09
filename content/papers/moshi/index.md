---
id: "moshi"
title: "Moshi: a speech-text foundation model for real-time dialogue"
authors: "Defossez et al."
year: 2024
venue: "arXiv preprint"
arxiv: "2410.00037"
url: "https://arxiv.org/abs/2410.00037"
track: speech
tier: core
est_hours: 5
status: full
summary: "Moshi is a speech-to-speech dialogue model that models its own voice and the user's voice as parallel token streams, so it listens and speaks at the same time and has no explicit turns. It uses Mimi, a streaming codec at 12.5 frames per second with 8 codebooks, where the first codebook is distilled from WavLM so it carries semantic content, and a large Temporal Transformer over time steps plus a small Depth Transformer over codebooks keeps each step cheap. Predicting time-aligned text tokens before the audio tokens, which the paper calls Inner Monologue, improves the linguistic quality of the speech, and the whole system reaches a theoretical latency of 160 ms."
related_course: "speech: speech LLMs"
prereqs:
  - id: audiolm
    reason: "Moshi keeps AudioLM's order of semantic tokens first and acoustic tokens after, but distills the semantic tokens into the first codebook of its own codec so they can be computed causally."
  - id: encodec
    reason: "Mimi starts from the EnCodec recipe of a causal convolutional encoder and decoder, a residual vector quantizer, and adversarial training, and changes its frame rate, bottleneck, and first codebook."
  - id: llama
    reason: "Helium, the text model that Moshi builds on, uses the LLaMA recipe of RMSNorm before each block, rotary position embeddings, and a gated feed-forward network."
background: [full-duplex, inner-monologue, streaming]
course_links:
  - title: "Full-duplex models and voice agents"
    url: "https://speech.ifkash.dev/lessons/065-full-duplex-and-voice-agents"
  - title: "Spoken language models"
    url: "https://speech.ifkash.dev/lessons/064-spoken-language-models"
self_check:
  - q: 'Mimi has encoder strides $(4, 5, 6, 8, 2)$ on 24 kHz audio. How many frames per second does it produce, and how many tokens per second does a single audio stream add with $Q = 8$ codebooks?'
    a: 'The hop is $4 \cdot 5 \cdot 6 \cdot 8 \cdot 2 = 1920$ samples, so $24000 / 1920 = 12.5$ frames per second, or 80 ms per frame. With 8 codebooks that is 100 tokens per second, which is why Moshi does not flatten the codebooks into one sequence.'
    see: encodec
  - q: 'AudioLM generates semantic tokens in a separate stage before acoustic tokens. Which part of that design does Moshi keep, and which part does it change?'
    a: 'It keeps the order: coarse semantic content first, acoustic detail after. It changes where the semantic tokens come from. Mimi distills them from WavLM into its first codebook, so a causal encoder produces them at 12.5 Hz, and one model generates semantic and acoustic tokens together at each step.'
    see: audiolm
  - q: 'Why does Moshi predict text tokens before the audio tokens of the same step, and what does it do with the text stream of the user?'
    a: 'The text acts as a prefix, so the model decides the words first and then the sounds, which improves the linguistic quality of the speech. Moshi models text only for its own speech. Transcribing the user in real time would need an external recognizer, which breaks the end-to-end speech-to-speech design.'
    see: inner-monologue
---

## The problem

A spoken assistant built as a pipeline chains voice activity detection, speech recognition, a text language model, and speech synthesis. The paper names three costs of that design (abstract and Section 1).

- **Latency adds up.** Every stage waits for the one before it, so the delay between the end of your sentence and the start of the reply is typically several seconds. The paper notes that natural conversation has an average gap of about 230 ms.
- **Text is the only channel.** Emotion, a laugh, or a sigh doesn't survive transcription, so the language model never sees it.
- **The design assumes turns.** Suppose the assistant is mid-sentence and you say "mm-hm". A turn-based system has to decide whether that was an interruption. It can't treat it as a back-channel and keep talking, because it has no representation of two people speaking at once.

Moshi drops the pipeline. One Transformer reads the user's audio and writes its own audio, one 80 ms frame at a time, and it also writes the text of what it says. This page follows the paper's three components: the Mimi codec (Section 3.3), the RQ-Transformer that generates audio tokens at low cost (Sections 3.4.1 and 3.4.2), and the joint sequence that adds the user's stream and the text stream (Sections 3.4.3 and 3.4.4). The AudioLM page, [AudioLM](/audiolm/), explains semantic and acoustic tokens. The [EnCodec](/encodec/) page explains RVQ. The [LLaMA](/llama/) page covers the architecture that Helium follows.

## Key ideas

The notation is the paper's. $S$ is the number of time steps and $K$ the number of sub-sequences at each step. $V_{s,k}$ is the token of sub-sequence $k$ at step $s$. $A_{t,q}$ are the audio tokens of Mimi with $Q = 8$ codebooks and $T$ frames, and $A'$ are the audio tokens of the user. $W_t$ is the text token at frame $t$. Every number comes from a script in `scripts/`. Run any script from the paper folder to reproduce the output.

### Mimi: a codec that runs at 12.5 Hz

**Intuition.** A language model generates one token per step, so every extra token per second of audio costs time at inference. Mimi cuts the cost twice. It runs at 12.5 frames per second, six times fewer frames than EnCodec's 75 per second, and it is causal, so it can encode and decode in a live stream. It also puts semantic content into its first codebook, so Moshi doesn't need a separate non-causal encoder like the w2v-BERT in AudioLM.

**Math.** The encoder is a causal convolutional autoencoder (Section 3.3.1). Four blocks with strides $(4, 5, 6, 8)$ and a final convolution with stride 2 turn a 24 kHz waveform into a latent of dimension $D = 512$ at

$$
\frac{24000}{4 \cdot 5 \cdot 6 \cdot 8 \cdot 2} = 12.5 \text{ frames per second.}
$$

A residual vector quantizer with $Q = 8$ codebooks of $N_A = 2048$ entries discretizes each frame, so the tokens live in $\{1, \dots, N_A\}^{S \times Q}$. The paper quotes the bitrate as 1.1 kbps. One difference from AudioLM: Mimi's tokens come from one model, and the first codebook is also the semantic one.

Section 3.3.2 distills that semantic information from WavLM, a self-supervised speech model that produces 1024-dimensional embeddings at 50 Hz. Average pooling with a kernel of 8 and a stride of 4 brings 50 Hz down to 12.5 Hz. A linear projection of the first quantizer's output is trained to match the pooled embedding with a cosine distance. The paper finds the pooling must be non-causal, which is fine, because the targets exist only during training.

A single RVQ with the distillation loss in level 1 trades audio quality for phonetic quality, because levels 2 to 8 only see the residual. The paper's fix is a **split RVQ**: one plain VQ trained to carry the semantic target, plus a 7-level RVQ run in parallel, with their outputs summed. That still gives 8 tokens per frame: 1 semantic and 7 acoustic.

**Tiny example, computed by `scripts/token_rates.py`.** The product of the strides is 1920 samples, so the frame rate is 12.5 Hz and each frame is 80 ms. A codebook of 2048 entries costs 11 bits per token, so 8 tokens at 12.5 frames per second give $12.5 \cdot 8 \cdot 11 = 1100$ bits per second. The WavLM target after pooling is $50 / 4 = 12.5$ Hz, and it matches. For comparison, an EnCodec-style hop of 320 samples gives 75 Hz, which is 6 times more frames.

Exercise 1 asks you to compute these numbers.

### RQ-Transformer: a large model over time, a small one over depth

**Intuition.** Mimi produces 8 tokens per frame. If you flatten them into one long sequence, a model has to take 8 steps for each 80 ms of audio. The RQ-Transformer splits the work. A large Temporal Transformer takes one step per frame and summarizes the past. A small Depth Transformer then spends a few cheap steps inside the frame, one for each codebook.

**Math.** Section 3.4.1 stacks the sub-sequences as $V_{s,k}$ with $1 \le s \le S$ and $1 \le k \le K$, and writes $V_s = (V_{s,1}, \dots, V_{s,K})$. The Temporal Transformer maps all earlier steps to a context vector (Equation 1):

$$
z_s = \mathrm{Tr}_{\mathrm{Temp}}(V_0, \dots, V_{s-1}) \in \mathbb{R}^d
$$

For $1 < k \le K$, the Depth Transformer maps $z_s$ and the tokens already generated in step $s$ to logits (Equation 2):

$$
l_{s,k} = \mathrm{Tr}_{\mathrm{Depth}}(z_s, V_{s,1}, \dots, V_{s,k-1}) \in \mathbb{R}^{N_k}
$$

For $k = 1$ the logits are $l_{s,1} = \mathrm{Lin}(z_s)$. Equation 3 states what training asks for: $\mathrm{softmax}(l_{s,1}) \approx P[V_{s,1} \mid V_0, \dots, V_{s-1}]$ and, for $k > 1$, $\mathrm{softmax}(l_{s,k}) \approx P[V_{s,k} \mid V_0, \dots, V_{s-1}, V_{s,1}, \dots, V_{s,k-1}]$. The Temporal Transformer takes $S$ steps and the Depth Transformer at most $K$ steps per frame. Table 1 gives the sizes: the Temporal Transformer is the 7B Helium model (4096 dimensions, 32 layers) and the Depth Transformer is small (1024 dimensions, 6 layers, 16 heads).

The paper also uses different weights for each codebook index $k$ in the Depth Transformer's linear and feed-forward layers. It argues that different codebooks need different transformations and that the small size makes this cheap (Section 3.4.1, and Table 6 for the ablation).

**Tiny example, computed by `scripts/token_rates.py`.** Take 5 minutes of one audio stream. Flattened, the sequence has $3750 \cdot 8 = 30000$ steps, which is the figure the paper gives in Section 3.4.1. The Temporal Transformer takes 3750 steps. Attention compares every step with every other step, so the number of pairs falls by $8^2 = 64$. The Temporal Transformer's context in Table 1 is 3000 steps, which is $3000 / 12.5 = 240$ seconds, or 4 minutes.

### Acoustic delay

**Intuition.** Given the past, the 8 tokens of one frame aren't independent: the acoustic tokens depend on the semantic token of the same frame. A model that predicts all of them at once has to capture that dependence in a single step. If the acoustic tokens are shifted later in time, they can be predicted after the semantic token has been seen, and the big Temporal Transformer handles that dependence.

**Math.** With a delay $\tau \in \mathbb{N}$ the model sequence $V$ is built from the codec output $A$ as (Section 3.4.2, Equation 4):

$$
V_{s,1} = A_{s,1}, \qquad
V_{s,q} = \begin{cases} A_{s-\tau,\,q} & s \ge \tau + 1 \\ 0 & s < \tau + 1 \end{cases}
\quad (q > 1)
$$

The semantic token, $q = 1$, isn't delayed. Every acoustic token comes $\tau$ steps late, and $0$ is the initial token before that. The delay has a price, which is latency. Each step is 80 ms, so the model can't finish a frame until $\tau + 1$ steps have passed. Section 5.3 reports the following: a pattern `[0, 1, 2, 3, 4, 5, 6, 7]` from earlier work gives 640 ms, `[0, 2, 2, 2, 2, 2, 2, 2]` gives 240 ms, and `[0, 0, 0, 0, 0, 0, 0, 0]` gives the minimum of 80 ms. One step of delay "significantly improves" speech quality over none, and two steps bring a further, smaller gain. The final model pretrains with $\tau = 2$ and finetunes with $\tau = 1$ for a theoretical latency of 160 ms.

**Tiny example, computed by `scripts/delay_pattern.py`.** Take $Q = 3$, $T = 5$ and $\tau = 1$. Row 1 stays `[1, 2, 3, 4, 5]`. Row 2 goes from `[16, 17, 18, 19, 20]` to `[0, 16, 17, 18, 19]` and row 3 from `[31, 32, 33, 34, 35]` to `[0, 31, 32, 33, 34]`. The latency is $(\max \text{delay} + 1) \cdot 80$ ms: the four patterns above give 80, 160, 240, and 640 ms. This formula is the page's reading of the paper's figures and it matches all of them.

Exercise 2 asks you to implement the delay and the latency.

### Two speakers in one sequence

**Intuition.** Moshi doesn't have a "user turn" and a "system turn". It has two audio streams that are always present. When the user talks and Moshi is quiet, Moshi's stream is near-silence. When both talk, both streams carry sound.

**Math.** Section 3.4.3 applies the acoustic delay to both streams and concatenates them. The two streams are $A$ for Moshi and $A'$ for the user. The paper's final sequence (Section 3.4.4, Equation 6) adds the text stream $W$ as the first row:

$$
\begin{aligned}
V_{s,1} &= W_s && \text{aligned text tokens} \\
V_{s,2} &= A_{s,1} && \text{semantic tokens of Moshi} \\
V_{s,1+q} &= A_{s-\tau,q} && s \ge \tau + 1,\ 1 < q \le Q \quad \text{delayed acoustic tokens of Moshi} \\
V_{s,1+Q+1} &= A'_{s,1} && \text{semantic tokens of the user} \\
V_{s,1+Q+q} &= A'_{s-\tau,q} && s \ge \tau + 1,\ 1 < q \le Q \quad \text{delayed acoustic tokens of the user}
\end{aligned}
$$

That makes $K = 2Q + 1 = 17$ sub-sequences for $Q = 8$. At inference, the model samples Moshi's own rows, which are the text row and Moshi's $Q$ audio rows, and it discards its prediction for the user rows and uses the real microphone audio instead. It predicts the user's rows during training, and the paper uses that to simulate dialogues for evaluation (Section 3.4.4, "Inference of Moshi").

A note on indices. The paper says Moshi samples $k \in \{2, \dots, 2 + Q\}$ for audio and that user audio is $k > 2 + Q$. Equation 6 puts Moshi's audio at $k = 2, \dots, Q + 1$ and the user's semantic row at $k = Q + 2$. Counting the rows of Equation 6 gives $Q + 1$ rows for Moshi including text, which matches the figure, so the page and the exercise use $k = 1, \dots, Q + 1$ and treat the paper's upper bound as an off-by-one.

**Tiny example, computed by `scripts/joint_sequence.py`.** With $Q = 3$, $T = 4$, $\tau = 1$, text $W = [101, 102, 103, 104]$, Moshi audio $1, \dots, 12$ and user audio $201, \dots, 212$, the $7 \times 4$ joint sequence is:

| Row | Content | Values |
|---|---|---|
| 1 | text | `101 102 103 104` |
| 2 | Moshi semantic | `1 2 3 4` |
| 3 | Moshi acoustic 2 | `0 5 6 7` |
| 4 | Moshi acoustic 3 | `0 9 10 11` |
| 5 | user semantic | `201 202 203 204` |
| 6 | user acoustic 2 | `0 205 206 207` |
| 7 | user acoustic 3 | `0 209 210 211` |

Rows 1 to 4 are sampled and rows 5 to 7 come from the microphone. At $Q = 8$ the same layout has 17 rows and 212.5 tokens per second, but the Temporal Transformer still takes 12.5 steps per second. Flattening all 17 rows would need 63,750 steps for 5 minutes (computed by `scripts/token_rates.py`).

Exercise 3 asks you to build this array.

### Inner Monologue

**Intuition.** Audio tokens carry sound first and meaning second. A model that must produce the audio directly has to decide what to say and how to say it in the same step. Moshi writes the words first. The text tokens of its own speech sit in the row above the semantic tokens, so each step the model decides the word piece and then generates the sound, and the paper says this scaffolding raises the linguistic quality of what it says (Section 3.4.4). The ablation in Section 5.3 calls it the most drastic improvement to the quality and length of generated speech.

**Math.** Whisper transcribes Moshi's speech with word-level timestamps. The $i$-th word becomes $n_i$ tokens $w_{i,j}$ with a start index $t_i$ in frames, which is its start time times 12.5 Hz. The stream $W \in \{1, \dots, N_W\}^T$ is built from two special tokens, PAD and EPAD (Equation 5). Start with PAD everywhere. Then, for each word, put EPAD in the slot just before the word and write the word's tokens from $t_i$ on. The paper doesn't insert an EPAD when it would overwrite a token of the previous word, and when $t_i = 1$ it inserts the EPAD at index 1 and shifts the tokens. As printed, Equation 5 writes the tokens at $t_i + j$ for $j$ from 1, which would leave a gap between the EPAD and the first token. The page follows Figure 4, where the first token comes right after the EPAD, which is the same as counting $j$ from 0.

Two uses of the text delay are in Section 3.4.4 and Section 5.7. If the text lags the audio, the text is decided by the audio, and sampling only text gives a streaming speech recognizer. If the text leads the audio, the audio follows the text, which gives streaming speech synthesis. The same loss, architecture, and data serve both, and only the delay changes.

**Tiny example, computed by `scripts/text_alignment.py`.** Take 12 frames and three words: "Hello," (2 tokens, start 1), "I'm" (3 tokens, start 5), and "Moshi" (1 token, start 9). The text stream is `EPAD Hello , EPAD I ' m EPAD Moshi PAD PAD PAD`. Three of the 12 slots are PAD, a fraction of 0.25, but that is a toy: the words are close together. The paper reports that in English conversational speech about 65% of the text slots are PAD (Section 3.4.4). At 12.5 slots per second, that leaves about 4.4 non-PAD slots per second, and the page reads this against the paper's 3 to 4 text tokens per second of speech (Section 3.4.1). The extra slots are the EPADs. A word that starts right after another one gets no EPAD: `[(2, ["a", "b"]), (4, ["c"])]` on 6 slots gives `EPAD a b c PAD PAD`.

Exercise 4 asks you to implement this alignment.

<!-- exercises -->

## Pitfalls

- **Flattening the codebooks.** At 12.5 Hz and 8 codebooks, a flattened sequence has 100 steps per second. Moshi keeps the Temporal Transformer at 12.5 steps per second and puts the codebooks in the Depth Transformer.
- **Confusing the frame rate with the token rate.** Mimi emits 12.5 frames per second, and each frame has 8 audio tokens. The token rate of one stream is 100 per second, and for the whole joint sequence it is 212.5 per second.
- **Delaying the semantic token.** Equation 4 delays only the acoustic codebooks. A delay on row 1 would delay the content the acoustic tokens are supposed to follow.
- **Counting latency from the delay alone.** The minimum is one frame, 80 ms, with no delay. The paper's 160 ms is a delay of one step plus that frame. It is a theoretical figure, and the paper reports 200 ms in practice.
- **Rolling instead of shifting.** `np.roll` wraps the last tokens to the front. The front of a delayed row must be the initial token $0$.
- **Writing an EPAD over a word.** Words that are close together leave no free slot, and the paper skips the EPAD in that case. An unconditional write corrupts the previous word.
- **Indexing step 1.** A word starting at the first frame has no slot at step 0. In Python, `W[t - 2]` with $t = 1$ is `W[-1]`, the last slot, so the bug doesn't raise an error.
- **Treating the user's rows as outputs.** At inference Moshi ignores its own prediction for the user's audio and uses the real signal.
- **Expecting a turn boundary.** Nothing in the architecture marks the change of speaker. When Moshi is silent, its audio stream decodes to near-silence and its text row is filled with PAD (Section 3.4.4). Forcing an EPAD token makes it start talking immediately.
- **Reading the latency as a benchmark.** The 160 ms comes from the frame length and the delay alone. The paper reports 200 ms in practice, and neither number is a benchmark of your own deployment.

## Further reading

- [The Moshi code and models](https://github.com/kyutai-labs/moshi), from Kyutai, is the official release, with reference inference code.
- [Moshi: a speech-text foundation model for real-time dialogue](https://arxiv.org/abs/2410.00037), the paper. Appendix C describes how to get a zero-shot padded text sequence for the ASR and TTS use.
- [SpeechTokenizer](https://arxiv.org/abs/2308.16692), by Zhang et al., is the earlier codec that distills semantic information into the first RVQ level. Mimi follows it.
- [Simple and Controllable Music Generation](https://arxiv.org/abs/2306.05284), by Copet et al., introduced the codebook delay pattern that Moshi shortens.
- [Autoregressive Image Generation using Residual Quantization](https://arxiv.org/abs/2203.01941), by Lee et al., proposed the RQ-Transformer.
- [Full-duplex models and voice agents](https://speech.ifkash.dev/lessons/065-full-duplex-and-voice-agents) covers the full-duplex idea in the speech course.
