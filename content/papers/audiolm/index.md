---
id: "audiolm"
title: "AudioLM: a Language Modeling Approach to Audio Generation"
authors: "Borsos et al."
year: 2022
venue: "IEEE/ACM TASLP 2023"
arxiv: "2209.03143"
url: "https://arxiv.org/abs/2209.03143"
track: speech
tier: extended
est_hours: 4
status: full
summary: "AudioLM turns audio generation into next-token prediction over discrete tokens, and it uses two kinds of tokens because neither kind is enough alone. Semantic tokens, from k-means clusters of an intermediate w2v-BERT layer at 25 per second, carry content and long-range structure but reconstruct poorly, while acoustic tokens, from a 12-layer SoundStream codec at 50 steps per second, carry speaker and recording detail. Three decoder-only Transformers run in sequence, semantic tokens first, then coarse acoustic tokens, then fine acoustic tokens, so that each sequence stays short enough to model and a 3-second prompt yields a continuation that keeps the speaker's voice."
related_course: "speech: speech language models"
prereqs:
  - id: encodec
    reason: "AudioLM's acoustic tokens come from a SoundStream residual vector quantizer, and EnCodec is the open codec built on the same design, so you need its steps-per-second, layers-per-step, and bitrate arithmetic to count tokens."
  - id: wav2vec2
    reason: "w2v-BERT, which supplies the semantic tokens, extends wav2vec 2.0's contrastive objective with masked prediction, so you need wav2vec 2.0's feature encoder, masking, and quantized targets."
  - id: gpt2
    reason: "Each of the three stages is a decoder-only Transformer language model trained with next-token prediction, which is the model and objective that GPT-2 set out."
background: [semantic-vs-acoustic-tokens, hierarchical-modeling]
course_links:
  - title: "Codec language models"
    url: "https://speech.ifkash.dev/lessons/061-codec-language-models"
self_check:
  - q: 'A 10-second clip at 16 kHz has $T = 160000$ samples. How many semantic tokens and how many acoustic tokens (with $Q = 12$ layers) does AudioLM produce, and what is the acoustic bitrate with only the $Q'' = 4$ coarse layers?'
    a: 'The semantic rate is 25 Hz, so $T_S = T/640 = 250$. The acoustic rate is 50 Hz, so $T_A = T/320 = 500$ steps and $500 \cdot 12 = 6000$ tokens. With four layers of 1024 symbols the bitrate is $50 \cdot 4 \cdot \log_2 1024 = 2000$ bps (Section III-B).'
    see: encodec
  - q: 'Table I compares token types. Why does the paper not simply use the 250 bps semantic tokens alone for generation?'
    a: 'They reconstruct poorly. Their ViSQOL score is 1.1, against 3.3 for the 2000 bps acoustic tokens, because they were trained to discriminate phonetic content and not to preserve signal detail. Their ABX error is much lower (6.7 / 7.6 against 22.4 / 28.7), so they keep the content and drop the voice (Section III-B, Table I).'
    see: semantic-vs-acoustic-tokens
  - q: 'In stage 2, the semantic tokens appear in the input sequence. Does the model predict them, and why do they come first?'
    a: 'No. The stage 2 model is trained only on the acoustic tokens, $p(y^q_t \mid z, y^{\le Q''}_{<t}, y^{<q}_t)$. The semantic tokens are conditioning for the acoustic prediction, and the semantic stage has already generated them for the whole clip (Section III-C).'
    see: hierarchical-modeling
---

## The problem

Audio carries information at several scales at once. In speech, one second holds 16,000 samples, tens of phonemes, and a fragment of a sentence, and a good generator has to be right about all three. The paper describes the tension as high audio quality against long-term structure (Section I, Section III-B).

You can't solve it by modeling the waveform directly. Self-attention costs grow with the square of the sequence length, and a 10-second clip has 160,000 samples. So the paper models discrete tokens that a pre-trained tokenizer produces, and decodes them back to audio afterward (Section III-A). That leaves a choice of tokenizer, and each choice fails in one direction.

- **Acoustic tokens from a codec** reconstruct well. A decoder-only Transformer trained on them alone produces speech that keeps the prompt's speaker and recording conditions, but the linguistic content is inconsistent, "often akin to babbling" (Section III-B).
- **Semantic tokens from a masked language model** capture content and long-term structure, but "they are poorly invertible" (Section II). Table I shows the gap. Semantic tokens at 250 bps have a ViSQOL reconstruction score of 1.1, and acoustic tokens at 2000 bps score 3.3.

AudioLM uses both. Section IV-E reports that it reaches 64.7 on the sBLIMP syntactic probe, against 57.1 for GSLM, with no text supervision. Section IV-G reports that raters identified a 10-second AudioLM sample as synthetic only 51.2 percent of the time, which isn't significantly different from guessing (p = 0.23).

## Key ideas

The paper writes a waveform as $x \in \mathbb{R}^T$. Semantic tokens are $z = (z_1, \dots, z_{T_S})$, each in $\{1, \dots, K\}$. Acoustic tokens form the matrix $Y \in \{1, \dots, N\}^{T_A \times Q}$, where $y^q_t$ is the token from quantizer $q$ at time step $t$. The paper uses Roman-numeral sections, such as Section III-C, and it doesn't number its equations, so this page cites sections, tables, and figures. The code and the exercises index symbols from 0, while the paper starts at 1. Every number below comes from a script in `scripts/`. Run any script from the paper folder to reproduce its output.

### Two tokenizers, two token rates

**Intuition.** The two token types differ in what they keep and in how often they appear. Semantic tokens come at half the rate of acoustic steps, and each acoustic step carries several tokens.

**Math.** The acoustic tokenizer is a SoundStream codec ([SoundStream](/soundstream/), [EnCodec](/encodec/) for the open version). Its encoder has strides $(2, 4, 5, 8)$, so it reduces the sampling rate by $320$. For 16 kHz input it gives embeddings at 50 Hz. A residual vector quantizer with $Q$ layers and $N$ symbols per layer then turns each embedding into $Q$ tokens. The paper's example is $N = 1024$ and $Q = 4$, which gives $50 \cdot 4 \cdot \log_2 1024 = 2000$ bps (Section III-B). The trained model uses $Q = 12$, which is 6000 bps and 600 tokens per second (Section IV-B).

The semantic tokenizer starts from w2v-BERT, which combines a masked language modeling loss with a contrastive loss ([wav2vec 2.0](/wav2vec2/) supplies the contrastive half). The paper takes the activations of the 7th layer of the MLM module, normalizes each dimension to zero mean and unit variance, and trains k-means with $K = 1024$ clusters. The cluster index is the semantic token (Section III-B, Section IV-B). The normalization "significantly improves" phonetic discriminability. The features arrive at 25 Hz, so $T_S = T/640$, and the bitrate is $25 \cdot \log_2 1024 = 250$ bps.

**Tiny example, computed by `scripts/tokens_example.py`.** With the paper's settings:

| Quantity | Value |
|---|---|
| Acoustic frame rate | 50 Hz |
| Semantic frame rate | 25 Hz |
| Acoustic steps per semantic step | 2 |
| Semantic bitrate | 250 bps |
| Acoustic bitrate, $Q' = 4$ | 2000 bps, 200 tokens per second |
| Acoustic bitrate, $Q = 12$ | 6000 bps, 600 tokens per second |

A 10-second clip has $T = 160{,}000$ samples, $T_S = 250$, and $T_A = 500$. That is 250 semantic tokens, 2000 coarse acoustic tokens, and 6000 acoustic tokens in all. One semantic token covers 640 samples.

**Tiny example, computed by `scripts/kmeans_tokens.py`.** Take three 2-dimensional frames $[10, 0.1]$, $[30, 0.9]$, and $[22, 0.2]$, with per-dimension mean $[20, 0.5]$ and standard deviation $[10, 0.4]$. Normalizing gives $[-1, -1]$, $[1, 1]$, and $[0.2, -0.75]$. With centroids $[-1, -1]$, $[1, 1]$, and $[0.2, -1]$, the nearest centroids are 0, 1, and 2. Without the normalization, the first dimension dominates the distance and all three frames map to centroid 1. Exercise 4 asks you to implement the tokenizer, and the removal of repeated tokens that the paper applies in the first two stages (Section IV-B).

### Flattening the residual layers into one stream

**Intuition.** A language model reads one sequence of tokens. The codec gives a matrix with $Q$ tokens per time step, so the paper lays the matrix out in a line.

**Math.** Section III-B flattens $Y$ in row-major order. The flat sequence is $y = (y^1_1, y^2_1, \dots, y^Q_1, y^1_2, \dots, y^Q_{T_A})$, which has length $T_A \cdot Q$. It adds a vector of offsets $o$ so that each layer gets its own range of ids, with

$$
o_i = \big((i - 1) \bmod Q\big) \cdot N
$$

The paper then drops the offsets from its notation and "assumes proper offsetting implicitly". The total vocabulary is $Q \cdot N$. Time stays the slow axis, so the first $Q$ tokens in the stream describe step 1 from coarse to fine. A model that predicts $y^q_t$ has already seen every coarser layer at step $t$.

**Tiny example, computed by `scripts/flatten_example.py`.** Let $N = 1024$ and $Q = 4$, with $T_A = 3$ steps and 0-indexed symbols:

$$
Y = \begin{bmatrix} 5 & 900 & 17 & 3 \\ 6 & 12 & 1000 & 4 \\ 5 & 7 & 2 & 1023 \end{bmatrix}
$$

The row-major stream is `[5, 900, 17, 3, 6, 12, 1000, 4, 5, 7, 2, 1023]`. The offsets cycle through `[0, 1024, 2048, 3072]`, and the final ids are `[5, 1924, 2065, 3075, 6, 1036, 3048, 3076, 5, 1031, 2050, 4095]`. Layer 1 symbol 5 and layer 2 symbol 5 would share an id without the offsets. With them they are 5 and 1029. A column-major flattening gives `[5, 6, 5, 900, 12, 7, ...]`, which puts all of layer 1 before any of layer 2.

Exercise 1 asks you to flatten and invert this stream.

### Three stages

**Intuition.** One model over the semantic tokens and all 12 acoustic layers would see very long sequences. Instead the paper splits the job by how much context each part needs. Semantic tokens need the whole clip. Coarse acoustic tokens need the semantic tokens. Fine acoustic tokens need only a local window.

**Math.** Section III-A trains each language model to maximize $\prod_{t} p(h_t \mid h_{<t})$ over its token sequence. Section III-C gives the three conditional distributions, with $Q'$ for the number of coarse layers:

1. Semantic modeling: $p(z_t \mid z_{<t})$.
2. Coarse acoustic modeling: $p\big(y^q_t \mid z,\, y^{\le Q'}_{<t},\, y^{<q}_t\big)$ for $q \le Q'$.
3. Fine acoustic modeling: $p\big(y^q_t \mid y^{\le Q'},\, y^{>Q'}_{<t},\, y^{<q}_t\big)$ for $q > Q'$.

Multiplying the three gives the factorization $p(z)\,p(y^{\le Q'} \mid z)\,p(y^{>Q'} \mid y^{\le Q'})$. This product is my summary of the three stages, and the paper doesn't write it as one formula. Two conditional independence assumptions back it up. The first stage ignores the acoustic past, $p(z_t \mid z_{<t}, y_{<t}) \approx p(z_t \mid z_{<t})$. The third stage ignores the semantic tokens, because "fine acoustic tokens are conditionally independent from semantic tokens when conditioned on coarse acoustic tokens". The paper also assumes that fine detail depends only on nearby coarse tokens. That assumption lets stage 3 run on non-overlapping 3-second chunks, whatever the clip length (Section III-C).

The default configuration is $Q = 12$, $Q' = 4$, and $K = N = 1024$. Each stage is a decoder-only Transformer with 12 layers, 16 heads, width 1024, feed-forward width 4096, dropout 0.1, and T5-style relative position embeddings, which comes to about 0.3B parameters per stage ([GPT-2](/gpt2/) describes the block). Training uses random crops of 30, 10, and 3 seconds for the three stages (Section IV-B).

**Tiny example, computed by `scripts/tokens_example.py`.** For the 10-second clip:

| Sequence | Tokens | $n^2$ |
|---|---|---|
| Stage 1, semantic only | 250 | 62,500 |
| Stage 2, $T_S + T_A Q'$ | 2250 | 5,062,500 |
| Stage 3, one 3-second chunk, $150 \cdot 12$ | 1800 | 3,240,000 |
| One stage, semantic plus all 12 layers | 6250 | 39,062,500 |

The single-stage sequence is 7.72 times as costly in attention as stage 2. Figure 2 notes that, for every semantic token, stage 2 holds $2Q' = 8$ acoustic tokens and stage 3 holds $2(Q - Q') = 16$. Stage 3 sees one chunk at a time, so four chunks cover the 10 seconds, and its sequence length doesn't grow with the clip. The stage 3 count is my reading of the paper: the 3-second chunk holds all 12 layers and no semantic tokens. The paper gives the conditioning and the target but not the exact token layout.

Exercise 2 asks you to compute these lengths.

### Conditioning: the stage 2 sequence

**Intuition.** In stage 2, the semantic tokens sit in the same sequence as the acoustic tokens, in front of them, and the model attends to them like any earlier tokens. They are an input. They aren't targets.

**Math.** Section III-C writes the stage 2 sequence as

$$
\big(z_1, \dots, z_{T_S},\; y^1_1, \dots, y^{Q'}_1,\; y^1_2, \dots, y^{Q'}_2,\; \dots,\; y^{Q'}_{T_A}\big)
$$

with "$y^1_1$ being the first token predicted during training". Only the coarse layers of $Y$ appear, and the loss applies only to them. The paper doesn't say how the two vocabularies are kept apart. This page and Exercise 3 give the semantic tokens the ids $[0, K)$ and put layer $q$ of the acoustic tokens at $K + qN + y$, which is a choice.

**Tiny example, computed by `scripts/stage_sequences.py`.** Let $K = 8$, $N = 16$, $Q' = 2$, with $z = [3, 3, 5]$ and a $6 \times 3$ matrix $Y$ whose first two columns are `[1, 2], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]`. The coarse acoustic ids are `[9, 26, 12, 29, 14, 31, 16, 33, 18, 35, 20, 37]`. The whole sequence is `[3, 3, 5, 9, 26, 12, ...]`, so it has $3 + 6 \cdot 2 = 15$ tokens, and the loss mask is 0 on the first 3 and 1 on the last 12.

Exercise 3 asks you to build this sequence and mask.

### Generating a continuation

**Intuition.** To continue a prompt, AudioLM first imagines what is said next, then who says it and how it sounds, then the fine detail.

**Math.** Section III-D describes three modes. Unconditional generation samples all of $\hat{z}$ in stage 1. Acoustic generation takes the ground-truth $z$ and samples only the acoustic tokens. Continuation maps the prompt to its semantic tokens $z_{\le t_s}$ and its coarse acoustic tokens $y^{\le Q'}_{\le t_a}$. Stage 1 samples $\hat{z}_{>t_s}$ from $z_{\le t_s}$. Stage 2 gets the entire semantic sequence $(z_{\le t_s}, \hat{z}_{>t_s})$ and the prompt's coarse tokens, and samples the continuation. Stage 3 refines the coarse tokens, and the SoundStream decoder turns the prompt and the samples into a waveform $\hat{x}$.

The prompt's acoustic tokens are what carry the speaker. In Section IV-F, a 3-second prompt gives 92.6 percent speaker-classification accuracy on the 7-second continuations (Table III). Sampling acoustic tokens from the semantic tokens alone gives 3.2 percent, against a 0.3 percent chance level. That 3.2 percent is the paper's evidence that semantic tokens carry little speaker information (Section IV-D). The paper samples with temperatures 0.6, 0.8, and 0.6 for the three stages (Section IV-B).

<!-- exercises -->

## Pitfalls

- **Treating the acoustic matrix as a flat sequence of independent symbols.** Without the per-layer offsets, the token ids from different layers collide. The model then can't tell a layer 1 symbol 5 from a layer 4 symbol 5.
- **Flattening column by column.** The order must be row-major, so that all $Q$ layers of one time step come before the next step. The factorization in stage 2 needs the coarser layers at step $t$ to precede the finer ones.
- **Counting stage 2 with all 12 layers.** Stage 2 predicts only the $Q' = 4$ coarse layers. Its sequence is 2250 tokens for 10 seconds, and not 6250.
- **Giving stage 3 the semantic tokens.** The paper drops them in stage 3 to shorten the sequence. The stage then conditions on coarse acoustic tokens alone.
- **Mixing up the two frame rates.** Semantic tokens come at 25 Hz and acoustic steps at 50 Hz. The factor of 2 between them is why stage 2 holds $2Q'$ acoustic tokens per semantic token.
- **Clustering unnormalized features.** The paper normalizes each w2v-BERT dimension before k-means. Without it, the dimensions with the largest scale decide the cluster, and phonetic discriminability drops.
- **Forgetting the repeat removal.** In the first two stages the paper removes consecutive repetitions of semantic tokens (Section IV-B). This shortens stage 1, and it is why the 250 tokens for a 10-second clip is an upper bound. The page's token counts don't apply the removal.
- **Reading AudioLM as built on EnCodec and wav2vec 2.0.** The paper uses SoundStream and w2v-BERT, and EnCodec appeared a month later. EnCodec and wav2vec 2.0 are the open prerequisites here because they share the design, but their numbers differ. EnCodec at 24 kHz runs at 75 Hz, and AudioLM's codec at 16 kHz runs at 50 Hz.
- **Reading the stage 3 chunk as a conditioning window on both sides.** Chunks are non-overlapping 3-second pieces, and the paper's assumption is that fine detail is determined locally by the coarse tokens.
- **Taking the human-evaluation result as a safety result.** Section IV-G says that raters can't tell AudioLM continuations from real speech, and Section IV-H says that a simple classifier detects them with 98.6 percent accuracy. Both numbers compare against SoundStream-compressed originals.

## Further reading

- The [AudioLM samples page](https://google-research.github.io/seanet/audiolm/examples) is the accompanying material that the paper points to.
- [SoundStream](https://arxiv.org/abs/2107.03312), by Zeghidour et al., is the codec that supplies the acoustic tokens.
- [w2v-BERT](https://arxiv.org/abs/2108.06209), by Chung et al., is the model that supplies the semantic tokens.
- [Generative Spoken Language Modeling from Raw Audio](https://arxiv.org/abs/2102.01192), by Lakhotia et al., is the textless NLP baseline that AudioLM compares against in Sections IV-C and IV-E.
- [Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers](https://arxiv.org/abs/2301.02111), by Wang et al., is VALL-E, which applies a similar coarse-then-fine token scheme to text-to-speech.
