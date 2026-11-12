---
id: "whisper"
title: "Robust Speech Recognition via Large-Scale Weak Supervision"
authors: "Radford et al."
year: 2022
venue: "ICML 2023"
arxiv: "2212.04356"
url: "https://arxiv.org/abs/2212.04356"
track: speech
tier: core
est_hours: 5
status: full
summary: "Whisper trains a standard encoder-decoder Transformer to predict the transcripts of 680,000 hours of audio collected from the internet. The data is noisy and only weakly supervised, but its size and variety make the model work on new datasets with no fine-tuning. One decoder handles transcription, translation to English, language identification, and voice activity detection, because the task is spelled out in special tokens at the start of the decoder's input."
related_course: "speech: ASR systems"
prereqs:
  - id: transformer
    reason: "Whisper is an encoder-decoder Transformer with cross-attention from the decoder to the encoder, so you need the attention and block structure from that paper."
  - id: bpe
    reason: "The decoder predicts byte-level BPE tokens, and the multilingual models refit the vocabulary at the same size to avoid fragmenting other languages."
  - id: wav2vec2
    reason: "wav2vec 2.0 is the self-supervised baseline that Whisper is measured against, and its need for task-specific fine-tuning is the weakness this paper sets out to remove."
background: [log-mel-features, multitask-token-format]
course_links:
  - title: "Whisper and large-scale recognition"
    url: "https://speech.ifkash.dev/lessons/041-whisper"
self_check:
  - q: 'The multilingual Whisper models refit the BPE vocabulary but keep its size. What goes wrong if you reuse the English-only GPT-2 vocabulary for Korean text?'
    a: 'The GPT-2 merges were learned on English, so Korean text falls apart into many short pieces (often single bytes). Sequences get longer, the decoder runs more steps, and each token carries less information. Refitting the merges on multilingual text fixes this, and keeping the size fixed leaves the embedding table and softmax cost unchanged (Section 2.2).'
    see: bpe
  - q: 'How many samples are in a 30 s window at 16 kHz, how many log-Mel frames does a 25 ms window with a 10 ms stride give, and how many positions does the encoder see after its stem?'
    a: 'There are $30 \cdot 16000 = 480{,}000$ samples. The 10 ms stride gives 3,000 frames of 80 channels. The second stem convolution has stride 2, so the encoder sees 1,500 positions of 20 ms each (Section 2.2).'
    see: log-mel-features
  - q: 'In a decoder block with cross-attention, which tensor supplies the queries and which supplies the keys and values? What is the shape of one head''s cross-attention weight matrix for a 10-token prefix?'
    a: 'The queries come from the decoder states, after causal self-attention. The keys and values come from the encoder output. With a 30 s window the encoder has 1,500 positions, so the weight matrix has shape $10 \times 1500$, and each row sums to 1 over the 1,500 encoder positions (Figure 1; Section 3.2.3 of the Transformer paper).'
    see: transformer
---

## The problem

By 2022, speech recognition models were good at the test sets they were trained for and bad at everything else. [wav2vec 2.0](/wav2vec2/) had shown that a model could learn strong audio representations from unlabeled speech, and had been scaled to 1,000,000 hours (Section 1). But a pre-trained encoder has no decoder. To get transcripts out of it, a practitioner has to fine-tune it on a labeled dataset, and the fine-tuned model inherits that dataset's quirks (Section 1).

Consider the gap that results. The paper compares a wav2vec 2.0 Large model fine-tuned on LibriSpeech with Whisper Large V2. Both score 2.7 word error rate (WER) on LibriSpeech clean. On the other 13 datasets, the wav2vec 2.0 model averages 29.3 WER, and Whisper averages 12.8, which is 55.2% fewer errors on average (Section 3.3, Table 2). On Common Voice the figures are 29.9 and 9.0. The same score on the training distribution hides a large difference everywhere else. `scripts/robustness_table.py` recomputes every column of that table.

The paper's answer is to stop chasing a clean training set. Existing supervised sets are small, for example 5,140 hours for SpeechStew, which merges seven datasets (Section 1). Transcripts exist on the internet in much larger quantities, and they're imperfect. So the paper collects 680,000 hours of audio paired with transcripts, filters out the worst, and trains a plain sequence-to-sequence Transformer to predict the text. It uses no self-supervision and no self-training (Section 1). The model is then evaluated zero-shot, with none of the evaluation sets' training data (Section 3.1).

## Key ideas

The paper has no numbered equations. It's a data and systems paper, and its core contributions are the dataset, a token format, and a set of decoding heuristics. So this page states each idea in words, cites sections, tables, and Figure 1, and uses small computations to make the formats concrete. The page's own notation is as follows: $T$ is the number of spectrogram frames, $d$ is the model width, and $L$ is the number of layers in each of the encoder and the decoder. Where the paper is silent, the page says "the released code" and means [github.com/openai/whisper](https://github.com/openai/whisper). Every number comes from a script in `scripts/`. Run any script from the `scripts/` folder to reproduce its output.

### Weak supervision at scale

**Intuition.** Self-supervised pre-training uses audio alone, which is plentiful but gives no decoder. Gold-standard transcripts give a decoder but are scarce. Internet audio with human-made captions sits in between. The transcripts are noisy, misaligned, or sometimes written by other ASR systems. The bet is that enough of them, across enough conditions, beats a small clean set.

**Method.** The pipeline is mostly about removing what would hurt (Section 2.1):

- It trains on raw transcript text, with no text normalization, so the model learns to produce punctuation and casing directly.
- It removes transcripts that look machine-generated. For example, an all-uppercase or all-lowercase transcript is very unlikely to be human-written.
- It runs a language detector on the audio and drops pairs where the spoken language doesn't match the transcript language. As an exception, English transcripts of non-English speech become X-to-English translation examples.
- It cuts audio into 30-second segments paired with the transcript text inside each segment. It keeps segments with no speech, at a lower rate, as training data for voice activity detection.

The mix is 680,000 hours in total, with 117,000 hours covering 96 other languages and 125,000 hours of X-to-English translation (Section 1). The English transcription share isn't stated directly, so `scripts/dataset_scaling.py` derives it as the remainder, 438,000 hours, or 64.4%.

**Tiny example, computed by `scripts/dataset_scaling.py`.** Table 6 in Section 4.2 trains a medium-sized model on subsets of the data. The subsets are 0.5%, 1%, 2%, 4%, and 8% of the full set, and the script confirms those fractions from the hour counts. Reading the table:

| Hours | English WER | Multilingual WER | X-to-en BLEU |
|---|---|---|---|
| 3,405 | 30.5 | 92.4 | 0.2 |
| 13,621 | 14.4 | 56.6 | 7.9 |
| 54,486 | 10.9 | 36.4 | 19.2 |
| 681,070 | 9.9 | 29.2 | 24.8 |

The last step multiplies the data by 12.5 and lowers English WER by only 1.0 point, a 9.2% relative drop. For multilingual recognition the same step lowers WER by 7.2 points, a 19.8% relative drop. The paper says multilingual WER follows a power law up to 54,486 hours and then deviates (Section 4.2). The script fits that power law on the first five rows and gets $\text{WER} \approx 1433 \cdot \text{hours}^{-0.338}$. Each doubling of data multiplies WER by 0.791. Extrapolated to 681,070 hours, the fit predicts 15.3, but the table reports 29.2. The paper offers two possible explanations: the models are under-trained for the dataset size, or the gains from data scale are nearly exhausted (Section 4.2). It doesn't settle between them.

### The log-Mel front end and the 30-second window

**Intuition.** The model doesn't read raw samples. It reads a picture of the sound: energy per frequency band, per 10 ms. Every input is exactly 30 seconds, padded with silence or cut, so the encoder always sees the same shape.

**Method.** All audio is resampled to 16,000 Hz. The model then computes an 80-channel log-magnitude Mel spectrogram on 25 ms windows with a 10 ms stride, and scales the input globally to between $-1$ and $1$, with approximately zero mean across the pre-training dataset (Section 2.2). The encoder starts with a stem of two convolution layers with filter width 3 and GELU, and the second layer has stride 2 (Section 2.2). Sinusoidal position embeddings are added after the stem.

The paper doesn't give the FFT size, the Mel scale, or the exact scaling. This page follows the released code: a 400-sample FFT with a periodic Hann window, Slaney-style Mel filters, a centered STFT with the last frame dropped, power spectra, `log10` of the Mel energies, a floor of 8 decades below the maximum, and then $(x + 4)/4$. So the code's scaling depends on each clip's maximum, which differs from the paper's description of one global scale.

**Tiny example, computed by `scripts/frontend_example.py`.**

| Quantity | Value |
|---|---|
| Samples in 30 s at 16 kHz | 480,000 |
| Window and hop | 400 samples (25 ms), 160 samples (10 ms) |
| FFT bins | 201 |
| STFT frames with centering | 3,001, and 3,000 after dropping the last |
| Frames without centering | 2,998 |
| Input to the encoder | `(80, 3000)`, which is 240,000 numbers |
| After the stride-2 convolution | 1,500 positions of 20 ms |

The 20 ms position spacing matters later. It's the grid of the timestamp tokens. For a 1 kHz sine tone, the script gets a spectrogram of shape `(80, 3000)` whose loudest band at the middle frame is band 26, whose filter peaks near 1,000 Hz. Silence maps to a constant, $(\log_{10} 10^{-10} + 4)/4 = -1.5$.

Exercise 3 asks you to write this function.

### An encoder-decoder Transformer, unchanged

**Intuition.** The paper deliberately uses an off-the-shelf architecture "to avoid confounding our findings with model improvements" (Section 2.2). The aim is to learn what data does, not what a new architecture does. The model is the [Transformer](/transformer/) encoder-decoder with a few choices that matter for audio.

**Method.** The details from Section 2.2 and Table 1:

- The encoder is the stem described above, then Transformer blocks with pre-activation residuals, then a final layer normalization.
- The decoder uses learned position embeddings and ties the input and output token embeddings. Each decoder block has causal self-attention, then cross-attention to the encoder output, then an MLP (Figure 1).
- The encoder and the decoder have the same width and the same number of blocks.
- The tokenizer is the byte-level [BPE](/bpe/) from GPT-2 for English-only models. The multilingual models refit the vocabulary at the same size.

Table 1 lists five sizes:

| Model | Layers $L$ | Width $d$ | Heads | Parameters (Table 1) |
|---|---|---|---|---|
| Tiny | 4 | 384 | 6 | 39M |
| Base | 6 | 512 | 8 | 74M |
| Small | 12 | 768 | 12 | 244M |
| Medium | 24 | 1024 | 16 | 769M |
| Large | 32 | 1280 | 20 | 1550M |

**Tiny example, computed by `scripts/model_size.py`.** Count the parameters from those dimensions. The paper doesn't state the vocabulary size, the decoder context length, or which projections have a bias. The script uses the released code's values: a vocabulary of 51,865 tokens, a decoder context of 448 tokens, and no bias on the key projection. It gets 37.2M for Tiny, 71.8M for Base, 240.6M for Small, 762.3M for Medium, and 1,541.4M for Large. Those are 0.6% to 4.7% below the table, and the paper doesn't say how it counted, so the script doesn't reproduce the table exactly. Two facts stand out. The decoder is bigger than the encoder at every size, because each decoder block has an extra attention layer. Also, the tied token embedding is a large part of a small model: 53.6% of Tiny's parameters, and 4.3% of Large's.

The cross-attention shape is the one new piece relative to the Transformer paper's encoder. At each decoder layer, a query from a token position scores all 1,500 encoder positions, which is the only way the text side reads the audio.

### One decoder, many tasks

**Intuition.** A speech system usually has separate parts: voice activity detection, language identification, recognition, translation, timestamp alignment. Whisper replaces the glue between them with a prompt. The decoder is an audio-conditioned language model, so the task is a prefix of special tokens, and the model completes it (Section 2.3).

**Format.** The sequence follows Section 2.3 and Figure 1:

1. Optionally, the transcript text that precedes this window, to give the model long-range context. Training adds it with some probability, and Table 17 gives the rate as 50%. The paper's Figure 1 marks it `PREV`, and the released code spells the marker `<|startofprev|>`.
1. `<|startoftranscript|>`.
1. A language token, one for each of the 99 languages in the training set. If there's no speech, the model predicts `<|nospeech|>` here instead.
1. A task token, `<|transcribe|>` or `<|translate|>`.
1. `<|notimestamps|>` if the model shouldn't predict timestamps.
1. The output: text tokens, with timestamp tokens interleaved in timestamp mode. The start time comes before each caption's text and the end time after it.
1. `<|endoftranscript|>`.

Times are relative to the 30-second window and quantized to the nearest 20 ms, which matches the encoder's resolution (Section 2.3). The loss is masked over the previous-context text and applies to every other token (Section 2.3). The paper uses these token spellings in the text; the exact spelling of the timestamp tokens, `<|2.40|>`, comes from the released code.

**Tiny example, computed by `scripts/prompt_example.py`.** The prompt for English transcription without timestamps is

`<|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>`

For Spanish audio translated to English with timestamps, it's `<|startoftranscript|> <|es|> <|translate|>`. The language token names the language that is spoken, not the language of the output, and `translate` always means into English. A 30 s window has 1,501 timestamp tokens, from `<|0.00|>` to `<|30.00|>`. A time of 0.031 s rounds to `<|0.04|>`, and 0.009 s rounds to `<|0.00|>`. With two segments, "Ask not" from 0.0 to 2.4 s and "what your country" from 2.4 to 5.0 s, the target is

`<|0.00|> Ask not <|2.40|> <|2.40|> what your country <|5.00|> <|endoftranscript|>`

With timestamps off, it is the text and `<|endoftranscript|>` only.

The format is also where the paper's multitask claim is tested. For small models, joint training on many tasks and languages hurts English recognition compared with an English-only model trained with the same compute. For the largest models, the joint model is better (Section 4.3, Figure 9). So the shared format pays off only with scale.

Exercise 1 asks you to build the prompt and the loss mask. Exercise 2 asks you to write the timestamp tokens.

### Decoding long audio

**Intuition.** The model was trained on 30-second windows, so it can't read an hour of audio at once. The system slides a window along the audio, using the predicted timestamps to decide where the next window starts. Then a mistake in one window can derail the next, and a few heuristics keep that from happening (Section 3.8 and Section 4.5).

**Method.** The procedure in Section 4.5 has these parts:

- Beam search with 5 beams, scored by log probability, which reduces repetition loops.
- Temperature fallback. Start at temperature 0. Raise it by 0.2, up to 1.0, when the average log probability of the generated tokens is below $-1$, or when the text has a gzip compression rate above 2.4.
- Previous-text conditioning when the applied temperature is below 0.5.
- A no-speech decision that combines the `<|nospeech|>` probability threshold 0.6 with the log probability threshold $-1$.
- A constraint that the initial timestamp is between 0.0 and 1.0 s, so the model can't skip the first words.

The compression test is a cheap detector of a looping decode. Repeated text compresses well, so its ratio is high. The released code computes the ratio as the byte length divided by the `zlib`-compressed byte length.

**Tiny example, computed by `scripts/fallback_example.py`.**

| Text | Bytes | Ratio | Fall back? |
|---|---|---|---|
| A normal 68-character sentence | 68 | 0.971 | No |
| "Thank you. " repeated 20 times | 220 | 10.000 | Yes |
| The same phrase repeated 4 times | 44 | 2.000 | No |
| A Korean sentence of 27 characters | 67 | 0.859 | No |

The Korean row shows why the ratio is computed on bytes. The text has 27 characters and 67 bytes. The schedule is `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`, six temperatures. Building it with floating-point steps is easy to get wrong. The script shows that `np.arange(0, 1.0, 0.2)` has only five values, and that $0.2 \cdot 3$ is `0.6000000000000001`.

Table 7 reports how much each heuristic helps. The average WER over seven long-form datasets is 11.0 with greedy decoding only, 10.6 after beam search (a 3.6% relative drop), 10.2 after adding voice activity detection, and 10.0 after previous-text conditioning, a 9.1% relative drop in all. The temperature fallback row has the same average as the beam search row, 10.6, so on average it changed nothing in this table. The paper calls these heuristics "a workaround for the noisy predictions of the model" (Section 4.5).

Exercise 4 asks you to write the fallback logic.

<!-- exercises -->

## Pitfalls

- **Treating the paper like a typical architecture paper.** There's no new architecture. The results come from the data, the format, and the decoding procedure. Reproducing the model from the paper alone isn't possible, because details such as the vocabulary size, the FFT size, and the Mel scale are in the released code.
- **Feeding anything other than a 30 s window.** The encoder expects `(80, 3000)` and a fixed 1,500-position positional embedding. Shorter audio has to be padded with silence, and longer audio has to be chunked. Padding and chunking choices change the transcript.
- **Off-by-one frames.** A centered STFT of 30 s gives 3,001 frames, not 3,000. Skipping the centering gives 2,998. Either mistake breaks the shape that the encoder checks.
- **Reading the log-Mel scaling as a fixed constant.** The paper says the input is scaled to between $-1$ and $1$ across the dataset. In the released code, the floor depends on each clip's maximum, so quiet clips get a different offset than loud ones.
- **Confusing the language token with the output language.** `<|es|>` with `<|translate|>` means Spanish speech and English text. `<|transcribe|>` keeps the language.
- **Training on the previous context.** The loss is masked over the previous-text tokens, and it applies to all other tokens (Section 2.3). If the loss covers the previous text, the model learns to predict text it only sees as context.
- **Using a 10 ms timestamp grid.** The encoder's positions are 20 ms apart, and the paper quantizes times to 20 ms.
- **Comparing WER without the normalizer.** The paper normalizes text before computing WER, so that formatting differences don't count as errors, and warns that this process can overfit to Whisper's output style (Section 3.2, Section 4.4, Appendix C). Numbers computed without the normalizer aren't comparable with the paper's tables.
- **Expecting fallback to rescue every failure.** In Table 7, the temperature fallback row doesn't improve the average WER over beam search. The heuristics reduce certain failure modes, such as repetition loops, and they don't fix a bad transcript in general.
- **Counting parameters from Table 1 alone.** The token embedding is a large share of a small model. A model with a different vocabulary has a different parameter count at the same layers and width.

## Further reading

- [openai/whisper](https://github.com/openai/whisper) is the official code and model release. `whisper/audio.py`, `whisper/tokenizer.py`, and `whisper/transcribe.py` define the front end, the special tokens, and the fallback loop that this page follows.
- [Introducing Whisper](https://openai.com/index/whisper/) is the authors' announcement post.
- [wav2vec 2.0](/wav2vec2/) is the self-supervised alternative that the paper compares against.
- [Hugging Face's Whisper fine-tuning guide](https://huggingface.co/blog/fine-tune-whisper), by Sanchit Gandhi, shows how to fine-tune the models on a new language.
