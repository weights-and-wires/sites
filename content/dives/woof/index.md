---
id: woof
title: "Underdog Woof: scoring, normalizing, and shrinking a voice"
summary: "Underdog Woof is a pair of English text-to-speech models, 31.7 MB with one voice and 40.6 MB with two, that the post says match a 41.5 GB model on word errors. The post reports results and does not explain training, so this page teaches only what you can check: how word error rate is scored, how text normalization turns numbers into speech, how int8 storage sets the file size, and how the released runtime generates audio frame by frame."
source:
  title: "Woof speaks in 31.7 MB"
  author: "The Underdog team"
  site: "Underdog AI"
  date: "2026-10-26"
  url: "https://underdog.ai/woof"
topics: [tts, evaluation, quantization]
est_minutes: 30
prereqs:
  - {id: transformer, reason: "The runtime's speech model is a causal transformer with a key-value cache, so you need to know what attention and a block look like."}
  - {id: whisper, reason: "The post scores audio by transcribing it with speech recognizers, and its everyday sets use Whisper large-v3."}
  - {id: llm-int8, reason: "It shows how absmax int8 quantization with a scale per row works, which is the storage idea behind Woof's file size."}
  - {id: flow-matching, reason: "The released code calls its per-frame head a one-step flow head, so the vocabulary of flows and noise helps."}
background: [quantization, kv-cache, rmsnorm, swiglu, autoregressive-decoding, adaln-conditioning]
---

<div class="callout callout-note"><span class="callout-title">What this page can and can't cover</span><p>The post and the released code do not disclose how Woof was trained or what data recipe it used, and this page doesn't speculate about either. The post reports results: size, word error rate, and speed. Everything about the architecture below comes from the released <code>runtime.py</code> in the <a href="https://huggingface.co/ConwayResearch/Underdog-Woof">Hugging Face repository</a>, and each such claim says "according to the released runtime".</p></div>

## The problem

Most open text-to-speech models are large. The post lists weight files from 327 MB (Kokoro) up to 41.5 GB (Fish Audio Drama 3, 35.9 billion parameters, stored mostly in 8 bits). A phone agent or an app that speaks on the device can't ship a 41.5 GB file.

Small models have a second problem: written text is not spoken text. A phone number such as `020 7946 0018` is a string of digits, but a person reads it as "zero two zero, seven nine four six, zero zero one eight". The post says small voice models "trip on these" cases: they read a phone number as one enormous figure, or lose half of a code.

The post's answer has three visible parts: a rule-based text normalizer in front of the model, weights stored in int8, and a model of 30.7 million parameters (one voice) or 38.2 million (two voices). The post then reports that the two-voice model gets 1.42% of words wrong on a hard set of 500 sentences, against 1.89% for Drama 3.

The post doesn't say why a model this small can do that. This page doesn't either. It teaches the pieces the post does show.

## Key ideas

### How the post scores a voice

**Intuition.** You can't compare two audio clips word for word, so the post measures a round trip. The model reads a script, a speech recognizer writes down what it heard, and you count how many words differ from the script. A better voice produces audio the recognizer transcribes correctly more often.

**Math.** Align the reference words $r$ with the transcript words $h$ using the smallest number of edits. Count substitutions $S$, deletions $D$, and insertions $I$, and divide by the reference length $N$:

$$
\mathrm{WER} = \frac{S + D + I}{N}
$$

Worked example. The script "call zero two zero now" has $N = 5$ words. A recognizer that hears "call zero two oh now" makes one substitution, so $\mathrm{WER} = 1/5 = 20\%$. A single wrong word is expensive in a short, digit-heavy sentence.

Test sets pool errors over all words. The post reports 90 errors in 4,377 words for one-voice Woof on the hard set, which is $90/4377 = 2.06\%$ (our arithmetic). The widget below runs the alignment for any script and transcript you type.

<div class="widget" data-widget="wer-aligner"></div>

**What the post reports.** All of the following come from the post unless a line says "our arithmetic".

| Set | Model | Result |
| --- | --- | --- |
| 500 hard sentences (numbers, dates, prices, codes, URLs) | Woof, one voice | 2.06% (90 in 4,377 words) |
| The same set | Woof, two voices, better voice | 1.42% |
| The same set | Fish Audio Drama 3 | 1.89% |
| 200 everyday sentences (LibriTTS-R test set, Whisper large-v3) | Woof, one voice | 2.05% (66 in 3,217 words) |
| 30 expressive lines (Whisper large-v3) | Woof, one voice | 0.91% (3 in 329 words) |

The post's "25% fewer" claim for two-voice Woof against Drama 3 matches $1 - 1.42/1.89 = 24.9\%$ (our arithmetic).

**Why a numbers-heavy set is hard.** Two reasons show in the post. First, a number has many correct written forms, so for the hard set the post says an LLM (gemma-4-31b-it) counts errors against the source text after qwen3.8-omni-flash transcribes each clip, "because a number like 14/03 can be written down correctly in more than one way". A plain string match would count "14/03" against "the fourteenth of March" as a miss. Second, a digit string has little redundancy, so one wrong digit changes the answer, as in the example above.

**Read the numbers with care.** The post reports one result per sentence and doesn't give confidence intervals. With 62 errors in 4,379 words, a binomial approximation (our arithmetic, assuming independent words) gives roughly 1.42% plus or minus 0.35 points. That is a rough guide, not a result from the post. Scores also depend on which recognizer and which counting LLM you use.

### Text normalization

**Intuition.** Normalization rewrites text the way people say it before any audio model sees it. The post shows these rewrites, each from its own list of examples: `14/03` becomes "the fourteenth of March", `$1,249.50` becomes "one thousand two hundred and forty nine dollars fifty cents", and `XK-4471` becomes "X K four four seven one". The post says the normalizer is rule-based and uses only the Python standard library.

**What a rule looks like.** Each rule is a pattern plus a function. Match a pattern such as `$` followed by digits, then spell out the number. Order matters: the price rule must run before a bare-digit rule, or it would split `$1,249.50` into pieces.

The post doesn't publish the rules, and we haven't read the released `normalize.py`. The cell in the next section is our own small normalizer that reproduces the post's examples, and it isn't Woof's.

### int8 storage and the size arithmetic

**Intuition.** File size is parameters times bytes per parameter. A weight stored as a 32-bit float takes 4 bytes. Stored as an 8-bit integer plus a shared scale, it takes about 1 byte.

**Math.** To store a row of weights $w$ in int8, choose a scale from the largest absolute value, then round:

$$
s = \frac{\max_j |w_j|}{127}, \qquad q_j = \operatorname{round}\!\left(\frac{w_j}{s}\right), \qquad \hat w_j = q_j \, s
$$

Worked example for the row $w = [0.9, -0.3, 0.05]$. The scale is $s = 0.9/127 = 0.00709$, the integers are $q = [127, -42, 7]$, and the recovered row is $[0.9, -0.2976, 0.0496]$. Each weight is off by at most $s/2$, and the typical error is about $s/4$.

"According to the released runtime," the speech model's linear layers are stored as int8 values `q` with one scale `s` per output row, and the loader rebuilds float32 weights as `q * s[:, None]`. The runtime only reads the scales. How the exporter chose them isn't in that file, so the absmax rule above is our assumption. The runtime's decoder uses either float tensors or int8 with a scale per group of values.

The widget below shows why a scale per row matters. Give one row large weights and switch between one scale per row and one per tensor.

<div class="widget" data-widget="int8-size"></div>

**Size arithmetic (our arithmetic on the post's numbers).**

- One voice: 30.7 million parameters at 1 byte is 30.7 MB. The post reports 31.7 MB, a gap of 1.0 MB.
- Two voices: 38.2 million parameters at 1 byte is 38.2 MB. The post reports 40.6 MB, a gap of 2.4 MB.
- At fp32, the same parameter counts would take 122.8 MB and 152.8 MB.
- The gaps are consistent with scales, biases, and tensors kept in higher precision, but the post doesn't break the files down, so we can't say how the gap splits.
- Ratios: $41{,}500\,\text{MB} / 40.6\,\text{MB} = 1022$ and $35.9\text{B}/30.7\text{M} = 1169$ match the post's 1,022x and 1,169x. For the one-voice file, $41{,}500/31.7 = 1309$ against the post's 1,308x, a difference that probably comes from unrounded sizes.

Drama 3 shows the same arithmetic at scale: the post says it stores most weights in 8 bits, and 35.9 billion parameters at about 1 byte is 35.9 GB, against the 41.5 GB it reports.

### The inference loop, according to the released runtime

This section reads `runtime.py`, the NumPy inference file in the Hugging Face repository. The post doesn't describe these steps. Sizes such as layer count, width, and frame rate come from the weights file, so they aren't stated here.

1. **Characters in.** Text becomes ids per character. The vocabulary has 92 listed characters (letters, digits, punctuation, a few symbols), with id 1 for a start marker and id 2 for any unknown character.
2. **One prefix pass.** The runtime concatenates the voice embedding vectors, the text embeddings, and a BOS vector, then runs the whole prefix through the transformer in one pass. That pass fills a key-value cache.
3. **A causal transformer.** According to the released runtime, each block applies RMS normalization, a fused query-key-value projection with rotary position embeddings, causal attention, an output projection, and a SwiGLU feed-forward layer. Because of the cache, generating frame $t$ attends to stored keys and values and costs one new position, not a full recomputation.
4. **One continuous frame per step.** The last hidden state is turned into a conditioning vector. A small head then produces one latent frame. The code draws noise $\epsilon \sim \mathcal{N}(0, I)$ with 32 values per frame, multiplies it by a temperature (default 0.5), and returns

   $$
   z = T\epsilon + \mathrm{head}(s,\, T\epsilon)
   $$

   where $s$ is the conditioning vector. The head is a stack of small MLP blocks whose shift, scale, and gate come from $s$. The code's docstring calls it a "one-step flow head", and it runs once per frame.
5. **Stop or continue.** The transformer also outputs an end-of-speech logit $e$ each step. The loop stops when $e$ exceeds $\log\frac{p}{1-p}$ for $p = 0.9$, which is the same as the sigmoid of $e$ exceeding 0.9, and only after the frame index passes 3. That check means at least four frames are generated. A frame cap also applies, which the weights file sets.
6. **Feed the frame back.** The new frame $z$ goes through a linear layer and becomes the next input to the transformer.
7. **Decode.** The runtime undoes a normalization of the latents (`lat * std + mean`) and passes them to a decoder: an input layer, a causal transformer with a limited attention window, and a stack of convolutions with upsampling that ends in one channel of 24 kHz audio. The post says this decoder follows the design of Mimi by Kyutai.

A seed makes the output repeat: the runtime draws noise from a generator built to match PyTorch's, and the post says a seed gives the same audio for the same text.

The widget below is a toy of this control flow, not Woof. The "latent" is one number and the stop signal is hand-made. Move the temperature and the threshold and watch when the loop stops.

<div class="widget" data-widget="toy-loop"></div>

**What `--steps` means.** The post lists `--steps` 16, 8, and 4 at 18x, 24x, and 29x real time and doesn't say what a step is. The word doesn't appear in the NumPy `runtime.py`, whose head runs once per frame. We don't know what `--steps` controls, so we don't claim a meaning. Our arithmetic only restates the speeds: 13.2 s of speech at 18x takes 0.73 s, at 29x about 0.46 s.

## Rebuild it

The four cells below are our reconstruction of the checkable pieces. None of them is Woof.

### A tiny normalizer

Each rule is a regular expression plus a function. Add a rule, for example for percentages, and run again.

<!-- cell: normalize -->

### Word error rate with alignment

The cell aligns each reference and transcript, counts errors, and pools them the way the post does. The pooled rate and the average of per-sentence rates differ slightly.

<!-- cell: wer -->

### Per-row int8 quantization

The cell quantizes a 64 by 64 matrix with one noisy row, compares one scale per row with one per tensor, and reports bytes. The scale overhead is why the ratio is 3.76x and not 4x.

<!-- cell: quantize -->

### A toy generation loop

This cell mirrors the stop test and the feedback in the runtime with a one-number "latent". Change `target` or the seed and rerun.

<!-- cell: toy_loop -->

## What the post doesn't say

- **Training and data.** The post names LibriTTS-R and Emilia-YODAS as training data, and says nothing about the recipe: model design choices, losses, schedules, or why the model is small.
- **What `--steps` does.** See above.
- **Naturalness.** The post reports word errors, not listener ratings. A low WER means the words are recognizable, not that the voice sounds good.
- **One seed, one scorer.** The post's results use one generation per sentence and specific recognizers and an LLM counter. The hard set also is not a general benchmark: it is built around numbers.
- **The size comparison.** "1,000x smaller" compares against one very large model. The post's own table includes models between 327 MB and 15.1 GB.
- **The license.** The post says the weights are open for non-commercial use under CC BY-NC 4.0. Check the license on the Hugging Face page before you use them.

## Further reading

- [Woof speaks in 31.7 MB](https://underdog.ai/woof), the post this page is based on.
- [Underdog Woof on Hugging Face](https://huggingface.co/ConwayResearch/Underdog-Woof), including `underdog_woof/runtime.py`, the file the inference-loop section reads.
- [Whisper](/whisper/), the recognizer the post uses for its everyday sets.
- [LLM.int8()](/llm-int8/), for absmax quantization with a scale per row.
- [Transformer](/transformer/) and [Flow matching](/flow-matching/), for the building blocks the runtime's vocabulary borrows.
