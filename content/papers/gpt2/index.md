---
id: "gpt2"
title: "Language Models are Unsupervised Multitask Learners"
authors: "Radford et al."
year: 2019
venue: "OpenAI technical report"
arxiv: null
url: "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf"
track: nlp-llm
tier: core
est_hours: 6
status: full
summary: "The paper trains one decoder-only Transformer on next-token prediction over WebText, 40 GB of text from pages that people linked on Reddit, and then tests it on many tasks with no task-specific training. A task is written as ordinary text, such as an article followed by the cue TL;DR:, and the model's continuation is the answer. A byte-level version of byte-pair encoding lets the model read and score any string, and moving layer normalization to the input of each sub-block lets the stack grow to 48 layers and 1.5 billion parameters, where zero-shot results improve steadily with size."
related_course: "ml: GPT from scratch"
prereqs:
  - id: transformer
    reason: "GPT-2 is a stack of Transformer decoder blocks, so it reuses multi-head causal self-attention and residual connections and changes only where layer normalization sits."
  - id: bpe
    reason: "GPT-2 builds its vocabulary with BPE merges learned from pair frequencies, and changes the starting symbols from characters to bytes."
background: [autoregressive-language-modeling, perplexity]
course_links:
  - title: "Training a language model: next-token loss, schedules, and clipping"
    url: "https://ml.ifkash.dev/lessons/068-next-token-loss-and-safe-steps"
  - title: "Train a tiny transformer to reverse digits"
    url: "https://ml.ifkash.dev/lessons/070p-project-train-a-tiny-transformer"
self_check:
  - q: 'A language model assigns $p(\text{the}) = 0.5$, $p(\text{cat} \mid \text{the}) = 0.4$, and $p(\text{sat} \mid \text{the cat}) = 0.5$. What is the probability of the sequence, and what is its perplexity?'
    a: 'The sequence probability is $0.5 \times 0.4 \times 0.5 = 0.1$, by the chain rule in Equation 1. Perplexity is $\exp$ of the average negative log-probability, which is $0.1^{-1/3} \approx 2.154$. A model that picked uniformly among $V$ tokens would have perplexity $V$.'
    see: perplexity
  - q: 'Why can a byte-level BPE vocabulary encode any string with no unknown token, and how many base symbols does it start from?'
    a: 'Every string is a sequence of UTF-8 bytes, and the 256 possible byte values are the base symbols. Merges only combine symbols that already exist, so any text decomposes into known symbols. The paper contrasts this with a code-point base vocabulary of over 130,000 (Section 2.2).'
    see: bpe
  - q: 'In `x + Sublayer(LayerNorm(x))`, what does the block return when every weight of the sub-layer is zero, and what would `LayerNorm(x + Sublayer(x))` return?'
    a: 'The pre-LN form returns $x$ unchanged, because the sub-layer adds zero to the residual stream. The post-LN form returns the normalized $x$, so a zero-weight block still rescales the stream. Section 2.3 moves layer normalization to the pre-LN position.'
    see: transformer
---

## The problem

By 2018, the best language-understanding systems followed one recipe: collect a labeled dataset for a task, fine-tune a model on it, and test on held-out examples from the same distribution. The paper calls the result "narrow experts" (Section 1). Each task needs its own dataset, and a model that does well on one distribution often fails on a slightly different one.

Multitask training looked like the fix, but the paper points out that it scales the same way. The two most ambitious efforts trained on 10 and 17 (dataset, objective) pairs. Viewed as meta-learning, each pair is one training example, and learners usually need hundreds to thousands of examples before they generalize (Section 1). Writing hundreds of datasets and objectives by hand is not a plan.

Consider a concrete failure. You have a model fine-tuned on news summaries. You ask it to translate a French sentence, and it has no way to know that is what you want, because its architecture and its training signal encode only one task. You need a new dataset, a new head, or at least a new fine-tuning run.

The paper takes a different route. The text of the web already contains examples of tasks written in natural language. Table 1 shows French and English translations that occur in the training set as quoted sentences, such as "Je ne suis pas un imbecile [I'm not a fool]". If a language model is large enough to predict that text well, the paper's conjecture is that it has to learn the task in order to predict the text (Section 2). Section 3 tests the idea on eight language-modeling datasets, reading comprehension, summarization, translation, and question answering, with no parameter update for any of them.

Section 6 is candid about the result. Reading comprehension is competitive with supervised baselines, summarization is "rudimentary", and the zero-shot performance is "still far from use-able". The paper's claim is that the trend with model size points somewhere, not that GPT-2 solves the tasks.

## Key ideas

The sections below use the paper's names where it gives one: $x$ for an example, $s_1, \dots, s_n$ for its symbols, $N$ for the number of residual layers, and $d_{\text{model}}$ for the model width. The paper has one numbered equation, so most of the math here is the paper's prose written out, and the page says so where it does that. Every number comes from a script in `scripts/`. Run any script from the paper folder to reproduce the output.

### A language model as a multitask learner

**Intuition.** A language model predicts each symbol from the symbols before it. If a document contains a question followed by its answer, predicting the answer well means answering the question. The task is part of the text, so the model needs no separate task input.

**Math.** Section 2 factorizes the probability of an example $x = (s_1, \dots, s_n)$ into next-symbol conditionals (Equation 1):

$$
p(x) = \prod_{i=1}^{n} p(s_i \mid s_1, \dots, s_{i-1})
$$

The paper prints $p(s_n \mid s_1, \dots, s_{n-1})$ inside the product. The index inside the product must be $i$, so this page writes it that way. The factorization allows tractable sampling and also gives any conditional, such as $p(s_{n-k}, \dots, s_n \mid s_1, \dots, s_{n-k-1})$.

A general system should model $p(\text{output} \mid \text{input}, \text{task})$. The paper writes the task into the sequence: a translation example is the sequence (translate to french, english text, french text), and a reading-comprehension example is (answer the question, document, question, answer) (Section 2). Training on the whole sequence and training only on the answer share a global minimum, because the supervised objective is the unsupervised objective evaluated on a subset of the sequence. The paper calls this a "toy-ish setup" and notes that the practical question is whether the objective can be optimized to convergence, and that learning in this setup is much slower than with explicit supervision.

Perplexity is the reporting metric. For a dataset, Section 3.1 divides the total log-probability by the number of canonical units, which is a word, a character, or a byte, and reports the exponential or a scaled version. GPT-2 predicts BPE tokens, but it can still report per-word perplexity or bits per byte, because the log-probability of a string is the same whichever way you slice it.

**Tiny example, computed by `scripts/lm_factorization.py`.** Let a toy model assign $p(\text{the}) = 0.5$, $p(\text{cat} \mid \text{the}) = 0.4$, and $p(\text{sat} \mid \text{the cat}) = 0.5$.

| Quantity | Value |
|---|---|
| $p(x)$, the product of the three | 0.1000 |
| Per-token negative log-probability (nats) | 0.6931, 0.9163, 0.6931 |
| Average negative log-probability | 0.7675 |
| Perplexity, $\exp(0.7675)$ | 2.1544 |
| Check: $p(x)^{-1/3}$ | 2.1544 |
| Uniform model over $V = 50{,}257$ tokens | 50,257 |
| Bits per byte, for the 11 bytes of "the cat sat" | 0.3020 |

Perplexity reads as an effective branching factor: the toy model is as unsure as a fair choice among about 2.15 tokens at each step. The uniform model, which knows nothing, has a perplexity equal to its vocabulary size.

The script also splits a five-token sequence into a three-token prompt and a two-token answer, with log-probabilities $-1.2, -0.7, -2.0, -0.3, -0.5$. The full-sequence log-probability is $-4.7$. The prompt contributes $-3.9$ and the answer given the prompt contributes $-0.8$, and $-3.9 + (-0.8) = -4.7$. A supervised loss on the answer alone is one term of the language-model loss.

A sequence of $T$ tokens gives $T - 1$ predictions in one forward pass, because position $t$ predicts the token at $t + 1$ and the last position has no target. Exercise 1 asks you to compute the loss and perplexity from logits, with the shift done correctly.

### Zero-shot task transfer

**Intuition.** If tasks are text, you invoke a task by writing the start of a document that looks like that task. The model continues the document.

**Math.** The paper has no new formula here. Each task in Section 3 is a prompt format and a decoding rule:

| Task | Prompt | Decoding | Section |
|---|---|---|---|
| Reading comprehension (CoQA) | Document, the conversation so far, then `A:` | Greedy | 3.5 |
| Summarization (CNN and Daily Mail) | Article followed by `TL;DR:` | Top-$k$ sampling with $k = 2$, 100 tokens; the first 3 sentences are the summary | 3.6 |
| Translation (WMT-14) | Example pairs `english sentence = french sentence`, then `english sentence =` | Greedy; the first generated sentence | 3.7 |
| Question answering (Natural Questions) | Example question-answer pairs to set the short-answer style | Greedy | 3.8 |

Language-modeling benchmarks need no prompt. Section 3.1 evaluates them by computing the probability of the test text directly, and Section 3.2 and 3.4 pick the answer to a multiple-choice or ambiguity task by the probability the model assigns to each option.

What the paper reports, with the table or section where each result appears:

- **Language modeling (Table 3).** The 1542M model sets a new state of the art on 7 of 8 datasets. The one it misses is the One Billion Word benchmark, which the paper blames on sentence-level shuffling that removes long-range structure. On LAMBADA, perplexity falls from 35.13 for the 117M model to 8.63 for the 1542M model, against a previous best of 99.8. On WikiText-2 it falls from 29.41 to 18.34.
- **Reading comprehension (Section 3.5).** 55 F1 on CoQA, which matches or exceeds 3 of 4 baselines that trained on more than 127,000 labeled question-answer pairs. The paper finds that answers often follow a simple retrieval heuristic, such as answering a "who" question with a name from the document.
- **Summarization (Section 3.6, Table 4).** An average ROUGE of 21.40 with the `TL;DR:` hint and 15.03 without it. The 6.4-point drop is the paper's evidence that a text cue invokes a task. The score is only slightly above 20.98 for three random sentences from the article.
- **Translation (Section 3.7).** 5 BLEU from English to French and 11.5 BLEU from French to English on WMT-14. The paper had filtered non-English pages out of WebText, and a language detector found only 10 MB of French.
- **Question answering (Section 3.8).** 4.1 percent exact match on Natural Questions, 5.3 times the smallest model's rate, and 63.1 percent accuracy on the 1 percent of questions where it is most confident.

The contamination check in Section 4 matters for how to read these numbers. The paper builds Bloom filters over 8-grams of the training set and finds the test sets of common benchmarks overlap with WebText between 1 and 6 percent, on average 3.2 percent, which is no more than they overlap with their own training splits.

### Byte-level BPE

**Intuition.** A word-level vocabulary can't represent a string it hasn't seen. A character-level vocabulary over Unicode is too big. Bytes give a base vocabulary of 256 symbols that can spell any string in UTF-8. Plain BPE over bytes wastes merges on punctuation variants of the same word, so the paper stops merges from crossing character categories.

**Math.** Section 2.2 gives the argument in prose. A code-point base vocabulary is over 130,000 symbols before any merge, which is large next to the 32,000 to 64,000 token vocabularies common with BPE. A byte-level version needs 256. The paper says that applying BPE directly to bytes gives suboptimal merges, because BPE is a greedy frequency heuristic and "dog", "dog.", "dog!", and "dog?" all occur often. The fix is to prevent merges across character categories for any byte sequence, with an exception for spaces that "significantly improves the compression efficiency" with little fragmentation of words. The final vocabulary has 50,257 entries (Section 2.3).

The paper does not give the rule as a formula. The released encoder implements it as a regular expression that splits text into pieces before BPE runs, and the merge procedure from the [BPE paper](/bpe/) then runs within each piece. This page and Exercise 4 use a simplified form of that split: maximal runs of letters, of digits, or of other non-space characters, where a single space directly before a run joins that run. The released pattern also handles contractions as a special case. The simplification here doesn't, which is why "don't" splits into three pieces below.

**Tiny example, computed by `scripts/byte_bpe.py`.** The split of `dog dog. dog! dog?` is `dog`, ` dog`, `.`, ` dog`, `!`, ` dog`, `?`. The punctuation never joins the word, so ` dog` is one reusable piece. Other splits:

| Input | Pieces |
|---|---|
| `I have 42 dogs, ok?` | `I`, ` have`, ` 42`, ` dogs`, `,`, ` ok`, `?` |
| `a  b` (two spaces) | `a`, ` `, ` b` |

Only the last space joins the next run. Now the byte counts, which show why the base vocabulary is 256 and not smaller:

| String | Code points | UTF-8 bytes |
|---|---|---|
| `dog` | 3 | 3 |
| `café` | 4 | 5 (`99, 97, 102, 195, 169`) |
| `世界` | 2 | 6 |
| a grinning-face emoji | 1 | 4 |

Every string, however unusual, is a list of values from 0 to 255, so no `<UNK>` token is needed. Section 3.1 notes that `<UNK>` occurs 26 times in 40 billion bytes of WebText, and that the model's ability to score any string lets it be tested on any benchmark with no tokenization step.

Exercise 4 asks you to implement the split and the byte conversion.

### The pre-LN decoder

**Intuition.** GPT-2 is the Transformer decoder from the [previous paper](/transformer/) with the encoder and the cross-attention removed. What changes is where normalization sits and how the weights start. Putting layer normalization at the input of each sub-block leaves the residual stream untouched, so many layers can add to it. Scaling the residual branches at initialization stops that stream from growing with depth.

**Math.** Section 2.3 lists the modifications to the original GPT, which itself follows the Transformer:

- Layer normalization moves to the input of each sub-block, like a pre-activation residual network, so a block is $x + \operatorname{Sublayer}(\operatorname{LayerNorm}(x))$ for attention and then for the feed-forward network.
- An extra layer normalization follows the final self-attention block.
- Weights of residual layers are scaled at initialization by $1/\sqrt{N}$, where $N$ is the number of residual layers.
- The vocabulary has 50,257 tokens, the context is 1024 tokens, and the batch size is 512.

Table 2 lists four sizes by layers and $d_{\text{model}}$: 12 layers at 768 (117M), 24 at 1024 (345M), 36 at 1280 (762M), and 48 at 1600 (1542M). The text doesn't give the head count, the feed-forward width, the activation, or whether positions are learned. The standard choices from the original GPT and the released code are a feed-forward width of $4\,d_{\text{model}}$, the GELU activation, learned position embeddings, biases on every linear layer, and one embedding matrix shared by the input and the output projection. The exercise and the count below assume them. The paper also doesn't say how many residual layers $N$ counts. This page reads it as $2L$ for $L$ blocks, one residual layer for attention and one for the feed-forward network in each.

The paper follows the original Transformer's attention, so you can reuse Exercise 4 of the Transformer page for the attention layer itself. Exercise 2 on this page supplies it and asks for the part that changes.

**Tiny example, computed by `scripts/param_count.py` and `scripts/residual_scale.py`.** First, parameter counts under the assumptions above, with token embeddings of $50{,}257 \times d_{\text{model}}$ and $1024 \times d_{\text{model}}$ positions. One block holds $12\,d_{\text{model}}^2 + 13\,d_{\text{model}}$ weights:

| Table 2 name | Layers | $d_{\text{model}}$ | Block parameters | Total parameters |
|---|---|---|---|---|
| 117M | 12 | 768 | 7,087,872 | 124,439,808 |
| 345M | 24 | 1024 | 12,596,224 | 354,823,168 |
| 762M | 36 | 1280 | 19,677,440 | 774,030,080 |
| 1542M | 48 | 1600 | 30,740,800 | 1,557,611,200 |

The counts differ from the names in Table 2. The smallest model comes out at 124.4M and the largest at 1.558B. The widely used `gpt2` checkpoint on Hugging Face, which follows these choices, has 124M parameters. Table 2's labels appear to undercount. Treat the sizes as names, and compute from the architecture when the number matters. In the smallest model, the token and position embeddings hold 38,597,376 and 786,432 parameters, 31.6 percent of the total. Without weight tying, the total would be 163,037,184.

Now the initialization. Suppose each residual layer adds a branch output with unit variance to a stream that starts with unit variance. After $N$ unscaled layers the variance is $1 + N$. With each branch scaled by $1/\sqrt{N}$, its variance drops to $1/N$, and the total is $1 + N \cdot (1/N) = 2$ whatever the depth. A seeded simulation with 4,096 independent values gives the following:

| Blocks | $N$ | Variance, no scaling | Variance, scaled by $1/\sqrt{N}$ |
|---|---|---|---|
| 12 | 24 | 24.55 (theory 25) | 1.99 (theory 2) |
| 48 | 96 | 92.10 (theory 97) | 1.91 (theory 2) |

The simulation uses independent random branches, which is a stand-in. Real branches correlate with the stream, so treat the table as the intuition and not a measurement of a trained model.

Last, the pre-LN identity property. Let the stream be $x = [100, 0, 0, 0]$ and let every sub-layer weight be zero. The pre-LN block returns `[100, 0, 0, 0]`. The post-LN block, $\operatorname{LayerNorm}(x + \operatorname{Sublayer}(x))$, returns `[1.7321, -0.5774, -0.5774, -0.5774]`. At initialization, with small branch weights, the whole stack is close to the identity on the residual stream, and each layer then learns what to add. The Transformer page's block example compares the two placements in more detail.

Exercise 2 builds the pre-LN block, the full forward pass with tied embeddings and a final layer normalization, and the residual scale.

### Sampling with temperature and top-k

**Intuition.** A language model gives a distribution over the next token. Greedy decoding picks the most likely token every time, and tends to repeat. Sampling from the full distribution picks unlikely tokens too often. Top-$k$ sampling keeps only the $k$ most likely tokens and renormalizes. Temperature reshapes the distribution before that.

**Math.** Section 3.6 uses top-$k$ sampling with $k = 2$ for summarization, citing Fan et al. (2018), and says it "reduces repetition and encourages more abstractive summaries than greedy decoding". The paper defines neither temperature nor top-$k$ by formula. The standard definitions, which the exercise follows, are as follows. For logits $z \in \mathbb{R}^V$ and temperature $T > 0$:

$$
p_i = \frac{\exp(z_i / T)}{\sum_{j \in S} \exp(z_j / T)} \quad \text{for } i \in S, \qquad p_i = 0 \text{ otherwise}
$$

Here $S$ is the set of the $k$ indices with the largest $z_i / T$, or all indices when no top-$k$ is used. Dividing by a positive $T$ doesn't change which indices are largest, so temperature and top-$k$ can apply in either order for the choice of $S$.

**Tiny example, computed by `scripts/sampling_example.py`.** Take logits $z = [2, 1, 0.5, -1]$:

| Setting | Probabilities |
|---|---|
| Softmax, $T = 1$ | `[0.6095, 0.2242, 0.1360, 0.0303]` |
| Softmax, $T = 0.5$ | `[0.8420, 0.1140, 0.0419, 0.0021]` |
| Softmax, $T = 2$ | `[0.4344, 0.2635, 0.2052, 0.0969]` |
| Top-2, $T = 1$ | `[0.7311, 0.2689, 0, 0]` |
| Top-1 | `[1, 0, 0, 0]` |
| Top-2, $T = 0.5$ | `[0.8808, 0.1192, 0, 0]` |

Check the top-2 row by hand. The two kept probabilities are $0.6095$ and $0.2242$, and $0.6095 / (0.6095 + 0.2242) = 0.7311$. That value is the same as $\sigma(2 - 1) = \sigma(1)$, because with two tokens the softmax is a sigmoid of the logit gap. Lower temperature lowers the entropy: 0.5381 nats at $T = 0.5$, 1.0144 at $T = 1$, and 1.2648 at $T = 2$. Top-1 is greedy decoding.

Exercise 3 asks you to implement the sampler's probability step.

<!-- exercises -->

## Pitfalls

- **Misaligned targets.** The logits at position $t$ score the token at position $t + 1$. Comparing them with the token at $t$ gives a loss that looks fine, trains, and learns to copy its input. Drop the last row of logits and the first target.
- **Averaging over the wrong count.** A sequence of $T$ tokens has $T - 1$ targets in a single window. Dividing by $T$ shifts every loss slightly, which matters when you compare with a published perplexity.
- **Comparing perplexities across tokenizers.** Table 3 reports perplexity per word for some datasets and bits per byte or per character for others. A per-token perplexity from one tokenizer isn't comparable with a per-token perplexity from another. Convert to a per-byte quantity first, as in the example above.
- **Reading Table 2 as exact sizes.** The names 117M, 345M, 762M, and 1542M are labels. Counting parameters from the architecture gives 124M, 355M, 774M, and 1.56B.
- **Applying layer normalization in the post-LN position.** The paper's modification moves it to the input of each sub-block and adds one more after the last block. Leaving out the final layer normalization is easy to miss, because the model still trains, with a differently scaled output.
- **Scaling every weight by $1/\sqrt{N}$.** The paper scales the weights of the residual layers, which are the output projections that write into the stream, not every weight in the network.
- **Forgetting that positions are not shared.** The same token at two positions needs different inputs, so the model adds a position embedding to the token embedding. Without it, the causal mask is the only source of order, and the model can't tell a token's position within the visible prefix.
- **Untying the embeddings by accident.** Weight tying means the output logits come from the same matrix as the input embeddings. A separate output matrix adds 38.6 million parameters for the smallest model and changes the behavior.
- **Letting BPE merge across categories.** Without the category rule, the vocabulary spends slots on `dog.`, `dog!`, and `dog?`. A byte-level vocabulary has no unknown-token fallback to hide such waste.
- **Using the wrong decoding for the task.** The paper uses greedy decoding for CoQA, translation, and question answering, and top-$k$ with $k = 2$ for summarization. Swapping them changes the scores. The summarization score of 21.40 also depends on the `TL;DR:` cue, and drops to 15.03 without it.
- **Treating zero-shot scores as a ceiling.** Section 6 says the zero-shot performance is far from useful in practice, and Section 4 reports a small but consistent benefit from train-test overlap. Read the scores as a trend with size.
- **Sampling with a temperature of 0.** The formula divides by $T$. Treat $T \to 0$ as greedy decoding and handle it as a separate branch.

## Further reading

- [openai/gpt-2](https://github.com/openai/gpt-2) is the official code release, with the byte-level encoder and the inference model.
- [Better language models and their implications](https://openai.com/index/better-language-models/) is the authors' announcement post. It explains the staged release of the larger models.
- [nanoGPT](https://github.com/karpathy/nanoGPT), by Andrej Karpathy, is a short, readable reimplementation that reproduces the 124M model.
- [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY), by Andrej Karpathy, is a video that builds a decoder-only Transformer.
- [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE), by Andrej Karpathy, is a video that builds byte-level BPE with the category-splitting pattern.
- [Hierarchical Neural Story Generation](https://arxiv.org/abs/1805.04833), by Fan et al., introduces top-$k$ sampling, which Section 3.6 cites.
- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751), by Holtzman et al., compares greedy, top-$k$, and nucleus sampling.
