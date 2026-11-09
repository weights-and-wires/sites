---
id: "gpt3"
title: "Language Models are Few-Shot Learners"
authors: "Brown et al."
year: 2020
venue: "NeurIPS 2020"
arxiv: "2005.14165"
url: "https://arxiv.org/abs/2005.14165"
track: nlp-llm
tier: core
est_hours: 4
status: full
summary: "The paper scales the GPT-2 decoder-only Transformer to 175 billion parameters, trains it on 300 billion tokens with plain next-token prediction, and tests it with no weight updates. A task is given as a natural-language instruction, one demonstration, or a few demonstrations placed in the prompt, and the model's continuation is the answer. The gap between zero-shot, one-shot, and few-shot results grows with model size, which the authors read as evidence that larger models are better at learning from context."
related_course: "ml: LLM pretraining"
prereqs:
  - id: gpt2
    reason: "GPT-3 keeps GPT-2's decoder-only, pre-layer-norm architecture and its idea that a task can be written as text for the model to continue, and changes the scale and the way the tasks are evaluated."
background: [in-context-learning, few-shot-prompting]
course_links:
  - title: "Scaling laws and the compute budget"
    url: "https://ml.ifkash.dev/lessons/055-scaling-laws"
  - title: "Training a language model: next-token loss, schedules, and clipping"
    url: "https://ml.ifkash.dev/lessons/052-next-token-loss-and-safe-steps"
self_check:
  - q: 'Section 2.1 says GPT-3 uses "the same model and architecture as GPT-2". Name the changes the paper lists, in the architecture and in the context length.'
    a: 'The paper keeps the modified initialization, pre-normalization, and tokenization, and adds alternating dense and locally banded sparse attention layers, like the Sparse Transformer. The context window is $n_{\text{ctx}} = 2048$ tokens for every model, against 1024 for GPT-2. The other changes are scale: 96 layers and $d_{\text{model}} = 12288$ for the largest model.'
    see: gpt2
  - q: 'In the few-shot setting, a model is given $K = 10$ solved examples and then a new question. How many gradient updates does the model take, and what does the paper mean by "in-context learning"?'
    a: 'None. All $K$ examples are text in the prompt, and the model produces the answer in one forward pass. The paper uses "in-context learning" for the inner loop of meta-learning, which occurs within the forward pass on each sequence (Section 1, Figure 1.1 and its footnote 1). The paper leaves open whether the model learns the task from the examples or only recognizes a task it saw during pre-training.'
    see: in-context-learning
  - q: 'Each demonstration takes 60 tokens, and you reserve 60 tokens for the final question and the answer. What is the largest $K$ that fits in a 2048-token context?'
    a: 'The demonstrations get $2048 - 60 = 1988$ tokens, and $\lfloor 1988 / 60 \rfloor = 33$. The paper says 10 to 100 examples typically fit (Section 2), and longer demonstrations push $K$ toward the low end.'
    see: few-shot-prompting
---

## The problem

By 2020 the standard recipe was to pre-train a Transformer on text and then fine-tune it on a labeled dataset for each task. The architecture was task-agnostic, but the data was not. The paper lists three costs (Section 1):

- Every new task needs a new labeled dataset of thousands to hundreds of thousands of examples, and for tasks such as grammar correction or critiquing a short story that data is hard to collect.
- A larger model fine-tuned on a narrow dataset can exploit spurious correlations in it, so a benchmark score "may exaggerate actual performance on the underlying task".
- People don't work this way. A short instruction, or at most a few examples, is enough for a person to start a new language task.

Consider a concrete failure. You have a model fine-tuned on English-to-French translation. You now want it to turn "Q: What is 48 plus 76?" into "124". The model has nothing to condition on except its weights, so you need arithmetic training data and another fine-tuning run. For the next task you need another dataset.

GPT-2 had already shown that a language model can attempt tasks that are written as text, and the [previous page](/gpt2/) covers how. The paper says the results were still "far inferior to fine-tuning". It quotes 4 percent on Natural Questions and 55 F1 on CoQA, more than 35 points behind the state of the art (Section 1). The authors' bet is that capacity is the missing ingredient. Transformer language models had grown from 100 million to 17 billion parameters in a few years, and validation loss follows a smooth trend with scale (the [Kaplan et al. scaling laws](/kaplan/)). So in-context learning might improve with scale too. GPT-3 tests the bet at 175 billion parameters.

## Key ideas

The paper uses these names, and this page follows them: $K$ for the number of demonstrations in the context, $n_{\text{ctx}}$ for the context length, $n_{\text{params}}$ for the parameter count, $n_{\text{layers}}$, $d_{\text{model}}$, $n_{\text{heads}}$, $d_{\text{head}}$, and $d_{\text{ff}}$ for the architecture in Table 2.1. The paper has no numbered equations, so the math on this page is the paper's prose written as formulas, and the page says so where it does that. Every number comes from a script in `scripts/`. Run any script from the paper folder to reproduce the output.

### In-context learning as the inner loop of meta-learning

**Intuition.** The paper describes a two-level process. The outer loop is ordinary pre-training. Across the web, a sequence often contains repeated sub-tasks, such as several translations in a row or a list of question-and-answer pairs, so predicting the next token well rewards picking up the pattern from earlier in the sequence. The inner loop is what happens in the forward pass of a single sequence: the model reads the pattern and continues it. The paper names the inner loop "in-context learning" and the whole structure "meta-learning" (Section 1, Figure 1.1, footnote 1).

**Math.** The paper gives no formula, so the following is this page's notation and not the paper's. Write the context as $c = (I, (x_1, y_1), \dots, (x_K, y_K), x_{K+1})$, where $I$ is an optional instruction, each $(x_k, y_k)$ is a solved example, and $x_{K+1}$ is the new input. The answer $y_{K+1} = (t_1, \dots, t_m)$ comes from the same next-token model as in pre-training:

$$
p_\theta(y_{K+1} \mid c) = \prod_{i=1}^{m} p_\theta(t_i \mid c, t_1, \dots, t_{i-1})
$$

The weights $\theta$ are fixed. The setting changes only $c$. Section 2 defines three settings by the size of $K$:

| Setting | Demonstrations | Instruction |
|---|---|---|
| Zero-shot (0S) | $K = 0$ | Natural-language description only |
| One-shot (1S) | $K = 1$ | A description and one demonstration |
| Few-shot (FS) | typically $10 \le K \le 100$, as many as fit in $n_{\text{ctx}} = 2048$ | Optional |

Fine-tuning is the fourth point on the spectrum, and the paper doesn't use it for GPT-3 (Section 2). The authors present the three settings "not as competing alternatives, but as different problem settings" that trade benchmark performance against the amount of task data.

**Tiny example, computed by `scripts/prompt_budget.py`.** The arithmetic prompts follow Appendix G, where a context is `Q: What is 98 plus 45?` and `A:` on the next line (Figure G.44), with demonstrations separated by blank lines (Section 2.4). The script counts tokens with a toy rule: a run of letters is one token, each digit is one token, each other non-space character is one token, and a run of newlines is one token. GPT-3's byte-pair encoding counts differently, so treat the counts as an illustration.

| Setting | Prompt | Toy tokens |
|---|---|---|
| Zero-shot | the final question only | 13 |
| One-shot | 1 demonstration, then the question | 30 |
| Few-shot, $K = 3$ | 3 demonstrations, then the question | 61 |

A demonstration takes 15 to 17 toy tokens, 16 on average. With 3 tokens reserved for the answer, $\lfloor (2048 - 13 - 3) / 16 \rfloor = 127$ demonstrations fit. A demonstration of 150 tokens, as in a reading-comprehension task, leaves room for 13. The paper's range of 10 to 100 follows from this budget: it depends on the task's length more than on any rule.

Exercise 1 asks you to build the prompt and the evaluation loop.

### Zero-, one-, and few-shot evaluation

**Intuition.** Evaluation is a forward pass per test example, with a prompt drawn from the training set. The paper uses a different decoding rule for each kind of answer: it ranks the options by likelihood when the answers are a fixed set, and it generates when the answer is free text.

**Math.** Section 2.4 specifies the protocol in prose. For each test example, draw $K$ examples at random from the task's training set as conditioning, delimited by one or two newlines. $K$ can be anything from 0 to what fits in $n_{\text{ctx}}$, and when a development set exists, the paper tries a few values of $K$ on it and runs the best value on the test set. The scoring rules are as follows:

- **Multiple choice.** Give $K$ solved examples and then one unanswered context, and compare the model's likelihood of each completion. Most tasks use the per-token likelihood, which normalizes for length. ARC, OpenBookQA, and RACE use the ratio

$$
\frac{P(\text{completion} \mid \text{context})}{P(\text{completion} \mid \text{answer context})}
$$

  where the answer context is the string "Answer: " or "A: ". The ratio asks how much the context raised the completion's probability over what a generic answer prompt gives it.

- **Binary classification.** Rename the labels to words such as "True" and "False", then treat the task like multiple choice.
- **Free-form completion.** Use beam search with a beam width of 4 and a length penalty of $\alpha = 0.6$, and score with F1, BLEU, or exact match, whichever the dataset uses.

**Tiny example, computed by `scripts/mc_scoring.py` and `scripts/fewshot_eval.py`.** First, the scoring rules. For the question "What is the capital of France?", the script uses invented per-token log-probabilities for three completions. The numbers are an illustration and aren't from the paper.

| Completion | Tokens | Sum of log-probabilities | Mean per token | Log ratio to the answer context |
|---|---|---|---|---|
| ` Paris` | 1 | -1.50 | -1.500 | 4.50 |
| ` a large city` | 4 | -2.00 | -0.500 | -0.20 |
| ` Lyon` | 1 | -2.50 | -2.500 | 4.00 |

The total log-likelihood picks ` Paris`, because the short completion pays for fewer tokens. The per-token mean picks ` a large city`, whose four tokens are each unsurprising whatever the question. The ratio picks ` Paris` again, because the context raised its probability the most. This is the reason the paper switches rules on some datasets. Exercise 2 implements all three.

Now the in-context behavior. The script builds a task with a made-up operation, written `blip` and meaning multiplication, and a stub model that infers the operation from the demonstrations in its prompt. The stub isn't a language model, but its weights never change, which is the property the paper's definition needs. Exact-match accuracy over 50 test questions with demonstrations drawn from a pool of 30 is as follows.

| $K$ | Accuracy |
|---|---|
| 0 | 0.00 |
| 1 | 1.00 |
| 2 | 1.00 |
| 5 | 1.00 |
| 10 | 1.00 |

A real model gives a curve instead of a step. Figure 1.2 plots this for a task that removes random symbols from a word, and finds that accuracy rises with $K$, rises further with a task description, and rises more steeply for larger models. The paper also reports results for the three settings on 175B directly. On the arithmetic battery of Section 3.9.1, exact-match accuracy goes as follows (Table 3.9):

| Task | Zero-shot | One-shot | Few-shot |
|---|---|---|---|
| 2-digit addition | 76.9 | 99.6 | 100.0 |
| 3-digit addition | 34.2 | 65.5 | 80.4 |
| 4-digit addition | 4.0 | 14.0 | 25.5 |
| 5-digit addition | 0.7 | 3.5 | 9.3 |
| 2-digit multiplication | 19.8 | 27.4 | 29.2 |

The paper's text gives 80.2 percent for 3-digit addition few-shot while Table 3.9 gives 80.4, so the page uses the table. Two spot checks from Section 3.9.1 bear on memorization: only 17 of 2,000 three-digit addition problems (0.8 percent) appear in the training data in the forms `<NUM1> + <NUM2> =` or `<NUM1> plus <NUM2>`, and wrong answers often miss a carried "1".

Other headline results from Section 1: on CoQA, 81.5, 84.0, and 85.0 F1 for zero-, one-, and few-shot. On TriviaQA, 64.3, 68.0, and 71.2 percent. Zero-shot perplexity on Penn Tree Bank is 20.5, against the previous best of 35.8 (Section 3.1.1, Table 3.1). Section 5 lists where few-shot stays weak: close to chance on the comparison tasks WiC and ANLI, and behind on some reading-comprehension sets such as QuAC and RACE.

### Scaling to 175 billion parameters

**Intuition.** The architecture is GPT-2's. What changes is size, and how the sizes are chosen. Previous work (Kaplan et al.) suggests that validation loss follows a smooth power law in model size, so the authors train eight models over three orders of magnitude, 125 million to 175 billion parameters, to see whether in-context learning scales the same way (Section 2.1).

**Math.** Table 2.1 lists the eight models, and all have $n_{\text{ctx}} = 2048$, $d_{\text{ff}} = 4\,d_{\text{model}}$, and 300 billion training tokens. The largest has $n_{\text{layers}} = 96$, $d_{\text{model}} = 12288$, $n_{\text{heads}} = 96$, $d_{\text{head}} = 128$, a batch of 3.2 million tokens, and a peak learning rate of $0.6 \times 10^{-4}$. Section 2.1 also says the layers alternate between dense and locally banded sparse attention patterns, as in the Sparse Transformer. The paper doesn't give the band width, and the parameter count doesn't depend on it.

Appendix D estimates training compute as follows. Each token costs one multiply and one add per parameter in the forward pass, which is 2 flops, and the backward pass costs twice the forward pass, so the total is $6$ flops per parameter per token. Attention is ignored. The compute is $C \approx 6\,n_{\text{params}}\,D$ for $D$ training tokens, and the paper reports it in petaflop/s-days of $8.64 \times 10^{19}$ flops each.

**Tiny example, computed by `scripts/model_size.py`.** The script counts parameters with the GPT-2 layout from the [previous page](/gpt2/): one block holds $12\,d_{\text{model}}^2 + 13\,d_{\text{model}}$ weights, plus a token matrix of $50{,}257 \times d_{\text{model}}$ shared with the output projection and $2048 \times d_{\text{model}}$ position embeddings. The paper states the $4\,d_{\text{model}}$ feed-forward width. It doesn't state the vocabulary size, the biases, the weight tying, or whether positions are learned, so those are assumptions carried over from GPT-2. They reproduce Table D.1 closely:

| Model | $n_{\text{layers}}$ | $d_{\text{model}}$ | Parameters (script) | Table D.1 | Difference |
|---|---|---|---|---|---|
| Small | 12 | 768 | 125,226,240 | 125M | +0.18% |
| Medium | 24 | 1024 | 355,871,744 | 356M | -0.04% |
| Large | 24 | 1536 | 760,300,032 | 760M | +0.04% |
| XL | 24 | 2048 | 1,315,723,264 | 1,320M | -0.32% |
| 2.7B | 32 | 2560 | 2,651,553,280 | 2,650M | +0.06% |
| 6.7B | 32 | 4096 | 6,658,404,352 | 6,660M | -0.02% |
| 13B | 40 | 5140 | 12,952,938,780 | 12,850M | +0.80% |
| 175B | 96 | 12288 | 174,604,259,328 | 174,600M | +0.00% |

For 175B, one block is 1,812,099,072 parameters, 96 blocks hold $12 \cdot 96 \cdot 12288^2 = 173{,}946{,}175{,}488$ of the total, and the token and position embeddings add 642,723,840. The weights of the layers dominate: embeddings are 0.4 percent of the model.

The 13B row is off by 0.8 percent because Table 2.1 prints $d_{\text{model}} = 5140$ for it, while $n_{\text{heads}} \cdot d_{\text{head}} = 40 \cdot 128 = 5120$. With $d_{\text{model}} = 5120$ the script gives 12,853,386,240, which matches Table D.1's 12,850M. This page treats 5140 as a typo in the table, and it is the only row that doesn't fit.

Compute then follows. With Table D.1's 174.6 billion parameters, $6 \times 174.6 \times 10^9 \times 300 \times 10^9 = 3.143 \times 10^{23}$ flops, which is 3,637 petaflop/s-days, against the table's $3.14 \times 10^{23}$ and 3,640. The script's other rows match the table within rounding. The 175B model sees $300 / 174.6 = 1.72$ tokens per parameter. Later work, including the [Chinchilla paper](/chinchilla/), argues that smaller models trained on more tokens use the same compute better.

Appendix D says attention "typically uses less than 10%" of the compute and ignores it. At full context with dense attention, the two attention matrix products cost $4\,n_{\text{ctx}}\,d_{\text{model}}$ flops per token per layer, against $24\,d_{\text{model}}^2$ for the weight matrices. That share is 2.7 percent for 175B, 14.3 percent for XL, and 30.8 percent for Small. The claim holds for the large models, and the small models' figures in Table D.1 are rougher than the large ones. The locally banded layers cost less than dense ones, and causal masking halves the average cost, so the real shares are lower than these. Exercise 3 builds the counts.

### Training on a weighted data mix

**Intuition.** Common Crawl is large but noisy. The paper filters it by similarity to high-quality reference text, removes near-duplicate documents, and then mixes it with curated corpora that it samples more often than their size warrants. This accepts a little overfitting for better data (Section 2.2).

**Math.** The mix in Table 2.2 gives the fraction of training examples drawn from each source. It isn't proportional to the source size.

| Dataset | Tokens | Weight in mix | Epochs in 300B tokens |
|---|---|---|---|
| Common Crawl (filtered) | 410 billion | 60% | 0.44 |
| WebText2 | 19 billion | 22% | 2.9 |
| Books1 | 12 billion | 8% | 1.9 |
| Books2 | 55 billion | 8% | 0.43 |
| Wikipedia | 3 billion | 3% | 3.4 |

Common Crawl and Books2 are seen less than once, and the other sources two to three times. The Table 2.2 epoch figures aren't exactly weight times 300B divided by size, for example, 0.22 · 300 / 19 = 3.5 and not 2.9, so the weights are probably rounded. Don't derive one column from the others.

Training uses Adam with $\beta_1 = 0.9$, $\beta_2 = 0.95$, and $\epsilon = 10^{-8}$, a global gradient-norm clip of 1.0, weight decay 0.1, a linear warmup over the first 375 million tokens, and cosine decay to 10 percent of the peak rate over 260 billion tokens (Appendix B). The [Adam page](/adam/) covers the optimizer. The batch size ramps from 32 thousand tokens to its full value over the first 4 to 12 billion tokens.

Every training sequence has the full $n_{\text{ctx}} = 2048$ tokens. Documents shorter than that are packed together and delimited by an end-of-text token, and the sequence isn't masked in any special way, so tokens attend across the boundary (Appendix B). The authors say the end-of-text token is enough for the model to infer that the text on either side is unrelated.

**Tiny example, computed by `scripts/packing_example.py`.** The paper doesn't say what happens to a document that crosses a sequence boundary or to a short remainder, so the script cuts one token stream into equal pieces and drops the remainder. With $n_{\text{ctx}} = 4$ and the end-of-text token written as 0, the documents `[1,2,3]`, `[4,5]`, and `[6,7,8,9]` pack into three sequences:

| Sequence | Document ids | Cross-document (query, key) pairs |
|---|---|---|
| `[1, 2, 3, 0]` | 0, 0, 0, 0 | 0 of 10 |
| `[4, 5, 0, 6]` | 0, 0, 0, 1 | 3 of 10 |
| `[7, 8, 9, 0]` | 0, 0, 0, 0 | 0 of 10 |

In the second sequence the last token, `6`, belongs to a new document but can attend to the three tokens of the one before. Scale this up: for 60 invented documents with a mean length of 643 tokens, packed to 2048, 57.0 percent of causal (query, key) pairs cross a boundary. The document lengths are invented, so read it as "a large share", not as a measurement of the real corpus. Exercise 4 asks you to implement the packing and the count.

<!-- exercises -->

## Pitfalls

- **Treating few-shot as training.** No weights change in any of the three settings. The accuracy gain with $K$ comes from the forward pass on a longer prompt, and the next prompt starts fresh. Fine-tuning is a separate setting that the paper doesn't run.
- **Reading "learning" too literally.** The paper's footnote 1 says the terms are "agnostic" on whether the model learns a new task at inference time or recognizes one it saw during training. Table 3.9 can't tell the two apart.
- **Drawing demonstrations that contain the test question.** When the demonstration pool overlaps the evaluation set, a demonstration can hand the answer to the model. Exclude the query from its own demonstrations, as Exercise 1 does.
- **Comparing likelihoods without normalizing.** A total log-likelihood prefers short completions. The paper normalizes per token for most tasks and by the answer-context probability for ARC, OpenBookQA, and RACE, and picked each on the development set.
- **Stripping nothing before exact match.** A model completes after `A:` with a leading space, so `" 124"` must equal `"124"`. Substring matching goes too far in the other direction, because `"1245"` contains `"124"`.
- **Using one $K$ for every task.** The paper picks $K$ on the development set when one exists, and most prompts hold 10 to 100 examples (Section 2.4). Longer examples leave room for fewer.
- **Reading Table 2.1 as exact.** The 13B row prints $d_{\text{model}} = 5140$, which doesn't equal $n_{\text{heads}} \cdot d_{\text{head}}$. The scale names are labels, and the counts in Table D.1 match the architecture only when you use 5120.
- **Ignoring attention in the compute estimate.** The $6\,n_{\text{params}}\,D$ rule drops attention. It is close for 175B and loose for small models at 2048 tokens.
- **Assuming the benchmarks are clean.** A bug in the overlap filtering left some benchmark text in the training data (Section 2.2). Section 4 flags examples with a 13-gram overlap and compares scores on the clean subsets. The paper reports that contamination has a minimal effect on most datasets and marks a few results with an asterisk.
- **Expecting every task to improve.** Few-shot performance on WiC and ANLI is near chance (Section 5). The paper ties part of this to the lack of bidirectional context in a left-to-right model, and calls that a conjecture.
- **Treating a bigger context as free.** Every demonstration spends tokens of the 2048-token window and attention over them on every generated token. The cost of a few-shot prompt is paid at inference time.

## Further reading

- [openai/gpt-3](https://github.com/openai/gpt-3) is the authors' repository with the synthetic datasets and the samples that the paper releases. It doesn't contain the model weights.
- [Scaling Laws for Neural Language Models](/kaplan/), by Kaplan et al., is the scaling-law work that the paper builds on, and Section 2 cites it for the choice of batch size and model shape.
- [Training Compute-Optimal Large Language Models](/chinchilla/), by Hoffmann et al., revisits the $n_{\text{params}}$ versus tokens trade-off that GPT-3 settled at 300 billion tokens.
- [Transformer Math 101](https://blog.eleuther.ai/transformer-math/), by EleutherAI, derives the parameter and compute formulas used in Exercise 3.
- [nanoGPT](https://github.com/karpathy/nanoGPT), by Andrej Karpathy, is a short reimplementation of the GPT-2 architecture that GPT-3 scales.
