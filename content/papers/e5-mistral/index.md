---
id: "e5-mistral"
title: "Improving Text Embeddings with Large Language Models"
authors: "Wang et al."
year: 2023
venue: "ACL 2024"
arxiv: "2401.00368"
url: "https://arxiv.org/abs/2401.00368"
track: nlp-llm
tier: extended
est_hours: 3
status: full
summary: "E5-Mistral turns a decoder-only language model into a text embedding model with one short contrastive fine-tuning stage. A proprietary LLM writes about 500 thousand synthetic query-document pairs across 93 languages, and the embedding of a text is the last-layer vector at an appended end-of-sequence token. A natural-language instruction in front of each query selects the task, and the model reaches 66.6 on the MTEB benchmark when it trains on synthetic data plus labeled data."
related_course: "ml: text embeddings and retrieval"
prereqs:
  - id: clip
    reason: "E5-Mistral trains with the same temperature-scaled InfoNCE loss and the same end-of-text-token feature that CLIP uses, so you need to know how in-batch negatives and the temperature shape the loss."
  - id: llama
    reason: "The backbone is a decoder-only Transformer with causal attention in the LLaMA family, so you need to know why only the last position of a causal model has read the whole text."
  - id: lora
    reason: "The fine-tuning stage trains LoRA adapters of rank 16 on a frozen 7B model, so you need to know what the adapters change and why so few parameters train."
background: [contrastive-learning, infonce, fine-tuning, cross-entropy]
course_links:
  - title: "Text embeddings and retrieval"
    url: "https://ml.ifkash.dev/lessons/077-text-embeddings-and-retrieval"
  - title: "Contrastive learning: InfoNCE, collapse, and CLIP"
    url: "https://ml.ifkash.dev/lessons/051-contrastive-learning"
self_check:
  - q: 'With $\tau = 0.02$, a query has cosine 0.80 with its positive and 0.78 with its only negative. What is the InfoNCE loss, and what is the loss at $\tau = 1$ with the same cosines?'
    a: 'The scaled gap is $(0.80 - 0.78) / 0.02 = 1$, so the loss is $\ln(1 + e^{-1}) \approx 0.3133$. At $\tau = 1$ the gap is $0.02$ and the loss is $\ln(1 + e^{-0.02}) \approx 0.6832$, close to the $\ln 2 \approx 0.6931$ of a coin flip. A small $\tau$ turns a small cosine gap into a clear preference.'
    see: infonce
  - q: 'A batch holds two texts, one with 4 real tokens and one with 2, padded to length 4. Which position does last-token pooling read in each row for right padding and for left padding?'
    a: 'With right padding the rows read positions 3 and 1 (zero-based), because the second row ends in two pad tokens. With left padding both rows read position 3. Reading position 3 for a right-padded second row would return a pad position, not the `[EOS]` token.'
    see: llama
  - q: 'LoRA adapters of rank 16 on every linear layer give about 42 million trainable parameters on a base of about 7 billion. What share of the weights is that?'
    a: 'About $42 / 7000 = 0.6$ percent. The base model stays frozen, so one adapter file is small enough to store per task.'
    see: lora
---

## The problem

A text embedding model maps a piece of text to a vector so that texts with related meaning land close together. Search, clustering, and retrieval-augmented generation all rely on it. Before this paper, the strongest models came from small bidirectional encoders such as E5 and BGE, and they took a long recipe to train. The recipe had a large contrastive pre-training stage on billions of weakly labeled text pairs, followed by fine-tuning on a few labeled datasets (Section 1).

That recipe has two weaknesses. First, the data is narrow. Labeled embedding datasets cover few task types and few languages, so a model trained on them handles a new kind of search poorly. Second, the pipeline is long and costly to rebuild.

Large language models seemed like a good fix, because they've read far more text than any encoder. Earlier work on LLM-based embeddings, such as SGPT and GTR, showed that scale helps, but those models still trailed the best small encoders (Section 2).

A concrete failing example is a search system that must answer two different questions with one index. The query "how much protein should a female eat" is a question, and the right document is a passage that answers it. A query such as a sentence from a news article asks for a near-duplicate. A single embedding of the query can't know which of the two tasks it serves, so one task's neighbors rank above the other's.

The paper's move has three parts (Section 3):

- Ask a proprietary LLM to write the training data, with a different task each time.
- Put a one-sentence task description in front of the query so that one model serves many tasks.
- Fine-tune an open decoder-only model, Mistral-7B, with a plain contrastive loss for one epoch.

The data step needs no human labels, and the synthetic-only model reaches an MTEB average of 63.1 (Table 1).

## Key ideas

This page uses the paper's notation. A query is $q^+$, its relevant document is $d^+$, and $q^+_{\text{inst}}$ is the query with the instruction in front. The embeddings are $h_{q^+_{\text{inst}}}$ and $h_{d^+}$, the temperature is $\tau$, and the set of negatives is $\mathcal{N}$. The page writes $\cos$ for cosine similarity. Every number in a worked example comes from a script in `scripts/`. Run any script from the paper folder to reproduce the output.

### Synthetic data from a two-step prompt

The idea is to let GPT-4 supply the diversity that labeled datasets lack. The paper sorts embedding tasks into two groups (Section 3.1).

- **Asymmetric tasks** have a query and a document that are related but not paraphrases. The paper splits them by length into short-long, long-short, short-short, and long-long matches.
- **Symmetric tasks** have a query and a document with similar meaning and different wording. They are monolingual semantic textual similarity (STS) and bitext retrieval.

For the asymmetric tasks, a two-step prompt first asks the LLM to brainstorm a list of task definitions. A second prompt then writes one concrete example for a chosen definition. The paper reports that a single prompt for both steps gave less diverse data (Section 3.1). Symmetric tasks skip the brainstorming step. Placeholders such as the query length and the language are sampled at runtime to add variety. The paper drops outputs that don't parse as the expected JSON and removes exact duplicates (Section 3.1).

The data set has about 500 thousand examples with about 150 thousand unique instructions. About 25 percent come from GPT-3.5-Turbo and the rest from GPT-4, and the total use is about 180 million tokens (Section 4.1). It covers 93 languages, with about 1 thousand examples per language for the 75 lowest-resource ones.

The synthetic data isn't the whole training set. The final model also trains on 13 public datasets, which together with the synthetic data give about 1.8 million examples (Section 4.2, Appendix A). For datasets that have no hard negatives, the authors mined the top 100 with mE5-base and used one hard negative per pair (Appendix A).

### The query instruction

The idea is to describe the task in words, on the query side only. Given a relevant pair, the paper builds the instructed query with a fixed template (Section 3.2, Equation 1):

$$
q^+_{\text{inst}} = \texttt{Instruct: \{task\_definition\}} \backslash n \; \texttt{Query: \{} q^+ \texttt{\}}
$$

Here `{task_definition}` is a one-sentence task description. For the synthetic data it is the output of the brainstorming step. For datasets such as MS-MARCO, the authors wrote the definitions by hand and reused one for every query in the dataset (Section 3.2).

The document gets no prefix. This choice has a practical payoff: you can build the document index once, and then change the task by changing only the query text (Section 3.2). It also fits the ablation in Table 5. Removing the instruction lowers the MTEB average from 64.5 to 60.3, a drop of 4.2 points. A short task-type prefix in place of a sentence also scores 60.3 (Section 5.3, Table 5).

Worked example, computed by `scripts/query_format.py`. With the task "Given a web search query, retrieve relevant passages that answer the query" and the query "how much protein should a female eat", the instruction adds 92 characters in front of the query. The document text is unchanged.

### Last-token pooling

The idea is to read the embedding from the one position that has seen the whole text. A causal Transformer lets position $t$ attend only to positions up to $t$. Only the last position has read every token, so the paper appends an `[EOS]` token to both the query and the document and takes the last-layer hidden vector at that token (Section 3.2). CLIP reads its text feature the same way.

For a batch, you have to find the `[EOS]` position in each row, because padding moves it.

- With right padding, the last real token sits at index `mask.sum() - 1`.
- With left padding, every row ends in a real token, so index $T - 1$ is correct, and `mask.sum() - 1` is wrong for any row that has padding.
- One rule covers both: take the last position where the attention mask is 1.

Worked example, computed by `scripts/pooling_example.py`. Take a batch of two rows and four positions. Row 0 has four real tokens and row 1 has two. With right padding, the last real indices are 3 and 1, and reading position $-1$ for row 1 returns a pad position. With left padding, both indices are 3. The expression `sum(mask) - 1` gives 3 and 1 on the left-padded batch, so it picks the wrong position for row 1.

The pooling choice doesn't matter much for the score. Mean pooling over all tokens gives an MTEB average of 64.1 against 64.5 for last-token pooling (Section 5.3, Table 5). The paper keeps last-token pooling for the default.

### InfoNCE with in-batch and hard negatives

The idea is to train the model to pick the relevant document out of a pool of distractors. The loss is the standard InfoNCE loss over in-batch negatives and hard negatives (Section 3.2, Equation 2):

$$
\mathcal{L} = -\log \frac{\phi(q^+_{\text{inst}}, d^+)}{\phi(q^+_{\text{inst}}, d^+) + \sum_{n_i \in \mathcal{N}} \phi(q^+_{\text{inst}}, n_i)}
$$

Here $\mathcal{N}$ is the set of all negatives. The matching score is a temperature-scaled cosine similarity (Section 3.2, Equation 3):

$$
\phi(q, d) = \exp\!\left(\frac{1}{\tau} \cos(h_q, h_d)\right)
$$

The temperature is fixed at $\tau = 0.02$, so the scale $1/\tau$ is 50. The paper doesn't spell out which documents enter $\mathcal{N}$ beyond "in-batch negatives and hard negatives". The exercise on this page uses every other document in the batch and every hard negative in the batch, which is a common implementation choice and not a claim about the authors' code.

Why a temperature as small as 0.02? Cosine values live in $[-1, 1]$. At $\tau = 1$, the softmax over cosines of 0.80, 0.78, and 0.70 is almost flat, so the positive gets a share of 0.3466 and the loss is 1.0595. At $\tau = 0.02$, the same cosines give the positive a share of 0.7275 and a loss of 0.3182 (computed by `scripts/infonce_example.py`). The small temperature stretches a gap of 0.02 in cosine into a gap of 1 in the exponent.

Worked example, computed by `scripts/infonce_example.py`. One query has cosine 0.80 with its positive and 0.78 with a single hard negative. At $\tau = 0.02$ the loss is $\ln(1 + e^{-1}) \approx 0.3133$. To push the loss against one negative down to 0.01, the cosine gap has to reach about 0.0920.

### Fine-tuning a 7B model for under a thousand steps

The idea is that a model pre-trained on a large amount of text already has good text representations, so a short, light fine-tune is enough. The paper fine-tunes Mistral-7B for one epoch with the loss above (Section 4.2). It follows the RankLLaMA recipe and trains LoRA adapters of rank 16 on all linear layers, which gives about 42 million trainable parameters (Section 4.2, Appendix A). The other settings are a batch size of 2048, a learning rate of $10^{-4}$ with 100 warmup steps and linear decay, a weight decay of 0.1, one hard negative per pair, and a maximum length of 512 tokens (Appendix A).

The step count follows from the data. About 1.8 million examples at a batch size of 2048 give about 879 steps, which matches the paper's "fewer than 1k steps" (Appendix A; computed by `scripts/scale_numbers.py`). The 42 million trainable parameters are about 0.6 percent of a 7 billion parameter base.

Contrastive pre-training, the long first stage of older recipes, matters little here. It helps XLM-R-large, but for Mistral-7B the effect is negligible, which the paper reads as evidence that autoregressive pre-training already builds good text representations (Section 5.1, Figure 3).

### Results and costs

On the 56 English datasets of MTEB, the full-data model scores 66.6, which beats the previous best of 64.2 by 2.4 points (Section 4.3, Table 1). The model trained on synthetic data plus MS-MARCO scores 64.5, and the model trained on synthetic data only scores 63.1. On the 15 BEIR retrieval datasets, the full-data model scores 56.9 nDCG@10 (Table 3). Retrieval metrics such as nDCG@10 use the ranking of the documents, which is the subject of the third exercise.

The model has costs that the paper states in its Limitations section. A 7B encoder is much more expensive to run than a BERT-style encoder. Each embedding has 4096 dimensions, and in float32 that is 16,384 bytes per document, or 16.4 GB for a million documents (computed by `scripts/scale_numbers.py`). The multilingual quality for low-resource languages lags, because Mistral-7B is pre-trained mostly on English text (Table 2).

<!-- exercises -->

## Pitfalls

- **Reading position -1 on a right-padded batch.** The last position is a pad token, so the embedding comes from padding. Use the attention mask to find the last real token, as in the first exercise.
- **Using `sum(mask) - 1` on a left-padded batch.** That index is correct only for right padding. The mask-based rule in the first exercise works for both.
- **Forgetting the `[EOS]` token.** The paper appends `[EOS]` to both queries and documents before the forward pass (Section 3.2). Many tokenizers don't add it by default, and the model then reads an ordinary token.
- **Putting the instruction on the documents.** The paper instructs only the query. An instruction on the documents breaks the prebuilt index, because the index would depend on the task.
- **Changing the instruction at test time without checking.** The wording of the task description affects quality (Table 5). Use a sentence that describes the task, as the training data did.
- **Forgetting to normalize.** Equation 3 uses cosine similarity. If you use a raw dot product, the vector length enters the score and the temperature of 0.02 no longer means what the paper intends.
- **Using a large temperature.** At $\tau = 1$, cosines in $[-1, 1]$ give a nearly flat softmax and a weak training signal.
- **Counting hard negatives as in-batch.** In-batch negatives are free, because the other pairs' documents are already encoded. Hard negatives add an extra encoded document per pair, so they raise the cost per step.
- **Comparing scores across different evaluation settings.** The paper evaluates only the first 512 tokens of each text, and one evaluation takes about 3 days on 8 V100 GPUs because of the document encoding (Section 4.2).

## Further reading

- [Official model and usage code on the Hugging Face Hub](https://huggingface.co/intfloat/e5-mistral-7b-instruct): the model card includes the last-token pooling code that handles both padding sides.
- [The unilm repository](https://github.com/microsoft/unilm/tree/master/e5): the authors' code for the E5 family.
- [MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316): the benchmark behind the headline numbers.
- [BEIR: a heterogeneous benchmark for zero-shot retrieval](https://arxiv.org/abs/2104.08663): the 15 retrieval datasets that MTEB reuses.
- [RepLLaMA and RankLLaMA](https://arxiv.org/abs/2310.08319): the LoRA recipe for LLM retrievers that this paper follows.
- [Text Embeddings by Weakly-Supervised Contrastive Pre-training (E5)](https://arxiv.org/abs/2212.03533): the encoder-based predecessor.
