---
id: "bert"
title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
authors: "Devlin et al."
year: 2018
venue: "NAACL 2019"
arxiv: "1810.04805"
url: "https://arxiv.org/abs/1810.04805"
track: nlp-llm
tier: extended
est_hours: 4
status: full
summary: "BERT pre-trains a stack of Transformer encoder layers that read text in both directions at once. To keep a token from seeing itself, it hides 15 percent of the input tokens and trains the model to recover them, and it adds a second task that asks whether two sentences were adjacent. You then fine-tune the whole model on each task with one small output layer, and this set new results on eleven language-understanding benchmarks."
related_course: "ml: transformers"
prereqs:
  - id: transformer
    reason: "BERT is a stack of Transformer encoder layers, so it reuses multi-head self-attention, residual connections, and layer normalization, and changes only the attention mask and the training objective."
background: [masked-language-modeling, fine-tuning]
course_links:
  - title: "Encoders, decoders, and cross-attention"
    url: "https://ml.ifkash.dev/lessons/065-encoders-and-cross-attention"
  - title: "Scaled attention, many heads, and the causal mask"
    url: "https://ml.ifkash.dev/lessons/059-scaled-attention-and-heads"
  - title: "The transformer block: SwiGLU, residuals, and layer norm"
    url: "https://ml.ifkash.dev/lessons/063-the-transformer-block"
self_check:
  - q: 'A sentence has 20 WordPiece tokens. About how many does BERT select for prediction, and of those, about how many are replaced by `[MASK]`, replaced by a random token, and left unchanged?'
    a: 'About 15 percent are selected, so 3 tokens. Of the selected tokens, 80 percent become `[MASK]`, 10 percent become a random token, and 10 percent stay the same (Section 3.1, Appendix A.1). In expectation that is 2.4, 0.3, and 0.3 tokens. The model predicts the original token at all 3 selected positions.'
    see: masked-language-modeling
  - q: 'In a causal Transformer, can the output at position 1 change if you edit token 3? What about in BERT, and which part of the layer differs?'
    a: 'In a causal model the output at position 1 cannot change, because the mask blocks attention to later positions. In BERT it can change, because every position attends to every other position in every layer. The attention formula is the same, and only the mask differs (Section 3 and Figure 3).'
    see: transformer
  - q: 'You fine-tune BERT on a sentence-pair classification task. Which hidden state feeds the classifier, and which parameters get updated?'
    a: 'The classifier reads the final hidden state of the `[CLS]` token, which sits first in the packed input (Section 3). Fine-tuning updates all the pre-trained parameters together with the new output layer, so no weights stay frozen (Section 3.2).'
    see: fine-tuning
---

## The problem

Before BERT, the strongest way to reuse unlabeled text was a language model, and a language model reads in one direction. The paper names this as the major limitation (Section 1): in OpenAI GPT, every token attends only to the tokens before it, a restriction that the paper calls sub-optimal for sentence-level tasks and very harmful for token-level tasks such as question answering, where you need context from both sides.

ELMo, the other well-known approach, did read both directions, but it trained a left-to-right model and a right-to-left model separately and concatenated their outputs. Neither half sees the whole sentence while it builds a token's representation, and ELMo supplies features to a task-specific architecture instead of being fine-tuned itself.

Consider a concrete failure. You want the model to resolve the blank in "The bank by the ___ was steep." A left-to-right model reads only "The bank by the" when it fills the blank, so it can't tell river from money. The word "steep" is the clue, and it comes after the gap. Conditioning every layer on both sides looks like the obvious fix, and it has a catch: if each position can attend to every position, then in a deep stack the position can see the very token you ask it to predict. The model learns to copy, and the loss goes to zero without learning anything about language.

BERT's answer is to change the task and leave the architecture alone. It hides some tokens and asks the model to recover them from the rest (Section 3.1). A hidden token can't leak through attention, because it isn't in the input. With that one change, you can drop the causal mask and read the whole sequence at once.

## Key ideas

The paper has no numbered equations. It describes its method in prose, tables, and figures, so this page cites sections, figures, and tables. Notation follows the paper: $L$ is the number of layers, $H$ the hidden size, and $A$ the number of attention heads (Section 3). Every number in an example comes from a script in `scripts/`. Run any of them from the paper folder to reproduce the output.

### Bidirectional encoder: the same layer, a different mask

**Intuition.** BERT is the encoder half of the Transformer from the [previous paper](/transformer/). Each layer is multi-head self-attention followed by a feed-forward network, wrapped in residual connections and layer normalization. GPT uses the same layer with a causal mask. BERT removes the mask, so each position reads all positions.

**Math.** Section 3 states that BERT's model architecture is a multi-layer bidirectional Transformer encoder, based on the original implementation, and gives two sizes:

| Model | $L$ | $H$ | $A$ | Parameters (paper) |
|---|---|---|---|---|
| BERT-BASE | 12 | 768 | 12 | 110M |
| BERT-LARGE | 24 | 1024 | 16 | 340M |

The feed-forward size is $4H$, which is 3072 for BERT-BASE and 4096 for BERT-LARGE (Section 3, footnote). BERT-BASE was chosen to match OpenAI GPT's size so the two can be compared, and the paper notes that the comparison differs mainly in the attention mask (Section 3, Section 4.1). Figure 3 states the consequence: only BERT conditions each representation on both left and right context in all layers. With the same scaled dot-product attention as before, written in the Transformer paper's notation,

$$
\operatorname{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}} + M\right)V
$$

the only difference is the additive mask $M$. For a causal model, $M_{ij} = -\infty$ when $j > i$ and $0$ otherwise. For BERT, $M = 0$ everywhere (apart from padding). This formula is the page's restatement, since the BERT paper doesn't repeat it.

**Tiny example, computed by `scripts/attention_masks.py`.** Take three tokens with embeddings $(1, 0)$, $(0, 1)$, $(1, 1)$, and set the query, key, and value projections to the identity. The attention weights of the first token are:

| Mask | Weights of token 1 over tokens 1, 2, 3 | Output for token 1 |
|---|---|---|
| Bidirectional | 0.4011, 0.1978, 0.4011 | (0.8022, 0.5989) |
| Causal | 1, 0, 0 | (1, 0) |

Now replace token 3 with $(-3, 2)$. With the bidirectional mask, the output for token 1 moves by 0.2722 in its largest coordinate. With the causal mask, it moves by exactly 0. That difference is the whole architectural gap between an encoder and a decoder. Exercise 3 asks you to implement attention with both masks.

The price is that BERT can't generate text by sampling the next token, because it has no left-to-right factorization of the probability of a sequence. You can still fine-tune it as a classifier or tagger, which is what the paper does.

### Input representation: `[CLS]`, `[SEP]`, and three embeddings

**Intuition.** One input format has to cover a single sentence and a sentence pair, so that one model can handle classification, entailment, and question answering. The model needs to know which tokens belong to which sentence and where each token sits.

**Math.** Section 3 and Figure 2 describe the input:

- Text is split with WordPiece, with a 30,000-token vocabulary.
- The first token of every sequence is `[CLS]`. Its final hidden state is the aggregate sequence representation for classification.
- A sentence pair is packed into one sequence, with the sentences separated by the special token `[SEP]`. A learned segment embedding marks each token as belonging to sentence A or sentence B.
- The input for each token is the sum of its token embedding, its segment embedding, and its position embedding (Figure 2). BERT learns the position embeddings. It doesn't use the sinusoidal encodings from the Transformer paper.

Writing $E_{\text{tok}}$, $E_{\text{seg}}$, and $E_{\text{pos}}$ for the three tables (this page's names, not the paper's), the input at position $i$ is

$$
x_i = E_{\text{tok}}[t_i] + E_{\text{seg}}[s_i] + E_{\text{pos}}[i]
$$

where $t_i$ is the token id and $s_i \in \{0, 1\}$ is the segment id. The packed sequence for sentences $a$ and $b$ is `[CLS] a [SEP] b [SEP]`. Appendix A.2 sets the maximum sequence length to 512 tokens for the pair together.

**Tiny example, computed by `scripts/input_packing.py`.** Pack sentence A with token ids 10, 11, 12 and sentence B with ids 20, 21, using `[CLS]` = 1 and `[SEP]` = 2:

| Position | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| Token id | 1 (`[CLS]`) | 10 | 11 | 12 | 2 (`[SEP]`) | 20 | 21 | 2 (`[SEP]`) |
| Segment id | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |

With 2-dimensional toy tables, position 4 (the first `[SEP]`) is $(0, 1) + (0.5, 0.5) + (0.8, 0.9) = (1.3, 2.4)$. The token embedding of `[SEP]` is the same at positions 4 and 7, but the sums differ, because the segment and position terms differ. Exercise 4 asks you to build the packed input and the sum.

### Masked language modeling

**Intuition.** Hide a few tokens, then predict them from everything else. A hidden token isn't in the input, so a deep bidirectional stack has nothing to copy.

**Math.** Section 3.1 (Task #1) says to select 15 percent of the WordPiece tokens in each sequence at random for prediction. The model then predicts only the selected tokens, using a softmax over the vocabulary on the final hidden vectors, rather than reconstructing the whole input.

The paper points out a mismatch: `[MASK]` never appears during fine-tuning. To reduce it, a selected token isn't always replaced by `[MASK]`. Instead (Section 3.1 and Appendix A.1):

- 80 percent of the time, the token becomes `[MASK]`.
- 10 percent of the time, it becomes a random token.
- 10 percent of the time, it stays unchanged.

The paper's example, with the sentence "my dog is hairy" and the fourth token selected, gives "my dog is [MASK]", "my dog is apple", and "my dog is hairy". The target at that position is always "hairy". Because the model can't tell which selected tokens were changed, it has to keep a good representation of every input token. The loss is the cross-entropy of the original token at the selected positions only (Section 3.1), so in notation of this page, with $S$ the selected positions and $t_i$ the original token,

$$
\mathcal{L}_{\text{MLM}} = -\frac{1}{|S|} \sum_{i \in S} \log p_\theta\!\left(t_i \mid \tilde{x}\right)
$$

where $\tilde{x}$ is the corrupted input. The division by $|S|$ is this page's choice of average. The paper's text doesn't spell out the normalization.

**Tiny example, computed by `scripts/masking_example.py`.** The script corrupts a seeded sequence of 1,000,000 random token ids with the rule above:

| Quantity | Value |
|---|---|
| Fraction of tokens selected | 0.1497 |
| Of the selected: replaced by `[MASK]` | 0.8018 |
| Of the selected: replaced by a random token | 0.0990 |
| Of the selected: left unchanged | 0.0992 |
| Random replacements, as a fraction of all tokens | 0.0148 |
| Label positions equal the selected positions | True |

The paper notes that random replacement affects only 1.5 percent of all tokens, which is 10 percent of 15 percent, and the script's 0.0148 agrees. Only 13.5 percent of tokens differ from the original after corruption, because the unchanged tokens count among the selected ones. Exercise 1 asks you to implement this rule.

Now the loss. Take four positions and a vocabulary of three, with logits $(2, 0, 0)$, $(1, 2, 0)$, $(0, 0, 3)$, and $(0, 1, 2)$. Select positions 2 and 4, with original tokens 1 and 0 (counting from 0). The script `scripts/mlm_loss.py` prints the negative log-probabilities 0.4076 and 2.4076, so the loss is $(0.4076 + 2.4076) / 2 = 1.4076$. If you average over all four positions with a dummy label of 0 at the unselected ones, you get 1.5374, and the sum over the selected positions is 2.8152. Both are wrong. Exercise 2 asks you to implement the masked loss.

The cost of this objective is the learning signal. A causal model gets a prediction at every position, so a 512-token sequence gives 511 targets. Masked training gives about 77. The paper says the masked model converges slightly slower than a left-to-right model, but that its accuracy exceeds the left-to-right model's almost immediately (Appendix C.1).

### Next-sentence prediction

**Intuition.** Many tasks, such as question answering and natural language inference, depend on how two sentences relate. A language-model objective doesn't capture that directly, so BERT adds a second pre-training task.

**Math.** Section 3.1 (Task #2) builds training pairs as follows. For each example, sentence B is the real next sentence 50 percent of the time (labeled `IsNext`) and a random sentence from the corpus 50 percent of the time (labeled `NotNext`). The final hidden state of `[CLS]` feeds a binary classifier. The paper reports that the model reaches 97 to 98 percent accuracy on this task (Section 3.1, footnote). The pre-training loss adds the masked-LM loss and this classification loss. Both losses are means over their own targets.

The ablation in Table 5 tests whether it helps, on the BERT-BASE dev set:

| Variant | MNLI-m | QNLI | MRPC | SST-2 | SQuAD F1 |
|---|---|---|---|---|---|
| BERT-BASE | 84.4 | 88.4 | 86.7 | 92.7 | 88.5 |
| No NSP | 83.9 | 84.9 | 86.5 | 92.6 | 87.9 |
| Left-to-right, no NSP | 82.1 | 84.3 | 77.5 | 92.1 | 77.8 |

Removing NSP hurts on QNLI, MNLI, and SQuAD 1.1 (Section 5.1). Making the model left-to-right is worse on every task, with large drops on MRPC and SQuAD. Later work questioned the value of NSP. Treat Table 5 as the paper's own evidence and the later results as outside the paper.

### Pre-train, then fine-tune

**Intuition.** Pre-training builds general representations from unlabeled text, at a large cost paid once. Fine-tuning adapts all of them to a task with a small labeled set, at a small cost paid per task.

**Math.** Pre-training (Section 3.1 and Appendix A.2):

- The corpus is BooksCorpus (800M words) and English Wikipedia (2,500M words), text passages only.
- Each batch holds 256 sequences of 512 tokens, which is 128,000 tokens per batch, trained for 1,000,000 steps, about 40 epochs over the 3.3 billion words.
- The optimizer is Adam with learning rate $10^{-4}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, L2 weight decay 0.01, linear warmup over the first 10,000 steps, then linear decay (see the [Adam paper](/adam/)).
- Dropout is 0.1 on all layers, and the activation is GELU rather than ReLU.
- To save time, 90 percent of the steps use sequences of length 128, and the last 10 percent use 512.
- BERT-BASE trained on 4 Cloud TPUs (16 chips) and BERT-LARGE on 16 Cloud TPUs (64 chips), each for 4 days.

Fine-tuning (Section 3.2 and Appendix A.3) adds one task-specific output layer and updates every parameter. The paper's recommended ranges are a batch size of 16 or 32, a learning rate of $5 \times 10^{-5}$, $3 \times 10^{-5}$, or $2 \times 10^{-5}$, and 2, 3, or 4 epochs, with dropout kept at 0.1. For classification, the `[CLS]` hidden state feeds the new layer. For span tasks such as SQuAD, the layer scores each token as a start or an end.

The headline results come from Section 4 and Tables 1 to 4:

- **GLUE:** a score of 80.5 on the leaderboard, a 7.7-point absolute improvement over the previous best (Section 4.1).
- **SQuAD v1.1:** a test F1 of 93.2 for the ensemble with TriviaQA fine-tuning (Table 2).
- **SQuAD v2.0:** a test F1 of 83.1 for a single BERT-LARGE model (Table 3).
- **SWAG:** a test accuracy of 86.3 for BERT-LARGE (Table 4).

Because the model reads the pair jointly, the same recipe covers four task shapes: sentence pairs (entailment), single sentences (sentiment), question and passage (extractive question answering), and single sentences with token-level labels (tagging).

**Tiny example, computed by `scripts/param_count.py`.** The paper gives 110M parameters for BERT-BASE. You can count them from the architecture. One layer has four attention projections, each with a bias, two feed-forward layers with biases, and two layer norms:

$$
4(H^2 + H) + (4H^2 + 4H) + (4H^2 + H) + 4H = 12H^2 + 13H
$$

For $H = 768$ that is 7,087,872 parameters per layer, and 12 layers hold 85,054,464. The embeddings add the token table, 512 position vectors, 2 segment vectors, and a layer norm. The script prints the counts for the paper's 30,000-token vocabulary and for the 30,522 tokens of the released uncased checkpoint:

| Vocabulary | Layers plus embeddings | Plus pooler, MLM head, NSP head |
|---|---|---|
| 30,000 (Section 3) | 108,490,752 | 109,705,010 |
| 30,522 (released checkpoint) | 108,891,648 | 110,106,428 |

Both round to the paper's 110M. For BERT-LARGE the same script gives 335.7M and 336.2M, against the paper's 340M. The count of 30,522 tokens is a property of the released English uncased checkpoint, which this page takes from memory of the released code. The paper itself says 30,000. The token embedding table holds 21.3 percent of the BERT-BASE total at 30,522 tokens, and the 12 layers hold 77.2 percent.

<!-- exercises -->

## Pitfalls

- **Computing the loss on every position.** Only the selected positions have a target. Averaging over all positions, or filling the others with a dummy label, changes the loss and trains the model to predict unchanged inputs.
- **Treating 80/10/10 as three independent draws.** First pick which 15 percent of positions to select. Then split the selected positions 80, 10, and 10. Drawing three separate 15 percent samples gives a different corruption rate.
- **Masking special tokens.** `[CLS]`, `[SEP]`, and padding have nothing to predict. Exclude them from selection, and exclude them from the pool of random replacement tokens.
- **Forgetting that the label is the original token.** For a position replaced by a random token or left unchanged, the target is still the original. If you build the labels from the corrupted input, you train an identity map.
- **Applying a causal mask by habit.** A decoder implementation often has the mask built in. BERT needs the full attention matrix, with a padding mask only.
- **Zeroing attention weights after the softmax.** The rows then no longer sum to 1. Add a large negative number to the scores before the softmax.
- **Dropping the segment embedding.** Without it, the model can find the sentence boundary only from the position of `[SEP]`. The paper's input is the sum of three embeddings (Figure 2).
- **Using sinusoidal positions.** BERT learns its position embeddings, so the position table has a fixed length of 512 rows, and a longer sequence has no row to look up.
- **Expecting `[MASK]` at fine-tuning time.** The token never appears when you fine-tune. That is why the 10 percent random and 10 percent unchanged cases exist (Section 3.1). Table 8 in Appendix C.2 shows fine-tuning is fairly robust to the mix, while a feature-based approach is more sensitive: a mask-only strategy hurts NER, and a random-only strategy does much worse.
- **Reading NSP as settled.** The paper's ablation says it helps, and later work found it matters little. Keep the two claims apart.
- **Reading the parameter counts loosely.** The paper rounds to 110M and 340M. A count from the architecture depends on the vocabulary size and on which heads you include.
- **Expecting to sample from BERT.** The model estimates a token given both sides, not the probability of a sequence, so it has no left-to-right factorization to sample from.

## Further reading

- [google-research/bert](https://github.com/google-research/bert) is the official code and checkpoint release, including the vocabulary files and the masking code in `create_pretraining_data.py`.
- [The Illustrated BERT, ELMo, and co.](https://jalammar.github.io/illustrated-bert/) by Jay Alammar is a diagram-first walk through the model and its inputs.
- [Hugging Face: BERT model documentation](https://huggingface.co/docs/transformers/model_doc/bert) shows the reference implementation and the `DataCollatorForLanguageModeling` that applies the 80/10/10 rule.
- [RoBERTa](https://arxiv.org/abs/1907.11692) by Liu et al. re-examines the recipe, including NSP and the choice of static or dynamic masking.
- [Deep contextualized word representations](https://arxiv.org/abs/1802.05365) by Peters et al. introduces ELMo, the feature-based baseline that Section 2 of the BERT paper discusses.
