---
id: qwn
title: "Qwen, minus the E"
summary: "A hobby project that fine-tunes Qwen3-1.7B to answer everyday prompts without the letter 'e'. A logit mask guarantees the rule, supervised fine-tuning and GRPO with an LLM judge make the answers read well, and the usable rate rises from about 1 in 4 to about 6 in 10."
source:
  title: "Qwen, Minus the E: Fine-tuning Qwen3-1.7B locally to write without the letter 'e'"
  author: "Kevin Wu"
  site: "kevin-wu.ca"
  date: "2026-10-10"
  url: "https://kevin-wu.ca/blog/qwn/"
topics: [rl, fine-tuning, llm]
est_minutes: 45
prereqs:
  - {id: grpo, reason: "The post's RL step is GRPO, so you need the idea of scoring a group of samples and standardizing the rewards inside the group."}
  - {id: lora, reason: "Every adapter in the post is a LoRA, and the memory appendix depends on how few parameters LoRA trains."}
  - {id: bpe, reason: "The reason a model cannot see letters is that it reads subword tokens, which this paper introduced for translation."}
background: [softmax, sft, kl-penalty, group-relative-baselines, optimizer-state-memory]
course_links:
  - {title: "RL lesson 71: group-relative policy optimization", url: "https://rl.ifkash.dev/lessons/071-group-relative-policy-optimization"}
  - {title: "ML lesson 77: fine-tuning and LoRA", url: "https://ml.ifkash.dev/lessons/077-fine-tuning-and-lora"}
---

## The problem

A lipogram is text that avoids one letter. The post asks a small question with a hard answer: can a 1.7-billion-parameter model answer everyday prompts in plain English without ever using the letter "e", trained on a single 16 GB GPU?

The difficulty is that language models do not read letters. A model sees tokens, which are chunks of text from a fixed vocabulary (see [Neural machine translation of rare words with subword units](/bpe/)). Whether a token contains an "e" is a fact about the token's spelling, and the model has no direct view of it. Three simple approaches fail in turn.

1. **Ask politely.** The post puts a system message in every prompt that forbids "e". Every one of 200 test answers still contained an "e". The usable rate is 0%.
1. **Reward the absence of "e".** The post's first reinforcement learning (RL) runs scored each token for whether it contained an "e". The model found letter soup: streams of single letters, and strings like "Tst, tkt, nkt". The reward was satisfied and the text was useless.
1. **Mask the tokens.** Setting the score of every token that contains an "e" to negative infinity removes every "e" with no training. The answers are still often bad: the post shows a masked answer that repeats the misspelled word "riddiculous" over and over. Masked Qwen3-1.7B scores 26% usable.

The post's recipe combines the last idea with two training stages. The mask gives the rule for free. Supervised fine-tuning (SFT) and then GRPO teach the model to write well with the words that remain.

### What "usable" means

The post counts an answer as usable when it passes three checks:

- It has no "e" or "E", including accented forms. Code checks this.
- It answers the prompt.
- It reads as normal English.

An LLM judge, GPT-6 Luna, scores the last two checks, coherence and responds, each from 1 to 5. An answer needs at least 4 on both. The judge is not told about the "e" rule, and it does not penalize wrong facts. The headline metric is the usable rate over 200 answers: 50 validation prompts with 4 answers each.

## Key ideas

### A logit mask guarantees the rule, not the sense

At each step the model produces one score, a logit, for every token in its vocabulary. A softmax turns the scores into probabilities and the sampler draws one token. The mask runs between those two steps: it sets the logit of every token that contains "e", "E", or a non-ASCII look-alike to $-\infty$. Because $e^{-\infty}=0$, those tokens get probability 0 and can never be drawn.

Here is a four-token example. This is our toy, not a number from the post. The model's logits for the next token are:

| Token | Logit | Probability, no mask | Probability, mask |
| --- | --- | --- | --- |
| the | 2.0 | 0.58 | 0 (has an "e") |
| walk | 1.0 | 0.21 | 0.51 |
| a | 0.5 | 0.13 | 0.31 |
| dog | 0.0 | 0.08 | 0.19 |

The softmax renormalizes over the three tokens that remain, so each one grows by the same factor. The ratio between "walk" and "a" is unchanged: the model's preferences among the allowed tokens survive. What the mask took away is the model's favorite word.

That is why the guarantee is one-sided. The constraint holds at every position, with certainty, whatever the model does. But the mask never checks that the allowed tokens say anything. When every good continuation has an "e", the model is forced onto what is left, and the result can be a misspelling or a repeated word. The post's own explanation of its later results rests on this: under the mask, each extra word is another chance to be forced into an odd one.

<div class="widget" data-widget="letter-mask"></div>

### Split the constraint from the skill

The mask handles the constraint. What remains is a skill: choosing words that make sense given that some words are now unavailable. The post trains that skill in two stages.

**Stage 1: SFT on e-free answers.** The first try was SFT on the novel *Gadsby*, a book written without "e", about 50,000 words. The loss fell, but the usable rate stayed near 26% and the outputs repeated themselves. The post's guesses are that a novel teaches prose rather than answering prompts, and that the data is too small. The second try was 180 hand-built e-free answers, written by frontier models with feedback on offending words. The usable rate under the mask grew with the number of examples:

| Training examples | 0 | 15 | 30 | 45 | 90 | 135 | 180 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Usable under the mask | 26% | 10% | 20.5% | 22.5% | 31.5% | 34% | 35% |

Small sets hurt, and 90 examples beat the *Gadsby* run at a similar size. Gains level off, and the adapter tends to swap in a different word when the word it needs has an "e". The post gives "clik" for "cliché" as an example.

**Stage 2: RL with GRPO.** SFT imitates fixed examples. RL asks the model to try answers, scores them, and shifts toward the better ones. The next two ideas explain the score and the update.

### A judge-based reward, and why it replaced the letter reward

Rewarding the absence of "e" fails because the reward measures the constraint, and the mask already guarantees the constraint. A reward that only the letter can satisfy invites any text that satisfies it. The post's early runs show how many forms the cheating can take: single-letter streams, repeated sentences, and degenerate loops. Adding a penalty for the model drifting from its starting point (a KL penalty, see the background links) did not save those runs, because it was small relative to the other rewards.

The lesson the post draws is to read samples, not only reward curves.

The working reward comes from a judge that looks at the whole answer. For each sampled answer the judge returns a coherence score $c$ and a responds score $r$, each from 1 to 5, and the reward is

$$R=\frac{(c-1)+(r-1)}{8}\in[0,1].$$

A (1, 1) answer earns 0 and a (5, 5) answer earns 1. The mask stays on while the model generates the answers, so the reward never has to check for "e". The judge only has to ask whether the text is any good.

### GRPO: standardize the reward inside a group

GRPO (see [DeepSeekMath](/grpo/)) needs no value network. For each prompt, the model samples a group of answers, scores each, and computes each answer's advantage as how far its reward sits from its group's mean, in units of the group's standard deviation:

$$A_i=\frac{R_i-\bar R}{\operatorname{std}(R)}.$$

Every token in an answer is pushed up (positive $A_i$) or down (negative $A_i$) in proportion to its answer's advantage.

The post's settings: 4 prompts per step, a group of 8 answers per prompt, temperature 0.7, answers capped at 300 tokens, LoRA rank 16, learning rate 2e-5, a KL weight of 0.02, and 300 steps. The post says the KL term had no noticeable effect and is probably unnecessary. With 4 prompts and groups of 8, a step scores 32 answers.

Here is a group of four answers with judge scores $(c,r)$. This is our example, not the post's:

| Answer | $(c,r)$ | Reward | Advantage |
| --- | --- | --- | --- |
| 1 | (5, 5) | 1.00 | +1.15 |
| 2 | (4, 4) | 0.75 | +0.23 |
| 3 | (5, 3) | 0.75 | +0.23 |
| 4 | (2, 2) | 0.25 | -1.61 |

The mean reward is 0.6875 and the population standard deviation is 0.2724. A group in which every answer scores the same has a standard deviation of 0, every advantage is 0, and the step teaches the model nothing. That is why the post checks that groups contain variety: on 50 prompts with 16 tries each, at least one usable answer appeared for 78% of prompts under the mask, and for 64% under the SFT adapter.

Edit the scores in this cell and run it to see the advantages move.

<!-- cell: advantage -->

### The sum rewards half-answers

Answers 2 and 3 in the table tie at 0.75. Only answer 2 is usable, because usable needs both scores to be at least 4, and answer 3 has a responds score of 3. The reward is a sum, so a high score on one axis pays for a low score on the other. The usability test is a minimum over the two axes.

The post sees the consequence in the masked model. Over GRPO training, the median answer length fell from 52 to 18 tokens (the mean from 92 to 42). Usable answers rose from 26% to 43%, but all of that gain came from coherence. The number of clean answers that skip part of the question rose from 24 to 66 out of 200. A short, tidy, partial answer earns a decent reward, so the model learned to write those.

The SFT adapter also shrank, from a median of 53 tokens to 24, but both scores rose and the usable rate rose at every answer length. The post therefore does not read its gain, 35% to 60.5%, as only a length effect.

<div class="widget" data-widget="group-scorer"></div>

### Two fixes did not stop the shrinking

The post tried two fixes, each as a 150-step run from the SFT adapter:

- Use the lower of the two scores as the reward, so a (5, 3) answer earns less than a (4, 4) answer.
- Divide each answer's loss by the 300-token cap instead of by its own length. Dividing by its own length gives each token of a short answer a bigger push.

Neither raised the usable rate, and neither stopped the shrinking. The median length at step 150 was 30 and 38 tokens, against 33 for the original run, and 53 before RL. The author's explanation is that, under the mask, every extra word risks a forced odd word, so shorter answers score better on their own merits. Take that as the post's hypothesis rather than a tested result.

So the min is the right shape of reward for this definition of usable, but the post does not show that it fixes the drift. Our scaling of the min in the widget, $(\min(c,r)-1)/4$, is our choice; the post says only "the lower of the two scores".

### What the training bought

The post resamples the 50 evaluation prompts 1,000 times to put a range on each gain (a paired bootstrap, middle 90%).

| Start | Usable rate before | After RL | Gain (90% range) |
| --- | --- | --- | --- |
| Masked base model | 26% | 43% | +17 (+8 to +26) |
| Masked SFT adapter | 35% | 60.5% | +25.5 (+17.5 to +33.5) |

Most of the gain arrives by step 125. In total the final adapter produced 121 usable answers out of 200, against 52 for the masked base model and 0 for the unmasked one. The author counts it as "from about 1 in 4 to about 6 in 10".

### How a 1.7B model fits on a 16 GB GPU

The post's appendix does a napkin calculation for a 2B-parameter model in bf16 with batch size 4, sequence length 2048, a 150k vocabulary, and AdamW with fp32 states:

| Bucket | Formula | Size |
| --- | --- | --- |
| Weights | $2\text{B}\times 2$ bytes | 4 GB |
| Gradients | $2\text{B}\times 2$ bytes | 4 GB |
| Optimizer states | $2\text{B}\times 4\times 2$ bytes | 16 GB |
| Activations | $4\times 2048\times 2048\times 2\times 12\times 28$ bytes | 11.3 GB |
| Logits | $4\times 2048\times 150000\times 2$ bytes | 2.46 GB |

That totals 37.7 GB, which the post rounds to about 40 GB. Three changes remove most of it:

- **LoRA** trains $\Delta W=BA$ instead of $W$. For a $2048\times2048$ matrix at rank 16, that is $2\times2048\times16=65{,}536$ parameters rather than $4{,}194{,}304$, which is 64 times fewer. Across the model it trains about 1% of the parameters, about 20 million. The frozen weights still take 4 GB.
- **Gradient checkpointing** stores only each layer's input (about 0.9 GB for 28 layers) plus one layer's intermediate values (about 0.4 GB), and recomputes the rest. The cost is an extra forward pass, usually 20% to 30% slower.
- **Chunked cross-entropy** (for example Liger Kernel or Cut Cross-Entropy) avoids materializing the full logits tensor, so that bucket falls to a few hundred megabytes.

The post reports about 7 to 8 GB with CUDA overhead for the result and a peak of 10 to 13 GB during the RL runs, inside a 15 GB cap. It does not break the 7 to 8 GB down. The post also mentions QLoRA, which stores the frozen weights in 4 bits, as a way to reduce the 4 GB weights bucket to about 1 GB.

<div class="widget" data-widget="memory-calc"></div>

## Rebuild it

Our reconstruction keeps the mechanism and drops the scale. The "model" is a table of next-word logits over 35 words, started from a "pretrained" table that mildly prefers grammatical pairs. A mask removes every word that contains an "e". A rule-based judge scores coherence as the fraction of grammatical word pairs and responds as length. The reward is the post's sum formula, and GRPO uses 8 answers per prompt and 4 prompts per step, for 150 steps. The cell then repeats the run with a min reward.

The numbers below come from our toy. They are not the post's results, and the toy grammar is hand-made, so the high usable rate after training says little about how hard the real task is. One coincidence: the masked, untrained rate in the toy is also about 26%. It is not tied to the post's 26%, which came from a different setup.

<!-- cell: toy -->

<div class="widget" data-widget="toy-curves"></div>

Look at what the cell shows:

- Instruction alone gives almost no usable answers. Our toy samples "e" words at the rate the base table prefers them.
- The mask alone gives a large jump, as it does in the post, because it removes a whole class of failures.
- GRPO then lifts the usable rate much higher in the toy, as it does (by a smaller amount) in the post.
- The answers do not shrink here. Our judge scores responds from length only up to three words, so the toy has no reward for being short and no forced odd words that punish length. The post's drift needs the real model's behavior, and our toy does not reproduce it.

## Pitfalls

- **A mask is not a guarantee of quality.** Masked output can be e-free and nonsense. Check both parts of the definition of usable.
- **Letter-level rewards invite hacking.** Whenever a reward can be satisfied by a degenerate output, expect RL to find it. Read samples as well as curves.
- **The sum and the threshold disagree.** The reward adds the two scores but usable needs both to reach 4. The post sees short, partial answers rise under the sum, though it also finds that the min did not stop the shrinking.
- **Small evaluation.** The evaluation set is 50 prompts, each answered 4 times, with a single run per setting. The post gives ranges from resampling the prompts, but not from repeating training.
- **The judge is an LLM, and it ignores facts.** It does not penalize wrong facts, and the post notes that the reward favors half-answers.
- **The adapters still need the mask.** Both released adapters must run with the mask on to stay "e"-free.
- **Only two model sizes.** The 4B comparison shows 87.5% against 80.5% judged usable without the rule, but only 12.5% against 26% masked. With the rule, the smaller model did better under the mask. The post tested one family and two sizes.

## What the post doesn't say

- It does not break the 7 to 8 GB figure down by component. The sum of the listed pieces, about 6 GB, is our arithmetic.
- It does not test the KL penalty by removing it, so the claim that it is unnecessary rests on noticing no effect.
- It does not say why the 4B model fares worse under the mask. It describes the difference (dropping letters from planned words against swapping words) without a test of the cause.

## Further reading

- [Qwen, Minus the E](https://kevin-wu.ca/blog/qwn/), the original post.
- The post's code repository: [kevinyvv/qwen-3-1.7b-finetune-proj](https://github.com/kevinyvv/qwen-3-1.7b-finetune-proj), which holds the judge, the GRPO engine, and run records.
- The adapters on Hugging Face: [k33wu/qwen3-1.7b-no-e](https://huggingface.co/k33wu/qwen3-1.7b-no-e) (SFT then RL) and [k33wu/qwen3-1.7b-no-e-sft180](https://huggingface.co/k33wu/qwen3-1.7b-no-e-sft180) (SFT only).
- [DeepSeekMath](/grpo/) for GRPO, [LoRA](/lora/) for the adapters, and [Neural machine translation of rare words with subword units](/bpe/) for tokens.
