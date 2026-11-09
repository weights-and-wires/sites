---
id: "r1"
title: "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"
authors: "DeepSeek-AI et al."
year: 2025
venue: "arXiv preprint (Nature 2025)"
arxiv: "2501.12948"
url: "https://arxiv.org/abs/2501.12948"
track: rl
tier: core
est_hours: 6
status: full
summary: "DeepSeek-R1-Zero applies GRPO directly to a pretrained base model with a reward that a program computes, an accuracy check plus a format check, and long step-by-step reasoning with self-checking emerges without any human-written reasoning traces. DeepSeek-R1 repairs the readability and language-mixing problems of that run with a four-stage recipe: a small cold-start fine-tune, reasoning RL, rejection sampling into a large fine-tuning set, and a final RL stage. The same data, about 800,000 samples, then teaches smaller open models by plain fine-tuning, which beats running the RL recipe on those small models."
related_course: "rl: reasoning models"
prereqs:
  - id: grpo
    reason: "Every RL stage of R1 runs GRPO unchanged, so you need the group-relative advantage and the clipped objective with a KL term to read Equations 1 to 3 here."
  - id: cot
    reason: "R1-Zero's thinking is a very long chain of thought that RL produces in place of the human-written exemplars that prompting needs."
background: [verifiable-rewards, distillation]
course_links:
  - title: "Reasoning with verifiable rewards"
    url: "https://rl.ifkash.dev/lessons/057-reasoning-with-verifiable-rewards"
  - title: "Verifiable rewards"
    url: "https://rl.ifkash.dev/lessons/056-verifiable-rewards"
self_check:
  - q: 'Four completions of one question earn rule rewards $[2, 1, 1, 0]$ under Equation 4. What are their GRPO advantages (Equation 3, population standard deviation)?'
    a: 'The mean is $1$ and the standard deviation is $\sqrt{0.5} \approx 0.7071$. The advantages are $[1.4142, 0, 0, -1.4142]$. The two middle completions sit exactly at the group mean, so they get no push in either direction.'
    see: grpo
  - q: 'Few-shot chain-of-thought exemplars improve a plain language model. What does the paper report for few-shot prompting of DeepSeek-R1, and what does it recommend?'
    a: 'Section 6 says few-shot prompting consistently degrades R1. It recommends a zero-shot prompt that describes the problem and the output format. The model already produces its own long chain of thought, so exemplars only get in the way.'
    see: cot
  - q: 'Why does Section 2.2 avoid neural reward models for the reasoning tasks, and what does it use?'
    a: 'The paper observes that neural reward models, outcome-based or process-based, are susceptible to reward hacking during large-scale RL, and that retraining them is expensive. It uses rule-based rewards: an answer checker or compiler for accuracy, and a tag check for format.'
    see: verifiable-rewards
---

## The problem

By late 2024, a pretrained model could solve harder problems when it wrote out its reasoning, either from few-shot exemplars (see [Chain-of-thought prompting](/cot/)) or from fine-tuning on human-written solutions. The paper's introduction names two limits of that route. Human reasoning traces are expensive and add the annotators' biases. And a model trained to imitate human reasoning has its performance capped by the human exemplars, which rules out reasoning patterns that humans don't use.

The paper asks whether reasoning can come from reinforcement learning alone, with the supervision reduced to one bit per answer: right or wrong. Consider a question such as "Compute $47 + 38$." A grader needs no reasoning trace to score it. The grader checks the final number. If a model samples many attempts and the policy gradient pushes toward the attempts that the checker accepted, does the model learn to think longer and check its work?

The answer in Section 2 is yes, with a cost. The model trained this way, DeepSeek-R1-Zero, raises its AIME 2024 pass@1 from 15.6% to 77.9% during training (Section 2.3, Figure 1a), and 86.7% with self-consistency over 16 samples. But its reasoning is hard to read and mixes languages (Section 3), and it is narrow: it is trained only on reasoning tasks. DeepSeek-R1 is the multi-stage recipe that fixes those two problems while keeping the reasoning (Section 3, Figure 2). Appendix F then asks whether small models should run this RL at all.

## Key ideas

The sections that follow use the paper's notation: a question $q$, a group of $G$ outputs $\{o_1, \dots, o_G\}$, rewards $r_i$, advantages $A_i$, the clip range $\varepsilon$, the KL coefficient $\beta$, and the reference policy $\pi_{ref}$. Section and equation numbers follow the arXiv v2 text, which is the Nature version. Earlier versions number them differently. Every computed number comes from a script in `scripts/`. Run any script from the `scripts/` folder to reproduce the output.

### Rule-based rewards: accuracy plus format

**Intuition.** R1-Zero never sees a reasoning trace or a learned judge. A program scores each completion on two questions. Is the answer right? Is the thinking wrapped in the requested tags? The two scores are added.

**Math.** The prompt template (Table 1) asks the model to put its reasoning in `<think>...</think>` and its answer in `<answer>...</answer>`. The reward is the sum of an accuracy term and a format term, "combined with the same weight" (Section 2.2, Equation 4):

$$
\text{Reward}_{rule} = \text{Reward}_{acc} + \text{Reward}_{format}
$$

The accuracy reward checks the final answer against ground truth: for math, the answer in a specified format such as a box, and for code, a compiler against test cases. The format reward checks the tags. The paper says it avoids neural reward models, outcome-based or process-based, for reasoning tasks because they are susceptible to reward hacking and costly to retrain.

The paper gives no code for either check. This page reads the format reward as a check on both tags, because the template shows both, and the paper's text names only `<think>` and `</think>` (Section 2.2). The exact rules are this page's choice, and Exercise 1 asks you to write them.

**Tiny example, computed by `scripts/rewards_example.py`.** The gold answer to "$47 + 38$" is 85. Five completions earn these rewards:

| Completion | Accuracy | Format | Total |
|---|---|---|---|
| Tags, right answer | 1 | 1 | 2 |
| Tags, wrong answer (75) | 0 | 1 | 1 |
| No tags, "The answer is 85." | 0 | 0 | 0 |
| Thinking tagged, answer outside the answer tags | 0 | 0 | 0 |
| `<answer>` before `<think>`, right answer | 1 | 0 | 1 |

The last row shows why the two terms are separate: a completion can be right in the wrong shape. Under Equation 4, such a completion earns one point, the same as a tagged wrong answer.

Exercise 1 asks you to write these verifiers.

### GRPO on whole completions

**Intuition.** For each question, sample a group of completions, score each one, and push the policy toward the completions that scored above the group's average. The paper reuses GRPO from [DeepSeekMath](/grpo/) without change. What is new is the setting: a base model, a rule reward, and thousands of steps.

**Math.** For each question $q$, GRPO samples $\{o_1, \dots, o_G\}$ from $\pi_{\theta_{old}}$ and maximizes (Section 2.1, Equation 1):

$$
\mathcal{J}_{GRPO}(\theta) = \mathbb{E}\left[ \frac{1}{G} \sum_{i=1}^{G} \left( \min\!\left( \frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{old}}(o_i \mid q)} A_i,\ \operatorname{clip}\!\left( \frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{old}}(o_i \mid q)},\ 1-\varepsilon,\ 1+\varepsilon \right) A_i \right) - \beta\, \mathbb{D}_{KL}\!\left( \pi_\theta \,\|\, \pi_{ref} \right) \right) \right]
$$

The KL term is the estimator of Equation 2, and the advantage is the group's standardized reward (Equation 3):

$$
\mathbb{D}_{KL}\!\left( \pi_\theta \,\|\, \pi_{ref} \right) = \frac{\pi_{ref}(o_i \mid q)}{\pi_\theta(o_i \mid q)} - \log \frac{\pi_{ref}(o_i \mid q)}{\pi_\theta(o_i \mid q)} - 1, \qquad
A_i = \frac{r_i - \operatorname{mean}(\{r_1, \dots, r_G\})}{\operatorname{std}(\{r_1, \dots, r_G\})}
$$

The notation differs from the [GRPO page](/grpo/) in two ways. This paper writes the ratio and the KL estimate once per output $o_i$, with no sum over tokens and no $1/|o_i|$, while DeepSeekMath writes them per token $o_{i,t}$ (its Equation 3). The R1 paper doesn't comment on the difference, and this page doesn't claim which form its code uses. The paper also doesn't say which standard deviation Equation 3 uses. The page's script uses the population standard deviation, as in the GRPO page. Appendix A.3 repeats the same objective as Equations 11 to 13.

The training settings for R1-Zero are in Section 2.1: learning rate $3 \times 10^{-6}$, $\beta = 0.001$, temperature 1, $G = 16$ outputs per question, and a maximum length of 32,768 tokens before step 8.2k and 65,536 afterward. A training step uses 32 unique questions, so a batch holds 512 outputs. The reference model is replaced with the latest policy every 400 steps, which is a way to let the policy travel far from its starting point (Appendix A.3 gives this reason).

**Tiny example, computed by `scripts/group_example.py`.** One group of $G = 8$ completions earns Equation 4 rewards $[2, 2, 1, 1, 0, 0, 2, 1]$. The mean is 1.125 and the standard deviation is 0.7806. The advantages are as follows:

| Reward | Advantage |
|---|---|
| 2 | $+1.1209$ |
| 1 | $-0.1601$ |
| 0 | $-1.4412$ |

The advantages sum to 0, because the baseline is the group mean. Format reward matters early and vanishes later. Once every completion in a group follows the format, the format reward is the same constant for all of them, and the advantages equal those of accuracy alone: $[1.291, 1.291, -0.7746, \dots]$ for three right answers in eight. So the format term shapes the policy until the model reliably writes the tags and then stops contributing.

`scripts/rollout_arithmetic.py` checks the batch arithmetic of Section 2.1: $32 \times 16 = 512$ outputs per step. A rollout of 8,192 outputs splits into 16 minibatches of 512, each used for one update, so a "step" is one update (the caption of Figure 1 defines a step as a single policy update). Ten thousand four hundred steps are 650 rollouts and about 5.3 million sampled completions.

Section 3.2.1 also states a clip ratio $\varepsilon$ of 10 for the first RL stage, and adds that a lower value can truncate the gradients of many tokens and hurt performance, while a higher one can destabilize training. Read literally, the clip interval is $[1-\varepsilon, 1+\varepsilon] = [-9, 11]$. The script shows that a ratio of 1.5 with a positive advantage passes through unclipped, where $\varepsilon = 0.2$ would cap it at 1.2. The paper doesn't explain the value or say whether it is a typo, so this page treats it as reported and flags it.

### What the RL run produced

**Intuition.** The reward says nothing about how long to think, yet response length grows and the model starts to second-guess itself.

**Math.** There is no new equation here. Section 2.3 reports the observations. Average response length grows steadily through training (Figure 1b). The model increasingly shows reflection and exploration of alternative solutions (Appendix C.2, Figure 9a). Table 2 shows an intermediate checkpoint that stops mid-solution, flags the moment in the first person, and re-derives the step. Figure 9b ties this to a sudden rise in the word "wait".

Two cautions apply. The paper calls the length growth "thinking time" and the "aha moment" a change in reasoning patterns. Both are descriptions of text statistics, and Appendix B.3.2 adds that the vivid reasoning patterns reflect engineered heuristics and not human-like intelligence in the model. And the prerequisite for the whole result is a strong base. Appendix G.1 reports that a 7B dense model and a 16B mixture-of-experts model, used as the RL base early in the project, failed to improve on AIME and tended to repeat themselves as responses grew.

### From R1-Zero to R1: four stages

**Intuition.** R1-Zero reasons well but writes badly, mixes English and Chinese, and was only trained on tasks with checkable answers. R1 adds stages that fix each problem in turn.

**Math.** The pipeline of Figure 2 has four training stages on DeepSeek-V3-Base:

1. **Cold start** (Section 3, Appendix B.3.2). A few thousand long chains of thought in a conversational, first-person style, produced by sampling R1-Zero, filtering to correct and readable outputs, and rewriting with DeepSeek-V3 and human checks. Fine-tune V3-Base on them. The result is the initial RL actor.
2. **Reasoning RL** (Section 3.2.1). GRPO with the Equation 4 rewards, plus a language-consistency reward (Equation 7), added directly to the final reward:

$$
\text{Reward}_{language} = \frac{\text{Num}(\text{Words}_{target})}{\text{Num}(\text{Words})}
$$

3. **Rejection sampling and SFT** (Appendix B.3.3). Sample responses from the stage-2 checkpoint, keep the correct and readable ones, add non-reasoning data, and fine-tune again. This stage uses a generative reward model (DeepSeek-V3 as a judge) for some data without rule-checkable answers.
4. **Second RL** (Section 3.2.2). Reasoning prompts use the rule reward, and general prompts use reward models for helpfulness and harmlessness (Section 3.1, Equations 5 and 6). The batch reward is (Equations 8 to 10):

$$
\text{Reward} = \text{Reward}_{reasoning} + \text{Reward}_{general} + \text{Reward}_{language}, \quad
\text{Reward}_{reasoning} = \text{Reward}_{rule}, \quad
\text{Reward}_{general} = \text{Reward}_{reward\_model} + \text{Reward}_{format}
$$

The second RL stage lowers the temperature to 0.7, runs 1,700 steps, and uses general data with the preference reward only in the last 400 steps. The paper reports that longer training on the model-based reward can lead to reward hacking (Appendix B.5, Figure 6).

**Tiny example, computed by `scripts/language_reward_example.py`.** Equation 7 needs a rule for what counts as a target-language word, and the paper doesn't give one. This page ignores tokens that contain no letter and calls a token English when all of its letters are ASCII. The chain of thought "First add the ones digits 7 and 8 to get 15 then carry the 1" has 11 counted words and reward 1.0000. Replace two phrases with Chinese ("我们先算" and "进位") and the count becomes 12 words, 10 of them English, so the reward is 0.8333.

`scripts/stage_table.py` reads Table 3. Each stage helps a different target:

| Stage | AIME 2024 pass@1 | Change |
|---|---|---|
| R1-Zero | 77.9 | |
| R1-Dev1 (cold-start SFT) | 59.0 | $-18.9$ |
| R1-Dev2 (reasoning RL) | 74.0 | $+15.0$ |
| R1-Dev3 (rejection sampling, SFT) | 78.1 | $+4.1$ |
| R1 (second RL) | 79.8 | $+1.7$ |

The drop at Dev1 is the paper's own finding: the small cold-start set degrades reasoning on AIME (Section 4). The last stage helps general benchmarks the most. On AlpacaEval 2.0, R1 scores 87.6 and Dev3 scores 62.1, a gain of 25.5 points.

`scripts/sft_mix.py` checks Table 5, the fine-tuning set of stage 3. The five domains sum to 804,745 samples. Of these, 626,933 are Math, Code, STEM, and Logic, and 177,812 are General (22.1%). The text of Appendix B.3.3 rounds these to "about 600k" reasoning and "about 200k" non-reasoning samples, so the table and the prose differ by a few percent. The same script shows why sampling several responses helps: if each sample passes the filters with probability $p = 0.3$, then 16 samples give at least one keeper with probability $1 - 0.7^{16} = 0.9967$, and 4 samples give $0.7599$.

Exercise 2 asks you to write the language-consistency reward and the filters of rejection sampling.

### Distilling into small models

**Intuition.** Instead of running RL on each small model, fine-tune it on the large model's reasoning traces.

**Math.** Distillation here is plain supervised fine-tuning, with no RL stage and no logit matching (Appendix F, Appendix B.4.3). The training set is the 800k samples of Appendix B.3.3, generated with R1. The recipe is 2 to 3 epochs, a cosine learning-rate schedule that decays to one-tenth of its start, a context of 32,768 tokens, and a batch size of 64. Table 6 lists six students, from Qwen2.5-Math-1.5B (initial learning rate $1 \times 10^{-4}$) to Llama-3.3-70B-Instruct ($2 \times 10^{-5}$).

The paper also tests whether RL alone could do the job on a small model (Appendix F.1, Table 16). It runs the R1-Zero recipe on Qwen2.5-32B-Base for over 10,000 steps and compares the result with the distilled model of the same size.

**Tiny example, computed by `scripts/stage_table.py`.** Using the Table 16 values:

| Benchmark | RL only (Qwen2.5-32B-Zero) | Distilled (R1-Distill-Qwen-32B) | Gap |
|---|---|---|---|
| AIME 2024 pass@1 | 47.0 | 72.6 | $+25.6$ |
| MATH-500 pass@1 | 91.6 | 94.3 | $+2.7$ |
| GPQA Diamond | 55.0 | 62.1 | $+7.1$ |
| LiveCodeBench | 40.2 | 57.2 | $+17.0$ |

The paper draws two conclusions (Appendix F.1). Distilling a stronger model into a smaller one works well, while small models that rely on large-scale RL need enormous compute and may not reach the performance of distillation. And advancing past the current frontier may still need larger base models and larger-scale RL. The comparison is one model size with one RL run, so treat it as evidence about this recipe, not as a law.

<!-- exercises -->

## Pitfalls

- **Treating the reward as a trained model.** In R1-Zero, the reward is a program: an answer check, a test run, and a tag check. A bug in the program is a bug in the training signal. If the answer check accepts any text that contains the gold number, the policy can learn to list many numbers. Exercise 1's mutants include a substring match.
- **Assuming the format reward is permanent.** Once every sample in a group follows the format, the format term adds a constant and the advantage ignores it. It teaches the tags and then drops out.
- **Reading $\varepsilon = 10$ as a normal clip.** The text of Section 3.2.1 gives a clip ratio of 10. With $1 - \varepsilon$ below zero, the lower clip can't bind. Check your own settings against the paper's text and the released code, and don't assume the value is a typo or a tuned choice without evidence.
- **Counting steps wrong.** A step is one policy update on one minibatch. The paper's 10,400 steps for R1-Zero are updates, not rollouts, and a rollout of 8,192 outputs is 16 updates.
- **Expecting the language reward to be free.** The paper says its own ablation (Appendix B.6) shows a slight drop in performance from the language-consistency reward (Section 3.2.1), and it keeps the reward because it makes the output more readable.
- **Mixing paper versions.** Section numbers, equation numbers, and tables can change between arXiv versions. This page reads the v2 text.
- **Prompting R1 with few-shot examples.** The paper recommends zero-shot prompts, because few-shot examples consistently degrade R1 (Section 6).
- **Concluding that RL is never useful for small models.** Appendix F.1 compares one distilled 32B model with one RL run at that size. It supports "distill first," and it doesn't show that RL on top of distillation can't help. The paper says that adding RL to the distilled models could boost them substantially and leaves the experiment to others.
- **Reading the toy loop as the algorithm.** The exercise omits the KL term and the clip, because each group is used once. It shows the learning signal, not the engineering of a 16-sample, 32,768-token rollout.

## Further reading

- The [paper's arXiv page](https://arxiv.org/abs/2501.12948) and the released models at [huggingface.co/deepseek-ai](https://huggingface.co/deepseek-ai), which the paper links.
- [DeepSeekMath](/grpo/) for the GRPO algorithm that every RL stage here uses, and [Chain-of-thought prompting](/cot/) for the behavior that R1-Zero learns without exemplars.
- Hinton, Vinyals, and Dean, "Distilling the Knowledge in a Neural Network" (2015), which the paper cites for the idea of distillation. Note that R1's distillation is fine-tuning on generated text, not matching soft labels.
