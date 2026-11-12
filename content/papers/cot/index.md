---
id: "cot"
title: "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
authors: "Wei et al."
year: 2022
venue: "NeurIPS 2022"
arxiv: "2201.11903"
url: "https://arxiv.org/abs/2201.11903"
track: nlp-llm
tier: core
est_hours: 2
status: full
summary: "The paper changes the few-shot exemplars from input-output pairs into input, chain of thought, output triples, so that the model writes out intermediate reasoning steps in natural language before its final answer. On arithmetic, commonsense, and symbolic tasks this lifts accuracy sharply for models of roughly 100 billion parameters or more, and PaLM 540B with eight exemplars beats a fine-tuned GPT-3 with a verifier on GSM8K. For smaller models the same prompts do not help or even hurt, which the authors describe as an emergent ability of scale."
related_course: "rl: reasoning models"
prereqs:
  - id: gpt3
    reason: "Chain-of-thought prompting is few-shot prompting with exemplars in the context and exact-match scoring, and GPT-3 supplies that setting and the finding that standard few-shot accuracy stays flat on reasoning tasks."
background: [prompting, exact-match-evaluation]
course_links:
  - title: "Reasoning with verifiable rewards"
    url: "https://rl.ifkash.dev/lessons/073-reasoning-with-verifiable-rewards"
  - title: "Process rewards and test-time search"
    url: "https://rl.ifkash.dev/lessons/074-process-rewards-and-search"
self_check:
  - q: 'Figure 1 shows a standard prompt and a chain-of-thought prompt for the same question. What differs between them, how many exemplars does the main arithmetic experiment use, and does any model weight change?'
    a: 'Only the exemplars differ. Each standard exemplar is an (input, answer) pair, and each chain-of-thought exemplar is an (input, chain of thought, answer) triple, with the chain written by hand. The main arithmetic experiments use a single set of eight hand-written exemplars for every benchmark except AQuA, which gets four exemplars from its training set (Section 3.1). No weights change: the paper states that no language model was fine-tuned (Section 6).'
    see: gpt3
  - q: 'A model writes "Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11." The scorer uses exact match. Which string does it compare with the gold answer, and why does the chain not need to match anything?'
    a: 'It compares the text after "The answer is", here "11", with the gold answer "11". The exemplars end every chain with that phrase, so the answer can be extracted by a fixed rule (Figure 1). The chain itself is not scored: the paper scores the final answer only, and checks the chains by hand on small samples (Section 3.2).'
    see: exact-match-evaluation
  - q: 'Table 2 gives GPT-3 (text-davinci-002) on GSM8K at 350M parameters as 2.2 with standard prompting and 0.5 with chain of thought, and at 175B as 15.6 and 46.9. What is the chain-of-thought gain at each size, and what does the paper conclude?'
    a: 'The gain is 0.5 - 2.2 = -1.7 points at 350M and 46.9 - 15.6 = +31.3 points at 175B. The paper concludes that chain-of-thought prompting is an emergent ability of model scale: it does not help small models, which write fluent but illogical chains, and gives gains at about 100B parameters and above (Section 3.2).'
    see: prompting
---

## The problem

By 2022, scaling a language model had steadily improved few-shot prompting, but not for every kind of task. The paper opens with this observation: "scaling up model size alone has not proved sufficient" for arithmetic, commonsense, and symbolic reasoning (Section 1). The [GPT-3 page](/gpt3/) covers the standard recipe, where $K$ solved examples go in the prompt and the model completes the next one. The recipe works for tasks that a model can answer in a single step, and it works poorly for tasks where a person would work through several steps.

Consider a concrete failure. Figure 1 gives the model a prompt that contains one solved question, "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?", with the answer "The answer is 11." The test question is "The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?" Under standard prompting the model answers "The answer is 27." The right answer is $23 - 20 + 6 = 9$. The model had to find three numbers, apply two operations, and write one number, all within the computation that produces a single token.

The authors name two earlier ideas that each fix half of this (Section 1). Training or fine-tuning a model to write natural-language rationales helps (Ling et al., 2017; Cobbe et al., 2021), but it needs a large set of high-quality rationales, which is far more expensive to collect than input-output pairs. Few-shot prompting needs no training set, but it "works poorly on tasks that require reasoning abilities". The paper combines them: keep the few-shot setting, and put a rationale in each exemplar.

## Key ideas

The paper has no numbered equations, so the math on this page is this page's notation and not the paper's. It writes an exemplar as the triple $\langle \text{input}, \text{chain of thought}, \text{output} \rangle$ (Section 1), which this page abbreviates to $(x, r, y)$. $K$ is the number of exemplars, as on the GPT-3 page. The paper's own numbers come from its appendix tables, and each is labeled with its table.

### Exemplars are input, chain of thought, output triples

**Intuition.** A chain of thought is "a series of intermediate natural language reasoning steps that lead to the final output" (Section 1). Standard prompting shows the model $K$ exemplars of the form (question, answer), and the model answers the new question directly. Chain-of-thought prompting keeps the same $K$ questions and changes only the answers: each one now holds the steps that lead to it. The model does what the exemplars do, so it writes a chain for the new question, and then the answer. Nothing else about the prompt changes, and the paper treats this as an advantage: a single frozen model can handle many tasks with no training set (Section 1, Section 2, property 4).

**Math.** Write the exemplar set as $E_{\text{std}} = ((x_1, y_1), \dots, (x_K, y_K))$ for standard prompting and $E_{\text{cot}} = ((x_1, r_1, y_1), \dots, (x_K, r_K, y_K))$ for chain of thought. The model's weights $\theta$ are fixed in both cases, and the prompt is the exemplars followed by the test input $x$. Standard prompting decodes the answer directly. Chain-of-thought prompting decodes a chain and then the answer, and the answer conditions on the chain it just wrote:

$$
p_\theta(r, y \mid x, E_{\text{cot}}) = p_\theta(r \mid x, E_{\text{cot}}) \; p_\theta(y \mid x, r, E_{\text{cot}})
$$

Each factor is the usual product over tokens from the GPT-3 page. The paper samples with greedy decoding, so $r$ is one chain, the most likely token at each position, and not a sum over all chains (Section 3.1). Averaging over many sampled chains is a follow-up that the paper only cites (Wang et al., 2022a, in Section 3.1). A scorer extracts the answer $a(r, y)$ from the generated text, and the score is exact match, $\mathbb{1}[a(r, y) = y^\star]$ for the gold answer $y^\star$.

For the arithmetic benchmarks, the authors "manually composed a set of eight few-shot exemplars with chains of thought", and used the same set for every benchmark except the multiple-choice AQuA, which gets four exemplars with solutions from its training set (Section 3.1, with the full prompts in Appendix Tables 20 and 21). They say these exemplars "did not undergo prompt engineering". Figure 3 shows triples for the other tasks, and the exemplars always end with a fixed phrase such as "The answer is" or "So the answer is".

**Tiny example, computed by `scripts/trace_example.py`.** The script takes the two model outputs from Figure 1 and checks each. The toy token count is one token per word.

| Output | Toy tokens | Extracted answer | Correct |
|---|---|---|---|
| Standard: "The answer is 27." | 4 | 27 | no |
| Chain of thought: the cafeteria chain | 37 | 9 | yes |

The chain holds two equations, $23 - 20 = 3$ and $3 + 6 = 9$, and both check out. The chain costs 33 more generated tokens than the bare answer. The paper's first listed property is that this extra output is where "additional computation can be allocated to problems that require more reasoning steps" (Section 2). The next idea examines that claim. Exercise 1 asks you to build the checker that the script uses, and to score whole traces with it.

### Intermediate tokens as extra computation

**Intuition.** A language model produces each token with a fixed amount of computation: one forward pass through $L$ layers. When the answer is the next token, the pass must find the whole solution at once. When the model writes the steps first, each step is its own pass, which reads the earlier steps from the context. The paper offers this as a reason in principle (Section 2, property 1), and then tests whether extra tokens alone explain the gain (Section 3.3).

**Math.** Suppose a task needs $k$ dependent operations, and one forward pass can reliably perform $b$ of them (the "budget"). This is a model for thinking about the claim and not a measurement of a real network, and the paper doesn't state it. Answer-only prompting has one pass, so it can solve the problem only when $k \le b$. A chain with one line per $b$ operations takes $\lceil k / b \rceil$ passes, each starting from the value that the previous line wrote down. Now give every pass an independent chance $\epsilon$ of a slip. Because a slip is written into the chain and later steps build on it, the chain is correct only if no pass slips:

$$
P(\text{chain correct}) = (1 - \epsilon)^{\lceil k / b \rceil}
$$

The chain trades an impossible problem for a long one. Accuracy falls with each extra line, which is the cost of the "no guarantee of correct reasoning paths" that the paper lists as its third limitation (Section 6).

**Tiny example, computed by `scripts/budget_curve.py`.** With $b = 1$ and $\epsilon = 0.05$, 20,000 random problems per length, seed 0:

| $k$ | Answer-only | Stepwise | $(1 - \epsilon)^k$ |
|---|---|---|---|
| 1 | 0.951 | 0.949 | 0.950 |
| 2 | 0.015 | 0.902 | 0.902 |
| 4 | 0.021 | 0.818 | 0.815 |
| 8 | 0.004 | 0.660 | 0.663 |

At $k = 1$ the two are the same, because the problem fits in one pass. Beyond the budget, answer-only is near zero, and a few correct answers appear only when a skipped step is an identity on the value, such as a multiplication of 0. Stepwise decays as predicted. In exercise 2 you build both methods and reproduce these rows.

The paper then tests whether the benefit is just more tokens, using LaMDA 137B and three ablations (Section 3.3, Figure 5, Appendix Table 6). **Equation only** prompts for the equation before the answer. **Variable compute only** prompts for a run of dots as long as the equation would be. **Reasoning after answer** puts the chain after the answer, which tests whether the chain only activates knowledge. The table is computed by `scripts/ablation_table.py`, with accuracy in percent and the difference from standard prompting in parentheses:

| Variant | GSM8K | SVAMP | ASDiv | MAWPS |
|---|---|---|---|---|
| Standard prompting | 6.5 | 29.5 | 40.1 | 43.2 |
| Equation only | 5.4 (-1.1) | 35.1 (+5.6) | 45.9 (+5.8) | 50.1 (+6.9) |
| Variable compute only | 6.4 (-0.1) | 28.0 (-1.5) | 39.4 (-0.7) | 41.3 (-1.9) |
| Reasoning after answer | 6.1 (-0.4) | 30.7 (+1.2) | 38.6 (-1.5) | 43.6 (+0.4) |
| Chain of thought | 14.3 (+7.8) | 36.7 (+7.2) | 46.6 (+6.5) | 57.9 (+14.7) |

Dots gain nothing, a chain written after the answer gains nothing, and equations help only where the problems are short. The paper reads this as evidence that "variable computation by itself is not the reason for the success", and that natural-language intermediate steps matter. The toy model agrees on dots: a pass that writes no value can't hand a result to the next pass. It also shows why the order matters, because a chain written after the answer can't feed the answer. Equations alone don't help on GSM8K, because the questions are "too semantically challenging" to translate straight into one equation (Section 3.3, Appendix A.4).

Table 6 gives chain of thought on SVAMP as 36.7, while Table 2 and Table 1 give 37.5 for the same model and benchmark. The paper doesn't explain the difference, and this page keeps each table's number with its table.

### Emergence with model scale

**Intuition.** The first of the three takeaways in Section 3.2 is that chain of thought "is an emergent ability of model scale", which the authors define in the sense of Wei et al. (2022b): it "does not positively impact performance for small models, and only yields performance gains when used with models of ∼100B parameters". Small models "produced fluent but illogical chains of thought, leading to lower performance than standard prompting". The other two takeaways are that gains are larger on harder problems (GSM8K gains more than doubled for the largest GPT and PaLM models, while the easiest subset of MAWPS gained little or lost), and that the largest models compare well with fine-tuned systems.

**Math.** For a model family with sizes $s_1 < s_2 < \dots$ and accuracies $A_{\text{std}}(s)$ and $A_{\text{cot}}(s)$, define the gain $g(s) = A_{\text{cot}}(s) - A_{\text{std}}(s)$. This is this page's formalization of the paper's wording. Say that chain of thought has emerged at $s_i$ when $g(s_j) > 0$ for every $j \ge i$ and $i$ is the smallest such index. Requiring every larger size to gain is stricter than finding the first positive gain, because a benchmark can show a small positive gain at a small size by noise.

**Tiny example, computed by `scripts/scale_gains.py`.** The accuracies are Table 2's, for GSM8K, and the gains are the chain-of-thought column minus the standard column:

| Family | Sizes (billions of parameters) | Gains (points) | Emergence |
|---|---|---|---|
| LaMDA | 0.42, 2, 8, 68, 137 | -2.2, -1.7, -1.6, +2.5, +7.8 | 68B |
| GPT-3 | 0.35, 1.3, 6.7, 175 | -1.7, -1.9, -1.6, +31.3 | 175B |
| PaLM | 8, 62, 540 | -0.8, +20.3, +39.0 | 62B |

Every small model loses points, and every large one gains. At 175B, GPT-3 reaches 46.9 against 15.6, which is 3.01 times the standard accuracy, and PaLM 540B reaches 3.18 times (56.9 against 17.9). The "about 100B" in the text is a loose statement, because on this data the switch happens between two sizes in each family, and the sizes tested are sparse: between 8B and 62B for PaLM, 8B and 68B for LaMDA, and 6.7B and 175B for GPT-3. The paper's GPT-3 models are `text-ada-001`, `text-babbage-001`, `text-curie-001`, and `text-davinci-002`, which the authors say "presumably correspond to" InstructGPT models of 350M, 1.3B, 6.7B, and 175B parameters (Section 3.1), so they aren't the original GPT-3 models from the previous page.

The benchmark matters too. PaLM on SVAMP gains +1.7 at 8B, -1.5 at 62B, and +9.6 at 540B, so the emergence point is 540B under the all-larger rule, and 8B under a first-positive rule. Exercise 3 asks you to code the rule and to see where the two disagree.

### Chains can be wrong, and they extend to longer inputs

**Intuition.** A chain of thought is text that the model generates, and nothing forces it to be correct. The authors read chains from LaMDA 137B on GSM8K by hand (Section 3.2, Appendix D). They also test whether a chain can be extended to inputs longer than the exemplars, using two toy symbolic tasks, and the tasks work for a reason that is easy to see in a script (Section 5).

**Math.** The paper has no formulas here, only counts. From 50 sampled examples with correct final answers, all but two had a logically and mathematically correct chain, and the two "coincidentally arrived at the correct answer". From 50 samples with wrong answers, 46 percent of the chains were almost correct, "barring minor mistakes (calculator error, symbol mapping error, or one reasoning step missing)", and 54 percent had major errors in semantic understanding or coherence. The authors also add a post-hoc external calculator that evaluates the equations in a chain, which raises GPT-3 175B on GSM8K from 46.9 to 49.6 and PaLM 540B from 56.9 to 58.6 (Appendix B, Table 1), so some failures are arithmetic slips and not reasoning errors.

For length, the exemplars of the last-letter task have two-word names, and the test sets have two (in domain), three, and four words (out of domain, or OOD). The chain template is the same at any length: one "The last letter of ... is ..." sentence per word, then the concatenation. The coin-flip exemplars have two people who each flip or don't flip the coin, and the tests go to three and four. Standard prompting fails on both OOD tasks, and chain of thought "facilitates length generalization beyond seen chains of thought for language models of sufficient scale" (Section 5).

**Tiny example, computed by `scripts/error_analysis.py`.** The sample counts follow from the percentages: 50 correct answers split into 48 sound chains and 2 coincidences, and 50 wrong answers split into 23 almost-correct and 27 major-error chains. The script also builds the last-letter chain of Figure 3 for longer names:

| Name | Answer | Chain length (words) | Lookup steps |
|---|---|---|---|
| Lady Gaga | ya | 23 | 2 |
| Amy Brown Lee | yne | 30 | 3 |
| Amy Brown Lee Ng | yneg | 37 | 4 |

Each extra word adds one fixed-size step, so the chain grows by 7 words per word. The model's job on a longer name is to repeat a step it has already seen written. For PaLM 540B, Table 5 gives last-letter accuracy of 99.4 on two words, 94.8 on three, and 63.0 on four with chain of thought, against 7.6, 0.2, and 0.0 with standard prompting. On coin flip, chain of thought gives 100.0, 98.6, and 90.2, against 98.1, 49.3, and 54.8. The paper adds that these tasks are "toy tasks" in which the exemplars already show the full structure, and that small models still fail at them (Section 5).

Exercise 1's `verify_chain` is a small version of the calculator check. It tests the arithmetic of each equation, and it can't tell whether the equations answer the right question. The scorer's `lucky_correct` number is the share of correct answers whose chain has a wrong step, which is the 2 in 50 that the authors found by hand.

<!-- exercises -->

## Pitfalls

- **Treating the chain as an explanation of the computation.** The paper says that the chain "provides an interpretable window into the behavior of the model", and also that "fully characterizing a model's computations that support an answer remains an open question" (Section 2). Its limitations section says that emulating a human thought process "does not answer whether the neural network is actually 'reasoning'" (Section 6).
- **Scoring only the answer.** A correct answer can come with a wrong chain: two of 50 in the manual check. Exact match on the answer can't see that, and it counts a lucky chain as a success.
- **Checking only the arithmetic.** A verifier like the one in exercise 1 accepts "4 * 6 = 24" for the wire question even when the model has misread the question (Appendix Figure 10, the 62B output). The paper's error analysis separates calculator errors from semantic errors for this reason (Section 3.2, Appendix D).
- **Expecting gains at any scale.** The gains are negative for small models in Table 2, and on GSM8K the PaLM 8B gain is -0.8. Prompting a small model this way can be worse than not doing it.
- **Expecting gains on any task.** Appendix A.3 says that chain of thought helps most when the task needs multiple steps, the model is large, and the scaling curve is flat. The gains on SingleOp, the one-step subset of MAWPS, were negative or very small (Section 3.2, Appendix Table 3). GPT-3 also loses points on CSQA and StrategyQA (Appendix Table 4).
- **Reading the exemplars as a fixed recipe.** The authors wrote one set of eight and report a spread across annotators: on coin flip, 99.6 for annotator A and 71.4 for annotator C (Appendix A.2, Table 7). Prompt engineering "still does matter".
- **Counting the ablations as a proof.** The ablations in Table 6 are run on LaMDA 137B only, because additional queries for GPT-3 and PaLM were "limited and expensive". Figure 5 adds PaLM 540B for GSM8K, and no ablation covers GPT-3.
- **Mixing the two accuracy tables.** SVAMP chain of thought is 36.7 in Table 6 and 37.5 in Tables 1 and 2. Cite the table you took the number from.
- **Forgetting the cost.** A chain is generated at inference time, so each answer costs more tokens, and the benefit appears only in models that are "costly to serve in real-world applications" (Section 6).
- **Letting the verifier see more than one operator.** The regular expression in exercise 1 handles `a op b = c`. In "2 * 3 + 4 = 10" it would read "3 + 4 = 10" and report a false error, so a verifier for real chains needs a small expression parser.

## Further reading

- [Language Models are Few-Shot Learners](/gpt3/) is the previous page, which defines the few-shot setting that this paper changes.
- [Let's Verify Step by Step](/prm/), by Lightman et al., trains a reward model that scores each step of a chain, which is the idea behind exercise 1 at scale.
- [DeepSeek-R1](/r1/) trains the model with reinforcement learning to produce long chains of thought, instead of prompting for them.
- [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168), by Cobbe et al., introduces GSM8K and the fine-tuned-with-a-verifier baseline that PaLM 540B beats in Figure 2.
- [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171), by Wang et al., samples many chains and takes the majority answer, which the paper cites as a follow-up in Section 3.1.
- [Show Your Work: Scratchpads for Intermediate Computation with Language Models](https://arxiv.org/abs/2112.00114), by Nye et al., trains a model to write intermediate steps for programs and arithmetic, and the paper cites it in the related work (Section 7).
