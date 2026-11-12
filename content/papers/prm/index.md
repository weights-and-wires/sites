---
id: "prm"
title: "Let's Verify Step by Step"
authors: "Lightman et al."
year: 2023
venue: "ICLR 2024"
arxiv: "2305.20050"
url: "https://arxiv.org/abs/2305.20050"
track: rl
tier: extended
est_hours: 4
status: full
summary: "The paper compares two ways of training a reward model that checks a model's math solutions: outcome supervision, which labels only the final answer, and process supervision, which labels every reasoning step up to the first mistake. A process-supervised reward model (PRM) scores a solution as the product of its step correctness probabilities, and picking the best of 1860 sampled solutions with it solves 78.2% of a 500-problem MATH test subset, against 72.4% for an outcome-supervised model and 69.6% for majority voting. The authors also release PRM800K, 800 thousand human step labels, and show that choosing which solutions to label with active learning makes the labels about 2.6 times more data efficient."
related_course: "rl: reasoning models"
prereqs:
  - id: instructgpt
    reason: "The PRM and the outcome-supervised baseline are both reward models trained on human or automatic labels, and InstructGPT supplies the reward-model setting that this paper refines from one score per response to one score per step."
  - id: cot
    reason: "The generator writes its solutions as a chain of thought with one step per line, and that step structure is what the PRM scores and the labelers judge."
background: [process-vs-outcome-supervision, best-of-n]
course_links:
  - title: "Process rewards and test-time search"
    url: "https://rl.ifkash.dev/lessons/074-process-rewards-and-search"
  - title: "Credit assignment for LLMs"
    url: "https://rl.ifkash.dev/lessons/075-credit-assignment-for-llms"
  - title: "Reward models"
    url: "https://rl.ifkash.dev/lessons/067-reward-models"
self_check:
  - q: 'A solution has five steps and the first mistake is in step 3. Which step labels does process supervision collect, and what does outcome supervision collect?'
    a: 'Process supervision labels steps 1 and 2 as correct and step 3 as incorrect, then stops (Section 2.6). Outcome supervision gives one label for the whole solution, here "wrong", and the ORM uses that same target at every position (Appendix E). The outcome label cannot say that steps 1 and 2 were fine.'
    see: process-vs-outcome-supervision
  - q: 'You sample $N = 5$ solutions and a scorer ranks them. How is the answer chosen, and how does that differ from majority voting?'
    a: 'Best-of-$N$ keeps the one solution with the highest score and reports its final answer (Section 2.1). Majority voting ignores scores and reports the final answer that most of the $N$ solutions share. A scorer can pick a rare correct answer that voting outvotes, and voting can beat a poor scorer.'
    see: best-of-n
  - q: 'The PRM gives a solution three step probabilities of $0.9$, $0.9$, and $0.9$. What is the solution score under the paper''s choice, and what would the minimum give?'
    a: 'The paper multiplies the step probabilities, so the score is $0.9^3 = 0.729$, the probability that every step is correct (Section 2.6). The minimum gives $0.9$. The product is lower for longer solutions, which Appendix F.2 calls a slight bias against solutions with more steps.'
    see: instructgpt
---

## The problem

A language model that writes its reasoning step by step (see [Chain-of-thought prompting](/cot/)) can still derail. One wrong step is enough, and the rest of the solution can look fluent. The paper notes that in multi-step reasoning a single logical error can derail a much larger solution (Section 1).

A common fix is to train a reward model that discriminates good solutions from bad ones, then sample many solutions and keep the one it ranks highest (Section 1). The question is what to train that reward model on. Take this made-up five-step solution to a math problem.

1. Set up the equation correctly.
1. Simplify it correctly.
1. Factor an expression wrongly.
1. Solve the wrong factorization correctly.
1. Report an answer that doesn't match the key.

An outcome-supervised reward model (ORM) sees one label for this solution, "wrong answer". It has to work out for itself that step 3 is the problem, and that steps 1 and 2 were fine. The paper calls this a credit-assignment task and says it is hard, because on hard problems most sampled solutions contain an error somewhere, so a negative label carries little information (Section 6.1).

Outcome labels have a second flaw. A solution can reach the right answer through a wrong step, and the label says "correct". The paper calls these false positives (Section 2.5).

A process-supervised reward model (PRM) is trained on a label for each step. It fixes both flaws, at the price of human labor, because no program can check a step the way it checks a final answer (Section 2). Earlier work by Uesato et al. (2022) compared the two on grade-school math and found similar final-answer error rates. This paper repeats the comparison with a stronger base model, much more human feedback, and the harder MATH dataset (Section 1). It finds that process supervision wins.

Two scope notes help you read the results. The paper never trains the generator with reinforcement learning (RL). It fixes one generator and measures how well a reward model picks the best solution from many samples (Section 2.1). And the headline test set is a 500-problem subset of MATH, not the full test set of 5,000 problems (Appendix C).

## Key ideas

The paper has no numbered equations, so this page names the formulas it writes down. A problem is $x$ and a sampled solution is $s = (s_1, \dots, s_k)$, a sequence of $k$ steps. The PRM's probability that step $i$ is correct is $p_i$. Every number below comes from a script in `scripts/`. Run a script from the paper folder to reproduce it.

### Process supervision and outcome supervision

**Intuition.** Outcome supervision asks "is the final answer right?" once per solution. Process supervision asks "is this step right?" once per step. The second question is more work for a human, and it points at the exact place where a solution goes wrong.

**Math.** An ORM is trained to predict whether a solution is correct. The paper follows Cobbe et al. (2021): the target is the same at every token of the solution, and at test time the ORM's prediction at the final token is the solution's score (Section 2.5, Appendix E). A PRM predicts the correctness of each step after the last token of that step. The prediction is a single token, and training maximizes the log-likelihood of the target token, which is the ordinary language-model pipeline (Section 2.6). In this page's notation, with a label $\ell_i \in \{\text{positive}, \text{neutral}, \text{negative}\}$ and $m$ the index of the first negative step, the loss is

$$
\mathcal{L}_{\text{PRM}} = -\sum_{i=1}^{\min(m,\,k)} \log q_\theta(\ell_i \mid x, s_1, \dots, s_i)
$$

The sum stops at the first mistake. The paper chooses to supervise only up to that step, so that the two kinds of supervision carry the same information on correct solutions, and process supervision adds only the location of the mistake on incorrect ones (Section 2.6). It also keeps labeling cost similar, because deciding that a solution is correct requires finding its first mistake anyway.

Human labelers mark each step positive, negative, or neutral (Section 2.4). Neutral means ambiguous, such as a step that is subtly misleading. The paper allows it to defer the choice: at test time it can count neutral as positive or as negative (Appendix F.2).

**Tiny example, computed by `scripts/step_labels.py`.** Take three solutions. Each row shows what each kind of supervision records.

| Solution | Steps correct | Final answer | Process labels | Outcome labels |
|---|---|---|---|---|
| Error at step 3, another at step 5 | 1, 1, 0, 1, 0 | wrong | 1, 1, 0 | 0, 0, 0, 0, 0 |
| No error | 1, 1, 1, 1 | right | 1, 1, 1, 1 | 1, 1, 1, 1 |
| Right answer, flawed step 2 | 1, 0, 1 | right | 1, 0 | 1, 1, 1 |

In row 1, the process labels stop at the first 0 and say nothing about step 5. The outcome labels are five zeros that don't say where. Row 3 is a false positive: the outcome label says every position is fine, and the process labels catch step 2.

The script also shows why a negative outcome label is weak evidence. With 10 steps and a 10% error rate per step, the chance that every step is right is $0.9^{10} = 0.3487$. With 20 steps and a 20% error rate, it is $0.0115$. On hard problems almost every sample fails, so "wrong" says little about which steps to blame (Section 6.1). Exercise 3 asks you to write these labels.

### Scoring a solution with a PRM

**Intuition.** To rank two solutions, you need one number per solution. The PRM gives one probability per step. The paper takes the chance that no step is wrong.

**Math.** The paper defines the PRM score of a solution as "the probability that every step is correct under the PRM" and implements it as the product of the step correctness probabilities (Section 2.6):

$$
S_{\text{prod}}(s) = \prod_{i=1}^{k} p_i
$$

Appendix F.2 describes the alternative, the minimum, and a second choice about neutral labels. The step-level score $p_i$ is the probability of the positive label, and a neutral label counts either as positive ($p_i = P(\text{pos}) + P(\text{neu})$) or as negative ($p_i = P(\text{pos})$):

$$
S_{\min}(s) = \min_{i} p_i
$$

Table 4 reports best-of-1860 accuracy for all four combinations:

| Reduction | Neutral = positive | Neutral = negative |
|---|---|---|
| Product | 78.2% | 77.4% |
| Minimum | 77.6% | 77.8% |

The best is the product with neutral counted as positive, which the paper uses from then on. The spread across the four is 0.8 points, which is 4 problems out of 500 (computed by `scripts/paper_numbers.py`). The paper says the difference between the strategies is minor. It also notes that the product is slightly biased against solutions with more steps.

**Tiny example, computed by `scripts/score_aggregation.py`.** Solution A has step probabilities $0.95, 0.90, 0.92$. Its product is $0.7866$ and its minimum is $0.9000$. Now compare a short and a long solution:

| Solution | Steps | Product | Minimum |
|---|---|---|---|
| Short | 0.90, 0.95 | 0.8550 | 0.9000 |
| Long | ten steps of 0.98 | 0.8171 | 0.9800 |

The product prefers the short solution, because each extra step multiplies in a number below 1. The minimum prefers the long one, because its weakest step is stronger. This is the length bias. The script also shows the neutral choice: a step with $P(\text{pos}) = 0.50$ and $P(\text{neu}) = 0.30$ scores $0.80$ or $0.50$, and four such steps multiply to $0.4096$ or $0.0625$. Exercise 1 asks you to implement both reductions.

### Best-of-N search against majority voting

**Intuition.** Sampling many solutions and keeping one trades compute for accuracy. The better the scorer, the more of that compute pays off. Majority voting is the baseline: it doesn't need a reward model, only agreement among the final answers.

**Math.** The paper evaluates a reward model by its ability to perform best-of-$N$ search over uniformly sampled solutions from the generator. For each test problem it keeps the solution ranked highest, grades its final answer automatically, and reports the fraction correct (Section 2.1):

$$
\hat s = \arg\max_{j = 1, \dots, N} S(s^{(j)}), \qquad \text{accuracy} = \Pr\big[\text{final answer of } \hat s \text{ is correct}\big]
$$

If each sample is correct with probability $p$ and the scorer is perfect, the accuracy is $1 - (1 - p)^N$, because the search succeeds when any sample is correct. A real scorer does worse, and a scorer that ignores correctness stays at $p$. The paper also tried reward-model-weighted voting, which adds up the scores of the solutions that share an answer, and reports that it didn't noticeably improve on the PRM alone (Section 3).

**Tiny example, computed by `scripts/best_of_n_toy.py`.** One problem, five sampled solutions, and the correct answer is 12:

| Solution | Final answer | ORM score | PRM product |
|---|---|---|---|
| c0 | 12 | 0.80 | 0.7866 |
| c1 | 15 | 0.85 | 0.1659 |
| c2 | 15 | 0.60 | 0.2511 |
| c3 | 12 | 0.70 | 0.2430 |
| c4 | 15 | 0.82 | 0.3110 |

The ORM picks c1 and answers 15, which is wrong. Majority voting also answers 15, three votes to two. The PRM picks c0 and answers 12. ORM-weighted voting still says 15 (2.2700 against 1.5000), while PRM-weighted voting says 12 (1.0296 against 0.7280). The PRM also ranks c3, which has the right answer and a weak step, only fourth of five. Exercise 2 asks you to implement the three selectors.

**What the paper measures.** At large scale, the generator and all reward models start from GPT-4 (Section 2.2). The PRM is trained on PRM800K. The ORM is trained on 100 uniformly sampled solutions per problem, an order of magnitude more data, and does not overlap with PRM800K (Section 3). With 1860 solutions per problem, the share of the 500 test problems solved is:

| Method | Best-of-1860 |
|---|---|
| Process-supervised reward model | 78.2% |
| Outcome-supervised reward model | 72.4% |
| Majority voting | 69.6% |

These are Figure 3's values, and `scripts/paper_numbers.py` computes the gaps: 5.8 points over the ORM and 8.6 over majority voting. The text says the ORM does slightly better than majority voting, the PRM beats both at every $N$, and the gap widens as $N$ grows (Section 3). On out-of-distribution questions from recent AP and AMC exams, the same ordering holds in aggregate with 100 samples: ORM 63.8%, PRM 72.9%, majority voting 61.3% (Section 5, Table 1).

**A simulation of the shape.** The curves in Figure 3 need a trained model, but a toy model reproduces the shape. `scripts/best_of_n_sim.py` is this page's own model and not the paper's. Each sample is correct with probability $p = 0.3$. The scorer returns $1 \cdot \text{correct} + \sigma \cdot z$ with $z \sim \mathcal{N}(0, 1)$, so a larger $\sigma$ is a worse reward model. Wrong solutions carry one of 20 equally likely wrong answers. Each cell averages 4000 trials with seed 0.

| $N$ | Oracle | $\sigma = 0.5$ | $\sigma = 1$ | $\sigma = 2$ | Random scorer | Majority vote | $1 - (1-p)^N$ |
|---|---|---|---|---|---|---|---|
| 1 | 0.299 | 0.290 | 0.284 | 0.296 | 0.301 | 0.297 | 0.300 |
| 4 | 0.750 | 0.681 | 0.512 | 0.411 | 0.310 | 0.430 | 0.760 |
| 16 | 0.997 | 0.939 | 0.699 | 0.498 | 0.286 | 0.864 | 0.997 |
| 64 | 1.000 | 0.989 | 0.818 | 0.579 | 0.293 | 1.000 | 1.000 |
| 256 | 1.000 | 0.997 | 0.881 | 0.647 | 0.305 | 1.000 | 1.000 |

At $N = 1$ every method equals $p$, because there is nothing to choose from. The oracle follows $1 - (1-p)^N$. A noisier scorer climbs more slowly, and the random scorer never leaves $p$. A better scorer converts a larger share of the extra samples into accuracy, which matches the paper's reading that the PRM is better than the ORM and voting at searching over many solutions (Section 3). Majority voting does well in this table only because the wrong answers are scattered over 20 values. If the wrong answers agree with each other, voting fails, as Exercise 4 tests. Exercise 4 asks you to write this simulation.

### PRM800K and active learning

**Intuition.** Human step labels are expensive, so spend them where they teach the PRM the most. A solution that the current PRM rates highly and that ends in a wrong answer must contain a step the PRM misjudges. Labeling it is more informative than labeling an obviously bad solution.

**Math.** The paper calls these solutions convincing wrong-answer solutions: rated highly by the current best PRM, with an incorrect final answer. Correctness here is judged only by the final answer, which is occasionally wrong (Section 2.4). In data collection, the authors generate $N$ solutions per problem, surface only the top $K$ most convincing wrong-answer ones, and retrain the PRM at several points (Section 2.4). The dataset is PRM800K: 800K step-level labels across 75K solutions to 12K problems. The 12K problems are the 7,500 MATH training problems plus 4,500 MATH test problems, which leaves 500 test problems for evaluation (Section 2.4 and Appendix C). `scripts/paper_numbers.py` confirms the arithmetic and computes 10.7 labels per solution and 6.25 solutions per problem.

The controlled study of active learning is at small scale (Section 4.2). A small selector PRM trained on one sample per problem scores 1000 samples per problem. To build a training set of $N$ solutions per problem, the authors pick 80% from the most convincing wrong-answer samples and 20% from the most convincing of the rest, right or wrong. The large PRM then supplies labels for those solutions. The authors compare the slopes of the fitted lines and estimate that this is about 2.6 times more data efficient than uniform labeling (Section 4.2, Figure 4a).

Section 4 also explains why a synthetic setup is needed. The large PRM supervises small models, because human labeling of ablations would be too costly (Section 4). The large PRM's steps count as incorrect when its negative-label probability exceeds 20% (Appendix H). Process supervision returns step labels up to the first negative one. Outcome supervision from the large PRM marks the solution correct only if every step passes.

**Tiny example, computed by `scripts/active_learning.py`.** A pool of 12 solutions has selector scores and wrong-answer flags. To label $k = 5$, take $\text{round}(0.8 \cdot 5) = 4$ convincing wrong-answer solutions and 1 more from the rest. The four are the wrong-answer solutions with scores 0.91, 0.86, 0.80, and 0.74. The remaining one is the best leftover, a right-answer solution scoring 0.88. The wrong-answer share is 0.667 in the pool and 0.800 in the labeled set. The paper doesn't say how it rounds the 80/20 split, so this page picks round-to-nearest.

The synthetic label step is in `scripts/step_labels.py`. With $P(\text{negative}) = 0.05, 0.10, 0.20, 0.25, 0.01$, a step is incorrect only above 0.20, so the first four steps give process labels $1, 1, 1, 0$ and the outcome label is $0$.

<!-- exercises -->

## Pitfalls

- **Reading 78.2% as a better model.** The paper doesn't improve the generator. It improves the choice among 1860 fixed samples per problem (Section 2.1). The number depends on a large sampling budget, and on a 500-problem subset of MATH (Appendix C), not the full test set.
- **Comparing the two reward models as if the data matched.** At large scale the PRM trains on PRM800K, which active learning biased toward wrong answers, and the ORM trains on 100 uniform samples per problem. The paper says the two training sets are not directly comparable (Section 3). The small-scale study (Section 4.1) controls for this. It also says it is unclear whether final-answer checking or large-PRM outcome labels is the fairer outcome baseline.
- **Ignoring the length bias.** The product shrinks as steps are added, and the paper says it biases slightly against solutions with more steps (Appendix F.2). The minimum doesn't have this bias. The measured difference is small, so check the effect on your own data before you pick one.
- **Treating missing labels as positive.** Process labels stop at the first incorrect step. The steps after it have no label at all, and not a positive one. Don't train on them.
- **Forgetting the neutral label.** A PRM predicts three labels. The solution score uses $P(\text{pos})$ or $P(\text{pos}) + P(\text{neu})$, and the choice changes the product by a lot when steps are uncertain: $0.4096$ against $0.0625$ in the script above.
- **Expecting voting to always be a weak baseline.** In the paper majority voting is a strong baseline (Section 3). In the simulation it wins only when wrong answers scatter. If a model makes the same mistake repeatedly, the vote converges on it.
- **Trusting small tables without a check.** Section 5 says 224 held-out STEM questions, and Table 1's rows add up to 234, which is also the table's aggregate count (`scripts/paper_numbers.py`). The problem-weighted average of the rows differs slightly from the aggregate row: 63.5 against 63.8 for the ORM, 73.0 against 72.9 for the PRM, and 61.4 against 61.3 for voting. This page quotes the table's aggregate row.
- **Assuming the benefits carry over beyond math.** The paper reports a "negative alignment tax" for process supervision in this setting, and says it is unknown how broadly the result generalizes beyond math (Section 6.2).
- **Ignoring contamination.** The MATH test set may overlap with the pretraining data. The authors found no clear sign of memorization and argue that any effect should be similar across methods (Section 6.3). The fresh exams in Section 5 are the stronger evidence.
- **Active learning is not a solved recipe.** The authors found the model trained on the largest active-learning set slightly under the trend, and found that iteratively retraining the selector was unstable in a way they couldn't diagnose (Section 4.2).

## Further reading

- [PRM800K](https://github.com/openai/prm800k) is the official release: the step-level labels, the 500-problem test split, and the labeling instructions.
- [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168), by Cobbe et al., is the outcome-supervised verifier recipe that this paper's ORM follows.
- [Solving math word problems with process- and outcome-based feedback](https://arxiv.org/abs/2211.14275), by Uesato et al., is the earlier comparison that this paper extends.
- [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171), by Wang et al., is the majority-voting baseline.
- [Measuring Mathematical Problem Solving With the MATH Dataset](https://arxiv.org/abs/2103.03874), by Hendrycks et al., is the benchmark.
- [Math-Shepherd](https://arxiv.org/abs/2312.08935), by Wang et al., trains a PRM without human step labels.
- This site covers test-time compute scaling at [Snell et al.](/snell/), which uses a PRM as the verifier for search.
