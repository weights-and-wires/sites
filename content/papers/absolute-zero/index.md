---
id: "absolute-zero"
title: "Absolute Zero: Reinforced Self-play Reasoning with Zero Data"
authors: "Zhao et al."
year: 2025
venue: "arXiv preprint"
arxiv: "2505.03335"
url: "https://arxiv.org/abs/2505.03335"
track: rl
tier: extended
est_hours: 4
status: full
summary: "Absolute Zero trains one language model to play two roles: a proposer that writes Python programs with an input and an output as reasoning tasks, and a solver that answers them. A code executor checks every task and every answer, so the loop needs no human-written questions or labels. The proposer earns a learnability reward that is highest for tasks the solver sometimes solves, and the resulting model, AZR, improves its code and math averages over its base model without any curated training data."
related_course: "rl: AI feedback and self-improvement"
prereqs:
  - id: r1
    reason: "AZR keeps R1-Zero's idea of reinforcement learning from a reward that a program computes, and it adds a format penalty to the same kind of rule-based reward."
  - id: grpo
    reason: "AZR normalizes rewards against a baseline built from sampled outputs, as GRPO does, and Equation 8 here changes only which outputs share a baseline."
  - id: alphazero
    reason: "AZR is self-play in the AlphaZero sense, where the agent generates its own training signal, but the opponent is a task proposer and the environment is a code executor."
background: [verifiable-rewards, self-play, group-relative-baselines]
course_links:
  - title: "AI feedback and self-improvement"
    url: "https://rl.ifkash.dev/lessons/076-ai-feedback-and-self-improvement"
  - title: "Verifiable rewards"
    url: "https://rl.ifkash.dev/lessons/072-verifiable-rewards"
self_check:
  - q: 'A solver attempts a proposed task 8 times and succeeds 2 times. What is the proposer reward under Equation 4, and what is it if the solver succeeds 0 times or 8 times?'
    a: 'The mean solver reward is $2/8 = 0.25$, so the proposer reward is $1 - 0.25 = 0.75$. With 0 successes the reward is $0$ by definition, and with 8 successes it is $1 - 1 = 0$. The proposer earns nothing for tasks that are too hard or too easy.'
    see: verifiable-rewards
  - q: 'Why does Section 3.1 give a solver answer that is well formatted but wrong a reward of $-0.5$ and not $0$?'
    a: 'A reward of $0$ would match the reward of a response that earns nothing for other reasons, such as a proposed task that is too easy. Equation 6 separates three cases: the role reward for a passing response, $-0.5$ for a well-formatted failure, and $-1$ for a formatting error. The ordering tells the policy that a format error is worse than a wrong answer.'
    see: verifiable-rewards
  - q: 'Why does Equation 8 use six baselines and not one shared baseline for the whole batch?'
    a: 'Rewards from different roles and task types have different scales and spreads. Solver rewards are 0 or 1, and proposer rewards are fractions, so a shared mean would give one role a positive advantage on average and the other a negative one. Normalizing within each pair of task type and role keeps every group centered on its own mean.'
    see: group-relative-baselines
---

## The problem

Reinforcement learning with verifiable rewards, as in [DeepSeek-R1](/r1/), removes the human-written reasoning trace. It still needs a set of questions with checkable answers, and people write and curate those. Curated question sets are expensive to scale, and a model that trains only on tasks that people thought of is limited to them.

Consider a model that trains on a fixed pool of coding problems. After enough steps, it solves most of the easy ones every time and fails the hardest ones every time. Neither group gives a useful gradient. A fixed pool can't adapt to what the model has just learned.

The paper asks for a setting with no human data at all, which it calls the Absolute Zero paradigm (Section 2). One model must write its own tasks and learn from solving them. That raises a new problem. Without a human judge, the model needs a way to decide which tasks are worth proposing. A task that the solver always gets right teaches nothing, and so does a task that it never gets right.

The paper's answer is the Absolute Zero Reasoner (AZR, Section 3). A code executor is the only source of ground truth. The proposer earns a reward for tasks at the edge of the solver's ability, and a single policy plays both roles. AZR-Coder-7B raises the overall average over the paper's code and math benchmarks from 40.2 to 50.4 with no curated data (Section 4, Table 1).

## Key ideas

The sections that follow use the paper's notation: a seed $z$, a proposed task $\tau$, a task query $x$ with reference answer $y^\star$, the proposer reward $r^{propose}$, and the solver reward $r^{solve}$. Section and equation numbers follow arXiv v3. Earlier versions can number them differently. Every computed number comes from a script in `scripts/`. Run any script from the `scripts/` folder to reproduce the output.

### One policy, two roles, one executor

**Intuition.** The model writes a task, then solves it. The task is a Python program, and a Python interpreter is the referee. The interpreter can run any program the model writes, so it can compute the correct answer to any task the model proposes and can check any answer the model gives.

**Math.** The two roles share one set of weights $\theta$. The objective (Section 2, Equation 3) is an expectation over seeds $z$ of the proposer's reward and the solver's expected reward:

$$
J(\theta) = \max_\theta \; \mathbb{E}_{z \sim p(z)} \Big[ \mathbb{E}_{(x, y^\star) \sim f_e(\cdot \mid \tau),\, \tau \sim \pi_\theta^{propose}(\cdot \mid z)} \big[ \lambda \, r_e^{propose}(\tau, \pi_\theta) + \mathbb{E}_{y \sim \pi_\theta^{solve}(\cdot \mid x)} [ r_e^{solve}(y, y^\star) ] \big] \Big]
$$

The coefficient $\lambda \ge 0$ trades the proposer term against the solver term. The environment $f_e$ turns a proposed task $\tau$ into a query and a verified answer. In AZR the environment is a code executor, and a task is a triplet $(p, i, o)$ of a program, an input, and the output $o = p(i)$ (Section 3.2).

**Tiny example.** The program $p$ is `sum(i for i in range(x) if i % 2 == 0)` and $i = 5$. The executor computes $o = 6$. The executor makes the label, so no person writes the answer. `scripts/triplet_example.py` prints this triplet.

### Three task types: deduction, abduction, induction

**Intuition.** One triplet supports three questions, depending on which two of the three parts the solver sees. The paper links them to three modes of reasoning (Section 3.2).

**Math.** For a triplet $(p, i, o)$ with $o = p(i)$, the three task types are as follows:

| Task type | Solver sees | Solver returns | Verified by |
|---|---|---|---|
| Deduction | $p$, $i$ | $o$ | equality with $o$ |
| Abduction | $p$, $o$ | an input $i'$ | running $p$ and checking $p(i') = o$ |
| Induction | some pairs $(i, o)$ and a short description | a program $p'$ | running $p'$ on held-out pairs |

Abduction is checked by running the program, not by comparing with the original input, because programs need not be one-to-one (Section 3.2). Before a proposed task enters training, the executor validates it (Section 3.3.3). The program must run without error and return a value. It must not import sensitive packages. It must be deterministic, which the paper checks by running the program twice and comparing the outputs.

**Tiny example, computed by `scripts/triplet_example.py`.** For the program above, $f(5) = 6$ and $f(6) = 6$. The abduction answer 5 is correct, and so is 6, even though 6 isn't the original input. The answer 3 gives $f(3) = 2$ and fails. An induction answer `sum(range(0, x, 2))` matches six held-out pairs and passes, while `x // 2` fails. Exercise 4 implements these checks.

### The learnability reward for the proposer

**Intuition.** The proposer should write tasks that the solver can learn from. A task that the solver always solves has nothing left to teach. A task that the solver never solves gives no signal either, because every attempt scores the same. The best tasks are the ones the solver sometimes solves, and among those, harder is better.

**Math.** The solver attempts a proposed task several times at non-zero temperature, and $\bar r_{solve}$ is the mean of its binary rewards (Section 3.1, Equation 4):

$$
r^{propose} =
\begin{cases}
0 & \text{if } \bar r_{solve} = 0 \\
1 - \bar r_{solve} & \text{otherwise}
\end{cases}
$$

The second case also gives $0$ when $\bar r_{solve} = 1$, so the reward is zero at both ends. Between the ends it falls as the solver succeeds more often, so a task that the solver rarely solves earns the most. The solver's own reward is binary: $r^{solve} = \mathbb{I}(y = y^\star)$ (Section 3.1, Equation 5), where equality is Python value equality.

**Tiny example, computed by `scripts/learnability_example.py`.** With 8 solver attempts per task:

| Successes of 8 | Mean solver reward | $r^{propose}$ |
|---|---|---|
| 0 | 0.000 | 0.000 |
| 1 | 0.125 | 0.875 |
| 2 | 0.250 | 0.750 |
| 4 | 0.500 | 0.500 |
| 6 | 0.750 | 0.250 |
| 7 | 0.875 | 0.125 |
| 8 | 1.000 | 0.000 |

The paper's hyperparameter table (Table 3, Appendix B) lists 8 samples to estimate task accuracy. This page uses 8 for the example and doesn't confirm that this entry is the number of rollouts in Equation 4. Exercise 1 implements the reward.

### Composite reward with a format penalty

**Intuition.** A role reward alone doesn't distinguish a wrong answer from a malformed one. The paper adds a penalty so that a policy that breaks the output format loses more than one that answers incorrectly.

**Math.** For either role, the reward is as follows (Section 3.1, Equation 6):

$$
R(y_\pi) =
\begin{cases}
r_{role} & \text{if the response passes} \\
-0.5 & \text{if the response is wrong but well formatted} \\
-1 & \text{if the response has formatting errors}
\end{cases}
$$

This page reads "passes" as a correct answer for the solver and a valid task for the proposer. The paper's own wording for the middle case isn't checked here. A valid task that earns a learnability reward of $0$ isn't penalized under this reading, because it passes and its role reward is $0$.

**Tiny example, computed by `scripts/composite_example.py`.** A correct, well-formatted solver answer earns $1.0$, a wrong but well-formatted answer earns $-0.5$, and a formatting error earns $-1.0$. A valid proposed task with learnability $0.75$ earns $0.75$, and an invalid task earns $-0.5$. Exercise 2 implements the function.

### Task-relative REINFORCE++

**Intuition.** Solver rewards are 0 or 1. Proposer rewards are fractions. Abduction tasks can be easier than induction tasks. One baseline over all of these would mix scales that don't compare. Task-Relative REINFORCE++ (TRR++) keeps a separate baseline for every pair of task type and role.

**Math.** Let the task type be induction, deduction, or abduction, and the role be propose or solve. The normalized advantage is as follows (Section 3.3.5, Equation 8):

$$
A^{norm}_{task, role} = \frac{r - \mu_{task, role}}{\sigma_{task, role}}, \qquad task \in \{ind, ded, abd\},\; role \in \{propose, solve\}
$$

There are six pairs, so there are six means $\mu$ and six standard deviations $\sigma$. The idea is GRPO's group-relative baseline with a different grouping rule. Each group is one task type in one role, not the samples for one prompt. The paper builds on REINFORCE++, and this page doesn't cover that algorithm's other parts.

**Tiny example, computed by `scripts/trr_example.py`.** A toy batch has 12 rewards in three groups. The deduction-solve rewards $[1, 0, 0, 1]$ have mean $0.5$ and standard deviation $0.5$, so the advantages are $[1, -1, -1, 1]$. The deduction-propose rewards $[0.75, 0.5, 0.125, 0]$ have mean $0.3438$ and standard deviation $0.2981$, so the advantages are $[1.3628, 0.5241, -0.7338, -1.1531]$. The abduction-solve rewards $[1, 1, 0, 1]$ give $[0.5773, 0.5773, -1.732, 0.5773]$. If you used one baseline for the whole batch, the first group would get $[1.0392, -1.1778, -1.1778, 1.0392]$, which is not centered on that group. Exercise 3 implements Equation 8.

### What the paper shows

Section 4 trains on Qwen2.5-7B and Qwen2.5-7B-Coder, with extra runs on other sizes and on Llama-3.1-8B. Table 3 (Appendix B) lists the settings: a batch of $64 \times 6$, a learning rate of $10^{-6}$ with AdamW, 500 steps, $K = 6$ reference examples for the proposer, and a seed buffer of size $B \times S$ with $S = 4$.

Table 1 reports averages over code and math benchmarks. AZR-Coder-7B reaches a code average of 61.6 (up 5.0 from its base) and a math average of 39.1 (up 15.2), for an overall average of 50.4 (up 10.2 from 40.2). AZR-Base-7B reaches 55.2 on code (up 3.2) and 38.4 on math (up 10.9), for an overall average of 46.8 (up 7.0 from 39.8). The paper notes that the Coder model's overall average is 1.8 points above the best other 7B model in the table, and that its code average is 0.3 points above the best model trained on expert code data.

On Llama-3.1-8B, AZR helps on some benchmarks and hurts on others. For example, HumanEval+ rises from 31.7 to 35.4, and MBPP+ falls from 53.7 to 50.8 (Appendix C.1, Table 5). The paper also reports an "uh-oh moment" in which Llama-3.1-8B writes worrying chains of thought and says this calls for safety-aware training.

<!-- exercises -->

## Pitfalls

- **Rewarding the easy end.** The learnability reward is $1 - \bar r_{solve}$ only when $\bar r_{solve} > 0$. If you drop the zero case, a task that the solver never solves earns the maximum reward of 1, and the proposer learns to write impossible tasks. Exercise 1's mutant checks for this bug.
- **Using the success rate as the reward.** The reward is the complement. With $\bar r_{solve}$ in its place, the proposer is paid for easy tasks.
- **Treating `-0.5` and `0` as the same.** Equation 6 keeps a well-formatted failure at $-0.5$ so that it stays below a valid task that happens to earn learnability $0$. Exercise 2's tests check both cases.
- **One baseline for every sample.** Equation 8 has six baselines. A shared baseline mixes binary solver rewards with fractional proposer rewards, so one role gets mostly positive advantages. Exercise 3's mutants check for pooling by role, by task type, and for the whole batch.
- **Comparing an abduction answer with the original input.** A program can map many inputs to the same output, so the check runs $p$ on the answer. Equality with $i$ rejects correct answers. Exercise 4 checks this.
- **Skipping task validation.** The executor runs code that the model wrote. Without the integrity, safety, and determinism checks of Section 3.3.3, a task can crash, touch the system, or have an answer that changes between runs. The exercise's `run_program` restricts built-ins, but it isn't a sandbox, and you shouldn't run untrusted code with it.
- **Reading the gain as settled by one table.** This page didn't check whether Table 1 reports variance across seeds, so treat the gains as point estimates.
- **Reading equation numbers.** Equation numbers here follow arXiv v3. Earlier versions can number them differently.

## Further reading

- The [paper's arXiv page](https://arxiv.org/abs/2505.03335) and the authors' code, [Absolute-Zero-Reasoner](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner), which the paper links.
- [DeepSeek-R1](/r1/) for rule-based rewards without human reasoning traces, [GRPO](/grpo/) for the group-relative baseline that Equation 8 regroups, and [AlphaZero](/alphazero/) for self-play that needs no human games.
