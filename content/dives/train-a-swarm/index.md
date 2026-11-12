---
id: train-a-swarm
title: "Train a swarm of agents"
summary: "Task reward alone gives no edge to agents that work in parallel, so the post makes faster work pay with latency pressure, and its main choice is a wall-clock deadline that grows during training. Every agent in the organization shares one policy and one flat collective advantage, so no per-agent credit estimate is needed. This page rebuilds the deadline idea in a toy where a policy learns a team size, using our own model."
source:
  title: "How to Train a Swarm"
  author: "General Reasoning team"
  site: "General Reasoning"
  date: "2026-11-02"
  url: "https://www.gr.inc/blog/how-to-train-a-swarm"
topics: [rl, multi-agent, agents]
est_minutes: 45
prereqs:
  - {id: grpo, reason: "The post's collective advantage is a group-relative advantage that is applied to every token, which is the idea you meet in GRPO."}
  - {id: gae, reason: "The post's example of an incentive that backfires is GAE with λ below 1, which makes reasoning length collapse."}
  - {id: r1, reason: "The post treats collective inference-time scaling as an extension of the longer-reasoning pattern that R1 showed."}
background: [policy-gradient-theorem, group-relative-baselines]
course_links:
  - {title: "The policy gradient", url: "https://rl.ifkash.dev/lessons/031-the-policy-gradient"}
  - {title: "Group-relative policy optimization", url: "https://rl.ifkash.dev/lessons/071-group-relative-policy-optimization"}
---

## The problem

An agent collective is a group of AI agents that work toward one goal. The post describes the current interest in swarms and notes that there is "little public work" on training them with reinforcement learning (RL). It then describes a public recipe from the same lab, called reinforcement learning from collective rewards (RLCR), and a model trained with it, Avalon-M1.

The post sets one goal: *collective inference-time scaling*. A trained collective should spend more compute on useful work done in parallel, and on the communication it needs to coordinate that work.

The difficulty is that a task reward does not ask for any of that. Suppose a lone agent and a team both solve a task, so both get the same reward. A reward that ignores time has no reason to prefer the team, which finishes sooner. In the post's words, the setup "must incentivise parallel work by making faster work pay."

The post makes the comparison with a diagram of three ways to spend about the same inference compute:

| Approach | Elapsed time in the post's diagram |
|---|---|
| One agent reasons for longer | 100 minutes |
| Four independent attempts, then a selection | 28 minutes |
| A communicating organization (CEO, investigator, builder, tester) | 46 minutes |

Independent attempts are fastest here, so when attempts are enough, communication is only overhead. The post argues that coordination pays when the work is imperfectly parallelizable, meaning agents must share intermediate results or combine complementary work. The rest of this page covers how to build an incentive that finds that case without hand-coded structure.

## Key ideas

The post has one equation and no code. Everything marked "our toy" or "our numbers" below is our own model or arithmetic, and is not a result from the post. Scripts in `scripts/` compute every number we state. Run any script from the `scripts/` folder to reproduce its output.

### Incentives decide the behavior

The post starts with the reasoning paradigm. RL taught models to spend more inference compute before answering, because checking work and backtracking raised expected return. That outcome was not guaranteed. The post points to Open-Reasoner-Zero's ablation: with PPO and GAE $\lambda = 1$, response length keeps growing, and with $\lambda = 0.95$ it collapses within about 50 steps. A strong base model also matters.

The lesson the post draws is that the RL setup, not only the algorithm, determines the behavior. For swarms, the setup must create a reason to work in parallel. See [GAE](/gae/) for what $\lambda$ does, and [R1](/r1/) for the length growth.

### Latency penalties

The first option is an explicit latency penalty. The post says DeepSeek-V4.1-Flash's agent teams were trained with one, but that the form wasn't shared. It offers one possible form, shaped like a length penalty:

$$
R(\tau) = R_{\text{task}}(\tau) - \beta\,\bar p(q)\,\frac{T(\tau)}{T_{\max}}
$$

Here $T(\tau)$ is the wall-clock time of rollout $\tau$, $T_{\max}$ is a time limit, $\bar p(q)$ is the average pass rate of task $q$ across its rollouts, and $\beta$ sets the strength.

**Tiny example, computed by `scripts/penalty_example.py` (our numbers).** Let both a single agent and a swarm earn $R_{\text{task}} = 0.8$, with $\beta = 0.5$, $\bar p = 0.5$, and $T_{\max} = 100$ minutes. The single agent takes 100 minutes and the swarm takes 46:

- Single agent: $0.8 - 0.5 \cdot 0.5 \cdot 1.00 = 0.55$.
- Swarm: $0.8 - 0.5 \cdot 0.5 \cdot 0.46 = 0.685$.

The swarm wins by 0.135. Without the penalty, the two tie at 0.8.

The post lists three drawbacks. The penalty can interact badly with other quantities such as reasoning effort. It is unclear that you want latency pressure at every rollout length. And $T/T_{\max}$ varies across tasks and depends heavily on $T_{\max}$.

### Time limits and anytime return

The second option, and the one Avalon-M1 uses, is a time limit as the only source of latency pressure. $T_{\max}$ is a maximum wall-clock time. When a rollout hits it, the rollout's return is the total return up to that point. The policy is rewarded for making as much progress as it can before the deadline.

This needs an environment with an "anytime-return" quality: you can stop a rollout at any time and read a measure of progress. The post's examples are the number of tests passed in a long software task, and returns in a financial trading environment.

The post is explicit about a limit. A deadline "does not reward every reduction in completion time": two rollouts that reach the same return before the deadline get the same score, even if one finishes earlier. The deadline and its schedule remain design choices.

### Why the deadline grows

The post increases the deadline during training. Avalon-M1's schedule starts at 2 hours and ends just over 5 hours by step 200. The post's reason is that a longer deadline gives a collective time to recover the initial costs of coordination, and that single agents progress faster at first because they have less coordination overhead.

To make this concrete, here is a model of our own, not the post's. Work has a parallel fraction $p$. By Amdahl's law, a team of $n$ agents works at speed

$$
s(n) = \frac{1}{(1 - p) + p/n}
$$

Each extra agent costs a setup time $c$ before the team does useful work. The progress by deadline $T$ is

$$
\text{progress}(n, T) = s(n)\,\max\bigl(0,\ T - c\,(n - 1)\bigr)
$$

**Tiny example, computed by `scripts/amdahl_cap.py` (our toy).** Take $p = 0.9$ and $c = 0.6$ minutes. Then $s(1) = 1.00$, $s(4) = 1/(0.1 + 0.225) = 3.08$, and $s(8) = 1/(0.1 + 0.1125) = 4.71$.

| Deadline | $n = 1$ | $n = 4$ | $n = 8$ |
|---|---|---|---|
| 5 minutes | 5.0 | 9.8 | 3.8 |
| 20 minutes | 20.0 | 56.0 | 74.4 |

At 5 minutes, eight agents lose: $4.71 \cdot (5 - 4.2) = 3.8$. At 20 minutes they win by a wide margin. Eight agents overtake one agent when $s(8)\,(t - 4.2) = t$, which gives $t = 5.33$ minutes. The same shape appears in the post's EdgeBench result, where an organization of up to 8 hires starts slowly and overtakes the single-agent baseline "at around 2.4 hours of derived latency." The technical report that the post links gives 1.7 hours in its abstract; this page uses the post's figure and does not know the reason for the difference.

Use the explorer to move the deadline and see the best team size change.

<div class="widget" data-widget="cap-explorer"></div>

<!-- cell: progress -->

### One flat advantage for every agent

In RLCR, one policy plays every agent, and all agents receive the same collective advantage. The post describes it as a "flat collective advantage applied to all agents." The post doesn't give the formula. The report that the post links says every agent receives the same collective advantage, which avoids per-agent credit assignment, and says most of its environments use group-relative advantages with a group size of 16, citing the GRPO paper. The example below is our reading of that description, in the style of [GRPO](/grpo/).

**Tiny example, computed by `scripts/advantage_example.py` (our numbers).** A group has four rollouts. The tests passed when the cap hits are $[3, 5, 5, 9]$. The mean is 5.5, the variance is $(2.5^2 + 0.5^2 + 0.5^2 + 3.5^2)/4 = 4.75$, and the population standard deviation is 2.18. The advantages are $[-1.15, -0.23, -0.23, +1.61]$ and they sum to 0. Every token of every agent in the rollout with return 9 receives $+1.61$, whether the agent wrote a key test or sent a greeting.

The post names the cost as part of the "uneasy triangle." A flat advantage over episodes of hours, and sometimes millions of tokens, is a coarse signal, because it applies to many tokens that did not affect the outcome.

Edit the returns in the next widget, then switch to the latency penalty.

<div class="widget" data-widget="flat-advantage"></div>

### Communication primitives

Latency pressure isn't enough. Agents also need ways to talk so they can divide work, avoid duplicating it, and hand it off. The post asks researchers to keep these primitives simple, so that agents discover how to cooperate without much bias from the harness. The primitives it names are direct messages, public channels (message boards), and, optionally, issue boards and knowledge archives.

The post compares three systems, which we summarize here:

| System | What is trained | Who talks to whom | Scale |
|---|---|---|---|
| Kimi K2.5 Agent Swarm | The orchestrator only; sub-agents are frozen | Orchestrator to sub-agents and back | Up to 300 sub-agents in K2.6 |
| Avalon-M1 | Every agent, on one collective reward | Anyone, through channels, direct messages, and an archive | A CEO and up to 8 specialists |
| OpenAI's Navier-Stokes system, as described | Not described | Within a group of agents; the post assumes a board | About 10,000 agents in the group that got the solution |

The post notes that each system still imposes some structure. Avalon-M1's CEO can hire and fire, although every agent can talk to every other.

### Signs of life

The post argues that reward curves alone don't show whether a collective is learning to use parallel compute well. It tracks three kinds of evidence, all from Avalon-M1:

- **Useful work.** The output and reasoning tokens of every turn that acts on the environment, such as running a command, editing a file, or running tests, compared with the environment's average. A collective with 5 to 8 hires starts below the average rollout and does nearly twice the average by step 200.
- **Communication.** Compute spent on communication rises as training progresses, normalized by rollout duration. The post says volume alone doesn't establish that communication is useful.
- **Performance against latency.** A single agent given as many output tokens as the collective recovers much of the gain but takes longer. At a matched token budget, the post says a single-agent baseline takes 4 times as long.

The post reports these on internal benchmarks, and it says the model is still in training. We can't check the numbers independently.

## Rebuild it

The reconstruction is our own toy, not the post's training run. A policy picks a team size $n$ from 1 to 8 with a softmax over eight logits. The environment returns the progress by the deadline from the model above. The update is REINFORCE with a flat group-relative advantage: sample 16 team sizes, standardize their returns, and push up the log-probability of each sampled size in proportion to its advantage. There are no tokens or real agents, so the toy keeps only the incentive.

Run the cell, then compare each learned team size with the best one.

<!-- cell: reconstruct -->

<div class="widget" data-widget="cell-result"></div>

With seed 0, the learned mean team size follows the best size: 2.00 at a deadline of 2, 4.00 at 5, 6.05 at 10, and 8.00 at 20 and 40. A longer deadline teaches a bigger team.

Next, train in the browser with a schedule of your own choice.

<div class="widget" data-widget="team-trainer"></div>

Now compare three ways to set the incentive in the same toy.

<!-- cell: variants -->

With seed 0, the results are:

- **Completion only.** The mean team size stays at 4.50, the value of a uniform policy, because every reward is equal and every advantage is 0. This is the "nothing favors parallelism" problem from the post.
- **Explicit latency penalty.** The mean is 8.00. In our toy, finish time falls with every extra agent, so the penalty always prefers the largest team. This is a property of our model and not a statement about the post's penalty.
- **Deadline ramped from 2 to 20.** With seed 0 the mean reaches 6.98 at step 300, short of the best size of 8. Across seeds 0 to 5 the final means are 6.98, 4.01, 4.02, 5.13, 4.01, and 7.97. Many runs lock in a team of about 4 while the deadline is short, and then stop learning, because a group of 16 identical team sizes has zero advantage.

Our toy doesn't model the post's reason for the schedule. The post says single agents progress faster at first because they have less coordination overhead, and a longer deadline gives a collective time to recover that cost. In our toy, a fixed deadline of 20 reaches 7.99 or 8.00 on every seed from 0 to 5, and the ramp does worse. So the toy doesn't support the curriculum, and it says nothing against it either: our model has no way to represent the post's learning dynamics. What it does show is that a policy trained under a moving deadline can lose its exploration, which is one reason a real recipe needs more than a flat advantage.

## What the post doesn't say

- **The training algorithm.** The post says only "a simple core policy gradient algorithm" with a shared policy and a flat collective advantage. For the base model, objective, and group size, read the technical report. The report is still described as under revision.
- **A tested latency penalty.** The penalty equation is "one possible functional form." The post says the form used for DeepSeek-V4.1-Flash wasn't shared, and it doesn't report a comparison between penalties and deadlines.
- **Proof that communication helps.** The post says communication volume "alone does not establish its usefulness."
- **Results independent of the lab.** The benchmarks include internal ones, and the post states that the model is still in training. The full paper is promised later.
- **How the result scales.** The "village problem" is the gap between the at most 9 agents in training and the hundreds or thousands in deployment. The post says it doesn't know what generalizes.
- **Shared environments.** In the current recipe, rollouts are isolated. The post lists as open how to reward a rollout for a public good that only helps later episodes.
- **Our toy.** Team size is one number in our model. The real setting has a CEO, specialists, and a message board. Don't read the toy's team sizes as predictions about Avalon-M1.

## Further reading

- [How to Train a Swarm](https://www.gr.inc/blog/how-to-train-a-swarm), the original post. It links the Avalon-M1 technical report at [gr.inc/avalonm1.pdf](https://gr.inc/avalonm1.pdf), which covers the recipe in detail.
- [DeepSeek-R1](/r1/), the reasoning pattern the post extends, and [GRPO](/grpo/), the group-relative advantage.
- [Open-Reasoner-Zero (arXiv 2503.24290)](https://arxiv.org/abs/2503.24290), the source of the GAE $\lambda$ ablation the post cites.
- [Just enough thinking (arXiv 2506.05256)](https://arxiv.org/abs/2506.05256), which the post cites for length-penalty shapes.
