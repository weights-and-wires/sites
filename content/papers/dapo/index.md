---
id: "dapo"
title: "DAPO: An Open-Source LLM Reinforcement Learning System at Scale"
authors: "Yu et al."
year: 2025
venue: "arXiv preprint"
arxiv: "2503.14476"
url: "https://arxiv.org/abs/2503.14476"
track: rl
tier: extended
est_hours: 4
status: full
summary: "DAPO is a GRPO-style reinforcement learning recipe for long chain-of-thought reasoning, and it fixes four failures of the naive method: entropy collapse, groups that give no gradient, long outputs that count too little in the loss, and noisy rewards for truncated outputs. It raises only the upper clip bound, filters out prompts whose sampled answers are all right or all wrong, averages the loss over tokens instead of over outputs, and adds a soft length penalty. On the Qwen2.5-32B base model it reaches 50 points on AIME 2024, against 30 for naive GRPO and 47 for DeepSeek-R1-Zero-Qwen-32B, in about half the training steps."
related_course: "rl: reasoning models"
prereqs:
  - id: r1
    reason: "DAPO sets out to reproduce R1-Zero-style training, pure RL with a rule-based accuracy reward on a base model, and measures itself against R1-Zero-Qwen-32B."
background: [entropy-collapse, dynamic-sampling]
course_links:
  - title: "Reasoning with verifiable rewards"
    url: "https://rl.ifkash.dev/lessons/073-reasoning-with-verifiable-rewards"
  - title: "Group-relative policy optimization"
    url: "https://rl.ifkash.dev/lessons/071-group-relative-policy-optimization"
self_check:
  - q: 'R1-Zero trains a base model with reinforcement learning and a reward that a program computes, with no human-written reasoning traces. What does DAPO use as its reward, and why does it avoid a learned reward model?'
    a: 'DAPO uses a rule: $+1$ if the predicted answer is equivalent to the ground truth and $-1$ otherwise (Section 2.4, Equation 7). The paper says reward models are prone to reward hacking, and a rule on a checkable answer is not. To make the rule reliable, the authors rewrite every training answer into an integer (Section 3.5).'
    see: r1
  - q: 'A policy has entropy $H = 0.05$ nats after 500 steps and was at $H = 1.5$ at the start. What is this called, what does it do to training, and which DAPO technique targets it?'
    a: 'This is entropy collapse: the policy becomes nearly deterministic early, so the sampled outputs for a prompt look alike and exploration stops. DAPO targets it with Clip-Higher, which raises the upper clip bound so that low-probability tokens can gain more probability per update (Section 3.1).'
    see: entropy-collapse
  - q: 'A group of $G = 4$ sampled outputs for one prompt earns rewards $[+1, +1, +1, +1]$. What are the group-relative advantages, what is the gradient, and what does dynamic sampling do with the group?'
    a: 'The mean is $1$ and the standard deviation is $0$, so every advantage is $0$ (the formula gives $0/0$, and an implementation must guard it). The group adds nothing to the gradient. Dynamic sampling drops the group and keeps sampling prompts until the batch holds only groups with at least one correct and one incorrect output (Section 3.2, Equation 11).'
    see: dynamic-sampling
---

## The problem

Reasoning models such as OpenAI o1 and DeepSeek-R1 get their long chains of thought from large-scale reinforcement learning (RL). The technical reports for these models leave out most of the training recipe. The DAPO paper starts from this gap. The authors train the Qwen2.5-32B base model with the naive GRPO recipe (see [DeepSeekMath](/grpo/)), and the model reaches only 30 points on AIME 2024. DeepSeek reports 47 points for the same base model with the R1 approach (Section 1).

The paper traces the gap to a set of failures in long chain-of-thought training.

- **Entropy collapse.** The policy's entropy falls quickly, and the sampled responses for a prompt become nearly identical. The paper reads this as limited exploration and an early deterministic policy (Section 3.1).
- **Groups with no gradient.** As training goes on, more prompts have every sampled answer correct. All rewards in the group are equal, so the advantages are 0 and the group adds nothing to the gradient. The effective batch shrinks, and its gradient becomes noisier (Section 3.2).
- **Long outputs count too little.** GRPO averages the loss over the tokens of each output first, and then over outputs. A 10,000-token output then has the same weight as a 500-token one, so each of its tokens counts less. Gibberish and repetition in long outputs are penalized too weakly (Section 3.3).
- **Noisy rewards for truncated outputs.** Generation stops at a maximum length. A sound but long chain of thought gets cut off and receives the punitive reward, as if the reasoning were wrong (Section 3.4).

Here is a concrete case of the second failure. With 16 samples per prompt, a prompt that the model solves 90% of the time gives a uniform group with probability about 0.19. At a 99% pass rate, that probability is about 0.85. `scripts/dynamic_sampling_example.py` computes both. As the policy improves, a growing share of the batch contributes nothing.

DAPO, short for Decoupled Clip and Dynamic sAmpling Policy Optimization, answers each failure with one technique. The authors release the training code (built on the verl framework) and a 17K-prompt math dataset. Starting from the same base model, DAPO reaches 50 points on AIME 2024 with about 50% of the training steps of DeepSeek-R1-Zero-Qwen-32B (Section 4.2, Figure 1).

## Key ideas

This page uses the paper's notation. A prompt $q$ has an answer $a$. The old policy $\pi_{\theta_{old}}$ samples a group of $G$ outputs $\{o_i\}_{i=1}^{G}$. Output $o_i$ has $|o_i|$ tokens, and $o_{i,t}$ is token $t$. The ratio is $r_{i,t}(\theta) = \pi_\theta(o_{i,t} \mid q, o_{i,<t}) / \pi_{\theta_{old}}(o_{i,t} \mid q, o_{i,<t})$, and $\hat A_{i,t}$ is the advantage. The paper writes the advantage with the reward of the output, $\hat A_{i,t} = (R_i - \operatorname{mean}(\{R_i\})) / \operatorname{std}(\{R_i\})$ (Equation 9). It is the same for every token of an output. This is the GRPO advantage, and the [GRPO page](/grpo/) explains it. This page doesn't repeat that explanation.

DAPO keeps the clipped surrogate of PPO (see [Proximal policy optimization](/ppo/)) and changes four things around it. It also removes the KL penalty that GRPO subtracts from the objective. The paper's argument is that a long-chain-of-thought model is expected to drift far from its starting point, so the restriction isn't needed (Section 2.3). The sections that follow take the four techniques one at a time. Every number comes from a script in `scripts/`. Run any script from the paper folder to reproduce the output.

### Clip-Higher

**Intuition.** PPO's clip stops a token's probability from rising much in one update, and the allowed rise is proportional to the old probability. A likely token can gain a lot of absolute probability. An unlikely token, which is the kind that opens a new line of reasoning, can gain very little. If you apply the same $\varepsilon$ to both sides, the policy tends to sharpen toward what it already does, and entropy falls. The fix is to give the upper side more room and leave the lower side as it is.

**Math.** Section 3.1 splits the single clip range of GRPO (Section 2.2, Equation 5) into two. For a token with a positive advantage, the term in the objective stops growing once the ratio passes $1 + \varepsilon_{high}$. For a token with a negative advantage, it stops once the ratio drops below $1 - \varepsilon_{low}$. Equation 10 is the objective with the two bounds:

$$
\min\Big(r_{i,t}(\theta)\hat A_{i,t},\ \operatorname{clip}\big(r_{i,t}(\theta),\ 1 - \varepsilon_{low},\ 1 + \varepsilon_{high}\big)\hat A_{i,t}\Big)
$$

Section 4.1 sets $\varepsilon_{low} = 0.2$ and $\varepsilon_{high} = 0.28$. The paper keeps $\varepsilon_{low}$ small on purpose. A larger value would let the policy push a token's probability toward 0, and that collapses the sampling space (Section 3.1). The paper also observes that the tokens that hit the upper clip have a low mean probability (Figure 3a), which supports the intuition.

**Tiny example, computed by `scripts/clip_higher_example.py`.** Take two tokens with positive advantage, as in Section 3.1. With $\varepsilon = 0.2$, the largest probability a token can reach in one update is $\pi_{old}(1 + \varepsilon)$:

| $\pi_{old}$ | bound at $\varepsilon = 0.2$ | bound at $\varepsilon_{high} = 0.28$ | gain |
|---|---|---|---|
| 0.01 | 0.0120 | 0.0128 | +0.0008 |
| 0.1 | 0.1200 | 0.1280 | +0.0080 |
| 0.5 | 0.6000 | 0.6400 | +0.0400 |
| 0.9 | 1.0800 | 1.1520 | +0.0720 |

The paper quotes the first and last bounds at $\varepsilon = 0.2$: 0.012 and 1.08. The 0.9 token is limited by the value 1 rather than by the clip, so the clip doesn't restrict a likely token at all, while it restricts the 0.01 token to a rise of 0.002. Now take the 0.01 token after an update that raises it to 0.0125, a ratio of 1.25. At $\varepsilon_{high} = 0.2$ the term is capped at 1.2 and the gradient is 0. At $\varepsilon_{high} = 0.28$ the term is 1.25 and the gradient still flows, so the token can keep growing. The same script shows the lower side: with $\hat A = -1$ and a ratio of 0.75, the term is $-0.8$ with no gradient at $\varepsilon_{low} = 0.2$.

Exercise 1 asks you to implement the asymmetric clip.

### Dynamic sampling

**Intuition.** If every output in a group is correct, or every output is wrong, the group says nothing about which behavior is better. Its advantages are all 0, and it contributes nothing to the gradient, but it still takes up space in the batch. DAPO draws more prompts and throws these groups away until the batch is full of groups that do teach something.

**Math.** Equation 11 adds a constraint to the objective: the number of outputs equivalent to the answer $a$ must be strictly between 0 and $G$.

$$
0 < \Big|\{o_i \mid \texttt{is\_equivalent}(a, o_i)\}\Big| < G
$$

Algorithm 1 implements it with a buffer. It samples a batch of prompts, filters out the uniform groups, and adds the rest to the buffer. If the buffer holds fewer groups than the batch size, it samples again. Only a full buffer goes on to the update. Because the rewards are $+1$ and $-1$ (Equation 7), a group is uniform exactly when its standard deviation is 0. The cost of a batch is therefore dynamic. The paper argues that this doesn't hurt much, because generation time in a synchronous system is dominated by the longest outputs, and because training reaches the same accuracy in fewer steps (Section 3.2, Figure 6).

**Tiny example, computed by `scripts/dynamic_sampling_example.py`.** With $G = 4$, the group $[+1, +1, +1, +1]$ gets advantages $[0, 0, 0, 0]$, and so does $[-1, -1, -1, -1]$. The group $[+1, -1, -1, -1]$ gets $[1.7321, -0.5774, -0.5774, -0.5774]$. Check by hand: the mean is $-0.5$, the standard deviation is $\sqrt{0.75} = 0.8660$, and $1.5 / 0.8660 = 1.7321$. With $G = 16$, a group is uniform with probability $p^{16} + (1 - p)^{16}$, which is 0.00003 at $p = 0.5$, 0.1853 at $p = 0.9$, and 0.8515 at $p = 0.99$. In a seeded simulation of a prompt pool where many prompts are nearly always right or nearly always wrong, filling a batch of 512 kept prompts takes 819 draws, so the filter costs 1.60 times as many prompts. This is an illustration of the mechanism and doesn't come from the paper.

Exercise 3 asks you to implement the filter and the loop.

### Token-level policy gradient loss

**Intuition.** Averaging the loss over each output and then over outputs gives every output the same total weight. A long output spreads that weight over many tokens, so each of its tokens counts less. Averaging over all tokens in the group gives every token the same weight, so long outputs get more of the total, in proportion to their length. Good reasoning patterns in a long output are then learned as strongly as in a short one, and a bad pattern in a long output, such as repetition, is penalized as strongly.

**Math.** GRPO's sample-level reduction is $\frac{1}{G}\sum_{i}\frac{1}{|o_i|}\sum_{t}(\cdot)$ (Section 2.2, Equation 5). DAPO's token-level reduction is the following (Section 3.3, Equation 12):

$$
\frac{1}{\sum_{i=1}^{G}|o_i|}\sum_{i=1}^{G}\sum_{t=1}^{|o_i|}\min\Big(r_{i,t}\hat A_{i,t},\ \operatorname{clip}(r_{i,t}, 1 - \varepsilon_{low}, 1 + \varepsilon_{high})\hat A_{i,t}\Big)
$$

The paper reports that the sample-level loss leads to an unhealthy rise in entropy and in response length (Figure 4). Token-level loss gives a smaller gain in accuracy than the other techniques, but it makes the training more stable and the length increase healthier (Section 4.2).

**Tiny example, computed by `scripts/loss_reduction_example.py`.** A group has two outputs. The first is short (2 tokens) and good, with $\hat A = +1$. The second is long (8 tokens) and bad, with $\hat A = -1$. Take $r = 1$ so that each token term equals its advantage.

| Reduction | weight per token, short | weight per token, long | objective |
|---|---|---|---|
| Sample-level (Equation 5) | 0.2500 | 0.0625 | $+0.0000$ |
| Token-level (Equation 12) | 0.1000 | 0.1000 | $-0.6000$ |

Check by hand: sample-level is $\frac{1}{2}(1 + (-1)) = 0$, and token-level is $(2 \cdot 1 + 8 \cdot (-1)) / 10 = -0.6$. Under the sample-level rule each output holds half of the total weight. Under the token-level rule the long output holds 0.80 of it. The long, bad output is no longer cancelled by the short, good one.

Exercise 2 asks you to implement both reductions.

### Overlong reward shaping

**Intuition.** Generation has a maximum length, and an output that reaches it is truncated before it can give an answer. A flat punishment for truncation treats a correct but long reasoning chain the same as a wrong one, and this is noise in the reward. The paper tests two fixes. The first, overlong filtering, masks the loss of truncated outputs, so they add no gradient. The second, soft overlong punishment, keeps them but lets the penalty grow with length, so the model learns to stay short without a cliff at the limit.

**Math.** Section 3.4 defines the length penalty in Equation 13, with a hard limit $L_{max}$ and a buffer $L_{cache}$ at its end:

$$
R_{length}(y) =
\begin{cases}
0, & |y| \le L_{max} - L_{cache} \\
\dfrac{(L_{max} - L_{cache}) - |y|}{L_{cache}}, & L_{max} - L_{cache} < |y| \le L_{max} \\
-1, & L_{max} < |y|
\end{cases}
$$

This penalty is added to the rule-based correctness reward of Equation 7. Section 4.1 sets the expected maximum length to 16,384 tokens and adds a 4,096-token punishment cache, so $L_{cache} = 4096$ and the hard generation limit, which is $L_{max}$ in Equation 13, is 20,480 tokens.

**Tiny example, computed by `scripts/overlong_example.py`.**

| length $\lvert y \rvert$ | $R_{length}$ |
|---|---|
| 8000 | 0.0000 |
| 16384 | 0.0000 |
| 18432 | $-0.5000$ |
| 20000 | $-0.8828$ |
| 20480 | $-1.0000$ |
| 25000 | $-1.0000$ |

A length of 18432 is halfway through the ramp, so the penalty is $(16384 - 18432) / 4096 = -0.5$. The function is continuous, with no jump at either end of the ramp. A correct answer of that length earns $1 - 0.5 = 0.5$, and a wrong answer of the same length earns $-1 - 0.5 = -1.5$. A correct answer that reaches 20,480 tokens earns $0$, which is less than a correct answer of 16,000 tokens (reward 1).

Exercise 4 asks you to implement the penalty.

### What each technique adds

Table 1 (Section 4.2) adds the techniques one at a time to the naive GRPO baseline and reports AIME 2024 avg@32 (the mean over 32 repeats of the evaluation set). `scripts/ablation_table.py` computes the gains.

| Configuration | AIME 2024 | gain |
|---|---|---|
| Naive GRPO | 30 | |
| + Overlong filtering | 36 | +6 |
| + Clip-Higher | 38 | +2 |
| + Soft overlong punishment | 41 | +3 |
| + Token-level loss | 42 | +1 |
| + Dynamic sampling (DAPO) | 50 | +8 |

The total gain over naive GRPO is 20 points, and the final score is 3 points above the 47 of DeepSeek-R1-Zero-Qwen-32B. The other training settings (Section 4.1) are a constant learning rate of $10^{-6}$ with 20 warm-up steps, 512 prompts per batch, 16 samples per prompt, and a mini-batch size of 512, which gives 16 gradient updates per rollout step.

<!-- exercises -->

## Pitfalls

- **Raising both clip bounds.** Clip-Higher changes only $\varepsilon_{high}$. The paper keeps $\varepsilon_{low} = 0.2$, because a larger lower bound would let tokens be pushed toward probability 0 and collapse the sampling space (Section 3.1).
- **Reading the 1.08 bound as reachable.** The paper's example gives 1.08 for a token with probability 0.9 at $\varepsilon = 0.2$. A probability can't exceed 1. The point of the example is that the clip doesn't limit likely tokens, so Exercise 1 caps the ceiling at 1.
- **Dividing by a zero standard deviation.** A uniform group has $\operatorname{std} = 0$, and Equation 9 gives $0/0$. Dynamic sampling removes these groups from the batch, but a guard that returns 0 is still needed wherever the formula runs before the filter. Without a guard, one `nan` poisons the whole batch loss.
- **Filtering after you build the batch.** The batch-filling loop keeps sampling until it has the full number of kept groups. If you filter a fixed batch instead, the batch shrinks, and you lose the consistent number of prompts that Equation 11 is meant to give.
- **Forgetting the cost of dynamic sampling.** The filter draws more prompts than it keeps. In the illustrative pool on this page, 1.60 times as many. The paper reports that the total training time isn't much affected, because convergence takes fewer steps (Section 4.2), but that depends on the pool and the system, so measure it.
- **Mixing up the two lengths.** Section 4.1 calls 16,384 tokens the expected maximum length, and the hard generation limit is 20,480. In Equation 13, $L_{max}$ is the hard limit (20,480) and $L_{max} - L_{cache}$ is where the penalty starts (16,384). Reading $L_{max}$ as 16,384 shifts the whole ramp.
- **Treating overlong filtering and soft punishment as one technique.** Filtering masks the loss of truncated outputs. Soft punishment adds a graded penalty to the reward. Table 1 lists them as separate rows, with gains of +6 and +3.
- **Averaging per output when you mean per token.** Equation 12 divides by the total token count of the group, $\sum_i |o_i|$, and not by $G$ and not by each $|o_i|$. The two reductions agree only when every output has the same length.
- **Expecting entropy to rise forever.** The paper wants entropy kept in an appropriate range. Too low means the distribution is sharp and exploration is lost. Too high goes with gibberish and repetition. Section 4.3 reports that a slow upward trend in entropy went with better accuracy.
- **Reading Table 1 as independent effects.** Each row adds a technique to the previous row, in one order. The gains depend on that order, so don't read +8 as "dynamic sampling is worth 8 points" in every setting.
- **Keeping the KL term from GRPO.** DAPO drops it (Section 2.3). If you carry over a GRPO implementation, remove the term, or the comparison to the paper isn't the same algorithm.
- **Assuming the reward is the R1 reward.** R1-Zero adds a format reward. DAPO's reward is only the $\pm 1$ accuracy rule of Equation 7, plus the length penalty of Equation 13, on answers that the authors rewrote to integers (Section 3.5).

## Further reading

- The [paper's arXiv page](https://arxiv.org/abs/2503.14476) and the [project page](https://dapo-sia.github.io/), which the paper links.
- The [verl framework](https://github.com/volcengine/verl), which the paper's implementation builds on.
- [DeepSeekMath](/grpo/) for the GRPO advantage and objective that DAPO modifies, [Proximal policy optimization](/ppo/) for the clipped surrogate, and [DeepSeek-R1](/r1/) for the R1-Zero setting that DAPO sets out to reproduce.
