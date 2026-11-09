---
id: "grpo"
title: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
authors: "Shao et al."
year: 2024
venue: "arXiv preprint"
arxiv: "2402.03300"
url: "https://arxiv.org/abs/2402.03300"
track: rl
tier: core
est_hours: 5
status: full
summary: "DeepSeekMath introduces Group Relative Policy Optimization (GRPO), a variant of PPO that has no value network. For each question it samples a group of outputs, scores them, and sets every token's advantage to its output's reward standardized by the group mean and standard deviation. The KL penalty moves out of the reward and into the loss, and the method lifts a 7B instruction-tuned model from 46.8% to 51.7% on MATH."
related_course: "rl: reasoning post-training"
prereqs:
  - id: ppo
    reason: "GRPO keeps PPO's clipped probability-ratio objective unchanged and replaces only the source of the advantage, so you need Equation 7 of PPO and its ratio and clip to read Equation 3 here."
  - id: instructgpt
    reason: "The paper takes the RLHF recipe of an SFT model, a reward model, and RL with a KL term to a reference model from InstructGPT, and then changes where the KL term goes and how the baseline is computed."
background: [group-relative-baselines, outcome-rewards]
course_links:
  - title: "Group-relative policy optimization"
    url: "https://rl.ifkash.dev/lessons/055-group-relative-policy-optimization"
self_check:
  - q: 'Four sampled outputs for one question earn rewards $[1, 0, 0, 0]$. What are the advantages under GRPO outcome supervision (use the population standard deviation), and what do they sum to?'
    a: 'The mean is $0.25$ and the standard deviation is $\sqrt{0.25 \cdot 0.75} \approx 0.4330$. The correct output gets $0.75 / 0.4330 \approx 1.7321$ and each wrong output gets $-0.25 / 0.4330 \approx -0.5774$. The advantages sum to 0, because the baseline is the group mean.'
    see: group-relative-baselines
  - q: 'PPO clips the ratio $r = \pi_\theta / \pi_{\theta_{old}}$ with $\epsilon = 0.2$. A sample has $r = 1.5$ and $\hat A = -1$. What is its term in the objective, and does GRPO change that?'
    a: 'The term is $\min(1.5 \cdot (-1),\ 1.2 \cdot (-1)) = -1.5$, the full penalty, with a gradient. GRPO changes nothing here: Equation 3 uses the same clipped term as PPO, with $\hat A_{i,t}$ from the group in place of the advantage from a critic.'
    see: ppo
  - q: 'In RLHF as InstructGPT does it, where does the KL penalty to the reference policy go, and where does GRPO put it?'
    a: 'InstructGPT subtracts a per-token KL term from the reward, so it flows into the advantage. GRPO leaves the reward alone and subtracts $\beta\,\mathbb{D}_{KL}[\pi_\theta \,\|\, \pi_{ref}]$ from the objective directly (Section 4.1.1, Equation 3), so $\hat A_{i,t}$ stays a function of the group rewards only.'
    see: instructgpt
---

## The problem

InstructGPT-style RLHF (see [Training language models to follow instructions with human feedback](/instructgpt/)) runs PPO (see [Proximal policy optimization](/ppo/)) on a language model. PPO is an actor-critic method. Its advantage $A_t$ comes from generalized advantage estimation (Section 4.1.1), which needs a learned value function $V_\psi$ as a baseline. For a language model, that value function is "typically another model of comparable size as the policy model" (Section 4.1.1), so it adds a second network's memory and compute to training.

The paper names a second difficulty. In a language-model task, usually only the last token receives a score from the reward model. A value function that must be accurate at every earlier token, from a single end-of-sequence signal, is hard to train (Section 4.1.1).

The same section describes where PPO puts the KL penalty to the reference model: inside the reward, as a per-token term (Section 4.1.1, Equation 2):

$$
r_t = r_\varphi(q, o_{\le t}) - \beta \log \frac{\pi_\theta(o_t \mid q, o_{<t})}{\pi_{ref}(o_t \mid q, o_{<t})}
$$

That choice feeds the penalty into the advantage and so into the critic's targets.

Here is the setting this paper targets. A question $q$ has a checkable answer. The model writes a chain of thought $o$, and a reward model (or a rule) returns one number for the whole output. Each question has many sampled outputs, and the reward is only meaningful relative to the other outputs for that question. The paper observes that reward models are trained on comparisons between outputs for the same question, so a comparison within a group fits what they were trained to do (Section 4.1.1).

The paper's answer is Group Relative Policy Optimization (GRPO). It removes the critic, builds the baseline from a group of samples, and moves the KL term out of the reward. DeepSeekMath-RL 7B, trained with GRPO from DeepSeekMath-Instruct 7B on about 144K chain-of-thought questions, scores 88.2% on GSM8K and 51.7% on MATH, up from 82.9% and 46.8% (Section 4.2, Table 5).

## Key ideas

The sections that follow use the paper's notation: a question $q$, a group of $G$ sampled outputs $\{o_1, \dots, o_G\}$, the output $o_i$ with $|o_i|$ tokens and token $o_{i,t}$, rewards $r_i$, the advantage $\hat A_{i,t}$, the clip range $\varepsilon$, the KL coefficient $\beta$, and the reference model $\pi_{ref}$. The paper writes the clip range as $\varepsilon$, and the [PPO page](/ppo/) writes it $\epsilon$. They are the same quantity. Every number comes from a script in `scripts/`. Run any script from the `scripts/` folder to reproduce the output.

### The group-relative advantage

**Intuition.** PPO asks a critic, "how good is this state?", and measures each action against the answer. GRPO asks the other samples instead. For one question, draw several outputs. An output that scores above its siblings was a good draw for this question, and one that scores below was a bad draw. The group's own average is the baseline, and no network predicts it.

**Math.** For each question $q$, GRPO samples a group $\{o_1, \dots, o_G\}$ from $\pi_{\theta_{old}}$ and scores it with a reward model, giving $\mathbf{r} = \{r_1, \dots, r_G\}$. Outcome supervision subtracts the group mean, divides by the group standard deviation, and gives the result to every token of the output (Section 4.1.2, unnumbered):

$$
\hat A_{i,t} = \widetilde r_i = \frac{r_i - \operatorname{mean}(\mathbf{r})}{\operatorname{std}(\mathbf{r})}
$$

The paper doesn't say whether $\operatorname{std}$ is the population or the sample standard deviation. This page and the exercises use the population standard deviation (`np.std` with `ddof=0`). The paper also doesn't say what happens when every reward is equal. The page's code adds $10^{-6}$ to the standard deviation, which makes the advantages 0 in that case. That is an implementation choice and not a statement from the paper.

Three properties follow from the formula. The advantages of a group have mean 0 and standard deviation 1. The advantage doesn't change when you rescale or shift all rewards. And the same $\hat A_i$ applies to all $|o_i|$ tokens of output $i$, so the algorithm credits or blames the whole response and not individual tokens.

**Tiny example, computed by `scripts/group_advantage_example.py`.** A group of $G = 8$ samples has 0/1 rewards $[1, 0, 0, 1, 0, 0, 0, 0]$. The mean is 0.25 and the standard deviation is 0.4330. Then:

| Output | Reward | Advantage |
|---|---|---|
| Correct | 1 | $+1.7321$ |
| Wrong | 0 | $-0.5773$ |

The correct samples share more credit, because they are rarer. Check the mean by hand: $2 \cdot 1.7321 + 6 \cdot (-0.5773) \approx 0$. Rewards $10\mathbf{r} + 3$ give the same advantages.

A group where all 8 rewards are 1 has advantage 0 everywhere, so it contributes no policy gradient. This happens often when the question is too easy or too hard. For a success rate $p = 0.05$, the probability that all 8 samples earn the same reward is $0.05^8 + 0.95^8 = 0.6634$. At $p = 0.5$ it is $0.0078$. The paper uses $G = 64$ (Section 4.2), which makes empty groups rarer at a given $p$.

Exercise 1 asks you to implement this function for a batch of groups.

### The GRPO objective

**Intuition.** Keep PPO's loop exactly as it is: a ratio between the new and old policy, a clip that stops a sample from pulling the policy too far, and an average over tokens. Swap in the group advantage, and subtract a KL penalty to a reference model.

**Math.** PPO maximizes (Section 4.1.1, Equation 1):

$$
\mathcal{J}_{PPO}(\theta) = \mathbb{E}\left[\frac{1}{|o|}\sum_{t=1}^{|o|}\min\!\Big(\frac{\pi_\theta(o_t \mid q, o_{<t})}{\pi_{\theta_{old}}(o_t \mid q, o_{<t})}A_t,\ \operatorname{clip}\Big(\frac{\pi_\theta(o_t \mid q, o_{<t})}{\pi_{\theta_{old}}(o_t \mid q, o_{<t})},\, 1-\varepsilon,\, 1+\varepsilon\Big)A_t\Big)\right]
$$

GRPO maximizes (Section 4.1.1, Equation 3):

$$
\mathcal{J}_{GRPO}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\Big\{\min\!\Big(\rho_{i,t}\,\hat A_{i,t},\ \operatorname{clip}\big(\rho_{i,t},\, 1-\varepsilon,\, 1+\varepsilon\big)\hat A_{i,t}\Big) - \beta\,\mathbb{D}_{KL}\big[\pi_\theta \,\|\, \pi_{ref}\big]\Big\}\right]
$$

Here $\rho_{i,t} = \pi_\theta(o_{i,t} \mid q, o_{i,<t}) / \pi_{\theta_{old}}(o_{i,t} \mid q, o_{i,<t})$. The paper writes the ratio out in full and doesn't name it, so $\rho_{i,t}$ is this page's shorthand, not the paper's. The expectation runs over questions $q \sim P(Q)$ and groups $\{o_i\} \sim \pi_{\theta_{old}}(O \mid q)$. Two details matter. First, the average is per output: each output's token terms are divided by its own length $|o_i|$, and then the $G$ outputs are averaged. Second, the KL term is inside the braces for each token, and $\hat A_{i,t}$ doesn't contain it. The paper says this avoids complicating the calculation of $\hat A_{i,t}$ (Section 4.1.1).

Algorithm 1 loops: sample $G$ outputs per question from $\pi_{\theta_{old}}$, score them, compute $\hat A_{i,t}$, and run $\mu$ gradient updates on the objective. Line 11 of Algorithm 1 cites "Equation 21" for the objective. Equation 21 in Appendix A.1.6 is the gradient coefficient, so Equation 3 is the objective the algorithm maximizes. In the paper's experiments, "the policy model only has a single update following each exploration stage" (Section 4.2), so $\mu = 1$, $\pi_{\theta_{old}} = \pi_\theta$ at the time of the update, and the ratio is 1.

**Tiny example, computed by `scripts/objective_example.py`.** A group of $G = 2$ outputs with $|o_1| = 2$ and $|o_2| = 4$ tokens. Rewards are $[1, 0]$, so $\hat A = [+1, -1]$. Take $\varepsilon = 0.2$ and $\beta = 0.04$ (the page chooses $\varepsilon$ for the example; the paper's $\beta$ is in Section 4.2). The ratios and the per-token clipped terms are:

| Output | Ratios $\rho$ | Clipped terms |
|---|---|---|
| 1 ($\hat A = +1$) | 1.3, 0.9 | 1.2, 0.9 |
| 2 ($\hat A = -1$) | 1.1, 1.0, 0.7, 1.25 | -1.1, -1.0, -0.8, -1.25 |

The first ratio of output 1 is above $1 + \varepsilon$ with a positive advantage, so its gain is capped at 1.2. The third ratio of output 2 is below $1 - \varepsilon$ with a negative advantage. Its term is $\min(-0.7, -0.8) = -0.8$, the lower value, and its gradient is 0. The KL estimates (next section) are small, 0 to 0.05 per token. Averaging each output over its own tokens gives 1.0495 and -1.0383, and the objective is $J = 0.0056$. Without the KL term, $J = 0.0062$. Averaging over all six tokens at once, which isn't what Equation 3 says, gives $-0.3423$, because the longer output gets 4 of the 6 votes. Exercise 2 asks you to implement this objective.

### The KL estimator and where the KL term goes

**Intuition.** The KL penalty keeps the policy close to a reference model, so the RL step can't wander far from the SFT model just to please the reward model. PPO folds the penalty into the reward, so it enters through the advantage. GRPO adds it to the loss as a separate term, which keeps the advantage purely about reward.

**Math.** GRPO estimates the KL for each token with the following estimator, which the paper attributes to Schulman (2020) and calls unbiased and guaranteed positive (Section 4.1.1, Equation 4):

$$
\mathbb{D}_{KL}\big[\pi_\theta \,\|\, \pi_{ref}\big] = \frac{\pi_{ref}(o_{i,t} \mid q, o_{i,<t})}{\pi_\theta(o_{i,t} \mid q, o_{i,<t})} - \log\frac{\pi_{ref}(o_{i,t} \mid q, o_{i,<t})}{\pi_\theta(o_{i,t} \mid q, o_{i,<t})} - 1
$$

Write $x = \pi_{ref}/\pi_\theta$ for the ratio at the sampled token. The estimator is $x - \log x - 1$. Since $\log x \le x - 1$, it is never negative, and it is 0 only when $x = 1$. Averaged over tokens sampled from $\pi_\theta$, it equals the true KL: $\mathbb{E}_{\pi_\theta}[x - 1] = 0$ and $\mathbb{E}_{\pi_\theta}[-\log x] = \mathrm{KL}(\pi_\theta \| \pi_{ref})$. Equation 2's term $\log(\pi_\theta / \pi_{ref}) = -\log x$ is the same quantity without the $x - 1$ part, and it has the same mean but can be negative on single samples.

Appendix A.1.6 gives the gradient with $\pi_{\theta_{old}} = \pi_\theta$ (Equations 19 and 20). The KL term turns into a per-token coefficient on $\nabla_\theta \log \pi_\theta$:

$$
\nabla_\theta \mathcal{J}_{GRPO}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\Big(\hat A_{i,t} + \beta\Big(\frac{\pi_{ref}(o_{i,t} \mid o_{i,<t})}{\pi_\theta(o_{i,t} \mid o_{i,<t})} - 1\Big)\Big)\nabla_\theta \log \pi_\theta(o_{i,t} \mid q, o_{i,<t})\right]
$$

The gradient coefficient (Equation 21) is the bracket $\hat A_{i,t} + \beta\,(\pi_{ref}/\pi_\theta - 1)$. When the policy has made a token less likely than the reference did, $\pi_{ref}/\pi_\theta > 1$ and the KL part pushes the probability back up.

**Tiny example, computed by `scripts/kl_example.py`.** Take one token with $\pi_\theta = 0.5$ and $\pi_{ref} = 0.25$. Then $x = 0.5$, and the estimator is $0.5 - \log 0.5 - 1 = 0.1931$. The Equation 2 term is $\log(0.5/0.25) = 0.6931$. Swap the two probabilities and the estimator becomes $0.3069$, so the direction of the ratio matters. For a 3-token vocabulary with $\pi_\theta = [0.6, 0.3, 0.1]$ and $\pi_{ref} = [0.2, 0.5, 0.3]$, the true KL is $0.3961$. Averaging the estimator over $\pi_\theta$ gives exactly $0.3961$. Its per-token values are $[0.4319, 0.1558, 0.9014]$, all positive, while the log-ratio takes the values $[1.0986, -0.5108, -1.0986]$. In 100,000 samples (seed 0), the estimator has mean 0.3955 and standard deviation 0.2083, and the log-ratio has mean 0.3973 and standard deviation 0.8745, with 39.96% of its samples below 0. Both means are close to 0.3961, so the benefit of Equation 4 is variance and a non-negative signal, and not a different expectation.

Exercise 2 asks you to implement the estimator. Exercise 4 checks the gradient coefficient numerically.

### Process supervision

**Intuition.** An outcome reward credits every token of a correct output equally, including the wrong turns before the lucky finish. A process reward model scores each reasoning step, so a good step can be credited and a bad step blamed. A token should be credited for the quality of everything from its step to the end.

**Math.** A process reward model scores the steps of each output. The reward of step $j$ is given at the end token of that step, $index(j)$, and output $i$ has $K_i$ steps, so the rewards are $R = \{\{r_1^{index(1)}, \dots, r_1^{index(K_1)}\}, \dots, \{r_G^{index(1)}, \dots, r_G^{index(K_G)}\}\}$. The paper normalizes with the mean and standard deviation of all of $R$, and sets the advantage of a token as the sum of the normalized rewards of the steps that end at or after it (Section 4.1.3, unnumbered):

$$
\widetilde r_i^{\,index(j)} = \frac{r_i^{index(j)} - \operatorname{mean}(R)}{\operatorname{std}(R)}, \qquad \hat A_{i,t} = \sum_{index(j) \ge t} \widetilde r_i^{\,index(j)}
$$

For a token inside step $j$, the sum covers step $j$ and every later step. In this page's code, the advantage is stored once per step, and every token in the step shares it. The group objective is still Equation 3.

**Tiny example, computed by `scripts/process_example.py`.** A group of 3 outputs with step rewards $[1, 1, 0]$, $[1, 0]$, and $[0, 0, 0, 0]$. Over all 9 steps, the mean is 0.3333 and the standard deviation is 0.4714. The normalized rewards are $+1.4142$ for a reward of 1 and $-0.7071$ for a reward of 0. The advantages by step are:

| Output | Advantage of step 1, 2, ... |
|---|---|
| 1 | 2.1213, 0.7071, -0.7071 |
| 2 | 0.7071, -0.7071 |
| 3 | -2.8284, -2.1213, -1.4142, -0.7071 |

Output 3 has four failed steps and its first step carries all four penalties. Normalizing each output on its own would erase this, because an all-zero output would normalize to zero. With one step per output, the formula reduces to the outcome advantage of the first idea.

Exercise 3 asks you to implement this.

### What the paper shows

Section 4.2 describes the training. The data are about 144K chain-of-thought questions from GSM8K and MATH. The policy starts from DeepSeekMath-Instruct 7B, and the learning rate is $10^{-6}$. The KL coefficient is $\beta = 0.04$, there are 64 outputs per question, the maximum length is 1024, and the batch size is 1024. Table 5 reports chain-of-thought accuracy for DeepSeekMath-Instruct 7B and DeepSeekMath-RL 7B: GSM8K 82.9% to 88.2%, MATH 46.8% to 51.7%, MGSM-zh 73.2% to 79.6%, and CMATH 84.6% to 88.8%. The paper says the model improves on all the benchmarks, including those whose questions weren't in the RL data.

Section 5.2.1 compares methods on a 1.3B model, with a shared gradient form (Equation 5) split into a data source, a reward function, and a gradient coefficient. Figure 5 shows that Online RFT beats offline RFT, that GRPO beats Online RFT, and that GRPO with process supervision beats GRPO with outcome supervision. The paper reads the GRPO result as the benefit of a gradient coefficient that depends on the reward: Online RFT gives every correct output the same positive weight and doesn't penalize incorrect outputs (the gradient coefficient of Equation 10 is the indicator of a correct answer), while GRPO can penalize a wrong output and reward a better one more than a weaker one. Figure 6 shows that a first iteration of iterative RL (Section 4.1.4) helps most. Section 5.2.2 finds that RL improves Maj@K but not Pass@K, and reads this as the RL step making the output distribution more robust and not as adding new capability.

#### A toy comparison with a learned baseline

The paper doesn't include a bandit experiment. This one is the page's own. The script `scripts/bandit_compare.py` runs a 3-arm Bernoulli bandit with success probabilities $[0.2, 0.5, 0.8]$, groups of $G = 8$, a learning rate of 0.5, and one update per group, as in the paper's setup, with 50 seeds. The three methods differ only in the advantage:

| Method | Advantage |
|---|---|
| GRPO | $(r_i - \operatorname{mean}) / \operatorname{std}$ over the group |
| Group mean | $r_i - \operatorname{mean}$ over the group |
| Critic | $r_i - V$, with $V$ a learned scalar that tracks the batch mean |

The mean probability of the best arm is:

| Step | GRPO | Group mean | Critic |
|---|---|---|---|
| 10 | 0.678 | 0.501 | 0.526 |
| 25 | 0.884 | 0.705 | 0.744 |
| 50 | 0.950 | 0.864 | 0.887 |
| 100 | 0.981 | 0.951 | 0.958 |
| 200 | 0.991 | 0.977 | 0.980 |

Two readings follow. The group-mean baseline, with no learned parameters, performs about as well as the learned critic here. The bandit has no state, so the critic is one number and is easy to learn, which makes this a weak test of the paper's claim that a critic is hard to learn for long sequences. GRPO is faster than both at the same learning rate, but part of that gain is the division by the standard deviation, which makes the update larger. It isn't a controlled test of the baseline alone, and you shouldn't read it as GRPO being a better estimator.

<!-- exercises -->

## Pitfalls

- **Averaging over all tokens in the batch.** Equation 3 divides each output's token sum by its own length $|o_i|$ and then averages over $G$. A single mean over every token weights long outputs more. On the page's example, the two ways give $0.0056$ and $-0.3423$. Exercise 2's mutant checks for this bug.
- **Putting the KL term in the advantage.** The point of Equation 3 is that the KL penalty is a separate loss term, and $\hat A_{i,t}$ depends on the group rewards only. If you also subtract $\beta \log(\pi_\theta/\pi_{ref})$ from the reward, as in Equation 2, the penalty counts twice.
- **Swapping the KL ratio.** The estimator uses $\pi_{ref}/\pi_\theta$. With $\pi/\pi_{ref}$ in its place, the number is still non-negative but is a different quantity ($0.3069$ and not $0.1931$ on the page's example), and it gives the wrong gradient. The estimate is also on tokens sampled from $\pi_\theta$, which is the policy that produced them.
- **Dividing by a zero standard deviation.** A group of equal rewards has $\operatorname{std}(\mathbf{r}) = 0$, and the formula gives $0/0$. The paper doesn't discuss it. Without a guard, one `nan` poisons the whole batch loss. With the guard, the group gives no gradient, which is also why small groups waste samples on very easy or very hard questions.
- **Sample versus population standard deviation.** The paper writes $\operatorname{std}$ without saying which. With $G = 2$, the two differ by a factor of $\sqrt 2$. Pick one and keep it the same between training and any test you write.
- **Normalizing process rewards per output.** Section 4.1.3 normalizes with the mean and standard deviation of the whole set $R$. Normalizing each output separately removes the signal that one output is worse than another. Exercise 3's mutant checks for this bug.
- **Using the advantage as a state value.** $\hat A_{i,t}$ is the same for every token of an output under outcome supervision. It is a statement about the whole response, not an estimate of the value of a prefix, and it doesn't distinguish a good first step from a bad last step.
- **Clipping when there is a single update.** With $\mu = 1$, as in the paper's training, the ratio is 1 at the time of the update, and the clip and the $\pi_{\theta_{old}}$ copy do nothing. They start to matter when you take several update steps on one batch.
- **Reading the equation numbers.** Algorithm 1 cites Equation 21 for the objective, but Equation 3 is the objective and Equation 21 is the gradient coefficient.

## Further reading

- The [paper's arXiv page](https://arxiv.org/abs/2402.03300) and the authors' code repository, [DeepSeek-Math](https://github.com/deepseek-ai/DeepSeek-Math), which the paper links.
- [Proximal policy optimization](/ppo/) for the clipped objective that GRPO keeps, and [Training language models to follow instructions with human feedback](/instructgpt/) for the KL-in-the-reward recipe that GRPO changes.
- John Schulman, "Approximating KL Divergence" (2020), the note the paper cites for the estimator in Equation 4.
