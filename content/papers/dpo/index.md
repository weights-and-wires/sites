---
id: "dpo"
title: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
authors: "Rafailov et al."
year: 2023
venue: "NeurIPS 2023"
arxiv: "2305.18290"
url: "https://arxiv.org/abs/2305.18290"
track: rl
tier: core
est_hours: 5
status: full
summary: "DPO removes the explicit reward model and the reinforcement learning loop from RLHF. The KL-regularized reward-maximization problem has a closed-form optimal policy, so a reward can be written as a scaled log ratio between the policy and a frozen reference model, and the Bradley-Terry preference likelihood then depends on the policy alone. Training becomes binary cross-entropy on preference pairs, which matches or beats PPO-based RLHF on sentiment control, summarization, and single-turn dialogue in the paper's experiments."
related_course: "rl: preference optimization"
prereqs:
  - id: instructgpt
    reason: "DPO starts from the InstructGPT recipe of SFT, a Bradley-Terry reward model, and a KL-penalized RL stage, and it replaces the last two stages with one loss."
background: [log-probability-ratios, closed-form-optimal-policy]
course_links:
  - title: "Direct preference optimization"
    url: "https://rl.ifkash.dev/lessons/070-direct-preference-optimization"
  - title: "PPO and DPO: two ways to reach the leashed optimum"
    url: "https://ml.ifkash.dev/lessons/080-ppo-and-dpo"
self_check:
  - q: 'Which stages of the InstructGPT pipeline does DPO keep, and which does it replace? What plays the role of the reward model?'
    a: 'DPO keeps supervised fine-tuning, because $\pi_{\text{ref}}$ is the SFT model, and it keeps the preference data. It replaces the separate reward model and the PPO stage with one classification loss. The policy itself is the reward model: the implicit reward is $\hat r_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$.'
    see: instructgpt
  - q: 'With $\beta = 0.1$, the log ratios $\log\frac{\pi_\theta}{\pi_{\text{ref}}}$ are $+0.8$ for the preferred response and $-0.5$ for the rejected one. What is the DPO loss for this pair, and what is it when $\pi_\theta = \pi_{\text{ref}}$?'
    a: 'The margin is $0.1 \cdot (0.8 - (-0.5)) = 0.13$, so the loss is $-\log\sigma(0.13) \approx 0.6303$. When $\pi_\theta = \pi_{\text{ref}}$ both log ratios are $0$, the margin is $0$, and the loss is $\log 2 \approx 0.6931$ for every pair.'
    see: log-probability-ratios
  - q: 'Two responses have $\pi_{\text{ref}} = [0.5, 0.5]$ and rewards $[0, 1]$. For $\beta = 1$, what is the optimal policy of Equation 4, and how does it relate to the Bradley-Terry probability that the second response wins?'
    a: 'The unnormalized weights are $[0.5, 0.5e]$, with $Z \approx 1.8591$, so $\pi^* \approx [0.2689, 0.7311]$. With a uniform reference this equals $[\sigma(-1), \sigma(1)]$, which is the Bradley-Terry preference probability for a reward gap of $1$.'
    see: closed-form-optimal-policy
---

## The problem

RLHF, as InstructGPT ran it (see [InstructGPT](/instructgpt/)), has three stages. Supervised fine-tuning gives a model $\pi^{\text{SFT}}$. A reward model $r_\phi$ is fit to human preference labels. Then a policy is trained with PPO to maximize that reward under a KL penalty to the SFT model (Section 3). The paper lists what this costs: the pipeline "involves training multiple LMs and sampling from the LM policy in the loop of training, incurring significant computational costs" (Section 1).

Consider one concrete failure. The RL stage needs a low-variance learning signal for each sampled response. Section 5.2 shows that the reward the actor-critic maximizes has a normalizing term, $\beta \log \sum_y \pi_{\text{ref}}(y \mid x)\exp(\frac{1}{\beta} r_\phi(x, y))$, that acts as the soft value function of the reference policy. This term doesn't change the optimal solution, but if you leave it out, "the policy gradient of the objective could have high variance, making learning unstable" (Section 5.2, Equation 10). Practitioners cover for it with a learned value function, which is hard to fit, or with a human-completion baseline, which is a one-sample estimate. The paper also tried a publicly released PPO-trained model on the Anthropic Helpful and Harmless data and found no prompt or sampling temperature that did better than the base Pythia-2.8B model (Section 6.2).

The paper asks whether the RL stage can be dropped while the same objective is kept. Section 4 answers yes. The result is a loss you can write in a few lines of PyTorch (Appendix B), with no sampling during training and almost no hyperparameter tuning (Abstract, Section 7). On the paper's tasks, DPO reaches a win rate of about 61% on Reddit TL;DR summarization at sampling temperature 0, against 57% for PPO at its best temperature (Section 6.2, Figure 2). It is also the only method in the paper that improves over the preferred completions on single-turn Anthropic-HH dialogue (Figure 3, Section 6.2).

## Key ideas

The sections that follow use the paper's notation: a prompt $x$, a response $y$, the preferred and dispreferred responses $y_w$ and $y_l$, the trainable policy $\pi_\theta$, the frozen reference policy $\pi_{\text{ref}}$, the KL coefficient $\beta$, the true reward $r^*$, and the implicit reward $\hat r_\theta$. The paper writes the partition function as $Z(x)$ and the logistic function as $\sigma$. Every number comes from a script in `scripts/`. Run any script from the repository root to reproduce the output.

One notation difference: the paper writes $\pi_\theta(y_w \mid x)$ for the probability of a whole response. Code works with the log of that, which is the sum of the token log-probabilities. The exercises take these sequence log-probabilities as plain numbers, called `pi_w`, `pi_l`, `ref_w`, and `ref_l`. The paper's Appendix B code takes them as one flat batch and gathers pairs with index arrays, and the exercises skip that step.

### The KL-regularized objective has a closed-form optimum

**Intuition.** RLHF asks for the policy that scores well on a reward but stays close to the reference model. If the policy can be any distribution over a finite set of responses, this tug-of-war has an exact answer: reweight the reference by $e^{r/\beta}$ and renormalize. No sampling or RL is needed to state the optimum, only to find it with a neural network.

**Math.** The RL objective is (Section 3, Equation 3)

$$
\max_{\pi_\theta}\; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(y \mid x)}\big[r_\phi(x, y)\big] - \beta\, \mathbb{D}_{\mathrm{KL}}\big[\pi_\theta(y \mid x)\,\|\,\pi_{\text{ref}}(y \mid x)\big]
$$

For a general reward $r$, its optimum is (Section 4, Equation 4)

$$
\pi_r(y \mid x) = \frac{1}{Z(x)}\,\pi_{\text{ref}}(y \mid x)\exp\!\Big(\frac{1}{\beta}\,r(x, y)\Big), \qquad Z(x) = \sum_y \pi_{\text{ref}}(y \mid x)\exp\!\Big(\frac{1}{\beta}\,r(x, y)\Big)
$$

Appendix A.1 derives it in three moves. First, rewrite the objective as $\min_\pi \mathbb{E}_{y\sim\pi}\big[\log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)} - \frac{1}{\beta}r(x,y)\big]$ (Equation 12). Second, define $\pi^*$ as the right-hand side of Equation 4, so that the expression equals $\mathbb{D}_{\mathrm{KL}}(\pi\,\|\,\pi^*) - \log Z(x)$ (Equations 13 and 14). Third, note that $Z(x)$ doesn't depend on $\pi$ and that the KL is zero only when $\pi = \pi^*$ (Gibbs' inequality), which gives Equation 15.

The paper points out that $Z(x)$ is a sum over every possible response, which is "expensive to estimate", so Equation 4 isn't usable directly (Section 4). The next idea gets around that.

**Tiny example, computed by `scripts/closed_form_example.py`.** Take three responses with $\pi_{\text{ref}} = [0.70, 0.25, 0.05]$ and rewards $r = [0, 1, 2]$. For $\beta = 1$, the unnormalized weights $\pi_{\text{ref}}\,e^{r/\beta}$ are $[0.7000, 0.6796, 0.3695]$, so $Z = 1.7490$ and

$$
\pi^* = [0.4002,\; 0.3885,\; 0.2112]
$$

Response 2 has the highest reward and began with 5 percent of the mass. It now holds 21 percent. For $\beta = 0.5$, $Z = 5.2772$ and $\pi^* = [0.1326, 0.3500, 0.5173]$. A smaller $\beta$ means a longer leash, and the optimum moves more mass toward the best response. These match the toy values on the InstructGPT page, which uses the same closed form without deriving it.

Exercise 1 asks you to compute $\pi_r$ and $\log Z$ stably.

### A reward is a log ratio, and the partition function cancels

**Intuition.** Equation 4 maps any reward to an optimal policy. Run it backward and any policy is the optimum for some reward. The preference data only ever compare two rewards for the same prompt, so the per-prompt constant that comes from $Z(x)$ drops out.

**Math.** Take the log of Equation 4 and solve for the reward (Section 4, Equation 5):

$$
r(x, y) = \beta \log\frac{\pi_r(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)
$$

The Bradley-Terry model says that human preferences come from a latent reward $r^*$ (Section 3, Equation 1):

$$
p^*(y_1 \succ y_2 \mid x) = \frac{\exp(r^*(x, y_1))}{\exp(r^*(x, y_1)) + \exp(r^*(x, y_2))} = \sigma\big(r^*(x, y_1) - r^*(x, y_2)\big)
$$

This depends only on the difference of the two rewards. Substitute Equation 5 for $r^*$: the term $\beta \log Z(x)$ is the same for both responses and cancels, which leaves an expression in the optimal policy $\pi^*$ and $\pi_{\text{ref}}$ alone (Section 4, Equation 6):

$$
p^*(y_1 \succ y_2 \mid x) = \frac{1}{1 + \exp\!\Big(\beta \log\frac{\pi^*(y_2 \mid x)}{\pi_{\text{ref}}(y_2 \mid x)} - \beta \log\frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)}\Big)}
$$

The paper's phrase for this is a change of variables: the preference loss becomes a function of the policy directly (Section 1). It then calls $\hat r_\theta(x, y) = \beta \log\frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$ "the reward implicitly defined by the language model $\pi_\theta$ and reference model $\pi_{\text{ref}}$" (Section 4).

**Tiny example, computed by `scripts/closed_form_example.py`.** Use the numbers from the last example with $\beta = 1$. The log ratios $\beta\log(\pi^*/\pi_{\text{ref}})$ are $[-0.5591, 0.4409, 1.4409]$. Adding $\beta\log Z = \log 1.7490 = 0.5591$ returns the true rewards $[0, 1, 2]$, which is Equation 5. The preference between responses 2 and 0 is $\sigma(2 - 0) = 0.8808$ from the rewards, and $\sigma(1.4409 - (-0.5591)) = 0.8808$ from the log ratios alone. The log-ratio route never needed $Z$.

Exercise 1's second function, `implicit_reward`, is the log ratio in the first row of that example.

### The DPO loss is binary cross-entropy on preference pairs

**Intuition.** Reward modeling fits $r_\phi$ by maximum likelihood on the Bradley-Terry model. DPO does the same with the reward parameterized as a log ratio, so the network being fit is the policy. The loss is a logistic regression in which the "feature" is how much more the policy has raised the winner's probability than the loser's, compared with the reference.

**Math.** The reward-modeling loss is (Section 3, Equation 2)

$$
\mathcal{L}_R(r_\phi, \mathcal{D}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\big[\log\sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)\big]
$$

Replace $r_\phi$ with $\hat r_\theta$ and the loss becomes (Section 4, Equation 7)

$$
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\Big[\log\sigma\Big(\beta\log\frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta\log\frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\Big)\Big]
$$

The paper's pipeline has two steps. Sample two completions per prompt from $\pi_{\text{ref}}$ and collect preference labels, or reuse a public dataset. Then minimize $\mathcal{L}_{\text{DPO}}$ for a chosen $\beta$ (Section 4, "DPO outline"). When $\pi_{\text{SFT}}$ is available, $\pi_{\text{ref}} = \pi_{\text{SFT}}$. When it isn't, the paper fits $\pi_{\text{ref}}$ by maximizing the likelihood of the preferred completions, to reduce the distribution shift between the data and the reference. Appendix B gives the loss as `-F.logsigmoid(beta * (pi_logratios - ref_logratios))` and uses $\beta = 0.1$, a batch size of 64, RMSprop with a learning rate of $10^{-6}$ warmed up over 150 steps, and $\beta = 0.5$ for TL;DR summarization.

**Tiny example, computed by `scripts/loss_example.py`.** One pair has sequence log-probabilities $\log\pi_\theta(y_w) = -12$, $\log\pi_\theta(y_l) = -15$, $\log\pi_{\text{ref}}(y_w) = -13$, and $\log\pi_{\text{ref}}(y_l) = -14$. The policy has raised the winner by 1 nat and lowered the loser by 1 nat.

| $\beta$ | Implicit rewards $(\hat r_w, \hat r_l)$ | Margin | $p(y_w \succ y_l)$ | Loss |
|---|---|---|---|---|
| 0.1 | $(0.1, -0.1)$ | 0.2 | 0.5498 | 0.5981 |
| 0.5 | $(0.5, -0.5)$ | 1.0 | 0.7311 | 0.3133 |

Three more checks:

- When $\pi_\theta = \pi_{\text{ref}}$, the margin is 0 and the loss is $\log 2 = 0.6931$, whatever the log-probabilities are.
- If the policy had moved the wrong way ($\log\pi_\theta(y_w) = -14.5$ and $\log\pi_\theta(y_l) = -13.5$), the margin is $-0.2$ and the loss is $0.7981$.
- With a margin of $-1000$, the naive $-\log\sigma(m)$ evaluates to `inf` in float64 because $\sigma$ underflows to 0. The stable form $\log(1 + e^{-m})$, written as `np.logaddexp(0, -m)`, returns 1000.

Exercise 2 asks you to write the loss and survive that last case.

### The gradient weights each pair by how wrong the implicit reward is

**Intuition.** Pushing up the winner and pushing down the loser is the obvious update. DPO adds a per-example weight: pairs the current policy already ranks correctly get a small weight, and pairs it ranks backward get a large one. The paper credits that weight with preventing degeneration.

**Math.** Section 4 gives the gradient

$$
\nabla_\theta \mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\beta\,\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\Big[\sigma\big(\hat r_\theta(x, y_l) - \hat r_\theta(x, y_w)\big)\,\big[\nabla_\theta \log\pi(y_w \mid x) - \nabla_\theta \log\pi(y_l \mid x)\big]\Big]
$$

The weight $\sigma(\hat r_\theta(x, y_l) - \hat r_\theta(x, y_w))$ is "higher when reward estimate is wrong" (the label in the paper's equation). Appendix A.4 derives it by writing $u$ for the argument of $\sigma$ and using $\sigma'(u) = \sigma(u)(1 - \sigma(u))$ and $\sigma(-u) = 1 - \sigma(u)$ (Equations 21 and 22). The paper reports that a "naive version of this method without the weighting coefficient can cause the language model to degenerate", and points to samples from its unlikelihood baseline in Appendix Table 3 (Section 4).

For a softmax policy over $K$ responses with logits $\theta$, $\nabla_\theta\log\pi_\theta(y) = e_y - \pi_\theta$, where $e_y$ is the one-hot vector. The $\pi_\theta$ terms cancel in the difference, and the gradient with respect to the logits for one pair is $-\beta\,\sigma(\hat r_l - \hat r_w)(e_w - e_l)$. This is the tabular case that the exercises use. The paper doesn't spell out this special case.

**Tiny example, computed by `scripts/gradient_example.py`.** The policy is a softmax over three responses, $\pi_{\text{ref}} = [0.70, 0.25, 0.05]$, and the pair prefers response 2 over response 0, with $\beta = 0.5$.

- At $\pi_\theta = \pi_{\text{ref}}$ the weight is $\sigma(0) = 0.5$ and the gradient is $[0.25, 0, -0.25]$. A central finite difference of the loss agrees to nine decimal places.
- If the policy has moved toward the loser (logits shifted by $[+2, 0, -2]$), then $\hat r_w - \hat r_l = -2$ and the weight is $\sigma(2) = 0.8808$. If it has moved toward the winner (shift $[-2, 0, +2]$), the margin is $+2$ and the weight is $\sigma(-2) = 0.1192$. A backward pair pulls 7.4 times as hard.
- Ten gradient-descent steps with learning rate 1 shrink the weight from 0.5000 to 0.3033 at step 5 and 0.1889 at step 10, while $\pi(y_2)$ rises from 0.05 to 0.3719. The loss falls from 0.6931 to 0.1922.

Exercise 3 asks you to implement this gradient from the formula and check it against finite differences.

### Nothing is lost by the reparameterization

**Intuition.** One worry is that restricting rewards to the form $\beta\log(\pi/\pi_{\text{ref}})$ limits which preferences the model can fit. Section 5.1 argues it doesn't, because a reward only matters up to a per-prompt constant, and the log-ratio form picks one representative from each class.

**Math.** Definition 1 calls two rewards equivalent when $r(x, y) - r'(x, y) = f(x)$ for some function $f$. Two lemmas follow (Appendix A.5 gives the proofs):

- **Lemma 1.** Rewards in the same class induce the same Plackett-Luce, and in particular Bradley-Terry, preference distribution.
- **Lemma 2.** Rewards in the same class induce the same optimal policy under the constrained RL problem.

**Theorem 1** states that, under mild assumptions, every reward class consistent with Plackett-Luce can be represented as $r(x, y) = \beta\log\frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$ for some $\pi$. The proof sketch projects each reward onto the class member $f(r; \pi_{\text{ref}}, \beta)(x, y) = r(x, y) - \beta\log\sum_y\pi_{\text{ref}}(y \mid x)\exp(\frac{1}{\beta}r(x, y))$ (Equation 8). Equivalently, the DPO reparameterization picks the member that satisfies $\sum_y\pi_{\text{ref}}(y \mid x)\exp(\frac{1}{\beta}r(x, y)) = 1$, so that $\pi$ is a valid distribution (Equation 9).

**Tiny example, computed by `scripts/closed_form_example.py`.** Add the constant 5 to every reward in the earlier example. The optimal policy is unchanged at $[0.4002, 0.3885, 0.2112]$, and the preference between responses 2 and 0 is still 0.8808. Only $Z$ changes, from 1.7490 to 259.5781, a factor of $e^{5/\beta} = 148.4132$. The DPO loss can't see $Z$ at all, which is why it works without estimating it.

### Training on preferences recovers the optimum

**Intuition.** If the loss really is the Bradley-Terry maximum likelihood for the reparameterized reward, then with enough data its minimizer should be the policy that Equation 4 predicts for the true reward. A tiny tabular policy lets you check that directly.

**Math.** With the loss in Equation 7 and a Bradley-Terry data distribution, the population minimizer has $\hat r_\theta(x, y_1) - \hat r_\theta(x, y_2) = r^*(x, y_1) - r^*(x, y_2)$ for every pair, because the logistic likelihood is maximized when the model's preference probabilities equal the data's. A policy must also sum to 1, which fixes the one free constant per prompt, and the result is $\pi_\theta = \pi^*$ of Equation 4 for $r^*$. This argument is this page's, specialized to the tabular case. The paper's Section 5.1 gives the general statement through Theorem 1.

**Tiny example, computed by `scripts/toy_training.py`.** Let $r^* = [0, 1, 2]$ and $\pi_{\text{ref}} = [0.70, 0.25, 0.05]$. Build exact Bradley-Terry counts for 1000 comparisons of each ordered pair, so the entry for response 2 beating response 0 is $1000\,\sigma(2) = 880.80$. Train the logits by gradient descent from $\theta = \log\pi_{\text{ref}}$, with learning rate $8/\beta^2$ and 3000 steps.

| $\beta$ | Trained $\pi_\theta$ | Closed-form $\pi^*$ (Equation 4) | Implicit reward gaps |
|---|---|---|---|
| 1 | $[0.4002, 0.3885, 0.2112]$ | $[0.4002, 0.3885, 0.2112]$ | $1.0000,\ 1.0000$ |
| 0.5 | $[0.1326, 0.3500, 0.5173]$ | $[0.1326, 0.3500, 0.5173]$ | $1.0000,\ 1.0000$ |

The largest difference between trained and closed-form probabilities is below $10^{-15}$. Both runs end at the same loss, 0.5099, and recover the reward gaps of 1 because the data fix the gaps. What $\beta$ changes is how far the policy has to move to express them: the log-ratio gaps must equal the reward gaps divided by $\beta$, so they are twice as large at $\beta = 0.5$ as at $\beta = 1$.

Exercise 4 asks you to write this training loop.

### What the experiments show

**Intuition.** The claim to test is that a classification loss reaches the same place as RL, with less machinery.

**Reward against KL (Section 6.1).** On controlled IMDb sentiment generation, with GPT-2-large and a sentiment classifier as the ground-truth reward, the paper sweeps hyperparameters over 22 runs (target KL in $\{3, 6, 9, 12\}$ for PPO and $\beta \in \{0.05, 0.1, 1, 5\}$ for DPO, along with baselines). DPO has the best frontier of expected reward against KL to the reference, including against PPO given the ground-truth reward (Figure 2, left).

**Summarization (Section 6.2).** On Reddit TL;DR, the GPT-4 win rate against human-written summaries is about 61% for DPO at temperature 0.0 and 57% for PPO at its best temperature, also 0.0 (Figure 2, right). The paper reports that DPO is much more robust to the sampling temperature, while PPO can fall to the level of the base GPT-J model at high temperatures. The paper notes that it didn't meaningfully tune $\beta$.

**Dialogue (Section 6.2).** On the one-turn subset of Anthropic-HH, with Pythia-2.8B and a reference trained on the chosen completions, DPO is the only computationally efficient method that improves over the preferred completions. It matches or beats the Best of 128 baseline, which is expensive at inference time (Figure 3).

**Out-of-distribution (Section 6.3, Table 1).** Moving the TL;DR policies to CNN/DailyMail articles, DPO's win rates are 0.36 at temperature 0 and 0.31 at 0.25, against 0.26 and 0.23 for PPO.

**Judge reliability (Section 6.4, Table 2).** In a human study, GPT-4 agrees with humans about as often as humans agree with each other, for example in the DPO column GPT-4 (C) agrees with humans 67 percent of the time and two humans agree 65 percent of the time. Humans preferred DPO samples at temperature 0.25 over PPO samples at temperature 0 in 58 percent of judgments.

The paper lists open questions in Section 7: how DPO policies generalize out of distribution, how reward over-optimization shows up without an explicit reward, whether the slight drop in Figure 3 (right) is an instance of it, scaling beyond 6B parameters, and the sensitivity of GPT-4 win rates to the prompt. These are results from models up to 6B parameters.

<!-- exercises -->

## Pitfalls

- **Using token-level log-probabilities as the sequence log-probability.** Equation 7 uses $\pi_\theta(y \mid x)$ for the whole response, which is the sum of the response-token log-probabilities. Count only the completion tokens, not the prompt or the padding. A mean over tokens instead of a sum is a different loss with a different effective $\beta$. The paper doesn't spell out the token-level sum; Appendix B's code takes the sequence-level values as input.
- **Letting the reference move.** The reference log-probabilities are constants. If you compute them with the model that you're training, or edit them in place, the log ratios are 0 and the loss is stuck at $\log 2$. Exercise 4's tests check that `ref_logits` isn't modified.
- **Starting from a different model than the reference.** The loss at the first step should be $\log 2 = 0.6931$ when $\pi_\theta = \pi_{\text{ref}}$. A different first-step loss means the policy and the reference disagree, which usually means a different checkpoint, tokenizer, or prompt template.
- **Swapping the winner and the loser.** The margin is $(\hat r_w - \hat r_l)$ with $\hat r = \beta\log(\pi_\theta/\pi_{\text{ref}})$. Reversing it trains the model to prefer the rejected response, and the loss still looks plausible. The sign of the gradient weight is easy to get wrong too: it is $\sigma(\hat r_l - \hat r_w)$.
- **Trusting the appendix version of the gradient.** In the arXiv v3 PDF, Appendix A.4 defines the loss in Equation 21 with the winner and loser terms of the sigmoid argument reversed relative to Equation 7, and the final line of that derivation has $\sigma(\hat r_w - \hat r_l)$ as the weight, which is the opposite of the Section 4 weight. Section 4's form is the gradient of Equation 7. Exercise 3's finite-difference test checks this directly.
- **Computing $-\log\sigma$ naively.** `np.log(1 / (1 + np.exp(-m)))` returns `inf` for large negative margins, which long sequences with a large $\beta$ can produce. Use `logaddexp` or `F.logsigmoid`, as Appendix B does.
- **Reading $\beta$ backward.** A larger $\beta$ keeps the policy closer to $\pi_{\text{ref}}$, because it is the same coefficient as in the KL term of Equation 3. In Equation 4 the reward is divided by $\beta$. The paper's default is 0.1, and 0.5 for TL;DR summarization (Appendix B).
- **Skipping the SFT step.** The derivation assumes preference pairs that were sampled from $\pi_{\text{ref}}$. When the pairs come from another model, the paper's advice is to fit $\pi_{\text{ref}}$ by maximum likelihood on the preferred completions first, to reduce the distribution shift (Section 4).
- **Calling DPO on-policy.** DPO trains on a fixed dataset. It never samples from $\pi_\theta$ during training, which is why it is cheap, and the paper says that how it handles the reward over-optimization seen in RL is an open question (Section 7).
- **Treating a GPT-4 win rate as a human win rate.** The paper finds that the prompt changes the win rates, and it uses the concise GPT-4 (C) prompt for its main results because that prompt tracks human judgments better (Section 6.4).

## Further reading

- [direct-preference-optimization](https://github.com/eric-mitchell/direct-preference-optimization) is the authors' reference implementation of DPO, with the loss, the data pipeline, and the training script.
- The [TRL DPO trainer documentation](https://huggingface.co/docs/trl/dpo_trainer) describes a widely used implementation, with the loss variants that came after the paper.
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155), by Ouyang et al., is the RLHF pipeline that DPO simplifies. See the [InstructGPT page](/instructgpt/) on this site.
- [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325), by Stiennon et al., collected the TL;DR preference data that Section 6 uses.
