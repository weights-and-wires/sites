---
id: "ppo"
title: "Proximal Policy Optimization Algorithms"
authors: "Schulman et al."
year: 2017
venue: "arXiv preprint"
arxiv: "1707.06347"
url: "https://arxiv.org/abs/1707.06347"
track: rl
tier: core
est_hours: 6
status: full
summary: "PPO makes a policy gradient update safe to repeat on the same batch of data by clipping the probability ratio between the new and old policy, so a sample stops pushing the policy once the ratio leaves a small interval around 1. This clipped surrogate objective gives a pessimistic lower bound on the unclipped objective and needs only first-order optimization, where TRPO needs a constrained second-order step. The full loss also adds a value-function error and an entropy bonus, and each iteration alternates between collecting a batch and running several epochs of minibatch gradient ascent on it."
related_course: "rl: policy gradients"
prereqs:
  - id: trpo
    reason: "PPO keeps the TRPO surrogate objective and the goal of limiting how far each update moves the policy, then enforces that goal with clipping in place of a KL constraint."
  - id: gae
    reason: "The advantage estimates in PPO's objective come from generalized advantage estimation truncated to a fixed-length segment (Equations 11 and 12)."
background: [importance-sampling, clipping]
course_links:
  - title: "Proximal policy optimization"
    url: "https://rl.ifkash.dev/lessons/034-proximal-policy-optimization"
  - title: "PPO in practice"
    url: "https://rl.ifkash.dev/lessons/035-ppo-in-practice"
self_check:
  - q: 'TRPO limits each update with a KL constraint, solved by conjugate gradient. What does PPO use in place of the constraint, and does the replacement guarantee that the new policy stays close to the old one?'
    a: 'PPO clips the ratio $r_t(\theta)$ inside the objective (Equation 7), so a sample has no gradient once the ratio has moved past $1 \pm \epsilon$ in the direction the advantage favors. That removes the incentive to move further, but it is not a constraint. The sum of many samples, and the shared parameters of a network, can still push the ratio past the interval, so the policy can drift beyond $\epsilon$. In the bandit example on this page, the ratio reaches 1.29 with $\epsilon = 0.2$.'
    see: trpo
  - q: 'The advantage estimator in Equation 11 sums $\delta_t + (\gamma\lambda)\delta_{t+1} + \dots$ over a segment of $T$ steps. What does it use for the value after the last step, and what does it become when $\lambda = 1$?'
    a: 'It stops at the end of the segment, and the final residual $\delta_{T-1} = r_{T-1} + \gamma V(s_T) - V(s_{T-1})$ bootstraps from the learned value $V(s_T)$. With $\lambda = 1$ the estimate is the discounted rewards in the segment plus $\gamma^{T-t} V(s_T)$, minus $V(s_t)$, which is Equation 10 from the A3C paper.'
    see: gae
  - q: 'With $\epsilon = 0.2$, what is the contribution of one sample to $L^{CLIP}$ when $r = 1.5$ and $\hat A = -1$? What about $r = 1.5$ and $\hat A = +1$? Why are they not symmetric?'
    a: 'For $\hat A = -1$, the terms are $r\hat A = -1.5$ and $\operatorname{clip}(r)\hat A = -1.2$, and the minimum is $-1.5$: the full penalty, with a gradient. For $\hat A = +1$ the terms are $1.5$ and $1.2$, and the minimum is $1.2$: the gain is capped and the gradient is 0. The minimum ignores a change in the ratio only when it would make the objective better, and keeps it when it would make the objective worse (Section 3).'
    see: clipping
---

## The problem

A vanilla policy gradient method collects a batch of trajectories with the current policy, takes one gradient step on $L^{PG}(\theta) = \hat{\mathbb{E}}_t[\log \pi_\theta(a_t \mid s_t)\hat A_t]$ (Section 2.1, Equation 2), and throws the batch away. That is wasteful, because simulation is the expensive part. The obvious fix is to take several steps on the same batch. Section 2.1 says this isn't well justified, and that it "often leads to destructively large policy updates."

TRPO ([Trust region policy optimization](/trpo/)) fixes the step size with a KL constraint, but Section 1 lists its costs: it is relatively complicated, and it isn't compatible with architectures that include noise such as dropout, or with parameter sharing between the policy and the value function. The paper asks for the reliability of TRPO using only first-order optimization.

Here is the failure in the smallest setting. The policy is a softmax over 3 arms with mean rewards 0, 0.5, and 1, with logits all 0. One batch has 64 pulls with noisy rewards, and the advantage is the reward minus the batch mean. Repeating the gradient step on that batch, with the surrogate $L^{CPI}$ that PPO starts from (Equation 6), gives the numbers below. They come from `scripts/bandit_ppo.py`, with learning rate 0.5.

| Epochs on the batch | Probability of the best arm | Ratio $\pi_\theta/\pi_{\theta_{old}}$ for the best arm | $\mathrm{KL}[\pi_{old}, \pi_\theta]$ |
|---|---|---|---|
| 1 | 0.3644 | 1.0931 | 0.0028 |
| 10 | 0.6654 | 1.9962 | 0.2525 |
| 50 | 0.9514 | 2.8541 | 1.4323 |
| 200 | 0.9901 | 2.9702 | 2.4797 |

Sixty-four noisy samples have turned a uniform policy into one that picks the best arm 95% of the time. Nothing in $L^{CPI}$ tells the optimizer to stop, because each sample's term $r_t \hat A_t$ keeps growing as the ratio moves. The estimate is only good near $r = 1$. After 50 epochs the new policy sits far from the one that produced the data. PPO adds a term that makes the objective flat once a sample has moved the ratio far enough.

## Key ideas

The sections that follow use the paper's notation: the probability ratio $r_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_{old}}(a_t \mid s_t)$, the advantage estimate $\hat A_t$, the clipping range $\epsilon$, the KL coefficient $\beta$, and the value and entropy coefficients $c_1$ and $c_2$. The paper writes the number of optimization epochs as $K$, and so does this page. In the exercises $K$ doesn't appear as a number of epochs; the bandit has 3 arms, and the code names the epoch count `epochs`. The paper's Equation 9 is a quantity to maximize, so the page does too, and the exercises give the negated loss where code needs a loss to minimize. Every number comes from a script in `scripts/`. Run any script from the paper folder to reproduce the output.

### The probability ratio and the surrogate $L^{CPI}$

**Intuition.** The batch comes from $\pi_{\theta_{old}}$, and you want to score a different policy $\pi_\theta$ on it. Importance sampling does this: weight each sample's advantage by how much more (or less) likely the new policy makes that action. At $\theta = \theta_{old}$ every weight is 1.

**Math.** Section 3 defines $r_t(\theta)$, so that $r_t(\theta_{old}) = 1$, and rewrites TRPO's surrogate (Section 2.2, Equation 3) as (Section 3, Equation 6):

$$
L^{CPI}(\theta) = \hat{\mathbb{E}}_t\!\left[\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}\hat A_t\right] = \hat{\mathbb{E}}_t\big[r_t(\theta)\hat A_t\big]
$$

The superscript refers to conservative policy iteration, where the objective was proposed. Because $\nabla_\theta r_t = r_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)$ and $r_t = 1$ at $\theta_{old}$, the gradient of $L^{CPI}$ at $\theta_{old}$ equals the policy gradient estimator $\hat g$ of Equation 1. The two objectives agree to first order and differ once $\theta$ moves.

**Tiny example, computed by `scripts/bandit_ppo.py`.** On the bandit from the problem section, one epoch is exactly one vanilla policy gradient step, because the ratio is 1. Both the clipped and the unclipped runs give the best-arm probability 0.3644 after 1 epoch, and they only separate once a second epoch has moved the ratios. Exercise 4 tests this equality.

### The clipped surrogate objective

**Intuition.** Make the term for each sample flat in the direction where the policy has already moved "enough" to improve it. If the advantage is positive, raising the ratio helps, so stop rewarding it past $1 + \epsilon$. If the advantage is negative, lowering the ratio helps, so stop rewarding it below $1 - \epsilon$. Don't remove the penalty in the other direction. A sample that made the policy worse must keep its full weight.

**Math.** Section 3, Equation 7:

$$
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\Big[\min\big(r_t(\theta)\hat A_t,\ \operatorname{clip}(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon)\hat A_t\big)\Big]
$$

The paper suggests $\epsilon = 0.2$. The minimum makes $L^{CLIP}$ a lower bound on $L^{CPI}$ (Figure 2 shows this along the update direction), and $L^{CLIP} = L^{CPI}$ to first order around $\theta_{old}$. Figure 1 plots one term as a function of $r$. For $\hat A > 0$ the term rises with slope $\hat A$ until $r = 1 + \epsilon$ and is flat after. For $\hat A < 0$ it falls with slope $\hat A$ until $r = 1 - \epsilon$ and is flat below. The terms are not flat the other way round: for $\hat A > 0$ and $r < 1 - \epsilon$, the minimum picks $r\hat A$, which has a nonzero gradient. A gradient with respect to $\theta$ comes through $r_t$, so it is $\hat A_t \nabla_\theta r_t$ where the min selects the unclipped term, and 0 where it selects the clipped one.

**Tiny example, computed by `scripts/clip_example.py`.** With $\epsilon = 0.2$:

| $\hat A$ | $r$ | $r\hat A$ | Term of $L^{CLIP}$ | $\partial L / \partial r$ |
|---|---|---|---|---|
| +1 | 1.1 | 1.10 | 1.10 | +1 |
| +1 | 1.5 | 1.50 | 1.20 | 0 |
| +1 | 0.5 | 0.50 | 0.50 | +1 |
| -1 | 0.9 | -0.90 | -0.90 | -1 |
| -1 | 0.5 | -0.50 | -0.80 | 0 |
| -1 | 1.5 | -1.50 | -1.50 | -1 |

Rows 3 and 6 are the pessimistic cases. The ratio moved the wrong way for the advantage, so the full term stays, and so does its gradient. Now take a batch of four samples with $r = [1.3, 0.7, 1.1, 0.5]$ and $\hat A = [2, 1, -1, -2]$. The unclipped terms are $[2.6, 0.7, -1.1, -1.0]$, with mean 0.3. The clipped terms are $[2.4, 0.7, -1.1, -1.6]$, with mean 0.1. The first and last samples have no gradient, and the middle two keep it. The script prints the gradient with respect to $\log\pi_\theta(a_t)$ as $[0, 0.175, -0.275, 0]$, which is $r_t \hat A_t / 4$ for the active samples.

Exercise 1 asks you to implement $L^{CLIP}$ and this gradient, and to check it against finite differences.

### Many epochs on one batch

**Intuition.** With the clipped objective, the same batch can feed several epochs of minibatch gradient ascent. A sample drops out of the gradient once its ratio has crossed the edge, so the update loses its drive as the policy moves, and the batch stops being useful.

**Math.** Algorithm 1 (Section 5) alternates between collecting data and optimizing. Each iteration, each of $N$ actors runs $\pi_{\theta_{old}}$ for $T$ timesteps and computes advantage estimates. Then the algorithm optimizes the surrogate $L$ on the $NT$ samples with $K$ epochs and minibatch size $M \le NT$, and sets $\theta_{old} \leftarrow \theta$. The old policy stays fixed during the $K$ epochs. That is what gives the ratio a meaning. Appendix A, Table 3, lists the MuJoCo settings: $T = 2048$, Adam stepsize $3 \times 10^{-4}$, 10 epochs, minibatch size 64, $\gamma = 0.99$, and $\lambda = 0.95$.

**Tiny example, computed by `scripts/bandit_ppo.py`.** The same bandit, batch, and learning rate as in the problem section, with $\epsilon = 0.2$:

| Epochs on the batch | Probability of the best arm | Ratio for the best arm | $\mathrm{KL}[\pi_{old}, \pi_\theta]$ |
|---|---|---|---|
| 1 | 0.3644 | 1.0931 | 0.0028 |
| 10 | 0.4296 | 1.2889 | 0.0238 |
| 50 | 0.4206 | 1.2618 | 0.0182 |
| 200 | 0.4279 | 1.2838 | 0.0226 |

The unclipped run reaches a KL of 1.43 after 50 epochs, and the clipped run stays near 0.02 even after 200. The clipped run is not a hard limit. The ratio settles at about 1.28, above $1 + \epsilon = 1.2$, because the arms share one softmax, so a step that lowers one arm's probability also raises another's. The sample-level clip doesn't see that. The clipped run also wobbles between epochs 10 and 200 instead of converging to one point, for the same reason.

Exercise 4 asks you to implement this loop, including the part that is easy to get wrong: the ratio compares against the policy from the start of the call, not the previous epoch.

### The adaptive KL penalty

**Intuition.** The paper also tries the other classic remedy from TRPO's theory, a penalty on the KL divergence, and makes the penalty coefficient track a target KL. It is a baseline in the paper, not the main method.

**Math.** Section 4, Equation 8:

$$
L^{KLPEN}(\theta) = \hat{\mathbb{E}}_t\Big[r_t(\theta)\hat A_t - \beta\,\mathrm{KL}\big[\pi_{\theta_{old}}(\cdot \mid s_t),\ \pi_\theta(\cdot \mid s_t)\big]\Big]
$$

After the epochs, compute $d = \hat{\mathbb{E}}_t[\mathrm{KL}[\pi_{\theta_{old}}(\cdot \mid s_t), \pi_\theta(\cdot \mid s_t)]]$. If $d < d_{targ}/1.5$, set $\beta \leftarrow \beta/2$. If $d > d_{targ} \times 1.5$, set $\beta \leftarrow \beta \times 2$. The paper says the numbers 1.5 and 2 are heuristics and the algorithm isn't sensitive to them. It also says the clipped version performed better. Section 2.2 explains why a fixed $\beta$ fails: one value doesn't work across problems, or within a single problem as learning proceeds.

**Tiny example, computed by `scripts/adaptive_kl_example.py`.** For a single state with $\pi_{old} = [0.5, 0.5]$ and $\pi_\theta = [0.6, 0.4]$, $\mathrm{KL}[\pi_{old}, \pi_\theta] = 0.02041$. The reverse direction is 0.02014, close but not equal. With $d_{targ} = 0.01$, the band is $[0.00667, 0.015]$ and $\beta$ starts at 1. Measured values of $d = 0.004, 0.012, 0.030, 0.0095, 0.0066$ give $\beta = 0.5, 0.5, 1, 1, 0.5$. The first measurement is below the band, so $\beta$ halves. The third is above it, so $\beta$ doubles.

Exercise 2 asks you to implement the KL, the penalized objective, and the update rule.

### One loss for policy, value, and entropy

**Intuition.** With a network that shares parameters between the policy and the value function, you can't optimize two separate losses with separate optimizers. You add the terms into one number, with coefficients that set the exchange rate between them.

**Math.** Section 5, Equation 9, to be maximized:

$$
L_t^{CLIP+VF+S}(\theta) = \hat{\mathbb{E}}_t\Big[L_t^{CLIP}(\theta) - c_1 L_t^{VF}(\theta) + c_2\, S[\pi_\theta](s_t)\Big]
$$

Here $L_t^{VF} = (V_\theta(s_t) - V_t^{targ})^2$ and $S$ is an entropy bonus that encourages exploration. The advantage $\hat A_t$ inside $L^{CLIP}$ comes from a truncated generalized advantage estimator over a segment of length $T$ (Equations 11 and 12), $\hat A_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \dots$ with $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ (see [GAE](/gae/) for the derivation). The paper's version of the last term in Equation 11 has the exponent $T - t + 1$ on $\delta_{T-1}$. Counting steps from $t$ gives $T - t - 1$, which is what the code in the exercises and the GAE page use. The paper's MuJoCo experiments don't share parameters, so $c_1$ is irrelevant there, and they use no entropy bonus (Section 6.1). The Atari settings in Appendix A, Table 5, use $c_1 = 1$ and $c_2 = 0.01$.

**Tiny example, computed by `scripts/combined_objective_example.py`.** Take a batch of 3 states with 3 actions, $\epsilon = 0.2$, and $c_1 = 0.5$ and $c_2 = 0.01$ (values chosen for the example, not taken from the paper). The ratios of the chosen actions are $[1.3, 1.25, 1.0]$, with advantages $[1, -0.5, 2]$. The terms are:

| Term | Per-sample values | Mean |
|---|---|---|
| $L^{CLIP}_t$ | $[1.2, -0.625, 2.0]$ | 0.8583 |
| $L^{VF}_t$ | $[0.25, 0.25, 0.25]$ | 0.25 |
| $S_t$ (nats) | $[0.8865, 1.0297, 0.639]$ | 0.8517 |

The objective is $0.8583 - 0.5 \cdot 0.25 + 0.01 \cdot 0.8517 = 0.7419$, and the loss to minimize is $-0.7419$. Sample 2 has a negative advantage and a ratio of 1.25, above $1 + \epsilon$. Its term is $-0.625$ and not $-0.6$, because the minimum keeps the worse value.

Exercise 3 asks you to implement the entropy and the full objective.

### What the paper shows

Section 6.1 compares surrogates on 7 MuJoCo tasks, 1 million timesteps each, with 3 seeds per task. The score is normalized so a random policy gets 0 and the best result gets 1 (Table 1). No clipping or penalty scores $-0.39$, which is worse than a random policy because it collapses on HalfCheetah. Clipping with $\epsilon = 0.1$, $0.2$, and $0.3$ scores 0.76, 0.82, and 0.70. The adaptive KL penalty scores at most 0.74 (with $d_{targ} = 0.01$), and the fixed KL penalty at most 0.72 (with $\beta = 3$). Section 6.2 compares PPO with TRPO, CEM, vanilla policy gradient with an adaptive stepsize, A2C, and A2C with a trust region on the same tasks, and reports that PPO outperforms the previous methods on almost all the continuous control environments (Figure 3). Section 6.3 trains 3D humanoids with Roboschool (Figure 4). Section 6.4 and Table 2 compare PPO with A2C and ACER on 49 Atari games. Counting games won by average reward over all of training, PPO wins 30, ACER 18, and A2C 1. Counting by the last 100 episodes, PPO wins 19, ACER 28, A2C 1, with 1 tie. The scores favor PPO for fast learning and ACER for final performance.

<!-- exercises -->

## Pitfalls

- **Recomputing the old log-probabilities.** The ratio needs $\log\pi_{\theta_{old}}$ from the policy that collected the data, fixed for all $K$ epochs. If you recompute it at each epoch, $r = 1$ every time and clipping never triggers. Exercise 4's mutant checks for this bug.
- **Treating the clip as a constraint.** The clip removes a sample's incentive. It doesn't bound the KL, and the page's bandit run ends at a ratio of 1.29 with $\epsilon = 0.2$. Monitor the measured KL or the fraction of clipped samples, and don't assume the policy stayed inside $[1-\epsilon, 1+\epsilon]$.
- **Clipping the wrong side.** For $\hat A > 0$, the objective is flat above $1 + \epsilon$ and not below $1 - \epsilon$. If you use only the clipped term and skip the `min`, a sample with a good advantage and a ratio of 0.5 loses its gradient, and the update can't repair a mistake. Exercise 1 has a test for each side.
- **Signs.** Equation 9 is maximized. In a framework that minimizes, negate the whole thing: subtract the surrogate and the entropy, and add $c_1 L^{VF}$.
- **Entropy and value terms with the wrong scale.** $c_1$ and $c_2$ only mean something relative to the scale of the advantages and rewards. The paper's values ($c_1 = 1$ and $c_2 = 0.01$ for Atari) don't transfer to tasks with different reward scales.
- **Details that are not in the paper.** Implementations commonly normalize advantages within each minibatch, clip the global gradient norm, and shuffle minibatches each epoch. The paper doesn't describe these, and the exercises omit them. The paper does anneal $\epsilon$ and the learning rate linearly in the Atari runs ($\epsilon = 0.1 \times \alpha$, Table 5).
- **Equation 11's exponent.** The last term as printed reads $(\gamma\lambda)^{T-t+1}\delta_{T-1}$. Counting steps from $t$ gives $T - t - 1$.

## Further reading

- The official code: the [OpenAI Baselines](https://github.com/openai/baselines) repository, which has the `ppo2` implementation that most later code follows.
- The [Spinning Up PPO page](https://spinningup.openai.com/en/latest/algorithms/ppo.html), a short explanation with pseudocode.
- Huang et al., [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/), which lists the details this page and the paper leave out.
