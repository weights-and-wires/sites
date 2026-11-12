---
id: "sac"
title: "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
authors: "Haarnoja et al."
year: 2018
venue: "ICML 2018"
arxiv: "1801.01290"
url: "https://arxiv.org/abs/1801.01290"
track: rl
tier: extended
est_hours: 5
status: full
summary: "This paper trains a stochastic actor off-policy to maximize reward plus the entropy of its own action distribution, so the agent succeeds at the task while acting as randomly as it can. A critic pair, a state-value network, and a policy trained through reparameterized samples make the method stable and sample-efficient on continuous control. It beats DDPG, PPO, and soft Q-learning on the hardest benchmarks and gives similar results across random seeds."
related_course: "rl: continuous control"
prereqs:
  - id: ddpg
    reason: "SAC keeps DDPG's replay buffer, slow target network, and actor that is trained by differentiating the critic through the action, and then replaces the deterministic actor with a stochastic one."
  - id: td3
    reason: "SAC takes the minimum of two Q-functions, the clipped double-Q idea from TD3, to curb overestimation, although it applies the minimum in its value and policy updates rather than in the critic target."
background: [entropy-regularization, reparameterization-trick]
course_links:
  - title: "Maximum-entropy reinforcement learning"
    url: "https://rl.ifkash.dev/lessons/041-maximum-entropy-rl"
  - title: "Soft actor-critic"
    url: "https://rl.ifkash.dev/lessons/042-soft-actor-critic"
self_check:
  - q: 'DDPG''s actor gradient multiplies $\nabla_a Q$ by $\nabla_\theta \mu$. Take SAC''s policy $a = \mu + \sigma\epsilon$ with $\mu = 0.5$, $\sigma = 1$, one noise draw $\epsilon = 1$, and the critic $Q(a) = -(a - 2)^2$. What does Equation 13 give for the gradient of $J_\pi$ with respect to $\mu$, and which way does a descent step move $\mu$?'
    a: 'The sample action is $a = 1.5$ and $\nabla_a Q = -2(1.5 - 2) = 1$. For the mean, the two $\log \pi$ terms in Equation 13 cancel, because $\partial_\mu \log \pi = \epsilon/\sigma$ and $\nabla_a \log \pi \cdot \partial_\mu f = -\epsilon/\sigma$. That leaves $\nabla_\mu J_\pi = -\nabla_a Q = -1$. Descent moves $\mu$ up, toward the critic''s peak at $a = 2$. This is DDPG''s signal on the mean. The entropy term only acts through $\sigma$.'
    see: ddpg
  - q: 'A state has two actions with $Q(s, a_1) = Q(s, a_2) = 0$ and the temperature is $\alpha = 1$. What are the best policy and its soft value, and why does the value exceed the ordinary value of 0?'
    a: 'The best policy is uniform, because the Boltzmann policy $\pi \propto \exp(Q/\alpha)$ gives equal mass to equal values. The soft value is $\log(e^0 + e^0) = \log 2 = 0.6931$. The ordinary value is 0. The extra $0.6931$ is the entropy bonus: a uniform policy over two actions has entropy $\log 2$.'
    see: entropy-regularization
  - q: 'Why does Section 4.2 choose the reparameterization trick over the likelihood-ratio gradient for the policy, and what does that choice require of the critic?'
    a: 'The target density in Equation 10 is the Q-function, which is a neural network that you can differentiate. Writing the action as $a = f_\phi(\epsilon; s)$ lets the gradient flow through $Q(s, a)$ and gives a lower-variance estimator. On a one-dimensional toy, the page measures a variance of 0.251 for the reparameterized estimator and 3.240 for the score-function estimator, a ratio of about 13. The price is that $Q$ must be differentiable in the action, and the sample must be a differentiable function of the policy parameters.'
    see: reparameterization-trick
---

## The problem

Deep RL for continuous control had two problems in 2018, and the paper opens with both (Section 1). The methods needed millions of environment steps, and they were brittle: learning rates, exploration constants, and other settings had to be tuned for each task.

The two main families each failed one way. On-policy methods such as TRPO, PPO, and A3C are stable, but they need fresh samples for every gradient step. Off-policy methods reuse the replay buffer, but the paper calls the leading one, DDPG, extremely brittle and sensitive to hyperparameters (Section 1). A deterministic actor and a Q-function that is trained on the actor's own actions interact badly.

A third family, soft Q-learning, already used the maximum entropy idea. Section 2 says it isn't a true actor-critic: its Q-function estimates the optimal Q-function, and the actor is only an approximate sampler. Its convergence depends on how well that sampler matches the target, and in continuous actions it needs approximate inference.

Figure 1 shows how this looks in practice. DDPG makes no progress on Ant-v1, Humanoid-v1, and the rllab Humanoid, which has a 21-dimensional action (Section 5.1). Figure 2 isolates the cause. A deterministic variant of SAC, close to DDPG with two Q-functions, shows very high variability between seeds on the rllab Humanoid, and SAC's stochastic policy does not (Section 5.2).

The paper's answer is an off-policy actor-critic whose objective includes the policy's entropy. Section 4 derives it from a tabular algorithm with a convergence proof, soft policy iteration, and then approximates that algorithm with neural networks. The result matches the other methods on the easy tasks and beats them by a wide margin on the hard ones, with similar learning curves across five seeds (abstract and Section 5.2).

## Key ideas

This page uses the paper's notation: the state-value network $V_\psi$ and its target $V_{\bar\psi}$, the two soft Q-functions $Q_{\theta_1}$ and $Q_{\theta_2}$, the policy $\pi_\phi$, the replay buffer $\mathcal{D}$, the discount $\gamma$, the target smoothing coefficient $\tau$, and the noise $\epsilon$. Two things differ from the paper. This page keeps the temperature $\alpha$ visible in Equation 1 and in the examples. The paper sets $\alpha = 1$ after Equation 1 and folds the temperature into the reward scale (Section 3.2, Section 5.2). This page also writes $d_i \in \{0, 1\}$ for the terminal flag, which the paper's equations omit.

This page covers the version of the paper that the stub names: the ICML 2018 paper, arXiv 1801.01290v2. It trains a value network $V_\psi$ and uses a fixed temperature. The follow-up paper, "Soft Actor-Critic Algorithms and Applications" (arXiv 1812.05905), learns the temperature automatically. This paper doesn't contain that method. The course lesson linked below covers it.

Every number comes from a script in `scripts/`. Run any script from the paper folder to reproduce its output.

### The maximum entropy objective and the soft value

**Intuition.** Standard RL rewards the agent for return only. Maximum entropy RL also pays it for being unpredictable. When two actions are equally good, the best policy gives both equal probability, and it never writes off an action that might turn out to be useful. This helps exploration and keeps several modes of good behavior (Section 3.2).

**Math.** The objective adds the entropy of the policy at every visited state (Section 3.2, Equation 1):

$$
J(\pi) = \sum_{t=0}^{T} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi}\big[\, r(s_t, a_t) + \alpha\, \mathcal{H}(\pi(\cdot \mid s_t)) \,\big]
$$

The temperature $\alpha$ sets the trade-off. As $\alpha \to 0$ you recover the ordinary objective. The paper then drops $\alpha$ from its notation, because scaling the reward by $\alpha^{-1}$ absorbs it. The infinite-horizon discounted version is in Appendix A.

The soft Bellman backup and the soft state value (Section 4.1, Equations 2 and 3) read:

$$
\mathcal{T}^\pi Q(s_t, a_t) = r(s_t, a_t) + \gamma\, \mathbb{E}_{s_{t+1} \sim p}\big[V(s_{t+1})\big], \qquad V(s_t) = \mathbb{E}_{a_t \sim \pi}\big[Q(s_t, a_t) - \log \pi(a_t \mid s_t)\big]
$$

The value is the expected Q-value plus the entropy of the policy. For a discrete action set, the policy that maximizes $V$ has a closed form. With the temperature put back, the maximizer is the Boltzmann policy $\pi(a \mid s) \propto \exp(Q(s, a)/\alpha)$ and the maximum is $V(s) = \alpha \log \sum_a \exp(Q(s, a)/\alpha)$. The paper reaches the same policy through the KL projection in Equation 4, which is the next idea.

**Tiny example, computed by `scripts/soft_value.py`.** Take three actions with $Q = (1, 2, 3)$ and $\alpha = 1$. The Boltzmann policy is $(0.0900, 0.2447, 0.6652)$ with entropy $0.8324$. Then $\mathbb{E}[Q - \log \pi] = 3.4076$, and the closed form $\log\sum_a e^{Q_a}$ gives the same $3.4076$. Other policies score lower on the same formula:

| Policy | $\mathbb{E}[Q]$ | Entropy | $V$ |
|---|---|---|---|
| Greedy | 3.0000 | 0 | 3.0000 |
| Uniform | 2.0000 | 1.0986 | 3.0986 |
| Boltzmann | 2.5752 | 0.8324 | 3.4076 |

The Boltzmann policy has a lower expected Q than the greedy one and still wins, because it earns 0.8324 in entropy. With $\alpha = 0.1$ the value is $3.0000$, which is $\max_a Q$. With $\alpha = 10$ it is $13.0194$, close to $\bar Q + \alpha \log 3 = 12.9861$. Two actions with $Q = (0, 0)$ give $V = 0.6931 = \log 2$.

Exercise 1 asks you to write the closed form and the policy without overflow.

### Soft policy iteration

**Intuition.** Policy iteration alternates between scoring the current policy and improving it. The soft version scores with the soft backup and improves by moving the policy toward the exponential of the new Q-function. Because you can restrict the policy to a family such as Gaussians, the improvement step projects back into that family.

**Math.** Soft policy evaluation applies $\mathcal{T}^\pi$ repeatedly and converges to the soft Q-value of $\pi$ (Lemma 1). The improvement step minimizes a KL divergence for each state (Section 4.1, Equation 4):

$$
\pi_{\text{new}} = \arg\min_{\pi' \in \Pi}\ D_{\mathrm{KL}}\!\left(\pi'(\cdot \mid s_t)\ \Big\|\ \frac{\exp\big(Q^{\pi_{\text{old}}}(s_t, \cdot)\big)}{Z^{\pi_{\text{old}}}(s_t)}\right)
$$

The partition function $Z$ depends only on the state and has no effect on the gradient. The projected policy has a soft Q-value at least as high everywhere (Lemma 2), and alternating the two steps converges to the best policy in $\Pi$ (Theorem 1). All three results assume $|\mathcal{A}| < \infty$, so that the entropy-augmented reward is bounded (Appendix B). The practical algorithm works in continuous actions without that guarantee.

**Tiny example, computed by `scripts/soft_value.py`.** On the three-action state above, the uniform policy has $V = 3.0986$. Improving it to the Boltzmann policy of its Q-values raises $V$ to $3.4076$. With a table for $Q$ and no policy restriction, the KL in Equation 4 is exactly 0 at the Boltzmann policy.

### Value network, twin critics, and targets

**Intuition.** Exact soft policy iteration is impossible with function approximators, so SAC trains three kinds of network by SGD on batches from the replay buffer. A value network $V_\psi$ learns the soft value. Two Q-networks learn the soft Q-values, and each regresses onto a target that uses a slowly moving copy of the value network. The minimum of the two Q-networks keeps a single critic's optimistic errors from feeding back.

**Math.** The value network minimizes a squared residual (Section 4.2, Equation 5):

$$
J_V(\psi) = \mathbb{E}_{s_t \sim \mathcal{D}}\Big[\tfrac12\big(V_\psi(s_t) - \mathbb{E}_{a_t \sim \pi_\phi}\big[Q_\theta(s_t, a_t) - \log \pi_\phi(a_t \mid s_t)\big]\big)^2\Big]
$$

The states come from the replay buffer, but the actions are sampled from the current policy. That is why a value network can learn off-policy (Equation 6 gives the gradient estimator). The paper notes that you could estimate the value from one action sample without a separate network, and that including one stabilizes training (Section 4.2). Each Q-function minimizes the soft Bellman residual (Equations 7 and 8):

$$
J_Q(\theta) = \mathbb{E}_{(s_t, a_t) \sim \mathcal{D}}\Big[\tfrac12\big(Q_\theta(s_t, a_t) - \hat Q(s_t, a_t)\big)^2\Big], \qquad \hat Q(s_t, a_t) = r(s_t, a_t) + \gamma\, \mathbb{E}_{s_{t+1} \sim p}\big[V_{\bar\psi}(s_{t+1})\big]
$$

The target network $V_{\bar\psi}$ is an exponential moving average of $V_\psi$ with smoothing constant $\tau$. Algorithm 1 applies $\bar\psi \leftarrow \tau\psi + (1 - \tau)\bar\psi$ after every gradient step, which is DDPG's soft update (see the DDPG page) applied to the value network only. There is no target actor and no target policy.

Section 4.2 uses two Q-functions, with parameters $\theta_1$ and $\theta_2$, trained independently. It takes their minimum for the value gradient in Equation 6 and the policy gradient in Equation 13, following TD3. This differs from TD3, where the minimum goes into the critic's own target. Here the minimum enters $J_V$ and $J_\pi$, and the Q target in Equation 8 uses the value network. The paper says SAC learns even a 21-dimensional Humanoid with one Q-function, but two Q-functions speed up training considerably on harder tasks.

Equation 8 has no terminal flag. This page and Exercise 3 use $\hat Q = r + \gamma (1 - d) V_{\bar\psi}(s_{t+1})$.

**Tiny example, computed by `scripts/soft_targets.py`.** Take $\gamma = 0.99$ and a batch of three transitions with $Q_{\theta_1} = (1.0, 0.5, 2.0)$, $Q_{\theta_2} = (0.8, 0.9, 1.5)$, and $\log \pi = (-1.2, -0.3, -0.5)$ at the sampled actions. The minimum is $(0.8, 0.5, 1.5)$, so the value targets are $\min(Q_1, Q_2) - \log \pi = (2.0, 0.8, 2.0)$. For $V_\psi = (1.5, 0.4, 1.9)$, the errors are $(-0.5, -0.4, -0.1)$ and $J_V = 0.07$. For the Q target, take $r = (1, 0, 2)$, $V_{\bar\psi}(s') = (3, 2, 1)$, and a terminal third row: $\hat Q = (3.97, 1.98, 2.0)$. Without the mask, the third target would be $2.99$.

**Why the minimum.** The script also tests the paper's reason for using two critics. Five actions are all worth 0, and each critic's estimate has independent noise with standard deviation 1. The expected maximum of one critic is $1.165$, which is pure overestimation. The expected maximum of the minimum of two critics is $0.380$. A policy that follows one critic's argmax picks an action whose other critic's value is, on average, $-0.001$: the two critics don't share the error. For the soft value $\log\sum_a e^{Q_a}$ on the same noisy critics, one critic gives $1.976$ against a true $\log 5 = 1.609$, and the twin minimum gives $1.297$. The minimum is a pessimistic estimate. The paper accepts that, because it wants to mitigate the positive bias that degrades value-based methods (Section 4.2).

Exercise 3 asks you to write both targets and the loss.

### A reparameterized stochastic policy

**Intuition.** The policy minimizes the KL divergence to the exponential of the Q-function. The Q-function is a network, so you can differentiate it. If you write the sampled action as a function of the policy's parameters and a fixed noise source, the gradient flows from $Q$ back through the action to the parameters, the way it does in DDPG.

**Math.** The policy loss is the expected KL from Equation 4 (Section 4.2, Equation 10). With $a_t = f_\phi(\epsilon_t; s_t)$ and $\epsilon_t$ drawn from a fixed distribution such as a spherical Gaussian (Equation 11), the loss becomes (Equation 12):

$$
J_\pi(\phi) = \mathbb{E}_{s_t \sim \mathcal{D},\, \epsilon_t \sim \mathcal{N}}\big[\log \pi_\phi(f_\phi(\epsilon_t; s_t) \mid s_t) - Q_\theta(s_t, f_\phi(\epsilon_t; s_t))\big]
$$

The partition function has dropped out. The gradient estimate (Equation 13) is

$$
\hat\nabla_\phi J_\pi(\phi) = \nabla_\phi \log \pi_\phi(a_t \mid s_t) + \big(\nabla_{a_t} \log \pi_\phi(a_t \mid s_t) - \nabla_{a_t} Q(s_t, a_t)\big)\nabla_\phi f_\phi(\epsilon_t; s_t)
$$

with $a_t = f_\phi(\epsilon_t; s_t)$. The paper says this extends DDPG-style policy gradients to any tractable stochastic policy. The $-\nabla_a Q \cdot \nabla_\phi f$ term is DDPG's actor gradient, and the $\log \pi$ terms are the entropy bonus. The paper writes the policy as a Gaussian whose mean and covariance come from neural networks (Section 4.2). It uses the minimum of the two Q-functions as $Q_\theta$ here.

The paper's alternative is the likelihood-ratio gradient (Williams, 1992), which doesn't need to differentiate $Q$. The paper says the reparameterized form has lower variance.

**Tiny example, computed by `scripts/reparam_vs_score.py`.** Take one state, $a \sim \mathcal{N}(\mu, \sigma^2)$, and the critic $Q(a) = -(a-2)^2$. Then $\mathbb{E}[Q] = -(\mu - 2)^2 - \sigma^2$, so the exact gradient with respect to $\mu$ at $\mu = 0.5$ is $3$. Both estimators average over a batch of 16 draws and are repeated for 20,000 batches:

| Estimator | Mean | Variance |
|---|---|---|
| Reparameterized, $\mathbb{E}[Q'(a)]$ | 2.999 | 0.251 |
| Score function, $\mathbb{E}[Q(a)\,\partial_\mu \log \pi]$ | 3.011 | 3.240 |

Both are unbiased. The score estimator's variance is 12.9 times larger. The reparameterized variance matches the theory, $4\sigma^2/16 = 0.250$. Subtracting the exact mean as a baseline cuts the score variance to $1.753$, which is still 7 times the reparameterized value.

**Policy loss check, computed by `scripts/policy_loss.py`.** For the same $Q$ with $\mu = 0.5$ and $\log\sigma = 0$, take four fixed noise draws $\epsilon = (-1, 0, 1, 0.5)$. Equation 12 gives $J_\pi = 1.23731$. For the mean, the $\log \pi$ terms in Equation 13 cancel, and the gradient is $-\overline{Q'(a)} = -2.75$. For $\log\sigma$ the gradient is $-1 - \sigma\,\overline{Q'(a)\epsilon} = -0.25$. Central finite differences give $-2.75000$ and $-0.25000$. The $-1$ is the entropy term, which pushes $\sigma$ up.

Here $\exp(Q) \propto \mathcal{N}(2, 1/2)$, so Equation 4 says the best Gaussian has $\mu = 2$ and $\sigma = 0.7071$. Gradient descent with 200,000 fixed noise draws and 2,000 steps of size 0.05 reaches $\mu = 2.000$ and $\sigma = 0.706$. With the entropy term removed, the same loop collapses to $\sigma = 0.0498$ and still falls.

Exercise 4 asks you to write both gradient estimators.

### Bounded actions with a tanh squash

**Intuition.** A Gaussian sample can be any real number, but torques have limits. SAC applies $\tanh$ to the sample. The action is then in $(-1, 1)$, and you rescale it to the real range. The squash changes the density, so the entropy bonus needs a corrected log-probability.

**Math.** Let $u \in \mathbb{R}^D$ have density $\mu(u \mid s)$ and let $a = \tanh(u)$ element-wise. The density of $a$ follows from the change of variables (Appendix C, Equation 20), and the Jacobian is diagonal, so the log-likelihood is (Equation 21):

$$
\log \pi(a \mid s) = \log \mu(u \mid s) - \sum_{i=1}^{D} \log\big(1 - \tanh^2(u_i)\big)
$$

The paper writes the Gaussian density as $\mu(u \mid s)$. This page writes $\mathcal{N}(u; \mu, \sigma^2)$ instead, with $\mu$ as the mean, and the exercise code uses `mu` for the mean. A numerically stable form of the correction is $\log(1 - \tanh^2 u) = 2\,(\log 2 - u - \mathrm{softplus}(-2u))$.

**Tiny example, computed by `scripts/squash_logprob.py`.** Take $u = 0.5$, $\mu = 0$, and $\sigma = 1$. Then $a = \tanh(0.5) = 0.4621$. The Gaussian log-density is $-1.0439$, the correction is $\log(1 - \tanh^2 0.5) = -0.2402$, and $\log \pi(a) = -1.0439 + 0.2402 = -0.8037$. The correction raises the density near the edges of $(-1, 1)$, where the squash compresses the values, and the density at $a = 0.4621$ is $0.4477$.

Two numerical checks agree. The slope of $P(\tanh(u) \le a)$ at $a = 0.4621$ is $0.447665$, and $\exp(-0.8037) = 0.447665$. The density integrates to $1.000000$ over $(-1, 1)$, and the version without the correction integrates to $0.6057$. In two dimensions, with $u = (0.5, -1)$, $\mu = (0, 0.5)$, and $\sigma = (1, 0.5)$, the log-probability is $-4.6619$.

At $u = 25$, $1 - \tanh^2 u$ rounds to $0$ in float64, so the naive correction is $-\infty$. The stable form gives $-48.6137$.

Exercise 2 asks you to write this log-probability and test it against these checks.

### Reward scale, evaluation, and results

**Intuition.** With $\alpha$ folded into the reward, the reward scale acts as an inverse temperature. A larger reward magnitude means a lower entropy weight, and a nearly deterministic policy. This is the one hyperparameter that needs tuning (Section 5.2).

**Math.** Table 1 gives the shared settings: Adam with a learning rate of $3 \times 10^{-4}$, $\gamma = 0.99$, a replay buffer of $10^6$ transitions, 2 hidden layers of 256 ReLU units in every network, and a minibatch of 256. The target smoothing coefficient is $\tau = 0.005$, with one target update and one gradient step per environment step. Table 2 lists the reward scale per environment: 5 for Hopper-v1, Walker2d-v1, HalfCheetah-v1, and Ant-v1, 20 for Humanoid-v1, and 10 for the rllab Humanoid. A reward scale of 5 is an entropy weight $\alpha = 1/5 = 0.2$ in this page's notation, and 20 is $\alpha = 0.05$.

The paper evaluates SAC with the mean of the policy, as an approximation of the maximum a posteriori action (Section 5.2). Figure 3(a) shows that this deterministic evaluation tends to return more than sampling from the policy. Training curves in Figure 1 report the sum of rewards, which is not the objective that SAC optimizes.

**Results.** Figure 1 compares SAC with DDPG, PPO, soft Q-learning (SQL), and the concurrent TD3 on Hopper, Walker2d, HalfCheetah, Ant, Humanoid-v1, and rllab Humanoid, with five seeds each and one evaluation rollout per 1,000 steps. SAC matches the baselines on the easier tasks and wins by a large margin on the harder ones. DDPG makes no progress on Ant-v1 and both Humanoids. SQL learns all the tasks but more slowly than SAC, and with worse asymptotic performance. Figure 2 shows much higher seed-to-seed variability for a deterministic variant, and Figure 3(b) shows poor results for reward scales that are too small (a nearly uniform policy) and too large (a nearly deterministic one). In Figure 3(c), a large $\tau$ causes instability and a small one slows learning. The paper used $\tau = 0.005$ on every task.

**Tiny example, computed by `scripts/soft_value.py`.** The reward-scale and temperature relationship is the one in the soft-value table. At $\alpha = 0.1$, the value of $Q = (1, 2, 3)$ is $3.0000$ and the policy is nearly greedy. At $\alpha = 10$ it is nearly uniform, and the value is $13.0194$.

<!-- exercises -->

## Pitfalls

- **The reward scale is the temperature.** The paper drops $\alpha$ and scales the reward instead. If you learn from a codebase that exposes $\alpha$, divide the reward scale into 1 to compare settings. Section 5.2 says too small a scale gives a nearly uniform policy and too large a scale gives a nearly deterministic one.
- **The target is a value network, not a Q target network.** This version has $V_\psi$ and $V_{\bar\psi}$. Later versions, including the one in the course lesson, remove the value network and use target Q-networks and a learned $\alpha$. The Equation 8 target uses only $V_{\bar\psi}(s_{t+1})$.
- **The minimum of the Q-functions belongs in two places.** The paper uses it in the value gradient (Equation 6) and the policy gradient (Equation 13). Each Q-function regresses onto its own Equation 8 target, which uses the value network.
- **Sample actions from the current policy for $V$ and $J_\pi$.** In Equations 5 and 6, the states come from the replay buffer but the actions don't. Using the buffer's action in the value target gives you the behavior policy's value.
- **Terminal states need a mask.** Equation 8 and Algorithm 1 omit it, as the DDPG paper does. Without it, the critic bootstraps through the end of an episode. The paper doesn't discuss time-limit truncation.
- **Backpropagate the policy loss into the policy only.** $J_\pi$ differentiates through $Q_\theta$ with respect to the action, and the Q-parameters must not receive this gradient.
- **Keep the squash correction in the policy loss.** Without the Jacobian term, the log-probability and the entropy bonus are wrong. The density integrates to $0.6057$ in the page example instead of 1.
- **The theory is for finite action sets.** Lemmas 1 and 2 and Theorem 1 assume $|\mathcal{A}| < \infty$, and Section 4.1 does the analysis in the tabular case. The deep algorithm approximates it without a guarantee.
- **The paper doesn't say how to bound the standard deviation.** It describes a Gaussian with a network-produced mean and covariance (Section 4.2) and doesn't give a minimum or maximum for the log standard deviation. Check an implementation's choice before you copy it.
- **Evaluate with the mean action, and report that choice.** Figure 3(a) shows evaluation with the mean differs from sampling. Compare algorithms under the same evaluation rule.
- **Keep the target schedule fixed when you compare runs.** The hard-update variant in Appendix E copies the target every 1,000 gradient steps and uses 4 gradient steps per environment step on all tasks except the humanoids. Mixing the two target schedules gives non-comparable runs.

## Further reading

- [Official implementation](https://github.com/haarnoja/sac), the code that the paper links (Section 5), and the [project page with videos](https://sites.google.com/view/soft-actor-critic).
- [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905), Haarnoja et al., 2018. The follow-up that learns the temperature automatically and drops the value network.
- [Spinning Up: Soft Actor-Critic](https://spinningup.openai.com/en/latest/algorithms/sac.html). A readable summary and reference implementation of the later formulation.
- [Reinforcement Learning with Deep Energy-Based Policies](https://arxiv.org/abs/1702.08165), Haarnoja et al., ICML 2017. The soft Q-learning method that this paper compares against and improves on.
