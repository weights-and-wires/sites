---
id: "td3"
title: "Addressing Function Approximation Error in Actor-Critic Methods"
authors: "Fujimoto, van Hoof, Meger"
year: 2018
venue: "ICML 2018"
arxiv: "1802.09477"
url: "https://arxiv.org/abs/1802.09477"
track: rl
tier: extended
est_hours: 4
status: full
summary: "This paper shows that DDPG's critic overestimates action values, because the actor climbs the critic's errors, and that the usual double-Q fix from discrete control does not remove the bias. It trains two critics and uses the smaller of their two estimates as the target, adds clipped noise to the target action, and updates the actor and the target networks only every other critic step. The resulting algorithm, TD3, beats DDPG and several other methods on seven MuJoCo tasks."
related_course: "rl: continuous control"
prereqs:
  - id: ddpg
    reason: "TD3 is DDPG with three changes, so you need DDPG's actor, critic, replay buffer, and soft target updates."
  - id: dqn
    reason: "The paper starts from the overestimation that comes with the max in DQN's target and from Double DQN's remedy, so you need that target."
background: [deterministic-policy-gradient, maximization-bias, q-learning]
course_links:
  - title: "Deterministic policy gradients: DDPG and TD3"
    url: "https://rl.ifkash.dev/lessons/040-deterministic-policy-gradients"
  - title: "Maximization bias and double learning"
    url: "https://rl.ifkash.dev/lessons/018-maximization-bias"
  - title: "Offline reinforcement learning (builds the TD3+BC actor loss)"
    url: "https://rl.ifkash.dev/lessons/053-offline-rl"
self_check:
  - q: 'An agent has 2 actions whose true values are both 0. A critic estimates each one as the truth plus independent $\mathcal{N}(0, 1)$ noise (Section 4). What is the expected value of the maximum estimate, and how does it change with 10 actions?'
    a: 'For two actions it is $1/\sqrt{\pi} \approx 0.5642$ even though the true maximum is 0. With 10 actions it is $1.5388$. Noise in the estimates is zero-mean, but the maximum picks the positive errors, so the bias grows with the number of candidates.'
    see: maximization-bias
  - q: 'At the next state, the two target critics score the target action as $Q''_1 = 5$ and $Q''_2 = 4$. The reward is 0, $\gamma = 0.9$, and the episode continues. What target does TD3 use (Equation 15), and what would DDPG-style training on the first critic alone use?'
    a: 'TD3 uses $y = 0 + 0.9 \cdot \min(5, 4) = 3.6$. Training on the first critic alone gives $0.9 \cdot 5 = 4.5$. The minimum can only lower the target, and when the critics disagree it takes the one that is less likely to be inflated.'
    see: ddpg
  - q: 'With $\tau = 0.005$ and a policy delay $d = 2$ (Section 6.1), roughly how many critic steps pass before a target network has closed half of a fixed gap to its source?'
    a: 'The targets move only on actor steps, and each one shrinks the gap by $1 - \tau = 0.995$. You need $0.995^k \le 0.5$, so $k = 139$ target updates. Two critic steps pass per target update, so that is 278 critic steps.'
    see: deterministic-policy-gradient
---

## The problem

DDPG trains a critic $Q_\theta(s, a)$ with a bootstrapped target and an actor that climbs the critic. The critic is a neural network, so its value estimates carry errors. Section 1 recalls the consequence from discrete control: a maximization over noisy estimates is biased upward (Thrun and Schwartz, 1993). The bias then feeds on itself, because each target contains the estimate of the next state.

DDPG has no explicit maximum, so it is tempting to think it avoids the problem. Section 4.1 argues that it does not. The actor takes a gradient step on the approximate critic, so the actor moves to where the critic looks best, and the critic looks best where its errors are most positive. Under the assumptions of that section, the value estimate of the new policy is at least its true value (Equations 5 to 7). Figure 1 measures this on Hopper and Walker2d: the average value that DDPG's critic reports sits well above the return the policy actually earns.

The obvious fix is Double DQN, which scores the action that the online network chooses with the target network. Section 4.2 explains why that fails here. The actor changes slowly, so the current and target networks stay too similar, and the estimates are not independent enough. Figure 2 shows that the actor-critic versions of Double DQN and Double Q-learning still overestimate.

A second problem sits next to the first. Each TD update leaves a residual error, and the errors add up over the horizon (Section 5.1). A policy that is trained on a high-variance critic then produces poor actions that the critic rates highly, which makes the next critic worse (Section 5.2). A concrete size of the effect comes from Equation 12: a constant residual error of 0.1 per step at $\gamma = 0.99$ shifts the value by $0.1 / (1 - 0.99) = 10$ (computed by `scripts/accumulating_error.py`).

The result, in Section 6: TD3 matches or beats every method in the comparison on final return, with a near tie on InvertedDoublePendulum, where DDPG is 18.05 higher. Table 1 reports a mean of 9636.95 on HalfCheetah against 3305.60 for DDPG, and 4372.44 on Ant against 1005.30. Those figures come from 10 seeds and 1 million steps.

## Key ideas

This page uses the paper's notation: the actor $\pi_\phi$, two critics $Q_{\theta_1}$ and $Q_{\theta_2}$, their target copies with primes, the replay buffer $B$, the discount $\gamma$, the soft-update rate $\tau$, and the policy delay $d$. The paper writes $\tilde{a}$ for the smoothed target action. The DDPG page writes the actor as $\mu$ with weights $\theta^\mu$, so the symbols differ, but the objects are the same. This page adds a terminal flag $\text{done}_i$ for the exercises. The paper's Algorithm 1 has none, and the flag is not the delay $d$.

### Overestimation bias in actor-critic

**Intuition.** Every estimate is the truth plus an error. If you pick the best-looking action, you tend to pick one whose error is positive, so the value you read is too high. A policy gradient step does the same thing, a little at a time.

**Math.** For discrete actions, Section 4 gives the starting inequality, $\mathbb{E}_\epsilon[\max_{a'}(Q(s', a') + \epsilon)] \ge \max_{a'} Q(s', a')$. For DPG, Section 4.1 compares two actor updates (Equation 4): $\phi_{\text{approx}}$, from the gradient of the approximate critic, and $\phi_{\text{true}}$, from the gradient of the true value. The gradients are normalized, and for a small enough step $\alpha$, the approximate critic prefers its own update (Equation 5), the true value prefers the other (Equation 6), and the approximate value ends up at or above the true value (Equation 7), provided the estimate was not already below the truth for $\phi_{\text{true}}$. The paper warns that each step may add only a little bias, but it can build up over many updates, and a wrong critic gives wrong policy updates.

**Tiny example, computed by `scripts/overestimation.py`.** Give an agent 2 actions, each with true value 0, and estimate each as the truth plus $\mathcal{N}(0, 1)$ noise. The maximum estimate has expectation $1/\sqrt{\pi} = 0.5642$. With 5 actions it is 1.1630, and with 10 actions it is 1.5388. A simulation of 200,000 trials gives 0.5665 and 1.5393. The bias scales with the noise level: it is 0.7697, 1.5393, and 3.0786 for $\sigma = 0.5, 1, 2$.

### Clipped double Q-learning

**Intuition.** Double Q-learning picks the action with one critic and scores it with another, so the noise that made the pick does not inflate the score. In an actor-critic setting the two critics still share a replay buffer and bootstrap from each other, so they are not fully independent. The paper's answer is to treat the less biased estimate as something that the more biased one can cap.

**Math.** The original Double Q-learning in an actor-critic form keeps two actors and two critics (Section 4.2, Equation 9). The clipped version keeps the pair of critics, takes the smaller of the two target values, and trains both critics toward that one target (Section 4.2, Equation 10, and Section 6, Equation 15):

$$
y = r + \gamma \min_{i=1,2} Q_{\theta'_i}\big(s', \pi_{\phi'}(s') + \epsilon\big)
$$

In the implementation form, one actor is optimized against $Q_{\theta_1}$ and both critics regress to the same $y$ (Section 4.2). If $Q_{\theta_2} > Q_{\theta_1}$, the update equals the standard one. If $Q_{\theta_2} < Q_{\theta_1}$, the critic's value is reduced. The paper says the target can't add overestimation beyond the standard target, and that it may underestimate. It argues that underestimation is safer, because the policy update doesn't push toward actions that the critic undervalues. It also notes that the expected minimum of random variables falls as their variance rises, so states with noisy estimates are valued lower and the policy prefers states with low-variance estimates.

**Tiny example, computed by `scripts/clipped_target.py`.** Take $\gamma = 0.9$, rewards $(1, 0, 2)$, and a terminal flag $(0, 0, 1)$. The target critics give $Q'_1 = (2, 5, -1)$ and $Q'_2 = (3, 4, -2)$ for the smoothed action. The minimum is $(2, 4, -2)$ and the targets are $(2.8, 3.6, 2.0)$. The first critic alone would give $(2.8, 4.5, 2.0)$, and their mean $(3.25, 4.05, 2.0)$. With critic outputs $(2.5, 3, 1.5)$ and $(2, 3.5, 2.5)$, the two mean squared errors are 0.23333 and 0.3, and the summed loss is 0.53333.

The toy from the previous idea also shows the sign of the correction. Critic $A$ picks the action and critic $B$ scores it. Using $B$ alone, as in Double Q-learning, gives a bias of $-0.0024$ (zero, within noise) with 2 actions. Taking the smaller of the two scores gives $-0.2833$, an underestimate, as the paper warns. For two independent estimates with standard deviation $s$, the expected minimum is $-s/\sqrt{\pi}$, which is $-0.5642$ for $s = 1$.

Exercises 1 and 3 ask you to write the target and the toy.

### Delayed policy updates

**Intuition.** A policy that moves while the critic is still wrong chases the critic's mistakes. Let the critic take several steps for each actor step, so the actor reads a steadier value.

**Math.** Section 5.1 explains why errors build up. A TD update leaves a residual $\delta(s, a)$ (Equation 11), and unrolling the recursion gives a value equal to the expected return minus the discounted sum of future residuals (Equation 12). So the variance of the estimate follows the variance of the future errors, and it can grow quickly when $\gamma$ is large. Section 5.2 adds that target networks limit this accumulation. Figure 3 shows the value of a state in Hopper with $\tau = 1, 0.1, 0.01$: with a fixed policy, all of them converge, but with a learning policy the fast-updating targets diverge.

From this, the paper updates the actor and the target networks only after every $d$ critic updates (Section 5.2, Algorithm 1), with a slow $\tau$ for the targets, $\theta' \leftarrow \tau\theta + (1 - \tau)\theta'$. The experiments use $d = 2$ and $\tau = 0.005$ (Section 6.1). The paper says a larger $d$ would help more against accumulating error, but for a fair comparison the critics train once per time step, and too few actor updates would stop learning. It also calls the scheme a two-timescale algorithm.

**Tiny example, computed by `scripts/accumulating_error.py`.** Over 1 million steps with $d = 2$, the critics make 1,000,000 updates and the actor and targets make 500,000. Among the first 10 steps, the actor updates at steps 2, 4, 6, 8, and 10. A target that tracks a fixed source with $\tau = 0.005$ has 8.16 percent of its gap left after 1,000 critic steps with $d = 2$, and 0.67 percent without the delay. Half the gap closes after 139 target updates, which is 278 critic steps with $d = 2$.

Exercise 4 asks you to write the schedule.

### Target policy smoothing

**Intuition.** A deterministic target action can land on a narrow spike in the critic, which is likely an error. If you average the target over a small neighborhood of actions, one spike counts for little. This also encodes the idea that similar actions should have similar values.

**Math.** Section 5.3 starts from the expectation over a small area around the target action (Equation 13), $y = r + \mathbb{E}_\epsilon[Q_{\theta'}(s', \pi_{\phi'}(s') + \epsilon)]$. In practice it draws one noise sample per transition and relies on the minibatch to average (Equation 14):

$$
y = r + \gamma\, Q_{\theta'}\big(s', \pi_{\phi'}(s') + \epsilon\big), \qquad \epsilon \sim \mathrm{clip}\big(\mathcal{N}(0, \sigma), -c, c\big)
$$

The clip keeps the target near the original action. The paper compares the result to Expected SARSA, with the noise on the target policy chosen apart from the exploration policy. The value being learned is that of the noisy policy defined by $\sigma$. The experiments use $\sigma = 0.2$ and $c = 0.5$ (Section 6.1). Algorithm 1 writes the noise scale of this step as $\tilde{\sigma}$, to tell it from the exploration noise $\sigma$. Algorithm 1 doesn't show a clip to the environment's action bounds. Implementations clip $\tilde{a}$ to the bounds, and the exercise does too. This page did not check the paper's supplementary material for this detail.

**Tiny example, computed by `scripts/smoothing.py`.** Take a target action 0.8, bounds $[-1, 1]$, $\sigma = 0.2$, $c = 0.5$, and standard normal draws $(0.5, 3, -4)$. The raw noise $\sigma z$ is $(0.1, 0.6, -0.8)$, which the clip changes to $(0.1, 0.5, -0.5)$. The actions are $(0.9, 1.0, 0.3)$, and the second one needed the bound clip as well. To see the smoothing, take a critic that is $0.5a$ plus a spike of height 1 and width 0.02 centered at 0.8. The unsmoothed target reads $Q'(0.8) = 1.4$. The mean over 100,000 smoothed draws is 0.4617, which is close to the smooth part, 0.4.

Exercise 2 asks you to write the target action.

### The full algorithm and the results

**Intuition.** TD3 is DDPG plus these three changes and a modest retuning. The ablations show that each piece matters on some tasks and that the pieces work best together.

**Math.** Algorithm 1 loops over time steps. It acts with $\pi_\phi(s) + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma)$, stores the transition, and samples a minibatch. It computes $\tilde{a}$ and $y$ from the target networks (Equation 15), and updates both critics by least squares toward $y$. Then, when the step number is a multiple of $d$, it updates $\phi$ by the deterministic policy gradient through $Q_{\theta_1}$ only (Equation 1 in Section 3) and soft-updates all three target networks.

The settings in Section 6.1 are as follows:

- Both the actor and the critics use two hidden layers of 400 and 300 ReLU units, with a tanh on the actor's output. Unlike the original DDPG, the critic reads the state and action together at its first layer.
- Adam with learning rate $10^{-3}$, a minibatch of 100, and a replay buffer holding the whole history.
- A purely exploratory policy for the first 10,000 steps on HalfCheetah and Ant and the first 1,000 on the other tasks, then Gaussian exploration noise $\mathcal{N}(0, 0.1)$.
- The paper chose uncorrelated exploration noise because Ornstein-Uhlenbeck noise offered no benefit.
- Evaluation every 5,000 steps over 10 episodes with no exploration noise, with results over 10 seeds.

Table 1 gives the maximum average return over 10 trials. TD3 has the best value on HalfCheetah (9636.95), Hopper (3564.07), Walker2d (4682.82), Ant (4372.44), and Reacher ($-3.60$), ties at 1000.00 on InvertedPendulum, and is close on InvertedDoublePendulum (9337.47 against 9355.52 for DDPG). The paper's own retuned DDPG ("Our DDPG") reaches 8577.29 on HalfCheetah, so part of the gain comes from the architecture and hyperparameters.

Table 2 ablates the three components. On Ant, the full TD3 gets 4185.06 and removing clipped double-Q drops it to 849.75. On Hopper, removing it drops 3304.75 to 1837.32. Adding a single component to the retuned DDPG ("AHE") often helps little, and the combination does better. Table 2 also shows that the actor-critic variants of Double Q-learning and Double DQN, with all of the other changes, do no better than TD3 except for DDQN-AC on HalfCheetah (10306.90 against 9532.99), which supports the choice of the clipped form.

<!-- exercises -->

## Pitfalls

- **The actor trains on one critic.** Algorithm 1 differentiates $Q_{\theta_1}$ only. The minimum applies to the target, not to the actor's objective.
- **Both critics share one target.** If you compute a separate target per critic from its own target network, you lose the clipping. Take the minimum first, then regress both critics to the result.
- **Delay the targets with the actor.** In Algorithm 1 the soft updates sit inside the `if` block. Updating the targets every step shrinks the effect of the delay (see the gap example above).
- **Two different noises.** Exploration noise ($\sigma = 0.1$) goes on the action that you send to the environment. Smoothing noise ($\sigma = 0.2$, clipped at $c = 0.5$) goes on the target action only. Mixing them up changes the algorithm.
- **Don't use the smoothed action in the actor loss.** The actor update uses the plain $\pi_\phi(s)$.
- **Terminal states and time limits.** The pseudocode omits the terminal flag. Mask the bootstrap at true terminals. Handling time-limit truncation is a separate choice that the paper doesn't discuss.
- **Clipping removes overestimation but adds underestimation.** The toy above shows a negative bias for the clipped target. The paper judges that trade-off to be a good one, but it is a trade-off, not an unbiased estimator.
- **Gains are uneven.** Table 2 shows that a single component helps little on some tasks and a lot on others, and the paper's own retuned DDPG already closes much of the gap on HalfCheetah. Compare against a tuned baseline.
- **Algorithm 1 says `if t mod d`.** Read it as "if $t \bmod d = 0$".

## Further reading

- [Official implementation](https://github.com/sfujim/TD3), the code that the paper links (Section 1), with the learning curves.
- [Spinning Up: Twin Delayed DDPG](https://spinningup.openai.com/en/latest/algorithms/td3.html). A readable summary and a reference implementation.
- [Double Q-learning](https://papers.nips.cc/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html), van Hasselt, NeurIPS 2010. The estimator that clipped double-Q builds on.
- [A Minimalist Approach to Offline Reinforcement Learning](https://arxiv.org/abs/2106.06860), Fujimoto and Gu, 2021. TD3+BC, which adds a cloning term to the TD3 actor loss. The rl course builds it in lesson 053.
- [Soft Actor-Critic](https://arxiv.org/abs/1801.01290), Haarnoja et al., 2018. The method that the paper compares against, which also uses two critics and the minimum.
