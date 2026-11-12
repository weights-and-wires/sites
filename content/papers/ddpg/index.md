---
id: "ddpg"
title: "Continuous control with deep reinforcement learning"
authors: "Lillicrap et al."
year: 2015
venue: "ICLR 2016"
arxiv: "1509.02971"
url: "https://arxiv.org/abs/1509.02971"
track: rl
tier: extended
est_hours: 4
status: full
summary: "This paper adapts DQN to continuous actions by pairing a Q-network, the critic, with a second network, the actor, that outputs one action per state and is trained by following the critic's slope. It borrows DQN's replay buffer and target networks, replaces the periodic copy of the target with a slow moving average, and adds correlated noise to the actor's action for exploration. One set of hyperparameters solves more than 20 simulated physics tasks, some of them from raw pixels."
related_course: "rl: continuous control"
prereqs:
  - id: dqn
    reason: "DDPG reuses DQN's replay buffer, TD-trained Q-network, and target network, so you need to know why each one is there."
background: [deterministic-policy-gradient, ornstein-uhlenbeck-noise]
course_links:
  - title: "Deep Q-networks"
    url: "https://rl.ifkash.dev/lessons/026-deep-q-networks"
  - title: "Deterministic policy gradients: DDPG and TD3"
    url: "https://rl.ifkash.dev/lessons/040-deterministic-policy-gradients"
self_check:
  - q: 'A target network follows its source with $\tau = 0.001$ (Section 7). Roughly how many updates pass before it has closed half of a fixed gap to the source?'
    a: 'The gap shrinks by a factor $(1 - \tau)$ per update, so you need the smallest $n$ with $0.999^n \le 0.5$. That is $n = 693$. A target that tracks a source moving at a steady speed lags it by about $1/\tau = 1000$ updates, which is why the targets change slowly and the critic sees a nearly fixed regression problem.'
    see: dqn
  - q: 'At one state $s = 3$ the actor is $\mu_\theta(s) = \theta s$ and outputs $0.5$. The critic is $Q(s, a) = -(a - 2)^2$. What is the single-sample gradient in Equation 6, and which way does $\theta$ move?'
    a: 'The critic slope is $\nabla_a Q = -2(a - 2) = 3$ at $a = 0.5$, and $\nabla_\theta \mu = s = 3$. The product is $9$. It is positive, so a gradient-ascent step raises $\theta$, which raises the action toward the critic''s peak at $a = 2$.'
    see: deterministic-policy-gradient
  - q: 'Exploration noise is an Ornstein-Uhlenbeck process with $\theta = 0.15$ and $\sigma = 0.2$ (Section 7). With the noise term switched off and a starting value of 1, what is the value after two steps with $dt = 1$, and why does the paper prefer this process to independent noise?'
    a: 'Each step multiplies the value by $1 - \theta = 0.85$, so after two steps it is $0.7225$. Consecutive values are correlated (a lag-1 correlation of 0.85), so the noise pushes the action the same way for several steps. On tasks with momentum, independent noise averages out before it moves the system.'
    see: ornstein-uhlenbeck-noise
---

## The problem

DQN picks an action by scoring every action and taking the best one. That works for 18 joystick positions. It does not work for a robot arm, where an action is a vector of real-valued torques (Section 1).

The obvious fix is to discretize each torque. Section 1 shows why that breaks. A 7-joint arm with the coarsest grid, three values per joint, already has $3^7 = 2{,}187$ actions. A grid of 10 values per joint has 10 million, and a DQN output layer fed by 256 hidden units would need about 2.6 billion weights to score them (computed by `scripts/discretization.py`). Fine control needs a fine grid, and the grid size grows exponentially with the number of joints. The grid also discards the fact that torque $0.51$ is close to torque $0.50$.

The other option is to keep a continuous $Q(s, a)$ and run an optimizer over $a$ at every step to find the best action. Section 3 calls this too slow with large networks.

The method the paper builds on, the deterministic policy gradient (DPG), avoids the maximization by learning a second network that outputs the action. Section 1 states the catch: applying DPG naively to neural networks is unstable on hard problems. The paper's contribution is the set of changes that make it stable: DQN's replay buffer and target networks, adapted to an actor-critic pair, plus batch normalization and correlated exploration noise.

The result, in Section 4: one algorithm, one architecture, and one set of hyperparameters (apart from the noise process for Torcs) solve more than 20 simulated physics tasks, with scores normalized so that a random policy gets 0 and the planner iLQG gets 1. Table 1 reports 26 normalized tasks plus Torcs. On pendulum swing-up the average low-dimensional score is 0.946. Section 6 notes that nearly all tasks were solved within 2.5 million steps, about 20 times fewer than DQN uses on Atari.

## Key ideas

This page uses the paper's notation: the actor $\mu(s \mid \theta^\mu)$, the critic $Q(s, a \mid \theta^Q)$, their target copies $\mu'$ and $Q'$ with weights $\theta^{\mu'}$ and $\theta^{Q'}$, the replay buffer $R$, the discount $\gamma$, the soft-update rate $\tau$, and the noise process $\mathcal{N}$. This page writes $N$ for the minibatch size. Every number comes from a script in `scripts/`. Run any script from the paper folder to reproduce its output.

### A deterministic actor, trained through the critic

**Intuition.** The actor outputs one action. The critic can say, for that action, which direction raises the value. The actor then moves its action that way. No search over actions happens, and no random action needs to be sampled and compared.

**Math.** If the policy is deterministic, $\mu : \mathcal{S} \to \mathcal{A}$, the Bellman equation loses its inner expectation over the next action (Section 2, Equation 3):

$$
Q^\mu(s_t, a_t) = \mathbb{E}_{r_t, s_{t+1} \sim E}\big[\, r(s_t, a_t) + \gamma\, Q^\mu(s_{t+1}, \mu(s_{t+1})) \,\big]
$$

The expectation depends only on the environment. So $Q^\mu$ can be learned off-policy from transitions that another behavior policy $\beta$ generated (Section 2). The critic minimizes (Section 2, Equations 4 and 5):

$$
L(\theta^Q) = \mathbb{E}\big[(Q(s_t, a_t \mid \theta^Q) - y_t)^2\big], \qquad y_t = r(s_t, a_t) + \gamma\, Q(s_{t+1}, \mu(s_{t+1}) \mid \theta^Q)
$$

The actor follows the chain rule applied to the expected return $J$ (Section 3, Equation 6):

$$
\nabla_{\theta^\mu} J \approx \mathbb{E}_{s_t \sim \rho^\beta}\Big[\, \nabla_a Q(s, a \mid \theta^Q)\big|_{s = s_t,\, a = \mu(s_t)}\ \nabla_{\theta^\mu} \mu(s \mid \theta^\mu)\big|_{s = s_t} \Big]
$$

Silver et al. (2014) proved that this is the policy gradient (Section 3). Two details matter:

- **The critic's parameters get no gradient from this loss.** You differentiate $Q$ with respect to the action $a$, and then $\mu$ with respect to $\theta^\mu$. Only $\theta^\mu$ is updated.
- **The state distribution is the behavior distribution $\rho^\beta$.** Footnote 2 says that, as in most policy gradient code, the paper ignores the discount in the state-visitation distribution.

**Tiny example, computed by `scripts/dpg_check.py`.** Take a linear actor $\mu(s) = \theta^\top s$ and a quadratic critic $Q(s, a) = -(a - k^\top s)^2$, with $k = (1, -0.5)$, $\theta = (0.2, 0.4)$, and three states $(1,0)$, $(0,1)$, $(1,1)$. The actions are $0.2, 0.4, 0.6$, and the critic's best actions are $1, -0.5, 0.5$. The slope $\nabla_a Q = -2(a - k^\top s)$ is $1.6, -1.8, -0.2$. Since $\nabla_\theta \mu = s$, the per-state products are $(1.6, 0)$, $(0, -1.8)$, and $(-0.2, -0.2)$. Their mean is $(0.4667, -0.6667)$. A central finite difference of $J(\theta) = \tfrac{1}{3}\sum_i Q(s_i, \theta^\top s_i)$ agrees to $5 \times 10^{-11}$. If you forget the factor $\nabla_\theta \mu$ and average only the slopes, you get the single number $-0.1333$, which has the wrong shape and the wrong value.

With learning rate 0.1, gradient ascent gets $\theta$ to $(0.579, -0.068)$ after 10 steps and to $(1.000, -0.500) = k$ after 300, where $J$ reaches 0.

Exercise 4 asks you to write this gradient and check it.

### Replay buffer and target networks for an actor-critic pair

**Intuition.** The critic's target contains the critic itself, so it moves whenever the critic does. Here the target also contains the actor. DQN fixed this with a frozen copy. DDPG keeps a copy of both networks, and the copies change slowly.

**Math.** Transitions $(s_t, a_t, r_t, s_{t+1})$ go into a finite buffer $R$ and the oldest ones are dropped when it is full. At every step the algorithm samples a minibatch uniformly from $R$ and updates the actor and the critic (Section 3). Because $Q^\mu$ is learned off-policy, the buffer can be large. The paper uses $10^6$ transitions (Section 7).

The target networks $Q'$ and $\mu'$ start as copies of $Q$ and $\mu$. Algorithm 1 computes each minibatch target from them:

$$
y_i = r_i + \gamma\, Q'\big(s_{i+1}, \mu'(s_{i+1} \mid \theta^{\mu'}) \mid \theta^{Q'}\big)
$$

and updates the critic on $L = \frac{1}{N} \sum_i (y_i - Q(s_i, a_i \mid \theta^Q))^2$. The actor update then uses the sampled form of Equation 6, averaged over the minibatch. The paper reports that having both target networks was required to train the critic without divergence (Section 3). Figure 2 and Section 4 add that training without a target network, as in the original DPG, is "very poor" in many environments.

Two details differ from what you might expect:

- **Algorithm 1 has no terminal flag.** This page uses $d_i \in \{0, 1\}$, so that $y_i = r_i + \gamma (1 - d_i) Q'(\cdot)$. Implementations add the mask, and the exercises do too.
- **The target actor chooses the next action.** There is no maximum over actions. That is the whole reason a continuous action space works.

**Tiny example, computed by `scripts/target_example.py`.** Take $\gamma = 0.9$ and a scalar state and action. The target actor is $\mu'(s) = 0.5s$, the target critic is $Q'(s, a) = s - (a - s)^2$, and the online critic is $Q(s, a) = 0.2s + a$. A minibatch of three transitions has $s' = (2, -1, 3)$, $r = (0, 1, 2)$, and the third transition is terminal:

| $i$ | $s_{i+1}$ | $\mu'(s_{i+1})$ | $Q'(s_{i+1}, \mu'(s_{i+1}))$ | $d_i$ | $y_i$ |
|---|---|---|---|---|---|
| 1 | 2 | 1.0 | 1.0 | 0 | $0 + 0.9 \cdot 1 = 0.9$ |
| 2 | -1 | -0.5 | -1.25 | 0 | $1 + 0.9 \cdot (-1.25) = -0.125$ |
| 3 | 3 | 1.5 | 0.75 | 1 | $2$ |

Without the mask, $y_3$ would be 2.675. The online critic gives $Q(s_i, a_i) = (0.7, 0.5, 1.4)$ for the stored $s = (1, 0, 2)$ and $a = (0.5, 0.5, 1)$. The errors $y_i - Q$ are $(0.2, -0.625, 0.6)$ and the critic loss is $0.26354$.

Exercise 1 asks you to write the target and the loss.

### Soft target updates

**Intuition.** DQN copies the weights into the target every few thousand steps. The target is frozen for a while and then jumps. DDPG instead moves the target a tiny step toward the online weights after every update, so it never jumps.

**Math.** After each minibatch update (Section 3 and Algorithm 1):

$$
\theta^{Q'} \leftarrow \tau\, \theta^Q + (1 - \tau)\, \theta^{Q'}, \qquad \theta^{\mu'} \leftarrow \tau\, \theta^\mu + (1 - \tau)\, \theta^{\mu'}, \qquad \tau \ll 1
$$

The paper uses $\tau = 0.001$ (Section 7). The target changes slowly, so the critic faces a regression problem that is closer to supervised learning. The cost is that value information reaches the target late. The paper says the delay may slow learning but that stability outweighs it (Section 3).

If the online weights stand still, the gap to the target shrinks by a factor $1 - \tau$ per update, so it is $(1 - \tau)^n$ after $n$ updates. If they move at a constant speed $v$ per step, the target settles at a constant lag of $v(1 - \tau)/\tau$ behind them.

**Tiny example, computed by `scripts/soft_update.py`.** With $\tau = 0.1$ and a source frozen at 1, a target that starts at 0 reads $0.100$, $0.190$, $0.271$ after one, two, and three updates. With the paper's $\tau = 0.001$, it takes 693 updates to close half the gap, 2,302 to close 90 percent, and 4,603 to close 99 percent. A source that moves at $10^{-3}$ per step leaves the target 0.999 behind, which is a lag of 999 steps, about $1/\tau$.

Exercise 2 asks you to write the update and the step count.

### Exploration with correlated noise

**Intuition.** A deterministic actor never tries anything new. Because DDPG is off-policy, you can add noise to the behavior without touching the learning rule. On a system with momentum, noise that changes sign at every step cancels out. Noise that drifts the same way for a few steps moves the system.

**Math.** The behavior policy adds a process $\mathcal{N}$ to the actor (Section 3, Equation 7):

$$
\mu'(s_t) = \mu(s_t \mid \theta^\mu_t) + \mathcal{N}
$$

The paper writes $\mu'$ for the exploration policy here and for the target actor elsewhere. This page uses $\mu'$ for the target actor only and calls the exploration policy the behavior policy. For $\mathcal{N}$, Section 7 uses an Ornstein-Uhlenbeck process with $\theta = 0.15$ and $\sigma = 0.2$, which it describes as the velocity of a Brownian particle with friction, centered at 0. The paper doesn't print a discretization. This page and the exercise use the Euler step

$$
x_{t+1} = x_t + \theta(\mu - x_t)\,dt + \sigma\sqrt{dt}\,\varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, I)
$$

with $dt = 1$ and $\mu = 0$. Here $\theta$ is the pull toward $\mu$, which is a different use of the symbol from the actor weights $\theta^\mu$. For this step, $x_{t+1} = (1 - \theta) x_t + \sigma \varepsilon_t$, so the lag-1 correlation is $1 - \theta = 0.85$ and the stationary standard deviation is $\sigma / \sqrt{1 - 0.85^2} \approx 0.380$.

**Tiny example, computed by `scripts/ou_noise.py`.** The theory gives a lag-1 correlation of 0.850 and a stationary standard deviation of 0.3797. A simulation of 4,000 paths measures 0.851 and 0.3801. Now average the noise over 20 consecutive steps. Independent noise with the same standard deviation gives a spread of 0.0851 for that average, and the OU noise gives 0.2535, about three times larger. The correlated noise holds a push in one direction long enough to change the state. With the noise term off and a start at 1, the value decays as $0.85, 0.7225, 0.6141, 0.5220$.

Exercise 3 asks you to write the noise process.

### Networks, batch normalization, and the results

**Intuition.** The recipe is short, and a few details keep it working: bounded actions, small initial outputs, and a critic that sees the action late.

**Math.** Section 7 gives the settings. The optimizer is Adam, with learning rates $10^{-4}$ for the actor and $10^{-3}$ for the critic. The critic has L2 weight decay of $10^{-2}$, $\gamma = 0.99$, and $\tau = 0.001$. Low-dimensional networks have two hidden layers of 400 and 300 ReLU units, and the actor ends in a tanh layer to bound the action. The action joins the critic at the second hidden layer. The last-layer weights start uniform in $[-3 \times 10^{-3}, 3 \times 10^{-3}]$, so initial policy and value outputs are near zero. The minibatch is 64 for low-dimensional inputs and 16 for pixels, and the buffer holds $10^6$ transitions.

Section 3 adds batch normalization on the state input and on all layers of $\mu$ and the layers of $Q$ before the action input. This lets one set of hyperparameters work across tasks whose observations have different units and ranges. For pixels, Section 4 repeats the action 3 times, stacks the 3 renderings into 9 feature maps, and downsamples frames to $64 \times 64$.

**Tiny example, computed by `scripts/network_size.py`.** The paper says the low-dimensional networks have about 130,000 parameters. With a 17-dimensional state and a 6-dimensional action, the actor has 129,306 and the critic 129,601. With a 3-dimensional state and a 1-dimensional action, each has about 122,000. The $400 \times 300$ middle layer holds 120,000 of those weights, and the action enters at that layer through $6 \times 300 = 1{,}800$ weights.

**Results.** Figure 2 compares four variants on a selection of tasks: DPG with batch normalization only, with target networks only, with both, and with both from pixels. The caption's verdict is that target networks are crucial. Section 4 adds that, to perform well across all tasks, both target networks and batch normalization are necessary. Figure 3 compares the critic's value estimates with observed returns. They are accurate on pendulum and cartpole and less accurate on harder tasks, where DDPG still learns competent policies. Table 1 reports average and best scores over 5 runs. On the best runs, several tasks score above 1, which means they beat the planner.

<!-- exercises -->

## Pitfalls

- **Soft updates apply to both targets.** The paper found that a target actor and a target critic were both needed (Section 3). Updating only the critic's target leaves the actor's contribution to $y_i$ moving at full speed.
- **Terminal states need a mask.** Algorithm 1 omits it. Without it, the critic bootstraps through the end of an episode. Time-limit truncation is a separate case that the paper doesn't cover.
- **The actor update must not change the critic.** In an autodiff framework, take the gradient of $-Q(s, \mu(s))$ with respect to the actor parameters only, and don't apply it to the critic.
- **Don't confuse the two gradients.** The critic's loss differentiates with respect to $\theta^Q$ and treats $y_i$ as a constant. The actor's objective differentiates $Q$ through the action. In DPG, $\nabla_\theta Q$ is the wrong quantity.
- **Noise can push an action out of range.** The actor's tanh bounds $\mu(s)$, but $\mu(s) + \mathcal{N}$ can exceed the bound. Clip the action before you send it to the environment. The paper doesn't discuss this.
- **Turn the noise off to evaluate.** The paper evaluates the policy periodically without exploration noise (Section 4).
- **Initialize the last layers small.** The paper uses small uniform last-layer weights so that early values and actions are near zero (Section 7).
- **The paper's pixel-network range $[3 \times 10^{-4}, 3 \times 10^{-4}]$ looks like a typo.** The interval as printed has zero width. Read it as $[-3 \times 10^{-4}, 3 \times 10^{-4}]$.
- **Convergence is not guaranteed.** The conclusion says non-linear approximators remove any convergence guarantee, and Figure 3 shows that the critic's estimates are less accurate on hard tasks. Later work addresses overestimation with twin critics (TD3, covered in the Deterministic policy gradients lesson). The paper doesn't.
- **Sample efficiency is still poor.** Section 6 says DDPG needs many training episodes, although fewer steps than DQN needs on Atari.

## Further reading

- [Deterministic policy gradient algorithms](http://proceedings.mlr.press/v32/silver14.pdf), Silver et al., ICML 2014. It proves the policy gradient that Equation 6 relies on.
- [Spinning Up: Deep Deterministic Policy Gradient](https://spinningup.openai.com/en/latest/algorithms/ddpg.html). It explains the algorithm and gives a short, readable reference implementation.
- The paper doesn't link an official implementation. A movie of learned policies is at the link in its footnote 1.
