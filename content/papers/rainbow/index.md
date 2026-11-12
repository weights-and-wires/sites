---
id: "rainbow"
title: "Rainbow: Combining Improvements in Deep Reinforcement Learning"
authors: "Hessel et al."
year: 2018
venue: "AAAI 2018"
arxiv: "1710.02298"
url: "https://arxiv.org/abs/1710.02298"
track: rl
tier: extended
est_hours: 6
status: full
summary: "This paper combines six independent improvements to DQN in one agent: double Q-learning, prioritized replay, dueling networks, multi-step returns, distributional value learning, and noisy networks. The pieces fit together through a multi-step distributional loss whose bootstrap action comes from the online network, and whose replay priority is the KL loss. One set of hyperparameters beats every earlier DQN variant on 57 Atari games, and an ablation study shows that prioritized replay and multi-step returns matter most."
related_course: "rl: deep value methods"
prereqs:
  - id: dqn
    reason: "Rainbow is a stack of changes to the DQN loss, replay memory, and network, so you need to know the baseline that each change modifies."
background: [mdp, bellman-equation, q-learning, epsilon-greedy, maximization-bias, n-step-returns, prioritized-replay, distributional-rl, kl-divergence]
course_links:
  - title: "Deep Q-networks"
    url: "https://rl.ifkash.dev/lessons/026-deep-q-networks"
  - title: "Double and dueling DQN"
    url: "https://rl.ifkash.dev/lessons/027-double-and-dueling-dqn"
  - title: "Prioritized experience replay"
    url: "https://rl.ifkash.dev/lessons/028-prioritized-replay"
  - title: "Distributional reinforcement learning"
    url: "https://rl.ifkash.dev/lessons/029-distributional-rl"
  - title: "Learning from pixels (noisy layers and Rainbow)"
    url: "https://rl.ifkash.dev/lessons/030-learning-from-pixels"
self_check:
  - q: 'Ten actions all have true value 0, and each value estimate carries independent noise with mean 0. Why is $\max_a \hat q(a)$ above 0 on average, and what does Double DQN change?'
    a: 'The maximum selects whichever estimate has the largest positive noise, so its average is above the true maximum (about 1.54 for ten unit-variance estimates). Double Q-learning picks the action with one set of estimates and reads its value from another, so the noise that wins the selection is not in the value that is read. The paper writes this as $q_{\bar\theta}(S_{t+1}, \arg\max_{a''} q_\theta(S_{t+1}, a''))$ (Section 3).'
    see: maximization-bias
  - q: 'A window holds the rewards $1, 0, 2$ and then the target network gives $\max_{a''} q = 5$. With $\gamma = 0.9$, what is the 3-step target, and what changes if the episode ended on the reward 2?'
    a: 'The return is $1 + 0.9 \cdot 0 + 0.81 \cdot 2 = 2.62$ and the bootstrap term is $0.9^3 \cdot 5 = 3.645$, so the target is $6.265$. If the episode ended on the reward 2, the discount $\gamma_t^{(n)}$ is 0 and the target is $2.62$ (Section 3, Equation 2).'
    see: n-step-returns
  - q: 'Why does Rainbow prioritize transitions by the KL loss and not by the absolute TD error, and what does the importance weight $(N P(i))^{-\beta}$ correct?'
    a: 'The KL loss is what the distributional agent minimizes, and the paper suggests it is more robust in stochastic environments because it can keep falling even when the returns are not deterministic (Section 4). Prioritized sampling draws some transitions more often than others, which biases the gradient. The importance weight scales each sampled loss down by how over-sampled the transition is, and $\beta$ rises from 0.4 to 1 so the correction is complete at the end of training.'
    see: prioritized-replay
---

## The problem

DQN made deep Q-learning work on Atari, and between 2015 and 2017 researchers published a series of fixes for its weaknesses. Each fix addressed a different weakness, and each was evaluated alone against plain DQN. The paper states the open question in its abstract: it is unclear which of the extensions are complementary and can be combined (Section 1). Two combinations existed (prioritized DDQN and dueling DDQN), and nobody had tried all of them at once.

The weaknesses are concrete. Here are two that you can run.

- **Overestimation.** The DQN target takes a maximum over noisy estimates. Suppose ten actions all have true value 0, and each estimate has unit Gaussian noise. `scripts/overestimation.py` shows that the average of $\max_a \hat q(a)$ is 1.54, although the true maximum is 0. The target is biased upward before any learning rule is involved.
- **Slow reward propagation.** A one-step target moves a reward one state back per update sweep. In a chain of 12 states with a reward at the end, `scripts/chain_propagation.py` shows that the first state learns of the reward after 12 sweeps with $n = 1$, and after 4 sweeps with $n = 3$.

Other weaknesses are listed in Section 3: uniform replay wastes samples on transitions with little left to learn, a scalar Q-value hides the shape of the return, and $\epsilon$-greedy exploration struggles when many actions must be right in a row. Rainbow's answer is one agent that carries all six remedies, with one set of hyperparameters for all 57 games (Section 5).

Here is the headline. The paper reports a median human-normalized score of 223% with no-op starts, against 79% for DQN and 164% for the best single extension in its table, distributional DQN (Section 6, Table 2). Rainbow matches DQN's final performance after 7 million frames (Figure 1).

## Key ideas

The notation follows the paper: $R_{t+1}$ is a reward, $\gamma_t$ is the discount (zero at termination), $q_\theta$ is the online network, and $q_{\bar\theta}$ is the target network, a periodic copy that receives no gradient (Section 2). Two differences are specific to this page. The exercises use a constant $\gamma$ and a `done` flag in place of the paper's $\gamma_t$. The paper's distributions $d_t = (\bm z, \bm p_\theta(S_t, A_t))$ appear here as an array `p` of probabilities on the atoms `z`.

### Double Q-learning

**Intuition.** A max over noisy estimates picks the estimate with the best luck. If one set of values chooses the action and another set evaluates it, the luck of the choice does not carry over to the value.

**Math.** DQN minimizes $(R_{t+1} + \gamma_{t+1} \max_{a'} q_{\bar\theta}(S_{t+1}, a') - q_\theta(S_t, A_t))^2$ (Section 2, Equation 1). Double DQN replaces the max by

$$
\big(R_{t+1} + \gamma_{t+1}\, q_{\bar\theta}\big(S_{t+1}, \arg\max_{a'} q_\theta(S_{t+1}, a')\big) - q_\theta(S_t, A_t)\big)^2
$$

The online network $\theta$ selects the action and the target network $\bar\theta$ evaluates it (Section 3).

**Tiny example, computed by `scripts/overestimation.py`.** With ten actions of true value 0 and unit noise, selecting and evaluating with the same estimates gives an average of 1.54. Selecting with one independent estimate and reading from another gives an average of $-0.00$ (that is, 0 up to sampling error).

### Prioritized replay

**Intuition.** Some transitions teach more than others. A transition that the network predicts badly has more to teach, so the agent replays it more often.

**Math.** Proportional prioritized replay samples transition $t$ with probability $p_t \propto |\delta_t|^{\omega}$, where $\delta_t$ is the last absolute TD error and $\omega$ shapes the distribution. New transitions enter with maximum priority (Section 3). Rainbow keeps this form and changes the priority to the KL loss, covered below. Sampling non-uniformly biases the gradient, and prioritized replay corrects it with importance weights $w_i = (N P(i))^{-\beta} / \max_j w_j$. Rainbow uses $\omega = 0.5$ and raises $\beta$ linearly from 0.4 to 1.0 over training (Section 5, Table 1). The paper cites the importance-sampling scheme without restating the formula, so the weight formula here comes from the prioritized replay paper (Schaul et al.) and the lesson linked above.

**Tiny example, computed by `scripts/dueling_priorities.py`.** Four transitions have priorities $[0.04, 0.25, 1.0, 0.16]$. With $\omega = 0.5$ the sampling probabilities are $[0.0952, 0.2381, 0.4762, 0.1905]$. The weights, normalized so the largest is 1, are $[1, 0.4, 0.2, 0.5]$ at $\beta = 1$, and $[1, 0.693, 0.525, 0.758]$ at $\beta = 0.4$. The most-sampled transition gets the smallest weight, and a lower $\beta$ compresses the weights toward 1.

### Dueling networks

**Intuition.** In many states the choice of action matters little. A network with a separate stream for the value of the state learns that value from every transition, whatever action was taken.

**Math.** A shared encoder $f_\xi$ feeds a value stream $v_\eta$ and an advantage stream $a_\psi$. They merge as (Section 3)

$$
q_\theta(s, a) = v_\eta(f_\xi(s)) + a_\psi(f_\xi(s), a) - \frac{\sum_{a'} a_\psi(f_\xi(s), a')}{N_{\text{actions}}}
$$

The subtracted mean makes the average of the Q-values over actions equal to $v$, so the split into value and advantage is identifiable.

**Tiny example, computed by `scripts/dueling_priorities.py`.** Take $v = 1$ and advantages $[0.5, -0.5, 3.0]$. The mean advantage is 1, so $q = [0.5, -0.5, 3.0]$. The average of $q$ is 1, which equals $v$, and the best action is the same as under the advantages.

### Multi-step returns

**Intuition.** A one-step target waits for the bootstrap to carry information back one step at a time. An $n$-step target reads $n$ real rewards before it bootstraps, so a reward reaches earlier states faster. The price is more variance, and bias when the data come from an older policy.

**Math.** The truncated $n$-step return is (Section 3, Equation 2)

$$
R_t^{(n)} \equiv \sum_{k=0}^{n-1} \gamma_t^{(k)} R_{t+k+1}, \qquad \gamma_t^{(k)} = \prod_{i=1}^{k} \gamma_{t+i}
$$

and the loss is $(R_t^{(n)} + \gamma_t^{(n)} \max_{a'} q_{\bar\theta}(S_{t+n}, a') - q_\theta(S_t, A_t))^2$. Rainbow compared $n = 1, 3, 5$ and chose $n = 3$: both 3 and 5 did well early, and 3 was best by the end (Section 5).

**Tiny example, computed by `scripts/nstep_example.py`.** With $\gamma = 0.9$, rewards $[1, 0, 2]$, and $\max_{a'} q = 5$, the return is 2.62, the discount is $0.9^3 = 0.729$, and the 3-step target is $2.62 + 0.729 \cdot 5 = 6.265$. The one-step target with the same $q$ is 5.5. If the episode ends after rewards $[1, 2]$, the target is 2.8 with no bootstrap term. Exercise 1 asks you to write this target.

### Distributional learning with the C51 projection

**Intuition.** Instead of one number per action, the network predicts probabilities on a fixed grid of possible returns. The Bellman update shifts and shrinks that distribution, and the result no longer sits on the grid, so it must be projected back.

**Math.** The support is $z^i = v_{\min} + (i-1)\frac{v_{\max} - v_{\min}}{N_{\text{atoms}} - 1}$ for $i = 1, \dots, N_{\text{atoms}}$, and the network outputs probabilities $p^i_\theta(S_t, A_t)$ on the atoms (Section 3). The target is the next-state distribution contracted by the discount and shifted by the reward, $d'_t = (R_{t+1} + \gamma_{t+1} \bm z, \bm p_{\bar\theta}(S_{t+1}, \bar a^*_{t+1}))$. The loss is the KL divergence from the projected target to the prediction (Section 3, Equation 3):

$$
D_{\text{KL}}(\Phi_{\bm z}\, d'_t \,\|\, d_t)
$$

Here $\Phi_{\bm z}$ is the projection onto the fixed support. Each shifted atom $\mathcal{T}z_j$ is clipped to $[v_{\min}, v_{\max}]$ and its probability splits between the two neighbouring atoms in proportion to proximity. The greedy action $\bar a^*_{t+1}$ uses the mean $\bm z^\top \bm p(S_{t+1}, a)$. Rainbow uses 51 atoms on $[-10, 10]$ (Table 1), which gives a spacing of 0.4 (`scripts/hyperparameters.py`).

**Tiny example, computed by `scripts/c51_projection.py`.** Use five atoms $z = [-2, -1, 0, 1, 2]$ and $p = [0.1, 0.2, 0.4, 0.2, 0.1]$. With $r = 0.5$ and a discount of 0.9, the shifted atoms are $[-1.3, -0.4, 0.5, 1.4, 2.3]$, and the last one clips to 2. The projected probabilities are $[0.03, 0.15, 0.32, 0.32, 0.18]$ and they sum to 1. The mean moves from 0 to 0.47, not to the exact 0.5, because clipping the top atom discards 0.03 of the mean. For a terminal transition the discount is 0 and every atom collapses onto the reward. Exercise 2 asks you to write the projection.

### Noisy networks

**Intuition.** $\epsilon$-greedy adds the same randomness in every state. A layer with learned noise can shrink its noise where the agent is confident and keep it where the agent is not.

**Math.** A noisy linear layer adds a second, noisy stream to the usual one (Section 3, Equation 4):

$$
\bm y = (\bm b + \mathbf{W}\bm x) + (\bm b_{\text{noisy}} \odot \epsilon^b + (\mathbf{W}_{\text{noisy}} \odot \epsilon^w)\bm x)
$$

The noise $\epsilon^b$ and $\epsilon^w$ are random variables, and the network can learn to ignore the noisy stream at different rates in different parts of the state space (Section 3). Rainbow replaces all linear layers with noisy ones, uses factorised Gaussian noise to cut the number of noise variables, initializes the noisy stream with $\sigma_0 = 0.5$, and then acts fully greedily with $\epsilon = 0$ (Section 4 and Section 5).

**Tiny example, computed by `scripts/noisy_example.py`.** Take $x = [1, 2]$, $W = [0.5, -0.5]$, $b = 0.1$, a noisy stream initialised to 0.5, and fixed noise. The deterministic stream gives $-0.4$ and the noisy stream gives $-2.0$, so $y = -2.4$. Factorised noise uses one value per input and per output and builds the weight noise as an outer product $f(\epsilon_{out}) f(\epsilon_{in})^\top$ with $f(x) = \operatorname{sgn}(x)\sqrt{|x|}$. A $512 \times 512$ layer then needs 1,024 noise values and not 262,656. The exact form of $f$ comes from the Noisy Nets paper, not from Rainbow, which only names the factorised noise.

### The combined Rainbow target

**Intuition.** The six pieces touch different parts of the agent, so they mostly plug in without conflict. Three of them meet at the loss: multi-step returns set the shift and the discount, the distributional loss sets the shape, and double Q-learning sets the bootstrap action.

**Math.** Rainbow builds the target distribution $d_t^{(n)} = (R_t^{(n)} + \gamma_t^{(n)} \bm z,\; \bm p_{\bar\theta}(S_{t+n}, a^*_{t+n}))$ and minimizes (Section 4)

$$
D_{\text{KL}}(\Phi_{\bm z}\, d_t^{(n)} \,\|\, d_t)
$$

The bootstrap action $a^*_{t+n}$ is the greedy action under the **online** network's mean values, and the **target** network supplies the distribution at that action. The replay priority is the same loss, $p_t \propto (D_{\text{KL}}(\Phi_{\bm z}\, d_t^{(n)} \| d_t))^{\omega}$. The paper argues this might be more robust to noisy, stochastic environments, because the loss can continue to decrease even when the returns are not deterministic (Section 4). The network is a dueling architecture adapted to distributions. A value stream with $N_{\text{atoms}}$ outputs and an advantage stream with $N_{\text{atoms}} \times N_{\text{actions}}$ outputs merge per atom, and a softmax over the atoms of each action gives the probabilities (Section 4):

$$
p^i_\theta(s, a) = \frac{\exp\big(v^i_\eta(\phi) + a^i_\psi(\phi, a) - \bar a^i_\psi(s)\big)}{\sum_j \exp\big(v^j_\eta(\phi) + a^j_\psi(\phi, a) - \bar a^j_\psi(s)\big)}
$$

where $\phi = f_\xi(s)$ and $\bar a^i_\psi(s)$ is the mean of the advantages over actions for atom $i$.

**Tiny example, computed by `scripts/rainbow_target.py` and `scripts/dueling_priorities.py`.** Use the five-atom support above, $\gamma = 0.9$, and a 3-step window with rewards $[1, 0, 0]$, so $R = 1$ and the discount is $0.729$. Two actions have next-state distributions from the online network with means $[0.45, -0.25]$ and from the target network with means $[-0.6, 0.75]$. The online network picks action 0. The target network's distribution at action 0 projects to $[0, 0.1374, 0.3813, 0.3084, 0.1729]$. If the target network picked its own action (action 1, as plain C51 does), the target would be $[0, 0.0229, 0.1, 0.3219, 0.5552]$, which is a very different shape. Against a prediction $[0.10, 0.20, 0.30, 0.25, 0.15]$, the KL loss is 0.1292 and the priority with $\omega = 0.5$ is 0.3594. For the dueling head, the logits for atoms of $v = [0, 1, 0]$ with advantages $[[1, 0, -1], [0, 2, 0]]$ give action probabilities $[0.5065, 0.3072, 0.1863]$ and $[0.0629, 0.7662, 0.1710]$. Exercise 3 asks you to write the target and the priority, and Exercise 4 asks for the dueling head and the replay weights.

### What the ablation shows

**Intuition.** A combination can hide which part does the work, so the paper removes one component at a time from the full agent and measures the loss.

**Result.** All numbers in this section come from the paper's text and are not computed by a script here. Figure 3 plots the median human-normalized score for Rainbow and six ablations, and Figure 4 shows the drop per game (Section 6):

- Prioritized replay and multi-step learning are the two most important components. Removing either one causes a large drop in median performance, and removing multi-step learning hurts final performance as well as early performance. Full Rainbow beats either ablation in 53 of 57 games.
- Distributional learning ranks next. For the first 40 million frames the ablation matches the full agent, and then it falls behind, mostly on games where the agent is at or above human level.
- Noisy networks help the median. Removing them gives a large drop on several games and a small gain on others.
- Dueling makes no significant difference to the median, although its effect differs by game.
- Double Q-learning also changes the median little, and it helps or hurts depending on the game. The paper's hypothesis is that the clipped support $[-10, 10]$ already counteracts overestimation, because real returns often exceed 10 and the distributions therefore underestimate. It adds that double Q-learning may matter more if the support grows.

The paper's reading is that the ranking applies to this particular combination. It does not rank the components in isolation.

**Tiny example, computed by `scripts/hyperparameters.py`.** With rewards clipped to $[-1, 1]$ and $\gamma = 0.99$, the largest discounted return is $1/(1 - \gamma) = 100$, ten times the edge of the support at 10. This is the setting behind the paper's observation that returns can fall outside the support.

<!-- exercises -->

## Pitfalls

- **Letting the target network pick the action.** The Rainbow target selects the action with the online network and reads the distribution from the target network. Using the target network for both is plain C51 and brings the overestimation back. Exercise 3 tests this with a case in which the two networks disagree.
- **Discounting the bootstrap by $\gamma$ and not $\gamma^n$.** The distribution at $S_{t+n}$ is $n$ steps away. A window cut short by the end of an episode has fewer than $n$ rewards, and a terminal window has no bootstrap at all.
- **Forgetting to clip before projecting.** Shifted atoms can leave $[v_{\min}, v_{\max}]$. Without the clip, the projection indexes outside the array or silently loses probability. The page example shows the mean drifting by 0.03 for this reason.
- **Computing the KL in the wrong direction.** The loss is $D_{\text{KL}}(\text{target} \,\|\, \text{prediction})$. The target distribution is fixed data, so the loss equals the cross-entropy $-\sum_i m_i \log p_i$ minus the target's entropy, and the gradient is the same.
- **Dropping the importance weights.** Priorities change the sampling distribution. Without the weights the gradient is biased, and $\beta$ must reach 1 by the end of training.
- **Assigning new transitions the wrong priority.** New transitions enter with maximum priority (Section 3). Without that rule, a new transition has no TD error yet and might never be sampled.
- **Adding $\epsilon$-greedy on top of noisy layers.** Rainbow sets $\epsilon = 0$ whenever Noisy Nets are used (Section 5). The ablation without Noisy Nets uses $\epsilon$-greedy annealed to 0.01 over the first 250,000 frames.
- **Averaging over the wrong axis in the dueling head.** The mean of the advantages runs over actions, for each atom. A mean over atoms gives a valid-looking softmax with the wrong logits.
- **Reading the ablation as a ranking of the components in general.** It removes one component from a specific combination. Double Q-learning looks unimportant here partly because the clipped support already limits overestimation.
- **Comparing wall-clock time.** The paper compares algorithms frame by frame and runs everything on a single GPU. A full 200-million-frame run takes about 10 days, and the variants differ by less than 20% (Section 6). Parallel setups are outside its scope.

## Further reading

- [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887), by Bellemare, Dabney, and Munos, introduces C51 and the projection that this page implements.
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952), by Schaul et al., defines the priorities and importance weights.
- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461), by van Hasselt, Guez, and Silver, and [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581), by Wang et al., cover the other two DQN fixes.
- [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295), by Fortunato et al., defines the noisy layer and its factorised noise.
- [Dopamine](https://github.com/google/dopamine), a research framework from Google, includes a Rainbow agent.
