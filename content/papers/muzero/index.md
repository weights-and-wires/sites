---
id: "muzero"
title: "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model"
authors: "Schrittwieser et al."
year: 2019
venue: "Nature 2020"
arxiv: "1911.08265"
url: "https://arxiv.org/abs/1911.08265"
track: rl
tier: extended
est_hours: 6
status: full
summary: "MuZero runs the AlphaZero kind of tree search without being told the rules of the game. It learns three functions, one that encodes an observation into a hidden state, one that moves the hidden state forward by an action and predicts the reward, and one that predicts a policy and a value, and it trains them only to get reward, value, and policy right. The same algorithm matched AlphaZero in Go, chess, and shogi and set a state of the art on 57 Atari games."
related_course: "rl: model-based RL"
prereqs:
  - id: alphazero
    reason: "You need AlphaZero's single network with a policy head and a value head, trained on search visit counts and game outcomes, because MuZero keeps that search and training signal and replaces only the rule-based simulator."
background: [model-based-rl, latent-dynamics]
course_links:
  - title: "World models and MuZero"
    url: "https://rl.ifkash.dev/lessons/044-world-models-and-muzero"
  - title: "Learning a model and planning with it"
    url: "https://rl.ifkash.dev/lessons/043-model-based-rl"
  - title: "n-step returns"
    url: "https://rl.ifkash.dev/lessons/019-n-step-returns"
  - title: "Monte Carlo tree search"
    url: "https://rl.ifkash.dev/lessons/022-monte-carlo-tree-search"
  - title: "AlphaZero: learning from self-play"
    url: "https://rl.ifkash.dev/lessons/062-alphazero-self-play"
self_check:
  - q: 'The pUCT rule of Equation 2 multiplies the bonus by $c_1 + \log\frac{\sum_b N(s,b) + c_2 + 1}{c_2}$, with $c_1 = 1.25$ and $c_2 = 19652$. How large is that factor at $\sum_b N = 0$ and at $\sum_b N = 800$, and what does that say about the log term at the search budgets of the paper?'
    a: 'The factor is 1.2501 at 0 visits and 1.2900 at 800 visits (computed by `scripts/pucb_example.py`). At 800 simulations the log term adds only 0.04 to $c_1$, so for board games it is almost a constant. It starts to matter only when a node has tens of thousands of visits, where it grows the exploration weight slowly, to 3.06 at 100,000 visits.'
    see: alphazero
  - q: 'AlphaZero masks illegal moves at every node and stops the search at terminal states. What does MuZero do instead, and why can it get away with it?'
    a: 'MuZero masks illegal moves only at the root, where it can ask the environment, and it never stops at a terminal node: it uses the network value everywhere (Appendix A). Inside the tree, the learned model has to predict actions that never occur in the data as unlikely, and it learns this because terminal states are treated as absorbing states during training and illegal actions never appear in the training trajectories.'
    see: model-based-rl
  - q: 'The Atari value target of 100 goes through $h$ and then the support of integers from -300 to 300. Which two supports get weight, and how much?'
    a: 'First $h(100) = 9.1499$ (computed by `scripts/transform_example.py`). The target then puts weight 0.8501 on support 9 and 0.1499 on support 10, because the encoding splits a number between its two neighbouring integers (Appendix F). Decoding takes the expected support value, 9.1499, and then inverts $h$ to get back 100. The transform is why a support of 601 integers can represent raw values up to about 58,700.'
    see: latent-dynamics
---

## The problem

The AlphaZero page ends with a search that needs two things from the outside world. The first is a network that scores positions. The second is a simulator that knows the rules. The search calls the simulator to move from a position to the next one, to list the legal moves at each node, and to detect that a game has ended. Appendix A of the paper names these three uses of the rules: state transitions, available actions, and episode termination.

Most problems don't ship with a perfect simulator. A robot, a recommender system, and a video game with no emulator access all lack one. Atari is the canonical case: the paper notes that model-based methods that predicted pixels, or that learned latent states, trailed well-tuned model-free methods there, even in data efficiency (Section 2). Predicting every pixel of the next frame spends model capacity on details that don't matter for winning. Model-free methods such as R2D2 do well on Atari, and they are far behind on games like chess and Go that need careful lookahead (Section 1).

The question is whether one algorithm can do both. The abstract's answer is that MuZero, with no knowledge of the dynamics, matched AlphaZero in Go, chess, and shogi and set a state of the art on 57 Atari games. In Atari, Table 1 reports a median human-normalized score of 2041.1% for MuZero against 1920.6% for R2D2, with 20.0 billion frames against 37.5 billion, in 12 hours against 5 days (the ratios are computed by `scripts/paper_numbers.py`).

The paper isn't a proof of general-purpose planning. The dynamics function is deterministic, and the paper leaves stochastic transitions to future work (Section 3). Imperfect-information games such as poker are outside its scope (footnote 2).

## Key ideas

The notation follows the paper: observations $o_t$, actions $a_t$, rewards $u_t$, a hidden state $s^k$ at unroll step $k$, and the three functions $h_\theta$, $g_\theta$, and $f_\theta$. The paper writes the hidden states as $s^k$ in Figure 1 and as $s_k$ in the text. The exercises use `obs`, `actions`, `u`, `z`, and `pi`. Every number in a worked example comes from a script in `scripts/`. Run any script from the paper folder to reproduce its output. The scripts need Python 3 and NumPy only. The scripts load the exercise solutions, so the page and the exercises share one implementation. Numbers quoted from the paper, such as Atari scores and training settings, don't come from a script, except for arithmetic on them in `scripts/paper_numbers.py`.

The table summarizes the changes against AlphaZero. The sections after it take them one at a time.

| | AlphaZero | MuZero |
|---|---|---|
| State in the tree | The true game position, from a simulator | A learned hidden state $s^k$ |
| Transition | Rules of the game | $r^k, s^k = g_\theta(s^{k-1}, a^k)$ |
| Legal moves | Mask at every node | Mask at the root only |
| Terminal nodes | Stop and use the true outcome | No special case, the network value is used |
| Rewards and discount | Terminal $\pm 1$, no discount | Any reward at any step, discount $\gamma$ |
| Value target | The game outcome | $n$-step return that bootstraps from the search value |
| Value scale in the search | Bounded in $[0, 1]$ | Min-max normalized $Q$ |
| Training | One position per sample | Sequence unrolled for $K = 5$ steps |

### Three functions in place of a simulator

**Intuition.** MuZero keeps the structure of a model, a state and a transition, and drops the requirement that the state mean anything. The hidden state is whatever the network needs to predict what the search needs: the reward for the move, the value of the position, and a good policy. The paper stresses that there is no constraint that the state reconstructs the observation, matches the true state, or has any particular semantics (Section 1). The model can invent its own rules, as long as planning with them works.

**Math.** At time step $t$, the model $\mu_\theta$ is made of three functions (Section 3, Figure 1):

$$
s^0 = h_\theta(o_1, \dots, o_t), \qquad r^k, s^k = g_\theta(s^{k-1}, a^k), \qquad p^k, v^k = f_\theta(s^k)
$$

The representation $h$ encodes the observation history into the root hidden state. The dynamics $g$ takes the previous hidden state and a hypothetical action and returns an immediate reward and the next hidden state. The prediction $f$ is AlphaZero's joint policy and value network, applied to a hidden state. The three outputs have targets (Section 3):

- $p^k_t \approx \pi(a_{t+k+1} \mid o_1, \dots, o_t, a_{t+1}, \dots, a_{t+k})$, the search policy.
- $v^k_t \approx \mathbb{E}[u_{t+k+1} + \gamma u_{t+k+2} + \dots \mid \dots]$, the discounted return.
- $r^k_t \approx u_{t+k}$, the observed reward.

The search makes at most one call of $g$ and one call of $f$ per simulation, so the cost per simulation is of the same order as AlphaZero's (Appendix B). This is the idea of a **value-equivalent model** from the predictron and value prediction networks that the paper cites (Section 2): the abstract model matches the real one in the quantities that planning needs, and in nothing else.

**Tiny example, computed by `scripts/unroll_example.py`.** The script builds a model with a 2-dimensional hidden state and two actions. All three functions are linear, and $f$ reads the policy logits from $s$ with a matrix and the value with a vector. It starts from $s^0 = [1, 2]$ and applies the actions $a^1 = 0$ and $a^2 = 1$. The dynamics function returns $r^1 = 1$ and $s^1 = [2, 2]$, then $r^2 = 2$ and $s^2 = [2, 4]$. The values are $v^0, v^1, v^2 = 3, 4, 6$, and the first policy is the softmax of the logits $[1, 2]$, which is $[0.2689, 0.7311]$. Exercise 4 asks you to write this unroll.

### Training through an unrolled model

**Intuition.** A model is only useful for planning if it stays accurate when you apply it several times in a row. So training doesn't score a single prediction. It samples a stretch of a stored game, encodes the first observation, feeds the real actions that followed into $g$ one after another, and scores the three outputs at every step against what the game actually produced (Figure 1C, Appendix G). Errors that compound over steps then show up in the loss, and gradients flow back through all the steps ("backpropagation-through-time").

**Math.** For a position at time $t$, the loss sums reward, value, and policy errors over the unroll (Section 3, Equation 1):

$$
l_t(\theta) = \sum_{k=0}^{K} \Bigl[\, l^r(u_{t+k}, r^k_t) + l^v(z_{t+k}, v^k_t) + l^p(\pi_{t+k}, p^k_t) \Bigr] + c\,\lVert\theta\rVert^2
$$

The targets are as follows. The policy target $\pi_{t+k}$ is the visit-count distribution that the search produced at step $t+k$ of the real game, as in AlphaZero. The reward target is the observed reward. The value target $z_{t+k}$ is described in the next section. The paper trains with $K = 5$ unrolled steps in all its experiments (Section 4, Appendix G).

Three details come from Appendix G and Section 3:

- *Which losses.* Board games use squared error for value, omit the reward loss because there are no intermediate rewards, and use cross-entropy for the policy. Atari uses cross-entropy over a categorical support for value and reward (see the transform section). The paper found that cross-entropy was more stable than squared error when values and rewards have a variable scale (footnote 1).
- *Gradient scaling.* Each head's loss is multiplied by $1/K$. The gradient at the start of the dynamics function is also scaled by $1/2$, so the total gradient into the dynamics function stays constant as the unroll grows.
- *Hidden-state scaling.* The hidden state is scaled to $[0, 1]$ with min-max scaling, $s_{\text{scaled}} = (s - \min s)/(\max s - \min s)$, to match the range of the action input.

Equation 1 sums all three losses from $k = 0$. The model predicts no reward for the root, because $r^k$ comes from $g$. This page and Exercise 4 skip the reward term at $k = 0$.

**Tiny example, computed by `scripts/unroll_example.py`.** Continue the model above with $K = 2$, the search policies $\pi_0 = [1, 0]$, $\pi_1 = [0, 1]$, $\pi_2 = [0.5, 0.5]$, value targets $z = [3, 5, 7]$, and reward targets $u_1 = 0$ and $u_2 = 1$. The terms are:

| $k$ | Reward $(u_k - r_k)^2$ | Value $(z_k - v_k)^2$ | Policy $-\pi_k^\top \log p_k$ |
|---|---|---|---|
| 0 | none | 0 | 1.3133 |
| 1 | 1 | 1 | 0.6931 |
| 2 | 1 | 1 | 1.1269 |

The sum is 7.1333, and $7.1333 / K$ is 3.5667. The model predicts a reward of 1 at step 1, and the target is 0, which is the whole step-1 reward error. Exercise 4 asks you to write this loss.

### n-step value targets with bootstrapping

**Intuition.** In board games, the value target is the final outcome, which is the same for every position of a game. Atari episodes last up to 30 minutes, or 108,000 frames, and rewards arrive at every step. Waiting for the end of the episode is too slow and has high variance. MuZero looks $n$ steps ahead, sums the discounted rewards it saw, and replaces the rest of the episode with the search's value estimate from that point (Section 3). It is the $n$-step return of the Sutton and Barto book, with the search value $\nu$ as the bootstrap.

**Math.** The value target is (Section 3, unnumbered):

$$
z_t = u_{t+1} + \gamma u_{t+2} + \dots + \gamma^{n-1} u_{t+n} + \gamma^n \nu_{t+n}
$$

Here $\nu_{t+n}$ is the value that the search estimated at the real state $n$ steps later. Final outcomes in board games are rewards $u_t \in \{-1, 0, +1\}$ at the final step, with $\gamma = 1$ and bootstrapping to the end of the game. Atari uses $n = 10$ and $\gamma = 0.997$ (Appendices C and G). MuZero Reanalyze uses $n = 5$ and a target network $f_{\theta^-}$ for $\nu^-$, so the bootstrap value is more recent and stable (Appendix H).

Two conventions differ from the paper. The page indexes rewards from 0, so that `u[t]` is the paper's $u_{t+1}$. At the end of an episode, the exercise adds no bootstrap term, which matches the paper's treatment of terminal states as absorbing with a constant value (Appendix A).

**Tiny example, computed by `scripts/nstep_example.py`.** Take rewards $u = [1, 0, 2, 0, 1]$ and values $\nu = [5, 4, 3, 2, 1]$, with $n = 2$ and $\gamma = 0.5$. For $t = 1$ the target is $0 + 0.5 \cdot 2 + 0.25 \cdot \nu_3 = 1.5$. The full list is $z = [1.75, 1.5, 2.25, 0.5, 1.0]$. The last two entries have no bootstrap term, because the window of $n$ steps reaches the end of the episode. With the Atari discount, the weight of the bootstrap term is $0.997^{10} = 0.9704$, so almost all of the target at $n = 10$ comes from the learned value, and the reward sum fixes only the first ten steps. Exercise 2 asks you to write these targets.

### pUCT with min-max normalized values

**Intuition.** The search must pick which action to simulate at each node. AlphaZero's rule adds an exploration bonus to $Q$, and that works because $Q$ lies in $[0, 1]$, the same scale as the prior. In Atari, $Q$ can be thousands. A bonus on the scale of a probability would then be invisible, and the search would never explore. The fix doesn't need to know the game's score range. MuZero rescales $Q$ with the smallest and largest $Q$ seen so far anywhere in the tree (Appendix B).

**Math.** Every edge $(s, a)$ stores $\{N, Q, P, R, S\}$: the visit count, mean value, prior, reward, and next state. A simulation starts at the root and, at each step $k$, picks (Appendix B, Equation 2):

$$
a^k = \arg\max_a \left[\, \bar Q(s, a) + P(s, a)\,\frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)} \left( c_1 + \log\frac{\sum_b N(s, b) + c_2 + 1}{c_2} \right) \right]
$$

with $c_1 = 1.25$ and $c_2 = 19652$. The paper writes the normalized value $\bar Q$ as $Q$ in Equation 2 and defines it in Equation 5:

$$
\bar Q(s_{k-1}, a_k) = \frac{Q(s_{k-1}, a_k) - \min_{s,a \in \text{Tree}} Q(s, a)}{\max_{s,a \in \text{Tree}} Q(s, a) - \min_{s,a \in \text{Tree}} Q(s, a)}
$$

The square-root term is AlphaGo's PUCT bonus with $c_{\text{puct}}$ replaced by the bracket. The log term makes the exploration weight grow slowly with the number of visits. The paper doesn't say what to return when the maximum equals the minimum. This page and Exercise 1 return 0, which is a choice.

**Tiny example, computed by `scripts/pucb_example.py`.** One node has $Q = [2, 5, 0]$, $N = [10, 2, 0]$, and $P = [0.6, 0.3, 0.1]$, and the tree range is $[0, 10]$. The normalized values are $[0.2, 0.5, 0]$. With $\sum_b N = 12$, the factor is $1.25 + \log\frac{12 + 19653}{19652} = 1.2507$ and $\sqrt{12} = 3.4641$. The bonuses are $[0.2363, 0.4332, 0.4332]$, so the scores are $[0.4363, 0.9332, 0.4332]$ and the search picks edge 1. The script also tabulates the factor:

| $\sum_b N$ | 0 | 50 | 800 | 19,652 | 100,000 |
|---|---|---|---|---|---|
| $c_1 + \log(\dots)$ | 1.2501 | 1.2526 | 1.2900 | 1.9432 | 3.0564 |

At the 800 simulations of the board-game experiments, the log term moves the factor from 1.25 to 1.29, so $c_2 = 19652$ matters only for searches far larger than the paper's. When the same $Q$ values are on a reward scale of a thousand times larger, normalized scores don't change. Without normalization, edge 2 scores 0.4 against 2000 for edge 0, and no bonus can compete. Exercise 1 asks you to write this rule.

### Backup with rewards and a discount

**Intuition.** AlphaZero's backup adds a single outcome to every edge on the path. With intermediate rewards and a discount $\gamma$, edges deeper in the path are worth less, and the rewards seen along the way matter. Each edge on the path gets the return that starts at that edge: the rewards from there to the leaf, plus the leaf value discounted by the remaining steps.

**Math.** Suppose a simulation reaches a leaf $s^l$ through edges $a^1, \dots, a^l$, and the prediction function returns $v^l$ for the leaf. For each edge $k = l, \dots, 1$ the search forms an $(l - k)$-step return (Appendix B, Equation 3):

$$
G^k = \sum_{\tau=0}^{l-1-k} \gamma^\tau r_{k+1+\tau} + \gamma^{\,l-k} v^l
$$

and updates the running mean and visit count of that edge (Equation 4):

$$
Q(s_{k-1}, a_k) := \frac{N(s_{k-1}, a_k)\, Q(s_{k-1}, a_k) + G^k}{N(s_{k-1}, a_k) + 1}, \qquad N(s_{k-1}, a_k) := N(s_{k-1}, a_k) + 1
$$

The tree-wide minimum and maximum in Equation 5 are taken over these $Q$ values. The expansion step stores $r_l$ and $s^l$ from the dynamics function in the edge's tables, so later simulations look them up instead of calling $g$ again.

The sum in Equation 3 starts at $r_{k+1}$, so as printed, $G^k$ doesn't include the reward $r_k$ on the edge itself. This page follows the printed form. The official pseudocode isn't in the PDF, and this page doesn't check how it places that reward.

**Tiny example, computed by `scripts/backup_example.py`.** Take $\gamma = 0.9$, rewards $r_1 = 1$, $r_2 = 0$, $r_3 = 2$ along a path of three edges, and a leaf value $v^3 = 5$. Then $G^3 = 5$, $G^2 = r_3 + 0.9 \cdot 5 = 6.5$, and $G^1 = r_2 + 0.9\, r_3 + 0.81 \cdot 5 = 5.85$. The edges start with $(N, Q) = (4, 3)$, $(1, 6)$, and $(0, 0)$. After the update they hold $(5, 3.57)$, $(2, 6.25)$, and $(1, 5)$. The script also checks that $G^k = r_{k+1} + \gamma\, G^{k+1}$, which is how an implementation computes it in one backward pass.

### Value and reward transforms for Atari

**Intuition.** Atari scores range from a few points to hundreds of thousands, and different games use different scales. A network that regresses raw values has gradients that vary by orders of magnitude. MuZero first squashes the values with a function that is roughly a square root, so large values become small, and then predicts a distribution over a fixed grid of integers instead of a single number (Appendix F). The squashing follows the R2D2 agent that MuZero compares against.

**Math.** The transform is (Appendix F):

$$
h(x) = \operatorname{sign}(x)\left(\sqrt{|x| + 1} - 1\right) + \epsilon x, \qquad \epsilon = 0.001
$$

The $\epsilon x$ term keeps $h$ strictly increasing with a slope of at least $\epsilon$, so it is invertible. The arXiv text prints the closing parenthesis after $\epsilon x$, which would place the linear term inside the sign factor. This page uses the form above, which is the one that is odd and increasing, and Exercise 3's tests check both properties. The paper doesn't print the inverse. Solving $y = w - 1 + \epsilon(w^2 - 1)$ for $w = \sqrt{|x| + 1}$ gives

$$
h^{-1}(y) = \operatorname{sign}(y)\left[\left(\frac{\sqrt{1 + 4\epsilon(|y| + 1 + \epsilon)} - 1}{2\epsilon}\right)^2 - 1\right]
$$

This inverse is derived on this page, and it isn't in the paper. The transformed scalar $h(x)$ is then encoded by a function $\phi$ over a support of 601 integers, $-300, \dots, 300$. The scalar becomes the weights of its two adjacent integers, so that the original value is recovered by $x = x_{\text{low}}\, p_{\text{low}} + x_{\text{high}}\, p_{\text{high}}$. The network outputs a softmax of size 601 for the value and for the reward. At inference, the code takes the expected value under the softmax and then inverts the transform. The loss on these outputs is the cross-entropy $\phi(z)^\top \log q$ (Figure S2).

**Tiny example, computed by `scripts/transform_example.py`.** The paper's example is a target of 3.7 on the integer support: weight 0.3 on 3 and 0.7 on 4. With the transform, $h(3.7) = 1.1716$ and $h(100) = 9.1499$. The target 100 puts weight 0.8501 on support 9 and 0.1499 on support 10, and decoding returns 100. The inverse of the largest support value is $h^{-1}(300) = 58705.6$, so the 601 supports cover raw values up to about 58,700. The order of the steps matters. The script puts equal mass on supports 0 and 10. Taking the mean first, 5, and inverting gives 34.59. Inverting each support first and then averaging gives 58.72. Exercise 3 asks you to write $h$, $h^{-1}$, $\phi$, and the decoder.

### Results

The paper trains for 1 million mini-batches of 2048 sequences in board games and 1024 in Atari, with 800 simulations per search in board games and 50 in Atari (Section 4). That is 2.05 billion and 1.02 billion sampled sequences (computed by `scripts/paper_numbers.py`). The networks use AlphaZero's convolutional residual design with 16 residual blocks in the representation and dynamics functions instead of 20, and 256 planes. For Atari, the representation function takes the last 32 RGB frames at 96 by 96 and the last 32 actions, and downsamples to a 6 by 6 hidden state (Appendices E and F).

- *Go, chess, shogi.* The paper reports that MuZero matched AlphaZero's Elo in chess and shogi and slightly exceeded it in Go, with less computation per node (16 residual blocks against 20). It reads this as a sign that MuZero uses the extra applications of the dynamics function to understand the position better (Figure 2, Section 4).
- *Atari.* Table 1 reports a median of 2041.1% and a mean of 4999.2% of the human score for MuZero, against 1920.6% and 4024.9% for R2D2. MuZero beat R2D2 in 42 of 57 games and beat the model-based SimPLe in all of them.
- *Sample efficiency.* MuZero Reanalyze re-runs the search on old trajectories with the latest parameters to get fresh targets. With 200 million frames per game, it reached a median of 731.1% against 192%, 231%, and 431% for IMPALA, Rainbow, and LASER (Section 4, Appendix H). The ratio to LASER is 1.70 (computed by `scripts/paper_numbers.py`).
- *Search matters less in Atari than in Go.* Playing the raw policy with a single simulation still did well, and performance plateaued at about 100 simulations (Figure 3B). The paper suggests that the model is less accurate in Atari. Training with only 6 simulations per move, fewer than the 8 actions in Ms. Pac-Man, still learned an effective policy (Figure 3D).

<!-- exercises -->

## Pitfalls

- **Treating the hidden state as the game state.** The state $s^k$ has no semantics. It isn't trained to reconstruct the observation or to match the true state, so you can't read a board position out of it, and you can't ask whether a hidden state is a legal position.
- **Backing up without the reward and the discount.** AlphaZero's backup adds one outcome to every edge. With rewards, each edge gets its own return $G^k$ (Equation 3), and the discount applies per step. Reusing AlphaZero's sign-flipping backup is a bug for single-agent domains.
- **Using Equation 2 without normalizing $Q$.** The bonus has the scale of a probability, so on a reward scale of thousands the search never explores. `scripts/pucb_example.py` shows 0.4 against 2000. Normalize with the min and max over the whole tree, and not over one node.
- **Dividing by zero in the normalization.** At the start of a search, the tree has a single value, so the maximum equals the minimum. The paper doesn't say what to do. Decide on a value, and test it.
- **Expecting the $c_2$ term to matter.** With $c_2 = 19652$ and 800 simulations, the log term moves the factor from 1.25 to 1.29. It exists for very large searches.
- **Masking illegal moves in the tree.** MuZero masks them at the root only, where the environment can answer (Appendix A). Inside the tree, the network is expected to have learned to give such moves low probability.
- **Stopping at terminal nodes.** MuZero doesn't. The search can pass a terminal state, and the network is expected to predict the same value there, because training treats terminal states as absorbing.
- **Mixing up the value target indices.** The target for the prediction at unroll step $k$ is $z_{t+k}$, the return from the real state $k$ steps after the sampled position, and not $z_t$. Likewise, the action fed into $g$ at step $k$ is the real action $a_{t+k}$. The unroll in `scripts/unroll_example.py` has a different result if you swap the order of the actions.
- **Averaging before or after the inverse transform.** The decoder takes the mean under the softmax and then inverts $h$. The two orders differ a lot, as `scripts/transform_example.py` shows.
- **Using the Atari losses for board games, or the reverse.** Board games use squared error for the value and no reward loss. Atari uses categorical cross-entropy for both value and reward (footnote 1, Appendix G).
- **Forgetting the gradient scaling.** Without the factor $1/K$ on each head and the factor $1/2$ on the gradient into the dynamics function, longer unrolls change the size of the update (Appendix G). Exercise 4 returns the unscaled sum of Equation 1.
- **Bootstrapping from a stale value.** The target $z_t$ uses the search value $\nu_{t+n}$ computed when the game was played. Reanalyze replaces it with a target network's value on re-run searches (Appendix H), and it lowers $n$ to 5 and weights the value loss by 0.25.
- **Copying the exercise's conventions as the paper's.** The flat-range rule in Exercise 1, the end-of-episode rule in Exercise 2, and the inverse of $h$ in Exercise 3 are choices or derivations on this page. Exercise 4's model is linear and uses squared errors everywhere.
- **Treating the toys as the paper.** The exercises use a few numbers and no search tree. The paper trains deep networks with 800 simulations per move on thousands of TPUs, with 16 for training and 1000 for self-play in each board game (Appendix G).

## Further reading

- [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265), the paper. The appendices hold the search, hyperparameters, and training details that this page cites. The paper lists pseudocode as supplementary material in the ancillary files of the arXiv submission, and this page doesn't use it.
- [AlphaZero](/alphazero/), the page that comes before this one on this site, and [AlphaGo](/alphago/), which introduced the PUCT rule that Equation 2 extends.
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html), by Sutton and Barto. Its chapters on planning and learning with models cover Dyna and Monte Carlo tree search, and its chapter on n-step methods covers the return that MuZero bootstraps.
- [David Silver's reinforcement learning course](https://www.davidsilver.uk/teaching/) includes a lecture on integrating learning and planning, given by the paper's last author.
