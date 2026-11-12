---
id: "alphago"
title: "Mastering the game of Go with deep neural networks and tree search"
authors: "Silver et al."
year: 2016
venue: "Nature 529"
arxiv: null
url: "https://doi.org/10.1038/nature16961"
track: rl
tier: extended
est_hours: 5
status: full
summary: "AlphaGo plays Go by running Monte Carlo tree search that is guided by two deep networks. A policy network trained on human games, then improved by self-play, proposes promising moves, and a value network trained on self-play positions replaces most of the search depth. The program won 494 of 495 games against other Go programs and beat the European champion Fan Hui 5 games to 0."
related_course: "rl: planning and search"
prereqs:
  - id: dqn
    reason: "You need DQN's picture of a deep convolutional network that reads a raw grid of pixels or stones, and its lesson that correlated consecutive samples hurt training, which returns here as value-network overfitting."
background: [monte-carlo-tree-search, supervised-policy-learning]
course_links:
  - title: "Monte Carlo tree search"
    url: "https://rl.ifkash.dev/lessons/022-monte-carlo-tree-search"
  - title: "The policy gradient"
    url: "https://rl.ifkash.dev/lessons/031-the-policy-gradient"
  - title: "Many agents and self-play"
    url: "https://rl.ifkash.dev/lessons/061-multi-agent-and-self-play"
self_check:
  - q: 'The value network is trained on 30 million positions, each taken from a different self-play game, and not on every position of fewer games. Why?'
    a: 'Successive positions of one game differ by a single stone but share one outcome label $z$. A large network can memorize which game a position comes from and reproduce $z$ without learning to evaluate. The paper measured a test MSE of 0.37 against a training MSE of 0.19 with whole KGS games, and 0.234 against 0.226 with one position per game. This is the same problem that DQN faces with consecutive frames.'
    see: dqn
  - q: 'A node has two moves with priors $P = [0.5, 0.5]$, visit counts $N = [3, 0]$, and values $Q = [0.1, 0]$. With $c_{\text{puct}} = 5$, which move does the selection rule pick?'
    a: 'The bonus is $u = c_{\text{puct}} P \sqrt{\sum_b N(s,b)} / (1 + N(s,a))$ and $\sqrt{3} \approx 1.732$. Move 0 scores $0.1 + 5 \cdot 0.5 \cdot 1.732 / 4 = 1.183$. Move 1 scores $0 + 5 \cdot 0.5 \cdot 1.732 / 1 = 4.330$. The search picks move 1, because an unvisited move with a decent prior gets a large bonus.'
    see: monte-carlo-tree-search
  - q: 'The SL policy network $p_\sigma$ is weaker than the RL policy network $p_\rho$ at playing. Why does AlphaGo still use $p_\sigma$ to supply the priors $P(s, a)$?'
    a: 'The paper reports that $p_\sigma$ performed better inside the search, presumably because human moves form a diverse set of reasonable candidates, while RL concentrates on the single best move. The search needs a broad beam of candidates. The stronger $p_\rho$ is used where strength matters, as the policy whose outcomes the value network learns to predict.'
    see: supervised-policy-learning
---

## The problem

A game of perfect information has an optimal value function $v^*(s)$, and you can compute it by searching the tree of all move sequences. The tree has about $b^d$ sequences, where $b$ is the breadth (legal moves per position) and $d$ is the depth (game length). The introduction gives $b \approx 35$ and $d \approx 80$ for chess, and $b \approx 250$ and $d \approx 150$ for Go. That is about $10^{123}$ sequences for chess and about $10^{360}$ for Go (computed by `scripts/search_space.py`). No computer can enumerate either tree.

The introduction names two ways to shrink the tree:

- **Reduce the depth with a position evaluator.** Cut the tree at a state $s$ and replace everything below by an estimate $v(s) \approx v^*(s)$. Chess, checkers, and othello programs do this, but people believed an evaluator for Go was intractable.
- **Reduce the breadth with a policy.** Sample moves from a distribution $p(a \mid s)$ and search only the likely ones. Monte Carlo rollouts take this to the extreme: they play to the end of the game with no branching.

The best Go programs before this paper combined MCTS with shallow policies and value functions that were linear in hand-made features. Those programs reached strong amateur level. The weak point is the evaluator. The paper says that earlier programs used a linear combination of input features for the value function, and that a value function for Go was believed to be intractable, so those programs scored leaves with fast rollouts from a weak policy.

AlphaGo replaces both pieces with deep convolutional networks. The paper reports that its policy network alone, with no search, won 85% of games against Pachi, an open-source MCTS program that runs 100,000 simulations per move. The previous state of the art, a convolutional network trained only by supervised learning, won 11% against Pachi (all numbers are from the paper, in "Reinforcement learning of policy networks"). The full system, with search, won 494 of 495 games against other Go programs and then beat the human European champion Fan Hui, a professional 2 dan, 5 games to 0 in October 2015.

The paper is an article in *Nature* with no numbered equations in the main text. This page cites the section headings instead, for example "(Methods, \"Search algorithm\")".

## Key ideas

The notation follows the paper: states $s$, moves $a$, the SL policy network $p_\sigma(a \mid s)$ with weights $\sigma$, the fast rollout policy $p_\pi$, the RL policy network $p_\rho$ with weights $\rho$, the value network $v_\theta(s)$ with weights $\theta$, the outcome $z \in \{+1, -1\}$, and the search statistics $Q(s,a)$, $N(s,a)$, and $P(s,a)$. Every number in the worked examples comes from a script in `scripts/`. Run any script from the paper folder to reproduce its output. The scripts need Python 3 and NumPy only. Numbers quoted from the paper, such as accuracies and win rates, don't come from a script.

### Three training stages for the networks

**Intuition.** Learning to play Go from self-play alone would start from random moves, and the reward arrives only at the end of a long game. So the pipeline first copies human experts, which gives fast, low-variance gradients. It then improves that copy by playing itself. Finally it trains a separate evaluator on the games the improved player plays.

**Math.** There are four trained models (Figure 1).

1. *Fast rollout policy $p_\pi$.* A linear softmax over small pattern features. It predicts human moves with 24.2% accuracy and takes 2 microseconds per move, against 3 milliseconds for the policy network ("Supervised learning of policy networks").
1. *SL policy network $p_\sigma$.* A 13-layer convolutional network trained on KGS expert positions by stochastic gradient ascent on the log likelihood of the human move:

$$
\Delta \sigma \propto \frac{\partial \log p_\sigma(a \mid s)}{\partial \sigma}
$$

   It reaches 57.0% test accuracy with all input features, and 55.7% with only the raw board and move history. The best earlier result the paper cites is 44.4%. The Methods section describes 29.4 million positions from 160,000 games by KGS 6 to 9 dan players, while the main text rounds this to 30 million.
1. *RL policy network $p_\rho$.* It starts from $\rho = \sigma$ and plays games against a randomly chosen earlier version of itself. The outcome $z_t = \pm r(s_T)$ is $+1$ for a win and $-1$ for a loss, seen from the player who moves at step $t$. The update follows the policy gradient:

$$
\Delta \rho \propto \frac{\partial \log p_\rho(a_t \mid s_t)}{\partial \rho}\, z_t
$$

   The Methods section adds a baseline: the update uses $z_t - b(s_t)$, with $b = 0$ on the first pass and $b = v_\theta$ on the second. The paper reports that $p_\rho$ won more than 80% of games against $p_\sigma$.
1. *Value network $v_\theta(s)$.* It estimates $v^{p_\rho}(s) = \mathbb{E}[z_t \mid s_t = s,\ a_{t \dots T} \sim p_\rho]$, the expected outcome when both players follow $p_\rho$. It is trained by stochastic gradient descent to minimize the mean squared error between $v_\theta(s)$ and the outcome $z$ ("Reinforcement learning of value networks"):

$$
\Delta \theta \propto \frac{\partial v_\theta(s)}{\partial \theta}\,\bigl(z - v_\theta(s)\bigr)
$$

The Methods give the cost: the SL network trained for about 3 weeks (340 million steps on 50 GPUs), and the RL network for 10,000 minibatches of 128 games on 50 GPUs for one day.

**Tiny example, computed by `scripts/policy_grad_example.py`.** For a softmax policy with logits $x$, the gradient of $\log p(a)$ with respect to $x$ is $e_a - \text{softmax}(x)$, where $e_a$ is the one-hot vector for $a$. With three equal logits and $a = 1$, the gradient is $[-1/3, 2/3, -1/3]$. The SL update ascends this direction. The RL update multiplies it by $z - b$. A win ($z = +1$, $b = 0$) gives the same vector, and a loss gives its negative, $[1/3, -2/3, 1/3]$. A step of size 0.5 raises the probability of the played move from 0.3333 to 0.4519 after a win, and lowers it to 0.2327 after a loss. With a baseline of 0.25, a win scales the gradient by 0.75. Exercise 4 asks you to write both updates and check them against finite differences.

### Value networks and correlated positions

**Intuition.** A game of 200 moves gives 200 positions, and they all get the same label $z$. Consecutive positions differ by one stone. A large network can learn to recognize the game and recite its outcome, which says nothing about how to evaluate a new position.

**Math.** The paper trained a value network on whole games from the KGS data set and saw this failure. The network memorized the game outcomes: the minimum test MSE was 0.37 against a training MSE of 0.19. The fix is a new data set of 30 million positions, each sampled from a separate self-play game of $p_\rho$ against itself. Training on it gave MSEs of 0.226 on the training set and 0.234 on the test set, which shows little overfitting ("Reinforcement learning of value networks"). Figure 2b shows that one evaluation of $v_\theta$ approaches the accuracy of 100 rollouts with $p_\rho$, using 15,000 times less computation.

**Tiny example, computed by `scripts/value_overfit.py`.** This toy model is mine and doesn't use Go. Each game has a hidden 100-dimensional vector $g$. Every position of the game is $g$ plus noise, and every position carries the game's outcome $z \in \{-1, +1\}$. A linear least-squares model trains on 400 positions in two ways and is tested on positions from fresh games. The numbers are means over 20 seeds:

| Training set | Train MSE | Test MSE |
|---|---|---|
| 20 positions from each of 20 games | 0.015 | 1.139 |
| 1 position from each of 400 games | 0.500 | 0.907 |

A predictor that always outputs 0 has an MSE of 1.000. The correlated set looks nearly perfect on the training data and is worse than a constant on new games. The independent set has a worse training error and a better test error. Both arms overfit somewhat, because 100 features and 400 samples leave room to fit noise. The gap between the arms is the point.

### Selection with a prior

**Intuition.** The tree search must decide which move to simulate next. A pure greedy choice by $Q$ never tries a move that looked bad once, and a pure uniform choice wastes simulations on bad moves. AlphaGo adds a bonus to $Q$ that is large for moves the policy network likes and moves that have few visits, and that shrinks as visits accumulate.

**Math.** Each edge $(s, a)$ of the tree stores a mean action value $Q(s,a)$, a visit count $N(s,a)$, and a prior probability $P(s,a)$. At step $t$ of a simulation, the search picks

$$
a_t = \operatorname*{arg\,max}_a \bigl( Q(s_t, a) + u(s_t, a) \bigr)
$$

The main text says only that $u(s,a) \propto P(s,a) / (1 + N(s,a))$. The Methods section gives the exact variant of the PUCT algorithm that the program uses ("Search algorithm"):

$$
u(s, a) = c_{\text{puct}}\, P(s, a)\, \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)}
$$

with $c_{\text{puct}} = 5$ (Extended Data Table 5). The paper says that the rule first prefers moves with a high prior and few visits, and asymptotically prefers moves with a high value.

**Tiny example, computed by `scripts/puct_example.py`.** Take $c_{\text{puct}} = 5$ and one node with three moves:

| Move | $P$ | $N$ | $Q$ | $u$ | $Q + u$ |
|---|---|---|---|---|---|
| 0 | 0.6 | 10 | 0.2 | 0.9448 | 1.1448 |
| 1 | 0.3 | 2 | 0.5 | 1.7321 | 2.2321 |
| 2 | 0.1 | 0 | 0.0 | 1.7321 | 1.7321 |

Here $\sqrt{\sum_b N} = \sqrt{12} = 3.4641$. The search picks move 1: its value is the highest, and its bonus is large because it has few visits. The script also runs a longer example on a node whose true move values are $[0.1, 0.3, 0.5]$ and whose prior $[0.7, 0.2, 0.1]$ favors the worst move. The share of visits is $[0.70, 0.20, 0.10]$ after 10 simulations, which copies the prior. After 100 it is $[0.55, 0.23, 0.22]$, and after 1,000 it is $[0.26, 0.14, 0.60]$, which has moved to the best move.

Exercise 1 asks you to write this rule.

### Two evaluations per leaf

**Intuition.** When a simulation reaches a new leaf $s_L$, AlphaGo scores it twice. The value network gives a fast, smooth estimate of what a strong player would achieve. A rollout with the fast policy $p_\pi$ plays the game out and gives a noisy but unbiased result for that weaker player. The two errors differ, so the program mixes them.

**Math.** The leaf position is processed once by the SL policy network, and the output probabilities become the priors, $P(s_L, a) = p_\sigma(a \mid s_L)$. The leaf is then evaluated as (main text, "Searching with policy and value networks"):

$$
V(s_L) = (1 - \lambda)\, v_\theta(s_L) + \lambda\, z_L
$$

where $z_L$ is the outcome of a rollout with $p_\pi$ from $s_L$ to the end of the game. The Methods section keeps separate statistics for the two evaluations, $N_v, W_v$ for the value network and $N_r, W_r$ for the rollouts, and combines them in the mean value of an edge:

$$
Q(s, a) = (1 - \lambda)\,\frac{W_v(s, a)}{N_v(s, a)} + \lambda\,\frac{W_r(s, a)}{N_r(s, a)}
$$

The two backward passes run asynchronously, so $N_v$ and $N_r$ differ at any moment. The paper sets $\lambda = 0.5$. In the tournament, the variant with $\lambda = 0$ (value network only) still beat every other Go program, which shows that value networks can replace Monte Carlo evaluation, and the mixed version won at least 95% of its games against the other variants ("Evaluating the playing strength of AlphaGo"). The paper concludes that the two evaluations are complementary.

**Tiny example, computed by `scripts/leaf_mix_example.py`.** A leaf has $v_\theta = 0.3$ and a rollout outcome $z = -1$. Then $V = 0.3$ for $\lambda = 0$, $V = -0.35$ for $\lambda = 0.5$, and $V = -1$ for $\lambda = 1$. For the edge form, take $\lambda = 0.5$, $W_v = 1.2$ over $N_v = 4$, and $W_r = -1$ over $N_r = 3$. Then $Q = 0.5 \cdot 0.3 + 0.5 \cdot (-0.3333) = -0.0167$. When every simulation adds one value-network result and one rollout, the two forms are the same number: for four leaves, the mean of the mixed values and the mix of the means are both $0.075$.

### Backup keeps a mean, and the search plays the most visited move

**Intuition.** Each edge records the average of every evaluation below it. Early simulations give a noisy average, and later ones sharpen it. The final move choice uses the visit counts and not the values.

**Math.** After a simulation, every edge on the path updates its statistics. In the paper's notation (Methods, "Search algorithm"), the edge accumulates the visit count and the mean evaluation of the simulations that passed through it:

$$
N(s, a) = \sum_{i=1}^{n} \mathbf{1}(s, a, i), \qquad Q(s, a) = \frac{1}{N(s, a)} \sum_{i=1}^{n} \mathbf{1}(s, a, i)\, V(s_L^i)
$$

Here $s_L^i$ is the leaf of the $i$th simulation and $\mathbf{1}(s, a, i)$ is 1 when the $i$th simulation traversed edge $(s, a)$. The paper writes the sum with the same $V$ for every edge and doesn't discuss whose point of view $V$ takes. For a two-player zero-sum game, this page uses the usual convention: the value is for the player to move at the leaf, so the edge into the leaf, which the opponent chose, gets $-V$, and the signs alternate toward the root. When the search ends, AlphaGo plays the root move with the maximum visit count, which the paper says is less sensitive to outliers than the maximum action value. The child of the chosen move becomes the new root, and its subtree is kept.

**Tiny example, computed by `scripts/backup_example.py`.** A two-edge path has leaf values $0.4$, $-0.2$, and $0.6$ in three simulations. The root edge ends with $N = 3$, $W = 0.8$, and $Q = 0.2667$, and the second edge ends with $W = -0.8$ and $Q = -0.2667$. The root $Q$ is the mean of the three leaf values, $0.2667$, whatever order they arrive in.

Exercise 2 asks you to write the leaf mix and the backup.

### A small search on Nim

**Intuition.** You can check the whole algorithm on a game small enough to solve by hand. In Nim, a pile of $n$ stones is shared by two players. Each turn takes 1, 2, or 3 stones, and whoever takes the last stone wins. The player to move at $n$ wins exactly when $n \bmod 4 \neq 0$, so from 7 stones the winning move is to take 3 and leave 4.

**Math.** This is a simplified APV-MCTS (asynchronous policy and value MCTS) in the paper's naming. It uses selection, expansion, evaluation, and backup exactly as described above, with these differences:

- It uses $\lambda = 0$ and no rollouts, so the leaf value is $v_\theta$ or the exact terminal result.
- A new node is expanded on its first visit. The paper expands a node only after its rollout count passes $n_{\text{thr}} = 40$, and it uses the tree policy $p_\tau$ for placeholder priors until the policy network answers.
- It runs one thread, so there is no virtual loss ($n_{\text{vl}} = 3$ in the paper).
- The prior comes from a function that you pass in, and the value comes from another function.

**Tiny example, computed by `scripts/nim_search.py`.** There are 44 complete games from 7 stones. With the exact value function (+1 when $n \bmod 4 \neq 0$, else $-1$) and a uniform prior, the root visit counts for taking 1, 2, and 3 stones are $[1, 1, 8]$ after 10 simulations, $[8, 9, 33]$ after 50, and $[20, 18, 162]$ after 200. The search plays "take 3". The prior matters too. With a prior of $[0.90, 0.05, 0.05]$ that favors the wrong move, the visits are $[10, 0, 0]$ after 10 simulations and $[111, 4, 85]$ after 200. At $c_{\text{puct}} = 5$, "take 3" first has the most visits between 200 and 300 simulations, and it has 872 of 1,000 visits after 1,000. With no value network at all (a value of 0 everywhere except the known terminal loss), the search still finds the winning move for 5, 6, and 7 stones, with 362 of 400 visits on "take 1" at 5 stones. It does so because it plays out to the end of the game and backs up the terminal loss with alternating signs.

Exercise 3 asks you to write the search loop.

### Scale and what the components add

**Intuition.** The networks are expensive. AlphaGo hides their cost by running the tree search on CPUs and the networks on GPUs at the same time.

**Math.** The final single-machine version used 40 search threads, 48 CPUs, and 8 GPUs. The distributed version used 40 threads, 1,202 CPUs, and 176 GPUs (the section after the search description, and Methods). The distributed version won 77% of its games against single-machine AlphaGo and 100% against other programs. Against Crazy Stone, Zen, and Pachi, single-machine AlphaGo won 77%, 86%, and 99% of games when the opponents had four handicap stones. The tournament uses an Elo scale in which a 230-point gap means a 79% win probability (Figure 4). The paper's parameter table (Extended Data Table 5) lists $\beta = 0.67$ (the softmax temperature for the SL prior), $\lambda = 0.5$, $n_{\text{vl}} = 3$, $n_{\text{thr}} = 40$, and $c_{\text{puct}} = 5$.

**Tiny example.** The Elo statement checks out with the standard formula, a win probability of $1 / (1 + 10^{-\Delta / 400})$, which gives 0.790 for $\Delta = 230$ (computed by `scripts/search_space.py`).

<!-- exercises -->

## Pitfalls

- **Expecting equation numbers.** The *Nature* article has no numbered equations in the main text. The formulas are in the text and in the Methods, so cite sections.
- **Using the main-text bonus where you need the Methods one.** The main text says $u \propto P / (1 + N)$. The code needs the Methods form with $c_{\text{puct}}$ and $\sqrt{\sum_b N(s,b)}$. Without the square root, the bonus doesn't grow with the total number of visits, and the search stops exploring too early.
- **Dividing by zero on a fresh edge.** The `1 +` in the denominator avoids this. At a node with no visits, $\sqrt{\sum_b N} = 0$, so every bonus is 0 and the first choice is a tie on $Q$. `scripts/puct_example.py` prints this case.
- **Forgetting that $V$ has a point of view.** The value is for the player to move at the leaf. If you don't negate it at every level, the search prefers moves that give the opponent a winning position. Exercise 3's tests are built to catch this.
- **Reading "mean of all evaluations" as "last evaluation".** The edge value $Q$ is an average over every simulation through the edge, so it moves slowly. A backup that overwrites $Q$ with the newest $V$ makes the search jumpy.
- **Playing the move with the highest $Q$.** AlphaGo plays the most visited root move. A move with one lucky simulation can have a high $Q$ and no support.
- **Using the strongest policy for the priors.** The paper reports that the weaker SL policy network gave better play than the RL network as the source of $P(s,a)$, and the stronger RL network gave a better value function. The two networks play different roles.
- **Sampling many positions per game for the value network.** All positions of one game share a label, so a large network memorizes it. The paper uses one position per game, and `scripts/value_overfit.py` shows the effect on a toy.
- **Mixing up the two position counts.** The main text says the SL network trained on 30 million positions. The Methods give 29.4 million positions from 160,000 games. The value network's separate set of 30 million self-play positions is a different data set.
- **Treating the toy search as the paper's search.** The Nim code here expands on the first visit, uses no rollouts, and runs one thread. The real program has an expansion threshold, a tree policy for placeholder priors, virtual losses, and asynchronous GPU queues.
- **Looking for the learned-from-scratch version.** This paper starts from human games. AlphaZero, which builds on this paper, trains without human data.

## Further reading

- [Mastering the game of Go with deep neural networks and tree search](https://www.nature.com/articles/nature16961), the *Nature* article. Its Methods section and Extended Data tables hold the details that this page cites.
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html), by Sutton and Barto. A section of the Applications and Case Studies chapter describes AlphaGo and AlphaGo Zero.
- [A Survey of Monte Carlo Tree Search Methods](https://ieeexplore.ieee.org/document/6145622), by Browne et al., covers UCT and the many variants of tree search that came before the neural networks.
- [David Silver's reinforcement learning course](https://www.davidsilver.uk/teaching/) includes a lecture on classic games and a lecture on simulation-based search, given by the paper's lead author.
