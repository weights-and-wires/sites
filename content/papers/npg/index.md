---
id: "npg"
title: "A Natural Policy Gradient"
authors: "Kakade"
year: 2001
venue: "NIPS 2001 (Advances in Neural Information Processing Systems 14, published 2002)"
arxiv: null
url: "https://papers.nips.cc/paper_files/paper/2001/hash/4b86abe48d358ecf194c56c69108433e-Abstract.html"
track: rl
tier: extended
est_hours: 4
status: full
summary: "The ordinary policy gradient is the steepest ascent direction only when you measure step length in raw parameter units, so its steps change with the units of the parameters and with how often each state is visited. This paper measures step length with the Fisher information matrix of the policy, which gives the natural gradient: the gradient multiplied by the inverse Fisher matrix. With a compatible function approximator, the natural gradient is exactly the weight vector of the least-squares fit to the action values, so a large natural step moves toward the greedy action of that fit, as in one step of policy iteration."
related_course: "rl: policy gradients"
prereqs: []
background: [policy-gradient-theorem, fisher-information, kl-divergence, softmax, least-squares]
course_links:
  - title: "Natural gradients and trust regions"
    url: "https://rl.ifkash.dev/lessons/035-trust-regions-and-trpo"
  - title: "Why policy optimization works"
    url: "https://rl.ifkash.dev/lessons/039-why-policy-optimization-works"
self_check:
  - q: 'A one-parameter policy $\pi(1) = \sigma(\theta)$ has $Q = [0, 1]$. You rescale the parameter to $\theta = 10u$ and take a plain gradient step of size $\alpha = 0.1$ in $u$. How does the change in $\theta$ compare with the same step in $\theta$ itself, and what does the natural gradient step do?'
    a: 'At $\theta = 0$ the gradient is 0.25. The plain step changes $\theta$ by 0.025 in the original units and by 2.5 in the rescaled units, because $\Delta\theta = c^2 \alpha\,\nabla_\theta\eta$. That moves $\pi(1)$ from 0.5 to 0.9241, with a KL of 0.636. The natural gradient is 1 in both units, so both steps change $\theta$ by 0.1 and give $\pi(1) = 0.5250$ (`scripts/invariance.py`).'
    see: fisher-information
  - q: 'Theorem 1 says that if $w$ minimizes the squared error of $w^\top\psi$ against $Q$, then $w = F^{-1}\nabla\eta$. In which weights is that error measured, and why does adding a constant to $Q(s, \cdot)$ in one state change neither side?'
    a: 'The error is weighted by $\rho(s)\,\pi(a; s, \theta)$, the on-policy distribution over state-action pairs. The features $\psi = \nabla\log\pi$ have zero mean under $\pi$ in each state, so a constant offset in $Q(s, \cdot)$ is orthogonal to every feature. It drops out of the normal equations $Fw = \nabla\eta$ and out of the gradient, and $w$ is unchanged to about $10^{-14}$ (`scripts/compatible_weights.py`).'
    see: least-squares
  - q: 'For the policy in `scripts/greedy_limit.py`, a plain gradient step of size 1000 puts all the probability on action 1 in state 2, but the natural gradient step puts it on action 2. Which of the two actions has the larger $f(s, a; w)$, and what does Theorem 2 say about the natural step?'
    a: 'Action 2 has $f = 0.4582$ and action 1 has $f = 0.3454$, and both beat the policy-weighted mean of 0. The plain step picks a better action. Theorem 2 says the natural step picks an action in $\arg\max_a f(s, a; w)$ in every state for an exponential-family policy, so it picks the best action of the fit, which is action 2.'
    see: softmax
---

## The problem

A policy gradient method follows $\nabla\eta$, the gradient of the average reward $\eta(\theta)$ with respect to the policy parameters. This page calls that the plain gradient. It has two problems, and both come from the same source: the plain gradient is the steepest direction only when you measure step length with the identity matrix, and that choice has no connection to the policy.

The first problem is that the plain gradient depends on the units of the parameters. Take a policy with one parameter, $\pi(1) = \sigma(\theta)$, and action values $Q = [0, 1]$, so $\eta = \sigma(\theta)$. At $\theta = 0$ the gradient is 0.25. Now write the same policy with the parameter rescaled to $\theta = 10u$. A plain step of size $\alpha = 0.1$ in $u$ changes $\theta$ by 2.5 instead of 0.025, and $\pi(1)$ jumps from 0.5 to 0.9241 (a KL of 0.636) instead of 0.5062 (a KL of 0.00008). The policy is the same and the step rule is the same, but the step is 100 times larger in $\theta$. The numbers come from `scripts/invariance.py`.

The second problem is that the plain gradient weights each state by how often you visit it. A state that the policy rarely visits gets a small gradient, so it learns slowly, and learning there gets slower as the policy stops visiting it. The paper's two-state example in Section 5 shows the failure. The plain gradient raises the self-loop probability in the frequently visited state first, which makes the other state even rarer. The stationary probability of that state falls as low as $10^{-7}$ from an initial 0.2, and the policy sits on a long plateau. Under the natural gradient it never drops below 0.05, and the paper reports a speedup of many orders of magnitude (Section 5, Figure 1C).

The paper's answer is to choose the metric from the policy instead of from the parameter units. It measures distance with the Fisher information matrix, and the resulting direction is the natural gradient. The paper also connects the direction to policy iteration: with a compatible function approximator, the natural gradient is the weight vector of the critic, and a large step along it picks the greedy action.

Section 5 reports results in three settings. In a one-dimensional linear quadratic regulator, the natural gradient reaches a score of about 22 roughly three orders of magnitude faster than the plain gradient when the parameters are rescaled by a factor of 10 (Figure 1A). In Tetris, with the features of the earlier greedy policy iteration work, the plain gradient learns poorly and the natural gradient improves quickly and doesn't degrade (Figure 2). In a smaller Tetris board, the plain gradient eventually reaches the same performance. This page doesn't reproduce those experiments.

## Key ideas

The paper works with the average reward $\eta(\pi) = \sum_{s,a}\rho^\pi(s)\,\pi(a; s)\,R(s, a)$ of an ergodic MDP, where $\rho^\pi$ is the stationary state distribution (Section 2). This differs from the discounted setting of the [TRPO page](/trpo/), which uses the discounted visitation $\rho_\pi$. The policy $\pi(a; s, \theta)$ is a smooth function of $\theta \in \mathbb{R}^m$, and $Q^\pi(s, a)$ is the differential action value. This page drops the superscript $\pi$ on $\rho$ and $Q$ and writes $\pi(a \mid s)$ as the paper's $\pi(a; s, \theta)$ when the parameters are clear from the context. The paper's year is the proceedings' meeting year, 2001. The volume appeared in 2002, so other sources cite it as Kakade (2002).

### The steepest direction depends on the metric

**Intuition.** "Steepest" means "largest improvement per unit of step length." The answer depends on what you call length. If you say a step of $d\theta$ has length $\lVert d\theta\rVert$, you get the plain gradient. If you say a step has the length that the policy feels, you get a different direction.

**Math.** The paper defines the steepest direction as the $d\theta$ that moves $\eta$ the most under a fixed squared length $\lvert d\theta\rvert^2 = d\theta^\top G(\theta)\, d\theta$ for a positive-definite matrix $G(\theta)$ (Section 2). Maximizing $\nabla\eta^\top d\theta$ under that constraint gives (Lagrange multipliers, same section):

$$
d\theta \;\propto\; G(\theta)^{-1}\,\nabla\eta(\theta)
$$

The plain gradient is the case $G = I$. The paper calls that an ad hoc choice that has no connection to the policy, and says it is "non-covariant": the update $\Delta\theta_i = \partial\eta/\partial\theta_i$ has units of $1/\theta_i$ on the right and units of $\theta_i$ on the left (Section 1). The gradient of Equation 1 is

$$
\nabla\eta(\theta) = \sum_{s,a}\rho(s)\,\nabla\pi(a; s, \theta)\,Q(s, a)
$$

Because $\rho(s)$ multiplies each state's contribution, the plain gradient weights the states by visit frequency.

**Tiny example, computed by `scripts/invariance.py`.** Part A is the one-parameter policy above. At $\theta = 0$, the gradient and the Fisher information both equal 0.25, so $F^{-1}\nabla\eta = 1$. Under $\theta = cu$, the gradient becomes $c\,\nabla_\theta\eta$ and the Fisher information becomes $c^2 F$. The table shows one step with $\alpha = 0.1$:

| Scale $c$ | Plain $\Delta\theta$ | Plain KL | Natural $\Delta\theta$ | Natural KL |
|---|---|---|---|---|
| 1 | 0.0250 | 0.00008 | 0.1000 | 0.00125 |
| 10 | 2.5000 | 0.63574 | 0.1000 | 0.00125 |

Check the natural step by hand for $c = 10$. In $u$, the gradient is $10 \cdot 0.25 = 2.5$ and the Fisher information is $100 \cdot 0.25 = 25$. The step is $\Delta u = 0.1 \cdot 2.5/25 = 0.01$, so $\Delta\theta = 10 \cdot 0.01 = 0.1$, the same as for $c = 1$.

Part B shows the state weighting. Two states have $\rho = [0.8, 0.2]$, a sigmoid policy $\pi(1; s) = [0.8, 0.1]$, and fixed gaps $Q(s, 1) - Q(s, 0) = [0.5, 1.0]$. The plain gradient is $[0.064, 0.018]$, so state 0 gets 3.556 times the weight of state 1 even though state 1 has the larger gap. The natural gradient is $[0.5, 1.0]$, which is the gap itself, because $\rho(s)$ cancels against the Fisher matrix. The paper makes the same statement for a table-lookup Boltzmann policy in Section 5. Treat $Q$ as a fixed snapshot in this example. In a real MDP, $\rho$ and $Q$ change with $\theta$.

Exercise 3 asks you to check the invariance numerically.

### The Fisher information matrix of a policy

**Intuition.** The right length for a step is how much it changes the action distributions, not how much it changes the numbers in $\theta$. The Fisher information matrix measures that. Averaging it over the states that the policy visits gives one metric for the whole policy.

**Math.** For each state $s$, the distribution $\pi(a; s, \theta)$ is a point on a probability manifold with coordinates $\theta$. Its Fisher information is (Section 2, Equation 2):

$$
F_s(\theta) = \mathbb{E}_{\pi(a;s,\theta)}\!\left[\frac{\partial\log\pi(a; s,\theta)}{\partial\theta_i}\,\frac{\partial\log\pi(a; s,\theta)}{\partial\theta_j}\right]
$$

The paper cites Amari for two facts. The matrix is positive definite, and, up to a scale, it defines the same distance between two distributions whatever coordinates you use. The metric for the whole policy averages over the stationary distribution (Equation 3), and the natural gradient is the steepest direction under it (Equation 4):

$$
F(\theta) = \mathbb{E}_{\rho}\big[F_s(\theta)\big], \qquad \tilde\nabla\eta(\theta) = F(\theta)^{-1}\,\nabla\eta(\theta)
$$

This page writes the natural gradient as $F^{-1}\nabla\eta$ and calls the score $\psi(s, a) = \nabla_\theta\log\pi(a; s, \theta)$ (Equation 5), so $F = \sum_{s,a}\rho(s)\,\pi(a; s)\,\psi\psi^\top$. The paper notes that $F$ depends on the transition model only through $\rho$ (Section 2). Section 5 notes that you can estimate $F$ online, because the derivatives $\nabla\log\pi$ are already computed for $\nabla\eta$.

A tabular softmax policy has a singular $F$. If $\pi(a) \propto \exp(\theta_a)$, then $\psi(a) = e_a - \pi$ and $F_s = \mathrm{diag}(\pi) - \pi\pi^\top$. Adding the same number to every logit changes nothing, so $F_s\mathbf{1} = 0$. The paper deals with a near-singular $F$ by adding about $10^{-3} I$ at every step, because the policies tend to become deterministic and $\nabla\log\pi \to 0$ (Section 5).

**Tiny example, computed by `scripts/softmax_fisher.py`.** One state, three actions, $\pi = [0.5, 0.3, 0.2]$ and $Q = [1, 0, 0.5]$, so $V = \pi\cdot Q = 0.6$ and the advantages are $[0.4, -0.6, -0.1]$. The Fisher matrix is

$$
F = \begin{bmatrix} 0.25 & -0.15 & -0.10 \\ -0.15 & 0.21 & -0.06 \\ -0.10 & -0.06 & 0.16 \end{bmatrix}
$$

The eigenvalues are 0 (to rounding), 0.2319, and 0.3881. Check $F\mathbf{1} = 0$ by hand: the first row sums to $0.25 - 0.15 - 0.10 = 0$. The gradient is $\nabla\eta = \pi\,(Q - V) = [0.2, -0.18, -0.02]$, and its entries sum to 0, so it lies in the range of $F$ and a solution of $Fx = \nabla\eta$ exists. With damping, the solution of $(F + \lambda I)x = \nabla\eta$ is:

| $\lambda$ | $x$ |
|---|---|
| 1 | `[0.1433, -0.1316, -0.0117]` |
| 0.1 | `[0.4012, -0.3889, -0.0123]` |
| 0.001 | `[0.4988, -0.4986, -0.0002]` |
| $10^{-6}$ | `[0.5, -0.5, 0.0]` |

A large $\lambda$ shrinks the step toward $\nabla\eta/\lambda$, and a small $\lambda$ recovers the pseudo-inverse solution $[0.5, -0.5, 0]$. Damping isn't free. It adds a term that depends on the parameterisation, and the next section shows what the solution means.

Exercises 1 and 2 ask you to build $F$ for a softmax policy with a state distribution, and to solve the damped system.

### With a compatible critic, the natural gradient is the critic's weights

**Intuition.** Suppose you train a critic to predict $Q$ using the features $\nabla\log\pi$. Those are the features that make the policy gradient come out exact when you plug the critic in for $Q$. The paper shows that the critic's weight vector, after a least-squares fit, is the natural gradient.

**Math.** Section 3.1 defines a compatible function approximator for $Q$ (Equation 5):

$$
\psi^\pi(s, a) = \nabla_\theta\log\pi(a; s, \theta), \qquad f^\pi(s, a; w) = w^\top\psi^\pi(s, a)
$$

Let $w$ minimize the squared error $\epsilon(w, \pi) = \sum_{s,a}\rho(s)\,\pi(a; s,\theta)\,\big(f^\pi(s, a; w) - Q^\pi(s, a)\big)^2$. The approximator is compatible with the policy in the sense of Sutton et al. (2000), the policy gradient theorem paper: using $f^\pi$ in place of $Q^\pi$ in Equation 1 still gives the exact gradient.

**Theorem 1** (Section 3.1). If $w$ minimizes $\epsilon(w, \pi_\theta)$, then $w = \tilde\nabla\eta(\theta)$. The proof is short. Setting $\partial\epsilon/\partial w = 0$ gives $\sum \rho\,\pi\,\psi\,(\psi^\top w - Q) = 0$. Because $\nabla\pi = \pi\psi$, the term $\sum\rho\,\pi\,\psi\,Q$ is $\nabla\eta$, and $\sum\rho\,\pi\,\psi\psi^\top = F$, so $Fw = \nabla\eta$.

So an actor-critic method with a compatible critic is forced to use the natural gradient as the critic's weights. Good actions, the ones with large $Q$, have features with a large inner product with the natural gradient.

**Tiny example.** Check the tabular case from `scripts/softmax_fisher.py` by hand. The vector $w = [0.5, -0.5, 0]$ is $Q$ minus its mean, and $Fw = [0.25\cdot0.5 + 0.15\cdot0.5,\; -0.15\cdot0.5 - 0.21\cdot0.5,\; -0.10\cdot0.5 + 0.06\cdot0.5] = [0.2, -0.18, -0.02]$, which equals $\nabla\eta$. The critic $f(a) = w^\top\psi(a)$ takes the values $[0.4, -0.6, -0.1]$, which are the advantages. The weight vector fits $Q$ up to a constant, and $\psi$ can't represent the constant because $\sum_a\pi\,\psi = 0$.

Now a case with three states and three features, computed by `scripts/compatible_weights.py`. The features and $Q$ are random with a fixed seed, and $\rho = [0.5, 0.3, 0.2]$.

| Quantity | Value |
|---|---|
| $\nabla\eta$ (Equation 1) | `[0.0529, -0.0643, -0.0500]` |
| Eigenvalues of $F$ | `[0.0912, 0.1552, 0.3370]` |
| $F^{-1}\nabla\eta$ | `[0.3890, -0.6130, -0.6926]` |
| Weighted least-squares $w$ | `[0.3890, -0.6130, -0.6926]` |
| Unweighted least-squares $w$ | `[0.2345, -0.6187, -0.4882]` |

The two $w$ values agree to about $10^{-15}$. The fit isn't perfect, because the weighted squared error at $w$ is 0.2469, against 0.3415 at $w = 0$, and Theorem 1 holds for the best fit anyway. The unweighted fit gives a different answer, so the weights $\rho(s)\,\pi(a; s)$ matter. Adding a constant to $Q(s, \cdot)$ in each state changes neither $w$ nor $\nabla\eta$.

Exercise 4 asks you to compute $w$ by weighted least squares and compare it with $F^{-1}\nabla\eta$.

### A natural gradient step moves toward the greedy action

**Intuition.** The plain gradient step makes every action with above-average value more likely. The natural gradient step moves toward the action that the critic ranks highest. A very large step ends up there.

**Math.** Section 3.2 compares the update with one improvement step of policy iteration, which picks $a \in \arg\max_{a'} f^\pi(s, a'; w)$ in each state. Take the exponential family $\pi(a; s,\theta) \propto \exp(\theta^\top\phi_{sa})$, whose geometry is flat, so a step along a tangent vector stays on the manifold.

**Theorem 2** (Section 3.2). Assume $\tilde\nabla\eta(\theta) \ne 0$ and that $w$ minimizes the approximation error. Let $\pi_\infty(a; s) = \lim_{\alpha\to\infty}\pi(a; s, \theta + \alpha\tilde\nabla\eta(\theta))$. Then $\pi_\infty(a; s) \ne 0$ if and only if $a \in \arg\max_{a'} f^\pi(s, a'; w)$.

The proof uses $\psi = \phi_{sa} - \mathbb{E}_\pi[\phi_{sa'}]$. The second term doesn't depend on $a$, so $\arg\max_a f = \arg\max_a\tilde\nabla\eta^\top\phi_{sa}$. After the step, the logit of $a$ is $\theta^\top\phi_{sa} + \alpha\tilde\nabla\eta^\top\phi_{sa}$, and the second term dominates as $\alpha\to\infty$. The paper says the plain gradient, in the same limit, picks only a better action: one with $f$ above its policy-weighted mean, but not necessarily the best. The paper uses the infinite step only to make the point. It is not a recommended step size.

**Theorem 3** (Section 3.2). For a general parameterisation, let $w$ minimize the error and take the step $\theta' = \theta + \alpha\tilde\nabla\eta(\theta)$. Then

$$
\pi(a; s, \theta') = \pi(a; s, \theta)\,\big(1 + \alpha\,f^\pi(s, a; w)\big) + O(\alpha^2)
$$

The proof follows from $\Delta\theta = \alpha w$ (Theorem 1) and $\nabla\pi = \pi\psi$. Each action's probability moves in proportion to the critic's value for it, so the natural gradient is locally moving toward the best action. The paper adds that the greedy action doesn't always improve the policy, and many studies examine that failure. With a line search the improvement is guaranteed to start, because $F$ is positive definite (end of Section 3.2). The paper treats the line search as an overhead. The [TRPO page](/trpo/) shows how a KL budget sets the step length instead.

**Tiny example, computed by `scripts/softmax_fisher.py`.** Use the single-state policy from before, with $w$ equal to $[0.5, -0.5, 0]$ and $f = [0.4, -0.6, -0.1]$. A step with $\alpha = 0.1$ gives the exact probabilities `[0.5199, 0.2823, 0.1978]`. Theorem 3 predicts $\pi\,(1 + \alpha f) = [0.5\cdot1.04,\; 0.3\cdot0.94,\; 0.2\cdot0.99] = [0.52, 0.282, 0.198]$. The largest error is $2.6\times10^{-4}$. At $\alpha = 0.01$ it is $2.6\times10^{-6}$, about 100 times smaller, which is the $O(\alpha^2)$ rate.

Now Theorem 2 with the three-state exponential-family policy from `scripts/greedy_limit.py`, which uses the same seed and $\rho$ as the previous example:

| Step size $\alpha$ | Natural: most likely action in states 0, 1, 2 | Plain: most likely action in states 0, 1, 2 |
|---|---|---|
| 10 | 2, 2, 2 | 1, 1, 1 |
| 1000 | 2, 2, 2 | 2, 2, 1 |

At $\alpha = 1000$, the natural step puts probability 1 on action 2 in every state, which is the action with the largest $f$ in each state. The plain step also picks action 2 in states 0 and 1, but in state 2 it puts all the probability on action 1. Action 1 has $f = 0.3454$, which beats the mean of 0 but not the maximum of 0.4582. The greedy action is greedy for $f$, not for $Q$. In state 1, $\arg\max Q$ is action 0 and $\arg\max f$ is action 2, because the fit is only an approximation.

## Pitfalls

- **Confusing the natural gradient with a step size rule.** The paper gives a direction, $F^{-1}\nabla\eta$, and a learning rate $\alpha$ that you choose. It doesn't say how large $\alpha$ may be. [TRPO](/trpo/) fixes the step by a KL budget $\delta$, with $\beta = \sqrt{2\delta/(s^\top F s)}$, and a line search. The two differ in the step length, not in the direction.
- **Reading Theorem 2 as an algorithm.** The infinite step is a limit used to show where the direction points. A real step has finite $\alpha$, and the greedy step of policy iteration can degrade the policy. The paper's Tetris results show that earlier greedy policy iteration with the same features degrades after an initial improvement.
- **Expecting exact invariance in practice.** The Fisher metric is invariant in the sense that it gives the same distance in any coordinates, but the natural gradient with damping isn't. In the invariance example, a damped step with $\lambda = 0.1$ changes $\theta$ by 0.0714 at $c = 1$ and 0.0996 at $c = 10$, while the undamped step gives 0.1 in both. The paper also notes that its natural gradient curves under different scalings aren't identical, and it attributes that to the weighting by the state distribution (Section 5).
- **Inverting a singular Fisher matrix.** The tabular softmax $F$ has the null vector $\mathbf{1}$, so $F^{-1}$ doesn't exist. Use a damped solve or a pseudo-inverse, and don't form an inverse. The paper adds about $10^{-3} I$.
- **Treating $w$ as an estimate of $Q$.** The features $\psi$ have zero mean under the policy in each state, so $w^\top\psi$ can represent only the part of $Q$ that varies across actions. It approximates the advantage, and the fit has an error even at the optimum (0.2469 in the example above). Theorem 1 says $w = F^{-1}\nabla\eta$ only at the least-squares optimum. A critic that you train for a few steps gives an approximate natural gradient.
- **Using the wrong weights in the fit.** The weights are $\rho(s)\,\pi(a; s,\theta)$, which are the state-action frequencies of the current policy. An unweighted fit returns a different vector (`[0.2345, -0.6187, -0.4882]` against `[0.3890, -0.6130, -0.6926]` above).
- **Assuming the metric captures curvature of $\eta$.** Section 4 shows that $F$ doesn't contain the terms of the Hessian that depend on $Q$, and that the Hessian isn't necessarily positive definite away from a maximum. The paper suggests that conjugate gradient methods may converge faster near a maximum, and that the natural gradient may be more efficient far from it.
- **Mixing the two Fisher matrices.** The matrix here is the exact expectation over actions, $\sum_a\pi\,\psi\psi^\top$. An estimate built from outer products of sampled $\nabla\log\pi$ is also a consistent estimate of $F$ (Section 5), but it's a different matrix at any finite sample size.

## Further reading

- [Trust Region Policy Optimization](/trpo/), by Schulman et al., turns this direction into a step with a KL budget, conjugate gradient solves, and a line search.
- Sutton, McAllester, Singh, and Mansour, "Policy gradient methods for reinforcement learning with function approximation" (NIPS 2000), gives the policy gradient theorem and the compatible function approximation that this paper builds on.
- Amari, "Natural gradient works efficiently in learning" (Neural Computation, 1998), is the source of the natural gradient that the paper cites.
- [Approximately Optimal Approximate Reinforcement Learning](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/KakadeLangford-icml2002.pdf), by Kakade and Langford, is the companion paper on conservative policy iteration.
- [Spinning Up: Trust Region Policy Optimization](https://spinningup.openai.com/en/latest/algorithms/trpo.html), by OpenAI, derives the natural gradient step and ships a reference implementation.
