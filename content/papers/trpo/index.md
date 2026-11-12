---
id: "trpo"
title: "Trust Region Policy Optimization"
authors: "Schulman et al."
year: 2015
venue: "ICML 2015"
arxiv: "1502.05477"
url: "https://arxiv.org/abs/1502.05477"
track: rl
tier: core
est_hours: 6
status: full
summary: "TRPO updates a policy by maximizing a surrogate objective that is accurate near the current policy, and it limits how far the new policy can move by bounding the average KL divergence between the two. The theory shows that penalizing the maximum KL divergence guarantees that the true return never decreases, but the penalty size it prescribes gives tiny steps, so the practical algorithm uses a hard constraint instead. Each update solves the constrained problem approximately with a natural gradient direction from conjugate gradient, a step length set by the constraint, and a backtracking line search."
related_course: "rl: policy gradients"
prereqs:
  - id: npg
    reason: "TRPO's step is a natural gradient step with a KL budget, so you need the Fisher matrix, the natural gradient, and why it ignores how the policy is parameterized."
background: [policy-gradient-theorem, kl-divergence, conjugate-gradient]
course_links:
  - title: "Natural gradients and trust regions"
    url: "https://rl.ifkash.dev/lessons/035-trust-regions-and-trpo"
self_check:
  - q: 'The surrogate $L_\pi(\tilde\pi)$ uses the state visitation $\rho_\pi$ of the old policy, where the true return uses $\rho_{\tilde\pi}$. What does that ignore, and what does Equation 4 guarantee anyway?'
    a: 'It ignores how the new policy changes which states get visited. Even so, $L$ and $\eta$ have the same value and the same gradient at the old parameters, which is the policy gradient theorem restated. So a small enough step that improves $L$ also improves $\eta$, but Equation 4 says nothing about how small "small enough" is.'
    see: policy-gradient-theorem
  - q: 'Two policies pick between two actions in one state: $\pi = [0.5, 0.5]$ and $\tilde\pi = [0.1, 0.9]$. Compute $D_{TV}$ and $D_{KL}(\pi \,\|\, \tilde\pi)$. Is $D_{KL}(\tilde\pi \,\|\, \pi)$ the same?'
    a: '$D_{TV} = \tfrac12(0.4 + 0.4) = 0.4$, and $D_{KL}(\pi \,\|\, \tilde\pi) = 0.5\ln(0.5/0.1) + 0.5\ln(0.5/0.9) \approx 0.5108$. The reverse direction is about 0.3681, so KL is not symmetric. The relation $D_{TV}^2 \le D_{KL}$ holds here: $0.16 \le 0.5108$.'
    see: kl-divergence
  - q: 'The Fisher matrix for a policy network with a million parameters has a trillion entries. How does TRPO still compute $F^{-1}g$?'
    a: 'Conjugate gradient solves $Fx = g$ using only products $Fv$, and each product costs about as much as one gradient computation, so $F$ is never stored. The paper runs 10 iterations (Appendix C.1). In exact arithmetic, CG on a $d \times d$ system finishes in at most $d$ iterations, which is why 2 iterations solve the $2 \times 2$ example exactly.'
    see: conjugate-gradient
---

## The problem

A policy gradient method repeats three steps: collect trajectories with the current policy, estimate the gradient of the expected return $\eta(\pi)$, and take a step. The gradient points in an improving direction only locally, and the method has no principled way to say how far to go. Both failure modes are costly.

- **A small step wastes samples.** Every update needs a fresh batch of trajectories, so a conservative step size makes learning slow.
- **A large step can destroy the policy.** The next batch comes from the new policy. If one bad update collapses it, for example into a policy that falls over immediately, the next gradient comes from the collapsed policy, and learning may not recover. The paper's Section 8.1 supports this with a comparison. TRPO learned swimming, hopping, and walking gaits, while the classic natural policy gradient, which uses a fixed penalty coefficient in place of a KL constraint, did not produce hopping and walking gaits that made forward progress. The authors swept the step size for the natural gradient and kept the best value.

The distance that matters is in policy space, not parameter space. Changing one parameter by 0.1 can leave the policy almost unchanged, or it can flip the action in every state.

Here is a milder version of the failure in a two-state MDP that the examples below use. A step that moves the probability of switching states from 0.5 toward 1 looks better and better to the surrogate objective, because the surrogate is linear in that probability, but the true return gains less and less. The numbers come from `scripts/bound_example.py`:

| New switch probability in state 0 | Surrogate gain $L - \eta$ | True gain $\eta(\tilde\pi) - \eta(\pi)$ |
|---|---|---|
| 0.60 | +0.4729 | +0.4210 |
| 0.80 | +1.4187 | +1.0356 |
| 0.99 | +2.3171 | +1.4445 |

The surrogate overshoots, and the gap grows with the step. The paper's contribution is a way to bound that gap, so you can choose a step that is provably safe, and then a practical algorithm that approximates the bound.

## Key ideas

The sections that follow use the paper's notation: states $s$, actions $a$, reward $r(s)$, discount $\gamma$, start distribution $\rho_0$, expected discounted return $\eta(\pi)$, advantage $A_\pi$, and the unnormalized discounted visitation $\rho_\pi$. Section 2 of the paper defines a reward $r(s)$ that depends on the state only, and so do the examples. Two notation differences: the paper calls the Fisher matrix $A$, and this page writes $F$ for it, because $A_\pi$ is already the advantage. The conjugate gradient example in Exercise 3 keeps the paper's letter $A$ for a generic matrix. Every number comes from a script in `scripts/`. Run any script from the paper folder to reproduce the output.

### The surrogate objective

**Intuition.** You want to score a candidate policy $\tilde\pi$ using data from the current policy $\pi$. The true return of $\tilde\pi$ depends on the states $\tilde\pi$ visits, and you have no samples from those states. The surrogate keeps the old state distribution and changes only the actions, which is accurate when the policies are close.

**Math.** The return of any policy $\tilde\pi$ equals the old return plus the advantages of the old policy, accumulated along trajectories of the new one (Section 2, Equation 1). Rewriting the sum over time as a sum over states gives (Equation 2):

$$
\eta(\tilde\pi) = \eta(\pi) + \sum_s \rho_{\tilde\pi}(s) \sum_a \tilde\pi(a \mid s)\, A_\pi(s, a)
$$

The unnormalized visitation is $\rho_\pi(s) = P(s_0 = s) + \gamma P(s_1 = s) + \gamma^2 P(s_2 = s) + \dots$, so it sums to $1/(1-\gamma)$. The new policy's visitation $\rho_{\tilde\pi}$ is hard to optimize through, so the paper replaces it with $\rho_\pi$ (Equation 3):

$$
L_\pi(\tilde\pi) = \eta(\pi) + \sum_s \rho_\pi(s) \sum_a \tilde\pi(a \mid s)\, A_\pi(s, a)
$$

For a differentiable parameterized policy $\pi_\theta$, $L$ matches $\eta$ to first order at the current parameters $\theta_0$ (Equation 4): $L_{\pi_{\theta_0}}(\pi_{\theta_0}) = \eta(\pi_{\theta_0})$, and $\nabla_\theta L_{\pi_{\theta_0}}(\pi_\theta)\big|_{\theta=\theta_0} = \nabla_\theta \eta(\pi_\theta)\big|_{\theta=\theta_0}$. The equal gradients are the policy gradient theorem. The paper states that this gives no guidance on step size.

**Tiny example, computed by `scripts/bound_example.py`.** The MDP has two states with $r = [0, 1]$, $\gamma = 0.9$, and $\rho_0 = [1, 0]$, so every episode starts in state 0. Action 0 stays in the current state and action 1 switches to the other state. The old policy $\pi$ switches with probability 0.5 in state 0 and 0.2 in state 1. The new policy $\tilde\pi$ switches with probability 0.9 in state 0 and still 0.2 in state 1. The results are as follows:

| Quantity | Value |
|---|---|
| $\eta(\pi)$ | 6.1644 |
| $\rho_\pi$ | `[3.8356, 6.1644]`, which sums to $10 = 1/(1-\gamma)$ |
| $V_\pi$ | `[6.1644, 7.5342]` |
| $A_\pi(s, a)$, rows are states | `[[-0.6164, 0.6164], [0.2466, -0.9863]]` |
| $L_\pi(\tilde\pi)$ | 8.0559 |
| $\eta(\tilde\pi)$ | 7.4312 |

Check $L$ by hand. The policies differ only in state 0, so $L_\pi(\tilde\pi) - \eta(\pi) = 3.8356 \cdot (0.1 \cdot (-0.6164) + 0.9 \cdot 0.6164) = 3.8356 \cdot 0.4931 = 1.8915$. Adding $\eta(\pi)$ gives 8.0559. The surrogate predicts a gain of 1.8915, and the true gain is 1.2668. Equation 2, evaluated with the true visitation $\rho_{\tilde\pi}$, returns 7.4312 exactly, which confirms the identity.

Exercise 1 asks you to implement $\rho_\pi$, $A_\pi$, and $L_\pi$, and to check Equations 2 and 4.

### The monotonic improvement bound

**Intuition.** The surrogate overshoots by an amount that depends on how far $\tilde\pi$ is from $\pi$. If you can bound the overshoot by a penalty that depends only on that distance, then maximizing "surrogate minus penalty" can't make the true return worse.

**Math.** The paper starts from Kakade and Langford's bound for mixtures of policies (Section 2, Equations 5 and 6), and extends it to every stochastic policy by replacing the mixture weight with a distance. The distance is the total variation divergence, $D_{TV}(p \,\|\, q) = \tfrac12 \sum_i |p_i - q_i|$, taken at the worst state (Section 3, Equation 7):

$$
D_{TV}^{\max}(\pi, \tilde\pi) = \max_s D_{TV}\big(\pi(\cdot \mid s) \,\|\, \tilde\pi(\cdot \mid s)\big)
$$

Theorem 1 (Equation 8) says that with $\alpha = D_{TV}^{\max}(\pi_{\text{old}}, \pi_{\text{new}})$ and $\epsilon = \max_{s,a} |A_\pi(s, a)|$,

$$
\eta(\pi_{\text{new}}) \;\ge\; L_{\pi_{\text{old}}}(\pi_{\text{new}}) - \frac{4\epsilon\gamma}{(1-\gamma)^2}\,\alpha^2
$$

The paper proves it twice in the appendix, once by coupling random variables (Appendix A) and once by perturbation theory (Appendix B). The inequality $D_{TV}(p \,\|\, q)^2 \le D_{KL}(p \,\|\, q)$ then gives the KL version (Equation 9), with $C = 4\epsilon\gamma/(1-\gamma)^2$:

$$
\eta(\tilde\pi) \;\ge\; L_\pi(\tilde\pi) - C\, D_{KL}^{\max}(\pi, \tilde\pi)
$$

Algorithm 1 maximizes the right-hand side $M_i(\pi) = L_{\pi_i}(\pi) - C\,D_{KL}^{\max}(\pi_i, \pi)$ at every iteration. Both sides are equal at $\pi = \pi_i$, so $\eta(\pi_{i+1}) - \eta(\pi_i) \ge M_i(\pi_{i+1}) - M_i(\pi_i) \ge 0$ (Equation 10), because $\pi_{i+1}$ maximizes $M_i$. The return never decreases. The paper calls this a minorization-maximization (MM) scheme, the same family as expectation maximization.

**Tiny example, computed by `scripts/bound_example.py`.** Use the same $\pi$ and $\tilde\pi$ as before. The constants are as follows:

| Quantity | Value |
|---|---|
| $\epsilon = \max\lvert A_\pi\rvert$ | 0.9863 |
| $C = 4\epsilon\gamma/(1-\gamma)^2$ | 355.07 |
| $D_{TV}^{\max}$ (state 0) | 0.4 |
| $D_{KL}^{\max}$ (state 0) | 0.5108 |
| Theorem 1 bound, Equation 8 | $-48.755$ |
| KL bound, Equation 9 | $-173.32$ |
| True $\eta(\tilde\pi)$ | 7.4312 |

Both bounds hold, but neither helps: they sit far below the true return. The squared-TV form is tighter than the KL form because $0.4^2 = 0.16$ is smaller than 0.5108. The penalty is the problem. With $C = 355.07$, the best penalized step moves the switch probability in state 0 only from 0.5 to 0.5033, and the true return gains 0.0155. A hard limit $D_{KL}^{\max} \le 0.01$ allows 0.5703 and gains 0.3059. Section 4 makes the same observation: the penalty coefficient the theory recommends gives very small steps, so TRPO swaps the penalty for a constraint.

Exercise 2 asks you to compute $D_{TV}^{\max}$, $D_{KL}^{\max}$, $C$, and both bounds, then test them on 200 random MDPs.

### The trust region constraint

**Intuition.** Treat the KL divergence as a budget. Spend as much of it as you like on whatever improves the surrogate most, but don't exceed $\delta$.

**Math.** Section 4 replaces the penalty with a constraint (Equation 11):

$$
\underset{\theta}{\text{maximize}}\; L_{\theta_{\text{old}}}(\theta) \quad \text{subject to} \quad D_{KL}^{\max}(\theta_{\text{old}}, \theta) \le \delta
$$

The maximum over all states means one constraint per state, which is impractical. The paper uses the average over states visited by the old policy instead (Section 4, Equation 12):

$$
\underset{\theta}{\text{maximize}}\; L_{\theta_{\text{old}}}(\theta) \quad \text{subject to} \quad \overline{D}_{KL}^{\rho_{\theta_{\text{old}}}}(\theta_{\text{old}}, \theta) \le \delta
$$

Here $\overline{D}_{KL}^{\rho}(\theta_1, \theta_2) = \mathbb{E}_{s \sim \rho}\big[D_{KL}(\pi_{\theta_1}(\cdot \mid s) \,\|\, \pi_{\theta_2}(\cdot \mid s))\big]$. This is a heuristic. The theorem needs the maximum, and the average is cheaper and, in the authors' experiments, performs about as well (Section 4 and Section 8.1). Section 6 lists the other deviations from the theory: a hard constraint with a tuned $\delta$ replaces the penalty, the average KL replaces the max KL, and the advantage estimation error is ignored.

Section 5 turns the objective into something you can estimate from samples. It replaces the sum over states by an expectation over $s \sim \rho_{\theta_{\text{old}}}$, replaces the advantage $A_{\theta_{\text{old}}}$ by the Q-value $Q_{\theta_{\text{old}}}$ (which changes the objective by a constant, so the gradient is unchanged), and replaces the sum over actions by an importance sampling estimator (Equation 14). The *single path* scheme (Section 5.1) samples whole trajectories from $\pi_{\theta_{\text{old}}}$, so $q = \pi_{\theta_{\text{old}}}$. The *vine* scheme (Section 5.2) branches several actions from each state in a rollout set, which lowers variance but needs a simulator that can reset to arbitrary states. The examples and exercises on this page use the single-path view.

**Tiny example, computed by `scripts/natural_gradient_example.py`.** The next section builds the update for a softmax policy with $K = 3$ actions, $d = 3$ parameters, and $N = 6$ sampled states, and all the numbers below come from there. The script uses the mean over the six states as its surrogate, $L(\theta) = \frac1N \sum_n \sum_a \pi_\theta(a \mid s_n) A(s_n, a)$, which is Equation 14 with the visitation weights replaced by sample weights and with the advantage in place of $Q$. With $\delta = 0.1$, the constraint $\overline{D}_{KL} \le \delta$ rules out the full natural gradient step computed below, whose real KL is 0.1316.

### The natural gradient step

**Intuition.** Maximizing a linear function of the step $\Delta = \theta - \theta_{\text{old}}$ under a quadratic constraint has a closed-form answer. The direction is the gradient, bent by the matrix that measures how much the KL divergence grows in each direction.

**Math.** Section 7 and Appendix C approximate the surrogate to first order and the KL to second order. Let $g = \nabla_\theta L_{\theta_{\text{old}}}(\theta)\big|_{\theta=\theta_{\text{old}}}$ and let $F$ be the Fisher matrix, which is the Hessian of the KL at $\theta_{\text{old}}$ (the paper's $A$; Equation 17 and Appendix C):

$$
\underset{\Delta}{\text{maximize}}\; g^\top \Delta \quad \text{subject to} \quad \tfrac12 \Delta^\top F \Delta \le \delta
$$

The solution points along $s = F^{-1} g$, and the largest step on the constraint boundary has length $\beta$ (Appendix C):

$$
\Delta = \beta s, \qquad \beta = \sqrt{\frac{2\delta}{s^\top F s}}
$$

The step length follows from $\delta = \tfrac12 \beta^2 s^\top F s$. The paper notes that $s^\top F s$ is a by-product of conjugate gradient. Appendix C.1 shows how to compute the Fisher matrix and its products. The Fisher matrix is $J^\top M J$, where $J$ is the Jacobian of the distribution parameters $\mu_\theta(x)$ and $M$ is the Fisher matrix of the distribution in terms of $\mu$ (Equations 56 and 57). The paper computes the Hessian of the KL analytically, and doesn't use the covariance of the policy gradients (the "empirical FIM", Section 6). The analytic version doesn't depend on the sampled action.

The matrix has one entry per pair of parameters, which is too large to store for a neural network. Conjugate gradient solves $Fx = g$ using only products $Fv$, and the paper runs $k = 10$ iterations and computes the products on a 10% subsample of the data (Appendix C.1).

The linear and quadratic models are only local, so the true objective and the true KL can disagree with them at the full step. Appendix C therefore ends with a line search. It starts from the maximal step, shrinks $\beta$ exponentially until the objective improves, and checks that the nonlinear KL constraint holds. The paper reports that without the line search the algorithm occasionally takes large steps that badly degrade performance.

Natural policy gradient is the special case that uses the same linear and quadratic models but a fixed step size $1/\lambda$ and no line search (Section 7). An $\ell_2$ constraint $\tfrac12\|\theta - \theta_{\text{old}}\|^2 \le \delta$ in place of the KL gives the plain policy gradient (Equation 18).

**Tiny example, computed by `scripts/natural_gradient_example.py`.** Start with conjugate gradient on a system you can solve by hand. Let $A = \begin{bmatrix} 4 & 1 \\ 1 & 3 \end{bmatrix}$ and $g = [1, 2]$. The exact solution is $[0.0909, 0.6364]$. CG starts from $x = 0$:

| Iterations | $x$ |
|---|---|
| 1 | `[0.25, 0.5]` |
| 2 | `[0.0909, 0.6364]` |

The residual after 2 iterations is zero, as the theory predicts for a $2 \times 2$ system. For $s = [0.0909, 0.6364]$ the script prints $s^\top A s = 1.3636$, so with $\delta = 0.01$, $\beta = \sqrt{0.02 / 1.3636} = 0.1211$. The quadratic KL of the step $\beta s$ is $\tfrac12 \beta^2 s^\top A s = 0.01$, as designed.

Now the softmax policy with $N = 6$ states, $K = 3$ actions, and $d = 3$ parameters. The gradient is $g = [0.0938, -0.0369, -0.0997]$. The analytic Fisher matrix agrees with a finite-difference Hessian of the mean KL to $1.6 \times 10^{-9}$, and its eigenvalues are 0.0078, 0.0312, and 0.2221. CG with 10 iterations returns $s = [0.1581, -1.5867, -0.8991]$, which equals the direct solve. The script then compares the natural step with a plain gradient step that has the same *predicted* KL:

| $\delta$ | Natural step: real KL, $L$ gain | Gradient step: real KL, $L$ gain |
|---|---|---|
| 0.01 | 0.01116, +0.0677 | 0.01143, +0.0565 |
| 0.1 | 0.13158, +0.2504 | 0.13256, +0.2040 |

At the same predicted KL, the natural step gains more, which is expected because it solves the model problem exactly. The real KL exceeds $\delta$ in all four rows, because the quadratic model underestimates the curvature at these step sizes. That's what the line search is for. At $\delta = 0.1$, the full step has KL 0.1316 and violates the constraint. The first backtrack, with step scale 0.5, has KL 0.0295 and a surrogate gain of +0.1154, and the search accepts it.

Exercise 3 implements CG and $\beta$. Exercise 4 implements the Fisher-vector product and the full update with the line search.

<!-- exercises -->

## Pitfalls

- **Treating the guarantee as the algorithm.** Theorem 1 holds for $D_{KL}^{\max}$ with the penalty coefficient $C$. The practical TRPO uses the average KL, a hard constraint, and sampled advantages, so the monotonic improvement guarantee doesn't carry over. The abstract describes monotonic improvement as something TRPO tends to show in practice, which is an empirical claim.
- **Reading $C$ as a hyperparameter you can tune.** The theory fixes it from $\epsilon$ and $\gamma$, and for $\gamma$ near 1 it grows like $1/(1-\gamma)^2$. In the example, $C = 355$ for $\gamma = 0.9$. With $\gamma = 0.99$ it would be about 110 times larger for the same $\epsilon$. That's why the paper switches to $\delta$.
- **Swapping the arguments of the KL divergence.** The constraint is $D_{KL}(\pi_{\text{old}} \,\|\, \pi_{\text{new}})$. The reverse direction gives a different number (0.3681 against 0.5108 in the example) and a different Fisher matrix away from $\theta_{\text{old}}$.
- **Forgetting that the surrogate is a model.** $L$ uses the old state distribution. Both the objective and the KL constraint are accurate only near $\theta_{\text{old}}$, so check both after the step. A step that satisfies the quadratic KL model can still violate the real constraint, as it does in all four rows of the table above.
- **Skipping the line search.** The full natural gradient step is a prediction. The paper reports occasional catastrophic steps without the line search (Appendix C). Check the improvement and the KL together, and return the old parameters when no candidate qualifies.
- **Building the Fisher matrix from squared gradients.** The paper uses the analytic Hessian of the KL, which averages over all actions in each state. The empirical estimate from outer products of sampled log-probability gradients is a different, noisier matrix (Section 6).
- **Forming the Fisher matrix for a large network.** Use matrix-vector products. Subsampling the data for these products, as in Appendix C.1, keeps the cost near the cost of one gradient.
- **Not centering the features in the Fisher-vector product.** The product uses features centered by the policy's mean feature. Without centering, a shift of all logits in a state, which changes nothing, appears to cost KL. The Exercise 4 tests catch this.
- **Confusing TRPO's constraint with a gradient clip.** A clip bounds the Euclidean norm of the update (Equation 18 is the $\ell_2$ case). The KL constraint bounds the change in the policy's action distribution, and so it scales each parameter direction by how much that direction moves the policy.

## Further reading

- [Proximal Policy Optimization Algorithms](/ppo/), by Schulman et al., replaces the hard constraint with a clipped surrogate objective that needs only first-order optimization.
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](/gae/), by Schulman et al., is the advantage estimator that the TRPO and PPO implementations use.
- [Approximately Optimal Approximate Reinforcement Learning](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/KakadeLangford-icml2002.pdf), by Kakade and Langford, is the conservative policy iteration paper that Section 2 of TRPO extends.
- [Spinning Up: Trust Region Policy Optimization](https://spinningup.openai.com/en/latest/algorithms/trpo.html), by OpenAI, is a short derivation with a reference implementation.
