---
id: "pg-theorem"
title: "Policy Gradient Methods for Reinforcement Learning with Function Approximation"
authors: "Sutton, McAllester, Singh, and Mansour"
year: 2000
venue: "NIPS 1999 (Advances in Neural Information Processing Systems 12)"
arxiv: null
url: "https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html"
track: rl
tier: extended
est_hours: 5
status: full
summary: "This paper proves that the gradient of expected return with respect to the parameters of a stochastic policy is a sum over states and actions of the policy's own gradient times the action value, weighted by how often the policy visits each state. The weights are held fixed, so the formula never needs the derivative of the state distribution, which depends on the unknown dynamics of the environment. The paper then shows that a learned critic gives the exact same gradient when its features are the gradient of the log policy and it is fit by least squares, and uses that result to prove that a form of policy iteration with function approximation converges to a local optimum."
related_course: "rl: policy gradients"
prereqs: []
background: [mdp, value-function, policy-gradient-theorem, softmax, least-squares]
course_links:
  - title: "The policy gradient"
    url: "https://rl.ifkash.dev/lessons/031-the-policy-gradient"
  - title: "Where the gradient comes from"
    url: "https://rl.ifkash.dev/lessons/032-where-the-gradient-comes-from"
self_check:
  - q: 'Theorem 1 has no term for how the policy changes the distribution of visited states. Why is that term not needed, and what would you have to know to compute it directly?'
    a: 'The state weights $d^\pi(s)$ do depend on $\theta$, but the appendix proof never differentiates them. It unrolls the recursion for $\partial V^\pi/\partial\theta$ through the dynamics, which builds the weights up from the gradient of the policy at each visited state. What remains is $\sum_s d^\pi(s)\sum_a \partial\pi(s,a)/\partial\theta\; Q^\pi(s,a)$. Computing $\partial d^\pi/\partial\theta$ directly would need the transition probabilities, which an agent learning from experience does not have.'
    see: policy-gradient-theorem
  - q: 'In the two-state example on this page, the discount is $\gamma = 0.5$ and the weights are $d^\pi = (1.5, 0.5)$. What is the sum of $d^\pi$ over states, and why is it not 1?'
    a: 'The sum is $2 = 1/(1-\gamma)$. In the start-state formulation, $d^\pi(s) = \sum_t \gamma^t \Pr(s_t = s \mid s_0, \pi)$ is a discounted count of visits (Section 1), not a probability distribution. Dividing by $1/(1-\gamma)$ gives a distribution, and then the gradient is $1/(1-\gamma)$ times an expectation under that distribution.'
    see: mdp
  - q: 'The critic $f_w$ has fewer parameters than there are state-action pairs, so it cannot equal $Q^\pi$. Why does Theorem 2 still give the exact gradient?'
    a: 'Equation 3 says that the weighted error $Q^\pi - f_w$ is orthogonal to $\partial f_w/\partial w$, and the compatibility condition (Equation 4) makes that vector equal to $\nabla_\theta \log\pi$. So the error is orthogonal to the only directions that the gradient formula multiplies it by, and the error term drops out (Equation 6). The least-squares fit of $w$ is what enforces Equation 3, and it only needs $f_w$ to get the relative values of actions right in each state.'
    see: least-squares
---

## The problem

Before this paper, the standard way to use function approximation in reinforcement learning was the value-function approach: learn $Q(s, a)$ with a function approximator, and act greedily with respect to it. The paper's introduction lists the problems. The greedy policy is deterministic, but the best policy is often stochastic. An arbitrarily small change in the estimated value of an action can flip which action is chosen, so the policy changes discontinuously as the values change. The paper cites results that Q-learning, Sarsa, and dynamic programming can fail to converge to any policy, even for simple MDPs and simple function approximators.

The alternative is to represent the policy directly, $\pi(s, a, \theta)$, and to move $\theta$ along the gradient of the performance measure $\rho$ (Equation 1, $\Delta\theta \approx \alpha\,\partial\rho/\partial\theta$). This page writes the performance measure as $J$ in the code and $\rho$ when it follows the paper. Small changes in $\theta$ then cause small changes in the policy. The obstacle is that $\rho$ depends on $\theta$ in two ways. The policy changes which action is chosen in each state, and it also changes which states the agent visits. The second effect depends on the transition probabilities of the environment, which the agent doesn't know.

Here is what goes wrong with a plausible shortcut. Take a 2-state MDP where each action moves the agent deterministically. In state 0, action 0 stays in state 0 and action 1 moves to state 1, both with reward 0. In state 1, action 0 stays and pays reward 1, and action 1 moves back to state 0 with reward 0. The discount is $\gamma = 0.5$, the start state is 0, and the policy is a softmax with logits $\theta_{s,a} = 0$. If you differentiate only the policy and weight every state equally, the gradient for state 1 comes out as $\pm 0.3125$. The true gradient, computed by `scripts/theorem1_example.py`, is $\pm 0.15625$. The shortcut is off by a factor of 2 in state 1 because the agent visits state 1 half as often as the shortcut assumes. Theorem 1 supplies the missing weights without differentiating them.

## Key ideas

This page uses the paper's notation: a policy $\pi(s, a)$ with parameter vector $\theta$, state-action value $Q^\pi(s, a)$, state weights $d^\pi(s)$, and critic $f_w(s, a)$ with parameter $w$. The paper treats two formulations (Section 1): average reward per step, and a designated start state $s_0$ with discount $\gamma \in [0, 1]$. The paper says its results apply to both. This page uses the start-state formulation, where $\rho(\pi) = E\{\sum_{t=1}^\infty \gamma^{t-1} r_t \mid s_0, \pi\}$, $Q^\pi(s, a) = E\{\sum_{k=1}^\infty \gamma^{k-1} r_{t+k} \mid s_t = s, a_t = a, \pi\}$, and $d^\pi(s) = \sum_{t=0}^\infty \gamma^t \Pr\{s_t = s \mid s_0, \pi\}$. The weights $d^\pi$ sum to $1/(1-\gamma)$ and are not a probability distribution. The code names the objective `J`. The paper keeps the parameterization $\pi(s, a, \theta)$ general, except for the Gibbs example in Section 3. This page uses a tabular softmax $\pi(s,a) = e^{\theta_{s,a}}/\sum_b e^{\theta_{s,b}}$ for Theorem 1 and a linear Gibbs policy for Theorem 2. Run any script from `content/papers/pg-theorem/scripts/` to reproduce the numbers.

### Theorem 1: the policy gradient theorem

**Intuition.** Changing $\theta$ has two effects: the action probabilities in each state change, and the visitation of states changes. The theorem says that, for the first-order change in $\rho$, you can act as if only the first effect exists. You weight each state by how often the current policy visits it, and you multiply the change in the action probabilities by the action's value. The change in state visitation is already accounted for inside $Q^\pi$.

**Math.** Section 1, Theorem 1, Equation 2. For any MDP, in either formulation:

$$
\frac{\partial \rho}{\partial \theta} = \sum_s d^\pi(s) \sum_a \frac{\partial \pi(s, a)}{\partial \theta}\, Q^\pi(s, a)
$$

The paper stresses that no term of the form $\partial d^\pi(s)/\partial\theta$ appears. That makes the gradient easy to estimate by sampling. If states are drawn by following $\pi$, then $\sum_a \frac{\partial\pi(s,a)}{\partial\theta} Q^\pi(s,a)$ is an unbiased estimate of $\partial\rho/\partial\theta$, which is how the paper puts it. In the start-state formulation, the sampled states also need the discount weights in $d^\pi$, which the pitfalls section covers. Replacing $Q^\pi(s_t, a_t)$ with the sampled return $R_t$ gives Williams's REINFORCE update, $\Delta\theta_t \propto \frac{\partial\pi(s_t,a_t)}{\partial\theta}\frac{1}{\pi(s_t,a_t)} R_t$. The factor $1/\pi(s_t, a_t)$ corrects for the oversampling of actions that $\pi$ prefers. The paper proves the theorem in its appendix. The proof differentiates $V^\pi(s) = \sum_a \pi(s,a)Q^\pi(s,a)$ and expands $\partial Q^\pi/\partial\theta$ into a term in $\partial V^\pi(s')/\partial\theta$ at the next state, because the rewards and transition probabilities don't depend on $\theta$. Unrolling that recursion produces the discounted visit weights $\sum_k \gamma^k \Pr(s \to x, k, \pi)$ and leaves only $\partial\pi/\partial\theta$ terms.

For a tabular softmax policy, $\partial\pi(s,a)/\partial\theta_{s,b} = \pi(s,a)(\mathbb{1}[a=b] - \pi(s,b))$. Because $\sum_a \partial\pi(s,a)/\partial\theta = 0$, Equation 2 simplifies to

$$
\frac{\partial \rho}{\partial \theta_{s,b}} = d^\pi(s)\,\pi(s,b)\,\big(Q^\pi(s,b) - V^\pi(s)\big)
$$

where $V^\pi(s) = \sum_a \pi(s,a) Q^\pi(s,a)$. This is the form that the code uses.

**Tiny example, computed by `scripts/theorem1_example.py`.** Use the 2-state MDP from the previous section, with $\gamma = 0.5$ and a uniform policy ($\theta = 0$). Solving the linear equations gives $V^\pi = (0.25, 0.75)$, so $\rho = V^\pi(s_0) = 0.25$. The action values are $Q^\pi(0, \cdot) = (0.125, 0.375)$ and $Q^\pi(1, \cdot) = (1.375, 0.125)$. The state weights are $d^\pi = (1.5, 0.5)$, which sum to $2 = 1/(1-\gamma)$. You can check $d^\pi$ by hand: $d_0 = 1 + 0.5(0.5 d_0 + 0.5 d_1)$ and $d_1 = 0.5(0.5 d_0 + 0.5 d_1)$, so $d_1 = d_0/3$ and $d_0 = 1.5$.

The gradient with respect to $\theta_{s,b}$ is $d^\pi(s)\,\pi(s,b)\,(Q^\pi(s,b) - V^\pi(s))$. In state 0, that is $1.5 \cdot 0.5 \cdot (0.125 - 0.25) = -0.09375$ for action 0 and $+0.09375$ for action 1. In state 1, it is $0.5 \cdot 0.5 \cdot (1.375 - 0.75) = +0.15625$ for action 0 and $-0.15625$ for action 1. Central finite differences of $\rho$ with step $10^{-6}$ agree to within $2 \times 10^{-11}$. On a random 3-state MDP with $\gamma = 0.9$, the difference is below $10^{-9}$.

Exercise 1 asks you to compute $d^\pi$, $Q^\pi$, and this gradient, and to check it against finite differences.

### Theorem 2: compatible function approximation

**Intuition.** Theorem 1 needs $Q^\pi$, which the agent has to estimate. If you plug in a learned approximation $f_w$, you get the right gradient only if the approximation error doesn't push the gradient off course. Theorem 2 picks the form of $f_w$ so that the error cannot matter. The error only has to be uncorrelated with the gradient of the log policy, and a least-squares fit makes it so.

**Math.** Section 2, Theorem 2, Equations 3 to 5. Let $f_w : \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ approximate $Q^\pi$, learned by following $\pi$ with updates $\Delta w_t \propto [\hat Q^\pi(s_t,a_t) - f_w(s_t,a_t)]\,\partial f_w(s_t,a_t)/\partial w$. At a local optimum of that process:

$$
\sum_s d^\pi(s) \sum_a \pi(s,a)\,\big[Q^\pi(s,a) - f_w(s,a)\big]\,\frac{\partial f_w(s,a)}{\partial w} = 0 \qquad \text{(3)}
$$

Suppose $f_w$ is compatible with the policy parameterization:

$$
\frac{\partial f_w(s,a)}{\partial w} = \frac{\partial \pi(s,a)}{\partial \theta}\,\frac{1}{\pi(s,a)} \qquad \text{(4)}
$$

Then replacing $Q^\pi$ by $f_w$ in Theorem 1 gives the same gradient:

$$
\frac{\partial \rho}{\partial \theta} = \sum_s d^\pi(s) \sum_a \frac{\partial \pi(s,a)}{\partial \theta}\, f_w(s,a) \qquad \text{(5)}
$$

The proof combines (3) and (4) to get $\sum_s d^\pi(s)\sum_a \frac{\partial\pi(s,a)}{\partial\theta}[Q^\pi(s,a) - f_w(s,a)] = 0$ (Equation 6), then subtracts this zero from Theorem 1. A footnote credits Tsitsiklis with the remark that $f_w$ linear in these features may be the only way to satisfy (4).

Section 3 works out one case. For a Gibbs policy $\pi(s,a) = e^{\theta^\top\phi_{sa}}/\sum_b e^{\theta^\top\phi_{sb}}$, condition (4) requires $\partial f_w/\partial w = \phi_{sa} - \sum_b \pi(s,b)\phi_{sb}$, so $f_w(s,a) = w^\top[\phi_{sa} - \sum_b \pi(s,b)\phi_{sb}]$. That is $f_w$ linear in the same features as the policy, normalized to mean zero in each state. Because $\sum_a \pi(s,a) f_w(s,a) = 0$, the paper notes that $f_w$ is better thought of as an approximation of the advantage $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$ than of $Q^\pi$. The theorems still hold if you add any function $v(s)$ of the state to $f_w$, because $\sum_a \partial\pi(s,a)/\partial\theta = 0$. The choice of $v$ does not affect any theorem but can change the variance of the estimators, like a baseline in earlier work. This is the paper's justification for using advantages as the target of value-function learning.

This page writes $\psi_{sa} = \phi_{sa} - \sum_b \pi(s,b)\phi_{sb}$ for the compatible features. The paper does not name them.

**Tiny example, computed by `scripts/compatible_example.py`.** Use the same MDP and a policy with one parameter: $\phi_{s,1} = 1$ and $\phi_{s,0} = 0$ in both states, so $\pi(s,1) = \sigma(\theta)$. At $\theta = 0$ the policy is uniform, and $\psi_{s,0} = -0.5$ and $\psi_{s,1} = +0.5$ in both states. Equation 3 with weights $d^\pi(s)\pi(s,a)$ is a one-dimensional least-squares problem:

$$
w = \frac{\sum d\pi\,\psi\,Q}{\sum d\pi\,\psi^2} = \frac{0.09375 - 0.15625}{0.5} = -0.125
$$

The numerator is $-0.0625$: state 0 contributes $0.75 \cdot (-0.5 \cdot 0.125 + 0.5 \cdot 0.375) = 0.09375$ and state 1 contributes $0.25 \cdot (-0.5 \cdot 1.375 + 0.5 \cdot 0.125) = -0.15625$. The denominator is $\sum_s d(s) \cdot 0.25 = 0.5$. The critic is $f_w = (+0.0625, -0.0625)$ in each state, which is far from $Q^\pi$ (for example, $Q^\pi(1, 0) = 1.375$). It even ranks the actions the wrong way in state 0, where $Q^\pi$ prefers action 1 and $f_w$ prefers action 0. The gradient with $f_w$ is $\sum d\pi\,\psi\,f_w = w \cdot 0.5 = -0.0625$. The true gradient, from Theorem 1 mapped to the single parameter, is $g_{0,1} + g_{1,1} = 0.09375 - 0.15625 = -0.0625$. They match.

The script also checks a case with 3 states, 3 actions, and 2 parameters, where the maximum gap $|Q^\pi - f_w|$ is 1.45. The two gradients are both $(0.95416, 1.10699)$.

Exercise 2 asks you to build $\psi$, fit $w$, and verify Equation 5.

### Theorem 3: convergence of policy iteration

**Intuition.** Once Theorem 2 says that the actor update $\theta \leftarrow \theta + \alpha\sum d\,\partial\pi/\partial\theta\,f_w$ follows the true gradient, standard results for gradient ascent with a diminishing step size apply.

**Math.** Section 4, Theorem 3. Let $\pi$ and $f_w$ be differentiable function approximators that satisfy the compatibility condition (4), with second derivatives of the policy bounded: $\max_{\theta,s,a,i,j}\left|\partial^2\pi(s,a)/\partial\theta_i\partial\theta_j\right| < B < \infty$. Let $\{\alpha_k\}$ be any step-size sequence with $\alpha_k \to 0$ and $\sum_k \alpha_k = \infty$. For any MDP with bounded rewards, take any $\theta_0$, let $\pi_k = \pi(\cdot,\cdot,\theta_k)$, let $w_k$ solve Equation 3 for $\pi_k$, and update

$$
\theta_{k+1} = \theta_k + \alpha_k \sum_s d^{\pi_k}(s) \sum_a \frac{\partial \pi_k(s,a)}{\partial \theta}\, f_{w_k}(s,a).
$$

Then $\lim_{k\to\infty} \partial\rho(\pi_k)/\partial\theta = 0$. The proof applies Theorem 2, which makes the update the true gradient direction, and then Proposition 3.5 on page 96 of Bertsekas and Tsitsiklis (1996), Neuro-Dynamic Programming. This page did not check that proposition. The result is convergence to a local optimum, not a global one, and the iteration needs the exact fit of $w_k$ at each step.

**Tiny example, computed by `scripts/policy_iteration_example.py`.** Run the iteration on the one-parameter policy above, with $\alpha_k = 4/\sqrt{k+1}$ (a choice for this example; it satisfies both conditions) and the exact $Q^\pi$ at each step:

| $k$ | $\theta_k$ | $\rho(\pi_k)$ | $\lvert\partial\rho/\partial\theta\rvert$ |
|---|---|---|---|
| 0 | 0.0000 | 0.250000 | 0.062500 |
| 1 | -0.2500 | 0.262452 | 0.036245 |
| 10 | -0.5183 | 0.267889 | 0.003852 |
| 100 | -0.5493 | 0.267949 | 0.000003 |
| 1000 | -0.5493 | 0.267949 | 0.000000 |

The return rises and the gradient goes to 0, as the theorem says. The stopping point is a stationary point of $\rho$ for this policy family. A scan of $\theta$ over $[-8, 8]$ in the same script finds the same maximum, 0.267949, so here the stopping point is also the best policy in the family. The theorem itself only promises a stationary point.

### What the paper shows

The paper has no experiments. Its contributions are the three theorems and the discussion in Sections 1 and 3: the policy gradient theorem, the compatible form of the critic, the observation that advantages are the right target, and the first convergence proof for a form of policy iteration with arbitrary differentiable function approximation. The Introduction compares the result with Jaakkola, Singh, and Jordan (1995), who proved a similar result for tabular POMDPs; with VAPS (Baird and Moore, 1999), which ascends a mixture of performance and value-function error and so need not reach a locally optimal policy; and with Gordon's fitted value iteration, which converges but not to a locally optimal policy.

<!-- exercises -->

## Pitfalls

- **Reading $d^\pi$ as a probability distribution.** In the start-state formulation, $d^\pi$ sums to $1/(1-\gamma)$ (2 in the example). Treating it as a distribution scales the gradient by $1-\gamma$. Exercise 1's mutant that normalizes $d^\pi$ checks this.
- **Dropping $\gamma^t$ in REINFORCE.** The sampled estimate is unbiased only with the factor $\gamma^t$ on step $t$ (the weights in $d^\pi$ come from the discounted visits). Many implementations leave it out and use the undiscounted state distribution instead. That is a different update, and it estimates a different quantity. In `scripts/reinforce_example.py`, leaving the factor out leaves a maximum error of 5.83 against the exact gradient, where the full estimator has 0.021 with 20,000 episodes.
- **Using $Q^\pi$ at the wrong policy.** Theorem 1 holds at the policy that generated the data. If you reuse $Q$ values from earlier parameters, the gradient is stale.
- **Treating Theorem 2 as a statement about any critic.** The critic must satisfy both Equation 3 (a converged weighted least-squares fit, with weights $d^\pi(s)\pi(s,a)$) and Equation 4 (the compatible form). A critic with other features, or one trained for a few steps, gives a biased gradient. Exercise 2's tests check that the fit satisfies Equation 3 and that a wrong weighting fails.
- **Expecting $f_w$ to estimate $Q^\pi$.** Compatible features are centered, so $f_w$ has mean zero in every state. It approximates the advantage up to the part of the advantage that the features can represent, and it can be far from $Q^\pi$ in absolute value, as in the example above.
- **Reading Theorem 3 as a result about practical actor-critic.** The theorem assumes an exact solution of Equation 3 at each iteration, a compatible critic, bounded second derivatives, and a diminishing step size. It is a convergence result for an idealized algorithm. It says nothing about the sampling noise, bootstrapped critics, or neural network critics that are used in practice.
- **Local optima.** The conclusion is $\partial\rho/\partial\theta \to 0$. It doesn't mean that the policy is optimal, and with a restricted policy class the best policy may not be reachable.

## Further reading

- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd edition), Chapter 13: the policy gradient theorem derived with an episodic start-state objective, and REINFORCE with a baseline.
- Williams, "Simple statistical gradient-following algorithms for connectionist reinforcement learning" (1992): the REINFORCE algorithm that the paper relates to Theorem 1.
- Konda and Tsitsiklis, "Actor-critic algorithms" (NIPS 1999): the independent, contemporary result that the paper mentions.
- OpenAI Spinning Up, "Intro to policy optimization": a derivation of the simplest policy gradient, with the reward-to-go and baseline forms.
- [Trust region policy optimization](/trpo/), [Generalized advantage estimation](/gae/), and [Proximal policy optimization](/ppo/): the later papers on this site that build on Theorem 1.
