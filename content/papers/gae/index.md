---
id: "gae"
title: "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
authors: "Schulman et al."
year: 2015
venue: "ICLR 2016"
arxiv: "1506.02438"
url: "https://arxiv.org/abs/1506.02438"
track: rl
tier: core
est_hours: 3
status: full
summary: "Generalized advantage estimation (GAE) scores each action in a trajectory with an exponentially weighted sum of one-step TD residuals, where the weight decays by a factor of gamma times lambda per step. The parameter lambda trades bias for variance: lambda of 0 gives the one-step TD residual, which has low variance but is biased when the value function is wrong, and lambda of 1 gives the full discounted return minus the value baseline, which is unbiased but noisy. The paper combines this estimator with trust region updates for both the policy and the value function, and uses the result to learn running gaits for simulated 3D robots."
related_course: "rl: advantage estimation"
prereqs:
  - id: trpo
    reason: "GAE supplies the advantage estimates that the TRPO surrogate objective and trust-region step consume."
background: [td-lambda, bias-variance-trade-off, value-function]
course_links:
  - title: "Generalized advantage estimation"
    url: "https://rl.ifkash.dev/lessons/034-generalized-advantage-estimation"
self_check:
  - q: 'If $V = V_{\pi,\gamma}$ exactly, why is the TD residual $\delta_t^V$ an unbiased estimate of the advantage? What goes wrong when $V$ is only approximate, and does $\text{GAE}(\gamma, 1)$ suffer from the same problem?'
    a: 'With the exact value function, $\mathbb{E}[r_t + \gamma V(s_{t+1}) \mid s_t, a_t] = Q_{\pi,\gamma}(s_t, a_t)$, so subtracting $V(s_t)$ gives $A_{\pi,\gamma}$ (Equation 10). An approximate $V$ makes $\gamma V(s_{t+1})$ a biased stand-in for the future return, and the bias goes straight into the policy gradient. $\text{GAE}(\gamma, 1)$ sums the actual rewards, so $V$ appears only as the baseline $-V(s_t)$. A baseline that depends on the state alone does not bias the gradient (Proposition 1), so $\text{GAE}(\gamma, 1)$ is $\gamma$-just for any $V$ (Equation 18).'
    see: value-function
  - q: 'With $\lambda = 0.5$, what weight does GAE put on the 1-step, 2-step, and 3-step advantage estimators $\hat A_t^{(1)}, \hat A_t^{(2)}, \hat A_t^{(3)}$, and what is the total weight on all estimators with $k \ge 4$? How does this differ from TD($\lambda$)?'
    a: 'The weights are $(1-\lambda)\lambda^{k-1}$, which gives 0.5, 0.25, and 0.125. The weights for $k \ge 4$ sum to $\lambda^3 = 0.125$, because the weights add up to 1. The weighting is the same as in TD($\lambda$), but TD($\lambda$) mixes $k$-step estimates of the value function, whereas GAE mixes $k$-step estimates of the advantage (Section 3, after Equation 16).'
    see: td-lambda
  - q: 'TRPO needs advantages of the old policy in its surrogate objective. Where does GAE plug in, and which parts of the TRPO step stay the same?'
    a: 'GAE replaces the advantage estimate $\hat A_n$ inside $L_{\theta_{\text{old}}}(\theta)$ (Equation 31 in this paper). That changes the gradient $g$, which becomes less noisy for $\lambda < 1$. The Fisher matrix, the conjugate gradient solve, the step length $\beta = \sqrt{2\delta / (s^\top F s)}$, and the line search are all unchanged.'
    see: trpo
---

## The problem

A policy gradient method needs a number $\Psi_t$ that says how good the action at step $t$ was, and it weights $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ by that number (Section 2, Equation 1). The paper lists six choices for $\Psi_t$. They run from the total reward of the trajectory, through the reward following the action with or without a baseline, to $Q_\pi$, the advantage $A_\pi$, and the TD residual. The advantage gives almost the lowest variance, but nobody knows it, so you have to estimate it. Both ways of estimating it fail in a different direction.

- **Sampled returns have high variance.** The sum of rewards after step $t$ is unbiased, but it adds up every random reward and action choice in the future. The paper notes that the variance of the gradient estimate scales unfavorably with the time horizon, because an action's effect is confounded with the effects of earlier and later actions (Section 1).
- **A learned value function trades that variance for bias.** The TD residual $r_t + \gamma V(s_{t+1}) - V(s_t)$ looks one step ahead and then trusts $V$. It has low variance, but when $V$ is wrong, so is the gradient. The paper calls this bias more pernicious than variance, because with unlimited samples it can still make the algorithm converge to a poor solution (Section 1). The paper reports that the one-step estimate, $\lambda = 0$, gave excessive bias and poor performance in its experiments (Section 7 and Appendix A.2).

You want a dial between the two extremes. The table shows the problem on a toy task. It has 5 states, 2 actions, $\gamma = 0.95$, and reward noise with standard deviation 0.3. Each row estimates $A_{\pi,\gamma}(s_0, a_0) = -0.2196$ from 20,000 sampled trajectories. The column "wrong $V$" adds a fixed per-state error to the exact value function. The bias column measures the bias that matters for the gradient, which excludes the state-only baseline (the script explains why). The numbers come from `scripts/bias_variance.py`:

| Estimator | Bias with exact $V$ | Std with exact $V$ | Bias with wrong $V$ | Std with wrong $V$ |
|---|---|---|---|---|
| One-step residual ($\lambda = 0$) | +0.002 | 0.357 | +0.113 | 0.567 |
| Return minus $V$ ($\lambda = 1$) | −0.008 | 1.999 | −0.008 | 1.999 |

The exact value function makes both estimators unbiased, as the theory says, but the sampled return is about 5.6 times noisier. The wrong value function leaves the return estimator alone and biases the one-step one. GAE interpolates between these two rows.

## Key ideas

This page uses the paper's notation: reward $r_t$, discount $\gamma$, value estimate $V$, TD residual $\delta_t^V$, $k$-step estimator $\hat A_t^{(k)}$, and the generalized advantage estimator $\hat A_t^{\text{GAE}(\gamma,\lambda)}$. The paper writes sums to $\infty$ on an episode that ends in an absorbing state, and it treats $\gamma$ as a variance-reduction parameter of an undiscounted problem (Section 2). In the examples and exercises, an episode of $T$ steps ends with $V(s_T) = 0$, so every sum stops at the last step. The page drops the superscript $V$ and writes $\delta_t$ where the context is clear. All numbers come from scripts in `scripts/`, and you can run any of them from the paper folder.

### The TD residual as an advantage estimate

**Intuition.** After you take action $a_t$ and see $r_t$ and $s_{t+1}$, you can ask whether the action did better than the policy's average at $s_t$. The value function already knows the average, so you can compare the immediate reward plus the value of where you landed against the value of where you started.

**Math.** With a learned value function $V$, the paper defines the TD residual (Section 3):

$$
\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

If $V = V_{\pi,\gamma}$, then $\delta_t^V$ is an unbiased estimator of the discounted advantage $A_{\pi,\gamma}$ (Equation 10), and it is $\gamma$-just. An estimator $\hat A_t$ is $\gamma$-just when swapping it for $A_{\pi,\gamma}$ in the policy gradient leaves the expected gradient unchanged (Definition 1, Equation 7). Proposition 1 gives a sufficient condition: $\hat A_t = Q_t - b_t$, where $Q_t$ is an unbiased estimate of $Q_{\pi,\gamma}(s_t, a_t)$ and the baseline $b_t$ depends only on what came before $a_t$. The proof is in Appendix B. For any other $V$, $\delta_t^V$ is biased.

**Tiny example, computed by `scripts/gae_example.py`.** The episode has four steps with $\gamma = 0.9$, rewards $r = [1, 0, 0, 2]$, and value estimates $V = [0.5, 0.4, 0.6, 0.3]$. The episode ends after step 3, so $V(s_4) = 0$. Check the residuals by hand: $\delta_0 = 1 + 0.9 \cdot 0.4 - 0.5 = 0.86$, $\delta_1 = 0 + 0.9 \cdot 0.6 - 0.4 = 0.14$, $\delta_2 = 0 + 0.9 \cdot 0.3 - 0.6 = -0.33$, and $\delta_3 = 2 + 0 - 0.3 = 1.7$. The script prints `[0.86, 0.14, -0.33, 1.7]`. The last step is the only one that gets credit for the reward of 2, and the earlier steps don't see it.

Exercise 1 asks you to compute these residuals as part of the estimator.

### The $k$-step estimators

**Intuition.** The one-step residual trusts $V$ immediately after one step. You can trust it later instead. Add up more real rewards, and only then fall back on $V$. The longer you wait, the less the error in $V$ matters, because it's discounted by $\gamma^k$.

**Math.** Summing $k$ discounted residuals gives $\hat A_t^{(k)}$ (Section 3, Equations 11 to 14):

$$
\hat A_t^{(k)} = \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V = -V(s_t) + r_t + \gamma r_{t+1} + \dots + \gamma^{k-1} r_{t+k-1} + \gamma^k V(s_{t+k})
$$

The two forms are equal because the $V$ terms telescope. The estimate is $k$ real rewards, then the bootstrap $\gamma^k V(s_{t+k})$, then the baseline $-V(s_t)$. The paper notes that the bias gets smaller as $k$ grows, because $\gamma^k V(s_{t+k})$ is more heavily discounted and the baseline $-V(s_t)$ doesn't affect the bias. As $k \to \infty$ the estimator becomes the empirical return minus the baseline (Equation 15):

$$
\hat A_t^{(\infty)} = \sum_{l=0}^{\infty} \gamma^l \delta_{t+l}^V = -V(s_t) + \sum_{l=0}^{\infty} \gamma^l r_{t+l}
$$

**Tiny example, computed by `scripts/gae_example.py`.** Use the same episode. Here $\hat A_0^{(2)} = 0.86 + 0.9 \cdot 0.14 = 0.986$, and $\hat A_0^{(3)} = 0.986 + 0.81 \cdot (-0.33) = 0.7187$. With only four steps, $\hat A_0^{(4)}$ is the whole return minus the baseline: $1 + 0.9^3 \cdot 2 - 0.5 = 1.958$. The table lists all four estimates:

| $k$ | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| $\hat A_0^{(k)}$ | 0.8600 | 0.9860 | 0.7187 | 1.9580 |

Exercise 3 asks you to implement $\hat A_t^{(k)}$ from the second form of Equation 14.

### GAE: an exponentially weighted average

**Intuition.** Don't pick $k$. Average all the $k$-step estimators, and give shorter ones more weight by a factor of $\lambda$ for each extra step. A single number $\lambda$ then controls how far the estimator looks before it trusts $V$.

**Math.** The generalized advantage estimator is the exponentially weighted average of the $k$-step estimators (Section 3):

$$
\hat A_t^{\text{GAE}(\gamma,\lambda)} = (1-\lambda)\left(\hat A_t^{(1)} + \lambda \hat A_t^{(2)} + \lambda^2 \hat A_t^{(3)} + \dots\right)
$$

Expand each $\hat A_t^{(k)}$ into residuals and collect the coefficient of each $\delta_{t+l}$. The geometric series $(1-\lambda)(\lambda^l + \lambda^{l+1} + \dots) = \lambda^l$ turns the double sum into a single one (Equation 16):

$$
\hat A_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \, \delta_{t+l}^V
$$

The estimator is a discounted sum of TD residuals with discount $\gamma\lambda$. That gives the backward recursion that implementations use, $\hat A_t = \delta_t + \gamma\lambda \hat A_{t+1}$, which computes every step in one pass. The two ends of the dial are the special cases in Equations 17 and 18:

$$
\text{GAE}(\gamma, 0): \hat A_t = \delta_t, \qquad\qquad \text{GAE}(\gamma, 1): \hat A_t = \sum_{l=0}^{\infty} \gamma^l \delta_{t+l} = \sum_{l=0}^{\infty} \gamma^l r_{t+l} - V(s_t)
$$

The paper states that $\text{GAE}(\gamma, 1)$ is $\gamma$-just regardless of the accuracy of $V$, but it has high variance because it sums many terms. $\text{GAE}(\gamma, 0)$ is $\gamma$-just only when $V = V_{\pi,\gamma}$, and otherwise it has bias, but it typically has much lower variance (Section 3). Values of $\lambda$ between 0 and 1 compromise. The construction mirrors TD($\lambda$), but TD($\lambda$) estimates the value function, where GAE estimates the advantage.

**Tiny example, computed by `scripts/gae_example.py`.** Take $\gamma = 0.9$ and $\lambda = 0.5$, so $\gamma\lambda = 0.45$. Work backward from the end: $\hat A_3 = \delta_3 = 1.7$, then $\hat A_2 = -0.33 + 0.45 \cdot 1.7 = 0.435$, then $\hat A_1 = 0.14 + 0.45 \cdot 0.435 = 0.33575$, then $\hat A_0 = 0.86 + 0.45 \cdot 0.33575 = 1.01109$. The weights $(1-\lambda)\lambda^{k-1}$ on $\hat A_0^{(1)}, \hat A_0^{(2)}, \hat A_0^{(3)}$ are 0.5, 0.25, and 0.125, and the remaining 0.125 goes to $\hat A_0^{(4)}$, because the episode has no more steps. The weighted mixture is $0.5 \cdot 0.86 + 0.25 \cdot 0.986 + 0.125 \cdot 0.7187 + 0.125 \cdot 1.958 = 1.01109$, which agrees with the recursion. For the special cases, the script prints $[0.86, 0.14, -0.33, 1.7]$ for $\lambda = 0$ and $[1.958, 1.22, 1.2, 1.7]$ for $\lambda = 1$. The second is the discounted return minus $V$, which starts at $2.458 - 0.5 = 1.958$.

Exercise 1 implements the recursion. Exercise 2 extends it to a buffer with several episodes. Exercise 3 checks that the mixture equals Equation 16.

### Two parameters: $\gamma$ and $\lambda$ control different biases

**Intuition.** The discount $\gamma$ changes which quantity you estimate, so it biases the gradient no matter how good $V$ is. The parameter $\lambda$ only changes how the estimate of that quantity is built from samples and $V$, so it biases the gradient only to the extent $V$ is wrong.

**Math.** The paper treats $\gamma$ as a variance-reduction parameter. Using $\gamma < 1$ means estimating the discounted advantage $A_{\pi,\gamma}$, which is a biased version of the undiscounted one (Section 2, Equations 4 to 6). The paper says that $\gamma < 1$ introduces bias into the policy gradient regardless of the value function's accuracy, and $\lambda < 1$ introduces bias only when the value function is inaccurate (Section 3). It also reports that the best $\lambda$ is much lower than the best $\gamma$ empirically, probably because $\lambda$ introduces far less bias for a reasonably accurate $V$. The GAE policy gradient is (Equation 19):

$$
g^\gamma \approx \mathbb{E}\left[\sum_{t=0}^{\infty} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \, \hat A_t^{\text{GAE}(\gamma,\lambda)}\right]
$$

The equality holds when $\lambda = 1$. For $\lambda < 1$ it is a biased estimate of $g^\gamma$, which is the gradient of the discounted objective.

**Tiny example, computed by `scripts/bias_variance.py`.** The same toy task as in the problem statement, with a value function that has a fixed error of up to about 1.3 in a state. The script computes the exact bias of GAE in closed form, using $\mathbb{E}[\delta_l] = \gamma\,\mathbb{E}\,e(s_{l+1}) - \mathbb{E}\,e(s_l)$ for the value error $e$, and compares it with the sampled mean. The two agree to within sampling noise (the standard error is about 0.014):

| $\lambda$ | Std, exact $V$ | Bias, wrong $V$ (sampled) | Bias, wrong $V$ (closed form) | Std, wrong $V$ |
|---|---|---|---|---|
| 0 | 0.357 | +0.113 | +0.111 | 0.567 |
| 0.5 | 0.467 | +0.135 | +0.134 | 0.427 |
| 0.9 | 1.084 | +0.115 | +0.119 | 1.033 |
| 0.95 | 1.374 | +0.085 | +0.091 | 1.342 |
| 0.99 | 1.817 | +0.023 | +0.031 | 1.809 |
| 1.0 | 1.999 | −0.008 | +0.000 | 1.999 |

Variance grows with $\lambda$, and the bias with a wrong $V$ falls toward zero as $\lambda \to 1$. The bias doesn't fall in a straight line. In this run it first rises between $\lambda = 0$ and $\lambda = 0.5$, because the error pattern of $V$ across states decides how the one-step and multi-step errors partly cancel. The root mean squared error with the wrong $V$ is smallest at $\lambda = 0.40$, where it is 0.441. At $\lambda = 0$ it is 0.578, and at $\lambda = 1$ it is 1.999. In this toy task the best $\lambda$ is intermediate, as it is in the paper's experiments (cart-pole: $\gamma \in [0.96, 0.99]$ and $\lambda \in [0.92, 0.99]$ in Section 6.3.1, and the biped: $\gamma \in [0.99, 0.995]$ and $\lambda \in [0.96, 0.99]$ in Section 6.3.2). Don't read the specific value 0.40 as typical, because it depends on how large the toy value error and the reward noise are. The paper's Section 7 summarizes that an intermediate $\lambda$ in $[0.9, 0.99]$ usually works best.

### The reward shaping view

**Intuition.** GAE is a return in a different MDP. If you reshape the rewards using the value function, the rewards become TD residuals, and those are concentrated near the step where something surprising happened. A discount of $\gamma\lambda$ then cuts off what's left of the noise.

**Math.** Reward shaping with a potential $\Phi : \mathcal{S} \to \mathbb{R}$ defines (Section 4, Equation 20):

$$
\tilde r(s, a, s') = r(s, a, s') + \gamma\Phi(s') - \Phi(s)
$$

This leaves the discounted advantage $A_{\pi,\gamma}$ unchanged. Along a trajectory the shaped rewards telescope (Equation 21), so $\sum_l \gamma^l \tilde r_{t+l} = \sum_l \gamma^l r_{t+l} - \Phi(s_t)$. The shaped value and action-value functions are $\tilde Q = Q - \Phi$ and $\tilde V = V - \Phi$ (Equations 22 and 23), so the advantage is the same (Equation 24). If $\Phi = V_{\pi,\gamma}$, then $\tilde V_{\pi,\gamma}$ is zero at every state. Setting $\Phi = V$ makes the shaped reward equal to $\delta_t^V$, so (Equation 25):

$$
\sum_{l=0}^{\infty} (\gamma\lambda)^l \, \tilde r_{t+l} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \, \delta_{t+l}^V = \hat A_t^{\text{GAE}(\gamma,\lambda)}
$$

The paper then introduces a response function $\chi(l; s_t, a_t) = \mathbb{E}[r_{t+l} \mid s_t, a_t] - \mathbb{E}[r_{t+l} \mid s_t]$ (Equation 26), which splits the advantage across time as $A_{\pi,\gamma}(s, a) = \sum_l \gamma^l \chi(l; s, a)$. Using $\gamma < 1$ drops the terms with $l \gg 1/(1-\gamma)$, and using $\lambda < 1$ drops the terms of the shaped rewards with $l \gg 1/(1-\gamma\lambda)$ (Section 4). With the exact value function as $\Phi$, the response is nonzero only at $l = 0$, so a good $V$ turns a temporally extended response into an immediate one.

**Tiny example, computed by `scripts/reward_shaping.py`.** On a random 4-state, 3-action MDP with $\gamma = 0.9$ and a random potential $\Phi$, the script solves for $Q$, $V$, and $A$ exactly before and after shaping. The largest difference from Equations 22, 23, and 24 is below $10^{-14}$ in each case, which is floating-point error. For the page's four-step episode with $\Phi = V$, the shaped rewards are $[0.86, 0.14, -0.33, 1.7]$, equal to the residuals. Summing them with discount $\gamma\lambda = 0.45$ gives $[1.01109, 0.33575, 0.435, 1.7]$, equal to the GAE values above.

Exercise 4 asks you to build GAE from the shaped rewards.

### GAE inside the policy and value updates

**Intuition.** GAE is only a way to score actions. The paper combines it with trust region updates for the policy, and also for the value function, which is the other half of the estimator.

**Math.** The policy update is TRPO with the sampled advantage $\hat A_n$ in the surrogate (Section 6.1, Equation 31). It's the TRPO objective from the [previous paper](/trpo/):

$$
L_{\theta_{\text{old}}}(\theta) = \frac{1}{N} \sum_{n=1}^{N} \frac{\pi_\theta(a_n \mid s_n)}{\pi_{\theta_{\text{old}}}(a_n \mid s_n)} \hat A_n
$$

Section 6.1 gives the loop: simulate the policy for $N$ timesteps, compute $\delta_t^V$ with $V = V_{\phi_i}$, compute $\hat A_t = \sum_l (\gamma\lambda)^l \delta_{t+l}^V$, take the TRPO step to get $\theta_{i+1}$, and then fit the value function to get $\phi_{i+1}$. The order matters. The policy update uses $V_{\phi_i}$, not $V_{\phi_{i+1}}$. The paper says updating the value function first adds bias.

The value function is fit by regression on the discounted returns $\hat V_t = \sum_l \gamma^l r_{t+l}$ (Section 5, Equation 28, the Monte Carlo or TD(1) target), under a trust region on how far the value function may move (Equation 29). That constraint equals a bound on the average KL divergence between the old and new value function, when the value function is read as a Gaussian with mean $V_\phi(s)$ and variance $\sigma^2$. The paper solves it with conjugate gradient, as in the policy step (Equation 30). The paper also tried a TD($\lambda$)-style target and saw no difference from the $\lambda = 1$ target (Section 5, footnote 2).

**Tiny example, computed by `scripts/gae_example.py`.** Take the page's episode, and imagine fitting $V$ so well that it reproduces the discounted return at every step: $V = [2.458, 1.62, 1.8, 2.0]$. Every residual is then $0$, so $\hat A_t = 0$ for every $\lambda$, and the policy gradient estimate vanishes. This is the extreme case that the paper uses to explain why the value function must be updated after the policy step (Section 6.1).

### What the paper shows

The experiments are simulated: cart-pole balancing, and 3D robot locomotion in MuJoCo, with policies and value functions that are neural networks. The 3D robot tasks use three hidden layers of 100, 50, and 25 tanh units (Section 6.2.1). GAE with intermediate $\gamma$ and $\lambda$ learned running gaits for a biped and a quadruped, and a standing-up behavior for the biped (Sections 6.3.1 to 6.3.3). The biped result took 1,000 iterations, which the paper converts to 5.8 days of simulated real time (Section 6.3.2). Where the paper compares against a baseline with no value function, the learned $V$ does better (Sections 6.3.2 and 6.3.3). The best cart-pole results sit at intermediate values of both $\gamma$ and $\lambda$ (Figure 2).

<!-- exercises -->

## Pitfalls

- **Treating a time limit as the end of the episode.** When an episode ends because the environment hit a step limit, the state isn't terminal and the future return isn't zero. Bootstrap from $V(s_T)$ there. When the episode really ends, use $0$. The paper's formulas assume an absorbing terminal state and don't discuss time limits, so this is an implementation point and not a paper claim. Exercise 2 handles the terminal case and the cut-off buffer.
- **Letting the recursion cross episode boundaries.** In a flat rollout buffer with several episodes, masking only the bootstrap term is not enough. The backward carry $\gamma\lambda\hat A_{t+1}$ must also stop at every episode end, or the last steps of one episode pick up the advantages of the next. Exercise 2's tests check that episode 1 doesn't change when episode 2 does.
- **Confusing the roles of $\gamma$ and $\lambda$.** The discount $\gamma$ changes the objective and biases the gradient even with a perfect $V$. The parameter $\lambda$ affects bias only when $V$ is imperfect (Section 3). Treating $\gamma\lambda$ as one knob hides this, because only the product enters the sum in Equation 16.
- **Expecting the bias to fall steadily as $\lambda$ rises.** The bias is zero at $\lambda = 1$ and nonzero for a wrong $V$ at $\lambda = 0$, but in between it depends on the error pattern. The table above has a bias that rises between $\lambda = 0$ and $\lambda = 0.5$. Variance is the quantity that rises steadily.
- **Using $\lambda = 0$ and expecting the best of both.** The paper found that the one-step estimate gave excessive bias and poor performance on its tasks (Section 7). Policy gradient methods that differentiate through a critic can work with a one-step target when tuned appropriately, but that's a different class of algorithm.
- **Computing advantages with a value function that has already been updated.** Use the $V$ that was in force when the data was collected, as in the algorithm box (Section 6.1). An overfit $V$ shrinks every residual toward zero, as the last example shows.
- **Forgetting that GAE is truncated in practice.** The sum in Equation 16 runs to infinity, but a finite rollout stops at $N$ steps. Implementations bootstrap from $V(s_N)$ at the end, which is what `last_value` does in the exercises.
- **Reading code that adds a value to the advantage.** Many libraries use $\hat A_t + V(s_t)$ as the regression target for the value function. That equals the $\lambda$-return, not the $\lambda = 1$ return the paper regressed on in its main experiments (Equation 28 and footnote 2). The paper reports no difference between them in its tests.
- **Normalizing advantages and assuming the paper does.** Subtracting the batch mean and dividing by the batch standard deviation is common in implementations. It is not part of the paper's estimator, and it changes the scale of the gradient, which interacts with the step size or trust region you choose.

## Further reading

- [Proximal Policy Optimization Algorithms](/ppo/), by Schulman et al., uses a truncated version of this estimator and replaces the TRPO constraint with a clipped objective.
- [Trust Region Policy Optimization](/trpo/), by Schulman et al., is the policy update this paper builds on, in Section 6.1 and Equation 31.
- [Supplementary videos for the paper](https://sites.google.com/site/gaepapersupp), linked from the paper, show the learned biped and quadruped gaits.
- [Spinning Up: Proximal Policy Optimization](https://spinningup.openai.com/en/latest/algorithms/ppo.html), by OpenAI, describes a PPO implementation that uses GAE-$\lambda$ advantage estimates.
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html), by Sutton and Barto, covers TD($\lambda$) and eligibility traces, the construction that GAE mirrors.
