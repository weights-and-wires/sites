---
id: "adam"
title: "Adam: A Method for Stochastic Optimization"
authors: "Kingma et al."
year: 2014
venue: "ICLR 2015"
arxiv: "1412.6980"
url: "https://arxiv.org/abs/1412.6980"
track: training-optimization
tier: core
est_hours: 3
status: full
summary: "Adam keeps two exponential moving averages for every parameter: one of the gradient and one of the squared gradient. It divides the first by the square root of the second, so each parameter takes a step of about the stepsize alpha, however large or small its gradients are. A bias correction removes the pull toward zero that the zero-initialized averages cause in the first steps."
related_course: "ml: optimizers"
prereqs: []
background: [sgd, momentum, exponential-moving-averages]
course_links:
  - title: "SGD, Adam, and learning-rate schedules"
    url: "https://ml.ifkash.dev/lessons/038-sgd-adam-and-schedules"
self_check:
  - q: 'Adam starts with $m_0 = 0$ and $\beta_1 = 0.9$. For a constant gradient $g_t = 1$, what is $m_1$, and what is $\hat m_1$ after the bias correction?'
    a: 'The result is $m_1 = 0.9 \cdot 0 + 0.1 \cdot 1 = 0.1$, ten times too small. Dividing by $1 - \beta_1^1 = 0.1$ gives $\hat m_1 = 1$, the true mean of the gradient.'
    see: exponential-moving-averages
  - q: 'In SGD with momentum, a parameter sees a gradient of $+1$ for many steps, and then the gradient flips to $-1$. What does the velocity do at the flip?'
    a: 'The velocity is a running average of past gradients, so it stays positive for a few steps and decays toward $-1$ instead of flipping at once. Adam keeps this behavior in its first moment $m_t$ and adds a second average, $v_t$, that rescales the step.'
    see: momentum
  - q: 'Plain SGD multiplies the gradient by a learning rate $\eta$. If you multiply the loss by 1000, what happens to the SGD step, and what does Adam do under the same change?'
    a: 'The SGD step grows 1000 times, so $\eta$ must shrink by the same factor to keep the same behavior. Adam divides by $\sqrt{\hat v_t}$, which also grows 1000 times, so its step is unchanged (Section 2.1).'
    see: sgd
---

## The problem

Stochastic gradient descent (SGD) moves every parameter by the same learning rate times its own gradient. That rule has two weaknesses when the problem is noisy or badly scaled.

- **The learning rate depends on the gradient scale.** Suppose you multiply a loss by 1000, for example by changing its units. The gradients grow by 1000, and an SGD run that worked now diverges. Parameters in different layers can also have gradients that differ by orders of magnitude (Section 6.3 notes that weight sharing in convolutional networks produces very different gradients in different layers), so no single learning rate suits all of them.
- **Sparse gradients starve rare features.** A parameter whose gradient is zero on most minibatches gets a tiny average step under a global learning rate. AdaGrad (Duchi et al., 2011) fixes this with a per-parameter step size but accumulates squared gradients forever. RMSProp (Tieleman and Hinton, 2012) uses a moving average instead, which suits non-stationary problems, but it has no bias correction and no momentum on the gradient itself (Section 5).

The paper proposes one method that combines the two ideas. Per-parameter step sizes come from an estimate of the gradient's second moment, as in RMSProp and AdaGrad. The direction comes from an estimate of its first moment, as in momentum. A correction term makes both estimates usable from the first step. Section 6 reports that Adam trains logistic regression, fully connected networks, and convolutional networks as fast as, or faster than, SGD with Nesterov momentum, AdaGrad, and RMSProp.

## Key ideas

The sections below use the paper's notation. $\alpha$ is the stepsize, $\beta_1, \beta_2 \in [0, 1)$ are the decay rates, $\epsilon$ is a small constant, $g_t$ is the gradient at timestep $t$, and $\theta_t$ is the parameter vector. The paper writes $g_t^2$ for the element-wise square $g_t \odot g_t$, and every operation is element-wise. The exercises use `theta`, `alpha`, `beta1`, `beta2`, and `eps` for the same quantities. Every number comes from a script in `scripts/`. Run any script from the paper folder to reproduce the output.

### Two moving averages of the gradient

**Intuition.** Keep a running average of the gradient: its direction is the signal, and averaging cancels part of the minibatch noise. Keep a second running average of the squared gradient: its square root tells you how large the gradients typically are for this parameter. Divide the first by the second, and every parameter takes a step measured in units of its own gradient size.

**Math.** Algorithm 1 (Section 2) updates two moment estimates, each initialized at zero:

$$
m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t
$$

$$
v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2
$$

It then corrects both for the zero initialization (next idea) and takes the step:

$$
\hat m_t = \frac{m_t}{1 - \beta_1^t}, \qquad
\hat v_t = \frac{v_t}{1 - \beta_2^t}, \qquad
\theta_t = \theta_{t-1} - \alpha \cdot \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}
$$

The paper's recommended defaults are $\alpha = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, and $\epsilon = 10^{-8}$. The moving averages estimate the first moment (the mean) and the second raw moment (the uncentered variance) of the gradient (Section 2). Section 2 also gives a slightly faster equivalent, $\alpha_t = \alpha \sqrt{1 - \beta_2^t} / (1 - \beta_1^t)$ and $\theta_t = \theta_{t-1} - \alpha_t \cdot m_t / (\sqrt{v_t} + \hat\epsilon)$. It changes the effective $\epsilon$ slightly, which is why the paper writes $\hat\epsilon$.

**Tiny example, computed by `scripts/adam_trace.py`.** Take one parameter and the defaults above, with gradients $g_1 = 2$, $g_2 = -1$, $g_3 = 1$. The table gives the step divided by $\alpha$:

| $t$ | $g_t$ | $m_t$ | $v_t$ | $\hat m_t$ | $\hat v_t$ | step / $\alpha$ |
|---|---|---|---|---|---|---|
| 1 | +2 | 0.2000 | 0.004000 | 2.0000 | 4.0000 | 1.0000 |
| 2 | -1 | 0.0800 | 0.004996 | 0.4211 | 2.4992 | 0.2663 |
| 3 | +1 | 0.1720 | 0.005991 | 0.6347 | 1.9990 | 0.4489 |

Check step 1 by hand: $m_1 = 0.1 \cdot 2 = 0.2$ and $v_1 = 0.001 \cdot 4 = 0.004$. Dividing by $0.1$ and $0.001$ gives $\hat m_1 = 2$ and $\hat v_1 = 4$, so the step is $\alpha \cdot 2 / 2 = \alpha$. The first step is always $\alpha$ times the sign of the gradient. At $t = 2$ the gradient flips sign, but the average still remembers the positive gradient, so the step stays positive and shrinks to about a quarter of $\alpha$.

Exercise 1 asks you to implement this update.

### Bias correction

**Intuition.** Both averages start at zero, so early on they are dominated by that zero. After one step with $\beta_1 = 0.9$, $m_1$ holds only 10% of the gradient. For $v_t$ the effect is far worse: with $\beta_2 = 0.999$, $v_1$ holds 0.1% of $g_1^2$. The correction divides the missing mass back out.

**Math.** Section 3 derives the correction for the second moment. Unrolling the recursion gives $v_t$ as a weighted sum of all earlier squared gradients (Section 3, Equation 1):

$$
v_t = (1 - \beta_2) \sum_{i=1}^{t} \beta_2^{\,t-i} \cdot g_i^2
$$

Take expectations and assume the second moment $\mathbb{E}[g^2]$ does not change over time. The geometric sum collapses (Equations 2 to 4):

$$
\mathbb{E}[v_t] = \mathbb{E}[g_t^2] \cdot (1 - \beta_2^t) + \zeta
$$

Here $\zeta = 0$ when the true second moment is stationary, and it stays small otherwise because old gradients get small weights. The factor $1 - \beta_2^t$ comes only from the zero initialization, so Algorithm 1 divides it out. The derivation for $m_t$ is the same with $\beta_1$.

The paper's text right after Equation 4 says that "the exponential decay rate $\beta_1$" should be chosen so old gradients get small weights. The derivation in that passage is about $\beta_2$, so this page treats the $\beta_1$ there as a typo.

**Tiny example, computed by `scripts/bias_correction.py`.** Feed a constant gradient $g_t = 1$ and compare steps with and without the correction, in units of $\alpha$:

| $t$ | corrected | uncorrected |
|---|---|---|
| 1 | 1.0000 | 3.1623 |
| 2 | 1.0000 | 4.2496 |
| 5 | 1.0000 | 5.7971 |
| 10 | 1.0000 | 6.5279 |

Without the correction, the first step is $0.1 / \sqrt{0.001} = 3.16$ times too large, because the denominator is small and the numerator is less small. The error grows before it fades, and the script shows a step of $6.53\,\alpha$ at $t = 10$. The factor $1 - \beta_2^t$ is 0.001 at $t = 1$, 0.0953 at $t = 100$, and 0.9933 at $t = 5000$, so the second moment needs thousands of steps to forget its initialization. The script also checks Equation 4 with seeded draws $g \sim \mathcal{N}(0, 1)$, for which $\mathbb{E}[g^2] = 1$. The mean of $v_{50}$ is 0.04886 against a predicted $1 - 0.999^{50} = 0.04879$, and dividing by that factor gives 1.0013.

Section 6.4 tests this on a variational autoencoder. When $\beta_2$ is close to 1 and there is no bias correction, training becomes unstable in the first epochs. Removing the correction yields RMSProp with momentum (Section 5). Exercise 2 asks you to reproduce this effect.

### The effective step size

**Intuition.** The step has the form $\alpha$ times a ratio $\hat m_t / \sqrt{\hat v_t}$. For a gradient with a consistent sign, that ratio is close to $\pm 1$, so every parameter moves by about $\alpha$ per step. For a noisy gradient that changes sign, the average $\hat m_t$ is small compared with the typical magnitude, and the step shrinks.

**Math.** Section 2.1 sets $\epsilon = 0$ and defines the effective step $\Delta_t = \alpha \cdot \hat m_t / \sqrt{\hat v_t}$. It states two upper bounds:

$$
|\Delta_t| \le \alpha \cdot \frac{1 - \beta_1}{\sqrt{1 - \beta_2}} \quad \text{if } 1 - \beta_1 > \sqrt{1 - \beta_2}, \qquad |\Delta_t| \le \alpha \quad \text{otherwise.}
$$

The first case happens only in the sparsest situation, when a gradient has been zero at every timestep except the current one. In common cases $|\mathbb{E}[g]| / \sqrt{\mathbb{E}[g^2]} \le 1$, so $|\Delta_t| \lessapprox \alpha$. The paper reads this as a trust region around the current parameter value, and it calls the ratio $\hat m_t / \sqrt{\hat v_t}$ a signal-to-noise ratio (SNR). A smaller SNR means a smaller step, which the paper describes as a form of automatic annealing as the optimizer nears an optimum.

The ratio is also invariant to the scale of the gradients. Multiplying $g$ by $c$ multiplies $\hat m_t$ by $c$ and $\hat v_t$ by $c^2$, so $(c \cdot \hat m_t) / \sqrt{c^2 \cdot \hat v_t} = \hat m_t / \sqrt{\hat v_t}$.

**Tiny example, computed by `scripts/step_bound.py`.** With the defaults, the first bound is $\alpha \cdot 0.1 / \sqrt{0.001} = 3.1623\,\alpha$. The script reports the following:

- A gradient that is 0 for 5000 steps and 1 at step 5001 gives a step of $3.1516\,\alpha$, which is just under the bound. After only 20 zeros, the same spike gives $0.5120\,\alpha$, because the correction terms haven't yet decayed.
- A first step with $g = 5$ is exactly $1.0\,\alpha$.
- On 2000 seeded random gradients from $\mathcal{N}(0, 1)$, the largest step is $1.0\,\alpha$, which comes from step 1.
- Multiplying 200 gradients by 1000 changes the steps by at most $4.34 \times 10^{-19}$ when $\epsilon = 0$.

Note the invariance is exact only at $\epsilon = 0$. When gradients become comparable to $\epsilon$, which the paper observes for the second moment in the CNN experiment of Section 6.3, the denominator is dominated by $\epsilon$ and the step shrinks.

### Adam against SGD with momentum

**Intuition.** On a well-conditioned problem, tuned SGD with momentum is hard to beat. Adam's advantage is that its main hyperparameter, $\alpha$, is a distance in parameter space and does not depend on the gradient's scale. The paper makes this point in Section 2.1: you can often pick the order of magnitude of $\alpha$ from what you know about the parameters.

**Math.** The momentum baseline in the exercise is the heavy-ball form, with $v \leftarrow \mu v + g$ and $\theta \leftarrow \theta - \eta v$. The paper's SGD baselines use Nesterov momentum (Section 6.1), which differs slightly. The Rosenbrock function $f(x, y) = (1 - x)^2 + 100\,(y - x^2)^2$ is a standard test with a curved, narrow valley and a minimum of 0 at $(1, 1)$. The paper does not use it, so this comparison is an illustration and does not reproduce a result from Section 6.

**Tiny example, computed by `scripts/rosenbrock.py`.** Start at $(-1.5, 1.5)$, where $f = 62.5$, and run 2000 steps of each method.

| Method | Setting | Final $f$ |
|---|---|---|
| Adam | $\alpha = 0.01$ | 0.914 |
| Adam | $\alpha = 0.05$ | $5.7 \times 10^{-5}$ |
| Adam | $\alpha = 0.1$ | $2.3 \times 10^{-9}$ |
| Momentum, $\mu = 0.9$ | $\eta = 10^{-4}$ | 0.108 |
| Momentum, $\mu = 0.9$ | $\eta = 10^{-3}$ | $6.2 \times 10^{-9}$ |
| Momentum, $\mu = 0.9$ | $\eta = 3 \times 10^{-3}$ | diverges (`nan`) |

Tuned momentum ties Adam on this problem, and both need a good setting. The difference shows when you change the units. Multiply the gradient by 1000 and the table changes as follows:

| Method | Setting | Final $f$ |
|---|---|---|
| Adam | $\alpha = 0.1$ | $2.3 \times 10^{-9}$ (unchanged) |
| Momentum | $\eta = 10^{-3}$ | diverges (`nan`) |
| Momentum | $\eta = 10^{-6}$ | $6.2 \times 10^{-9}$ |

Adam's setting still works, and momentum needs a learning rate 1000 times smaller. Exercise 3 asks you to build this comparison.

### AdaMax

**Intuition.** Adam divides by an $L_2$-style norm of the past gradients. You can generalize to an $L_p$ norm, and as $p \to \infty$ the norm becomes a running maximum, which is simple and stable.

**Math.** Section 7.1 derives the limit (Equations 6 to 12) and gives the recursion $u_t = \max(\beta_2 \cdot u_{t-1}, |g_t|)$ (Equation 12). Algorithm 2 then updates

$$
\theta_t = \theta_{t-1} - \frac{\alpha}{1 - \beta_1^t} \cdot \frac{m_t}{u_t}
$$

with $m_t$ as in Adam. Only $m_t$ needs bias correction. The paper notes that $u_t$ doesn't, and that the step obeys the simpler bound $|\Delta_t| \le \alpha$. The suggested defaults are $\alpha = 0.002$, $\beta_1 = 0.9$, and $\beta_2 = 0.999$.

**Tiny example, computed by `scripts/adamax_example.py`.** With gradients $2, -1, 0.5, 4$ the step divided by $\alpha$ is $1.0000$, $0.2107$, $0.2255$, and $0.3706$. The value $u_t$ is $2.0$, $1.998$, $1.996$, and $4.0$. The first step is again $\alpha$ times the sign of the gradient, and $u_t$ decays by $\beta_2$ each step until a larger gradient replaces it. Exercise 4 asks you to implement it.

<!-- exercises -->

## Pitfalls

- **Skipping the bias correction.** Without it, the first steps are several times too large when $\beta_2$ is near 1. The script shows $3.16\,\alpha$ on step 1 and $6.53\,\alpha$ on step 10 for a constant gradient. Frameworks include the correction by default, but a hand-written optimizer often leaves it out.
- **Starting the timestep at 0.** The correction divides by $1 - \beta_1^t$, which is 0 at $t = 0$. Count from $t = 1$.
- **Putting $\epsilon$ inside the square root.** Algorithm 1 adds $\epsilon$ after the root. Moving it inside changes the step for small gradients: for $g = 10^{-6}$ the correct step is $0.99\,\alpha$, and the wrong version gives about $0.01\,\alpha$.
- **Expecting the bound to hold exactly.** Section 2.1 says the step is only approximately bounded by $\alpha$. With $\epsilon > 0$ it is slightly smaller, and in the sparse case it can reach $3.16\,\alpha$.
- **Treating $\alpha$ as scale-free.** Adam removes the dependence on the gradient scale, and it doesn't remove the dependence on the problem. On Rosenbrock, $\alpha = 0.01$ leaves $f$ at 0.914 after 2000 steps, while $\alpha = 0.1$ reaches $2.3 \times 10^{-9}$.
- **Reading the regret bound as a guarantee for deep learning.** Theorem 4.1 (Section 4) covers online convex optimization, and the paper says its analysis does not apply to non-convex networks (Section 6.2). A later paper, Reddi et al., "On the Convergence of Adam and Beyond" (ICLR 2018), showed a flaw in this convergence proof and constructed cases where Adam doesn't converge.
- **Confusing Adam with AdamW.** Adding L2 regularization to the gradient interacts with $v_t$ in Adam. Decoupling weight decay from the gradient is a separate change; see [AdamW](/adamw/).
- **Comparing against untuned baselines.** Section 6 reports each optimizer at its best hyperparameters from a dense grid. Match that protocol, or a comparison says more about the tuning than the optimizer.

## Further reading

- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101), by Loshchilov and Hutter, is the AdamW paper. This site covers it at [AdamW](/adamw/).
- [On the Convergence of Adam and Beyond](https://arxiv.org/abs/1904.09237), by Reddi, Kale, and Kumar, gives the counterexample to the convergence proof and proposes AMSGrad.
- [An overview of gradient descent optimization algorithms](https://www.ruder.io/optimizing-gradient-descent/), by Sebastian Ruder, compares SGD variants, AdaGrad, RMSProp, and Adam.
- [Why Momentum Really Works](https://distill.pub/2017/momentum/), by Gabriel Goh, explains the first-moment half of Adam.
- [`torch.optim.Adam`](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) is the reference implementation most people use.
