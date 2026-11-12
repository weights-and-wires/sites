---
id: "adamw"
title: "Decoupled Weight Decay Regularization"
authors: "Loshchilov et al."
year: 2017
venue: "ICLR 2019"
arxiv: "1711.05101"
url: "https://arxiv.org/abs/1711.05101"
track: training-optimization
tier: core
est_hours: 2
status: full
summary: "For plain SGD, adding an L2 penalty to the loss and shrinking the weights directly at each step are the same update after a rescaling, and under Adam they are not. With L2 in the gradient, Adam divides the penalty by the running gradient scale, so weights with large gradients are regularized less. AdamW shrinks the weights outside the adaptive update, which gives every weight the same decay rate and makes the best decay factor depend much less on the learning rate."
related_course: "ml: optimizers"
prereqs:
  - id: adam
    reason: "AdamW changes one line of Adam's update, so you need the moment estimates $m_t$ and $v_t$ and the per-parameter division by $\\sqrt{\\hat v_t}$ to see what the L2 term gets divided by."
background: [l2-regularization, weight-decay]
course_links:
  - title: "Weight decay and early stopping: two ways to keep weights small"
    url: "https://ml.ifkash.dev/lessons/041-weight-decay-and-early-stopping"
  - title: "SGD, Adam, and learning-rate schedules"
    url: "https://ml.ifkash.dev/lessons/038-sgd-adam-and-schedules"
self_check:
  - q: 'Plain SGD with step $\alpha$ runs on the loss $f(\theta) + \frac{\lambda''}{2}\lVert\theta\rVert_2^2$. Write one update. What decay factor $\lambda$ does it match in $\theta \leftarrow (1 - \lambda)\theta - \alpha\nabla f(\theta)$?'
    a: 'The update is $\theta \leftarrow \theta - \alpha\nabla f(\theta) - \alpha\lambda''\theta = (1 - \alpha\lambda'')\theta - \alpha\nabla f(\theta)$. It matches decay $\lambda = \alpha\lambda''$, so $\lambda'' = \lambda / \alpha$ (Proposition 1). The two knobs are tied together by the learning rate.'
    see: l2-regularization
  - q: 'A weight decay of $\lambda = 0.01$ is applied for 100 steps with no gradient from the loss. What fraction of the weight is left, and what makes this a different operation from "make the weights small" through the loss?'
    a: 'Each step multiplies the weight by $0.99$, so $0.99^{100} \approx 0.366$ is left. Decay acts on the weights directly at a fixed rate. An L2 penalty acts through the gradient, so its effect depends on whatever the optimizer does with gradients.'
    see: weight-decay
  - q: 'In Adam, a parameter has had loss gradients about 100 times larger than another for a long time. Both receive the same $\lambda\theta$ term in the gradient. Which one does the optimizer shrink faster, and by roughly how much?'
    a: 'The shrinkage per step is about $\alpha\lambda\theta / \sqrt{\hat v_t}$, and $\sqrt{\hat v_t}$ is about 100 times larger for the first parameter, so it shrinks about 100 times more slowly. Adam normalizes the penalty together with the loss gradient (Section 2).'
    see: adam
---

## The problem

Weight decay is a standard regularizer for neural networks, and for years the usual advice was that "weight decay" and "L2 regularization" are two names for one thing. Deep learning libraries followed that advice, and an optimizer option called `weight_decay` meant "add $\lambda\theta$ to the gradient". The paper shows that this holds for SGD and fails for Adam, which is why Adam with this option regularized poorly.

The failure has a concrete form. Suppose a parameter has large loss gradients, so Adam's second moment $\hat v_t$ for it is large. The L2 term $\lambda\theta$ goes into the same gradient, so Adam divides it by the same large $\sqrt{\hat v_t}$. The parameter that the loss pushes hardest gets the weakest pull toward zero. A parameter with tiny gradients gets the strongest pull. The regularizer ends up depending on the gradient scale, which is not a property you chose and not one you can tune.

A second problem appears even for SGD. In Equation 1, the weight decay of Hanson and Pratt (1988) multiplies the weights by $1 - \lambda$ per step. The equivalent L2 factor is $\lambda' = \lambda / \alpha$ (Proposition 1), so the best $\lambda'$ moves whenever you change the learning rate $\alpha$. Section 4.2 and Figure 2 show this as a diagonal valley in the plane of the two hyperparameters: changing only one of them makes results worse.

The paper has two goals. It shows exactly where the equivalence breaks (Propositions 2 and 3), and it proposes AdamW and SGDW, which apply the decay outside the gradient step. Section 4 reports that this improves Adam's generalization on CIFAR-10 and ImageNet32x32, with about a 15% relative improvement in test error (Section 4.4), and that the best learning rate and decay factor become more independent (Figure 2).

## Key ideas

The sections below use the paper's notation. $\alpha$ is the stepsize, $\beta_1$ and $\beta_2$ are the decay rates, $\epsilon$ is a small constant, $\theta_t$ is the parameter vector, $\nabla f_t(\theta_{t-1})$ is the batch gradient of the loss, and $\lambda$ is the weight decay factor. The paper writes $\lambda'$ for an L2 factor in the loss. The paper also has a schedule multiplier $\eta_t$ that scales the step and the decay; this page sets $\eta_t = 1$ except where noted. The exercises use `theta`, `grad`, `alpha`, `lam`, `lam_l2` (for $\lambda'$), and `beta1`, `beta2`, `eps`. Every number comes from a script in `scripts/`. Run any script from the paper folder to reproduce the output.

### Decay and L2 are one thing for SGD

**Intuition.** Weight decay shrinks every weight by a fixed fraction each step. An L2 penalty adds the gradient $\lambda'\theta$ to the loss gradient. For SGD, the penalty's gradient step is the weight shrinkage, because SGD multiplies every gradient by the same $\alpha$. The only catch is the exchange rate: $\lambda' = \lambda / \alpha$.

**Math.** Equation 1 gives the weight decay of Hanson and Pratt:

$$
\theta_{t+1} = (1 - \lambda)\,\theta_t - \alpha\,\nabla f_t(\theta_t)
$$

Proposition 1 says that SGD with this decay takes the same steps as SGD without it on the loss $f_t^{\text{reg}}(\theta) = f_t(\theta) + \frac{\lambda'}{2}\lVert\theta\rVert_2^2$, with $\lambda' = \lambda / \alpha$. The proof in Appendix A (Equations 5 and 6) expands the two updates and compares them:

$$
\theta_{t+1} = \theta_t - \alpha\,\nabla f_t(\theta_t) - \alpha\lambda'\,\theta_t \qquad\text{against}\qquad \theta_{t+1} = \theta_t - \alpha\,\nabla f_t(\theta_t) - \lambda\,\theta_t
$$

The two match when $\alpha\lambda' = \lambda$. Equivalence is therefore a statement about a pair $(\lambda', \alpha)$, and the decay per step $\alpha\lambda'$ changes when you change $\alpha$ and leave $\lambda'$ alone. Decoupling the two hyperparameters is the paper's motivation for SGDW, Algorithm 1 in Section 2.

**Tiny example, computed by `scripts/sgd_equivalence.py`.** Take $f(\theta) = \frac12\sum_i h_i(\theta_i - b_i)^2$ with $h = (1, 4, 0.25)$ and $b = (1, -2, 3)$. Start at $\theta = (5, 5, 5)$ with $\alpha = 0.1$ and $\lambda = 0.01$, and run 50 steps.

| Update | Final $\theta$ | Largest difference from decay |
|---|---|---|
| Decay, $\lambda = 0.01$ | (0.92115, -1.95122, 2.624026) | not applicable |
| L2, $\lambda' = \lambda / \alpha = 0.1$ | (0.92115, -1.95122, 2.624026) | $2.2 \times 10^{-16}$ |
| L2, $\lambda' = \lambda = 0.01$ | (1.009648, -1.995012, 3.451294) | $0.83$ |

Now keep $\lambda' = 0.1$ and halve the learning rate. The decay per step is $\alpha\lambda'$, which is $0.01$ at $\alpha = 0.1$, $0.005$ at $\alpha = 0.05$, and $0.0025$ at $\alpha = 0.025$. With decoupled decay, $\lambda = 0.01$ gives $0.01$ at every $\alpha$.

Momentum breaks even this match. Algorithm 1 puts $\lambda\theta$ into the gradient (line 6) for the L2 version, and the gradient then passes through the momentum $m_t$ (line 8). The SGDW version subtracts $\lambda\theta$ in line 9, after the momentum. With $\beta_1 = 0.9$, the same script shows the two runs end at different points: (0.805119, -2.264453, 1.947613) against (0.791984, -2.126078, 2.908657). The paper's Proposition 1 covers plain SGD, so it makes no claim about this case. Exercise 2 asks you to implement both variants.

### Under Adam, L2 is divided by the gradient scale

**Intuition.** Adam divides each parameter's update by $\sqrt{\hat v_t}$, a running estimate of its typical gradient size. If $\lambda\theta$ is part of the gradient, it is divided by the same number. A parameter that the loss tugs hard has a large $\sqrt{\hat v_t}$, so the penalty is small relative to it.

**Math.** Algorithm 2 (Section 2) lists both variants in one box. Line 6 forms $g_t = \nabla f_t(\theta_{t-1}) + \lambda\theta_{t-1}$ and applies to the L2 variant only. In the same box, line 12 for AdamW is:

$$
\theta_t = \theta_{t-1} - \eta_t\left(\alpha\,\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} + \lambda\,\theta_{t-1}\right)
$$

AdamW uses $g_t = \nabla f_t(\theta_{t-1})$ in lines 7 and 8, so $m_t$ and $v_t$ never see the decay, and the decay term in line 12 is not divided by anything. In the L2 variant, the same $\lambda\theta_{t-1}$ term enters $m_t$ and $v_t$, and the update becomes $\alpha\hat m_t / (\sqrt{\hat v_t} + \epsilon)$ with the penalty folded in.

Proposition 2 states the general result. Let $\mathcal{O}$ be an optimizer with iterates $\theta_{t+1} \leftarrow \theta_t - \alpha M_t \nabla f_t(\theta_t)$, where $M_t \ne kI$ for every scalar $k$. Then no L2 coefficient $\lambda'$ makes running $\mathcal{O}$ on $f_t + \frac{\lambda'}{2}\lVert\theta\rVert_2^2$ equal to running it with decay $\lambda$. The proof (Appendix A, Equations 7 and 8) reduces to requiring $\lambda\theta_t = \alpha\lambda' M_t\theta_t$ for every $\theta_t$, which needs $M_t = kI$. Adam's $M_t$ is a per-parameter diagonal with different entries, so the condition fails.

**Tiny example, computed by `scripts/adam_effective_decay.py`.** Use the defaults $\alpha = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, and $\epsilon = 10^{-8}$.

First, give the optimizer no loss gradient, with $\theta = 1$. In the L2 variant the gradient is $g = \lambda'$, so $\hat m_1 = \lambda'$ and $\sqrt{\hat v_1} = \lambda'$. The first step is $\alpha \cdot \lambda' / (\lambda' + \epsilon) \approx \alpha$, whatever the value of $\lambda'$. The script shows a shrinkage of $0.001000$ for $\lambda' = 0.1$, $0.01$, and $0.001$. Adam normalized the penalty away. In AdamW, $\lambda = 0.1$ shrinks the weight by $0.1$, and $\lambda = 0.001$ shrinks it by $0.001$.

Second, add loss gradients of a given scale. The script holds $\theta = 1$ and draws $g_t = s \cdot \xi_t + \lambda'\theta$, with $\xi_t \sim \mathcal{N}(0, 1)$ from seed 0, $\lambda' = 0.1$, and 2000 steps. It averages the shrinkage $\alpha\lambda' / (\sqrt{\hat v_t} + \epsilon)$ that comes from the penalty over the steps after the first 200:

| Gradient scale $s$ | L2 shrinkage per step | AdamW, $\lambda = 10^{-4}$ |
|---|---|---|
| 0.1 | $7.18 \times 10^{-4}$ | $1.00 \times 10^{-4}$ |
| 1 | $1.00 \times 10^{-4}$ | $1.00 \times 10^{-4}$ |
| 10 | $1.01 \times 10^{-5}$ | $1.00 \times 10^{-4}$ |

For $s \gg \lambda'$, the L2 shrinkage falls about tenfold for each tenfold increase in $s$. This is the sense in which weights with large gradient amplitudes are regularized less (Section 1 and Section 2). At $s = 0.1$ the penalty itself is comparable to the noise, so the scaling flattens. The AdamW decay is the same for every $s$. This table isolates the decay term and doesn't train a model, so it illustrates the mechanism rather than reproducing a result from Section 4. Exercise 1 asks you to implement both variants.

### The fixed-preconditioner case makes this exact

**Intuition.** Adam changes its preconditioner at every step, which makes it hard to state exactly what loss it optimizes. The paper therefore freezes the preconditioner and asks what the L2 version would need to look like to match decoupled decay.

**Math.** Proposition 3 takes the optimizer of Proposition 2 with $M_t = \mathrm{diag}(s)^{-1}$ and every $s_i > 0$. It says the optimizer with decay $\lambda$ on $f_t$ takes the same steps as the optimizer without decay on the scale-adjusted loss, with $\lambda' = \lambda / \alpha$. The proof (Appendix A, Equations 9 to 13) differentiates the penalty and divides by $s$:

$$
f_t^{\text{sreg}}(\theta) = f_t(\theta) + \frac{\lambda'}{2}\,\bigl\lVert\theta \odot \sqrt{s}\bigr\rVert_2^2
\quad\Rightarrow\quad
\theta_{t+1} = \theta_t - \alpha\,\frac{\nabla f_t(\theta_t)}{s} - \alpha\lambda'\,\theta_t
$$

The gradient of the penalty is $\lambda'\,s \odot \theta$, and the division by $s$ cancels the factor $s$. The coefficient on $\frac12\theta_i^2$ is therefore $\lambda' s_i$, which grows with $s_i$: a parameter with a large inverse preconditioner (historically large gradients) is regularized more strongly than under uniform L2, in proportion to $\sqrt{s_i}$ on the weight. The paper notes that this does not apply directly to Adam, whose preconditioner changes, but it gives the intuition.

On notation, the arXiv v3 text of Equation 2 appears to carry an extra factor of $\alpha$ in the penalty's denominator that the proof does not use. This page follows the proof, where the penalty is $\frac{\lambda'}{2}\lVert\theta \odot \sqrt{s}\rVert_2^2$ with $\lambda' = \lambda / \alpha$.

**Tiny example, computed by `scripts/fixed_preconditioner.py`.** Let $s = (0.1, 1, 10)$, $\alpha = 0.01$, and $\lambda = 0.01$, so $\lambda' = 1$. Take $f(\theta) = \frac12\lVert\theta - \mathbf{1}\rVert^2$ and start at $\theta = (2, 2, 2)$.

- The L2 term shrinks the three weights by $\alpha\lambda' / s = (0.1, 0.01, 0.001)$ per step. The decoupled decay shrinks them by $0.01$ each.
- After 200 steps the decoupled run ends at $(0.909091, 0.526382, 0.29988)$. The L2 run with per-coordinate coefficients $\lambda' s_i$ ends at the same point, with a largest difference of $5.6 \times 10^{-16}$.
- The L2 run with one uniform $\lambda' = 1$ ends at $(0.5, 0.526382, 1.505077)$, a difference of $1.21$. Only the middle coordinate, where $s = 1$, agrees.

Exercise 3 asks you to build both steps and the coefficient conversion.

### Decoupling the decay from the learning rate

**Intuition.** In the decoupled form the decay per step is $\lambda$, and the learning rate $\alpha$ is in a different term. Changing one doesn't directly change the other, so you can tune them one at a time.

**Math.** Section 2 says that decoupling "explicitly decouples $\lambda$ and $\alpha$", and it adds that some problem-dependent coupling can remain. The paper tests this in Section 4.2. For SGD with L2 (Figure 2, top left), the best settings lie on a diagonal. For SGDW (top right), a learning rate that is not well tuned still gives a good decay factor when held fixed. For Adam with L2 (bottom left), the best results with a nonzero L2 factor were comparable to those without any. For AdamW (bottom right), the settings separate, and the best result is substantially better and rivals SGD and SGDW. These are results on a 26 2x64d ResNet on CIFAR-10 after 100 epochs, and the paper says in its conclusion that they must be verified on a wider range of tasks (Section 5).

The paper adds a schedule multiplier $\eta_t$ that scales both the gradient step and the decay (Algorithm 2, line 12), and it reports that a schedule such as cosine annealing helps Adam substantially (Section 4.1, Figure 1). The multiplier is part of why the decay stays tied to the schedule.

### Normalized weight decay

**Intuition.** The best decay factor shrinks as training gets longer, because the total shrinkage is a product over every step. The paper rescales $\lambda$ so that one value works across training budgets.

**Math.** Appendix B.1 replaces $\lambda$ with a normalized factor $\lambda_{\text{norm}}$:

$$
\lambda = \lambda_{\text{norm}}\sqrt{\frac{b}{B\,T}}
$$

Here $b$ is the batch size, $B$ is the number of training points, and $T$ is the number of epochs. This equation is unnumbered in Appendix B.1. Appendix D refers to it as "Eq. (15)", but Equation 15 in the paper is the cosine schedule multiplier of Appendix B.2, so treat that reference as an error. The paper says the square-root form is one choice informed by few experiments, and that some normalization matters more than this exact form.

**Tiny example, computed by `scripts/normalized_decay.py`.** Take $b = 128$ and $\lambda_{\text{norm}} = 0.025$, a value Appendix D reports as near-optimal. The page uses the standard 50,000 CIFAR-10 training images.

| Run | Batch passes | $\lambda$ |
|---|---|---|
| CIFAR-10, $T = 100$ | about 39,000 | $1.26 \times 10^{-4}$ |
| CIFAR-10, $T = 400$ | about 156,000 | $6.3 \times 10^{-5}$ |
| CIFAR-10, $T = 1800$ | about 702,000 | $3.0 \times 10^{-5}$ |

The raw $\lambda$ for $T = 1800$ is 4.24 times smaller than for $T = 100$. For ImageNet32x32 with 1,281,167 training images, the same $T = 100$ gives $\lambda = 2.5 \times 10^{-5}$, and the ratio between the CIFAR-10 and ImageNet32x32 values is 5.06. This matches Appendix D, which says that the raw CIFAR-10 value would have been about 5 times too large for ImageNet32x32. The paper does not state the dataset sizes, so the counts here are the standard ones.

<!-- exercises -->

## Pitfalls

- **Treating `weight_decay` in an Adam optimizer as the paper's decay.** In an optimizer that adds $\lambda\theta$ to the gradient, the option is L2 regularization. The script shows it giving a first step of about $\alpha$ for any $\lambda'$ when the loss gradient is zero, and a shrinkage that falls about tenfold per tenfold in gradient scale. Check whether your library applies the term to the gradient or to the weights.
- **Copying $\lambda$ between libraries without checking the scale.** Algorithm 2 subtracts $\eta_t\lambda\theta_{t-1}$, with no $\alpha$ in that term. A library may multiply the decay by the learning rate, as `torch.optim.AdamW` does in its documented update ($\theta \leftarrow \theta - \gamma\lambda\theta$ with learning rate $\gamma$). Then the paper's $\lambda$ equals $\gamma$ times the library's value, and a schedule that lowers the learning rate also lowers the decay. Read the documentation for your version before you transfer a value.
- **Assuming $\lambda' = \lambda$ for SGD.** Proposition 1 needs $\lambda' = \lambda / \alpha$. The script shows a difference of $0.83$ after 50 steps when you skip the division.
- **Assuming the equivalence survives momentum.** Proposition 1 covers plain SGD. With momentum, the L2 term passes through $m_t$ and the decay doesn't (Algorithm 1, lines 6 to 9), and the runs diverge.
- **Putting $\lambda\theta$ into $m_t$ and $v_t$ in AdamW.** The moments must see only the loss gradient. If they see the decay too, the decay gets normalized by $\sqrt{\hat v_t}$ again and you have Adam with L2.
- **Applying decay to every parameter.** The paper decays all weights at the same rate $\lambda$ and doesn't discuss exempting biases or normalization parameters. Many LLM training setups exempt them, which is a practice outside this paper.
- **Reading Proposition 3 as a result about Adam.** The paper says it doesn't directly apply, because Adam's preconditioner changes at every step. It illustrates the direction of the effect.
- **Treating the 15% figure as general.** The improvement in Section 4.4 comes from a ResNet on CIFAR-10 and ImageNet32x32. Section 5 asks for verification on other tasks. Section 4.5 cites reports from other groups, including a Transformer, but the paper's own controlled sweeps use images.
- **Dropping the schedule.** Section 4.1 reports that the benefit is larger with a better learning rate schedule, and that cosine annealing clearly beats a fixed rate or a step drop in Figure 1. A comparison at a fixed learning rate understates the effect.

## Further reading

- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) is the paper. The authors' code for the experiments is at [loshchil/AdamW-and-SGDW](https://github.com/loshchil/AdamW-and-SGDW).
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) defines the update that AdamW modifies. This site covers it at [Adam](/adam/).
- [A Unified Theory of Adaptive Stochastic Gradient Descent as Bayesian Filtering](https://arxiv.org/abs/1507.02030), by Aitchison, is the framework in Section 3 of the paper where decay appears as part of the state transition.
- [Three Mechanisms of Weight Decay Regularization](https://arxiv.org/abs/1810.12281), by Zhang, Wang, Xu, and Grosse, compares L2 regularization and decay for SGD, Adam, and K-FAC, and is cited in Section 4.5.
- [`torch.optim.AdamW`](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) is the reference implementation most people use. Read its update rule to see how the decay is scaled.
- [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983), by Loshchilov and Hutter, introduces the cosine schedule and restarts that Appendix B.2 applies to AdamW.
