---
id: "flow-matching"
title: "Flow Matching for Generative Modeling"
authors: "Lipman et al."
year: 2022
venue: "ICLR 2023"
arxiv: "2210.02747"
url: "https://arxiv.org/abs/2210.02747"
track: generative
tier: core
est_hours: 5
status: full
summary: "The paper trains continuous normalizing flows, which turn noise into data by integrating a learned velocity field, without ever simulating the flow during training. It does this by regressing the network onto a per-example target velocity for a chosen Gaussian path from noise to one data point, which gives the same gradients as regressing onto the intractable velocity field of the whole data distribution. Diffusion paths are one choice of Gaussian path, and a straight-line optimal-transport path trains faster and samples with fewer solver steps, reaching 2.99 bits per dimension and an FID of 6.35 on CIFAR-10 with about half the function evaluations of a DDPM baseline."
related_course: "ml: flow models"
prereqs:
  - id: score-sde
    reason: "The paper's diffusion paths are the variance-preserving case of that paper's probability-flow ODE, and Appendix D recovers that ODE's vector field from Theorem 3 after reversing time so that t = 0 is noise."
background: [odes, continuous-normalizing-flows, probability-paths]
course_links:
  - title: "Flow matching"
    url: "https://speech.ifkash.dev/lessons/066-flow-matching"
  - title: "Straighter paths and better solvers"
    url: "https://speech.ifkash.dev/lessons/067-straighter-paths-and-better-solvers"
self_check:
  - q: 'Integrate $\mathrm{d}x/\mathrm{d}t = x$ from $x(0) = 1$ to $t = 1$ with two Euler steps. What do you get, and what is the exact answer?'
    a: 'The step size is $h = 0.5$. The first step gives $1 + 0.5 \cdot 1 = 1.5$ and the second gives $1.5 + 0.5 \cdot 1.5 = 2.25$. The exact solution is $e \approx 2.7183$, so two steps leave an error of about 0.47. Sampling a flow means solving this kind of ODE with the learned field in place of $x$ (Section 2, Equations 1 and 2).'
    see: odes
  - q: 'A flow is $\phi_t(x) = e^{t}x$ and the prior is $p_0 = \mathcal{N}(0, 1)$. What is the vector field $v_t(y)$ that generates it, and what is the standard deviation of $p_t$ at $t = \ln 2$?'
    a: 'Differentiating gives $\frac{\mathrm{d}}{\mathrm{d}t}\phi_t(x) = e^t x = \phi_t(x)$, so $v_t(y) = y$ for every $t$. The push-forward of $\mathcal{N}(0, 1)$ under $x \mapsto e^t x$ is $\mathcal{N}(0, e^{2t})$, so the standard deviation at $t = \ln 2$ is $e^{\ln 2} = 2$ (Section 2, Equations 1 to 4).'
    see: continuous-normalizing-flows
  - q: 'A path of densities $p_t$ and a vector field $u_t$ are given. What equation must they satisfy for $u_t$ to generate $p_t$, and what does the conditional version of the vector field look like for a mixture path?'
    a: 'They must satisfy the continuity equation $\partial_t p_t + \mathrm{div}(p_t u_t) = 0$ (Appendix B, Equation 26). For a mixture path $p_t(x) = \int p_t(x \mid x_1)\,q(x_1)\,\mathrm{d}x_1$, the generating field is the posterior-weighted average of the conditional fields, $u_t(x) = \int u_t(x \mid x_1)\,\frac{p_t(x \mid x_1)\,q(x_1)}{p_t(x)}\,\mathrm{d}x_1$ (Section 3.1, Equations 6 and 8).'
    see: probability-paths
---

## The problem

A continuous normalizing flow (CNF) is the most flexible kind of flow model. A neural network $v_t(x;\theta)$ defines a velocity at every point and time, and integrating it carries samples from a simple prior $p_0$ to a complicated distribution $p_1$ (Section 2). In principle a CNF can represent any smooth probability path. The paper's introduction lists why that flexibility went unused (Section 1 and Section 5):

- **Maximum-likelihood training needs ODE simulation.** To get a gradient of $\log p_1(x_1)$ you solve the ODE in Equation 28 forward and backward, with a divergence of $v_t$ along the way (Appendix C, Equations 27 and 28). Every training step costs a full adaptive solve, so CNFs didn't scale to large images.
- **The earlier simulation-free alternatives didn't fit.** Rozen et al. (2021) need integrals that are hard to estimate in high dimension, and Ben-Hamu et al. (2022) get biased gradients when the batch is a small sample (Section 5).
- **Diffusion models did scale, but only on a narrow family of paths.** Their paths come from a fixed noising process, and the noise never reaches a true Gaussian in finite time, so $p_0$ is only approximated (Section 4.1, Example I). The paper argues that this restriction leads to long training times and to special-purpose fast samplers (Section 1).

The paper's answer is a regression objective for CNFs. Training needs no ODE solve: sample a time, a noise point, and a data point, compute a closed-form target velocity, and take a gradient step on a squared error (Section 3.2). Sampling is still an ODE solve, but now the path is yours to design, and the paper's straight-line choice cuts the number of solver steps.

The headline results, on the same U-Net architecture for every method (Section 6.1, Table 1), are the following. Each cell is NLL in bits per dimension, FID, and the average number of function evaluations (NFE) of the adaptive dopri5 solver:

| Dataset | DDPM loss | Score matching | Flow matching with diffusion path | Flow matching with OT path |
|---|---|---|---|---|
| CIFAR-10 | 3.12, 7.48, 274 | 3.16, 19.94, 242 | 3.10, 8.06, 183 | 2.99, 6.35, 142 |
| ImageNet 32x32 | 3.54, 6.99, 262 | 3.56, 5.68, 178 | 3.54, 6.37, 193 | 3.53, 5.02, 122 |
| ImageNet 64x64 | 3.32, 17.36, 264 | 3.40, 19.74, 441 | 3.33, 16.88, 187 | 3.31, 14.45, 138 |

Read the middle columns carefully. Flow matching on the diffusion path isn't uniformly better than DDPM: its CIFAR-10 FID of 8.06 is worse than DDPM's 7.48. The OT path is what wins on every metric. A model trained with OT on ImageNet 128x128 reaches 2.90 bits per dimension and an FID of 20.9 (Table 1, right).

## Key ideas

The paper's notation is the page's notation, with a few spots called out below. A sample $x_1 \sim q(x_1)$ is data, and $x_0 \sim p(x) = \mathcal{N}(0, I)$ is noise, so **time runs from noise at $t = 0$ to data at $t = 1$** (Section 3). This is the reverse of the [score-based SDE paper](/score-sde/), where $t = 0$ is data. Appendix D handles the conversion with Lemma 1: if $u_t$ generates $p_t$, then $-u_{1-t}$ generates $p_{1-t}$. The vector field being regressed is $v_t(x;\theta)$, the target is $u_t(x)$ or $u_t(x \mid x_1)$, the CNF flow is $\phi_t$, and the per-example (conditional) flow is $\psi_t$. This page writes $x_t$ for a point on a path at time $t$ and $\sigma_{\min}$ explicitly, where the paper's equations leave the point as $x$.

Every number below comes from a script in `scripts/`. Run any script from the paper folder to reproduce its output.

### A vector field that generates a path, and a marginal field built from per-example ones

**Intuition.** A vector field moves particles, and a moving crowd of particles has a density that changes in time. The continuity equation is the bookkeeping rule: density can only change by flowing in or out. The hard part of training a CNF is that you want the field for the whole data distribution, and nobody can write that down. The paper's first observation is that you don't need to. Pick a simple path for each single data point, then average the per-point fields with the right weights, and the average generates the average path.

**Math.** A vector field $v_t$ defines a flow $\phi_t$ by the ODE $\frac{\mathrm{d}}{\mathrm{d}t}\phi_t(x) = v_t(\phi_t(x))$ with $\phi_0(x) = x$ (Section 2, Equations 1 and 2). The flow pushes the prior forward, $p_t = [\phi_t]_* p_0$ (Equations 3 and 4), and $v_t$ generates $p_t$ exactly when the continuity equation holds (Appendix B, Equation 26):

$$
\frac{\partial}{\partial t}\,p_t(x) + \mathrm{div}\big(p_t(x)\,v_t(x)\big) = 0
$$

Now build the target. Give each data point $x_1$ a conditional path $p_t(x \mid x_1)$ with $p_0(x \mid x_1) = p(x)$, and let $p_1(x \mid x_1)$ be a narrow Gaussian around $x_1$. The marginal path and the marginal field are (Section 3.1, Equations 6 and 8):

$$
p_t(x) = \int p_t(x \mid x_1)\,q(x_1)\,\mathrm{d}x_1, \qquad
u_t(x) = \int u_t(x \mid x_1)\,\frac{p_t(x \mid x_1)\,q(x_1)}{p_t(x)}\,\mathrm{d}x_1
$$

At $t = 1$ the marginal $p_1$ is a mixture that closely approximates $q$ (Equation 7). **Theorem 1** (Section 3.1, proof in Appendix A) says that if each $u_t(\cdot \mid x_1)$ generates $p_t(\cdot \mid x_1)$, then $u_t$ generates $p_t$. The proof exchanges the integral and the derivative in the continuity equation.

**Tiny example, computed by `scripts/marginal_field.py`.** Take one-dimensional data with two equally likely points, $x_1 = -2$ and $x_1 = +2$, and the OT path with $\sigma_{\min} = 0.1$ (the path itself is the topic of a later idea). At $t = 0.5$ the standard deviation is $\sigma_t = 0.55$. At $x = 0.5$ the script prints:

| Quantity | $x_1 = -2$ | $x_1 = +2$ |
|---|---|---|
| Conditional density $p_t(x \mid x_1)$ | 0.017596 | 0.479829 |
| Conditional field $u_t(x \mid x_1)$ | $-4.4545$ | $2.8182$ |
| Posterior weight $p_t(x \mid x_1)\,q(x_1) / p_t(x)$ | 0.0354 | 0.9646 |

The marginal density is $p_t(0.5) = 0.248712$ and the marginal field is $u_t(0.5) = 0.0354 \cdot (-4.4545) + 0.9646 \cdot 2.8182 = 2.5609$. The point sits near $+2$'s path, so the field mostly follows that conditional field, with a small pull from the far one. At $x = 0$ the two weights are equal and the field is exactly $0$. The script also checks the continuity equation by finite differences at $t = 0.2, 0.5, 0.8$ on 41 grid points. The largest residual is $1.37 \times 10^{-8}$, so Theorem 1 holds on this example. Exercise 3 asks you to implement this field.

### Conditional flow matching regresses on the easy target

**Intuition.** With $u_t$ in hand you could regress onto it directly, but you can't evaluate it, because it needs the integral over all data. Denoising score matching has the same structure, and the paper borrows its trick: regress onto the per-example field instead. Each individual target is noisy and often wrong for the point it's attached to, but the *average* target at any given $(x, t)$ is the marginal field. A squared-error regression learns the conditional mean, so the noise averages out.

**Math.** The Flow Matching (FM) objective regresses on the marginal field (Section 3, Equation 5):

$$
\mathcal{L}_{\mathrm{FM}}(\theta) = \mathbb{E}_{t,\,p_t(x)}\,\lVert v_t(x) - u_t(x)\rVert^2
$$

The Conditional Flow Matching (CFM) objective regresses on the per-example field (Section 3.2, Equation 9), with $t \sim \mathcal{U}[0, 1]$, $x_1 \sim q(x_1)$, and $x \sim p_t(x \mid x_1)$:

$$
\mathcal{L}_{\mathrm{CFM}}(\theta) = \mathbb{E}_{t,\,q(x_1),\,p_t(x \mid x_1)}\,\lVert v_t(x) - u_t(x \mid x_1)\rVert^2
$$

**Theorem 2** (Section 3.2, proof in Appendix A) says that, up to a constant independent of $\theta$, the two are equal, so $\nabla_\theta \mathcal{L}_{\mathrm{FM}} = \nabla_\theta \mathcal{L}_{\mathrm{CFM}}$. The proof expands both squared norms. The $\lVert v_t\rVert^2$ terms match because the marginal path is the mixture in Equation 6. The cross terms match because the marginal field is the posterior-weighted average in Equation 8. The $\lVert u\rVert^2$ terms don't depend on $\theta$.

**Tiny example, computed by `scripts/marginal_field.py` and `scripts/conditional_paths.py`.** Use the same two-point data at $t = 0.5$ and compute both losses by numerical integration over $x$ for four candidate fields:

| Candidate $v_t(x)$ | $\mathcal{L}_{\mathrm{FM}}$ | $\mathcal{L}_{\mathrm{CFM}}$ | Difference |
|---|---|---|---|
| $0$ | 3.4423 | 4.8100 | 1.3677 |
| $1$ | 4.4423 | 5.8100 | 1.3677 |
| $2x$ | 2.6323 | 4.0000 | 1.3677 |
| $u_t$ (the marginal field) | 0.0000 | 1.3677 | 1.3677 |

The difference is the same $1.3677$ for every candidate, which is Theorem 2. It also exposes a fact that matters in practice: at the optimum, $\mathcal{L}_{\mathrm{CFM}}$ doesn't reach zero. It stops at $1.3677$ at this $t$, the variance of the per-example targets around their average. So a flat CFM curve is not a sign of a bad model.

For a single training pair the loss is cheap to evaluate. With $\sigma_{\min} = 0.1$, $x_0 = -1$, $x_1 = 2$, and $t = 0.5$, the target is $2.9$ (Equation 23). A field that outputs $0$ gets loss $8.41$, a field that outputs $1$ gets $3.61$, and one that outputs $2.9$ gets $0$. Exercise 1 asks you to build this pair and loss.

### Gaussian conditional paths: any $\mu_t$ and $\sigma_t$ give a field (Theorem 3)

**Intuition.** To use CFM you need a conditional path and its field for each data point. The paper picks Gaussian paths because they come with a canonical flow: shift and scale the noise. If the path is a Gaussian whose mean and standard deviation you choose, the vector field that moves a noise sample along it has a closed form. Diffusion paths are then one choice of $\mu_t$ and $\sigma_t$, and so is the optimal-transport path in the next idea.

**Math.** The conditional path is (Section 4, Equation 10)

$$
p_t(x \mid x_1) = \mathcal{N}\big(x \mid \mu_t(x_1),\ \sigma_t(x_1)^2 I\big),
$$

with $\mu_0(x_1) = 0$ and $\sigma_0(x_1) = 1$, so every path starts at standard noise, and $\mu_1(x_1) = x_1$ and $\sigma_1(x_1) = \sigma_{\min}$, a small number, so every path ends in a narrow Gaussian at $x_1$. The conditional flow is the affine map $\psi_t(x) = \sigma_t(x_1)\,x + \mu_t(x_1)$ (Equation 11), which pushes $\mathcal{N}(0, I)$ to $p_t(\cdot \mid x_1)$ (Equation 12). Substituting $x_t = \psi_t(x_0)$ turns the CFM loss into a regression on $\frac{\mathrm{d}}{\mathrm{d}t}\psi_t(x_0)$ (Equation 14):

$$
\mathcal{L}_{\mathrm{CFM}}(\theta) = \mathbb{E}_{t,\,q(x_1),\,p(x_0)}\,\Big\lVert v_t\big(\psi_t(x_0)\big) - \tfrac{\mathrm{d}}{\mathrm{d}t}\psi_t(x_0)\Big\rVert^2
$$

**Theorem 3** (Section 4, proof in Appendix A) solves for the field in closed form (Equation 15):

$$
u_t(x \mid x_1) = \frac{\sigma_t'(x_1)}{\sigma_t(x_1)}\,\big(x - \mu_t(x_1)\big) + \mu_t'(x_1)
$$

Here a prime is the time derivative. The proof inverts $\psi_t$ to get $x_0 = (x - \mu_t)/\sigma_t$ and substitutes into $\psi_t'(x_0) = \sigma_t' x_0 + \mu_t'$. The first term rescales the offset from the mean, and the second moves the mean.

The paper's diffusion examples fit this template. The variance-preserving (VP) path, with the time reversal from Lemma 1 applied, is (Section 4.1, Example I, Equation 18)

$$
p_t(x \mid x_1) = \mathcal{N}\big(x \mid \alpha_{1-t}\,x_1,\ (1 - \alpha_{1-t}^2)\,I\big), \qquad \alpha_s = e^{-\frac12 T(s)},\quad T(s) = \int_0^s \beta(r)\,\mathrm{d}r
$$

and Theorem 3 turns it into Equation 19:

$$
u_t(x \mid x_1) = -\frac{T'(1-t)}{2}\left[\frac{e^{-T(1-t)}\,x - e^{-\frac12 T(1-t)}\,x_1}{1 - e^{-T(1-t)}}\right]
$$

The paper shows in Appendix D that this is the vector field of the probability-flow ODE from the score-based SDE paper, in the $t = 0$ is noise convention. So flow matching on a diffusion path trains the *same* target as a diffusion model's ODE, with a regression loss in place of score matching. The variance-exploding path (Equations 16 and 17) fits the template the same way. Two caveats from the paper. These paths don't reach pure noise in finite time, so $p_0$ is only approximated by a Gaussian at sampling (Section 4.1). And at $t = 1$ the VP standard deviation is $0$, not $\sigma_{\min}$, so the paper's training and sampling stop short of $t = 1$ by $\epsilon$ (Appendix E.1).

**Tiny example, computed by `scripts/conditional_paths.py`.** Use the paper's schedule, $\beta(s) = \beta_{\min} + s(\beta_{\max} - \beta_{\min})$ with $\beta_{\min} = 0.1$ and $\beta_{\max} = 20$ (Appendix E.1), data $x_1 = 2$, and the noise draw $x_0 = -1$. The script prints the path and the field at three times, and it checks Equation 19 against Equation 15 with finite-difference derivatives:

| $t$ | $\alpha_{1-t}$ | $\mu_t$ | $\sigma_t$ | $u_t(x_t \mid x_1)$ from Eq. 19 | from Eq. 15 |
|---|---|---|---|---|---|
| 0.25 | 0.058664 | 0.1173 | 0.9983 | 0.9073 | 0.9073 |
| 0.50 | 0.281183 | 0.5624 | 0.9597 | 3.2399 | 3.2399 |
| 0.75 | 0.723657 | 1.4473 | 0.6902 | 5.5980 | 5.5980 |

The two equations agree. The velocity changes by a factor of six between $t = 0.25$ and $t = 0.75$, because the diffusion path barely moves the sample at first and then rushes toward the data. Exercise 2 asks you to write Theorem 3 and the closed-form derivatives for this path.

### The optimal-transport path: straight lines at constant speed

**Intuition.** The diffusion path was designed for a noising process, not for a sampler. If all you want is a path from noise to data, the simplest choice is to change the mean and the standard deviation linearly in time. Each noise sample then travels in a straight line at constant velocity. Straight lines are the shortest route and the easiest thing for an ODE solver to follow.

**Math.** The paper sets (Section 4.1, Example II, Equation 20)

$$
\mu_t(x) = t\,x_1, \qquad \sigma_t(x) = 1 - (1 - \sigma_{\min})\,t.
$$

Theorem 3 gives the field (Equation 21), the flow (Equation 22), and the CFM loss (Equation 23):

$$
u_t(x \mid x_1) = \frac{x_1 - (1 - \sigma_{\min})\,x}{1 - (1 - \sigma_{\min})\,t}, \qquad
\psi_t(x_0) = \big(1 - (1 - \sigma_{\min})\,t\big)\,x_0 + t\,x_1,
$$

$$
\mathcal{L}_{\mathrm{CFM}}(\theta) = \mathbb{E}_{t,\,q(x_1),\,p(x_0)}\,\big\lVert v_t(\psi_t(x_0)) - \big(x_1 - (1 - \sigma_{\min})\,x_0\big)\big\rVert^2.
$$

The regression target in Equation 23 doesn't depend on $t$. The paper calls this flow the optimal-transport displacement map between the two Gaussians $p_0(x \mid x_1)$ and $p_1(x \mid x_1)$, by Example 1.7 in McCann (1997) (Section 4.1). It also states a limit, and so should you: the conditional flow is optimal, but that does not make the *marginal* field an optimal-transport solution. The paper only expects the marginal field to stay relatively simple (end of Section 4.1).

**Tiny example, computed by `scripts/conditional_paths.py`.** Take $\sigma_{\min} = 0.1$, $x_1 = 2$, and $x_0 = -1$. Along the path:

| $t$ | $\mu_t$ | $\sigma_t$ | $\psi_t(x_0)$ | $u_t(\psi_t(x_0) \mid x_1)$ |
|---|---|---|---|---|
| 0.00 | 0.0000 | 1.0000 | $-1.0000$ | 2.9000 |
| 0.25 | 0.5000 | 0.7750 | $-0.2750$ | 2.9000 |
| 0.50 | 1.0000 | 0.5500 | 0.4500 | 2.9000 |
| 0.75 | 1.5000 | 0.3250 | 1.1750 | 2.9000 |
| 1.00 | 2.0000 | 0.1000 | 1.9000 | 2.9000 |

The velocity is $2.9 = x_1 - 0.9\,x_0$ at every time, and a finite-difference derivative of $\psi_t$ matches it. The point ends at $x_1 + \sigma_{\min} x_0 = 1.9$. Compare the VP column above: $0.9$, $3.2$, and $5.6$. The script also measures how far a point travels along the path, divided by the straight distance between its ends, for three noise draws. The OT path gives $1.0000$ every time. The VP path gives $1.0000$ for $x_0 = -1$ and $0$, but for $x_0 = +1$ it gives $1.4705$: the sample rises to $2.2361$, overshoots the data point at $2$, and comes back. That is the overshoot the paper describes in Figure 3 (Section 4.1).

### Sampling is an ODE solve, and straighter paths need fewer steps

**Intuition.** To draw a sample, draw $x_0 \sim \mathcal{N}(0, I)$ and integrate the learned field from $t = 0$ to $t = 1$ (Section 6.2). Every step costs one network evaluation. A solver with big steps follows a curved trajectory poorly, because it extrapolates along the tangent and falls off the curve. A straighter trajectory has a tangent that stays valid for longer.

**Math.** Euler's method takes $x \leftarrow x + h\,v_t(x)$ with $h = 1/n$, evaluating at the start of each step. The midpoint method takes $x \leftarrow x + h\,v_{t + h/2}\big(x + \tfrac h2 v_t(x)\big)$ and costs two evaluations per step. Euler has global error $O(h)$ and midpoint has $O(h^2)$. The paper's experiments use the adaptive dopri5 solver for Table 1, and fixed-step Euler, midpoint, and fourth-order Runge-Kutta for the low-NFE comparisons (Section 6.2). It reports that the OT model needs roughly 60% of the NFEs of the diffusion models to reach the same per-pixel error against a 1000-NFE solution (Figure 7, left).

**Tiny example, computed by `scripts/moons_euler.py`, `scripts/vp_vs_ot.py`, and `scripts/gaussian_solvers.py`.** Training a network isn't needed to see the effect, because on a finite dataset the CFM minimizer is just Equation 8 with $q$ the empirical distribution. The scripts use 200 points on two interleaved half-circle "moons" (noise $0.05$, built in NumPy) and the exact marginal field for the OT path with $\sigma_{\min} = 0.05$. They integrate from the same 1000 noise samples and compare with a 1000-step midpoint solution (the same reference idea as Figure 7). Euler with a few steps gives:

| Euler steps | Mean squared distance to reference | Mean distance to nearest data point |
|---|---|---|
| 5 | 0.05354 | 0.0958 |
| 10 | 0.02107 | 0.0583 |
| 20 | 0.00740 | 0.0470 |
| 50 | 0.00298 | 0.0424 |

The reference itself sits 0.0401 from the nearest data point on average. Five steps leaves samples more than twice that far off the curves, and 50 steps brings them within 6 percent. Midpoint at 10 calls (5 steps) gets a squared distance of $0.00372$, against $0.02107$ for 10 Euler steps. The moment check agrees: with 50 Euler steps the sample standard deviation is $[0.875, 0.494]$ against $[0.873, 0.500]$ for the data.

For the diffusion path (Equation 19), `scripts/vp_vs_ot.py` uses the same data and the mean distance to the nearest data point as a rough quality score, with 500 samples:

| Euler steps | OT path | VP path |
|---|---|---|
| 5 | 0.0968 | 0.1558 |
| 10 | 0.0595 | 0.0886 |
| 20 | 0.0467 | 0.0476 |
| 50 | 0.0419 | 0.0313 |
| 200 | 0.0399 | 0.0206 |

At 5 and 10 steps, OT is much closer to the data, which matches the paper's low-NFE finding (Section 6.2). At 50 steps and above the VP score is *lower*, and that is a property of the metric, not a win for VP. The VP path ends at $\sigma \to 0$ and lands almost exactly on training points, while the OT path keeps its $\sigma_{\min} = 0.05$ spread around the curve. Treat the table as evidence for the low-step regime only.

Finally, for a Gaussian target $\mathcal{N}(2, 0.5^2)$ the field and the exact flow are known, so the solver error can be measured directly. Each time the number of steps doubles, the Euler error falls by a factor of about 1.9 to 2.0, which is first order. The midpoint error falls by about 7.7 to 8.0 here. That is better than the guaranteed second-order factor of 4, because this field is affine in $x$. At 5 steps Euler's sample standard deviation is $0.3549$, against an exact $0.4806$, and at 50 steps it is $0.4666$. Exercise 4 asks you to write both solvers.

<!-- exercises -->

## Pitfalls

- **The time direction is the opposite of the score-based SDE paper.** Here $t = 0$ is noise and $t = 1$ is data, and sampling integrates *forward* with a positive step. Porting code from a diffusion library, where $t = 1$ is noise, flips the sign of the field (Lemma 1, Appendix D) and the direction of the loop.
- **The CFM loss doesn't go to zero.** The target $u_t(x \mid x_1)$ depends on which data point the noisy $x$ came from, so even a perfect model has a floor. In the page's example that floor is $1.3677$ at $t = 0.5$. Judge training by sample quality or by the gap to the floor, not by a loss near zero.
- **Regressing on the conditional target, not the marginal one.** $\mathcal{L}_{\mathrm{FM}}$ needs $u_t(x)$, which needs the data density. The training code computes $x_t = \psi_t(x_0)$ and the target $x_1 - (1 - \sigma_{\min})\,x_0$ from a pair $(x_0, x_1)$ and never forms a marginal field (Equation 23).
- **Dropping $\sigma_{\min}$ from the target.** For the OT path the target is $x_1 - (1 - \sigma_{\min})\,x_0$ (Equation 23), not $x_1 - x_0$. With $\sigma_{\min}$ small the difference is small, but the endpoint of the flow is then $x_1 + \sigma_{\min}x_0$, and the target and the point on the path must use the same $\sigma_{\min}$. Exercise 1's tests check both.
- **Straight conditional paths aren't straight marginal trajectories.** Every conditional trajectory of the OT path is a straight line, but the learned flow follows the *marginal* field, whose trajectories bend where data modes compete. The paper says the conditional flow being optimal doesn't make the marginal one optimal transport (Section 4.1). Models of this kind still need several solver steps, and `scripts/moons_euler.py` shows a clear error at 5 steps.
- **Averaging the loss over coordinates.** The loss is a squared Euclidean norm, so sum over the $d$ coordinates and average over the batch (Equations 5 and 9). Averaging over every element divides the loss by $d$, which silently changes the effective learning rate.
- **Evaluating the solver at the wrong time.** Euler uses the field at the start of the step, and midpoint uses $t + h/2$ for its second call. Using the end of the step turns the explicit method into a different, implicit scheme and gives no error message.
- **Weights in Equation 8 use the variance.** The posterior weight of a data point is a Gaussian in $x - t\,x_1$ with variance $\sigma_t^2$, centered at $t\,x_1$ rather than at $x_1$. Exercise 3's continuity-equation test catches a wrong center, a standard deviation where the variance belongs, and unnormalized weights. Compute the weights as a softmax: near $t = 1$ the variance is tiny and a direct exponential underflows.
- **The exact optimum memorizes.** On a finite dataset, the field from Equation 8 generates a mixture of narrow Gaussians around the training points (Equation 7). `scripts/moons_euler.py` ends with samples about 0.04 from a training point. A real network generalizes because it is smooth, and this page doesn't test that.
- **Likelihoods still need a divergence.** Training avoids ODE solves, but computing $\log p_1(x_1)$ for a data point integrates the divergence of $v_t$ along the flow (Appendix C, Equations 27 and 28), with the Hutchinson trace estimator in practice. FM improves training, not evaluation of likelihood.
- **The VP path is singular at $t = 1$.** Its standard deviation is $0$ there, so Equation 19 divides by zero, and training and sampling stop at $1 - \epsilon$ (Appendix E.1). The OT path has $\sigma_1 = \sigma_{\min} > 0$ and is defined for all $t \in [0, 1]$ (Section 4.1).

## Further reading

- [Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264), by Lipman et al. (2024), is a long tutorial from the same authors, with a PyTorch library at [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching).
- [An introduction to flow matching](https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html), by Fjelde, Mathieu, and Dutordoir, is a blog post that builds the same objective step by step, with animations.
- [TorchCFM](https://github.com/atong01/conditional-flow-matching) is a widely used library of conditional flow matching variants, including minibatch optimal-transport couplings.
- [Score-Based Generative Modeling through Stochastic Differential Equations](/score-sde/) is the paper whose probability-flow ODE this paper's diffusion path recovers in Appendix D.
- [Rectified flow](/rectified-flow/) reaches a similar straight-line objective from a different starting point and is the paper this one cites as concurrent work (Section 5).
- [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366), by Chen et al. (2018), introduced continuous normalizing flows (Section 2).
