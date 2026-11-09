---
id: "rectified-flow"
title: "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow"
authors: "Liu et al."
year: 2022
venue: "ICLR 2023"
arxiv: "2209.03003"
url: "https://arxiv.org/abs/2209.03003"
track: generative
tier: core
est_hours: 4
status: full
summary: "The paper learns an ODE that carries samples from one distribution to another by regressing a network onto the direction of the straight line between a noise sample and a data sample, taken at a random point along that line. Because an ODE flow can't cross itself, the learned flow rewires the lines where they intersect, and the paper proves that this lowers every convex transport cost and keeps the distribution at every time. Repeating the procedure on the flow's own noise-to-sample pairs, which the paper calls reflow, straightens the paths, so one Euler step on CIFAR-10 goes from an FID of 378 after the first round to 12.21 after the second, and 4.85 once that model is distilled."
related_course: "ml: flow models"
prereqs:
  - id: flow-matching
    reason: "The training loss here is that paper's conditional flow matching regression with the straight path $x_t = t\\,x_1 + (1 - t)\\,x_0$ and no minimum noise, so you need its argument that regressing onto a per-pair target learns the marginal field."
background: [optimal-transport-intuition]
course_links:
  - title: "Straighter paths and better solvers"
    url: "https://speech.ifkash.dev/lessons/057-straighter-paths-and-better-solvers"
  - title: "Flow matching"
    url: "https://speech.ifkash.dev/lessons/056-flow-matching"
self_check:
  - q: 'A training pair has $x_0 = -1$ and $x_1 = 3$. At $t = 0.25$, what is the point $X_t$ where the network is evaluated, what is its regression target, and what loss does a network that outputs $1$ there incur?'
    a: 'The point is $X_t = 0.25 \cdot 3 + 0.75 \cdot (-1) = 0$. The target is $X_1 - X_0 = 4$, the same at every $t$ on this line. An output of $1$ gives a squared error of $(1 - 4)^2 = 9$ (Section 2.1, Equation 1).'
    see: flow-matching
  - q: 'The straight lines from $X_0$ to $X_1$ can cross, but the rectified flow is an ODE. What does the ODE do at a crossing, and what does that change about the pairing of endpoints?'
    a: 'An ODE with a unique solution can''t have two paths pass through the same point at the same time. The flow therefore "rewires" the paths at the crossing: it follows the average direction of the lines there, so a particle ends at a different data point than its own line pointed to. The new pairing $(Z_0, Z_1)$ is deterministic and has no larger expected cost $\mathbb{E}[c(Z_1 - Z_0)]$ than the old one for every convex $c$ (Section 2.1, "Flows avoid crossing", and Theorem 3.5).'
    see: optimal-transport-intuition
  - q: 'The noise $X_0$ and the data $X_1$ are independent. What is $v^X(z, 0)$, and where does a single Euler step $Z_1 = Z_0 + v(Z_0, 0)$ from $Z_0 = z$ land?'
    a: 'At $t = 0$ the point $X_t$ is $X_0$, so $v^X(z, 0) = \mathbb{E}[X_1 - X_0 \mid X_0 = z] = \mathbb{E}[X_1] - z$ by independence (Section 2.1, Equation 2). One step lands at $z + \mathbb{E}[X_1] - z = \mathbb{E}[X_1]$ for every $z$: the mean of the data, not a data point. This is why a first rectified flow needs more than one step (Section 2.2, Figure 5).'
    see: flow-matching
---

## The problem

A continuous-time generative model turns noise into data by solving an ODE or an SDE with a learned drift. The paper names the cost in its introduction: drawing one sample takes many calls to the expensive network, while a GAN or a VAE takes one (Section 1). Score-based models and DDPM made training stable, and the probability-flow ODE and DDIM turned their SDEs into ODEs, but sampling still needs tens to hundreds of steps.

The paper's diagnosis is that the ODE's paths are the problem. If the path from a noise sample to a data sample were a straight line at constant speed, one Euler step would solve the ODE exactly. Real learned paths aren't straight, for two reasons that the paper separates:

- **The training pairs cross.** Draw a noise point and a data point independently, join them with a line, and train a network on the lines' directions. Two such lines through one point have different directions, and a network can output only one velocity there. The flow it learns has to bend (Section 2.1, "Flows avoid crossing").
- **The usual diffusion paths aren't straight to begin with.** The variance-preserving (VP) ODE of the score-based SDE paper and its sub-VP variant follow curves, and they move slowly early and fast late, so most of the update happens after $t = 0.5$ (Section 2.3.1, Figure 5).

The failure is easy to see on the paper's CIFAR-10 table. Take the first rectified flow, the one trained directly on independent pairs, and sample it with a single Euler step. Its FID is 378, because one step lands on the mean of the data, and the VP ODE with one step scores 451 (Section 5.2, Table 1). With one more round of the procedure in this paper, the same one-step sampler reaches an FID of 12.21, and a distilled version reaches 4.85. The ODE is the same kind of object. Its paths are straighter.

The paper's answer has two parts. First, a training loss that is a plain least-squares regression and that treats the generative problem as transporting any $\pi_0$ to any $\pi_1$ (Section 2.1, Equation 1). Second, *reflow*: train again on the pairs that the learned flow produces, which provably straightens the paths (Section 3.3). The paper also shows image-to-image translation and domain adaptation with the same algorithm (Sections 5.3 and 5.4); this page stays with generation and the theory.

## Key ideas

The paper's notation is this page's notation. $X_0 \sim \pi_0$ is the source (noise for generation) and $X_1 \sim \pi_1$ is the target (data). $X_t = t X_1 + (1 - t) X_0$ is the linear interpolation, $v$ is the velocity field, and $Z_t$ is the process that follows $\mathrm{d}Z_t = v(Z_t, t)\,\mathrm{d}t$. Time runs from $t = 0$ at noise to $t = 1$ at data, as in the [flow matching paper](/flow-matching/). The two papers name the same objects differently. What that paper calls the conditional target $u_t(x \mid x_1)$ with $\sigma_{\min} = 0$ is $X_1 - X_0$ here, and what it calls the marginal field $u_t$ is the paper's $v^X$ (Equation 2). Where this page writes a norm, the paper's $\lVert\cdot\rVert^2$ is a squared Euclidean norm. $N$ is the number of Euler steps and $h$ is the kernel bandwidth of Equation 5 (the paper's $h$ too), so the step size is always written $1/N$.

Every number below comes from a script in `scripts/`. Run any script from the repository root to reproduce its output. The first two examples share one toy problem: $X_0 \sim \mathcal{N}(0, 1)$ and $X_1 = \pm 2$ with probability $\tfrac12$ each, independent. It is small enough that the exact rectified field has a closed form.

### Regress onto the line direction, then follow the average

**Intuition.** The straight line from $X_0$ to $X_1$ is a perfect path, but following it needs $X_1$, which you don't have at sampling time. The paper calls this path *non-causal*: $X_t$ is defined by the future. The fix is to train a velocity field $v(x, t)$ that takes only the current position and time, and to ask it to match the line's direction as well as one function of $(x, t)$ can. At each $(x, t)$ many lines pass, and the best single answer under squared error is their average direction.

**Math.** The rectified flow induced from $(X_0, X_1)$ is the ODE $\mathrm{d}Z_t = v(Z_t, t)\,\mathrm{d}t$ with $Z_0 \sim \pi_0$. The velocity solves a least-squares problem (Section 2.1, Equation 1):

$$
\min_v \int_0^1 \mathbb{E}\Big[\big\lVert (X_1 - X_0) - v(X_t, t)\big\rVert^2\Big]\,\mathrm{d}t, \qquad X_t = t X_1 + (1 - t) X_0
$$

Algorithm 1 draws $t \sim \mathrm{Uniform}([0, 1])$ for the integral. The exact minimizer is the average direction of the lines through $x$ at time $t$ (Section 2.2, Equation 2):

$$
v^X(x, t) = \mathbb{E}\big[X_1 - X_0 \mid X_t = x\big]
$$

If $X_0$ given $X_1$ has a density $\rho$, Equation 4 writes this as $v^X(z, t) = \mathbb{E}\big[\tfrac{X_1 - z}{1 - t}\,\eta_t(X_1, z)\big]$, with $\eta_t$ a normalized density weight. The $1/(1 - t)$ factor comes from solving $X_t = z$ for $X_0$, and it makes the field singular at $t = 1$. **Theorem 3.3** (Section 3.1) says the flow reproduces the interpolation's marginal law at every time: $\mathrm{Law}(Z_t) = \mathrm{Law}(X_t)$ for all $t \in [0, 1]$. The proof (Equations 10 and 11) shows that both processes solve the same continuity equation with the same velocity $v^X$, and that the solution is unique.

The training loss is the conditional flow matching loss of the previous paper with $\sigma_{\min} = 0$. What this paper adds is the coupling view: it asks where $(Z_0, Z_1)$ lands, not only whether the marginals match.

**Tiny example, computed by `scripts/rewire_1d.py`.** For the toy problem, $X_0 = (x - t a)/(1 - t)$ when $X_1 = a$ and $X_t = x$, so $X_1 - X_0 = (a - x)/(1 - t)$. The posterior odds of $a = +2$ against $a = -2$ are $\exp\!\big(4 t x/(1 - t)^2\big)$, so

$$
v^X(x, t) = \frac{2\tanh\!\big(2 t x/(1 - t)^2\big) - x}{1 - t}.
$$

At $x = 0.5$ and $t = 0.5$ the argument of $\tanh$ is $2$, so the posterior mean of $X_1$ is $2 \tanh 2 = 1.9281$ and the field is $(1.9281 - 0.5)/0.5 = 2.8561$. The point $0.5$ sits closer to the line toward $+2$, so the field points up, but it is less than the $4$ that the nearest line alone would give, because some of the lines through that point head the other way. The script integrates the ODE from $Z_0 \sim \mathcal{N}(0, 1)$ with RK4 and compares the marginals with $X_t$, whose variance is $4t^2 + (1 - t)^2$:

| $t$ | $\mathrm{Var}\,X_t$ | $\mathrm{Var}\,Z_t$ | $\mathbb{E}\lvert X_t\rvert$ | $\mathbb{E}\lvert Z_t\rvert$ |
|---|---|---|---|---|
| 0.25 | 0.8125 | 0.8125 | 0.7267 | 0.7267 |
| 0.50 | 1.2500 | 1.2500 | 1.0085 | 1.0085 |
| 0.75 | 2.3125 | 2.3125 | 1.5000 | 1.5000 |
| 0.90 | 3.2500 | 3.2500 | 1.8000 | 1.8000 |

Two statistics match to four digits at every time, which is Theorem 3.3. Exercise 1 asks you to build the field without a network, with the kernel estimator of Equation 5.

### Flows can't cross, so the endpoints get rewired and the cost drops

**Intuition.** The lines of $X_t$ cross, but the paths of an ODE can't, so the flow must break at each crossing. The paper's picture is a road network: $X_t$ builds straight roads between $\pi_0$ and $\pi_1$, and the rectified flow is traffic on them that obeys one rule, no two cars in the same place at the same time heading in different directions. At an intersection the cars swap roads. The marginal traffic density is unchanged (Theorem 3.3), but each car now ends up at a different destination than the road it started on, and the new routes are, on average, shorter.

**Math.** Call $(Z_0, Z_1) = \mathrm{Rectify}((X_0, X_1))$ the rectified coupling (Section 3.2, Definition 3.4). **Theorem 3.5** says that for every convex cost $c$,

$$
\mathbb{E}\big[c(Z_1 - Z_0)\big] \le \mathbb{E}\big[c(X_1 - X_0)\big].
$$

The proof chains two applications of Jensen's inequality. The first moves $c$ inside $\int_0^1 v^X(Z_t, t)\,\mathrm{d}t$. The second moves it inside the conditional expectation $v^X(X_t, t) = \mathbb{E}[X_1 - X_0 \mid X_t]$. The marginals in Theorem 3.3 connect the two. The theorem holds for all convex $c$ at once, so rectification doesn't target any one cost. The paper adds that the result is not an optimal transport coupling in general (Section 3.4, Theorems 3.8 and 3.10, which this page returns to below).

**Tiny example, computed by `scripts/rewire_1d.py`.** For the toy problem, independent pairs have $\mathbb{E}\lvert X_1 - X_0\rvert^2 = \mathbb{E}X_1^2 + \mathbb{E}X_0^2 = 5$. The flow from Equation 2 carries every $Z_0 > 0$ to $+2$ and every $Z_0 < 0$ to $-2$, so $Z_1 = 2\,\mathrm{sign}(Z_0)$ up to $7 \times 10^{-10}$ on the script's grid. This is the monotone coupling. The script integrates both sides:

| Cost $c$ | $\mathbb{E}\,c(X_1 - X_0)$, independent | $\mathbb{E}\,c(Z_1 - Z_0)$, rectified |
|---|---|---|
| $\lvert x\rvert^2$ | 5.0000 | 1.8085 |
| $\lvert x\rvert$ | 2.0170 | 1.2361 |

The squared-cost value has the closed form $5 - 4\sqrt{2/\pi} = 1.8085$, so the numerical result matches it to four digits. Both costs fall, as Theorem 3.5 predicts. Compare the last line of the script's output: because $X_0$ and $X_1$ are independent, $v^X(z, 0) = \mathbb{E}[X_1] - z = -z$, so one Euler step sends every sample to $0$, the mean of $\pi_1$. The first rectified flow is a good model that can't be sampled in one step. That is the problem the next idea solves.

### Reflow straightens the flow, and the identity that proves it

**Intuition.** After one rectification the pairs $(Z_0, Z_1)$ are deterministic and cross less, but the *flow* that connects them can still bend, because its velocity changes along each path. Train again on the new pairs, with the same loss, and you get the rectified flow of a coupling whose lines cross less. Do it repeatedly and the paths straighten, because each round removes crossings and crossings are exactly what force the bending.

**Math.** Define *straightness* for a differentiable process $Z$ (Section 2.2, Equation 3):

$$
S(Z) = \int_0^1 \mathbb{E}\Big[\big\lVert (Z_1 - Z_0) - \dot Z_t\big\rVert^2\Big]\,\mathrm{d}t
$$

$S(Z) = 0$ means $Z_t = t Z_1 + (1 - t) Z_0$: a line at *constant speed*. A flow with $S(Z) = 0$ is solved exactly by one Euler step, $Z_1 = Z_0 + v(Z_0, 0)$. The second measure is $V((X_0, X_1))$ (Section 3.3, Theorem 3.6, Equation 12), which is zero exactly when the lines of $X_t$ don't intersect:

$$
V\big((X_0, X_1)\big) = \int_0^1 \mathbb{E}\Big[\big\lVert X_1 - X_0 - \mathbb{E}[X_1 - X_0 \mid X_t]\big\rVert^2\Big]\,\mathrm{d}t
$$

**Theorem 3.6** says a coupling is a fixed point of $\mathrm{Rectify}$ (a *straight* coupling) if and only if $V = 0$. Taking $c(x) = \lVert x\rVert^2$ in the proof of Theorem 3.5 gives the identity at the center of the argument (Section 3.3, Equation 13, which the paper prints without the squares; the proof uses them):

$$
\mathbb{E}\big[\lVert X_1 - X_0\rVert^2\big] - \mathbb{E}\big[\lVert Z_1 - Z_0\rVert^2\big] = S(Z) + V\big((X_0, X_1)\big)
$$

The cost that rectification removes is spent on straightness and on non-crossing. Applying it to each round $Z^{k+1} = \mathrm{RectFlow}((Z^k_0, Z^k_1))$ and summing gives **Theorem 3.7**:

$$
\sum_{k=0}^{K} \Big[S(Z^{k+1}) + V\big((Z^k_0, Z^k_1)\big)\Big] \le \mathbb{E}\big[\lVert X_1 - X_0\rVert^2\big], \qquad \min_{k \le K} S(Z^k) \le \frac{\mathbb{E}\lVert X_1 - X_0\rVert^2}{K}.
$$

The paper writes the second form as $O(1/K)$ (Section 2.2, after Equation 3). Because the cost can only fall, the sum of the straightness values and the crossing measures over all rounds is bounded, so they must shrink.

**Tiny example, computed by `scripts/straightness_measure.py` and `scripts/straightness_1d.py`.** First, $S$ on single paths from $(0, 0)$. A constant-speed path along the chord to $(3, 4)$ has $S = 0$. The same chord at growing speed, $Z_t = t^2(3, 4)$, has $\dot Z_t = 2t(3, 4)$, so $S = 25\int_0^1 (1 - 2t)^2\,\mathrm{d}t = 25/3 = 8.3333$, even though the path is a straight line. The half circle from $(0, 0)$ to $(2, 0)$ has $S = \pi^2 - 4 = 5.8696$. The script computes all three with the discrete formula of Exercise 2 and matches those values to four digits. Straightness in the paper's sense includes the speed.

Second, Equation 13 on the toy problem from before. The script computes $V$ from the posterior variance, $\mathrm{Var}(X_1 - X_0 \mid X_t = x) = 4\,\mathrm{sech}^2\!\big(2 t x/(1 - t)^2\big)/(1 - t)^2$, and $S(Z^1)$ from the simulated flow:

| Quantity | Value |
|---|---|
| $\mathbb{E}\lvert X_1 - X_0\rvert^2 - \mathbb{E}\lvert Z_1 - Z_0\rvert^2$ | 3.1915 |
| $S(Z^1)$ | 1.1562 |
| $V((X_0, X_1))$ | 2.0352 |
| $S(Z^1) + V$ | 3.1914 |

The two sides agree to the quadrature error. The first round spends most of the saved cost on removing crossings ($V = 2.04$), and the flow it produces still has $S(Z^1) = 1.16$. The second round starts from the coupling $(Z_0, 2\,\mathrm{sign}\,Z_0)$, which is monotone, so its lines don't cross and $V = 0$. Its rectified flow has $v(x, t) = (2\,\mathrm{sign}(x) - x)/(1 - t)$ on the support of $X_t$, and the script finds $S(Z^2) \approx 10^{-21}$ and a largest gap of $1.6 \times 10^{-14}$ from the straight line. In one dimension this is Lemma 3.9: a straight coupling is deterministic and monotone.

### Few-step sampling, and what distillation adds

**Intuition.** Straightness pays off when you discretize. Euler with $N$ steps is exact on a straight constant-speed path for every $N$, and the error of an Euler step grows with how much the velocity changes along a path. So the flow after reflow needs fewer steps for the same quality. Distillation is a separate, final step: fit a network $\hat T$ that maps $Z_0$ to $Z_1$ directly, so sampling is one network call.

**Math.** The paper simulates with Euler steps of size $1/N$, $\hat Z_{t + 1/N} = \hat Z_t + v(\hat Z_t, t)/N$ (Section 5, "Algorithm"). For a $k$-rectified flow, distillation uses $\hat T(z_0) = z_0 + v(z_0, 0)$ and the loss $\mathbb{E}\big[\lVert (Z^k_1 - Z^k_0) - v(Z^k_0, 0)\rVert^2\big]$, which is the $t = 0$ term of Equation 1 (Section 2.2, "Distillation"). Distillation and rectification differ in kind. Distillation fits the same coupling $(Z^k_0, Z^k_1)$, and rectification produces a new coupling with lower cost. The paper recommends distilling only at the end.

**Tiny example, computed by `scripts/straightness_1d.py` and `scripts/toy_reflow_2d.py`.** On the 1-D toy problem, the table gives the squared Wasserstein-2 distance between the $N$-step Euler samples and $\pi_1$. Here $4$ is a collapse to the mean and $0$ is exact. The second flow is the exact rectified flow of the monotone coupling:

| $N$ | 1-rectified | 2-rectified |
|---|---|---|
| 1 | 4.0000 | 0.0000 |
| 2 | 0.6015 | 0.0000 |
| 4 | 0.0121 | 0.0000 |

The 2-D toy puts the same effect into a setting with a non-trivial coupling. The data $\pi_1$ are two blobs at $(\pm 3, 0)$ with standard deviation $0.3$, $\pi_0 = \mathcal{N}(0, I)$, and the velocity is the kernel estimator of Equation 5 over $n = 400$ pairs with $h = 0.25$. The script runs one reflow round with 100 Euler steps standing in for the exact flow:

| Quantity | Independent pairs | 1-rectified | 2-rectified |
|---|---|---|---|
| Transport cost $\mathbb{E}\lVert Z_1 - Z_0\rVert^2$ | 11.0096 | 6.0185 | 6.1627 |
| Straightness $S(Z)$ | | 8.9738 | 0.1965 |

| $N$ | Euler error, 1-rectified | Euler error, 2-rectified |
|---|---|---|
| 1 | 9.1219 | 0.6646 |
| 2 | 4.5405 | 0.0138 |
| 4 | 1.3318 | 0.0016 |
| 8 | 0.5469 | 0.0003 |

The error is $\mathbb{E}\lVert Z^{(N)}_1 - Z^{(100)}_1\rVert^2$, the squared distance from a flow's own 100-step endpoint. The first rectification drops the cost from 11.01 to 6.02. The straightness falls by a factor of 46 in the second round, and two Euler steps of the 2-rectified flow (error 0.0138) beat eight steps of the 1-rectified one (0.5469). One step of the 1-rectified flow lands near $(-0.22, 0.02)$, between the blobs, as the 1-D example predicts. The 2-rectified cost, 6.16, is slightly above 6.02. Theorem 3.5 holds for the exact flow, and the kernel estimate on 400 points isn't the exact flow. Exercise 3 asks you to run this round.

The paper's CIFAR-10 numbers show the same shape at scale (Section 5.2, Table 1; each cell is the FID, with the distilled one-step FID in parentheses):

| Model, one Euler step | FID | One-step FID after distillation |
|---|---|---|
| 1-rectified flow | 378 | 6.18 |
| 2-rectified flow | 12.21 | 4.85 |
| 3-rectified flow | 8.15 | 5.21 |
| VP ODE | 451 | 16.23 |

The third round improves the undistilled one-step model and not the distilled one. With a full adaptive RK45 solver, more reflow *hurts*: the 1-, 2-, and 3-rectified flows score FIDs of 2.58, 3.36, and 3.96, using 127, 110, and 104 function evaluations (same table). The paper attributes this to error in estimating $v^X$ that builds up over rounds, and says that too many rounds aren't recommended (Section 2.2, after Theorem 3.7; Section 5.2). Reflow trades accuracy at many steps for accuracy at few.

### Why the linear path, and not the diffusion ODE

**Intuition.** Nothing forces the interpolation to be a line. The paper's more general version replaces $X_t$ by any differentiable curve, and the same regression still gives a flow with the right marginals. What the line buys is the straightening effect: it is the geodesic of Euclidean space, and the cost-reduction theorem depends on it. The paper's point about diffusion is that the probability-flow ODE is one such curve, a poor one.

**Math.** Section 2.3 defines the nonlinear rectified flow with weight $w_t > 0$ as the minimizer of (Equation 6)

$$
\min_v \int_0^1 \mathbb{E}\Big[w_t\,\big\lVert v(X_t, t) - \dot X_t\big\rVert^2\Big]\,\mathrm{d}t
$$

for any time-differentiable interpolation $X_t$. Theorem 3.3 still holds, so the marginals are preserved, but Theorem 3.5 and the straightening are no longer guaranteed. **Proposition 3.11** (Section 3.5) says that every probability-flow ODE is an instance of Equation 6 with $X_t = \alpha_t X_1 + \beta_t \xi$ and $\xi \sim \mathcal{N}(0, I)$. For the VP ODE, $\alpha_t = \exp\!\big(-\tfrac14 a (1 - t)^2 - \tfrac12 b (1 - t)\big)$ with $a = 19.9$ and $b = 0.1$, and $\beta_t = \sqrt{1 - \alpha_t^2}$ (Section 2.3.1, Equations 7 and 8). The rectified flow has $\alpha_t = t$ and $\beta_t = 1 - \alpha_t$. The paper names two defects of the VP choice: $\beta_t \ne 1 - \alpha_t$ makes the paths curved, and the exponential $\alpha_t$ makes the speed non-uniform. Its own figure (Figure 4) shows that reflow can't straighten the VP ODE.

**Tiny example, computed by `scripts/vp_vs_linear.py`.** The coefficient pairs $(\alpha_t, \beta_t)$ for the two paths:

| $t$ | Linear $(\alpha_t, \beta_t)$ | VP ODE $(\alpha_t, \beta_t)$ | VP $\alpha_t + \beta_t$ |
|---|---|---|---|
| 0.25 | (0.25, 0.75) | (0.0587, 0.9983) | 1.0569 |
| 0.50 | (0.50, 0.50) | (0.2812, 0.9597) | 1.2408 |
| 0.75 | (0.75, 0.25) | (0.7237, 0.6902) | 1.4138 |
| 0.90 | (0.90, 0.10) | (0.9467, 0.3221) | 1.2688 |

For a straight path, $\alpha_t + \beta_t = 1$ at every $t$. The VP sum rises to $1.41$, and the curve's largest distance from the straight segment is $0.29$. At $t = 0.5$ the VP signal coefficient has covered 28.1% of its way to 1, against 50% for the line, and only 27.6% of its total change happens in the first half of time. That is the "slow early, fast late" speed profile of Figure 5. Replacing the exponential $\alpha_t$ by $\alpha_t = t$ in the VP ODE gives uniform speed on the same continuous trajectories (Figure 5, last column).

<!-- exercises -->

## Pitfalls

- **Equation 13 is printed without squares.** The paper writes $\mathbb{E}[\lVert X_1 - X_0\rVert] - \mathbb{E}[\lVert Z_1 - Z_0\rVert] = S(Z) + V$. The proof takes $c(x) = \lVert x\rVert^2$, and `scripts/straightness_1d.py` confirms the identity with squared norms, where both sides are 3.19.
- **The first rectified flow isn't a one-step model.** Section 2.2 and Figure 5 say a single Euler step of the first flow generates the mean of $\pi_1$, and Table 1 gives an FID of 378 for it. The few-step claim needs a reflow round or distillation, and the headline one-step FID of 4.85 is the distilled 2-rectified flow (Section 5.2).
- **Reflow isn't distillation, and more rounds aren't better.** Rectification changes the coupling and lowers its cost. Distillation fits the existing coupling and speeds up inference (Section 2.2, "Distillation"). Too many reflow rounds accumulate estimation error in $v^X$, and the RK45 FIDs rise from 2.58 to 3.96 over three rounds (Table 1).
- **Straight in the paper's sense includes constant speed.** $S(Z)$ compares the velocity with the chord, so a path along a straight line at changing speed has $S > 0$ (the $t^2$ example above gives $25/3$). The paper's one-step guarantee needs $Z_t = t Z_1 + (1 - t) Z_0$ exactly (Section 2.2). Code that checks only whether a path is a line passes paths that one Euler step still gets wrong.
- **Straight doesn't mean optimal.** Rectification lowers every convex cost at once and doesn't target any one. In one dimension, straight couplings coincide with the monotone coupling, which is optimal for all convex costs (Lemma 3.9 and Theorem 3.10). In any dimension, a coupling that minimizes a strictly convex cost is straight (Theorem 3.8). Above one dimension the converse fails, and a straight coupling needn't minimize any cost (Section 3.4). The 2-D toy on this page isn't an optimal transport map.
- **Convex-cost guarantees are for the exact flow.** Theorems 3.3 and 3.5 assume that the ODE has a unique solution and that you solve Equation 1 exactly. With a kernel estimate on 400 points, the 2-D toy's cost rose from 6.02 to 6.16 on the second round. With a network, expect the same kind of small violation.
- **The exact field memorizes.** With a finite $\pi_1$ and a known $\rho$, the exact $v^X$ carries $Z_0$ to the training points, which overfits (Section 2.2, "Smooth function approximation"). The kernel estimator in Exercise 1 does this when $h$ is small against the spacing of the data. A network generalizes only because it is smooth, and the paper reports that more smoothing (a larger L2 penalty) also straightens the flow (Section 5.1, Figure 7).
- **Singular factor at $t = 1$.** The field in Equations 4 and 5 divides by $1 - t$. An Euler solver that evaluates the field at the end of a step divides by zero on the last step. Use the time at the start of the step, as Exercise 3's tests check. The network in Algorithm 2 never evaluates the formula at $t = 1$, because it regresses onto $X_1 - X_0$ directly.
- **Reflow data is generated, not sampled.** Round $k + 1$ trains on pairs $(Z^k_0, Z^k_1)$ that you create by solving the previous flow from each noise point and pairing the endpoint with that same noise point (Algorithm 4). On CIFAR-10 the paper generates 4 million such pairs and fine-tunes for 300,000 steps per round (Appendix A). Pairing an endpoint with a different noise point reverts the coupling to independent.
- **Sampling backward flips the sign.** To go from data to noise you solve $\mathrm{d}\tilde X_t = -v(\tilde X_t, t)\,\mathrm{d}t$ from $\tilde X_0 \sim \pi_1$ and set $X_t = \tilde X_{1-t}$ (Section 2.1). Reusing the forward loop with the same sign steps away from the data.
- **Euler error depends on the field and not only on $N$.** On the 1-D toy, 4 steps of the first flow reach a squared Wasserstein error of 0.0121, because that example is almost straight to begin with. In the 2-D toy, 8 steps still leave an error of 0.55. Judge a step count from measured error and don't copy it from another problem.

## Further reading

- [RectifiedFlow](https://github.com/gnobitab/RectifiedFlow) is the paper's code release, by the first author.
- [Flow Matching for Generative Modeling](/flow-matching/), by Lipman et al., trains the same objective from a different starting point and treats this paper as concurrent work (Section 5 of that paper).
- [Score-Based Generative Modeling through Stochastic Differential Equations](/score-sde/) is the paper whose probability-flow ODE this paper reinterprets in Section 2.3.1 and Section 3.5.
- [InstaFlow](https://arxiv.org/abs/2309.06380), by Liu et al. (2023), applies reflow and distillation to a text-to-image model for one-step generation.
- [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206), by Esser et al. (2024), is the paper behind Stable Diffusion 3, which trains rectified flow with the straight-line path.
