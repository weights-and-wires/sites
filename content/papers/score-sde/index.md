---
id: "score-sde"
title: "Score-Based Generative Modeling through Stochastic Differential Equations"
authors: "Song et al."
year: 2020
venue: "ICLR 2021"
arxiv: "2011.13456"
url: "https://arxiv.org/abs/2011.13456"
track: generative
tier: core
est_hours: 6
status: full
summary: "The paper replaces the finite chain of noise levels in diffusion models with a stochastic differential equation that gradually turns data into noise, and shows that running it backward in time only needs the score of each noisy distribution. A network trained with score matching supplies those scores, and SDE solvers then generate samples. The same scores also define a deterministic probability-flow ODE with the same marginal distributions, which enables exact likelihoods and adaptive fast sampling, and the framework reaches an FID of 2.20 on CIFAR-10."
related_course: "ml: diffusion"
prereqs:
  - id: ddpm
    reason: "The variance-preserving SDE is the continuous-time limit of the DDPM noising chain, so this paper reuses its closed-form kernel q(x_t|x_0) and reads its epsilon-prediction loss as score matching."
background: [sdes, score-function, fokker-planck]
course_links:
  - title: "Sampling faster: DDIM and the probability-flow ODE"
    url: "https://speech.ifkash.dev/lessons/053-fast-samplers"
  - title: "Diffusion from scratch"
    url: "https://speech.ifkash.dev/lessons/052-diffusion-from-scratch"
self_check:
  - q: 'The data are $x \sim \mathcal{N}(2, 0.25)$. What is the score $\nabla_x \log p(x)$ at $x = 1$, and which way does it point?'
    a: 'The score of $\mathcal{N}(\mu, v)$ is $-(x - \mu)/v = -(1 - 2)/0.25 = 4$. It is positive, so it points toward larger $x$, which is toward the mean and the region of higher density.'
    see: score-function
  - q: 'One Euler-Maruyama step of $\mathrm{d}x = -\tfrac12\beta\,x\,\mathrm{d}t + \sqrt{\beta}\,\mathrm{d}w$ uses $\beta = 2$, $x = 1$, $\Delta t = 0.1$, and a noise draw $z = 0.5$. What is the new $x$, and why does the noise term use $\sqrt{\Delta t}$?'
    a: 'The step is $x + (-\tfrac12 \cdot 2 \cdot 1)(0.1) + \sqrt{2 \cdot 0.1}\,(0.5) = 1 - 0.1 + 0.2236 = 1.1236$. Brownian motion has variance proportional to elapsed time, so its increment scales with $\sqrt{\Delta t}$, while the drift scales with $\Delta t$.'
    see: sdes
  - q: 'The Fokker-Planck equation for $\mathrm{d}x = f\,\mathrm{d}t + g\,\mathrm{d}w$ in one dimension is $\partial_t p = -\partial_x(f p) + \tfrac12 g^2\,\partial_x^2 p$. Why does the ODE $\mathrm{d}x = [f - \tfrac12 g^2\,\partial_x \log p]\,\mathrm{d}t$ give the same $p_t$?'
    a: 'Because $\partial_x^2 p = \partial_x(p\,\partial_x \log p)$, the diffusion term equals $\tfrac12 g^2\,\partial_x(p\,\partial_x\log p)$. Moving it inside the first derivative gives $\partial_t p = -\partial_x\{[f - \tfrac12 g^2\,\partial_x\log p]\,p\}$, which is the continuity equation of that ODE (Appendix D.1).'
    see: fokker-planck
---

## The problem

Before this paper there were two successful ways to generate data by reversing a noising process, and they were developed separately (Section 1). Score matching with Langevin dynamics (SMLD) learns the score at a fixed ladder of noise scales $\sigma_1 < \dots < \sigma_N$ and samples with Langevin MCMC at each scale (Section 2.1, Equations 1 and 2). Denoising diffusion probabilistic modeling (DDPM) trains a chain of $N$ reverse Gaussian steps and samples by ancestral sampling (Section 2.2, Equations 3 and 4). The paper calls the two together "score-based generative models" because the DDPM objective is also a weighted sum of denoising score matching terms (Section 2.2).

Both methods commit to a grid of $N$ noise levels at training time, and that has concrete costs:

- **One sampler per model.** A DDPM trained on $N = 1000$ steps has one way to sample: the 1000-step chain of Equation 4. If you want fewer network evaluations, nothing in the framework says how to take bigger steps.
- **No exact likelihood.** DDPM's training objective is a bound. Table 2 reports ELBO values of at most 3.70 and 3.75 bits per dimension on CIFAR-10 for DDPM trained with $L$ and with $L_{\text{simple}}$ (marked with an asterisk). The exact likelihood that this paper computes for a DDPM model is 3.28 bits per dimension (Table 2, row "DDPM"), and Section 4.3 notes that exact likelihoods beat the ELBO for the same DDPM model.
- **No deterministic map.** A sample path from noise to image is random at every step, so the same noise vector $x_T$ gives a different image each time. You can't use $x_T$ as a latent code that you edit and decode.

The paper's answer is to let the number of noise levels go to infinity. The noise grid becomes a continuous time $t \in [0, T]$, and the corruption becomes an Itô SDE (Section 3.1). Reversing it needs only the score of each noisy distribution (Section 3.2). This one framework contains SMLD and DDPM as discretizations (Section 3.4), admits any SDE solver and a Predictor-Corrector sampler (Sections 4.1 and 4.2), and has a deterministic twin, the probability-flow ODE, that gives exact likelihoods and fast adaptive sampling (Section 4.3). With architecture improvements the paper reports an FID of 2.20 and an Inception score of 9.89 on unconditional CIFAR-10 (Table 3, NCSN++ cont. (deep, VE)), and 2.99 bits per dimension (Table 2, DDPM++ cont. (deep, sub-VP)).

## Key ideas

The sections that follow use the paper's notation: the state is $x(t)$ for $t \in [0, T]$, with $x(0) \sim p_0$ the data distribution and $x(T) \sim p_T$ the prior; $p_t(x)$ is the marginal density of $x(t)$, and $p_{st}(x(t) \mid x(s))$ is the transition kernel (Section 3.1). The forward SDE has drift $f(x, t)$ and a scalar diffusion coefficient $g(t)$, and $w$ is the standard Wiener process. The score network is $s_\theta(x, t)$. This page works with the variance-preserving (VP) SDE and the paper's DDPM-matching schedule, so $T = 1$, $\bar\beta_{\min} = 0.1$, and $\bar\beta_{\max} = 20$ (Appendix C, Equation 32).

Two notation differences matter if you read the paper next to the [DDPM](/ddpm/) page. First, the paper writes $\alpha_i := \prod_{j \le i}(1 - \beta_j)$ for what the DDPM paper and the DDPM page call $\bar\alpha_i$ (Section 2.2). Second, the paper never names the mean scale and variance of the VP kernel. This page writes them $m(t) := e^{-\frac12 \int_0^t \beta(s)\,\mathrm{d}s}$ and $\sigma^2(t) := 1 - e^{-\int_0^t \beta(s)\,\mathrm{d}s}$, so that Equation 33 reads $p_{0t}(x(t) \mid x(0)) = \mathcal{N}(m(t)\,x(0),\ \sigma^2(t) I)$. In DDPM terms, $m(t)^2$ plays the role of $\bar\alpha_t$.

Every number below comes from a script in `scripts/`. Run any script from the paper folder to reproduce its output.

### Noise as an SDE, and DDPM as the VP-SDE

**Intuition.** Picture a particle that is pulled slightly toward the origin and jostled by random kicks. Run that long enough and it forgets where it started: its position is a draw from a standard normal. A noising process is exactly that, and the only design choices are how strong the pull and the kicks are at each moment. The paper treats the DDPM chain as this process with its time step shrunk to zero.

**Math.** The forward process is the solution of an Itô SDE (Section 3.1, Equation 5):

$$
\mathrm{d}x = f(x, t)\,\mathrm{d}t + g(t)\,\mathrm{d}w
$$

It has no trainable parameters. Taking the DDPM chain $x_i = \sqrt{1-\beta_i}\,x_{i-1} + \sqrt{\beta_i}\,z_{i-1}$ (Equation 10) and letting $N \to \infty$ with $\beta_i = \bar\beta(i/N)/N$ gives the VP SDE (Section 3.4, Equation 11; the limit is in Appendix B, Equations 23 to 25):

$$
\mathrm{d}x = -\tfrac12\beta(t)\,x\,\mathrm{d}t + \sqrt{\beta(t)}\,\mathrm{d}w, \qquad \beta(t) = \bar\beta_{\min} + t(\bar\beta_{\max} - \bar\beta_{\min})
$$

So $f(x, t) = -\tfrac12\beta(t)\,x$ and $g(t) = \sqrt{\beta(t)}$. Because the drift is affine, the kernel is Gaussian in closed form (Appendix B, Equation 29; for this schedule, Equation 33):

$$
p_{0t}(x(t) \mid x(0)) = \mathcal{N}\!\Big(x(t);\ x(0)\,e^{-\frac14 t^2(\bar\beta_{\max} - \bar\beta_{\min}) - \frac12 t\bar\beta_{\min}},\ \big(1 - e^{-\frac12 t^2(\bar\beta_{\max} - \bar\beta_{\min}) - t\bar\beta_{\min}}\big) I\Big)
$$

The paper's other main case, the Variance Exploding (VE) SDE, is the limit of the SMLD chain: $\mathrm{d}x = \sqrt{\mathrm{d}[\sigma^2(t)]/\mathrm{d}t}\,\mathrm{d}w$ (Section 3.4, Equation 9). The name "variance preserving" is literal. If the data have covariance $\Sigma(0)$, then $\Sigma(t) = I + e^{-\int_0^t \beta}\,(\Sigma(0) - I)$ (Appendix B, Equation 26), which equals $I$ for all $t$ when $\Sigma(0) = I$. The paper also introduces a sub-VP SDE (Equation 12) that does better on likelihoods; this page doesn't use it.

**Tiny example, computed by `scripts/vp_marginal.py` and `scripts/em_forward.py`.** With the paper's schedule, the script prints these kernel values:

| $t$ | $\beta(t)$ | $m(t)$ | $\sigma^2(t)$ |
|---|---|---|---|
| 0.1 | 2.090 | 0.946722 | 0.103718 |
| 0.5 | 10.050 | 0.281183 | 0.920936 |
| 1.0 | 20.000 | 0.006572 | 0.999957 |

At $t = 1$ the signal is down to 0.0066 and the variance is 0.99996, so $x(1)$ is almost exactly $\mathcal{N}(0, I)$. This matches the DDPM chain: with $N = 1000$ and $\beta_i = \bar\beta(i/N)/N$, the script gets $\bar\alpha_N = 3.996 \times 10^{-5}$, close to the $4.036 \times 10^{-5}$ of the DDPM paper's linear schedule from $10^{-4}$ to $0.02$ and to the continuous value $4.319 \times 10^{-5}$. Equation 26 checks out too. For data variance 0.25 the variance at $t = 0, 0.1, 0.5, 1$ is 0.2500, 0.3278, 0.9407, 1.0000, and for data variance 4 it is 4.0000, 3.6888, 1.2372, 1.0001. Unit variance stays at 1.0000 throughout.

Now simulate the SDE directly. Start every sample at $x(0) = 2$ and apply Euler-Maruyama, $x \leftarrow x - \tfrac12\beta(t)\,x\,\Delta t + \sqrt{\beta(t)\,\Delta t}\;z$, with 400,000 samples. The theory at $t = 0.5$ is mean 0.5624 and variance 0.9209. The script reports:

| Steps $N$ | Mean at $t = 0.5$ | Variance at $t = 0.5$ | Variance at $t = 1$ (theory 1.0000) |
|---|---|---|---|
| 10 | 0.5923 | 1.1148 | 1.8247 |
| 100 | 0.5665 | 0.9396 | 1.0534 |
| 1000 | 0.5623 | 0.9217 | 1.0050 |

A coarse grid is a poor approximation: with 10 steps the variance at $t = 1$ is off by 82 percent, because $\beta(t)\,\Delta t$ is not small. This is why DDPM needs about 1000 steps and why the continuous view helps. Exercise 1 asks you to implement the kernel and this simulator.

### The reverse-time SDE

**Intuition.** Time reversal of a random process usually isn't a nice process. For a diffusion it is: Anderson (1982) showed that running a diffusion backward gives another diffusion. The new drift is the old one plus a correction that pushes samples toward regions where the noisy distribution $p_t$ has more mass. That correction is the score, and it is the only piece that depends on the data.

**Math.** A remarkable result from Anderson (1982), which the paper quotes, says that the reverse of the diffusion in Equation 5 is the reverse-time SDE (Section 3.2, Equation 6):

$$
\mathrm{d}x = \big[f(x, t) - g(t)^2\,\nabla_x \log p_t(x)\big]\,\mathrm{d}t + g(t)\,\mathrm{d}\bar w
$$

Here $\bar w$ is a Wiener process when time flows backward from $T$ to $0$, and $\mathrm{d}t$ is an infinitesimal negative timestep. Once you know $\nabla_x \log p_t(x)$ for every $t$, you can start from $x(T) \sim p_T$ and simulate this equation to sample from $p_0$. The paper notes that the DDPM ancestral sampler (Equation 4) is one particular discretization of the reverse-time VP SDE (Section 4.1 and Appendix E). A generic alternative is the reverse diffusion sampler, which discretizes the reverse SDE the same way as the forward one (Appendix E, Equation 46).

**Tiny example, computed by `scripts/drift_example.py` and `scripts/reverse_toy.py`.** Take data $x(0) \sim \mathcal{N}(2, 0.5^2)$ in one dimension. Under the VP SDE every $p_t$ is Gaussian with mean $2\,m(t)$ and variance $0.25\,m(t)^2 + \sigma^2(t)$, so the score is known exactly. At $t = 0.5$ and $x = 1$ the script prints:

| Quantity | Value |
|---|---|
| $\beta(0.5)$ | 10.0500 |
| Mean of $p_t$ | 0.5624 |
| Variance of $p_t$ | 0.9407 |
| Score $\nabla_x \log p_t(1)$ | $-0.4652$ |
| Forward drift $f = -\tfrac12\beta x$ | $-5.0250$ |
| Reverse-SDE drift $f - g^2 \cdot \text{score}$ | $-0.3495$ |

The forward drift pulls $x$ hard toward 0, at $-5.025$. The reverse drift is nearly flat at $-0.3495$, because the score term $-g^2 \cdot \text{score} = +4.6755$ almost cancels it. Going backward in time, that cancellation is what stops samples from simply being pushed away from 0.

Now integrate Equation 6 with Euler-Maruyama from $t = 1$ down to $\epsilon = 10^{-3}$, starting from $\mathcal{N}(0, 1)$ with 100,000 samples. The target at $t = \epsilon$ is mean 1.9999 and standard deviation 0.5001. With 1000 steps the script gets mean 2.0004 and standard deviation 0.5006. With 50 steps it gets 2.0253 and 0.5100, and with 10 steps it gets 2.1196 and 0.5728. Exercise 2 asks you to implement the score and the reverse drift.

### Score matching trains the reverse drift

**Intuition.** The reverse SDE needs $\nabla_x\log p_t(x)$, which is the score of a distribution you can't write down, because $p_t$ mixes the unknown data with noise. The trick from denoising score matching is that the score of the *conditional* kernel $p_{0t}(x(t) \mid x(0))$ is known in closed form. A network trained to match that conditional score at random noisy points learns the marginal score on average.

**Math.** Train a time-dependent score model $s_\theta(x, t)$ with a continuous generalization of the SMLD and DDPM objectives (Section 3.3, Equation 7):

$$
\theta^* = \arg\min_\theta\ \mathbb{E}_t\Big\{\lambda(t)\,\mathbb{E}_{x(0)}\mathbb{E}_{x(t) \mid x(0)}\Big[\big\|s_\theta(x(t), t) - \nabla_{x(t)} \log p_{0t}(x(t) \mid x(0))\big\|_2^2\Big]\Big\}
$$

Here $t$ is uniform on $[0, T]$, $x(0) \sim p_0$, $x(t) \sim p_{0t}(\cdot \mid x(0))$, and $\lambda(t) > 0$ is a weighting. With enough data and capacity, the optimum satisfies $s_{\theta^*}(x, t) = \nabla_x \log p_t(x)$ almost everywhere (Section 3.3). The paper typically chooses $\lambda \propto 1 / \mathbb{E}[\|\nabla_{x(t)} \log p_{0t}(x(t) \mid x(0))\|_2^2]$, so that every time contributes at a comparable scale.

For the VP SDE the kernel is Gaussian, so with $x(t) = m(t)\,x(0) + \sigma(t)\,\epsilon$ and $\epsilon \sim \mathcal{N}(0, I)$ the target is

$$
\nabla_{x(t)} \log p_{0t}(x(t) \mid x(0)) = -\frac{x(t) - m(t)\,x(0)}{\sigma^2(t)} = -\frac{\epsilon}{\sigma(t)}
$$

Its expected squared norm per dimension is $1/\sigma^2(t)$, so $\lambda = \sigma^2(t)$ is the choice above. Then $\lambda\,\|s_\theta + \epsilon/\sigma\|^2 = \|\sigma\,s_\theta + \epsilon\|^2$, which is the $\epsilon$-prediction loss of DDPM with $\epsilon_\theta = -\sigma\,s_\theta$. This is the sense in which the paper says Equation 3 is a weighted sum of denoising score matching objectives (Section 2.2). The paper stresses that Equation 7 needs the kernel $p_{0t}$ in closed form, which holds when the drift is affine (Section 3.3). For general SDEs it points to sliced score matching (Appendix A, Equation 19).

**Tiny example, computed by `scripts/dsm_example.py`.** Take $t = 0.5$, $x(0) = 2$, and a noise draw $\epsilon = -1$. Then $m = 0.2812$, $\sigma^2 = 0.9209$, and $x(t) = 0.2812 \cdot 2 + 0.9597 \cdot (-1) = -0.3973$. The target score is $-(x(t) - m\,x(0))/\sigma^2 = 1.0420$, which equals $-\epsilon/\sigma = 1.0420$. The weight is $\lambda = \sigma^2 = 0.9209$. Three candidate models give these weighted losses:

| Model | Weighted loss | $(\hat\epsilon - \epsilon)^2$ with $\hat\epsilon = -\sigma s$ |
|---|---|---|
| Exact, $s = -\epsilon/\sigma$ | 0.0000 | 0.0000 |
| Zero score | 1.0000 | 1.0000 |
| $s = -x(t)$ | 0.3828 | 0.3828 |

The two columns agree row by row, which is the equivalence with $\epsilon$-prediction. Exercise 3 asks you to implement the loss.

### The probability-flow ODE

**Intuition.** The reverse SDE is random: it adds fresh noise at every step. But the only thing you need from it is that, at each time $t$, the cloud of samples has density $p_t$. The paper shows that a deterministic flow can carry the same cloud through the same sequence of densities. You get the same marginals with no randomness, so a fixed starting point $x(T)$ maps to a fixed output, and you can run the map in either direction.

**Math.** For every diffusion there is a deterministic process whose trajectories share the marginals $\{p_t(x)\}_{t=0}^T$ of the SDE. It satisfies the probability-flow ODE (Section 4.3, Equation 13):

$$
\mathrm{d}x = \Big[f(x, t) - \tfrac12 g(t)^2\,\nabla_x \log p_t(x)\Big]\,\mathrm{d}t
$$

Compare this with Equation 6: the score coefficient is halved and the noise term is gone. The reason is the Fokker-Planck equation, which the paper calls Kolmogorov's forward equation (Appendix D.1, Equation 35). For the scalar-$g$ case in $d$ dimensions it reads

$$
\frac{\partial p_t(x)}{\partial t} = -\nabla\cdot\big[f(x, t)\,p_t(x)\big] + \tfrac12 g(t)^2\,\Delta p_t(x)
$$

Since $\Delta p = \nabla\cdot(p\,\nabla\log p)$, the diffusion term can be absorbed into the drift. The result is $\partial_t p_t = -\nabla\cdot\{[f - \tfrac12 g^2\nabla\log p_t]\,p_t\}$, which is the Liouville equation of an ODE with that drift (Appendix D.1, Equation 37). The general matrix-valued version is Equation 17.

Replacing the score with $s_\theta$ turns this into a neural ODE (Section 4.3). The paper lists four consequences:

- **Exact likelihood.** The instantaneous change of variables formula gives $\log p_0(x(0)) = \log p_T(x(T)) + \int_0^T \nabla\cdot\tilde f_\theta(x(t), t)\,\mathrm{d}t$ (Appendix D.2, Equation 39).
- **Latent codes.** Integrating forward encodes a datapoint as $x(T)$, which you can interpolate or rescale, then decode by integrating backward (Section 4.3).
- **Unique encoding.** The forward SDE has no trainable parameters, so with enough data and capacity the encoding depends only on the data distribution (Section 4.3, Appendix D.5).
- **Efficient sampling.** A black-box ODE solver can cut the number of score evaluations by over 90 percent without visibly hurting quality (Section 4.3, Figure 3). The paper uses RK45 with `atol = rtol = 1e-5` and $\epsilon = 10^{-3}$ for VP SDEs (Appendix D.4).

**Tiny example, computed by `scripts/drift_example.py`, `scripts/reverse_toy.py`, and `scripts/fokker_planck.py`.** At the same point as before ($t = 0.5$, $x = 1$), the probability-flow drift is $f - \tfrac12 g^2 \cdot \text{score} = -5.0250 + 2.3377 = -2.6873$, compared with $-0.3495$ for the reverse SDE. Run both on the same Gaussian data with 100,000 samples and $\epsilon = 10^{-3}$ (the target is mean 1.9999 and standard deviation 0.5001):

| Steps $N$ | SDE mean | SDE std | ODE mean | ODE std |
|---|---|---|---|---|
| 10 | 2.1196 | 0.5728 | 2.0475 | 0.4852 |
| 50 | 2.0253 | 0.5100 | 2.0035 | 0.4942 |
| 1000 | 2.0004 | 0.5006 | 1.9917 | 0.4981 |

Both recover the data distribution. The ODE is closer at small $N$ and the SDE is closer at $N = 1000$. The ODE is a single deterministic trajectory per starting point, so Euler's discretization error does not average out with noise, and a small bias remains at 1000 steps (mean 1.9917 against 1.9999). To see determinism, draw five prior samples once and run each method twice. The ODE returns the same five outputs, `[2.0885 1.7327 1.7875 0.7736 2.8936]`, both times. The SDE returns `[1.8963 0.8139 2.2205 2.6524 1.7282]` the first time and `[1.9965 2.5103 2.7146 2.0904 2.6962]` the second.

The Fokker-Planck claim is also testable. For the same Gaussian data, `scripts/fokker_planck.py` estimates $\partial_t p_t$ by a central difference in $t$ and compares it with both right-hand sides, using finite differences in $x$. At $t = 0.3$ the largest $|\partial_t p|$ is 1.4325. The largest residual is $1.02 \times 10^{-5}$ for the SDE's Fokker-Planck equation and $4.18 \times 10^{-6}$ for the ODE's continuity equation, so the two equations agree with the true $\partial_t p_t$ to within discretization error. Exercise 2 includes this check as a test, and it fails if the ODE keeps the full factor $g^2$.

Finally, a 2-D toy model, computed by `scripts/toy2d_pf_ode.py`. The data are a three-mode Gaussian mixture with weights $(0.5, 0.3, 0.2)$, means $(-2, -1)$, $(2, -1)$, $(0, 2)$, and standard deviation 0.3 per mode. Under the VP SDE every $p_t$ is also a mixture, so its score is exact. Euler steps on Equation 13 from $t = 1$ to $\epsilon = 10^{-3}$ with 5000 starting points give these mode weights:

| Sampler | Mode 0 | Mode 1 | Mode 2 |
|---|---|---|---|
| Target | 0.500 | 0.300 | 0.200 |
| ODE, 20 steps | 0.484 | 0.312 | 0.205 |
| ODE, 500 steps | 0.488 | 0.312 | 0.200 |
| SDE, 500 steps | 0.504 | 0.304 | 0.192 |

With 500 ODE steps the sample means are within 0.007 of the true centers and the per-axis standard deviations are between 0.289 and 0.303, against 0.3. Even 20 steps are enough to land in the right modes, which is the efficiency argument for the ODE. Exercise 4 asks you to implement this sampler.

<!-- exercises -->

## Pitfalls

- **The sign of $\mathrm{d}t$.** In Equations 6 and 13, $\mathrm{d}t$ is a negative timestep (Section 3.2). If you integrate from $t = 1$ to $t = \epsilon$ with a positive step size, the update moves the samples the wrong way along the drift, so they do not denoise.
- **Halving the score in the SDE or not halving it in the ODE.** The reverse SDE uses $g^2\,\nabla\log p_t$ and the probability-flow ODE uses $\tfrac12 g^2\,\nabla\log p_t$. Both versions run without error. Exercise 2 shows that a full coefficient in the ODE breaks the Fokker-Planck equation, so the marginals drift away from $p_t$.
- **$\Delta t$ versus $\sqrt{\Delta t}$.** An Euler-Maruyama step multiplies the drift by $\Delta t$ and the noise by $\sqrt{\beta\,\Delta t}$. Using $\Delta t$ for the noise makes the process nearly deterministic at small step sizes. Using $\beta$ in place of $\sqrt\beta$ gives the wrong variance.
- **Mixing up the two scores.** The network learns the score of the marginal $p_t$, but the training target is the score of the kernel $p_{0t}(x(t) \mid x(0))$ (Equation 7). Putting the marginal score into the loss needs $p_t$, which you don't have. Substituting $-\epsilon/\sigma$ without the $\sigma^2$ weight changes the objective, not just the scale.
- **Forgetting $\lambda(t)$.** Without $\lambda$, the target has norm about $1/\sigma(t)$, which blows up as $t \to 0$. Small times then dominate the gradient. The paper picks $\lambda \propto 1/\mathbb{E}\|\nabla\log p_{0t}\|^2$ (Section 3.3).
- **Integrating all the way to $t = 0$.** The kernel variance vanishes as $t \to 0$, which makes the score blow up. The paper observes numerical instability for training and sampling at $t = 0$ and stops at $\epsilon$: $10^{-3}$ for VP SDEs with ODE solvers and $10^{-5}$ for VE SDEs (Appendices C and D.4). For sampling, the choice of $\epsilon$ matters for Inception score and FID, and for likelihood a smaller $\epsilon$ is generally better (Appendix C).
- **Expecting the ODE to match the SDE path by path.** The two share the marginals $p_t$ at every $t$, not the joint distribution across times. A given $x(T)$ goes to a different output under each method.
- **Expecting ODE samples to be better.** Without a corrector, probability-flow samples typically have a worse (higher) FID than SDE samples, and the quality depends on the SDE: VE ODE samplers are much worse than VP ones, especially in high dimensions (Appendix D.4).
- **Treating the VP and VE choices as interchangeable.** VE SDEs gave better sample quality in the paper's experiments, and VP and sub-VP gave better likelihoods (Section 4.4). Picking an SDE is a modeling decision.
- **A mismatched time range.** The schedule $\beta(t)$ is defined on $t \in [0, 1]$ with $\bar\beta_{\min} = 0.1$ and $\bar\beta_{\max} = 20$. A discrete DDPM schedule from $10^{-4}$ to $0.02$ over 1000 steps corresponds approximately to $\beta_i = \bar\beta(i/N)/N$. Mixing the per-step $\beta_i$ with the rate $\beta(t)$ is off by a factor of $N$.

## Further reading

- [The official implementation](https://github.com/yang-song/score_sde) accompanies the paper.
- [Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.net/blog/2021/score/), by Yang Song, is the author's blog post that walks through score matching, Langevin dynamics, and the SDE view.
- [What are diffusion models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/), by Lilian Weng, covers the DDPM and score-based views side by side.
- [Denoising Diffusion Probabilistic Models](/ddpm/) is the discrete chain that Section 3.4 and Appendix B take to the continuous limit.
- [Reverse-time diffusion equation models](https://doi.org/10.1016/0304-4149(82)90051-5), by Anderson (1982), is the source of the reverse-time SDE in Equation 6.
