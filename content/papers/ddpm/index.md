---
id: "ddpm"
title: "Denoising Diffusion Probabilistic Models"
authors: "Ho et al."
year: 2020
venue: "NeurIPS 2020"
arxiv: "2006.11239"
url: "https://arxiv.org/abs/2006.11239"
track: generative
tier: core
est_hours: 6
status: full
summary: "A diffusion model fixes a forward chain that adds a little Gaussian noise at each of 1000 steps until the data is pure noise, then trains a network to run the chain backward. Because the forward chain has a closed form, training picks a random step, noises a datapoint in one shot, and asks the network to predict the noise that was added. That choice turns the variational bound into a plain mean squared error, and sampling by repeated denoising reaches an FID of 3.17 on CIFAR10."
related_course: "ml: diffusion"
prereqs:
  - id: vae
    reason: "A diffusion model is a latent-variable model trained on an evidence lower bound, with a fixed encoder and KL terms between Gaussians as in the VAE objective."
  - id: unet
    reason: "The network that predicts the noise is a U-Net with skip connections between matching resolutions, so its output has the same shape as its input."
background: [gaussian-noise, markov-chains, kl-divergence]
course_links:
  - title: "Diffusion from scratch"
    url: "https://speech.ifkash.dev/lessons/052-diffusion-from-scratch"
  - title: "Sampling faster: DDIM and the probability-flow ODE"
    url: "https://speech.ifkash.dev/lessons/053-fast-samplers"
self_check:
  - q: 'Two Gaussians over the same variable have means $\mu_1 = 0$ and $\mu_2 = 1$ and the same variance $\sigma^2 = 0.5$. What is $D_{KL}(\mathcal{N}(\mu_1, \sigma^2) \,\|\, \mathcal{N}(\mu_2, \sigma^2))$?'
    a: 'With equal variances the KL divergence is $(\mu_1 - \mu_2)^2 / (2\sigma^2) = 1 / (2 \cdot 0.5) = 1$. The paper uses this fact to turn each KL term of the bound into a squared distance between two means.'
    see: kl-divergence
  - q: 'Start from $x_0 = 2$ and add noise twice: $x_1 = \sqrt{0.9}\,x_0 + \sqrt{0.1}\,\epsilon_1$, then $x_2 = \sqrt{0.8}\,x_1 + \sqrt{0.2}\,\epsilon_2$, with independent $\epsilon_1, \epsilon_2 \sim \mathcal{N}(0, 1)$. What are the mean and variance of $x_2$?'
    a: 'The mean is $\sqrt{0.9 \cdot 0.8}\cdot 2 = \sqrt{0.72}\cdot 2 \approx 1.6971$. The variance is $0.8 \cdot 0.1 + 0.2 = 0.28 = 1 - 0.72$. Variances of independent Gaussian terms add, and a constant scale multiplies the variance by its square.'
    see: gaussian-noise
  - q: 'In a VAE, the encoder $q_\phi(z|x)$ is learned and the KL term to the prior is part of the loss. In a diffusion model, the term $L_T = D_{KL}(q(x_T|x_0) \,\|\, p(x_T))$ is dropped from training. Why?'
    a: 'The forward process has fixed variances $\beta_t$ and no learnable parameters, so $q$ has nothing to optimize and $L_T$ is a constant (Section 3.1). It is also tiny, because the schedule leaves almost no signal at $x_T$.'
    see: vae
---

## The problem

By 2020, several families of generative models produced striking samples: GANs, autoregressive models, flows, and VAEs (Section 1). Energy-based models and score matching had also reached image quality comparable to GANs. Diffusion probabilistic models had been proposed in 2015 (reference [53] in the paper), and they are straightforward to define and efficient to train. But the paper says that, to the authors' knowledge, nobody had shown they can generate high quality samples (Section 1).

Two design questions stood in the way:

- **What does the network predict?** The variational bound (Equation 5) has one KL term per step. It isn't obvious which output makes those terms easy to learn: a mean, a clean image, or something else.
- **Which loss gives the best samples?** The bound weights every step in a particular way. Whether that weighting is the right one for sample quality is a separate question.

The paper makes the network predict the noise $\epsilon$, shows that this choice connects the bound to denoising score matching, and trains on a simplified, unweighted version (Sections 3.2 and 3.4). On unconditional CIFAR10 the result reaches an Inception score of 9.46 and an FID of 3.17 (Table 1), better than most models in the literature, including class-conditional ones. The negative log likelihood is not competitive: at most 3.75 bits per dimension (Table 1). The authors say so directly, and they show that most of the codelength describes imperceptible details (Section 4.3).

## Key ideas

The sections that follow use the paper's notation. The data is $x_0 \sim q(x_0)$ and the latents $x_1, \dots, x_T$ have the same dimension. The noise variances are $\beta_t$, with $\alpha_t := 1 - \beta_t$ and $\bar\alpha_t := \prod_{s=1}^t \alpha_s$ (Section 2). The network is $\epsilon_\theta(x_t, t)$. Every number comes from a script in `scripts/`. Run any script from the paper folder to reproduce the output.

Timesteps run from $t = 1$ to $T$ in the paper. The exercise code stores them in 0-based arrays, so array index $t - 1$ holds timestep $t$. This page also uses $\bar\alpha_0 := 1$, which is a convention for the first step and not the paper's own definition.

### The fixed forward process, in closed form

**Intuition.** Take a datapoint and add a tiny amount of Gaussian noise. Repeat 1000 times, shrinking the signal slightly each time so that the total variance stays under control. By the end, nothing of the datapoint is left, and what remains is a sample from a standard normal. Nobody learns this process; it is a fixed recipe for destroying data. The learned part is its reverse.

**Math.** The forward process, which the paper also calls the diffusion process, is a Markov chain that adds Gaussian noise according to a schedule $\beta_1, \dots, \beta_T$ (Section 2, Equation 2):

$$
q(x_{1:T} | x_0) := \prod_{t=1}^{T} q(x_t | x_{t-1}), \qquad q(x_t | x_{t-1}) := \mathcal{N}\!\left(x_t;\, \sqrt{1-\beta_t}\,x_{t-1},\, \beta_t I\right)
$$

A sum of Gaussian steps is Gaussian, so you can jump from $x_0$ to any $x_t$ at once (Section 2, Equation 4):

$$
q(x_t | x_0) = \mathcal{N}\!\left(x_t;\, \sqrt{\bar\alpha_t}\,x_0,\, (1-\bar\alpha_t) I\right)
$$

Equivalently, $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ with $\epsilon \sim \mathcal{N}(0, I)$. This is the reparameterization $x_t(x_0, \epsilon)$ that Section 3.2 uses before Equation 9. The closed form is what makes training cheap: you never simulate the chain to reach step $t$.

The paper fixes $T = 1000$ and sets the $\beta_t$ to constants increasing linearly from $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$ (Section 4). It scales the data to $[-1, 1]$ (Section 3.3), so that the network always sees inputs on the same scale as the standard normal prior.

**Tiny example, computed by `scripts/forward_example.py`.** Take a four-step schedule with $\beta = (0.1, 0.2, 0.3, 0.4)$, so that you can check it by hand. Then $\alpha = (0.9, 0.8, 0.7, 0.6)$ and $\bar\alpha = (0.9, 0.72, 0.504, 0.3024)$. For $t = 3$ the signal scale is $\sqrt{0.504} = 0.7099$ and the noise variance is $1 - 0.504 = 0.496$.

Now start at $x_0 = 2$ and run the chain with Equation 2 for 400,000 samples. After three steps the sample mean is 1.4209 and the variance is 0.4954. Equation 4 predicts a mean of $2 \cdot 0.7099 = 1.4199$ and a variance of 0.4960. The two routes agree to within sampling error. Exercise 1 asks you to implement the schedule and the closed form and to run this comparison.

With the paper's schedule, the script prints these values:

| $t$ | $\bar\alpha_t$ | Signal $\sqrt{\bar\alpha_t}$ | Noise $\sqrt{1-\bar\alpha_t}$ |
|---|---|---|---|
| 1 | 0.999900 | 0.9999 | 0.0100 |
| 100 | 0.897018 | 0.9471 | 0.3209 |
| 500 | 0.078587 | 0.2803 | 0.9599 |
| 1000 | 0.000040 | 0.0064 | 1.0000 |

At $t = 1000$ the signal scale is 0.0064. For a coordinate $x_0 = 1$, the script computes $D_{KL}(q(x_T | x_0) \,\|\, \mathcal{N}(0, I)) = 2.9 \times 10^{-5}$ bits. The paper reports $L_T \approx 10^{-5}$ bits per dimension averaged over its data (Section 4). Both say the same thing: $x_T$ is nearly pure noise, so starting the reverse chain from $\mathcal{N}(0, I)$ costs almost nothing.

### The reverse process and the bound

**Intuition.** To generate, run the chain backward: start from noise and remove a little of it at a time. Each backward step is a Gaussian whose mean a network predicts. You train the network the usual latent-variable way, by maximizing a lower bound on the likelihood. It is the same bound as in the VAE (see [Auto-Encoding Variational Bayes](/vae/)), with the fixed forward chain acting as the encoder.

**Math.** The reverse process, or generative model, is a Markov chain with learned Gaussian transitions that starts at $p(x_T) = \mathcal{N}(x_T; 0, I)$ (Section 2, Equation 1):

$$
p_\theta(x_{0:T}) := p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1} | x_t), \qquad p_\theta(x_{t-1} | x_t) := \mathcal{N}\!\left(x_{t-1};\, \mu_\theta(x_t, t),\, \Sigma_\theta(x_t, t)\right)
$$

Training minimizes the usual variational bound on the negative log likelihood (Equation 3). The paper then rewrites it to reduce variance (Section 2, Equation 5; the derivation is in Appendix A):

$$
\mathbb{E}_q\Big[\underbrace{D_{KL}\big(q(x_T|x_0) \,\|\, p(x_T)\big)}_{L_T} + \sum_{t>1} \underbrace{D_{KL}\big(q(x_{t-1}|x_t, x_0) \,\|\, p_\theta(x_{t-1}|x_t)\big)}_{L_{t-1}} \underbrace{- \log p_\theta(x_0|x_1)}_{L_0}\Big]
$$

The rewrite works because the forward process posterior is tractable once you also condition on $x_0$ (Section 2, Equations 6 and 7):

$$
q(x_{t-1} | x_t, x_0) = \mathcal{N}\!\left(x_{t-1};\, \tilde\mu_t(x_t, x_0),\, \tilde\beta_t I\right)
$$

$$
\tilde\mu_t(x_t, x_0) := \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,x_0 + \frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t, \qquad \tilde\beta_t := \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t
$$

Every KL term in Equation 5 now compares two Gaussians, so it has a closed form and needs no Monte Carlo estimate (Section 2). The term $L_T$ has no learnable parameters, because the $\beta_t$ are fixed, so it is a constant (Section 3.1). The paper also fixes $\Sigma_\theta(x_t, t) = \sigma_t^2 I$ to an untrained constant. It reports that $\sigma_t^2 = \beta_t$ and $\sigma_t^2 = \tilde\beta_t$ gave similar results (Section 3.2). The first is optimal for $x_0 \sim \mathcal{N}(0, I)$, and the second for $x_0$ set to one point.

**Tiny example, computed by `scripts/posterior_example.py`.** Use the four-step schedule and take $t = 3$, $x_0 = 2$, $x_3 = 0.5$. Here $\bar\alpha_3 = 0.504$ and $\bar\alpha_2 = 0.72$.

| Quantity | Value |
|---|---|
| Coefficient on $x_0$ in Equation 7 | 0.5132 |
| Coefficient on $x_t$ in Equation 7 | 0.4723 |
| $\tilde\mu_3(x_3, x_0)$ | $0.5132 \cdot 2 + 0.4723 \cdot 0.5 = 1.2626$ |
| $\tilde\beta_3$ | 0.1694 |
| $\beta_3$, for comparison | 0.3000 |

As a check, the script multiplies the prior $q(x_2 | x_0)$ by the likelihood $q(x_3 | x_2)$ using Bayes' rule on precisions. The result is a Gaussian with mean 1.2626 and variance 0.1694, the same as Equations 6 and 7. The posterior variance $\tilde\beta_t$ is always smaller than $\beta_t$, because $x_0$ is extra information. At $t = 1$ you get $\bar\alpha_0 = 1$ and so $\tilde\beta_1 = 0$.

Exercise 2 asks you to implement $\tilde\mu_t$ and $\tilde\beta_t$ and to check them against Bayes' rule.

### Predict the noise

**Intuition.** The network must output a mean $\mu_\theta(x_t, t)$ that matches $\tilde\mu_t$. That formula contains the clean image $x_0$, which the network doesn't have. But $x_t$ is just $x_0$ plus noise, so knowing $x_t$ and the noise $\epsilon$ tells you $x_0$. The paper therefore asks the network to guess $\epsilon$ and builds the mean from that guess.

**Math.** With $p_\theta(x_{t-1}|x_t) = \mathcal{N}(\mu_\theta, \sigma_t^2 I)$, the term $L_{t-1}$ is a squared distance between means (Section 3.2, Equation 8):

$$
L_{t-1} = \mathbb{E}_q\!\left[\frac{1}{2\sigma_t^2}\,\big\|\tilde\mu_t(x_t, x_0) - \mu_\theta(x_t, t)\big\|^2\right] + C
$$

The constant $C$ doesn't depend on $\theta$. The paper substitutes $x_t(x_0, \epsilon)$ from Equation 4 and uses Equation 7. The target mean simplifies (Equation 10) to $\frac{1}{\sqrt{\alpha_t}}\big(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon\big)$. Since $x_t$ is an input to the network, the paper chooses the parameterization (Equation 11):

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t, t)\right)
$$

Here $\epsilon_\theta$ is a function approximator meant to predict $\epsilon$ from $x_t$. The paper notes that predicting $x_0$ is possible too, but that it led to worse sample quality in early experiments (Section 3.2).

**Tiny example, computed by `scripts/posterior_example.py`.** Reuse $t = 3$, $x_0 = 2$, $x_3 = 0.5$. The noise that links them is $\epsilon = (x_3 - \sqrt{\bar\alpha_3}\,x_0) / \sqrt{1-\bar\alpha_3} = -1.3061$. Plug that $\epsilon$ into Equation 11:

$$
\frac{1}{\sqrt{0.7}}\left(0.5 - \frac{0.3}{\sqrt{0.496}} \cdot (-1.3061)\right) = 1.2626
$$

This equals the $\tilde\mu_3$ from the previous example. An exact $\epsilon$ prediction reproduces the exact posterior mean, which is the content of Equation 10. Exercise 2 tests this equality at every timestep.

### The simplified loss and the two algorithms

**Intuition.** With the $\epsilon$ parameterization, each KL term becomes a weighted squared error between true and predicted noise. The weights shrink for the noisy steps and grow for the nearly clean ones. The paper throws the weights away and trains on the unweighted squared error. This is not the exact bound any more, but it samples better.

**Math.** Substituting Equation 11 into Equation 10 gives (Section 3.2, Equation 12):

$$
\mathbb{E}_{x_0, \epsilon}\!\left[\frac{\beta_t^2}{2\sigma_t^2\,\alpha_t\,(1-\bar\alpha_t)}\,\big\|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\, t)\big\|^2\right]
$$

Section 3.4 drops that weight and defines the simplified objective (Equation 14):

$$
L_{\text{simple}}(\theta) := \mathbb{E}_{t, x_0, \epsilon}\!\left[\big\|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\, t)\big\|^2\right]
$$

where $t$ is uniform between 1 and $T$. The case $t = 1$ stands in for $L_0$: the paper approximates the discrete decoder of Equation 13 by the Gaussian density times the bin width (Section 3.4). Algorithm 1 trains with one gradient step per sample of $(x_0, t, \epsilon)$. Algorithm 2 samples: draw $x_T \sim \mathcal{N}(0, I)$, then for $t = T, \dots, 1$ compute

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t, t)\right) + \sigma_t z, \qquad z \sim \mathcal{N}(0, I) \text{ if } t > 1, \text{ else } z = 0
$$

Note that $1 - \alpha_t = \beta_t$, so line 4 of Algorithm 2 is Equation 11 plus noise. The paper says this loop resembles Langevin dynamics with $\epsilon_\theta$ as a learned gradient of the data density (Section 3.2).

**Tiny example, computed by `scripts/loss_weights.py` and `scripts/oracle_sampler.py`.** First, the weights. With $\sigma_t^2 = \beta_t$ and the linear schedule, the weight in Equation 12 is

| $t$ | 1 | 10 | 100 | 1000 |
|---|---|---|---|---|
| $w_t$ | 0.5001 | 0.0737 | 0.0101 | 0.0102 |

The bound weights step 1 about 49 times more than step 1000. $L_{\text{simple}}$ gives every step weight 1, so relative to the bound it scales term $t$ by $1/w_t$. That is a factor of 2.00 at $t = 1$ and 98.00 at $t = 1000$. This is the sense in which the simplified objective down-weights small $t$ (Section 3.4): it focuses the network on the harder denoising tasks at large $t$. Exercise 4 checks Equation 12 against a directly computed Gaussian KL.

Second, the sampler. Suppose the data are one-dimensional with $x_0 \sim \mathcal{N}(2, 0.5^2)$. Then $x_t$ is Gaussian and the best noise predictor has a closed form: $\epsilon_\theta(x, t) = \sqrt{1-\bar\alpha_t}\,(x - \sqrt{\bar\alpha_t}\cdot 2) / (\bar\alpha_t \cdot 0.25 + 1 - \bar\alpha_t)$. No network is needed. Running Algorithm 2 with this predictor and the paper's schedule on 20,000 samples gives:

| Reverse noise | Sample mean | Sample std |
|---|---|---|
| $\sigma_t^2 = \beta_t$ | 2.004 | 0.499 |
| $\sigma_t^2 = \tilde\beta_t$ | 2.004 | 0.494 |
| none ($z = 0$ everywhere) | 2.000 | 0.002 |

The target is mean 2 and standard deviation 0.5. Both variance choices recover it. Without the noise term the chain collapses to a single point, because every sample follows the same deterministic path to the mean. The noise term is what carries the spread of the data. Exercise 3 asks you to implement $L_{\text{simple}}$ and the sampler and to run this check.

<!-- exercises -->

## Pitfalls

- **Off-by-one in $\bar\alpha$.** The paper's $t$ starts at 1, and arrays start at 0. Reading `alpha_bars[t]` for step $t$ is wrong, and at $t = T$ it raises an error. Reading `alpha_bars[t - 2]` for $\bar\alpha_{t-1}$ at $t = 1$ silently wraps to $\bar\alpha_T$. Use $\bar\alpha_0 = 1$ for the first step.
- **Using $\alpha_t$ where $\bar\alpha_t$ belongs.** Equation 4 uses the cumulative product, not the single-step value. With the paper's schedule, $\alpha_{1000} = 0.98$ but $\bar\alpha_{1000} = 0.00004$.
- **Forgetting a square root.** Equation 4 scales the signal by $\sqrt{\bar\alpha_t}$ and the noise by $\sqrt{1-\bar\alpha_t}$. The variances add up to 1, which is what keeps unit-variance data from drifting. A missing square root still runs, and produces the wrong noise level.
- **Adding noise at the last step.** Algorithm 2 sets $z = 0$ at $t = 1$. The paper says the final output is $\mu_\theta(x_1, 1)$ and displays it noiselessly (Section 3.3). Adding $\sigma_1 z$ puts visible grain on every sample.
- **Feeding the network the clean input.** The network sees $x_t$ and $t$, never $x_0$. Without the timestep it can't tell how much noise to expect. The paper passes $t$ through the Transformer sinusoidal embedding and shares the parameters across all timesteps (Section 4).
- **Treating the weight in Equation 12 as optional.** The simplified loss is a different objective, not the same one rewritten. Table 2 shows the effect. Predicting $\epsilon$ with the true bound and fixed variance gives FID 13.51, and with $L_{\text{simple}}$ it gives 3.17. Predicting $\tilde\mu$ works only with the true bound (FID 13.22) and was unstable with unweighted mean squared error.
- **Learning the variance.** The paper reports that learning a diagonal $\Sigma_\theta$ led to unstable training and poorer sample quality than fixed variances (Section 4.2).
- **Mean versus sum over dimensions.** Equation 14 is a squared norm, which sums over the data dimensions. Many codebases average over every entry instead, which divides the loss by the dimension $D$. The minimizer is the same, but learning rates and logged values differ by a factor of $D$.
- **Expecting a good likelihood.** The model's lossless codelength is at most 3.75 bits per dimension on CIFAR10 (Table 1), which is not competitive with other likelihood-based models. The paper trains for sample quality. Training on the true bound gives better codelengths (Section 4.1).
- **Skipping the data scaling.** The network is designed for data in $[-1, 1]$ (Section 3.3). Feeding raw pixel values in $[0, 255]$ breaks the assumption that $x_T$ is close to $\mathcal{N}(0, I)$ at the same noise level.

## Further reading

- [The official implementation](https://github.com/hojonathanho/diffusion), linked from the paper's abstract, is the code that produced the results.
- [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585), by Sohl-Dickstein et al., is the 2015 paper that introduced diffusion probabilistic models (reference [53] in the paper).
- [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600), by Song and Ermon, is the denoising score matching work that Section 3.2 connects to (reference [55] in the paper).
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502), by Song et al., keeps the same trained network and samples in far fewer steps. The course lesson "Sampling faster: DDIM and the probability-flow ODE" covers it.
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](/unet/) is the backbone design that the network in Section 4 follows.
