---
id: "cfg"
title: "Classifier-Free Diffusion Guidance"
authors: "Ho et al."
year: 2022
venue: "NeurIPS 2021 Workshop"
arxiv: "2207.12598"
url: "https://arxiv.org/abs/2207.12598"
track: generative
tier: extended
est_hours: 3
status: full
summary: "Classifier guidance sharpens a conditional diffusion model's samples by adding the gradient of a separately trained classifier, which costs an extra noisy-data classifier and a pipeline change. This paper trains one network on both the conditional and the unconditional task by replacing the condition with a null token some of the time, then samples with the extrapolated prediction (1 + w) times the conditional minus w times the unconditional. Raising w trades diversity for fidelity, and on ImageNet it reaches the FID and Inception score trade-off of classifier guidance with no classifier at all."
related_course: "ml: diffusion"
prereqs:
  - id: ddpm
    reason: "Guidance edits the noise prediction of a trained diffusion model, so you need the forward noising process, the epsilon-prediction objective, and the ancestral sampler from DDPM."
background: [conditional-generation]
course_links:
  - title: "Classifier-free guidance"
    url: "https://speech.ifkash.dev/lessons/064-classifier-free-guidance"
self_check:
  - q: 'In DDPM notation, $\bar\alpha_3 = 0.504$. This paper writes the same noise level with $\alpha_\lambda^2 = \bar\alpha_t$ and $\sigma_\lambda^2 = 1 - \bar\alpha_t$. What is the log-SNR $\lambda = \log(\alpha_\lambda^2 / \sigma_\lambda^2)$?'
    a: 'The signal variance is $0.504$ and the noise variance is $1 - 0.504 = 0.496$, so $\lambda = \ln(0.504 / 0.496) = 0.016$. The noise level is almost exactly the point where signal and noise are equal. The two parameterizations describe the same forward process; $\lambda$ increases as noise decreases.'
    see: ddpm
  - q: 'A $\epsilon$-prediction model sees $z = 0.5$ at $\bar\alpha_t = 0.504$ and predicts $\epsilon = -1.3061$. What clean value $x$ does that prediction imply, using $x = (z - \sigma\epsilon) / \alpha$?'
    a: 'With $\alpha = \sqrt{0.504} = 0.7099$ and $\sigma = \sqrt{0.496} = 0.7043$, $x = (0.5 + 0.7043 \cdot 1.3061) / 0.7099 = 2.0$. This is the same inversion of $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ that DDPM uses, and Algorithm 2 line 4 applies it to the guided $\tilde\epsilon$.'
    see: ddpm
  - q: 'A conditional model answers a question about a class label $c$. What do the two special settings of the guided prediction, $w = 0$ and $w = -1$, return in terms of $\epsilon(z, c)$ and $\epsilon(z)$?'
    a: 'With $\tilde\epsilon = (1+w)\,\epsilon(z, c) - w\,\epsilon(z)$, $w = 0$ gives $\epsilon(z, c)$ (the ordinary conditional model) and $w = -1$ gives $\epsilon(z)$ (the unconditional model, which ignores the label). Positive $w$ extrapolates beyond the conditional prediction, away from the unconditional one.'
    see: conditional-generation
---

## The problem

A class-conditional diffusion model draws samples that cover the whole class, including its odd, blurry, and ambiguous members. GANs and flows have a knob for this: truncating the input noise or lowering the sampling temperature gives sharper, more typical samples at the price of diversity (Section 3). The paper says the obvious versions of that knob do not work for diffusion models. Scaling the model's scores or shrinking the variance of the noise added during the reverse process produces blurry, low-quality samples (Section 3, citing Dhariwal and Nichol).

Classifier guidance was the first fix (Section 3.1). It trains a separate classifier $p_\theta(c | z_\lambda)$ on noisy data and modifies the diffusion score with the classifier's input gradient, which up-weights samples that the classifier assigns confidently to the right class. That works, but the paper lists the costs (Section 1):

- **An extra model.** The classifier must be trained on noisy $z_\lambda$, so you can't plug in an off-the-shelf classifier.
- **An adversarial flavor.** Sampling steps along classifier gradients resembles an adversarial attack on that classifier. It raises the question of whether FID and Inception score improve only because the sampler fools the classifiers inside those metrics.

The paper asks whether guidance can be done without a classifier. The answer is a one-line change to training and a one-line change to sampling (Section 5). On class-conditional ImageNet, sweeping the guidance strength traces a trade-off between FID and Inception score like the one from classifier guidance or GAN truncation (Section 4.1). At 128x128, FID reaches 2.43 at $w = 0.3$ and the Inception score reaches 421.03 at $w = 4$ (Table 2, $T = 256$). The paper says that the FID at $w = 0.3$ beats the classifier-guided ADM-G, and that at $w = 4$ the model beats BigGAN-deep at both FID and IS (Section 4.1).

## Key ideas

This page uses the paper's notation, which differs from DDPM's. The data is $x \sim p(x)$, with optional conditioning $c$. The forward process works in terms of the log signal-to-noise ratio $\lambda$ and defines $z_\lambda = \alpha_\lambda x + \sigma_\lambda \epsilon$. The network is $\epsilon_\theta(z_\lambda, c)$. The guidance strength is $w$, and the probability of dropping the condition in training is $p_{\text{uncond}}$. The null token is written $\varnothing$. Every number comes from a script in `scripts/`. Run any script from the paper folder to reproduce the output.

The paper trains in continuous time (Section 2). Where DDPM indexes steps $t = 1, \dots, T$ and uses $\bar\alpha_t$, the paper indexes noise levels by $\lambda$. The two descriptions are the same forward process: $\alpha_\lambda^2$ plays the role of $\bar\alpha_t$ and $\sigma_\lambda^2$ plays the role of $1 - \bar\alpha_t$, so $\lambda = \log\big(\bar\alpha_t / (1 - \bar\alpha_t)\big)$. Exercise 1 implements this mapping.

### The log-SNR parameterization

**Intuition.** DDPM names a noise level by a step number, and that number means different things for different schedules. The log-SNR names it by how much of the variance is signal. A large $\lambda$ means a nearly clean input, and a very negative $\lambda$ means nearly pure noise. Guidance then has one description at every noise level.

**Math.** The forward process is the variance-preserving process (Section 2, Equations 1 and 2):

$$
q(z_\lambda | x) = \mathcal{N}(\alpha_\lambda x,\, \sigma_\lambda^2 I), \qquad \alpha_\lambda^2 = \frac{1}{1 + e^{-\lambda}}, \qquad \sigma_\lambda^2 = 1 - \alpha_\lambda^2
$$

Then $\lambda = \log(\alpha_\lambda^2 / \sigma_\lambda^2)$, and the forward process runs in the direction of decreasing $\lambda$. The paper parameterizes the denoiser by $\epsilon$-prediction, $x_\theta(z_\lambda) = (z_\lambda - \sigma_\lambda \epsilon_\theta(z_\lambda)) / \alpha_\lambda$, and trains on (Section 2, Equation 5):

$$
\mathbb{E}_{\epsilon, \lambda}\big[\|\epsilon_\theta(z_\lambda) - \epsilon\|_2^2\big], \qquad z_\lambda = \alpha_\lambda x + \sigma_\lambda \epsilon, \quad \lambda \sim p(\lambda)
$$

This is the same objective as DDPM's $L_{\text{simple}}$ with a different way of choosing the noise level. The paper's $p(\lambda)$ comes from the cosine schedule: draw $u$ uniformly on $[0, 1]$ and set $\lambda = -2\log\tan(au + b)$, with $b = \arctan(e^{-\lambda_{\max}/2})$ and $a = \arctan(e^{-\lambda_{\min}/2}) - b$ (Section 2). This is a hyperbolic secant distribution truncated to $[\lambda_{\min}, \lambda_{\max}]$. All the paper's models use $\lambda_{\min} = -20$ and $\lambda_{\max} = 20$ (Section 4.1). Note that $u = 0$ gives $\lambda_{\max}$, so $\lambda$ falls as $u$ grows. For sampling, the paper uses $\lambda$ values from uniformly spaced $u$, and Algorithm 2 needs them in increasing order.

**Tiny example, computed by `scripts/guidance_example.py` (Part A).**

| $\lambda$ | $\alpha_\lambda^2$ | $\sigma_\lambda^2$ |
|---|---|---|
| $-20$ | 0.000000 | 1.000000 |
| $-4$ | 0.017986 | 0.982014 |
| $0$ | 0.500000 | 0.500000 |
| $4$ | 0.982014 | 0.017986 |
| $20$ | 1.000000 | 0.000000 |

At $\lambda = 0$ the signal and noise variances are both $0.5$. The DDPM schedule from the previous page maps onto this scale: at $t = 1, 100, 500, 1000$ the script gives $\lambda = 9.210, 2.165, -2.462, -10.118$. Nine uniformly spaced $u$ values give $\lambda = -20, -3.229, -1.763, -0.806, 0, 0.806, 1.763, 3.229, 20$ when sorted in increasing order. The mass sits mostly near $\lambda = 0$, with two long tails out to $\pm 20$.

### Train both models at once with condition dropout

**Intuition.** Guidance needs two predictions at the same noisy input: one that knows the class and one that doesn't. You could train two networks. The paper trains one, and each time it shows the network a training example it flips a biased coin. With small probability the label is replaced by a null token $\varnothing$, so the network learns to denoise with no label. Otherwise it keeps the label. At sampling time, giving the network $\varnothing$ gets the unconditional prediction.

**Math.** Algorithm 1 is the DDPM training loop with one extra line. Given $(x, c) \sim p(x, c)$:

1. $c \leftarrow \varnothing$ with probability $p_{\text{uncond}}$ (line 3).
2. $\lambda \sim p(\lambda)$ and $\epsilon \sim \mathcal{N}(0, I)$ (lines 4 and 5).
3. $z_\lambda = \alpha_\lambda x + \sigma_\lambda \epsilon$ (line 6).
4. Take a gradient step on $\nabla_\theta \|\epsilon_\theta(z_\lambda, c) - \epsilon\|^2$ (line 7).

So $\epsilon_\theta(z_\lambda) := \epsilon_\theta(z_\lambda, c = \varnothing)$ (Section 3.2). A single network represents both models, and the total parameter count does not grow. The paper notes that separate models would also work, and prefers joint training because it is simple and does not complicate the pipeline (Section 3.2).

The hyperparameter $p_{\text{uncond}}$ sets how much of the batch trains the unconditional task. The paper compares 0.1, 0.2, and 0.5 on 64x64 ImageNet. It finds that 0.5 is consistently worse across the FID and Inception score frontier, and that 0.1 and 0.2 perform about equally well (Section 4.2). The conclusion in the paper is that only a small share of the model's capacity needs to go to the unconditional task. Exercise 1 asks you to implement the dropout and the corruption step.

### The guided score

**Intuition.** Look at what each prediction says. The conditional prediction $\epsilon(z, c)$ says which noise would explain $z$ given that it's a member of class $c$. The unconditional prediction $\epsilon(z)$ says the same for any class. Their difference isolates what the label adds. Guidance starts at the conditional prediction and moves further along that difference by $w$ times its length.

**Math.** Sampling uses the linear combination (Section 3.2, Equation 6):

$$
\tilde\epsilon_\theta(z_\lambda, c) = (1 + w)\,\epsilon_\theta(z_\lambda, c) - w\,\epsilon_\theta(z_\lambda)
$$

The motivation is classifier guidance. Classifier guidance modifies the conditional score to $\tilde\epsilon_\theta(z_\lambda, c) = \epsilon_\theta(z_\lambda, c) - w\,\sigma_\lambda \nabla_{z_\lambda}\log p_\theta(c | z_\lambda)$, which approximately samples from $\tilde p_\theta(z_\lambda | c) \propto p_\theta(z_\lambda | c)\,p_\theta(c | z_\lambda)^w$ (Section 3.1). The effect is to up-weight data that the classifier labels confidently and to lower diversity.

Bayes' rule gives an implicit classifier, $p^i(c | z_\lambda) \propto p(z_\lambda | c) / p(z_\lambda)$. If you had the exact scores $\epsilon^*(z_\lambda, c)$ and $\epsilon^*(z_\lambda)$, its gradient would be (Section 3.2)

$$
\nabla_{z_\lambda}\log p^i(c | z_\lambda) = -\frac{1}{\sigma_\lambda}\big[\epsilon^*(z_\lambda, c) - \epsilon^*(z_\lambda)\big]
$$

Substituting this into classifier guidance gives $(1 + w)\,\epsilon^*(z_\lambda, c) - w\,\epsilon^*(z_\lambda)$, which has the form of Equation 6. The paper stresses that the resemblance is only formal. A network is an unconstrained function, so its score estimates are not conservative vector fields, and $\epsilon_\theta(z_\lambda, c) - \epsilon_\theta(z_\lambda)$ is in general not the gradient of any classifier (Section 3.2 and Section 5). That is why the sampler can't be read as an adversarial attack on a classifier. It also means that Equation 6 comes with no guarantee: Bayes-rule classifiers from a misspecified generative model can be inconsistent, so the paper checks empirically (Section 3.2).

The paper uses $(1 + w)$, so that $w = 0$ is the ordinary conditional model. Many libraries use a guidance scale $s = 1 + w$ instead. See the pitfalls.

**Tiny example, computed by `scripts/guidance_example.py` (Parts B and C).** Take two classes with $x | c \sim \mathcal{N}(\mu_c, 1)$, $\mu = (-2, +2)$, and equal class probabilities. The noise predictions are exact, in closed form. At $\lambda = 0$, $z = 1$, and class $c = 1$:

| Quantity | Value |
|---|---|
| $\epsilon(z, c)$ | $-0.2929$ |
| $\epsilon(z)$ | $-0.1813$ |
| $\epsilon(z, c) - \epsilon(z)$ | $-0.1116$ |
| $\tilde\epsilon$ at $w = 0$ | $-0.2929$ |
| $\tilde\epsilon$ at $w = 1$ | $-0.4045$ |
| $\tilde\epsilon$ at $w = 2$ | $-0.5161$ |
| $\tilde\epsilon$ at $w = 4$ | $-0.7394$ |

Each unit of $w$ adds another copy of the difference, $-0.1116$. A sampler that read $w$ as the scale $s$ would return $w\,\epsilon_c + (1-w)\,\epsilon_u$ and, at $w = 2$, give $-0.4045$ instead.

For Part C, the script checks the implicit classifier identity. Here $\sigma_\lambda = 0.7071$, so $-(1/\sigma_\lambda)\,[\epsilon(z, c) - \epsilon(z)] = 0.1578$. A finite-difference gradient of the true log posterior $\log p(c = 1 | z)$ at the same point also gives $0.1578$, where $p(c = 1 | z) = 0.9442$. Exercise 2 implements Equation 6, the implicit classifier gradient, and the classifier-guided score, and checks that the last two combine back into the first.

### Sampling with Algorithm 2

**Intuition.** Guided sampling is ordinary ancestral sampling with one change. At every step, run the network twice, once with the label and once with the null token, and combine the two noise predictions before taking the step.

**Math.** Algorithm 2 takes a strength $w$, a label $c$, and an increasing sequence $\lambda_1 = \lambda_{\min} < \dots < \lambda_T = \lambda_{\max}$. It draws $z_1 \sim \mathcal{N}(0, I)$ and for $t = 1, \dots, T$ does the following:

1. $\tilde\epsilon_t = (1 + w)\,\epsilon_\theta(z_t, c) - w\,\epsilon_\theta(z_t)$ (line 3).
2. $\tilde x_t = (z_t - \sigma_{\lambda_t}\tilde\epsilon_t) / \alpha_{\lambda_t}$ (line 4).
3. If $t < T$, sample $z_{t+1}$ from a Gaussian with mean $\tilde\mu_{\lambda_{t+1}|\lambda_t}(z_t, \tilde x_t)$ and the variance below. Otherwise return $\tilde x_T$ (line 5).

The mean comes from the posterior of the forward process, with $\tilde x_t$ standing in for the data (Section 2, Equation 3):

$$
\tilde\mu_{\lambda'|\lambda}(z_\lambda, x) = e^{\lambda - \lambda'}\frac{\alpha_{\lambda'}}{\alpha_\lambda}\,z_\lambda + \big(1 - e^{\lambda - \lambda'}\big)\,\alpha_{\lambda'}\,x, \qquad \tilde\sigma^2_{\lambda'|\lambda} = \big(1 - e^{\lambda - \lambda'}\big)\,\sigma_{\lambda'}^2
$$

The variance is a log-space interpolation between $\tilde\sigma^2_{\lambda'|\lambda}$ and $\sigma^2_{\lambda|\lambda'} = (1 - e^{\lambda - \lambda'})\,\sigma_\lambda^2$ with a constant coefficient $v$ (Section 2, Equation 4), as suggested by Nichol and Dhariwal. The paper uses $v = 0.3$ on 64x64 ImageNet and $v = 0.2$ on 128x128 (Section 4.1). The paper notes that this is DDPM's ancestral sampler. It says the step in Algorithm 2 "could be replaced by another sampler, e.g. DDIM", so the guided $\tilde\epsilon$ can go into any sampler. Note that the paper's Equation 3 gives the posterior from $\lambda$ to $\lambda' > \lambda$, which is the direction of generation.

**Tiny example, computed by `scripts/toy_sweep.py` and `scripts/guidance_example.py` (Part D).** The toy data has three classes at the vertices of a triangle with radius 2 and spread 1 in 2-D, similar to the setup of the paper's Figure 2. Everything has a closed form, so no network is needed. The script runs Algorithm 2 with $v = 0$, $T = 256$, and 20,000 samples for class 0, and measures the samples along the direction pointing away from the other two classes:

| $w$ | Mean along the axis | Std along the axis | Std across | Fraction nearest class 0 |
|---|---|---|---|---|
| 0.0 | 1.989 | 0.994 | 0.990 | 0.9266 |
| 0.5 | 2.328 | 0.882 | 0.922 | 0.9878 |
| 1.0 | 2.506 | 0.824 | 0.913 | 0.9981 |
| 2.0 | 2.713 | 0.812 | 0.892 | 0.9999 |
| 4.0 | 2.974 | 0.799 | 0.896 | 1.0000 |

The unguided samples reproduce the class, with mean 2 (the distance of the class mean from the origin) and standard deviation 1. As $w$ grows, the samples move away from the other classes and the spread shrinks, so that nearly every sample sits closer to its own class than to the others. That is the diversity-for-fidelity trade in miniature. It also shows what "concentrates on the mode" means precisely: the mass concentrates in a smaller region, but not on the class mean, because it is pushed outward. This matches the paper's description of Figure 2, where mass moves away from other classes toward directions of high confidence (Section 3.1). Exercise 3 asks you to implement Algorithm 2 and to check this behavior.

Is the sampler producing the tilted density $\tilde p(x | c) \propto p(x | c)^{1+w}\,p(x)^{-w}$? Part D computes that density on a grid for a 1-D version, with $\mu = (-2, +2)$ and unit variance, and compares it to the sampler:

| $w$ | Grid mean | Grid std | Sampler mean | Sampler std |
|---|---|---|---|---|
| 0 | 2.0000 | 1.0000 | 2.0019 | 0.9915 |
| 1 | 2.0710 | 0.9366 | 2.3364 | 0.8753 |
| 2 | 2.1023 | 0.9135 | 2.4793 | 0.8525 |
| 4 | 2.1394 | 0.8889 | 2.6481 | 0.8476 |

The grid density is the idealized target at clean data, and the guided sampler moves farther and concentrates more than it does. The paper calls the effect approximate (Section 3.1), and this is the reason: guidance acts at every noise level, and a guided score at a noisy level isn't the score of the tilted density convolved with noise. The grid is still the right picture of the direction of the effect. Both columns show a mean that moves away from the other class and a standard deviation that shrinks. Exercise 4 asks you to compute the grid density.

### What the knob does and what it costs

The paper sweeps $w \in \{0, 0.1, \dots, 4\}$ with 50,000 samples per value (Section 4.1). FID is best at small $w$, and Inception score is best at large $w$. Between the two extremes there is a clear trade-off: as $w$ grows, FID gets worse and IS gets better. The values are:

| Model | $w$ | FID | IS |
|---|---|---|---|
| 64x64, $p_{\text{uncond}} = 0.1$ (Table 1) | 0.0 | 1.80 | 53.71 |
| | 0.1 | 1.55 | 66.11 |
| | 4.0 | 26.22 | 260.2 |
| 128x128, $T = 256$ (Table 2) | 0.0 | 7.27 | 82.45 |
| | 0.3 | 2.43 | 158.47 |
| | 4.0 | 21.53 | 421.03 |

The paper reads these numbers as a proof of concept, not as a state-of-the-art attempt. It reuses the architecture and hyperparameters of the classifier-guided ADM models, which were tuned for classifier guidance (Section 4). The cost is sampling speed: each step evaluates the network twice, so a fair comparison to ADM-G at the same cost is the $T = 128$ setting, where it does not beat ADM-G on FID (Section 4.3). The paper suggests injecting the condition late in the network to share work between the two passes, and leaves it for future work (Section 5).

<!-- exercises -->

## Pitfalls

- **Two conventions for the strength.** The paper's Equation 6 is $(1 + w)\,\epsilon_c - w\,\epsilon_u$, where $w = 0$ is no guidance. Many code bases and papers write $\epsilon_u + s\,(\epsilon_c - \epsilon_u)$, where $s = 1$ is no guidance. They are the same family with $s = 1 + w$. Copying a published value of $s$ into $w$, or the reverse, shifts the effect by one full unit. At $w = 2$ the page example above gives $-0.5161$, and reading it as $s = 2$ gives $-0.4045$.
- **Dropping the condition for the whole batch.** Algorithm 1 flips the coin for each example (line 3). One coin per batch makes every gradient step purely conditional or purely unconditional, which adds variance to the training signal.
- **A null token that the network never saw.** The unconditional prediction at sampling time is only meaningful if training used the same token $\varnothing$ on the same input channel. Passing zeros or a random label instead gives an unrelated prediction, and $\tilde\epsilon$ turns into noise.
- **Setting $p_{\text{uncond}}$ too high.** The paper finds 0.5 worse than 0.1 or 0.2 across the whole frontier (Section 4.2). A small share of the training budget on the unconditional task is enough.
- **Treating Equation 6 as a classifier gradient.** It isn't one. With a trained network, the difference $\epsilon_\theta(z, c) - \epsilon_\theta(z)$ has no scalar potential in general (Section 3.2). Do not expect the theory of classifier guidance, such as exact sampling from $p(z|c)\,p(c|z)^w$, to carry over. The toy table above shows that even with exact scores the sampler does not match the tilted density.
- **Reading the sweep as monotone quality.** Only one of the two metrics improves. FID is best at small $w$ (0.1 to 0.3 in the paper) and gets worse after that, while IS keeps rising up to $w = 4$ (Section 4.1). Large $w$ also gives saturated colors, as the paper notes for its 128x128 samples (Figure 3).
- **Forgetting the doubled cost.** Every sampling step needs the conditional and the unconditional pass (Section 4.3 and Section 5). Batch them into one forward pass if memory allows. Skipping the second pass at $w = 0$ is the only free saving.
- **Giving up diversity without checking.** Guidance reduces sample variety by design. The paper warns that this may matter in deployed models where parts of the data are underrepresented (Section 5).
- **Walking $\lambda$ in the wrong order.** With the paper's formula, $u = 0$ gives $\lambda_{\max}$. Algorithm 2 needs $\lambda_1 = \lambda_{\min}$ first. If you sweep $u$ upward, you run the sampler from clean to noisy.
- **Adding noise at the last step.** Algorithm 2 returns $\tilde x_T$ at $t = T$ without sampling $z_{T+1}$. The deterministic final estimate is the sample.

## Further reading

- [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233), by Dhariwal and Nichol, introduced classifier guidance and the ADM-G models that the paper compares against.
- [Variational Diffusion Models](https://arxiv.org/abs/2107.00630), by Kingma et al., is the continuous-time view of the log-SNR that Section 2 builds on.
- [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672), by Nichol and Dhariwal, proposed the log-space variance interpolation that Equation 4 uses and the cosine noise schedule behind $p(\lambda)$.
- [The workshop version of the paper](https://openreview.net/pdf?id=qw8AKxfYbI) is the short paper that appeared at the NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications.
- [Denoising Diffusion Probabilistic Models](/ddpm/) is the prerequisite on this site. It derives the training loss and ancestral sampler that guidance modifies.
