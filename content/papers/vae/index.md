---
id: "vae"
title: "Auto-Encoding Variational Bayes"
authors: "Kingma et al."
year: 2013
venue: "ICLR 2014"
arxiv: "1312.6114"
url: "https://arxiv.org/abs/1312.6114"
track: generative
tier: core
est_hours: 4
status: full
summary: "Variational inference for a latent-variable model normally needs integrals that have no closed form, and the usual gradient estimator for the approximate posterior has very high variance. The paper writes a latent sample as a deterministic function of the posterior parameters and a noise variable, which makes the evidence lower bound differentiable and lets ordinary stochastic gradient descent train it. It then trains an encoder network that outputs the approximate posterior for any datapoint, so one forward pass replaces a per-datapoint optimization."
related_course: "ml: generative models"
prereqs: []
background: [kl-divergence, gaussian-distributions, elbo, expectation]
course_links:
  - title: "Autoencoders and variational autoencoders"
    url: "https://ml.ifkash.dev/lessons/061-autoencoders-and-vaes"
self_check:
  - q: 'For $q = \mathcal{N}(\mu, \sigma^2)$ and $p = \mathcal{N}(0, 1)$ in one dimension, compute $D_{KL}(q \,\|\, p)$ for $\mu = 1$ and $\sigma = 1$. What does the result say about the two terms of the closed form?'
    a: 'The result is $\frac{1}{2}\mu^2 = 0.5$. With $\sigma = 1$ the variance part $\frac{1}{2}(\sigma^2 - 1 - \log\sigma^2)$ is exactly 0, so only the mean offset from the prior contributes.'
    see: kl-divergence
  - q: 'Equation 1 says $\log p_\theta(x) = D_{KL}(q_\phi(z|x) \,\|\, p_\theta(z|x)) + \mathcal{L}(\theta, \phi; x)$. Why is $\mathcal{L}$ a lower bound, and when is it tight?'
    a: 'The KL divergence is never negative, so $\mathcal{L} \le \log p_\theta(x)$. The bound is tight exactly when $q_\phi(z|x)$ equals the true posterior $p_\theta(z|x)$, because only then is the KL term 0.'
    see: elbo
  - q: 'The score-function identity $\nabla_\phi \mathbb{E}_{q_\phi}[f(z)] = \mathbb{E}_{q_\phi}[f(z)\nabla_\phi \log q_\phi(z)]$ and the reparameterized estimator both have the same expectation. Which one differentiates $f$, and which has the lower variance in the paper?'
    a: 'The reparameterized estimator differentiates $f(g_\phi(\epsilon, x))$ through the sample, so it needs a differentiable $f$. The score-function estimator only evaluates $f$. Section 2.2 calls the score-function estimator very high variance, and the reparameterized estimator avoids that.'
    see: expectation
---

## The problem

Suppose a datapoint $x$ comes from a hidden code $z$ through a neural network: $z \sim p_\theta(z)$, then $x \sim p_\theta(x|z)$. To train $\theta$ by maximum likelihood you need $\log p_\theta(x)$, which is $\log \int p_\theta(z)\,p_\theta(x|z)\,dz$. With a nonlinear network inside $p_\theta(x|z)$ this integral has no closed form (Section 2.1, problem 1). The posterior $p_\theta(z|x)$ is out of reach for the same reason, so the EM algorithm can't be used. Mean-field variational Bayes needs expectations that are also intractable.

Two workarounds existed before this paper, and each one fails on a large dataset (Section 2.1, problem 2):

- **Per-datapoint sampling**, such as Monte Carlo EM, runs an expensive sampling loop for every datapoint on every update.
- **Variational inference with the score-function gradient** is cheaper. The naive estimator in Section 2.2 is $f(z)\,\nabla_{q_\phi(z)} \log q_\phi(z)$, and the paper calls its variance very high and impractical.

The paper asks for one method that makes parameter updates from small minibatches, that works when the integrals are intractable, and that gives you an encoder $q_\phi(z|x)$ for free. Section 5 compares it with the wake-sleep algorithm and Monte Carlo EM on MNIST and Frey Face. The new method converges faster and reaches a better lower bound in every setting of Figure 2. It uses minibatches of $M = 100$ with $L = 1$ sample per datapoint.

## Key ideas

The sections that follow keep the paper's notation: generative parameters $\theta$, variational parameters $\phi$, latent code $z$ of dimension $J$, auxiliary noise $\epsilon$, posterior mean $\mu$, and posterior standard deviation $\sigma$. The paper writes the lower bound as $\mathcal{L}(\theta, \phi; x^{(i)})$, and this page drops the superscript $(i)$ when one datapoint is in play.

Two notation choices differ from the paper. Code usually takes the log variance $\log\sigma^2$ as the encoder output, because it keeps $\sigma$ positive without a constraint, so the exercises do the same. The worked examples also use a one-dimensional toy model, $p(z) = \mathcal{N}(0, 1)$ and $p(x|z) = \mathcal{N}(x; z, 1)$, which the paper doesn't use. Its posterior is known in closed form, so every claim can be checked against an exact answer. For this model $p(x) = \mathcal{N}(x; 0, 2)$ and $p(z|x) = \mathcal{N}(z; x/2, 1/2)$.

### The evidence lower bound

**Intuition.** You can't compute $\log p_\theta(x)$, but you can compute something that sits underneath it and gets closer as your approximate posterior $q_\phi(z|x)$ improves. Maximizing that quantity also pushes $\log p_\theta(x)$ up. The gap between the two is exactly how far $q_\phi$ is from the true posterior.

**Math.** Equation 1 splits the log marginal likelihood of one datapoint into two parts (Section 2.2):

$$
\log p_\theta(x^{(i)}) = D_{KL}\big(q_\phi(z|x^{(i)}) \,\|\, p_\theta(z|x^{(i)})\big) + \mathcal{L}(\theta, \phi; x^{(i)})
$$

The KL term is non-negative, so Equation 2 follows:

$$
\log p_\theta(x^{(i)}) \ge \mathcal{L}(\theta, \phi; x^{(i)}) = \mathbb{E}_{q_\phi(z|x)}\big[-\log q_\phi(z|x) + \log p_\theta(x, z)\big]
$$

Equation 3 regroups the same quantity into a regularizer and a reconstruction term:

$$
\mathcal{L}(\theta, \phi; x^{(i)}) = -D_{KL}\big(q_\phi(z|x^{(i)}) \,\|\, p_\theta(z)\big) + \mathbb{E}_{q_\phi(z|x^{(i)})}\big[\log p_\theta(x^{(i)}|z)\big]
$$

The first term of Equation 3 pulls $q_\phi$ toward the prior. The second is the expected log likelihood of the datapoint, which is a negative reconstruction error in autoencoder terms (Section 2.3).

**Tiny example, computed by `scripts/elbo_identity.py`.** Take $x = 1$ in the toy model. Then $\log p(x) = -1.5155$. Pick a poor $q = \mathcal{N}(0.2, 0.8^2)$:

| Quantity | Value |
|---|---|
| $D_{KL}(q \,\|\, p(z\|x))$ | 0.1066 |
| $\mathcal{L}$ from Equation 3, both terms in closed form | $-1.6221$ |
| Sum, which is the right side of Equation 1 | $-1.5155$ |

The sum matches $\log p(x)$. If you set $q$ to the true posterior $\mathcal{N}(0.5, 0.5)$, the KL term is 0.0000 and $\mathcal{L} = -1.5155$, so the bound touches $\log p(x)$.

Exercise 3 asks you to implement the estimate of $\mathcal{L}$ and test both facts.

### The SGVB estimator

**Intuition.** The expectations in Equation 3 are integrals over $z$. Replace them with an average over a few samples of $z$. The only question is how to keep the average differentiable in $\phi$, and the next idea answers it.

**Math.** Section 2.3 draws samples as $z^{(i,l)} = g_\phi(\epsilon^{(i,l)}, x^{(i)})$ with $\epsilon^{(l)} \sim p(\epsilon)$ and builds two estimators of $\mathcal{L}$. The generic one uses Equation 2 directly (Equation 6):

$$
\widetilde{\mathcal{L}}^{A}(\theta, \phi; x^{(i)}) = \frac{1}{L}\sum_{l=1}^{L} \log p_\theta(x^{(i)}, z^{(i,l)}) - \log q_\phi(z^{(i,l)}|x^{(i)})
$$

When the KL term can be integrated analytically, the second estimator replaces it with the exact value (Equation 7):

$$
\widetilde{\mathcal{L}}^{B}(\theta, \phi; x^{(i)}) = -D_{KL}\big(q_\phi(z|x^{(i)}) \,\|\, p_\theta(z)\big) + \frac{1}{L}\sum_{l=1}^{L} \log p_\theta(x^{(i)}|z^{(i,l)})
$$

For a dataset of $N$ points and a minibatch of $M$, Equation 8 scales the average: $\mathcal{L}(\theta, \phi; X) \simeq \frac{N}{M}\sum_{i=1}^{M} \widetilde{\mathcal{L}}(\theta, \phi; x^{(i)})$. Algorithm 1 is the loop that follows from it: draw a minibatch, draw noise, take the gradient of the estimator with respect to $\theta$ and $\phi$, and update.

The paper says estimator B typically has less variance than A (Section 2.3). That is a statement about typical cases, and the toy model shows it can reverse. `scripts/elbo_identity.py` draws 100,000 single-sample estimates for $q = \mathcal{N}(0.2, 0.8^2)$. The standard deviation is 0.5204 for A and 0.7857 for B. At the true posterior, A has a standard deviation of exactly 0, because $\log p(x, z) - \log q(z|x)$ equals $\log p(x)$ for every $z$, while B still has 0.5012. Treat the paper's claim as a guideline for a model with an exact KL term, not as a guarantee.

With $L = 100{,}000$ samples both estimators agree with the exact value: $-1.6226$ for A and $-1.6227$ for B, against $-1.6221$.

### The reparameterization trick

**Intuition.** If you sample $z$ from $q_\phi$ and then use $z$ in a loss, the sampling step blocks the gradient, because you can't differentiate through a random draw. Move the randomness into a variable $\epsilon$ that doesn't depend on $\phi$. Then $z$ becomes an ordinary function of $\phi$, and gradients flow through it.

**Math.** Section 2.4 writes $z = g_\phi(\epsilon, x)$ with $\epsilon \sim p(\epsilon)$, so that $q_\phi(z|x)\prod_i dz_i = p(\epsilon)\prod_i d\epsilon_i$. The expectation then moves onto the noise (Equation 5):

$$
\mathbb{E}_{q_\phi(z|x^{(i)})}\big[f(z)\big] = \mathbb{E}_{p(\epsilon)}\big[f(g_\phi(\epsilon, x^{(i)}))\big] \simeq \frac{1}{L}\sum_{l=1}^{L} f\big(g_\phi(\epsilon^{(l)}, x^{(i)})\big)
$$

For a Gaussian, $z = \mu + \sigma\epsilon$ with $\epsilon \sim \mathcal{N}(0, 1)$ is a valid choice. The paper gives three general recipes: an inverse CDF, a location-scale family, and composition of transformations of simple variables. The multivariate model in Section 3 uses $z^{(i,l)} = \mu^{(i)} + \sigma^{(i)} \odot \epsilon^{(l)}$ with $\epsilon^{(l)} \sim \mathcal{N}(0, I)$, where $\odot$ is the element-wise product.

**Tiny example, computed by `scripts/reparam_gradient.py`.** Let $f(z) = z^2$ and $z \sim \mathcal{N}(\mu, \sigma^2)$ with $\mu = \sigma = 1$. Then $\mathbb{E}[f] = \mu^2 + \sigma^2$, so the true gradients are $\partial/\partial\mu = 2\mu = 2$ and $\partial/\partial\sigma = 2\sigma = 2$. The script draws 200,000 samples:

| Estimator | Mean of $\partial/\partial\mu$ | Std of one sample | Mean of $\partial/\partial\sigma$ | Std of one sample |
|---|---|---|---|---|
| Score function, $f(z)\nabla\log q$ | 2.002 | 5.500 | 2.016 | 11.791 |
| Reparameterized, $\nabla f(\mu + \sigma\epsilon)$ | 2.000 | 2.002 | 2.005 | 3.472 |

Both are unbiased. The score-function estimator for $\mu$ has 7.5 times the variance of the reparameterized one. The cost of reparameterization is that $f$ must be differentiable in $z$ and $z$ must be continuous (the paper says so in the abstract and in Section 2.4).

Exercise 2 asks you to implement the sample and its pathwise gradient.

### The Gaussian VAE and the closed-form KL

**Intuition.** Choose a prior and an encoder family for which the KL term in Equation 3 needs no sampling. Gaussian choices give this, and the formula is simple enough to differentiate by hand.

**Math.** Section 3 takes the prior $p_\theta(z) = \mathcal{N}(z; 0, I)$ and the encoder $\log q_\phi(z|x^{(i)}) = \log \mathcal{N}(z; \mu^{(i)}, \sigma^{2(i)} I)$ (Equation 9), where $\mu^{(i)}$ and $\sigma^{(i)}$ are outputs of an MLP. Appendix B integrates the KL term. With $J$ the dimension of $z$:

$$
-D_{KL}\big(q_\phi(z) \,\|\, p_\theta(z)\big) = \frac{1}{2}\sum_{j=1}^{J}\Big(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\Big)
$$

The full estimator for this model is Equation 10:

$$
\mathcal{L}(\theta, \phi; x^{(i)}) \simeq \frac{1}{2}\sum_{j=1}^{J}\Big(1 + \log(\sigma_j^{(i)})^2 - (\mu_j^{(i)})^2 - (\sigma_j^{(i)})^2\Big) + \frac{1}{L}\sum_{l=1}^{L}\log p_\theta(x^{(i)}|z^{(i,l)})
$$

The decoder term $\log p_\theta(x|z)$ is a Bernoulli or Gaussian MLP, depending on the data (Section 3, Appendix C). In the toy model, the decoder is the fixed Gaussian from the notation note.

**Tiny example, computed by `scripts/kl_gaussian.py`.** Take $J = 2$, $\mu = (0.5, -1.0)$, and $\sigma = (0.8, 1.5)$:

| Dimension $j$ | Term $\frac{1}{2}(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2)$ |
|---|---|
| 1 | $-0.1681$ |
| 2 | $-0.7195$ |

The sum is $-0.8877$, so $D_{KL} = 0.8877$. A Monte Carlo average of $\log q(z) - \log p(z)$ over 1,000,000 samples gives 0.8884. With $\sigma_j = 1$ in both dimensions, the variance terms cancel and the KL is $\frac{1}{2}\sum_j \mu_j^2 = 0.6250$.

Exercise 1 asks you to implement the closed form and the Monte Carlo check.

### The amortized inference network

**Intuition.** Classical variational inference fits a separate $q$ for every datapoint, which means an optimization loop per datapoint. The paper instead trains one network that maps $x$ to the parameters of $q_\phi(z|x)$. After training, inference for a new datapoint is one forward pass. The network's weights $\phi$ are shared across all datapoints, which is why the paper calls $q_\phi(z|x)$ a recognition model, or a probabilistic encoder (Section 2.1).

**Math.** Nothing new is needed in the objective. The same $\widetilde{\mathcal{L}}$ is optimized jointly over $\theta$ and $\phi$ (Algorithm 1). The price is an amortization gap: if the encoder family can't represent the true posterior for every $x$ at once, the bound stays below $\log p_\theta(x)$ by $D_{KL}(q_\phi(z|x) \,\|\, p_\theta(z|x))$ (Equation 1). The paper doesn't use the term amortization gap. It's the standard name for this part of Equation 1.

**Tiny example, computed by `scripts/amortized_example.py`.** In the toy model the true posterior mean is $x/2$ and its variance is $1/2$. A linear encoder $\mu(x) = wx$ with $\sigma^2 = 0.5$ can match that for every $x$ with a single parameter $w = 0.5$. The gap $\log p(x) - \mathcal{L}$ for $x = -2, 1, 3$ is:

| Encoder | Gap at $x = -2$ | Gap at $x = 1$ | Gap at $x = 3$ |
|---|---|---|---|
| $w = 0.5$ | 0.0000 | 0.0000 | 0.0000 |
| $w = 0.3$ | 0.1600 | 0.0400 | 0.3600 |

The gap grows with $x^2$, because the mean error $(0.5 - w)x$ grows with $x$. Training the encoder by SGD on the reparameterized estimator, with $x$ drawn from $p(x)$, 3,000 steps, learning rate 0.02, and batch size 128, finds $w = 0.504$ and $\log\sigma^2 = -0.702$. The optimum is $w = 0.5$ and $\log\sigma^2 = \log 0.5 = -0.693$. The paper uses Adagrad (Section 5). This toy uses plain SGD, because plain SGD is enough here.

Exercise 4 asks you to write the gradient and fit the encoder.

<!-- exercises -->

## Pitfalls

- **Treating the paper's $\sigma$ as the network output.** The paper's MLP outputs $\mu$ and $\sigma$ (Appendix C). Code that outputs $\log\sigma^2$ must use $\sigma = \exp(\tfrac{1}{2}\log\sigma^2)$. Using $\exp(\log\sigma^2)$ as the standard deviation squares the variance and still trains, only to the wrong answer.
- **Sampling without reparameterizing.** If the code draws $z$ with a library call that has no gradient, the encoder gets no learning signal from the reconstruction term. Only the KL term reaches $\phi$, and the encoder collapses toward the prior.
- **Mixing up the two KL terms.** Equation 1 has $D_{KL}(q \,\|\, p(z|x))$, which is the gap and can't be computed. Equation 3 has $D_{KL}(q \,\|\, p(z))$, which is the regularizer. Only the second has the closed form in Appendix B.
- **Forgetting the dataset scaling.** Equation 8 multiplies the minibatch sum by $N/M$. A plain batch average optimizes the same objective divided by $N$, so it changes the effective learning rate, and it changes the weight of any prior on $\theta$ that you add to the loss.
- **Assuming estimator B always wins.** The page above shows a model where estimator A has the lower variance. Check the variance if you rely on the claim.
- **Reading a rising lower bound as better samples.** The bound can rise while the generated samples stay blurry. The bound measures likelihood, and the paper's experiments report the bound and the marginal likelihood, not sample quality (Section 5).
- **Expecting reparameterization for discrete latents.** The trick needs a continuous $z$ and a differentiable $g_\phi$. The paper contrasts its method with wake-sleep, which also applies to discrete latent variables (Section 4).

## Further reading

- Kingma and Welling, [An Introduction to Variational Autoencoders](https://arxiv.org/abs/1906.02691) (2019), the authors' longer treatment of the same derivation.
- Rezende, Mohamed, and Wierstra, [Stochastic Backpropagation and Approximate Inference in Deep Generative Models](https://arxiv.org/abs/1401.4082) (2014). The paper cites this concurrent, independent work in Section 4.
- Ranganath, Gerrish, and Blei, [Black Box Variational Inference](https://arxiv.org/abs/1401.0118) (2014), a score-function approach with variance reduction.
