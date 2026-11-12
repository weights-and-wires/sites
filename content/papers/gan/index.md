---
id: "gan"
title: "Generative Adversarial Networks"
authors: "Goodfellow et al."
year: 2014
venue: "NeurIPS 2014"
arxiv: "1406.2661"
url: "https://arxiv.org/abs/1406.2661"
track: generative
tier: extended
est_hours: 3
status: full
summary: "A generator network turns noise into samples, and a discriminator network learns to tell those samples from real data. The two play a minimax game whose optimal discriminator is a density ratio, and with that discriminator in place the generator's objective equals a Jensen-Shannon divergence between the data and model distributions. The game needs only backpropagation and no likelihood, but its training can stall or oscillate, and a generator can collapse many noise inputs onto few outputs."
related_course: "ml: generative models"
prereqs: []
background: [minimax-games, jensen-shannon-divergence]
course_links:
  - title: "GAN vocoders"
    url: "https://speech.ifkash.dev/lessons/055-gan-vocoders"
self_check:
  - q: 'In $\min_G \max_D V(D, G)$, which player controls which variable, and what does the inner $\max_D$ do for a fixed generator?'
    a: 'The generator $G$ controls $p_g$ and minimizes $V$. The discriminator $D$ maximizes $V$. For a fixed $G$, the inner maximization finds the best possible classifier of data against samples, $D^*_G(x) = p_{\text{data}}(x) / (p_{\text{data}}(x) + p_g(x))$.'
    see: minimax-games
  - q: 'Compute $\mathrm{JSD}(p \Vert q)$ for $p = [1, 0]$ and $q = [0, 1]$ in nats. What does the result say about the Jensen-Shannon divergence of two distributions with disjoint support?'
    a: 'The mixture is $m = [0.5, 0.5]$, and each KL term is $\log 2$, so $\mathrm{JSD} = \tfrac12\log 2 + \tfrac12\log 2 = \log 2 \approx 0.6931$. That is the maximum value of the divergence, and it is the same for every pair of disjoint distributions, however far apart they are.'
    see: jensen-shannon-divergence
  - q: 'At the global optimum, $p_g = p_{\text{data}}$. What is $D^*_G(x)$ there, and what is the value of $V$?'
    a: '$D^*_G(x) = 1/2$ for every $x$, because the two densities are equal. Then $V = \log\tfrac12 + \log\tfrac12 = -\log 4 \approx -1.3863$ (Theorem 1).'
    see: minimax-games
---

## The problem

Before 2014, a generative model usually meant a model of the density $p(x)$, and the training methods that fit one paid for it in compute. Section 2 of the paper lists the costs. A restricted Boltzmann machine or a deep Boltzmann machine defines an unnormalized density divided by a partition function, and the partition function and its gradient are intractable except for trivial models. Estimating them with Markov chain Monte Carlo (MCMC) runs into mixing problems. Deep belief networks inherit the difficulties of both directed and undirected models. Score matching and noise-contrastive estimation need a density that is analytic up to a constant. Generative stochastic networks avoid the density but still sample by running a Markov chain.

Consider generating a 28 by 28 digit. Any model that needs a Markov chain to draw a sample pays for many network passes per sample, and the chain's samples are correlated until it mixes. The paper asks for a model that draws a sample with one forward pass and trains with plain backpropagation, with no inference step and no partition function.

The answer is to skip the density. Section 1 compares the setup to counterfeiters and police. A generator $G$ maps noise to samples, and a discriminator $D$ tries to tell its output from real data. If $D$ can't, $G$ has matched the data. Section 5 reports Parzen-window log-likelihood estimates on MNIST and TFD (Table 1). On MNIST, adversarial nets score 225 ± 2, against 214 ± 1.1 for the deep GSN and 138 ± 2 for the DBN. On TFD they score 2057 ± 26, which is below the stacked CAE's 2110 ± 50. The paper calls the Parzen estimate high-variance and says it does well only in low dimensions (Section 5), so treat the numbers as a first sketch and not as a benchmark.

## Key ideas

The sections that follow use the paper's notation: data distribution $p_{\text{data}}$, noise prior $p_z$, generator $G(z; \theta_g)$ with induced distribution $p_g$, and discriminator $D(x; \theta_d)$ with one scalar output. Two notes on notation. The paper writes the value function as $V(D, G)$ in Equation 1 and as $V(G, D)$ elsewhere, and this page uses $V(G, D)$ throughout. Algorithm 1 samples noise from "$p_g(z)$", which is a typo for the prior $p_z(z)$ defined in Section 3, so this page writes $p_z$. All logs are natural, and the scripts print numbers in nats. Run any script from the paper folder to reproduce the output.

### The minimax game

**Intuition.** $D$ is a binary classifier whose two classes are "came from the data" and "came from $G$". $G$ is trained to make that classifier fail. Each player's loss is the other's gain, so the two can't both win.

**Math.** $D(x)$ is the probability that $x$ came from the data and not from $p_g$. Section 3 trains $D$ to maximize the probability of the correct label on both kinds of samples, and trains $G$ to minimize $\log(1 - D(G(z)))$. Together this is a two-player minimax game (Section 3, Equation 1):

$$
\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]
$$

The first term rewards $D$ for scoring real data near 1. The second rewards $D$ for scoring generated data near 0, and it penalizes $G$ when $D$ succeeds.

In practice the game is played with minibatches (Algorithm 1). The paper alternates $k$ steps of ascent on $D$ with one step of descent on $G$, and it used $k = 1$ in its experiments. It explains that optimizing $D$ to completion in the inner loop is computationally prohibitive and overfits on finite datasets, so $k$ steps keep $D$ near its optimum while $G$ changes slowly (Section 3).

**Tiny example, computed by `scripts/optimal_discriminator.py`.** You can check these claims without sampling, on a discrete $x$ with four values. When $D$ outputs $1/2$ for every $x$, the value is $\log\tfrac12 + \log\tfrac12 = -\log 4 = -1.3863$, whatever $p_{\text{data}}$ and $p_g$ are. A discriminator that does no better than a coin flip earns that score, and a better one scores higher. Exercise 1 implements $V$ and checks this property.

### The optimal discriminator

**Intuition.** Fix the generator. At a point $x$, the discriminator sees $p_{\text{data}}(x)$ real examples for every $p_g(x)$ fake ones, and the best it can do is report the fraction that is real.

**Math.** Proposition 1 (Section 4.1, Equation 2) states the result:

$$
D^*_G(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}
$$

The proof rewrites the value function as an integral over $x$ (Equation 3), using the change of variables $x = G(z)$ so that the expectation over $z$ becomes an expectation over $p_g$:

$$
V(G, D) = \int_x p_{\text{data}}(x)\log D(x) + p_g(x)\log(1 - D(x))\,dx
$$

For any $(a, b) \in \mathbb{R}^2 \setminus \{0, 0\}$, the function $y \mapsto a\log y + b\log(1 - y)$ reaches its maximum on $[0, 1]$ at $a/(a+b)$. The integrand has that form at every $x$, so each point is maximized separately. The paper also notes that the discriminator's objective is the log-likelihood of the conditional probability $P(Y = y \mid x)$, where $Y$ records whether $x$ came from the data.

**Tiny example, computed by `scripts/optimal_discriminator.py`.** Take four values of $x$ with

$$
p_{\text{data}} = [0.4,\ 0.4,\ 0.2,\ 0.0], \qquad p_g = [0.1,\ 0.2,\ 0.3,\ 0.4].
$$

Then $D^*_G = [0.8,\ 0.6667,\ 0.4,\ 0.0]$. Check the first entry by hand: $0.4 / (0.4 + 0.1) = 0.8$. The last entry is 0, because $G$ produces a value that the data never contains, so $D$ rejects it with certainty. Plugging $D^*_G$ into $V$ gives $-0.9686$, which is higher than the $-1.3863$ of the constant discriminator. Exercise 1 implements $D^*_G$ and $V$, including the $0 \log 0 = 0$ convention that the fourth entry needs.

### The criterion is a Jensen-Shannon divergence

**Intuition.** Once $D$ is optimal, the game's value depends only on how far apart $p_g$ and $p_{\text{data}}$ are. The generator is then minimizing a distance between two distributions, even though it never sees a density.

**Math.** Substitute $D^*_G$ into the value function to get the virtual training criterion for the generator (Section 4.1, Equation 4):

$$
C(G) = \max_D V(G, D) = \mathbb{E}_{x \sim p_{\text{data}}}\!\left[\log\frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}\right] + \mathbb{E}_{x \sim p_g}\!\left[\log\frac{p_g(x)}{p_{\text{data}}(x) + p_g(x)}\right]
$$

Theorem 1 says the global minimum of $C(G)$ is reached if and only if $p_g = p_{\text{data}}$, and that the value there is $-\log 4$. The proof subtracts $-\log 4$ and recognizes two Kullback-Leibler divergences (Equation 5):

$$
C(G) = -\log 4 + \mathrm{KL}\!\left(p_{\text{data}} \,\Big\Vert\, \frac{p_{\text{data}} + p_g}{2}\right) + \mathrm{KL}\!\left(p_g \,\Big\Vert\, \frac{p_{\text{data}} + p_g}{2}\right)
$$

The two KL terms sum to twice the Jensen-Shannon divergence (Equation 6):

$$
C(G) = -\log 4 + 2\,\mathrm{JSD}(p_{\text{data}} \Vert p_g)
$$

The divergence is non-negative and zero only for equal distributions, which gives the theorem. This page defines $\mathrm{JSD}(p \Vert q) = \tfrac12\mathrm{KL}(p \Vert m) + \tfrac12\mathrm{KL}(q \Vert m)$ with $m = (p+q)/2$, which is the convention Equation 6 needs. In nats, the JS divergence is at most $\log 2$.

**Tiny example, computed by `scripts/optimal_discriminator.py`.** For the same four-point distributions, with $m = [0.25, 0.3, 0.25, 0.2]$, the two KL terms are $\mathrm{KL}(p_{\text{data}} \Vert m) = 0.2584$ and $\mathrm{KL}(p_g \Vert m) = 0.1592$. Their sum is $0.4176$, so $\mathrm{JSD} = 0.2088$. Equation 6 predicts $C(G) = -1.3863 + 2 \cdot 0.2088 = -0.9686$, which matches the value computed directly from $D^*_G$ in the previous idea.

Two edge cases from the same script matter in practice. With $p_g = p_{\text{data}}$, $D^* = 0.5$ on the support and $C(G) = -1.3863$. With disjoint supports, the JSD equals $\log 2 = 0.6931$ and $C(G) = 0$, whatever the distance between the supports. Exercise 1 asks you to verify Equation 6 on random distributions.

Section 4.2 adds a convergence argument (Proposition 2). If $G$ and $D$ have enough capacity, $D$ reaches its optimum at each step, and $p_g$ is updated to improve $C(G)$, then $p_g$ converges to $p_{\text{data}}$. The proof uses the fact that $V$ is convex in $p_g$. The paper states the limits itself: a real $G$ is a multilayer perceptron with parameter space $\theta_g$, not a free density, and the neural-network case has "multiple critical points" and no guarantee (end of Section 4.2).

### The non-saturating generator loss

**Intuition.** Early in training, $G$ is poor, and $D$ rejects its samples with high confidence. The generator loss $\log(1 - D(G(z)))$ is flat there, so $G$ gets almost no learning signal at the moment it needs the most.

**Math.** Section 3 notes that Equation 1 "may not provide sufficient gradient for $G$ to learn well", and that $\log(1 - D(G(z)))$ saturates when $D$ is confident. The fix is to train $G$ to maximize $\log D(G(z))$ instead. The paper says this gives the same fixed point of the dynamics of $G$ and $D$, with much stronger gradients early in learning. Write $D = \sigma(a)$ with logit $a$. The two generator losses, each to be minimized, have these derivatives:

$$
\frac{\partial}{\partial a}\log(1 - \sigma(a)) = -\sigma(a), \qquad
\frac{\partial}{\partial a}\bigl[-\log\sigma(a)\bigr] = -(1 - \sigma(a))
$$

The first vanishes where $D$ is near 0, which is where $G$ is losing. The second vanishes where $D$ is near 1, which is where $G$ is winning.

**Tiny example, computed by `scripts/saturation.py`.** Let $D$ be confident that a sample is fake, with logit $a = -4.6$ so that $D(G(z)) = 0.00995$. The saturating gradient is $-0.00995$ and the non-saturating gradient is $-0.99005$, a ratio of 99.5. At $a = 0$ both equal $-0.5$. At $a = 2$ the picture flips, and the saturating gradient ($-0.8808$) is the larger one. Exercise 2 implements both losses on logits and checks the gradients against finite differences.

### Instability and the Helvetica scenario

**Intuition.** The game has a saddle point, and plain gradient steps on a saddle don't have to settle there. They can circle it. A second failure happens when $G$ learns a few outputs that $D$ accepts and sends many different $z$ to them.

**Math.** Section 6 names the second failure the "Helvetica scenario": $G$ collapses too many values of $z$ to the same value of $x$ to have enough diversity to model $p_{\text{data}}$. The paper's remedy is to keep $D$ synchronized, so that $G$ isn't trained too much without updating $D$, the same reason Algorithm 1 uses $k$ discriminator steps. The paper doesn't use the later term "mode collapse" or analyze either failure in depth. This page keeps the paper's claims separate from the standard follow-up picture below.

*A bilinear game.* Take $\min_x \max_y\, xy$, whose saddle is $(0, 0)$. Simultaneous gradient steps with learning rate $\eta$ are $x \leftarrow x - \eta y$ and $y \leftarrow y + \eta x$. Each step multiplies the distance from the saddle by $\sqrt{1 + \eta^2}$, so the iterates spiral outward. `scripts/bilinear_game.py` starts at $(1, 1)$ with $\eta = 0.1$. The distance grows from 1.4142 to 1.4864 after 10 steps, 1.8136 after 50, and 2.3259 after 100, matching the predicted $\sqrt{2}(1 + 0.01)^{50} = 2.3259$. This game is not a GAN. It is the smallest case that shows how alternating ascent and descent can fail to converge.

*A collapsed generator.* `scripts/mixture_collapse.py` takes data from an equal mixture of $N(-2, 0.5^2)$ and $N(2, 0.5^2)$, and compares generators with a 40-bin histogram estimate of the JS divergence. A generator that matches the mixture has $\mathrm{JSD} = 0.0001$. A generator that puts all its mass on the $+2$ mode has $0.2160$, and the best single Gaussian (mean $-0.002$, standard deviation 2.06) has $0.1940$. Both are well under the $\log 2 = 0.6931$ ceiling. A generator stuck on one mode still produces samples that look real, and the divergence is the only thing that reveals the missing mode. Exercise 4 builds these diagnostics.

*What the training run shows.* Exercise 3 trains the full loop on $N(3, 1)$. `scripts/gan_1d.py` runs it twice. With $k = 5$ and learning rate 0.02 for 4000 iterations, the generator reaches a mean of 2.96 and a standard deviation of 1.05 on seed 0. With $k = 1$ and learning rate 0.05 for 2000 iterations, the mean is 3.14, but the standard deviation shrinks to 0.27. The two runs differ in more than $k$, so don't read this as a controlled test. It shows that the settings of the alternation can leave the generator with the right center and far too little spread.

<!-- exercises -->

## Pitfalls

- **Swapping the roles in the optimal discriminator.** $D^*_G(x) = p_{\text{data}}(x) / (p_{\text{data}}(x) + p_g(x))$ puts the data density on top. Putting $p_g$ on top gives the probability of "fake", and Theorem 1's value changes.
- **Reading $-\log 4$ as a loss that training should reach.** It is the value of $C(G)$ at the optimum, and it holds only when $D$ is optimal. In a real run, the measured losses of $D$ and $G$ are noisy estimates against a changing opponent, so they don't track sample quality.
- **Computing the logs on probabilities.** Compute $\log\sigma(a)$ from logits with `logaddexp`. A naive $\log(1/(1 + e^{-a}))$ overflows for $a = -1000$, and $\log(1 - D)$ loses precision as $D \to 1$.
- **Training the generator on $\log(1 - D(G(z)))$ from the start.** This is Equation 1 and the gradient is tiny when $D$ is confident. Use $-\log D(G(z))$, as Section 3 suggests.
- **Dropping $0 \log 0 = 0$.** A point where $p_{\text{data}}$ or $p_g$ is zero adds nothing to the value. Without the convention, a discrete implementation returns `nan` on any pair of distributions with different supports.
- **Treating the JS divergence as a distance that shrinks smoothly.** For distributions with disjoint support it equals $\log 2$ however far apart they are (see the disjoint case above). The value of $C(G)$ then gives $G$ no signal about which direction to move. The paper doesn't analyze this case, and later work does.
- **Trusting the theory for the networks.** Theorem 1 and Proposition 2 hold in the non-parametric setting, in the space of density functions. The paper says explicitly that a multilayer perceptron generator has no such guarantee (end of Section 4.2).
- **Forgetting that $D$ is evaluated on both classes.** Each $D$ step needs real and generated samples. Training $D$ on one class only teaches it to output a constant.
- **Comparing Parzen scores across papers.** Table 1 uses a Gaussian Parzen window fit to generated samples, with $\sigma$ chosen on a validation set. The estimate depends on that choice and the paper calls it high-variance.

## Further reading

- [The authors' code](https://github.com/goodfeli/adversarial) is the original Pylearn2 implementation that the paper's footnote links to.
- [NIPS 2016 Tutorial: Generative Adversarial Networks](https://arxiv.org/abs/1701.00160), by Ian Goodfellow, covers the theory, the non-saturating loss, and the training difficulties in more depth than the paper.
- [Wasserstein GAN](https://arxiv.org/abs/1701.07875), by Arjovsky, Chintala, and Bottou, argues that the JS divergence gives poor gradients on distributions with disjoint supports and proposes a different distance.
- [Unrolled Generative Adversarial Networks](https://arxiv.org/abs/1611.02163), by Metz et al., studies mode collapse and a training procedure that reduces it.
- [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784), by Mirza and Osindero, develops the first extension that the paper lists in Section 7.
