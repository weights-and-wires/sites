---
id: "ldm"
title: "High-Resolution Image Synthesis with Latent Diffusion Models"
authors: "Rombach et al."
year: 2021
venue: "CVPR 2022"
arxiv: "2112.10752"
url: "https://arxiv.org/abs/2112.10752"
track: generative
tier: core
est_hours: 4
status: full
summary: "A latent diffusion model splits image generation into two stages. A pretrained autoencoder first compresses an image by a factor of 4 to 16 per side, keeping what the eye notices and dropping imperceptible detail, and then a diffusion model trained in that smaller latent space generates the content. Because the U-Net runs on a 64 by 64 grid and not a 512 by 512 one, training and sampling cost far less, and a cross-attention layer lets text, layouts, or other inputs steer the denoiser."
related_course: "ml: diffusion"
prereqs:
  - id: ddpm
    reason: "The latent diffusion objective is the DDPM noise-prediction loss, unchanged, applied to encoder outputs, and it relies on the closed-form noising of $q(x_t|x_0)$."
  - id: vae
    reason: "The first-stage autoencoder is a VAE-style encoder with a small KL penalty toward a standard normal, sampled with the reparameterization trick."
  - id: unet
    reason: "The denoiser is a time-conditional U-Net on a 2D latent grid, and the cross-attention layers sit inside it."
background: [autoencoders, cross-attention]
course_links:
  - title: "Diffusion for audio"
    url: "https://speech.ifkash.dev/lessons/055-diffusion-for-audio"
  - title: "Diffusion from scratch"
    url: "https://speech.ifkash.dev/lessons/052-diffusion-from-scratch"
self_check:
  - q: 'The paper''s KL-regularized autoencoder returns a mean $\mu$ and a variance $\sigma^2$ for every latent entry. How do you draw the latent $z$ that the diffusion model trains on, and why is the draw written that way?'
    a: 'Draw $\varepsilon \sim \mathcal{N}(0, I)$ and set $z = \mu + \sigma \cdot \varepsilon$. Writing the sample as a deterministic function of the parameters and an independent noise draw is the reparameterization trick, and it keeps the encoder differentiable.'
    see: vae
  - q: 'For a latent $z_0$, what does the closed-form forward process give for $z_t$, and what does the network $\epsilon_\theta$ predict?'
    a: 'With $\bar\alpha_t$ as in DDPM, $z_t = \sqrt{\bar\alpha_t}\, z_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon$ for $\epsilon \sim \mathcal{N}(0, I)$. The network predicts the noise $\epsilon$ from $z_t$ and $t$, and the loss is the squared error between the two.'
    see: ddpm
  - q: 'In a cross-attention layer, which of $Q$, $K$, and $V$ come from the image features and which from the condition? What shape is the attention matrix?'
    a: 'The queries come from the image features. The keys and values come from the condition tokens. For $N$ image positions and $M$ tokens, the attention matrix is $N \times M$ and each row sums to 1.'
    see: cross-attention
---

## The problem

Before this paper, the best image generators were diffusion models that ran in pixel space (see [DDPM](/ddpm/)). A pixel-space model has two costs, and both grow with the number of pixels.

- **Every step touches every pixel.** The paper's introduction (Section 1) notes that the loss lets a diffusion model ignore imperceptible detail, but the gradients and the network still run on all pixels. The paper's Figure 2 shows that most bits of a digital image are imperceptible detail, so most of that work is wasted.
- **Sampling repeats the cost.** A sampler calls the U-Net once per step, with tens to hundreds of steps, each at full resolution.

Take one 512 x 512 RGB image. It has 262,144 pixel positions and 786,432 values. A 1000-step sampler pushes 262 million positions through the U-Net, and a self-attention layer on the full grid would need a 262,144 x 262,144 score matrix, which is about $6.9 \times 10^{10}$ entries. The previous remedy was to compress the image into a short sequence of discrete codes and model it with an autoregressive transformer (see [VQGAN](https://arxiv.org/abs/2012.09841)). Section 3 argues that this forces aggressive compression, which costs reconstruction quality.

The paper's answer is to separate two jobs. A one-time autoencoder removes imperceptible detail. A diffusion model then spends its capacity on the semantic content, in a space that is much smaller but still a 2D grid, so a U-Net still fits it. Section 4.2 reports a CelebA-HQ FID of 5.11, which the paper calls a new state of the art for that dataset, and Section 4.1 reports a 38-point FID gap between pixel-space diffusion and an $f = 8$ latent model after 2 million steps with the same compute budget.

## Key ideas

The page uses the paper's notation. An image is $x \in \mathbb{R}^{H \times W \times 3}$, the encoder is $\mathcal{E}$, the decoder is $\mathcal{D}$, and the latent is $z = \mathcal{E}(x) \in \mathbb{R}^{h \times w \times c}$ with downsampling factor $f = H/h = W/w$ (Section 3.1). The denoiser is $\epsilon_\theta$, the condition is $y$, and its encoder is $\tau_\theta$. Where the paper writes the forward process in Appendix B as $q(x_t|x_0) = \mathcal{N}(x_t \mid \alpha_t x_0, \sigma_t^2 I)$ (Equation 4), this page also uses the [DDPM](/ddpm/) symbol $\bar\alpha_t$. The two agree when $\alpha_t^2 = \bar\alpha_t$ and $\sigma_t^2 = 1 - \bar\alpha_t$. Every number comes from a script in `scripts/`. Run any script from the paper folder to reproduce the output.

### Perceptual compression with an autoencoder

**Intuition.** Train an autoencoder that shrinks the image a little, only until the details that no viewer would notice are gone. The paper calls this *perceptual compression*, and it keeps the code a 2D grid, so spatial structure survives. The autoencoder is trained once, and any number of diffusion models can reuse it.

**Math.** Section 3.1 defines $z = \mathcal{E}(x)$ and $\tilde{x} = \mathcal{D}(z) = \mathcal{D}(\mathcal{E}(x))$ with $z \in \mathbb{R}^{h \times w \times c}$. The factor is $f = 2^m$ for an integer $m$. Reconstructions are trained with a perceptual loss and a patch-based adversarial objective, not only with $L_1$ or $L_2$, which would give blurry output. Appendix G gives the full objective (Equation 25):

$$
L_{\text{Autoencoder}} = \min_{\mathcal{E}, \mathcal{D}} \max_{\psi} \Big( L_{rec}(x, \mathcal{D}(\mathcal{E}(x))) - L_{adv}(\mathcal{D}(\mathcal{E}(x))) + \log D_\psi(x) + L_{reg}(x; \mathcal{E}, \mathcal{D}) \Big)
$$

Here $D_\psi$ is the patch discriminator. The term $L_{reg}$ stops the latent from taking an arbitrary scale, and Section 3.1 offers two variants. *KL-reg* adds a slight KL penalty toward $\mathcal{N}(0, 1)$, as in the [VAE](/vae/). *VQ-reg* puts a vector-quantization layer inside the decoder. Appendix G says the KL term is weighted by about $10^{-6}$, which makes it a very weak regularizer, so the latent is close to a plain autoencoder code. For a KL-regularized model, the encoder outputs $\mathcal{E}_\mu$ and $\mathcal{E}_{\sigma^2}$, and the latent is $z = \mathcal{E}_\mu(x) + \mathcal{E}_\sigma(x) \cdot \varepsilon$ with $\varepsilon \sim \mathcal{N}(0, 1)$ (Appendix G).

**Tiny example, computed by `scripts/compute_savings.py`.** For a $512 \times 512 \times 3$ image, the number of latent values and positions follows from $f$ and $c$. The channel counts come from the z-shape row of Table 13 (for example, $64 \times 64 \times 3$ at $f = 4$ and $32 \times 32 \times 4$ at $f = 8$, written for $256 \times 256$ inputs):

| $f$ | Latent shape | Positions | Values | Fewer positions | Fewer values |
|---|---|---|---|---|---|
| 1 (pixels) | 512x512x3 | 262,144 | 786,432 | 1x | 1x |
| 4 | 128x128x3 | 16,384 | 49,152 | 16x | 16x |
| 8 | 64x64x4 | 4,096 | 16,384 | 64x | 48x |
| 16 | 32x32x8 | 1,024 | 8,192 | 256x | 96x |
| 32 | 16x16x32 | 256 | 8,192 | 1,024x | 96x |

Check the $f = 8$ row by hand: $512 / 8 = 64$, so the latent has $64 \cdot 64 = 4{,}096$ positions and $4{,}096 \cdot 4 = 16{,}384$ values. The ratios are $262{,}144 / 4{,}096 = 64 = f^2$ and $786{,}432 / 16{,}384 = 48$.

These ratios are a cost model, and the paper doesn't state them as FLOP counts. The model assumes that a convolutional layer's cost is proportional to the number of positions, which holds when the U-Net's channel widths stay fixed. Table 13 supports that assumption for Section 4.1, because its model sizes stay between 391M and 396M parameters from $f = 1$ to $f = 32$. A full self-attention layer on $N$ positions costs $O(N^2)$, so it shrinks by $f^4$: 4,096 times at $f = 8$. The U-Net applies attention only at coarse resolutions (Table 13 lists the cross-attention resolutions as 32, 16, and 8 for most models), so the real saving from attention is smaller. Over 1000 sampling steps at $f = 8$, the U-Net processes $4{,}096 \cdot 1000 = 4{,}096{,}000$ positions, against $262{,}144{,}000$ for pixels.

Exercise 1 implements the shape and ratio calculations.

### The compression tradeoff

**Intuition.** Compress too little and the diffusion model still has to do the perceptual work, so it trains slowly. Compress too much and the autoencoder throws away information that the diffusion model can't recover.

**What the paper measures.** Section 4.1 trains class-conditional models with $f \in \{1, 2, 4, 8, 16, 32\}$ (written LDM-$f$, with LDM-1 being pixel-space diffusion), each on one A100 with the same number of steps and parameters. Small factors ($f = 1, 2$) train slowly, and an overly large factor (LDM-32) stops improving after relatively few steps. LDM-4 to LDM-16 strike a good balance, and LDM-4 and LDM-8 give the best sample quality (Section 4.1). Table 8 shows why for the autoencoders alone. Its KL-regularized rows give a reconstruction FID of 0.27 at $f = 4$ ($c = 3$), 0.90 at $f = 8$ ($c = 4$), 2.63 at $f = 16$ ($c = 8$), and 7.3 at $f = 32$ ($c = 16$).

### Diffusion in latent space

**Intuition.** Treat the latent as the data. Noise it, ask the U-Net to predict the noise, and decode the final sample with $\mathcal{D}$ in a single pass. The paper changes nothing in the diffusion model itself, only the space it runs in.

**Math.** Section 3.2 writes the pixel-space objective as (Equation 1)

$$
L_{DM} = \mathbb{E}_{x,\, \epsilon \sim \mathcal{N}(0,1),\, t}\left[ \lVert \epsilon - \epsilon_\theta(x_t, t) \rVert_2^2 \right]
$$

with $t$ uniform on $\{1, \dots, T\}$, and the latent objective as (Equation 2)

$$
L_{LDM} := \mathbb{E}_{\mathcal{E}(x),\, \epsilon \sim \mathcal{N}(0,1),\, t}\left[ \lVert \epsilon - \epsilon_\theta(z_t, t) \rVert_2^2 \right]
$$

The two equations differ only in what the network sees: $x_t$ in Equation 1 and $z_t$ in Equation 2. Section 3.2 notes that the forward process stays fixed, so $z_t$ comes from $z = \mathcal{E}(x)$ in closed form during training. In DDPM notation, $z_t = \sqrt{\bar\alpha_t}\, z + \sqrt{1-\bar\alpha_t}\, \epsilon$. The backbone $\epsilon_\theta(\circ, t)$ is a time-conditional U-Net, and samples of $z$ decode to an image with one pass through $\mathcal{D}$. Table 12 and Table 13 use $T = 1000$ steps with a linear noise schedule.

**Rescaling the latent.** Appendix G and Appendix D.1 add a detail that matters in practice. The signal-to-noise ratio at step $t$ is $\mathrm{SNR}(t) = \alpha_t^2 / \sigma_t^2$ (Appendix B), and Appendix D.1 notes that the SNR the network sees also depends on the latent's variance, $\mathrm{Var}(z)/\sigma_t^2$. A KL-regularized latent can have a large variance, which makes the SNR high, and the model then commits to semantic detail early in the reverse process. Appendix G fixes this by estimating one scalar from the first batch,

$$
\hat\sigma^2 = \frac{1}{bchw} \sum_{b,c,h,w} (z_{b,c,h,w} - \hat\mu)^2, \qquad \hat\mu = \frac{1}{bchw} \sum_{b,c,h,w} z_{b,c,h,w}
$$

and training on $z / \hat\sigma$. The VQ-regularized space already has a variance close to 1 and needs no rescaling (Appendix D.1).

**Tiny example, computed by `scripts/latent_noising_example.py`.**

- **Sampling.** With $\mu = (1, -2)$, variances $(1, 4)$, and $\varepsilon = (0.5, -1)$, the latent is $z = (1.5, -4.0)$. The second entry is $-2 + 2 \cdot (-1)$.
- **KL penalty.** Each component contributes $\tfrac12(\mu^2 + \sigma^2 - 1 - \log \sigma^2)$, which is 0.5 and 2.8069, so the total is 3.3069. Weighted by $10^{-6}$, as in Appendix G, it adds $3.3 \times 10^{-6}$ to the loss, which is why the penalty barely constrains the latent.
- **Rescaling.** For a batch of shape $(4, 3, 8, 8)$ drawn with standard deviation 5 and mean 1, the script finds $\hat\sigma = 4.9949$, and the rescaled batch has standard deviation 1.0.
- **Why the scale matters.** At $\alpha_t = 0.6$ and $\sigma_t = 0.8$, $\mathrm{SNR}(t) = 0.36 / 0.64 = 0.5625$. A latent with standard deviation 1 carries a signal-to-noise ratio of 0.535 at that step, and a latent with standard deviation 5 carries 14.991, nearly 28 times more.

Exercise 2 implements the sampling, the KL term, and the rescaling.

**A toy pipeline, computed by `scripts/toy_latent_diffusion.py`.** To see the whole two-stage idea with no training, replace the autoencoder with PCA and the U-Net with an exact formula. Take 16-dimensional data near a 2D subspace. PCA with $k = 2$ is the pair $(\mathcal{E}, \mathcal{D})$, and its mean squared reconstruction error per sample is 0.03492, which equals the sum of the discarded covariance eigenvalues, 0.03492. If the latents are Gaussian, $z_0 \sim \mathcal{N}(m, \Sigma)$, the minimizer of Equation 2 has a closed form, because $\epsilon$ and $z_t$ are jointly Gaussian:

$$
\epsilon_\theta(z_t, t) = \sqrt{1-\bar\alpha_t}\; \Sigma_t^{-1} \left( z_t - \sqrt{\bar\alpha_t}\, m \right), \qquad \Sigma_t = \bar\alpha_t \Sigma + (1 - \bar\alpha_t) I
$$

This formula is not in the paper. It is the Gaussian special case of Equation 2, and it stands in for a trained network. Sampling 4,000 latents with 200 DDPM steps (a linear schedule from $10^{-3}$ to 0.05, so $\bar\alpha_T = 0.0056$) and decoding them gives pixel-space samples whose covariance differs from the reconstructed data's by 3.6 percent in relative Frobenius norm. The denoiser handles 2 numbers per step, against 16 in pixel space. Exercise 4 builds this pipeline.

### Cross-attention conditioning

**Intuition.** To steer generation with text, a layout, or another input $y$, encode $y$ into a set of tokens, and let every spatial position of the U-Net's feature map look at those tokens through attention. The image supplies the questions, and the condition supplies the answers.

**Math.** Section 3.3 turns the unconditional denoiser into $\epsilon_\theta(z_t, t, y)$. A domain-specific encoder $\tau_\theta$ maps $y$ to $\tau_\theta(y) \in \mathbb{R}^{M \times d_\tau}$, and a cross-attention layer inside the U-Net computes

$$
\mathrm{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{Q K^\top}{\sqrt{d}}\right) \cdot V, \qquad Q = W_Q^{(i)} \cdot \phi_i(z_t),\quad K = W_K^{(i)} \cdot \tau_\theta(y),\quad V = W_V^{(i)} \cdot \tau_\theta(y)
$$

where $\phi_i(z_t) \in \mathbb{R}^{N \times d_\epsilon^i}$ is a flattened intermediate representation of the U-Net (Section 3.3). The conditional objective is (Equation 3)

$$
L_{LDM} := \mathbb{E}_{\mathcal{E}(x),\, y,\, \epsilon \sim \mathcal{N}(0,1),\, t}\left[ \lVert \epsilon - \epsilon_\theta(z_t, t, \tau_\theta(y)) \rVert_2^2 \right]
$$

and $\tau_\theta$ and $\epsilon_\theta$ are optimized jointly through it. For text, Section 4.3.1 uses a BERT tokenizer and implements $\tau_\theta$ as a transformer, in a 1.45B-parameter KL-regularized model trained on LAION-400M.

**Notation differences.** This page writes the projections with row vectors, $Q = \phi\, W_Q^\top$, because that matches NumPy. The paper's printed shapes for the projection matrices don't fit its own products. It lists $W_V^{(i)} \in \mathbb{R}^{d \times d_\epsilon^i}$ and $W_Q^{(i)} \in \mathbb{R}^{d \times d_\tau}$, but $Q$ multiplies $\phi_i$ (width $d_\epsilon^i$) and $V$ multiplies $\tau_\theta(y)$ (width $d_\tau$). The only consistent reading is $W_Q \in \mathbb{R}^{d \times d_\epsilon^i}$ and $W_K, W_V \in \mathbb{R}^{d \times d_\tau}$, which is the one used here and in Exercise 3. The paper also doesn't mention an output projection back to width $d_\epsilon^i$, which the layer needs before it rejoins the U-Net. Section 3.3 and Figure 3 also describe a second option, plain concatenation, for dense conditions such as semantic maps.

**Tiny example, computed by `scripts/cross_attention_example.py`.** Take $N = 2$ image positions $\phi = I$, $M = 3$ tokens $\tau = [[1,0],[0,1],[1,1]]$, $W_Q = W_K = I$, and $W_V = \operatorname{diag}(1, 2)$, with $d = 2$. The scores are $QK^\top/\sqrt{2}$, which gives $0.7071$ for a matching token and $0$ otherwise. For the first position the scores are $(0.7071, 0, 0.7071)$. The softmax is $(0.4011, 0.1978, 0.4011)$, because $e^{0.7071} = 2.028$ and the denominator is $2 \cdot 2.028 + 1 = 5.056$. The values are $(1,0)$, $(0,2)$, and $(1,2)$, so the output is $0.4011 \cdot (1,0) + 0.1978 \cdot (0,2) + 0.4011 \cdot (1,2) = (0.8022, 1.1978)$. The second position gives $(0.5989, 1.6044)$. Reordering the tokens leaves both outputs unchanged, because cross-attention has no positional information of its own. Any order information has to come from $\tau_\theta$, such as the positional encodings in its transformer.

Exercise 3 implements this layer.

<!-- exercises -->

## Pitfalls

- **Reading $f$ as an image size.** The factor $f$ divides each side. A latent for $f = 8$ has 64 times fewer positions, not 8 times fewer. Because the latent also has $c$ channels, the value ratio is $3 f^2 / c$, which is 48 for $f = 8$, $c = 4$.
- **Taking the attention saving at face value.** The $f^4$ figure applies to full self-attention on the whole grid. The U-Net uses attention only at some resolutions, so the end-to-end speed-up is smaller than that figure.
- **Forgetting that the autoencoder is frozen.** The diffusion model trains on latents from a fixed $\mathcal{E}$. The paper stresses that this stage is trained once and reused (Section 1), and that it avoids the delicate weighting of reconstruction against prior that joint training needs.
- **Sampling the KL latent with the variance.** The encoder reports $\sigma^2$ (or its log), and the latent uses $\sigma$. Passing the variance as the scale gives noise with the wrong spread.
- **Skipping the rescaling for a KL latent.** A latent with a large standard deviation changes the effective signal-to-noise ratio at every step (Appendix D.1). The scalar $\hat\sigma$ is one number over all of $b, c, h, w$, not one per channel.
- **Softmax over the wrong axis in cross-attention.** Each image position must distribute its attention over the condition tokens, so the rows of the $N \times M$ matrix sum to 1. Normalizing over positions instead still returns plausible-looking numbers.
- **Expecting token order to matter by itself.** Cross-attention treats the tokens as a set. Word order reaches the U-Net only through whatever $\tau_\theta$ encodes.
- **Expecting a PCA latent to match the paper.** PCA is linear and has no adversarial loss, so it can't reproduce fine texture. It makes the idea checkable, nothing more.
- **Treating the toy denoiser as a trained model.** The closed form in Exercise 4 is exact only because the latent is Gaussian. A real image latent isn't, which is why the paper trains a U-Net.

## Further reading

- [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion) is the official code release, which the paper links in its introduction.
- [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion) is the text-to-image model built on this method.
- [The Illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/), by Jay Alammar, is a visual walkthrough of the autoencoder, the denoiser, and the text encoder.
- [Taming Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2012.09841), by Esser et al., is the VQGAN paper that the first stage builds on.
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598), by Ho and Salimans, is the guidance method that the paper's text-to-image results use (Section 4.3.1).
