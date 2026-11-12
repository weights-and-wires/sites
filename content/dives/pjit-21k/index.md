---
id: pjit-21k
title: "Training a text-to-image model for $21K (Pyramid-JiT)"
summary: "Linum trains a 2B-parameter diffusion transformer that works directly on pixels, with 32x32 patches, so a 512x512 image becomes only 256 tokens. The recipe adds low-resolution readouts along the trunk and a timestep schedule that starts noisy and widens later. The post reports 660M training samples for about $21K of rented GPUs."
source:
  title: "Training a Text-to-Image Model for $21K"
  author: "Sahil Chopra and Manu Chopra (Linum)"
  site: "Linum Field Notes"
  date: "2026-11-04"
  url: "https://www.linum.ai/field-notes/pjit-21k"
topics: [diffusion, pixel-space, efficiency, training-cost]
est_minutes: 45
prereqs:
  - {id: ddpm, reason: "The page adds noise to an image and trains a network to undo it, so you need the noisy-sample and timestep vocabulary from diffusion."}
  - {id: flow-matching, reason: "The post's timesteps, velocity, and v-loss come from the flow-matching family, where t interpolates between data and noise."}
  - {id: vit, reason: "P-JiT cuts an image into 32x32 patches and treats each as one token, exactly the patch embedding a Vision Transformer uses."}
  - {id: dit, reason: "The model is a diffusion transformer with timestep conditioning through adaptive layer norm that starts as the identity."}
  - {id: ldm, reason: "The post's whole motivation is to remove the VAE that latent diffusion puts in front of the transformer."}
background: [patchify, adaln-conditioning, exponential-moving-averages, gaussian-noise, downsampling-and-upsampling]
course_links:
  - {title: "Diffusion from scratch", url: "https://speech.ifkash.dev/lessons/062-diffusion-from-scratch"}
  - {title: "Flow matching", url: "https://speech.ifkash.dev/lessons/066-flow-matching"}
---

This page covers two posts from Linum. The main source is [Training a Text-to-Image Model for $21K](https://www.linum.ai/field-notes/pjit-21k), which reports a large training run. It leans on an earlier post, [Pyramid-JiT: Low-Res Drafts, High-Res Images](https://www.linum.ai/field-notes/pyramid-jit), where the method (called P-JiT) is described and ablated. Each claim names its source, and anything that is our own inference or our own toy number is marked.

## The problem

Video models are slow and expensive because attention costs $O(n^2)$ in the number of tokens $n$. The post's authors want a cheaper video model, and they use text-to-image as a testbed: a video model has to learn "nouns" before "verbs", so video models start as image models.

The usual way to shrink $n$ is a separate autoencoder (a VAE) that compresses the image before the diffusion transformer sees it. See [Latent diffusion](/ldm/). The post says VAEs seem to hit an empirical ceiling of about 16x16 token reduction, and that a VAE with better reconstruction can still give worse generations.

So the post removes the VAE. The diffusion model works on raw pixels and does the compressing itself, using bigger patches. Linum v2 (their earlier model, a Wan VAE plus a standard DiT) used a 16x16 reduction. P-JiT uses 32x32, so a 512x512 image becomes 256 tokens, the same count v2 spent on a 256x256 image.

The widget shows why this matters. Try 16 px against 32 px patches at 512 px, then at 1024 px.

<div class="widget" data-widget="token-budget"></div>

The cost of that choice shows up in the last line of the readout. A 32x32 RGB patch holds $32 \cdot 32 \cdot 3 = 3{,}072$ numbers, and a single token must carry them. A naive model would squeeze them into a narrow vector. The Pyramid-JiT post says it widened the linear bottleneck from 256 to 1,024 dimensions, which changes the compression from 12:1 to 3:1 (that post also reports the two gave "pretty much the same" results).

Two numbers to keep straight. The post states a 4x reduction in attention context (1,024 tokens to 256). The score matrix has $n^2$ entries, so it shrinks 16x (1,048,576 to 65,536 pairs); that 16x is our arithmetic, not a claim in the post. In the real model, 256 text tokens are added, giving 512 tokens at 512 px and 1,280 at 1024 px (the post's figure).

## Key ideas

### Pixel-space x-prediction

In a latent model, the network predicts noise or velocity in the VAE's latent space. P-JiT keeps pixels and predicts the clean image $x$ directly ("x-prediction"), following the JiT paper that the post builds on.

The post's plots run from clean on the left to pure noise on the right, and it clamps the loss weight at $t = 0.1$, so $t = 1$ is pure noise. This page uses the following form. It is our inference from the JiT-style setup, not a formula printed in the post:

$$z_t = (1 - t)\,x + t\,\sigma\,\epsilon, \qquad v = \sigma\epsilon - x.$$

Here $\sigma$ is the noise scale: the post says it is 2 at 512 px and 4 at 1024 px. If the network outputs $\hat x$, the velocity it implies is $\hat v = (z_t - \hat x)/t$. Subtract the true velocity and the noise cancels:

$$\hat v - v = \frac{x - \hat x}{t}.$$

The post trains with a "v-loss", and the Pyramid-JiT post writes it as $\|(\hat x - x)/\max(t, 0.1)\|^2$. That matches the identity above, and it means an error in $\hat x$ costs $1/t^2$ times more at low noise. A worked check: an $x$ error of $0.1$ at $t = 0.3$ gives $(0.1/0.3)^2 \approx 0.111$, at $t = 0.9$ gives $0.012$, and at $t = 0.05$ the clamp acts as if $t = 0.1$ and gives $1.0$. The clamp stops the weight from growing without bound near clean images.

### Readouts along the trunk

The trunk has 22 transformer blocks after two refiner blocks each for text and image. Because every block works toward a clean image, the post attaches small linear heads partway up and scores them too. After block 10 a head predicts the image at one quarter of the side length (128x128 for a 512x512 image); after block 16, one half (256x256); the last head predicts the full image. Targets are bilinear, antialiased downsamples of $x_0$, each scored with the same v-weighted loss at weight 1. The two extra heads add 2.85M parameters, and the post drops them at sampling time.

The post's reasoning, from the Pyramid-JiT post: the early blocks of a DiT adapt slowly, a low-resolution target boosts their gradient, and it makes them learn structure first. It cites related work (REPA, RAE v2, Self-Flow). Treat this as their hypothesis. The post does not measure per-block gradients.

Here is a 1-D version of the targets. Average-pooling $x = [1, 3, 5, 7, 2, 2, 4, 4]$ by 4 gives $[4, 3]$ and by 2 gives $[2, 6, 2, 4]$. A head that predicts $[3, 3]$ for the first target has mean squared error $((3-4)^2 + 0)/2 = 0.5$.

<div class="widget" data-widget="pyramid-targets"></div>

The Pyramid-JiT post ablates this on 512x512 images with the same 22-block trunk after 75M samples. The metric is the v-space loss at the final block (lower is better):

| Variant | Loss | Change |
| --- | --- | --- |
| 1 readout (baseline) | 0.1037 | none |
| 3 readouts: 128, 256, 512 at blocks 10, 16, 22 | 0.0934 | -10.0% |
| 3 readouts, all at full resolution | 0.0948 | -8.5% |
| 2 readouts: 128 at block 10, 512 at 22 | 0.0960 | -7.4% |
| 3 readouts, weights 0.25, 0.5, 1.0 | 0.0983 | -5.1% |

Equal weights and a growing resolution did best. The post reports each as a single run, with no error bars, so small gaps (for example 0.0934 against 0.0948) deserve caution.

### A schedule that starts noisy and then widens

Training needs a distribution over timesteps. The post samples $t = \sigma(z)$ with $z \sim \mathcal N(\mu, s^2)$, a logit-normal distribution. For the first 80M samples it uses $(\mu, s) = (0.8, 0.8)$, which concentrates on high noise, so the model learns basic structure. Then it switches to $(-0.2, 1.0)$ for the rest of training to learn fine detail. The Pyramid-JiT post says it found widening the schedule in its earlier JiT-DDT post, which this page doesn't cover, and this post gives no ablation of the schedule.

The probability that $t$ falls in an interval has a closed form, where $\Phi$ is the standard normal CDF:

$$P(a < t < b) = \Phi\!\left(\frac{\operatorname{logit} b - \mu}{s}\right) - \Phi\!\left(\frac{\operatorname{logit} a - \mu}{s}\right).$$

The post's plot reports, for the band around $t = 0.1$, about 3.0% of timesteps in phase 2 against under 0.05% in phase 1, a ratio of 107x. It labels the band 0.08 to 0.13. Our computation on exactly that band gives 3.2% and 0.034%, a ratio of 94x. Our computation on 0.075 to 0.125 gives 3.0% and 0.028%, or 107x. The post's labels probably round the band's edges, which is our inference. The other statistics on the post's plot match ours exactly (mean $t$ of 0.67 and 0.46, $P(t<0.1) = 2.3\%$, $P(t>0.9)$ of 4.0% and 0.83%), which also confirms that $t = 1$ is noise.

<div class="widget" data-widget="schedule"></div>

### Moving to 1024 px

The post trains 610.27M samples at 512-class sizes, then 49.97M at 1024-class sizes, with global batch 2,048 throughout. Four details of the switch:

- **Fresh optimizer, old weights.** It resumes from the 512 px EMA weights with a new optimizer.
- **Noise scale 4.** The noise scale doubles with side length, citing JiT's appendix: the post's reason is "redundancy of information" (our gloss: at higher resolution, neighboring pixels repeat each other, so the same noise is easier to see through).
- **RoPE interpolation.** A fixed position grid stays aligned to 512 px. The 1,024 px model's 32 patch indices (0 to 31) are multiplied by $1/2$, so they land on 0, 0.5, ..., 15.5, inside the range the model saw.
- **A second EMA.** An EMA of the weights has half-life 1,954 steps at 512 px (about 4M samples; per-step decay $0.5^{1/1954} \approx 0.99965$) and starts at 100M samples. At 1024 px it halves to 977 steps and starts 2M samples in, so the average has adapted before it begins. The post also pauses EMA updates for 250 steps when the gradient norm exceeds 10 times its rolling median.

The post says the 1024 px loss is higher than the 512 px loss, even as quality improves, because the noise is higher and fine detail must be exact.

### The bill

The post reports 8,139 H100-hours for pretraining and 276 for post-training, at $2.50 per hour, for 8,415 hours and $21,038, restarts included. It uses 64 H100s with bf16 and FSDP2, 32 images per GPU. Post-training used a variant of FDFO (finite-difference flow optimization) with decision models and human-in-the-loop judges, which the post says it will describe later.

<div class="widget" data-widget="budget"></div>

## Rebuild it

Our reconstruction has three small cells. They use NumPy only, and none of them reproduces the post's results.

The first cell does the shape and schedule arithmetic: patch tokens, how each readout head turns tokens back into an image, the v-loss identity, and the schedule shares.

<!-- cell: shapes_and_schedule -->

The second cell is a toy training run with our own numbers. It uses a 1-D signal of 32 values, a six-block residual trunk, and hand-written backprop. It compares a final head alone with a final head plus two pyramid readouts, using the same $t$ schedule and loss weight as above. Run it, then read the widget below.

<!-- cell: toy_pyramid -->

<div class="widget" data-widget="toy-result"></div>

The readouts do not help here. Over three seeds at 200 steps, the final-head error with readouts is within about one seed spread of the error without them, at every $t$ in the run.

Our reading of why, which is an inference: the post's claim concerns a 22-block, 2B-parameter trunk whose early blocks adapt slowly at scale, and a six-block toy trunk with 64-wide blocks has no such problem to fix. The cell does show the mechanism that the post points to: the extra heads raise the gradient norm reaching block 0 at initialization, by about 1.7x here. Whether that turns into lower loss needs a deep network and far more training. The toy neither confirms nor refutes the post's ablation.

The third cell redoes the bill from the post's numbers.

<!-- cell: budget -->

The step counts differ from the post's by 1 to 2 (297,983 against 297,985), because the post's sample counts are rounded.

## What the post doesn't say

- **Parameter counts differ between the two posts.** This post says 2.18B parameters at sampling and about 2.42B in training, with a PixelREPA head of about 241M. The Pyramid-JiT post says 2.33B in training, with a PixelREPA head of 145M. Adding each head to 2.18B gives 2.42B and 2.33B, so the posts disagree about the head, not the trunk. The posts don't say which is current.
- **No new benchmark.** This post offers a gallery of 249 images and no new numbers. The Pyramid-JiT post reports FD-DINOv2 of 27.9 for P-JiT against 79.9 for Linum v2 and 39.1 for JiT-DDT (real images against real images scores 25.4), and that P-JiT reaches v2's final score in 11.3x fewer samples. It also reports that Linum v2 still beats P-JiT on GenEval (0.855 against 0.817), mostly on counting (0.80 against 0.56); GenEval 2 and DPG-Bench are close.
- **Known limits.** The model never saw rendered text, so it can't write text or posters. The post lists struggles with full bodies, many entities, multi-person scenes, and world knowledge.
- **"$21K" is a compute bill.** The post's total covers pretraining and post-training for this release, restarts included. It doesn't mention earlier experiments, data work, or staff time.
- **Single runs.** The post's ablations and cost are single runs. Treat small gaps as noise until repeated.
- **The post's code release.** The post says an AI assistant wrote the model card and repository by extracting code from the internal repo.

## Further reading

- [Training a Text-to-Image Model for $21K](https://www.linum.ai/field-notes/pjit-21k), the post this page is about.
- [Pyramid-JiT: Low-Res Drafts, High-Res Images](https://www.linum.ai/field-notes/pyramid-jit), the earlier post that describes the method, the ablations, and the architecture appendix.
- [Code](https://github.com/Linum-AI/pjit-21k) and [weights](https://huggingface.co/Linum-AI/pjit-21k) (Apache 2.0).
- The JiT paper ([arXiv 2511.13720](https://arxiv.org/abs/2511.13720)), which both posts cite for pixel-space x-prediction and the noise-scale rule. This page doesn't cover it, so details attributed to JiT here are the posts' descriptions.
- [Latent diffusion](/ldm/), [DiT](/dit/), [Vision Transformer](/vit/), and [Flow matching](/flow-matching/) on this site.
