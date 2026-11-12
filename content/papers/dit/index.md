---
id: "dit"
title: "Scalable Diffusion Models with Transformers"
authors: "Peebles et al."
year: 2022
venue: "ICCV 2023"
arxiv: "2212.09748"
url: "https://arxiv.org/abs/2212.09748"
track: generative
tier: core
est_hours: 4
status: full
summary: "DiT replaces the convolutional U-Net in a latent diffusion model with a plain Vision Transformer that reads the noised latent as a sequence of patch tokens. The timestep and class label condition each block through adaptive layer norm whose output gate starts at zero, so every block begins as the identity function, and this beat cross-attention and in-context conditioning. Sample quality tracks the forward-pass Gflops of the network, from more depth and width or from smaller patches, and the largest model reaches an FID of 2.27 on class-conditional ImageNet 256x256."
related_course: "ml: diffusion"
prereqs:
  - id: ldm
    reason: "DiT runs in the latent space of a frozen pretrained autoencoder, so you need to know what the 32x32x4 latent is and why diffusing there costs a fraction of pixel-space compute."
  - id: vit
    reason: "DiT is a ViT on the noised latent and reuses its patch embedding, sine-cosine position embeddings, and Small, Base, and Large configurations."
  - id: cfg
    reason: "The paper's headline FID uses classifier-free guidance, and the class label is dropped during training and replaced with a learned null embedding."
background: [adaln-conditioning, patchify]
course_links:
  - title: "Diffusion for audio"
    url: "https://speech.ifkash.dev/lessons/065-diffusion-for-audio"
  - title: "Diffusion from scratch"
    url: "https://speech.ifkash.dev/lessons/062-diffusion-from-scratch"
self_check:
  - q: 'A $512 \times 512$ image goes through the Stable Diffusion autoencoder (downsample factor 8, 4 latent channels) and DiT with patch size $p = 4$. How many tokens does the transformer see, and how long is each flattened patch before the linear embedding?'
    a: 'The latent is $64 \times 64 \times 4$. With $p = 4$ there are $(64/4)^2 = 256$ tokens, and each patch holds $4 \cdot 4 \cdot 4 = 64$ numbers. Halving $p$ would quadruple the token count.'
    see: patchify
  - q: 'In an adaLN-Zero block with hidden size $d = 384$, the layer that regresses the scale, shift, and gate vectors is a linear map from $d$ to $6d$. How many parameters does it hold per block, and what does it output at initialization?'
    a: 'It holds $384 \cdot 2304 + 2304 = 887{,}040$ parameters, counting the bias. The paper zero-initializes it, so all six vectors, including both gates $\alpha$, are zero and the block returns its input unchanged.'
    see: adaln-conditioning
  - q: 'For one latent entry, the unconditional noise prediction is $0.2$ and the conditional one is $0.5$. What does guidance with scale $s = 4$ give, and what does $s = 1$ give?'
    a: 'With $\hat\epsilon = \epsilon_\varnothing + s(\epsilon_c - \epsilon_\varnothing)$, $s = 4$ gives $0.2 + 4 \cdot 0.3 = 1.4$, and $s = 1$ gives $0.5$, which is the plain conditional prediction.'
    see: cfg
---

## The problem

By 2022, diffusion models led image generation, and every strong one used a convolutional U-Net as its backbone. Ho et al. inherited the U-Net from PixelCNN++, and Dhariwal and Nichol tuned it, but the high-level design stayed the same (Section 1). Transformers had replaced domain-specific networks in language, vision (see [Vision Transformer](/vit/)), and reinforcement learning, and they showed clean scaling there. Diffusion had not joined that trend.

That left two gaps.

- **No scaling recipe.** Papers compared U-Nets by parameter count. The paper argues that parameter count is a poor proxy for image models, because it ignores resolution (Section 2, "Architecture complexity"). Nobody had a clean rule for how sample quality grows with compute.
- **Unclear conditioning.** A transformer needs a way to take the timestep $t$ and the class $c$ as input. The obvious choices, extra tokens or cross-attention, were untested at scale for diffusion.

The paper asks whether the U-Net inductive bias is needed at all (Section 1). Its answer is no. It builds Diffusion Transformers (DiTs) on top of the latent diffusion framework of [LDM](/ldm/), measures cost in forward-pass Gflops, and finds a strong link between Gflops and FID. The largest model, DiT-XL/2 at 118.6 Gflops, reaches an FID of 2.27 on ImageNet 256x256 with classifier-free guidance (Table 2). The best earlier diffusion model, LDM-4-G, scored 3.60 at 103.6 Gflops (Section 5.1).

## Key ideas

The paper doesn't number its equations, so this page cites sections, figures, tables, and the appendix. The notation follows the paper: $I$ for the latent side length, $C$ for its channels, $p$ for the patch size, $T$ for the token count, $d$ for the hidden size, $N$ for the number of blocks, and $\gamma$, $\beta$, $\alpha$ for the regressed scale, shift, and gate. Every number comes from a script in `scripts/`. Run any script from the paper folder to reproduce its output.

### Latent patches as tokens

**Intuition.** A diffusion model denoises a spatial tensor. DiT cuts that tensor into small squares, flattens each square into a vector, and treats the vectors as a sentence. The denoiser is then an ordinary transformer, with no downsampling path, upsampling path, or skip connections.

**Math.** The input is the noised latent $z$ of shape $I \times I \times C$. For a $256 \times 256 \times 3$ image, the Stable Diffusion autoencoder has downsample factor 8, so $z$ has shape $32 \times 32 \times 4$ (Section 3.2 and Section 4, "Diffusion"). The first layer, "patchify," linearly embeds each $p \times p$ patch into a token of dimension $d$ (Section 3.2, Figure 4):

$$
T = \left(\frac{I}{p}\right)^2
$$

The paper then adds the standard ViT sine-cosine position embeddings to every token (Section 3.2). The design space includes $p = 2, 4, 8$. Halving $p$ quadruples $T$, and so at least quadruples the Gflops, but changing $p$ has no meaningful effect on the parameter count (Section 3.2). The linear embedding holds $p^2 C d$ weights, so only that one layer changes.

After the last block, the paper applies a final layer norm and a linear layer that decodes each token into a $p \times p \times 2C$ tensor, then rearranges the tokens into the original layout (Section 3.2, "Transformer decoder"). The $2C$ channels hold the predicted noise and a diagonal covariance, because the paper follows Nichol and Dhariwal and trains $\Sigma_\theta$ as well as $\epsilon_\theta$ (Section 3.1).

**Tiny example, computed by `scripts/patchify_example.py`.** Take a $4 \times 4$ latent with one channel that holds the numbers 0 to 15 in raster order, and $p = 2$:

| Token | Patch contents |
|---|---|
| 0 (top left) | `[0, 1, 4, 5]` |
| 1 (top right) | `[2, 3, 6, 7]` |
| 2 (bottom left) | `[8, 9, 12, 13]` |
| 3 (bottom right) | `[10, 11, 14, 15]` |

For the paper's $32 \times 32 \times 4$ latent, $p = 8$ gives $T = 16$, $p = 4$ gives $T = 64$, and $p = 2$ gives $T = 256$. With $d = 1152$ and $p = 2$, the embedding holds only $2 \cdot 2 \cdot 4 \cdot 1152 = 18{,}432$ weights. At $512 \times 512$, the latent is $64 \times 64 \times 4$, and $p = 2$ gives $T = 1024$ (Section 5.1).

Exercise 1 asks you to implement `patchify` and `unpatchify`, and to check that no pixels cross between patches.

### adaLN-Zero conditioning

**Intuition.** The block needs to know the timestep (how noisy) and the class (what to draw). Instead of feeding them in as tokens or through a second attention layer, DiT lets the condition rewrite the layer norm. A small network reads the condition and outputs a scale and a shift for each feature. This is cheap, because it runs on one vector and not on every token. It applies the same modulation to all tokens, which the paper notes is a restriction that cross-attention doesn't have (Section 3.2).

**Math.** The paper compares four block designs (Section 3.2, Figure 3):

- **In-context.** Append the embeddings of $t$ and $c$ as two extra tokens.
- **Cross-attention.** Add a cross-attention layer after self-attention.
- **adaLN.** Replace each layer norm with an adaptive one. It regresses $\gamma$ and $\beta$ from the sum of the $t$ and $c$ embeddings and doesn't learn them directly.
- **adaLN-Zero.** Also regress a per-feature gate $\alpha$ that multiplies each sub-layer output right before its residual connection, and initialize the regression layer to output zeros.

The paper doesn't write the block as an equation. In the paper's symbols, one half-block computes

$$
x \leftarrow x + \alpha \odot F\big(\mathrm{LN}(x) \odot (1 + \gamma) + \beta\big),
$$

where $F$ is self-attention for the first half and the pointwise MLP for the second. The $(1 + \gamma)$ form is the convention of the authors' released code, not something the paper states, and it makes $\gamma = 0$ mean "no change." The appendix gives the size of the regression: the sum of the embeddings goes through a SiLU and a linear layer with $4d$ outputs for adaLN or $6d$ for adaLN-Zero (Appendix A, "DiT model details"). The six outputs are $\gamma_1, \beta_1, \alpha_1, \gamma_2, \beta_2, \alpha_2$, as labeled in Figure 3. Both $\alpha$ vectors start at zero, so the block starts as the identity. The paper motivates this with ResNet work that found zero-initialized residual branches help (Section 3.2).

The experiment backs the design. Four DiT-XL/2 variants trained for 400K steps reach these FID-50K values without guidance (Table 4): in-context 35.24, cross-attention 26.14, adaLN 25.21, and adaLN-Zero 19.47. Cross-attention costs roughly 15% more Gflops (137.6 against 118.6), and adaLN-Zero adds negligible Gflops (Section 3.2, Section 5). The paper reports that the adaLN-Zero FID is nearly half the in-context FID, and concludes that the conditioning mechanism and the initialization both matter (Section 5).

**Tiny example, computed by `scripts/adaln_example.py`.** Take one token $x = [1, 2, 3, 4]$. Its layer norm is `[-1.3416, -0.4472, 0.4472, 1.3416]`. Let the condition produce $\gamma = [0.5, 0, -0.5, 1]$ and $\beta = [1, 1, 0, 0]$. The modulated value $\mathrm{LN}(x) \odot (1 + \gamma) + \beta$ is `[-1.0125, 0.5528, 0.2236, 2.6833]`. Check the first entry by hand: $-1.3416 \cdot 1.5 + 1 = -1.0125$.

Now the gate. Let $F(h) = 2h$ stand in for the sub-layer, and set $\gamma = \beta = 0$:

| Gate $\alpha$ | $x + \alpha F(\mathrm{LN}(x))$ |
|---|---|
| 0 | `[1, 2, 3, 4]` |
| 0.1 | `[0.7317, 1.9106, 3.0894, 4.2683]` |
| 1 | `[-1.6833, 1.1056, 3.8944, 6.6833]` |

At $\alpha = 0$ the output equals the input, whatever $F$ computes. The script also shows that a zero-initialized regression layer gives $\alpha = 0$ for any condition vector. For DiT-XL ($d = 1152$), the six-way regression holds 7,969,536 parameters per block, and 28 blocks hold 223,147,008 of the model's roughly 675M (Table 4).

Exercise 2 asks you to implement the adaLN-Zero block and check that it is the identity at zero initialization.

### Gflops predicts FID

**Intuition.** Parameter count says little about a DiT. Hold the width and depth fixed and shrink the patch size, and the parameters stay about the same while the token count quadruples and the quality improves a lot. What improves quality is the compute spent per image.

**Math.** The paper names four sizes (Table 1): DiT-S ($N = 12$, $d = 384$, 6 heads), DiT-B (12, 768, 12), DiT-L (24, 1024, 16), and DiT-XL (28, 1152, 16). It crosses them with $p \in \{8, 4, 2\}$ to get 12 models, and writes DiT-XL/2 for the XLarge config with $p = 2$ (Section 4). All 12 train with the same hyperparameters: a constant learning rate of $10^{-4}$, no weight decay, and a batch size of 256 (Section 4, "Training"). The paper plots FID-50K at 400K steps against forward-pass Gflops and reports a correlation of $-0.93$ (Section 5, Figure 8). It also estimates training compute as

$$
\text{model Gflops} \times \text{batch size} \times \text{training steps} \times 3
$$

where the factor of 3 approximates the backward pass as twice the forward pass (Section 5, "Larger DiT models are more compute-efficient"). The paper doesn't number this expression.

The paper doesn't derive its Gflops. Counting one multiply-add as one flop, one block with $T$ tokens costs $4Td^2$ for the Q, K, V, and output projections, $2T^2 d$ for $QK^\top$ and the attention-weighted sum, $8Td^2$ for the MLP with a $4d$ hidden size, and $6d^2$ for the adaLN-Zero regression on one vector. Add the patchify layer and the final decoder. This count is the page's derivation, not the paper's, and the check against Table 4 below is what justifies it.

**Tiny example, computed by `scripts/gflops.py` and `scripts/scaling_fit.py`.** The count matches Table 4 to within 1% for all 12 models. Five of them:

| Model | $T$ | Computed Gflops | Table 4 |
|---|---|---|---|
| DiT-S/4 | 64 | 1.41 | 1.41 |
| DiT-B/4 | 64 | 5.56 | 5.56 |
| DiT-L/4 | 64 | 19.69 | 19.70 |
| DiT-XL/4 | 64 | 29.04 | 29.05 |
| DiT-XL/2 | 256 | 118.62 | 118.64 |

At $512 \times 512$, where $I = 64$, the formula gives 524.5 for DiT-XL/2, against 524.6 in the paper (Section 5.1). The $T^2$ term explains why halving $p$ multiplies the cost by a little more than 4: DiT-XL/2 over DiT-XL/4 is 4.08.

With the 12 (Gflops, FID-50K) pairs from Table 4, `scripts/scaling_fit.py` finds a correlation of $-0.93$ between $\log$ Gflops and FID, which reproduces the paper's number. The correlation with raw Gflops is $-0.73$, so the $-0.93$ holds on the log axis that Figure 8 uses. A straight-line fit gives $\text{FID} \approx -54.3 \log_{10}(\text{Gflops}) + 126.1$. That fit summarizes 12 points and isn't a law from the paper. Two models with similar cost land at similar FID even though their shapes differ: DiT-S/2 (6.06 Gflops) has an FID of 68.40, and DiT-B/4 (5.56 Gflops) has 68.38 (Section 5).

Sampling compute can't make up for a small model. DiT-L/2 with 1000 sampling steps costs 80.7 Tflops per image, and DiT-XL/2 with 128 steps costs $118.64 \cdot 128 / 1000 = 15.2$ Tflops. The larger model still has the better FID-10K, 23.7 against 25.9 (Section 5.2, Figure 10).

Exercise 3 asks you to count Gflops from the config table.

### Guidance and the final result

**Intuition.** DiT trains with the class label dropped at random and replaced by a learned null embedding $\varnothing$. At sampling time, the model runs twice and extrapolates from the unconditional prediction toward the conditional one (see [Classifier-free guidance](/cfg/)).

**Math.** The paper restates the rule in Section 3.1 (the formula isn't numbered):

$$
\hat\epsilon_\theta(x_t, c) = \epsilon_\theta(x_t, \varnothing) + s \cdot \big(\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \varnothing)\big), \qquad s > 1
$$

Here $s = 1$ recovers standard sampling. The authors applied guidance to only the first three of the four latent channels, because three-channel and four-channel guidance give similar FID once the scale is adjusted (Appendix A, "Classifier-free guidance on a subset of channels"). Exercise 4 uses a simplified version of this: the guided channels come from the formula, and every other output channel, including the covariance half, copies the conditional pass.

**Tiny example, computed by `scripts/cfg_example.py`.** Take unconditional noise `[0.2, -0.4, 1.0, 0.5]` and conditional noise `[0.5, -0.1, 0.6, 0.9]`. Then $s = 1$ returns the conditional values, $s = 1.5$ gives `[0.65, 0.05, 0.4, 1.1]`, and $s = 4$ gives `[1.4, 0.8, -0.6, 2.1]`. Check the first entry for $s = 4$: $0.2 + 4 \cdot (0.5 - 0.2) = 1.4$.

On ImageNet 256x256, DiT-XL/2 trained for 7M steps gets an FID-50K of 9.62 without guidance and 2.27 with $s = 1.5$ (Table 2). At 512x512, trained for 3M steps, it gets 3.04, against 3.85 for the best earlier diffusion model, ADM-G with ADM-U (Table 3).

Exercise 4 asks you to implement guidance on a subset of channels.

<!-- exercises -->

## Pitfalls

- **Looking for numbered equations.** The paper has none. If another source cites an equation number for DiT, check it against the text.
- **Reshaping without the grid transpose.** `z.reshape(T, p*p*C)` runs without an error and mixes pixels from different patches. Reshape to `(g, p, g, p, C)`, transpose the grid axes to the front, then flatten.
- **Forgetting the `1 +` in the modulation.** With $\mathrm{LN}(x) \odot \gamma + \beta$, a zero-initialized $\gamma$ would zero the sub-layer input. The released code uses $(1 + \gamma)$, and that is what makes the zero-initialized layer harmless.
- **Zeroing the wrong layer.** The zero initialization applies to the layer that regresses the modulation (and so the gates $\alpha$) and to the final linear layer (Section 3.2, Section 4, "Training"). Zeroing the attention or MLP weights themselves would block gradients to those layers.
- **Gating the sum and not the branch.** The gate multiplies $F(\cdot)$ before the residual add. Gating $x + F(\cdot)$ discards $x$ at $\alpha = 0$, and the block stops being the identity.
- **Comparing models by parameters.** DiT-XL/2 and DiT-XL/4 have the same 675M parameters but a 4x difference in Gflops and a large FID gap, 19.47 against 43.01 (Table 4).
- **Charging the adaLN layer per token.** The regression sees one conditioning vector per image, so it costs $6d^2$ and not $6Td^2$. The mistake is small, but it breaks the 1% check against Table 4.
- **Quoting the wrong FID.** The scaling plots use FID-50K at 400K steps without guidance. The headline 2.27 is at 7M steps with $s = 1.5$ and the ft-EMA VAE decoder (Table 2, Table 5). Section 5.2 uses FID-10K, which isn't comparable.
- **Expecting adaLN to supply position.** Patchify drops the patch layout. The sine-cosine position embeddings restore it, and the condition doesn't.
- **Guiding the covariance.** Guidance extrapolates the noise prediction. The learned covariance channels aren't extrapolated.

## Further reading

- [facebookresearch/DiT](https://github.com/facebookresearch/DiT) is the official PyTorch release. Its block, patchify, and sampling code are the source of the conventions that this page flags as the code's and not the paper's.
- [DiT project page](https://www.wpeebles.com/DiT) has sample images and links to the code.
- [Denoising diffusion probabilistic models](/ddpm/) covers the training objective that DiT keeps unchanged.
- [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206), by Esser et al., is a later paper that scales a DiT-style backbone for text-to-image generation.
