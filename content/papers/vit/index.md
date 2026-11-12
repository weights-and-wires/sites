---
id: "vit"
title: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
authors: "Dosovitskiy et al."
year: 2020
venue: "ICLR 2021"
arxiv: "2010.11929"
url: "https://arxiv.org/abs/2010.11929"
track: vision
tier: core
est_hours: 4
status: full
summary: "The Vision Transformer cuts an image into fixed-size patches, projects each patch linearly to a vector, and feeds the sequence to a standard Transformer encoder, with a learnable class token for classification and learned position embeddings. It has almost no built-in knowledge of images, so it trails ResNets when trained on ImageNet alone. Pre-trained on 14M to 300M images, it matches or beats them while using about 2 to 4 times less pre-training compute."
related_course: "ml: vision Transformers"
prereqs:
  - id: transformer
    reason: "ViT reuses the Transformer encoder (multi-head self-attention plus an MLP, each in a residual connection) unchanged except for where LayerNorm sits, so you need to know what one block computes."
background: [patch-embedding, image-classification]
course_links:
  - title: "Scaled attention, many heads, and the causal mask"
    url: "https://ml.ifkash.dev/lessons/059-scaled-attention-and-heads"
  - title: "Convolution: slide a filter, count the weights, count the view"
    url: "https://ml.ifkash.dev/lessons/046-convolution-and-receptive-field"
self_check:
  - q: 'A $224 \times 224$ RGB image is cut into $16 \times 16$ patches. How many patches $N$ are there, how many numbers does each flattened patch hold, and how long is the Transformer input sequence once a class token is added?'
    a: '$N = 224 \cdot 224 / 16^2 = 196$ patches. Each holds $16 \cdot 16 \cdot 3 = 768$ numbers. The class token makes the sequence length $N + 1 = 197$.'
    see: patch-embedding
  - q: 'Self-attention builds an $n \times n$ score matrix per head. How many score entries is that if every pixel of a $224 \times 224$ image is a token, and how many if every $16 \times 16$ patch is a token (plus a class token)?'
    a: 'With pixels, $n = 50{,}176$ and the matrix has $50{,}176^2 = 2{,}517{,}630{,}976$ entries. With patches, $n = 197$ and it has $38{,}809$ entries, about 65,000 times fewer. The cost grows with the square of the sequence length.'
    see: transformer
  - q: 'A linear classification head maps a $D = 768$ feature vector to $K = 1000$ class scores. How many weights and biases does it have?'
    a: '$768 \cdot 1000 + 1000 = 769{,}000$. Each of the $K$ classes has its own weight vector over the $D$ features, plus one bias.'
    see: image-classification
---

## The problem

By 2020, Transformers dominated language. The standard recipe was to pre-train a large model on a huge text corpus and fine-tune it on a small task, and the paper notes there was no sign of saturating performance as models passed 100B parameters (Section 1). Computer vision still ran on convolutional networks. Earlier attempts to use self-attention on images either combined it with convolutions or replaced them with specialized attention patterns, and those patterns had not scaled well on accelerators (Section 1, Section 2).

The obvious approach fails on cost. Take a $224 \times 224$ image and let every pixel attend to every other pixel (Section 2 makes this point). The sequence has $224 \cdot 224 = 50{,}176$ tokens, so one attention head builds a score matrix with $50{,}176^2 = 2{,}517{,}630{,}976$ entries. That is about 2.5 billion numbers per head, per layer, per image. Computed by `scripts/patch_counts.py`.

The paper asks a plainer question: what if you take a standard Transformer with as few changes as possible and feed it patches in place of words (Section 1)? On ImageNet alone, the answer is mildly disappointing. The models land a few percentage points below ResNets of similar size, because a Transformer lacks "translation equivariance and locality" and has to learn them from data (Section 1). The paper's claim is that this stops mattering at scale: "large scale training trumps inductive bias". Pre-trained on ImageNet-21k (14M images) or JFT-300M, ViT-H/14 reaches 88.55% on ImageNet (Section 1, Table 2).

## Key ideas

This page follows the paper's notation: image $x \in \mathbb{R}^{H \times W \times C}$, patch size $P$, patch count $N$, patch embedding matrix $E$, position embeddings $E_{pos}$, hidden size $D$, layer count $L$, and $z_\ell$ for the sequence after $\ell$ blocks. The paper writes the attention weights of its MSA operator in Appendix A. This page uses the Transformer paper's $W^Q, W^K, W^V, W^O$ for them instead, and the [Transformer page](/transformer/) explains them. Every number comes from a script in `scripts/`.

### Split the image into patches and embed them linearly

**Intuition.** A Transformer reads a sequence of vectors. An image is a grid, so cut the grid into non-overlapping squares, unroll each square into one long vector, and treat each vector like a word. Pixels would give far too many tokens, and patches give a few hundred.

**Math.** Section 3.1 reshapes the image into a sequence of flattened patches $x_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$, where $N = HW/P^2$. A trainable matrix $E \in \mathbb{R}^{(P^2 \cdot C) \times D}$ maps each patch to $D$ dimensions (Equation 1). The paper calls the result the *patch embeddings*. There is no bias term in the equation, and $D$ stays constant through every layer.

**Tiny example, computed by `scripts/patch_counts.py` and `scripts/embed_example.py`.** For a $224 \times 224$ RGB image:

| Patch size $P$ | $N$ | Sequence length $N+1$ |
|---|---|---|
| 32 | 49 | 50 |
| 16 | 196 | 197 |
| 14 | 256 | 257 |

With $P = 16$ the flattened patch has $16 \cdot 16 \cdot 3 = 768$ numbers, and for ViT-B/16 $D = 768$ as well, so $E$ is $768 \times 768$ with 589,824 weights. The two 768s are a coincidence of the Base configuration: ViT-L/16 has $D = 1024$, and ViT-H/14 has a patch width of $14 \cdot 14 \cdot 3 = 588$ and $D = 1280$. The paper notes that the sequence length shrinks with the *square* of the patch size, so smaller patches are much more expensive (Section 4.1).

Now a case you can check by hand. Take a $4 \times 4$ one-channel image holding $0, 1, \dots, 15$ row by row, with $P = 2$. The four patches in raster order, each flattened row by row, are:

| Patch | Pixels |
|---|---|
| 1 | `[0, 1, 4, 5]` |
| 2 | `[2, 3, 6, 7]` |
| 3 | `[8, 9, 12, 13]` |
| 4 | `[10, 11, 14, 15]` |

A plain `reshape(4, 4)` on the same array gives `[0, 1, 2, 3]` as the first row instead. That row holds a whole image row and no square. The transpose in `reshape(H//P, P, W//P, P, C).transpose(0, 2, 1, 3, 4)` is what groups the pixels into squares. Exercise 1 asks you to write `patchify` and its exact inverse.

The linear map has a second reading. Applying $E$ to non-overlapping patches is the same as a convolution with a $P \times P$ kernel, stride $P$, and $D$ output channels. The paper doesn't state this, but the script checks it numerically: the largest difference between the two computations on random data is about $10^{-15}$. Many implementations use a convolution for speed.

### The class token and learned position embeddings

**Intuition.** Classification needs one vector for the whole image. Following BERT, the paper prepends one extra learnable vector to the sequence and lets attention write the answer into it. Attention is also order-blind, so each token gets a learned vector that tells it where its patch came from.

**Math.** Section 3.1 defines the input to the encoder (Equation 1):

$$
z_0 = [x_{\text{class}};\; x_p^1 E;\; x_p^2 E;\; \cdots;\; x_p^N E] + E_{pos}, \qquad E \in \mathbb{R}^{(P^2 \cdot C) \times D},\; E_{pos} \in \mathbb{R}^{(N+1) \times D}
$$

The semicolons stack rows. The class token $x_{\text{class}} \in \mathbb{R}^{D}$ is row 0, and $E_{pos}$ has $N + 1$ rows because the class token gets a position embedding too. The paper uses "standard learnable 1D position embeddings": patches are numbered in raster order, and nothing in the table encodes the 2D grid. It reports that more advanced 2D-aware embeddings gave no significant gain (Section 3.1, Appendix D.4).

**Tiny example, computed by `scripts/embed_example.py`.** Continue the $4 \times 4$ image with $D = 2$ and $E = \begin{bmatrix} 1 & 0 \\ 1 & 0 \\ 0 & 1 \\ 0 & 1 \end{bmatrix}$, which sums the top row of each patch into the first coordinate and the bottom row into the second. The patch embeddings are:

| Patch | $x_p^i E$ |
|---|---|
| 1 | `[1, 9]` |
| 2 | `[5, 13]` |
| 3 | `[17, 25]` |
| 4 | `[21, 29]` |

Check patch 1 by hand: the top row is $0 + 1 = 1$ and the bottom row is $4 + 5 = 9$. With $x_{\text{class}} = [0.5, -0.5]$ and a made-up $E_{pos}$ whose rows are `[0.1, 0.1]`, `[0, 0]`, `[1, 0]`, `[0, 1]`, `[-1, -1]`, the sequence $z_0$ has shape $(5, 2)$ and its first rows are `[0.6, -0.4]` and `[1, 9]`. Exercise 2 asks you to build $z_0$.

How much does the position table matter? Appendix D.4 (Table 8) reports ImageNet 5-shot accuracy for ViT-B/16: 0.614 with no position embeddings and 0.642 with 1D embeddings. Treating the patches as a bag costs about 2.8 points. The 1D, 2D, and relative variants are within about 0.3 points of each other.

Without position embeddings the model cannot tell patch order. `scripts/permutation_example.py` builds a two-block encoder with random weights and shuffles six patches. With no position embeddings, the class-token output changes by at most $1.4 \times 10^{-14}$, which is round-off. With position embeddings, the change is $12.5$. Exercise 3 includes this test.

### Use a standard Transformer encoder with pre-LN

**Intuition.** After Equation 1, the sequence goes through the same kind of block you met in the [Transformer paper](/transformer/): attention mixes information across tokens, an MLP transforms each token, and a residual connection wraps each step. ViT keeps that design and moves the LayerNorm to the front of each step.

**Math.** Section 3.1 states the encoder as $L$ alternating blocks (Equations 2 and 3), and then the image representation (Equation 4):

$$
z'_\ell = \operatorname{MSA}(\operatorname{LN}(z_{\ell-1})) + z_{\ell-1}, \qquad \ell = 1 \dots L
$$

$$
z_\ell = \operatorname{MLP}(\operatorname{LN}(z'_\ell)) + z'_\ell, \qquad \ell = 1 \dots L
$$

$$
y = \operatorname{LN}(z_L^0)
$$

The superscript 0 selects row 0, the class token. The paper says "Layernorm (LN) is applied before every block, and residual connections after every block", and the MLP has two layers with a GELU non-linearity. The original Transformer used post-LN, $\operatorname{LayerNorm}(x + \operatorname{Sublayer}(x))$ (Section 3.1 of that paper), so this is a real difference. In pre-LN, the residual stream $z$ is never normalized inside the stack. An input that is large stays large along the identity path, which is why a block with all-zero weights returns its input unchanged. Exercise 3 uses that property as a test.

At pre-training time, a classification head with one hidden layer reads $y$. At fine-tuning time, the paper replaces it with a single linear layer that is zero-initialized and has shape $D \times K$ for $K$ downstream classes (Section 3.1, Section 3.2). Appendix D.3 reports that average-pooling the patch tokens in place of using a class token works about as well, with a different learning rate (Figure 9). The class token is a design inherited from NLP, not a requirement.

**Tiny example, computed by `scripts/param_count.py`.** Table 1 lists three variants:

| Model | Layers $L$ | Hidden size $D$ | MLP size | Heads | Params in Table 1 | Counted here |
|---|---|---|---|---|---|---|
| ViT-Base | 12 | 768 | 3072 | 12 | 86M | 85.8M |
| ViT-Large | 24 | 1024 | 4096 | 16 | 307M | 303.3M |
| ViT-Huge | 32 | 1280 | 5120 | 16 | 632M | 630.8M |

The count adds the patch projection $E$, the class token, $E_{pos}$ for $224 \times 224$ input, per block four attention matrices, two MLP matrices, two LayerNorms and all biases, and the final LayerNorm. It leaves out the classification head. For ViT-B/16 one block holds 7,087,872 weights and the 12 blocks hold 85,054,464 of the 85,797,888 total. The Large count falls 1.2% short of Table 1 and the Huge count 0.2% short, and the table doesn't say what it includes, so treat the gap as unexplained. A $K = 1000$ head would add 769,000 weights for ViT-B.

### Higher-resolution fine-tuning

**Intuition.** Fine-tuning at a larger image size often helps. The paper keeps the patch size fixed, so a bigger image means more patches and a longer sequence. The weights don't depend on length, but the position table does, so it has to be resized.

**Math.** Section 3.2 says the pre-trained position embeddings "may no longer be meaningful" at the new length, so the paper performs "2D interpolation of the pre-trained position embeddings, according to their location in the original image." It doesn't give the interpolation formula. This page and Exercise 4 use bilinear interpolation with aligned corners. The class-token row is not part of the grid. The paper also notes that this resizing and the patch extraction are the only points where 2D image structure is injected into the model by hand (Section 3.2).

**Tiny example, computed by `scripts/interpolate_example.py`.** Resize the $2 \times 2$ grid $\begin{bmatrix} 0 & 1 \\ 2 & 3 \end{bmatrix}$ to $3 \times 3$. New index $i$ reads source coordinate $i(g-1)/(g_{new}-1) = i/2$, so the middle index reads 0.5. The result is $\begin{bmatrix} 0 & 0.5 & 1 \\ 1 & 1.5 & 2 \\ 2 & 2.5 & 3 \end{bmatrix}$. The corners keep their values, and the middle row is the average of the two rows. For ViT-B/16, fine-tuning at 384 changes the grid from $14 \times 14$ to $24 \times 24$, so the table grows from 197 to 577 rows (151,296 to 443,136 parameters, and 85,797,888 to 86,089,728 in total). Exercise 4 asks you to write the resize.

### Scale beats inductive bias

**Intuition.** A CNN starts with strong assumptions about images: nearby pixels matter together, and a pattern means the same thing wherever it appears. Those assumptions help when data is scarce. A ViT must learn them. With enough data, learning them is better than assuming them.

**Math.** There is no new formula. The evidence comes from two experiments (Section 4.3). In the first, models are pre-trained on ImageNet (1.3M images), ImageNet-21k (14M), and JFT-300M (303M) and then transferred to ImageNet. With the smallest set, ViT-Large underperforms ViT-Base, and both fall below BiT ResNets. Only with JFT-300M do the larger ViT models pull ahead (Figure 3). In the second, models train on random JFT subsets of 9M, 30M, 90M, and 300M images, and ViT-B/32 does "much worse" than ResNet50 on the 9M subset but better on 90M and above (Figure 4).

**What to check, from the paper's tables.** The numbers below come from the paper, not from a script on this page:

| Model | Pre-training data | ImageNet top-1 | TPUv3-core-days |
|---|---|---|---|
| ViT-H/14 | JFT-300M | 88.55 | 2.5k |
| ViT-L/16 | JFT-300M | 87.76 | 0.68k |
| ViT-L/16 | ImageNet-21k | 85.30 | 0.23k |
| BiT-L (ResNet152x4) | JFT-300M | 87.54 | 9.9k |
| Noisy Student (EfficientNet-L2) | ImageNet plus unlabeled JFT | 88.4 or 88.5 | 12.3k |

(Table 2.) ViT-L/16 beats BiT-L on ImageNet at roughly 1/15 of the compute: $9.9\text{k} / 0.68\text{k} \approx 14.6$. The paper cautions that pre-training cost also depends on the schedule, optimizer, and weight decay, so Section 4.4 runs a controlled comparison on JFT-300M and finds that ViT uses approximately 2 to 4 times less compute than ResNets for the same average accuracy over five datasets (Figure 5). It also finds that hybrids (a ResNet stem feeding a ViT) edge out pure ViT at small budgets and that the gap disappears for larger models.

The recipe is Adam with $\beta_1 = 0.9$ and $\beta_2 = 0.999$, batch size 4096, and a weight decay of 0.1, which the authors found useful for transfer (Section 4.1).

<!-- exercises -->

## Pitfalls

- **Reshaping without the transpose.** `img.reshape(N, P*P*C)` runs and returns the right shape, but each row holds pixels from a strip of the image and not from a square. Group the axes first, as in Exercise 1.
- **Wrong flatten order inside a patch.** Any fixed order works if the model is trained with it, but pretrained weights assume a specific one. Check the order against the checkpoint you load.
- **Forgetting that the class token has a position embedding.** $E_{pos}$ has $N + 1$ rows, not $N$. Off-by-one shapes are a common error when you load a checkpoint.
- **Reading ViT-B/16 as "16 patches".** In `ViT-L/16`, the 16 is the patch side in pixels. It's not the number of patches, and not the number of heads. Smaller $P$ means a longer sequence.
- **Using post-LN because the Transformer paper does.** ViT is pre-LN (Equations 2 and 3). A post-LN stack behaves differently and needs different learning-rate handling.
- **Reusing position embeddings at a new resolution without resizing.** A $14 \times 14$ table has no rows for a $24 \times 24$ grid. Resize the grid, not the class row, and keep the patch size fixed (Section 3.2).
- **Judging ViT from a small-data run.** On ImageNet alone, the paper's own ViT models trail ResNets. The claim holds only with large-scale pre-training (Section 4.3).
- **Expecting the position table to start with 2D structure.** At initialization the position embeddings carry no information about 2D position. The trained model learns that nearby patches get similar embeddings (Section 3.1, Section 4.5, Figure 7).
- **Skipping the learning-rate change when swapping the class token for pooling.** Appendix D.3 says the two heads work similarly well only with different learning rates.

## Further reading

- [vision_transformer](https://github.com/google-research/vision_transformer) is the official code and pre-trained models, linked from the paper's first page.
- [Training data-efficient image transformers and distillation through attention](https://arxiv.org/abs/2012.12877), by Touvron et al., trains ViT on ImageNet alone with heavy augmentation and distillation.
- [Better plain ViT baselines for ImageNet-1k](https://arxiv.org/abs/2205.01580), by Beyer et al., shows how a plainer recipe closes much of the small-data gap.
- [Scalable Diffusion Models with Transformers](/dit/) on this site applies the same patch-and-Transformer design to image generation.
