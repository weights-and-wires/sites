---
id: "clip"
title: "Learning Transferable Visual Models From Natural Language Supervision"
authors: "Radford et al."
year: 2021
venue: "ICML 2021"
arxiv: "2103.00020"
url: "https://arxiv.org/abs/2103.00020"
track: vision
tier: core
est_hours: 3
status: full
summary: "CLIP trains an image encoder and a text encoder together on 400 million image-text pairs from the internet. For a batch of N pairs, it pushes the N matching image and text embeddings together and the N squared minus N mismatched pairs apart, with a symmetric cross-entropy loss over the scaled cosine similarities. At test time, the text encoder turns class names or prompts into a linear classifier, so the model classifies images into categories it never saw labeled, and zero-shot CLIP matches a supervised ResNet-50 on ImageNet."
related_course: "ml: multimodal"
prereqs:
  - id: vit
    reason: "The strongest CLIP image encoders are Vision Transformers, so you need to know how an image becomes a sequence of patch tokens and how one output vector represents the image."
  - id: transformer
    reason: "The text encoder is a Transformer with masked self-attention, and the activation at its end-of-text token becomes the sentence embedding."
background: [contrastive-learning, infonce]
course_links:
  - title: "Self-supervised speech"
    url: "https://speech.ifkash.dev/lessons/036-self-supervised-speech"
  - title: "Attention as a soft lookup, with a dot product or a learned scorer"
    url: "https://ml.ifkash.dev/lessons/045-attention-as-soft-lookup"
self_check:
  - q: 'A batch holds $N = 4$ aligned (image, text) pairs. How many positive pairs and how many negative pairs does the contrastive loss see, and what loss does a model that scores every pair equally get?'
    a: 'There are 4 positive pairs (the diagonal) and $N^2 - N = 12$ negative pairs. If every logit is equal, each softmax is uniform over 4 entries, so each cross-entropy term is $\ln 4 \approx 1.3863$ and so is the average.'
    see: contrastive-learning
  - q: 'The CLIP image encoder ViT-L/14 reads a $224 \times 224$ image. How many patch tokens does it get, and how many after the paper''s extra pre-training epoch at $336 \times 336$?'
    a: 'Patch size 14 gives $224 / 14 = 16$, so $16 \times 16 = 256$ patches at 224 pixels. At 336 pixels it is $24 \times 24 = 576$ patches. Each patch is flattened and projected linearly, and a class token adds one more position.'
    see: vit
  - q: 'CLIP''s text encoder uses masked (causal) self-attention and reads its feature from the activation at the [EOS] token. Why that position?'
    a: 'With a causal mask, position $i$ attends only to positions $\le i$. The last token is the only position that has attended to the whole text, so its activation can summarize the full sentence.'
    see: transformer
---

## The problem

In 2021, the best image classifiers learned from a fixed list of labels. A ResNet or a ViT trained on ImageNet picks one of 1000 classes through a softmax head, and it has no way to answer a question about a class outside that list. Extending it means collecting and labeling new images, and a single majority-vote gold label per image is expensive to produce at scale (Section 2.1). Natural language is a richer supervision signal that already exists on the internet, because people write text next to pictures (Section 2.1).

Earlier work had tried to learn from image-text pairs, and the results were weak at scale. Visual N-Grams reached 11.5% zero-shot accuracy on ImageNet (Section 1; Table 1). The labeled alternatives limited their label sets: Mahajan et al. trained to predict 1000 ImageNet-related hashtag classes, and Kolesnikov et al. predicted 18,291 JFT classes. Both used a static softmax classifier with a fixed output layer (Section 1).

Predicting the caption from the image is the obvious fix, and it trains slowly. Section 2.3 reports that a 63M-parameter Transformer language model that predicts the exact words of the caption learns to recognize ImageNet classes three times more slowly than a bag-of-words baseline (Figure 2). The reason is that the task is too hard and too specific. An image of a dog can have thousands of valid captions, and the model has to commit to the exact words.

The paper's move is to make the task easier. Don't predict which words describe an image. Predict only which text, as a whole, goes with which image. Section 2.3 reports that this swap gives another 4x efficiency gain on zero-shot ImageNet transfer (Figure 2). With 400 million pairs and a batch of 32,768, the resulting model matches a ResNet-50 on ImageNet without using any of its 1.28 million labeled training images. Table 1 compares the zero-shot ImageNet accuracy against Visual N-Grams:

| Dataset | Visual N-Grams | CLIP |
|---|---|---|
| aYahoo | 72.4 | 98.4 |
| ImageNet | 11.5 | 76.2 |
| SUN | 23.0 | 58.5 |

The paper cautions that this comparison isn't controlled for dataset size, compute, or architecture (Section 3.1.3).

## Key ideas

This page uses the paper's names from the Figure 3 pseudocode: image features `I_f`, text features `T_f`, learned projections `W_i` and `W_t`, normalized embeddings `I_e` and `T_e` of width `d_e`, batch size `n`, and a learned `t`. The paper has no numbered equations, so the citations below point to sections and to Figure 3. Section 2.5 writes the temperature as $\tau$ and Figure 3 writes the logit scale as $\exp(t)$, so $\exp(t) = 1/\tau$. The page writes $s = \exp(t)$ for the scale. Every number comes from a script in `scripts/`. Run any script from the paper folder to reproduce the output.

### The symmetric contrastive loss

**Intuition.** Put $n$ images and their $n$ captions in a batch. Score every image against every caption, which gives an $n \times n$ table. The $n$ cells on the diagonal are true pairs, and the other $n^2 - n$ cells are mismatches that you get for free. Training pushes each diagonal score up against the rest of its row, and against the rest of its column.

**Math.** Each encoder's features go through a learned linear projection into a joint space, and both are normalized to unit length (Figure 3):

$$
I_e = \frac{I_f W_i}{\lVert I_f W_i \rVert}, \qquad T_e = \frac{T_f W_t}{\lVert T_f W_t \rVert}
$$

The logit matrix is the table of scaled cosine similarities:

$$
L = e^{t}\, I_e T_e^{\top} \in \mathbb{R}^{n \times n}
$$

Figure 3 computes two cross-entropies, `loss_i` along `axis=0` and `loss_t` along `axis=1`, and averages them. The order doesn't matter in the average. Written for one pair $k$, with $L_{ij}$ scoring image $i$ against text $j$, the row term comes first:

$$
\mathcal{L} = \frac{1}{2n} \sum_{k=1}^{n}\left[ -\log \frac{e^{L_{kk}}}{\sum_{j} e^{L_{kj}}} \;-\; \log \frac{e^{L_{kk}}}{\sum_{i} e^{L_{ik}}} \right]
$$

The first term classifies which of the $n$ texts belongs to image $k$, and the second classifies which of the $n$ images belongs to text $k$. The paper cites the multi-class N-pair loss (Sohn, 2016) as the origin of this construction and says the InfoNCE loss of Oord et al. (2018) popularized it (Section 2.3). The paper describes the goal as maximizing the cosine similarity of the $n$ real pairs and minimizing it for the $n^2 - n$ incorrect pairings (Section 2.3). The cross-entropy form does that through a softmax, so the incorrect pairings are weighed by how confusable they are.

**Tiny example, computed by `scripts/infonce_example.py`.** Take $n = 2$ with $I_e = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$ and $T_e = \begin{bmatrix} 1 & 0 \\ 0.6 & 0.8 \end{bmatrix}$, so text 2 leans toward image 1. The cosine similarities are $\begin{bmatrix} 1 & 0.6 \\ 0 & 0.8 \end{bmatrix}$. With scale $s = 10$, the logits are $\begin{bmatrix} 10 & 6 \\ 0 & 8 \end{bmatrix}$.

| Direction | Diagonal probabilities | Mean loss |
|---|---|---|
| Image to text (softmax along each row) | 0.98201, 0.99966 | 0.00924 |
| Text to image (softmax along each column) | 0.99995, 0.88080 | 0.06349 |

The symmetric loss is $(0.00924 + 0.06349)/2 = 0.03636$. The two directions disagree because the table isn't symmetric. Text 2 has a high similarity with image 1 (0.6), and the column for text 2 is the one that pays for it. A loss with one direction misses this.

Two reference points help when you debug. With orthonormal embeddings, every diagonal logit is $s$ and every other is 0, so the loss is $\ln(1 + (n-1)e^{-s})$. The script prints 0.74367 for $n = 4$ and $s = 1$, and 1.27401 for $n = 8$ and $s = 1$. With $s = 0$ every logit is 0 and the loss is $\ln n$, which is 1.3863 for $n = 4$.

Exercise 1 asks you to implement this loss.

### The temperature is learned

**Intuition.** Cosine similarities lie in $[-1, 1]$. A softmax over numbers that small is nearly uniform, so the loss barely distinguishes right from wrong. The scale $s$ stretches the scores. A large $s$ makes the softmax sharp, and a small $s$ makes it flat.

**Math.** The paper doesn't tune the temperature as a hyperparameter. It optimizes it as a log-parameterized multiplicative scalar (Section 2.3), which is the `t` in Figure 3. Section 2.5 initializes $\tau$ to the equivalent of 0.07 and clips it so the logits are never scaled by more than 100, which it found necessary to prevent unstable training. With $s = 1/\tau$, the initial scale is $1/0.07 = 14.2857$, so $t = \ln(14.2857) = 2.6593$, and the clip is $t \le \ln 100 = 4.6052$. Because $L$ is proportional to $e^t$, the gradient is $\partial \mathcal{L} / \partial t = \sum_{ij} G_{ij} L_{ij}$, where $G = \partial \mathcal{L} / \partial L$. This is the `dt` of Exercise 3.

**Tiny example, computed by `scripts/temperature_example.py`.** A seeded batch of 4 noisy pairs has diagonal cosines 0.808, 0.774, 0.711, 0.660 and a largest off-diagonal 0.413.

| Scale $s$ | Loss | Mean largest probability per image |
|---|---|---|
| 1 | 0.93044 | 0.39617 |
| 14.29 (the initial value) | 0.00483 | 0.99719 |
| 100 (the clip) | 0.00000 | 1.00000 |

At scale 0 the loss would be $\ln 4 = 1.3863$. The loss falls because the same similarities become more decisive, and that is why the scale is part of what the optimizer learns: the model can raise $s$ as its embeddings become more separable. A small batch of random embeddings can show a loss well above $\ln n$ at the initial scale, because 14.29 magnifies random mistakes. Exercise 3 starts at a loss of 5.2232 for $n = 8$, where $\ln 8 = 2.0794$.

### Batch size sets the number of negatives

**Intuition.** Every other pair in the batch is a negative, so a bigger batch gives each pair more negatives to beat. The cost is the $n \times n$ table, which grows with the square of the batch.

**Math.** There are $n$ positives and $n^2 - n$ negatives per batch (Section 2.3). The paper trains with a minibatch of 32,768 (Section 2.5) and shards the similarity computation, so each GPU computes only the subset of pairwise similarities that its local batch needs. The paper also uses gradient checkpointing, mixed precision, and half-precision Adam statistics to save memory (Section 2.5).

**Tiny example, computed by `scripts/scale_numbers.py`.** At $n = 32{,}768$ the batch has 1,073,709,056 negative pairs, and the table has 1,073,741,824 entries. In float32 that is 4.0 GiB for one table. Sharded over 256 GPUs, each holds a $128 \times 32768$ block of 4,194,304 entries. The count of 256 is illustrative and isn't claimed by the paper for this step. Training for 32 epochs on 400 million pairs sees 12.8 billion images, which is 390,625 optimizer steps. At one image per second, that takes 405.6 years, which matches the 405 years in Section 6. Section 3.1.2 offers a way to read the loss: every pre-training step is like optimizing a classifier for a random dataset with 1 example per class and 32,768 classes defined by language.

### Zero-shot classification by prompt

**Intuition.** After pre-training, the text encoder maps any sentence to a point near the images it describes. To classify an image, write one sentence per class, embed each, and pick the class whose sentence lands closest to the image. The class list can change at test time with no retraining.

**Math.** Section 3.1.2 computes the image embedding and the embedding of each class's text, takes the cosine similarities, scales them by the temperature, and applies a softmax. The paper points out that this is a multinomial logistic regression classifier with L2-normalized inputs, L2-normalized weights, no bias, and temperature scaling. The text encoder acts as a hypernetwork that generates the classifier weights: row $c$ of the weight matrix is the normalized embedding of the text for class $c$ (Section 3.1.2). For class texts $T_e^{(1)}, \dots, T_e^{(C)}$ and an image embedding $I_e$:

$$
p(c \mid \text{image}) = \operatorname{softmax}_c\!\left( e^{t}\, I_e \cdot T_e^{(c)} \right)
$$

The paper caches the classifier after the first computation, so the cost of the text encoder is amortized over every image in the dataset (Section 3.1.2).

The class name alone is a weak prompt. Section 3.1.4 gives two reasons. A single word is rare in the pre-training text, because most captions are full sentences. A name also has several senses, such as "crane" the machine and the bird, or "boxer" the dog and the athlete. The default template "A photo of a {label}." gives a 1.3% gain on ImageNet. Per-task templates can help further, for example "A photo of a {label}, a type of pet." on Oxford-IIIT Pets.

**Tiny example, computed by `scripts/zero_shot_example.py`.** The exercise setting has two classes in a 2-D embedding space, so each embedding is an angle. The image sits at 30 degrees, and class 0 is correct. Prompt A puts class 0 at 60 degrees and class 1 at 5 degrees. At scale 100 it gives class probabilities [0.0175, 0.9825], which is wrong. Prompt B puts class 0 at 20 degrees and class 1 at 100 degrees, and gives [1.0, 0.0].

### Prompt ensembling in embedding space

**Intuition.** Different prompts see different parts of a class: "a big {label}" and "a small {label}" are two views. Averaging them smooths out a single prompt's quirks. Averaging the text embeddings is cheaper than averaging the predictions.

**Math.** Section 3.1.4 builds the ensemble over the embedding space and not over probabilities. This lets the paper cache a single set of averaged text embeddings, so the cost matches one classifier once it is amortized. On ImageNet it ensembles 80 context prompts for another 3.5% over the single default prompt, and prompt engineering plus ensembling together improve ImageNet accuracy by almost 5% (Figure 4 shows the effect across models). The paper doesn't say how the average is normalized. This page and Exercise 2 normalize each prompt's embedding, average, and normalize again, because the cosine similarity assumes unit-length class vectors. This is a choice made here.

**Tiny example, computed by `scripts/zero_shot_example.py`.** Average prompt A and prompt B from above. The ensemble puts class 0 at 40 degrees and class 1 at 52.5 degrees. The image at 30 degrees has cosines 0.9848 and 0.9239 with them, so the probabilities at scale 100 are [0.9977, 0.0023]: the ensemble recovers the right class. Averaging the probability vectors instead gives [0.5087, 0.4913], which is nearly a tie. Before the second normalization, the averaged embeddings have lengths 0.9397 and 0.6756, and the shorter one comes from the two prompts that disagree more. The script also counts the one-time cost: 1000 classes with 80 prompts take 80,000 text-encoder calls, and each image afterward takes one image-encoder pass and 1000 dot products.

Exercise 2 asks you to build the classifier and the ensemble.

### The architecture and the linear projection

**Intuition.** CLIP doesn't invent a new network. It reuses a vision backbone and a text Transformer and connects them with one linear layer each.

**Math.** Section 2.4 describes two families of image encoder: modified ResNet-50 variants with attention pooling in place of global average pooling, and Vision Transformers that follow the ViT paper with an extra layer normalization after the combined patch and position embeddings. The text encoder is a Transformer with the modifications of GPT-2: 12 layers, 512 wide, 8 heads, 63M parameters, a lowercased BPE vocabulary of 49,152, and a maximum sequence length of 76. The sequence is wrapped in [SOS] and [EOS] tokens. The text uses masked self-attention, so it could be initialized from a pre-trained language model or take a language-modeling objective later, and the feature is the activation of the top layer at [EOS], layer-normalized and linearly projected (Section 2.4). The paper trains 5 ResNets and 3 Vision Transformers (ViT-B/32, ViT-B/16, ViT-L/14) for 32 epochs (Section 2.5), and "CLIP" in the results means ViT-L/14@336px, which gets one additional epoch at 336 pixels. The ViT-L/14 took 12 days on 256 V100 GPUs, and the largest ResNet, RN50x64, took 18 days on 592 V100 GPUs.

Section 2.3 also removes things that other contrastive work uses. CLIP trains from scratch, with no ImageNet or pre-trained text weights. It uses a linear projection and no non-linear projection head, because the authors saw no difference in efficiency. The only augmentation is a random square crop. The dataset, WIT, holds 400 million (image, text) pairs found by searching for 500,000 queries with up to 20,000 pairs per query to balance the classes (Section 2.2).

**Tiny example, computed by `scripts/scale_numbers.py`.** The patch counts for the ViT encoders at 224 pixels are 49 tokens for ViT-B/32, 196 for ViT-B/16, and 256 for ViT-L/14. ViT-L/14@336px has $24 \times 24 = 576$ patches. Add one class token and the sequences are 50, 197, 257, and 577 long.

### Alignment with linear encoders

**Intuition.** The paper's code has no backward pass, because the training framework supplies one. Deriving the gradient once makes it clear what the loss does: it moves each embedding toward its partner and away from the confusable ones, only along the sphere's surface.

**Math.** Let $P^{r}$ be the softmax of $L$ along rows, $P^{c}$ the softmax along columns, and $I$ the identity. The gradient of $\mathcal{L}$ with respect to the logits is $G = (P^{r} + P^{c} - 2I)/(2n)$. Then $\partial \mathcal{L}/\partial I_e = s\, G\, T_e$ and $\partial \mathcal{L}/\partial T_e = s\, G^{\top} I_e$. Normalization contributes a projection: for $e = f/\lVert f \rVert$, $\partial \mathcal{L}/\partial f = \left(g - e\,(e \cdot g)\right)/\lVert f \rVert$ with $g = \partial \mathcal{L}/\partial e$. The step removes the part of the gradient that points along $e$. The loss doesn't depend on a feature's length, so the gradient has no component along it. This derivation is mine, and it is not in the paper.

**Tiny example, computed by `scripts/align_toy.py`.** Two views of the same 3-D vectors (8 pairs, 6 and 5 raw features each) go through linear encoders into 4 dimensions, at scale 14.2857 with plain gradient descent at rate 1.0.

| Step | Loss | Image-to-text accuracy | Text-to-image accuracy |
|---|---|---|---|
| 0 | 5.2232 | 0.000 | 0.125 |
| 100 | 0.3771 | 1.000 | 0.750 |
| 500 | 0.1716 | 1.000 | 1.000 |
| 1500 | 0.1498 | 1.000 | 1.000 |

Exercise 3 asks you to write `loss_and_grads` and checks it against finite differences.

### What zero-shot buys, and what it doesn't

**Intuition.** A zero-shot classifier is strong where language describes the category, and weak where the pre-training data has no such text.

**What to check, from the paper.** These numbers come from the paper and not from a script on this page:

- Zero-shot CLIP beats a fully supervised linear classifier on ResNet-50 features on 16 of 27 datasets, including ImageNet (Section 3.1.5, Figure 5). It trails by more than 10 points on EuroSAT, KITTI Distance, PatchCamelyon, GTSRB, CLEVRCounts, DTD, Flowers102, RESISC45, and FGVCAircraft.
- Zero-shot CLIP reduces the gap between ImageNet accuracy and accuracy under natural distribution shift by up to 75% (Section 3.3, Figure 13). A linear classifier fit on ImageNet features raises ImageNet accuracy by 9.2% to 85.4% and leaves the average accuracy under shift slightly lower (Section 3.3, Figure 14).
- The paper estimates that zero-shot CLIP needs about 1000x more compute to reach overall state of the art, and calls that infeasible with current hardware (Section 6).
- On handwritten MNIST digits, zero-shot CLIP reaches 88%, and logistic regression on raw pixels does better. The paper reads this as brittle generalization: images unlike the pre-training data stay out of distribution (Section 6).

<!-- exercises -->

## Pitfalls

- **Forgetting to normalize.** Without the L2 normalization the logits are raw dot products, and the model can lower the loss by growing the features. The temperature then has no stable meaning, and the zero-shot classifier no longer measures cosine similarity.
- **Treating `t` as the scale.** The scale is $\exp(t)$ (Figure 3). At the paper's initial value $t = 2.6593$, using $t$ directly gives a scale of 2.66 in place of 14.29, so the softmax stays flat.
- **Mixing up the axes.** The two terms normalize over different axes of the same table. If both use the same axis, you train a one-directional loss, which the symmetric loss is meant to avoid.
- **Summing in place of averaging.** The loss is a mean over the batch. A sum multiplies the loss and its gradient by $n$, so the best learning rate changes with the batch size.
- **Overflow in the softmax.** With a scale up to 100, logits reach 100, and $e^{100} \approx 2.7 \times 10^{43}$ overflows float32, which tops out near $3.4 \times 10^{38}$. Subtract the row maximum, or use a log-sum-exp.
- **Forgetting that duplicates are false negatives.** If two pairs in a batch share a caption, or two captions describe the same image, the loss still treats the off-diagonal cell as a negative. The sections of the paper cited on this page don't address this.
- **Using the class name as the whole prompt.** A bare label is rare in the training text and can be ambiguous. Use a template such as "A photo of a {label}." and ensemble several (Section 3.1.4).
- **Ensembling probabilities and not embeddings.** Averaging probability vectors needs one softmax per prompt for every image, and you cannot merge the prompts into one cached classifier. Averaging normalized embeddings leaves one classifier to cache (Section 3.1.4).
- **Reading zero-shot as "never saw the images".** The paper's sense of zero-shot is generalization to unseen datasets, not unseen categories (Section 3.1.1). Pre-training data can overlap with an evaluation set, and Section 5 measures this. It finds the overall accuracy change from overlap is small, at most 0.6% on Birdsnap (Figure 17).
- **Expecting generation.** CLIP picks from the concepts you supply. It doesn't write captions, and the paper says so in Section 6.

## Further reading

- [openai/CLIP](https://github.com/OpenAI/CLIP) is the official code and pre-trained weights, linked from the paper's abstract.
- [Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748), by Oord et al., introduces the InfoNCE loss that the paper cites in Section 2.3.
- [Contrastive Learning of Medical Visual Representations from Paired Images and Text](https://arxiv.org/abs/2010.00747), by Zhang et al., is ConVIRT, the paper that CLIP simplifies.
- [An Image is Worth 16x16 Words](/vit/) and [Attention Is All You Need](/transformer/) on this site cover the two encoders CLIP combines.
