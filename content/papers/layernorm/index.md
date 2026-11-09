---
id: "layernorm"
title: "Layer Normalization"
authors: "Ba et al."
year: 2016
venue: "arXiv preprint"
arxiv: "1607.06450"
url: "https://arxiv.org/abs/1607.06450"
track: foundations
tier: core
est_hours: 2
status: full
summary: "Layer normalization standardizes the summed inputs of a layer using the mean and variance across the features of one example, so no example depends on the rest of the batch. A learned gain and bias per feature restore the freedom to represent any scale and offset. Because the computation is identical at training and test time and works with a batch of one, it applies to recurrent networks, variable-length sequences, and the Transformer."
related_course: "ml: normalization"
prereqs: []
background: [backprop, mean-and-variance, batch-statistics]
course_links:
  - title: "Normalisation layers: batch norm, layer norm, and their gradients"
    url: "https://ml.ifkash.dev/lessons/033-normalisation-layers"
  - title: "The transformer block: SwiGLU, residuals, and layer norm"
    url: "https://ml.ifkash.dev/lessons/050-the-transformer-block"
self_check:
  - q: 'A batch holds 2 examples with 4 features each. For batch normalization and for layer normalization, over which numbers is the mean of the first example''s first feature computed?'
    a: 'Batch normalization averages that feature over the batch: 2 numbers, one from each example (Equation 2). Layer normalization averages the first example over its 4 features (Equation 3), so the second example never enters the calculation.'
    see: batch-statistics
  - q: 'Layer normalization has $H$ features per example. How many numbers does it compute for a batch of $N$ examples, and how many learned parameters does it have?'
    a: 'It computes $N$ means and $N$ standard deviations, one pair per example. It learns $2H$ parameters: a gain $g$ and a bias $b$ for each feature, shared by all examples (Equation 5).'
    see: mean-and-variance
  - q: 'Why must the backward pass of layer normalization subtract a term from $g \odot dy / \sigma$ to get $da$?'
    a: 'The mean $\mu$ and the standard deviation $\sigma$ are functions of every input in the example, so a change to one input moves the statistics, and through them every output. Backpropagation has to add those paths. They contribute the terms $-\operatorname{mean}(d\hat{a})$ and $-\hat{a}\,\operatorname{mean}(d\hat{a} \odot \hat{a})$.'
    see: backprop
---

## The problem

Batch normalization (Ioffe and Szegedy, 2015) made deep feed-forward networks train faster by standardizing each hidden unit's summed input over the mini-batch. The paper lists what goes wrong with that when the network is recurrent, or when batches are small (Section 1 and Section 2):

- **The statistics depend on the batch.** Equation 2 takes an expectation over the training distribution, and in practice over a mini-batch. The paper notes that this constrains the mini-batch size (Section 2). The output for one example changes when its batch partners change.
- **A batch of one has no batch statistics.** Every feature has zero variance across a single example. Online learning and very large distributed models, where each machine sees a tiny batch, can't use it (Section 1).
- **Recurrent networks need different statistics at each step.** Sequences have different lengths, so batch normalization needs separate running statistics for each time step. A test sequence longer than any training sequence has no statistics at all (Section 3.1).

**Computed by `scripts/batch_dependence.py`.** Take the example $x = [1, 2, 3, 6]$ and put it in a batch of two with one of two partners. With batch normalization and no $\epsilon$, the normalized $x$ is `[-1, 1, 1, 1]` next to the first partner, and `[1, -1, -1, 1]` next to the second. The same input gets two different outputs. In a batch of one, batch normalization with $\epsilon = 10^{-5}$ returns `[0, 0, 0, 0]`, because every feature has zero variance across one example. Layer normalization returns `[-1.069, -0.5345, 0, 1.6036]` in all three cases.

## Key ideas

The sections that follow use the paper's notation (Section 2 and Section 3): $a^l$ is the vector of summed inputs to layer $l$, $H$ is the number of hidden units in the layer, $\mu^l$ and $\sigma^l$ are the normalization statistics, $g$ is the gain, $b$ is the bias, and $f$ is the element-wise nonlinearity. Where the paper writes the normalized value inline, this page names it $\hat{a}$. The paper's Equation 3 has no $\epsilon$. The code on this page, like all practical implementations, adds $\epsilon$ inside the square root; the hand examples set $\epsilon = 0$ to match the paper. Every number comes from a script in `scripts/`. Run any script from the paper folder to reproduce the output.

### Normalize across features, per example

**Intuition.** Batch normalization asks "how does this unit behave across the batch?" Layer normalization asks "how does this layer behave for this one example?" Each example is standardized against its own features, so no example needs to know about any other.

**Math.** For one example, the layer's summed inputs $a^l_i = {w^l_i}^\top h^l$ (Section 2, Equation 1) are standardized with statistics taken over the $H$ hidden units (Section 3, Equation 3):

$$
\mu^l = \frac{1}{H}\sum_{i=1}^{H} a^l_i, \qquad
\sigma^l = \sqrt{\frac{1}{H}\sum_{i=1}^{H} \left(a^l_i - \mu^l\right)^2}
$$

The paper contrasts this with Equation 2: under batch normalization each hidden unit has its own $\mu_i$ and $\sigma_i$ and the training cases share them, while under layer normalization all hidden units share one $\mu$ and $\sigma$ and each training case has its own (Section 3). Note that the variance divides by $H$, not by $H - 1$.

**Tiny example, computed by `scripts/layernorm_example.py`.** Let $H = 4$ and $a = [1, 2, 3, 6]$.

| Step | Result |
|---|---|
| $\mu = (1 + 2 + 3 + 6)/4$ | 3 |
| $\sigma^2 = (4 + 1 + 0 + 9)/4$ | 3.5 |
| $\sigma = \sqrt{3.5}$ | 1.8708 |
| $\hat{a} = (a - \mu)/\sigma$ | `[-1.069, -0.5345, 0, 1.6036]` |

Check the first entry by hand: $(1 - 3)/1.8708 = -1.069$. The result has mean 0 and variance 1. Now scale and shift the whole example: $a' = 10a + 5 = [15, 25, 35, 65]$ has $\mu' = 35$ and $\sigma' = 18.7083$, and its $\hat{a}'$ is the same four numbers. This is the data-rescaling invariance of Equation 7 (Section 5.1).

Exercise 1 asks you to implement this function with a gain, a bias, and an $\epsilon$.

### Learned gain and bias

**Intuition.** Forcing every layer's summed input to mean 0 and variance 1 would remove the network's ability to use a different scale or offset where it needs one. The learned parameters put that ability back, one pair per unit.

**Math.** After the normalization and before the nonlinearity, the paper applies a per-unit gain $g_i$ and bias $b_i$ (Section 5.1, Equation 5), the same form it uses for batch normalization and weight normalization:

$$
h_i = f\!\left(\frac{g_i}{\sigma_i}\left(a_i - \mu_i\right) + b_i\right)
$$

For a recurrent layer, Equation 4 writes the same thing in vector form with $\odot$ for the element-wise product, and $a^t = W_{hh} h^{t-1} + W_{xh} x_t$:

$$
h^t = f\!\left[\frac{g}{\sigma^t} \odot \left(a^t - \mu^t\right) + b\right]
$$

Section 6 initializes the gains to 1 and the biases to 0, so that a fresh layer normalizes and does nothing else.

**Tiny example, computed by `scripts/layernorm_example.py`.** Keep $\hat{a} = [-1.069, -0.5345, 0, 1.6036]$ and set $g = [1, 2, 1, 0.5]$ and $b = [0, 0.5, 0, -1]$. Then $g \odot \hat{a} + b = [-1.069,\ -0.569,\ 0,\ -0.1982]$. For the second entry, $2 \cdot (-0.5345) + 0.5 = -0.569$. The gain and the bias are indexed by feature, so they have $H$ entries each. They are shared by every example, and in a recurrent network by every time step (Section 3.1).

### Independent of batch size

**Intuition.** Layer normalization is a function of one example. Training and test time use the same computation, a batch of one gives the same answer as a batch of 1,000, and a recurrent network can recompute the statistics at every step.

**Math.** Section 3 states that layer normalization imposes no constraint on the mini-batch size and works with batch size 1. Section 3.1 applies it to a recurrent network: the statistics $\mu^t$ and $\sigma^t$ depend only on the summed inputs at the current time step, and there is one set of $g$ and $b$ for all time steps. A test sequence that is longer than any training sequence needs no new parameters.

**Tiny example, computed by `scripts/batch_dependence.py`.** In the problem section, the same example got three different batch-normalized outputs and one layer-normalized output. The script prints all four rows so that you can compare them. The Transformer relies on this property. It normalizes each position of each sequence on its own, which matters when an autoregressive decoder runs with one new token at a time (see [Attention Is All You Need](/transformer/), Section 3.1).

### Invariances, and what they buy

**Intuition.** Normalization throws away some information about the scale of the summed inputs. That makes the network insensitive to changes you would like it to ignore.

**Math.** Section 5.1 compares the three normalization methods in Table 1. Layer normalization is invariant to:

- re-scaling the whole weight matrix, and to adding the same shift to every incoming weight, meaning $W' = \delta W + \mathbf{1}\gamma^\top$ (Equation 6). The shift adds the same constant $\gamma^\top x$ to every $a_i$, which the mean removes, and the factor $\delta$ is divided out by $\sigma$.
- re-scaling and re-centering an individual training case, because $\mu$ and $\sigma$ depend only on the current input (Equation 7, which writes the re-scaling case).

It is not invariant to re-scaling a single row of the weight matrix. Batch normalization and weight normalization are, because those methods compute a separate scalar for each unit (Table 1). The invariance to data rescaling has a consequence for gradients: scaling $x$ by 10 scales $\partial y/\partial x$ by $1/10$. Exercise 2 tests this property.

The paper's geometric analysis (Section 5.2) argues that dividing by $\sigma$ lets the weight norm act as an implicit learning-rate control: when the weight vector grows, the Fisher information along it shrinks, so the same parameter update changes the output less. The page doesn't repeat that derivation.

### The backward pass

**Intuition.** The statistics $\mu$ and $\sigma$ depend on every input of the example. When you perturb one input, the output moves directly, and it also moves because the mean and the standard deviation moved. The backward pass must include both effects.

**Math.** The paper gives no backward formulas, so this derivation is this page's own and uses the paper's symbols. Let $\hat{a} = (a - \mu)/\sigma$ and $y = g \odot \hat{a} + b$, and let $d\hat{a} = g \odot dy$. Differentiating through $\mu$ and $\sigma$ gives, for one example,

$$
da = \frac{1}{\sigma}\Big(d\hat{a} - \operatorname{mean}(d\hat{a}) - \hat{a}\,\operatorname{mean}(d\hat{a} \odot \hat{a})\Big)
$$

where both means run over the $H$ features. The parameter gradients sum over the examples in the batch: $dg = \sum_n dy \odot \hat{a}$ and $db = \sum_n dy$. The three terms inside the parentheses are the direct path, the path through $\mu$, and the path through $\sigma$.

Two properties follow when $\epsilon = 0$. The row $da$ sums to 0, because a constant shift of $a$ doesn't change the output. It is also orthogonal to $\hat{a}$, because scaling $a$ doesn't change the output either.

**Tiny example, computed by `scripts/gradient_check.py`.** The script draws a $3 \times 4$ input and random $g$, $b$, and upstream gradient, and compares these formulas with central finite differences. The largest difference is $3.7 \times 10^{-10}$ for $da$, $2.4 \times 10^{-10}$ for $dg$, and $4.5 \times 10^{-10}$ for $db$. Rows of $da$ sum to 0 to four decimals. If you keep only the direct path, $g \odot dy/\sigma$, the largest error in $da$ is 2.9234, which is far outside the finite-difference tolerance.

Exercise 2 asks you to implement this backward pass.

### Statistics through depth

**Intuition.** In a deep random network, each layer multiplies the signal by a factor that is slightly off 1. The error compounds. Normalization resets the scale at every layer, so the compounding stops.

**Math.** The paper observes that in a standard RNN the average magnitude of the summed inputs tends to grow or shrink at every step, which leads to exploding or vanishing gradients. A layer-normalized RNN is invariant to re-scaling all of the summed inputs of a layer, which the paper credits with much more stable hidden-to-hidden dynamics (Section 3.1). A feed-forward stack shows the same effect across layers.

**Tiny example, computed by `scripts/depth_statistics.py`.** A 20-layer ReLU network has 256 units per layer and a batch of 64. Each weight comes from $N(0, s^2/256)$. The table shows the standard deviation of the summed inputs $a^l$ at layers 1, 5, 10, and 20. With layer normalization, $g = 1$ and $b = 0$, and the script applies it before the ReLU.

| Weight scale $s$ | Layer normalization | Layer 1 | Layer 5 | Layer 10 | Layer 20 |
|---|---|---|---|---|---|
| 2 | no | 2.02 | 8.55 | 51.3 | 1,280 |
| 2 | yes | 1 | 1 | 1 | 1 |
| 1 | no | 1.01 | 0.267 | 0.0501 | 0.00122 |
| 1 | yes | 1 | 1 | 1 | 1 |

For ReLU, a scale of $s = \sqrt{2}$ keeps the second moment about constant. A scale of 2 doubles the variance at each layer and a scale of 1 halves it, so the standard deviation drifts by a factor near 1.4 per layer in the two directions. After 20 layers, that gap is three orders of magnitude. Layer normalization holds the standard deviation at 1 for both scales.

Exercise 3 asks you to measure these statistics.

<!-- exercises -->

## Pitfalls

- **Normalizing over the wrong axis.** For an input of shape `(batch, seq, features)`, layer normalization reduces over the last axis only. Reducing over the batch axis gives batch normalization, and the output for each example then depends on the rest of the batch.
- **Dividing by $H - 1$.** The paper's Equation 3 divides by $H$. NumPy's `var` does that by default. PyTorch's `Tensor.var` divides by $H - 1$ unless you pass `unbiased=False`. Check the library you use.
- **Leaving out $\epsilon$, or adding it in the wrong place.** Equation 3 has none, but a constant row of activations has $\sigma = 0$. Add $\epsilon$ inside the square root, as in $\sqrt{\sigma^2 + \epsilon}$. Adding it after the root works for most inputs and fails the tests of exercise 1.
- **Forgetting the gain and the bias.** Without them, every layer is pinned to mean 0 and variance 1 whatever the task needs. The paper's analysis in Section 5 includes both parameters.
- **Dropping the correction terms in the backward pass.** The direct term $g \odot dy / \sigma$ looks like the whole gradient and passes a shape check. It misses the two paths through $\mu$ and $\sigma$, and the gradient is wrong by a large margin.
- **Summing the parameter gradients over the wrong axis.** $dg$ and $db$ sum over the batch, so they have $H$ entries. Summing over the features gives $N$ entries, and the shapes only agree when $N = H$.
- **Applying it to convolutional networks without checking.** Section 6.7 reports that layer normalization sped up a convolutional baseline in preliminary experiments, but batch normalization did better. The authors suggest that units near the image boundary, which are rarely active, have different statistics from the rest of the layer.
- **Reading the Transformer's placement into this paper.** The paper normalizes the summed inputs before the nonlinearity inside an RNN. The Transformer applies it around residual sub-layers. Neither choice follows from the other; see [Attention Is All You Need](/transformer/).
- **Expecting exact invariance with $\epsilon$.** Equations 6 and 7 hold for $\epsilon = 0$. With $\epsilon = 10^{-5}$ the outputs agree to many digits, but only when the variance is much larger than $\epsilon$.

## Further reading

- [The order-embedding code](https://github.com/ivendrov/order-embedding) and [the attentive-reader code](https://github.com/cooijmanstim/Attentive_reader/tree/bn) are the public Theano projects that the authors modified for their experiments (Section 6.1 and Section 6.2, footnotes 1 and 2).
- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167), by Ioffe and Szegedy, is the method that this paper transposes.
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467), by Zhang and Sennrich, drops the mean subtraction and keeps the rescaling. Many recent language models use it.
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745), by Xiong et al., explains why the placement of layer normalization changes how a Transformer trains.
- [Understanding and Improving Layer Normalization](https://arxiv.org/abs/1911.07013), by Xu et al., argues that the derivative of the mean and variance matters more than the forward normalization.
