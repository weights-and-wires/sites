---
id: "muon"
title: "Muon: An optimizer for hidden layers in neural networks"
authors: "Jordan et al."
year: 2024
venue: "Blog post (Keller Jordan)"
arxiv: null
url: "https://kellerjordan.github.io/posts/muon/"
track: training-optimization
tier: core
est_hours: 3
status: full
summary: "Muon takes the momentum update for each hidden weight matrix and replaces it with the nearest matrix whose singular values are all one, so every direction in the update gets the same step size. It computes that matrix with five steps of a tuned Newton-Schulz iteration, which uses only matrix multiplications and adds about one percent to the training FLOPs in the post's two examples. Embeddings, the output head, and all scalar and vector parameters stay on AdamW, and the post reports speed records on CIFAR-10 and NanoGPT training with this setup."
related_course: "ml: optimizers"
prereqs:
  - id: adamw
    reason: "Muon handles only the hidden matrices, so the embeddings, the output head, and every scalar or vector parameter still need the AdamW update from that page."
background: [svd, spectral-norm, newton-schulz-iteration]
course_links:
  - title: "Lost digits, and how QR and the SVD keep them"
    url: "https://ml.ifkash.dev/lessons/004-stable-least-squares"
  - title: "SGD, Adam, and learning-rate schedules"
    url: "https://ml.ifkash.dev/lessons/031-sgd-adam-and-schedules"
self_check:
  - q: 'A matrix $G = U S V^\top$ has singular values $(3, 1, 0.1)$. What is $\mathrm{Ortho}(G)$, and what are its singular values?'
    a: 'It is $U V^\top$, the nearest matrix with orthonormal rows or columns to $G$ in Frobenius norm. Its three singular values are all $1$, so the direction that carried $3$ and the direction that carried $0.1$ now have the same size.'
    see: svd
  - q: 'Newton-Schulz needs every singular value of its input in $[0, 1]$. Why does the post divide by $\lVert G\rVert_F$ and not compute the largest singular value?'
    a: 'The Frobenius norm is the square root of the sum of squared singular values, so it is at least the spectral norm. Dividing by it puts every singular value in $[0, 1]$ without an SVD. It is also benign, because $\mathrm{Ortho}(cG) = \mathrm{Ortho}(G)$ for any $c > 0$.'
    see: spectral-norm
  - q: 'A transformer has a $50257 \times 768$ token embedding, twelve blocks of $768 \times 768$ and $768 \times 3072$ matrices, and LayerNorm weights. Which of these does Muon update, and which does AdamW update?'
    a: 'Muon updates only the block matrices. AdamW updates the embedding, the output head, and the LayerNorm weights and biases, because Muon applies to 2D hidden-layer parameters and the post finds AdamW better for the input and output layers ("Empirical considerations").'
    see: adamw
---

## The problem

Adam and SGD with momentum treat a weight matrix as a bag of independent numbers. Adam rescales each entry by its own gradient history, and SGD does not rescale at all. Neither uses the fact that the matrix is a linear map with rows, columns, and singular directions. The post ("Why is it good to orthogonalize the update?") reports an empirical observation on that point. Inspecting the updates from SGD-momentum and Adam for the 2D parameters of transformers by hand, the author finds that they have a very high condition number, so they are almost low-rank and a few directions dominate every neuron's update. The post speculates, and does not prove, that a more even update helps the "rare directions" that are small in the update but important for learning.

Here is a concrete version, computed by `scripts/update_geometry.py`. Take a $6 \times 4$ update built as one strong direction plus a little noise. Its singular values are $(8.87, 0.31, 0.22, 0.05)$, so the condition number is $190$, and the top direction carries $99.81\%$ of the squared Frobenius norm. A step along this update moves the weights almost entirely along one direction. The other three directions get a fraction of a percent of the step. That is the situation the post describes, built by hand and not measured on a trained model, so it shows the mechanism and doesn't confirm that real updates look like this.

The obvious fix is to even out the singular values, and the exact way to do that is an SVD of every update at every step. The post rules this out ("Eliminating alternatives to NS iteration") because the SVD is "far too slow". The alternative that Shampoo implementations use, a coupled Newton iteration, must run in at least float32 to stay stable, which the post says makes it slow on modern GPUs. Muon's contribution is an orthogonalization that is cheap and stable in bfloat16.

This is a blog post, not a conference paper, and it differs from the other pages on this site in two ways. It has no section or equation numbers, so this page cites its section headings, such as "Definition" and "Runtime analysis". Its algorithm box is an image that is labeled Algorithm 2, and this page cites it by that label. The post also gives many of its results as links to logs and figures instead of tables, and this page repeats them as the author's reports. It does not reproduce them.

## Key ideas

The notation follows the post. $G_t$ is the gradient of the loss with respect to a weight matrix, $B_t$ is the momentum buffer, $O_t$ is the orthogonalized update, $\eta$ is the learning rate, and $\mu$ is the momentum coefficient. The post's Algorithm 2 writes the parameter as $\theta$, and its Shampoo section writes $W$. This page writes $W$ for one matrix, and so do the exercises, which name the variables `W`, `G`, `B`, `lr`, and `mu`. The post writes the singular value decomposition as $G = USV^\top$, and so does this page. Every number comes from a script in `scripts/`. Run any script from `scripts/` to reproduce it.

### Orthogonalize the momentum update

**Intuition.** Run momentum as usual and keep a buffer $B_t$ for each weight matrix. Before the step, replace $B_t$ with its closest orthogonal matrix. That keeps the directions of $B_t$ and throws away how much of each one there was. Every singular direction then moves the weights by the same amount.

**Math.** Algorithm 2 ("Definition") is short. It sets $B_0 \leftarrow 0$ and then loops:

$$
\begin{aligned}
G_t &\leftarrow \nabla_\theta \mathcal{L}_t(\theta_{t-1}) \\
B_t &\leftarrow \mu B_{t-1} + G_t \\
O_t &\leftarrow \mathrm{NewtonSchulz5}(B_t) \\
\theta_t &\leftarrow \theta_{t-1} - \eta\, O_t
\end{aligned}
$$

These are lines 3 to 6 of the box. The function the loop approximates is defined in "The design of Muon", where the equation is not numbered:

$$
\mathrm{Ortho}(G) = \arg\min_O \{ \lVert O - G \rVert_F : O^\top O = I \ \text{or}\ O O^\top = I \}
$$

The post states that this is the same as replacing $G$ with $UV^\top$, where $G = USV^\top$. The singular values all become $1$ and the singular vectors stay. For a matrix with more rows than columns the minimizer has orthonormal columns ($O^\top O = I$), and for a wide matrix it has orthonormal rows. Because $\mathrm{Ortho}(cG) = \mathrm{Ortho}(G)$ for $c > 0$, the size of the gradient doesn't matter, and only $\eta$ sets the step size.

The post gives a theoretical motivation only by reference. It points to Bernstein and Newhouse (2024), "Old optimizer, new norm: An anthology", and to its own Shampoo section, and its list of contributors mentions Bernstein's work on steepest descent under the spectral norm as a closely related algorithm. The Shampoo derivation in "Relationship to prior optimizers" is short and easy to check. If you drop Shampoo's accumulation, its update is $(GG^\top)^{-1/4}\,G\,(G^\top G)^{-1/4}$. Substituting $G = USV^\top$ gives $US^{-1/2}U^\top \cdot USV^\top \cdot VS^{-1/2}V^\top = UV^\top$. The script confirms this on a random $5 \times 3$ matrix, with a largest entry difference of $3.3 \times 10^{-16}$ from the SVD-based $UV^\top$. The post therefore reads Muon with momentum off as a kind of accumulation-free Shampoo.

**Tiny example, computed by `scripts/muon_step_example.py`.** Take a $2 \times 2$ weight matrix, $\eta = 0.1$, $\mu = 0.9$, $W = 0$, and two gradients $G_1 = \mathrm{diag}(3, 1)$ and $G_2 = \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}$. This script uses an exact SVD for $\mathrm{Ortho}$, so that you can check the numbers by hand.

- Step 1. $B_1 = G_1$ has singular values $3$ and $1$. Its $UV^\top$ is the identity, so $W_1 = -0.1 I$.
- Step 2. $B_2 = 0.9\,G_1 + G_2 = \begin{pmatrix} 2.7 & 1 \\ -1 & 0.9 \end{pmatrix}$, with singular values $(2.9591, 1.1591)$. Its orthogonalization is the rotation $\begin{pmatrix} 0.8742 & 0.4856 \\ -0.4856 & 0.8742 \end{pmatrix}$, whose singular values are $(1, 1)$. So $W_2 = \begin{pmatrix} -0.1874 & -0.0486 \\ 0.0486 & -0.1874 \end{pmatrix}$.

The order of momentum and orthogonalization matters. The post describes Orthogonal-SGDM (Tuddenham et al., 2022) as the variant that orthogonalizes the gradient first and applies momentum afterward ("Relationship to prior optimizers"). On the same two gradients it ends at $W_2 = \begin{pmatrix} -0.19 & -0.1 \\ 0.1 & -0.19 \end{pmatrix}$, which is $0.0514$ away from Muon's result in the largest entry. Its step direction at step 2 has singular values $(1.3454, 1.3454)$, so it no longer has unit singular values, because a sum of two orthogonal matrices is not orthogonal. The post says that putting momentum before the orthogonalization performs better empirically. Exercise 2 asks you to write this step and checks the order.

### Newton-Schulz iteration approximates the orthogonalization

**Intuition.** Computing an SVD is slow and awkward on a GPU, but a matrix polynomial is only matrix multiplications. A polynomial in $GG^\top$ times $G$ acts on each singular value on its own, and the singular vectors pass through unchanged. If you pick the polynomial so that repeated application pushes every number in $[0, 1]$ toward $1$, the result is close to $UV^\top$.

**Math.** One step with coefficients $(a, b, c)$ ("Proving that NS iteration orthogonalizes the update") gives:

$$
G' = aG + b\,(GG^\top)G + c\,(GG^\top)^2 G = U\,(aS + bS^3 + cS^5)\,V^\top
$$

The post defines the quintic $\varphi(x) = ax + bx^3 + cx^5$. Then $N$ steps give $U\varphi^N(S)V^\top$, where $\varphi^N(S)$ applies $\varphi$ $N$ times to each diagonal entry. The post says two things are needed for convergence to $\mathrm{Ortho}(G)$. First, every singular value must start in $[0, 1]$, which the post achieves by replacing $G$ with $G / \lVert G \rVert_F$. Second, $\varphi^N(x) \to 1$ as $N \to \infty$ for all $x \in [0, 1]$. The baseline $(a, b, c) = (2, -1.5, 0.5)$ meets the second condition (the plot labeled "Baseline coefficients for Newton-Schulz iteration" in the same section).

The post then tunes the coefficients ("Tuning the coefficients"). It wants $a$ as large as possible, because $\varphi'(0) = a$ sets how fast small singular values grow. It accepts a limit of $\varphi^N(x)$ anywhere in $[0.7, 1.3]$, since $\varepsilon$ up to about $0.3$ didn't harm the loss curves in its runs. An "ad-hoc gradient based approach" gives $(a, b, c) = (3.4445, -4.7750, 2.0315)$, and five steps suffice. The tuned iteration does not converge to $1$. It oscillates in a band.

**Tiny example, computed by `scripts/ns_example.py`.** First, scalars. With the tuned coefficients, $\varphi(1) = 3.4445 - 4.7750 + 2.0315 = 0.7010$, which is the bottom of the band, and the baseline has $\varphi(1) = 1$. Here are five steps for several starting values $x_0$:

| $x_0$ | Tuned, after 5 steps | Baseline $(2, -1.5, 0.5)$, after 5 steps |
|---|---|---|
| 1.0 | 0.6964 | 1.0000 |
| 0.5 | 0.7654 | 1.0000 |
| 0.1 | 0.7121 | 1.0005 |
| 0.01 | 0.6989 | 0.312 |
| 0.001 | 0.4705 | 0.032 |

For $x_0 = 0.01$, the tuned path is $0.01, 0.0344, 0.1184, 0.4001, 1.0931, 0.6989$, so it rises fast and then bounces. The baseline needs many more steps for small starting values. This is the tradeoff behind the tuning. The value in the last row also shows a limit that the post does not discuss. After normalization, a singular value near $10^{-3}$ is still only $0.47$ after five steps, so very small directions are lifted but not fully.

Second, a matrix. Build a $4 \times 3$ matrix with singular values $(3, 1, 0.1)$. After the Frobenius normalization they are $(0.9482, 0.3161, 0.0316)$. After five tuned steps they are $(1.1340, 1.1336, 0.7530)$, all inside the post's $[0.7, 1.3]$ band, though no longer equal. The largest entry of the difference from the exact $UV^\top$ is $0.1752$. With the baseline coefficients and 25 steps the difference is $1.25 \times 10^{-15}$. Scaling $G$ by $100$ changes the result by at most $5.1 \times 10^{-9}$. The last number is the eps-sized effect of normalizing by $\lVert G \rVert_F + \epsilon$, with the post's $\epsilon = 10^{-7}$.

**Implementation details from the code in "Definition".** The `newtonschulz5` function casts to bfloat16, divides by the norm plus `eps`, and transposes when the matrix has more rows than columns, so that $A = XX^\top$ is the smaller Gram matrix. The post says bfloat16 is stable for Newton-Schulz and not for the coupled Newton iteration. The exercises run in float64 so the tests can compare against `np.linalg.svd`. Exercise 1 asks you to implement this function.

### The cost is small

**Intuition.** The iteration is a handful of matrix multiplications on a matrix the same size as the weights. A training step already multiplies by the batch of activations, which is much larger, so the extra work is a small fraction.

**Math.** For an $n \times m$ parameter with $m \le n$, one step of the iteration needs $2(2nm^2 + m^3)$ matmul FLOPs, which is at most $6nm^2$ for a square matrix ("Runtime analysis"). Counting the three products is a quick check: $XX^\top$ costs $2nm^2$, $A \cdot A$ costs $2m^3$, and $BX$ costs $2nm^2$, which add up to $4nm^2 + 2m^3$. Over $T$ steps the extra cost is at most $6Tnm^2$. A linear layer's forward and backward pass costs $6nmB$ for $B$ inputs, so the overhead is at most $Tm/B$.

**Tiny example, computed by `scripts/cost_and_split.py`.** The post gives two cases, and the script reproduces both.

- NanoGPT speedrunning, with $m = 768$ and $B = 524288$ tokens per batch: $5 \cdot 768 / 524288 = 0.73\%$. The post rounds this to 0.7%.
- Llama 405B, with $m = 16384$ and $B = 16{,}000{,}000$: $0.51\%$, which the post rounds to 0.5%.

For one square $768 \times 768$ matrix, five steps cost $13{,}589{,}544{,}960$ FLOPs, against $1{,}855{,}425{,}871{,}872$ for the forward and backward pass at the speedrun batch size, a ratio of $0.73\%$. For a tall $3072 \times 768$ matrix, the count is $40{,}768{,}634{,}880$ FLOPs, below the $6Tnm^2$ bound of $54{,}358{,}179{,}840$. The ratio to the baseline step is $0.55\%$.

This is a FLOP count and not a time. The same post says that Muon has a slower per-step wallclock time than AdamW ("Discussion"), and its open questions include whether the iterations can be distributed across a large GPU cluster.

### Muon for hidden matrices, AdamW for the rest

**Intuition.** The orthogonalization only makes sense for a matrix that acts as a linear map between hidden activations. A vector has no singular directions, and the embedding and the output head sit at the edges of the network.

**Math.** There is no equation here, but the rule is explicit ("Definition" and "Empirical considerations"). Muon is for the 2D parameters of hidden layers. Scalar and vector parameters, and the input and output layers, use AdamW. The post says the embedding part follows from the modular norm theory (Large et al., 2024), and that the output layer part "does not seem to follow from the theory, and is instead driven by empirics". A 4D convolutional filter can use Muon after its last three dimensions are flattened. Two more empirical findings are in the same section. Nesterov-style momentum works a bit better than plain momentum in every case the author tested, and it is the default in the public implementation. Muon also works better for a transformer if it is applied separately to the Q, K, and V matrices, and not to a joint QKV matrix. The post does not write the Nesterov update as a formula, so check the implementation for the exact form.

**Tiny example, computed by `scripts/cost_and_split.py`.** Take a GPT-2-small-shaped model with $d = 768$, 12 blocks, a vocabulary of 50257, a context of 1024, separate Q, K, and V matrices, no linear biases, and an untied output head. This is a configuration chosen for this page, not the speedrun's model.

| Group | Parameters | Optimizer |
|---|---|---|
| Hidden matrices (Q, K, V, output, two MLP matrices in each block) | 84,934,656 | Muon |
| Token and position embeddings | 39,383,808 | AdamW |
| Output head | 38,597,376 | AdamW |
| LayerNorm weights and biases | 38,400 | AdamW |

Only $52.1\%$ of the $162{,}954{,}240$ parameters go to Muon here, because the embedding and an untied head are large for a vocabulary this size. The optimizer for those tables is therefore a real part of the recipe. Exercise 3 asks you to write the routing rule, and exercise 4 trains a toy network with the split.

**What the post reports.** All of these are the author's results, with logs linked from the post and no independent replication on this page. Muon improved the speed record for 94% accuracy on CIFAR-10 from 3.3 to 2.6 A100-seconds, and the record for 3.28 validation loss on FineWeb (NanoGPT speedrunning) by a factor of 1.35. It kept helping at 774M and 1.5B parameters, and trained a 1.5B transformer to GPT-2 XL-level HellaSwag in 10 8xH100-hours against 13.3 hours with AdamW. The post says both optimizers were tuned.

The post's final section argues for judging optimizers on competitive tasks, where the baseline is the previous record and a worse optimizer can be dropped. It also lists open questions: scaling past 20B parameters, distributing the iteration, and whether Muon helps in fine-tuning and reinforcement learning.

<!-- exercises -->

## Pitfalls

- **Treating Muon as a replacement for AdamW on the whole model.** The post's recipe is a split. Applying Muon to the embedding, the output head, or a bias vector goes against it, and for vectors the operation has no meaning. Exercise 3 tests the routing.
- **Expecting exact orthogonality.** The tuned coefficients give singular values in about $[0.7, 1.13]$ after five steps for inputs that aren't tiny, in the examples here (the post targets $[0.7, 1.3]$), and they never settle at $1$. If you need an exactly orthogonal matrix, use the SVD or the baseline coefficients with many steps.
- **Normalizing by the wrong norm.** Dividing by the largest entry, or not dividing, can leave singular values above $1$, and the iteration then diverges. The Frobenius norm is at least the spectral norm, so it is safe. Exercise 1 includes a test with a matrix whose largest entry is small but whose spectral norm is large.
- **Orthogonalizing before the momentum.** That is Orthogonal-SGDM, and the post says it performs worse. The accumulated direction is also no longer a unit-singular-value matrix. Exercise 2 tests the order.
- **Using $(1 - \mu)$ in the buffer.** Algorithm 2 uses $B_t = \mu B_{t-1} + G_t$, with no $(1-\mu)$ factor. Because Muon discards the buffer's scale, the missing factor changes nothing in the direction at the first step, but it does change the weighting of past gradients against the new one after that. Check which form your library uses.
- **Forgetting the transpose in the implementation.** For a tall matrix, $XX^\top$ is $n \times n$ and the cost is no longer the $4nm^2 + 2m^3$ above. The result is mathematically the same, so the tests can't catch it, but it is slow.
- **Flattening a convolution the wrong way.** The post flattens the last three dimensions, so a filter bank of shape $(\text{out}, \text{in}, k_h, k_w)$ becomes $(\text{out}, \text{in} \cdot k_h k_w)$. Merging the first two dimensions orthogonalizes a different matrix.
- **Reading small singular values as fixed.** In the scalar table, $x_0 = 0.001$ reaches $0.47$ in five steps. Directions that are tiny relative to the Frobenius norm are amplified, but not equalized. The toy network in exercise 4 shows it: the first Muon step on the $32 \times 32$ hidden matrix has top singular value $1.201$ (divided by `lr_muon`) and bottom $0.022$, because some ReLU units are inactive and their rows of the gradient are almost zero. Orthogonalization can't create a direction the gradient doesn't have.
- **Reading the toy comparison as evidence.** `scripts/toy_training.py` trains the exercise network with untuned learning rates. AdamW leads at step 20 (loss $0.0273$ against $0.0353$), Muon leads at step 60 ($3.2 \times 10^{-3}$ against $4.1 \times 10^{-3}$), and AdamW is far ahead at step 200 ($1.5 \times 10^{-5}$ against $1.5 \times 10^{-3}$). Neither optimizer was tuned, and the model has three small matrices. The post's own argument is that an undertuned baseline makes optimizer comparisons unreliable ("Discussion").
- **Assuming the post's results are peer-reviewed.** The results come from a blog post and linked logs. The strongest evidence it cites is the speedrun record history, where AdamW could have won back the record with better tuning.
- **Copying details the post doesn't give.** The post's algorithm has no weight decay and no update-scale correction, and the post does not give a learning rate or momentum value in the text. Later work on scaling Muon to larger models, which this page doesn't cover, adds decoupled weight decay as in [AdamW](/adamw/) and a rescaling of the update. Check the implementation you use, and don't assume the algorithm box is complete.

## Further reading

- [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/) is the post. It links a ready-to-use PyTorch implementation and the NanoGPT speedrun record that uses it, and the code lives in the [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) repository.
- [Old Optimizer, New Norm: An Anthology](https://arxiv.org/abs/2409.20325), by Bernstein and Newhouse, is the theory the post builds on. Its Appendix A, which the post cites, lists methods for orthogonalization.
- [Shampoo: Preconditioned Stochastic Tensor Optimization](https://arxiv.org/abs/1802.09568), by Gupta, Koren, and Singer, is the second-order method that Muon reduces to when accumulation is dropped.
- [Scalable Optimization in the Modular Norm](https://arxiv.org/abs/2405.14813), by Large et al., is the theory the post cites for treating the embedding layer differently.
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) is the AdamW paper, and the prerequisite page on this site is [AdamW](/adamw/). The Adam update that AdamW modifies is on [Adam](/adam/).
- [Orthogonalising gradients to speed up neural network optimisation](https://arxiv.org/abs/2202.07052), by Tuddenham, Prügel-Bennett, and Hare, is the Orthogonal-SGDM method that the post contrasts with Muon.
