---
id: "mixed-precision"
title: "Mixed Precision Training"
authors: "Micikevicius et al."
year: 2017
venue: "ICLR 2018"
arxiv: "1710.03740"
url: "https://arxiv.org/abs/1710.03740"
track: training-optimization
tier: core
est_hours: 2
status: full
summary: "Mixed precision training stores weights, activations, and gradients in 16-bit floating point, which nearly halves memory and speeds up arithmetic on GPUs with fast half-precision math. Three techniques keep accuracy at the FP32 level without new hyperparameters: an FP32 master copy of the weights that receives the updates, loss scaling that shifts small gradients into the FP16 range, and FP16 products that accumulate in FP32. The paper matches FP32 accuracy on image classification, detection, speech recognition, translation, language modeling, and GAN training."
related_course: "gpu: numerics"
prereqs: []
background: [ieee-754-floating-point, underflow-and-overflow]
course_links:
  - title: "Training kernels"
    url: "https://gpu.ifkash.dev/c/19-training-kernels"
  - title: "Floating point on the GPU"
    url: "https://gpu.ifkash.dev/c/09-floating-point-on-the-gpu"
self_check:
  - q: 'What are the smallest positive subnormal and the largest finite values of FP16? What does `np.float16(1e-8)` return?'
    a: 'The smallest subnormal is $2^{-24} \approx 5.96 \times 10^{-8}$, and the largest finite value is 65,504. The value $10^{-8}$ is below $2^{-25}$, which is half of the smallest subnormal, so it rounds to 0.'
    see: underflow-and-overflow
  - q: 'FP16 has 10 mantissa bits. Why does `1.0 + 2**-11` evaluate to `1.0` in FP16, and what does that say about adding a small update to a large weight?'
    a: 'Near 1.0 the spacing between FP16 numbers is $2^{-10}$. The value $1 + 2^{-11}$ is exactly halfway between 1 and $1 + 2^{-10}$, and round to nearest even picks 1. Any update smaller than half the spacing of the weight vanishes, so a weight that is more than about 2,048 times larger than its update never moves.'
    see: ieee-754-floating-point
  - q: 'A float32 gradient of $10^{-8}$ is stored in FP16 and becomes 0. Multiply by $S = 1024$ before the cast and divide after it. What do you get back, and why is the result not exactly $10^{-8}$?'
    a: 'You get about $1.001 \times 10^{-8}$. The product $1.024 \times 10^{-5}$ is a subnormal in FP16 with a spacing of $2^{-24}$, so it rounds to $1.0252 \times 10^{-5}$, and dividing by 1024 gives the slightly different value. Scaling rescues the value from 0 but does not give it full precision.'
    see: underflow-and-overflow
---

## The problem

The speech models in the paper's introduction grew from 11 million parameters in 2014 to 116 million in 2016 (Section 1). Training them in single precision (FP32) is limited by memory capacity, memory bandwidth, and arithmetic throughput. A 16-bit format halves the storage and the bandwidth, and the paper notes that half-precision arithmetic on recent GPUs runs 2 to 8 times faster than single precision (Section 1).

Naively switching a network to FP16 fails, and it fails because of the format's range. The following facts come from `scripts/fp16_limits.py`:

| Quantity | Value |
|---|---|
| Largest finite FP16 value | 65,504 |
| Smallest normal value, $2^{-14}$ | $6.1035 \times 10^{-5}$ |
| Smallest subnormal value, $2^{-24}$ | $5.9605 \times 10^{-8}$ |
| `np.float16(1e-8)` | 0.0 |
| `np.float16(65520)` | `inf` |

Training produces values on both sides of that range. Weight gradients are often tiny. The paper's Figure 2b shows that about 5% of the weight-gradient values of a Mandarin speech model have exponents below $-24$ (Section 3.1), and Figure 3 shows that 67% of the activation gradients of an SSD detector are zero when stored in FP16 (Section 3.2). The SSD network diverges in FP16 (Section 4.2, Table 2).

Mixed precision training keeps FP16 for the bulk of the work and uses FP32 in the three places where FP16 breaks. The paper's contribution is that this recipe needs no change to the model or to the hyperparameters (Section 1).

## Key ideas

The paper has no numbered equations. It describes each technique in prose and in Figure 1, so this page writes the formulas out. The notation is mine: $w$ for a weight, $g$ for its gradient, $\eta$ for the learning rate, and $S$ for the loss scale. The paper doesn't use those symbols. Every number comes from a script in `scripts/`. Run any script from the paper folder to reproduce the output.

### FP32 master copy of the weights

**Intuition.** The forward and backward passes use an FP16 copy of the weights, so they are fast and small. The optimizer keeps a second copy in FP32 and updates that one. After every step, it rounds the FP32 copy to FP16 for the next iteration. Small updates accumulate in the high-precision copy, where they can't be rounded away.

**Math.** Section 3.1 and Figure 1 describe one iteration. Write $w^{(32)}$ for the master weights and $w^{(16)}$ for the working copy:

$$
w^{(16)}_t = \operatorname{fl}_{16}\!\left(w^{(32)}_t\right), \qquad
w^{(32)}_{t+1} = w^{(32)}_t - \eta\, g_t
$$

Here $\operatorname{fl}_{16}$ rounds to the nearest FP16 value, and $g_t$ is the gradient from the FP16 backward pass, converted to FP32. Update the master, not the working copy.

The paper gives two reasons why FP16 weights can't take the update directly (Section 3.1):

1. The update $\eta g$ can be too small for FP16. The paper says that any magnitude below $2^{-24}$ becomes zero. The exact rule is that values below $2^{-25}$ round to 0 and values in $[2^{-25}, 2^{-24})$ round to $2^{-24}$, so the paper's statement is slightly loose, and `scripts/fp16_limits.py` shows the boundary.
1. The update can be representable and still vanish when it is added to the weight. FP16 has 10 mantissa bits, so the update's implicit bit is shifted right by 11 or more places when the weight is at least 2,048 times larger. The sum then rounds back to the weight.

**Tiny example, computed by `scripts/master_weights.py`.** Take $w = 1.0$, $g = 0.2$, $\eta = 10^{-3}$, so that $\eta g = 2 \times 10^{-4}$. That update is representable in FP16 (stored as $2.0003 \times 10^{-4}$). The spacing of FP16 numbers just below 1.0 is $2^{-11} \approx 4.9 \times 10^{-4}$, so $1.0 - 2 \times 10^{-4}$ rounds back to 1.0 on every step.

| After 1,000 steps | Weight |
|---|---|
| FP16 weights only | 1.0 |
| FP32 master | 0.800026 |
| FP16 copy of the master | 0.7998 |

The FP16-only weight hasn't moved. The master has moved by 0.2, the amount SGD should move. Check by hand: $1000 \times 2 \times 10^{-4} = 0.2$. The script also shows case 1 of the list above. With $\eta = g = 10^{-4}$ the update is $10^{-8}$, which is 0.0 in FP16.

The paper's Figure 2a shows the effect at scale. Training a Mandarin speech model with FP16 weights and no master copy gives an 80% relative accuracy loss, and the master copy recovers the FP32 result (Section 3.1). The cost is 50% more weight memory than FP32 training, since each weight takes 4 bytes plus 2 bytes. The paper argues that the total memory use roughly halves anyway, because activations dominate memory and are stored in FP16.

Exercise 2 asks you to implement both steps and watch the FP16-only weight stall.

### Loss scaling

**Intuition.** Gradients in practice cluster at small magnitudes, and FP16's range at the small end is narrow. Multiplying the loss by a constant $S$ before backpropagation multiplies every gradient by $S$, because the chain rule is linear. The gradients move up into the representable range. Before the weight update, you divide by $S$ so the update is the same size as in FP32 training.

**Math.** Section 3.2 describes the procedure in prose. Let $L$ be the loss and $S$ the loss scale:

$$
\tilde{g} = \frac{\partial (S L)}{\partial w} = S\,\frac{\partial L}{\partial w}, \qquad
g = \frac{\operatorname{fl}_{16}(\tilde{g})}{S}
$$

The division happens in FP32 and "right after the backward pass but before gradient clipping or any other gradient-related computations" (Section 3.2), so clipping thresholds and weight decay need no retuning. The paper offers two ways to choose $S$:

- A constant found by experiment. The paper tried constants from 8 to 32K, and many networks needed none (Section 3.2).
- A constant picked from gradient statistics, so that $S \cdot \max|g| < 65{,}504$. There is no penalty for a large $S$ unless the gradients overflow.

An overflow produces infinities and NaNs "which will irreversibly damage the weights after an update" (Section 3.2). The paper suggests that you detect overflow when you unscale the weight gradients and skip the update for that iteration. The paper doesn't describe a dynamic scale that adapts during training. Later frameworks add that, and Pitfalls says more.

**Tiny example, computed by `scripts/loss_scaling.py`.** Take one gradient $g = 10^{-8}$. In FP16 it is 0. With $S = 1024$, the scaled value $1.024 \times 10^{-5}$ is stored as $1.0252 \times 10^{-5}$ (a subnormal, which has coarse spacing of $2^{-24}$), and dividing by 1024 in FP32 returns $1.0012 \times 10^{-8}$. The gradient is alive, with about 0.1% error.

Now 100,000 gradients whose exponents are spread uniformly from $-30$ to $-10$ (a made-up distribution that has the same shape as Figure 3):

| Scale $S$ | Gradients that become zero |
|---|---|
| 1 | 25.01% |
| 8 | 10.18% |
| 128 | 0.00% |
| 1,024 | 0.00% |
| 32,768 | 0.00% |

At $S = 1$, the 25% are the ones below $2^{-25}$, one quarter of the exponent range. At $S = 8$, the cut-off moves down by 3 exponent steps, which is the shift the paper uses for the SSD network ("increasing the exponents by 3", Section 3.2). The paper reports that this one change makes SSD train to 77.1% mAP, against 76.9% for FP32 (Section 4.2, Table 2). The bigLSTM language model needs a factor of 128 (Section 4.5).

For the overflow side, take $\max|g| = 2$. The scale $2^{15}$ already gives $2 \times 2^{15} = 65{,}536$, which is `inf` in FP16. The largest safe power of two is $2^{14} = 16{,}384$, and the script prints that value.

Exercise 1 asks you to implement the round trip and the scale-selection rule.

### Accumulate in higher precision

**Intuition.** A dot product adds thousands of small products. If the running sum lives in FP16, each addition rounds to 10 mantissa bits, and once the sum is large the new terms are smaller than the spacing and are lost. The fix is to multiply in FP16, keep the running sum in FP32, and convert the final result to FP16 when it is written to memory.

**Math.** Section 3.3 sorts the arithmetic into three categories:

- **Dot products** (matrix multiplies and convolutions). Accumulate the FP16 partial products into an FP32 value, then convert to FP16 before the write. Volta Tensor Cores do this: they multiply FP16 input matrices and accumulate into FP16 or FP32 outputs. The paper reports that without FP32 accumulation, some FP16 models didn't match the baseline.
- **Reductions**, such as the statistics in batch normalization and the sum in softmax. Do the arithmetic in FP32. The layers still read and write FP16 tensors, and they are memory-bound, so FP32 math costs no time.
- **Point-wise operations**, such as nonlinearities and element-wise products. Use either precision, since they are also memory-bound.

For a dot product of length $n$, the accumulator is $s_k = s_{k-1} + a_k b_k$, and the result is $\operatorname{fl}_{16}(s_n)$.

**Tiny example, computed by `scripts/accumulate.py`.** Take the dot product of a vector of 4,096 ones with itself. The exact answer is 4,096.

| Accumulator | Result |
|---|---|
| FP32 | 4,096.0 |
| FP16 | 2,048.0 |

Above 2,048 the spacing of FP16 numbers is 2, so $2048 + 1$ is a tie that rounds to even, which is 2,048. The sum stalls halfway. For two random vectors of 4,096 values from $[0, 1)$, seeded with `default_rng(0)`, the exact dot product is 1,009.62. The FP32 accumulator gives 1,009.62 (relative error $4.5 \times 10^{-7}$) and the FP16 accumulator gives 925.50 (relative error 8.3%).

Exercise 3 asks you to write the loop with a configurable accumulator.

### What the paper doesn't cover: bfloat16

The paper uses IEEE FP16 only. Its limited exponent range (5 bits) is the reason for loss scaling. A later format, bfloat16, keeps the 8 exponent bits of FP32 and cuts the mantissa to 7 bits, so it has FP32's range and needs no loss scaling. The master weights still help, because the mantissa is short. This page includes bfloat16 because it explains why modern training differs from the 2017 recipe, and the paper itself doesn't mention it.

**Tiny example, computed by `scripts/bf16_regression.py`.** A bfloat16 number is the upper 16 bits of an FP32 number. Rounding to nearest even gives these values:

| Input | bfloat16 | FP16 |
|---|---|---|
| $1 + 2^{-8}$ (a tie) | 1.0 | 1.00390625 |
| $1 + 3 \cdot 2^{-8}$ (a tie) | 1.015625 | 1.01171875 |
| $10^{-30}$ | $9.984 \times 10^{-31}$ | 0.0 |
| $10^{38}$ | $9.969 \times 10^{37}$ | `inf` |

The script also trains a 4-weight linear regression with 256 samples for 200 steps of gradient descent at learning rate 0.05, seeded with `default_rng(0)`, and quantizes the activations and gradients to bfloat16. The final mean squared error is $2.0 \times 10^{-7}$ in FP32, $5.6 \times 10^{-5}$ with bfloat16 and an FP32 master, and $3.2 \times 10^{-2}$ when the weights themselves are bfloat16 and no master exists. That is the same lesson as Section 3.1, in a different format.

Exercise 4 asks you to write the bfloat16 rounding from the bit pattern.

<!-- exercises -->

## Pitfalls

- **Scaling the loss and forgetting to unscale the gradients.** The update is $S$ times too large. The paper unscales right after the backward pass, before clipping and weight decay.
- **Unscaling in FP16.** Dividing the FP16 gradient by $S$ in FP16 pushes the small values back to 0 or to subnormals. Convert to FP32 first.
- **Choosing $S$ too large.** There is no cost until a gradient exceeds 65,504. Then the gradients hold `inf` or `NaN`, and one update ruins the weights. Check for non-finite gradients and skip that step (Section 3.2).
- **Expecting a static scale to last.** The paper uses constants, chosen once per network. Gradient magnitudes drift during training, which is why later frameworks, such as PyTorch's `GradScaler`, raise and lower the scale automatically. That is an addition to the paper's recipe.
- **Updating the FP16 weights.** The working copy is a rounded view of the master. If you apply the update to it and copy it back, you lose the point of Section 3.1.
- **Computing the update in FP16.** Even with an FP32 master, `lr * grad` in FP16 flushes small products to 0 before the master sees them. Convert the gradient to FP32 first.
- **Accumulating a long dot product in FP16.** The sum stalls once the terms drop below half the spacing. The 4,096-ones example stops at 2,048.
- **Applying loss scaling to every model.** Many of the paper's networks don't need it. The ILSVRC CNNs in Section 4.1 and the DCGAN in Section 4.6 train without it. Scaling is for models with many tiny gradients, such as SSD, bigLSTM, and the translation LSTM.
- **Counting memory wrongly.** The master copy adds 50% to the weights, not to the whole run. Activations dominate training memory, and the paper stores those in FP16.
- **Treating the 2,048 ratio as exact.** The paper says that an update vanishes when the weight is at least 2,048 times larger. Rounding can recover the update in some cases (Section 3.1), so treat the number as a rule of thumb.

## Further reading

- [Mixed Precision Training of Deep Neural Networks](https://developer.nvidia.com/blog/mixed-precision-training-deep-neural-networks/), a blog post from NVIDIA that summarizes the method and the hardware support.
- [Automatic Mixed Precision package](https://pytorch.org/docs/stable/amp.html), the PyTorch documentation for `autocast` and `GradScaler`, which add dynamic loss scaling.
- [Training with Mixed Precision, NVIDIA deep learning performance guide](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html), which covers Tensor Core requirements and the choice of loss scale.
- [IEEE 754 floating-point converter](https://www.h-schmidt.net/FloatConverter/IEEE754.html), for checking by hand how a number is stored.
