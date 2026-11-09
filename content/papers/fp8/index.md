---
id: "fp8"
title: "FP8 Formats for Deep Learning"
authors: "Micikevicius et al."
year: 2022
venue: "arXiv preprint"
arxiv: "2209.05433"
url: "https://arxiv.org/abs/2209.05433"
track: gpu-systems
tier: extended
est_hours: 3
status: full
summary: "The paper defines two 8-bit floating point encodings: E4M3, with 4 exponent bits and 3 mantissa bits, and E5M2, with 5 exponent bits and 2 mantissa bits. It recommends E4M3 for weight and activation tensors and E5M2 for gradient tensors, and it leaves the choice of a scaling factor for each tensor to software. Simulated FP8 training matches 16-bit training on image classification, translation, and language models up to 175 billion parameters without any change to the hyperparameters."
related_course: "gpu: numerics"
prereqs:
  - id: mixed-precision
    reason: "FP8 reuses the idea of scaling values into the representable range before a cast, and the FP32 or FP16 master weights stay wide while only the inputs of matrix multiplies are cast to 8 bits."
background: [floating-point-formats, dynamic-range]
course_links:
  - title: "Floating point on the GPU"
    url: "https://gpu.ifkash.dev/c/09-floating-point-on-the-gpu"
  - title: "Quantized kernels"
    url: "https://gpu.ifkash.dev/c/18-quantized-kernels"
self_check:
  - q: 'What are the largest normal values of E4M3 and E5M2, and why is the E4M3 maximum 448 instead of the 240 that IEEE 754 conventions would give?'
    a: 'E4M3 tops out at $1.75 \times 2^{8} = 448$ and E5M2 at $1.75 \times 2^{15} = 57{,}344$ (Table 1). IEEE conventions would reserve the top exponent field of E4M3 for infinities and NaNs, which caps the maximum at 240. The paper instead keeps only one NaN pattern, `S.1111.111`, and uses the other seven patterns of that exponent for 256, 288, 320, 352, 384, 416, and 448 (Section 3.1). The range grows from 17 to 18 binades.'
    see: floating-point-formats
  - q: 'A gradient tensor has values around $10^{-5}$. What happens when you cast it to E4M3 without a scaling factor, and what happens with E5M2?'
    a: 'The smallest E4M3 subnormal is $2^{-9} \approx 1.95 \times 10^{-3}$, so every value below half of that rounds to 0 and the whole tensor vanishes. The smallest E5M2 subnormal is $2^{-16} \approx 1.5 \times 10^{-5}$, so a bit more than half of the values still flush to zero (55% for the Gaussian tensor in `scripts/scaling.py`). A per-tensor scaling factor moves the values into the representable range of either format.'
    see: dynamic-range
  - q: 'FP16 automatic mixed precision skips the weight update and lowers the loss scale when it detects an overflow. Why does the paper say this is a poor choice for FP8?'
    a: 'FP8 has a much narrower dynamic range, so overflows are much more likely, and the training run would skip too many updates (Section 2). The paper has FP8 casts saturate to the largest representable value, and it picks each scaling factor so that the maximum magnitude in the tensor lands near the format maximum.'
    see: mixed-precision
---

## The problem

Mixed precision training moved most of the arithmetic of training from 32 bits to 16 bits. The paper's argument is that the next step is 8 bits, which "reduces the compute requirements of neural network training" and the memory bandwidth that the tensors consume (Section 1, Section 5). Eight bits is a hard budget, because a floating point number has to split those bits between range and precision.

The two ways to split them are the two encodings the paper defines (Section 3):

| Format | Exponent bits | Mantissa bits | Max normal | Min subnormal |
|---|---|---|---|---|
| E4M3 | 4 | 3 | 448 | $2^{-9} \approx 1.95 \times 10^{-3}$ |
| E5M2 | 5 | 2 | 57,344 | $2^{-16} \approx 1.53 \times 10^{-5}$ |

These values come from `scripts/fp8_format.py`, which decodes all 256 bit patterns and reproduces Table 1 of the paper. Neither format works for every tensor:

- Cast a tensor of gradients near $10^{-5}$ to E4M3 and every value rounds to 0.
- Cast a tensor with activation outliers near 1,000 to E4M3 and the outliers clip to 448.

`scripts/scaling.py` runs both experiments. With plain casting the gradients lose 100% of their values in E4M3 and 55% in E5M2, and the outlier-heavy activations end up with a relative RMSE of 0.56. The paper's answer has two parts. It defines two encodings, so a tensor can pick the one that suits its range. It also scales each tensor by a software-chosen factor, so the tensor's values overlap the format's range (Section 2).

The paper's claim is empirical. Training with the matrix-multiply inputs rounded to FP8 matches 16-bit training across a wide set of models, and E4M3 post-training quantization of BERT recovers an accuracy that int8 doesn't (Section 4). The training runs in the paper are simulated: tensor values are clipped to the values that FP8 can represent, but the arithmetic runs in FP16 or bfloat16 (Section 4).

## Key ideas

The paper has no numbered equations. It specifies the formats in Table 1 and describes scaling in prose, so this page writes the formulas out. The notation is mine: $e$ and $m$ for the numbers of exponent and mantissa bits, $s$ for a per-tensor scaling factor, $F_{\max}$ for the largest representable magnitude of a format, and $\operatorname{fl}_8$ for rounding to FP8. The paper doesn't use those symbols. Every number comes from a script in `scripts/`, and each script runs from the paper folder with `python3 scripts/<name>.py`.

### Two encodings, E4M3 and E5M2

**Intuition.** One mantissa bit buys a factor of two in precision. One exponent bit buys a lot more range. E4M3 keeps three mantissa bits, so its neighboring values are about 12% apart at worst, and it covers about 18 powers of two. E5M2 keeps two mantissa bits, so its values are about 25% apart at worst, and it covers 32 powers of two. The paper recommends "E4M3 for weight and activation tensors, and E5M2 for gradient tensors" (Section 3).

**Math.** A byte is written $S.E.M$ (Section 3): a sign bit, an exponent field of $e$ bits, and a mantissa field of $m$ bits. The mantissa is the trailing significand field, without the implied leading 1. With exponent bias $b$ (7 for E4M3 and 15 for E5M2, Section 3.2), a byte decodes as:

$$
x = (-1)^S \cdot \left(1 + \frac{M}{2^m}\right) \cdot 2^{E - b} \quad (E > 0), \qquad
x = (-1)^S \cdot \frac{M}{2^m} \cdot 2^{1 - b} \quad (E = 0)
$$

The second case is the subnormal range. Table 1 lists the layout of the extreme values, and the formula reproduces them. For example, the E4M3 maximum is `S.1111.110`, which is $1.75 \cdot 2^{8} = 448$.

The paper chose E5M2 to be "IEEE half precision with fewer mantissa bits", so converting between E5M2 and FP16 is straightforward (Section 3). It adds that 5 exponent bits give "sufficient per tensor dynamic range (32 binades, including the subnormal values) for DL" (Section 3.1).

**Tiny example, computed by `scripts/fp8_format.py`.** Decoding every pattern gives these counts and extremes:

| | E4M3 | E5M2 |
|---|---|---|
| Max normal | 448 | 57,344 |
| Min normal | $2^{-6} = 0.015625$ | $2^{-14} \approx 6.10 \times 10^{-5}$ |
| Max subnormal | 0.0136719 ($0.875 \cdot 2^{-6}$) | $4.5776 \times 10^{-5}$ ($0.75 \cdot 2^{-14}$) |
| Min subnormal | 0.00195312 ($2^{-9}$) | $1.5259 \times 10^{-5}$ ($2^{-16}$) |
| Binades covered | 18 | 32 |
| Finite codes (of 256) | 254 | 248 |

The values in $[1, 2)$ show the precision difference. E4M3 has eight: 1, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875. E5M2 has four: 1, 1.25, 1.5, 1.75. Exercise 1 asks you to write the decoder and check it against Table 1.

### E4M3 reclaims the special values

**Intuition.** An IEEE-style format spends a whole exponent field on infinities and NaNs. For an 8-bit format with only 4 exponent bits, that costs a binade of range. The paper gives E4M3 the extra range back by dropping infinities and keeping a single NaN pattern per sign (Section 3.1).

**Math.** In E5M2 the top exponent field, `11111`, encodes infinity when $M = 0$ and NaN otherwise. In E4M3, only `S.1111.111` is NaN. The other patterns with $E = 15$ are ordinary numbers $(1 + M/8) \cdot 2^{8}$ for $M = 0, \ldots, 6$. The paper counts the gain as "seven more magnitudes (256, 288, 320, 352, 384, 416, 448)" and a range that grows "from 17 to 18 binades" (Section 3.1). Without the change, the maximum would be 240.

The paper considered going one step further. A single zero and a single NaN would add the magnitude 480, but it rejects that option because it would break the symmetry between positive and negative values, which lets software compare and sort floats with integer operations (Section 3.1).

**Tiny example, computed by `scripts/fp8_format.py`.** The top binade of E4M3 decodes to `[256, 288, 320, 352, 384, 416, 448, nan]`. The largest value below the reserved NaN pattern is 448, and the largest value with an IEEE-style reserved exponent would be 240. The script also counts 2 NaN codes and 0 infinity codes for E4M3, and 6 NaN codes and 2 infinity codes for E5M2.

Because E4M3 has no infinity, the conversion rules matter. A finite value that overflows saturates to the maximum, and an infinity or NaN in the wider type becomes NaN in E4M3 (Section 2).

### Rounding to FP8 and saturating

**Intuition.** Within a binade $[2^k, 2^{k+1})$ the representable values are evenly spaced. A cast picks the nearest one and, on a tie, the one with an even mantissa. A value beyond the maximum saturates to the maximum instead of becoming infinity. The paper says rounding mode "is orthogonal to the interchange format" and leaves it to the implementation (Section 2). This page uses round to nearest even, the common default.

**Math.** For a normal value $x$ with $2^k \le |x| < 2^{k+1}$, the spacing is $2^{k-m}$, and

$$
\operatorname{fl}_8(x) = \operatorname{sign}(x) \cdot \min\!\left(2^{k-m}\,\operatorname{round}\!\left(\frac{|x|}{2^{k-m}}\right),\; F_{\max}\right)
$$

with $k$ clamped from below at $1 - b$, which gives the subnormal range its fixed spacing $2^{1-b-m}$. Rounding to nearest bounds the relative error of a normal value by half the spacing over the value, which is at most $2^{-(m+1)}$. That bound is my derivation, not a claim from the paper. It gives 6.25% for E4M3 and 12.5% for E5M2.

**Tiny example, computed by `scripts/rounding_error.py`.** Ties in E4M3: 1.0625 sits between 1.0 and 1.125 and rounds to 1.0, which has the even mantissa. 1.1875 sits between 1.125 and 1.25 and rounds to 1.25. 1.3125 rounds to 1.25. In E5M2, 1.125 rounds to 1.0 and 1.375 rounds to 1.5.

On 100,000 draws from a standard Gaussian (seeded with `default_rng(0)`), counting only values with $|x| > 0.1$:

| Format | Mean relative error | Max relative error | Bound $2^{-(m+1)}$ |
|---|---|---|---|
| E4M3 | 0.0225 | 0.0588 | 0.0625 |
| E5M2 | 0.0449 | 0.1111 | 0.1250 |

E5M2 has about twice the error of E4M3 on the same data, which is the price of one fewer mantissa bit. Exercise 2 asks you to write `quantize` and check it against a brute-force search over all representable values.

### Per-tensor scaling

**Intuition.** A scale factor slides a tensor along the number line so that it overlaps the range of the format. Multiply before the cast, divide after it. The paper compares this to loss scaling in mixed precision training, which moves gradients into the FP16 range (Section 2). The difference is that FP8's range is so narrow that one scale per tensor type isn't enough for some networks, so each tensor gets its own factor (Section 2, Section 4.3).

**Math.** For a tensor $x$ and a format with maximum $F_{\max}$, the paper describes the heuristic in words. The scale is chosen "such that the maximum magnitude in the tensor becomes close to the maximum representable magnitude" (Section 2). Written as a formula, with $\operatorname{amax}(x) = \max_i |x_i|$:

$$
s = \frac{F_{\max}}{\operatorname{amax}(x)}, \qquad
x_8 = \operatorname{fl}_8(s\,x), \qquad
\hat{x} = \frac{x_8}{s}
$$

The paper says that the details of picking the factor are "beyond the scope of this paper", so the formula for $s$ is the simplest instance of its description. Section 4.2 names three calibration methods for activations, which are max, percentile, and mean squared error, and it uses max calibration for weights. Values that still overflow saturate (Section 2). For a matrix multiply, the unscaling happens once per dot product on the higher-precision output, so its cost is amortized over many multiply-accumulates (Section 2).

Section 3.2 explains why the factor is left to software. An exponent bias moves the representable range along the number line, so it "is equivalent to allowing only powers of 2 as scaling factors", while a software scale can be any real number.

Section 2 also explains why FP16's overflow policy doesn't carry over. FP16 training skips the update and lowers the loss scale when it sees an overflow. In FP8, overflows are much more likely, so "too many skipped updates" would result.

**Tiny example, computed by `scripts/scaling.py`.** Two tensors of 10,000 values, both seeded with `default_rng(0)`. These are made-up distributions, not the paper's data.

| Tensor | Cast | Result |
|---|---|---|
| Gaussian with 10 values multiplied by 1,000 (amax 1,304) | E4M3, no scale | 6 values exceed 448 and clip. Relative RMSE 0.5619 |
| Same | E4M3, $s = 448 / 1304 = 0.3436$ | No clipping. Relative RMSE 0.0215 |
| Gaussian with standard deviation $10^{-5}$ (amax $4.02 \times 10^{-5}$) | E4M3, no scale | 100% zeros |
| Same | E4M3, amax scale | 0% zeros. Relative RMSE 0.0264 |
| Same | E5M2, no scale | 55.2% zeros. Relative RMSE 0.4457 |
| Same | E5M2, amax scale | 0% zeros. Relative RMSE 0.0525 |

Scaling is what rescues the small-gradient tensor. Section 4.3 shows the same effect on a trained network, which is the paper's evidence that scaling matters. A bfloat16-trained 1.3B GPT-3 model has a wikitext103 perplexity of 10.19. Casting only the matrix-multiply inputs to E4M3 with no calibration and a suitable exponent bias (in the range 7 to 10) matches that baseline. Also casting the residual connections leaves no single exponent bias that works, and even bias 7 gives a perplexity of 12.59. With per-tensor scales, the two cases give 10.29 and 10.44 (Section 4.3, Figure 2). Section 4.2 adds that casting to FP8 without a scaling factor raised a perplexity result to 11.0.

Exercise 3 asks you to write `amax_scale` and the scaled round trip.

### Which format for which tensor

**Intuition.** Weights and activations are bounded and well-behaved after scaling, so the extra mantissa bit of E4M3 pays off. Gradients span many orders of magnitude, so the extra exponent bit of E5M2 pays off. The paper's recommendation is E4M3 for weights and activations and E5M2 for gradients (Section 3). It adds that some networks train with only one of the two types, and others need both or must keep fewer tensors in FP8 (Section 3).

**Math.** There is no new formula. The choice depends on the spread of the tensor's magnitudes against the dynamic range of the format. Each format covers $\log_2(F_{\max} / \text{min subnormal})$ binades, which is about 18 for E4M3 and 32 for E5M2. A tensor whose useful values span more binades than the format covers loses its smallest values to 0, even when its largest value is scaled to $F_{\max}$.

**Tiny example, computed by `scripts/scaling.py`.** Both tensors are scaled by amax first. The Gaussian tensor stands in for weights, and the wide-range tensor, which multiplies a Gaussian by $10^u$ with $u$ uniform on $[-6, 0]$, stands in for gradients. Both are made up.

| Tensor | Format | Zeros | Median per-element error |
|---|---|---|---|
| Gaussian | E4M3 | 0.0% | 0.0214 |
| Gaussian | E5M2 | 0.0% | 0.0436 |
| Wide-range | E4M3 | 16.0% | 0.0312 |
| Wide-range | E5M2 | 0.0% | 0.0433 |

On the Gaussian tensor, E4M3 halves the error, and nothing is lost to 0. On the wide-range tensor, E4M3 sends 16% of the values to 0 because their magnitudes fall below its smallest subnormal after scaling, while E5M2 keeps every value at the cost of more rounding error. The relative RMSE doesn't show the difference, because large values dominate it: 0.0254 for E4M3 and 0.0557 for E5M2. The zeros and the per-element error show what is lost.

### What the paper measured

The paper's experiments clip the inputs of matrix multiplies and convolutions (the "GEMM" inputs): activations, weights, and activation gradients. Outputs stay in higher precision (Section 4.1). All numbers below come from the paper's tables, not from this page's scripts.

- **Image classification** (Section 4.1, Table 2). FP8 is within run-to-run variation for most models. ResNet-50 v1.5 scores 76.76 against a baseline of 76.71. MobileNet v2 falls from 71.65 to 71.04, and the authors say they continue to work on recovering that accuracy.
- **Translation** (Section 4.1, Table 3). Transformer Large scores 28.35 BLEU against 28.43, and GNMT scores 24.65 against 24.83.
- **Language models** (Section 4.1, Table 4, Figure 1). The GPT-3 style models range from 126M to 175B parameters. At 175B the perplexity is 6.68 for FP8 and 6.65 for the baseline, measured at 75% of training because the bfloat16 run hadn't finished.
- **Post-training quantization** (Section 4.2, Table 5). For BERT Base on SQuAD v1.1 the F1 score is 88.19 in 16-bit, 76.89 in int8, and 88.09 in E4M3.

<!-- exercises -->

## Pitfalls

- **Using IEEE conventions for E4M3.** An E4M3 that reserves the top exponent for infinities and NaNs has a maximum of 240, and decoders based on it disagree with Table 1 for 7 magnitudes. The paper's E4M3 has no infinity and one NaN pattern (Section 3.1).
- **Assuming E4M3 has infinities.** A finite overflow saturates to 448. An infinity in the wider type becomes NaN (Section 2). If your code checks `isinf` on an E4M3 tensor, it never fires.
- **Casting without a scale.** The tensor either flushes to zero or clips to the maximum. The paper says some networks need a per-tensor factor, and a single factor for all tensors of a type isn't enough (Section 4.3).
- **Forgetting to unscale.** The scale multiplies before the cast and divides after the matrix multiply. A forgotten division makes the output $s$ times too large.
- **Computing the scale from a signed maximum.** Use the largest magnitude. A tensor whose largest element is negative gets the wrong scale otherwise.
- **Dividing by zero for an all-zero tensor.** `amax` is 0, and $F_{\max} / 0$ is not a usable scale. Pick 1.
- **Copying FP16's overflow policy.** Skipping the update and lowering the scale on overflow worked for FP16, and the paper calls it a poor choice for FP8 (Section 2).
- **Treating the paper's accuracy results as measurements of a hardware speedup.** The runs are simulated. Tensors are clipped to FP8 values, and the arithmetic runs in FP16 or bfloat16 (Section 4). The paper reports no throughput numbers.
- **Reading "matches 16-bit" as "matches everywhere".** The paper casts only the inputs of matrix multiplies and convolutions, and it leaves other tensors, such as residual connections, to later study or per-tensor calibration (Section 4.1, Section 4.3). The authors also note a remaining gap for MobileNet v2.
- **Taking the heuristic for the scale as the paper's algorithm.** The paper describes the goal, a maximum magnitude close to the format maximum, and leaves the heuristics out of scope (Section 2). Calibration with percentile or mean squared error methods, as in Section 4.2, can do better than the amax rule.
- **Equating a power-of-2 scale with an arbitrary scale.** An exponent bias is the same as a power-of-2 scale (Section 3.2). A real-valued scale gives a closer fit to the format maximum.

## Further reading

- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433), the paper. Table 1 and Section 3.1 are the parts to read twice.
- [Using FP8 with Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html), NVIDIA's primer on the two formats and on delayed per-tensor scaling in training. That recipe is a later addition to this paper's ideas.
- [OCP 8-bit Floating Point Specification (OFP8)](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1), the later industry specification of the same two encodings.
- [`ml_dtypes`](https://github.com/jax-ml/ml_dtypes), a NumPy extension with FP8 dtypes, handy for checking your `quantize` against a library.
