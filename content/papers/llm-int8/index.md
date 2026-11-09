---
id: "llm-int8"
title: "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale"
authors: "Dettmers et al."
year: 2022
venue: "NeurIPS 2022"
arxiv: "2208.07339"
url: "https://arxiv.org/abs/2208.07339"
track: inference
tier: extended
est_hours: 3
status: full
summary: "LLM.int8() runs the feed-forward and attention projection layers of a transformer in 8-bit integers, which halves inference memory, and it matches 16-bit quality up to 175B parameters. Absmax quantization with one constant per row of the activations and one per column of the weights confines the damage of a large value to a single inner product. Beyond about 6.7B parameters a few feature dimensions carry values above 6 in most tokens, and the method multiplies exactly those dimensions in 16-bit while the other 99.9% of values stay in 8-bit."
related_course: "inference: quantization"
prereqs:
  - id: transformer
    reason: "The method quantizes the query, key, value, and output projections and the feed-forward layers of a transformer, so you need to know where those matrix multiplications sit in a block."
  - id: mixed-precision
    reason: "The method keeps FP16 inputs and outputs around the int8 products and accumulates in a wider type, which builds on the FP16 range and accumulation facts of mixed precision training."
background: [quantization, outliers]
course_links:
  - title: "Quantization on Ampere"
    url: "https://qwen.ifkash.dev/c/18-quantization"
  - title: "Quantized kernels"
    url: "https://gpu.ifkash.dev/c/18-quantized-kernels"
self_check:
  - q: 'Quantize the vector $[0.30, -0.70, 0.45, 0.10, -0.20, 0.60]$ with absmax quantization to int8, then add one value of 60.0 and quantize again. What happens to the value 0.10 in each case?'
    a: 'Without the outlier, $s_x = 127 / 0.70 \approx 181.4$ and 0.10 maps to the code 18. With the outlier, $s_x = 127 / 60 \approx 2.117$ and 0.10 maps to the code $\lfloor 0.2117 \rceil = 0$, so it is gone. One int8 step grows from 0.0055 to 0.4724, and any value below 0.2362 rounds to 0.'
    see: quantization
  - q: 'An FP16 accumulator sums terms of $60 \times 60 = 3600$, as in a dot product with a feature of magnitude 60. After how many terms does it overflow, and what does the paper hypothesize about 16-bit training?'
    a: 'The largest finite FP16 value is 65,504, and 18 terms give 64,800, so the sum becomes `inf` at term 19. Section 4.2 (point 3) hypothesizes that regular 16-bit training becomes unstable beyond 6.7B parameters for this reason. The paper quotes the limit as 65535. The largest finite FP16 value is 65,504, which is the limit that matters here.'
    see: mixed-precision
  - q: 'In $X W$, row-wise (or vector-wise) quantization gives every row of $X$ its own constant. Why does that not help when an outlier feature occupies one column of $X$ in nearly every token?'
    a: 'Each row of $X$ is a token, and nearly every token contains a value from the outlier column. Each row therefore has a large maximum, and each row''s constant is stretched by that outlier, just as a single tensor-wide constant would be. The small features in that row still lose their bins (Section 4.3).'
    see: outliers
---

## The problem

Most of the memory of a large transformer holds the weights of the feed-forward and attention projection layers. For models of 6.7B parameters and larger, those layers hold about 95% of the parameters and account for 65% to 85% of the computation (Section 1). Quantizing those weights and activations to 8-bit integers (int8) would halve the memory. Earlier 8-bit methods did that, but they were tested on models below 350M parameters, and they lost accuracy, usually needing extra tuning after training (Section 1).

The failure is easy to reproduce. Take a hidden vector of seven values and quantize it with absmax quantization, which scales the tensor so that its largest magnitude lands on 127 (Section 2.1). The following results come from `scripts/absmax_example.py`.

| Input | Scale $s_x$ | Codes of the six small values | Largest error on the small values |
|---|---|---|---|
| Six values between $-0.70$ and $0.60$ | 181.4 | `[54, -127, 82, 18, -36, 109]` | 0.0024 |
| The same six values plus one value of 60.0 | 2.117 | `[1, -1, 1, 0, 0, 1]` | 0.2276 |

One outlier of 60.0 makes a quantization step 0.4724 wide, up from 0.0055. Two of the six small values round to 0, and the rest snap to $\pm 0.4724$. The paper says the same thing about real models: the range of the quantization distribution becomes so large that most bins are empty and small values are quantized to zero (Section 4.2, point 3).

That is what happens in large transformers. Beyond roughly 6.7B parameters, a few feature dimensions of the hidden states hold values of 20 or more in most tokens, and every earlier 8-bit method fails. In Table 1, the 13B model scores a C4 perplexity of 12.45 in 32-bit float and 19.08 with int8 absmax, which is worse than the 14.59 of the 6.7B model with the same method. Figure 1 shows the same pattern for zero-shot accuracy. LLM.int8() gets 12.45 at 13B and keeps 16-bit accuracy up to 175B.

## Key ideas

The paper's notation: $X_{f16} \in \mathbb{R}^{s \times h}$ for the hidden states with sequence dimension $s$ and feature dimension $h$, $W_{f16} \in \mathbb{R}^{h \times o}$ for the weights, and $C$ for the output. A scaling constant is $c$ (also written $s_x$ for absmax), and $Q(\cdot)$ is a quantizer. The subscripts $f16$, $i8$, and $i32$ name the data types. This page drops the subscripts when the type is clear. One difference: Figure 2 of the paper writes $C_x$ for the absolute maximum and multiplies by $127 / C_x$, while Equations 6 and 7 define $c_x$ as the factor that already includes the 127. This page uses the factor form of the equations. Every number comes from a script in `scripts/`. Run any script from the paper folder to reproduce the output.

### Absmax quantization and the tensor-wise matmul

**Intuition.** Map the largest magnitude in the tensor to 127 and scale everything else by the same factor. Round to the nearest integer. The scheme is symmetric around 0 and wastes no bits on a zero point, which is why it is the most commonly used.

**Math.** Section 2.1 gives absmax quantization, without an equation number:

$$
X_{i8} = \left\lfloor \frac{127 \cdot X_{f16}}{\max_{ij}\left(|X_{f16,ij}|\right)} \right\rceil = \left\lfloor s_{x_{f16}} X_{f16} \right\rceil
$$

The brackets $\lfloor\cdot\rceil$ mean rounding to the nearest integer. An int8 matrix product multiplies the integer codes in int32 and divides by the product of the two constants (Section 2.1, Equation 6):

$$
X_{f16} W_{f16} = C_{f16} \approx \frac{1}{c_{x_{f16}} c_{w_{f16}}} C_{i32} = S_{f16} \cdot C_{i32}
$$

Here $c_x$ and $c_w$ are the constants of the two whole tensors. The paper also covers zeropoint quantization (Equations 1 to 5), which shifts an asymmetric range into $[-127, 127]$. This page skips it. Section 4.3 explains why it helped the baselines and why the gain vanishes once outliers are decomposed.

**Tiny example, computed by `scripts/absmax_example.py`.** The Problem section already gave this example. The important number is the step size, $1 / s_x$. It equals the largest magnitude divided by 127, so the error of every value in the tensor grows with the one largest value.

### Vector-wise quantization

**Intuition.** A matrix product is a table of independent inner products, one per (row of $X$, column of $W$) pair. Each inner product can use its own constants, so a large value only spoils the rows or columns that contain it. After the integer product, divide each output entry by the product of its row constant and its column constant.

**Math.** Assign a constant $c_x$ to every row of $X$ and a constant $c_w$ to every column of $W$, with $c_x \in \mathbb{R}^s$ and $c_w \in \mathbb{R}^o$. The denormalization is the outer product $c_x \otimes c_w$ (Section 3.1, Equation 7):

$$
C_{f16} \approx \frac{1}{c_{x_{f16}} \otimes c_{w_{f16}}} C_{i32} = S \cdot C_{i32} = S \cdot A_{i8} B_{i8} = S \cdot Q(A_{f16}) Q(B_{f16})
$$

In the equation, $A$ and $B$ stand for the two operands, so $A = X$ and $B = W$. The division is element-wise. Row-wise quantization, the earlier method the paper compares to (Section 3), puts a constant on each row of the hidden states. Vector-wise quantization, in the paper's account, also gives each column of $W$ its own constant, so each inner product has its own pair. Section 4.3 reports that vector-wise keeps an advantage over row-wise even with the decomposition, so the finer weight precision is needed to recover full perplexity. The gap in Table 1 is small: at 13B, row-wise plus decomposition gives 12.46 and vector-wise plus decomposition gives 12.45.

**Tiny example, computed by `scripts/vectorwise_example.py`.** Take

$$
X = \begin{bmatrix} 0.2 & -0.4 & 0.1 \\ 20 & 30 & -10 \end{bmatrix}, \quad
W = \begin{bmatrix} 0.5 & -1.0 \\ 1.0 & 0.3 \\ -0.2 & 0.8 \end{bmatrix}.
$$

Row 1 of $X$ is tiny and row 2 is large. The exact product is `[[-0.32, -0.24], [42.0, -19.0]]`. The steps are as follows:

| Step | Result |
|---|---|
| Row constants $c_x = 127 / \max\lvert\text{row}\rvert$ | `[317.5, 4.2333]` |
| Column constants $c_w$ | `[127, 127]` |
| $X_{i8}$ | `[[64, -127, 32], [85, 127, -42]]` |
| $W_{i8}$ | `[[64, -127], [127, 38], [-25, 102]]` |
| $C_{i32}$ | `[[-12833, -9690], [22619, -10253]]` |
| Vector-wise result | `[[-0.3183, -0.2403], [42.0714, -19.0706]]` |
| Tensor-wise result (Equation 6) | `[[-0.3534, -0.3776], [42.0714, -19.0706]]` |

The second row is identical in both methods, because it already sets the tensor-wide scale. The first row is where the methods differ. Its absolute errors are 0.0017 and 0.0003 with vector-wise constants and 0.0334 and 0.1376 with a tensor-wide constant. Check one entry by hand: $-12833 / (317.5 \cdot 127) = -0.3183$.

Exercise 1 asks you to implement this function.

### Emergent outlier features

**Intuition.** Vector-wise constants follow rows of $X$, which are tokens. Large transformers have outliers in columns of $X$, which are features, and those columns are present in almost every token. Every row contains a piece of an outlier column, so every row constant is stretched, and vector-wise quantization is back to the problem of the first example.

**Math.** Section 4.1 defines the outliers it tracks. A feature is a dimension $h_i$ of a hidden state $X^l \in \mathbb{R}^{s \times h}$. It counts as an outlier feature when all three conditions hold:

- Its magnitude is at least $\alpha = 6.0$ in at least one position.
- It occurs in the same dimension $h_i$ in at least 25% of the layers.
- It appears in at least 6% of the sequence positions.

The paper chose the thresholds so that the criteria detect a single outlier in its smallest model, with 125M parameters. It analyzes only the inputs of the attention projections (query, key, value, and output) and of the first feed-forward layer (Section 4.1). The paper's findings (Section 4.2) are as follows:

- **A phase shift at about 6.7B parameters.** The share of layers with outliers jumps from 65% to 100% between 6B and 6.7B parameters, and the share of sequence positions jumps from 35% to 75% (Figure 3a, point 1).
- **A smooth trend in perplexity.** Against perplexity, the emergence is gradual and exponential, so the paper suggests that perplexity, and not size alone, drives it (Figure 3b, points 2 and 4).
- **Sparse and systematic.** A 6.7B model with a sequence length of 2048 has about 150,000 outlier values per sequence, concentrated in 6 hidden dimensions, and no more than 7 dimensions for the models up to 13B (Section 3.2 and Section 4.2).
- **Essential.** Zeroing the outlier dimensions cuts the mean top-1 attention probability from about 40% to about 20% and raises validation perplexity by 600% to 1000%. Zeroing 7 random dimensions changes the probability by 0.02% to 0.3% and the perplexity by 0.1% (Section 4.2).

**Tiny example, computed by `scripts/outlier_scan.py`.** The script builds synthetic hidden states with 512 tokens and 1,024 features, adds three outlier dimensions with a magnitude near 60, and multiplies by a random weight matrix. The data is synthetic and mimics the pattern, so don't read the error values as the paper's. The "values flushed to 0" column is the share of non-outlier entries of $X$ that quantize to the code 0 in row-wise absmax:

| Outlier magnitude | Vector-wise error | Decomposed error | Dimensions with max at least 6 | Values flushed to 0 |
|---|---|---|---|---|
| none | 0.0111 | 0.0111 | 0 | 1.1% |
| 10 | 0.0232 | 0.0099 | 3 | 3.4% |
| 20 | 0.0347 | 0.0075 | 3 | 6.9% |
| 60 | 0.0448 | 0.0033 | 3 | 20.2% |

The errors are relative, $\lVert \hat{C} - C \rVert / \lVert C \rVert$. With three outlier dimensions out of 1,024, the largest channel maxima are 77.3, 76.4, and 74.0, and the next channel is at 4.9, so a threshold of 6 separates them cleanly.

Exercise 2 asks you to find the outlier dimensions.

### Mixed-precision decomposition

**Intuition.** Outlier dimensions are few and systematic, so handle them separately. Pull the outlier columns out of $X$ and the matching rows out of $W$, and multiply them in 16-bit. Quantize the remaining columns with vector-wise int8, where the row constants are now small. Add the two products.

**Math.** Let $O$ be the set of feature dimensions that hold at least one value larger than $\alpha$ in magnitude. The paper finds $\alpha = 6.0$ enough to bring the degradation close to zero (Section 3.2). The decomposition is (Section 3.2, Equation 8, in the paper's Einstein notation, where $X^h$ is column $h$ of $X$ and $W^h$ is row $h$ of $W$):

$$
C_{f16} \approx \sum_{h \in O} X^h_{f16} W^h_{f16} + S_{f16} \cdot \sum_{h \notin O} X^h_{i8} W^h_{i8}
$$

The sums run over the inner dimension. The set $O$ has no more than 7 members for models up to 13B parameters, so the 16-bit part adds about 0.1% memory (Section 3.2). The first sum is a small FP16 matmul, and the second is vector-wise int8 from Equation 7 on the remaining columns. This page defines $O$ with "at least $\alpha$", following Section 4.1. Section 3.2 says "larger than". The two differ only at an exact tie, and the paper doesn't say which it implements.

**Tiny example, computed by `scripts/decomposition_example.py`.** Take

$$
X = \begin{bmatrix} 0.30 & 40 & -0.50 & 0.20 \\ -0.40 & 55 & 0.10 & 0.60 \\ 0.25 & -48 & 0.35 & -0.15 \end{bmatrix}, \quad
W = \begin{bmatrix} 0.5 & -0.2 \\ 0.3 & 0.4 \\ -0.6 & 0.1 \\ 0.2 & 0.7 \end{bmatrix}.
$$

Column 1 holds the outliers, so $O = \{1\}$. The exact product is `[[12.49, 16.03], [16.36, 22.51], [-14.515, -19.32]]`. The results are as follows:

| Method | Result | Largest error |
|---|---|---|
| Vector-wise only | `[[12.6927, 16.1900], [16.4990, 22.5190], [-14.5509, -19.3509]]` | 0.2027 |
| Decomposed (Equation 8) | `[[12.4897, 16.0316], [16.3584, 22.5095], [-14.5139, -19.3192]]` | 0.0016 |

The largest error falls from 0.2027 to 0.0016. The outlier values 40, 55, and $-48$ are multiplied in float, so only the three small columns are rounded to int8. Exercise 3 asks you to write this function.

### What the result shows, and what it doesn't

Table 1 compares the methods at five sizes. The rows below are perplexities on C4, from the paper's table:

| Method | 125M | 1.3B | 2.7B | 6.7B | 13B |
|---|---|---|---|---|---|
| 32-bit float | 25.65 | 15.91 | 14.43 | 13.30 | 12.45 |
| Int8 absmax | 87.76 | 16.55 | 15.11 | 14.59 | 19.08 |
| Int8 zeropoint | 56.66 | 16.24 | 14.76 | 13.49 | 13.94 |
| Int8 absmax vector-wise | 35.84 | 16.82 | 14.98 | 14.13 | 16.48 |
| Absmax LLM.int8() (vector-wise + decomposition) | 25.83 | 15.93 | 14.44 | 13.24 | 12.45 |

Beyond perplexity, Figure 1 shows that the mean zero-shot accuracy of OPT models stays at the 16-bit level from 125M to 175B parameters, while the best earlier 8-bit method degenerates toward random after the emergence. For BLOOM-176B, the paper reports a 1.96 times smaller memory footprint (Section 3). The paper's Table 2 turns that into hardware: OPT-175B and BLOOM fit in 8 RTX 3090 GPUs with 24 GB each in 8-bit, while the 16-bit limit on the same server is OPT-66B.

The limits are stated in Section 6. The paper studies the int8 data type only, not FP8. It tests models up to 175B parameters, and other emergent properties could break the method at larger scales. It doesn't use int8 for the attention function, because that function has no parameters, so the quantization covers the linear layers only. It also studies inference only, with an initial analysis of training in Appendix E. On speed, the quantization overhead can slow models below 6.7B parameters compared with FP16. The paper reports about twice the speed for the large matrix multiplications of 175B-scale models (Section 3.4). The method's main goal is memory.

<!-- exercises -->

## Pitfalls

- **Quantizing per tensor and calling it int8.** One constant for the whole tensor lets one value set the step size for every value (the first table). Use per-row constants for $X$ and per-column constants for $W$.
- **Mixing up the axes.** Row constants belong to $X$, because they follow the tokens. Column constants belong to $W$, because they follow the output channels. If you quantize the rows of $W$, the shapes break, or worse, they match by luck when $h = o$.
- **Accumulating in int8.** The product of two int8 codes reaches 16,129, which doesn't fit in int8, and a sum of $h$ of them is larger still. Accumulate in int32, as Equation 6 shows. NumPy wraps int8 products silently.
- **Thinking vector-wise handles outliers.** It follows rows, and outlier features sit in columns (Section 4.3). It is the first half of the method. The decomposition is the second half.
- **Taking the outliers from the weights.** The outlier dimensions come from the activations $X$, which change with every input. Equation 8 defines $O$ from $X$, and the same indices select the rows of $W$.
- **Counting an outlier column in both parts.** If the column stays in the int8 matmul, the row constants are still large and you add its contribution twice.
- **Copying the threshold blindly.** The value 6.0 is a magnitude that the paper chose empirically, because perplexity degradation stops once features at or above it are treated as outliers (Section 4.1). Activation scales depend on the model, and the paper doesn't claim 6.0 is universal.
- **Quantizing the attention function.** The paper doesn't (Section 6). The method covers the weight-bearing linear layers: the query, key, value, and output projections and the feed-forward layers.
- **Expecting a speedup at small scale.** The paper's main aim is memory. Models below 6.7B parameters can run slower because of the quantization overhead (Section 3.4).
- **Treating 6.7B as a law.** The paper shows that emergence tracks perplexity as a smooth trend, and that model size is one covariate among many (Section 4.2, points 2 and 4). The 6.7B boundary is for the models the paper tested.
- **Dividing by zero.** A row or column of all zeros has no maximum to divide by. The exercises' solutions assume that each row and column has a nonzero value, and a production kernel must handle the case.

## Further reading

- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes), the paper's open-source implementation, which Hugging Face Transformers uses for 8-bit loading.
- [A Gentle Introduction to 8-bit Matrix Multiplication for transformers at scale](https://huggingface.co/blog/hf-bitsandbytes-integration), the Hugging Face blog post that introduced the integration, with the same figures and code examples.
- [Emergent Features and Outliers](https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features/), the author's blog post on the emergent features behind the method.
- [ZeroQuant](https://arxiv.org/abs/2206.01861) and [GPTQ](/gptq/), related ways to quantize large models. ZeroQuant uses finer, group-wise constants (Section 5), and GPTQ quantizes weights to lower bit widths.
