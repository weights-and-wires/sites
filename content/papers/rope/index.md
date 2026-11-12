---
id: "rope"
title: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
authors: "Su et al."
year: 2021
venue: "Neurocomputing 2024"
arxiv: "2104.09864"
url: "https://arxiv.org/abs/2104.09864"
track: architectures
tier: core
est_hours: 3
status: full
summary: "RoPE encodes position by rotating each pair of query and key coordinates by an angle proportional to the token position. Because a rotation by m followed by the inverse of a rotation by n equals a rotation by m minus n, the attention score depends on the relative offset only, and no position vectors are added to the input. The method adds no learned parameters, and later long-context extension methods work by rescaling its angles."
related_course: "ml: positional encoding"
prereqs:
  - id: transformer
    reason: "RoPE changes only the query-key product inside scaled dot-product attention, so you need that layer and the sinusoidal encoding it replaces."
background: [complex-numbers-and-rotations, dot-products]
course_links:
  - title: "Position encodings: sinusoidal clocks and rotary embeddings"
    url: "https://ml.ifkash.dev/lessons/062-position-encodings"
  - title: "Rotary position embeddings"
    url: "https://qwen.ifkash.dev/c/05-rotary-embeddings"
self_check:
  - q: 'Rotate the vector $[1, 0]$ by an angle $a$, and rotate another copy of $[1, 0]$ by an angle $b$. What is the dot product of the two results?'
    a: 'The results are $[\cos a, \sin a]$ and $[\cos b, \sin b]$, so the dot product is $\cos a \cos b + \sin a \sin b = \cos(a - b)$. It depends only on the difference of the angles, which is the fact RoPE builds on.'
    see: complex-numbers-and-rotations
  - q: 'The score $q^\top k$ is a sum over coordinates. Why does RoPE rotate pairs of coordinates and not single coordinates?'
    a: 'A sum over coordinates splits into independent 2D dot products, one per pair, so each pair can be handled on its own. A rotation needs two coordinates to act on, and a 2D rotation keeps the length of the pair and changes only its angle.'
    see: dot-products
  - q: 'In the Transformer, a sinusoidal vector $p_{pos}$ is added to the embedding before the $W_q$ and $W_k$ projections. Besides the two tokens, what does $q_m^\top k_n$ then depend on?'
    a: 'It depends on both absolute positions $m$ and $n$, through cross terms such as $x_m^\top W_q^\top W_k p_n$ (Equation 6 of the RoFormer paper expands them). The RoFormer paper asks for a score that depends on the relative position $m - n$ alone (Section 3.1, Equation 11).'
    see: transformer
---

## The problem

The Transformer ([Attention Is All You Need](/transformer/)) adds a position vector $p_i$ to each token embedding before the projections $W_q$ and $W_k$ (Section 2.2 of the RoFormer paper, Equation 3). Expanding the score $q_m^\top k_n$ then produces four terms (Equation 6). Three of them mix word content with absolute positions $m$ and $n$. The model has to learn, from data, that only the difference $m - n$ matters for most language patterns, such as "the next word" or "two words back".

Relative-position methods fix this by editing the expanded score: they replace $p_n$ with a relative vector $\tilde p_{m-n}$ (Equations 5 to 10), or add a learned bias $b_{i,j}$ (Equation 8). The paper's complaint is that all of these still add position information to the content terms (Section 2.3), which also makes them hard to combine with linear attention (Section 1).

**A concrete failure, computed by `scripts/additive_vs_rotary.py`.** Give every position the same token vector $x$ and set $W_q = W_k = I$. Then every pair of tokens two positions apart should score the same. With additive sinusoidal encodings, the score $(x + p_m)^\top (x + p_n)$ for offset 2 is 7.1617 at $m = 2$, 7.2629 at $m = 10$, 8.3396 at $m = 50$, and 7.5612 at $m = 200$. With RoPE it is 3.0855 every time.

The paper asks for a function pair $f_q, f_k$ whose inner product depends only on the word embeddings and the relative position (Section 3.1, Equation 11):

$$
\langle f_q(x_m, m),\, f_k(x_n, n) \rangle = g(x_m, x_n, m - n)
$$

## Key ideas

The sections that follow use the paper's notation: $x_m$ for the embedding at position $m$, $q_m = f_q(x_m, m)$, $k_n = f_k(x_n, n)$, $W_q$ and $W_k$ for the projections, $d$ for the head width (even), and $\theta_i$ for the angles. The paper calls the matrix $R^d_{\Theta,m}$. This page writes $R_m$ when $d$ and $\Theta$ are clear. Every number comes from a script in `scripts/`. Run any script from the paper folder to reproduce the output.

### Rotate a 2D vector by an angle proportional to its position

**Intuition.** Think of a query as an arrow in a plane. RoPE turns the arrow by $m$ steps of a fixed angle $\theta$, where $m$ is the token position. Two arrows turned by $m$ and $n$ steps differ in angle by $(m - n)\theta$, whatever $m$ and $n$ are. The dot product of two arrows depends on their lengths and the angle between them, so it sees only $m - n$.

**Math.** Section 3.2.1 solves Equation 11 for $d = 2$. It writes the 2D vector as a complex number and shows that the solution is (Equation 12)

$$
f_q(x_m, m) = (W_q x_m)\,e^{im\theta}, \quad
f_k(x_n, n) = (W_k x_n)\,e^{in\theta}, \quad
g(x_m, x_n, m - n) = \operatorname{Re}\!\left[(W_q x_m)(W_k x_n)^{*} e^{i(m-n)\theta}\right]
$$

where $\operatorname{Re}[\cdot]$ takes the real part and $^{*}$ is the complex conjugate. The constant $\theta \neq 0$ is preset. As a matrix, multiplying by $e^{im\theta}$ is a rotation applied after the projection (Equation 13):

$$
f_{\{q,k\}}(x_m, m) =
\begin{bmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{bmatrix}
W_{\{q,k\}} x_m
$$

Section 3.4.1 derives this. The argument has three steps. Write $f_q$, $f_k$, and $g$ in polar form. Equation 11 forces the radii not to depend on position (Equation 27) and forces the angle to grow by the same constant at every step, an arithmetic progression $\phi(m) = m\theta + \gamma$ (Equation 30). Then $\gamma = 0$ is a free choice (Equation 33).

**Tiny example, computed by `scripts/rotation_2d_example.py`.** Take $\theta = 0.5$, $W_q x_m = q = [1, 2]$, and $W_k x_n = k = [3, 1]$. Without rotation, $q \cdot k = 5$. With rotation:

| $m$ | $n$ | $m - n$ | $q_m \cdot k_n$ |
|---|---|---|---|
| 5 | 2 | 3 | -4.6338 |
| 8 | 5 | 3 | -4.6338 |
| 13 | 10 | 3 | -4.6338 |
| 2 | 5 | -3 | 5.3412 |
| 7 | 2 | 5 | -6.9981 |

The first three rows share the offset 3 and give the same score. Row 4 has the opposite offset and gives a different score, so the score keeps the direction of the offset, which matters in a causal language model. The script also checks the complex form: $\operatorname{Re}[q k^{*} e^{i(m-n)\theta}]$ gives -4.6338 as well. A rotation keeps the length: $\|q\|$ is 2.2361 before and after rotation by $5\theta$.

Exercise 2 asks you to check this for the general case.

### Rotate every pair of coordinates, with a different angle per pair

**Intuition.** A real head has $d = 64$ or $128$ coordinates, not 2. RoPE splits them into $d/2$ adjacent pairs and rotates each pair by its own angle. Pair 1 spins fast, so it distinguishes neighbors. The last pair spins slowly, so it still tells apart tokens that are thousands of positions apart. This is the sinusoidal encoding's frequency ladder, applied as a rotation.

**Math.** Section 3.2.2 divides the $d$-dimensional space into $d/2$ subspaces and uses the linearity of the inner product to combine the 2D solutions (Equation 14):

$$
f_{\{q,k\}}(x_m, m) = R^d_{\Theta,m}\, W_{\{q,k\}}\, x_m
$$

The matrix $R^d_{\Theta,m}$ is block diagonal, with $d/2$ rotation blocks (Equation 15). The $i$th block is

$$
\begin{bmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{bmatrix},
\qquad
\Theta = \left\{\theta_i = 10000^{-2(i-1)/d},\; i \in [1, 2, \dots, d/2]\right\}
$$

The base 10000 and the exponent $2i/d$ come from the sinusoidal encoding in Section 3.5 of the Transformer paper. A note on indexing: Equation 15 counts pairs from $i = 1$, so $\theta_1 = 1$. Section 3.3 writes $\theta_i = 10000^{-2i/d}$ instead. The two differ by one in the index, and the code in this page and in the exercises uses the 0-indexed form $\theta_j = 10000^{-2j/d}$ with $j = i - 1$, which matches Equation 15.

The matrix is orthogonal, which the paper says "ensures stability" (Section 3.2.2). Plugging $f_q$ and $f_k$ into the score gives the central identity (Equation 16):

$$
q_m^\top k_n = (R^d_{\Theta,m} W_q x_m)^\top (R^d_{\Theta,n} W_k x_n) = x_m^\top W_q^\top R^d_{\Theta,n-m} W_k x_n, \qquad R^d_{\Theta,n-m} = (R^d_{\Theta,m})^\top R^d_{\Theta,n}
$$

The paper prints the middle expression as $x^\top W_q R^d_{\Theta,n-m} W_k x$ and leaves out the transpose on $W_q$ and the subscripts on $x$. The version here writes them out. Note the order: $R_{n-m}$, not $R_{m-n}$, with $m$ the query position and $n$ the key position.

Multiplying a dense $d \times d$ matrix is wasteful, because most entries are zero. Section 3.4.2 gives an elementwise form (Equation 34):

$$
R^d_{\Theta,m}\, x =
\begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_{d-1} \\ x_d \end{bmatrix} \otimes
\begin{bmatrix} \cos m\theta_1 \\ \cos m\theta_1 \\ \vdots \\ \cos m\theta_{d/2} \\ \cos m\theta_{d/2} \end{bmatrix} +
\begin{bmatrix} -x_2 \\ x_1 \\ \vdots \\ -x_d \\ x_{d-1} \end{bmatrix} \otimes
\begin{bmatrix} \sin m\theta_1 \\ \sin m\theta_1 \\ \vdots \\ \sin m\theta_{d/2} \\ \sin m\theta_{d/2} \end{bmatrix}
$$

where $\otimes$ is the elementwise product. This costs $O(d)$ per vector. RoPE has no learned parameters. It applies to $q$ and $k$ only, after $W_q$ and $W_k$, and leaves $v$ untouched (the paper's Equation 1 allows a position-dependent $v_n$, but RoPE doesn't use that freedom).

**Tiny example, computed by `scripts/rope_full_example.py`.** For $d = 8$ the angles are $\theta = [1, 0.1, 0.01, 0.001]$, so the wavelengths $2\pi/\theta_i$ are 6.28, 62.83, 628.3, and 6283 positions. Take $x = [1, 2, \dots, 8]$ at $m = 3$. Both Equation 15 and Equation 34 give

```text
[-1.2722, -1.8389, 1.6839, 4.7079, 4.8178, 6.1473, 6.976, 8.021]
```

Check the first pair by hand: the angle is $3 \cdot 1 = 3$ radians, so $x_1' = 1\cos 3 - 2\sin 3 = -0.9900 - 0.2822 = -1.2722$ and $x_2' = 1\sin 3 + 2\cos 3 = 0.1411 - 1.9800 = -1.8389$. The last pair turns by only $0.003$ radians, so $[7, 8]$ barely moves: it becomes $[6.976, 8.021]$.

Now the relative property. With random $q, k \in \mathbb{R}^8$ (seed 0), the script prints $q_m \cdot k_n$ for several position pairs:

| $m$, $n$ | $m - n$ | $q_m \cdot k_n$ |
|---|---|---|
| (0, 0), (4, 4), (9, 9) | 0 | -1.467987 |
| (3, 1), (10, 8), (50, 48) | 2 | -1.763129 |
| (1, 4), (6, 9), (100, 103) | -3 | -1.628434 |

For $m = 7$, $n = 3$, the script gets -1.302115 both from rotating the two vectors and from $q^\top R_{n-m} k$ (Equation 16). The script also confirms $R_m^\top R_n = R_{n-m}$ and $R_m^\top R_m = I$.

Exercise 1 asks you to implement Equation 34. Exercise 2 asks you to build Equation 15 and verify Equation 16.

### Attention weights shrink with distance, but not by a clean rule

**Intuition.** The slow pairs of a $d$-dimensional vector keep contributing the same value at any distance, while the fast pairs fall in and out of phase. When the distance grows, the fast pairs average to zero, and only the slow ones add. Section 3.3 says this is a "long-term decay": tokens far apart should connect less.

**Math.** Section 3.4.3 groups $q = W_q x_m$ and $k = W_k x_n$ into 2D pieces and writes the score as a sum of complex products (Equation 35):

$$
q_m^\top k_n = \operatorname{Re}\!\left[\sum_{i=0}^{d/2-1} q_{[2i:2i+1]}\, k^{*}_{[2i:2i+1]}\, e^{i(m-n)\theta_i}\right]
$$

With $h_i = q_{[2i:2i+1]}k^{*}_{[2i:2i+1]}$ and $S_j = \sum_{i=0}^{j-1} e^{i(m-n)\theta_i}$, an Abel transformation (Equation 36) gives the bound (Equation 37)

$$
\left|\sum_{i=0}^{d/2-1} h_i\, e^{i(m-n)\theta_i}\right| \le \left(\max_i |h_{i+1} - h_i|\right) \sum_{i=0}^{d/2-1} |S_{i+1}|
$$

The paper then says the average $\frac{1}{d/2}\sum_i |S_i|$ decays as $m - n$ grows, and shows this in Figure 2.

**Tiny example, computed by `scripts/long_term_decay.py`.** With $d = 256$ and $\theta_i = 10000^{-2i/d}$, the average of $|S_j|$ is 64.5 at distance 0, 36.2 at distance 10, 18.6 at distance 100, and 10.5 at distance 500. At distance 1000 it is 11.3, higher than at 500. Over distances 1 to 2000 it ranges from about 5.2 (at distance 1906) to about 13.9.

So the decay holds on average over the first few hundred positions, which is the paper's claim, but the curve oscillates at long range. Also, Equation 37 is an upper bound on the size of the score, not a statement that every score falls. The paper's own limitations section says it has no faithful explanation for why RoFormer does well on long text (Section 4.5.5). Treat "decays with distance" as a loose motivation and not as a guarantee.

### Change the angles to run on longer inputs

**Intuition.** The paper's Section 1 lists "sequence length flexibility" as a benefit: unlike a learned position table with a fixed number of rows, a rotation exists for every integer $m$. But a model trained on positions up to $L$ has never seen the angles $m\theta_i$ for $m > L$, for the fast pairs especially. Extension methods leave the model alone and rescale the angles so that long inputs look familiar. The RoFormer paper doesn't describe these methods. They come later, from Chen et al. (2023) and the open-source community.

**Math.** Position interpolation divides every position by the extension factor $s = L'/L$, where $L'$ is the new length (Chen et al., 2023). The angle becomes $(m/s)\,\theta_i$, and the largest new position maps near the largest trained one. NTK-aware scaling leaves the positions alone and enlarges the base instead (a community method, folded into YaRN by Peng et al., 2023):

$$
\text{base}' = \text{base} \cdot s^{d/(d-2)}, \qquad \theta_i' = (\text{base}')^{-2i/d}
$$

With this choice the fastest pair keeps $\theta_0' = 1$, and the slowest pair, $i = d/2 - 1$, becomes exactly $\theta_{d/2-1}/s$. Pairs in between shrink by factors between 1 and $1/s$. The idea is to leave the high frequencies, which resolve neighboring tokens, nearly intact, and stretch only the low ones.

**Tiny example, computed by `scripts/extrapolation_example.py`.** Take $d = 8$, $L = 2048$, $L' = 8192$, so $s = 4$.

- Without scaling, position 8191 gives the fast pair an angle of 8191 radians. With interpolation, the position becomes 2047.75, and the angle is 2047.75. Neighboring tokens now differ by $\theta_0/s = 0.25$ radians in that pair, down from 1, so the model has to tell finer differences apart.
- NTK-aware scaling gives $\text{base}' = 63{,}496.04$ and $\theta' = [1, 0.063, 0.004, 0.0003]$. Against the original $[1, 0.1, 0.01, 0.001]$, the ratios are $[1, 0.63, 0.397, 0.25]$.
- For $d = 128$ with base 10000, 41 of the 64 pairs have a wavelength shorter than 2048 positions. The longest wavelength is about 54,410 positions.

Exercise 3 asks you to implement both rescalings.

<!-- exercises -->

## Pitfalls

- **Pairing the wrong coordinates.** The paper rotates adjacent pairs $(x_{2i-1}, x_{2i})$ (Equation 15). Many open-source implementations, including the Hugging Face Llama code, pair coordinate $i$ with coordinate $i + d/2$ (a "rotate half" convention). Both give relative-offset scores, and they are equivalent when the weights $W_q$ and $W_k$ are permuted to match. Mixing the two conventions when you load checkpoints produces garbage that doesn't crash. Exercise 1 follows the paper's pairing.
- **Rotating $v$ too.** RoPE applies to $q$ and $k$ only. Rotating $v$ would make the output depend on absolute position.
- **Rotating before the projection.** The rotation comes after $W_q$ and $W_k$ (Equation 14). Rotating $x$ first and then projecting doesn't give Equation 16, because $W$ doesn't commute with $R$.
- **Using $R_{m-n}$ for $R_{n-m}$.** The score is $q^\top R_{n-m} k$ with $m$ the query. The two differ by a transpose and agree only when $q = k$ in every pair. Exercise 2 tests the sign.
- **Computing angles in low precision.** The product $m\theta_i$ reaches the thousands for the first pair. In float16 or bfloat16, the rounding error of that product can reach whole radians once positions are in the thousands. Compute the angles in float32 and cast after the sine and cosine.
- **Treating decay as a guarantee.** Equation 37 bounds the score, and the computed average $|S_j|$ rises again after distance 500 in the `long_term_decay.py` run. Don't expect the model to ignore far tokens because of the rotation alone.
- **Reading Table 2 as a clean win.** In Table 2, RoFormer beats BERT on MRPC, STS-B, and QQP, and loses on SST-2 (90.7 against 93.5), QNLI (88.0 against 90.5), and MNLI (80.2/79.8 against 84.6/83.4). Table 1 shows a gain of 0.2 BLEU (27.5 against 27.3) on WMT 2014 English-to-German, a small gain, and the paper reports no variance across runs. The paper's strongest evidence is the faster pre-training convergence in Figure 3 and the long-text result in Table 5.
- **Assuming extension is free.** Interpolation and NTK-aware scaling change what the model sees at the positions it was trained on. Models usually need some fine-tuning, or a method like YaRN, to recover quality at the new length.

## Further reading

- [The RoFormer code release](https://github.com/ZhuiyiTechnology/roformer), linked from the paper's introduction, has the authors' implementation.
- [Rotary Embeddings: A Relative Revolution](https://blog.eleuther.ai/rotary-embeddings/), by EleutherAI, is a short explainer with code.
- [Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595), by Chen et al., introduces position interpolation.
- [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071), by Peng et al., combines per-frequency scaling with an attention temperature.
- [Attention Is All You Need](/transformer/) defines the sinusoidal encoding that RoPE's angles come from.
