---
id: "lora"
title: "LoRA: Low-Rank Adaptation of Large Language Models"
authors: "Hu et al."
year: 2021
venue: "ICLR 2022"
arxiv: "2106.09685"
url: "https://arxiv.org/abs/2106.09685"
track: training-optimization
tier: core
est_hours: 3
status: full
summary: "LoRA freezes a pretrained weight matrix and learns its change as the product of two thin matrices, so each new task trains and stores a tiny fraction of the weights. The thin factor B starts at zero, so training begins exactly at the base model. After training, the product folds into the original weight, so inference costs the same as the base model."
related_course: "ml: parameter-efficient fine-tuning"
prereqs:
  - id: transformer
    reason: "LoRA is applied to the query, key, value, and output projections of self-attention, so you need to know where those weight matrices sit and what shapes they have."
background: [low-rank-factorization, fine-tuning, matrix-rank]
course_links:
  - title: "Fine-tuning on a budget with LoRA"
    url: "https://ml.ifkash.dev/lessons/077-fine-tuning-and-lora"
self_check:
  - q: 'A GPT-3 attention projection $W_q$ is $12288 \times 12288$. How many weights do $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ hold for $r = 4$, and how does that compare with the full matrix?'
    a: 'They hold $4 \cdot (12288 + 12288) = 98{,}304$ weights, against $12288^2 = 150{,}994{,}944$ in the full matrix. That is 1,536 times fewer.'
    see: low-rank-factorization
  - q: 'What is the largest possible rank of $\Delta W = BA$ when $B$ is $d \times r$ and $A$ is $r \times k$? Why does that matter for the claim that LoRA "constrains" the update?'
    a: 'The rank is at most $r$, because every column of $BA$ is a combination of the $r$ columns of $B$. The update is limited to an $r$-dimensional column space, however large $d$ and $k$ are, and that limit is the assumption LoRA makes about adaptation.'
    see: matrix-rank
  - q: 'In the Transformer, which matrices does the LoRA paper adapt, and which does it leave frozen?'
    a: 'It adapts the self-attention projections ($W_q$, $W_k$, $W_v$, $W_o$) and freezes the MLP, the layer norms, and the biases (Section 4.2). Most of its experiments adapt only $W_q$ and $W_v$.'
    see: transformer
---

## The problem

Fine-tuning updates every weight of a pretrained model. That gives the best quality, but each downstream task then produces a new full copy of the model. The paper puts it this way: the new model "contains as many parameters as in the original model" (Section 1), and for GPT-3 that means 175 billion parameters per task (Section 2).

Consider a service with 100 customers, each with a task-specific GPT-3.

- **Storage.** A full copy in 16-bit precision takes about 350 GB, so 100 copies take 35 TB. Section 4.2 gives the 350 GB figure for the checkpoint.
- **Training memory.** Adam keeps two extra values for every trainable weight. The paper reports that fine-tuning GPT-3 175B needs 1.2 TB of VRAM, and that LoRA needs 350 GB (Section 4.2).
- **Switching tasks.** Moving a different 350 GB model into GPU memory for each request is slow.

The earlier fixes each give something up (Section 3). Adapter layers, small bottleneck modules inserted into each block, add depth, and the extra layers can't be skipped at inference. Table 1 measures up to 30 percent more latency on GPT-2 medium at batch size 1 and sequence length 128. Prompt-based methods such as prefix tuning take positions away from the usable sequence length, and the paper reports that they are hard to optimize (Section 3). Both families often fail to match full fine-tuning in quality (Section 1).

The paper asks for a method that trains few parameters, matches fine-tuning, and adds no latency at inference. The idea is to constrain the update to each weight matrix to a low rank. Section 5.5 and Table 4 show GPT-3 175B results. LoRA with 4.7 million trainable parameters reaches 73.4 percent on WikiSQL, against 73.8 percent for full fine-tuning with 175,255.8 million, and it beats fine-tuning on MNLI-matched (91.7 against 89.5 percent).

## Key ideas

The sections that follow use the paper's notation. A pretrained weight is $W_0 \in \mathbb{R}^{d \times k}$ and its update is $\Delta W = BA$, with $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and $r \ll \min(d, k)$ (Section 4.1). The paper writes $x$ as a column vector. The exercises use a batch $X \in \mathbb{R}^{n \times k}$ with one example per row, so the code computes `X @ W0.T`. That is a transposition of the same formula, and the paper doesn't write it. Every number comes from a script in `scripts/`. Run any script from the paper folder to reproduce the output.

### Freeze W and learn a low-rank update

**Intuition.** Fine-tuning moves a weight matrix from $W_0$ to $W_0 + \Delta W$. LoRA bets that the useful $\Delta W$ is simple: it points along only a few directions. If that is true, you can store those few directions and skip the rest.

**Math.** Section 2 defines the general setup. Full fine-tuning maximizes the conditional language-modeling objective over all parameters $\Phi$ (Equation 1). The parameter-efficient version encodes the change as $\Delta\Phi = \Delta\Phi(\Theta)$ with $|\Theta| \ll |\Phi_0|$ and maximizes over $\Theta$ only (Equation 2). LoRA picks the following form for each adapted matrix (Section 4.1):

$$
W_0 + \Delta W = W_0 + BA
$$

$W_0$ receives no gradient. Both $W_0$ and $BA$ multiply the same input, and the outputs add (Section 4.1, Equation 3):

$$
h = W_0 x + \Delta W x = W_0 x + BAx
$$

Compute $Ax$ first. That gives an $r$-vector, and $B$ expands it to $d$ entries. The code never needs to build the $d \times k$ matrix $BA$.

The paper also scales the update by $\alpha / r$, where $\alpha$ is a constant in $r$ (Section 4.1): $h = W_0 x + \frac{\alpha}{r} BAx$. The paper says this scaling "helps to reduce the need to retune hyperparameters when we vary $r$", and that with Adam, tuning $\alpha$ is roughly the same as tuning the learning rate. It sets $\alpha$ to the first $r$ it tries and doesn't tune it. Equation 3 omits the scale factor, and this page and the exercises include it.

**Tiny example, computed by `scripts/lora_example.py`.** Take $d = 2$, $k = 3$, $r = 1$, and

$$
W_0 = \begin{bmatrix} 1 & 0 & 2 \\ 0 & 1 & 1 \end{bmatrix}, \quad
B = \begin{bmatrix} 1 \\ 2 \end{bmatrix}, \quad
A = \begin{bmatrix} 1 & 0 & -1 \end{bmatrix}, \quad
x = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}.
$$

The steps and results are as follows:

| Step | Result |
|---|---|
| $W_0 x$ | `[7, 5]` |
| $Ax$ | `[-2]` |
| $B(Ax)$ | `[-2, -4]` |
| $h = W_0 x + BAx$ | `[5, 1]` |
| $\Delta W = BA$ | `[[1, 0, -1], [2, 0, -2]]`, rank 1 |

With $\alpha = 2$ and $r = 1$, the scale is 2 and the output is `[3, -3]`, because $[7, 5] + 2 \cdot [-2, -4]$.

Exercise 1 asks you to write this layer.

### Initialize B at zero so training starts at the base model

**Intuition.** A new adapter must not damage the pretrained model before it has learned anything. Zero in one factor makes the update exactly zero, and the layer starts as the base layer.

**Math.** Section 4.1 uses a random Gaussian initialization for $A$ and zero for $B$, "so $\Delta W = BA$ is zero at the beginning of training." Figure 1 labels the two factors $B = 0$ and $A = \mathcal{N}(0, \sigma^2)$. Zero in only one factor is enough. The model output starts equal to the base model, and the gradients still flow, as the next example shows.

Write $s = \alpha / r$ and let $g = \partial L / \partial h$ be the gradient flowing into the layer output. For one example the chain rule gives

$$
\frac{\partial L}{\partial B} = s \, g \,(Ax)^\top, \qquad
\frac{\partial L}{\partial A} = s \, (B^\top g)\, x^\top
$$

The paper doesn't print these. They follow from Equation 3. The gradient for $A$ contains $B$. At initialization $B = 0$, so $A$ gets exactly zero gradient and only $B$ moves in the first step. After that step $B$ is nonzero and $A$ starts to learn. The converse fails: if $A$ started at zero too, then $\partial L / \partial B = s\,g\,(Ax)^\top$ would be zero, and neither factor could ever move.

**Tiny example, computed by `scripts/lora_example.py`.** Use the matrices above with $\alpha = 2$, $r = 1$, so $s = 2$, and take $g = [1, 1]$. Then $Ax = -2$, and:

| Case | $\partial L / \partial B$ | $\partial L / \partial A$ |
|---|---|---|
| $B = 0$ | `[[-4], [-4]]` | `[[0, 0, 0]]` |
| $B = [1, 2]^\top$ | `[[-4], [-4]]` | `[[6, 12, 18]]` |

The second row of the right column is $s \cdot (B^\top g) \cdot x^\top = 2 \cdot 3 \cdot [1, 2, 3]$. Exercise 3 asks you to write both gradients and check them against finite differences.

### Merge the update into W at inference for zero overhead

**Intuition.** The update $BA$ is just a $d \times k$ matrix, the same shape as $W_0$. After training, add it into the weight once. The deployed model has the original architecture and no extra layer.

**Math.** Section 4.1 ("No Additional Inference Latency") says to compute and store $W = W_0 + BA$ and run inference as usual. To switch tasks, subtract $BA$ to recover $W_0$, then add a different $B'A'$:

$$
W = W_0 + \tfrac{\alpha}{r} BA, \qquad W' = W - \tfrac{\alpha}{r} BA + \tfrac{\alpha}{r} B'A'
$$

The scale factor $\alpha / r$ goes into the merge, because the forward pass includes it. The paper names a limitation. After merging, you can't batch inputs for different tasks through one forward pass. The alternative is to keep the adapters unmerged and select the right one per sample when latency isn't critical (Section 4.2).

**Tiny example, computed by `scripts/lora_example.py` and `scripts/param_counts.py`.** With the matrices from the first example ($\alpha = r = 1$), the merged weight is

$$
W = \begin{bmatrix} 2 & 0 & 1 \\ 2 & 1 & -1 \end{bmatrix}, \qquad Wx = [5, 1],
$$

which equals the unmerged output. Subtracting $BA$ gives back $W_0$ exactly. Swapping in $B' = [0, 1]^\top$ and $A' = [2, 0, 0]$ gives $W' = \begin{bmatrix} 1 & 0 & 2 \\ 2 & 1 & 1 \end{bmatrix}$.

What does skipping the merge cost? For GPT-3's $12288 \times 12288$ matrices at $r = 4$, an unmerged adapter adds 98,304 multiply-adds per token per matrix, which is 0.065 percent of the 150,994,944 multiply-adds of the matrix itself. The count is small. The paper's latency argument in Section 3 is about sequential kernel launches at small batch sizes, not about the arithmetic. Merging removes those extra steps entirely.

Exercise 2 asks you to write merge, unmerge, and swap.

### Count the parameters, and why a low rank is enough

**Intuition.** The factorization trades rank for size. A $d \times k$ update needs $dk$ numbers. A rank-$r$ update needs $r(d + k)$. For GPT-3, that is thousands of times fewer weights.

**Math.** Section 4.2 applies LoRA to the self-attention projections only, and freezes the MLP, the layer norms, and the biases. It treats each of $W_q$, $W_k$, $W_v$ as one $d_{\text{model}} \times d_{\text{model}}$ matrix, even though the output is sliced into heads. The number of trainable weights for $L$ layers with $m$ adapted matrices per layer, each $d \times d$, is

$$
|\Theta| = L \cdot m \cdot r \cdot 2d
$$

The paper reports the following results:

- Section 4.2 reports a checkpoint reduction of roughly 10,000 times (350 GB to 35 MB) for $r = 4$ with $W_q$ and $W_v$ adapted, and a training-VRAM drop from 1.2 TB to 350 GB.
- Section 7.1 and Table 5 fix a budget of 18 million parameters on GPT-3. Adapting $W_q$ and $W_v$ at $r = 4$ gives 73.7 percent on WikiSQL and 91.3 percent on MultiNLI. Adapting $W_q$ alone at $r = 8$ gives 70.4 and 91.0. The paper's conclusion is that "it is preferable to adapt more weight matrices than adapting a single type of weights with a larger rank."
- Section 7.2 and Table 6 show that rank barely matters. For $\{W_q, W_v\}$ on WikiSQL, $r = 1$ gives 73.4 percent and $r = 8$ gives 73.8. The paper reads this as evidence that $\Delta W$ has a very small "intrinsic rank," and adds that a small $r$ may not work for every task (footnote 6).
- Section 7.3 and Table 7 compare $\Delta W_q$ with $W_q$ at $r = 4$. The update amplifies directions that $W_q$ doesn't emphasize, by a factor of about 21.5 ($6.91 / 0.32$).

**Tiny example, computed by `scripts/param_counts.py` and `scripts/rank_example.py`.** For one $12288 \times 12288$ matrix, $r = 4$ trains 98,304 weights against 150,994,944, which is 1,536 times fewer. For all 96 layers with $W_q$ and $W_v$ adapted, the totals are:

| $r$ | Trainable weights | Share of 175B | Size in FP16 |
|---|---|---|---|
| 1 | 4,718,592 | 0.0027% | 9.4 MB |
| 4 | 18,874,368 | 0.0108% | 37.7 MB |

The 18.9 million for $r = 4$ matches the paper's 18M budget in Section 7.1, and the 0.0108 percent is consistent with the paper's "as small as 0.01%" in Section 2. The 37.7 MB matches the paper's roughly 35 MB. The $r = 1$ row equals the 4.7M budget in Table 4, which Appendix D.4 describes as $r_q = r_v = 1$.

A rank-$r$ product can't hold more than $r$ directions. Take $\Delta W = \operatorname{diag}(3, 1, 0, 0)$, which has rank 2. The best rank-1 update keeps only the 3, and its Frobenius error is exactly 1, against a total norm of 3.1623. Rank 2 reproduces $\Delta W$ exactly. This is the Eckart–Young result for the best low-rank approximation, a standard fact the paper doesn't cite. Exercise 4 shows the same thing in a trained layer.

Exercise 4 fits a frozen random $12 \times 12$ layer to a new target whose update has rank 2. `scripts/fit_frozen_layer.py` reports the following results:

| $r$ | Trainable weights | Final loss |
|---|---|---|
| 1 | 24 | 0.955 |
| 2 | 48 | 0.000000 |
| 4 | 96 | 0.000000 |

The starting loss is 2.8468. A full update would train 144 weights. Rank 1 stalls above 0.9 because the target has two directions, and rank 2 fits exactly.

<!-- exercises -->

## Pitfalls

- **Initializing $B$ with noise.** The model then starts away from the base model. Zero in $B$ and Gaussian in $A$ is the paper's choice (Section 4.1). Zero in both factors blocks all gradients.
- **Forgetting the $\alpha / r$ scale in the merge.** The merged weight must use the same scale as the forward pass. A merge with $BA$ only gives different outputs whenever $\alpha \ne r$, and the unmerged forward pass still looks right.
- **Treating $\alpha$ as a learning rate you can ignore.** The paper says tuning $\alpha$ is roughly the same as tuning the learning rate (Section 4.1), and that it fixes $\alpha$ to the first $r$ it tries. If you copy a learning rate from another recipe, check the value of $\alpha / r$ as well.
- **Building $BA$ in the forward pass during training.** It gives the right result and costs $O(dk)$ per token for the update, which defeats the compute savings. Compute $B(Ax)$.
- **Expecting a large $r$ to help.** Table 6 shows no gain from $r = 8$ or $r = 64$ over $r = 4$ on the paper's GPT-3 tasks. Spend the budget on more weight matrices (Table 5), and raise $r$ only when the task needs it.
- **Reading the results as a proof that updates are low rank.** Section 7 supports the hypothesis with experiments on GPT-3 175B and two datasets, and footnote 6 says a small $r$ may not work for every task.
- **Merging and then batching several adapters.** A merged weight serves one task. To serve many adapters in one batch, keep them unmerged and pay the extra operations (Section 4.2).
- **Expecting full optimizer savings on the whole model.** The VRAM saving comes from skipping gradients and optimizer states for frozen weights. The frozen weights still sit in memory.
- **Assuming other libraries use the paper's recipe exactly.** The paper describes a Gaussian $A$. Other implementations, including the official `loralib` code, pick their own initialization for $A$, so check the source if you must reproduce a number.

## Further reading

- [microsoft/LoRA](https://github.com/microsoft/LoRA) is the authors' package, with the code and checkpoints for RoBERTa, DeBERTa, and GPT-2.
- [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://arxiv.org/abs/2012.13255), by Aghajanyan et al., is the work that inspired the low-rank hypothesis (Section 4.1).
- [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751), by Houlsby et al., is the adapter design that the paper compares against in Section 3.
- [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190), by Li and Liang, is the prompt-based alternative.
- [Hugging Face PEFT](https://github.com/huggingface/peft) is a widely used library that implements LoRA for many model families.
