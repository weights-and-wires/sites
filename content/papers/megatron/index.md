---
id: "megatron"
title: "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"
authors: "Shoeybi et al."
year: 2019
venue: "arXiv preprint"
arxiv: "1909.08053"
url: "https://arxiv.org/abs/1909.08053"
track: gpu-systems
tier: core
est_hours: 5
status: full
summary: "Megatron-LM splits each Transformer layer across GPUs so that a model too large for one GPU can still train. It cuts the first matrix multiplication of each block by columns and the second by rows, so the nonlinearity and the attention heads run on each GPU with no communication. Each layer then needs only two all-reduces in the forward pass and two in the backward pass, and the authors train an 8.3 billion parameter model on 512 GPUs at 76 percent scaling efficiency."
related_course: "gpu: distributed training"
prereqs:
  - id: transformer
    reason: "Megatron-LM splits the Transformer's multi-head attention and feed-forward blocks, so you need their weight shapes and the order of operations inside a layer."
background: [collective-communication, all-reduce, matrix-partitioning]
course_links:
  - title: "Tensor parallelism"
    url: "https://qwen.ifkash.dev/c/19-tensor-parallelism"
  - title: "Overlap and parallelism"
    url: "https://gpu.ifkash.dev/c/22-overlap-and-parallelism"
self_check:
  - q: 'You split $A$ by rows across 2 GPUs in $Y = \operatorname{GeLU}(XA)$, so GPU 1 computes $X_1A_1$ and GPU 2 computes $X_2A_2$. What must happen before the GeLU, and why does the paper split $A$ by columns instead?'
    a: 'The two partial products must be summed first, because $\operatorname{GeLU}(X_1A_1 + X_2A_2) \ne \operatorname{GeLU}(X_1A_1) + \operatorname{GeLU}(X_2A_2)$. For one pair of partial sums, $3$ and $-4$, the left side is $-0.1588$ and the right side is $2.9963$. That sum is a synchronization point (an all-reduce). A column split gives each GPU complete pre-activations for its own columns, so the GeLU runs locally and the synchronization point goes away (Section 3, Equation 3).'
    see: matrix-partitioning
  - q: 'For $b = 8$, $s = 1024$, and padded vocabulary $v = 51{,}200$, how many elements does an all-gather of the output logits move, and how many does the fused cross-entropy approach move?'
    a: 'The all-gather moves $b \cdot s \cdot v = 419{,}430{,}400$ elements. Fusing the cross-entropy loss into the parallel GEMM leaves only $b \cdot s = 8{,}192$ elements to communicate, which is smaller by a factor of $v$ (Section 3).'
    see: collective-communication
  - q: 'A layer has $d_{\text{model}} = 1024$ and 16 heads, and runs on 4 GPUs. How many heads does each GPU hold, and what is the shape of each GPU''s shard of $W^Q$?'
    a: 'Each GPU holds $16 / 4 = 4$ heads. Each head has width $d_k = 1024 / 16 = 64$, so a shard of $W^Q$ has shape $1024 \times 256$, which is 4 heads of 64 columns each. The shards of $W^K$ and $W^V$ have the same shape, and the matching shard of $W^O$ is $256 \times 1024$.'
    see: transformer
---

## The problem

Suppose you want to train an 8.3 billion parameter Transformer. In 2019 the standard recipe was data parallelism: every GPU keeps a full copy of the model and processes a different slice of the batch. That recipe fails here for a memory reason, not a speed reason. Data parallelism requires the whole model to fit on one worker (Section 2.3), and Adam stores extra memory per parameter for its momentum and other state (Section 1). The 8.3 billion parameter configuration needs 8-way model parallelism, while the largest configuration that fits on one 32 GB V100 in the authors' setup has 1.2 billion parameters (Section 5.1).

Two ways of splitting the model existed.

- **Pipeline parallelism** puts different layers on different GPUs. It needs extra scheduling logic, and it leaves GPUs idle in "pipeline bubbles" (Section 2.3).
- **Distributed tensor computation** splits each operation across GPUs. Mesh-TensorFlow did this with a new language and compiler, so you had to rewrite your model (Section 2.3).

The paper asks whether you can split every layer of an existing PyTorch Transformer by adding a few communication calls, with no compiler and no rewrite. The answer is yes, and the cost is two all-reduces per layer in each direction. The authors sustain 15.1 petaFLOPs on 512 GPUs for an 8.3 billion parameter model. They measure it against a single-GPU baseline of 1.2 billion parameters at 39 teraFLOPs, which is 30 percent of the hardware peak (Section 5.1).

## Key ideas

The sections that follow use the paper's notation: an input $X$, a weight matrix $A$, an activation $Y = \operatorname{GeLU}(XA)$, hidden size $H$ (written $h$ below), batch size $b$, sequence length $s$, vocabulary size $v$, and $t$ GPUs. The paper writes the number of GPUs in prose and doesn't give it a symbol. The paper doesn't name the second MLP weight matrix or the block output, so this page calls them $B$ and $Z = YB$. Every number comes from a script in `scripts/`. Run any script from the paper folder to reproduce the output.

The paper's own equations are few. Equations 1 to 3 in Section 3 cover the MLP split, and the rest of the method is described in prose and in Figures 3 and 4.

### Column-parallel then row-parallel linear layers

**Intuition.** A Transformer MLP is two matrix multiplications with a nonlinearity between them. You can split a matrix product across GPUs in two ways: cut the weight by columns, or cut it by rows. One choice forces an all-reduce right after the first product. The other choice does not, and you take it.

**Math.** The first part of the MLP block is a GEMM followed by GeLU (Section 3, Equation 1):

$$
Y = \operatorname{GeLU}(XA)
$$

*Option 1, split by rows.* Cut $X$ by columns and $A$ by rows (Equation 2):

$$
X = [X_1, X_2], \qquad A = \begin{bmatrix} A_1 \\ A_2 \end{bmatrix}
$$

Then $XA = X_1A_1 + X_2A_2$, so $Y = \operatorname{GeLU}(X_1A_1 + X_2A_2)$. GeLU is nonlinear, so $\operatorname{GeLU}(X_1A_1 + X_2A_2) \ne \operatorname{GeLU}(X_1A_1) + \operatorname{GeLU}(X_2A_2)$, and the GPUs must sum their partial products before they apply GeLU. That is a synchronization point.

*Option 2, split by columns.* Cut $A$ into $A = [A_1, A_2]$ (Equation 3):

$$
[Y_1, Y_2] = [\operatorname{GeLU}(XA_1),\ \operatorname{GeLU}(XA_2)]
$$

Every GPU receives the full $X$ and computes the complete pre-activation for its own columns. GeLU runs locally with no communication.

The paper then splits the second GEMM by rows, so it consumes $Y_i$ directly. With $B = \begin{bmatrix} B_1 \\ B_2 \end{bmatrix}$, GPU $i$ computes the partial output $Z_i = Y_i B_i$, and the true output is $Z = Z_1 + Z_2$. A single all-reduce computes that sum, before the result goes to dropout (Section 3, Figure 3a). In the paper's words, the block "requires only a single all-reduce operation in the forward pass" (the $g$ operator).

**Tiny example, computed by `scripts/mlp_split.py`.** First the bad option. Take $X = [1, 2]$ and $A = [3, -2]^\top$, and split by rows. GPU 1 computes $3$ and GPU 2 computes $-4$, and the true pre-activation is $-1$. Then $\operatorname{GeLU}(-1) = -0.1588$, but $\operatorname{GeLU}(3) + \operatorname{GeLU}(-4) = 2.9963$. Skipping the synchronization gives the wrong answer. This page uses the tanh approximation of GeLU in every script and exercise. The paper doesn't say which form it uses.

Now the good option, on 2 GPUs, with 2 tokens, hidden size 2, and inner width 4:

$$
X = \begin{bmatrix} 1 & 2 \\ 0 & -1 \end{bmatrix}, \quad
A = \begin{bmatrix} 1 & -1 & 0.5 & 2 \\ 0 & 1 & -2 & 1 \end{bmatrix}, \quad
B = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \\ -1 & 2 \end{bmatrix}
$$

GPU 1 takes the first two columns of $A$ and the first two rows of $B$. GPU 2 takes the rest.

| Quantity | Value |
|---|---|
| Single-device $Z$ | `[[-3.1594, 8.8404], [2.1134, 1.4782]]` |
| GPU 1 partial $Z_1$ | `[[0.8412, 0.8412], [0.0, -0.1588]]` |
| GPU 2 partial $Z_2$ | `[[-4.0005, 7.9992], [2.1134, 1.637]]` |
| All-reduce $Z_1 + Z_2$ | `[[-3.1594, 8.8404], [2.1134, 1.4782]]` |

The partials are full-sized, `(tokens, h)`, but neither equals $Z$. The all-reduce completes them. On a random 4-way example with $h = 8$, the largest difference from the single-device result is $7 \times 10^{-15}$, which is floating-point rounding, and one GPU's partial alone is off by 21.7.

Exercise 1 asks you to build this block and to check that it calls the reduction exactly once.

### One all-reduce per block, in each of forward and backward

**Intuition.** The forward pass needs a sum of partial outputs. The backward pass needs a sum of partial input gradients, for the mirror-image reason. The paper wraps each of the two communications in an operator, so the model code stays unchanged.

**Math.** The paper defines two conjugate operators (Section 3, Figure 3):

- $f$ is the identity in the forward pass and an all-reduce in the backward pass.
- $g$ is an all-reduce in the forward pass and the identity in the backward pass.

Code 1 shows $f$ as a PyTorch `autograd.Function` of a few lines. The MLP block places $f$ at the input of the column-parallel GEMM and $g$ after the row-parallel GEMM.

To see why $f$ all-reduces in the backward pass, use the column split. Every GPU holds the same $X$, so the gradient with respect to $X$ is the sum of what each GPU's columns contribute. With $dY_i = \partial L / \partial Y_i$ on GPU $i$ and $H_i = XA_i$:

$$
\frac{\partial L}{\partial X} = \sum_i \frac{\partial L}{\partial H_i} A_i^\top
$$

Each GPU computes only its own term, so the terms must be summed, which is an all-reduce. The weight gradients need no communication, because $dA_i = X^\top\, \partial L / \partial H_i$ uses only local data, and the row-parallel $dB_i = Y_i^\top\, dZ$ does too. This page uses $dX$ and $dA$ for gradients. The paper doesn't derive these expressions; they follow from the chain rule.

**Tiny example, computed by `scripts/backward_fg.py`.** Take $X = [1, 2]$ and a column-split $A$ on 2 GPUs. With the incoming gradient $dY = [1, 2, -1, 0.5]$, GPU 1 computes the local gradient `[1.0, 5.0]` and GPU 2 computes `[-2.5, 0.5]`. Neither is the answer. Their sum is `[-1.5, 5.5]`, which equals $dY\,A^\top$ on a single device. The weight-gradient blocks $dA_1$ and $dA_2$ concatenate to the single-device $dA$ with no communication.

**Counting per layer.** The attention block (next idea) has the same structure, so a whole Transformer layer needs 2 all-reduces in the forward pass and 2 in the backward pass, 4 in total (Section 3, Figure 4). The layer norms, dropout, and residual additions sit between the parallel regions. The paper does not split them. It copies the layer-norm parameters onto every GPU and has every GPU compute the same result, because duplicating that cheap work costs less than communicating it (Section 3). The optimizer then updates each GPU's own parameters, and no updated values need to cross the network.

Exercise 2 asks you to implement the backward pass and apply $f$.

### Split attention heads across devices

**Intuition.** Multi-head attention already is a set of independent computations. Head $i$ reads its own columns of $W^Q$, $W^K$, and $W^V$, and no head reads another head's scores. You can hand a few whole heads to each GPU.

**Math.** Section 3 and Figure 3b partition the $Q$, $K$, and $V$ projections in a column-parallel fashion, "such that the matrix multiply corresponding to each attention head is done locally on one GPU". Each GPU computes attention for its own heads with no communication. The output projection $W^O$ is then split by rows. It takes each GPU's concatenated head outputs directly, and the GPU's partial result is summed with an all-reduce. The notation follows the [Transformer](/transformer/) page. With $h_{\text{heads}}$ heads and $t$ GPUs, GPU $r$ holds $h_{\text{heads}} / t$ heads:

$$
Z = \sum_{r=1}^{t} \operatorname{Concat}\big(\operatorname{head}_i : i \in \text{GPU } r\big)\, W^O_r
$$

Here $W^O_r$ is the block of rows of $W^O$ that matches GPU $r$'s heads. The number of heads must be divisible by $t$. The paper doesn't write this formula; it describes the partition in words.

**Tiny example, computed by `scripts/heads_split.py`.** Take $d_{\text{model}} = 16$, 4 heads, so $d_k = 4$, and 2 GPUs. GPU $r$ takes columns 0 to 7 (GPU 0) or 8 to 15 (GPU 1) of $W^Q$, $W^K$, and $W^V$. Each shard is $16 \times 8$, and the matching shard of $W^O$ is $8 \times 16$. The sum of the two partial outputs matches the single-device multi-head attention to within $4 \times 10^{-16}$.

One detail trips people up. Each GPU holds 2 heads of width 4, so the local width is 8. The scale in each head is still $1/\sqrt{d_k} = 0.5$. Dividing by the square root of the local width gives $0.3536$ and silently changes the model.

For the paper's 8.3 billion parameter model, with $h = 3072$, attention holds $4h^2 = 37{,}748{,}736$ weights per layer and each of 8 GPUs holds $4{,}718{,}592$ of them.

The paper also reports that more heads slightly lower efficiency, because the per-head GEMMs shrink and the softmax grows (Appendix D.1, Table 7). With 8-way parallelism on the 8.3B model, 16 heads reach 82 percent scaling efficiency, 24 heads reach 80 percent, and 32 heads reach 77 percent.

Exercise 3 asks you to build the head-parallel attention.

### The embeddings and the output layer

**Intuition.** The output embedding is an $H \times v$ matrix with $v$ in the tens of thousands, so it is worth splitting too. It shares weights with the input embedding, so both sides change together.

**Math.** The paper splits the embedding $E_{H \times v}$ along the vocabulary dimension, $E = [E_1, E_2]$ (Section 3). Each GPU holds a slice of the table, so the input side needs an all-reduce, the $g$ operator, after the lookup. On the output side, the plain approach computes the parallel GEMM $[Y_1, Y_2] = [XE_1, XE_2]$ and runs an all-gather to assemble the logits. That all-gather moves $b \times s \times v$ elements. The paper instead fuses the parallel GEMM with the cross-entropy loss, which reduces the data to $b \times s$ elements, "scalar losses instead of logits".

The paper also pads the vocabulary. GPT-2's vocabulary has 50,257 tokens. For efficient GEMMs, each GPU's slice should be a multiple of 128, so the vocabulary is padded to a multiple of $128 \times 8 = 1024$, giving 51,200 (Section 5.1).

**Tiny example, computed by `scripts/comm_volume.py`.** For $b = 8$, $s = 1024$, and $v = 51{,}200$:

| Output-layer approach | Elements communicated |
|---|---|
| All-gather the logits ($b \cdot s \cdot v$) | 419,430,400 |
| Fused cross-entropy ($b \cdot s$) | 8,192 |

The fused version moves 51,200 times fewer elements.

### Counting communication per layer

**Intuition.** Tensor parallelism pays for its memory savings in bandwidth. Four all-reduces per layer is small in a count, but every one moves an activation tensor of $b \times s \times h$ elements, which is far bigger than a gradient shard. That is why the authors run model parallelism inside one DGX-2H server, where GPUs share 300 GB/s of NVSwitch bandwidth, and use the slower 100 GB/s InfiniBand links between servers for data parallelism (Section 5, Appendix B.1).

**Math.** Per layer, forward and backward together, the all-reduced volume is

$$
V_{\text{layer}} = 4 \cdot b \cdot s \cdot h \ \text{elements}
$$

A bandwidth-optimal ring all-reduce over $t$ GPUs makes each GPU send and receive $2(t-1)/t$ times the tensor size. The paper doesn't give this factor. It is the standard ring cost, and the real cost depends on the NCCL algorithm and the topology.

**Tiny example, computed by `scripts/comm_volume.py`.** Use the paper's 8.3B configuration from Table 1: $h = 3072$, 72 layers, 8 GPUs, with $b = 8$, $s = 1024$, and fp16 activations at 2 bytes per element.

- One all-reduced tensor holds $8 \cdot 1024 \cdot 3072 = 25{,}165{,}824$ elements, which is 48 MiB.
- One layer all-reduces $4 \times 25{,}165{,}824 = 100{,}663{,}296$ elements, which is 192 MiB.
- With the ring factor $2 \cdot 7/8 = 1.75$, each GPU moves 336 MiB per layer, or 23.62 GiB across 72 layers, for one forward and backward pass.
- The parameter count checks out: $72 \cdot 12h^2 + vh = 8{,}311{,}013{,}376$, which is the paper's 8.3 billion. Here $12h^2$ is the weight count of one layer, $4h^2$ in attention and $8h^2$ in the MLP. The count ignores biases and layer norms.

Exercise 4 asks you to write these counting functions.

### Combining with data parallelism, and what the paper measured

**Intuition.** Tensor parallelism and data parallelism multiply. Inside a group of 8 GPUs, the model is split. Across 64 such groups, each group processes different data.

**Math.** Appendix B.1 describes the layout. GPUs in one model-parallel group split one copy of the model. GPUs at the same position in each group hold the same parameters and form a data-parallel group, and gradients are all-reduced within it. The total number of GPUs is the product, $8 \times 64 = 512$ for the 8.3B model.

The scaling results in Section 5.1 use a fixed per-GPU size of about 1 billion parameters (weak scaling), with the hidden size per head fixed at 96. The baseline is a 1.2 billion parameter model on one GPU at 39 teraFLOPs, which is 30 percent of peak.

| Model parameters (Table 1) | GPUs (model parallel) | GPUs (model + data parallel) |
|---|---|---|
| 1.2 billion | 1 | 64 |
| 2.5 billion | 2 | 128 |
| 4.2 billion | 4 | 256 |
| 8.3 billion | 8 | 512 |

Against that baseline, the 8.3 billion parameter model on 8 GPUs reaches 77 percent of linear scaling, and on 512 GPUs it reaches 74 percent (Section 5.1.1, Figure 5). The abstract and introduction quote 76 percent for the 512-GPU run. The page uses the section's figures and notes the difference, because the paper doesn't explain it. Strong scaling is weaker. A fixed 1.2 billion parameter model speeds up by 1.64 times on 2 GPUs and 2.98 times on 8, because the per-GPU compute shrinks while communication stays (Appendix D.2, Table 8).

### A normalization detail for BERT

The paper's Section 5.3 and Figure 7 report a second result that is not about parallelism. The original BERT architecture degrades as it grows. Rearranging the layer normalization and the residual connection (Figure 7b) trains stably and improves downstream results monotonically with model size. This is the same post-LN versus pre-LN question that the [Transformer](/transformer/) page raises.

<!-- exercises -->

## Pitfalls

- **Splitting the first GEMM by rows.** The result is mathematically correct, but it needs an all-reduce before the GeLU. The column split exists to remove that synchronization.
- **Applying GeLU after the all-reduce.** GeLU belongs on each GPU's own columns, before the second GEMM. $\operatorname{GeLU}(\sum_i Z_i) \ne \sum_i \operatorname{GeLU}(Z_i)$, and neither equals the correct output.
- **Forgetting the backward all-reduce.** Without $f$, each GPU's gradient with respect to $X$ covers only its own columns. The forward pass is correct, and the model trains on wrong gradients.
- **Averaging instead of summing.** Both all-reduces are sums. Data-parallel gradients are often averaged, so the habit carries over by mistake.
- **Splitting $Q$, $K$, and $V$ inconsistently.** A head needs its query, key, and value columns on the same GPU. Slice $W^K$ and $W^V$ with the same indices as $W^Q$, and slice $W^O$ by the matching rows.
- **Scaling by the local width.** Each GPU divides scores by $\sqrt{d_k}$, the width of a single head, and not by the square root of the width of its shard.
- **Replicated and sharded dropout.** Dropout outside the parallel regions must use the same random numbers on every GPU, because that tensor is replicated. Dropout inside the attention block must use different numbers on each GPU. The paper keeps two random-number generators for this reason (Appendix B.2).
- **Treating all-reduce volume as small.** Four all-reduces per layer sounds cheap, but each moves $b \cdot s \cdot h$ elements. The approach relies on NVSwitch-class bandwidth, so extending it across slow links costs much more than the formula suggests.
- **Expecting a linear speedup on a fixed model.** The paper's near-linear numbers are weak scaling, where the model grows with the GPU count. A fixed 1.2 billion parameter model gains 2.98 times on 8 GPUs.
- **Forgetting to pad the vocabulary.** A vocabulary of 50,257 doesn't divide across 8 GPUs, and slices that aren't multiples of 128 slow the output GEMM.

## Further reading

- [Megatron-LM on GitHub](https://github.com/NVIDIA/Megatron-LM) is the official code, listed in the paper's contributions (Section 1).
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473), by Narayanan et al., combines tensor, pipeline, and data parallelism, and adds interleaved pipeline schedules.
- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198), by Korthikanti et al., adds sequence parallelism and turns the layer's all-reduce into a reduce-scatter and an all-gather.
- [ZeRO](/zero/) shards optimizer state, gradients, and parameters across data-parallel workers, a different way to relieve the same memory limit.
- [GPT-2](/gpt2/) and [Mixed precision training](/mixed-precision/) cover the model family and the training recipe that Megatron-LM scales.
