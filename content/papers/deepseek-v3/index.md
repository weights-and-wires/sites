---
id: "deepseek-v3"
title: "DeepSeek-V3 Technical Report"
authors: "DeepSeek-AI et al."
year: 2024
venue: "arXiv preprint"
arxiv: "2412.19437"
url: "https://arxiv.org/abs/2412.19437"
track: architectures
tier: core
est_hours: 5
status: full
summary: "DeepSeek-V3 is a 671B-parameter mixture-of-experts language model that activates 37B parameters per token and trains on 14.8T tokens in 2.788M H800 GPU hours. It keeps the MLA attention and DeepSeekMoE layers of DeepSeek-V2 and adds three things: expert load balancing with a routing-only bias instead of an auxiliary loss, a multi-token prediction training objective, and FP8 mixed-precision training with tile-wise and block-wise scaling. The report prices the full run at $5.576M, assuming $2 per GPU hour."
related_course: "ml: frontier training recipe"
prereqs:
  - id: deepseek-v2
    reason: "DeepSeek-V3 reuses MLA and the DeepSeekMoE layer from DeepSeek-V2 unchanged, so its cache size, shared and routed experts, and node-limited routing are all read against that paper."
  - id: fp8
    reason: "DeepSeek-V3 trains its matrix multiplies in E4M3 with scaling factors, so you need the format, its saturation behavior, and the idea of scaling a tensor into range before a cast."
background: [multi-token-prediction, load-balancing]
course_links:
  - title: "Mixtures of experts: routing, the router's gradient, and balance"
    url: "https://ml.ifkash.dev/lessons/060-mixtures-of-experts"
  - title: "Reasoning with verifiable rewards"
    url: "https://rl.ifkash.dev/lessons/057-reasoning-with-verifiable-rewards"
self_check:
  - q: 'In auxiliary-loss-free balancing, the bias $b_i$ decides which experts a token visits. Why does the paper keep it out of the gating value that multiplies the expert output?'
    a: 'The paper says the bias "is only used for routing" (Section 2.1.2). The gating value still comes from the original affinity $s_{i,t}$ (Equations 13 and 16). The bias then changes which experts see a token, and it doesn''t change how much an expert that is chosen contributes to the output.'
    see: load-balancing
  - q: 'The FP8 paper recommends E4M3 for weights and activations and E5M2 for gradients. DeepSeek-V3 uses E4M3 for every tensor. What does the report credit for making that work?'
    a: 'Fine-grained scaling (Section 3.3.2, "Mantissa over Exponents"). Each 1x128 activation tile or 128x128 weight block has its own scale, so the group effectively shares exponent bits and the limited range of E4M3 matters less than it does under one scale for the whole tensor.'
    see: fp8
  - q: 'A sequence has tokens $t_1, \ldots, t_7$, so $T = 6$. Which token does position 3 predict in the main model, which in MTP module 1, and what happens to the MTP modules at inference?'
    a: 'The main model predicts $t_4$ and module 1 predicts $t_5$ (Section 2.2, Equations 22-24). The MTP modules can be discarded at inference so the main model runs alone, or reused for speculative decoding (Section 2.2, "MTP in Inference").'
    see: multi-token-prediction
---

## The problem

DeepSeek-V2 showed that attention with a compressed KV cache and a fine-grained mixture-of-experts feed-forward layer make a large model cheap to run and to train. DeepSeek-V3 scales that design to 671B total parameters, of which 37B are activated for each token, and trains it on 14.8T tokens (Section 1, Section 4.2). Three costs get in the way at that scale, and the report answers each of them.

**Balancing the experts costs quality.** An MoE router can send most tokens to a few experts. The report calls this routing collapse, and in expert-parallel training an overloaded expert makes every other GPU wait (Section 2.1.2). The usual fix is an auxiliary loss, but "too large an auxiliary loss will impair the model performance". The loss pulls the router toward balance and away from what the language-modeling loss wants.

**Training is expensive.** The report's budget is 2.788M H800 GPU hours for the whole run (Table 1). Most of that is arithmetic in the matrix multiplies, and 16-bit arithmetic leaves the speedup of 8-bit hardware unused. The [FP8](/fp8/) page shows why a plain 8-bit cast is risky: one outlier forces a scale that crushes the other values.

**Each token teaches little.** A next-token objective gives one training signal per position. The report adds a second one, predicting a further token (Section 2.2).

The architecture is mostly inherited. Section 2.1.1 repeats the MLA equations of [DeepSeek-V2](/deepseek-v2/), with different numbers. With $n_h = 128$, $d_h = 128$, $d_c = 512$, $d_h^R = 64$, and 61 layers (Section 4.2), MLA caches 35,136 elements per token against 1,998,848 for multi-head attention, a ratio of 56.9. Both numbers come from `scripts/config_counts.py`. The same script counts the experts: 58 of the 61 layers are MoE layers, each with 1 shared and 256 routed experts, and a token uses 9 of those 257. An expert is a SwiGLU feed-forward network of width 2048, so it has $3 \cdot 7168 \cdot 2048 = 44{,}040{,}192$ weights. The experts alone then hold 656.5B parameters and the activated ones 23.0B. These are my estimates, not the paper's. The remainder of the 671B and the 37B is attention, embeddings, the three dense layers, and the MTP module, which the report doesn't itemize.

This page covers what is new in V3: the router (Section 2.1.2), the MTP objective (Section 2.2), the FP8 framework (Section 3.3), and the cost (Table 1). It skips DualPipe, the all-to-all kernels, the inference deployment, the data, and post-training.

## Key ideas

The notation follows the paper. Token $t$ has an FFN input $u_t$, and the routed experts are indexed by $i = 1, \ldots, N_r$. This page writes $T$ for the sequence length, as the paper does. In code, arrays are 0-indexed, which the exercises note. Where the page uses its own symbol, it says so.

### Balance experts with a routing-only bias

**Intuition.** An auxiliary loss changes the gradients, so balance fights with accuracy. A bias doesn't touch the gradients at all. Each expert has a number that is added to its affinity when the router ranks experts. A popular expert gets a lower number until the ranking spreads the tokens out. The bias is a controller that runs outside the loss.

**Math.** V3 computes the affinity with a sigmoid and normalizes the selected scores, which differs from V2's softmax (Section 2.1.2, Equations 12-15). The change that matters here is Equation 16. A bias $b_i$ for each routed expert is added to the affinity "to determine the top-K routing":

$$
g'_{i,t} =
\begin{cases}
s_{i,t}, & s_{i,t} + b_i \in \operatorname{Topk}\big(\{s_{j,t} + b_j \mid 1 \le j \le N_r\},\, K_r\big),\\
0, & \text{otherwise,}
\end{cases}
$$

and the gating value is still $g_{i,t} = g'_{i,t} / \sum_{j=1}^{N_r} g'_{j,t}$ (Equation 13). The paper says "the bias term is only used for routing." At the end of each training step, the bias of an overloaded expert drops by $\gamma$ and the bias of an underloaded expert rises by $\gamma$. The paper sets $\gamma = 0.001$ for the first 14.3T tokens and 0.0 for the last 500B (Section 4.2). The paper doesn't say how "overloaded" is measured beyond monitoring "the expert load on the whole batch". This page reads it as load above or below the mean load, and uses $b \leftarrow b - \gamma \operatorname{sign}(\text{load} - \overline{\text{load}})$. That formula and the tie rule (no change at exactly the mean) are my reading, not the paper's text.

V3 still keeps a small sequence-wise balance loss, with $\alpha = 0.0001$, "just to avoid extreme imbalance within any single sequence" (Equations 17-20, Section 4.2). With the paper's definitions, perfectly balanced routing gives $f_i = 1$ and $P_i = 1/N_r$, so $\mathcal{L}_{\mathrm{Bal}} = \alpha$. That value is my derivation.

**Tiny example, computed by `scripts/balance_sim.py`.** Four experts, $K_r = 2$, one token with $s = [0.9, 0.8, 0.3, 0.2]$. Without a bias, the token goes to experts 1 and 2 (0-indexed: 0 and 1), with gates $[0.5294, 0.4706]$. With $b = [-0.5, 0, 0.2, 0]$, the ranking scores are $s + b = [0.4, 0.8, 0.5, 0.2]$. The token goes to experts 2 and 3 (0-indexed: 1 and 2), and the gates are computed from $s$ alone: $0.8 / 1.1 = 0.7273$ and $0.3 / 1.1 = 0.2727$.

**A toy simulation, from the same script.** The router has 64 experts, $K_r = 8$, and 2048 tokens per step. Each expert has a fixed random preference, and the affinities are a sigmoid of Gaussian noise plus that preference. The data are made up. The script uses the paper's $\gamma = 0.001$ and measures the coefficient of variation (CV) of the expert load and the maximum load over the mean:

| Step | CV with bias | Max/mean with bias | CV without bias | Max/mean without bias |
|---|---|---|---|---|
| 0 | 1.0835 | 4.281 | 1.0835 | 4.281 |
| 50 | 0.7348 | 3.391 | 1.0735 | 4.227 |
| 100 | 0.4385 | 2.602 | 1.0750 | 4.297 |
| 200 | 0.1271 | 1.172 | 1.0834 | 4.266 |
| 400 | 0.0507 | 1.129 | 1.0845 | 4.191 |

Balance takes a few hundred steps here, because each step moves a bias by only 0.001 while the preferences shift affinities by about 0.1. The bias range at step 400 is $-0.21$ to $0.396$. A real run has thousands of steps, so the delay costs little. Exercise 1 asks you to write the router and the update.

**What the paper measured.** Section 4.5.2 trains two baselines with only auxiliary losses (15.7B parameters on 1.33T tokens, and 228.7B on 578B tokens) and swaps in the bias method. Table 5 shows gains on most benchmarks. At the large scale, Pile-test goes from 0.656 to 0.652 bits per byte, HumanEval from 40.2 to 46.3, GSM8K from 70.7 to 74.5, and MATH from 37.2 to 39.6, while MMLU goes down from 68.3 to 67.2. Section 4.5.3 argues that the gain comes from balancing over a whole batch instead of each sequence. A batch-wise auxiliary loss gets the same validation loss as the bias in the paper's 1B and 3B experiments (2.253 and 2.080), against 2.258 and 2.085 for a sequence-wise loss. The paper lists the weakness of batch-wise balancing too: imbalance inside a small batch and under a domain shift at inference.

### Predict more than one token

**Intuition.** Predicting the token after next forces each hidden state to carry information about the future, and every position gives two training signals. The paper says MTP "densifies the training signals" and "may enable the model to pre-plan its representations" (Section 2.2). It doesn't claim either as proven.

**Math.** Gloeckle et al. predict $D$ extra tokens in parallel with independent heads. V3 predicts them sequentially and "keep[s] the complete causal chain at each prediction depth". There are $D$ modules. Module $k$ owns a Transformer block $\mathrm{TRM}_k$ and a projection $M_k \in \mathbb{R}^{d \times 2d}$, and it shares the embedding layer and the output head with the main model. For token position $i$ at depth $k$, the module combines the previous depth's hidden state with the embedding of token $t_{i+k}$ (Equation 21):

$$
h'^{\,k}_i = M_k\big[\operatorname{RMSNorm}(h^{k-1}_i);\ \operatorname{RMSNorm}(\operatorname{Emb}(t_{i+k}))\big],
$$

where $h^0_i$ is the main model's output. The block and head then give (Equations 22-23)

$$
h^k_{1:T-k} = \mathrm{TRM}_k(h'^{\,k}_{1:T-k}), \qquad P^k_{i+k+1} = \operatorname{OutHead}(h^k_i).
$$

Each depth has a cross-entropy loss, and the total averages them with a weight $\lambda$ (Equations 24-25):

$$
\mathcal{L}^k_{\mathrm{MTP}} = -\frac{1}{T} \sum_{i=2+k}^{T+1} \log P^k_i[t_i], \qquad
\mathcal{L}_{\mathrm{MTP}} = \frac{\lambda}{D} \sum_{k=1}^{D} \mathcal{L}^k_{\mathrm{MTP}}.
$$

V3 sets $D = 1$, with $\lambda = 0.3$ for the first 10T tokens and 0.1 for the remaining 4.8T (Section 4.2). The tokens run $t_1, \ldots, t_{T+1}$, and Equation 24 divides by $T$, not by the number of terms. That is what the paper prints, and this page and the exercise follow it. For depth $k$ it sums $T - k$ terms.

**Tiny example, computed by `scripts/mtp_example.py`.** Take $T = 6$, so the tokens are $t_1, \ldots, t_7$. The main model has 6 positions, and position $i$ predicts $t_{i+1}$. Depth 1 has 5 positions, and position $i$ predicts $t_{i+2}$ (1 predicts $t_3$, and 5 predicts $t_7$). Depth 2 has 4 positions. With uniform predictions over $V = 100$ tokens, every term is $\ln 100$, so depth 1 gives $5/6 \cdot \ln 100 = 3.8376$ and depth 2 gives $4/6 \cdot \ln 100 = 3.0701$. With $\lambda = 0.3$, $D = 2$, $\mathcal{L}^1 = 2$, and $\mathcal{L}^2 = 3$, Equation 25 gives $0.15 \cdot 5 = 0.75$. For Equation 21 with $d = 4$, $M_k$ is $4 \times 8$.

**At inference.** The paper says the modules can be discarded, so the main model runs alone with the same inference cost, or they can drive speculative decoding (Section 2.2). Section 5.4.3 reports that the acceptance rate of the second token is 85% to 90% and the decoding speed is 1.8 times higher. With one extra token, each decoding step yields $1 + p$ tokens in expectation for acceptance rate $p$, which is 1.85 to 1.90 by `scripts/mtp_example.py`. The reported 1.8 is lower, which is consistent with the cost of running the extra module. That reading is mine.

**What the paper measured.** Section 4.5.1 adds a 1-depth module to two baselines (15.7B parameters on 1.33T tokens and 228.7B on 540B tokens). Table 4 shows gains on most benchmarks, and the inference cost is the same. Exercise 3 asks you to write Equations 21, 24, and 25.

### Train in FP8 with fine-grained scaling

**Intuition.** The [FP8](/fp8/) paper scales a whole tensor so that its largest value lands at the format maximum. If one value is far larger than the rest, the scale crushes the rest. V3 gives every small group of values its own scale, so an outlier only hurts its own group.

**Math.** The report has no numbered equations for FP8, so this page states the formulas itself. The notation is mine: $s_g = F_{\max} / \max_{x \in g} |x|$ for a group $g$, with $F_{\max} = 448$ for E4M3, and the stored values are $\operatorname{fl}_8(s_g x)$. The paper's choices are as follows.

- **Where FP8 is used.** All three GEMMs of a linear layer, Fprop, Dgrad, and Wgrad, run in FP8, with outputs in BF16 or FP32 (Section 3.3.1, Figure 6). The embedding, output head, MoE gating, normalization, and attention operators keep BF16 or FP32. The master weights, weight gradients, and optimizer states are in higher precision.
- **Groups.** Activations are scaled per $1 \times 128$ tile, which is "per token per 128 channels". Weights are scaled per $128 \times 128$ block (Section 3.3.2, "Fine-Grained Quantization", Figure 7a). The maximum is computed online for each group, not from a history of earlier steps ("Online Quantization").
- **Format.** E4M3 for all tensors, where the earlier hybrid recipe used E5M2 for gradients. The report credits the fine-grained groups, which "effectively share exponent bits among these grouped elements" ("Mantissa over Exponents").
- **Accumulation.** FP8 GEMMs on the H800 retain about 14 bits of accumulation precision, and the report measures a maximum relative error near 2% for $K = 4096$. Partial sums are copied to FP32 registers on CUDA cores every $N_C = 128$ elements, where the group scales are also applied (Section 3.3.2, "Increasing Accumulation Precision", Figure 7b). This page doesn't simulate the accumulation, and the 2% and 14-bit figures are the paper's.
- **Storage and communication.** The AdamW moments are BF16. Some cached activations are FP8, with an E5M6 format for the input of the linear layer after attention. MoE dispatch is FP8 and combine is BF16 (Section 3.3.3).

**Tiny example, computed by `scripts/fp8_tiles.py`.** The matrix is $256 \times 512$ with Gaussian entries and one outlier of 100,000 at row 3, column 7. The data are made up. The script reproduces E4M3 rounding with the decoder logic of the [FP8](/fp8/) page. On rows 100 to 255, which are away from the outlier, the results are:

| Scaling | Relative RMSE | Values rounded to 0 |
|---|---|---|
| One scale for the tensor | 0.1254 | 17.19% |
| $1 \times 128$ tiles | 0.0258 | 0.00% |
| $128 \times 128$ blocks | 0.0373 | 0.83% |

The tile holding the outlier has a maximum of 100,000, and its neighbor tile in the same row has 3.115. That neighbor has a relative RMSE of 0.0247 with tiles, which is about the same as clean rows. The blocks do worse than the tiles because the outlier shares its block with 127 other rows, and rows 100 to 127 are in that block. Appendix B.2 explains why V3 doesn't use blocks for activation gradients: in a 16B MoE model trained on about 300B tokens, block-wise quantization of the activation gradients "leads to model divergence". The authors hypothesize that token-correlated outliers cause it. Exercise 2 asks you to write the tile and block round trips.

**What the paper measured.** Compared with BF16 on a 16B and a 230B MoE model, the relative loss error of FP8 training stays "consistently below 0.25%" (Section 3.3, Appendix B.1, Figure 10).

### Read the cost table

**Intuition.** The headline number is a product of three figures, so you can check it.

**Math.** Table 1 lists GPU hours in thousands: 2664K for pre-training, 119K for context extension, 5K for post-training, and 2788K in total. At \$2 per H800 GPU hour, that is \$5.576M. Section 1 says that each trillion tokens costs 180K GPU hours, or 3.7 days on 2048 GPUs. The report notes that these numbers cover the official run only and exclude "prior research and ablation experiments".

**Tiny example, computed by `scripts/training_cost.py`.** $180\mathrm{K} \times 14.8 = 2664\mathrm{K}$, which matches the table. A trillion tokens take $180{,}000 / 2048 / 24 = 3.66$ days, so the pre-training takes 54.2 days, which is about 1.8 months. The shares of the total hours are 95.6% for pre-training, 4.3% for context extension, and 0.2% for post-training, and they cost \$5.328M, \$0.238M, and \$0.010M. The total depends linearly on the price. At \$1 per GPU hour it is \$2.788M, and at \$3 it is \$8.364M. Pre-training spends about 0.648 GPU-milliseconds per token. Exercise 4 asks you to write these calculations.

<!-- exercises -->

## Pitfalls

- **Putting the bias in the gate.** The bias is for ranking in Equation 16. The gating value $g_{i,t}$ comes from the original $s_{i,t}$ (Equation 13). Using $s_{i,t} + b_i$ in the gate lets the balancing controller change the model's output.
- **Treating the bias as a trained parameter.** The paper updates it by a fixed step $\gamma$ at the end of each step, based on the observed load. It has no gradient. A learning rate or an optimizer state for it is a different method.
- **Reading "auxiliary-loss-free" as "no balance loss".** V3 also trains with a sequence-wise loss, $\alpha = 0.0001$ (Equations 17-20), to avoid extreme imbalance in a single sequence.
- **Forgetting that balance is per batch.** The bias reacts to the load over a whole batch, not to each sequence. Section 4.5.3 gives this as the reason for better specialization, and as a risk for small batches and domain shifts at inference.
- **Moving the bias in proportion to the error.** The paper's step is $\pm\gamma$. A step proportional to the load gap is a different controller, and it can oscillate.
- **Counting the MTP prediction from the wrong position.** The module at depth $k$ takes $h^{k-1}_i$ and the embedding of $t_{i+k}$ and predicts $t_{i+k+1}$ (Equations 21-23). Using $t_{i+k}$ as the target leaks the input.
- **Dividing Equation 24 by the number of terms.** The paper divides by $T$ for every depth. If you want a mean over positions, say so, because the numbers differ by $T/(T-k)$.
- **Assuming MTP costs inference time.** With $D = 1$ the module is dropped at inference. Speculative decoding is an option, not the default (Section 2.2).
- **Assuming "MTP helps" for every benchmark.** Table 4 shows gains on most benchmarks, not all.
- **Applying the FP8 recipe to everything.** Embedding, output head, MoE gating, normalization, and attention operators stay in BF16 or FP32, and the master weights, gradients, and optimizer states are in higher precision (Section 3.3.1).
- **Using E5M2 for gradients because the FP8 paper did.** V3 uses E4M3 everywhere and relies on the fine-grained groups (Section 3.3.2).
- **Skipping the accumulation fix.** The scales are per group along the inner dimension of the GEMM, and the partial sums are promoted to FP32 every 128 elements. Without promotion, the report measures an error near 2% for $K = 4096$.
- **Using block scaling for activation gradients.** Appendix B.2 reports divergence at the 16B scale, and activations use tiles for that reason.
- **Reading \$5.576M as the cost of building the model.** It is the rental cost of the final run at an assumed \$2 per hour. It excludes research, ablations, data, and staff (Section 1).
- **Treating the parameter estimates as the paper's.** The 656.5B and 23.0B in `scripts/config_counts.py` count only expert weights and come from this page's arithmetic.

## Further reading

- The official repository is [deepseek-ai/DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3), where the abstract points for the checkpoints.
- Wang et al. (2024a), the paper that the report cites for the bias method, describes the update in more detail. Read it with Section 2.1.2 open.
- Appendix B has the FP8 versus BF16 curves (B.1) and the block-wise quantization experiment (B.2).
- Section 3.2 covers DualPipe and the all-to-all kernels, which this page leaves out.
