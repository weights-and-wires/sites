---
id: "deepseekmoe"
title: "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models"
authors: "Dai et al."
year: 2024
venue: "ACL 2024"
arxiv: "2401.06066"
url: "https://arxiv.org/abs/2401.06066"
track: scaling-efficiency
tier: extended
est_hours: 3
status: full
summary: "DeepSeekMoE changes the mixture-of-experts layer in two ways that keep the parameter count and the compute per token fixed. It splits each expert into m smaller ones and activates m times as many, which multiplies the number of expert combinations a token can use, and it makes a few experts shared, so every token uses them and the routed experts don't have to relearn common knowledge. At the 2B scale this layer matches a GShard model with 1.5 times the expert parameters and compute, and DeepSeekMoE 16B matches LLaMA2 7B with about 40% of the computation."
related_course: "ml: mixture of experts"
prereqs:
  - id: mixtral
    reason: "DeepSeekMoE changes the top-K routed layer that Mixtral uses, so you need the router, the sparse gate, and the split between active and total parameters."
background: [expert-specialization]
course_links:
  - title: "Mixtures of experts: routing, the router's gradient, and balance"
    url: "https://ml.ifkash.dev/lessons/081-mixtures-of-experts"
  - title: "Mixture-of-experts and sparse kernels"
    url: "https://gpu.ifkash.dev/c/20-moe-and-sparse-kernels"
self_check:
  - q: 'A conventional MoE layer has $N = 16$ experts and routes each token to $K = 2$. DeepSeekMoE with $m = 4$ and no shared experts has the same total and active expert parameters. How many experts does it have, how many does a token use, and how many distinct expert sets can a token choose in each design?'
    a: 'It has $mN = 64$ experts of a quarter of the size, and a token uses $mK = 8$ of them. The conventional layer allows $\binom{16}{2} = 120$ sets, and the fine-grained layer allows $\binom{64}{8} = 4{,}426{,}165{,}368$ sets (Section 3.1). The budget is unchanged: 16 FFNs of expert parameters in total and 2 FFNs active per token.'
    see: expert-specialization
  - q: 'Mixtral turns the top-2 logits into gates with a softmax over just those two. A DeepSeekMoE token has router affinities that sum to 1 over the routed experts, and the top three are $0.4197$, $0.2546$, and $0.1544$. What are the three gates, and what is the difference from Mixtral?'
    a: 'The gates are $0.4197$, $0.2546$, and $0.1544$, which sum to $0.8287$ (Equation 10). The paper uses the affinity $s_{i,t}$ itself as the gate and does not renormalize it. Mixtral applies the softmax after selecting, so its two gates always sum to 1.'
    see: mixtral
  - q: 'In the 2B validation model, why does a token activate 7 routed experts and not 8, and what does the shared expert add to that?'
    a: 'The layer activates $mK = 8$ experts in total (Section 3.1). One of the 8 is the shared expert, which every token uses, so the router picks $mK - K_s = 7$ of the 63 routed experts (Equation 10). The shared expert adds no routing choice. Its output enters the sum with weight 1 for every token (Equation 9).'
    see: expert-specialization
---

## The problem

A mixture-of-experts layer replaces one feed-forward network (FFN) with $N$ copies of it and a router that sends each token to $K$ of them. [Mixtral](/mixtral/) is the open decoder-only example: 8 experts and top-2 routing in every layer. The idea behind the design is expert specialization. Each expert is meant to hold knowledge that the others don't, so the total capacity grows with $N$ while the compute per token follows $K$. The paper argues that the conventional layer, which it illustrates with GShard-style top-2 routing, falls short of this goal for two reasons (Section 1):

- **Knowledge hybridity.** With only 8 or 16 experts, each expert receives tokens that need very different kinds of knowledge. It has to pack all of them into one set of weights.
- **Knowledge redundancy.** Tokens sent to different experts still need common knowledge, such as how to continue a sentence. Several experts learn the same thing in their own weights, which wastes parameters.

The paper's goal is to remove both problems without changing the budget. Figure 2 shows the plan in three panels: conventional top-2 routing, then fine-grained experts, then fine-grained experts plus shared experts. The number of expert parameters and the compute per token are the same in all three panels.

The evidence comes from a 2B-parameter validation model trained on 100B tokens. Table 1 compares four MoE designs with the same 2.0B total parameters:

| Model | Activated parameters | Pile loss | HellaSwag | TriviaQA |
|---|---|---|---|---|
| Hash Layer (top-1) | 0.2B | 1.932 | 46.2 | 6.5 |
| Switch (top-1) | 0.2B | 1.881 | 49.1 | 8.9 |
| GShard (top-2) | 0.3B | 1.867 | 50.5 | 10.2 |
| DeepSeekMoE | 0.3B | 1.808 | 54.8 | 16.6 |

DeepSeekMoE and GShard have the same activated parameters, so the gap in the last row comes from the layer design alone (Section 4.2). Table 2 puts the gap in context. A GShard model with 1.5 times the expert size, so 1.5 times the expert parameters and compute, reaches the same Pile loss of 1.808. A dense model with 16 times the FFN parameters reaches 1.806, and the paper calls it the upper bound for an MoE with the same parameter count (Section 4.3). All of these figures are from the paper's tables, which report no variation across runs.

## Key ideas

The page uses the paper's notation (Section 2): $T$ tokens, $d$-dimensional hidden states $\mathbf{u}_t^l$ after the attention module of layer $l$, $N$ experts $\operatorname{FFN}_i$, an expert centroid $\mathbf{e}_i^l$ for each expert, an affinity $s_{i,t}$, and a gate value $g_{i,t}$. The paper omits layer normalization in its equations, and so does this page. The exercises store tokens as rows, so the logits are `u @ centroids.T`, where the paper writes $\mathbf{u}_t^{l\top} \mathbf{e}_i^l$.

The conventional layer, which the paper takes as the starting point, is (Section 2, Equations 3-5)

$$
\mathbf{h}_t^l = \sum_{i=1}^{N} g_{i,t} \operatorname{FFN}_i(\mathbf{u}_t^l) + \mathbf{u}_t^l,
\qquad
g_{i,t} =
\begin{cases}
s_{i,t}, & s_{i,t} \in \operatorname{Topk}(\{s_{j,t} \mid 1 \le j \le N\}, K), \\
0, & \text{otherwise},
\end{cases}
\qquad
s_{i,t} = \operatorname{Softmax}_i(\mathbf{u}_t^{l\top} \mathbf{e}_i^l).
$$

Two features of this formula carry over to the rest of the page. The affinities come from a softmax over all experts, and the gate is the affinity itself. There is no renormalization after the top-$K$ selection. This differs from Mixtral, which takes the softmax over only the selected logits (see the [Mixtral](/mixtral/) page). Both choices send each token to $K$ experts, but they give different weights.

### Fine-grained expert segmentation

**Intuition.** Take each expert and cut its hidden layer into $m$ pieces. The pieces are $m$ smaller experts, each with $1/m$ of the parameters. A token now activates $m$ times as many experts, so it still touches the same number of parameters. The router can combine many small pieces of knowledge, where before it had to choose whole large blocks.

**Math.** Each expert FFN is split into $m$ experts by reducing the FFN intermediate hidden dimension to $1/m$ of its size, and the number of activated experts rises to $mK$ (Section 3.1, Equations 6-8):

$$
\mathbf{h}_t^l = \sum_{i=1}^{mN} g_{i,t} \operatorname{FFN}_i(\mathbf{u}_t^l) + \mathbf{u}_t^l,
\qquad
g_{i,t} =
\begin{cases}
s_{i,t}, & s_{i,t} \in \operatorname{Topk}(\{s_{j,t} \mid 1 \le j \le mN\}, mK), \\
0, & \text{otherwise},
\end{cases}
$$

with $s_{i,t}$ defined as in Equation 5, now over $mN$ experts. The total expert parameters stay at $N$ times those of a standard FFN, and the number of nonzero gates grows to $mK$.

The gain is combinatorial. A token picks a set of experts, and the number of possible sets is a binomial coefficient. The paper compares $N = 16$ with $K = 2$ against the same layer with $m = 4$ (Section 3.1):

$$
\binom{16}{2} = 120
\quad\text{versus}\quad
\binom{64}{8} = 4{,}426{,}165{,}368.
$$

**Tiny example, computed by `scripts/combinatorics.py`.** The script prints the two counts above, and their ratio is about 36.9 million. It also checks that the budget doesn't move. Splitting each expert into 4 gives 64 experts of size $1/4$, and the token activates 8 of them. That is $64 \times \frac14 = 16$ FFNs of expert parameters in total, and $8 \times \frac14 = 2$ FFNs active, the same as the 16 and 2 of the conventional layer. The number of sets is only an upper bound on useful variety, since the router may not use most of them, and the paper offers it as an argument for flexibility and not as a measurement (Section 3.1). The paper's evidence is the ablation: with 1 shared expert, splitting into 32 or 64 experts raises the overall performance step by step (Section 4.4, Figure 3).

Exercise 1 asks you to compute this layout for any $N$, $K$, $m$, and $K_s$.

### Shared expert isolation

**Intuition.** Some knowledge is needed by almost every token. If the router has to find it, many routed experts learn their own copy. The fix is to set aside $K_s$ experts that every token always uses, so the routed experts are free to differ from each other.

**Math.** The first $K_s$ experts are shared. Every token goes to them regardless of the router, and the router chooses from the remaining $mN - K_s$ experts. To keep the compute constant, the number of routed experts that a token activates drops by $K_s$ (Section 3.2, Equations 9-11):

$$
\mathbf{h}_t^l = \sum_{i=1}^{K_s} \operatorname{FFN}_i(\mathbf{u}_t^l) + \sum_{i=K_s+1}^{mN} g_{i,t} \operatorname{FFN}_i(\mathbf{u}_t^l) + \mathbf{u}_t^l,
$$

$$
g_{i,t} =
\begin{cases}
s_{i,t}, & s_{i,t} \in \operatorname{Topk}(\{s_{j,t} \mid K_s + 1 \le j \le mN\}, mK - K_s), \\
0, & \text{otherwise},
\end{cases}
\qquad
s_{i,t} = \operatorname{Softmax}_i(\mathbf{u}_t^{l\top} \mathbf{e}_i^l).
$$

The paper then states that the number of routed experts is $mN - K_s$ and the number of nonzero gates is $mK - K_s$. Equation 11 writes the softmax with the index $i$ only, without saying over which set. This page takes it over the routed experts $K_s + 1, \dots, mN$, because only those have centroids that a router needs. That is my reading of the notation, so check it against the released code when you need the exact behavior. Shared experts have no gate, which is equivalent to a gate of exactly 1.

The validation model has $N = 16$, $K = 2$, $m = 4$, and $K_s = 1$. It has 1 shared and 63 routed experts of size $1/4$, and each token uses the shared expert plus 7 routed ones (Section 4.2, Table 2, Table 7). The larger models differ. The 16B model has 2 shared and 64 routed experts, activating 2 and 6, and the 145B model has 4 shared and 128 routed, activating 4 and 12 (Section 5.1.2, Section 7.1, Table 7). In those two cases the shared experts are listed on top of the routed ones, so the total expert parameters are 16.5 FFNs and not 16 (`scripts/combinatorics.py` prints this). The expert size is $1/4$ of an FFN for the 16B model and $1/8$ for the 145B model.

**Tiny example, computed by `scripts/gate_example.py`.** One token has the six routed-expert logits $[2.0, 0.5, 1.0, -1.0, 0.0, 1.5]$, and the layer activates 3 routed experts. The softmax over the six logits gives affinities $[0.4197, 0.0936, 0.1544, 0.0209, 0.0568, 0.2546]$. The top three are routed experts 0, 5, and 2 (counting from 0), so the gates are $0.4197$, $0.2546$, and $0.1544$ and sum to $0.8287$. A Mixtral-style gate on the same token would be $[0.5065, 0.3072, 0.1863]$, renormalized to sum to 1. Now give the stand-in experts scalar outputs: $\mathbf{u}_t = 1$, a shared expert that returns 100, and routed expert $i$ returns $10(i+1)$. The layer output is $1 + 100 + 0.4197 \cdot 10 + 0.2546 \cdot 60 + 0.1544 \cdot 30 = 125.1025$. With renormalized gates it would be $130.0863$. The two differ because the paper keeps the raw softmax values.

The paper's evidence for the shared expert is in Section 4.5. Turning the shared expert off and activating one more routed expert, which keeps the compute constant, raises the Pile loss from 1.808 to 2.414. The paper concludes that the shared expert holds knowledge that the routed experts can't replace. It also tests the ratio of shared to routed experts. With 1, 2, and 4 shared experts, out of 64 total, the Pile losses are 1.808, 1.806, and 1.811, so the choice barely matters (Section 4.4). The paper uses a ratio of 1 shared to 3 activated routed experts for the larger models.

Exercise 2 asks you to implement this layer.

### Load balance at two levels

**Intuition.** A learned router tends to collapse onto a few experts, which leaves the rest untrained (routing collapse, Section 3.3). If experts live on different devices, uneven loads also leave some devices idle while others are still working. The paper therefore uses two auxiliary losses: one that spreads the load across experts, and one that spreads it across devices.

**Math.** Let $N' = mN - K_s$ be the number of routed experts and $K' = mK - K_s$ the number of activated routed experts. The expert-level balance loss is (Section 3.3, Equations 12-14)

$$
\mathcal{L}_{\text{ExpBal}} = \alpha_1 \sum_{i=1}^{N'} f_i P_i,
\qquad
f_i = \frac{N'}{K' T} \sum_{t=1}^{T} \mathbb{1}(\text{Token } t \text{ selects Expert } i),
\qquad
P_i = \frac{1}{T} \sum_{t=1}^{T} s_{i,t}.
$$

Here $f_i$ is the fraction of selections that go to expert $i$, scaled so that perfect balance gives $f_i = 1$, and $P_i$ is the mean affinity of expert $i$. With perfect balance, $f_i = 1$ and $P_i = 1/N'$, so the loss equals $\alpha_1$. Any imbalance raises it. The [Switch Transformers](/switch/) balance loss has the same $\sum_i f_i P_i$ form, with its own scaling.

If the routed experts are partitioned into $D$ groups $\{\mathcal{E}_1, \dots, \mathcal{E}_D\}$, with one group per device, the device-level balance loss is (Section 3.3, Equations 15-17)

$$
\mathcal{L}_{\text{DevBal}} = \alpha_2 \sum_{i=1}^{D} f'_i P'_i,
\qquad
f'_i = \frac{1}{|\mathcal{E}_i|} \sum_{j \in \mathcal{E}_i} f_j,
\qquad
P'_i = \sum_{j \in \mathcal{E}_i} P_j.
$$

Note that $f'_i$ is a mean and $P'_i$ is a sum. The paper's reason for the second loss is that strict balance across experts hurts quality, while balance across devices is what removes the computation bottleneck. In practice it sets a small $\alpha_1$ against collapse and a larger $\alpha_2$ for the devices (Section 3.3). This paper has no rule that limits the devices a token can reach. The pitfalls section returns to that point.

**Tiny example, computed by `scripts/gate_example.py`.** Take $T = 4$ tokens, $N' = 4$ routed experts, $K' = 2$, and $\alpha = 1$. Balanced routing, with each expert chosen by exactly 2 tokens and every affinity equal to $0.25$, gives $f_i = 1$ for all experts and a loss of exactly $1.0000$. In the collapsed case every token selects experts 0 and 1 and has affinities $[0.55, 0.30, 0.10, 0.05]$. Then $f = [2, 2, 0, 0]$ and $\mathcal{L}_{\text{ExpBal}} = 2 \cdot 0.55 + 2 \cdot 0.30 = 1.7000$. For the device loss, put experts 0 and 1 on device 1 and experts 2 and 3 on device 2. Then $f' = [2, 0]$, $P' = [0.85, 0.15]$, and the loss is $1.7000$. If you place experts 0 and 2 together instead, $f' = [1, 1]$, $P' = [0.65, 0.35]$, and the device loss drops to $1.0000$ while the expert-level loss stays at $1.7000$. That is the point of the second loss: it only asks for balanced devices.

The paper uses the device-level loss only when it needs it. For the 2B and 16B models, all experts of a layer sit on one device, so it doesn't use the device-level loss and sets $\alpha_1 = 0.01$ and $0.001$, respectively (Section 4.1.3, Section 5.1.2). For the 145B model, the routed experts of each layer are spread over 4 devices, with $\alpha_2 = 0.05$ and $\alpha_1 = 0.003$ (Section 7.1).

Exercises 3 and 4 ask you to implement the two losses.

### Why the layer specializes

The paper tests the claim of specialization with three experiments on the 2B model, all on the Pile loss (Section 4.5):

- **Less redundancy among routed experts.** The test masks the routed experts that have the highest affinity for each token and picks the top-$K$ from the rest. DeepSeekMoE gets worse faster than GShard with 1.5 times the expert size (Figure 4). The paper reads this as each routed expert being harder to replace.
- **Fewer activated experts suffice.** With 4 activated routed experts, DeepSeekMoE has a Pile loss comparable to GShard with its full top-2 (Figure 5). A model trained from scratch with only 3 activated routed experts, half the activated expert parameters of GShard, still beats GShard (Figure 6).
- **The shared expert can't be replaced**, as the previous idea shows (1.808 to 2.414).

These experiments show that disabling or removing components of this particular model hurts. They are consistent with the paper's explanation, but they don't directly show what each expert knows.

At scale, the paper reports that DeepSeekMoE 16B has 16.4B total and about 2.8B activated parameters and uses about 39.6% of the computation of LLaMA2 7B while performing comparably (Section 5.1.2, Section 5.2, Table 4). A preliminary 145B model, trained on only 245B tokens, matches DeepSeek 67B (a dense model) with 28.5% of the computation (Section 7.2).

<!-- exercises -->

## Pitfalls

- **Renormalizing the gates.** Mixtral applies a softmax after the top-$K$ selection, so its gates sum to 1. DeepSeekMoE uses the affinity from the softmax over the routed experts as the gate (Equation 10). In the page's example the gates sum to $0.8287$. If you renormalize, the layer output changes from $125.1025$ to $130.0863$. Exercise 2 catches it.
- **Looking for device-limited routing.** Some summaries list device-limited routing as part of DeepSeekMoE, but this paper doesn't describe it. Section 3.3 describes a device-level balance loss, and Section 7.1 says the 145B model spreads the routed experts across 4 devices. A rule that limits how many devices a token's experts can sit on comes from the later DeepSeek-V2 paper. Don't attribute it here.
- **Counting the shared experts twice.** In the 2B model the shared experts are part of the $mN$ experts, and the router chooses $mK - K_s$ of the other $mN - K_s$ (Equations 9-10). The 2B model has 1 shared and 63 routed experts, and a token uses $1 + 7 = 8 = mK$. The larger models in Table 7 list the shared experts separately from the 64 or 128 routed ones, so their expert totals are 16.5 FFNs and not 16.
- **Treating "more combinations" as a result.** The count $\binom{64}{8}$ shows room for flexibility. The paper's support for it is the ablation in Figure 3 and the experiments in Section 4.5, not the count.
- **Reading the "upper bound" as a theorem.** The dense model with 16 times the FFN parameters is the paper's reference for the best that an MoE with the same parameters could do, and it sits at nearly the same loss (1.806 against 1.808). The paper states this only for about 2B parameters and 100B tokens (Section 4.3).
- **Assuming the balance losses are always on.** The device-level loss is not used for the 2B and 16B models because every layer's experts are on one device. The expert-level factor is small, 0.01 and 0.001 (Section 4.1.3, Section 5.1.2), because larger values hurt quality at 16B.
- **Mixing up $f_i$ and $P_i$.** $f_i$ counts hard selections, so it carries no gradient. $P_i$ is the mean of the soft affinities, which carries the gradient to the router. Using the affinities of only the selected tokens, or the same indicator for both, changes the loss and removes the signal.
- **Forgetting that the first layer is dense.** The 16B and 145B models substitute all FFNs except the first with MoE layers, because the load balance converges more slowly in the first layer (Section 5.1.2, Section 7.1).

## Further reading

- The paper's code and the DeepSeekMoE 16B checkpoint are linked from the paper at [github.com/deepseek-ai/DeepSeek-MoE](https://github.com/deepseek-ai/DeepSeek-MoE).
- [Mixtral](/mixtral/) is the conventional top-2 layer that this page compares with, and [Switch Transformers](/switch/) is the source of the balance loss that Equations 12-14 resemble.
- [GShard](https://arxiv.org/abs/2006.16668) is the top-2 baseline in every comparison of Section 4.
- The paper credits [DeepSpeed-MoE](https://arxiv.org/abs/2201.05596) with a prototype of the shared expert, which it derives from an engineering view and this paper from an algorithmic one (Section 3.2).
