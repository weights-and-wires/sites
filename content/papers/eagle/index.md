---
id: "eagle"
title: "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"
authors: "Li et al."
year: 2024
venue: "ICML 2024"
arxiv: "2401.15077"
url: "https://arxiv.org/abs/2401.15077"
track: inference
tier: extended
est_hours: 5
status: full
summary: "EAGLE speeds up decoding without changing the output distribution by drafting with a tiny head that predicts the target model's next hidden feature instead of its next token. The head also reads the token that was just sampled, which removes most of the uncertainty that makes feature prediction hard. The draft is a tree, which the target checks in one pass with a tree attention mask, and the paper reports a 2.7x to 3.5x latency speedup on LLaMA2-Chat 70B."
related_course: "inference: speculative decoding"
prereqs:
  - id: spec-decoding
    reason: "EAGLE keeps the draft-then-verify loop and the accept/reject rule from speculative sampling, and it changes only where the draft comes from and how many candidates it offers."
background: [feature-level-drafting, tree-attention]
course_links:
  - title: "Speculative decoding"
    url: "https://qwen.ifkash.dev/c/20-speculative-decoding"
  - title: "Tree speculation and speculation under load"
    url: "https://qwen.ifkash.dev/c/20a-tree-speculation"
self_check:
  - q: 'A draft model sees the feature $f_i$ at the word "I" and has to predict $f_{i+1}$. Why is that hard when the model only sees $f_i$, and what input does EAGLE add?'
    a: 'The target samples the next token from a distribution, and "am" and "always" lead to different next features. One $f_i$ therefore maps to several possible $f_{i+1}$ values, and a regression head can only learn their average. EAGLE adds the token sequence advanced by one step, so the head sees which token was sampled. Then $f_{i+1}$ is a function of its inputs again.'
    see: feature-level-drafting
  - q: 'A draft tree has nodes $a$ and $b$ as two children of the last context token, and $c$ as a child of $a$. Which keys can the query for $c$ attend to, and which position id does it get if the context holds 3 tokens?'
    a: 'The query for $c$ attends to all 3 context tokens, to $a$ (its parent), and to $c$ itself. It does not attend to $b$, because $b$ is a sibling branch. Its position id is 4: with 3 context tokens at positions 0 to 2, $a$ and $b$ sit at position 3, and $c$ sits one step later.'
    see: tree-attention
  - q: 'Speculative sampling accepts a draft token $x$ with probability $\min(1, p(x)/q(x))$. With two candidates for one slot, what does the target distribution become after the first candidate is rejected, and why does the output stay exact?'
    a: 'It becomes $\operatorname{norm}(\max(0, p - q))$. The first test accepts token $x$ with mass $\min(p(x), q(x))$, and the rejection leaves exactly the mass $p(x) - \min(p(x), q(x))$ that the first test under-delivered. The second test then treats this residual as its target, so the total mass for each token is still $p(x)$. This is the same argument as for one candidate, applied twice.'
    see: spec-decoding
---

## The problem

Speculative sampling (see the `spec-decoding` page) needs a draft that is cheap and often right. There are two usual choices, and both fall short.

A separate small model can draft, but the target's family often has no model that is both small and well matched. The paper notes that for the 7B model of the LLaMA2-Chat family there is no smaller draft to use. Even where one exists, it is a separate network that runs its own forward pass for every drafted token.

Heads on top of the target, such as Medusa, are cheap but weak. They predict several future tokens from one feature, without seeing the tokens in between. The paper reports a draft accuracy of about 0.6 for Medusa and about 0.8 for EAGLE (Section 1).

The failing example is the one in Figure 3. The prefix is "I". The target can continue with "am" or with "always". Those two choices lead to two different feature sequences. A draft that predicts the next feature from $f_{\text{I}}$ alone has to guess which one will happen. It learns the average of both, and that average matches neither.

Figure 4 shows the cost on Vicuna 7B with MT-bench at temperature 0. Drafting at the token level gives a 1.5x speedup. Drafting on features alone gives 1.9x. Adding the token that was just sampled gives 2.8x. The rest of this page explains why that last step matters and what else EAGLE does.

## Key ideas

The paper uses these symbols. A token is $t_i$ and its embedding is $e_i$. The feature $f_i$ is the target's second-to-top-layer hidden state at position $i$, the vector that the LM head turns into logits. The draft model predicts a feature $\hat f_{i}$ and reads it out with the target's own LM head. The paper does not number most of its equations, so this page cites sections, figures, and tables.

### Draft on the second-to-top-layer features

**Intuition.** A feature is smoother than a token. A token is one choice out of a large vocabulary. A feature is a continuous vector with a lot of structure, and its next value is a simple function of the last one. Predicting features is an easier job for a small model, and the target's own LM head turns the predicted features into token distributions for free (Section 1, Section 3.1).

**Math.** Vanilla decoding runs $t_{1:j} \to e_{1:j} \to f_j \to p_{j+1} \to t_{j+1}$ (Section 2). The EAGLE draft model has three parts (Section 3.1, Figure 6):

1. The embedding layer of the target, frozen.
1. An autoregression head: one fully connected layer and one Transformer decoder layer, trained.
1. The LM head of the target, frozen.

Two things follow. The draft head is small: the paper lists 0.24B trainable parameters for the 7B target and 0.99B for the 70B target (Section 4). And the draft needs no vocabulary of its own, since it reuses the target's embedding and LM head.

**Tiny example, computed by `scripts/drafter_step.py`.** The script builds a toy draft step with feature size $d = 3$. A fully connected layer with weights of shape $(2d, d) = (6, 3)$ maps the concatenated input to a predicted feature of size 3, so the shape of the feature does not change. Exercise 4 asks you to build this step.

### Resolve uncertainty with the advanced token

**Intuition.** The feature $f_i$ was computed before the target sampled $t_{i+1}$. So $f_i$ cannot tell the draft which word was chosen. Feed the draft the sampled token too, and the uncertainty is gone. The token is one step ahead of the feature, and the paper calls the input the "advanced token sequence" (Section 1, Section 3.1).

**Math.** At position $i$, the draft model gets the feature sequence $f_{1:i}$ and the token sequence $t_{2:i+1}$ (Section 3.2). It concatenates each feature with the embedding of the token one step ahead, which gives a tensor of shape (batch, sequence length, $2 \times$ hidden size). The fully connected layer brings it back to the hidden size. The decoder layer then predicts $\hat f_{i+1}$ (Section 3.1, Figure 6).

The ablation in Section 4.3.2 compares four inputs on Vicuna 7B. They are feature with shifted token (EAGLE), feature with unshifted token, token only, and feature only. The paper's text says that advancing the token by one step gives the largest gain. The exact values are in Figure 8, and this page does not reproduce them.

**Tiny example, computed by `scripts/drafter_step.py`.** The script builds a toy case with two possible next tokens. The next feature is $f_{i+1} = f_i A + e(t_{i+1}) B$ for random matrices $A$ and $B$, with $d = 4$ and 4,000 samples. The mean square of the feature is 3.9538. A linear regression from $f_i$ alone leaves a mean squared error of 3.0308, nearly all of the feature's size. A regression from $f_i$ and the embedding of the next token leaves 0.000000. The toy is linear and so it is a cartoon, but it shows the mechanism: the leftover error is exactly the part that depends on the sampled token.

### Train the head with a feature loss and a token loss

**Intuition.** The head has two jobs. Its predicted feature should be close to the real feature, and the token distribution it implies should be close to the real distribution. The first is a regression, and the second is a classification.

**Math.** Section 3.2 defines the regression loss with Smooth L1 and the classification loss with cross-entropy. Write $\hat f_{i+1}$ for the draft's output and $p_{i+2} = \operatorname{softmax}(\text{LM\_head}(f_{i+1}))$ for the target's distribution. This page writes $\hat p_{i+2}$ for the distribution that the LM head gives from the draft's feature $\hat f_{i+1}$:

$$
L_{\text{reg}} = \operatorname{SmoothL1}\bigl(f_{i+1},\; \hat f_{i+1}\bigr), \qquad L_{\text{cls}} = \operatorname{CrossEntropy}\bigl(p_{i+2},\; \hat p_{i+2}\bigr),
$$

$$
L = L_{\text{reg}} + w_{\text{cls}} \, L_{\text{cls}}, \qquad w_{\text{cls}} = 0.1.
$$

The classification term has a weight of 0.1, so the regression term dominates. Two choices in the paper make training cheap:

- **A fixed dataset.** The head trains on a fixed ShareGPT dataset, about 68,000 dialogues, with no text generated by the target (Section 3.2, Section 4). Table 6 compares this with target-generated answers on LLaMA2-Chat 7B: 2.78x with the fixed data against 2.88x with generated data. The paper judges the gap small enough to keep the cheap option (Section 4.3.3).
- **Noise on the features.** At training time the draft reads real features from the target. At inference time it reads its own predicted features, which carry errors. To bridge this gap, the paper adds uniform noise from $U(-0.1, 0.1)$ to the target's features during training (Section 3.2).

The optimizer is AdamW with $\beta = (0.9, 0.95)$, a learning rate of $3 \times 10^{-5}$, and gradient clipping at 0.5 (Section 4).

**Tiny example, computed by `scripts/drafter_step.py`.** Smooth L1 with $\beta = 1$ is $\tfrac12 x^2$ for $|x| < 1$ and $|x| - \tfrac12$ otherwise. The script prints 0.125 for $x = 0.5$ and 2.5 for $x = 3$. This page uses $\beta = 1$, the common library default, because the paper does not state it. Exercise 4 includes this loss.

### Draft a tree, verify it with tree attention

**Intuition.** A chain stops at its first rejection, and everything after is wasted. A tree offers several continuations at some positions, so a rejection at one branch can still be rescued by a sibling. The tree has more tokens than the draft has forward passes, because every pass can grow several branches at once.

**Math.** Drafting a tree of depth $m$ takes $m$ forward passes of the draft head and gives more than $m$ tokens. Figure 6 shows a tree of 10 tokens from 3 passes (Section 3.1). Appendix A.1 gives the setup used in the experiments: 5 forward passes for drafting, with $k = 4$ children at the root. The paper chose the tree shape by hand, with likelier branches deeper and wider. In greedy mode a node's children are the top-$k$ tokens, and in sampling mode they are $k$ samples.

The target then scores the whole tree in a single forward pass (Section 3.3). Each node must see its own root-to-node path and nothing else, or the logits for a branch would depend on tokens that are not in its prefix. A tree attention mask does this: a node attends to the whole context, itself, and its ancestors. The position id of a node is the position it would have if its path were a plain chain. The paper does not spell out the mask, so this construction is the standard one that the description implies.

**Tiny example, computed by `scripts/tree_mask_example.py`.** Take a tree of six nodes $a, \dots, f$ with parents $(-1, a, a, b, c, -1)$, so $a$ and $f$ are roots, $b$ and $c$ are children of $a$, $d$ is a child of $b$, and $e$ is a child of $c$. The context has 3 tokens. The script prints:

- depth $(1, 2, 2, 3, 3, 1)$ and position ids $(3, 4, 4, 5, 5, 3)$. Siblings share a position.
- A tree block with 12 entries set, against 21 for a causal chain of six tokens. Row $d$ sees $a$, $b$, and $d$, and row $e$ sees $a$, $c$, and $e$. Neither sees the other.

Exercise 1 asks you to build this mask and position ids, and its tests check that one masked pass gives the same output as running each path alone.

### Verify several candidates and stay exact

**Intuition.** With one candidate per slot, speculative sampling uses the accept/reject rule and, on a rejection, the residual $\operatorname{norm}(\max(0, p - q))$. With $k$ candidates for the same slot, test them one after another. After each rejection, the residual becomes the new target. Only if all $k$ fail do you draw a token from the last residual.

**Math.** Section 3.3 refers to multi-round speculative sampling, and Appendix A.2 gives the algorithm (Algorithm 1). The paper follows SpecInfer for this. The pseudocode in the appendix writes the adjustment with a scalar at one token. The text describes the elementwise adjustment, which is what this page uses. Let the candidates be drawn i.i.d. from $q$. Write $p^{(1)} = p$, and after the $r$-th rejection

$$
p^{(r+1)} = \operatorname{norm}\bigl(\max(0,\; p^{(r)} - q)\bigr).
$$

Candidate $r$ is accepted with probability $\min(1, p^{(r)}(x)/q(x))$. If all $k$ are rejected, the token comes from $p^{(k+1)}$. The output is exactly $p$, by induction on $k$: each round delivers the mass $\min(p^{(r)}, q)$ and leaves the remaining mass for the next round, in the same way as in the one-candidate proof of the `spec-decoding` page.

This page's argument assumes i.i.d. draws from $q$. The paper's pseudocode does not make clear whether a without-replacement tree is handled the same way. In greedy mode, the paper takes the top-$k$ tokens and verification is an argmax check, which needs no residuals.

**Tiny example, computed by `scripts/multi_candidate.py`.** Use $p = [0.5, 0.4, 0.1]$ and $q = [0.2, 0.3, 0.5]$ from the `spec-decoding` page.

| Candidates $k$ | P(some candidate accepted) | Exact output distribution |
|---|---|---|
| 1 | 0.6000 | `[0.5, 0.4, 0.1]` |
| 2 | 0.7800 | `[0.5, 0.4, 0.1]` |
| 3 | 0.8240 | `[0.5, 0.4, 0.1]` |

After one rejection the target becomes $[0.75, 0.25, 0]$, and the second candidate is accepted with rate $\sum \min(p^{(2)}, q) = 0.45$. So $P = 0.6 + 0.4 \times 0.45 = 0.78$. A seeded run of 200,000 samples with $k = 2$ gives frequencies `[0.4992, 0.4, 0.1008]`, an acceptance rate of 0.7814, and a chi-square statistic of 1.52 against $p$ with 2 degrees of freedom. If you forget to update the target between candidates, the frequencies are `[0.4009, 0.4592, 0.1398]`, and the output is no longer $p$. Exercise 2 builds this step.

### How much does a tree buy?

**Intuition.** The average accepted length $\tau$ is what a round yields, counting the extra token that the target adds. A tree raises $\tau$ at a small extra cost, because the target scores the extra nodes in a pass that was memory bound anyway.

**Math.** Use the model of Exercise 3, which is an assumption of this page and not a result of the paper. A node is tested only if its parent was accepted and its earlier siblings were rejected. Then

$$
\tau = 1 + \sum_{i \in \text{nodes}} P(\text{node } i \text{ accepted}).
$$

For a chain of depth $d$ with the same acceptance $\alpha$ at each node, this is $\sum_{k=0}^{d} \alpha^k$, the formula of the `spec-decoding` page.

**Tiny example, computed by `scripts/expected_length.py`.** Take $\alpha = 0.7$ and depth 5. The chain gives $\tau = 2.9412$. Now add a second child to every node on the spine, accepted with probability 0.5 when the first child was rejected. This tree has 10 nodes and gives $\tau = 3.3571$. A simulation of 200,000 rounds gives 3.3532. If one draft pass costs $c = 0.05$ of a target pass, and the larger verification pass is treated as free, the speedup $\tau / (1 + 5c)$ is 2.35 for the chain and 2.69 for the tree. The assumptions are strong, so treat these as an illustration of the direction and not as a prediction.

The paper measures the same effect in Table 5 (MT-bench, temperature 0, $\tau$ for chain then tree). For example, Vicuna 7B goes from 3.20 to 3.94. Across the six models, the script computes gains from 0.62 to 0.75, which agrees with the paper's statement of gains of 0.6 to 0.8 (Section 4.3.1). The paper also reports that EAGLE with a chain only still gives 2.3x to 2.7x.

### What the paper reports

All numbers in this section are from the paper's tables, with batch size 1 unless noted.

- **Headline.** On LLaMA2-Chat 70B, EAGLE gives a latency speedup of 2.7x to 3.5x and doubles throughput, with the distribution of the text unchanged (abstract). On MT-bench, the paper reports 2.1x to 3.8x over vanilla decoding, 1.7x to 2.1x over Lookahead, and 1.5x to 1.6x over Medusa (Section 4.1).
- **Table 1.** At temperature 0, LLaMA2-Chat 70B gets 3.52x on HumanEval ($\tau$ 4.42), 3.03x on GSM8K ($\tau$ 3.93), and 2.97x on Alpaca ($\tau$ 3.77). At temperature 1, the speedups are lower, 2.92x, 2.74x, and 2.65x for the same three tasks.
- **Table 7 (Section 4.4).** At batch size 4, Vicuna 7B keeps a 2.76x speedup and LLaMA2-Chat 70B keeps 2.40x. Throughput at each method's maximum batch size doubles, to 1.97x and 1.99x. EAGLE needs a little more memory, so its maximum batch is 7 against 8 on Vicuna 7B.
- **Table 3.** Mixtral 8x7B Instruct gets only 1.50x ($\tau$ 3.25). The paper links this to a lower $\tau$ and to the cost of loading experts during verification.
- **Table 4 (Section 4.2).** With gpt-fast on a single RTX 3090, LLaMA2-Chat 7B at FP16 goes from 55.1 to 100.2 tokens per second. In int4 it goes from 106.9 to 160.4.

## Implement it

<!-- exercises -->

## Pitfalls

- **Aligning the token with the feature.** The draft input at position $i$ is $f_i$ with the embedding of $t_{i+1}$, not $t_i$. If you use $t_i$, the head can't know which token was sampled, and you lose the gain that Figure 4 shows. Exercise 4 tests for this.
- **Letting branches see each other.** A tree mask built as a lower triangle over node indices lets siblings see each other. The logits of one branch then depend on a token from another. Check the mask by comparing a masked pass with the per-path runs, as the tests of Exercise 1 do.
- **Wrong position ids.** Siblings share a position, because they compete for one slot. Using the node index, or counting from 0 instead of from the context length, gives the target the wrong positions for a position-dependent encoding such as rotary embeddings.
- **Not updating the target between candidates.** Testing every candidate against the original $p$ gives frequencies of `[0.4009, 0.4592, 0.1398]` instead of `[0.5, 0.4, 0.1]` in the example above. Shapes and means look fine, and only a distribution test catches it.
- **Assuming the tree is free.** The target scores every tree node. That is cheap when the pass is memory bound and costly when the batch already saturates compute. The paper's batch-size study stops at small batches, and the speedup falls from 3.01x at batch 1 to 2.40x at batch 4 for LLaMA2-Chat 70B (Table 7).
- **Reading speedups as model properties.** The numbers depend on the task, the temperature, and the hardware. Table 1 spans 2.78x to 3.76x at temperature 0 for the LLaMA2-Chat and Vicuna families, and Mixtral reaches only 1.50x. Don't carry one number to another model.
- **Comparing against the wrong baselines.** Lookahead supports only greedy decoding, and Medusa's non-greedy mode doesn't preserve the output distribution. The paper compares with them at temperature 0 only for that reason (Figures 1 and 2).
- **Calling a measured $\tau$ a speedup.** $\tau$ counts tokens per target pass. The speedup also pays for the draft passes and the larger verification pass.
- **Treating the toy models as measurements.** The expected-length model on this page assumes independent acceptance with fixed probabilities. Real acceptance depends on the context. Use it to see the direction of an effect, not its size.

## Further reading

- [Li, Wei, Zhang, and Zhang, "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty" (arXiv 2401.15077)](https://arxiv.org/abs/2401.15077). Read Section 3.1 with Figure 6 for the draft and Appendix A.2 for the multi-round verification.
- [The official EAGLE code](https://github.com/SafeAILab/EAGLE) from the authors' group.
- [Li et al., "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees" (arXiv 2406.16858)](https://arxiv.org/abs/2406.16858) and [Li et al., "EAGLE-3" (arXiv 2503.01840)](https://arxiv.org/abs/2503.01840), the follow-ups that make the tree dynamic and drop the feature objective.
- [Miao et al., "SpecInfer" (arXiv 2305.09781)](https://arxiv.org/abs/2305.09781), the source of the tree verification that EAGLE follows.
- [Tree speculation and speculation under load](https://qwen.ifkash.dev/c/20a-tree-speculation) in the course, which covers tree masks, multi-candidate verification, and EAGLE's variants with runnable labs.
