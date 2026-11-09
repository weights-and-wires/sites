---
id: "instructgpt"
title: "Training language models to follow instructions with human feedback"
authors: "Ouyang et al."
year: 2022
venue: "NeurIPS 2022"
arxiv: "2203.02155"
url: "https://arxiv.org/abs/2203.02155"
track: rl
tier: core
est_hours: 6
status: full
summary: "InstructGPT fine-tunes GPT-3 in three stages: supervised fine-tuning on labeler demonstrations, a reward model trained on labeler rankings of several responses, and PPO against that reward model with a per-token KL penalty toward the supervised policy. Mixing pretraining gradients into the PPO updates (PPO-ptx) limits regressions on public NLP benchmarks. Labelers prefer outputs from the 1.3B InstructGPT model to outputs from the 175B GPT-3, a model with more than 100 times as many parameters."
related_course: "rl: RLHF post-training"
prereqs:
  - id: christiano
    reason: "InstructGPT reuses the pairwise-comparison reward model of this paper, extends it from two clips to a ranking of K responses, and moves the loop from game agents to text."
  - id: gpt3
    reason: "The pretrained GPT-3 models are the starting point for every stage, and the paper's headline result compares the aligned 1.3B model with the 175B GPT-3 and its few-shot prompting."
  - id: ppo
    reason: "The RL stage optimizes the reward with PPO, using the clipped surrogate objective with a clip ratio of 0.2, and the KL term of Equation 2 sits on top of it."
background: [kl-penalty, sft, reward-model]
course_links:
  - title: "RLHF with PPO"
    url: "https://rl.ifkash.dev/lessons/053-rlhf-with-ppo"
  - title: "Supervised fine-tuning and the post-training pipeline"
    url: "https://rl.ifkash.dev/lessons/049-sft-and-the-post-training-pipeline"
self_check:
  - q: 'A reward model scores a labeler-preferred response at $r_w = 2$ and the rejected one at $r_l = 1$. What does Equation 1 give as the loss for this pair, and what is the loss if you add $100$ to both scores?'
    a: 'The loss is $-\log\sigma(2 - 1) = \log(1 + e^{-1}) \approx 0.3133$. Adding $100$ to both scores changes nothing, because only the difference enters the sigmoid. The paper uses this fact when it adds a bias to the reward model so that labeler demonstrations score $0$ on average before RL (Section 3.5).'
    see: christiano
  - q: 'A sampled response has per-token log ratios $\log\pi^{RL} - \log\pi^{SFT}$ that sum to $1.1$, and the reward model scores it $1.8$. With $\beta = 0.02$, what reward does Equation 2 assign, and at which token does the reward model''s score arrive?'
    a: 'The reward is $1.8 - 0.02 \cdot 1.1 = 1.778$. The environment is a bandit: the reward model scores the whole response once, so its score is paid at the final token, while the KL penalty is charged at every token.'
    see: kl-penalty
  - q: 'The RL stage uses PPO with a clip ratio of $0.2$. For a token with a positive advantage, at which probability ratio $r_t$ does the clipped objective stop pushing the policy?'
    a: 'At $r_t = 1 + \epsilon = 1.2$. Above it the term is constant, so the gradient for that token is $0$. For a negative advantage the cutoff is $1 - \epsilon = 0.8$ in the other direction.'
    see: ppo
---

## The problem

GPT-3 (see [Language models are few-shot learners](/gpt3/)) is trained to predict the next token of text from the internet. That objective is different from "follow the user's instructions helpfully and safely", and the paper calls the mismatch misalignment (Section 1). A base model asked to "Summarize this article in two paragraphs or less" can continue the prompt as if it were the first line of a list of exercises, ignore the length limit, or invent details. The model isn't malfunctioning. In the training data, a line like that is often followed by more lines like it.

You can patch this with a few-shot prompt, and the paper includes that baseline, "GPT-3 (prompted)". Prompting doesn't change what the model optimizes. The paper's fix is to change the objective, using human feedback as the reward signal (Section 1, following Christiano et al. and Stiennon et al.).

The payoff is large. On the paper's API test prompts, outputs from the 175B InstructGPT are preferred to 175B GPT-3 outputs 85 ± 3% of the time, and to few-shot 175B GPT-3 outputs 71 ± 4% of the time (Section 1 and Section 4.1). Outputs from the 1.3B InstructGPT are preferred to the 175B GPT-3, a model with more than 100 times as many parameters (Figure 1, Section 1). The alignment compute is small next to pretraining. Section 5.1 gives 4.9 petaflops/s-days for the 175B SFT model and 60 for the 175B PPO-ptx model, against 3,640 for GPT-3. `scripts/pipeline_numbers.py` prints that the first two sum to 64.9, or 1.78% of the third, and that 175 / 1.3 is 134.6.

The method has a cost, which the paper calls an "alignment tax": plain RLHF lowers scores on public NLP benchmarks such as SQuAD and DROP (Section 1 and Section 4.2). The PPO-ptx variant in the second key idea below is the paper's mitigation.

## Key ideas

The sections that follow use the paper's notation: a prompt $x$, a response $y$, the reward model $r_\theta(x, y)$, the RL policy $\pi^{RL}_\phi$, the supervised policy $\pi^{SFT}$, the KL reward coefficient $\beta$, and the pretraining loss coefficient $\gamma$. A labeler's preferred response in a pair is $y_w$ and the rejected one is $y_l$. The paper says "KL reward coefficient" for $\beta$ and uses $\beta = 0.02$ throughout (Appendix C.4). Every number comes from a script in `scripts/`. Run any script from the repository root to reproduce the output.

### Three stages: SFT, reward model, PPO

**Intuition.** Each stage gets its signal from a different kind of human effort. Writing a good response is slow, so the paper uses demonstrations only to get a reasonable starting policy. Ranking responses is faster and still tells you which direction is better, so the paper uses rankings to train a reward model. The reward model can then score millions of samples that no human reads, and RL uses those scores.

**Math.** Section 3.1 lists the three steps (Figure 2):

1. Collect demonstrations and train a supervised policy $\pi^{SFT}$ by fine-tuning GPT-3. The paper trains for 16 epochs with a cosine learning-rate schedule and residual dropout of 0.2, and selects the final model by reward-model score on a validation set (Section 3.5, Appendix C.1).
1. Collect comparisons between model outputs and train a reward model $r_\theta(x, y)$ that outputs a scalar. The reward model starts from the SFT model with the final unembedding layer removed (Section 3.5). In practice, the paper's final 6B reward model is initialized from a GPT-3 6B model fine-tuned on public NLP datasets, and the paper reports similar results when it starts from GPT-3 or the SFT model (Appendix C.2).
1. Optimize a policy against the reward model with PPO. The paper initializes the RL policy from an SFT model that also mixes in 10% pretraining data (Appendix C.3). It initializes the value function from the reward model (Section 3.5).

Steps 2 and 3 can be repeated. Most of the comparison data comes from the supervised policies, with some from PPO policies (Section 3.1). Table 6 in Appendix A.3 gives the sizes in prompts: the SFT training split has 11,295 labeler-written and 1,430 customer prompts, the reward-model training split has 6,623 labeler and 26,584 customer prompts, and the PPO training split has 31,144 customer prompts.

**Tiny example, computed by `scripts/pipeline_numbers.py`.** Adding up Table 6 gives 12,725 SFT prompts, 33,207 reward-model prompts, and 31,144 PPO prompts. The reward-model split has 2.61 times as many prompts as the SFT split. Because each reward-model prompt carries a ranking of 4 to 9 responses, the number of training comparisons is, as Appendix A.3 puts it, an order of magnitude larger than the number of prompts. The paper also reports that the RL stage runs for 256,000 episodes with a batch of 512 and a minibatch of 64, which the script turns into 500 iterations of 8 minibatches each (Appendix C.4).

### The reward model from rankings

**Intuition.** Christiano et al. show two clips and ask which is better. A labeler for language responses can rank several at once, and the ranking holds more information per minute of labeling. A ranking of $K$ responses gives $\binom{K}{2}$ pairwise comparisons. The paper shows that you can't treat those pairs as independent data points: they are strongly correlated within one prompt, so shuffling them into one dataset makes the reward model overfit after a single pass.

**Math.** Section 3.5 presents labelers with between $K = 4$ and $K = 9$ responses. The loss treats all $\binom{K}{2}$ comparisons from one prompt as a single batch element (Section 3.5, Equation 1):

$$
\text{loss}(\theta) = -\frac{1}{\binom{K}{2}}\, \mathbb{E}_{(x, y_w, y_l)\sim D}\Big[\log\sigma\big(r_\theta(x, y_w) - r_\theta(x, y_l)\big)\Big]
$$

This is the Bradley-Terry loss of [Christiano et al.](/christiano/) averaged over all pairs in a ranking. The difference of rewards is the log odds that a human prefers $y_w$ to $y_l$. The batching has a compute payoff. The reward model needs one forward pass for each of the $K$ completions, where separate pairs would need $\binom{K}{2}$ passes (Section 3.5). The paper reports that it also gives better validation accuracy and log loss. Ties are dropped (Appendix C.2). Because the loss depends only on differences, the paper adds a bias to the reward model before RL so that labeler demonstrations score 0 on average (Section 3.5).

**Tiny example, computed by `scripts/rm_loss_example.py`.** One prompt, $K = 4$ responses A, B, C, D, with scores $r = [0.2, 1.5, -0.7, 0.9]$ and the labeler ranking B $\succ$ D $\succ$ A $\succ$ C. These scores are made up.

| Pair $(y_w \succ y_l)$ | $r_w - r_l$ | $-\log\sigma(r_w - r_l)$ |
|---|---|---|
| B $\succ$ D | +0.6 | 0.4375 |
| B $\succ$ A | +1.3 | 0.2410 |
| B $\succ$ C | +2.2 | 0.1051 |
| D $\succ$ A | +0.7 | 0.4032 |
| D $\succ$ C | +1.6 | 0.1839 |
| A $\succ$ C | +0.9 | 0.3412 |

The six values sum to 1.7118, and Equation 1 divides by $\binom{4}{2} = 6$, so the loss is 0.2853. Add 100 to every score and the loss is still 0.2853. Reverse the ranking, so the labeler disagrees with every score gap, and the loss rises to 1.5020. For $K = 9$ there are 36 pairs, and the largest batch the paper describes is $64 \times 36 = 2{,}304$ comparisons (Appendix C.2).

The paper uses only 6B reward models, because a 175B reward model was unstable to train and made a poor value-function initialization, and because it adds a lot of PPO compute (Section 3.5, Appendix C.2). Labelers agree with each other on 72.6 ± 1.5% of pairs, and held-out labelers on 77.3 ± 1.3% (Section 3.4), so the reward model is fitting a noisy target.

Exercise 1 asks you to implement this loss.

### The KL penalty against the SFT policy

**Intuition.** A reward model is a learned approximation that is only reliable near the data it was trained on. A policy that maximizes it will find responses the model overrates. The paper adds a penalty that grows as the policy drifts from the SFT model, "to mitigate over-optimization of the reward model" (Section 3.5). Think of the SFT model as a tether: the longer the leash ($1/\beta$), the further RL can wander.

**Math.** The environment is a bandit. It presents a random customer prompt, the policy writes a response, the reward model scores it, and the episode ends (Section 3.5). The reward model's scalar enters Equation 2 together with a log ratio of the two policies. Written for one sampled pair $(x, y)$, the RL part of the objective is

$$
r_\theta(x, y) - \beta \log\!\Big(\pi^{RL}_\phi(y \mid x) \,/\, \pi^{SFT}(y \mid x)\Big)
$$

(Section 3.5, Equation 2, first term). A response is a sequence of tokens, so $\log\pi(y \mid x) = \sum_t \log\pi(y_t \mid x, y_{<t})$, and the log ratio of the whole response is the sum of per-token log ratios. The paper describes the penalty as "per-token" (Section 3.5). Writing the penalty at each token gives a per-token reward

$$
R_t = -\beta\big(\log\pi^{RL}(y_t \mid x, y_{<t}) - \log\pi^{SFT}(y_t \mid x, y_{<t})\big) \;+\; \mathbb{1}[t = T]\; r_\theta(x, y)
$$

The paper doesn't write $R_t$. It is this page's way to see where the terms go, and it sums to the integrand above. Since $y$ is sampled from $\pi^{RL}$, the expected log ratio is the KL divergence $\mathrm{KL}(\pi^{RL} \,\|\, \pi^{SFT})$, so the penalty measures how far the policy has moved. A single sample can be negative, because it is a sample of a quantity whose mean is non-negative.

**Tiny example, computed by `scripts/kl_reward_example.py`.** A response of 5 tokens, with sampled-token log-probabilities under the RL policy $[-1.2, -0.8, -2.1, -0.5, -1.0]$ and under the SFT policy $[-1.5, -1.0, -1.9, -0.9, -1.4]$. The per-token log ratios are $[0.3, 0.2, -0.2, 0.4, 0.4]$, which sum to 1.1. The reward model scores the response 1.8. These numbers are made up.

| $\beta$ | Per-token rewards $R_1, \dots, R_5$ | Sum |
|---|---|---|
| 0.02 (the paper's value) | $[-0.006, -0.004, 0.004, -0.008, 1.792]$ | 1.778 |
| 0.1 | $[-0.03, -0.02, 0.02, -0.04, 1.76]$ | 1.690 |
| 0.5 | $[-0.15, -0.10, 0.10, -0.20, 1.60]$ | 1.250 |

Each sum equals $1.8 - \beta \cdot 1.1$, which is the Equation 2 integrand. Token 3 has a negative log ratio, so its penalty is a small bonus.

`scripts/kl_optimum.py` shows what $\beta$ controls, on a three-response toy problem. For the objective $\mathbb{E}[r] - \beta\,\mathrm{KL}(\pi\,\|\,\pi^{SFT})$ over a finite set of responses, the maximizer is $\pi^*(y) \propto \pi^{SFT}(y)\,e^{r(y)/\beta}$. This closed form is a standard result, and the InstructGPT paper doesn't derive it. Take $\pi^{SFT} = [0.70, 0.25, 0.05]$ and rewards $[0, 1, 2]$. The SFT policy's expected reward is 0.35.

| $\beta$ | Optimal policy $\pi^*$ | $\mathbb{E}[r]$ | $\mathrm{KL}(\pi^*\,\|\,\pi^{SFT})$ |
|---|---|---|---|
| 10 | $[0.6748, 0.2663, 0.0589]$ | 0.3841 | 0.0017 |
| 1 | $[0.4002, 0.3885, 0.2112]$ | 0.8110 | 0.2520 |
| 0.5 | $[0.1326, 0.3500, 0.5173]$ | 1.3847 | 1.1059 |
| 0.1 | $[0.0000, 0.0002, 0.9998]$ | 1.9998 | 2.9932 |
| 0.02 | $[0, 0, 1]$ | 2.0000 | 2.9957 |

A large $\beta$ keeps $\pi^*$ near the SFT policy. A small $\beta$ sends it to the highest-scoring response, whose SFT probability was only 0.05. The KL reaches $-\log 0.05 = 2.9957$, the cost of putting all the mass on a response the SFT model rarely wrote. In this toy the reward is exact. In the real system the reward model is wrong in places the policy can reach, so the same collapse picks out errors in the reward model. The paper's $\beta = 0.02$ is small, and the paper doesn't report an ablation on $\beta$ that supports a comparison with these toy values.

Exercise 2 asks you to write the per-token reward.

### Mixing in pretraining gradients: PPO-ptx

**Intuition.** RL on the reward model makes the model better at the labelers' task and can make it worse at tasks the labelers never rated. The paper notes regressions on SQuAD, DROP, HellaSwag, and WMT 2015 French to English (Section 1). The remedy is to keep training on pretraining text at the same time, so the policy stays a good language model.

**Math.** The full objective adds a pretraining log-likelihood term (Section 3.5, Equation 2):

$$
\text{objective}(\phi) = \mathbb{E}_{(x,y)\sim D_{\pi^{RL}_\phi}}\Big[r_\theta(x,y) - \beta\log\!\big(\pi^{RL}_\phi(y\mid x)/\pi^{SFT}(y\mid x)\big)\Big] + \gamma\,\mathbb{E}_{x\sim D_{\text{pretrain}}}\Big[\log\pi^{RL}_\phi(x)\Big]
$$

The paper maximizes this objective. For the models it calls "PPO", $\gamma = 0$. The models it calls "PPO-ptx" use $\gamma = 27.8$ and draw 8 times as many pretraining examples as there are RL episodes (Appendix C.4). Unless it says otherwise, "InstructGPT" in the paper means PPO-ptx (Section 3.5). In the implementation, each minibatch computes the PPO gradient and the pretraining gradient in consecutive steps, accumulates both in the gradient buffer, and multiplies the pretraining gradient by $\gamma$ (Appendix C.4). The pretraining term as printed in the paper is $\log\pi^{RL}_\phi(x)$, which this page reads as the log-likelihood of a pretraining text $x$ under the RL policy.

**Tiny example, computed by `scripts/kl_reward_example.py`.** With $\beta = 0.02$ and the response above, the RL part is $1.8 - 0.02 \cdot 1.1 = 1.778$. Suppose one pretraining text has log-likelihood $-2.3$ under the RL policy. With $\gamma = 27.8$ the pretraining term is $27.8 \cdot (-2.3) = -63.94$, and the combined value is $-62.162$. The two terms are on different scales on purpose. The large $\gamma$ makes the pretraining gradient comparable to the PPO gradient (Appendix C.4). The paper reports that PPO-ptx greatly reduces the benchmark regressions without compromising labeler preference scores (Section 1, Section 4.2).

Exercise 3 asks you to compute this objective from samples.

### What the experiments show

**Intuition.** The claim isn't that RLHF produces a better language model on benchmarks. It is that for the task people actually send to an API, human preferences are a better target than next-token prediction or public instruction datasets.

**Results the page doesn't compute.** The numbers below come from the paper. The page's scripts don't reproduce them, because they need human labelers.

- Labelers prefer InstructGPT to GPT-3 across model sizes, and the 1.3B PPO-ptx model beats the 175B GPT-3 (Figure 1, Section 4.1).
- On TruthfulQA, InstructGPT produces truthful and informative answers about twice as often as GPT-3. On closed-domain tasks, it makes up information not present in the input at a 21% rate against 41% for GPT-3 (Section 1).
- The paper reports about 25% fewer toxic outputs than GPT-3 when prompted to be respectful, and no significant improvement on Winogender or CrowS-Pairs (Section 1).
- Fine-tuning 175B GPT-3 on FLAN or T0 does worse than the SFT baseline on the API distribution. InstructGPT has a 73.4 ± 2% win rate against the SFT baseline, and the paper's T0 and FLAN versions have 26.8 ± 2% and 29.8 ± 2% (Section 1).
- Held-out labelers, who produced no training data, prefer InstructGPT at about the same rate as training labelers, a preliminary result (Section 1, Section 4.1).

<!-- exercises -->

## Pitfalls

- **Treating the pairs from one ranking as independent.** Section 3.5 and footnote 5 say that shuffling all $\binom{K}{2}$ comparisons into one dataset makes each completion appear in $K - 1$ gradient updates, and the reward model overfits after one epoch. Train on all pairs of a prompt as one batch element, for one epoch.
- **Dividing by $K$, not by $\binom{K}{2}$.** The $1/\binom{K}{2}$ factor in Equation 1 makes a prompt with 9 responses count the same as one with 4. Summing instead of averaging weights long rankings more. Exercise 1 tests the mean.
- **Flipping the sign of the difference.** The winner's score comes first in $r_\theta(x, y_w) - r_\theta(x, y_l)$. With the sign flipped, the reward model learns what labelers rejected, and RL optimizes that faithfully. Exercise 1's mutants for this bug fail seven tests.
- **Paying the reward model's score at every token.** The environment is a bandit, and the reward model scores the finished response once. Adding the score to every token multiplies it by the response length. Exercise 2 tests where the score goes.
- **Averaging the log ratio over tokens.** The KL term in Equation 2 is the log ratio of whole-response probabilities, which is a sum over tokens. A mean shrinks the penalty by the length $T$, so long responses are barely tethered. Exercise 3 tests the sum.
- **Reading a single-sample log ratio as a KL.** The per-token log ratio is a sample of a quantity with a non-negative mean. Individual values are often negative, as token 3 is above. Clipping them at zero changes the objective.
- **Letting the KL tether drift.** The penalty is measured against the frozen SFT policy, and the SFT model must stay fixed. If you compare against the previous policy iterate instead, the tether moves with the policy and no longer limits total drift. PPO's own clip ratio compares against the previous iterate, so the two comparisons use different reference policies.
- **Forgetting that the reward is only defined up to a shift.** Equation 1 depends only on differences. The paper adds a bias so that labeler demonstrations score 0 on average before RL (Section 3.5). A reward that drifts in position changes the scale of the KL term relative to the reward.
- **Reading "aligned" as "aligned to everyone".** The paper says the procedure aligns the model to the preferences of a specific group, mostly its labelers and researchers, and not to a broader notion of human values (Section 1, Section 5.2). Labelers agree on 72.6% of pairs, so the training signal is noisy.
- **Expecting PPO-ptx numbers from a plain PPO run.** The paper's regression fix needs $\gamma = 27.8$ and 8 times more pretraining examples than RL episodes (Appendix C.4). A small $\gamma$ barely changes the PPO gradient.
- **Reading the figures as measurements of harm reduction.** The paper says InstructGPT still makes simple mistakes: it can fail to follow instructions, make up facts, and give long hedging answers (Section 1).

## Further reading

- The [paper's arXiv page](https://arxiv.org/abs/2203.02155), and OpenAI's [code and samples](https://github.com/openai/following-instructions-human-feedback) that the paper links in a footnote in Section 4.
- Stiennon et al., "Learning to summarize from human feedback" (2020), the summarization study whose method this paper follows (Section 3.1).
- Ziegler et al., "Fine-Tuning Language Models from Human Preferences" (2019), the first use of this three-part recipe with a KL penalty on text.
- [Direct preference optimization](/dpo/), which removes the explicit reward model and the RL loop from this pipeline.
