---
id: "christiano"
title: "Deep reinforcement learning from human preferences"
authors: "Christiano et al."
year: 2017
venue: "NeurIPS 2017"
arxiv: "1706.03741"
url: "https://arxiv.org/abs/1706.03741"
track: rl
tier: core
est_hours: 4
status: full
summary: "The paper trains an agent without a reward function by asking a human which of two short trajectory clips is better. A reward predictor is fit to those comparisons with a Bradley-Terry cross-entropy loss, and a standard RL algorithm maximizes the predicted reward while new queries keep arriving. A few hundred to a few thousand comparisons, under 1% of the agent's interactions, are enough for MuJoCo tasks, Atari games, and new behaviors such as a backflip."
related_course: "rl: RLHF"
prereqs:
  - id: ppo
    reason: "This paper's policy step uses A2C and TRPO, and PPO is the first-order policy optimizer that later RLHF systems put in the same slot, so its clipped update is the policy half of the loop."
background: [bradley-terry-model, reward-modeling]
course_links:
  - title: "Reward models from preferences"
    url: "https://rl.ifkash.dev/lessons/051-reward-models"
  - title: "RLHF with PPO"
    url: "https://rl.ifkash.dev/lessons/053-rlhf-with-ppo"
self_check:
  - q: 'A reward predictor assigns return $3$ to segment $\sigma^1$ and $1$ to segment $\sigma^2$. What does Equation 1 give for $\hat P[\sigma^1 \succ \sigma^2]$, and what happens to it if you add $100$ to both returns?'
    a: 'It gives $e^3 / (e^3 + e^1) = \sigma(2) \approx 0.8808$. Adding $100$ to both returns changes nothing, because the probability depends only on the difference. This is why the learned reward is defined only up to a constant shift.'
    see: bradley-terry-model
  - q: 'The loss compares summed rewards of two segments. Why does the paper normalize the predicted reward to a fixed standard deviation before the policy sees it (Section 2.2.1), and what does the normalization not change?'
    a: 'The comparison loss fixes only differences of returns, so the position and scale of the reward are arbitrary. A reward that drifts in scale while the predictor trains would act like a changing learning rate for the policy. Shifting and rescaling by a positive constant keeps the ranking of segments, so it doesn''t change which clip the predictor prefers.'
    see: reward-modeling
  - q: 'The reward predictor keeps changing while the policy trains. Which policy optimizers does the paper use, and why does the paper prefer policy gradient methods here?'
    a: 'It uses A2C for Atari and TRPO for MuJoCo (Section 2.2.1). The reward function is non-stationary, so the paper prefers policy gradient methods, which it treats as robust to changes in the reward function. PPO, a later first-order method, fills the same role.'
    see: ppo
---

## The problem

Reinforcement learning (RL) needs a reward function, and for many goals nobody can write one. Section 1 gives the example of a robot that cleans a table or scrambles an egg: a simple hand-written reward tends to be optimized without the behavior it was meant to describe actually satisfying the designer's preferences. Demonstrations don't always help either. A human can't easily demonstrate how to steer a robot with many joints and a body that is nothing like a human body.

The obvious fix is to let a human score the agent's behavior directly. That doesn't scale. Section 1 notes that deep RL needs hundreds or thousands of hours of experience, so human feedback on every step is prohibitively expensive. The paper asks for several orders of magnitude fewer human queries.

Here is the budget the paper actually reaches on Atari. The runs train for 50 million timesteps (Appendix A.2) and ask the human 5,500 questions (Section 3.1.2). Each question shows two 25-timestep segments. `scripts/feedback_budget.py` prints the totals.

| Quantity | Value |
|---|---|
| Timesteps shown to the human (5,500 questions, 2 segments of 25 steps each) | 275,000 |
| Fraction of the 50,000,000 RL timesteps | 0.55% |
| Labels per RL timestep | 1 in 9,091 |
| Human time at 3 to 5 seconds per query (Section 3.1) | 4.6 to 7.6 hours |

The paper reports roughly 5 hours for 5,000 labels (Section 4, footnote 6). The point of the table is the ratio. The agent has orders of magnitude more interaction data than the human can label, so the human's judgments must be spread through a learned reward model and not applied one step at a time.

## Key ideas

The sections that follow use the paper's notation: observations $o_t$, actions $a_t$, a trajectory segment $\sigma = ((o_0, a_0), \dots, (o_{k-1}, a_{k-1}))$, a preference $\sigma^1 \succ \sigma^2$, the reward estimate $\hat r$, and the human's label $\mu$, which the paper defines as a distribution over $\{1, 2\}$. This page writes $\mu$ as the probability mass on segment 1, so $\mu(2) = 1 - \mu$, and the exercises use the same convention. The paper's loss sums over the database $D$ of labeled triples $(\sigma^1, \sigma^2, \mu)$. The exercises take the mean over a batch, which rescales the gradient and nothing else. Every number comes from a script in `scripts/`. Run any script from the paper folder to reproduce the output.

### Learn a reward from pairwise comparisons

**Intuition.** Treat the reward as a hidden score that explains the human's choices. A human who compares two clips picks the one with the higher total score most of the time, and picks it more reliably when the gap is larger. This is how Elo ratings work in chess: the difference in two ratings predicts the chance that one player beats the other (Section 2.2.3). Comparing clips is also easier for people than assigning each clip an absolute score. Section 3.3 reports that predicting comparisons worked much better than predicting scores on the continuous control tasks.

**Math.** Section 2.1 defines the setting. The agent sees observations and takes actions, and a human compares segments $\sigma^1$ and $\sigma^2$. The reward estimate $\hat r : O \times A \to \mathbb{R}$ is a neural network. The paper models the human as choosing a segment with probability that depends exponentially on its summed reward (Section 2.2.3, Equation 1):

$$
\hat P[\sigma^1 \succ \sigma^2] = \frac{\exp \sum_t \hat r(o^1_t, a^1_t)}{\exp \sum_t \hat r(o^1_t, a^1_t) + \exp \sum_t \hat r(o^2_t, a^2_t)}
$$

Write $R^i = \sum_t \hat r(o^i_t, a^i_t)$ for the return of segment $i$. Dividing the numerator and denominator by $\exp R^1$ gives $\hat P[\sigma^1 \succ \sigma^2] = \sigma(R^1 - R^2)$, where $\sigma(x) = 1/(1+e^{-x})$ is the logistic function. The probability depends only on the difference of returns. The paper fits $\hat r$ by minimizing the cross-entropy between this prediction and the human's labels. The loss appears just below Equation 1 and has no equation number:

$$
\operatorname{loss}(\hat r) = -\sum_{(\sigma^1, \sigma^2, \mu) \in D} \Big( \mu(1) \log \hat P[\sigma^1 \succ \sigma^2] + \mu(2) \log \hat P[\sigma^2 \succ \sigma^1] \Big)
$$

The paper names this the Bradley-Terry model and says it specializes the Luce-Shepard choice rule to segments. The label $\mu$ handles the three answers a human can give (Section 2.2.2): if they pick a segment, $\mu$ puts all its mass there. If they say the segments are equally good, $\mu$ is uniform, so a tie is $\mu = 0.5$. If they can't compare the segments, the comparison is dropped.

**Tiny example, computed by `scripts/bt_example.py`.** Segment 1 has return 3 and segment 2 has return 1, so $\hat P[\sigma^1 \succ \sigma^2] = \sigma(2) = 0.8808$. The loss for this pair depends on the label:

| Human says | $\mu$ | Loss |
|---|---|---|
| Segment 1 | 1 | 0.1269 |
| Tie | 0.5 | 1.1269 |
| Segment 2 | 0 | 2.1269 |

Equal returns give $\hat P = 0.5$. Adding 100 to both returns leaves $\hat P$ at 0.8808, so the learned reward is defined only up to a constant shift. Exercise 1 asks you to write this probability and loss, with an overflow-safe log.

**A modification to Equation 1.** Section 2.2.3 lists several changes the authors made to the basic loss. One is to assume a 10% chance that the human answers uniformly at random instead of applying the softmax of Equation 1 directly. With $\epsilon = 0.1$ the probability becomes $(1-\epsilon)\hat P + \epsilon/2$. The same script shows the effect. The prediction for the example above falls from 0.8808 to 0.8427. More importantly, a confident wrong answer now has bounded cost. With a return gap of 50 and a label that disagrees, the plain loss is 50.0000 and the adjusted loss is 2.9957, which equals $-\log(\epsilon/2)$. The adjustment keeps a few wrong labels from dominating the fit, because a human rater has a constant chance of an error that doesn't shrink as the reward gap grows.

**Fit it, computed by `scripts/fit_reward.py`.** The paper's reward is a neural network. To see the learning problem without one, the script uses a linear reward $\hat r(o, a) = w \cdot \phi(o, a)$ with 6 features, segments of 25 steps, and a hidden true $w$. Labels come from the true segment returns. The script averages 20 seeds and measures two things on 2,000 held-out pairs: the correlation between the learned and the true step reward, and the fraction of held-out pairs ranked correctly.

| Queries | Correlation, clean labels | Pair accuracy, clean labels | Correlation, 10% random labels | Pair accuracy, 10% random labels |
|---|---|---|---|---|
| 10 | 0.846 | 0.829 | 0.760 | 0.786 |
| 25 | 0.941 | 0.898 | 0.855 | 0.843 |
| 50 | 0.982 | 0.941 | 0.881 | 0.853 |
| 100 | 0.994 | 0.971 | 0.903 | 0.868 |
| 200 | 0.999 | 0.987 | 0.920 | 0.874 |
| 400 | 1.000 | 0.992 | 0.896 | 0.863 |
| 800 | 1.000 | 0.997 | 0.925 | 0.876 |

A hundred comparisons recover this small reward almost exactly. This isn't evidence about the paper's networks and tasks, whose rewards are far richer. It shows why the fit is data-cheap: each comparison of two 25-step segments constrains the whole reward through a sum over steps. Label noise lowers the ceiling. The noisy columns don't climb smoothly, because the fit is plain gradient descent for a fixed 400 steps with no noise model, and each row draws a fresh set of pairs. Exercise 2 asks you to write the gradient and the fitting loop.

### Alternate reward learning and policy optimization

**Intuition.** The reward model only sees clips that the agent produces. If the agent changes, the clips change, and a model trained on old clips says nothing reliable about the new ones. So the paper doesn't collect labels once and train. It runs three processes at once (Section 2.2, Figure 1): the policy interacts with the environment, pairs of segments from those trajectories go to the human, and the reward estimate is refit on all labels so far.

**Math.** There is no new equation here, only a loop. The policy $\pi$ maximizes $\sum_t \hat r(o_t, a_t)$ using any RL algorithm that suits the domain (Section 2.2.1). The paper uses A2C for Atari and TRPO for MuJoCo, and chooses policy gradient methods because $\hat r$ may be non-stationary. The paper normalizes the output of $\hat r$ to zero mean and constant standard deviation before the policy sees it, which suits the problem because the position of the reward is underdetermined by the comparison loss. The loop runs asynchronously, so trajectories flow from the policy to the query selector, labels flow to the predictor, and new predictor parameters flow back to the policy.

Appendix A adds the details that make this work in practice. Labels arrive at a rate that decays roughly as $1/T$ in the number of frames $T$ (Appendix A.1 and A.2), so early training has many labels per timestep and late training has few. The Atari predictor is pretrained for 200 epochs before RL starts, and it keeps a buffer of the last 3,000 labels.

**Tiny example, computed by `scripts/online_vs_offline.py`.** Section 3.3 reports that collecting labels only at the start ("no online queries") performs poorly. The predictor captures only part of the true reward, and maximizing that partial reward can produce undesirable behavior. The paper's Pong runs show this as endless volleys. The script builds the smallest version of this failure. The agent picks one of 21 points $x \in [0, 1]$. The true reward is $1 - 4(x - 0.5)^2$, which peaks at $x = 0.5$ and is 0 at both ends. The reward model is linear in $x$, so it can only say "more $x$ is better" or "less $x$ is better". The starting policy sits near $x = 0.2$, where more $x$ really is better. Both runs get the same 400 labels from a labeler who sees the true reward.

| Labels collected | Final mean $x$ | Expected true reward |
|---|---|---|
| Start policy | 0.201 | 0.616 |
| Offline, all 400 from the start policy | 1.000 | 0.000 |
| Online, 8 rounds of 50 from the current policy | 0.519 | 0.956 |

The offline predictor learns a slope of about $+15.6$, so the policy runs to the end of the line, where the true reward is 0. The online predictor sees comparisons from the region around the peak, where the true preference reverses, and its slope turns negative (about $-7.8$ in the last round, averaged over 20 seeds). It pulls the policy back toward the peak. The toy is deliberately crude: the model class is misspecified and the policy step is an idealized natural-gradient update. It isolates the mechanism the paper describes, which is that a reward model is only trustworthy where it has seen labeled data.

### Choose which pairs to ask about

**Intuition.** A query is expensive, so ask about pairs where the answer would teach the predictor the most. If several predictors trained on different data agree about a pair, a label adds little. If they disagree, a label settles something.

**Math.** The paper trains an ensemble of predictors, each on $|D|$ triples sampled from $D$ with replacement, and normalizes each one independently before it averages them (Section 2.2.3, first bullet). To choose queries, it samples many candidate pairs of segments of length $k$, lets each predictor say which segment it expects to be preferred, and selects the pairs whose predictions have the highest variance across members (Section 2.2.4). Appendix A says it draws a factor of 10 more candidates than it shows the human, with an ensemble of 3 predictors unless stated otherwise. The paper calls this a crude approximation and notes that its ablations show it hurts performance on some tasks (Section 2.2.4). This page measures the disagreement as the variance of $\hat P[\sigma^1 \succ \sigma^2]$ across members, which is one reading of "variance in predictions". Other choices, such as the variance of return differences, also fit the sentence but rank pairs differently.

**Tiny example, computed by `scripts/query_example.py`.** Three one-feature predictors have weights 1, $-1$, and 2. The table lists four candidate pairs by the feature gap between their two segments.

| Pair | Feature gap | Member probabilities | Variance |
|---|---|---|---|
| 0 | 0 | 0.5, 0.5, 0.5 | 0.0000 |
| 1 | 3 | 0.953, 0.047, 0.998 | 0.1916 |
| 2 | 0.1 | 0.525, 0.475, 0.550 | 0.0010 |
| 3 | 1 | 0.731, 0.269, 0.881 | 0.0678 |

With two queries to spend, the rule picks pairs 1 and 3. Pair 1 has the largest gap and also the biggest disagreement, because the members disagree about its sign. Exercise 3 asks you to write the selection rule, and the per-member normalization and averaging that the ensemble uses.

### What the experiments show

**Intuition.** The paper's claim is not that the learned reward is better than the true reward. It is that human preferences can replace the true reward at a feedback cost low enough to be practical.

**Results the page doesn't compute.** Section 3.1.1 reports that with 700 labels the method nearly matches RL on eight simulated robotics tasks, and with 1,400 synthetic labels it performs slightly better than RL with the true reward, a result the authors attribute to the learned reward being slightly better shaped. Section 3.1.2 reports that on Atari, with 5,500 labels, the method learns substantially on most games but has more difficulty than on MuJoCo. Section 3.2 shows novel behaviors for which no reward function is known: a Hopper backflip (900 queries, under an hour), a Half-Cheetah that moves forward standing on one leg (800 queries), and keeping alongside other cars in Enduro (about 1,300 queries). Section 4 concludes that the method cuts interaction complexity by roughly 3 orders of magnitude. These numbers come from the paper, and the page's scripts don't reproduce them.

<!-- exercises -->

## Pitfalls

- **Label orientation.** In the paper, $\mu(1)$ is the mass on $\sigma^1$, and $\hat P[\sigma^1 \succ \sigma^2]$ is the probability that goes with it. If you swap the labels, or the order of the two returns, the reward learns to prefer the segments that the human rejected, and the policy then optimizes it faithfully. Exercise 1's mutant for this bug fails four tests.
- **Overflow in the exponentials.** Equation 1 as printed exponentiates a sum of rewards. For long segments or a large reward scale the exponent exceeds the range of a float and the loss turns to `nan`. Compute the loss through $\log\sigma$, for example with `logaddexp`.
- **Summing versus averaging over the segment.** Equation 1 sums $\hat r$ over the segment, and it doesn't average. Averaging rescales the logit by $1/k$, so the learned reward ends up $k$ times too large. It also changes the effect of segments of different lengths. Exercise 2 tests the sum.
- **Treating ties as a hard label.** A tie is $\mu = 0.5$, not a choice of either side, and the paper drops "incomparable" answers instead. Rounding a tie to 0 or 1 teaches the predictor a preference that the human didn't express. Section 3.1 footnote 4 notes that ties still carry information when they are not too common.
- **Trusting the learned reward off distribution.** The reward model has only seen labeled behavior. The policy will find whatever the model scores highly, including behavior it has never been labeled for. The paper names this risk in Section 3.3, and it concludes that human feedback has to be interleaved with RL learning and can't be provided statically. The Pong volleys are the paper's example.
- **Ignoring the reward's scale and offset.** Only differences of returns matter in Equation 1, so the scale and the offset of $\hat r$ are arbitrary. A policy sees them. The paper normalizes the predictions before the policy uses them (Section 2.2.1, and Appendix A for the target standard deviations: 1 for MuJoCo and 0.05 for Atari). A drifting scale acts like a drifting learning rate.
- **Leaking the reward through the environment.** Episode termination, life-loss signals, and the on-screen score all carry reward information. Appendix A says the authors removed variable-length episodes, replaced termination with a learned penalty, and blanked the score area on Atari, because otherwise the experiment would not test learning from preferences alone. A reimplementation that keeps these signals can look better than the paper's method for the wrong reason.
- **Reading the query selection as settled.** Section 2.2.4 calls the ensemble-disagreement rule a crude approximation, and the ablations show that it hurts on some tasks. Random queries are a strong baseline.
- **Expecting PPO or an RLHF language-model pipeline here.** This paper predates PPO and works with video clips of control and game agents. The policy step is A2C or TRPO, and the human sees clips of 1 to 2 seconds. The idea of a learned reward from pairwise comparisons carries over to later work on language models, and those systems differ in the policy optimizer, the reward architecture, and the comparison data.

## Further reading

- OpenAI's blog post [Learning from human preferences](https://openai.com/index/learning-from-human-preferences/), which has the Hopper backflip and the Enduro videos that the paper links.
- The [paper's arXiv page](https://arxiv.org/abs/1706.03741), whose latest version (v4) is the one this page cites.
- Bradley and Terry, "Rank analysis of incomplete block designs: I. The method of paired comparisons", Biometrika, 1952, the source of the model in Equation 1.
