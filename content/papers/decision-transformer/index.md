---
id: "decision-transformer"
title: "Decision Transformer: Reinforcement Learning via Sequence Modeling"
authors: "Chen et al."
year: 2021
venue: "NeurIPS 2021"
arxiv: "2106.01345"
url: "https://arxiv.org/abs/2106.01345"
track: rl
tier: core
est_hours: 5
status: full
summary: "Decision Transformer treats offline reinforcement learning as sequence modeling. A GPT-style causal transformer reads trajectories written as return-to-go, state, and action tokens and learns to predict each action with a plain supervised loss, with no value function and no bootstrapping. At test time you pick a target return, feed it in as the first token, and the model generates actions that aim to achieve it, which matches or beats strong offline RL baselines on Atari, OpenAI Gym, and Key-to-Door."
related_course: "rl: RL as sequence modeling"
prereqs:
  - id: transformer
    reason: "Decision Transformer is a GPT, so it reuses causal self-attention, the mask, and the residual block from that paper and changes only what the tokens are."
  - id: dqn
    reason: "DQN is the temporal-difference approach whose bootstrapped target and replay-based training this paper replaces with a supervised loss, so you need its target to see what disappears."
background: [offline-rl, return-to-go]
course_links:
  - title: "Offline reinforcement learning"
    url: "https://rl.ifkash.dev/lessons/045-offline-rl"
  - title: "Imitation learning"
    url: "https://rl.ifkash.dev/lessons/043-imitation-learning"
self_check:
  - q: 'An episode has rewards $r = [1, 0, 2, -1]$. What are the returns-to-go $\hat{R}_1, \dots, \hat{R}_4$, and what does Decision Transformer do with them at test time?'
    a: 'Each entry sums the rewards from that step to the end: $\hat{R} = [2, 1, 1, -1]$. At test time you set $\hat{R}_1$ to the return you want, then after every step you subtract the reward you just received, so $\hat{R}_{t+1} = \hat{R}_t - r_t$ (Section 3).'
    see: return-to-go
  - q: 'The hidden state at the state token $s_t$ predicts $a_t$. Why can the model not read $a_t$ from the input, and which tokens can that hidden state see?'
    a: 'The tokens are ordered $\hat{R}_1, s_1, a_1, \dots$, and the transformer uses a causal mask, so position $i$ attends only to positions $j \le i$ (Section 2.2). The token $a_t$ comes after $s_t$ and is hidden. For $K = 3$ and $t = 2$, the hidden state at $s_2$ sees 5 tokens: $\hat{R}_1, s_1, a_1, \hat{R}_2, s_2$.'
    see: transformer
  - q: 'DQN trains on the target $y = r + \gamma \max_{a''} Q(s'', a'')$. What does Decision Transformer train on instead, and which problem from the DQN pipeline does that avoid?'
    a: 'It trains the output at each state token to match the action in the dataset, with cross-entropy for discrete actions or mean squared error for continuous actions (Section 3). No target depends on the network''s own value estimate, so there is no bootstrapping. The paper says this avoids one part of the "deadly triad" and the need for discounting (Section 1).'
    see: dqn
---

## The problem

Offline RL learns a policy from a fixed dataset of trajectories from other policies. There is no environment to try ideas in, so every mistake in the learned values goes unchecked (Section 2.1). The standard approach is temporal-difference learning: fit a value function with a bootstrapped target, as in [DQN](/dqn/), and improve the policy against it. The paper lists two costs of that approach in Section 1 and Section 5.7. Bootstrapping is one of the "deadly triad" that destabilizes RL. Optimizing against a learned value function also exploits its errors, so offline methods add value pessimism or policy constraints to stay safe.

The obvious alternative is behavior cloning: train a network to predict the dataset's action from the state. It has no values to get wrong, but it copies whatever the data does, good or bad.

Here is a failing example that you can run. Take a corridor of 7 cells, numbered 0 to 6. The agent starts in cell 3, the goal is cell 6, and every step gives reward $-1$ until the episode ends at the goal. The dataset holds only random walks, 2,000 of them with 60,132 timesteps. In state 3, the dataset's actions are almost a coin flip: behavior cloning picks "right" with probability 0.495. A cloned policy takes 30.56 steps on average to reach the goal, and the best possible is 3. This is the same setup as the paper's shortest-path example (Section 1, Figure 2, and Appendix A.3). The script is `scripts/conditioning_example.py`. The paper's graph is a general directed graph with reward 0 at the goal, so the corridor and its constant $-1$ reward are my simplification.

The paper's answer is to keep the supervised loss and add one input: the return the trajectory went on to collect. Section 1 argues that this gives policy improvement with no dynamic programming. The results, as the paper reports them:

- **Atari (Section 4.1, Table 1).** With 1% of the DQN replay data, Decision Transformer scores 267.5 on Breakout against 211.1 for CQL, and 106.1 on Pong against 111.9.
- **D4RL locomotion (Section 4.2, Table 2).** The average over the 9 non-Reacher settings is 74.7 for Decision Transformer, 63.9 for CQL, and 46.4 for behavior cloning.
- **Long-term credit assignment (Section 5.4, Table 6).** On Key-to-Door with 1,000 random trajectories, the success rate is 71.8% against 13.1% for CQL.
- **Delayed rewards (Section 5.6, Table 7).** With the reward moved to the final step of Hopper medium-replay, Decision Transformer scores 78.5 and CQL scores 2.0.

## Key ideas

The sections that follow use the paper's notation: states $s_t$, actions $a_t$, rewards $r_t$, the return-to-go $\hat{R}_t$, a context of $K$ timesteps, and episode length $T$. Section 2.1 writes the return as $R_t$ and Section 3 writes the return-to-go as $\hat{R}_t$. This page uses $\hat{R}_t$ for both. The paper starts timesteps at 1 in Equation 2 and at 0 in Section 2.1. The scripts count from 0, because they are NumPy code, and the page says so where it matters. Every number below comes from a script in `scripts/`. Run any script from the paper folder to reproduce its output.

### Condition on return-to-go

**Intuition.** A reward says what just happened. The return-to-go says what is still to come. A model that sees $\hat{R}_t$ in front of each state can learn which behavior produced a good ending and which produced a bad one. The same dataset then contains both skilled and unskilled behavior, labeled by outcome.

**Math.** The return-to-go is the undiscounted sum of the rewards from step $t$ to the end of the episode (Section 3, Trajectory representation):

$$
\hat{R}_t = \sum_{t'=t}^{T} r_{t'}
$$

The paper drops the discount on purpose, since it avoids short-sighted behavior (Section 1). The entry for step $t$ includes the reward $r_t$ at that step. A trajectory in the paper's representation is (Section 3, Equation 2)

$$
\tau = \left(\hat{R}_1, s_1, a_1, \hat{R}_2, s_2, a_2, \dots, \hat{R}_T, s_T, a_T\right)
$$

At test time you choose $\hat{R}_1$, which is the performance you want. After each action, you subtract the reward you got: $\hat{R}_{t+1} = \hat{R}_t - r_t$ (Section 3 and Algorithm 1). This equation has no number in the paper.

**Tiny example, computed by `scripts/rtg_example.py`.** Take rewards $[1, 0, 2, -1]$. The returns-to-go are $[2, 1, 1, -1]$. Check the first entry by hand: $1 + 0 + 2 - 1 = 2$. If you set the target to 2 and subtract each reward as it arrives, you get $[2, 1, 1, -1]$ again, so the target and the dataset agree.

Two more rows show why the choice of signal matters. For a sparse episode with rewards $[0, 0, 0, 1]$, every return-to-go is 1, so the final reward reaches every earlier step in one pass with no backup. The running total of rewards, $[1, 1, 3, 2]$ for the first episode, is the wrong signal, because it looks backward. An episode with rewards $[2, 0, 0, 0]$ has returns-to-go $[2, 0, 0, 0]$, while $[0, 0, 0, 2]$ gives $[2, 2, 2, 2]$. The two episodes have the same total, and their conditioning tokens differ in time.

Exercise 1 asks you to compute returns-to-go for single episodes and batches.

### Trajectory tokens: return, state, action

**Intuition.** The model is a GPT. It needs a sequence of vectors, so each of the three kinds of data becomes its own token. Because the sequence is causal, the order decides what each prediction can use. Putting the return first and the action last means the model reads the goal and the situation, then the action it must produce.

**Math.** The model takes the last $K$ timesteps, which is $3K$ tokens, one each for return-to-go, state, and action (Section 3, Architecture, and Figure 1). Each modality has its own linear embedding. The paper follows each embedding with layer normalization, and replaces it with Tanh in Atari (Appendix A.1). Image states go through a convolutional encoder instead. A learned embedding of the timestep is added to every token of that timestep. The paper stresses that this differs from the standard positional embedding, because one timestep corresponds to three tokens. In Algorithm 1 the embeddings are

$$
e^{s}_t = \text{embed\_s}(s_t) + \text{embed\_t}(t), \quad
e^{a}_t = \text{embed\_a}(a_t) + \text{embed\_t}(t), \quad
e^{R}_t = \text{embed\_R}(\hat{R}_t) + \text{embed\_t}(t)
$$

The tokens are interleaved as $e^R_1, e^s_1, e^a_1, \dots$ and run through a transformer with the causal mask of the [Transformer paper](/transformer/). In the notation of Section 2.2, Equation 1, the sum and softmax run only over $j \in [1, i]$. The hidden state at each state token goes through a linear layer that predicts the action $a_t$. The prediction at $s_t$ can therefore use $\hat{R}_1, s_1, a_1, \dots, \hat{R}_t, s_t$ and nothing after.

**Tiny example, computed by `scripts/token_layout.py`.** With $K = 3$ the token order is `R_1, s_1, a_1, R_2, s_2, a_2, R_3, s_3, a_3`, nine tokens at positions 0 to 8. The timestep embedding indices are `1, 1, 1, 2, 2, 2, 3, 3, 3`, while the token positions run 0 to 8. The state tokens sit at positions 1, 4, and 7. They see 2, 5, and 8 tokens, which is $3t - 1$ for step $t$. The state token $s_2$ sees `R_1, s_1, a_1, R_2, s_2`.

The sequence is three times longer than the timestep count, so each attention layer has $(3K)^2 = 9K^2$ entries. For the paper's Atari setting of $K = 30$ (Appendix A.1, Table 8), the attention matrix has 8,100 entries, against 900 for a sequence of 30 tokens.

Exercise 2 asks you to build the token sequence and check that the prediction at $s_t$ can't see $a_t$.

### A supervised loss instead of bootstrapping

**Intuition.** Once the return is an input, learning a policy is a prediction problem. Given what came before, including how well the episode ends, predict the action that the dataset took. Nothing in the target depends on the model's own estimate of the future.

**Math.** Section 3 (Training) samples minibatches of length $K$ from the dataset. The output at the state token $s_t$ is trained to predict $a_t$, with cross-entropy for discrete actions and mean squared error for continuous actions, and the per-timestep losses are averaged. For continuous actions, Algorithm 1 writes the loss as `mean((a_preds - a)**2)`. In symbols, with $\hat{a}_t$ the model's prediction:

$$
\mathcal{L} = \frac{1}{K}\sum_{t=1}^{K} \ell\!\left(\hat{a}_t,\, a_t\right)
$$

The paper doesn't number this equation. The paper also reports that predicting states or returns-to-go did not improve performance (Section 3, Training).

Compare it with DQN's target $y = r + \gamma \max_{a'} Q(s', a'; \theta)$. That target uses the network's own value estimate, which is bootstrapping. Decision Transformer has no such term and no discount. Section 5.7 conjectures that this is why it needs no pessimism or behavior regularization: it never optimizes against a learned function that can be exploited.

**Tiny example, computed by `scripts/conditioning_example.py`.** A count table stands in for the network. It is a one-timestep version of the idea and not a transformer: $\pi(a \mid s, \hat{R}) = \operatorname{count}(s, \hat{R}, a) / \operatorname{count}(s, \hat{R})$. On the corridor from the problem section, the table at state 3 reads as follows.

| Return-to-go $\hat{R}$ | Count of left | Count of right | $P(\text{right})$ |
|---|---|---|---|
| $-3$ | 0 | 1,464 | 1.000 |
| $-5$ | 351 | 758 | 0.683 |
| $-7$ | 357 | 477 | 0.572 |
| $-9$ | 320 | 311 | 0.493 |
| any (behavior cloning) | 6,072 | 5,956 | 0.495 |

The first row shows hindsight at work. Every dataset timestep in state 3 with exactly 3 steps left went right, because only the path right, right, right fits, so conditioning on $\hat{R} = -3$ selects that behavior. Behavior cloning averages over all rows and gets a coin flip.

Now roll out the policy from state 3, subtracting each reward from the target (500 episodes per row):

| Policy | Mean steps to the goal | Fewest steps |
|---|---|---|
| Behavior cloning | 30.56 | 3 |
| Conditioned on $\hat{R} = -3$ | 3.00 | 3 |
| Conditioned on $\hat{R} = -5$ | 5.00 | 5 |
| Conditioned on $\hat{R} = -9$ | 9.00 | 9 |

The target controls the behavior. This is the effect that Section 5.2 and Figure 4 measure: the return that the agent achieves tracks the return you ask for. The table has a limit that a neural network softens but doesn't remove. At state 3 with $\hat{R} = -2$, a target that no path can reach, the dataset has 0 entries, so the table has nothing to say. Section 5.2 reports that Decision Transformer can sometimes extrapolate past the best return in the dataset, for example on Seaquest, which a table can't.

Exercise 4 asks you to build this table and use it to act.

### Generating actions with a target return

**Intuition.** At test time the model is a prompted generator. The prompt is the target return plus the first state. Each new action changes the world, and each reward changes how much of the target is left.

**Math.** Algorithm 1 starts the lists with `R = [target_return]`, `t = [1]`, and the first observation. In each step it runs the model, takes the last action output, steps the environment, appends `R[-1] - r` as the new return-to-go and the new state, and keeps only the last $K$ entries. The paper specifies the target at test time as the expert-level return, "1 for success or 0 for failure" in the example of Section 3. For Atari, Appendix A.1 conditions on about 1 or 5 times the best return in the dataset. For Gym, Appendix A.2 uses expert returns, except for HalfCheetah, where 50% of expert performance worked better.

**Tiny example, computed by `scripts/rtg_example.py`.** Take $K = 3$ and an environment that gives reward $+1$ at each of 6 steps. A start target of 6 gives the current return-to-go $6, 5, 4, 3, 2, 1$ at the six steps. The context holds one timestep at step 1, two at step 2, and three from step 3 onward. At step 5 the window holds timesteps 3, 4, and 5, with returns-to-go $4, 3, 2$, and the timestep embedding uses the numbers 3, 4, 5 and not 1, 2, 3.

The pseudocode in Algorithm 1 is a sketch, and one line is easy to misread. It appends `len(R)` as the next timestep after `R` was already cut to $K$ entries, so a literal copy gives every late step the same timestep. The exercise uses the absolute step count instead, which is what the learned timestep embedding is for.

Exercise 3 asks you to write the loop and check the context, the timestep numbers, and the decrement.

<!-- exercises -->

## Pitfalls

- **Using the reward as the token.** The model needs the return still to come, $\hat{R}_t$, and not the reward $r_t$ or the running total. The running total looks backward and is unrelated to what the agent is about to do.
- **Leaving the current reward out of the sum.** $\hat{R}_t$ includes $r_t$. An exclusive sum shifts every token by one step and breaks the test-time update $\hat{R}_{t+1} = \hat{R}_t - r_t$.
- **Forgetting to decrement.** If the target stays at its starting value, the model is told at step 50 that it still has to collect everything. The paper's loop subtracts each reward (Section 3).
- **Ordering the tokens wrongly.** If the action comes before the state in a timestep, the hidden state that predicts $a_t$ can read $a_t$. The loss goes to zero and the model learns nothing.
- **Using a per-token position embedding.** The paper adds one learned embedding per timestep to all three tokens, not one per token position (Section 3, Architecture).
- **Asking for an unreachable return.** The model was trained on returns that occurred in the data. A target far above the data is outside the training distribution. Section 5.2 shows that it sometimes works, and a count table shows that it need not.
- **Mistaking luck for skill.** A high return-to-go in the dataset might come from a lucky transition, and the model conditions on it anyway. The corridor is deterministic, so the example cannot show this. Treat it as a caution about stochastic environments.
- **Comparing numbers across settings.** Atari scores are normalized so that 100 is a professional gamer and 0 is a random policy (Section 4.1). D4RL scores are normalized so that 100 is an expert (Section 4.2). The two scales aren't comparable.
- **Treating the Atari and Gym hyperparameters as one recipe.** Atari uses 6 layers, 8 heads, width 128, and $K = 30$ ($K = 50$ for Pong). Gym uses 3 layers, 1 head, width 128, and $K = 20$ ($K = 5$ for Reacher) (Appendix A, Tables 8 and 9).
- **Calling the exercise 4 table a transformer.** It has one timestep and no learned parameters. It shows what conditioning does. It doesn't show what attention adds, such as the context-length effect in Section 5.3, Table 5, where $K = 1$ drops Pong from 106.1 to 2.5.

## Further reading

- [The paper's project page](https://sites.google.com/berkeley.edu/decision-transformer) links the authors' code, which the paper cites as its code source.
- [kzl/decision-transformer](https://github.com/kzl/decision-transformer) is the authors' public repository.
- [Introducing Decision Transformers on Hugging Face](https://huggingface.co/blog/decision-transformers) is a short walkthrough with an implementation in the Transformers library.
- [minGPT](https://github.com/karpathy/minGPT), by Andrej Karpathy, is the GPT implementation that the paper's Atari model builds on (Appendix A.1).
- [Offline Reinforcement Learning as One Big Sequence Modeling Problem](https://arxiv.org/abs/2106.02039), by Janner et al., is the concurrent Trajectory Transformer, which adds state and return prediction and discretization (Section 6.2).
- [Reinforcement Learning Upside Down](https://arxiv.org/abs/1912.02875), by Schmidhuber, is the return-conditioned supervised learning that Section 6.2 compares with this work.
