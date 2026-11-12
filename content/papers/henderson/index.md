---
id: "henderson"
title: "Deep Reinforcement Learning that Matters"
authors: "Henderson, Islam, Bachman, Pineau, Precup, Meger"
year: 2018
venue: "AAAI 2018"
arxiv: "1709.06560"
url: "https://arxiv.org/abs/1709.06560"
track: rl
tier: extended
est_hours: 5
status: full
summary: "This paper shows that the reported performance of deep policy gradient methods on MuJoCo tasks depends heavily on details that papers rarely control or report: network size and activation, reward scale, random seed, environment, and the codebase. Two groups of five seeds of the same algorithm can give learning curves that a significance test calls different. The authors recommend many seeds, confidence intervals, significance tests, and a full report of hyperparameters and implementation details."
related_course: "rl: evaluating RL"
prereqs:
  - id: ppo
    reason: "PPO is one of the four baselines, and the paper's large-network result measures the KL divergence of its updates."
  - id: trpo
    reason: "TRPO supplies the seed-split and codebase experiments, and its KL constraint is the quantity the paper compares across network sizes."
  - id: ddpg
    reason: "The reward-scale and environment experiments depend on how DDPG's off-policy critic and exploration noise behave in unstable tasks."
background: [bootstrap-confidence-interval, statistical-significance, mean-and-variance]
course_links:
  - title: "Evaluating RL honestly"
    url: "https://rl.ifkash.dev/lessons/059-evaluating-rl"
  - title: "Debugging RL runs"
    url: "https://rl.ifkash.dev/lessons/060-debugging-rl-runs"
self_check:
  - q: 'You train a new algorithm on 3 seeds and a baseline on 3 seeds, and the new algorithm has the higher mean final return. The paper ran 10 seeds of one TRPO configuration on HalfCheetah and split them into two groups of 5. What did that experiment show, and what does it imply for your 3-versus-3 comparison?'
    a: 'The two groups of 5 gave learning curves that a two-sample t-test over the training distribution called different ($t = -9.0916$, $p = 0.0016$ in the Figure 5 caption), although they came from the same algorithm and hyperparameters. The only difference was the random seeds. A gap between two small groups can be seed luck, so 3 versus 3 with a higher mean is not evidence. You need more seeds and an interval or a test.'
    see: trpo
  - q: 'The paper reports a mean with a 95% bootstrap confidence interval (Table 3, 10,000 bootstrap iterations). Describe how you build a percentile interval from 5 final returns, and say what can go wrong with it at this sample size.'
    a: 'Draw $B$ resamples of 5 returns with replacement, take the mean of each, and report the $2.5$th and $97.5$th percentiles of those $B$ means. With 5 returns the resamples can only combine the values you have, so the interval tends to be too narrow. The page measures 83.75% coverage at 5 seeds for a nominal 95% interval, against 93.95% at 30 seeds.'
    see: bootstrap-confidence-interval
  - q: 'DDPG is trained on HalfCheetah with the reward multiplied by $\hat\sigma = 10^{-4}, \dots, 100$. What does the paper find for small $\hat\sigma$, and why does that make a comparison of DDPG against another algorithm fragile?'
    a: 'Figure 3 shows a failure to learn for scales below $\hat\sigma = 0.01$, and layer normalization changes how the scale affects the result. The scale is a hyperparameter that the algorithm does not choose for itself, so a baseline run at the wrong scale looks weak and the comparison measures the tuning effort, not the algorithm.'
    see: ddpg
---

## The problem

Suppose you train a candidate policy gradient method for 5 seeds and the baseline for 5 seeds. The candidate's mean return is higher, and the plot of mean plus or minus one standard error shows two bands that do not touch. You write that the candidate is better. Would a second set of 5 seeds say the same?

The paper's answer is that it often would not. The authors ran 10 seeds of one TRPO configuration on HalfCheetah-v1, split them into two sets of 5, and plotted the two averages (Figure 5). The curves differ enough that a two-sample t-test over the whole training distribution reports $t = -9.0916$ and $p = 0.0016$, although the algorithm, the hyperparameters, and the code are identical.

This page measures the same effect in a model you can inspect. It draws 10 same-algorithm "final returns" from a normal distribution with mean 3000 and standard deviation 800 (made-up numbers), splits them 5 and 5, and applies the mean plus or minus one standard error rule that the paper's plots use. The bands do not overlap in 20.46% of 10,000 experiments, computed by `scripts/seed_split_sim.py`. Every one of those is a false alarm, because both groups come from the same algorithm.

The paper does not stop at seeds. It also finds large swings from the following choices, each of which a paper can make without saying so:

- The size and activation of the policy and value networks.
- The scale of the rewards.
- The environment that you pick.
- The codebase that supplies the baseline.

For example, Table 1 reports PPO on Hopper-v1 with a $(64, 64)$ policy at $2790 \pm 62$ and with a $(400, 300)$ policy at $61 \pm 33$, after 2M timesteps and with all else left at the defaults.

The authors' conclusion is that judging a new method against a baseline requires more than a table of means. The paper has no numbered sections or equations, so this page cites its headings, tables, and figures. It follows the arXiv v3 text (January 2019).

## Key ideas

### Hyperparameters and network architecture

**Intuition.** A policy gradient method is a recipe with many knobs. If the knob settings that you use for the baseline are not the ones its authors tuned, the baseline looks weak.

**What the paper does** (section "Network Architecture", Tables 1 and 2, Figure 2). Using the OpenAI Baselines code, it runs TRPO, PPO, DDPG, and ACKTR on Hopper-v1 and HalfCheetah-v1 for 2M timesteps with 5 seeds. It varies one setting at a time. The policy and value networks take the hidden sizes $(64, 64)$, $(100, 50, 25)$, and $(400, 300)$, and the activations tanh, ReLU, and Leaky ReLU. The tables report the mean and standard error of the return over the last 100 trajectories.

The paper finds that ReLU or Leaky ReLU usually does best but that the effects are not consistent across algorithms or environments. In a footnote, it says that PPO updates with the $(400, 300)$ network had a KL divergence on average 33.52 times higher than with $(64, 64)$. This suggests that a larger network needs a different clipping range or learning rate, so the architecture is not independent of the other settings.

**Tiny example, computed by `scripts/paper_numbers.py`.** The 33.52 times higher KL is a ratio of two averages from the paper's runs, which this page can't recompute. You can check the size of the Table 1 gap against the seed noise next to it. The two PPO Hopper-v1 entries ($2790 \pm 62$ and $61 \pm 33$) differ by 2729 return units, which is 44.0 times the larger standard error. The seed noise within a cell is small compared with the effect of the architecture.

### Reward scale

**Intuition.** Gradient-based critics respond to the size of their targets. Multiply the reward by a constant and you change the loss surface the critic sees, even though the optimal policy does not change.

**Math** (section "Reward Scale"). The paper rescales the reward as

$$
\hat r = r \hat\sigma
$$

and notes that a scale of $\hat\sigma = 0.1$ is common in the DDPG literature. Figure 3 shows DDPG on HalfCheetah-v1 with $\hat\sigma$ from $10^{-4}$ to $100$, with and without layer normalization. Training fails for scales below $0.01$, and layer normalization changes how the scale acts. The paper also notes that some environments have untuned reward scales, for example HumanoidStandup-v1 in OpenAI Gym, where returns reach millions.

**Tiny example.** Multiplying every reward by $\hat\sigma$ multiplies every return, every Q-value target, and every squared critic error by $\hat\sigma$ or $\hat\sigma^2$. With $\hat\sigma = 0.01$, a return of 3000 becomes 30 and a squared error of $10^4$ becomes $1$. A critic with a fixed learning rate and initialization reacts very differently to those two target sizes. The paper's point is that the choice is a hidden hyperparameter, and its suggested direction is an algorithm that adapts to the scale instead of leaving the choice to the user.

### Random seeds and the number of trials

**Intuition.** The seed sets the network initialization, the exploration noise, and the environment's randomness. If the learning process is sensitive to them, two groups of seeds are two samples from a wide distribution, and their means differ by chance.

**What the paper does** (section "Random Seeds and Trials", Figure 5). It runs 10 seeds of TRPO on HalfCheetah-v1 with one hyperparameter setting, and splits them into two groups of 5. It reports that averaging different seeds of the same configuration can produce learning curves that do not look like draws from one distribution. It also notes two common practices that make this worse: reporting the top $N$ trials out of several, and averaging fewer than 5 trials.

**Tiny example, computed by `scripts/seed_split_sim.py`.** Using the model from the problem section, 10,000 experiments, and two groups of 5 from the same distribution:

| Rule for "the groups differ" | Normal returns | Bimodal returns |
|---|---|---|
| Mean $\pm$ 1 standard error bands do not overlap | 20.46% | 21.08% |
| Welch's t-test, two-sided $p < 0.05$ | 4.23% | 4.34% |
| Welch calls A significantly better ($p < 0.05$, one side), A reports all 5 seeds | 2.07% | 2.08% |
| Same test, but A reports its best 3 of 5 seeds and B reports all 5 | 10.89% | 14.28% |

The bimodal column models seeds that land in a weak basin (mean 1500) or a strong one (mean 4000) with equal chance, plus noise. Both distributions are made up. The band rule fires about four times as often as the nominal 5%, and a proper test fires near 5%. Reporting only the best 3 of 5 seeds raises the rate of a false "A is significantly better" from about 2% to 11% (normal) or 14% (bimodal). The mean gain of the cherry-picked group is 439 in the normal model.

A caveat on the paper's own number: its $p = 0.0016$ in the Figure 5 caption comes from a test across the whole training distribution, and the paper doesn't say how it forms the samples, so this page doesn't claim that a test at the right level would reproduce it. The paper's point stands in the simulation above: with only 5 seeds, the band rule that plots show makes same-algorithm gaps look real.

Exercise 3 asks you to implement the band rule and the Welch rule over many experiments.

### Environment and codebase

**Intuition.** A benchmark task is a proxy. A method that wins on one task can lose on another, and an implementation that differs in details you did not list is a different baseline.

**What the paper does** (sections on environments and codebases, Figures 4 and 6, Tables 1 and 2). On Hopper-v1, HalfCheetah-v1, Swimmer-v1, and Walker2d-v1, the best algorithm changes with the environment. DDPG does best on stable HalfCheetah-v1 and loses its edge on the less stable Hopper-v1. On Swimmer-v1, TRPO does best, and a return of about 130 corresponds to a policy that curls up and flails instead of swimming, which the authors confirm by watching replays. The paper therefore asks you to show the learned behavior and not only the return.

For the codebase experiment, the paper compares three TRPO implementations and three DDPG implementations under the same hyperparameters. In Table 1, the $(400, 300)$ TRPO result on HalfCheetah-v1 is $1791 \pm 224$ for the original code, $738 \pm 240$ for the rllab implementation, and $-155 \pm 188$ for the OpenAI Baselines implementation used in the PPO paper. The authors say details such as the optimizer for the value baseline differ across codebases and are rarely published.

**Tiny example.** Take those three cells as three samples of "TRPO on HalfCheetah-v1 with a $(400, 300)$ policy". Their means span 1946 return units, which is 8.1 times the largest standard error among them (240), computed by `scripts/paper_numbers.py`. A paper that compares against one of these as "the TRPO baseline" has picked a number from a range this wide.

### Confidence intervals from the bootstrap

**Intuition.** You have a handful of final returns and want to know how much their mean would move on a rerun. Resample the returns with replacement many times, take the mean of each resample, and read the spread of those means.

**Math** (section "Confidence Bounds", Table 3). The paper reports the bootstrap mean and 95% bounds from 10,000 bootstrap iterations with the pivotal method. This page and Exercise 1 use the percentile method instead, which is simpler to compute. With $B$ resamples $\bar x^*_1, \dots, \bar x^*_B$ of the $n$ final returns, the interval is the $\alpha/2$ and $1-\alpha/2$ percentiles of the resample means. Take care when you compare with the paper. The standard pivotal interval (a textbook definition, not spelled out in the paper) reflects the percentiles about the sample mean, $(2\bar x - q_{1-\alpha/2},\ 2\bar x - q_{\alpha/2})$, and it differs from the percentile interval for skewed samples.

**Tiny example, computed by `scripts/bootstrap_example.py`.** Take $x = [10, 20, 30, 40, 50]$ and four resamples of indices, $[0,0,1,1,2]$, $[3,4,4,2,1]$, $[0,1,2,3,4]$, and $[4,4,4,3,3]$. The resample means are $18$, $38$, $30$, and $46$. Their 25th and 75th percentiles are $27$ and $40$, which is the central 50% interval.

A percentile interval is not a guarantee. When the script draws 2000 samples of $n$ normal returns (mean 3000, standard deviation 800) and builds a nominal 95% interval from each, the interval covers the true mean this often:

| Seeds $n$ | Coverage | Mean width |
|---|---|---|
| 5 | 83.75% | 1154.1 |
| 10 | 89.60% | 908.7 |
| 30 | 93.95% | 555.2 |

The paper notes that bounds from the bootstrap can be very wide, and that you may then need more trials. The table shows a second effect: with 5 seeds the interval is both wide and still too optimistic.

Exercise 1 asks you to implement this interval.

### Significance tests

**Intuition.** An interval says how much a mean can move. A test asks whether the difference between two algorithms' means is bigger than that movement.

**Math** (section "Significance"). The paper considers a two-sample t-test on the final returns across seeds, a Kolmogorov-Smirnov test, and the bootstrapped percent difference with a 95% interval. It gives one worked case, ACKTR against DDPG on Walker2d-v1: $t = 1.03$, $p = 0.334$, $\mathrm{KS} = 0.40$, $p = 0.697$, and a bootstrapped difference of 44.47% with a 95% interval of $(-80.62\%, 111.72\%)$. The interval includes 0, which agrees with the tests that the gap is not significant.

The paper doesn't print the t-test formula. This page uses Welch's version, which doesn't assume equal variances. With sample variances $s_a^2$ and $s_b^2$ and sizes $n_a$ and $n_b$, let $v_a = s_a^2/n_a$ and $v_b = s_b^2/n_b$. Then

$$
t = \frac{\bar a - \bar b}{\sqrt{v_a + v_b}}, \qquad \mathrm{df} = \frac{(v_a + v_b)^2}{v_a^2/(n_a - 1) + v_b^2/(n_b - 1)}
$$

The paper also warns that corrected tests from supervised learning, such as the corrected resampled t-test, rest on assumptions that don't necessarily hold in RL, and that better tests are an open problem.

**Tiny example, computed by `scripts/welch_example.py`.** Take $a = [1,2,3,4,5]$ and $b = [2,4,6,8,10]$. The means are 3 and 6, and the sample variances are 2.5 and 10, so $v_a + v_b = 0.5 + 2 = 2.5$. Then $t = -3/\sqrt{2.5} = -1.8974$ and $\mathrm{df} = 6.25/1.0625 = 5.8824$, with a two-sided $p = 0.1075$. A pooled-variance test gives the same $t$ but $\mathrm{df} = 8$ and $p = 0.0943$, which is more optimistic because it ignores that $b$ has four times the variance. When the two groups have equal variance and size, Welch's df is $2n - 2 = 8$.

Exercise 2 asks you to implement the statistic and its degrees of freedom.

### Power analysis with a lift

**Intuition.** Before you trust "no significant difference", ask whether your sample could have shown one. If an effect of the size you care about would almost never reach significance with your seeds, the test says nothing.

**What the paper does** (section "Power Analysis"). Take the sample, scale a copy by a uniform lift (the paper's example is 1.25), run many bootstrap simulations, and count the share that are significant. A small share means you need more trials. The paper reports that unstable settings lean toward insignificant results while a stable setting such as TRPO on Hopper-v1 needs no more trials, and defers the numbers to its supplement.

**Tiny example, computed by `scripts/power_example.py`.** Two made-up samples of 5 returns: a stable one with mean 3000 and a coefficient of variation of 0.026, and a noisy one with mean 2800 and a coefficient of variation of 0.670. With a lift of 1.25 and a bootstrap of 2000 pairs:

| Seeds per group | Power, stable sample | Power, noisy sample |
|---|---|---|
| 5 | 1.000 | 0.060 |
| 10 | 1.000 | 0.121 |
| 20 | 1.000 | 0.194 |
| 40 | 1.000 | 0.376 |

For the noisy sample, a 25% improvement is detected about 6% of the time with 5 seeds, which is close to the false-alarm rate. Even 40 seeds detect it fewer than 4 times in 10. Bootstrapping from 5 returns also limits what the table can say about larger groups, since it only reuses those 5 values.

Exercise 4 asks you to implement the power calculation.

### What the paper recommends

The conclusion (sections "What recommendations can we draw from our experiments?" and "How can we ensure that deep RL matters?") lists the following steps:

1. Search for hyperparameters until a baseline matches its originally reported performance, and match the original codebase's results when it exists.
1. Run many trials with different seeds, and average them, unless seed choice is part of the algorithm.
1. Test significance before you claim that a higher mean is a better algorithm.
1. Use bootstrapping and power analysis to decide how many trials you need.
1. Report all hyperparameters, implementation details, the experimental setup, and the evaluation methods for baselines and for your own method.
1. Show results on a range of environments and say in what setting the method is useful.

The paper also points to hyperparameter-agnostic algorithms as future work.

<!-- exercises -->

## Pitfalls

- **Treating the paper's numbers as a benchmark.** The runs use Gym v1 MuJoCo environments, 5 seeds, and 2M timesteps, and the codebases of 2017. Current implementations and environments differ, so use the paper for the method of checking and not as a table of expected returns.
- **Comparing means with overlapping error bars.** With 5 seeds, mean plus or minus one standard error is a narrow band. The page's simulation shows non-overlapping bands for 20% of same-algorithm splits.
- **Calling a t-test over a learning curve a test over seeds.** The points along one learning curve are correlated, so they are not independent samples of the algorithm. The unit of replication is the seed. The paper's Significance section sorts the final returns across trials, which gives one sample per seed.
- **Reading "not significant" as "equal".** The ACKTR against DDPG case has a wide interval, from -80.62% to 111.72%. That says the data cannot tell the two apart, and not that they perform the same. The power table shows how often a real 25% gain goes undetected with 5 seeds.
- **Selecting seeds.** Reporting the best 3 of 5 seeds is a reporting choice that the page's simulation shows raising the false-win rate from about 2% to 11% or 14%. Fix the seed list before you run and report all of it.
- **Trusting a percentile interval at 5 seeds.** The measured coverage is 83.75%, not 95%. Use more seeds, or say that the interval is approximate.
- **Treating the variance as noise that a better implementation removes.** The paper separates intrinsic sources (seeds, environment) from extrinsic ones (hyperparameters, codebase). Only the extrinsic ones are in your control.
- **Forgetting that Welch's df is not an integer.** The statistic's null distribution has $\mathrm{df}$ degrees of freedom, and a p-value from a table needs the exact value or a round-down. The exercises compare the statistic and the degrees of freedom, not a p-value.

## Further reading

- The paper's code, linked in its first footnote: [https://git.io/vFHnf](https://git.io/vFHnf). The baselines it benchmarks are in [OpenAI Baselines](https://github.com/openai/baselines).
- Islam et al., [Reproducibility of benchmarked deep reinforcement learning tasks for continuous control](https://arxiv.org/abs/1708.04133), the workshop paper that this one extends.
- Colas, Sigaud, and Oudeyer, [How many random seeds? Statistical power analysis in deep reinforcement learning experiments](https://arxiv.org/abs/1806.08295), a follow-up on the seed-count question.
- Agarwal et al., [Deep reinforcement learning at the edge of the statistical precipice](https://arxiv.org/abs/2108.13264), which proposes the interquartile mean and performance profiles, with the `rliable` library.
