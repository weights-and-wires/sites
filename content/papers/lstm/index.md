---
id: "lstm"
title: "Long Short-Term Memory"
authors: "Hochreiter et al."
year: 1997
venue: "Neural Computation 9(8)"
arxiv: null
url: "https://doi.org/10.1162/neco.1997.9.8.1735"
track: foundations
tier: extended
est_hours: 3
status: full
summary: "Error signals in a conventional recurrent network shrink exponentially as they travel back in time, so the network can't learn dependencies that span more than a few steps. The paper builds a memory cell around a linear self-connection of weight 1, the constant error carousel, so the error through that cell neither grows nor shrinks. A multiplicative input gate controls what enters the cell and a multiplicative output gate controls what leaves it, and with these gates the network learns tasks with time lags above 1000 steps."
related_course: "ml: sequence models"
prereqs: []
background: [backprop-through-time, vanishing-gradients]
course_links:
  - title: "The LSTM, a belt with three gates"
    url: "https://ml.ifkash.dev/lessons/054-the-lstm"
  - title: "Backpropagation through the LSTM, and the GRU"
    url: "https://ml.ifkash.dev/lessons/055-lstm-backward-and-the-gru"
self_check:
  - q: 'A recurrent unit with a logistic sigmoid activation has a self-connection of weight $w = 3$. By what factor does Equation 3.2 scale an error that travels back $q = 10$ steps, at most?'
    a: 'Each step scales the error by $f''(\mathit{net}) \, w \le 0.25 \cdot 3 = 0.75$, so ten steps scale it by at most $0.75^{10} \approx 0.056$. After 50 steps the factor is about $5.7 \times 10^{-7}$. The unit has no gate or other mechanism that could keep the error alive.'
    see: vanishing-gradients
  - q: 'Why does full backpropagation through time need memory that grows with the sequence length, and which property of LSTM does Section 4.6 claim in contrast?'
    a: 'BPTT has to store the activations of every time step so that it can replay them backward, so storage grows with the length $T$. Section 4.6 says LSTM is local in time: its update needs only the stored derivatives $\partial s_{c_j} / \partial w$, so storage does not depend on the sequence length. The price is that the algorithm truncates the gradient (Section 4.5).'
    see: backprop-through-time
  - q: 'The constant error carousel needs $f_j''(\mathit{net}_j)\,w_{jj} = 1$. What does that force on the activation function $f_j$ and on $w_{jj}$ in the experiments?'
    a: 'Section 3.2.2 integrates the condition and finds that $f_j$ must be linear. The experiments use the identity function with $w_{jj} = 1.0$, so $y_j(t+1) = y_j(t)$ and the error passes back unchanged.'
    see: vanishing-gradients
---

## The problem

Before this paper, the standard ways to train a recurrent network were backpropagation through time (BPTT) and real-time recurrent learning (RTRL). The introduction states the trouble in one sentence: the error signal that flows backward in time either blows up or vanishes, and it does so exponentially in the size of the weights. Blow-ups make the weights oscillate. Vanishing makes learning to bridge a long lag take a prohibitive amount of time, or fail outright.

Section 3.1 derives the effect for a fully connected net. An error at unit $u$ at time $t$ travels back $q$ steps to unit $v$. The factor that scales it is (Equation 3.1, unrolled in Equation 3.2):

$$
\frac{\partial \vartheta_v(t-q)}{\partial \vartheta_u(t)} = \sum_{l_1=1}^{n} \cdots \sum_{l_{q-1}=1}^{n} \prod_{m=1}^{q} f'_{l_m}\!\left(\mathit{net}_{l_m}(t-m)\right) w_{l_m l_{m-1}}, \qquad l_q = v,\; l_0 = u
$$

Here $\vartheta$ is the backpropagated error signal, $f'$ is the derivative of an activation function, and $w_{ij}$ is the weight on the connection from unit $j$ to unit $i$. Every term of the sum is a product of $q$ factors $f'(\mathit{net})\,w$. If each factor has magnitude above 1, the error grows exponentially with $q$. If each has magnitude below 1, it shrinks exponentially (Section 3.1.3).

A logistic sigmoid has $f' \le 0.25$. The paper shows that the factor is below 1 whenever $|w| < 4$, and that a larger initial weight or a larger learning rate doesn't help (Section 3.1.3). Section 3.1.5 gives a bound. With $n$ units and every $|w_{ij}| \le w_{\max} < 4/n$, set $\tau = n\,w_{\max}/4 < 1$. Then the backflow over $q$ steps is at most $n\,\tau^q$.

**Concrete failure, computed by `scripts/error_decay.py`.** Take a single sigmoid unit with a self-connection of weight $w = 3$. Each step scales the error by at most $0.25 \cdot 3 = 0.75$, and the table shows the product over $q$ steps:

| Lag $q$ | $0.75^q$ | $0.25^q$ (weight $w = 1$) |
|---|---|---|
| 10 | $5.63 \times 10^{-2}$ | $9.54 \times 10^{-7}$ |
| 50 | $5.66 \times 10^{-7}$ | $7.89 \times 10^{-31}$ |
| 100 | $3.21 \times 10^{-13}$ | $6.22 \times 10^{-61}$ |

For the bound, take $n = 2$ and $w_{\max} = 1$. Then $\tau = 0.5$, and the bound $n\,\tau^q$ is $1.95 \times 10^{-3}$ at $q = 10$ and $1.78 \times 10^{-15}$ at $q = 50$.

The paper's experiments confirm the practical effect. Experiment 2c has minimal time lags of 1000 steps. The outline in Section 5 says that BPTT and RTRL already fail with 10-step minimal lags on this family of tasks, and LSTM solves it.

## Key ideas

This page uses the paper's notation: $y_u(t)$ is the activation of unit $u$, $\mathit{net}_u(t) = \sum_v w_{uv}\,y_v(t-1)$, $c_j$ is the $j$th memory cell, $s_{c_j}$ is its internal state, and $\mathit{in}_j$ and $\mathit{out}_j$ are its gates. Every number comes from a script in `scripts/`. Run any script from the paper folder to reproduce the output.

The paper numbers only some equations. Section 3 numbers Equations 3.1 and 3.2, the appendix numbers Equations A.1 to A.26, and the cell definition in Section 4.1 has no number. This page cites the unnumbered formulas by section and gives their appendix twin where one exists.

### The constant error carousel

**Intuition.** The error vanishes because each step multiplies it by $f'(\mathit{net})\,w$. To keep the error alive, make that factor exactly 1. A unit that copies its own value to the next step does it.

**Math.** Consider one unit $j$ with a single connection to itself. Section 3.2.1 gives the local error backflow as $\vartheta_j(t) = f'_j(\mathit{net}_j(t))\,\vartheta_j(t+1)\,w_{jj}$. Constant error flow requires

$$
f'_j(\mathit{net}_j(t))\,w_{jj} = 1.0
$$

Section 3.2.2 integrates this differential equation and finds $f_j(\mathit{net}_j) = \mathit{net}_j / w_{jj}$ for arbitrary $\mathit{net}_j$. The function $f_j$ must be linear, and the activation must stay constant. The experiments use $f_j(x) = x$ and $w_{jj} = 1.0$, so $y_j(t+1) = f_j(w_{jj}\,y_j(t)) = y_j(t)$. The paper calls this the constant error carousel (CEC), and says it will be LSTM's central feature.

**Tiny example, computed by `scripts/error_decay.py`.** With $f' = 1$ and $w = 1$, the factor is $1 \cdot 1 = 1$, and after $q = 1000$ steps the error is still exactly 1. The script also builds a bank of 16 CECs and measures the spectral norm of $\partial s_t / \partial s_0$. It is 1.0 at lags 1, 10, 25, 50, and 100.

For contrast, the script measures a conventional network, $h_t = \tanh(W h_{t-1} + U x_t)$, with 16 units, $W$ drawn with entries of variance $1/16$, and 100 random inputs. The spectral radius of $W$ is 1.21, yet the norm of $\partial h_t / \partial h_0$ falls as follows. The units saturate, and a saturated tanh unit has a derivative near 0.

| Lag $t$ | $\lVert \partial h_t / \partial h_0 \rVert_2$ |
|---|---|
| 1 | $1.56$ |
| 10 | $1.32 \times 10^{-2}$ |
| 25 | $3.91 \times 10^{-8}$ |
| 50 | $1.31 \times 10^{-16}$ |
| 100 | $1.98 \times 10^{-30}$ |

This network uses tanh, where the paper uses logistic sigmoids, so the numbers illustrate the effect and don't reproduce a figure from the paper. Exercise 1 asks you to compute these Jacobians.

### Input and output gates

**Intuition.** A bare CEC has two problems, and Section 3.2.2 names both. The paper calls them the input weight conflict and the output weight conflict.

- **Input weight conflict.** The same incoming weight must both write the useful input into the unit and ignore irrelevant later inputs. It receives conflicting update signals.
- **Output weight conflict.** The same outgoing weight must both read the stored value at the right time and keep it from disturbing other units at other times.

The fix is context-sensitive control. An input gate decides when to write, and an output gate decides when to read.

**Math.** Section 4.1 wraps the CEC into a memory cell $c_j$ with two multiplicative gate units. A gate is a logistic sigmoid of a weighted sum of the previous activations (Appendix A.1.1, Equations A.5 and A.6):

$$
y^{\mathit{in}_j}(t) = f_{\mathit{in}_j}\!\left(\mathit{net}_{\mathit{in}_j}(t)\right), \qquad y^{\mathit{out}_j}(t) = f_{\mathit{out}_j}\!\left(\mathit{net}_{\mathit{out}_j}(t)\right)
$$

Gate activations lie in $(0, 1)$. The paper writes the gate activations as $y_{\mathit{in}_j}$ and $y_{\mathit{out}_j}$, with the gate name as a subscript. This page writes the gate name as a superscript to keep it apart from the time and unit subscripts.

**Tiny example, computed by `scripts/cell_example.py`.** An input gate with bias $2$ has $y^{\mathit{in}} = f(2) = 0.8808$ and passes 88% of its input. An output gate with bias $-1$ has $y^{\mathit{out}} = f(-1) = 0.2689$ and passes 27%. Section 4.7 uses this negative output-gate bias on purpose, for the reason given under "State drift and gate bias" below.

### The memory cell

**Intuition.** The cell stores a number $s$ and only ever adds to it. The input gate scales what is added, and the output gate scales what is shown to the rest of the network.

**Math.** The internal state starts at $s_{c_j}(0) = 0$ and then updates by addition (Section 4.1, repeated in Appendix A.1.1, Equation A.7):

$$
s_{c_j}(t) = s_{c_j}(t-1) + y^{\mathit{in}_j}(t)\; g\!\left(\mathit{net}_{c_j}(t)\right)
$$

$$
y^{c_j}(t) = y^{\mathit{out}_j}(t)\; h\!\left(s_{c_j}(t)\right)
$$

The weight on $s_{c_j}(t-1)$ is 1, which is the CEC. The functions are the logistic sigmoid for the gates (Equation A.1, range $[0, 1]$), $h(x) = 2/(1 + e^{-x}) - 1$ with range $[-1, 1]$ (Equation A.2), and $g(x) = 4/(1 + e^{-x}) - 2$ with range $[-2, 2]$ (Equation A.3). Both $h$ and $g$ are rescaled logistic sigmoids. The experiments in Section 5 use them, and the paper doesn't use $\tanh$.

Section 4.4 groups $S$ cells that share one input gate and one output gate into a memory cell block of size $S$. A block of size 1 is a single cell. The exercises use blocks of size 1.

**Tiny example, computed by `scripts/cell_example.py`.** One cell with scalar input $x_t$, cell-input weight 1 on $x_t$, input-gate bias 2, output-gate bias $-1$, and all recurrent weights 0. The inputs are $x = (1, 2, -1, 0, 0)$.

| $t$ | $x_t$ | $y^{\mathit{in}}$ | $g(\mathit{net}_c)$ | $s$ | $y^{\mathit{out}}$ | $h(s)$ | $y^c$ |
|---|---|---|---|---|---|---|---|
| 1 | 1.0 | 0.8808 | 0.9242 | 0.8141 | 0.2689 | 0.3859 | 0.1038 |
| 2 | 2.0 | 0.8808 | 1.5232 | 2.1557 | 0.2689 | 0.7924 | 0.2131 |
| 3 | -1.0 | 0.8808 | -0.9242 | 1.3416 | 0.2689 | 0.5855 | 0.1575 |
| 4 | 0.0 | 0.8808 | 0.0000 | 1.3416 | 0.2689 | 0.5855 | 0.1575 |
| 5 | 0.0 | 0.8808 | 0.0000 | 1.3416 | 0.2689 | 0.5855 | 0.1575 |

Check step 1 by hand: $g(1) = 4 f(1) - 2 = 4 \cdot 0.7311 - 2 = 0.9242$, and $s = 0 + 0.8808 \cdot 0.9242 = 0.8141$. At steps 4 and 5, $g(0) = 0$ adds nothing, and the cell holds its state with no decay. In a conventional unit, the same zero input would leave a value that gets squashed on every step.

The paper's index convention is $\mathit{net}_u(t) = \sum_v w_{uv}\,y_v(t-1)$, so every unit reads the activations of the previous step. Section 5 notes a slight deviation: at each time step the experiments first set the input units, then compute the hidden units, then compute the outputs. This page and Exercise 2 use that convention, which is the same as feeding $z_t = [x_t, y_{t-1}]$.

Exercise 2 asks you to implement this cell and run it over a sequence.

### Truncated backpropagation

**Intuition.** The cell is a CEC only if the error that leaks out of the cell is cut off. If an error could go through the gates and the cell input back to earlier cell outputs, it would travel the old, decaying paths. The paper cuts those paths on purpose. The error is kept only inside the cell and on the output connections.

**Math.** Section 4.5 states the rule. Errors arriving at a cell's net inputs, which include $\mathit{net}_{c_j}$, $\mathit{net}_{\mathit{in}_j}$, and $\mathit{net}_{\mathit{out}_j}$, are not propagated further back in time, though they still change the incoming weights. Only inside a memory cell is the error propagated back through earlier internal states $s_{c_j}$. Within the cell, the factor from $s_{c_j}(t)$ to $s_{c_j}(t-1)$ is 1, so once an error arrives at the cell output it is scaled by the output gate and $h'$, flows back through the CEC without any scaling, and is scaled once more by the input gate and $g'$ when it leaves (Section 4.5). The Appendix gives the formulas as approximate derivatives, marked with $\approx_{\mathrm{tr}}$, in A.1.2 and the backward pass in A.1.3 (Equations A.14 to A.26).

The paper is explicit that this is a design choice and not an approximation to be removed. Section 6.1 says the constant error flow through CECs can be shown only for truncated LSTM, and that the authors found no significant difference in a few experiments with the full gradient, because outside the CECs the error vanishes quickly anyway.

**Tiny example, computed by `scripts/error_decay.py`.** Under truncation, $\partial s_t / \partial s_{t-1} = 1$ exactly, because the input gate and $g$ are treated as constants with respect to the earlier state. After 100 steps the product is 1, as in the CEC example above.

The cost is small. Section 4.6 says the update complexity per time step and weight is $O(1)$, which is $O(W)$ per step for $W$ weights, the same as BPTT. Unlike full BPTT, LSTM is local in time: it doesn't store the activations of earlier steps.

### State drift and gate bias

**Intuition.** Because the state is a sum, a cell fed mostly positive input keeps growing. A large $|s|$ saturates $h$, and the derivative $h'(s)$ goes to 0. That kills the error through the output of the cell.

**Math.** Section 4.8 describes this internal state drift. The remedy for the start of learning is to bias the input gate toward zero. Section 4.7 treats a related failure, the abuse problem. Early in training, the network can reduce error without storing information over time, so it tends to use cells as bias units. One of the two fixes the paper offers gives each output gate a negative initial bias, which pushes the cell activations toward zero at first and frees the cells for later use.

The scaled logistic $h$ has derivative $h'(x) = 2 f(x)\,(1 - f(x))$.

**Tiny example, computed by `scripts/drift_example.py`.**

| State $s$ | $h'(s)$ |
|---|---|
| 0 | $0.5$ |
| 2 | $0.21$ |
| 5 | $1.33 \times 10^{-2}$ |
| 10 | $9.08 \times 10^{-5}$ |

A cell that receives a constant input of $+0.5$ per step reaches $s = 5$ after 10 steps. With an input gate bias of $0$, the gate is $0.5$ and $s$ reaches 25 after 100 steps, where $h'(s) = 2.8 \times 10^{-11}$. With a bias of $-3$, the gate is $0.0474$ and $s$ is $2.37$ after 100 steps, where $h'(s) = 0.156$. With a bias of $-6$, the gate is $0.00247$, $s$ is $0.124$, and $h'(s) = 0.498$. Section 5.4.3 uses biases of $-3.0$ and $-6.0$ for the first and second input gates on exactly this account.

### The adding problem

**Intuition.** A test of long-lag memory should have no short-lag shortcuts, and should need real numbers stored precisely. The adding problem does both.

**Math.** Section 5.4.1 defines it. Each sequence element is a pair. The first component is a real value drawn from $[-1, 1]$. The second is a marker in $\{1.0, 0.0, -1.0\}$. Sequences have random length between $T$ and $T + T/10$. Exactly two pairs are marked with $1.0$. One is among the first 10 pairs, with value $X_1$. The other is among the first $T/2 - 1$ still unmarked pairs, with value $X_2$. The first and the last pair have marker $-1.0$, and the rest have 0. The net outputs a single value at the sequence end, with target

$$
0.5 + \frac{X_1 + X_2}{4}
$$

which scales the sum into $[0, 1]$. A sequence is processed correctly if the absolute error at the end is below 0.04. The minimal time lag is $T/2$, because the second marked pair is always in the first half.

An LSTM can solve this with the machinery above. Let the input gate open when the marker is 1, let $g$ pass the value, and let the CEC add it to $s$. At the end, $s = X_1 + X_2$. The network has to learn this behavior of the gates. Exercise 3 implements the idealized version with hard 0/1 gates, as a statement of what the learned solution computes.

**Tiny example, computed by `scripts/adding_example.py`.** With seed 0 and $T = 100$, one sequence has length 109 and marked positions 6 and 9. Its values are $X_1 = 0.4590$ and $X_2 = 0.6317$, so $X_1 + X_2 = 1.0907$ and the target is $0.7727$. The idealized CEC adder returns the same $0.7727$. Over 2,560 random sequences with $T = 100$, the constant answer 0.5 has a mean absolute error of $0.1626$ and is wrong on 2,180 sequences. The idealized adder is wrong on 0.

The paper's own results for the trained LSTM are in Table 7. For $T = 100$, it makes 1 wrong prediction out of 2,560 test sequences, after 74,000 training sequences. For $T = 1000$ (minimal lag 500), it makes 1 wrong prediction, after 853,000 sequences. The numbers are means over 10 trials, and the architecture has 93 weights.

### What the 1997 cell leaves out

Descriptions of LSTM often list a forget gate, and so does this site's paper list. The 1997 paper has no forget gate. Its cell has the input gate and the output gate only, and the state weight on $s_{c_j}(t-1)$ is fixed at 1. Gers, Schmidhuber, and Cummins added the forget gate in 2000, which multiplies the old state by a learned value in $(0, 1)$:

$$
s(t) = y^{\varphi}(t)\,s(t-1) + y^{\mathit{in}}(t)\,g\!\left(\mathit{net}_c(t)\right)
$$

(This is the 2000 form, written with the paper's symbols and with $\varphi$ for the forget gate. It isn't in the 1997 paper.) The 1997 state can only accumulate, which is why Section 4.8 treats drift as a danger, and why the experiments reset the net activations after each input sequence (Section 5). The cell in current libraries has all three gates, and it also uses $\tanh$ in place of $g$ and $h$ and trains with the full gradient. The paper's analysis of error flow still explains why the additive state works.

<!-- exercises -->

## Pitfalls

- **Looking for the forget gate.** It isn't in this paper. The cell in a modern library is the 2000 variant. If you implement the 1997 cell and compare it with `torch.nn.LSTM`, the outputs differ even with matching weights.
- **Treating the state update as a convex combination.** The 1997 update adds to $s$ with weight 1. A cell that sums for a long time can drift (Section 4.8), so initialize the input-gate biases negative, as in Section 5.4.3.
- **Multiplying the old state by something near 1.** A decay such as $0.9\,s(t-1)$ looks like a harmless change, but it is the forget gate in disguise. The factor $0.9^{100} = 2.7 \times 10^{-5}$ brings the vanishing error back for lags near 100. Exercise 2 tests for exact addition for this reason.
- **Feeding the output gate the wrong state.** $y^c = y^{\mathit{out}}\,h(s)$ uses the updated state $s_{c_j}(t)$, not $s_{c_j}(t-1)$.
- **Mixing up the function ranges.** $g$ has range $(-2, 2)$ and $h$ has range $(-1, 1)$, and both are rescaled logistic sigmoids. Swapping in $\tanh$ for both works in practice but changes the numbers in the paper's tables.
- **Expecting an autograd framework to reproduce the paper's gradient.** Autograd computes the full gradient, with error flowing through the gates and the cell input to earlier steps. The paper's algorithm truncates those paths (Section 4.5), so the weight updates differ. Section 6.1 reports no significant difference in a few experiments, so the full gradient is a reasonable substitute in practice, but it isn't the same algorithm.
- **Reading "constant error flow" as "no vanishing anywhere".** The guarantee holds inside the CEC. Outside it, the error still vanishes quickly (Section 6.1).
- **Expecting LSTM to count steps exactly.** Section 6.1 says all gradient-based approaches have a practical inability to count discrete time steps precisely. A task that depends on whether a signal came 99 or 100 steps ago needs more than the cell.
- **Expecting LSTM to solve every long-lag task.** Section 6.1 lists the strongly delayed XOR problem, which the truncated algorithm doesn't solve easily, and the 500-step parity problem, which random weight guessing solves faster than LSTM.
- **Marking the first pair in the adding problem.** One of the first 10 pairs can be index 0, which also carries marker $-1.0$. The paper's rule is to set $X_1$ to zero in that case. A generator that skips the rule gives targets that don't match the marked values.
- **Counting the final marker.** The first and last pairs carry marker $-1.0$, and the gate opens only for $+1.0$. An adder that opens the gate for any nonzero marker adds two random values to the sum.

## Further reading

- [Learning to Forget: Continual Prediction with LSTM](https://doi.org/10.1162/089976600300015015), by Gers, Schmidhuber, and Cummins, adds the forget gate.
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/), by Christopher Olah, is an illustrated walkthrough of the modern three-gate cell.
- [LSTM: A Search Space Odyssey](https://arxiv.org/abs/1503.04069), by Greff et al., compares LSTM variants and finds that the forget gate and the output activation function matter most.
- [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850), by Alex Graves, shows LSTM at scale on text and handwriting.
- [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/), by Andrej Karpathy, is a blog post with character-level examples.
