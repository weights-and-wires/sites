---
id: "resnet"
title: "Deep Residual Learning for Image Recognition"
authors: "He et al."
year: 2015
venue: "CVPR 2016"
arxiv: "1512.03385"
url: "https://arxiv.org/abs/1512.03385"
track: vision
tier: core
est_hours: 3
status: full
summary: "Deeper plain networks reach higher training error than shallower ones, which the paper calls the degradation problem. A residual block fixes this by learning the difference between the desired output and its input and adding the input back through an identity shortcut, so a block that should do nothing only has to push its weights toward zero. With this change, networks of 50 to 152 layers train well and win ILSVRC 2015, and bottleneck blocks keep the cost of the deepest ones low."
related_course: "ml: deep CNNs"
prereqs: []
background: [convolution, vanishing-gradients, backprop]
course_links:
  - title: "Residual networks: the shortcut, the gradient, and the bottleneck"
    url: "https://ml.ifkash.dev/lessons/048-residual-networks"
self_check:
  - q: 'A 3×3 convolution maps 64 channels to 64 channels and has no bias. How many weights does it have, and how many does a 1×1 convolution from 256 channels to 64 channels have?'
    a: 'The 3×3 convolution has $3 \cdot 3 \cdot 64 \cdot 64 = 36{,}864$ weights. The 1×1 convolution has $1 \cdot 1 \cdot 256 \cdot 64 = 16{,}384$. The kernel size enters squared, which is why a 1×1 layer is cheap and why the bottleneck block puts the 3×3 layer at the narrow width.'
    see: convolution
  - q: 'A chain of 20 scalar layers each multiplies the gradient by 0.3. What is the gradient of the output with respect to the input? What changes if each pair of layers is wrapped in an identity shortcut?'
    a: 'The plain chain gives $0.3^{20} \approx 3.5 \times 10^{-11}$, so the early layers get almost no signal. With a shortcut around each pair, each block contributes $1 + 0.3^2 = 1.09$, and ten blocks give $1.09^{10} \approx 2.37$. The "1" is the shortcut path, which no weight scales.'
    see: vanishing-gradients
  - q: 'For $y = \sigma(\mathcal{F}(x) + x)$ with a ReLU $\sigma$, and an upstream gradient $g = \partial L / \partial y$, write the gradient with respect to $x$.'
    a: 'Let $z = \mathcal{F}(x) + x$ and $g_z = g \odot [z > 0]$. Then $\partial L / \partial x = (\partial \mathcal{F} / \partial x)^{\top} g_z + g_z$. The first term runs through the weights of $\mathcal{F}$. The second term runs through the shortcut and passes $g_z$ unchanged.'
    see: backprop
---

## The problem

By 2015, depth was the main lever for image recognition. Leading ImageNet models had 16 to 30 layers (Section 1). The obvious next step was to stack more layers. It failed in a way that nobody expected.

Section 1 states the failure. Past a certain depth, accuracy saturates and then degrades. Adding layers to a suitably deep model raises the **training** error, so the cause isn't overfitting. Figure 1 shows it on CIFAR-10: a 56-layer plain network has higher training error and higher test error than a 20-layer one. Table 2 shows it on ImageNet, with 10-crop top-1 validation error:

| Depth | Plain | ResNet |
|---|---|---|
| 18 layers | 27.94% | 27.88% |
| 34 layers | 28.54% | 25.03% |

The 34-layer plain network is worse than the 18-layer one. The 34-layer ResNet is better.

The paper's argument that this is an optimization failure is a construction. Take a shallow network that works. Add layers that compute the identity function. The deeper network now computes the same function, so a solver that can search the deeper network's weights should find training error no higher than the shallow one's. Experiments show that the solvers don't find it (Section 1).

Two other explanations are easy to reach for, and the paper rules them out or sets them aside:

- **Overfitting.** The training error goes up, not just the test error (Figure 1).
- **Vanishing gradients.** The plain networks use batch normalization, which keeps the forward signals at non-zero variance. The authors also verify that the backward gradients have healthy norms, and they conjecture that deep plain networks have exponentially low convergence rates (Section 4.1). The reason isn't established in this paper.

## Key ideas

The sections that follow use the paper's notation: an input $x$, a block output $y$, a residual function $\mathcal{F}(x, \{W_i\})$ with weights $W_i$, a projection $W_s$, and a ReLU $\sigma$. The paper omits biases in its formulas, and so does this page. Every number comes from a script in `scripts/`. Run any script from the repository root to reproduce the output.

### Learn the residual, not the mapping

**Intuition.** Suppose a few stacked layers must produce a desired mapping $\mathcal{H}(x)$. Instead of asking the layers to fit $\mathcal{H}(x)$ directly, ask them to fit the difference $\mathcal{F}(x) := \mathcal{H}(x) - x$ and add $x$ back (Section 3.1). If the best thing a block can do is nothing, the layers only need to output zero. That is easier than making a stack of nonlinear layers reproduce the identity.

**Math.** A building block is (Section 3.2, Equation 1)

$$
y = \mathcal{F}(x, \{W_i\}) + x
$$

For the two-layer block in Figure 2, $\mathcal{F} = W_2\,\sigma(W_1 x)$. The paper puts the second nonlinearity after the addition, so the block's output is $\sigma(y)$. The shortcut adds no parameters and almost no computation, which lets the paper compare plain and residual networks with the same depth, width, and cost (Section 3.2). The same formulas apply to convolutional layers. The addition is then element-wise on two feature maps, channel by channel.

The paper doesn't claim that identity mappings are optimal. It claims that the optimal function is more likely to be close to an identity than to a zero mapping, so the solver should find small perturbations around the identity more easily than a whole new function (Section 3.1). Figure 7 supports this: the learned residual functions have small responses, and deeper ResNets have smaller ones.

**Tiny example, computed by `scripts/residual_example.py`.** Take $W_1 = I$, $W_2 = \begin{bmatrix} 1 & -1 \\ 0.5 & 0.5 \end{bmatrix}$, and $x = [2, -1]$:

| Step | Result |
|---|---|
| $\sigma(W_1 x)$ | `[2, 0]` |
| $\mathcal{F}(x) = W_2\,\sigma(W_1 x)$ | `[2, 1]` |
| $\mathcal{F}(x) + x$ | `[4, 0]` |
| $y = \sigma(\mathcal{F}(x) + x)$ | `[4, 0]` |

Now set $W_1 = W_2 = 0$ and $x = [0.5, 2, 7]$. The residual block returns `[0.5, 2, 7]`, because $\mathcal{F} = 0$. A plain layer $\sigma(Wx)$ with $W = 0$ returns `[0, 0, 0]`. It has to learn $W = I$ to match. This is the constructed identity solution from the previous section, and the residual form reaches it at zero weights. The inputs here are non-negative, so the ReLU after the addition leaves them unchanged.

Exercise 1 asks you to implement the block, with and without a projection.

### Identity shortcuts and projections

**Intuition.** The shortcut can only add $x$ to $\mathcal{F}(x)$ when the two have the same shape. When a stage changes the number of channels, the shortcut needs a way to match.

**Math.** The paper gives a projection with a learned matrix $W_s$ (Section 3.2, Equation 2):

$$
y = \mathcal{F}(x, \{W_i\}) + W_s x
$$

In the convolutional networks, $W_s$ is a 1×1 convolution (Section 3.3). The paper says an identity is enough to address degradation, and uses $W_s$ only to match dimensions (Section 3.2). For the dimension-increasing shortcuts in ImageNet networks, it compares three options (Section 4.1, Table 3):

| Option | Shortcuts | ResNet-34 top-1 validation error |
|---|---|---|
| A | Identity, with zero-padding for new channels | 25.03% |
| B | Projection only where dimensions change | 24.52% |
| C | Projection everywhere | 24.19% |

All three are far better than the plain 34-layer network at 28.54%. B beats A, and the authors attribute that to the zero-padded channels having no residual learning. C beats B slightly, and they attribute that to the extra parameters. They conclude that projections aren't essential, and they use B for ResNet-50/101/152.

**Tiny example, computed by `scripts/residual_example.py`.** With $x = [1, 2]$, a $\mathcal{F}$ that maps 2 dimensions to 3 gives $\mathcal{F}(x) = [12, 12, 12]$. The projection $W_s = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}$ gives $W_s x = [1, 2, 3]$, so $y = [13, 14, 15]$.

### Why the gradient survives

**Intuition.** On the backward pass, the shortcut is a wire. The gradient at a block's output reaches its input through two paths. One goes through $\mathcal{F}$ and gets multiplied by weights. The other goes straight through the addition and isn't multiplied by anything.

**Math.** Differentiate Equation 1 with respect to $x$. Let $I$ be the identity matrix:

$$
\frac{\partial y}{\partial x} = I + \frac{\partial \mathcal{F}}{\partial x}
$$

Over a stack of blocks, the $I$ in each factor keeps a path that doesn't decay. The paper doesn't derive this formula. It motivates residual learning by the degradation experiments, and Section 4.1 argues that the plain networks' problem isn't vanishing gradients. The gradient-flow analysis of the identity path came later, in the same authors' follow-up paper listed under Further reading. Treat this section as the mechanism behind the paper's design, not as the paper's own claim.

**Tiny example, computed by `scripts/gradient_example.py`.** Start with a scalar chain of 20 weight layers, each with weight 0.3, and no ReLU (every value is positive, so the ReLU does nothing):

| Network | $\partial y / \partial x$ |
|---|---|
| Plain, 20 layers | $0.3^{20} = 3.487 \times 10^{-11}$ |
| Residual, 10 blocks of 2 layers | $(1 + 0.3^2)^{10} = 2.3674$ |

Now a 20-matrix ReLU network of width 32, with weights drawn from $N(0, (0.5^2 \cdot 2 / 32))$ and the loss $L = \tfrac12 \lVert h_n \rVert^2$. The script runs the plain version with 20 layers and the residual version with 10 blocks of 2 layers. With seed 0:

| Network | $\lVert \partial L / \partial h_0 \rVert$ | $\lVert \partial L / \partial h_n \rVert$ | Ratio |
|---|---|---|---|
| Plain, 20 layers | $1.04 \times 10^{-11}$ | $4.30 \times 10^{-6}$ | $2.4 \times 10^{-6}$ |
| Residual, 10 blocks | $9.47$ | $5.76$ | $1.64$ |

Across seeds 0 to 3, the plain ratio stays between $2 \times 10^{-7}$ and $2.4 \times 10^{-6}$, and the residual ratio stays between 1.38 and 2.45. This toy has no batch normalization and a deliberately small weight scale, so it shows the mechanism and not the paper's experiment. See Pitfalls.

Exercise 2 asks you to write the backward pass for one block. Exercise 3 asks you to measure the norms across a whole network.

### Bottleneck blocks

**Intuition.** A block with two 3×3 convolutions at width 256 is expensive. A cheaper design shrinks the width with a 1×1 convolution, runs the 3×3 convolution at the small width, and restores the width with another 1×1 convolution.

**Math.** For ResNet-50, 101, and 152, the paper replaces each two-layer block with a three-layer one: 1×1, 3×3, and 1×1 convolutions, where the 1×1 layers reduce and then restore the dimensions, leaving the 3×3 layer as a bottleneck with smaller input and output dimensions (Section 4.1, Figure 5, right). Table 1 lists the layouts. For example, the first stage (`conv2_x`) uses 1×1 with 64 filters, 3×3 with 64, and 1×1 with 256. The paper says both designs in Figure 5 have similar time complexity.

The paper also says identity shortcuts matter more here. If the shortcut in Figure 5 (right) were a projection, the time complexity and model size would double, because the shortcut connects the two high-dimensional ends (Section 4.1).

**Tiny example, computed by `scripts/bottleneck_params.py`.** Count weights with no biases and no batch normalization:

| Block | Weights |
|---|---|
| Basic, two 3×3 at 64 channels | 73,728 |
| Bottleneck, 256 → 64 → 64 → 256 | 16,384 + 36,864 + 16,384 = 69,632 |
| Bottleneck plus a 1×1 projection from 256 to 256 | 69,632 + 65,536 = 135,168 |

The first two rows agree with "similar complexity". The projection raises the bottleneck's size by a factor of 1.94, which matches "doubled". A basic block at the bottleneck's 256-channel width would have 1,179,648 weights, which is 16.9 times the bottleneck's. The bottleneck is what makes depth 152 affordable: the paper reports 11.3 billion FLOPs for ResNet-152, against 15.3 and 19.6 billion for VGG-16 and VGG-19 (Section 4.1).

Exercise 4 asks you to write these counts.

### What the results say

With the residual form, depth helps instead of hurting. Table 3 lists single-model top-1 and top-5 validation errors (10-crop) that fall with depth: ResNet-34 B at 24.52% top-1, ResNet-50 at 22.85%, ResNet-101 at 21.75%, and ResNet-152 at 21.43%. An ensemble of six models, only two of them 152 layers deep, reaches 3.57% top-5 error on the ImageNet test set (Table 5), which won ILSVRC 2015 classification. On CIFAR-10 (Table 6), ResNet-110 reaches 6.43% test error. A 1202-layer network trains to under 0.1% training error but tests at 7.93%, and the authors attribute the gap to overfitting on a small dataset (Section 4.2).

<!-- exercises -->

## Pitfalls

- **Reading the paper as a vanishing-gradient fix.** The paper trains its plain networks with batch normalization, checks that their backward gradients have healthy norms, and says the degradation is unlikely to come from vanishing gradients (Section 4.1). Exercise 3 uses a network with no normalization and a small weight scale to isolate what the shortcut does to gradient flow. A plain net with batch normalization and a good initialization won't show that collapse, and still degrades.
- **Putting the ReLU in the wrong place.** In this paper's block, the second nonlinearity comes after the addition: $\sigma(\mathcal{F}(x) + x)$ (Section 3.2, Figure 2). Applying it to $\mathcal{F}$ before the addition gives a different network. The same authors' follow-up paper studies a variant where nothing follows the addition.
- **Mismatched shapes.** Equation 1 needs $x$ and $\mathcal{F}(x)$ to have the same shape. At a stage boundary, where the width doubles and the map halves (Table 1, `conv3_1`, `conv4_1`, and `conv5_1` use stride 2), you need Equation 2 or Option A.
- **Using a one-layer residual function.** If $\mathcal{F}$ has a single layer, Equation 1 becomes $y = W_1 x + x$, a linear layer, and the paper reports no benefit (Section 3.2). Use two or three layers.
- **Forgetting the projection's cost in a bottleneck.** A projection at the 256-wide ends nearly doubles the block's weights (Exercise 4). Keep identity shortcuts there.
- **Comparing against a different baseline.** The plain and residual networks in Table 2 have the same depth, width, and parameter count only because Option A adds no parameters. Option B and C results (Table 3) include extra parameters.
- **Expecting more depth to keep helping.** The 1202-layer CIFAR-10 network optimizes well and still tests worse than the 110-layer one (Table 6).
- **Carrying the idea to Transformers carelessly.** Each Transformer sub-layer sits in a residual connection (see [the Transformer paper](/transformer/)), but it places layer normalization around the addition. Where the normalization goes changes how well deep stacks train.

## Further reading

- [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027), by He et al., is the follow-up that analyzes the signal path through a stack of blocks and moves the normalization and activation before the weights.
- [deep-residual-networks](https://github.com/KaimingHe/deep-residual-networks) is the authors' Caffe release of the ImageNet models.
- [Residual Networks Behave Like Ensembles of Relatively Shallow Networks](https://arxiv.org/abs/1605.06431), by Veit et al., studies the paths through a residual network.
- [torchvision's ResNet implementation](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py) shows the basic and bottleneck blocks in PyTorch.
