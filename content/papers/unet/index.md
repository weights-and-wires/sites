---
id: "unet"
title: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
authors: "Ronneberger et al."
year: 2015
venue: "MICCAI 2015"
arxiv: "1505.04597"
url: "https://arxiv.org/abs/1505.04597"
track: vision
tier: extended
est_hours: 3
status: full
summary: "The U-Net labels every pixel of an image by pairing a contracting path, which pools to gather context, with a symmetric expanding path, which up-samples to recover resolution. Feature maps from the contracting path are copied across and concatenated at the matching resolution, so the expanding path can combine what it knows about context with exact locations. With heavy elastic-deformation augmentation and a weighted pixel-wise loss, the network trains from a few dozen annotated images and wins the ISBI 2012 and 2015 microscopy challenges."
related_course: "ml: CNNs"
prereqs:
  - id: resnet
    reason: "The U-Net predates ResNet and doesn't cite it, but ResNet's addition-based identity shortcut is the closest relative of the U-Net's concatenation-based skip connection, and comparing the two shows what a skip connection does."
background: [convolution, downsampling-and-upsampling]
course_links:
  - title: "Convolution: slide a filter, count the weights, count the view"
    url: "https://ml.ifkash.dev/lessons/038-convolution-and-receptive-field"
  - title: "Diffusion for audio"
    url: "https://speech.ifkash.dev/lessons/055-diffusion-for-audio"
self_check:
  - q: 'A 572 x 572 single-channel input passes through two unpadded 3x3 convolutions and one 2x2 max pool with stride 2. What is the height and width of the result?'
    a: 'Each unpadded 3x3 convolution trims 2 pixels from each side length, so the sizes go $572 \to 570 \to 568$. The pool halves 568 to 284. Padding would keep 572 through the convolutions, but the U-Net uses none.'
    see: convolution
  - q: 'A 2x2 up-convolution with stride 2 maps a $28 \times 28 \times 1024$ feature map to 512 channels. What is the output shape, and how many weights does the layer have, not counting biases?'
    a: 'The output is $56 \times 56 \times 512$, because stride 2 doubles each side length. The kernel has $2 \cdot 2 \cdot 1024 \cdot 512 = 2{,}097{,}152$ weights.'
    see: downsampling-and-upsampling
  - q: 'Two 64-channel maps meet at a skip connection. How many channels leave the join when ResNet adds them, and when the U-Net concatenates them? How many weights does the next 3x3 convolution to 64 channels need in each case?'
    a: 'Addition keeps 64 channels, so the next convolution has $3 \cdot 3 \cdot 64 \cdot 64 = 36{,}864$ weights. Concatenation gives 128 channels, so it has $3 \cdot 3 \cdot 128 \cdot 64 = 73{,}728$. Concatenation keeps both maps separate and lets the next layer decide how to combine them.'
    see: resnet
---

## The problem

Classification networks produce one label per image. Many biomedical tasks need one label per **pixel**: which pixels belong to a cell, a membrane, a neuron. The paper also notes that these tasks rarely come with thousands of annotated images (Section 1). The training set for the ISBI 2012 electron microscopy challenge has 30 images of 512 x 512 pixels (Section 4).

The strongest approach before this paper was the sliding-window network of Ciresan et al. (Section 1). It classifies each pixel from a patch around it. The paper names two drawbacks.

- **It is slow and redundant.** The network runs once per patch, and neighboring patches overlap almost completely. For one 512 x 512 image, that is 262,144 forward passes, against 4 tiles for the U-Net, computed by `scripts/shapes.py`.
- **Context and localization trade off.** Larger patches need more max pooling, which blurs where things are. Small patches see little context (Section 1).

The fully convolutional network of Long et al. removes the per-patch loop by up-sampling the contracting path's output. The paper builds on it and changes it so that it works with very few training images (Section 1).

The results show it works. On the EM challenge, the U-Net reaches a warping error of 0.0003529 and a Rand error of 0.0382, and ranks first by warping error against 0.000420 for the sliding-window network (Table 1). On the two ISBI 2015 cell-tracking datasets it reaches an intersection over union (IOU) of 0.9203 on PhC-U373, against 0.83 for the second best, and 0.7756 on DIC-HeLa, against 0.46 (Table 2). Training takes about 10 hours on one NVidia Titan GPU, and segmenting a 512 x 512 image takes under a second (Abstract, Section 5).

## Key ideas

The sections that follow use the paper's notation: pixel positions $x \in \Omega \subset \mathbb{Z}^2$, $K$ classes, activations $a_k(x)$, the softmax output $p_k(x)$, the true label $\ell(x)$, and the weight map $w(x)$. Every number comes from a script in `scripts/`. Run any script from the repository root to reproduce the output.

### Contracting and expanding paths

**Intuition.** Pooling throws away location to gain context: after three pools, one pixel summarizes an 8 x 8 area. A segmentation needs both the context (what is this blob?) and the location (which pixels?). The U-Net first shrinks the image to collect context, then grows it back with a mirror-image path that is also wide, so the context can reach the high-resolution layers (Section 1).

**Math.** The paper describes the two paths in words and in Figure 1 (Section 2). The paper doesn't give formulas for them.

- **Contracting path.** Repeat the following step. Apply two 3x3 convolutions without padding, each followed by a ReLU, then a 2x2 max pool with stride 2. Double the number of channels at every pool.
- **Expanding path.** Repeat the following step. Up-sample the map and apply a 2x2 "up-convolution" that halves the channels. Concatenate the matching cropped map from the contracting path. Apply two 3x3 convolutions, each followed by a ReLU.
- **Output.** A final 1x1 convolution maps each 64-component vector to $K$ class scores.

In total the network has 23 convolutional layers (Section 2). The count includes the up-convolutions, as Figure 1 shows.

**Tiny example, computed by `scripts/shapes.py`.** Follow a 572 x 572 input through Figure 1, with 4 pools:

| Stage | Side length | Channels |
|---|---|---|
| Input | 572 | 1 |
| After the first two convolutions | 568 | 64 |
| After pool 1 | 284 | 64 |
| After the next two convolutions | 280 | 128 |
| After pool 2 | 140 | 128 |
| After pool 3 | 68 | 256 |
| After pool 4 | 32 | 512 |
| Bottom, after two convolutions | 28 | 1024 |
| After the first up-convolution | 56 | 512 |
| Last up-convolution | 392 | 64 |
| After the last two convolutions | 388 | 64 |
| After the 1x1 convolution | 388 | 2 |

Every size matches Figure 1. The script counts 18 3x3 convolutions, 4 up-convolutions, and one 1x1 convolution, which is 23 layers. With biases, one input channel, and two output classes, the script counts 31,030,658 parameters. The paper doesn't state a parameter count, so treat that number as derived from Figure 1.

One notation difference: the paper says "upsampling of the feature map followed by a 2x2 convolution" (Section 2). Exercise 4 implements the up-convolution as a stride-2 transposed convolution, where each input pixel paints one 2x2 block. That is a common implementation of the Figure 1 arrow, but it isn't identical to nearest-neighbor up-sampling followed by a 2x2 convolution.

### Skip connections between matching resolutions

**Intuition.** After the bottom, the network knows a lot about what is in the image and little about where. The contracting path knew where, before the pooling discarded it. The paper copies the contracting path's maps across to the expanding path, so the next convolution can see both (Section 1).

**Math.** The expanding path concatenates the up-convolution output with "the correspondingly cropped feature map from the contracting path" (Section 2). The cropping is needed because every unpadded convolution loses border pixels. Concatenation is along the channel axis, so a $C_1$-channel copy and a $C_2$-channel up-convolution output give $C_1 + C_2$ channels. This differs from the ResNet shortcut (see [ResNet](/resnet/)), which adds the two maps and needs equal channel counts. The U-Net paper doesn't compare the two, and it predates ResNet.

**Tiny example, computed by `scripts/skip_example.py`.** A 6 x 6 copied map holds the numbers 0 to 35 row by row, and the up-convolution output is 4 x 4. The crop removes 1 pixel per side, and the first row of the cropped map is `[7, 8, 9, 10]`. A crop from the top-left corner instead returns `[0, 1, 2, 3]`, which pairs each pixel with a neighbor that is 1 pixel off in each direction. At level 0 of Figure 1, the 568 x 568 map is cropped to 392 x 392, which is 88 pixels per side (`scripts/shapes.py`; the figure's caption labels the crop only as "copy and crop").

At the last expansion step, two 64-channel maps concatenate into 128 channels. The next 3x3 convolution then has $9 \cdot 128 \cdot 64 = 73{,}728$ weights, against 36,864 without the skip.

Exercise 2 asks you to write the crop and concatenation. Exercise 4 uses it inside a one-level U-Net.

### Unpadded convolutions and the overlap-tile strategy

**Intuition.** The network uses only the "valid" part of each convolution, so every output pixel has its full context inside the input. A large image doesn't fit in GPU memory in one piece, so the paper cuts it into tiles. Each tile's input is larger than its output, and the missing border context at the image edge is filled in by mirroring (Section 1, Figure 2).

**Math.** The paper gives no formula. Each 3x3 convolution without padding removes 2 from the side length, and each 2x2 pool needs an even size, so "it is important to select the input tile size such that all 2x2 max-pooling operations are applied to a layer with an even x- and y-size" (Section 2). Only some input sizes are valid.

**Tiny example, computed by `scripts/shapes.py`.** An input of 572 gives an output of 388, so each side loses $(572 - 388)/2 = 92$ pixels of border. A 512 x 512 image needs $\lceil 512 / 388 \rceil = 2$ tiles per axis, which is 4 tiles, and a mirrored image of side $512 + 2 \cdot 92 = 696$. Between 560 and 600, only 572 and 588 are valid for 4 pools. The size 573 is invalid, because the first pool would see 569.

Exercise 1 asks you to write the size calculator.

### Dense per-pixel prediction and the weighted loss

**Intuition.** The loss is a cross entropy summed over all pixels. Two problems arise in cell images. Background is far more common than cells, and thin background gaps between touching cells are easy to ignore. The weight map $w(x)$ makes the gap pixels count much more (Section 3, Figure 3).

**Math.** The energy is built on a softmax over the $K$ channels at each pixel (Section 3):

$$
p_k(x) = \frac{\exp(a_k(x))}{\sum_{k'=1}^{K} \exp(a_{k'}(x))}
$$

The paper then writes the energy as (Section 3, Equation 1)

$$
E = \sum_{x \in \Omega} w(x) \log\bigl(p_{\ell(x)}(x)\bigr)
$$

where $\ell : \Omega \to \{1, \dots, K\}$ is the true label of each pixel. As printed, Equation 1 has no minus sign, so minimizing it would push $p_{\ell(x)}$ toward 0. The text calls it a cross-entropy loss, so this page and Exercise 3 minimize $-E$.

The weight map is (Section 3, Equation 2)

$$
w(x) = w_c(x) + w_0 \exp\!\left(-\frac{(d_1(x) + d_2(x))^2}{2\sigma^2}\right)
$$

where $w_c$ balances class frequencies, $d_1$ is the distance to the border of the nearest cell, and $d_2$ is the distance to the border of the second nearest cell. The paper sets $w_0 = 10$ and $\sigma \approx 5$ pixels. It doesn't give $w_c$ beyond "compensate the different frequency of pixels from a certain class", so the examples here use $w_c = 1$.

**Tiny example, computed by `scripts/weight_map.py`.** With $w_c = 1$, $w_0 = 10$, $\sigma = 5$:

| $d_1$ | $d_2$ | $w(x)$ |
|---|---|---|
| 1 | 1 | 10.2312 |
| 2 | 3 | 7.0653 |
| 5 | 5 | 2.3534 |
| 10 | 10 | 1.0034 |

A pixel in a 2-pixel-wide gap between two touching cells is worth about 10 times a pixel far from any border. Check the row $(2, 3)$ by hand: $(2+3)^2 = 25$, $25 / 50 = 0.5$, $e^{-0.5} = 0.6065$, so $w = 1 + 6.065 = 7.065$.

Now the loss. Take a 1 x 2 image with $K = 2$, activations $a(x_1) = [2, 0]$ and $a(x_2) = [0, 1]$, true labels $\ell = 0$ at both pixels, and weights $w = [1, 7]$. The softmax gives $p_0(x_1) = 0.8808$ and $p_0(x_2) = 0.2689$. The weighted loss is $1 \cdot 0.1269 + 7 \cdot 1.3133 = 9.3198$. Without weights it is 1.4402, so the weight puts almost all of the loss on the badly predicted gap pixel.

Exercise 3 asks you to write both formulas.

### Few images: augmentation and initialization

**Intuition.** With 30 training images, the network can't learn that a cell is still a cell after it is squeezed or stretched, unless it sees squeezed and stretched cells. The paper generates them. It also initializes the weights so that no part of a network with many paths starts with exploding or dead activations (Section 3).

**Math.** The paper says random elastic deformation is "the key concept" for training with few images (Section 3.1). It draws random displacement vectors on a coarse 3 x 3 grid from a Gaussian with a standard deviation of 10 pixels, and interpolates per-pixel displacements bicubically. Dropout at the end of the contracting path adds more implicit augmentation.

For the initial weights, the paper draws from a Gaussian with standard deviation $\sqrt{2/N}$, where $N$ is the number of incoming nodes of one neuron (Section 3, citing He et al.). For a 3x3 convolution on 64 channels, $N = 9 \cdot 64 = 576$.

**Tiny example, computed by `scripts/weight_map.py`.** The standard deviation is $\sqrt{2/576} = 0.0589$. In a check with random unit-variance input of 64 channels, this initialization gives a pre-ReLU output variance of 2.00 and a post-ReLU mean square of 1.00. The paper says each feature map should have "approximately unit variance". The more exact statement is that the second moment after the ReLU stays near 1, which is what lets 23 layers keep their scale. A smaller standard deviation, $\sqrt{1/N}$, gives a pre-ReLU variance of 1.01 here, and the ReLU then halves the second moment at every layer.

The paper also trains with a single-image batch and a momentum of 0.99, so that many earlier samples shape each update (Section 3). Exercises 1 to 4 don't train a network, so none of them covers this.

<!-- exercises -->

## Pitfalls

- **Assuming `same` padding.** The paper's network uses valid convolutions, so the output is smaller than the input. Most reimplementations pad instead, which makes the crop unnecessary but changes the border behavior and the tiling strategy.
- **Cropping from a corner.** A top-left crop of the skip runs without an error and misaligns every pixel by the crop offset. Center the crop.
- **Mixing up concatenation and addition.** The U-Net concatenates, so the next convolution takes twice the channels. A reimplementation that adds the maps changes the layer shapes and the parameter count.
- **Picking an input size that fails halfway.** A size that isn't valid raises an error at a pool deep in the network, not at the input. Compute the size before you build the tile loader.
- **Losing the minus sign.** Equation 1 as printed lacks it. Copied literally, the loss rewards wrong predictions.
- **Squaring the distances separately.** Equation 2 squares the sum $d_1 + d_2$, not each distance.
- **Applying a ReLU to the up-convolution or the 1x1 head.** Section 2 puts the ReLU after each 3x3 convolution only. The class scores need to stay signed before the softmax.
- **Copying the sampling hyperparameters from a segmentation library.** The batch size of 1 and the momentum of 0.99 belong together (Section 3). Keeping the momentum with a larger batch changes the training dynamics.
- **Treating the weight map as free.** Equation 2 needs the distances to the two nearest cells, which the paper pre-computes for each ground-truth segmentation with morphological operations. The preprocessing is part of the method.

## Further reading

- [The official U-Net page](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net), cited in the paper, hosts the original Caffe implementation and the trained networks. The page may have moved since the paper's publication.
- [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038), by Long et al., is the architecture that the paper builds on.
- [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852), by He et al., derives the $\sqrt{2/N}$ initialization.
- [3D U-Net](https://arxiv.org/abs/1606.06650), by Cicek et al., extends the design to volumetric data.
- [Denoising Diffusion Probabilistic Models](/ddpm/) uses a U-Net as the network that predicts the noise, which is why this paper matters outside biomedical imaging.
