---
title: 'CSWin Transformer: How Cross-Shaped Windows Connect an Image'
date: 2026-10-29
permalink: /posts/2026/10/cswin-transformer-cross-shaped-windows/
excerpt: 'An illustrated guide to parallel stripe attention, locally enhanced positional encoding, and the accuracy–computation trade-off in CSWin Transformer.'
tags:
  - CSWin Transformer
  - vision transformer
  - attention
  - computer vision
---

A vision backbone needs both nearby detail and context from distant image regions. Global self-attention provides direct communication between every pair of tokens, but its attention matrix grows quadratically with the token count. Small windows reduce that cost while restricting which locations can interact in a single layer.

**CSWin Transformer** organizes attention into horizontal and vertical stripes. Introduced by Dong et al. at CVPR 2022, its name stands for **Cross-Shaped Window**. Different groups of attention heads follow the two orientations in parallel. The [original paper](https://arxiv.org/abs/2107.00652) and [official Microsoft implementation](https://github.com/microsoft/CSWin-Transformer) describe a hierarchical backbone for classification, detection, and segmentation.

This post develops the attention geometry, its computational cost, and the implementation details that matter when adapting the idea. It also connects to my earlier discussion of [Transformer-based super-resolution]({{ '/posts/2025/06/transformer-sisr/' | relative_url }}), where the balance between spatial precision and contextual coverage is especially relevant.

## Two head groups, two spatial routes

Consider a feature map with height `H`, width `W`, and `C` channels. Let `s` be the stripe width, assuming the relevant dimensions are divisible by `s`.

- A **horizontal stripe** contains `s` rows across the full width: `s × W` tokens.
- A **vertical stripe** contains `s` columns across the full height: `H × s` tokens.

CSWin assigns half of its attention heads to each orientation. Every head computes attention within its own stripe. After attention, the branch outputs return to their image positions, concatenate along the channel dimension, and pass through an output projection.

<figure>
  <img src="{{ '/images/blog/cswin-cross-shaped-windows.svg' | relative_url }}" width="480" height="820" alt="Two parallel attention head groups on an 8 by 8 grid: one attends within a horizontal stripe and the other within a vertical stripe, before concatenation and projection." loading="lazy">
  <figcaption>Separate head groups see the same query through different spatial routes. Each colored stripe has its own attention normalization.</figcaption>
</figure>

For this `8 × 8` example with `s = 2`, each head attends to 16 locations. Across both orientations, the query can receive information from 28 distinct locations: the two 16-token stripes overlap in a `2 × 2` region. That union describes the block's attention connectivity. It is not a single 28-token softmax shared by all heads.

The stripe is also different from a sliding cross centered independently at every query. Nearby queries inside the same stripe share the stripe's spatial extent.

## What one attention branch computes

For head `j`, partition the projected queries, keys, and values into the corresponding stripes. If the head dimension is `d`, its attention weights and output can be written as:

$$
\begin{aligned}
A_j &= \operatorname{softmax}\!\left(
\frac{Q_jK_j^\top}{\sqrt d}\right), \\
Y_j &= A_jV_j + P_j.
\end{aligned}
$$

Here `P_j` is the locally enhanced positional term discussed below. Each stripe gets its own attention matrix. Once the spatial arrangement has been restored, head outputs are combined:

$$
Y=\operatorname{Concat}(Y_1,\ldots,Y_K)W^O.
$$

This output projection mixes channels from the two orientations. The surrounding block follows the familiar pre-normalization residual structure: attention updates the token features, then a token-wise MLP updates them again. The [released model code](https://github.com/microsoft/CSWin-Transformer/blob/main/models/cswin.py) makes the partition, attention, reconstruction, and residual operations explicit.

## Long-range communication across blocks

A horizontal head can read far across the image immediately, while a vertical head can read far along its height. A location outside both stripes needs an indirect path.

For example, information from a distant token can first travel vertically to an intermediate location, then horizontally toward the query in a later block. Mixing the branch channels between blocks makes these complementary routes useful. This explains how stacked stripe blocks can develop broad spatial dependencies while each individual attention matrix remains restricted.

The existence of a path describes connectivity. The learned attention weights, feature capacity, and network depth determine how effectively information travels along it.

## LePE supplies local spatial structure

Attention weights depend on feature similarity. CSWin additionally uses **Locally-enhanced Positional Encoding (LePE)** to introduce a learned local operation on the projected values:

$$
P_j=\operatorname{DWConv}_{3\times3}(V_j).
$$

The depthwise convolution acts separately on channels, and its result is added to the attention-weighted values. This placement combines contextual aggregation with a local spatial signal before the output projection.

An important reproduction detail is the order of operations. In the [official `LePEAttention` implementation](https://github.com/microsoft/CSWin-Transformer/blob/main/models/cswin.py), values are first reshaped into stripe windows; the convolution then operates on those windows. Convolution over the entire feature map would use different neighbors at stripe boundaries.

LePE does not require a learned absolute-position table with one entry per token. That is useful when transferring the design to other resolutions. The surrounding implementation still needs correct shape handling: the released classification model stores fixed square resolutions, so arbitrary rectangular inputs require adaptations to partitioning and, where necessary, padding and masking.

## A hierarchy for vision features

CSWin begins with a `7 × 7`, stride-4 convolution and builds four feature stages. Between stages, stride-2 convolutions reduce the spatial size and increase the channel width. This creates a feature pyramid that downstream heads can consume.

For a `224 × 224` input, the released Tiny configuration illustrates the progression:

| Stage | Feature size | Channels | Stripe setting |
|---|---|---:|---|
| 1 | 56 × 56 | 64 | `s = 1` |
| 2 | 28 × 28 | 128 | `s = 2` |
| 3 | 14 × 14 | 256 | `s = 7` |
| 4 | 7 × 7 | 512 | Global attention |

The 224-resolution models use the stripe-size schedule `[1, 2, 7, 7]`; the official code explicitly switches the final stage to a single global-attention branch. At that point there are only 49 spatial tokens. Classification finishes with normalization, token averaging, and a linear classifier.

For detection and segmentation, intermediate feature scales are useful because small structures and large semantic regions need different spatial representations. The backbone supplies those features; a detection or segmentation system still needs its task-specific head.

## Deriving the attention cost

Let `N = HW`. In a horizontal branch, there are `H/s` stripes, each containing `sW` tokens. The vertical branch similarly has `W/s` stripes of `Hs` tokens. Allocate half of the channel capacity to each orientation.

Counting the main multiply-accumulate terms for both score computation and value aggregation gives:

$$
\begin{aligned}
T_{\mathrm{horizontal}} &\approx HsW^2C, \\
T_{\mathrm{vertical}} &\approx WH^2sC, \\
T_{\mathrm{mix}} &\approx HWC\,s(H+W).
\end{aligned}
$$

The QKV and output projections contribute approximately `4HWC²` more. Together, these recover the [paper's attention-module expression](https://arxiv.org/html/2107.00652v3#S3.SS2):

$$
T_{\mathrm{CSWin}}\approx HWC\,[4C+s(H+W)].
$$

This expression omits the MLP, LePE, and implementation overhead. Different FLOP-counting conventions may also multiply arithmetic counts by a constant.

For a square `L × L` feature map, stripe attention mixing scales as `O(sL³C)`. In terms of token count, that is `O(sN^{3/2}C)` for fixed stripe width. Full attention mixing scales as `O(N²C)`.

For the earlier `8 × 8`, `s = 2` example, the mixing-arithmetic ratio relative to full attention is `s/L = 1/4`. This ratio applies to the attention mixing operations, rather than the entire model's runtime. Projections, convolutions, tensor rearrangements, and the MLP still consume computation.

## CSWin, Swin, and axial attention

[Swin Transformer](https://arxiv.org/abs/2103.14030) uses local square windows and shifts the partition between consecutive blocks to exchange information across window boundaries. CSWin's stripe heads instead span an entire feature-map dimension within a block.

[Axial attention](https://arxiv.org/abs/1912.12180) organizes attention along individual tensor axes. A width-one stripe resembles a row or column operation, but CSWin's parallel division of heads and its wider stripe schedule specify a particular architecture. The diagrams should be read with that distinction in mind.

| Attention pattern | Spatial region for one head | Route to broader context |
|---|---|---|
| Global | Entire feature map | Direct all-pairs attention |
| Swin window | Local square | Shifted partitions across blocks |
| CSWin stripe | Full-width or full-height stripe | Complementary head groups and stacked blocks |

For a fixed square-window size, Swin's attention mixing is linear in token count. CSWin's stripe lengths grow with the image dimensions. This creates a different trade-off between the reach of one block and its cost as resolution increases.

## A small PyTorch partitioning experiment

The following educational example isolates spatial partitioning. It accepts channels-last tensors and handles rectangular feature maps when the stripe dimension is divisible by `s`. It omits QKV projections, attention, LePE, and padding.

```python
import torch


def partition_stripes(x, s, direction):
    b, h, w, c = x.shape
    if s <= 0:
        raise ValueError("s must be positive")
    if direction == "horizontal":
        if h % s:
            raise ValueError("height must be divisible by s")
        return x.reshape(b, h // s, s, w, c).reshape(-1, s * w, c)
    if direction == "vertical":
        if w % s:
            raise ValueError("width must be divisible by s")
        return (x.reshape(b, h, w // s, s, c)
                .permute(0, 2, 1, 3, 4)
                .reshape(-1, h * s, c))
    raise ValueError("unknown direction")


def merge_stripes(tokens, shape, s, direction):
    b, h, w, c = shape
    if direction == "horizontal":
        return tokens.reshape(b, h // s, s, w, c).reshape(shape)
    if direction == "vertical":
        return (tokens.reshape(b, w // s, h, s, c)
                .permute(0, 2, 1, 3, 4)
                .reshape(shape))
    raise ValueError("unknown direction")


x = torch.arange(64).reshape(1, 8, 8, 1)
for direction in ("horizontal", "vertical"):
    stripes = partition_stripes(x, 2, direction)
    restored = merge_stripes(stripes, x.shape, 2, direction)
    print(direction, tuple(stripes.shape))  # (4, 16, 1)
    assert torch.equal(restored, x)
```

Both orientations produce four 16-token stripes here, but their token memberships differ. Printing the token IDs makes this visible. These functions operate on an entire tensor for clarity; a complete CSWin block applies the corresponding partitions to different projected head groups.

## What the reported results establish

The [official checkpoint table](https://github.com/microsoft/CSWin-Transformer#main-results-on-imagenet) lists these ImageNet-1K results at `224 × 224`, using ImageNet-1K training:

| Model | Parameters | Listed FLOPs | Top-1 accuracy |
|---|---:|---:|---:|
| CSWin-T | 23M | 4.3G | 82.8% |
| CSWin-S | 35M | 6.9G | 83.6% |
| CSWin-B | 78M | 15.0G | 84.2% |

The repository also reports COCO detection and instance segmentation with Mask R-CNN variants, and ADE20K semantic segmentation with Semantic FPN or UPerNet. These demonstrate use as a general vision backbone. Downstream scores depend on the head, training schedule, input size, and inference protocol; single-scale and multiscale segmentation results should be kept separate.

## Implications for restoration and high-resolution images

For my restoration interests, the useful design idea is directional context: a stripe can connect distant edges, repeated patterns, or structures aligned across rows and columns. This suggests an experiment with stripe attention inside a restoration network, while preserving local processing and a reconstruction path.

The original recognition backbone progressively downsamples its features. A super-resolution system needs a design that preserves and reconstructs fine detail, so adapting the attention mechanism requires more than attaching an upsampler to a classifier. I would test stripe width, orientation balance, boundary behavior, memory, and reconstruction quality while keeping the training and degradation settings fixed.

Tiled inference needs special attention. A stripe spanning a training crop or inference tile sees a different context extent from a stripe spanning the full image. Compare full-image and tiled outputs where feasible, inspect seams, and record the overlap and padding policies.

CSWin offers a concrete way to organize long-range communication through geometry. Its value for a new task should be judged through that task's accuracy, spatial fidelity, and measured cost at the intended resolution.
