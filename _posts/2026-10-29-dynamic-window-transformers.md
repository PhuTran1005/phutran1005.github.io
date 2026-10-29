---
title: 'Dynamic Window Transformers: Adapting Attention Across Scales and Shapes'
date: 2026-10-29
permalink: /posts/2026/10/dynamic-window-transformers/
excerpt: 'From DW-ViT’s content-dependent multi-scale fusion to DWT’s window-shape routing for super-resolution: mechanisms, computational costs, and a PyTorch example.'
tags:
  - dynamic window transformer
  - attention
  - super-resolution
  - computer vision
---

An attention window determines which image features can interact directly. A small square emphasizes nearby detail. A larger square provides more context. A rectangle changes the balance between horizontal and vertical coverage. The useful choice can depend on the image and the task.

**Dynamic window transformers** make some part of this choice depend on the input. The name appears in different architectures, so this post distinguishes two specific methods:

- **DW-ViT**, from [Beyond Fixation: Dynamic Window Visual Transformer, CVPR 2022](https://arxiv.org/abs/2203.12856), computes several window scales and dynamically weights their features.
- **DWT**, from [Dynamic Window Transformer for Image Super-Resolution, ACCV 2024](https://openaccess.thecvf.com/content/ACCV2024/html/Xie_Dynamic_Window_Transformer_for_Image_Super-Resolution_ACCV_2024_paper.html), learns to route features among candidate window shapes.

Both adapt contextual processing, but their routing and computation differ. Together, they extend the discussion in my earlier posts on [CSWin Transformer]({{ '/posts/2026/10/cswin-transformer-cross-shaped-windows/' | relative_url }}) and [Transformer-based super-resolution]({{ '/posts/2025/06/transformer-sisr/' | relative_url }}).

## Specify what actually changes

The word *dynamic* becomes useful once three things are clear: the candidate windows, the granularity of the decision, and which branches execute.

| Method | Candidate geometry | Input-dependent operation |
|---|---|---|
| Swin | Fixed local windows with shifted partitions | Attention weights within each window |
| CSWin | Horizontal and vertical stripes | Attention weights within each head's stripe |
| DW-ViT | Several predetermined square sizes | Soft weighting across scale branches |
| DWT for SR | Several predetermined rectangular/square shapes | Shape-expert routing |

Changing a learned attention matrix inside a fixed window is different from adapting the contribution of whole window branches. Neither of the two dynamic models discussed here predicts an unrestricted polygon around every pixel.

For a candidate window, the familiar attention operation remains. Omitting positional terms and dropout for clarity:

$$
A_i=\operatorname{softmax}\!\left(
\frac{Q_iK_i^\top}{\sqrt d}\right)V_i.
$$

The additional design question is how to construct and combine those candidate computations.

## DW-ViT: compute multiple scales, then adapt the fusion

DW-ViT divides its attention heads equally among window scales. Each group operates on its own square partition, and its output is restored to the same feature-map coordinates. With `G` scales and `C` total channels, a branch produces `C/G` channels.

The [paper](https://arxiv.org/html/2203.12856v2#S3.SS2) studies window sides such as `[7, 14, 21]`. A small-window group captures local interactions, while larger-window groups connect more distant locations. Different stages can use different schedules; the candidate sizes are architectural settings.

<figure>
  <img src="{{ '/images/blog/dynamic-window-fusion.svg' | relative_url }}" width="480" height="798" alt="DW-ViT evaluates several fixed window scales, predicts input-dependent channel weights, sums the weighted branch features, and combines the expanded result with projected concatenated features." loading="lazy">
  <figcaption>DW-ViT's dynamic multi-scale fusion. Every scale branch runs; the input determines how its features contribute.</figcaption>
</figure>

### The gate varies by sample and channel

Let `Y_i` be the spatially restored feature map for scale `i`. DW-ViT forms a descriptor from projected concatenated branch features, followed by GELU, global average pooling, and a small MLP. The resulting logits produce softmax weights across scales:

$$
\begin{aligned}
\alpha_{b,:,c} &= \operatorname{softmax}_{i}(z_{b,:,c}), \\
R_b &= \sum_{i=1}^{G}\alpha_{b,i}\odot Y_{b,i}.
\end{aligned}
$$

Here `b` indexes the sample and `c` a channel within a group. The weights are shared over spatial positions. Thus one channel can emphasize a small window while another emphasizes a larger window, but this gate does not make separate decisions for every image location.

The weighted sum `R` has `C/G` channels. As shown in the [official `SKConv` code](https://github.com/pzhren/DW-ViT/blob/main/DW-ViT.py), the complete fusion expands it back to `C` channels and adds a projected concatenation:

$$
\begin{aligned}
Z &= P\!\left(\operatorname{Concat}_iY_i\right), \\
Y &= E(R)+Z.
\end{aligned}
$$

`P` maps `C` channels to `C`, and `E` maps `C/G` to `C`. This distinction matters when implementing the module: a weighted sum alone would leave the output too narrow.

DW-ViT also retains a four-stage hierarchy and Swin-style alternating ordinary and shifted window blocks. Dynamic fusion adapts scale preferences, while shifted partitions support communication across window boundaries.

## DWT for super-resolution: choose the window shape

The [ACCV 2024 DWT paper](https://openaccess.thecvf.com/content/ACCV2024/papers/Xie_Dynamic_Window_Transformer_for_Image_Super-Resolution_ACCV_2024_paper.pdf) focuses on image anisotropy: useful structures can extend farther in one direction than another. Its candidate window shapes, expressed as **height × width**, are:

- `4 × 16`: wider horizontal coverage.
- `8 × 8`: balanced coverage.
- `16 × 4`: taller vertical coverage.

All three contain 64 tokens. Their attention matrices therefore have the same size per head, while their spatial memberships differ. Changing aspect ratio changes which structures can be connected directly without increasing the token budget of an individual window.

### A feature-map router, rather than a per-pixel decision

The router globally average-pools the input feature map and applies linear layers with GELU to produce expert weights through Gumbel–Softmax. The decision is made for each input sample at each dynamic-attention block. It does not independently classify every pixel or local window.

Using `E_i(X)` to denote a shape expert, the paper's training mixture and inference algorithm can be summarized as:

$$
\begin{aligned}
Y_{\mathrm{train}} &= \sum_i\omega_i E_i(X), \\
k^* &= \operatorname*{argmax}_i\omega_i, \\
Y_{\mathrm{infer}} &= \omega_{k^*}E_{k^*}(X).
\end{aligned}
$$

Training evaluates all shape experts to form the weighted mixture. At inference, Algorithm 1 executes the highest-weight expert and retains its routing weight in the output. QKV projections are shared before shape-specific partitioning.

This is a different execution policy from DW-ViT, whose soft multi-scale fusion evaluates every branch. It also introduces a training/inference difference worth checking when reproducing the model. A faithful implementation must preserve the router, temperature behavior, and expert weighting, and explicitly specify its batching policy.

## A geometry example with a fixed token budget

Imagine a query on a thin horizontal railing. An `8 × 8` window covers nearby rows and columns. A `4 × 16` window spends more of its 64 positions along the railing, while a `16 × 4` window covers a longer vertical span. Which grouping helps depends on the feature content and the window boundaries.

This illustration explains a possible benefit, rather than proving that a router will identify the best shape for every structure. DWT's globally pooled decision summarizes the whole feature map, so a single image containing both horizontal and vertical patterns can create competing preferences.

By comparison, [CSWin](https://arxiv.org/abs/2107.00652) processes both stripe orientations in parallel head groups. DWT's rectangles have a bounded area and its inference router selects a shape expert. Their spatial reach and computational schedules should be evaluated separately.

## Computational cost: distinguish weighting from skipping

Let `N = HW`, with `C` channels. A local window containing `A` tokens has approximately `N/A` windows. Computing scores and aggregating values therefore costs about `2NAC`, excluding projections and other operations.

For DW-ViT, equally divide channels over `G` scales with square sides `m_i`:

$$
\begin{aligned}
T_{\mathrm{DW\text{-}ViT,mix}}
&\approx \frac{2NC}{G}\sum_i m_i^2, \\
T_{\mathrm{one\ shape,mix}}
&\approx 2NAC.
\end{aligned}
$$

Fixed candidate sizes keep this mixing cost linear in `N`, ignoring padding overhead. The weighted fusion does not eliminate the computation of a low-weight branch. Projections, the gate, expansion layers, and the MLP add further work.

In the DWT shape mixture, each expert processes the shared projected features at full channel width. With three equal-area candidates, training evaluates three sets of attention mixing operations; inference evaluates one selected set per sample. That does not imply a threefold difference in total runtime, because projections and other network components remain.

On accelerators, mixed routing decisions within a batch can also require regrouping samples or launching smaller kernels. Padding to multiple window sizes, memory layout changes, and framework support affect the measured speed. Report end-to-end latency alongside analytic operation counts.

## A small PyTorch example of dynamic fusion

The following educational module illustrates **DW-ViT-style fusion** for already-computed scale features. Each input branch has shape `[B, H, W, C/G]`. It omits the attention branches, positional biases, shifted windows, and training recipe; it is not a complete reproduction of either paper.

```python
import torch
from torch import nn


class DynamicScaleFusion(nn.Module):
    def __init__(self, channels, n_scales):
        super().__init__()
        if n_scales < 1 or channels < 1 or channels % n_scales:
            raise ValueError("channels must divide into nonempty groups")
        self.n_scales = n_scales
        self.group_channels = channels // n_scales
        hidden = max(1, self.group_channels // 2)
        self.project = nn.Linear(channels, channels)
        self.gate = nn.Sequential(
            nn.Linear(channels, hidden), nn.GELU(),
            nn.Linear(hidden, channels),
        )
        self.activate = nn.GELU()
        self.expand = nn.Linear(self.group_channels, channels)

    def forward(self, branches):
        if len(branches) != self.n_scales:
            raise ValueError("unexpected number of scale branches")
        joined = torch.cat(branches, dim=-1)
        shortcut = self.project(joined)
        descriptor = self.activate(shortcut).mean(dim=(1, 2))
        b = joined.shape[0]
        logits = self.gate(descriptor).reshape(
            b, self.n_scales, self.group_channels
        )
        weights = logits.softmax(dim=1)
        features = torch.stack(branches, dim=1)
        mixed = (features * weights[:, :, None, None, :]).sum(dim=1)
        return self.expand(mixed) + shortcut, weights


branches = [torch.randn(2, 8, 8, 4) for _ in range(3)]
fusion = DynamicScaleFusion(channels=12, n_scales=3)
output, weights = fusion(branches)
assert output.shape == (2, 8, 8, 12)
assert torch.allclose(weights.sum(dim=1), torch.ones(2, 4))
```

The returned weights have shape `[B, G, C/G]`, which makes their granularity visible. A different sample can produce different channel preferences. Because the branch tensors are inputs to this module, their computation has already happened before the gate produces the fused result.

## What the papers demonstrate

The [DW-ViT paper](https://arxiv.org/html/2203.12856v2) evaluates ImageNet-1K classification, ADE20K semantic segmentation, and COCO detection and instance segmentation. For example, its 224-resolution classification comparison reports **82.0% top-1 for DW-T**, versus **81.3% for Swin-T**, with listed costs of 5.2G and 4.5G FLOPs respectively. Those are results from the paper's training setup, including its 300-epoch ImageNet recipe. They describe an accuracy–cost trade-off, rather than an identical-budget comparison.

For SR, [DWT reports](https://openaccess.thecvf.com/content/ACCV2024/papers/Xie_Dynamic_Window_Transformer_for_Image_Super-Resolution_ACCV_2024_paper.pdf) **26.56 dB on Urban100 at ×4**, versus **26.47 dB for the lightweight SwinIR baseline** in its table. The setting uses bicubic degradation, DIV2K training, and Y-channel evaluation. The reported models have approximately 901K versus 897K parameters. This result should be read within that synthetic-SR protocol; it does not establish the same improvement for real-world degradations or every SwinIR configuration.

The [DW-ViT repository](https://github.com/pzhren/DW-ViT) provides an implementation. The SR discussion above follows the published DWT method and algorithm, rather than claiming that the illustrative fusion code reproduces its shape router.

## How I would evaluate the dynamic part

I would compare a strong fixed-window baseline, static multi-window fusion, and the dynamic version under the same data and training settings. This separates the benefit of additional branches from the benefit of input-dependent weights.

For a soft gate, inspect average weights, their variation between images and channels, and whether one scale dominates throughout training. For an expert router, also inspect selection frequencies, expert utilization, and the difference between the training mixture and inference policy. A visualization of gate values can help diagnose behavior, but task metrics must establish its usefulness.

In restoration, I would additionally test unseen degradations, image rotations, and tiled inference. Rotating an image exchanges horizontal and vertical structures; tiling changes the globally pooled descriptor and can change the selected expert. Compare full-image and tiled outputs, inspect seams, and report the overlap policy alongside PSNR, SSIM, perceptual quality, runtime, and memory.

The central design choice is how to allocate contextual processing to the input. Multi-scale soft fusion and shape-expert routing offer different answers, with different training behavior and deployment costs.
