---
title: 'Mamba in Computer Vision: From Selective State Spaces to Image and Video Restoration'
date: 2026-10-29
permalink: /posts/2026/10/mamba-in-computer-vision/
excerpt: 'How selective state spaces work, how visual scans preserve spatial context, and what Mamba brings to recognition, image restoration, and video super-resolution.'
tags:
  - Mamba
  - state space models
  - computer vision
  - image restoration
  - video super-resolution
---

A high-resolution image contains many locations that should inform one another. In video, useful information also arrives in neighboring frames. Capturing these relationships becomes expensive as the number of pixels, patches, and frames grows.

Mamba approaches sequence modeling through a compact state that is updated according to the input. For vision, this raises a useful question: **how can we carry information over large spatial and temporal distances while preserving the local details that a task needs?**

This post explains the underlying mechanism, the changes needed for visual data, and representative applications. I will also connect these ideas to our work on video super-resolution, [VSRM]({{ '/projects/vsrm/' | relative_url }}).

## Why sequence length matters in vision

Consider an image divided into non-overlapping 16 × 16 patches. Doubling its height and width produces four times as many tokens:

| Image resolution | Patch tokens | Token pairs in dense global attention |
|---|---:|---:|
| 512 × 512 | 1,024 | 1,048,576 |
| 1,024 × 1,024 | 4,096 | 16,777,216 |

The last column counts pairs for one attention head; it excludes channels, batches, and layers. It illustrates the quadratic arithmetic of full self-attention. Restoration can be more demanding because a model may retain a token at every feature-map location.

Efficient attention implementations change the memory picture. [FlashAttention](https://arxiv.org/abs/2205.14135), for example, computes exact attention without storing the entire attention matrix in GPU high-bandwidth memory. Fixed-window attention also has linear token scaling when the window size stays fixed. These distinctions matter when comparing architectures.

## A state space model carries a running representation

A state space model describes how an internal state evolves as inputs arrive. In its basic continuous form, suppressing channel and batch indices:

$$
\begin{aligned}
\frac{d h(t)}{d t} &= A h(t) + B x(t), \\
y(t) &= C h(t).
\end{aligned}
$$

The input `x(t)` changes the state `h(t)`, and the output reads from that state. The matrix `A` determines the state's dynamics. Structured state space models such as [S4](https://arxiv.org/abs/2111.00396) made this formulation practical for learning from long sequences.

After discretization, a useful schematic recurrence is:

$$
\begin{aligned}
h_t &= \overline{A}_t h_{t-1} + \overline{B}_t x_t, \\
y_t &= C_t h_t + D x_t.
\end{aligned}
$$

The first term carries previous information forward; the second writes new information into the state. The output combines a state readout with a direct input path. The bars indicate discretized coefficients; this equation leaves the input-discretization details abstract.

For an intuitive visual example, imagine scanning across a building facade. The state could retain information about repeated window structures while processing the current patch. Its capacity is finite, so the model must compress the preceding context into information useful for subsequent predictions.

## What makes Mamba selective?

In the original Mamba formulation, the discretization step `Δ_t` and the input/output coefficients `B_t` and `C_t` depend on the current input. The learned continuous-time matrix `A` is shared across sequence positions, while the discretized transition changes through `Δ_t`:

$$
\overline{A}_t = \exp(\Delta_t A).
$$

This gives the model input-dependent control over how it retains information, writes to its state, and reads an output. With a stable negative diagonal `A`, increasing `Δ_t` makes the previous state decay more rapidly. The [Mamba paper](https://arxiv.org/abs/2312.00752) explains the selection mechanism and its relationship to gating.

The recurrence supports an associative parallel scan when the sequence is available. Mamba combines this with a hardware-aware implementation, including kernel fusion and recomputation. During recurrent inference, it can instead update a fixed-size state and a short convolution buffer. Training activations still depend on sequence length; constant recurrent state does not imply constant training memory. The authors provide the implementation in [state-spaces/mamba](https://github.com/state-spaces/mamba).

## Images need a spatial scanning strategy

An image is a two-dimensional grid. A raster scan places horizontally adjacent locations close together, but vertically adjacent locations can be many sequence steps apart. It also places the end of one row next to the beginning of another.

As a result, the choice of sequence ordering becomes part of the model design. Bidirectional processing supplies context from both ends of an ordering. Complementary horizontal and vertical routes improve spatial coverage. Local convolutions can retain short-range interactions alongside the state-space computation.

The following schematic illustrates the scan-and-merge idea used in visual state-space designs. It abstracts away implementation details and is not the exact block diagram of every vision Mamba model.

<pre><code class="language-mermaid">flowchart TD
    A["2D feature map"] --> B["Choose spatial scan routes"]
    B --> C["Selective state-space processing"]
    C --> D["Restore spatial positions and merge routes"]
    D --> E["Features for the vision task"]
</code></pre>

A receptive field covering the image describes where information can travel. The strength and precision of that information still depend on the state capacity, ordering, and surrounding architecture.

## Vision backbones: Vim and VMamba

**Vision Mamba, usually called Vim**, embeds image patches with positional information and processes them using bidirectional Mamba blocks. This adapts sequence modeling to visual representations that benefit from context on both sides of a token. Its evaluation includes image classification, object detection, and semantic segmentation. The [ICML paper](https://proceedings.mlr.press/v235/zhu24f.html) and [official code](https://github.com/hustvl/Vim) describe the architecture.

**VMamba** introduces a two-dimensional selective scan, SS2D, inside a hierarchical vision backbone. It forms four complementary routes: row-wise and column-wise traversal, together with their reversals. The resulting sequences are scanned, mapped back to their spatial positions, and merged. This gives a concrete solution to the mismatch between ordered sequences and image grids. See the [VMamba paper](https://arxiv.org/abs/2401.10166) and [implementation](https://github.com/MzeroMiko/VMamba).

For dense prediction, the arrangement of intermediate features matters as much as the sequence mixer. A segmentation decoder needs spatially organized representations, and a detection head benefits from features at multiple scales. The backbone must expose those representations in a useful form.

## Image restoration: global context with local detail

Super-resolution and denoising require precise spatial reconstruction. A representation can capture an object's overall structure while losing the edge or texture detail needed to reconstruct it.

[MambaIR](https://arxiv.org/abs/2402.15648) addresses this by combining state-space processing with local enhancement and channel attention. The local operations help preserve neighboring-pixel interactions, while channel attention refines the feature representation. The paper evaluates the design on restoration tasks including image super-resolution and denoising; its [official repository](https://github.com/csguoh/MambaIR) contains the models and training code.

This suggests a useful division of responsibilities: a scan carries contextual information over long distances, while local operations support spatial precision. In restoration, I would judge that combination using both numerical metrics and visual inspection. Thin lines, repeated patterns, and difficult degradations can reveal failures that an average score obscures.

## Video applications: understanding and reconstruction

Video adds motion and a temporal dimension. The same object may move across many spatial positions, disappear behind another object, and reappear later.

### Video understanding with VideoMamba

[VideoMamba](https://arxiv.org/abs/2403.06977) applies bidirectional Mamba processing to spatiotemporal patch tokens for video understanding. Its design investigates spatial-first scanning and self-distillation, among other choices. The [OpenGVLab implementation](https://github.com/OpenGVLab/VideoMamba) provides a concrete example of adapting the architecture to video representations.

Bidirectional clip processing uses context from later frames within the clip. A deployment requiring immediate outputs must separately account for future-frame access and latency.

### Video super-resolution with VSRM

In our ICCV 2025 work, [VSRM](https://arxiv.org/abs/2506.22762), we combine Mamba-based aggregation with explicit motion alignment and reconstruction objectives:

- **Dual Aggregation Mamba Block:** combines spatial-first bidirectional scans with a temporal-first forward scan to aggregate complementary spatiotemporal information.
- **Deformable Cross-Mamba Alignment:** uses optical flow, learned reference offsets, and cross-Mamba interaction to align features from neighboring frames.
- **Frequency Charbonnier-like loss:** adds a Fourier-domain training signal alongside spatial Charbonnier loss, comparing real and imaginary frequency components.

The [project page]({{ '/projects/vsrm/' | relative_url }}) includes the architecture and visual results.

These components address separate parts of reconstruction. Context aggregation carries information across locations and frames. Alignment establishes useful correspondences under motion. The training objective shapes the reconstructed output. My interpretation is that a large context range helps most when the model can also associate that context with the correct geometry.

## Mamba-2 and hybrid vision architectures

[Mamba-2](https://proceedings.mlr.press/v235/dao24a.html) develops structured state-space duality, connecting particular state-space and attention formulations. Its algorithms make more effective use of matrix multiplication. This provides useful background for understanding later state-space architectures; individual vision models can use different generations of Mamba blocks and different scan implementations.

[MambaVision](https://arxiv.org/abs/2407.08083) explores a hybrid design. Early stages use convolutions, while later stages combine redesigned Mamba mixers with self-attention. Its evaluation covers classification, detection, and segmentation. The [NVIDIA repository](https://github.com/NVlabs/MambaVision) provides the implementation.

Hybrids offer a way to allocate different operations to different stages. When reading their efficiency claims, the cost of the entire architecture matters, including any attention layers that remain.

## How I would evaluate a visual Mamba model

For a practical comparison, I would start with the target task and input size, then examine the following questions:

| Question | What to examine |
|---|---|
| Does the scan preserve relevant geometry? | Ordering, directional coverage, boundaries, and motion correspondence. |
| Is local detail retained? | Edges, small structures, repeated textures, and temporal consistency. |
| Is the runtime comparison representative? | Resolution, clip length, batch size, precision, hardware, and warm-up procedure. |
| Where is memory used? | Training activations, feature maps, temporary scan buffers, and inference state. |
| Which operations dominate the full model? | Projections, convolutions, scan/merge transforms, attention, and reconstruction. |

Linear token scaling describes how a scan's arithmetic grows when its width and state size are fixed. End-to-end performance also depends on layout transformations, scan directions, implementation quality, and the surrounding network. Measurements at the intended resolution and clip length are therefore more informative than the architecture label alone.

For restoration, I would additionally check the degradation model and evaluation protocol. For video, I would test motion boundaries, occlusions, flicker, and the number of future frames used. These checks help distinguish an efficient context mechanism from a complete solution to the task.

Readers interested in the attention side of this discussion can continue with my earlier post on [Transformer-based single-image super-resolution]({{ '/posts/2025/06/transformer-sisr/' | relative_url }}). Together, these approaches provide different ways to organize context, computation, and reconstruction detail.
