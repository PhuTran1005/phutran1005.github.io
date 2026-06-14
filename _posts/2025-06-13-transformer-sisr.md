---
title: 'Single Image Super-Resolution with Transformer-Based Architectures: A Deep Dive'
date: 2025-06-13
permalink: /posts/2025/06/transformer-sisr/
tags:
  - super-resolution
  - transformers
  - computer vision
  - low-level vision
---

Single Image Super-Resolution (SISR) is one of the fundamental problems in low-level computer vision: given a low-resolution (LR) input, reconstruct a high-resolution (HR) output that is visually faithful and perceptually sharp. For years, convolutional neural networks (CNNs) dominated this space. But since the rise of the Vision Transformer (ViT), the field has shifted dramatically — and Transformer-based SR models now hold state-of-the-art results across virtually every benchmark.

This post walks through the core ideas, key architectures, and open challenges in Transformer-based SISR — written from the perspective of someone actively working in this space.

---

## Why CNNs Weren't Enough

Before diving into Transformers, it's worth understanding what CNNs struggle with in SISR.

CNNs process images through stacked local convolutions. Each layer aggregates information from a small neighborhood defined by the kernel size (typically 3×3 or 5×5). To capture long-range dependencies — say, the relationship between a texture patch in one corner and a corresponding region elsewhere in the image — CNNs need many stacked layers, which means deep architectures with large parameter counts.

This **locality constraint** is the central limitation. Super-resolution, especially at high upscaling factors (×3, ×4, ×8), often requires reasoning about self-similar structures across the entire image. A patch of brick wall in one region should inform the reconstruction of a similar patch elsewhere. CNNs can approximate this with very deep networks (e.g., EDSR with 32 residual blocks), but it's fundamentally expensive and indirect.

Transformers, by contrast, compute **global self-attention** — every position attends to every other position in a single operation. This makes long-range dependency modeling a first-class citizen rather than an emergent property of depth.

---

## The Attention Mechanism in Vision

The core operation in a Transformer is the scaled dot-product attention:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where Q (queries), K (keys), and V (values) are linear projections of the input features, and $d_k$ is the key dimension used for scaling.

In the context of images, each spatial position (or patch) produces a query, key, and value vector. The attention matrix captures pairwise similarity between all positions — effectively learning *which parts of the image are most relevant to reconstruct each location*.

For SR, this has a natural interpretation: pixels in the LR image can attend to globally similar textures and structures, enabling the network to "copy" high-frequency details from distant but related regions rather than hallucinating them from local context alone.

---

## SwinIR: The Foundational Architecture

The most influential Transformer-based SR model is **SwinIR** (Liang et al., ICCV 2021), which adapts the Swin Transformer for image restoration tasks.

### The Swin Transformer Block

The key insight in Swin is **shifted window attention**. Instead of computing global self-attention (which is $O(N^2)$ in the number of pixels), attention is computed within local non-overlapping windows of size $M \times M$. This reduces complexity to $O(N \cdot M^2)$, making it tractable for high-resolution images.

To enable cross-window interaction, consecutive Transformer layers alternate between two configurations:
- **Regular window partitioning**: windows aligned to a grid
- **Shifted window partitioning**: windows offset by $(\lfloor M/2 \rfloor, \lfloor M/2 \rfloor)$ pixels

This alternation means information flows across window boundaries every two layers, providing a form of quasi-global attention without the quadratic cost.

### SwinIR's Overall Design

SwinIR consists of three stages:

**1. Shallow Feature Extraction**  
A single 3×3 convolution maps the LR input to a feature embedding. This is important — the initial convolution provides translation equivariance and low-level edge detection that pure Transformers lack.

**2. Deep Feature Extraction**  
A stack of Residual Swin Transformer Blocks (RSTBs). Each RSTB contains several Swin Transformer layers followed by a 3×3 convolution and a residual connection from the block input. The convolution inside each RSTB serves a critical role: it re-introduces local inductive bias between the global attention operations.

**3. Image Reconstruction**  
For super-resolution, a sub-pixel convolution (PixelShuffle) upsamples the features by the target scale factor $s$, converting a $C \times H \times W$ feature map into a $3 \times (sH) \times (sW)$ RGB output.

A long skip connection from the shallow features to the reconstruction head helps preserve low-frequency content, letting the deep features focus on high-frequency residuals.

### Why SwinIR Works

The combination of local windowed attention and the long skip connection strikes a balance that CNNs miss:
- **Global context** via shifted windows and multi-scale receptive fields across RSTB depth
- **Local precision** via the convolutions embedded in each RSTB
- **Training stability** via residual connections at every level

SwinIR surpassed all prior CNN-based methods (EDSR, RCAN, DAN) at ×2, ×3, and ×4 on the standard DIV2K, BSD100, Set5, and Set14 benchmarks.

---

## HAT: Hybrid Attention for SR

A significant limitation of SwinIR is that its windowed attention still operates locally — the window size (typically 8×8) constrains how much global context each position can see per layer.

**HAT** (Chen et al., CVPR 2023) addresses this with two key innovations:

### Hybrid Attention Module (HAM)

HAT introduces a module that combines:
- **Window-based self-attention (WSA)**: identical to SwinIR's shifted window attention, capturing local structure
- **Channel attention (CA)**: a squeeze-and-excitation mechanism that aggregates global context through channel-wise statistics

The channel attention computes global average pooling across all spatial positions, producing a global descriptor of the feature map. A small MLP then outputs per-channel scaling weights. Because channel attention integrates information from the entire spatial extent, it effectively gives each position access to global context — complementing the local window attention.

The HAM processes features through both branches in parallel and combines them via addition:

$$\text{HAM}(X) = \text{WSA}(X) + \text{CA}(X)$$

### Overlapping Cross-Attention (OCA)

The second innovation is overlapping cross-attention, applied at a coarser level. While WSA operates on non-overlapping $8 \times 8$ windows, OCA introduces overlapping windows with a larger context region (e.g., $12 \times 12$) from which keys and values are drawn, while queries come from the inner $8 \times 8$ region.

This allows the attention in each query window to incorporate features from neighboring windows without the discontinuities that non-overlapping partitioning can create. The overlap size is a hyperparameter — larger overlaps capture more context at greater memory cost.

### Pre-training on Large-Scale Data

HAT also demonstrates that Transformer-based SR models benefit significantly from pre-training on large datasets (e.g., the ImageNet-based task of image restoration) before fine-tuning on standard SR benchmarks. This is a departure from prior CNN-based work, which typically trained from scratch on relatively small SR datasets (DIV2K: 800 images). The pre-training strategy is now standard in SOTA SR models.

HAT achieves notable gains over SwinIR, especially at ×4 — where accurate high-frequency prediction is hardest.

---

## CRAFT and CAT: Cross-Attention for Non-Local Similarity

A recurring theme in SR is **non-local self-similarity**: textures and structures repeat across an image, and explicit exploitation of this can boost reconstruction quality. CRAFT (Zhao et al., NeurIPS 2022) and CAT (Chen et al., NeurIPS 2022) formalize this through cross-attention mechanisms.

### CAT: Cross Aggregation Transformer

CAT introduces **rectangle window attention** — windows of varying aspect ratios that capture anisotropic structure. A horizontal strip window is good for capturing horizontal edges and textures, while a vertical strip captures vertical structures. By using multiple window shapes, the model builds a richer set of reference features for each query location.

The key technical contribution is a **cross-attention aggregation** operation: for each query window, CAT computes attention not only within the window but also against keys from adjacent windows of different orientations. This is implemented efficiently using a shift-and-share operation that reuses key/value computations across overlapping windows.

CAT also introduces a **token reduction** technique — not every spatial position needs to generate a query at every scale. By selectively computing attention for high-uncertainty regions (estimated from the magnitude of intermediate features), CAT reduces FLOPs while maintaining quality.

---

## CPAT: My Own Work

I want to mention our own contribution, **CPAT** (BMVC 2024), which takes a different perspective on the attention mechanism in SR.

### Channel-Partitioned Windowed Attention

The core observation in CPAT is that not all feature channels benefit from the same attention pattern. Low-level channels (e.g., those encoding edges and textures) benefit most from local, fine-grained attention. High-level channels (encoding structural patterns and global statistics) benefit from wider receptive fields.

Rather than applying a single attention operation uniformly across all channels, CPAT partitions the channel dimension into groups and applies windowed attention with different window sizes per group. Smaller windows for texture-sensitive channels, larger windows for structural channels. This multi-scale channel partitioning allows the model to allocate its attention budget more efficiently than architectures using a single uniform window size.

### Frequency Learning

Beyond attention, CPAT incorporates explicit frequency learning: a parallel branch that applies a Discrete Cosine Transform (DCT) to the features, processes the frequency-domain representation through a small MLP, and projects back to spatial domain. The frequency branch specializes in recovering high-frequency components (sharp edges, fine textures) that attention-based mechanisms can struggle with, since self-attention is inherently a weighted averaging operation that tends to produce over-smooth outputs.

The frequency branch output is fused with the attention branch output through a learned gating mechanism, allowing the model to dynamically balance spatial and frequency-domain information.

CPAT achieves competitive results on standard benchmarks while being more parameter-efficient than HAT, making it more practical for real-world deployment.

---

## Training Recipes: What Actually Matters

Architecture is only part of the story. Training procedures have a major impact on SR model performance.

### Loss Functions

**L1 / L2 pixel loss** remains the standard baseline. L1 (mean absolute error) tends to produce sharper results than L2 (mean squared error), which is known to over-smooth due to its implicit assumption of Gaussian noise.

**Perceptual loss** (Johnson et al., 2016) uses a pretrained VGG network to compare intermediate feature representations of the SR output and HR target. This produces perceptually sharper results at the cost of some PSNR — a well-known trade-off.

**Charbonnier loss** is a smooth approximation of L1:
$$\mathcal{L}_\text{Charb}(x, y) = \sqrt{(x - y)^2 + \epsilon^2}$$
where $\epsilon$ is a small constant (typically $10^{-3}$). It is differentiable everywhere, unlike L1, and more robust to outliers than L2. Most recent SOTA models (including SwinIR, HAT, CPAT) use Charbonnier loss.

**Frequency-domain losses** (used in VSRM and CPAT) compute the loss in the Fourier or DCT domain, explicitly penalizing errors in high-frequency components. Since PSNR is dominated by low-frequency reconstruction accuracy, frequency-domain losses help recover fine texture detail that pixel-domain losses miss.

### Data Augmentation

Standard augmentations for SR:
- **Random cropping**: LR/HR patch pairs, typically 64×64 LR (256×256 HR at ×4)
- **Horizontal flipping** and **90° rotations**: 8-fold augmentation via all dihedral symmetries of the square
- **Mixup / CutMix**: less common but used in some recent works

### Degradation Models

This is increasingly important for real-world SR. The standard **bicubic downsampling** assumption is unrealistic — real LR images have complex noise, blur, JPEG compression, and sensor artifacts. **Real-ESRGAN** (Wang et al., 2021) introduced a high-order degradation pipeline that randomly composes multiple degradation types, enabling robust SR on in-the-wild images. Most recent SOTA models include training on such synthetic degradations.

---

## Benchmarks and Evaluation

### Standard Benchmarks

| Dataset | # Images | Notes |
|---------|----------|-------|
| Set5 | 5 | Small, saturated — almost all methods >40 dB PSNR at ×2 |
| Set14 | 14 | More diverse content |
| BSD100 | 100 | Natural images from Berkeley Segmentation Dataset |
| Urban100 | 100 | Buildings with repetitive geometric patterns — hardest for SR |
| Manga109 | 109 | Japanese manga — extreme high-frequency content |
| DIV2K | 100 (val) | High-res diverse images; standard training set too |

### Metrics

**PSNR** (Peak Signal-to-Noise Ratio) is the dominant metric. Computed in the Y channel (luma) of YCbCr space. Higher is better. Most papers report improvements of 0.1–0.5 dB as significant; 0.5+ dB is a major jump.

$$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{255^2}{\text{MSE}}\right)$$

**SSIM** (Structural Similarity Index) measures luminance, contrast, and structural similarity jointly. Also computed in Y channel. Less sensitive than PSNR to small pixel-level differences and better correlated with human perception in some cases.

**LPIPS** (Learned Perceptual Image Patch Similarity) uses deep features from AlexNet or VGG to measure perceptual similarity. More relevant for GAN-based SR where outputs may be perceptually sharper but have lower PSNR.

A critical note: PSNR/SSIM are positively correlated with smoothness. GAN-based methods often have *lower* PSNR but *better* perceptual quality — this is the fundamental **distortion-perception tradeoff** (Blau & Michaeli, CVPR 2018).

---

## Open Challenges

Despite impressive benchmark numbers, several fundamental challenges remain.

### 1. The PSNR-Perception Gap

The best PSNR/SSIM models produce results that are technically accurate but often look blurry to human observers. GAN-based methods (ESRGAN, Real-ESRGAN) produce sharper, more realistic textures, but their outputs can contain hallucinated details that don't correspond to the ground truth. For medical imaging or remote sensing, such hallucinations are unacceptable. Closing this gap — high perceptual quality *and* high fidelity — is an open problem.

### 2. Real-World Degradation

Bicubic SR is effectively solved. The frontier is in-the-wild SR, where degradation types are unknown. Current approaches (Real-ESRGAN, BSRGAN) use synthetic degradation pipelines, but the distribution shift between synthetic and real degradations remains a significant challenge.

### 3. Computational Efficiency

SwinIR at ×4 takes ~6.6 GFlops for a 720p image. HAT takes more. For mobile or edge deployment, this is prohibitive. Efficient SR (IMDN, RFDN, etc.) remains an active research area — and Mamba-based architectures (like our own VSRM) offer a promising direction with linear rather than quadratic complexity.

### 4. Scale Generalization

Most SR models are trained and evaluated at a fixed scale factor. Continuous and arbitrary-scale SR (LIIF, ITSRN) allow a single model to handle any upscaling factor, but with some quality degradation relative to scale-specific models.

### 5. Reference-Based SR (RefSR)

Given a high-resolution reference image of a similar scene, can we further improve reconstruction quality? RefSR models (TTSR, MASA) use cross-attention between the LR input and the HR reference to transfer textures. This is particularly relevant for applications where reference images are available (e.g., time-lapse video, satellite imagery with prior acquisitions).

---

## Summary: Key Milestones in Transformer SR

| Year | Model | Key Innovation |
|------|-------|---------------|
| 2021 | SwinIR | Shifted window attention for SR |
| 2022 | CAT | Cross-window aggregation, rectangle attention |
| 2022 | CRAFT | Non-local cross-attention for texture transfer |
| 2023 | HAT | Hybrid attention (window + channel) + pre-training |
| 2024 | CPAT | Channel-partitioned attention + frequency learning |
| 2025 | VSRM | Mamba-based spatiotemporal SR (video) |

---

## Further Reading

- Liang et al., *SwinIR: Image Restoration Using Swin Transformer*, ICCV 2021 — [[paper](https://arxiv.org/abs/2108.10257)]
- Chen et al., *Activating More Pixels in Image Super-Resolution Transformer (HAT)*, CVPR 2023 — [[paper](https://arxiv.org/abs/2205.04437)]
- Chen et al., *Cross Aggregation Transformer for Image Restoration (CAT)*, NeurIPS 2022 — [[paper](https://arxiv.org/abs/2211.13654)]
- Wang et al., *Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data*, ICCV 2021 — [[paper](https://arxiv.org/abs/2107.10833)]
- Tran et al., *Channel-Partitioned Windowed Attention and Frequency Learning for Single Image Super-Resolution (CPAT)*, BMVC 2024 — [[paper](https://arxiv.org/abs/2407.16232)]
- Blau & Michaeli, *The Perception-Distortion Tradeoff*, CVPR 2018 — [[paper](https://arxiv.org/abs/1711.06077)]
