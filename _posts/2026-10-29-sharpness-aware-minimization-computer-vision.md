---
title: 'Sharpness-Aware Minimization for Computer Vision: From Loss Geometry to Practice'
date: 2026-10-29
permalink: /posts/2026/10/sharpness-aware-minimization-computer-vision/
excerpt: 'How SAM searches for weights with low loss in their neighborhood, what vision research has demonstrated, and how to implement and evaluate its two-pass update.'
tags:
  - Sharpness-Aware Minimization
  - optimization
  - computer vision
  - generalization
---

Architecture and training objectives receive much of the attention in computer vision. The optimizer also influences which solution a network reaches. Two models can fit their training images similarly while behaving differently on new images, imaging devices, or degradation patterns.

**Sharpness-Aware Minimization (SAM)** makes the neighborhood around the current weights part of the training objective. Introduced by [Foret et al. at ICLR 2021](https://arxiv.org/abs/2010.01412), it encourages solutions whose loss stays low under small parameter perturbations. Here, SAM refers to the optimization method; the Segment Anything family uses the same acronym for a different purpose.

This post explains the update, examines evidence from vision research, and develops a practical experiment for image restoration. It complements my posts on [visual Mamba models]({{ '/posts/2026/10/mamba-in-computer-vision/' | relative_url }}) and [Transformer-based super-resolution]({{ '/posts/2025/06/transformer-sisr/' | relative_url }}): the focus here is how to train a model rather than how to build its backbone.

## Optimize a neighborhood of weights

Let `w` denote the trainable parameters, and let `L_B` be the average loss on minibatch `B`. For clarity, start without weight decay or another explicit regularizer. The robust objective is:

$$
\begin{aligned}
&\min_w F_B(w), \\
&F_B(w)=\max_{\lVert\epsilon\rVert_2\leq\rho}
L_B(w+\epsilon).
\end{aligned}
$$

The radius `ρ` specifies the parameter neighborhood. A practical approximation uses two gradient evaluations on the same minibatch:

$$
\begin{aligned}
g &= \nabla_w L_B(w), \\
\hat\epsilon &= \rho\frac{g}{\lVert g\rVert_2}, \\
g_{\mathrm{SAM}} &= \left.\nabla_u L_B(u)\right|_{u=w+\hat\epsilon}, \\
w_{t+1} &= w_t-\eta g_{\mathrm{SAM}}.
\end{aligned}
$$

The first gradient defines an ascent perturbation. The second supplies the descent direction, treating that perturbation as constant. The weights must return to `w` before applying the base optimizer. Momentum SGD or AdamW can supply that final update; the displayed last line uses plain SGD.

The normalization follows directly from a first-order expansion:

$$
L_B(w+\epsilon)\approx L_B(w)+g^\top\epsilon.
$$

By Cauchy–Schwarz, the largest linearized increase inside an L2 ball is obtained by moving along `g`. This solves the **linearized** inner problem. It does not exactly maximize a nonlinear network's loss over the ball. When the gradient is zero, the implementation below uses a zero perturbation.

The [authors' implementation](https://github.com/google-research/sam) provides the full training recipe. Regularization, normalization state, and distributed gradient aggregation are additional choices that matter when reproducing results.

## A numerical picture of the trade-off

Consider two illustrative local loss functions. Here `u` is a displacement from each basin's own center; these are separate neighborhoods, rather than two complete neural-network loss landscapes:

$$
\begin{aligned}
L_A(u)&=0.10+10u^2, \\
L_B(u)&=0.15+u^2.
\end{aligned}
$$

With `ρ = 0.10`, the exact neighborhood maxima are:

| Local basin | Loss at its center | Worst loss within the radius |
|---|---:|---:|
| A: steeper | 0.10 | 0.20 |
| B: flatter | 0.15 | 0.16 |

<figure>
  <img src="{{ '/images/blog/sam-local-loss-neighborhood.svg' | relative_url }}" alt="Illustrative quadratic losses: basin A has lower center loss but rises faster than basin B within a parameter radius of 0.10." loading="lazy">
  <figcaption>Author-created illustration of neighborhood loss. The curves are analytic examples, not measurements from a trained vision model.</figcaption>
</figure>

Optimizing only center loss favors A. Optimizing the exact neighborhood objective favors B. However, the one-gradient ascent approximation at either exact center produces zero perturbation, so this example explains the objective, rather than guaranteeing that practical SAM escapes every sharp stationary point.

For a smooth loss near a stationary local minimum, a second-order expansion gives another useful interpretation:

$$
\begin{aligned}
&\max_{\lVert\epsilon\rVert_2\leq\rho}
\big[L(w+\epsilon)-L(w)\big] \\
&\qquad\approx \frac{\rho^2}{2}\lambda_{\max}(H).
\end{aligned}
$$

This local approximation assumes a positive-semidefinite Hessian `H` and a sufficiently small neighborhood. It relates the worst local increase to curvature, without turning SAM into an explicit Hessian-eigenvalue optimizer.

## Flatness requires a coordinate system

Euclidean distance between weights depends on the network's parameterization. [Dinh et al.](https://arxiv.org/abs/1703.04933) show that rescaling equivalent networks can change apparent sharpness while preserving the represented function. A flat-looking plot alone therefore cannot establish better generalization.

For experiments, I would hold architecture, parameter scaling, loss reduction, and perturbation definition fixed. Validation performance remains the deciding measurement. A weight-space neighborhood also gives no automatic guarantee about robustness to corrupted or adversarially perturbed input images.

## What has SAM demonstrated in computer vision?

### Image classification and Vision Transformers

The original SAM paper reports improved generalization on image benchmarks including CIFAR-10, CIFAR-100, and ImageNet, as well as robustness to label noise in its evaluated settings.

A particularly clear example is [When Vision Transformers Outperform ResNets without Pre-training or Strong Data Augmentations](https://arxiv.org/abs/2106.01548), published at ICLR 2022. With basic Inception-style preprocessing, its authors report ImageNet top-1 improvements of **5.3 percentage points for ViT-B/16** and **11.0 points for Mixer-B/16** using SAM. Those gains belong to that training setting. A pretrained model with a different augmentation and regularization recipe needs a fresh comparison.

For a classification experiment, I would measure top-1 accuracy alongside training time and calibration. Label-noise experiments should state the noise process and rate explicitly, since different annotation errors can produce different behavior.

### Domain generalization

Changes in camera, style, environment, or acquisition process can shift the test distribution. [Sharpness-Aware Gradient Matching (SAGM), CVPR 2023](https://arxiv.org/abs/2303.10353), develops a related objective that considers empirical loss, perturbed loss, and their gap. Its evaluation covers PACS, VLCS, OfficeHome, TerraIncognita, and DomainNet.

This work also highlights a limitation: ordinary SAM need not find a flat region with a sufficiently small loss. Domain-generalization improvements from SAGM should be attributed to its complete method. In a new application, evaluate held-out domains directly rather than infer domain robustness from training sharpness.

### Medical image segmentation

[UG-CEMT, WACV 2025](https://arxiv.org/abs/2412.15380), incorporates uncertainty-emphasizing SAM into a semi-supervised medical segmentation framework with ensemble mean teachers and consistency regularization. It evaluates multi-center prostate MRI and cardiac MRI.

This is evidence that sharpness-aware training can be a component of a dense-prediction system. Its reported performance belongs to the combined framework. A clean study of SAM itself would keep the decoder, labeled-data split, consistency losses, and augmentation fixed, then change the optimization method. Patient-level separation and evaluation across acquisition sites are especially useful for detecting misleading gains.

## How I would apply SAM to super-resolution

For restoration, I would treat SAM as an experimental training choice. Classification gains do not establish a PSNR improvement for an SR network.

Start with the existing reconstruction objective, such as L1 or Charbonnier loss, and apply the two-pass update to exactly that objective. Keep the same low-resolution/high-resolution pair and crop for both passes. If training combines pixel, perceptual, and frequency losses, state whether the ascent uses their full weighted sum or only selected terms; those choices define different methods.

My evaluation would include:

| Question | Measurement |
|---|---|
| Does reconstruction fidelity improve? | PSNR and SSIM with a fixed crop, color space, and evaluation protocol. |
| Are perceptual details preserved? | Perceptual metrics and close inspection of edges, textures, and repeated patterns. |
| Does it transfer to new degradations? | Tests with unseen blur, noise, compression, and real-image degradation patterns. |
| Is video behavior stable? | Temporal consistency and motion-boundary inspection for video restoration. |
| Is the improvement worth the cost? | Wall-clock training time, memory, and quality under the same compute budget. |

These are proposed experiments, rather than results from my own SAM-trained restoration model. An architecture such as Mamba or a Transformer can be held fixed while testing this optimizer question.

## Variants address different limitations

[ASAM, ICML 2021](https://proceedings.mlr.press/v139/kwon21b.html), adapts the perturbation neighborhood to parameter scale. Its motivation addresses the sensitivity of fixed-radius sharpness to rescaling. Changing the neighborhood also changes the meaning of `ρ`, so a radius selected for ordinary SAM should be retuned.

[LookSAM, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Towards_Efficient_and_Scalable_Sharpness-Aware_Minimization_CVPR_2022_paper.html), reduces additional computation by periodically calculating the inner ascent and approximating its contribution between those steps. The paper studies CNNs and ViTs, including large-batch training. Actual speed depends on the model, implementation, and hardware.

Vanilla SAM needs two sequential forward/backward evaluations per update. Inference uses the resulting network normally; the perturbation search is a training operation.

## A minimal PyTorch training step

The following educational implementation assumes **one device, FP32 parameters, dense gradients, and an ordinary SGD or AdamW optimizer**. It computes a single global gradient norm over the optimized parameters. To isolate the mechanism, start with `weight_decay=0` and a scalar data loss without an explicit regularizer.

```python
import torch


def sam_step(model, optimizer, images, targets, criterion, rho=0.05):
    if rho < 0:
        raise ValueError("rho must be nonnegative")
    params = [p for group in optimizer.param_groups
              for p in group["params"] if p.requires_grad]

    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(images), targets)
    loss.backward()
    active = [(p, p.grad.detach().clone())
              for p in params if p.grad is not None]
    if not active:
        raise RuntimeError("No gradients for the optimized parameters")
    norm = torch.linalg.vector_norm(torch.stack([
        torch.linalg.vector_norm(g) for _, g in active
    ]))
    if not torch.isfinite(norm):
        raise RuntimeError("Non-finite first-pass gradient")

    # Preserve weights and the buffers produced by the first forward.
    weights = [(p, p.detach().clone()) for p in params]
    buffers = [(b, b.detach().clone()) for b in model.buffers()]
    try:
        with torch.no_grad():
            for p, g in active:
                p.add_(rho * g / (norm + 1e-12))
        optimizer.zero_grad(set_to_none=True)
        perturbed_loss = criterion(model(images), targets)
        perturbed_loss.backward()
    finally:
        with torch.no_grad():
            for p, original in weights:
                p.copy_(original)
            for buffer, original in buffers:
                buffer.copy_(original)

    # Use the second gradient at the original weights.
    optimizer.step()
    return loss.detach(), perturbed_loss.detach()
```

Call `model.train()` before the training loop and pass images whose random cropping and augmentation have already been sampled. The snapshot approach uses additional memory to make restoration explicit. If the second pass raises an exception, `finally` restores the weights and buffers before the exception propagates.

BatchNorm needs particular care. Both passes can use training batch statistics, while running buffers retain the first pass's updates. That is the state-handling pattern in the [authors' training code](https://github.com/google-research/sam/blob/main/sam_jax/training_utils/flax_training.py); the buffer snapshots above implement it directly. Switching the whole model to evaluation mode would change the second loss computation. Dropout and stochastic depth also introduce randomness between passes, so record the stochastic-layer policy when comparing implementations.

For production mixed-precision or distributed training, use an implementation that explicitly supports those settings. [PyTorch's AMP documentation](https://docs.pytorch.org/docs/stable/notes/amp_examples.html) requires careful gradient-scaling state management, including at most one `unscale_` call per optimizer between steps. Distributed implementations must also specify whether the first gradient is synchronized: a local-worker perturbation and a globally aggregated perturbation define different updates.

Regularization deserves its own ablation. An explicit L2 term can affect the ascent if included in `criterion`; base-optimizer weight decay acts at the update. AdamW's decoupled decay is another design choice. Specify the recipe rather than treating all three as interchangeable.

## An experiment I would trust

I would first establish a strong SGD or AdamW baseline, then tune SAM's radius and learning rate using validation data only. An initial radius grid such as `0.01, 0.05, 0.10` is an experiment proposal, not a universal default. Parameter scaling changes the geometry of the neighborhood. Loss weighting can change the ascent direction, while loss reduction affects the base optimizer's update scale.

Use multiple seeds and report both **equal-update** and **equal-compute** comparisons. Equal updates expose the change in the optimization trajectory; equal compute tests whether spending the same training budget another way is more useful. Keep data splits, augmentation, architecture, and evaluation code fixed, and log both ordinary and perturbed losses during training.

For my vision work, the most interesting question is whether neighborhood-aware optimization improves the behavior that matters for the task: unseen degradations in restoration, cross-domain performance in recognition, or consistency of dense predictions. The evidence should come from those measurements, supported by a clearly specified training procedure.
