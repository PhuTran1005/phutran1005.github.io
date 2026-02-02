---
title: "VSRM: A Robust Mamba-Based Framework for Video Super-Resolution"
collection: projects
permalink: /projects/vsrm/
layout: single
author_profile: true
---


## SeedVR2: One-Step Video Restoration via Diffusion Adversarial Post-Training

**Dinh Phu Tran¹, Dao Duy Hung¹, Daeyoung Kim¹**

¹ School of Computing, KAIST, South Korea

---

## 🎬 Video Results

<p align="center">
  <video width="90%" controls>
    <source src="reds4_000_SRx4.mp4" type="video/mp4">
  </video>
</p>

<p align="center">
  <video width="90%" controls>
    <source src="reds4_011_SRx4.mp4" type="video/mp4">
  </video>
</p>

---

## Abstract

Video super-resolution remains a major challenge in lowlevel vision tasks. To date, CNN- and Transformer-based methods have delivered impressive results. However, CNNs
are limited by local receptive fields, while Transformers struggle with quadratic complexity, posing challenges for processing long sequences in VSR. Recently, Mamba has drawn attention for its long-sequence modeling, linear complexity, and large receptive fields. In this work, we propose VSRM, a novel Video Super-Resolution framework that leverages the power of Mamba. VSRM introduces Spatial-to-Temporal Mamba and Temporal-to-Spatial Mamba blocks to extract long-range spatio-temporal features and enhance receptive fields efficiently. To better align adjacent frames, we propose Deformable Cross-Mamba Alignment module. This module utilizes a deformable crossmamba mechanism to make the compensation stage more dynamic and flexible, preventing feature distortions. Finally, we minimize the frequency domain gaps between reconstructed and ground-truth frames by proposing a simple yet effective Frequency Charbonnier-like loss that better preserves high-frequency content and enhances visual quality. Through extensive experiments, VSRM achieves stateof-the-art results on diverse benchmarks, establishing itself as a solid foundation for future research.

---

## Method

<p align="center">
  <img src="overall_architecture.png" width="95%">
</p>

**Contribution.** 1) We successfully adapt Mamba-based model for VSR for the first time and propose a new framework, VSRM. VSRM introduces Dual Aggregation Mamba
Block (DAMB), which aggregates spatial and temporal features from sequences by using multi-scan direction mechanisms to obtain powerful representation ability and enlarge the model’s receptive field. 2) We design Deformable Cross-mamba Alignment (DCA) module to provide a more flexible and learnable alignment through a deformable window cross-mamba mechanism in the compensation stage. 3) We propose Frequency Charbonnier-like loss (FCL), which is computed in the frequency domain to preserve and generate high-frequency details better. 4) VSRM outperforms all state-of-the-art (SOTA) methods on various benchmarks and shows the potential as the solid backbone for VSR.

---

## Qualitative Results

<p align="center">
  <img src="viz.png" width="95%">
</p>

<p align="center">
  <img src="erf.png" width="95%">
</p>

---

## 📄 Paper & Code

- **Paper**: [arXiv](https://openaccess.thecvf.com/content/ICCV2025/papers/Tran_VSRM_A_Robust_Mamba-Based_Framework_for_Video_Super-Resolution_ICCV_2025_paper.pdf)
- **Code**: [GitHub](https://github.com/PhuTran1005/phutran1005.github.io)

---

## BibTeX

```bibtex
@InProceedings{Tran_2025_ICCV,
    author    = {Tran, Dinh Phu and Hung, Dao Duy and Kim, Daeyoung},
    title     = {VSRM: A Robust Mamba-Based Framework for Video Super-Resolution},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {14711-14721}
}