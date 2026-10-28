---
title: 'Gated DeltaNet and Gated DeltaNet-2: How Recurrent Models Edit Their Memory'
date: 2026-10-29
permalink: /posts/2026/10/gated-deltanet-and-gated-deltanet-2/
excerpt: 'From the delta rule to independent erase and write gates: an illustrated explanation of Gated DeltaNet, Kimi Delta Attention, and Gated DeltaNet-2.'
tags:
  - Gated DeltaNet
  - linear attention
  - recurrent memory
  - sequence modeling
---

Long-context models need to manage the information accumulated from earlier tokens. A recurrent architecture stores that information in a state whose size can remain fixed as the sequence grows. The difficult part is deciding how to revise the state when new information arrives.

Gated DeltaNet and Gated DeltaNet-2 approach this as a memory-editing problem. Their updates combine forgetting, reading an existing association, and committing new content. The second model gives the erase and write operations separate controls.

This post develops that idea from a small associative memory. It connects naturally to my earlier post on [Mamba and selective state spaces]({{ '/posts/2026/10/mamba-in-computer-vision/' | relative_url }}), while focusing on a different question: **how should a model edit information already stored in its recurrent state?**

## Start with a key-value memory

For one head, let `S_t` be a matrix with `d_v` rows and `d_k` columns. A key `k_t` and query `q_t` have `d_k` components; a value `v_t` and output `o_t` have `d_v` components.

An additive associative update can be written as:

$$
\begin{aligned}
S_t &= S_{t-1} + v_t k_t^\top, \\
o_t &= S_t q_t.
\end{aligned}
$$

The outer product writes a key-value association, and multiplication by a query reads from the accumulated matrix. This illustrates the recurrent viewpoint of [linear attention](https://arxiv.org/abs/2006.16236). It omits feature-map and denominator-normalization details that appear in some formulations.

This memory is distributed across matrix entries. Similar keys can interact, so it is useful to think about interference between associations. Appending another outer product may reinforce a repeated fact, but it can also accumulate outdated content when the value associated with a key changes.

## DeltaNet corrects the current prediction

The delta rule first asks what the memory currently returns for the incoming key. It then uses the difference between the incoming value and that prediction:

$$
\begin{aligned}
r_t &= v_t - S_{t-1} k_t, \\
S_t &= S_{t-1} + \beta_t r_t k_t^\top.
\end{aligned}
$$

Here `β_t` controls the size of the correction. This memory interpretation is developed in [Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174).

There is also a simple optimization interpretation. For the temporary memory objective below, the gradient points along the prediction error and the current key:

$$
\begin{aligned}
\ell(S) &= \tfrac{1}{2}\lVert S k_t-v_t\rVert_2^2, \\
\nabla_S\ell &= (S k_t-v_t)k_t^\top.
\end{aligned}
$$

A gradient step on this objective produces the delta update. This is a way to understand the changing memory state; it does not mean that inference runs backpropagation over all of the model's trained parameters.

For a unit-length key, a correction with `β_t = 1` replaces the value read at that key. Other queries can still be affected when they correlate with the edited key. Orthogonal-key examples isolate the mechanism, but real learned representations rarely form a perfectly separated set of memory slots.

## Gated DeltaNet adds adaptive forgetting

[Gated DeltaNet](https://arxiv.org/abs/2412.06464), published at ICLR 2025, combines a scalar decay gate with the delta correction. Using value-by-key storage throughout this post:

$$
\begin{aligned}
\bar S_t &= \alpha_t S_{t-1}, \\
r_t &= v_t - \bar S_t k_t, \\
S_t &= \bar S_t + \beta_t r_t k_t^\top, \\
o_t &= S_t q_t.
\end{aligned}
$$

The state is decayed **before** computing the residual. The gate `α_t` scales the previous memory, while `β_t` scales a targeted correction at the current key. In the original formulation, both gates are scalars for each head and token.

For a scalar example, take an old state of `2`, a new value of `5`, a unit key, `α = 0.8`, and `β = 0.25`. Decay gives `1.6`; the residual is `3.4`; the correction is `0.85`. The updated state is **2.45**.

One limitation becomes visible in the residual: `β_t` scales both the subtraction of the old read and the addition of the incoming value. Those two parts share a control. The [official implementation](https://github.com/NVlabs/GatedDeltaNet) supplies the surrounding projections, normalization, convolution, and output processing as well.

## Kimi Delta Attention refines the decay

Kimi Delta Attention, or KDA, is an important intermediate step. It replaces scalar forgetting with decay that varies across key channels, while keeping a scalar delta-edit gate. This lets different parts of the memory retain information at different rates. The [Kimi Linear paper](https://arxiv.org/abs/2510.26692) describes KDA and the larger hybrid architecture built around it; the [official repository](https://github.com/MoonshotAI/Kimi-Linear) provides code and models.

In the storage convention used here, key-channel decay acts on the columns:

$$
\begin{aligned}
D_t &= \operatorname{Diag}(\boldsymbol{\alpha}_t), \\
\bar S_t &= S_{t-1}D_t.
\end{aligned}
$$

This separates the lifetimes of different key channels. It still leaves a second design question: should removing an existing association and writing an incoming value always use the same strength?

## Gated DeltaNet-2 separates erase and write

[Gated DeltaNet-2](https://arxiv.org/abs/2605.22791) introduces a key-channel erase gate `b_t` and a value-channel write gate `w_t`. The following is the paper's recurrence transposed into the value-by-key convention used above:

$$
\begin{aligned}
\bar S_t &= S_{t-1}D_t, \\
e_t &= b_t \odot k_t, \\
z_t &= w_t \odot v_t, \\
r_t &= z_t - \bar S_t e_t, \\
S_t &= \bar S_t + r_t k_t^\top, \\
o_t &= S_t q_t.
\end{aligned}
$$

The symbol `⊙` denotes elementwise multiplication. The erase gate changes the key-side read that will be subtracted. The write gate changes the value-side content that will be added. The final correction is applied along the current key.

The diagram shows the roles of these operations in a single update:

<pre><code class="language-mermaid">flowchart TD
    A["Old state"] --> B["Decay"]
    B --> C["Read with b"]
    D["Value"] --> E["Write with w"]
    C --> F["New minus old"]
    E --> F
    F --> G["Key correction"]
    B --> G
    G --> H["New state"]
</code></pre>

The order matters: decay precedes the read and correction. Key-dependent erasure is also a structured matrix edit, rather than a separate mask on every state entry. The [official code](https://github.com/NVlabs/GatedDeltaNet-2) contains the full layer and optimized implementations.

## A small example makes the difference visible

Consider a memory with one value channel and two orthogonal key channels. It stores `[10, 7]`. The incoming key selects the first channel, the incoming value is `10`, and decay is fixed to one.

With tied erase/write strength, the current read already equals the incoming value. The residual is zero, so Gated DeltaNet leaves `[10, 7]` unchanged for any choice of `β`.

Now let Gated DeltaNet-2 use erase strength `0.9` on the selected key channel and write strength `0.1`. It subtracts `9` from the selected association and adds `1`. The new memory is **`[2, 7]`**.

| Part of the selected association | Tied gate, β = 0.9 | Independent erase/write gates |
|---|---:|---:|
| Old content retained | 1 | 1 |
| New content added | 9 | 1 |
| Resulting value | 10 | 2 |

This is a deliberately chosen numerical example, not an accuracy benchmark. It holds the key, value, and decay fixed to expose the gate coupling. A complete network can also change its learned projections. The example shows what the independent controls do inside this particular update.

## The models form a sequence of special cases

Setting every erase channel and every write channel to the same scalar `β_t` gives the KDA-style edit. Additionally making every decay channel equal to the same `α_t` recovers the Gated DeltaNet recurrence. With the erase and write gates still tied, setting decay to one gives the ordinary delta-rule update.

| Model | Decay | Erase/write control |
|---|---|---|
| DeltaNet | None | Shared scalar β |
| Gated DeltaNet | Scalar α | Shared scalar β |
| KDA | Per-key-channel α | Shared scalar β |
| Gated DeltaNet-2 | Per-key-channel α | Separate key-side b and value-side w |

These reductions compare the core recurrences. Full architectures can differ in projections, activations, normalization, head dimensions, and the arrangement of layers.

## A readable reference update

This NumPy function follows the value-by-key convention and implements one head for one token:

```python
import numpy as np

def gated_delta2_step(S, q, k, v, decay, erase, write):
    # S: (value_dim, key_dim)
    # q, k, decay, erase: (key_dim,)
    # v, write: (value_dim,)
    decayed = S * decay[None, :]
    erase_key = erase * k
    new_value = write * v
    correction = new_value - decayed @ erase_key
    updated = decayed + np.outer(correction, k)
    return updated, updated @ q
```

The snippet assumes that the inputs and gates have already been computed. It omits query/key normalization and the surrounding neural-network block. It is intended for tracing the equations, not as a GPU training implementation. Gate parameterization and numerical precision should follow the model being reproduced.

## Recurrent updates can still support parallel training

Reading the equations token by token gives a useful reference implementation, but training efficiency requires more structure. The [parallel DeltaNet work](https://arxiv.org/abs/2406.06484) develops a hardware-efficient chunkwise computation using compact representations of products of low-rank updates.

For Gated DeltaNet-2, channel-wise decay and independent edit factors must be preserved within those products. Its [implementation](https://github.com/NVlabs/GatedDeltaNet-2) uses a WY-style representation, triangular solves, and matrix operations, together with a backward pass that accounts for the vector gates. Moving vector gates outside products as though they were scalars would generally change the computation.

The persistent memory per recurrent head has `d_v × d_k` entries. At fixed dimensions, recurrent sequence mixing scales linearly with token count. Training still stores activations, and hybrid architectures can add an attention cache. A fixed-size recurrent state therefore describes one part of the system's memory use.

## Reading the experimental evidence

The [Gated DeltaNet-2 report](https://arxiv.org/html/2605.22791v1) compares nominal 1.3B-parameter models trained on 100B FineWeb-Edu tokens with 4K training sequences. Its evaluations cover language modeling, common-sense reasoning, and retrieval. Recurrent-only and hybrid models are separate settings; the hybrids include sliding-window attention.

I would read these results alongside the tested context lengths, recurrent-state sizes, and throughput setup. Better retrieval in the reported evaluations supports the memory-editing design under those conditions. It does not establish lossless storage of an arbitrary history or performance on a different modality.

For my own comparisons, I would check whether a benefit comes from the edit mechanism, additional projections, a different state size, or a changed training recipe. A fair runtime comparison should also distinguish training, prompt processing, and one-token decoding. These questions help connect the mathematics of an update to the behavior of a complete model.

Readers can inspect the [original Gated DeltaNet paper](https://openreview.net/forum?id=r8H7xhYPwz) and the [Gated DeltaNet-2 report](https://arxiv.org/html/2605.22791v1) for the complete derivations and experiments. The core progression is a useful lens for reading them: adaptive forgetting, targeted correction, and increasingly independent control over the contents of recurrent memory.
