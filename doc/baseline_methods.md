# Baseline Defense Methods for Procrustes-MPC Comparison

> Last updated: 2026-06-14

## Overview

This document catalogs candidate baseline methods for comparing against our **Procrustes-MPC** (Procrustes Multimodal Perturbation Correction) defense. Methods are organized into three tiers by comparison priority, and grouped by defense paradigm.

Our method, Procrustes-MPC, is an **inference-time, training-free** defense that applies per-layer orthogonal Procrustes rotation in a low-dimensional SVD subspace to correct modality-induced perturbations in VLM hidden states, optionally augmented with mean-shift along the refusal direction.

---

## Tier 1: Activation / Representation Steering (Same Paradigm — Most Directly Comparable)

These methods operate on model internal representations at inference time, making them the closest methodological comparisons to Procrustes-MPC.

| Method | Paper | Venue | arXiv | GitHub | Training-Free | Key Mechanism | Selection Rationale |
|--------|-------|-------|-------|--------|:---:|---------------|---------------------|
| **ASTRA** | Steering Away from Harm: An Adaptive Approach to Defending Vision Language Model Against Jailbreaks | CVPR 2025 | [2411.16721](https://arxiv.org/abs/2411.16721) | [ASTRAL-Group/ASTRA](https://github.com/ASTRAL-Group/ASTRA) | Yes | Contrastive steering vector + adaptive projection | **Most direct competitor.** Same paradigm (inference-time activation steering for VLM), but uses image attribution + PGD to construct steering vectors and adaptive projection to control strength. Comparing reveals Procrustes subspace rotation vs. contrastive vector subtraction. Open-source, CVPR 2025 accepted. |
| **SafeSteer** | SafeSteer: Adaptive Subspace Steering for Efficient Jailbreak Defense in Vision-Language Models | arXiv 2025 | [2509.21400](https://arxiv.org/abs/2509.21400) | Not released | Yes | SVD safety subspace + vector purification + harm-aware classifier | **Closest technical design.** Also uses SVD to construct a low-dimensional safety subspace, but performs steering vector purification (projection & reconstruction) rather than Procrustes rotation. Comparing demonstrates the geometric advantage of orthogonal rotation over linear projection/purification. |
| **Refusal Direction (Abliteration)** | Refusal in Language Models Is Mediated by a Single Direction | NeurIPS 2024 | [2406.11717](https://arxiv.org/abs/2406.11717) | [andyrdt/refusal_direction](https://github.com/andyrdt/refusal_direction) | Yes | Single refusal direction addition/ablation in residual stream | **Natural ablation baseline.** Procrustes-MPC's mean-shift component is itself a generalization of single-direction steering. This baseline isolates the question: does full subspace rotation provide value beyond single-direction manipulation? Foundational work (NeurIPS 2024), widely reproduced. |
| **InferAligner** | InferAligner: Inference-Time Alignment for Harmlessness through Cross-Model Guidance | EMNLP 2024 | [2401.11206](https://arxiv.org/abs/2401.11206) | [Jihuai-wpy/InferAligner](https://github.com/Jihuai-wpy/InferAligner) | Yes | Safety steering vectors extracted from aligned model + guidance gate | **Cross-model vs. self-contained comparison.** Extracts safety steering vectors (SSVs) from a separate aligned model and injects them into the target model via a guidance gate. Procrustes-MPC is self-contained (no external model needed). Comparing reveals whether self-derived subspace rotation matches or exceeds cross-model transfer. EMNLP 2024, open-source. |

---

## Tier 2: Fine-Tuning-Based Defenses (Different Paradigm — Training Cost Comparison)

These methods require model weight modifications. Comparing against them demonstrates whether our training-free approach can match the safety performance of methods that modify model parameters.

| Method | Paper | Venue | arXiv | GitHub | Training-Free | Key Mechanism | Selection Rationale |
|--------|-------|-------|-------|--------|:---:|---------------|---------------------|
| **VLGuard** | Safety Fine-Tuning at (Almost) No Cost: A Baseline for Vision Large Language Models | ICML 2024 | [2402.02207](https://arxiv.org/abs/2402.02207) | [ys-zong/VLGuard](https://github.com/ys-zong/VLGuard) | No | Safety SFT with ~2k curated safety data | **Standard fine-tuning baseline.** The canonical VLM safety SFT method; used as a baseline in most VLM safety papers (MMJ-Bench, etc.). Comparing shows that Procrustes-MPC (training-free) can match or exceed fine-tuning-based safety with zero weight modification and zero training data. ICML 2024, open-source with released model weights. |
| **Circuit Breakers (RR)** | Improving Alignment and Robustness with Circuit Breakers | arXiv 2024 | [2406.04313](https://arxiv.org/abs/2406.04313) | [GraySwanAI/circuit-breakers](https://github.com/GraySwanAI/circuit-breakers) | No | Train model to reroute harmful representations to be orthogonal to original | **Representation-level fine-tuning counterpart.** Both methods operate on internal representations, but Circuit Breakers train the model to make harmful representations orthogonal (permanent weight change), while Procrustes-MPC applies orthogonal rotation at inference time (no weight change). This is the most informative "training vs. inference" comparison for representation-based methods. Open-source by GraySwanAI. |

---

## Tier 3: Prompt-Based / Generation-Level Defenses (Different Mechanism — Completeness)

These methods operate at the input or output level rather than on internal representations. They provide upper/lower bounds and highlight the advantages of representation-level intervention.

| Method | Paper | Venue | arXiv | GitHub | Training-Free | Key Mechanism | Selection Rationale |
|--------|-------|-------|-------|--------|:---:|---------------|---------------------|
| **AdaShield** | AdaShield: Safeguarding Multimodal Large Language Models from Structure-based Attack via Adaptive Shield Prompting | ECCV 2024 | [2403.09513](https://arxiv.org/abs/2403.09513) | [SaFo-Lab/AdaShield](https://github.com/SaFo-Lab/AdaShield) | Yes | Adaptive defense prompt prepended to input | **Lightest-weight baseline.** Pure prompt-level defense with zero parameter or representation modification. Serves as a lower bound for representation-level methods and validates that hidden-state intervention (Procrustes-MPC) provides genuine advantages over prompt engineering, especially against perturbation-based attacks. ECCV 2024, open-source. |
| **ECSO** | Eyes Closed, Safety On: Protecting Multimodal LLMs via Image-to-Text Transformation | ECCV 2024 | [2403.09572](https://arxiv.org/abs/2403.09572) | [gyhdog99/ECSO](https://github.com/gyhdog99/ECSO) | Yes | Detect unsafe input, replace image with text caption, re-generate | **"Avoid vision" baseline.** When unsafe content is detected, ECSO discards the image entirely and substitutes a caption. This sacrifices visual understanding for safety. Comparing demonstrates that Procrustes-MPC achieves safety without sacrificing visual grounding — a critical advantage for real-world deployment. ECCV 2024, open-source. |
| **ETA** | ETA: Evaluating Then Aligning Safety of Vision Language Models at Inference Time | ICLR 2025 | [2410.06625](https://arxiv.org/abs/2410.06625) | [DripNowhy/ETA](https://github.com/DripNowhy/ETA) | Yes | Two-phase: safety evaluation + interference prefix + best-of-N search | **Generation-level inference-time SOTA.** Operates at the generation level (prefix injection + sampling search) rather than the representation level. Comparing reveals whether direct hidden-state rotation (Procrustes-MPC) is more efficient and effective than generation-level search strategies. ICLR 2025, latest published SOTA, open-source. |

---

## Comparison Dimensions

The following dimensions should be evaluated for each baseline:

| Dimension | Metric | Evaluation Tool |
|-----------|--------|-----------------|
| **Safety (Cell A)** | Unsafe rate (%) on pure-text harmful queries | `judge_actionable_safety.py` |
| **Safety (Cell B)** | Unsafe rate (%) on text+image joint harm | `judge_actionable_safety.py` |
| **Safety (Cell C)** | Unsafe rate (%) on vision-only harm (SD-TYPO, MMSB) | `judge_actionable_safety.py` |
| **Safety (Cell D)** | Unsafe rate (%) on emergent/context harm (SIUO, MSSBench) | `judge_context_aware.py` |
| **Utility** | ScienceQA accuracy (%), MMStar accuracy (%), MathVista accuracy (%) | `score_scienceqa.py`, letter-match |
| **Over-Refusal** | Over-refusal rate (%) on MOSSBench, XSTest, benign queries | `judge_over_refusal.py` |
| **Efficiency** | Inference latency overhead (ms/sample), GPU memory overhead | Wall-clock timing |
| **Cost** | Training required? Training data needed? Extra model needed? | Qualitative |

---

## Recommended Experiment Priority

| Priority | Baselines | Justification |
|----------|-----------|---------------|
| **P0 (Must-have)** | No Defense (Vanilla), ASTRA, Refusal Direction | Minimal viable comparison: undefended baseline + closest competitor + ablation baseline |
| **P1 (Strongly recommended)** | SafeSteer, VLGuard, ETA | Complete the story: subspace peer, training-based peer, generation-level peer |
| **P2 (Supplementary)** | AdaShield, Circuit Breakers, InferAligner, ECSO | Full picture across all paradigms |

---

## Reference Summary

| # | Method | arXiv ID | Venue | Year |
|---|--------|----------|-------|------|
| 1 | ASTRA | 2411.16721 | CVPR 2025 | 2024 |
| 2 | SafeSteer | 2509.21400 | arXiv | 2025 |
| 3 | Refusal Direction | 2406.11717 | NeurIPS 2024 | 2024 |
| 4 | InferAligner | 2401.11206 | EMNLP 2024 | 2024 |
| 5 | VLGuard | 2402.02207 | ICML 2024 | 2024 |
| 6 | Circuit Breakers | 2406.04313 | arXiv | 2024 |
| 7 | AdaShield | 2403.09513 | ECCV 2024 | 2024 |
| 8 | ECSO | 2403.09572 | ECCV 2024 | 2024 |
| 9 | ETA | 2410.06625 | ICLR 2025 | 2024 |
