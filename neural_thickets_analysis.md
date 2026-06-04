# Neural Thickets & Gradient Subspace Search: Deep Technical Analysis

> **Papers covered:**
> 1. *Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights* — Gan & Isola, MIT CSAIL, 2026
> 2. *Gradient Subspace Search: Compute-Efficient Post-Training via Low-Dimensional Weight Perturbation* — Follow-up proposal to Neural Thickets

---

## Table of Contents

1. [Neural Thickets — Executive Summary](#1-neural-thickets--executive-summary)
2. [Main Contributions](#2-main-contributions)
3. [Notation Table](#3-notation-table)
4. [Key Ideas Dependency Graph](#4-key-ideas-dependency-graph)
5. [The Three Most Important Equations](#5-the-three-most-important-equations)
6. [The Problem Being Solved](#6-the-problem-being-solved)
7. [Assumptions](#7-assumptions)
8. [Core Findings](#8-core-findings)
9. [The RandOpt Algorithm](#9-the-randopt-algorithm)
10. [Experimental Setup & Results](#10-experimental-setup--results)
11. [Types of Thickets](#11-types-of-thickets)
12. [Limitations & Open Questions](#12-limitations--open-questions)
13. [Most Likely Points of Confusion](#13-most-likely-points-of-confusion)
14. [Critical Reviewer Questions](#14-critical-reviewer-questions)
15. [Gradient Subspace Search — Overview](#15-gradient-subspace-search--overview)
16. [How Both Papers Relate](#16-how-both-papers-relate)
17. [Subspace RandOpt — The Proposed Method](#17-subspace-randopt--the-proposed-method)
18. [Pilot Experimental Results](#18-pilot-experimental-results)
19. [Critical Tensions Between the Papers](#19-critical-tensions-between-the-papers)
20. [Summary Assessment](#20-summary-assessment)

---

## 1. Neural Thickets — Executive Summary

Large pretrained language models, when viewed through the lens of their weight space, are surrounded by a dense "thicket" of task-specific solutions — a property that **scales with model size**. Small models live in a *needle in a haystack* regime where gradient-based search is necessary, but large pretrained models have so many good weight perturbations nearby that simple random sampling suffices.

The authors quantify this with two measures:
- **Solution density** δ(m): what fraction of random perturbations improve task performance by at least margin m
- **Spectral discordance** 𝒟: how specialized/diverse those perturbations are across tasks

Exploiting both properties, they propose **RandOpt**: sample N random Gaussian weight perturbations, evaluate each on a small training set, keep the top K, and ensemble their predictions via majority vote — **no gradient steps required**. Despite its simplicity, RandOpt is competitive with PPO, GRPO, and Evolution Strategies across math, coding, writing, and chemistry benchmarks, while being O(1) in training steps and fully parallelizable.

---

## 2. Main Contributions

| # | Contribution | Location |
|---|---|---|
| C1 | Solution density around pretrained weights scales monotonically with model size | §2.1, Fig 3a |
| C2 | Solution diversity (spectral discordance) also scales with model size | §2.2, Fig 3b |
| C3 | Minimal mechanistic model: thickets emerge from diverse pretraining (1D signals) | §3, Fig 5 |
| C4 | RandOpt: gradient-free, fully parallel post-training via random guessing + ensembling | §4, Alg 1 |
| C5 | RandOpt matches/exceeds PPO/GRPO/ES under equal training FLOPs | §5, Fig 6, Table 4 |
| C6 | Distillation of top-K ensemble into a single model to reduce inference cost | §7, Table 2 |
| C7 | Taxonomy of thicket types: reasoning, format, color thickets | §8, §J |

---

## 3. Notation Table

| Symbol | Type | Meaning |
|---|---|---|
| θ ∈ ℝᵈ | Vector | Pretrained model weight vector |
| ε | Random vector | Gaussian noise perturbation |
| σ | Scalar | Noise scale controlling perturbation magnitude |
| Σ = {σ₁, …, σₘ} | Set | Set of noise scales explored |
| s : ℝᵈ → ℝ | Function | Performance evaluation metric (e.g., accuracy) |
| δ(m) | Scalar ∈ [0,1] | Solution density: probability a random perturbation improves score by margin m |
| N | Integer | Population size (number of random seeds sampled) |
| K | Integer | Ensemble size (top-K models selected) |
| **P** ∈ [0,1]^{N×M} | Matrix | Percentile-rank matrix: N seeds × M tasks |
| **C** ∈ ℝ^{M×M} | Matrix | Pearson correlation matrix of task performance rankings |
| 𝒟 | Scalar | Spectral Discordance: measures task specialist diversity |
| fθ : 𝒳 → 𝒴 | Function | The model being adapted |
| θ' = θ + σ·ε(s) | Vector | Perturbed model weights for seed s |
| vᵢ | Scalar | Performance score of i-th perturbed model on D_train |
| ℐ_top | Index set | Indices of top-K performing perturbations |
| ŷ | Output | Final prediction via majority vote over ensemble |

---

## 4. Key Ideas Dependency Graph

```
PRETRAINING ON DIVERSE DATA
         │
         ▼
LARGE MODEL SCALE
         │
    ┌────┴────┐
    ▼         ▼
HIGH           HIGH
SOLUTION       SOLUTION
DENSITY        DIVERSITY
(δ(m) scales)  (𝒟 scales)
    │               │
    │   [thicket regime]
    │               │
    └────┬──────────┘
         ▼
   RANDOM SAMPLING IS
   SUFFICIENT TO FIND
   GOOD PERTURBATIONS
         │
         ▼
   SELECTED EXPERTS
   ARE SPECIALISTS
   (not generalists)
         │
         ▼
   ENSEMBLING COMPLEMENTS
   THEIR STRENGTHS
         │
         ▼
      RandOpt
  (N samples → top K → majority vote)
         │
    ┌────┴─────┐
    ▼          ▼
COMPETITIVE    O(1) WALL
WITH PPO/GRPO  CLOCK TIME
    │
    ▼
DISTILLATION
(K passes → 1 pass)
```

---

## 5. The Three Most Important Equations

### Equation 1 — Solution Density

$$\delta(m) = \mathbb{P}_{\epsilon \sim \mathcal{N}(0,\sigma^2 I)} \left[ s(\theta + \epsilon) \geq s(\theta) + m \right]$$

**Why it matters:** This is the paper's central measurable claim. It formalizes "how often does a random guess work?" as a probability. The empirical finding that δ(m) increases monotonically with model scale (Fig 3a) is the core empirical result underpinning everything else. Without this, RandOpt has no justification.

**Intuition:** Think of it as the "hit rate" of random guessing — out of all Gaussian perturbations of your weights, what fraction actually improve performance by at least m?

**Key finding:** For a 32B model on GSM8K, ~64% of random perturbations meet or beat baseline accuracy. For a 0.5B model: 0%.

---

### Equation 2 — Spectral Discordance

$$\mathcal{D} = 1 - \frac{1}{M(M-1)} \sum_{j \neq k} \mathbf{C}_{jk}$$

**Why it matters:** This measures whether different perturbations are all doing the same thing (generalists, 𝒟 → 0) or improving different tasks (specialists, 𝒟 → 1). If 𝒟 were near 0, ensembling would offer no benefit over just picking the best single perturbation. The finding that 𝒟 is high and grows with scale **justifies the ensembling step** of RandOpt.

**Bounds:** 𝒟 ∈ [0, M/(M−1)]. For M=7 tasks, the theoretical max is ≈1.17. Values near 1 mean the specialists are nearly orthogonal in their capabilities.

---

### Equation 3 — RandOpt Inference (Majority Vote)

$$\hat{y} = \text{mode}\left(\arg\max_y f_{\theta_i}(y|x) \;\Big|\; i \in \mathcal{I}_\text{top}\right)$$

**Why it matters:** This is the operational output — majority voting over K specialist models. The mode aggregation converts diverse specialists into a single robust prediction. The simplicity of this step (no learned aggregation, just counting) is both a strength and a limitation (doesn't generalize to open-ended outputs).

---

## 6. The Problem Being Solved

**Standard assumption (pre-2026):** The pretrained weight vector θ is a *starting point* for iterative gradient-based adaptation (fine-tuning, RLHF, PPO, GRPO).

**This paper's reframe:** The outcome of pretraining is better viewed as a **distribution over parameter vectors**, whose support already contains task-specific experts. The question is not how to optimize from θ — it's how to *sample* from this implicit distribution efficiently.

**The historical assumption being challenged:** Schmidhuber, Hochreiter & Bengio (2001) argued random guessing "cannot be viewed as a reasonable learning algorithm." The paper shows this is wrong *after* sufficient pretraining, because the density of solutions around the pretrained weights becomes high enough for random sampling to work.

---

## 7. Assumptions

| Assumption | Where it's made | How critical | What breaks if violated |
|---|---|---|---|
| Large enough pretrained model (≥1.5B) | §6, Fig 8 | Critical | RandOpt offers no improvement on GPT-2 0.1B or Qwen 0.5B |
| Gaussian perturbation structure (σ·ε, ε ~ N(0,I)) | §4, Eq 3 | Moderate | Different noise distributions might work equally well or better |
| Majority vote is a valid aggregation | §4, Eq 5 | Task-dependent | Fails for structured/open-ended outputs (stories, molecules) |
| Small training set D_train is representative | §4 | Moderate | Selection may overfit to unrepresentative examples |
| Fixed σ across model sizes for density measurement | §2.1 | Hidden/concerning | Larger models have more parameters, so same σ means different L₂ perturbation magnitude |
| Pretraining was on diverse data | §3 | Critical | Thickets only emerge when pretraining distribution is broad |

---

## 8. Core Findings

### Finding 1: Solution Density Scales with Model Size (Fig 3a)
On GSM8K, the fraction of random perturbations that match or beat baseline accuracy grows from **0% (0.5B) → 18% (1.5B) → 37% (3B) → 64% (32B)**. This is the "thicket" phenomenon.

### Finding 2: Solution Diversity (Spectral Discordance) Also Scales (Fig 3b)
Spectral discordance 𝒟 increases monotonically with model size across the Qwen2.5 family. Large models don't just have more good solutions — they have more *different* good solutions, each specializing in different tasks.

### Finding 3: Specialists, Not Generalists (Fig 4)
PCA of 7-dimensional performance vectors (across 7 tasks) shows **distinct clusters** — perturbations within a cluster share strengths (e.g., good at math, bad at chemistry) while different clusters offer complementary capabilities. The "performance spectra" are spiky, not flat.

### Finding 4: Thickets Require Diverse Pretraining (Fig 5, §3)
In a minimal 1D signal model, thickets only emerge when the base model is pretrained on **multiple signal types**. Pretraining on a single function type → plateau regime (already at ceiling). No pretraining → needle-in-haystack regime.

### Finding 5: RandOpt is Competitive (Fig 6, Table 4)
With K=50 and equal training FLOPs, RandOpt mostly matches or outperforms PPO, GRPO, and ES across 6 models (0.5B–8B) and 7 tasks. The ensemble size is critical: K=1 substantially underperforms K=50.

### Finding 6: Scaling of RandOpt (Fig 7, 8)
- Performance improves log-linearly with population size N
- Optimal selection ratio K/N decreases as N grows
- Thickets don't emerge until ~1.5B parameters
- RandOpt from scratch (no pretraining) stays near 0% across all scales

---

## 9. The RandOpt Algorithm

### Pseudocode (PyTorch-style)

```python
# Training: Select top-K seeds based on D_train performance
seeds = [sample_seed() for _ in range(N)]
sigmas_per_seed = [sigmas[i // (N // len(sigmas))] for i in range(N)]

# Evaluate all perturbed models
scores = [evaluate(theta + sigmas_per_seed[i] * eps(seed[i]), D_train)
          for i in range(N)]
top_indices = topk(scores, K).indices

# Inference: Ensemble predictions on test input x
answers = [generate(theta + sigmas_per_seed[i] * eps(seed[i]), x)
           for i in top_indices]
prediction = majority_vote(answers)
```

### Key Properties

| Property | RandOpt | GRPO/PPO | ES |
|---|---|---|---|
| Training steps | O(1) | O(T) | O(T) |
| Backpropagation | None | Required | None |
| Parallelism | Fully parallel | Sequential | Sequential |
| Inference cost | K forward passes | 1 pass | 1 pass |
| Communication | Scores once | Per-step gradients | Per-step scores |

### Distillation (§7)

To reduce inference cost from K passes to 1 pass, the top-50 models generate 25,000 responses on 500 training examples. A distilled model is trained via SFT on hard examples using:

$$\mathcal{L}_\text{Distill}(\theta) = -\sum_{t=T_x+1}^{T} \log p_\theta(s_t | x, s_{<t})$$

Cost of distillation is ~2% of the training cost. Results: Qwen2.5-3B-Instruct goes from 79.8% (base) → 84.3% (distill) vs. 87.1% (full RandOpt ensemble).

---

## 10. Experimental Setup & Results

### Models Tested
- Qwen2.5-Instruct: 0.5B, 1.5B, 3B
- OLMo3-7B (base and instruct) — chosen for open-source transparency, ruling out sandbagging
- Llama-3.1-8B-Instruct

### Benchmarks

| Benchmark | Domain | Type |
|---|---|---|
| Countdown | Math (symbolic reasoning) | Discrete integer |
| GSM8K | Math (grade school) | Discrete integer |
| MATH-500 | Math (competition) | Discrete |
| OlympiadBench | Math (Olympiad) | Discrete |
| MBPP | Code generation | Pass/fail |
| ROCStories | Creative writing | Ranking |
| USPTO-50k | Chemistry | Classification (1–10) |
| GQA | Visual reasoning (VLM) | Discrete |

### Selected Results (Table 4)

**Qwen2.5-1.5B-Instruct on GSM8K:**

| Method | Accuracy |
|---|---|
| Base | 58.8% |
| GRPO | 72.1% |
| ES | 71.7% |
| RandOpt (K=50) | **76.4%** |
| ES + TT-MV | 80.4% |

**Qwen2.5-3B-Instruct on GSM8K:**

| Method | Accuracy |
|---|---|
| Base | 79.8% |
| GRPO | 83.2% |
| ES | 85.8% |
| RandOpt (K=50) | **87.1%** |
| ES + TT-MV | 87.9% |

### VLM Result (Table 1)
Qwen2.5-VL-3B-Instruct on GQA: Base 56.6% → RandOpt **69.0%** (+12.4%)

### FLOPs Comparison (Appendix E)

| Method | FLOPs formula |
|---|---|
| GRPO | 8 · T · B · G · PL |
| PPO | 14 · T · B · G · PL |
| ES / RandOpt | 2 · T · N · D · PL |

RandOpt uses only forward passes (no backward), making it FLOP-efficient per unit of parallelism.

---

## 11. Types of Thickets

The paper identifies that thickets are not monolithic — different types of "improvements" coexist in the neighborhood:

### Reasoning Thicket
Base model answered incorrectly; perturbation finds the correct reasoning path. On GSM8K with Qwen2.5-3B: RandOpt K=50 contributes **12.3%** of gains from genuine reasoning improvement.

### Format Thicket
Base model had the right answer but formatted it incorrectly (e.g., not placing answer after `####`). RandOpt K=50 contributes **19.0%** of gains just from fixing format. This is also true for GRPO (~20.4% format gains).

### Color Thickets (Appendix J)
In diffusion models (Stable Diffusion XL), parameter regions preferentially generate images with specific color palettes. RandOpt with a "blue" or "yellow" target text reward steers generation toward desired color domains.

**Key implication:** The definition of "task expert" in this paper is broad — any perturbation that scores well on the benchmark, regardless of *why* it scores well.

---

## 12. Limitations & Open Questions

| Limitation | Details |
|---|---|
| Requires strong pretraining | RandOpt fails completely on models trained from scratch; thickets need ≥1.5B well-pretrained parameters |
| Inference cost | K forward passes at test time; distillation helps but adds complexity |
| Majority vote only for discrete outputs | Doesn't generalize cleanly to story generation, molecule design, etc. |
| Scaling appears to saturate | Both N scaling (Fig 7) and model scale scaling (Fig 10) show diminishing returns on a log scale |
| Cluster-scale compute | N=5000 on 200 GH200 GPUs; inaccessible to most practitioners |
| Mechanism unclear | The paper shows *that* thickets exist and scale, but not *why* pretraining creates them |
| Fixed σ across model sizes | Same noise scale σ=0.005 induces different L₂ perturbation norms in models of different dimensionality |
| Format thicket confound | A portion of all methods' gains (including GRPO) is from format fixing, not reasoning |

---

## 13. Most Likely Points of Confusion

**CF1: "RandOpt is just Best-of-N"**
Best-of-N operates in *output space* at test time — it samples different *answers* from the same model. RandOpt operates in *weight space* at training time — it selects different *models*. The selection is over parameter vectors, not generated responses.

**CF2: Why does ensembling help if we already selected the best perturbation?**
"Best" is measured on a small training set (200 examples). Individual top-1 models may overfit to that set's quirks. Diverse specialists complement each other's blind spots across the test distribution. This is why K=50 dramatically outperforms K=1 (Fig 11).

**CF3: What does "density scales with model size" mean exactly?**
The *fraction* of random perturbations that improve performance grows with model size — not just the absolute number. A 32B model has ~64% of perturbations beating baseline on GSM8K; a 0.5B model has 0% (Fig 12).

**CF4: Is the Gaussian neighborhood meaningful in high dimensions?**
The paper uses small σ=0.005, so perturbations live in a *local shell* around θ. The L₂ radius of a typical perturbation is σ√d, which grows with dimension — meaning the "neighborhood" is not comparably sized across model scales (a hidden assumption).

**CF5: Format thickets vs. reasoning thickets**
Some gains come merely from fixing output formatting, not from improved reasoning. Fig 9 decomposes this for RandOpt K=50: ~19% format gains, ~12% genuine reasoning gains, ~0.7% regression. This is also true for GRPO, so it's not unique to RandOpt — but it does mean accuracy numbers are partially a measure of formatting compliance.

---

## 14. Critical Reviewer Questions

**Q1: Is this "post-training" or just test-time compute?**
RandOpt requires evaluating N=5000 perturbed models on training data. The paper normalizes for FLOPs, but the *type* of compute differs (embarrassingly parallel forward passes vs. sequential gradient steps). The practical advantage depends entirely on cluster topology.

**Q2: How sensitive are results to σ?**
The paper uses σ ∈ {1, 2, 3} × 10⁻³ for main experiments. This is a critical hyperparameter — too small and no perturbation helps; too large and the model breaks. Sensitivity analysis across different model families is limited.

**Q3: Does the format thicket confound all results?**
If standard baselines were also evaluated with format-correcting post-processing, would RandOpt's relative advantage shrink substantially? The paper decomposes this for one model/task combination (Fig 9) but not systematically.

**Q4: Why does majority voting work for chemistry (USPTO) or writing (ROCStories)?**
For ROCStories (story ordering), the "answer" is a permutation — majority vote over orderings is non-trivial. The paper doesn't explain the voting implementation for non-integer outputs.

**Q5: Does solution density measure what we think it measures?**
δ(m) is measured with fixed σ=0.005 across all model sizes. But larger models have more parameters, so ‖ε‖₂ ≈ σ√d grows with d. The "neighborhood" being sampled is not comparably sized across scales — a 32B model's neighborhood is geometrically much larger than a 0.5B model's at the same σ.

---

## 15. Gradient Subspace Search — Overview

**Full title:** *Gradient Subspace Search: Compute-Efficient Post-Training via Low-Dimensional Weight Perturbation*

**Type:** Research proposal (not a completed paper) — a direct follow-up to Neural Thickets.

**Central research question:**
> Do the top-K weight perturbations found by RandOpt concentrate in the gradient subspace of the task reward — and if so, can searching directly in this subspace reduce N by 10×?

### Additional Notation (Proposal-specific)

| Symbol | Meaning |
|---|---|
| g = ∇θ𝒧(θ, D_loc) | Task gradient computed on small localization set |
| G | Reshaped gradient matrix |
| U, S, V | Thin SVD decomposition of G |
| Pᵣ = U_{:,:r} | Top-r left singular vectors (the subspace basis) |
| ρᵣ(Δ) | Subspace energy ratio for perturbation Δ |
| Δᵢ = θ'ᵢ − θ | Perturbation delta for the i-th candidate |
| zᵢ ~ N(0, Iᵣ) | Low-dimensional noise vector |
| r | Subspace rank (target: r ~ 100) |
| D_loc | Small localization set for gradient computation (~50 examples) |

---

## 16. How Both Papers Relate

### The Core Relationship: Problem → Gap → Proposed Fix

```
Neural Thickets (Gan & Isola, 2026)
├── FINDING: Large pretrained models are surrounded
│   by dense thickets of task-improving perturbations
├── ALGORITHM: RandOpt — sample N=5000 random
│   Gaussian perturbations, keep top-K, majority vote
├── RESULT: Competitive with PPO/GRPO/ES
└── BOTTLENECK: Needs N=5000 forward passes
            → requires 200 GH200 GPUs
            → inaccessible to most practitioners
                        │
                        ▼
Gradient Subspace Search (proposal)
└── QUESTION: Can we search SMARTER instead of MORE?
    └── HYPOTHESIS: Top-K perturbations concentrate
        in the low-dimensional gradient subspace
        → Search there with N=300-500 instead
        → One backward pass + ~500 forward passes
        → Runs on a SINGLE GPU
```

### Idea-by-Idea Correspondence

| Concept in Neural Thickets | How the Proposal Uses/Extends It |
|---|---|
| RandOpt: isotropic Gaussian search in ℝᵈ | Identified as the **inefficiency** — most energy goes to irrelevant directions |
| Solution density scales with model size | Motivates that good solutions exist; proposal asks *where* they are geometrically |
| Spectral discordance / specialist diversity | Proposal adds Hypothesis 5: verify subspace search preserves diversity 𝒟 |
| Low-dimensional curvature (Liang et al., 2026) | Neural Thickets mentions this as post-hoc; proposal promotes it to **core design principle** |
| N=5000 on 200 GH200 cluster | Identified as the **practical barrier** the proposal aims to remove |
| Morris et al.: 13 parameters for math reasoning | Used to support the claim that the effective subspace is very small |
| Log-linear N vs. accuracy scaling (Fig 7) | Implies diminishing returns → motivation for smarter search geometry |

### The Key Intellectual Leap

Neural Thickets asks: **"Do good perturbations exist nearby?"** — answers yes.

This proposal asks: **"Where exactly nearby do they live?"** — hypothesizes: in the **gradient subspace**.

---

## 17. Subspace RandOpt — The Proposed Method

### The Algorithm Change

**Neural Thickets — RandOpt (isotropic):**
```
θ'ᵢ = θ + σᵢ · ε(sᵢ),   ε(sᵢ) ~ N(0, Iᵈ)
```
Search is isotropic: equal probability in all d ~ 7×10⁹ directions.

**Gradient Subspace Search — Subspace RandOpt:**
```
θ'ᵢ = θ + σ · Pᵣ zᵢ,   zᵢ ~ N(0, Iᵣ),   r << d
```
Search is confined to an r-dimensional subspace defined by the task gradient.

### Subspace Energy Ratio (Core Diagnostic)

$$\rho_r(\Delta) = \frac{\|P_r P_r^\top \Delta\|^2}{\|\Delta\|^2}$$

This measures what fraction of a perturbation delta's energy lies within the gradient subspace. The hypothesis is that top-K perturbations have significantly higher ρᵣ than random perturbations.

### Sample Complexity Argument

To cover a fraction α of the relevant neighborhood, samples scale roughly with search dimensionality:

$$N_\text{subspace} \approx N_\text{full} \cdot \sqrt{r/d}$$

For r=100, d≈7×10⁹, N_full=5000:
- This gives N_subspace ≈ 19 (theoretical minimum)
- Proposal conservatively targets N ≈ 300–500 (10× reduction)

### Full Algorithm (Alg 2)

```
Require: θ, D_loc, D_train, r, N, K, σ

// Step 1: Subspace construction (1 backward pass)
g ← ∇θ𝒧(θ, D_loc)
U, S, V ← SVD(reshape(g))
Pᵣ ← U_{:,:r}

// Step 2: Random search in subspace
for i = 1, ..., N do
    zᵢ ~ N(0, Iᵣ)
    θ'ᵢ ← θ + σ · Pᵣ zᵢ
    vᵢ ← evaluate(θ'ᵢ, D_train)
end for
ℐ_top ← topK({vᵢ})

// Step 3: Inference (identical to vanilla RandOpt)
ŷ ← mode{ argmax_y f_{θ'ᵢ}(y|x) | i ∈ ℐ_top }
```

### Comparison Table

| Method | Search space | N | Backprop | Target acc. (GSM8K) |
|---|---|---|---|---|
| GRPO (200 steps) | — | — | Yes (200×) | ~82% |
| RandOpt (N=5000) | ℝᵈ | 5,000 | No | ~87% |
| RandOpt (N=500) | ℝᵈ | 500 | No | lower |
| **Subspace RandOpt** | Sᵣ ⊂ ℝᵈ | 300–500 | 1× only | ~87%? |

### Hypotheses

| # | Hypothesis |
|---|---|
| H2 | Top-K deltas have ρ̄⁺ᵣ ≥ 2·ρ̄⁻ᵣ for r ≤ 200 (gradient subspace alignment holds) |
| H3 | Subspace RandOpt (N=500) within 2% of RandOpt (N=5000) and beats RandOpt (N=500 isotropic) |
| H4 | Subspace Pᵣ computed on 50 examples is stable across random subsets |
| H5 | Spectral discordance 𝒟 of subspace perturbations is not significantly lower than full-space |

### Residual-Guided Search (Minor Extension, §3.3)

After building ensemble E₁, identify residual set R = {j : ŷ⁽¹⁾ⱼ ≠ yⱼ} (still-incorrect examples). Second round evaluates perturbations only on R:

$$v^{(2)}_i = \frac{1}{|R|} \sum_{j \in R} \mathbf{1}[f_{\theta'_i}(x_j) = y_j]$$

Reduces evaluation cost in Round 2 by factor |R|/|D_train|. Framed as stratified selection (not AdaBoost — no reweighting, no theoretical guarantees).

### Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Assumption fails (ρ̄⁺ ≈ ρ̄⁻) | Switch to empirical Fisher or PCA of top-K deltas; report null result |
| Subspace search reduces specialist diversity | Measure 𝒟 vs. r; report Pareto frontier between diversity and compute |
| Gradient unstable at small D_loc | Increase \|D_loc\|; ablate stability |
| Performance gap vs. full RandOpt is large | Compare at equal FLOPs, not equal N |

---

## 18. Pilot Experimental Results

Section 9 of the proposal contains preliminary results. The implementation differs from the clean §3 proposal:

| Aspect | Proposed (§3) | Implemented (§9) |
|---|---|---|
| Subspace construction | SVD of ∇θ𝒧 | SFT gradient + binary mask on top-K parameters |
| Perturbation | θ' = θ + σPᵣz | Noise only in masked (top-K) parameters |
| Updates | Pure selection, no gradient | Greedy updates every 50 samples |

### Training Results (Table 1, N=200)

| Checkpoint | Baseline Acc | Sparsity Acc | Sparsity (No Update) Acc |
|---|---|---|---|
| 100 | 15.00% | **15.50%** | 12.00% |
| 200 | 15.00% | **21.00%** | 12.00% |
| 300–500 | 15.00% | **25.00%** | 12.00% |
| 700 | 14.00% | **25.00%** | 12.00% |

Training accuracy: Sparsity method shows improvement (+10pp over baseline).

### Test Results — Critical Finding (Table 2)

At K=1 and K=5, the sparsity method **underperforms the baseline**:

| K | Baseline (K=50) | Sparsity (K=50) |
|---|---|---|
| 1 | 12.50% | 7.95–8.35% |
| 5 | 27.81–31.20% | 17.22–23.84% |
| 50 | 45.35–47.00% | 40.45–45.65% |

**The authors' own observation:** "Our method has much more overfitting than the baseline."

### Table 3 (50-sample training, Table 3)
More positive result: "Ours" reaches 36% vs. baseline's 26% at checkpoint 2000, with consistent gains throughout training.

### Figure 3 — 2D Perturbation Landscape
Five 2D perturbation landscape plots showing accuracy as a function of noise along two orthogonal directions. The landscapes are noisy/mottled, with best-performing points scattered — visually consistent with the Neural Thickets view that high-performing regions are not concentrated along any obvious axis.

---

## 19. Critical Tensions Between the Papers

### Tension 1: Diversity vs. Concentration

Neural Thickets' key insight is that **specialist diversity** (high spectral discordance 𝒟) is what makes ensembling valuable. Subspace RandOpt, by restricting search to a single gradient subspace, may inadvertently **collapse diversity**: all perturbations explore the same low-dimensional manifold, potentially producing correlated specialists rather than orthogonal ones.

The pilot data in §9 — where the sparsity method underperforms at low K — is consistent with exactly this failure mode.

### Tension 2: Gradient Direction vs. Good Perturbation Direction

The gradient ∇θ𝒧 points in the direction of steepest improvement under a first-order approximation. But RandOpt's power comes from finding perturbations that are *not* along the gradient — they are diverse random perturbations that happen to land in the thicket. The relationship between the gradient subspace and the geometry of the thicket is the core empirical question of Phase 1, and it is not at all obvious that they align.

### Tension 3: Random Success vs. Structured Success

Neural Thickets' philosophical claim is that the thicket regime makes gradient-based structure *unnecessary* for post-training. The proposal attempts to re-introduce gradient structure to make random search more efficient. If the proposal succeeds, it arguably partially rehabilitates gradient information — which is both useful practically and philosophically interesting for what it implies about the geometry of the thicket.

---

## 20. Summary Assessment

### Neural Thickets

| Dimension | Assessment |
|---|---|
| Novelty | High — reframes post-training as sampling from an implicit distribution |
| Empirical support | Strong across 6 models, 7 tasks, multiple model families |
| Practical impact | Requires 200 GH200 GPUs — high compute barrier |
| Theoretical depth | Descriptive (scaling laws) rather than explanatory (mechanism) |
| Limitations acknowledged | Yes, clearly in §11 |

### Gradient Subspace Search

| Dimension | Assessment |
|---|---|
| Motivation | Strong — the compute bottleneck of Neural Thickets is real |
| Core hypothesis | Plausible but unverified — no direct measurement of ρᵣ yet |
| Proposed method | Clean and well-specified in §3 |
| Pilot results | Concerning — overfitting and underperformance at low K |
| Risk of diversity collapse | High — not yet mitigated |
| Publishability if Phase 1 fails | Correctly noted as a publishable null result |

### The Big Picture

The two papers together tell a coherent story:

1. **Neural Thickets** establishes that the weight space neighborhood of large pretrained models is geometrically rich — dense with diverse specialists
2. **Gradient Subspace Search** asks whether this richness is *structured* — concentrated in directions the gradient already knows about — or *unstructured* — spread across the full high-dimensional weight space

If the gradient subspace alignment holds, it suggests the thicket has interpretable geometry and random search can be made practical. If it doesn't hold, it deepens the mystery of what makes certain regions of weight space so thicket-like, and points toward alternative geometric characterizations (Fisher information, Hessian eigenvectors, or data-driven PCA of top-K deltas from related tasks).

---

*Document compiled from: Gan & Isola (2026), arXiv:2603.12228v1, and the associated follow-up proposal "Gradient Subspace Search."*
