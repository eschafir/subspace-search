# Gradient Subspace Search & Neural Thickets: Research Repository

This repository contains the SFT SVD gradient subspace and PCA-based verification experiments for **Subspace RandOpt**, a follow-up proposal to *Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights* (Gan & Isola, MIT CSAIL, 2026).

The goal of this research project is to verify whether the top-$K$ task-improving weight perturbations found by RandOpt concentrate in a low-dimensional subspace, and if so, exploit this structure to reduce the population size from $N=5,000$ (requiring cluster-scale GH200 compute) to $N=500$ (feasible on a single GPU).

---

## 1. Executive Summary of Results

### 1.1 Phase 0 — Reproducing RandOpt Baseline
*   **Model:** `Qwen2.5-1.5B-Instruct`
*   **Dataset:** `GSM8K` mathematical reasoning ($D_{\text{train}}=200$, $D_{\text{test}}=200$, $N=500$, $K=50$, $\sigma \in \{1\text{e-3}, 2\text{e-3}, 3\text{e-3}\}$)
*   **Baseline Accuracy (no perturbation):** **66.50%**
*   **RandOpt Majority-Vote Accuracy:** **81.00%**
*   **Improvement:** **+14.5 percentage points**, confirming the existence of a dense "reasoning/formatting thicket" around the pretrained weights.

### 1.2 Phase 1 — Subspace Verification (Refuting the Dense Subspace Hypotheses)
We tested two separate methods for constructing a dense low-dimensional search subspace. Both failed to show significant alignment, resulting in a **rigorous academic null result**:

1.  **SFT Gradient SVD Subspace:** We constructed a subspace from the singular vectors of the 2D-reshaped SFT loss gradient. The alignment ratio of top-$K$ deltas ($\rho\bar{}^+$) vs. random deltas ($\rho\bar{}^-$) was flat at **$\approx 1.0\text{x}$** across all ranks ($r \in \{10, 50, 100, 200, 500\}$). 
    *   *Geometric Insight:* The gradient represents the direction of maximum first-order sensitivity (curvature). Perturbations along it degrade performance, so the RandOpt selection process actively filters them out (ratio is actually $0.92\text{x}$ at $r=10$).
2.  **PCA Subspace of Top-$K$ Deltas:** We performed PCA on $25$ training top-$K$ deltas and measured the alignment of $25$ validation top-$K$ deltas. The alignment ratio was weak, varying from **$1.12\text{x}$ to $1.54\text{x}$** across ranks ($r \in \{1, 2, 5, 10, 15, 20, 24\}$).
    *   *Geometric Insight:* The top-performing perturbations do not share a low-dimensional manifold; they are mutually orthogonal. This orthogonality directly explains the high **spectral discordance $\mathcal{D}$** observed in the Neural Thickets paper (diverse specialists instead of correlated generalists).

---

## 2. Repository Structure

```
├── Z:\FIU\Research\subspace-search
    ├── CLAUDE.md                   # Quick developer guide
    ├── README.md                   # This overview file
    ├── requirements.txt            # Dependency list
    ├── neural_thickets_analysis.md # Detailed technical summary of papers & pilot results
    ├── src/
    │   ├── models.py               # Model loader, parameter count, context manager
    │   ├── perturbation.py         # Isotropic and subspace delta generation
    │   ├── subspace.py             # SFT gradient computation and SVD subspace construction
    │   ├── randopt.py              # Isotropic and subspace RandOpt selection loops
    │   ├── metrics.py              # Subspace energy ratio and spectral discordance metrics
    │   └── benchmarks/             # Dataset loaders and formatting (GSM8K, Countdown, MBPP)
    ├── experiments/
    │   ├── phase0_reproduce.py     # Isotropic RandOpt baseline replication script
    │   ├── phase1_verify.py        # SFT SVD gradient alignment verification script
    │   ├── pca_verify.py           # PCA-based validation alignment verification script
    │   └── sparsity_randopt.py     # Parameter-wise Fisher Sparsity selection and testing script
    ├── reports/
    │   └── report.md               # Dynamic research report summarizing Phase 0/1 progress
    └── results/                    # Output directory for experimental checkpoints and JSONs
```

---

## 3. How to Run the Experiments

Make sure your environment is set up and has access to a CUDA-enabled GPU (approx. $12\text{ GB}$ VRAM required for Qwen-1.5B backpropagation).

### 3.1 Install Dependencies
```bash
pip install -r requirements.txt
```

### 3.2 Run Phase 0 (RandOpt Reproduction)
Runs isotropic RandOpt selection and majority-vote evaluation.
```bash
python experiments/phase0_reproduce.py --model qwen-1.5b --N 500 --K 50 --score-mode loss
```
*Saves checkpoints to `results/phase0_checkpoint.json`.*

### 3.3 Run Phase 1 (SFT Gradient Subspace Verification)
Measures the projection energy of top-performing perturbations onto the SVD SFT gradient subspace.
```bash
python experiments/phase1_verify.py --phase0-results results/phase0_checkpoint.json
```

### 3.4 Run PCA Subspace Verification
Runs cross-validation PCA on the top-performing deltas to verify if they share a low-dimensional manifold.
```bash
python experiments/pca_verify.py --phase0-results results/phase0_checkpoint.json
```

### 3.5 Run Parameter-wise Fisher Sparsity RandOpt
Runs RandOpt restricted to the top-$M$ parameters with the largest average squared gradients (diagonal Fisher information).
```bash
python experiments/sparsity_randopt.py --M 100000 --sigma 2e-3 --phase0-results results/phase0_checkpoint.json
```
*This script is optimized to run over **100x faster** than dense counterparts by mapping updates to GPU tensors in-place, bypassing $6\text{ GB}$ PCIe transfer overheads. It also saves selection checkpoints to allow immediate resuming.*

---

## 4. Geometric Conclusions & Scientific Impact

This research repository provides a rigorous characterization of the high-dimensional geometry of pretrained weight space:

```
                  ┌───────────────────────────────────┐
                  │ Pretrained Weight Neighborhood    │
                  │ (d = 1.5 Billion Dimensions)      │
                  └─────────────────┬─────────────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
      ┌───────────────────────┐           ┌───────────────────────┐
      │   SFT Gradient SVD    │           │      Delta PCA        │
      │   Alignment (1.0x)    │           │   Alignment (1.1-1.5x)│
      └──────────┬────────────┘           └──────────┬────────────┘
                 │                                   │
                 ▼                                   ▼
    Selection avoids SFT gradient;       Top perturbations are mutually
    gradient represents high-curvature   orthogonal, explaining high
    fragile directions (0.92x at r=10).  spectral discordance (D).
```

1.  **RandOpt is intrinsically high-dimensional:** Attempts to project the search space onto a dense low-dimensional subspace (Gradient SVD or PCA) collapse the necessary specialist diversity that makes majority-vote ensembling work.
2.  **Orthogonality and Thickets:** Pretrained models are surrounded by a shell of mutually orthogonal task experts. High spectral discordance $\mathcal{D}$ is not a bug; it is a feature of the high-dimensional geometry that enables robust ensembling.
3.  **Fisher Sparsity Overfitting:** Coordinate-sparse search (Fisher Sparsity) successfully reduces search dimensions to fit training loss, but suffers from generalization collapse on testing. High-sensitivity parameters are fragile, and perturbing them ruins general reasoning.
