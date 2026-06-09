"""
Generate all figures for the advisor report.
Run from the project root: python reports/generate_figures.py
Reads:  results/phase0.json (or phase0_checkpoint.json)
        results/phase1.json  (optional, for Phase 1 figures)
Writes: reports/figures/fig*.png
"""
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from pathlib import Path

# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":      "DejaVu Serif",
    "font.size":        12,
    "axes.labelsize":   13,
    "axes.titlesize":   14,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.dpi":       150,
})

BLUE   = "#4a90d9"
RED    = "#e84545"
GREEN  = "#2ecc71"
GRAY   = "#aaaaaa"
DARK   = "#2c3e50"

ROOT    = Path(__file__).parent.parent
RESULTS = ROOT / "results"
FIGS    = Path(__file__).parent / "figures"
FIGS.mkdir(exist_ok=True)


def load_phase0():
    for name in ("phase0.json", "phase0_checkpoint.json"):
        p = RESULTS / name
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            print(f"Loaded {p.name}")
            return data
    sys.exit("ERROR: no phase0 results found in results/")


def load_phase1():
    p = RESULTS / "phase1.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Figure 1 — Accuracy comparison bar chart
# --------------------------------------------------------------------------
def fig1_accuracy(p0):
    baseline = p0["baseline_acc"]
    mv_acc   = p0.get("mv_acc", None)

    fig, ax = plt.subplots(figsize=(6, 5))

    if mv_acc is not None:
        labels = ["Baseline\n(no perturbation)", "RandOpt\n(K=50, N=500)"]
        values = [baseline * 100, mv_acc * 100]
        colors = [BLUE, RED]
    else:
        labels = ["Baseline\n(no perturbation)"]
        values = [baseline * 100]
        colors = [BLUE]

    bars = ax.bar(labels, values, color=colors, width=0.45,
                  edgecolor=DARK, linewidth=0.8, zorder=3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.8,
                f"{val:.1f}%", ha="center", va="bottom",
                fontweight="bold", fontsize=13, color=DARK)

    if mv_acc is not None:
        delta = (mv_acc - baseline) * 100
        mid_y = (baseline + mv_acc) / 2 * 100
        ax.annotate(
            f"+{delta:.1f} pp",
            xy=(1, mv_acc * 100), xytext=(1.35, mid_y),
            fontsize=12, color="green", fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color="green", lw=1.5),
        )

    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy on GSM8K (%)")
    ax.set_title("Phase 0 Results — RandOpt vs Baseline\n"
                 "Qwen2.5-1.5B-Instruct · D_train=200 · D_test=200")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    plt.tight_layout()
    out = FIGS / "fig1_accuracy_comparison.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out.name}")


# --------------------------------------------------------------------------
# Figure 2 — Score distribution histogram
# --------------------------------------------------------------------------
def fig2_score_distribution(p0):
    all_scores = np.array(p0["all_scores"])
    top_scores = np.array(p0["top_scores"])
    threshold  = top_scores.min()
    top_mask   = all_scores >= threshold

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(all_scores[~top_mask], bins=40, color=BLUE, alpha=0.75,
            label=f"Non-selected  (n={int((~top_mask).sum())})", zorder=2)
    ax.hist(all_scores[top_mask], bins=15, color=RED, alpha=0.85,
            label=f"Top-K selected  (K=50)", zorder=3)
    ax.axvline(threshold, color=DARK, linestyle="--", linewidth=1.5,
               label=f"Selection threshold", zorder=4)

    ax.set_xlabel("Perturbation Score  (−loss, higher = better)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of 500 Random Perturbation Scores\n"
                 "Qwen2.5-1.5B-Instruct · GSM8K D_train (200 examples)")
    ax.legend(framealpha=0.9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)
    plt.tight_layout()
    out = FIGS / "fig2_score_distribution.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out.name}")


# --------------------------------------------------------------------------
# Figure 3 — RandOpt algorithm schematic
# --------------------------------------------------------------------------
def fig3_randopt_diagram():
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis("off")

    def box(x, y, w, h, label, sub="", color=BLUE):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.1",
            facecolor=color, edgecolor=DARK, linewidth=1.5, alpha=0.85, zorder=2
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + (0.15 if sub else 0), label,
                ha="center", va="center", fontsize=11, fontweight="bold",
                color="white", zorder=3)
        if sub:
            ax.text(x + w/2, y + h/2 - 0.25, sub,
                    ha="center", va="center", fontsize=9, color="white",
                    alpha=0.9, zorder=3)

    def arrow(x1, x2, y=2.0):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.8),
                    zorder=4)

    box(0.2,  1.25, 1.8, 1.5, "Pretrained\nWeights θ",     color=DARK)
    box(2.5,  1.25, 2.0, 1.5, "N=500 Random\nPerturbations", "ε ~ N(0, σ²I)", color=BLUE)
    box(5.1,  1.25, 2.0, 1.5, "Score on\nD_train",          "−loss proxy",    color=BLUE)
    box(7.7,  1.25, 1.8, 1.5, "Select\nTop-K=50",           "by score",       color="#e67e22")
    box(10.1, 1.25, 1.7, 1.5, "Majority\nVote",             "81.0% acc.",     color=RED)

    arrow(2.0,  2.5)
    arrow(4.5,  5.1)
    arrow(7.1,  7.7)
    arrow(9.5, 10.1)

    ax.text(6, 3.7, "Vanilla RandOpt  (Phase 0 — complete ✓)",
            ha="center", fontsize=13, fontweight="bold", color=DARK)

    plt.tight_layout()
    out = FIGS / "fig3_randopt_diagram.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out.name}")


# --------------------------------------------------------------------------
# Figure 4 — Research plan timeline
# --------------------------------------------------------------------------
def fig4_timeline():
    phases = [
        ("Phase 0\nReproduce RandOpt",       "1 week",   "✓ Done",   GREEN, True),
        ("Phase 1\nSubspace Alignment",       "2 weeks",  "In Progress", "#f39c12", False),
        ("Phase 2\nSubspace RandOpt",         "3 weeks",  "Pending",  GRAY,  False),
        ("Phase 3\nAblations",                "3 weeks",  "Pending",  GRAY,  False),
        ("Phase 4\nWrite-up",                 "3 weeks",  "Pending",  GRAY,  False),
    ]

    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.set_xlim(-0.5, len(phases) - 0.5)
    ax.set_ylim(-1, 3)
    ax.axis("off")

    for i, (title, duration, status, color, done) in enumerate(phases):
        rect = mpatches.FancyBboxPatch(
            (i - 0.42, 0.3), 0.84, 1.8,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor=DARK, linewidth=1.5, alpha=0.85
        )
        ax.add_patch(rect)
        ax.text(i, 1.55, title, ha="center", va="center",
                fontsize=10, fontweight="bold", color="white" if done else DARK)
        ax.text(i, 0.55, duration, ha="center", va="center",
                fontsize=9, color="white" if done else "#555")
        ax.text(i, 2.3, status, ha="center", va="center",
                fontsize=9, fontweight="bold",
                color=GREEN if done else ("#e67e22" if "Progress" in status else GRAY))

        if i < len(phases) - 1:
            ax.annotate("", xy=(i + 0.43, 1.2), xytext=(i + 0.43 + 0.14, 1.2),
                        arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.5))

    ax.text(2, 2.85, "Gradient Subspace Search — Research Timeline  (12 weeks total)",
            ha="center", fontsize=13, fontweight="bold", color=DARK)

    plt.tight_layout()
    out = FIGS / "fig4_timeline.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out.name}")


# --------------------------------------------------------------------------
# Figure 5 — Phase 1 subspace alignment concept
# --------------------------------------------------------------------------
def fig5_subspace_concept():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, title, highlight_center, center_color, label_top, label_bottom in [
        (axes[0],
         "Isotropic Search  (Vanilla RandOpt)",
         False, BLUE,
         "All directions equally likely",
         "Needs N=5,000 to find good solutions"),
        (axes[1],
         "Subspace Search  (Proposed)",
         True,  RED,
         "Search confined to gradient subspace S_r",
         "Hypothesis: N=300–500 sufficient"),
    ]:
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)

        # Weight space background
        circle_bg = plt.Circle((0, 0), 2.7, color="#ecf0f1", zorder=0)
        ax.add_patch(circle_bg)

        if highlight_center:
            # Gradient subspace direction
            theta = np.linspace(-np.pi/6, np.pi/6, 100)
            r_outer = 2.5
            xs = np.concatenate([[0], r_outer * np.cos(theta), [0]])
            ys = np.concatenate([[0], r_outer * np.sin(theta), [0]])
            wedge = plt.Polygon(list(zip(xs, ys)), color=RED, alpha=0.15, zorder=1)
            ax.add_patch(wedge)
            ax.annotate("", xy=(2.2, 0), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.5), zorder=3)
            ax.text(2.3, 0.25, "∇θ L", fontsize=11, color=RED, fontweight="bold")

        # Scatter good and bad perturbations
        rng = np.random.default_rng(7)
        n_pts = 60
        if highlight_center:
            angles_good = rng.uniform(-np.pi/5, np.pi/5, 18)
            r_good      = rng.uniform(0.5, 2.3, 18)
            angles_bad  = rng.uniform(np.pi/4, 2*np.pi - np.pi/4, n_pts - 18)
            r_bad       = rng.uniform(0.3, 2.5, n_pts - 18)
        else:
            angles_good = rng.uniform(0, 2*np.pi, 18)
            r_good      = rng.uniform(0.5, 2.3, 18)
            angles_bad  = rng.uniform(0, 2*np.pi, n_pts - 18)
            r_bad       = rng.uniform(0.3, 2.5, n_pts - 18)

        ax.scatter(r_bad  * np.cos(angles_bad),  r_bad  * np.sin(angles_bad),
                   s=30, color=BLUE, alpha=0.4, zorder=2, label="Poor perturbation")
        ax.scatter(r_good * np.cos(angles_good), r_good * np.sin(angles_good),
                   s=60, color=GREEN, alpha=0.9, zorder=3,
                   marker="*", label="Good perturbation")
        ax.scatter(0, 0, s=120, color=DARK, zorder=4, marker="D", label="θ (pretrained)")

        ax.text(0, -3.0, label_top,    ha="center", fontsize=9.5, color=DARK)
        ax.text(0, -3.55, label_bottom, ha="center", fontsize=9,  color="#666",
                style="italic")

        if ax is axes[0]:
            ax.legend(loc="upper left", fontsize=8, framealpha=0.8)

    fig.suptitle("Phase 1 Hypothesis: Good Perturbations Concentrate in Gradient Subspace",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = FIGS / "fig5_subspace_concept.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out.name}")


# --------------------------------------------------------------------------
# Figure 6 — Phase 1 results (only if phase1.json exists)
# --------------------------------------------------------------------------
def fig6_phase1_alignment(p1):
    if p1 is None:
        print("  Skipping fig6 (phase1.json not found yet)")
        return

    rbr = p1["results_by_rank"]
    ranks      = sorted(int(r) for r in rbr)
    rho_plus   = [rbr[str(r)]["mean_rho_plus"]  for r in ranks]
    rho_minus  = [rbr[str(r)]["mean_rho_minus"] for r in ranks]
    ratios     = [rbr[str(r)]["ratio"]           for r in ranks]
    spearmans  = [rbr[str(r)]["spearman_r"]      for r in ranks]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: ρ̄⁺ vs ρ̄⁻
    ax = axes[0]
    ax.plot(ranks, rho_plus,  "o-", color=RED,  linewidth=2, label="ρ̄⁺ (top-K deltas)",    markersize=7)
    ax.plot(ranks, rho_minus, "s--", color=BLUE, linewidth=2, label="ρ̄⁻ (random deltas)", markersize=7)
    ax.set_xlabel("Subspace rank  r")
    ax.set_ylabel("Mean subspace energy ratio  ρ̄")
    ax.set_title("Subspace Alignment: Top-K vs Random Deltas")
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_xscale("log")

    # Right: alignment ratio
    ax2 = axes[1]
    bar_colors = [GREEN if r >= 2 else "#e67e22" for r in ratios]
    ax2.bar([str(r) for r in ranks], ratios, color=bar_colors,
            edgecolor=DARK, linewidth=0.8)
    ax2.axhline(2.0, color=RED, linestyle="--", linewidth=1.5,
                label="Decision gate (2×)")
    ax2.set_xlabel("Subspace rank  r")
    ax2.set_ylabel("Alignment ratio  ρ̄⁺ / ρ̄⁻")
    ax2.set_title("Alignment Ratio by Rank\n(≥2× → proceed to Phase 2)")
    ax2.legend()
    ax2.yaxis.grid(True, linestyle="--", alpha=0.4)

    decision = p1.get("decision", "")
    color = GREEN if "PROCEED" in decision else "#e67e22"
    fig.suptitle(f"Phase 1 Result: {decision.replace('_', ' ')}",
                 fontsize=13, fontweight="bold", color=color)

    plt.tight_layout()
    out = FIGS / "fig6_phase1_alignment.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out.name}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating figures...")
    p0 = load_phase0()
    p1 = load_phase1()

    fig1_accuracy(p0)
    fig2_score_distribution(p0)
    fig3_randopt_diagram()
    fig4_timeline()
    fig5_subspace_concept()
    fig6_phase1_alignment(p1)

    print(f"\nAll figures saved to {FIGS}/")
