"""
evaluate.py
-----------
Evaluation and visualization — always at the coarse OBAFGKM level.

models.py guarantees that every ClassificationResult already has coarse
y_true / y_pred before reaching here. This file never sees subclass labels.

Two evaluation passes:
    Pass 1 — Full OBAFGKM (all 7 classes)
        Summary table, confusion matrix, per-class F1, feature importance,
        HR diagram.

    Pass 2 — FGKM only (O/B/A filtered out)
        Same metrics but restricted to the hard photometric region.

CSV export:
    predictions_all_models.csv — one row per test star, true coarse class,
    and each model's predicted coarse class.

Usage (via Star_Classification.py):
    from evaluate import run_evaluation
    run_evaluation(df, results, out_dir=args.out)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score
)

from models import ClassificationResult, SPECTRAL_CLASSES, COARSE_CLASSES

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CLASS_COLORS = {
    "O": "#6A8EF0",
    "B": "#8FB8ED",
    "A": "#FFFFFF",
    "F": "#FFFFD0",
    "G": "#FFE87C",
    "K": "#FFA040",
    "M": "#FF4500",
}

# Figures are written to a "figures" folder next to this script,
# regardless of the current working directory you run from.
FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def _ensure_figure_dir():
    os.makedirs(FIGURE_DIR, exist_ok=True)


def _save(fig, filename: str) -> None:
    _ensure_figure_dir()
    path = os.path.join(FIGURE_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"      Saved: {path}")


def _safe(s: str) -> str:
    """Filename-safe version of a model name string."""
    return s.replace(" ", "_").replace("[", "").replace("]", "") \
            .replace("(", "").replace(")", "").replace(",", "").lower()


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    result: ClassificationResult,
    normalize: bool = True,
    ax=None,
    title: str = None,
) -> plt.Figure:
    """Heatmap of true vs predicted coarse class."""
    classes = result.classes
    cm = confusion_matrix(result.y_true, result.y_pred, labels=classes)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot  = np.where(row_sums == 0, 0, cm / row_sums)
        vmax     = 1.0
    else:
        cm_plot = cm
        vmax    = cm.max()

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 6))
        fig.patch.set_facecolor("#1a1a2e")
    else:
        fig = ax.get_figure()

    ax.set_facecolor("#1a1a2e")
    im = ax.imshow(cm_plot, interpolation="nearest",
                   cmap="Blues", vmin=0, vmax=vmax)

    if standalone:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    n  = len(classes)
    fs = max(8, 11 - n // 2)
    ax.set_xticks(range(n))
    ax.set_xticklabels(classes, fontsize=fs, color="white")
    ax.set_yticks(range(n))
    ax.set_yticklabels(classes, fontsize=fs, color="white")
    ax.tick_params(colors="white")

    thresh = cm_plot.max() / 2.0
    for i in range(n):
        for j in range(n):
            val = cm_plot[i, j]
            ax.text(j, i, f"{val:.0%}" if normalize else str(int(val)),
                    ha="center", va="center", fontsize=max(7, 10 - n // 2),
                    color="white" if val < thresh else "black")

    ax.set_xlabel("Predicted class", color="white", fontsize=11)
    ax.set_ylabel("True class",      color="white", fontsize=11)
    ax.set_title(title or result.model_name, color="white", fontsize=12, pad=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    if standalone:
        fig.tight_layout()
        _save(fig, f"confusion_{_safe(result.model_name)}.png")
        plt.show()

    return fig


def plot_all_confusion_matrices(
    results: dict[str, ClassificationResult],
    normalize: bool   = True,
    filename_suffix: str = "",
) -> plt.Figure:
    print("\n[Evaluate] Plotting confusion matrices...")
    n   = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5))
    fig.patch.set_facecolor("#1a1a2e")
    if n == 1:
        axes = [axes]

    for ax, (_, result) in zip(axes, results.items()):
        plot_confusion_matrix(result, normalize=normalize,
                              ax=ax, title=result.model_name)

    fig.suptitle("Confusion matrices — predicted vs true spectral class",
                 color="white", fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, f"confusion_all_models{filename_suffix}.png")
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Per-class F1
# ---------------------------------------------------------------------------

def plot_class_metrics(
    results: dict[str, ClassificationResult],
    classes: list[str]   = None,
    filename_suffix: str = "",
    title: str           = None,
) -> plt.Figure:
    """Bar chart of per-class F1 for each model, side by side."""
    if classes is None:
        classes = SPECTRAL_CLASSES

    print("\n[Evaluate] Plotting per-class F1 scores...")

    n_models  = len(results)
    n_classes = len(classes)

    f1_data = {
        name: f1_score(r.y_true, r.y_pred, labels=classes,
                       average=None, zero_division=0)
        for name, r in results.items()
    }

    fig, ax = plt.subplots(figsize=(max(9, 2.2 * n_classes), 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    x       = np.arange(n_classes)
    width   = 0.8 / n_models
    offsets = np.linspace(-(n_models-1)/2, (n_models-1)/2, n_models) * width
    colors  = ["#6A8EF0", "#50C878", "#FF8C42", "#E040FB"]

    for i, (name, result) in enumerate(results.items()):
        ax.bar(x + offsets[i], f1_data[name], width * 0.9,
               label=result.model_name,
               color=colors[i % len(colors)],
               alpha=0.85, edgecolor="#1a1a2e")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=12, color="white")
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_yticklabels([f"{v:.0%}" for v in np.arange(0, 1.1, 0.2)],
                       color="white")
    ax.tick_params(colors="white")
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("Spectral class", color="white", fontsize=11)
    ax.set_ylabel("F1 score",       color="white", fontsize=11)
    ax.set_title(title or "Per-class F1 score by model",
                 color="white", fontsize=13)
    ax.legend(fontsize=10, facecolor="#2a2a3e", labelcolor="white",
              edgecolor="#444", loc="lower right")
    ax.grid(axis="y", color="#333", linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    # Highlight F/G/K overlap zone
    if all(c in classes for c in ["F", "G", "K"]):
        fi, ki = classes.index("F"), classes.index("K")
        ax.axvspan(fi - 0.5, ki + 0.5, alpha=0.08, color="yellow")
        ax.text((fi + ki) / 2, 1.08, "F/G/K overlap zone",
                color="yellow", fontsize=9, ha="center", alpha=0.8)

    fig.tight_layout()
    _save(fig, f"per_class_f1{filename_suffix}.png")
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(result: ClassificationResult) -> plt.Figure | None:
    """Horizontal bar chart for tree-based models."""
    model = result.model
    if model is None:
        return None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "named_steps"):
        inner = list(model.named_steps.values())[-1]
        if not hasattr(inner, "feature_importances_"):
            print(f"  Skipping feature importance for {result.model_name} (MLP)")
            return None
        importances = inner.feature_importances_
    else:
        return None

    features = result.feature_names
    indices  = np.argsort(importances)
    color_indices = ["u_g", "g_r", "r_i", "i_z", "u_r", "g_i"]

    fig, ax = plt.subplots(figsize=(7, max(5, len(features) * 0.4)))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    bar_colors = ["#6A8EF0" if features[i] in color_indices
                  else ("#FFD700" if features[i] == "elodieTEff"
                  else "#50C878")
                  for i in indices]

    ax.barh(range(len(indices)), importances[indices],
            color=bar_colors, edgecolor="#1a1a2e", alpha=0.85)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([features[i] for i in indices], color="white", fontsize=10)
    ax.tick_params(colors="white")
    ax.set_xlabel("Importance", color="white", fontsize=11)
    ax.set_title(f"Feature importance — {result.model_name}",
                 color="white", fontsize=12)

    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor="#6A8EF0", label="Color indices"),
        Patch(facecolor="#50C878", label="Raw magnitudes"),
    ]
    if "elodieTEff" in features:
        legend_els.append(Patch(facecolor="#FFD700", label="T_eff (spectroscopic)"))
    ax.legend(handles=legend_els, facecolor="#2a2a3e",
              labelcolor="white", edgecolor="#444", fontsize=9)
    ax.grid(axis="x", color="#333", linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    fig.tight_layout()
    _save(fig, f"feature_importance_{_safe(result.model_name)}.png")
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# HR diagram
# ---------------------------------------------------------------------------

def plot_hr_diagram(
    df: pd.DataFrame,
    result: ClassificationResult,
    sample_n: int = 5000,
) -> plt.Figure:
    """Color-magnitude diagram, true vs predicted coarse class."""
    print("\n[Evaluate] Plotting HR diagram...")

    needed   = ["g_r", "r", "spectral_class"]
    df_clean = df.dropna(subset=needed)
    df_clean = df_clean[df_clean["spectral_class"].isin(SPECTRAL_CLASSES)]

    if result.X_test is not None:
        idx     = result.X_test.index
        df_plot = df_clean.loc[df_clean.index.isin(idx)].copy()
        # y_pred is already coarse — align by index
        pred_series = pd.Series(result.y_pred, index=idx)
        df_plot["predicted_class"] = pred_series.reindex(df_plot.index)
        df_plot = df_plot.dropna(subset=["predicted_class"])
    else:
        df_plot = df_clean.copy()
        df_plot["predicted_class"] = result.y_pred[:len(df_clean)]

    if len(df_plot) > sample_n:
        df_plot = df_plot.sample(sample_n, random_state=42)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#1a1a2e")

    for ax, col, ttl in zip(
        axes,
        ["spectral_class", "predicted_class"],
        ["True labels", f"Predicted — {result.model_name}"]
    ):
        ax.set_facecolor("#0d0d1a")
        for cls in SPECTRAL_CLASSES:
            mask = df_plot[col] == cls
            if not mask.any():
                continue
            ax.scatter(df_plot.loc[mask, "g_r"], df_plot.loc[mask, "r"],
                       c=CLASS_COLORS[cls], s=3, alpha=0.5,
                       label=cls, linewidths=0)
        ax.set_xlim(-0.5, 2.0)
        ax.set_ylim(23, 14)
        ax.set_xlabel("g − r  (blue ← hot    cool → red)",
                      color="white", fontsize=10)
        ax.set_ylabel("r magnitude  (bright at top)",
                      color="white", fontsize=10)
        ax.set_title(ttl, color="white", fontsize=12)
        ax.tick_params(colors="white")
        leg = ax.legend(title="Class", fontsize=9, markerscale=3,
                        facecolor="#2a2a3e", labelcolor="white",
                        edgecolor="#444", title_fontsize=9)
        leg.get_title().set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    fig.suptitle("Observational HR diagram — color vs magnitude",
                 color="white", fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, f"hr_diagram_{_safe(result.model_name)}.png")
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# K-class highlight
# ---------------------------------------------------------------------------

def plot_k_class_highlight(
    results: dict[str, ClassificationResult],
) -> plt.Figure:
    """
    Dedicated precision / recall / F1 spotlight for K-type stars.

    This is the project's headline result: ML photometric classification
    dramatically improves K-star identification over classical color cuts.

    Layout — two panels:
        Left:  grouped bar chart — precision, recall, F1 for K, per model
        Right: improvement over baseline shown as delta bars (±pp vs baseline)

    Why precision AND recall matter here:
        Precision — when the model says K, how often is it right?
                    High precision = reliable K flag for follow-up spectroscopy.
        Recall    — of all true K stars, how many did we catch?
                    High recall = we're not missing K stars in surveys.
        F1        — harmonic mean; the single number to quote in a paper.
    """
    print("\n[Evaluate] Plotting K-class highlight...")

    from sklearn.metrics import precision_recall_fscore_support

    model_names, precisions, recalls, f1s = [], [], [], []

    for name, result in results.items():
        p, r, f, _ = precision_recall_fscore_support(
            result.y_true, result.y_pred,
            labels=["K"], average=None, zero_division=0
        )
        model_names.append(result.model_name)
        precisions.append(float(p[0]))
        recalls.append(float(r[0]))
        f1s.append(float(f[0]))

    n       = len(model_names)
    x       = np.arange(n)
    bar_colors = ["#6A8EF0", "#50C878", "#FF8C42", "#E040FB"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor("#1a1a2e")

    # ---- Left panel: absolute metrics ----
    ax = axes[0]
    ax.set_facecolor("#1a1a2e")

    width   = 0.22
    offsets = [-width, 0, width]
    metrics = [precisions, recalls, f1s]
    mlabels = ["Precision", "Recall", "F1"]
    mcolors = ["#6A8EF0", "#50C878", "#FF8C42"]

    for off, vals, mlbl, mcol in zip(offsets, metrics, mlabels, mcolors):
        bars = ax.bar(x + off, vals, width * 0.9,
                      label=mlbl, color=mcol, alpha=0.85, edgecolor="#1a1a2e")
        # Annotate value on top of each bar
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + 0.015, f"{v:.0%}",
                    ha="center", va="bottom", fontsize=8, color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=9, color="white",
                       rotation=15, ha="right")
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_yticklabels([f"{v:.0%}" for v in np.arange(0, 1.1, 0.2)], color="white")
    ax.tick_params(colors="white")
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score", color="white", fontsize=11)
    ax.set_title("K-class: Precision / Recall / F1 per model",
                 color="white", fontsize=12)
    ax.legend(fontsize=10, facecolor="#2a2a3e", labelcolor="white",
              edgecolor="#444", loc="lower right")
    ax.grid(axis="y", color="#333", linewidth=0.5)
    ax.axhline(0.9, color="#FFA040", linewidth=1.2,
               linestyle="--", alpha=0.6, label="90% threshold")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    # ---- Right panel: delta vs baseline ----
    ax2 = axes[1]
    ax2.set_facecolor("#1a1a2e")

    # Baseline is first entry — use it as reference
    base_p, base_r, base_f = precisions[0], recalls[0], f1s[0]

    delta_p = [p - base_p for p in precisions[1:]]
    delta_r = [r - base_r for r in recalls[1:]]
    delta_f = [f - base_f for f in f1s[1:]]
    ml_names = model_names[1:]
    ml_colors = bar_colors[1:]

    x2 = np.arange(len(ml_names))

    for off, deltas, mlbl, mcol in zip(offsets, [delta_p, delta_r, delta_f],
                                        mlabels, mcolors):
        bars2 = ax2.bar(x2 + off, deltas, width * 0.9,
                        label=mlbl, color=mcol, alpha=0.85, edgecolor="#1a1a2e")
        for bar, v in zip(bars2, deltas):
            sign = "+" if v >= 0 else ""
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     v + (0.01 if v >= 0 else -0.03),
                     f"{sign}{v:.0%}",
                     ha="center", va="bottom", fontsize=8, color="white")

    ax2.axhline(0, color="white", linewidth=0.8, alpha=0.5)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(ml_names, fontsize=9, color="white",
                        rotation=15, ha="right")
    ax2.tick_params(colors="white")
    ax2.set_ylabel("Δ vs Baseline (color cuts)", color="white", fontsize=11)
    ax2.set_title("K-class: Improvement over baseline",
                  color="white", fontsize=12)
    ax2.legend(fontsize=10, facecolor="#2a2a3e", labelcolor="white",
               edgecolor="#444", loc="lower right")
    ax2.grid(axis="y", color="#333", linewidth=0.5)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#444")

    # Annotate baseline reference values in subtitle
    ax2.set_xlabel(
        f"Baseline reference — Precision: {base_p:.0%}  "
        f"Recall: {base_r:.0%}  F1: {base_f:.0%}",
        color="#aaa", fontsize=9
    )

    fig.suptitle(
        "K-type Star Classification  —  Photometric ML vs Classical Color Cuts",
        color="white", fontsize=13, y=1.02
    )
    fig.tight_layout()
    _save(fig, "k_class_highlight.png")
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(
    results: dict[str, ClassificationResult],
    label: str = "",
) -> None:
    hdr = f"  {label}" if label else ""
    print("\n" + "=" * 72)
    print(f"{'Model':<34} {'Weighted F1':>12} {'Precision':>10} {'Recall':>8}{hdr}")
    print("-" * 72)
    for _, result in results.items():
        f1  = f1_score(result.y_true, result.y_pred,
                       average="weighted", zero_division=0)
        pre = precision_score(result.y_true, result.y_pred,
                              average="weighted", zero_division=0)
        rec = recall_score(result.y_true, result.y_pred,
                           average="weighted", zero_division=0)
        cv  = (f"  cv:{result.cv_scores.mean():.3f}±{result.cv_scores.std():.3f}"
               if result.cv_scores is not None else "")
        print(f"{result.model_name:<34} {f1:>12.3f} {pre:>10.3f} {rec:>8.3f}{cv}")
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Predictions CSV
# ---------------------------------------------------------------------------

def save_predictions(
    df: pd.DataFrame,
    results: dict[str, ClassificationResult],
    out_dir: str = ".",
) -> None:
    """
    Write predictions_all_models.csv to out_dir.

    Columns:
        spectral_class       -- true coarse label
        spectral_subclass    -- true granular label (if available)
        g_r, u_g, r_i, i_z  -- key color indices
        pred_<model>         -- predicted coarse class per model
    """
    print("\n[Evaluate] Saving predictions CSV...")

    ref = next((r for r in results.values() if r.X_test is not None), None)
    if ref is None:
        print("      No ML test set found — skipping CSV.")
        return

    idx       = ref.X_test.index
    keep_cols = ["spectral_class", "spectral_subclass",
                 "g_r", "u_g", "r_i", "i_z"]

    # Defensive: only keep test labels that actually exist in df. With the
    # index-preserving split in models._prepare_data this should be all of
    # them, but the intersection means any future index change degrades to a
    # smaller CSV with a warning rather than a hard KeyError.
    valid_idx = idx.intersection(df.index)
    if len(valid_idx) < len(idx):
        print(f"      Note: {len(idx) - len(valid_idx):,} of {len(idx):,} test "
              f"rows not found in df index; writing the {len(valid_idx):,} that align.")
    out_df = df.loc[valid_idx, [c for c in keep_cols if c in df.columns]].copy()

    for name, result in results.items():
        col = f"pred_{name}"
        if result.X_test is not None:
            # y_pred is already coarse — align by index
            out_df[col] = pd.Series(result.y_pred,
                                    index=result.X_test.index).reindex(out_df.index)
        else:
            # Baseline — align by df position
            s = pd.Series(result.y_pred, index=df.index[:len(result.y_pred)])
            out_df[col] = s.reindex(out_df.index)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "predictions_all_models.csv")
    out_df.to_csv(path)
    print(f"      Saved  : {path}")
    print(f"      Rows   : {len(out_df):,}")
    print(f"      Columns: {list(out_df.columns)}\n")


# ---------------------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------------------

def run_evaluation(
    df: pd.DataFrame,
    results: dict[str, ClassificationResult],
    out_dir: str = FIGURE_DIR,
) -> None:
    """
    Two-pass coarse evaluation. All ClassificationResults are already coarse.

    Pass 1 — Full OBAFGKM
        Summary table, confusion matrix, per-class F1,
        feature importance (tree models), HR diagram (best ML model).

    Pass 2 — FGKM only (O/B/A rows filtered)
        Summary table, confusion matrix, per-class F1.
        This is the hard region — where model differences actually matter.

    CSV export: predictions_all_models.csv
    """

    # ------------------------------------------------------------------
    # Pass 1: Full OBAFGKM
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("PASS 1 — FULL OBAFGKM")
    print("=" * 72)

    print_summary_table(results, label="(all 7 classes)")
    plot_all_confusion_matrices(results, filename_suffix="_obafgkm")
    plot_class_metrics(results, classes=SPECTRAL_CLASSES,
                       filename_suffix="_obafgkm",
                       title="Per-class F1 — full OBAFGKM (all models)")

    for name, result in results.items():
        if name in ("random_forest", "xgboost"):
            plot_feature_importance(result)

    best_key = "xgboost" if "xgboost" in results else "random_forest"
    if best_key in results:
        plot_hr_diagram(df, results[best_key])

    # K-class headline result
    plot_k_class_highlight(results)

    # ------------------------------------------------------------------
    # Pass 2: FGKM only
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("PASS 2 — FGKM ONLY  (O/B/A filtered out)")
    print("=" * 72)
    print("  O/B/A are photometrically trivial — all models agree on them.")
    print("  This pass isolates the hard region where confusion matters.\n")

    fgkm_results = {}
    for name, result in results.items():
        mask = np.isin(result.y_true, COARSE_CLASSES)
        fgkm_results[name] = ClassificationResult(
            model_name = result.model_name + " [FGKM]",
            y_true     = result.y_true[mask],
            y_pred     = result.y_pred[mask],
            classes    = COARSE_CLASSES,
            cv_scores  = result.cv_scores,
        )

    print_summary_table(fgkm_results, label="(FGKM only)")
    plot_all_confusion_matrices(fgkm_results, filename_suffix="_fgkm")
    plot_class_metrics(fgkm_results, classes=COARSE_CLASSES,
                       filename_suffix="_fgkm",
                       title="Per-class F1 — FGKM only (all models)")

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------
    save_predictions(df, results, out_dir=out_dir)
