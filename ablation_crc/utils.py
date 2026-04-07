"""
utils.py
========
Visualisations et utilitaires pour l'étude ablation.

Fonctions exportées :
    plot_training_curves          — loss + F1 au cours des epochs
    plot_ablation_barplot         — comparaison Acc/F1 entre variantes
    plot_f1_minority_comparison   — F1 classes minoritaires par variante
    print_ablation_table          — tableau ASCII dans le terminal
    save_ablation_csv             — export CSV du tableau de résultats
    plot_roc_curves               — courbes ROC one-vs-rest (optionnel)
"""

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from models import VARIANT_LABELS

# ----------------------------------------------------------------
# Palette cohérente avec la communication LaTeX
# ----------------------------------------------------------------

PALETTE = {
    "V1": "#BAB9B4",   # gris clair (baseline)
    "V2": "#20808D",   # teal      (ours — dominant)
    "V3": "#1B474D",   # teal foncé
    "V4": "#A84B2F",   # terra/rust
    "V5": "#944454",   # mauve
    "V6": "#FFC553",   # gold
}

ALL_VARIANTS = ["V1", "V2", "V3", "V4", "V5", "V6"]


# ----------------------------------------------------------------
# 1. Courbes d'entraînement
# ----------------------------------------------------------------

def plot_training_curves(history: dict,
                         save_path: Path,
                         variant: str = ""):
    """
    Trace loss et val F1-macro au cours des epochs.
    """
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # ── Loss ──
    ax = axes[0]
    ax.plot(epochs, history["train_loss"],
            color=PALETTE.get(variant, "#20808D"),
            linewidth=1.8, label="Train loss")
    ax.plot(epochs, history["val_loss"],
            color=PALETTE.get(variant, "#20808D"),
            linewidth=1.8, linestyle="--", label="Val loss")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title(f"Training Loss — {variant}", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)

    # ── Val F1-macro ──
    ax = axes[1]
    ax.plot(epochs, history["val_f1_macro"],
            color=PALETTE.get(variant, "#20808D"),
            linewidth=1.8, label="Val F1-macro")
    ax.plot(epochs, history["val_f1_minority"],
            color="#A84B2F", linewidth=1.4,
            linestyle=":", label="Val F1-minority")
    best_ep = int(np.argmax(history["val_f1_macro"])) + 1
    best_f1 = max(history["val_f1_macro"])
    ax.axvline(best_ep, color="#1B474D", linewidth=1,
               linestyle="--", alpha=0.7,
               label=f"Best ep={best_ep} (F1={best_f1:.4f})")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("F1-score", fontsize=11)
    ax.set_title(f"Validation F1 — {variant}", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(f"Training curves — {variant} ({VARIANT_LABELS.get(variant,'')})",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[utils] Courbes sauvegardées : {save_path}")


# ----------------------------------------------------------------
# 2. Barplot de comparaison ablation
# ----------------------------------------------------------------

def plot_ablation_barplot(results: list,
                          save_path: Path,
                          metrics=("test_acc", "f1_macro")):
    """
    Barplot groupé : Accuracy et F1-macro pour chaque variante.
    results : liste de dicts issus de evaluate_variant / evaluate_test
    """
    variants = [r["variant"] for r in results]
    labels   = [VARIANT_LABELS.get(v, v) for v in variants]

    acc_vals = [r.get("test_acc",   r.get("accuracy", 0)) for r in results]
    f1_vals  = [r.get("f1_macro",   0) * 100               for r in results]
    f1m_vals = [r.get("f1_minority",0) * 100               for r in results]

    x      = np.arange(len(variants))
    width  = 0.26
    colors = [PALETTE.get(v, "#888") for v in variants]

    fig, ax = plt.subplots(figsize=(12, 5))

    bars1 = ax.bar(x - width, acc_vals,  width,
                   color=colors, alpha=0.90,
                   edgecolor="white", label="Accuracy (%)")
    bars2 = ax.bar(x,          f1_vals,  width,
                   color=colors, alpha=0.65,
                   edgecolor="white", label="F1-macro (%)")
    bars3 = ax.bar(x + width,  f1m_vals, width,
                   color=colors, alpha=0.40,
                   edgecolor="white", label="F1-minority (%)")

    # Annotations
    for bars in (bars1, bars2, bars3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + 0.3, f"{h:.1f}",
                    ha="center", va="bottom", fontsize=7.5)

    # Highlight V2
    v2_idx = variants.index("V2") if "V2" in variants else None
    if v2_idx is not None:
        ax.axvspan(v2_idx - 1.5*width, v2_idx + 1.5*width,
                   alpha=0.08, color="#20808D")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=9)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title("Ablation Study — Performance Comparison",
                 fontsize=13)
    ax.set_ylim(0, 105)
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=10, loc="lower right")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[utils] Barplot sauvegardé : {save_path}")


# ----------------------------------------------------------------
# 3. Comparaison F1 classes minoritaires
# ----------------------------------------------------------------

def plot_f1_minority_comparison(results: list, save_path: Path):
    """
    Barplot horizontal comparant le F1 des classes minoritaires
    (DEB + EMP) entre variantes.
    """
    variants = [r["variant"] for r in results]
    labels   = [f"{v}\n{VARIANT_LABELS.get(v,'')}" for v in variants]
    f1m_vals = [r.get("f1_minority", 0) for r in results]
    colors   = [PALETTE.get(v, "#888") for v in variants]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(labels, f1m_vals, color=colors,
                   edgecolor="white", height=0.55)

    for bar, val in zip(bars, f1m_vals):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)

    ax.set_xlabel("F1-score (moyenne DEB + EMP)", fontsize=11)
    ax.set_title("F1 des classes minoritaires par variante",
                 fontsize=12)
    ax.set_xlim(0, 1.0)
    ax.axvline(0.8, color="#A84B2F", linewidth=1.2,
               linestyle="--", alpha=0.7, label="Seuil 0.80")
    ax.legend(fontsize=9)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[utils] F1-minority comparison sauvegardé : {save_path}")


# ----------------------------------------------------------------
# 4. Tableau ASCII (terminal)
# ----------------------------------------------------------------

def print_ablation_table(results: list):
    """
    Affiche un tableau de résultats formaté dans le terminal.
    """
    header = (
        f"\n{'─'*75}\n"
        f"{'ID':<5} {'Variante':<30} {'Acc%':>7} "
        f"{'F1-macro':>10} {'F1-minor':>10} {'Params':>8}\n"
        f"{'─'*75}"
    )
    print(header)
    for r in results:
        v    = r.get("variant", "?")
        lbl  = VARIANT_LABELS.get(v, r.get("label", ""))
        acc  = r.get("test_acc",    r.get("accuracy", 0))
        f1   = r.get("f1_macro",    0)
        f1m  = r.get("f1_minority", 0)
        par  = r.get("params_M",    "?")
        best = " ◀" if v == "V2" else ""
        print(f"{v:<5} {lbl:<30} {acc:>7.2f} "
              f"{f1:>10.4f} {f1m:>10.4f} {str(par):>7}M{best}")
    print(f"{'─'*75}\n")


# ----------------------------------------------------------------
# 5. Export CSV
# ----------------------------------------------------------------

def save_ablation_csv(results: list, save_path: Path):
    """
    Exporte le tableau de résultats en CSV.
    """
    fieldnames = [
        "variant", "label",
        "test_acc", "f1_macro", "f1_weighted",
        "f1_minority", "params_M", "test_loss",
    ]
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames,
                                extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = dict(r)
            row.setdefault("label",    VARIANT_LABELS.get(r["variant"], ""))
            row.setdefault("f1_weighted", "")
            writer.writerow(row)
    print(f"[utils] CSV sauvegardé : {save_path}")


# ----------------------------------------------------------------
# 6. Résumé multi-variantes : grille de confusion matrices
# ----------------------------------------------------------------

def plot_confusion_grid(results_dir: Path,
                        variants: list,
                        save_path: Path):
    """
    Charge les matrices de confusion déjà générées (PNG) et les
    assemble en une grille 2×3 pour la publication.
    Nécessite matplotlib et Pillow.
    """
    from PIL import Image

    imgs = []
    for v in variants:
        p = results_dir / v / "confusion_matrix.png"
        if p.exists():
            imgs.append((v, Image.open(p)))

    if not imgs:
        print("[utils] Aucune confusion matrix trouvée.")
        return

    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 11))
    axes = axes.flatten()

    for idx, (v, img) in enumerate(imgs):
        axes[idx].imshow(np.array(img))
        axes[idx].set_title(f"{v} — {VARIANT_LABELS.get(v,'')}",
                            fontsize=11, pad=6)
        axes[idx].axis("off")

    # Masquer les axes en surplus
    for idx in range(len(imgs), len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Confusion Matrices — Ablation Study on Kather 2016",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[utils] Grille de confusion matrices sauvegardée : {save_path}")


# ----------------------------------------------------------------
# Test standalone
# ----------------------------------------------------------------

if __name__ == "__main__":
    # Génère des figures de test avec données synthétiques
    print("[utils] Test avec données synthétiques …")

    # Courbes d'entraînement factices
    np.random.seed(42)
    n_ep = 30
    history = {
        "train_loss":    list(np.linspace(1.2, 0.25, n_ep) + np.random.randn(n_ep)*0.04),
        "val_loss":      list(np.linspace(1.1, 0.31, n_ep) + np.random.randn(n_ep)*0.05),
        "val_acc":       list(np.linspace(0.60, 0.947, n_ep) + np.random.randn(n_ep)*0.01),
        "val_f1_macro":  list(np.linspace(0.55, 0.931, n_ep) + np.random.randn(n_ep)*0.01),
        "val_f1_minority": list(np.linspace(0.40, 0.872, n_ep) + np.random.randn(n_ep)*0.015),
    }
    plot_training_curves(history, Path("test_outputs/V2_curves.png"), "V2")

    # Résultats factices
    fake_results = [
        {"variant":"V1","test_acc":91.3,"f1_macro":0.880,"f1_minority":0.791,"params_M":23.5},
        {"variant":"V2","test_acc":94.7,"f1_macro":0.931,"f1_minority":0.872,"params_M":24.1},
        {"variant":"V3","test_acc":94.1,"f1_macro":0.921,"f1_minority":0.861,"params_M":24.8},
        {"variant":"V4","test_acc":93.8,"f1_macro":0.915,"f1_minority":0.853,"params_M":47.0},
        {"variant":"V5","test_acc":93.2,"f1_macro":0.908,"f1_minority":0.844,"params_M":24.1},
        {"variant":"V6","test_acc":94.3,"f1_macro":0.927,"f1_minority":0.869,"params_M":24.1},
    ]
    print_ablation_table(fake_results)
    plot_ablation_barplot(fake_results, Path("test_outputs/ablation_barplot.png"))
    plot_f1_minority_comparison(fake_results, Path("test_outputs/f1_minority.png"))
    save_ablation_csv(fake_results, Path("test_outputs/ablation_table.csv"))
    print("[utils] Test terminé. Figures dans test_outputs/")
