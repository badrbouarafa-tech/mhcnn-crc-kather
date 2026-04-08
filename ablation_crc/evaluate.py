"""
evaluate.py
===========
Évaluation complète d'un modèle entraîné sur le test set Kather 2016.

Produit :
    - Accuracy globale, macro-F1, weighted-F1
    - F1 par classe
    - F1 classes minoritaires (DEB, EMP)
    - Matrice de confusion (normalisée)
    - Rapport de classification complet (sklearn)

Usage standalone :
    python evaluate.py --variant V2 --checkpoint checkpoints/V2/best_model.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import (
    CLASS_NAMES, NUM_CLASSES, MINORITY_IDX, get_dataloaders
)
from models import build_model


# ----------------------------------------------------------------
# Inférence
# ----------------------------------------------------------------

@torch.no_grad()
def predict(model: nn.Module, loader, device: str = "cpu"):
    """
    Retourne (all_preds, all_targets) sur l'ensemble du loader.
    Utilise toujours les logits_l2 pour la prédiction fine-grained.
    """
    model.eval()
    model.to(device)
    all_preds, all_targets = [], []

    for imgs, _l1, l2 in loader:
        imgs = imgs.to(device, non_blocking=True)
        out  = model(imgs)
        preds = out["logits_l2"].argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(l2.numpy())

    return np.array(all_preds), np.array(all_targets)


# ----------------------------------------------------------------
# Calcul des métriques
# ----------------------------------------------------------------

def compute_metrics(preds: np.ndarray,
                    targets: np.ndarray) -> dict:
    """
    Retourne un dict avec toutes les métriques.
    """
    acc          = accuracy_score(targets, preds)
    f1_macro     = f1_score(targets, preds, average="macro",     zero_division=0)
    f1_weighted  = f1_score(targets, preds, average="weighted",  zero_division=0)
    f1_per_class = f1_score(targets, preds, average=None,        zero_division=0)
    f1_minority  = float(np.mean(f1_per_class[MINORITY_IDX]))

    report = classification_report(
        targets, preds,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    return {
        "accuracy":     round(float(acc) * 100, 2),
        "f1_macro":     round(float(f1_macro), 4),
        "f1_weighted":  round(float(f1_weighted), 4),
        "f1_minority":  round(f1_minority, 4),
        "f1_per_class": {
            CLASS_NAMES[i]: round(float(v), 4)
            for i, v in enumerate(f1_per_class)
        },
        "report":       report,
    }


# ----------------------------------------------------------------
# Matrice de confusion
# ----------------------------------------------------------------

def plot_confusion_matrix(preds: np.ndarray,
                          targets: np.ndarray,
                          save_path: Path,
                          variant: str = ""):
    """
    Sauvegarde une matrice de confusion normalisée.
    """
    cm = confusion_matrix(targets, preds, normalize="true")

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt=".2f",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cmap="Blues",
        vmin=0, vmax=1,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(
        f"Confusion Matrix (normalised) — {variant}",
        fontsize=13, pad=12)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] Matrice de confusion sauvegardée : {save_path}")


# ----------------------------------------------------------------
# Rapport F1 par classe (barplot)
# ----------------------------------------------------------------

def plot_f1_per_class(metrics: dict,
                      save_path: Path,
                      variant: str = ""):
    """
    Barplot du F1 par classe.
    """
    names  = list(metrics["f1_per_class"].keys())
    values = list(metrics["f1_per_class"].values())

    colors = [
        "#A84B2F" if n in ("DEB", "EMP") else "#20808D"
        for n in names
    ]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(names, values, color=colors, edgecolor="white", width=0.6)
    ax.axhline(metrics["f1_macro"], color="#1B474D",
               linewidth=1.5, linestyle="--",
               label=f"Macro-F1 = {metrics['f1_macro']:.4f}")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1-score", fontsize=11)
    ax.set_title(
        f"F1-score par classe — {variant}\n"
        f"Acc={metrics['accuracy']:.1f}%  "
        f"F1-macro={metrics['f1_macro']:.4f}  "
        f"F1-minority={metrics['f1_minority']:.4f}",
        fontsize=11)
    ax.legend(fontsize=10)

    # Annotations sur les barres
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + 0.01, f"{val:.3f}",
                ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] F1 par classe sauvegardé : {save_path}")


# ----------------------------------------------------------------
# Évaluation complète d'une variante
# ----------------------------------------------------------------

def evaluate_variant(variant: str,
                     checkpoint: Path,
                     loaders: dict,
                     out_dir: Path,
                     device: str = "cpu") -> dict:
    """
    Charge un checkpoint, prédit sur le test set, calcule et
    sauvegarde toutes les métriques.

    Retourne le dict de métriques (sans le rapport textuel).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Chargement du modèle
    model = build_model(variant, pretrained=False)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Prédictions
    preds, targets = predict(model, loaders["test"], device=device)

    # Métriques
    metrics = compute_metrics(preds, targets)

    # Rapport textuel
    print(f"\n{'='*60}")
    print(f"Résultats {variant} ({out_dir})")
    print(f"{'='*60}")
    print(metrics["report"])

    # Sauvegarde JSON (sans le rapport, trop verbeux pour JSON)
    metrics_json = {k: v for k, v in metrics.items() if k != "report"}
    metrics_json["variant"] = variant
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)

    # Figures
    plot_confusion_matrix(
        preds, targets,
        save_path=out_dir / "confusion_matrix.png",
        variant=variant)
    plot_f1_per_class(
        metrics,
        save_path=out_dir / "f1_per_class.png",
        variant=variant)

    return metrics_json


# ----------------------------------------------------------------
# CLI
# ----------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Évaluer une variante sur le test set Kather 2016")
    p.add_argument("--variant",    required=True,
                   choices=["V1","V2","V3","V4","V5","V6"])
    p.add_argument("--checkpoint", required=True,
                   help="Chemin vers best_model.pt")
    p.add_argument("--data_root",  default="data/kather2016")
    p.add_argument("--out_dir",    default=None,
                   help="Répertoire de sortie (défaut: results/<variant>)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available()
                                            else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    loaders = get_dataloaders(
        root=args.data_root, batch_size=args.batch_size)
    out_dir = args.out_dir or f"results/{args.variant}"
    metrics = evaluate_variant(
        variant=args.variant,
        checkpoint=Path(args.checkpoint),
        loaders=loaders,
        out_dir=Path(out_dir),
        device=args.device,
    )
    print(f"\n[evaluate] Résumé {args.variant} :")
    for k, v in metrics.items():
        if k not in ("f1_per_class", "variant"):
            print(f"  {k:20s} : {v}")
