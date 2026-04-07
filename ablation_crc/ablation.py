"""
ablation.py
===========
Script principal — orchestre les 6 expériences de l'étude ablation.

Usage :
    # Toutes les variantes
    python ablation.py

    # Une seule variante
    python ablation.py --variants V2

    # Plusieurs variantes
    python ablation.py --variants V1 V2 V6

    # Reprendre après interruption (skip si checkpoint existe)
    python ablation.py --resume

    # GPU multi
    python ablation.py --device cuda:0

Sorties :
    checkpoints/<variant>/best_model.pt
    results/<variant>/metrics.json
    results/<variant>/confusion_matrix.png
    results/<variant>/f1_per_class.png
    results/training_curves/<variant>_curves.png
    results/ablation_summary.json
    results/ablation_table.csv
    results/ablation_barplot.png
    results/ablation_f1_minority.png
"""

import argparse
import json
import csv
import time
from pathlib import Path

import torch

from dataset import get_dataloaders
from models import build_model, VARIANT_LABELS
from train import Trainer, DEFAULT_CONFIG
from evaluate import evaluate_variant
from utils import (
    plot_training_curves,
    plot_ablation_barplot,
    plot_f1_minority_comparison,
    print_ablation_table,
    save_ablation_csv,
)

# ----------------------------------------------------------------
# Configuration globale de l'expérience
# ----------------------------------------------------------------

EXP_CONFIG = {
    **DEFAULT_CONFIG,
    "epochs":            50,
    "patience":          10,
    "lr":                1e-4,
    "weight_decay":      1e-5,
    "alpha":             0.3,
    "focal_gamma":       2.0,
    "batch_size":        32,
    "seq_phase1_epochs": 20,
    "seq_phase2_epochs": 30,
    "verbose":           True,
    "save_dir":          "checkpoints",
}

RESULTS_DIR   = Path("results")
CKPT_DIR      = Path(EXP_CONFIG["save_dir"])
ALL_VARIANTS  = ["V1", "V2", "V3", "V4", "V5", "V6"]


# ----------------------------------------------------------------
# Entraînement d'une variante
# ----------------------------------------------------------------

def run_variant(variant: str, loaders: dict,
                config: dict, resume: bool = False) -> dict:
    """
    Entraîne la variante et évalue sur le test set.
    Si resume=True et le checkpoint existe, on passe l'entraînement.
    Retourne le dict de métriques.
    """
    ckpt_path = CKPT_DIR / variant / "best_model.pt"
    out_dir   = RESULTS_DIR / variant

    if resume and ckpt_path.exists():
        print(f"\n[ablation] {variant} : checkpoint trouvé, "
              f"skip entraînement.")
    else:
        # Construction du modèle
        model = build_model(variant, pretrained=True)

        # Entraînement
        trainer = Trainer(model, loaders, config)
        history = trainer.fit()

        # Sauvegarde des courbes d'entraînement
        curves_dir = RESULTS_DIR / "training_curves"
        curves_dir.mkdir(parents=True, exist_ok=True)
        plot_training_curves(
            history,
            save_path=curves_dir / f"{variant}_curves.png",
            variant=variant,
        )

        # Évaluation sur test set (le Trainer a déjà chargé le best ckpt)
        metrics = trainer.evaluate_test()

        # Évaluation complète (confusion matrix, figures)
        evaluate_variant(
            variant=variant,
            checkpoint=ckpt_path,
            loaders=loaders,
            out_dir=out_dir,
            device=config["device"],
        )
        return metrics

    # Si resume : charger les métriques existantes
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    else:
        # Pas de métriques sauvegardées → réévaluer
        metrics = evaluate_variant(
            variant=variant,
            checkpoint=ckpt_path,
            loaders=loaders,
            out_dir=out_dir,
            device=config["device"],
        )
        return metrics


# ----------------------------------------------------------------
# Orchestration complète
# ----------------------------------------------------------------

def run_ablation(variants: list, config: dict,
                 resume: bool = False):
    """
    Lance toutes les expériences séquentiellement.
    Sauvegarde un résumé global à la fin.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Préparation des données (une seule fois)
    print("[ablation] Chargement des données …")
    loaders = get_dataloaders(
        batch_size=config["batch_size"],
        num_workers=4,
    )

    all_results = []
    total_t0 = time.time()

    for variant in variants:
        t0 = time.time()
        print(f"\n{'#'*60}")
        print(f"# Variante {variant} : {VARIANT_LABELS[variant]}")
        print(f"{'#'*60}")

        metrics = run_variant(variant, loaders, config, resume=resume)
        metrics["label"] = VARIANT_LABELS[variant]
        all_results.append(metrics)

        elapsed = time.time() - t0
        print(f"[ablation] {variant} terminé en {elapsed/60:.1f} min")

    # ── Résumé global ──────────────────────────────────────────
    summary_path = RESULTS_DIR / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[ablation] Résumé sauvegardé : {summary_path}")

    # ── Tableau console ────────────────────────────────────────
    print_ablation_table(all_results)

    # ── CSV ────────────────────────────────────────────────────
    csv_path = RESULTS_DIR / "ablation_table.csv"
    save_ablation_csv(all_results, csv_path)
    print(f"[ablation] CSV sauvegardé : {csv_path}")

    # ── Figures de comparaison ─────────────────────────────────
    plot_ablation_barplot(
        all_results,
        save_path=RESULTS_DIR / "ablation_barplot.png",
    )
    plot_f1_minority_comparison(
        all_results,
        save_path=RESULTS_DIR / "ablation_f1_minority.png",
    )

    total_elapsed = time.time() - total_t0
    print(f"\n[ablation] Étude complète terminée en "
          f"{total_elapsed/60:.1f} min.")
    print(f"[ablation] Résultats dans : {RESULTS_DIR}/")

    return all_results


# ----------------------------------------------------------------
# CLI
# ----------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Étude ablation Multi-Head CNN — Kather 2016")
    p.add_argument(
        "--variants", nargs="+",
        choices=ALL_VARIANTS, default=ALL_VARIANTS,
        help="Variantes à entraîner (défaut : toutes)")
    p.add_argument(
        "--resume", action="store_true",
        help="Reprendre depuis les checkpoints existants")
    p.add_argument(
        "--device", default=None,
        help="Device PyTorch (défaut : cuda si disponible, sinon cpu)")
    p.add_argument(
        "--epochs", type=int, default=None,
        help="Override du nombre max d'epochs")
    p.add_argument(
        "--batch_size", type=int, default=None,
        help="Override de la taille de batch")
    p.add_argument(
        "--alpha", type=float, default=None,
        help="Override de alpha pour la combined loss")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = dict(EXP_CONFIG)
    if args.device:
        config["device"] = args.device
    if args.epochs:
        config["epochs"] = args.epochs
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.alpha is not None:
        config["alpha"] = args.alpha

    print("[ablation] Configuration :")
    for k, v in config.items():
        print(f"  {k:25s} = {v}")

    results = run_ablation(
        variants=args.variants,
        config=config,
        resume=args.resume,
    )
