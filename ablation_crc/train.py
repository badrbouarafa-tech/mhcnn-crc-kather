"""
train.py
========
Boucle d'entraînement universelle pour les 6 variantes.

Gère :
    - Combined loss  : L = alpha*L1 + (1-alpha)*L2
    - Focal Loss     : pour V6
    - Entraînement séquentiel : pour V5 (2 phases)
    - Early stopping sur val macro-F1
    - Sauvegarde du meilleur checkpoint

Usage (via ablation.py) :
    from train import Trainer
    trainer = Trainer(model, loaders, config)
    history = trainer.fit()
"""

import os
import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score

from models import FocalLoss, count_parameters
from dataset import NUM_CLASSES, NUM_L1, MINORITY_IDX

# ----------------------------------------------------------------
# Configuration par défaut
# ----------------------------------------------------------------

DEFAULT_CONFIG = {
    # Optimisation
    "lr":           1e-4,
    "weight_decay": 1e-5,
    "epochs":       50,
    "batch_size":   32,

    # Loss
    "alpha":        0.3,    # pondération L = alpha*L1 + (1-alpha)*L2
    "focal_gamma":  2.0,    # pour V6

    # Early stopping
    "patience":     10,     # epochs sans amélioration du val macro-F1

    # Séquentiel (V5)
    "seq_phase1_epochs": 20,   # phase 1 : entraîner L1 head
    "seq_phase2_epochs": 30,   # phase 2 : entraîner L2 head (backbone gelé)

    # Divers
    "device":       "cuda" if torch.cuda.is_available() else "cpu",
    "save_dir":     "checkpoints",
    "verbose":      True,
}


# ----------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------

class Trainer:
    """
    Entraîne un modèle selon la variante détectée automatiquement.

    Args:
        model   : instance de nn.Module (V1–V6)
        loaders : dict {'train', 'val', 'test'} de DataLoader
        config  : dict de configuration (fusionne avec DEFAULT_CONFIG)
    """

    def __init__(self, model: nn.Module, loaders: dict, config: dict = None):
        cfg = {**DEFAULT_CONFIG, **(config or {})}
        self.cfg     = cfg
        self.model   = model.to(cfg["device"])
        self.loaders = loaders
        self.device  = cfg["device"]
        self.variant = getattr(model, "variant", "V?")

        # Répertoire de sauvegarde
        self.save_dir = Path(cfg["save_dir"]) / self.variant
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Fonctions de perte
        self.ce_l2 = nn.CrossEntropyLoss()
        self.ce_l1 = nn.CrossEntropyLoss()
        self.fl_l1 = FocalLoss(gamma=cfg["focal_gamma"])   # pour V6

        # Historique
        self.history = {
            "train_loss": [], "val_loss": [],
            "val_acc": [], "val_f1_macro": [], "val_f1_minority": [],
        }

    # ────────────────────────────────────────────────────────────
    # Calcul de la loss selon la variante
    # ────────────────────────────────────────────────────────────

    def _compute_loss(self, outputs: dict,
                      labels_l1: torch.Tensor,
                      labels_l2: torch.Tensor,
                      phase: int = 2) -> torch.Tensor:
        """
        Args:
            outputs  : dict de logits selon la variante
            labels_l1: labels binaires (L1)
            labels_l2: labels 8 classes (L2)
            phase    : 1 ou 2 (pour V5 entraînement séquentiel)
        """
        alpha = self.cfg["alpha"]

        # V1 — seulement L2
        if self.variant == "V1":
            return self.ce_l2(outputs["logits_l2"], labels_l2)

        # V2 — combined loss
        if self.variant == "V2":
            loss_l1 = self.ce_l1(outputs["logits_l1"], labels_l1)
            loss_l2 = self.ce_l2(outputs["logits_l2"], labels_l2)
            return alpha * loss_l1 + (1 - alpha) * loss_l2

        # V3 — 3 niveaux
        if self.variant == "V3":
            # Labels L2_mid : TUM+COM→0, STR+LYM→1, DEB+MUC+ADI+EMP→2
            l2_mid_map = torch.tensor(
                [0, 1, 0, 1, 2, 2, 2, 2], device=labels_l2.device)
            labels_l2_mid = l2_mid_map[labels_l2]
            loss_l1     = self.ce_l1(outputs["logits_l1"],     labels_l1)
            loss_l2_mid = self.ce_l2(outputs["logits_l2_mid"], labels_l2_mid)
            loss_l3     = self.ce_l2(outputs["logits_l2"],     labels_l2)
            return 0.2*loss_l1 + 0.3*loss_l2_mid + 0.5*loss_l3

        # V4 — combined, deux backbones
        if self.variant == "V4":
            loss_l1 = self.ce_l1(outputs["logits_l1"], labels_l1)
            loss_l2 = self.ce_l2(outputs["logits_l2"], labels_l2)
            return alpha * loss_l1 + (1 - alpha) * loss_l2

        # V5 — séquentiel : phase 1 → L1, phase 2 → L2
        if self.variant == "V5":
            if phase == 1:
                return self.ce_l1(outputs["logits_l1"], labels_l1)
            else:
                return self.ce_l2(outputs["logits_l2"], labels_l2)

        # V6 — focal sur L1 + CE sur L2
        if self.variant == "V6":
            loss_l1 = self.fl_l1(outputs["logits_l1"], labels_l1)
            loss_l2 = self.ce_l2(outputs["logits_l2"], labels_l2)
            return alpha * loss_l1 + (1 - alpha) * loss_l2

        raise ValueError(f"Variante inconnue : {self.variant}")

    # ────────────────────────────────────────────────────────────
    # Étape d'une epoch
    # ────────────────────────────────────────────────────────────

    def _run_epoch(self, loader, optimizer=None,
                   training: bool = True, phase: int = 2):
        """
        Retourne (mean_loss, accuracy, f1_macro, f1_minority).
        Si training=False (val/test), optimizer est ignoré.
        """
        self.model.train(training)
        total_loss = 0.0
        all_preds, all_targets = [], []

        with torch.set_grad_enabled(training):
            for imgs, l1, l2 in loader:
                imgs = imgs.to(self.device, non_blocking=True)
                l1   = l1.to(self.device)
                l2   = l2.to(self.device)

                outputs = self.model(imgs)
                loss    = self._compute_loss(outputs, l1, l2, phase=phase)

                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item() * imgs.size(0)

                # Prédictions L2 pour les métriques
                preds = outputs["logits_l2"].argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(l2.cpu().numpy())

        n = len(all_targets)
        mean_loss = total_loss / n
        acc       = sum(p == t for p, t in zip(all_preds, all_targets)) / n
        f1_macro  = f1_score(all_targets, all_preds,
                              average="macro", zero_division=0)
        f1_min    = f1_score(all_targets, all_preds,
                              labels=MINORITY_IDX,
                              average="macro", zero_division=0)
        return mean_loss, acc, f1_macro, f1_min

    # ────────────────────────────────────────────────────────────
    # Fit standard (V1, V2, V3, V4, V6)
    # ────────────────────────────────────────────────────────────

    def _fit_standard(self):
        cfg = self.cfg
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        scheduler = CosineAnnealingLR(
            optimizer, T_max=cfg["epochs"], eta_min=1e-6)

        best_f1   = -1.0
        no_improve = 0
        best_state = None

        for epoch in range(1, cfg["epochs"] + 1):
            t0 = time.time()

            tr_loss, tr_acc, tr_f1, _ = self._run_epoch(
                self.loaders["train"], optimizer, training=True)
            va_loss, va_acc, va_f1, va_f1min = self._run_epoch(
                self.loaders["val"], training=False)

            scheduler.step()

            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(va_loss)
            self.history["val_acc"].append(va_acc)
            self.history["val_f1_macro"].append(va_f1)
            self.history["val_f1_minority"].append(va_f1min)

            if cfg["verbose"]:
                elapsed = time.time() - t0
                print(f"  [{self.variant}] Ep {epoch:02d}/{cfg['epochs']} | "
                      f"loss {tr_loss:.4f}/{va_loss:.4f} | "
                      f"acc {va_acc*100:.1f}% | "
                      f"F1 {va_f1:.4f} | {elapsed:.1f}s")

            # Early stopping
            if va_f1 > best_f1 + 1e-4:
                best_f1    = va_f1
                no_improve = 0
                best_state = copy.deepcopy(self.model.state_dict())
                torch.save(best_state,
                           self.save_dir / "best_model.pt")
            else:
                no_improve += 1
                if no_improve >= cfg["patience"]:
                    if cfg["verbose"]:
                        print(f"  [{self.variant}] Early stopping "
                              f"à l'epoch {epoch}.")
                    break

        # Restauration du meilleur état
        if best_state is not None:
            self.model.load_state_dict(best_state)

    # ────────────────────────────────────────────────────────────
    # Fit séquentiel (V5)
    # ────────────────────────────────────────────────────────────

    def _fit_sequential(self):
        cfg = self.cfg

        # ── Phase 1 : entraîner backbone + head_l1 ──────────────
        if cfg["verbose"]:
            print(f"  [{self.variant}] Phase 1 : entraînement L1 "
                  f"({cfg['seq_phase1_epochs']} epochs)")
        opt1 = optim.Adam(
            self.model.parameters(),
            lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        sch1 = CosineAnnealingLR(
            opt1, T_max=cfg["seq_phase1_epochs"], eta_min=1e-6)

        best_state1, best_f1_1 = None, -1.0
        for ep in range(1, cfg["seq_phase1_epochs"] + 1):
            self._run_epoch(self.loaders["train"], opt1,
                            training=True, phase=1)
            _, va_acc, va_f1, _ = self._run_epoch(
                self.loaders["val"], training=False, phase=1)
            sch1.step()
            if va_f1 > best_f1_1:
                best_f1_1 = va_f1
                best_state1 = copy.deepcopy(self.model.state_dict())
            if cfg["verbose"]:
                print(f"    Phase1 ep {ep:02d} | val_acc={va_acc*100:.1f}%")

        if best_state1:
            self.model.load_state_dict(best_state1)

        # ── Phase 2 : geler backbone, entraîner head_l2 ─────────
        if cfg["verbose"]:
            print(f"  [{self.variant}] Phase 2 : entraînement L2 "
                  f"({cfg['seq_phase2_epochs']} epochs)")
        self.model.freeze_for_stage2()
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        opt2 = optim.Adam(
            trainable, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        sch2 = CosineAnnealingLR(
            opt2, T_max=cfg["seq_phase2_epochs"], eta_min=1e-6)

        best_state2, best_f1_2 = None, -1.0
        no_improve = 0
        for ep in range(1, cfg["seq_phase2_epochs"] + 1):
            tr_loss, tr_acc, tr_f1, _ = self._run_epoch(
                self.loaders["train"], opt2, training=True, phase=2)
            va_loss, va_acc, va_f1, va_f1min = self._run_epoch(
                self.loaders["val"], training=False, phase=2)
            sch2.step()

            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(va_loss)
            self.history["val_acc"].append(va_acc)
            self.history["val_f1_macro"].append(va_f1)
            self.history["val_f1_minority"].append(va_f1min)

            if va_f1 > best_f1_2 + 1e-4:
                best_f1_2  = va_f1
                no_improve = 0
                best_state2 = copy.deepcopy(self.model.state_dict())
                torch.save(best_state2, self.save_dir / "best_model.pt")
            else:
                no_improve += 1
                if no_improve >= cfg["patience"]:
                    break

            if cfg["verbose"]:
                print(f"    Phase2 ep {ep:02d} | F1={va_f1:.4f}")

        self.model.unfreeze_all()
        if best_state2:
            self.model.load_state_dict(best_state2)

    # ────────────────────────────────────────────────────────────
    # Point d'entrée principal
    # ────────────────────────────────────────────────────────────

    def fit(self) -> dict:
        """Lance l'entraînement et retourne l'historique."""
        params = count_parameters(self.model)
        if self.cfg["verbose"]:
            print(f"\n{'='*60}")
            print(f"Entraînement {self.variant} | "
                  f"{params/1e6:.1f}M params | "
                  f"device={self.device}")
            print(f"{'='*60}")

        if self.variant == "V5":
            self._fit_sequential()
        else:
            self._fit_standard()

        return self.history

    def evaluate_test(self) -> dict:
        """Évaluation finale sur le test set."""
        loss, acc, f1_macro, f1_minority = self._run_epoch(
            self.loaders["test"], training=False)
        results = {
            "variant":      self.variant,
            "test_loss":    round(loss, 4),
            "test_acc":     round(acc * 100, 2),
            "f1_macro":     round(f1_macro, 4),
            "f1_minority":  round(f1_minority, 4),
            "params_M":     round(count_parameters(self.model) / 1e6, 1),
        }
        return results
