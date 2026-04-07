"""
models.py
=========
Définition des 6 variantes architecturales pour l'étude ablation.

    V1 : Flat baseline        — 1 tête, 8 classes, CE standard
    V2 : mHC-2 (Ours)        — 2 têtes (L1 binaire + L2 8-class), shared backbone
    V3 : mHC-3               — 3 têtes (L1 binaire + L2 tumoral/non-tum + L3 8-class)
    V4 : Independent branches — 2 backbones ResNet-50 séparés
    V5 : Sequential training  — identique à V2 en architecture, flag pour train.py
    V6 : mHC-2 + Focal Loss  — identique à V2, focal loss sur L1 (flag pour train.py)

Toutes les variantes partagent la même interface :
    model(x)  →  dict avec clés selon la variante
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights

from dataset import NUM_CLASSES, NUM_L1

# ----------------------------------------------------------------
# Utilitaires
# ----------------------------------------------------------------

'''def _resnet50_backbone(pretrained: bool = True):
    """ResNet-50 sans la tête finale (fc), retourne (backbone, feat_dim)."""
    weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.resnet50(weights=weights)
    feat_dim = net.fc.in_features          # 2048
    net.fc = nn.Identity()                 # supprime la couche linéaire finale
    return net, feat_dim'''

# models.py — remplacer _resnet50_backbone par cette version universelle

from torchvision.models import (
    resnet18,  ResNet18_Weights,
    resnet50,  ResNet50_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
)

# ── Choisir ici ────────────────────────────────────────────────
BACKBONE_NAME = "resnet18"   # "resnet50" | "resnet18" | "efficientnet_b0" | "mobilenet_v3_small"
# ──────────────────────────────────────────────────────────────

def _resnet50_backbone(pretrained: bool = True):
    """Backbone interchangeable selon BACKBONE_NAME."""

    if BACKBONE_NAME == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet50(weights=weights)
        feat_dim = net.fc.in_features          # 2048
        net.fc = nn.Identity()

    elif BACKBONE_NAME == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=weights)
        feat_dim = net.fc.in_features          # 512
        net.fc = nn.Identity()

    elif BACKBONE_NAME == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = efficientnet_b0(weights=weights)
        feat_dim = net.classifier[1].in_features   # 1280
        net.classifier = nn.Identity()

    elif BACKBONE_NAME == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        net = mobilenet_v3_small(weights=weights)
        feat_dim = net.classifier[0].in_features   # 576
        net.classifier = nn.Identity()

    else:
        raise ValueError(f"Backbone inconnu : {BACKBONE_NAME}")

    return net, feat_dim
    
def _classification_head(in_features: int, num_classes: int,
                          hidden: int = 512, dropout: float = 0.5):
    """FC head : BN → Dropout → Linear(hidden) → ReLU → Linear(num_classes)."""
    return nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(dropout),
        nn.Linear(in_features, hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, num_classes),
    )


# ================================================================
# V1 — Flat baseline
# ================================================================

class FlatCNN(nn.Module):
    """
    ResNet-50 + une seule tête softmax à 8 classes.
    Forward → {'logits_l2': Tensor[B, 8]}
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone, feat_dim = _resnet50_backbone(pretrained)
        self.head = _classification_head(feat_dim, NUM_CLASSES)
        self.variant = "V1"

    def forward(self, x):
        feat = self.backbone(x)
        return {"logits_l2": self.head(feat)}


# ================================================================
# V2 — mHC-2 : shared backbone + 2 têtes (L1 + L2)
# ================================================================

class MultiHeadCNN2(nn.Module):
    """
    ResNet-50 partagé → GAP 2048-d
    Head L1 : binaire (Tumoral / Non-tumoral)
    Head L2 : 8 classes fine-grained
    Forward → {'logits_l1': Tensor[B,2], 'logits_l2': Tensor[B,8]}
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone, feat_dim = _resnet50_backbone(pretrained)
        self.head_l1 = _classification_head(feat_dim, NUM_L1)
        self.head_l2 = _classification_head(feat_dim, NUM_CLASSES)
        self.variant = "V2"

    def forward(self, x):
        feat = self.backbone(x)
        return {
            "logits_l1": self.head_l1(feat),
            "logits_l2": self.head_l2(feat),
        }


# ================================================================
# V3 — mHC-3 : 3 niveaux hiérarchiques
# ================================================================
# L1 : 2 classes  (Tumoral / Non-tumoral)
# L2 : 3 classes  (Tumoral_Complex / Stroma_Lympho / Debris_Mucosa_Adip_Empty)
# L3 : 8 classes  (fine-grained)

NUM_L2_V3 = 3   # groupes intermédiaires

class MultiHeadCNN3(nn.Module):
    """
    ResNet-50 partagé → 3 têtes hiérarchiques.
    Forward → {'logits_l1', 'logits_l2_mid', 'logits_l2'}
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone, feat_dim = _resnet50_backbone(pretrained)
        self.head_l1     = _classification_head(feat_dim, NUM_L1)
        self.head_l2_mid = _classification_head(feat_dim, NUM_L2_V3)
        self.head_l3     = _classification_head(feat_dim, NUM_CLASSES)
        self.variant = "V3"

    def forward(self, x):
        feat = self.backbone(x)
        return {
            "logits_l1":     self.head_l1(feat),
            "logits_l2_mid": self.head_l2_mid(feat),
            "logits_l2":     self.head_l3(feat),
        }


# ================================================================
# V4 — Independent branches : 2 backbones séparés
# ================================================================

class IndependentBranchCNN(nn.Module):
    """
    Backbone L1 (ResNet-50) → tête L1
    Backbone L2 (ResNet-50) → tête L2
    Pas de poids partagés.
    Forward → {'logits_l1', 'logits_l2'}
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone_l1, feat_dim = _resnet50_backbone(pretrained)
        self.backbone_l2, _        = _resnet50_backbone(pretrained)
        self.head_l1 = _classification_head(feat_dim, NUM_L1)
        self.head_l2 = _classification_head(feat_dim, NUM_CLASSES)
        self.variant = "V4"

    def forward(self, x):
        feat_l1 = self.backbone_l1(x)
        feat_l2 = self.backbone_l2(x)
        return {
            "logits_l1": self.head_l1(feat_l1),
            "logits_l2": self.head_l2(feat_l2),
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ================================================================
# V5 — Sequential training (même archi que V2, flag dans train.py)
# ================================================================

class MultiHeadCNNSequential(MultiHeadCNN2):
    """
    Identique à V2 en architecture.
    L'entraînement séquentiel est géré dans train.py via le flag
    `sequential=True` : phase 1 = backbone + head_l1 gelé head_l2,
                         phase 2 = backbone gelé, head_l2.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__(pretrained)
        self.variant = "V5"

    def freeze_for_stage2(self):
        """Gèle backbone et head_l1 pour la phase 2 (entraîn. head_l2)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.head_l1.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True


# ================================================================
# V6 — mHC-2 + Focal Loss (même archi que V2, loss différente)
# ================================================================

class MultiHeadCNNFocal(MultiHeadCNN2):
    """
    Identique à V2. La Focal Loss est appliquée dans train.py
    via le flag `focal=True` sur la tête L1.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__(pretrained)
        self.variant = "V6"


# ================================================================
# Focal Loss
# ================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss multi-classe.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha  : pondération par classe (Tensor ou None)
        gamma  : paramètre de modulation (défaut 2.0)
        reduction: 'mean' | 'sum' | 'none'
    """
    def __init__(self, alpha=None, gamma: float = 2.0,
                 reduction: str = "mean"):
        super().__init__()
        if alpha is not None:
            self.alpha = torch.as_tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            fl = alpha_t * (1 - pt) ** self.gamma * ce
        else:
            fl = (1 - pt) ** self.gamma * ce

        if self.reduction == "mean":
            return fl.mean()
        elif self.reduction == "sum":
            return fl.sum()
        return fl


# ================================================================
# Registre des variantes
# ================================================================

VARIANTS = {
    "V1": FlatCNN,
    "V2": MultiHeadCNN2,
    "V3": MultiHeadCNN3,
    "V4": IndependentBranchCNN,
    "V5": MultiHeadCNNSequential,
    "V6": MultiHeadCNNFocal,
}

VARIANT_LABELS = {
    "V1": "Flat baseline",
    "V2": "mHC-2 (Ours)",
    "V3": "mHC-3 (3 levels)",
    "V4": "Independent branches",
    "V5": "Sequential training",
    "V6": "mHC-2 + Focal Loss",
}


def build_model(variant: str, pretrained: bool = True) -> nn.Module:
    """Instancie la variante choisie."""
    assert variant in VARIANTS, \
        f"Variante inconnue : {variant}. Choisir parmi {list(VARIANTS.keys())}."
    return VARIANTS[variant](pretrained=pretrained)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------------------------------------------
# Test standalone
# ----------------------------------------------------------------
if __name__ == "__main__":
    x = torch.randn(4, 3, 224, 224)
    for v in VARIANTS:
        m = build_model(v, pretrained=False)
        out = m(x)
        params = count_parameters(m)
        print(f"{v} ({VARIANT_LABELS[v]}) | "
              f"params={params/1e6:.1f}M | "
              f"output keys={list(out.keys())}")
