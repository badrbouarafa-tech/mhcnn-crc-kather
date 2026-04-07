"""
dataset.py
==========
Téléchargement, organisation et chargement du dataset Kather 2016.

Structure attendue après téléchargement :
    data/kather2016/
        TUM/  STR/  COM/  LYM/  DEB/  MUC/  ADI/  EMP/

Usage :
    python dataset.py          # télécharge et prépare le dataset
"""

import os
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

# ----------------------------------------------------------------
# Constantes
# ----------------------------------------------------------------

KATHER_URL = (
    "https://zenodo.org/record/53169/files/"
    "Kather_texture_2016_image_tiles_5000.zip"
)

# 8 classes Kather 2016
CLASS_NAMES = ["TUM", "STR", "COM", "LYM", "DEB", "MUC", "ADI", "EMP"]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

# Hiérarchie L1 : Tumoral (1) vs Non-tumoral (0)
# TUM=0, COM=2  → Tumoral (L1=1)
# STR, LYM, DEB, MUC, ADI, EMP → Non-tumoral (L1=0)
L1_MAPPING = {
    "TUM": 1, "COM": 1,                          # Tumoral
    "STR": 0, "LYM": 0, "DEB": 0,
    "MUC": 0, "ADI": 0, "EMP": 0,               # Non-tumoral
}
L1_NAMES   = ["Non-tumoral", "Tumoral"]
NUM_L1     = 2

# Classes minoritaires pour F1-minority
MINORITY_CLASSES = ["DEB", "EMP"]
MINORITY_IDX     = [CLASS_TO_IDX[c] for c in MINORITY_CLASSES]

DATA_ROOT = Path("data/kather2016")

# ----------------------------------------------------------------
# Téléchargement
# ----------------------------------------------------------------

def download_kather2016(root: Path = DATA_ROOT):
    """Télécharge et extrait le dataset Kather 2016 si absent."""
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "kather2016.zip"

    if not any(root.glob("*/")) :
        print(f"[dataset] Téléchargement depuis {KATHER_URL} …")
        urllib.request.urlretrieve(KATHER_URL, zip_path)
        print(f"[dataset] Extraction dans {root} …")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(root)
        zip_path.unlink()
        print("[dataset] Prêt.")
    else:
        print(f"[dataset] Dataset déjà présent dans {root}.")


def discover_images(root: Path = DATA_ROOT):
    """
    Retourne deux listes parallèles :
        paths  : chemins absolus des images
        labels : index de classe (L2, 0–7)
    """
    paths, labels = [], []
    # Le zip extrait un sous-dossier — chercher récursivement
    for cls_name in CLASS_NAMES:
        cls_dirs = list(root.rglob(cls_name))
        for cls_dir in cls_dirs:
            if cls_dir.is_dir():
                for img_path in sorted(cls_dir.iterdir()):
                    if img_path.suffix.lower() in {".tif", ".png", ".jpg"}:
                        paths.append(str(img_path))
                        labels.append(CLASS_TO_IDX[cls_name])
    print(f"[dataset] {len(paths)} images trouvées, "
          f"{len(set(labels))} classes.")
    return paths, labels


# ----------------------------------------------------------------
# Dataset PyTorch
# ----------------------------------------------------------------

class KatherDataset(Dataset):
    """
    Dataset Kather 2016 avec labels L1 (binaire) et L2 (8 classes).

    Chaque item retourne :
        image  : Tensor [3, 224, 224]
        label_l1 : int  (0=Non-tumoral, 1=Tumoral)
        label_l2 : int  (0–7, fine-grained)
    """

    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, paths, labels, augment: bool = False):
        self.paths  = paths
        self.labels = labels   # L2 labels

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2,
                    saturation=0.1, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(self.MEAN, self.STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(self.MEAN, self.STD),
            ])

        # L1 labels dérivés du L2
        l2_to_l1 = [L1_MAPPING[CLASS_NAMES[i]] for i in range(NUM_CLASSES)]
        self.l1_labels = [l2_to_l1[lb] for lb in labels]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)
        return img, self.l1_labels[idx], self.labels[idx]


# ----------------------------------------------------------------
# Splits stratifiés
# ----------------------------------------------------------------

def make_splits(paths, labels, val_ratio=0.15, test_ratio=0.15,
                seed=42):
    """
    Split stratifié 70 / 15 / 15.
    Retourne trois listes d'indices.
    """
    indices = list(range(len(paths)))

    idx_trainval, idx_test = train_test_split(
        indices, test_size=test_ratio,
        stratify=labels, random_state=seed)

    val_frac = val_ratio / (1 - test_ratio)
    labels_trainval = [labels[i] for i in idx_trainval]

    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_frac,
        stratify=labels_trainval, random_state=seed)

    print(f"[dataset] Split → train={len(idx_train)}, "
          f"val={len(idx_val)}, test={len(idx_test)}")
    return idx_train, idx_val, idx_test


def get_dataloaders(root: Path = DATA_ROOT, batch_size: int = 32,
                    num_workers: int = 4, seed: int = 42):
    """
    Retourne un dict {'train', 'val', 'test'} de DataLoaders.
    Télécharge le dataset si nécessaire.
    """
    download_kather2016(root)
    paths, labels = discover_images(root)

    idx_train, idx_val, idx_test = make_splits(paths, labels, seed=seed)

    train_labels = [labels[i] for i in idx_train]
    val_labels   = [labels[i] for i in idx_val]
    test_labels  = [labels[i] for i in idx_test]

    ds_train = KatherDataset(
        [paths[i] for i in idx_train], train_labels, augment=True)
    ds_val   = KatherDataset(
        [paths[i] for i in idx_val],   val_labels,   augment=False)
    ds_test  = KatherDataset(
        [paths[i] for i in idx_test],  test_labels,  augment=False)

    loaders = {
        "train": DataLoader(ds_train, batch_size=batch_size,
                            shuffle=True,  num_workers=num_workers,
                            pin_memory=True),
        "val":   DataLoader(ds_val,   batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True),
        "test":  DataLoader(ds_test,  batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True),
    }
    return loaders


# ----------------------------------------------------------------
# Test standalone
# ----------------------------------------------------------------
if __name__ == "__main__":
    loaders = get_dataloaders()
    imgs, l1, l2 = next(iter(loaders["train"]))
    print(f"Batch shape : {imgs.shape}")
    print(f"L1 labels   : {l1[:8].tolist()}")
    print(f"L2 labels   : {l2[:8].tolist()}")
    print(f"Class names : {[CLASS_NAMES[i] for i in l2[:8].tolist()]}")
