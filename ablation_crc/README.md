# Ablation Study — Multi-Head CNN for CRC WSI Classification
## Dataset : Kather 2016 (8 classes, 5 000 patches)

---

## Structure du projet

```
ablation_crc/
├── dataset.py        # Téléchargement, KatherDataset, DataLoaders
├── models.py         # V1–V6 architectures + FocalLoss
├── train.py          # Trainer (combined loss, early stopping, séquentiel)
├── evaluate.py       # Métriques, confusion matrix, F1 par classe
├── utils.py          # Courbes, barplots, CSV, grille de confusion
├── ablation.py       # Script principal (orchestre tout)
└── requirements.txt
```

---

## Installation
```bash
conda create -n mhcnn_crc python=3.10 -y
conda activate mhcnn_crc
#conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
#fermer vs et ouvrire a nouveau

#pour GPU GTX 1050 Ti
conda activate mhcnn_crc
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#pour GPU RTX 3060
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

conda activate mhcnn_crc
pip install -r requirements.txt

```

---

## Utilisation

### Lancer toute l'étude (6 variantes)
```bash
python ablation.py
```

### Une seule variante
```bash
python ablation.py --variants V2
```

### Reprendre après interruption
```bash
python ablation.py --resume
```

### Évaluer un checkpoint existant
```bash
python evaluate.py --variant V2 --checkpoint checkpoints/V2/best_model.pt
```

### Tester les visualisations
```bash
python utils.py
```

---

## Variantes

| ID | Architecture         | Backbone          | Loss               |
|----|----------------------|-------------------|--------------------|
| V1 | Flat baseline        | ResNet-50 shared  | CE (8 classes)     |
| V2 | mHC-2 **(Ours)**     | ResNet-50 shared  | α·CE_L1 + CE_L2    |
| V3 | mHC-3 (3 niveaux)   | ResNet-50 shared  | 3-level combined   |
| V4 | Branches indép.     | ResNet-50 × 2     | CE each            |
| V5 | Séquentiel          | ResNet-50 shared  | CE stage-wise      |
| V6 | mHC-2 + Focal Loss  | ResNet-50 shared  | α·FL_L1 + CE_L2    |

---

## Sorties

```
checkpoints/<variant>/best_model.pt
results/<variant>/metrics.json
results/<variant>/confusion_matrix.png
results/<variant>/f1_per_class.png
results/training_curves/<variant>_curves.png
results/ablation_summary.json
results/ablation_table.csv
results/ablation_barplot.png
results/ablation_f1_minority.png
```

---

## Hiérarchie des labels

```
Level 1 (L1) — Binaire
├── Tumoral      : TUM, COM
└── Non-tumoral  : STR, LYM, DEB, MUC, ADI, EMP

Level 2 (L2) — 8 classes fine-grained
TUM | STR | COM | LYM | DEB | MUC | ADI | EMP
```

---

## Configuration (ablation.py)

```python
EXP_CONFIG = {
    "lr":            1e-4,
    "weight_decay":  1e-5,
    "epochs":        50,
    "patience":      10,      # early stopping sur val F1-macro
    "alpha":         0.3,     # L = alpha*L1 + (1-alpha)*L2
    "focal_gamma":   2.0,     # pour V6
    "batch_size":    32,
    "seq_phase1_epochs": 20,  # V5 phase 1 (L1)
    "seq_phase2_epochs": 30,  # V5 phase 2 (L2)
}
```
