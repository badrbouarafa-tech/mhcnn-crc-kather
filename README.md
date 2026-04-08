# Multi-Head CNN for Hierarchical Classification of Colorectal Cancer WSIs

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset: Kather 2016](https://img.shields.io/badge/Dataset-Kather%202016-lightgrey.svg)](https://doi.org/10.5281/zenodo.53169)

> **Official implementation** of the paper:  
> *"Multi-Head Convolutional Neural Network for Hierarchical Classification of Colorectal Cancer Whole Slide Images: A Systematic Ablation Study on Kather 2016"*  
> Badr Bouarafa et al.
>
> 📄 **Paper:** Under review at *Computers in Biology and Medicine* (Elsevier), 2026.  
> The full article link will be added here upon acceptance.

---

## Overview

We propose a **Multi-Head Classification (mHC)** framework that couples a shared ResNet-18 backbone with two specialised classification heads:

- **Head L1** — coarse binary decision: *Tumoral* (TUM + COM) vs. *Non-tumoral*
- **Head L2** — fine-grained 8-class tissue classification

Both heads are trained **jointly** via a weighted combined loss:

$$\mathcal{L} = 0.3 \cdot \mathcal{L}_{L1} + 0.7 \cdot \mathcal{L}_{L2}$$

A **systematic 6-variant ablation study** on [Kather 2016](https://doi.org/10.5281/zenodo.53169) (5,000 patches, 8 classes) isolates the contribution of hierarchy, backbone sharing, training strategy, and loss design.

---

## Key Results

| Variant | Description | Acc (%) | F1-macro | Params (M) |
|---|---|---|---|---|
| V1 | Flat baseline | 95.47 | 0.9553 | 11.4 |
| **V2** | **mHC-2 (Ours)** | **95.87** | **0.9588** | **11.7** |
| V3 | mHC-3 (3 levels) | 94.93 | 0.9495 | 12.0 |
| V4 | Independent branches | 95.20 | 0.9521 | 22.9 |
| V5 | Sequential training | 91.20 | 0.9133 | 11.7 |
| V6 | mHC-2 + Focal Loss | 93.33 | 0.9332 | 11.7 |

**Key findings:**
- mHC-2 (V2) achieves the best accuracy **and** uses 49% fewer parameters than V4
- Sequential training (V5) **collapses** during Phase-1 (val-acc ≤ 14.8%) — joint training is mandatory
- Hierarchical supervision improves *Complex stroma* F1 by **+2.70%** over the flat baseline
- Architecture is compatible with **federated learning** via selective backbone synchronisation (−95.7% communication cost)

---

## Repository Structure

```
mhcnn-crc-kather/
├── ablation_crc/
│   ├── dataset.py        # Kather 2016 download + KatherDataset class
│   ├── models.py         # 6 ablation variants + FocalLoss
│   ├── train.py          # Trainer (combined loss, early stopping)
│   ├── evaluate.py       # Metrics, confusion matrix, per-class F1
│   ├── ablation.py       # Main orchestrator — runs all 6 variants
│   └── utils.py          # Visualisations, barplots, CSV export
├── results/
│   ├── ablation_summary.json
│   ├── ablation_barplot.png
│   ├── ablation_table.csv
│   ├── ablation_f1_minority.png
│   ├── V1/
│   │   ├── confusion_matrix.png
│   │   └── f1_per_class.png
│   │   └── metrics.json
│   ├── V2/
│   │   ├── confusion_matrix.png
│   │   └── f1_per_class.png
│   │   └── metrics.json
│   ├── V3/
│   │   ├── confusion_matrix.png
│   │   └── f1_per_class.png
│   │   └── metrics.json
│   ├── V4/
│   │   ├── confusion_matrix.png
│   │   └── f1_per_class.png
│   │   └── metrics.json
│   ├── V5/
│   │   ├── confusion_matrix.png
│   │   └── f1_per_class.png
│   │   └── metrics.json
│   ├── V6/
│   │   ├── confusion_matrix.png
│   │   └── f1_per_class.png
│   │   └── metrics.json
│   └── training_curves/
│       ├── V1_curves.png
│       └── V2_curves.png
│       ├── V3_curves.png
│       └── V4_curves.png
│       ├── V5_curves.png
│       └── V6_curves.png
├── requirements.txt
├── environment.yml
└── README.md
```

---

## Installation

### Option 1 — Conda (recommended)

```bash
# 1. Clone the repository
git clone https://github.com/bouarafa/mhcnn-crc-kather.git
cd mhcnn-crc-kather

# 2. Create the conda environment
conda create -n mhcnn_crc python=3.10 -y
conda activate mhcnn_crc

# 3. Install PyTorch (adjust CUDA version if needed)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 4. Install remaining dependencies
pip install -r requirements.txt
```

### Option 2 — pip only

```bash
pip install torch==2.1.0 torchvision==0.16.0
pip install -r requirements.txt
```

### Requirements

```
torch>=2.1.0
torchvision>=0.16.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
numpy>=1.24.0
tqdm>=4.65.0
Pillow>=9.5.0
```

---

## Dataset

The **Kather 2016** dataset is publicly available on Zenodo:

> Kather, J.N. et al. (2016). *Multi-class texture analysis in colorectal cancer histology.* Scientific Reports, 6, 27988.  
> DOI: [10.5281/zenodo.53169](https://doi.org/10.5281/zenodo.53169)

The dataset will be **downloaded automatically** when you run `ablation.py`.  
It contains **5,000 patches** (150×150 px, 20× magnification) across **8 tissue classes**:

| Class | Abbreviation | Count |
|---|---|---|
| Tumor epithelium | TUM | 625 |
| Complex stroma | COM | 625 |
| Simple stroma | STR | 625 |
| Lymphocytes | LYM | 625 |
| Debris | DEB | 625 |
| Normal mucosa | MUC | 625 |
| Adipose | ADI | 625 |
| Empty | EMP | 625 |

---

## Usage

### Run the full ablation study (all 6 variants)

```bash
conda activate mhcnn_crc
python ablation_crc/ablation.py
```

This will:
1. Download Kather 2016 automatically (if not already present)
2. Train and evaluate all 6 variants sequentially
3. Save results to `results/ablation_summary.txt`
4. Generate figures in `results/`

**Estimated runtime** (single GPU, 8 GB VRAM):

| Backbone | Time per variant | Total (6 variants) |
|---|---|---|
| ResNet-18 | ~1 hour | ~6 hours |
| ResNet-50 | ~2h15 | ~13h30 |

> ResNet-18 is the recommended backbone for reproducibility.  
> To switch backbone: change `BACKBONE_NAME = "resnet18"` in `models.py`.

### Run a single variant

```python
from ablation_crc.ablation import run_variant
run_variant(variant_id="V2", data_dir="./data", results_dir="./results")
```

### Evaluate a saved checkpoint

```bash
python ablation_crc/evaluate.py \
  --checkpoint results/V2/best_model.pth \
  --variant V2 \
  --data_dir ./data
```

---

## Architecture

```
Input (224×224×3)
       │
  ┌────▼────────────────────────────┐
  │   ResNet-18 Backbone (shared)   │
  │         11.2M parameters        │
  │   Conv → BN → ReLU → MaxPool    │
  │   Stage 1–4 (64/128/256/512 ch) │
  └────────────────┬────────────────┘
                   │ GAP → 512-d
          ┌────────┴────────┐
          │                 │
   ┌──────▼──────┐   ┌──────▼──────┐
   │  Head L1    │   │  Head L2    │
   │  (binary)   │   │  (8-class)  │
   │ BN→Drop→FC  │   │ BN→Drop→FC  │
   │  512→256→2  │   │ 512→256→8   │
   └──────┬──────┘   └──────┬──────┘
          │                 │
   Tumoral / Non-tum   8-class tissue
```

**Combined loss:** `L = 0.3 · L_L1 + 0.7 · L_L2`  
**Optimizer:** Adam (lr=1e-4, wd=1e-5)  
**Schedule:** CosineAnnealingLR (T=50, lr_min=1e-6)  
**Batch size:** 32 | **Early stopping:** patience=10 on val macro-F1

---

## Federated Learning

The mHC-2 architecture is designed for federated deployment:

- **Backbone (11.2M params, 95.7%)** → synchronised across hospital nodes via FedAvg
- **Classification heads (0.5M, 4.3%)** → kept local, adapt to site-specific staining

This reduces per-round communication bandwidth by **95.7%** compared to full model synchronisation, while preserving privacy (raw WSI data never leaves the hospital).

A federated evaluation on Kather 2019 (100,000 patches) with non-IID virtual hospital nodes is planned as the direct continuation of this work.

---

## Reproducibility

All experiments were run with:
- PyTorch 2.1 + CUDA 12.1
- Random seeds fixed: `torch.manual_seed(42)`, `numpy.random.seed(42)`
- Single NVIDIA GPU (8 GB VRAM)

Results may vary slightly across hardware due to CUDA non-determinism in certain operations (e.g., `atomicAdd` in convolutional layers). To enable full determinism (at the cost of speed):

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## Citation

If you use this code or results in your research, please cite:

> The paper is currently **under review**. Once accepted, the BibTeX entry with DOI and volume/pages will be provided here.

```bibtex
@article{bouarafa2026mhcnn,
  author  = {Bouarafa, Badr and others},
  title   = {Multi-Head Convolutional Neural Network for Hierarchical
             Classification of Colorectal Cancer Whole Slide Images:
             A Systematic Ablation Study on {Kather} 2016},
  journal = {Computers in Biology and Medicine},
  year    = {2026},
  note    = {Under review — link will be updated upon acceptance}
}
```

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

The Kather 2016 dataset is distributed under its own license (CC BY 4.0).  
See [doi.org/10.5281/zenodo.53169](https://doi.org/10.5281/zenodo.53169) for dataset terms.

---

## Contact

**Badr Bouarafa**  
Informatics and Applications Laboratory, Faculty of Sciences  
Moulay Ismail University, Meknes, Morocco  
📧 ba.bouarafa@edu.umi.ac.ma  
🔗 ORCID: [0009-0004-1624-5894](https://orcid.org/0009-0004-1624-5894)
