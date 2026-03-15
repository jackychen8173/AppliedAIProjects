# Mini Project 7: Blood Cell Detection using YOLO26

> COMP 9130 — Applied Artificial Intelligence | Mini Project VII

## Problem Description

MedScan Diagnostics is a healthcare technology company developing AI-assisted screening tools for pathology labs. This project builds a proof-of-concept object detection system for **automated blood cell detection** in blood smear images using the YOLO26 (Ultralytics) model.

The goal is to detect and classify **Red Blood Cells (RBC)**, **White Blood Cells (WBC)**, and **Platelets** to support automated complete blood count (CBC) analysis for clinical laboratories processing thousands of samples daily.

---

## Dataset

- **Name:** BCCD (Blood Cell Count and Detection)
- **Source:** [Roboflow Universe — Joseph Nelson, Version 3](https://universe.roboflow.com/joseph-nelson/bccd) (MIT License)
- **Classes:** 3 — Platelet, RBC, WBC
- **Split:** 255 train / 73 validation / 36 test images
- **Class distribution (train):** Platelet: 249 | RBC: 2,938 | WBC: 263
- **Note:** Significant class imbalance — RBC accounts for ~85% of all instances

### How to Download the Dataset

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("joseph-nelson").project("bccd")
version = project.version(3)
dataset = version.download("yolov8", location="../data/raw/BCCD-3")
```

> Get your free API key at [roboflow.com](https://roboflow.com). Select **YOLOv8** format when downloading.

---

## Environment Setup

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended) — tested on NVIDIA GeForce RTX 3060 12GB
- Ultralytics 8.4.19, PyTorch 2.5.1+cu121

### Installation

```bash
# Clone the repository
git clone https://github.com/<username>/MiniProject7-BloodCellDetection.git
cd MiniProject7-BloodCellDetection

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## How to Run

```bash
# 1. Download the dataset (see above) into data/raw/BCCD-3/

# 2. Open the training notebook
jupyter notebook notebooks/medscan_bccd.ipynb

# 3. Run all cells in order:
#    - Sections 1–6:  Setup, data loading, EDA, visualisation
#    - Section 7:     Hyperparameter experiments (4 runs)
#    - Sections 8–9:  Results comparison, best model selection
#    - Sections 10–15: Full evaluation, training curves, confusion matrix,
#                      prediction visualisation, error analysis
#    - Section 16:    Final summary
```

---

## Project Structure

```
MiniProject7-BloodCellDetection/
├── README.md                        # Project documentation (this file)
├── requirements.txt                 # Python dependencies
├── notebooks/
│   └── medscan_bccd.ipynb           # Main training & evaluation notebook
├── data/
│   ├── raw/                         # Downloaded BCCD-3 dataset (git-ignored)
│   └── processed/                   # Reserved for preprocessing outputs
├── models/                          # Saved best.pt weights (git-ignored)
│   ├── run_A_batch16_imgsz416_best.pt
│   ├── run_B_batch16_imgsz640_best.pt
│   ├── run_C_batch8_imgsz416_best.pt
│   └── run_D_batch8_imgsz640_best.pt
├── results/
│   ├── figures/                     # All charts and visualisations
│   │   ├── class_distribution.png
│   │   ├── hyperparameter_comparison_map.png
│   │   ├── hyperparameter_comparison_prec_recall.png
│   │   ├── per_class_ap50_heatmap.png
│   │   ├── training_curves_best.png
│   │   ├── confusion_matrix_best.png
│   │   ├── val_predictions_best.png
│   │   └── incorrect_predictions_best.png
│   └── metrics/
│       └── hyperparameter_comparison.csv
└── docs/
    └── Mini_Project_VII_Report.pdf  # Written report
```

---

## Hyperparameter Experiments

All runs used YOLO26n pre-trained on COCO, trained for **25 epochs** with AdamW optimizer, early stopping patience = 10.

| Run | Batch | Img Size | mAP@50 | mAP@50-95 | Precision | Recall |
|-----|-------|----------|--------|-----------|-----------|--------|
| A   | 16    | 416      | 0.8622 | 0.6082    | 0.7779    | 0.8527 |
| **B** ⭐ | **16** | **640** | **0.8671** | **0.6270** | **0.8195** | **0.8253** |
| C   | 8     | 416      | 0.8542 | 0.6109    | 0.7570    | 0.8289 |
| D   | 8     | 640      | 0.8541 | 0.6092    | 0.7839    | 0.8153 |

**Best configuration: Run B — batch=16, imgsz=640**

---

## Results Summary

### Overall Performance (Best Model — Run B)

| Metric    | Value  |
|-----------|--------|
| mAP@50    | 0.8671 |
| mAP@50-95 | 0.6270 |
| Precision | 0.8195 |
| Recall    | 0.8253 |

### Per-Class Performance

| Class    | AP@50  | AP@50-95 | Notes                              |
|----------|--------|----------|------------------------------------|
| Platelet | 0.8168 | 0.4959   | Small object, limited training examples |
| RBC      | 0.8057 | 0.5795   | Dense overlap — model's weakest class   |
| WBC      | 0.9787 | 0.8057   | Visually distinctive, near-perfect      |

### Confusion Matrix Highlights

- **Platelet:** 65/75 correctly detected; 10 missed as background, 28 false positives
- **RBC:** 617/819 correctly detected; 202 missed as background, 240 false positives
- **WBC:** 72/72 correctly detected; only 5 false positives

---

## Key Findings

- **Best hyperparameters:** Batch 16 + image size 640 (Run B) outperformed all other configurations. Larger input resolution provides more spatial detail for small cell detection, and batch size 16 offers more stable gradient updates than batch 8 for this dataset size (~255 training images).
- **Class imbalance effect:** Despite RBC dominating 85% of labels, WBC achieved the highest AP (0.9787). Visual distinctiveness matters more than instance count — WBCs are large, uniquely stained, and well-separated. RBC's dense overlapping layout is the primary challenge.
- **Most challenging class:** RBC, due to dense packing and overlapping cells making precise localisation difficult.
- **Recommended confidence threshold:** 0.55 for clinical screening — balances false positive reduction with sufficient recall to catch genuine pathology. The development threshold of 0.25 is too permissive for clinical deployment.
- **False positives vs. false negatives:** False negatives are the more dangerous error in this context. Missing a WBC could delay leukemia or infection diagnosis; missed Platelets risk undetected thrombocytopenia.
- **Deployment recommendation:** Suitable as a **screening aid with mandatory pathologist oversight**. Not ready for standalone clinical diagnosis. Recommended next steps: expand dataset to 2,000–5,000 images, apply SAHI for small object detection, validate on an independent clinical test set, and pursue Health Canada Class II medical device approval.

---

## Team Member Contributions

| Member | Contributions |
|--------|---------------|
| Jacky Chen | Data preprocessing, model training, hyperparameter experiments|
| Savina Cai | Evaluation, analysis, report writing |

---

## References

- Nelson, J. (2022). BCCD Dataset (Version 3). Roboflow Universe. https://universe.roboflow.com/joseph-nelson/bccd
- Ultralytics. (2026). YOLO26 Documentation. https://docs.ultralytics.com
- Lin, T.-Y., et al. (2015). Microsoft COCO: Common Objects in Context. https://cocodataset.org
- Akyon, F. C., et al. (2022). Slicing Aided Hyper Inference and Fine-Tuning for Small Object Detection. IEEE ICIP 2022. https://github.com/obss/sahi

---

## Disclaimer

This is an educational exercise for COMP 9130 — Applied Artificial Intelligence. Real medical AI systems require extensive validation, regulatory approval, and clinical trials before deployment. The model described here is a proof-of-concept only and is not validated for clinical use.
