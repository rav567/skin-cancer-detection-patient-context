# Image-Level Patient Context for Skin Cancer Detection on Cropped 3D-TBP Tiles

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EfficientNet--B0-EE4C2C?logo=pytorch&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-tabular%20baseline-9ACD32)
![Dataset](https://img.shields.io/badge/Dataset-ISIC%202024%20SLICE--3D-20BEFF?logo=kaggle&logoColor=white)
![Tests](https://img.shields.io/badge/tests-11%20pytest-blue)
![Status](https://img.shields.io/badge/status-complete-brightgreen)

> **Dermatologists spot melanoma by comparing a lesion to the patient's other moles: the "ugly duckling" sign. Can a neural network learn that comparison from images alone?**
>
> This project tests whether aggregating image embeddings across a patient's *other* lesions improves skin cancer detection on the **ISIC 2024 SLICE-3D** dataset, compared with an image-only model and a tabular metadata baseline. No published work had modelled patient context visually on cropped 3D total-body-photography tiles; the ISIC 2024 winners only encoded it through engineered tabular features.

**TL;DR:** the hypothesis was **not supported**, and the study shows why. Image-level patient context was significantly worse than the image + tabular model (**p < 0.001**) and made results **3.5× more variable** between runs, while the plain image-only model matched the best configuration. Representation analysis shows the ugly-duckling signal *does* exist in the embedding space, but it is too weak and unstable to exploit this way.

Built for the Applied Machine Learning Systems II module (ELEC0135) at UCL, 2025/26.

---

## Results

Validation pAUC above 80% TPR (the official ISIC 2024 metric) under patient-grouped cross-validation. Values are mean with a 95% bootstrap confidence interval. The metric's maximum is 0.20.

| Model | Type | Mean pAUC ↑ | 95% CI | SD (runs) |
|---|---|:---:|:---:|:---:|
| Naive (majority) | Baseline | 0.020 | [0.020, 0.020] | 0.000 (5) |
| Tabular (LightGBM) | Baseline | 0.121 | [0.115, 0.128] | 0.013 (15) |
| Image only (EfficientNet-B0) | Baseline | 0.131 | [0.126, 0.136] | 0.008 (10) |
| **A** · Image only | Config | 0.141 | [0.138, 0.146] | **0.006** (6) |
| **B** · Image + tabular | Config | **0.144** | [0.137, 0.153] | 0.010 (6) |
| **C** · Image + patient context | Config | 0.130 | [0.114, 0.146] | 0.022 (6) |
| **D** · Combined (tabular + context) | Config | 0.133 | [0.121, 0.146] | 0.018 (6) |

### Statistical significance

Paired bootstrap tests over shared folds (1,000 resamples). Δ is the second model minus the first.

| Comparison | Δ pAUC | p-value | Significant |
|---|:---:|:---:|:---:|
| A vs B | +0.0028 | 0.356 | no |
| A vs C | −0.0112 | 0.106 | no |
| A vs D | −0.0080 | 0.163 | no |
| B vs C | −0.0140 | < 0.001 | **yes** |
| B vs D | −0.0108 | < 0.001 | **yes** |
| C vs D | +0.0032 | 0.093 | no |

---

## Key findings

1. **Patient context hurt rather than helped.** Both context configurations (C and D) were significantly worse than image + tabular (p < 0.001), and no configuration significantly beat the image-only model.
2. **Simplest is best here.** The image-only model (0.141) matched the top configuration, while the most elaborate one (D) underperformed it. The extra context machinery bought no accuracy.
3. **The "ugly duckling" residual was the most harmful part.** In the ablation, adding the context vector `[h, c]` lowered pAUC, and adding the deviation term `[h, c, h − c]` lowered it significantly (Δ = −0.0095, p = 0.016).
4. **The aggregator doesn't matter.** Mean and attention pooling were statistically indistinguishable (Δ = +0.0014, p = 0.79), so the problem lies in the context signal itself, not in how lesions are pooled.
5. **Context made training a lottery.** Run-to-run standard deviation rose from 0.006 (A) to 0.022 (C), with individual C runs ranging from 0.103 to 0.155.
6. **The signal is real, just weak.** In a UMAP of learned embeddings, the context residual ‖h − c‖ is largest exactly where malignant lesions cluster. Because the context is recomputed every epoch from an encoder that is still changing, it acts as a moving target and behaves like noise.

---

## Approach

### Shared architecture
- **Encoder:** EfficientNet-B0 pretrained on ImageNet (via `timm`), producing a 1,280-d lesion embedding `h`
- **Head:** 2-layer MLP (256 hidden units, ReLU, dropout 0.3) producing the malignancy logit

### Four configurations
| Config | Inputs | Purpose |
|---|---|---|
| **A** | Image only | Anchor baseline |
| **B** | Image + 9 tabular features (age, lesion long diameter, one-hot sex and anatomical site) | Lightweight structured-data comparison |
| **C** | Image + image-level patient context | **The novel contribution** |
| **D** | Image + tabular + patient context | Tests whether the two sources are complementary |

### Patient-context module (C, D)
- Embeddings of the patient's other lesions are aggregated into a context vector `c`, either by **mean pooling** or by **scaled dot-product attention** with the current lesion as the query
- The classifier sees `[h, c, h − c]`, where the residual `h − c` is designed to capture how much a lesion stands out from the patient's typical mole
- Single-lesion patients use a learnable default context vector
- The context embedding cache is rebuilt every epoch with the current encoder

### Handling extreme class imbalance
- **Focal loss** (γ = 2) with a per-fold class weight
- **Online hard negative mining:** all positives plus the hardest 50% of negatives in each batch
- **Balanced batch sampler** with a 1:3 positive-to-negative ratio

### Training
| Learning rate | Batch | Image size | Epochs | SWA from | Seeds | Optimiser |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1e-4 | 96 | 96 × 96 | 6 | epoch 5 | 42, 123 | AdamW |

Mixed precision, early stopping on validation pAUC (patience 2) and stochastic weight averaging. Hyperparameters are held fixed across all configurations, so differences reflect architecture rather than tuning.

### Evaluation protocol
- **Patient-grouped stratified 5-fold CV** (`StratifiedGroupKFold`), with a runtime assertion that halts the pipeline if any patient appears in more than one fold
- **Metric:** partial AUC above 80% TPR, isolating the high-sensitivity region that matters for screening. AUC, sensitivity at 90% specificity and F1 are also reported.
- **Statistics:** 95% bootstrap confidence intervals and paired bootstrap tests
- **Ablations:** aggregation (mean vs attention) and fusion structure (`[h]` vs `[h, c]` vs `[h, c, h − c]`)

---

## Dataset

[**ISIC 2024 – Skin Cancer Detection with 3D-TBP**](https://www.kaggle.com/competitions/isic-2024-challenge) (SLICE-3D) contains roughly 400,000 cropped 15 mm lesion tiles from 1,042 patients, at about 0.1% malignancy. It is licensed CC BY-NC 4.0.

To keep the patient structure the hypothesis depends on, the working set keeps **all 259 patients with a malignant lesion** plus 25 random benign-only patients, capped at 100 lesions per patient. That gives **27,595 lesions across 284 patients** (393 malignant, 1.4%).

> The dataset is heavily skewed towards fair skin types, so results may not generalise to darker skin tones.

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/rav567/skin-cancer-detection-patient-context.git
cd skin-cancer-detection-patient-context
conda env create -f environment.yml
conda activate amls2-final
```

Prefer pip? Use Python 3.11 and run `pip install -r requirements.txt` instead.

### 2. Get the data

The dataset downloads automatically through the Kaggle API on the first run. You need to:

1. Join the [ISIC 2024 competition](https://www.kaggle.com/competitions/isic-2024-challenge) on Kaggle and accept its rules.
2. Create an API token under **Kaggle → Settings → API → Create New Token**, and save `kaggle.json` to `~/.kaggle/`.

> **No Kaggle account?** The pipeline still runs. It generates a small synthetic dataset and a 1-epoch demo of every configuration, so you can check that the whole codebase works end to end in minutes.

### 3. Run

```bash
python main.py
```

One command runs the unit tests, downloads and validates the data, builds the patient-grouped folds, generates the EDA figures, and trains every baseline, configuration and ablation before producing all the result plots.

> **Runtime:** the full experiments took about 6 hours per configuration on a 16 GB consumer laptop.

### Run the tests on their own

```bash
pytest -q tests
```

Eleven unit tests check the custom pAUC metric against synthetic ground truth.

## Outputs

Everything is written to `results/`:

| Location | Contents |
|---|---|
| `results/eda/` | Class distribution, lesions per patient and other EDA figures |
| `results/baselines/` | Naive, LightGBM and image-only baseline metrics, plus a comparison table and forest plot |
| `results/configurations/` | Per-run metrics and predictions for configs A–D |
| `results/ablations/` | Aggregation and fusion ablation results |
| `results/figures/` | Training curves, master results dot plot, ablation chart, variance plot and UMAP |
| `results/training.log` | Full training log |

## Project structure

```
├── main.py                     # Single entry point: runs the full pipeline
├── environment.yml             # Conda environment
├── requirements.txt            # Pip dependencies
├── datasets/
│   ├── download.py             # Kaggle API download with retry logic
│   ├── schema.py               # Pydantic metadata validation
│   ├── splits.py               # Patient-grouped stratified k-fold with leakage check
│   ├── exploration.py          # EDA figures
│   ├── isic_dataset.py         # PyTorch Dataset (image, tabular, patient context)
│   ├── samplers.py             # Balanced 1:3 batch sampler
│   └── transforms.py           # Albumentations augmentation
├── models/
│   ├── backbones.py            # EfficientNet-B0 / ResNet-50 / ViT-Tiny classifiers
│   ├── patient_context.py      # Mean and attention context aggregation
│   └── full_pipeline.py        # Multimodal model (image + tabular + context)
├── training/
│   ├── trainer.py              # Multi-seed CV training loop (SWA, AMP, early stopping)
│   ├── losses.py               # Focal loss and online hard negative mining
│   ├── metrics.py              # pAUC, AUC, sensitivity at specificity, F1
│   └── stats.py                # Bootstrap confidence intervals and paired tests
├── baselines/                  # Naive, LightGBM and image-only baselines, comparison plot
├── experiments/                # Configurations A–D and ablation runners
├── analysis/                   # Training curves, master results, ablation, variance, UMAP
├── utils/seed.py               # Global seeding
└── tests/test_metrics.py       # pAUC unit tests
```

## Reproducibility

Seeds are fixed across Python, NumPy and PyTorch (CPU and CUDA), cuDNN runs in deterministic mode, and DataLoader workers get seeded initialisation. Splits are saved as deterministic CSVs, and `main.py` runs everything with zero arguments.

## Limitations and future work

- **Moving-target context:** recomputing context from a still-training encoder makes the residual feature non-stationary. A frozen or decoupled context encoder is the natural next step.
- **Sparse patient neighbourhoods:** subsampling leaves many patients with few reference lesions. Denser per-patient lesion sets, like the wide-field settings where image context has worked before, may change the outcome.
- **Few positives per fold:** each validation fold has only 70–90 malignant cases, so the high-sensitivity region of the ROC is noisy.
- **Richer tabular features:** the ISIC 2024 winners' patient-relative features (e.g. Local Outlier Factor, lesion-to-patient ratios) were not reimplemented and are worth benchmarking.
- **Compute budget:** EfficientNet-B0 at 96 × 96 was chosen to fit a 16 GB laptop, which limits fine-grained feature capacity.

## Tech stack

Python · PyTorch · timm · albumentations · scikit-learn · LightGBM · pydantic · UMAP · pytest · Matplotlib

## Author

**Ravpreet Gill** · [@rav567](https://github.com/rav567)

## References

- Kurtansky, N. R. et al. *The SLICE-3D dataset: 400,000 skin lesion image crops extracted from 3D TBP for skin cancer detection.* Scientific Data, 2024.
- Grob, J. J. & Bonerandi, J. J. *The 'Ugly Duckling' sign: identification of the common characteristics of nevi in an individual as a basis for melanoma screening.* Archives of Dermatology, 1998.
- Ilse, M., Tomczak, J. M. & Welling, M. *Attention-based deep multiple instance learning.* ICML, 2018.
- Lin, T.-Y. et al. *Focal loss for dense object detection.* ICCV, 2017.
