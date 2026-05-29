# ELEC0135 AMLS II -- ISIC 2024 Skin Cancer Detection

**Competition:** [ISIC 2024 – Skin Cancer Detection with 3D-TBP](https://www.kaggle.com/competitions/isic-2024-challenge)  
**Task:** Binary classification — benign vs. malignant skin lesions  
**Metric:** Partial AUC above 80% TPR (pAUC)

---

## Research Hypothesis

Patient context — the set of other lesions belonging to the same patient — provides a complementary signal to image features alone, and incorporating it via a learned aggregation module will improve melanoma detection performance over image-only and image+tabular baselines.

---

## Repository Structure

```
main.py                     # Single entry point — runs the full pipeline
environment.yml             # Conda environment specification
requirements.txt            # Python package dependencies
datasets/
  download.py               # Kaggle API download with retry logic
  schema.py                 # Pydantic-based metadata validation
  splits.py                 # Patient-grouped stratified k-fold splitting
  exploration.py            # EDA figures (class distribution, lesions per patient, etc.)
  isic_dataset.py           # PyTorch Dataset with image, tabular, and context support
  samplers.py               # Balanced batch samplers for class imbalance
  transforms.py             # Albumentations augmentation pipelines
models/
  backbones.py              # EfficientNet-B0 / ResNet-50 / ViT-Tiny classifiers
  patient_context.py        # Mean and attention-based context aggregation modules
  full_pipeline.py          # End-to-end multimodal model (image + tabular + context)
training/
  trainer.py                # Multi-seed cross-validated training loop (SWA, AMP, early stopping)
  losses.py                 # Focal loss and OHEM loss
  metrics.py                # pAUC, AUC, sensitivity at specificity, F1
  stats.py                  # Bootstrap confidence intervals and variance summaries
baselines/
  naive.py                  # Majority-class baseline
  tabular.py                # LightGBM on metadata features
  image_only.py             # EfficientNet-B0 image-only baseline
  compare.py                # Baseline comparison table and forest plot
experiments/
  run_configurations.py     # Configs A–D: image / tabular / context / combined
  run_ablations.py          # Ablation: aggregation (mean vs attention) and fusion strategies
analysis/
  training_curves.py        # Training loss and validation pAUC plots
  master_results.py         # Master summary table and dot plot with 95% CIs
  ablation_summary.py       # Ablation bar chart
  variance_analysis.py      # Seed/fold variance plots
  umap_analysis.py          # UMAP projection of learned embeddings
utils/
  seed.py                   # Global seed setting for reproducibility
tests/
  test_metrics.py           # Unit tests for evaluation metrics
results/                    # All outputs saved here (created automatically)
```

---

## Setup

```bash
conda env create -f environment.yml
conda activate amls2-final
```

---

## Running the Pipeline

```bash
python main.py
```

The pipeline is fully automated and requires no user input. It will:

1. Attempt to download the ISIC 2024 dataset via the Kaggle API (requires `~/.kaggle/kaggle.json`). If credentials are absent it falls back to a small synthetic dataset that exercises all code paths.
2. Validate metadata with Pydantic schema checks.
3. Create patient-grouped stratified 5-fold splits (no patient leakage).
4. Generate EDA figures.
5. Run naive and LightGBM tabular baselines.
6. Train all model configurations (A: image-only, B: image+tabular, C: image+context, D: combined).
7. Run ablation studies over aggregation and fusion strategies.
8. Save all result figures and CSV summaries to `results/`.

All random seeds are fixed (seed = 42). Plots are saved as PDFs under `results/figures/` and `results/eda/`.

---

## Kaggle Authentication (for real data)

Place your Kaggle API token at `~/.kaggle/kaggle.json`:

```json
{"username": "your_username", "key": "your_api_key"}
```

The token can be generated from your Kaggle account settings page.

---

## Key Design Choices

- **Class imbalance:** ~0.4% malignant prevalence handled via balanced batch sampling (1:3 ratio) and focal loss (γ = 2).
- **Patient context:** Lesion embeddings from co-located lesions on the same patient are aggregated (mean or attention) and fused with the query embedding via residual difference.
- **Stochastic Weight Averaging (SWA):** Applied in later epochs to improve generalisation.
- **Evaluation metric:** Partial AUC above 80% TPR, matching the competition metric, to reflect clinical relevance at high sensitivity.
