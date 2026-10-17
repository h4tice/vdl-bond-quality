# VDL Cement Bond Quality — Leakage-Free Evaluation

Code for the manuscript *"Deep Learning-Based Bond Quality Assessment from Variable Density
Log (VDL) Waveforms for Wellbore Integrity"* (H. Karaca, A. Tekerek).

Three-class cement bond quality (Good / Partial / Poor) is predicted from VDL waveform
windows of 100 depth × 256 time samples (95 % overlap) from Utah FORGE Well 78B-32. Labels
are derived from CBL and bond-index thresholds. Utah FORGE Well 16B(78)-32 is used as an
external test well.

## Evaluation design

| Issue | Implementation |
|---|---|
| Overlapping windows leaking between train and test | `src/blocked_cv.py`: 10 contiguous depth blocks assigned to 5 folds. Training and validation windows that share any depth row with a test window are purged. Every split asserts a gap of at least 100 rows between partitions. |
| Single split / single seed | Purged blocked 5-fold CV repeated with 5 seeds (42, 123, 456, 789, 2024). Out-of-fold predictions are pooled, so every window is tested exactly once per seed. |
| Normalisation leakage | z-score statistics are computed on the training partition only (`src/rev2_engine.py`). |
| Minority class hidden by weighted metrics | Macro-F1 (primary), balanced accuracy, per-class precision / recall / F1. |
| Statistical comparison | Paired Wilcoxon signed-rank tests with Holm correction. |
| Labels derived from the input measurement | Classical baselines, including a CBL-proxy decision tree on two waveform scalars. Error rate as a function of distance from the CBL/BI thresholds. |
| Generalisation to another well | Train on all of 78B-32, test on 16B(78)-32. |

The random and stratified-depth splits of the original submission are included as reference
protocols (`--protocols random stratdepth`) to measure the leakage effect directly.

## Models

| Model | Parameters |
|---|---|
| 2D-CNN | 422.1K |
| ResNet1D | 248.5K |
| Bi-LSTM | 580.6K |
| AttentionNet | 48.6K |

Baselines: majority class, logistic regression and random forest on 12 engineered waveform
features, and a depth-3 CBL-proxy decision tree. Training uses AdamW (lr 1e-3,
weight decay 1e-4), batch size 16, class-weighted cross-entropy, ReduceLROnPlateau and early
stopping (patience 10, at most 50 epochs). Class-proportional augmentation is optional
(`src/augmentation.py`).

## Main results

Pooled out-of-fold metrics, purged blocked CV, mean ± std over 5 seeds (`results/tables/T1_blocked_cv_main.md`):

| Method | Macro-F1 | Weighted F1 |
|---|---|---|
| AttentionNet + aug | 0.640 ± 0.048 | 0.891 ± 0.034 |
| Bi-LSTM + aug | 0.633 ± 0.009 | 0.897 ± 0.043 |
| ResNet1D + aug | 0.632 ± 0.030 | 0.878 ± 0.028 |
| 2D-CNN + aug | 0.610 ± 0.031 | 0.880 ± 0.046 |
| Engineered features + LR | 0.620 ± 0.022 | 0.837 ± 0.028 |
| Engineered features + RF | 0.582 ± 0.022 | 0.859 ± 0.031 |
| CBL-proxy tree | 0.553 ± 0.033 | 0.746 ± 0.048 |

- The four networks are statistically equivalent (Holm-corrected p ≥ 0.54).
- Leaky splits (random, stratified-depth) give macro-F1 of 0.70–0.83 for the same models.
- In the external well, augmented networks recall 88–97 % of the Poor windows, compared with
  55–77 % for the classical baselines.

All tables (T1–T9) and figures are in `results/`.

## Data

The DLIS files are public in the Geothermal Data Repository. Download them and place them as:

```
data/raw/forge_78b32/UniversityofUtah_Forge78B-32_6Oct21_CBL_Main.dlis                # doi:10.15121/1814488
data/raw/forge_16b32/1044903_UnivUtah_Forge_16B 78 -32_IBC-CBL_MainPass_rev1.dlis    # doi:10.15121/2001059
```

## Reproduce

```bash
python -m venv venv
source venv/bin/activate              # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt
bash run_revision2.sh                 # Windows: .\run_revision2.ps1
```

Individual steps:

```bash
python scripts/rev2_build_data.py                                # DLIS -> data/processed/rev2_<well>.npz
python scripts/rev2_run_cv.py --protocols blocked random stratdepth
python scripts/rev2_run_cv.py --protocols lc --seeds 42 123 456  # learning curve
python scripts/rev2_run_crosswell.py                             # 78B-32 -> 16B(78)-32
python scripts/rev2_run_improve.py                               # focal loss, positional encoding
python scripts/rev2_analysis.py                                  # -> outputs/revision2/{tables,figures}
python scripts/rev2_improve_analysis.py
```

Each run is saved to its own file under `outputs/revision2/runs/`, so an interrupted run
resumes where it stopped. A full run takes about 12–14 hours on an NVIDIA RTX 4060.

## Repository layout

```
config.py                      wells, label thresholds, training and augmentation settings
src/multi_well_loader.py       DLIS reading, null handling, decimation, windowing
src/labeling.py                CBL/BI threshold labels
src/blocked_cv.py              purged blocked K-fold with separation checks
src/splits.py                  random and stratified-depth reference splits
src/rev2_engine.py             training loop, metrics, baseline features
src/models_zoo.py              2D-CNN, ResNet1D, Bi-LSTM, AttentionNet
src/augmentation.py            six waveform augmentations, class-proportional targets
scripts/rev2_*.py              experiment and analysis scripts
results/                       tables and figures reported in the manuscript
```
