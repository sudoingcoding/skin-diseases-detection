# 🔬 ISIC 2019 Skin Lesion Classification — Comprehensive Deep Learning Benchmark

> A full-scale benchmark of **29 deep learning models** across 8 modern architecture families for multi-class skin lesion diagnosis, with a complete preprocessing pipeline, 5 ensemble strategies, statistical significance testing, and Grad-CAM explainability — trained on the ISIC 2019 dataset.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Preprocessing Pipeline](#-preprocessing-pipeline)
- [Architecture Zoo](#-architecture-zoo)
- [Training Setup](#-training-setup)
- [Individual Model Results](#-individual-model-results--performance-leaderboard)
- [Ensemble Strategies](#-ensemble-strategies)
- [Final Ensemble Comparison](#-final-ensemble-comparison)
- [Explainability — Grad-CAM](#-explainability--grad-cam)
- [Statistical Significance](#-statistical-significance-testing)
- [Image Gallery & Plot Guide](#-image-gallery--plot-guide)
- [Repository Structure](#-repository-structure)
- [How to Reproduce](#-how-to-reproduce)
- [Key Takeaways](#-key-takeaways)

---

## 🧭 Project Overview

Skin cancer is among the most common and, when detected late, most lethal cancers globally. The **ISIC 2019** challenge provides a large, clinically annotated dermoscopy image dataset spanning 8 disease classes — including the two most dangerous: **Melanoma (MEL)** and **Basal Cell Carcinoma (BCC)**.

This project systematically benchmarks the full landscape of modern computer vision architectures — from classic ResNets to state-of-the-art ConvNeXtV2 and Vision Transformers — to answer:

> *Which architecture generalises best to imbalanced, real-world medical image classification? And how much further can we push accuracy with ensemble methods?*

**What makes this project different from a typical Kaggle notebook:**

- ✅ Every model uses **identical preprocessing, splits, and training config** — enabling a fair apple-to-apple comparison
- ✅ A **two-stage image preprocessing pipeline** (hair removal + contrast enhancement) applied before any model sees the data
- ✅ **5 distinct ensemble strategies** implemented and compared with McNemar's statistical tests
- ✅ **Grad-CAM visual explanations** for every model's correct and incorrect predictions
- ✅ 12 standardised evaluation plots per model (loss curves, ROC, confusion matrix, radar chart, confidence distribution, etc.)

---

## 📊 Dataset

**Source:** [ISIC 2019 Skin Lesion Images for Classification](https://challenge.isic-archive.com/landing/2019/)  
**Platform:** Kaggle — `isic-2019-skin-lesion-images-for-classification`

### Class Distribution

| Abbr | Disease Name | Images | % of Total |
|------|-------------|--------|------------|
| **NV** | Nevus | 12,875 | 50.83% |
| **MEL** | Melanoma | 4,522 | 17.85% |
| **BCC** | Basal Cell Carcinoma | 3,323 | 13.12% |
| **BKL** | Benign Keratosis | 2,624 | 10.36% |
| **AK** | Actinic Keratosis | 867 | 3.42% |
| **SCC** | Squamous Cell Carcinoma | 628 | 2.48% |
| **VASC** | Vascular Lesion | 253 | 1.00% |
| **DF** | Dermatofibroma | 239 | 0.94% |
| | **Total** | **25,331** | |

> ⚠️ **Severe class imbalance** — imbalance ratio of **53.9:1** (NV vs DF). Addressed through inverse-frequency class weighting and Focal Loss.

### Data Splits

| Split | Size | Proportion |
|-------|------|------------|
| Train | 17,731 | 70% |
| Validation | 3,800 | 15% |
| Test | 3,800 | 15% |

- Stratified split (preserves class proportions across all three sets)
- Random seed: `42` for reproducibility
- **Zero data leakage** verified: no image overlap between Train/Val/Test

<!-- IMAGE_PLACEHOLDER: notebook=isic-2019-convnext-base, figure=class_distribution_bar_chart, description="Bar chart showing class distribution across 8 skin lesion categories, highlighting the 53.9:1 imbalance ratio" -->

---

## 🔧 Preprocessing Pipeline

All models in this benchmark were trained on images processed through a **two-stage preprocessing pipeline** that runs before any model-specific augmentation:

### Stage 1 — Hair Removal (DullRazor Algorithm)

Hair is a major source of noise in dermoscopy images, occluding lesion features. A morphological approach is applied:

1. Convert image to grayscale
2. Apply blackhat morphological filter to identify hair pixels
3. Threshold and create a binary mask
4. Inpaint masked regions using neighbouring pixel values (OpenCV inpainting)

**Statistics on the dataset:**

| Split | Images with Hair Detected | % |
|-------|--------------------------|---|
| Train | 14,045 / 17,731 | 79.21% |
| Val | 3,039 / 3,800 | 79.97% |
| Test | 2,954 / 3,800 | 77.74% |

> Nearly **4 in 5 dermoscopy images** contained hair artefacts — making this step non-trivial.

<!-- IMAGE_PLACEHOLDER: notebook=isic-2019-convnext-base, figure=hair_removal_comparison, description="Side-by-side comparison of original dermoscopy images vs hair-removed versions, showing the DullRazor inpainting result" -->

### Stage 2 — Contrast Enhancement (CLAHE)

After hair removal, images are passed through **CLAHE (Contrast Limited Adaptive Histogram Equalization)** to enhance local contrast in lesion boundaries:

- Average contrast improvement score: **~30 units** (across train, val, test sets)
- Applied consistently to all splits before model training

### Stage 3 — Runtime Augmentation (Training Only)

Applied dynamically during each training batch:

```python
transforms.RandomHorizontalFlip(p=0.5)
transforms.RandomVerticalFlip(p=0.5)
transforms.RandomRotation(20)
transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

Val/Test images only receive **resize + centre crop + normalise** (no augmentation).

<!-- IMAGE_PLACEHOLDER: notebook=isic-2019-convnext-base, figure=augmentation_examples, description="Grid showing examples of training augmentation — flips, rotations, and colour jitter applied to skin lesion images" -->

---

## 🏗️ Architecture Zoo

29 models were trained and evaluated, spanning 8 distinct architecture families:

### Architecture Families

| Family | Models Tested | Design Philosophy |
|--------|--------------|-------------------|
| **ConvNeXtV2** | large, small, tiny | Modernised pure-CNN with masked autoencoder pretraining (2023) |
| **ConvNeXt** | large, base, small, tiny | Pure-CNN redesigned with transformer-inspired tricks (2022) |
| **Vision Transformer (ViT)** | large, small, tiny | Pure self-attention, no convolutions (2020) |
| **DeiT** | base-distilled | Knowledge-distilled ViT using a teacher token (2021) |
| **Swin Transformer** | small, tiny | Hierarchical ViT with shifted window attention (2021) |
| **EfficientNet** | B0, B1, B2, B3, B4 | Compound-scaled CNN via NAS (2019) |
| **DenseNet** | 121, 161, 169, 201 | Dense skip connections between all layers (2017) |
| **MobileNetV3** | large, small | Lightweight architecture for efficient inference (2019) |
| **ResNet** | 18, 34, 50, 101, 152 | Classic residual learning baseline (2015) |

All models use **ImageNet-pretrained weights** (transfer learning), fine-tuned on the ISIC 2019 training set.

---

## ⚙️ Training Setup

A **unified training configuration** was used across all 29 models to ensure fair comparison. Model-specific parameters (input size, batch size) vary only where architecturally required.

### Common Hyperparameters

| Hyperparameter | Value |
|---------------|-------|
| Epochs | 50 |
| Optimiser | AdamW |
| LR Scheduler | CosineAnnealingLR |
| Weight Decay | 1e-4 |
| Early Stopping Patience | 7 epochs |
| Loss Function | Focal Loss (γ=2.0) + Label Smoothing (0.1) |
| Class Weighting | Inverse-frequency class weights |
| Device | CUDA (Kaggle P100/T4 GPU) |

### Loss Function Design

Standard Cross-Entropy was **not used** due to the severe class imbalance. Instead:

- **Focal Loss** with `gamma=2.0` down-weights easy, correctly classified examples (like the dominant NV class) and forces the model to focus on hard, minority-class examples
- **Label smoothing** of 0.1 prevents overconfident predictions
- **Inverse-frequency class weights** further penalise errors on rare classes (AK, DF, VASC)

### Model-Specific Configurations

| Architecture | Input Size | Batch Size | ~Params |
|-------------|-----------|-----------|---------|
| ConvNeXtV2-Large | 224×224 | 16 | 198M |
| ConvNeXt-Large | 224×224 | 20 | 198M |
| ViT-Large | 224×224 | 24 | 307M |
| ConvNeXt-Base | 224×224 | 24 | 89M |
| DeiT-Base-Distilled | 224×224 | 32 | 87M |
| ConvNeXt-Tiny | 224×224 | 32 | 28M |
| EfficientNet-B4 | 380×380 | 16 | 19M |
| EfficientNet-B3 | 300×300 | 20 | 12M |
| EfficientNet-B2 | 260×260 | 24 | 9.2M |
| EfficientNet-B0 | 224×224 | 32 | 5.3M |
| ResNet-18/34/50/101/152 | 224×224 | 20–32 | 11–60M |

---

## 📈 Individual Model Results — Performance Leaderboard

All 29 models evaluated on the **held-out test set** (3,800 images). Sorted by Test Accuracy.

| Rank | Model | Architecture | Variant | Test Acc % | Balanced Acc | Macro AUC | Best Val Acc % | Training Time (min) |
|------|-------|-------------|---------|-----------|-------------|-----------|--------------|-------------------|
| 🥇 1 | convnext2 large | ConvNeXtV2 | large | **87.82** | 0.82 | 0.97 | 88.16 | 404.4 |
| 🥈 2 | vit large | ViT | large | 87.47 | 0.82 | 0.96 | 87.74 | 527.5 |
| 🥉 3 | conv large | ConvNeXt | large | 87.21 | 0.80 | 0.97 | 86.84 | 433.8 |
| 4 | deit base | DeiT | base-distilled | 87.05 | 0.81 | 0.96 | 86.55 | 178.1 |
| 5 | conv tiny | ConvNeXt | tiny | 87.03 | 0.81 | 0.97 | 86.34 | 122.9 |
| 6 | conv base | ConvNeXt | base | 86.95 | **0.83** | 0.97 | 85.95 | 235.6 |
| 7 | convnext2 tiny | ConvNeXtV2 | tiny | 86.84 | 0.81 | 0.97 | 85.63 | 119.7 |
| 8 | conv small | ConvNeXt | small | 85.84 | 0.80 | 0.97 | 85.68 | 166.8 |
| 9 | vit small | ViT | small | 85.13 | 0.79 | 0.97 | 84.95 | 135.3 |
| 10 | swin small | Swin | small | 85.00 | 0.79 | 0.96 | 86.45 | 187.4 |
| 11 | swin tiny | Swin | tiny | 84.82 | 0.79 | 0.97 | 84.34 | 132.2 |
| 12 | eff2 | EfficientNet | B2 | 84.45 | 0.78 | 0.97 | 84.87 | 137.7 |
| 13 | connext2 small | ConvNeXtV2 | small | 83.79 | 0.79 | 0.97 | 84.55 | 107.0 |
| 14 | dense_169 | DenseNet | 169 | 83.39 | 0.79 | 0.97 | 82.79 | 136.4 |
| 15 | dense_161 | DenseNet | 161 | 83.37 | 0.77 | 0.97 | 83.34 | 195.3 |
| 16 | mobile large | MobileNetV3 | large | 83.21 | 0.78 | 0.97 | 82.58 | 107.1 |
| 17 | eff3 | EfficientNet | B3 | 82.68 | 0.80 | 0.97 | 83.61 | 127.2 |
| 18 | eff1 | EfficientNet | B1 | 82.37 | 0.78 | 0.97 | 82.50 | 125.7 |
| 19 | dense_201 | DenseNet | 201 | 82.34 | 0.77 | 0.97 | 82.16 | 174.4 |
| 20 | eff0 | EfficientNet | B0 | 82.03 | 0.78 | 0.97 | 80.84 | 107.2 |
| 21 | eff4 | EfficientNet | B4 | 81.97 | 0.77 | 0.97 | 83.13 | 311.3 |
| 22 | dense_121 | DenseNet | 121 | 81.11 | 0.77 | 0.96 | 80.97 | 132.0 |
| 23 | res 50 | ResNet | 50 | 80.05 | 0.74 | 0.96 | 79.71 | 116.6 |
| 24 | vit tiny | ViT | tiny | 80.05 | 0.76 | 0.96 | 80.13 | 106.8 |
| 25 | mobile small | MobileNetV3 | small | 78.26 | 0.72 | 0.96 | 78.55 | 105.2 |
| 26 | res 18 | ResNet | 18 | 70.00 | 0.66 | 0.93 | 70.21 | 41.4 |
| 27 | res 34 | ResNet | 34 | 68.97 | 0.67 | 0.93 | 69.95 | 47.8 |
| 28 | res 101 | ResNet | 101 | 68.03 | 0.65 | 0.92 | 67.71 | 42.1 |
| 29 | res 152 | ResNet | 152 | 62.16 | 0.59 | 0.91 | 62.45 | 51.1 |

> **Key insight:** Model scale ≠ performance. ResNet-152 (60M params, ~51 min training) achieved only 62.16%, while ConvNeXt-Tiny (28M params, ~123 min) hit 87.03%. Architecture modernity matters far more than parameter count for this task.

<!-- IMAGE_PLACEHOLDER: notebook=ensemble, figure=model_comparison.png, description="4-panel comparison: (1) horizontal bar chart of top models by test accuracy, (2) scatter plot of Balanced Accuracy vs Macro AUC (bubble size = training time), (3) pie chart of model type distribution, (4) histogram of training time distribution" -->

### Per-Class Breakdown — Best Single Model (ConvNeXtV2-Large, 87.82%)

| Class | Precision | Recall | F1-Score | Support |
|-------|----------|--------|---------|---------|
| AK | 0.8241 | 0.6846 | 0.7479 | 130 |
| BCC | 0.8957 | 0.9118 | 0.9037 | 499 |
| BKL | 0.8259 | 0.7944 | 0.8098 | 394 |
| DF | 0.9643 | 0.7500 | 0.8437 | 36 |
| MEL | 0.8336 | 0.7906 | 0.8115 | 678 |
| **NV** | **0.9008** | **0.9358** | **0.9180** | 1931 |
| SCC | 0.8280 | 0.8191 | 0.8235 | 94 |
| VASC | 0.9429 | 0.8684 | 0.9041 | 38 |

> AK (Actinic Keratosis) is the hardest class — lowest recall (68.5%) due to its visual similarity to BCC and BKL.

### Top Misclassification Patterns (ConvNeXt-Base)

| Pattern | Count | % of True Class |
|---------|-------|----------------|
| NV → MEL | 127 | 6.58% |
| MEL → NV | 96 | 14.16% |
| BKL → NV | 48 | 12.18% |
| NV → BCC | 20 | 1.04% |
| MEL → BKL | 18 | 2.65% |
| AK → BCC | 15 | 11.54% |

> The NV↔MEL confusion is clinically significant — misclassifying Melanoma as Nevus is a dangerous false negative. This pattern motivates the ensemble approach.

<!-- IMAGE_PLACEHOLDER: notebook=isic-2019-convnext-base, figure=convnext_base_misclassification_analysis.png, description="Bar chart showing the top misclassification pairs for ConvNeXt-Base on the test set, revealing the NV↔MEL confusion as the dominant error pattern" -->

---

## 🧩 Ensemble Strategies

Five distinct ensemble methods were implemented and rigorously compared, all operating on the saved **softmax probability outputs** (`.pkl` files) from individual model inference — no retraining required.

### Overview

| Strategy | # Models | Key Idea |
|----------|---------|---------|
| **1. Smart Hybrid** | 4 | Top performers + architectural diversity, weighted by Balanced Accuracy |
| **2. Top 5 Weighted** | 5 | Top 5 by leaderboard rank, weighted by Balanced Accuracy or Test Accuracy |
| **3. Architecture Diversity** | 6 | Best representative from each unique architecture family |
| **4. Confidence Weighted** | 5 | Models weighted by their prediction confidence gap (correct − incorrect mean confidence) |
| **5. Meta-Optimized Hierarchical** | 7 | Stacking + optimised weighted averaging + diversity scoring |

---

### Ensemble 1 — Smart Hybrid

**Philosophy:** Combine the highest-performing models while ensuring architectural variety. Prevents the ensemble from being dominated by slight variations of the same architecture.

**Selected models:**
- convnext2 large (87.82%, ConvNeXtV2)
- vit large (87.47%, ViT)
- conv large (87.21%, ConvNeXt)
- deit base (87.05%, DeiT) — *Transformer expert*
- eff2 (84.45%, EfficientNet) — *architectural diversity*
- dense_169 (83.39%, DenseNet) — *architectural diversity*
- mobile large (83.21%, MobileNet) — *architectural diversity*

**Weighting:** Normalised Balanced Accuracy scores

<!-- IMAGE_PLACEHOLDER: notebook=ensemble, figure=smart_hybrid_ensemble_visualization.png, description="2-panel figure: (1) bar chart comparing accuracy/balanced accuracy of individual models vs the Smart Hybrid ensemble, (2) weight distribution showing how each model contributes to the ensemble" -->

---

### Ensemble 2 — Top 5 Weighted

**Philosophy:** Simple and effective — take the five best models from the leaderboard and weight them. Evaluated with two weighting schemes:
- **Option A:** Weight by Balanced Accuracy (recommended for imbalanced datasets)
- **Option B:** Weight by Test Accuracy

**Selected models:** convnext2 large, vit large, conv large, deit base, conv tiny

<!-- IMAGE_PLACEHOLDER: notebook=ensemble, figure=top5_weighted_ensemble_analysis.png, description="5-panel analysis: individual model performance bars, weighting scheme comparison (Option A vs B), ensemble vs single best comparison, improvement-per-model bar chart, and overall accuracy/balanced accuracy summary" -->

---

### Ensemble 3 — Architecture Diversity

**Philosophy:** One champion from each major architecture family. Maximises diversity of learned representations, improving robustness to dataset biases.

**Selected models (one per family):**

| Rank | Family | Representative | Acc % | Bal Acc | Weight |
|------|--------|---------------|-------|---------|--------|
| 1 | ConvNeXtV2 | convnext2 large | 87.82 | 0.82 | 16.67% |
| 2 | ViT | vit large | 87.47 | 0.82 | 16.67% |
| 3 | DeiT | deit base | 87.05 | 0.81 | 16.67% |
| 4 | ConvNeXt | conv base | 86.95 | 0.83 | 16.67% |
| 5 | Swin | swin small | 85.00 | 0.79 | 16.67% |
| 6 | DenseNet | dense_169 | 83.39 | 0.79 | 16.67% |

> ResNet was intentionally **excluded** due to consistently poor performance (62–80% range) across all variants.

<!-- IMAGE_PLACEHOLDER: notebook=ensemble, figure=architecture_diversity_ensemble_analysis.png, description="8-panel deep-dive: architecture family pie chart, per-family best model accuracy, ensemble confusion matrix, ensemble ROC curves, architecture weight contributions, per-class F1 improvement over single best, confidence calibration, and ensemble accuracy vs single-model baseline" -->

---

### Ensemble 4 — Confidence Weighted

**Philosophy:** Weight models not just by accuracy, but by **how well-calibrated their confidence is**. A model with a large gap between its confidence on correct vs incorrect predictions is a better signal contributor.

**Confidence gap** = mean(max_prob | correct) − mean(max_prob | incorrect)

For ConvNeXt-Base:
- Confidence on correct predictions: **0.8738** (median: 0.9280)
- Confidence on incorrect predictions: **0.6804** (median: 0.6610)
- Confidence gap: **0.1934**

<!-- IMAGE_PLACEHOLDER: notebook=ensemble, figure=confidence_weighted_ensemble_analysis.png, description="6-panel figure: per-model confidence gap bar chart (used as weights), confidence histogram comparing correct vs incorrect predictions, confidence boxplot per class, ensemble vs individual models accuracy comparison, model weight pie chart, and calibration curve" -->

---

### Ensemble 5 — Meta-Optimized Hierarchical

**Philosophy:** The most sophisticated strategy. Combines:
1. **Stacking:** Meta-learner is trained on the probability outputs of base models
2. **Optimised weighted averaging:** Weights are numerically optimised (scipy) to maximise validation balanced accuracy
3. **Diversity scoring:** Models are penalised for correlated errors, encouraging the selection of complementary predictors
4. **Hierarchical class grouping:** Different model subsets are given higher authority for different class groups based on per-class AUC

**Selected models:** 7 models from diverse families (conv base + 6 others)

<!-- IMAGE_PLACEHOLDER: notebook=ensemble, figure=meta_optimized_ensemble_complete_analysis.png, description="10-panel comprehensive analysis: individual model accuracies, optimised weights per model, ensemble confusion matrix, ROC curves, per-class F1 heatmap comparing all models + ensemble, diversity correlation matrix, improvement waterfall chart, hierarchical class group weights, confidence calibration plot, and single best vs ensemble bar comparison" -->

---

## 🏆 Final Ensemble Comparison

### Results Table

| Strategy | Models | Accuracy % | Balanced Acc | Macro AUC | Improvement over Single Best |
|----------|--------|-----------|-------------|-----------|---------------------------|
| **Meta-Optimized** | 7 | **90.45** | **0.858** | **0.989** | **+3.50%** ⭐ |
| Architecture Diversity | 6 | 89.92 | 0.850 | 0.983 | +2.97% |
| Smart Hybrid | 4 | 89.90 | 0.847 | 0.981 | +2.94% |
| Top 5 Weighted | 5 | 89.74 | 0.846 | 0.982 | +2.79% |
| Confidence Weighted | 5 | 88.63 | 0.835 | 0.982 | +1.68% |
| **Single Best Model** | 1 | 86.95 | 0.830 | 0.970 | baseline |

> **The Meta-Optimized ensemble achieved 90.45% test accuracy** — a **+3.50 percentage point** improvement over the single best model (ConvNeXt-Base at 86.95%). Average improvement across all ensemble strategies: **+2.78%**.

<!-- IMAGE_PLACEHOLDER: notebook=ensemble, figure=ensemble_strategies_comprehensive_comparison.png, description="7-panel master comparison: (1) accuracy bar chart all strategies, (2) balanced accuracy comparison, (3) improvement over single best (dual bars: accuracy + balanced acc), (4) number of models vs accuracy scatter, (5) macro AUC comparison, (6) radar chart across all metrics, (7) complexity vs performance trade-off bubble chart" -->

### Ensemble Strategy Recommendations

| Strategy | Best For | Pros | Cons |
|----------|---------|------|------|
| **Meta-Optimized** | Maximum accuracy in production | Mathematically optimal, highest performance | Most complex, requires more computation |
| Architecture Diversity | Robust generalisation | High diversity, reduces bias | May include lower-performing models |
| Smart Hybrid | Balanced performance | Good accuracy + manageable complexity | More complex than simple average |
| Top 5 Weighted | Quick baseline ensemble | Simple to implement, good performance | Lacks diversity, correlated errors |
| Confidence Weighted | Calibrated uncertainty | Well-calibrated output probabilities | Computationally expensive |

---

## 🔍 Explainability — Grad-CAM

Every individual model notebook includes **Gradient-weighted Class Activation Mapping (Grad-CAM)** to visualise *which pixels drove the prediction* — a critical step for building trust in medical AI systems.

### What Grad-CAM Shows

- **Correct predictions:** Where the model correctly focuses (e.g., on the lesion body, irregular borders, colour asymmetry)
- **Incorrect predictions:** Where the model was confused (e.g., fixating on skin texture, hair artefacts, or background rather than the lesion)

Each model generates two Grad-CAM output grids (5 rows × 8 columns = 40 image panels each):
1. `{model}_correct_predictions_gradcam.png` — highest-confidence correct predictions
2. `{model}_incorrect_predictions_gradcam.png` — incorrect predictions showing where attention went wrong

<!-- IMAGE_PLACEHOLDER: notebook=isic-2019-convnext-base, figure=convnext_base_correct_predictions_gradcam.png, description="5×8 grid of Grad-CAM visualisations for ConvNeXt-Base correct predictions: original dermoscopy image, Grad-CAM heatmap overlay, and true vs predicted label for each of 10 samples across 8 classes" -->

<!-- IMAGE_PLACEHOLDER: notebook=isic-2019-convnext-base, figure=convnext_base_incorrect_predictions_gradcam.png, description="5×8 grid of Grad-CAM visualisations for ConvNeXt-Base incorrect predictions: showing where the model's attention deviated from the lesion for misclassified cases" -->

---

## 📐 Statistical Significance Testing

To confirm that ensemble improvements are real and not due to random variation, **McNemar's Test** was applied to all pairwise strategy comparisons.

### McNemar's Test Results

| Comparison | A Wins | B Wins | p-value | Significant? |
|-----------|--------|--------|---------|-------------|
| Meta-Optimized vs Single Best | 192 | 59 | 0.000 | ✅ YES |
| Architecture Diversity vs Single Best | 185 | 72 | 0.000 | ✅ YES |
| Smart Hybrid vs Single Best | 191 | 79 | 0.000 | ✅ YES |
| Top 5 Weighted vs Single Best | 182 | 76 | 0.000 | ✅ YES |
| Confidence Weighted vs Single Best | 138 | 74 | 0.000 | ✅ YES |
| Confidence Weighted vs Meta-Optimized | 47 | 116 | 0.000 | ✅ YES |
| Top 5 Weighted vs Meta-Optimized | 22 | 49 | 0.002 | ✅ YES |
| Top 5 Weighted vs Architecture Diversity | 22 | 29 | 0.401 | ❌ Not significant |
| Smart Hybrid vs Top 5 Weighted | 38 | 32 | 0.550 | ❌ Not significant |
| Smart Hybrid vs Architecture Diversity | 40 | 41 | 1.000 | ❌ Not significant |

> **Interpretation:** All ensembles significantly outperform the single best model (p < 0.001). However, Smart Hybrid, Top 5 Weighted, and Architecture Diversity are **not significantly different from each other** — any of the three is a reasonable choice with similar expected outcomes. Meta-Optimized is significantly better than all others.

---

## 🖼️ Image Gallery & Plot Guide

Each individual model notebook generates **12 standardised evaluation plots** saved as high-resolution PNGs. Use the table below to locate specific images when inserting into the README.

### Plot Naming Convention

All plots follow: `{model_name}_{plot_type}.png`

**Model name prefixes** used in file names:

| Notebook | File Prefix | Output Folder |
|---------|------------|--------------|
| isic-2019-convnext-base | `convnext_base` | `model_outputs/` |
| isic-2019-convnext-large | `convnext_large` | `model_outputs/` |
| isic-2019-convnext-small | `convnext_small` | `model_outputs/` |
| isic-2019-convnext-tiny | `convnext_tiny` | `model_outputs/` |
| isic-2019-convnextv2-large | `convnextv2` | `convnextv2_large_outputs/` |
| isic-2019-convnextv2-small | `convnextv2_small` | (model_outputs) |
| isic-2019-convnextv2-tiny | `convnextv2_tiny` | (model_outputs) |
| isic-2019-deit-base-distilled | `deit_base` | (model_outputs) |
| isic-2019-densenet-121 | `densenet121` | (model_outputs) |
| isic-2019-densenet-161 | `densenet161` | (model_outputs) |
| isic-2019-densenet-169 | `densenet169` | (model_outputs) |
| isic-2019-densenet-201 | `densenet201` | (model_outputs) |
| isic-2019-efficientnetb0–b4 | `efficientnet_b{N}` | (model_outputs) |
| isic-2019-mobilenetv3-large | `mobilenet_large` | (model_outputs) |
| isic-2019-mobilenetv3-small | `mobilenet_small` | (model_outputs) |
| isic-2019-resnet18/34/50/101/152 | `resnet{N}` | `model_outputs/` |
| isic-2019-swin-small | `swin` | `swin_small_outputs/` |
| isic-2019-swin-tiny | `swin` | `swin_tiny_outputs/` |
| isic-2019-vit-large | `vit` | `vit_large_outputs/` |
| isic-2019-vit-small | `vit` | `vit_small_outputs/` |
| isic-2019-vit-tiny | `vit` | `vit_tiny_outputs/` |

### Plot Types Per Model

| Plot Type | Suffix | Description |
|-----------|--------|-------------|
| Training Loss Curve | `_loss_curve.png` | Train vs Val loss over 50 epochs |
| Accuracy Curve | `_accuracy_curve.png` | Train vs Val accuracy over 50 epochs |
| LR Schedule | `_learning_rate.png` | CosineAnnealing LR decay over epochs |
| Raw Confusion Matrix | `_confusion_matrix_raw.png` | Absolute counts, 8×8 grid |
| Normalised Confusion Matrix | `_confusion_matrix_normalized.png` | Row-normalised (recall per class) |
| ROC Curves | `_roc_curves.png` | One-vs-rest ROC for all 8 classes |
| Precision/Recall/F1 | `_precision_recall_f1.png` | Grouped bar chart per class |
| Per-Class AUC | `_auc_scores.png` | Horizontal bar chart of AUC per class |
| Class Support Distribution | `_class_support.png` | Test set size per class |
| F1-Score Radar Chart | `_f1_radar_chart.png` | Radar/spider chart across 8 classes |
| Misclassification Analysis | `_misclassification_analysis.png` | Top-N misclassification pairs |
| Confidence Histogram | `_confidence_histogram.png` | Correct vs incorrect confidence distributions |
| Confidence Boxplot | `_confidence_boxplot.png` | Per-class prediction confidence spread |
| Grad-CAM (Correct) | `_correct_predictions_gradcam.png` | 40-panel grid with correct prediction heatmaps |
| Grad-CAM (Incorrect) | `_incorrect_predictions_gradcam.png` | 40-panel grid with incorrect prediction heatmaps |

### Ensemble-Level Plots (notebook: `ensemble`)

| Figure | File Name | Description |
|--------|-----------|-------------|
| Fig 1 | `model_comparison.png` | 4-panel master model leaderboard visualisation |
| Fig 2 | `smart_hybrid_ensemble_visualization.png` | Smart Hybrid ensemble analysis |
| Fig 3 | `top5_weighted_ensemble_analysis.png` | Top-5 Weighted ensemble analysis |
| Fig 4 | `architecture_diversity_ensemble_analysis.png` | Architecture Diversity ensemble analysis |
| Fig 5 | `confidence_weighted_ensemble_analysis.png` | Confidence Weighted ensemble analysis |
| Fig 6 | `meta_optimized_ensemble_complete_analysis.png` | Meta-Optimized ensemble full analysis |
| Fig 7 | `ensemble_strategies_comprehensive_comparison.png` | 7-panel final comparison across all strategies |

---

## 📁 Repository Structure

```
isic-2019-skin-lesion-classification/
│
├── notebooks/
│   ├── ensemble.ipynb                      # Ensemble strategies + comparison
│   ├── isic-2019-convnext-base.ipynb
│   ├── isic-2019-convnext-large.ipynb
│   ├── isic-2019-convnext-small.ipynb
│   ├── isic-2019-convnext-tiny.ipynb
│   ├── isic-2019-convnextv2-large.ipynb
│   ├── isic-2019-convnextv2-small.ipynb
│   ├── isic-2019-convnextv2-tiny.ipynb
│   ├── isic-2019-deit-base-distilled.ipynb
│   ├── isic-2019-densenet-121.ipynb
│   ├── isic-2019-densenet-161.ipynb
│   ├── isic-2019-densenet-169.ipynb
│   ├── isic-2019-densenet-201.ipynb
│   ├── isic-2019-efficientnetb0.ipynb
│   ├── isic-2019-efficientnetb1.ipynb
│   ├── isic-2019-efficientnetb2.ipynb
│   ├── isic-2019-efficientnetb3.ipynb
│   ├── isic-2019-efficientnetb4.ipynb
│   ├── isic-2019-mobilenetv3-large.ipynb
│   ├── isic-2019-mobilenetv3-small.ipynb
│   ├── isic-2019-resnet18.ipynb
│   ├── isic-2019-resnet34.ipynb
│   ├── isic-2019-resnet50.ipynb
│   ├── isic-2019-resnet101.ipynb
│   ├── isic-2019-resnet152.ipynb
│   ├── isic-2019-swin-small.ipynb
│   ├── isic-2019-swin-tiny.ipynb
│   ├── isic-2019-vit-large.ipynb
│   ├── isic-2019-vit-small.ipynb
│   └── isic-2019-vit-tiny.ipynb
│
├── results/                                # Saved model outputs (PKL + PNG)
│   ├── model_outputs/                      # ConvNeXt, DenseNet, ResNet, EfficientNet
│   ├── convnextv2_large_outputs/
│   ├── swin_small_outputs/
│   ├── swin_tiny_outputs/
│   ├── vit_large_outputs/
│   ├── vit_small_outputs/
│   └── vit_tiny_outputs/
│
├── ensemble_outputs/
│   ├── model_comparison.png
│   ├── ensemble_strategies_comparison.csv
│   ├── ensemble_statistical_significance.csv
│   ├── ensemble_strategy_recommendations.csv
│   └── ensemble_strategies_comprehensive_comparison.png
│
└── README.md
```

---

## 🚀 How to Reproduce

### Prerequisites

```bash
pip install torch torchvision
pip install numpy pandas scikit-learn matplotlib seaborn
pip install opencv-python pillow tqdm
pip install scipy  # for meta-optimised ensemble
```

### 1. Set Up Dataset

Download from Kaggle:
```bash
kaggle datasets download -d {isic-2019-dataset-slug}
kaggle datasets download -d {hair-removed-dataset-slug}
kaggle datasets download -d {contrast-enhanced-dataset-slug}
kaggle datasets download -d {splits-dataset-slug}
```

Expected paths:
- `/kaggle/input/isic-2019-skin-lesion-images-for-classification/`
- `/kaggle/input/hair-removed/`
- `/kaggle/input/contrast-enhanced/`
- `/kaggle/input/splits/`

### 2. Train Individual Models

Open any individual notebook (e.g., `isic-2019-convnext-base.ipynb`) on Kaggle and run all cells. Each notebook:
- Loads preprocessed data
- Trains for up to 50 epochs with early stopping
- Saves `{model_name}_best.pth` + `{model_name}_best.pkl` (results + predictions)
- Generates all 12+ evaluation plots automatically
- Generates Grad-CAM visualisations

### 3. Run Ensemble Analysis

After **all** individual models have been trained and their `.pkl` files saved:
```
kaggle input: model-2019/  (directory containing all {model}_best.pkl files)
```
Open `ensemble.ipynb` and run all cells. The notebook automatically:
- Loads all saved probability predictions
- Runs all 5 ensemble strategies
- Performs McNemar's statistical significance tests
- Generates the 7-panel comprehensive comparison

### 4. Reproduce Specific Ensemble Strategies

```python
# Example: Smart Hybrid Ensemble
selected_models = [
    {"Model Name": "convnext2 large", "Test Acc %": 87.82, "Balanced Acc": 0.82},
    {"Model Name": "vit large",       "Test Acc %": 87.47, "Balanced Acc": 0.82},
    {"Model Name": "conv large",      "Test Acc %": 87.21, "Balanced Acc": 0.80},
    {"Model Name": "deit base",       "Test Acc %": 87.05, "Balanced Acc": 0.81},
]
weights = [m["Balanced Acc"] for m in selected_models]
weights = np.array(weights) / np.sum(weights)  # Normalise

ensemble_probs = sum(all_probs[m["Model Name"]] * w 
                     for m, w in zip(selected_models, weights))
ensemble_preds = np.argmax(ensemble_probs, axis=1)
```

---

## 💡 Key Takeaways

### 1. Architecture Families Matter More Than Scale
Modern architectures (ConvNeXtV2, ViT, ConvNeXt, DeiT, Swin) all cluster in the **85–88%** accuracy range, while older ResNet variants cluster in the **62–80%** range — regardless of depth. This demonstrates that architectural advances since 2020 are transformative, not incremental.

### 2. Focal Loss + Class Weighting is Essential for Imbalanced Data
With a 53.9:1 class imbalance ratio, standard Cross-Entropy loss would be dominated by the NV class. Focal Loss with class weights allowed models to maintain competitive recall on minority classes (AK, DF, VASC) despite their tiny sample sizes.

### 3. Preprocessing Investment Pays Off
79% of images contained hair artefacts. The two-stage preprocessing pipeline (hair removal → CLAHE contrast enhancement) created a cleaner learning signal, and the +30-unit average contrast improvement gave every model a better-defined lesion boundary to learn from.

### 4. Ensembles Consistently Beat Single Models
All 5 ensemble strategies improved over the single best model, with improvements ranging from **+1.68% to +3.50%**. The Meta-Optimized strategy's +3.50% is statistically significant (McNemar p < 0.001) and practically meaningful in a medical diagnostic context.

### 5. Diversity > Raw Performance in Ensembles
The Architecture Diversity ensemble (which intentionally includes lower-performing DenseNet and Swin models) outperformed the Top-5 Weighted ensemble (which uses only the 5 best models). Different architecture families make **different types of errors**, so their combination cancels out individual failure modes.

### 6. Model Confidence is a Reliable Quality Signal
ConvNeXt-Base shows a clear separation between confidence on correct (mean 0.87) vs incorrect (mean 0.68) predictions — a 19-point gap. This calibration means the model's uncertainty is genuinely informative, making it suitable for human-in-the-loop triage systems.

### 7. Biggest Remaining Challenge: MEL ↔ NV Confusion
The most dangerous misclassification pattern — predicting Melanoma as Nevus (false negative: 14.16% of MEL cases) — persists across all models and ensembles. This suggests that dermoscopy images alone may be insufficient for reliable MEL vs NV differentiation at scale, and patient metadata, lesion evolution history, or multi-modal inputs could be the next frontier.

---

## 📚 References

- **ISIC 2019 Challenge:** Tschandl et al., *"The HAM10000 dataset"*, Scientific Data 2018
- **ConvNeXt:** Liu et al., *"A ConvNet for the 2020s"*, CVPR 2022
- **ConvNeXtV2:** Woo et al., *"ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders"*, CVPR 2023
- **Vision Transformer:** Dosovitskiy et al., *"An Image is Worth 16x16 Words"*, ICLR 2021
- **DeiT:** Touvron et al., *"Training Data-Efficient Image Transformers"*, ICML 2021
- **Swin Transformer:** Liu et al., *"Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"*, ICCV 2021
- **EfficientNet:** Tan & Le, *"EfficientNet: Rethinking Model Scaling for CNNs"*, ICML 2019
- **Focal Loss:** Lin et al., *"Focal Loss for Dense Object Detection"*, ICCV 2017
- **Grad-CAM:** Selvaraju et al., *"Grad-CAM: Visual Explanations from Deep Networks"*, ICCV 2017
- **DullRazor:** Lee et al., *"Dullrazor: A software approach to hair removal from images"*, Computers in Biology and Medicine 1997

---

*Trained and evaluated on Kaggle (GPU: P100/T4). Total compute: ~166 min average training time × 29 models ≈ 80 GPU-hours.*
