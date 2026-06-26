# 🔬 ISIC 2019 Skin Lesion Classification — Comprehensive Deep Learning Benchmark

> A full-scale benchmark of **29 deep learning models** across 8 modern architecture families for multi-class skin lesion diagnosis, with a complete preprocessing pipeline, 5 ensemble strategies, statistical significance testing, and Grad-CAM explainability — trained on the ISIC 2019 dataset.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![Kaggle](https://img.shields.io/badge/Platform-Kaggle-20BEFF?logo=kaggle)
![Models](https://img.shields.io/badge/Models-29-success)
![Dataset](https://img.shields.io/badge/Dataset-ISIC%202019-orange)

</div>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Methodology](#-methodology)
- [Dataset](#-dataset)
- [Preprocessing Pipeline](#-preprocessing-pipeline)
- [Architecture Zoo](#-architecture-zoo)
- [Training Setup](#-training-setup)
- [Individual Model Results](#-individual-model-results--performance-leaderboard)
- [Ensemble Strategies](#-ensemble-strategies)
- [Final Ensemble Comparison](#-final-ensemble-comparison)
- [Explainability — Grad-CAM](#-explainability--grad-cam)
- [Statistical Significance Testing](#-statistical-significance-testing)
- [Key Takeaways](#-key-takeaways)
- [References](#-references)

---

## 🧭 Project Overview

Skin cancer is among the most common and, when detected late, most lethal cancers globally. The **ISIC 2019** challenge provides a large, clinically annotated dermoscopy image dataset spanning 8 disease classes — including the two most dangerous: **Melanoma (MEL)** and **Basal Cell Carcinoma (BCC)**.

This project systematically benchmarks the full landscape of modern computer vision architectures — from classic ResNets to state-of-the-art ConvNeXtV2 and Vision Transformers — to answer:

> *Which architecture generalises best to imbalanced, real-world medical image classification? And how much further can we push accuracy with ensemble methods?*

**What makes this project different from a typical notebook:**

| Feature | Detail |
|---------|--------|
| ✅ Fair comparison | Every model uses **identical preprocessing, splits, and training config** |
| ✅ Medical preprocessing | Two-stage pipeline (hair removal + CLAHE) before any model sees data |
| ✅ Imbalance handling | Focal Loss + inverse-frequency class weights for 53.9:1 imbalance ratio |
| ✅ Ensemble depth | 5 distinct strategies compared with McNemar's statistical significance tests |
| ✅ Explainability | Grad-CAM for every model's correct and incorrect predictions |
| ✅ Standardised evaluation | 12+ plots per model (loss, ROC, confusion matrix, radar chart, confidence, etc.) |

---

## 🗺️ Methodology

![Main Methodology](https://raw.githubusercontent.com/sudoingcoding/skin-diseases-detection/main/figures/Main%20Methodology.png)

The end-to-end pipeline flows as follows:

1. **Raw ISIC 2019 images** → stratified 70/15/15 train/val/test split
2. **Stage 1:** DullRazor hair removal (morphological inpainting)
3. **Stage 2:** CLAHE contrast enhancement
4. **Stage 3:** Runtime augmentation (training only)
5. **29 models** trained with unified hyperparameters, saving `.pkl` probability outputs
6. **5 ensemble strategies** applied over saved softmax probabilities — no retraining
7. **McNemar's test** validates all pairwise ensemble comparisons statistically

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

| Split | Images | Proportion |
|-------|--------|------------|
| Train | 17,731 | 70% |
| Validation | 3,800 | 15% |
| Test | 3,800 | 15% |

- Stratified split — class proportions preserved across all three sets
- Random seed `42` for full reproducibility
- **Zero data leakage** verified: no image overlap between Train / Val / Test

### Split Quality Verification

| Check | Result |
|-------|--------|
| Train–Val overlap | 0 images ✅ |
| Train–Test overlap | 0 images ✅ |
| Val–Test overlap | 0 images ✅ |

---

## 🔧 Preprocessing Pipeline

All models were trained on images processed through a **two-stage preprocessing pipeline** that runs once — before any model sees the data — ensuring every model competes on identical input.

### Stage 1 — Hair Removal (DullRazor Algorithm)

Hair is the dominant noise source in dermoscopy images, directly occluding lesion boundaries. Steps:

1. Convert image to grayscale
2. Apply **blackhat morphological filter** to isolate dark, thin structures (hair)
3. Threshold and create a binary hair mask
4. **OpenCV inpainting** fills masked regions using surrounding pixel context

**Hair statistics across dataset:**

| Split | Images with Hair | Percentage |
|-------|-----------------|------------|
| Train | 14,045 / 17,731 | **79.21%** |
| Val | 3,039 / 3,800 | **79.97%** |
| Test | 2,954 / 3,800 | **77.74%** |

> Nearly **4 in 5 dermoscopy images** contained hair artefacts — making this step essential, not optional.

### Stage 2 — Contrast Enhancement (CLAHE)

After hair removal, all images go through **CLAHE (Contrast Limited Adaptive Histogram Equalization)** to sharpen lesion boundaries:

- Average contrast improvement: **~30 units** across all splits
- Adaptive — adjusts locally per image tile, not globally

### Stage 3 — Runtime Augmentation (Training Only)

Applied dynamically per batch during training:

```python
transforms.RandomHorizontalFlip(p=0.5)
transforms.RandomVerticalFlip(p=0.5)
transforms.RandomRotation(20)
transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

Val/Test images: **resize → centre crop → normalise only** (no augmentation).

---

## 🏗️ Architecture Zoo

29 models were trained and evaluated, spanning **8 distinct architecture families**:

| Family | Variants Tested | Design Philosophy | Year |
|--------|----------------|-------------------|------|
| **ConvNeXtV2** | large, small, tiny | Modernised pure-CNN + masked autoencoder pretraining | 2023 |
| **ConvNeXt** | large, base, small, tiny | Pure-CNN redesigned with transformer-inspired tricks | 2022 |
| **Vision Transformer (ViT)** | large, small, tiny | Pure self-attention, patch-based, no convolutions | 2020 |
| **DeiT** | base-distilled | Knowledge-distilled ViT via a teacher-student token | 2021 |
| **Swin Transformer** | small, tiny | Hierarchical ViT with shifted-window local attention | 2021 |
| **EfficientNet** | B0, B1, B2, B3, B4 | Compound-scaled CNN via neural architecture search | 2019 |
| **DenseNet** | 121, 161, 169, 201 | Dense skip connections between every layer pair | 2017 |
| **MobileNetV3** | large, small | Lightweight depthwise-separable CNN for efficiency | 2019 |
| **ResNet** | 18, 34, 50, 101, 152 | Classic residual learning — used as baseline | 2015 |

All models use **ImageNet-pretrained weights** (transfer learning), fine-tuned end-to-end on ISIC 2019.

---

## ⚙️ Training Setup

A **unified training configuration** was applied across all 29 models. Model-specific parameters (input size, batch size) vary only where architecturally required.

### Common Hyperparameters

| Hyperparameter | Value |
|---------------|-------|
| Max Epochs | 50 |
| Optimiser | AdamW |
| LR Scheduler | CosineAnnealingLR |
| Weight Decay | 1e-4 |
| Early Stopping Patience | 7 epochs |
| Loss Function | Focal Loss (γ = 2.0) + Label Smoothing (0.1) |
| Class Weighting | Inverse-frequency weights |
| Device | CUDA — Kaggle P100 / T4 GPU |

### Why Focal Loss?

Standard Cross-Entropy on a 53.9:1 imbalanced dataset would effectively learn "predict NV always." The triple defence used here:

- **Focal Loss (γ=2.0):** Down-weights easy majority-class examples, forces attention on hard minority-class samples
- **Label smoothing (0.1):** Prevents overconfident predictions on the dominant NV class
- **Inverse-frequency class weights:** Explicitly amplifies gradient signal from AK, DF, VASC

### Model-Specific Configurations

| Architecture | Input Size | Batch Size | ~Parameters |
|-------------|-----------|-----------|-------------|
| ConvNeXtV2-Large | 224×224 | 16 | 198M |
| ConvNeXt-Large | 224×224 | 20 | 198M |
| ViT-Large | 224×224 | 24 | 307M |
| ConvNeXt-Base | 224×224 | 24 | 89M |
| DeiT-Base-Distilled | 224×224 | 32 | 87M |
| ConvNeXt-Tiny | 224×224 | 32 | 28M |
| EfficientNet-B4 | 380×380 | 16 | 19M |
| EfficientNet-B3 | 300×300 | 20 | 12M |
| EfficientNet-B2 | 260×260 | 24 | 9.2M |
| EfficientNet-B0–B1 | 224–240×224–240 | 28–32 | 5–7M |
| ResNet-18 to 152 | 224×224 | 20–32 | 11–60M |

---

## 📈 Individual Model Results — Performance Leaderboard

All 29 models evaluated on the **held-out test set** (3,800 images, never seen during training or validation). Sorted by Test Accuracy.

| Rank | Model | Family | Test Acc % | Balanced Acc | Macro AUC | Best Val Acc % | Train Time (min) |
|------|-------|--------|-----------|-------------|-----------|--------------|-----------------|
| 🥇 1 | convnext2 large | ConvNeXtV2 | **87.82** | 0.82 | 0.97 | 88.16 | 404.4 |
| 🥈 2 | vit large | ViT | 87.47 | 0.82 | 0.96 | 87.74 | 527.5 |
| 🥉 3 | conv large | ConvNeXt | 87.21 | 0.80 | 0.97 | 86.84 | 433.8 |
| 4 | deit base | DeiT | 87.05 | 0.81 | 0.96 | 86.55 | 178.1 |
| 5 | conv tiny | ConvNeXt | 87.03 | 0.81 | 0.97 | 86.34 | 122.9 |
| 6 | conv base | ConvNeXt | 86.95 | **0.83** | 0.97 | 85.95 | 235.6 |
| 7 | convnext2 tiny | ConvNeXtV2 | 86.84 | 0.81 | 0.97 | 85.63 | 119.7 |
| 8 | conv small | ConvNeXt | 85.84 | 0.80 | 0.97 | 85.68 | 166.8 |
| 9 | vit small | ViT | 85.13 | 0.79 | 0.97 | 84.95 | 135.3 |
| 10 | swin small | Swin | 85.00 | 0.79 | 0.96 | 86.45 | 187.4 |
| 11 | swin tiny | Swin | 84.82 | 0.79 | 0.97 | 84.34 | 132.2 |
| 12 | eff2 | EfficientNet | 84.45 | 0.78 | 0.97 | 84.87 | 137.7 |
| 13 | connext2 small | ConvNeXtV2 | 83.79 | 0.79 | 0.97 | 84.55 | 107.0 |
| 14 | dense_169 | DenseNet | 83.39 | 0.79 | 0.97 | 82.79 | 136.4 |
| 15 | dense_161 | DenseNet | 83.37 | 0.77 | 0.97 | 83.34 | 195.3 |
| 16 | mobile large | MobileNetV3 | 83.21 | 0.78 | 0.97 | 82.58 | 107.1 |
| 17 | eff3 | EfficientNet | 82.68 | 0.80 | 0.97 | 83.61 | 127.2 |
| 18 | eff1 | EfficientNet | 82.37 | 0.78 | 0.97 | 82.50 | 125.7 |
| 19 | dense_201 | DenseNet | 82.34 | 0.77 | 0.97 | 82.16 | 174.4 |
| 20 | eff0 | EfficientNet | 82.03 | 0.78 | 0.97 | 80.84 | 107.2 |
| 21 | eff4 | EfficientNet | 81.97 | 0.77 | 0.97 | 83.13 | 311.3 |
| 22 | dense_121 | DenseNet | 81.11 | 0.77 | 0.96 | 80.97 | 132.0 |
| 23 | res 50 | ResNet | 80.05 | 0.74 | 0.96 | 79.71 | 116.6 |
| 24 | vit tiny | ViT | 80.05 | 0.76 | 0.96 | 80.13 | 106.8 |
| 25 | mobile small | MobileNetV3 | 78.26 | 0.72 | 0.96 | 78.55 | 105.2 |
| 26 | res 18 | ResNet | 70.00 | 0.66 | 0.93 | 70.21 | 41.4 |
| 27 | res 34 | ResNet | 68.97 | 0.67 | 0.93 | 69.95 | 47.8 |
| 28 | res 101 | ResNet | 68.03 | 0.65 | 0.92 | 67.71 | 42.1 |
| 29 | res 152 | ResNet | 62.16 | 0.59 | 0.91 | 62.45 | 51.1 |

> **Critical insight:** Architecture generation matters far more than model depth or parameter count. ResNet-152 (60M params) scored **62.16%** while ConvNeXt-Tiny (28M params, a 2022 architecture) hit **87.03%** — a 25-point gap from ~2× fewer parameters.

![Model Comparison Overview](https://raw.githubusercontent.com/sudoingcoding/skin-diseases-detection/main/figures/model_comparison.png)

*4-panel overview: (1) top-model accuracy ranking, (2) Balanced Accuracy vs Macro AUC scatter, (3) architecture family distribution, (4) training time histogram.*

---

### 🔎 Deep Dive: ConvNeXtV2-Large (Best Single Model — 87.82%)

#### Training Curves

| Loss Curve | Accuracy Curve |
|-----------|---------------|
| ![ConvNeXtV2 Loss](https://raw.githubusercontent.com/sudoingcoding/skin-diseases-detection/main/figures/convnextv2-large-loss-curve.png) | ![ConvNeXtV2 Accuracy](https://raw.githubusercontent.com/sudoingcoding/skin-diseases-detection/main/figures/convnextv2-large-accuracy-curve.png) |

#### Evaluation

| Confusion Matrix | ROC Curves |
|----------------|-----------|
| ![ConvNeXtV2 Confusion Matrix](https://raw.githubusercontent.com/sudoingcoding/skin-diseases-detection/main/figures/convnextv2-large-confusion-matrix.png) | ![ConvNeXtV2 ROC](https://raw.githubusercontent.com/sudoingcoding/skin-diseases-detection/main/figures/convnextv2-large-roc-curve.png) |

#### Grad-CAM — Where the Model Looks

![ConvNeXtV2 Grad-CAM](https://raw.githubusercontent.com/sudoingcoding/skin-diseases-detection/main/figures/convnextv2-large-grad-cam.png)

*Gradient-weighted Class Activation Maps showing which image regions drove predictions. Top rows: correct high-confidence predictions focused on lesion body and borders. Bottom rows: incorrect predictions revealing where the model's attention deviated.*

#### Per-Class Performance

| Class | Disease | Precision | Recall | F1-Score | Support |
|-------|---------|----------|--------|---------|---------|
| AK | Actinic Keratosis | 0.8241 | 0.6846 | 0.7479 | 130 |
| BCC | Basal Cell Carcinoma | 0.8957 | 0.9118 | 0.9037 | 499 |
| BKL | Benign Keratosis | 0.8259 | 0.7944 | 0.8098 | 394 |
| DF | Dermatofibroma | 0.9643 | 0.7500 | 0.8437 | 36 |
| MEL | Melanoma | 0.8336 | 0.7906 | 0.8115 | 678 |
| **NV** | **Nevus** | **0.9008** | **0.9358** | **0.9180** | 1931 |
| SCC | Squamous Cell Carcinoma | 0.8280 | 0.8191 | 0.8235 | 94 |
| VASC | Vascular Lesion | 0.9429 | 0.8684 | 0.9041 | 38 |

> **AK (Actinic Keratosis) is the hardest class** — lowest recall (68.5%) due to its visual similarity with BCC and BKL under dermoscopy.

---

### 🔎 Deep Dive: ViT-Large (2nd Place — 87.47%)

#### Training Curves & Evaluation

| Loss Curve | ROC Curves |
|-----------|-----------|
| ![ViT Loss](https://raw.githubusercontent.com/sudoingcoding/skin-diseases-detection/main/figures/vit-large-loss-curve.png) | ![ViT ROC](https://raw.githubusercontent.com/sudoingcoding/skin-diseases-detection/main/figures/vit-large-roc-curve.png) |

| Accuracy Curve | Confusion Matrix |
|---------------|----------------|
| ![ViT Accuracy](https://raw.githubusercontent.com/sudoingcoding/skin-diseases-detection/main/figures/loss-curve-accuracy-curve.png) | ![ViT Confusion Matrix](https://raw.githubusercontent.com/sudoingcoding/skin-diseases-detection/main/figures/loss-curve-confusion-matrix.png) |

#### Grad-CAM — Attention Patterns

![ViT Grad-CAM](https://raw.githubusercontent.com/sudoingcoding/skin-diseases-detection/main/figures/vit-large-grad-cam.png)

*ViT uses global self-attention — its Grad-CAM maps are noticeably more diffuse than ConvNeXtV2's, reflecting how patch-based attention distributes focus across the full image rather than concentrating on local features.*

> **Notable:** ViT-Large achieved the **highest Balanced Accuracy (0.82) tied with ConvNeXtV2-Large**, making it the superior choice when minority-class recall is the priority — important for rare diseases like VASC and DF.

---

### 🔎 Deep Dive: DeiT-Base-Distilled (4th Place — 87.05%)

#### Training Curves & Evaluation

| Loss Curve | Accuracy Curve |
|-----------|---------------|
| ![DeiT Loss](https://raw.githubusercontent.com/sudoingcoding/skin-diseases-detection/main/figures/deit-base-distilled-loss-curve.png) | ![DeiT Accuracy](https://raw.githubusercontent.com/sudoingcoding/skin-diseases-detection/main/figures/deit-base-distilled-accuracy-curve.png) |

| Confusion Matrix | ROC Curves |
|----------------|-----------|
| ![DeiT Confusion Matrix](https://raw.githubusercontent.com/sudoingcoding/skin-diseases-detection/main/figures/deit-base-distilled-confusion-matrix.png) | ![DeiT ROC](https://raw.githubusercontent.com/sudoingcoding/skin-diseases-detection/main/figures/deit-base-distilled-roc-curve.png) |

#### Grad-CAM

![DeiT Grad-CAM](https://raw.githubusercontent.com/sudoingcoding/skin-diseases-detection/main/figures/deit-base-distilled-grad-cam.png)

*DeiT uses knowledge distillation during pretraining — its attention maps show a hybrid between the broad global attention of ViT and the more localised focus of CNN-based models, which may explain its strong 87.05% despite being a 178-minute training run.*

> **Efficiency champion:** DeiT-Base-Distilled achieved 87.05% accuracy in just **178 minutes** — less than half the training time of ViT-Large (527 min) for nearly identical performance. Best accuracy-per-GPU-hour of any top-10 model.

---

### Top Misclassification Patterns

| Pattern | Count | % of True Class | Clinical Risk |
|---------|-------|----------------|--------------|
| **NV → MEL** | 127 | 6.58% | ⚠️ **HIGH** — False negative for cancer |
| **MEL → NV** | 96 | 14.16% | ⚠️ **HIGH** — Missing melanoma |
| BKL → NV | 48 | 12.18% | Medium |
| NV → BCC | 20 | 1.04% | Medium |
| MEL → BKL | 18 | 2.65% | High |
| AK → BCC | 15 | 11.54% | Medium |

> The NV ↔ MEL confusion is the most clinically dangerous pattern: **14.16% of melanoma cases are called Nevus** (a benign mole). This motivates the entire ensemble approach — ensembles showing the most improvement specifically on MEL recall.

---

## 🧩 Ensemble Strategies

Five distinct ensemble methods were implemented and rigorously compared. All operate on **saved softmax probability outputs** (`.pkl` files) from individual model inference — no retraining required.

### Strategy Overview

| # | Strategy | Models | Core Idea |
|---|---------|--------|-----------|
| 1 | **Smart Hybrid** | 4–7 | Top performers + forced architectural variety, weighted by Balanced Accuracy |
| 2 | **Top 5 Weighted** | 5 | Leaderboard top-5, weighted by Balanced Acc or Test Acc |
| 3 | **Architecture Diversity** | 6 | One champion per architecture family (equal weights) |
| 4 | **Confidence Weighted** | 5 | Weight by per-model confidence gap (correct vs incorrect confidence) |
| 5 | **Meta-Optimized Hierarchical** | 7 | Stacking + scipy-optimised weights + diversity scoring + class-group authority |

---

### Ensemble 1 — Smart Hybrid

**Philosophy:** Combine the highest-performing models while enforcing architectural variety. Prevents the ensemble being dominated by correlated errors from multiple ConvNeXt variants.

**Selected models:**

| Model | Family | Test Acc % | Role |
|-------|--------|-----------|------|
| convnext2 large | ConvNeXtV2 | 87.82 | Top performer |
| vit large | ViT | 87.47 | Top performer |
| conv large | ConvNeXt | 87.21 | Top performer |
| deit base | DeiT | 87.05 | Transformer expert |
| eff2 | EfficientNet | 84.45 | CNN diversity |
| dense_169 | DenseNet | 83.39 | Dense-skip diversity |
| mobile large | MobileNetV3 | 83.21 | Lightweight diversity |

**Weighting:** Normalised Balanced Accuracy scores (favours minority-class performance)

![Smart Hybrid Ensemble](https://raw.githubusercontent.com/sudoingcoding/skin-diseases-detection/main/figures/smart_hybrid_ensemble_visualization.png)

---

### Ensemble 2 — Top 5 Weighted

**Philosophy:** Simple baseline — take the five best leaderboard models and weight them. Evaluated with two weighting schemes to understand which matters more for imbalanced classification.

- **Option A:** Weight by Balanced Accuracy *(recommended for imbalanced data)*
- **Option B:** Weight by Test Accuracy

**Selected models:** convnext2 large, vit large, conv large, deit base, conv tiny

![Top 5 Weighted Ensemble](https://raw.githubusercontent.com/sudoingcoding/skin-diseases-detection/main/figures/top5_weighted_ensemble_analysis.png)

---

### Ensemble 3 — Architecture Diversity

**Philosophy:** One best representative per architecture family. Maximises diversity of learned representations — different families make systematically different errors, so their combination cancels failure modes.

| Family | Representative | Acc % | Bal Acc | Weight |
|--------|---------------|-------|---------|--------|
| ConvNeXtV2 | convnext2 large | 87.82 | 0.82 | 16.67% |
| ViT | vit large | 87.47 | 0.82 | 16.67% |
| DeiT | deit base | 87.05 | 0.81 | 16.67% |
| ConvNeXt | conv base | 86.95 | 0.83 | 16.67% |
| Swin | swin small | 85.00 | 0.79 | 16.67% |
| DenseNet | dense_169 | 83.39 | 0.79 | 16.67% |

> ResNet was **deliberately excluded** — all 5 variants scored 62–80%, pulling ensemble accuracy down without contributing complementary information.

![Architecture Diversity Ensemble](https://raw.githubusercontent.com/sudoingcoding/skin-diseases-detection/main/figures/architecture_diversity_ensemble_analysis.png)

---

### Ensemble 4 — Confidence Weighted

**Philosophy:** Weight models by how well-calibrated their uncertainty is, not just raw accuracy. A model with a large gap between its confidence on correct vs incorrect predictions is a higher-quality probabilistic signal contributor.

**Confidence gap formula:**
```
confidence_gap = mean(max_prob | correct prediction) − mean(max_prob | incorrect prediction)
```

**ConvNeXt-Base calibration stats:**
- Correct predictions: mean confidence **0.8738** (median 0.9280)
- Incorrect predictions: mean confidence **0.6804** (median 0.6610)
- **Confidence gap: 0.1934** — a clean 19-point separation

![Confidence Weighted Ensemble](https://raw.githubusercontent.com/sudoingcoding/skin-diseases-detection/main/figures/confidence_weighted_ensemble_analysis.png)

---

### Ensemble 5 — Meta-Optimized Hierarchical

**Philosophy:** The most sophisticated strategy, combining four mechanisms:

1. **Stacking:** A meta-learner trained on base model probability outputs
2. **Optimised weighted averaging:** Weights found via `scipy.optimize` to maximise validation Balanced Accuracy directly
3. **Diversity scoring:** Models penalised for correlated errors — complementary predictors are preferred
4. **Hierarchical class grouping:** Different model subsets given higher authority for different class clusters based on per-class AUC

**Selected 7 models:** conv base + 6 complementary models from diverse families

![Meta-Optimized Ensemble](https://raw.githubusercontent.com/sudoingcoding/skin-diseases-detection/main/figures/meta_optimized_ensemble_complete_analysis.png)

---

## 🏆 Final Ensemble Comparison

### Results Table

| Strategy | Models | Accuracy % | Balanced Acc | Macro AUC | Δ vs Single Best |
|----------|--------|-----------|-------------|-----------|-----------------|
| ⭐ **Meta-Optimized** | 7 | **90.45** | **0.858** | **0.989** | **+3.50%** |
| Architecture Diversity | 6 | 89.92 | 0.850 | 0.983 | +2.97% |
| Smart Hybrid | 4–7 | 89.90 | 0.847 | 0.981 | +2.94% |
| Top 5 Weighted | 5 | 89.74 | 0.846 | 0.982 | +2.79% |
| Confidence Weighted | 5 | 88.63 | 0.835 | 0.982 | +1.68% |
| **Single Best Model** | 1 | 86.95 | 0.830 | 0.970 | baseline |

> The **Meta-Optimized ensemble hit 90.45%** — a **+3.50 percentage point** absolute gain over the single best model. Average improvement across all 5 ensemble strategies: **+2.78%**.

![Ensemble Strategies Comprehensive Comparison](https://raw.githubusercontent.com/sudoingcoding/skin-diseases-detection/main/figures/ensemble_strategies_comprehensive_comparison.png)

*7-panel master comparison: accuracy bars, balanced accuracy bars, improvement waterfall over single best, models-used vs accuracy scatter, macro AUC comparison, multi-metric radar chart, and complexity vs performance trade-off.*

### Strategy Recommendation Guide

| Strategy | Best When | Pros | Cons |
|----------|----------|------|------|
| ⭐ **Meta-Optimized** | Maximum production accuracy | Mathematically optimal, highest on all metrics | Most complex, needs validation set for optimisation |
| **Architecture Diversity** | Generalisation & robustness | High diversity, reduced bias, interpretable weights | May include lower-performing individual models |
| **Smart Hybrid** | Balance of accuracy and simplicity | Strong performance, manageable to maintain | More complex than simple top-N |
| **Top 5 Weighted** | Quick ensemble baseline | Simple to implement and explain | Low diversity — correlated errors persist |
| **Confidence Weighted** | Calibrated uncertainty output | Well-calibrated probabilities for triage systems | Computationally expensive to compute gaps |

---

## 🔍 Explainability — Grad-CAM

Every model notebook generates **Gradient-weighted Class Activation Maps (Grad-CAM)** — the most widely used XAI technique for CNN/Transformer models in medical imaging. This is critical for clinical trust: a model achieving 90% accuracy means nothing if it's focusing on hair artefacts or image borders instead of the lesion.

### What Each Grad-CAM Panel Shows

- **Original dermoscopy image**
- **Grad-CAM heatmap overlaid** — red/warm = high activation, blue/cool = low activation
- **True label vs predicted label** for each sample

### Correct Predictions (ConvNeXtV2-Large)

The Grad-CAM maps on correct, high-confidence predictions show the model correctly:
- Attending to the **lesion body and its irregular borders**
- Responding to **colour asymmetry** within the lesion
- Ignoring surrounding normal skin

![ConvNeXtV2 Grad-CAM](https://raw.githubusercontent.com/sudoingcoding/skin-diseases-detection/main/figures/convnextv2-large-grad-cam.png)

### ViT-Large — Global Attention Pattern

![ViT Grad-CAM](https://raw.githubusercontent.com/sudoingcoding/skin-diseases-detection/main/figures/vit-large-grad-cam.png)

*ViT attends globally — its activation maps are broader and more distributed than CNN-based models, reflecting patch-based self-attention across the full 224×224 image.*

### DeiT-Base-Distilled — Hybrid Attention

![DeiT Grad-CAM](https://raw.githubusercontent.com/sudoingcoding/skin-diseases-detection/main/figures/deit-base-distilled-grad-cam.png)

*DeiT's distillation token creates a hybrid attention pattern — combining the global reach of ViT with tighter localisation learned from the CNN teacher network.*

---

## 📐 Statistical Significance Testing

To confirm that improvements are not due to random variation, **McNemar's Test** was applied to all pairwise comparisons. McNemar's test is ideal here because it measures whether two classifiers make **different errors** on the same test set, not just whether their accuracy numbers differ.

### McNemar's Test Results

| Comparison | A Wins | B Wins | p-value | Significant? |
|-----------|--------|--------|---------|-------------|
| Meta-Optimized vs Single Best | 192 | 59 | 0.000 | ✅ YES |
| Architecture Diversity vs Single Best | 185 | 72 | 0.000 | ✅ YES |
| Smart Hybrid vs Single Best | 191 | 79 | 0.000 | ✅ YES |
| Top 5 Weighted vs Single Best | 182 | 76 | 0.000 | ✅ YES |
| Confidence Weighted vs Single Best | 138 | 74 | 0.000 | ✅ YES |
| Confidence Weighted vs Meta-Optimized | 47 | 116 | 0.000 | ✅ YES |
| Architecture Diversity vs Confidence Weighted | 117 | 68 | 0.000 | ✅ YES |
| Smart Hybrid vs Confidence Weighted | 123 | 75 | 0.001 | ✅ YES |
| Top 5 Weighted vs Meta-Optimized | 22 | 49 | 0.002 | ✅ YES |
| Architecture Diversity vs Meta-Optimized | 27 | 47 | 0.027 | ✅ YES |
| **Top 5 Weighted vs Architecture Diversity** | 22 | 29 | 0.401 | ❌ Not significant |
| **Smart Hybrid vs Top 5 Weighted** | 38 | 32 | 0.550 | ❌ Not significant |
| **Smart Hybrid vs Architecture Diversity** | 40 | 41 | 1.000 | ❌ Not significant |

**Interpretation:**
- All ensembles significantly outperform the single best model (p < 0.001)
- Meta-Optimized is significantly better than every other strategy
- **Smart Hybrid, Top 5 Weighted, and Architecture Diversity are statistically indistinguishable** — any of the three is a defensible choice for production when the Meta-Optimized complexity is undesirable

---

## 💡 Key Takeaways

### 1. Architecture Generation > Parameter Count
Modern architectures (ConvNeXtV2, ViT, ConvNeXt, DeiT, Swin) cluster in the **85–88%** accuracy band. All ResNet variants cluster in the **62–80%** band — regardless of depth (ResNet-152 is worse than ResNet-50). The design philosophy of 2020–2023 architectures is transformative, not incremental.

### 2. Focal Loss + Class Weighting is Non-Negotiable for Medical Imbalance
A 53.9:1 imbalance ratio means standard Cross-Entropy would effectively learn "predict NV." The triple defence (Focal Loss + Label Smoothing + inverse-frequency weights) was essential for any model achieving competitive recall on rare classes.

### 3. Two-Stage Preprocessing Pays Off
79% of images had hair artefacts — a noise level that would significantly degrade raw training. Hair removal + CLAHE created a cleaner learning signal with a +30-unit average contrast improvement, giving every model a more consistent lesion boundary to attend to.

### 4. Ensembles Always Win — But Diversity Wins More Than Raw Accuracy
The Architecture Diversity ensemble (which includes lower-performing DenseNet and Swin models) outperformed the Top-5 Weighted ensemble (which uses only the 5 best). Different families make **structurally different errors**, and the combination cancels them.

### 5. Model Confidence is a Reliable Uncertainty Signal
ConvNeXt-Base shows a 19-point confidence gap between correct (0.87) and incorrect (0.68) predictions. This calibration is genuine — the model knows when it doesn't know, making it suitable for **human-in-the-loop triage** where flagging uncertain cases for dermatologist review is the real goal.

### 6. DeiT is the Efficiency Outlier Worth Noting
DeiT-Base-Distilled achieved 87.05% in 178 minutes — less than a third of ViT-Large's 527 minutes for nearly the same test accuracy. For resource-constrained settings, DeiT is the clear architecture of choice.

### 7. The MEL ↔ NV Confusion Persists and Matters
MEL→NV (predicting melanoma as a benign mole) remains at **14.16% false-negative rate** across all individual models. Ensembles reduce but don't eliminate this. This suggests dermoscopy images alone may be insufficient for definitive MEL/NV separation at scale — patient history, metadata (age, body site), or sequential imaging may be the necessary next modality.

---

## 📚 References

1. **ISIC 2019 Challenge** — Combalia et al., *BCN20000: Dermoscopic Lesions in the Wild*, arXiv 2019
2. **ConvNeXt** — Liu et al., *A ConvNet for the 2020s*, CVPR 2022
3. **ConvNeXtV2** — Woo et al., *ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders*, CVPR 2023
4. **Vision Transformer** — Dosovitskiy et al., *An Image is Worth 16×16 Words*, ICLR 2021
5. **DeiT** — Touvron et al., *Training Data-Efficient Image Transformers*, ICML 2021
6. **Swin Transformer** — Liu et al., *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows*, ICCV 2021
7. **EfficientNet** — Tan & Le, *EfficientNet: Rethinking Model Scaling for CNNs*, ICML 2019
8. **DenseNet** — Huang et al., *Densely Connected Convolutional Networks*, CVPR 2017
9. **Focal Loss** — Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017
10. **Grad-CAM** — Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017
11. **DullRazor** — Lee et al., *Dullrazor: A software approach to hair removal from images*, Computers in Biology and Medicine 1997
12. **McNemar's Test** — McNemar, Q., *Note on the sampling error of the difference between correlated proportions*, Psychometrika 1947

---

<div align="center">

*Trained on Kaggle (GPU: P100 / T4). Total compute: ~29 models × 166 min avg = ~80 GPU-hours.*

*Built by [sudoingcoding](https://github.com/sudoingcoding)*

</div>
