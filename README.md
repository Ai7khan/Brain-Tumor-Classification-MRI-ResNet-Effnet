# 🧠 Brain Tumor Classification from MRI
### Pattern Recognition — Final Project

> Automated 4-class brain tumor classification using Transfer Learning with ResNet50 and EfficientNetB3, trained on the Kaggle Brain Tumor MRI dataset.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pattern Recognition Pipeline](#pattern-recognition-pipeline)
- [Model Architectures](#model-architectures)
- [Training Strategy](#training-strategy)
- [Results](#results)
- [Key Findings](#key-findings)
- [Installation & Usage](#installation--usage)
- [Requirements](#requirements)
- [References](#references)

---

## Overview

This project applies a complete **4-step pattern recognition pipeline** to classify brain MRI images into four tumor categories using deep convolutional neural networks. Two pre-trained architectures — **ResNet50** and **EfficientNetB3** — are compared using a two-phase transfer learning strategy: feature extraction warmup followed by selective fine-tuning.

The project was developed and run entirely on **Google Colab** using free GPU compute.

---

## Dataset

**Source:** [Brain Tumor Classification (MRI) — Kaggle](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)  
**Author:** Sartaj Bhuvaji  
**Downloaded via:** `kagglehub`

| Class | Training | Testing |
|---|---|---|
| Glioma Tumor | 826 | 100 |
| Meningioma Tumor | 822 | 115 |
| No Tumor | 395 | 105 |
| Pituitary Tumor | 827 | 74 |
| **Total** | **2,870** | **394** |

- Images: JPEG, variable native resolution → resized to **224×224**
- Train/Validation split: **80/20** stratified (seed=123)
- Mild class imbalance addressed with **computed class weights**

---

## Project Structure

```
brain-tumor-classification/
│
├── brain_tumor_reseff.ipynb   # Main notebook (run on Colab)
├── README.md
│
└── outputs/                           # Generated during runtime
    ├── confusion_matrix_resnet50.png
    ├── confusion_matrix_efficientnet.png
    └── training_curves.png
```

> The dataset is downloaded automatically via `kagglehub` — no manual download needed.

---

## Pattern Recognition Pipeline

This project follows the four standard steps of a pattern recognition system:

### Step 1 — Sensing / Data Acquisition
Raw MRI JPEG images loaded from disk using `tf.keras.utils.image_dataset_from_directory`. Images decoded, resized to 224×224, and batched (batch size = 32).

### Step 2 — Preprocessing & Feature Extraction
- **ResNet50:** pixels normalized using `resnet50.preprocess_input` (BGR conversion + ImageNet mean subtraction)
- **EfficientNetB3:** raw [0, 255] pixels fed directly — model has built-in internal rescaling
- **Data Augmentation** applied to training pipeline:
  - `RandomFlip("horizontal")` — left-right symmetry
  - `RandomRotation(0.15)` — ±54° rotation, simulates head positioning variation
  - `RandomZoom(0.15)` — ±15% zoom, simulates scan framing variation
- **Class weights** computed using `sklearn.utils.class_weight.compute_class_weight('balanced')` to handle imbalance

### Step 3 — Classification
Two CNN models with custom classification heads trained using two-phase transfer learning. See [Model Architectures](#model-architectures) and [Training Strategy](#training-strategy).

### Step 4 — Evaluation
- Accuracy, Precision (macro), Recall (macro), F1-Score (macro)
- Per-class classification report
- Confusion matrix heatmap

---

## Model Architectures

Both models use ImageNet pretrained weights with a custom classification head:

```
Pretrained Base (frozen in Phase 1)
        ↓
GlobalAveragePooling2D
        ↓
BatchNormalization
        ↓
Dense(256, activation='relu')
        ↓
Dropout(0.5)
        ↓
Dense(4, activation='softmax')
```

### ResNet50
- **Parameters:** ~25M
- **Key feature:** Residual (skip) connections solve vanishing gradient in deep networks
- **Preprocessing:** `resnet50.preprocess_input` applied as functional layer inside model graph

### EfficientNetB3
- **Parameters:** ~12M
- **Key feature:** Compound scaling (width + depth + resolution) for efficiency
- **Preprocessing:** Built-in internal rescaling — expects raw [0, 255] pixel input
- **Note:** Must use `inputs=base.input` (not a separate `layers.Input`) to avoid Keras 3 graph disconnection error

---

## Training Strategy

### Two-Phase Transfer Learning

**Phase 1 — Warmup (frozen base)**
```python
base.trainable = False
model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy')
model.fit(..., epochs=12, class_weight=class_weight_dict)
```
The pretrained base acts as a fixed feature extractor. Only the new classification head is trained. Prevents chaotic random-weight gradients from corrupting ImageNet representations.

**Phase 2 — Fine-Tuning (top 30 layers unfrozen)**
```python
for layer in model.layers[-30:]:
    layer.trainable = True
model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy')
model.fit(..., epochs=10, class_weight=class_weight_dict)
```
The final ~30 layers of the base are gently adapted toward MRI-specific features. Learning rate is 100× smaller to prevent catastrophic forgetting.

### Callbacks
```python
EarlyStopping(patience=10, monitor='val_accuracy', restore_best_weights=True)
ReduceLROnPlateau(patience=2, factor=0.2, monitor='val_loss')
```

### Pipeline Optimization
```python
AUTOTUNE = tf.data.AUTOTUNE
train_gen = train_gen.map(augment_fn, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
```
AUTOTUNE dynamically tunes parallel CPU threads and prefetch buffer size at runtime to keep the GPU fully utilized.

---

## Results

### Overall Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| ResNet50 | 0.7411 | 0.8264 | 0.7376 | 0.7153 |
| EfficientNetB3 | 0.7208 | 0.7419 | 0.7144 | 0.7105 |

### Per-Class Performance

| Class | ResNet Prec | ResNet Recall | ResNet F1 | EffNet Prec | EffNet Recall | EffNet F1 |
|---|---|---|---|---|---|---|
| Glioma | 0.96 | 0.24 | 0.38 | 0.75 | 0.41 | 0.53 |
| Meningioma | 0.65 | 0.94 | 0.77 | 0.63 | 0.86 | 0.73 |
| No Tumor | 0.70 | 0.93 | 0.80 | 0.77 | 0.86 | 0.81 |
| Pituitary | 1.00 | 0.84 | 0.91 | 0.82 | 0.73 | 0.77 |

---

## Key Findings

**1. Glioma is the dominant failure mode**  
Both models severely underperform on glioma classification. ResNet50 achieves only 24% recall — 76/100 glioma cases misclassified, mostly as meningioma. EfficientNetB3 improves to 41% but still classifies the majority incorrectly. Glioma's high morphological variability across WHO grades makes it the hardest class in the dataset.

**2. ResNet50 overfits**  
Training accuracy reaches ~99% while validation stays at ~92% — a clear overfitting signature. The train/val gap reflects the model over-adapting to specific training examples rather than generalizing tumor features.

**3. EfficientNetB3 generalizes better**  
Despite slightly lower overall accuracy, EfficientNetB3 shows more balanced per-class recall and a tighter train/val curve. Its more efficient architecture adapts more evenly across all four classes.

**4. Preprocessing matters critically**  
An early version of the code double-normalized EfficientNetB3 (adding `Rescaling(1./255)` on top of its internal normalization), collapsing performance to 27% accuracy. Removing the redundant layer restored it to 72%.

### Proposed Improvements

- **Focal Loss** — down-weights easy examples, forces learning on hard glioma cases
- **Unfreeze more layers** — top 50-60 instead of 30 for deeper domain adaptation
- **MRI-specific augmentation** — brightness, contrast, Gaussian noise to simulate scanner variation
- **Higher resolution** — 300×300 or 380×380 to preserve fine tumor boundary details
- **Ensemble** — soft-vote ResNet50 + EfficientNetB3 predictions to combine complementary strengths
- **RadImageNet weights** — pretrained on 1.35M radiology images, far more domain-aligned than ImageNet

---

## Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/brain-tumor-classification.git
cd brain-tumor-classification
```

### 2. Open in Google Colab
Upload `brain_tumor_classification.ipynb` to [colab.research.google.com](https://colab.research.google.com) or use the badge:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ai7khan/brain-tumor-classification/blob/main/brain_tumor_reseff.ipynb)

### 3. Set up Kaggle credentials (first time only)
In Colab, mount your Drive and upload your `kaggle.json` API key, or set environment variables:
```python
import os
os.environ['KAGGLE_USERNAME'] = 'your_username'
os.environ['KAGGLE_KEY'] = 'your_api_key'
```

### 4. Enable GPU
`Runtime → Change runtime type → T4 GPU`

### 5. Run all cells
`Runtime → Run all`  
The dataset downloads automatically. Full training takes approximately **25-35 minutes** on a T4 GPU.

---

## Requirements

```
tensorflow >= 2.15
kagglehub
numpy
matplotlib
seaborn
scikit-learn
```

All dependencies are pre-installed in Google Colab. If running locally:

```bash
pip install tensorflow kagglehub numpy matplotlib seaborn scikit-learn
```

> **Note:** If you encounter `AttributeError: module 'keras' has no attribute 'KerasTensor'`, this is a Keras 3 compatibility issue with EfficientNetB3. The notebook handles this by using `inputs=base.input` instead of a separate `layers.Input`. Ensure you are using the latest version of this notebook.

---

## References

1. He, K. et al. "Deep residual learning for image recognition." CVPR 2016.
2. Tan, M. & Le, Q. V. "EfficientNet: Rethinking model scaling for CNNs." ICML 2019.
3. Bhuvaji, S. et al. "Brain Tumor Classification (MRI)." Kaggle, 2020.
4. Sultan, H. H. et al. "Multi-classification of brain tumor images using deep neural network." IEEE Access, 2019.
5. Lin, T.-Y. et al. "Focal loss for dense object detection." ICCV 2017.
6. Kingma, D. P. & Ba, J. "Adam: A method for stochastic optimization." ICLR 2015.

---

## Course

**Pattern Recognition** — Final Project  
Department of Computer Engineering
Ankara University
2025–2026 Academic Year
