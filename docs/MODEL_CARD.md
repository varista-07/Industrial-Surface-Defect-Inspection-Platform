# Model Card: Surface Defect Inspection Platform Models

## 1. Model Details
* **Developed By**: Student Engineering Team, CSAI3027P Deep Learning PBL Project 02.
* **Evaluation Date**: 23 November 2026.
* **Model Versions**:
  - Primary Classifier: `ResNet50_DefectClassifier` (Two-stage transfer learning with ImageNet weights, Stage 1 frozen backbone, Stage 2 fine-tuning `conv5_block*` with all `BatchNormalization` layers explicitly frozen).
  - Localization Model: `ResNet50_DefectDetector` (Multi-task Classification & Bounding Box Regression Head with Smooth L1 + Cross-Entropy Loss).
  - Benchmark Models: `InceptionV3_Benchmark` and `From-Scratch MLP (NumPy)`.
  - Segmentation: `Lightweight_UNet` (Trained on weak Otsu-derived pseudo-masks at $128 \times 128$).
  - Generative & Language: `Conv-VAE` ($64 \times 64$) and `DefectReportTransformer` (4-head self-attention).
* **Framework**: TensorFlow 2.21, Keras 3.15, Keras-CV 0.9.0, and pure NumPy.
* **License**: MIT Open Source for Academic Research.

---

## 2. Intended Use
* **Primary Intended Use**: Real-time automated inspection of hot-rolled steel strip surface defect images to identify defect class, localized bounding box, confidence score, and visual explainability heatmap.
* **Primary Users**: Quality control engineers, line operators, and academic evaluators in manufacturing AI.
* **Deployment Context**: Live interactive web platform (`http://localhost:8000`) integrated into post-rolling visual inspection stations.

---

## 3. Quantitative Performance Summary (Test Set: 270 Unseen Images)

All metrics reported below were empirically measured on the held-out test split (45 samples/class across 6 classes) with random seed 42:

| Architecture | Model Paradigm | Input Resolution | Test Accuracy | Test Macro F1 | CPU Latency / Image | Throughput (FPS) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **From-Scratch MLP** | Pure NumPy Backprop | $32 \times 32$ | 56.67% | 0.5480 | ~0.40 ms | ~2,500.0 FPS |
| **InceptionV3 Benchmark** | Transfer Learning | $200 \times 200 \times 3$ | 97.78% | 0.9778 | 192.80 ms | 5.19 FPS |
| **ResNet50 (Fine-Tuned)** | Two-Stage (Frozen BN) | $200 \times 200 \times 3$ | **100.00%** | **1.0000** | 318.95 ms | 3.13 FPS |
| **Multi-Task Detector** | ResNet50 Shared Head | $200 \times 200 \times 3$ | 75.19% | 0.7410 | 21.22 ms | 47.12 FPS |
| **U-Net Weak Segmentation**| ResNet Encoder (Otsu) | $128 \times 128 \times 3$ | — | Dice: 0.1950 | 27.40 ms | 36.50 FPS |
| **Report Transformer** | 4-Head Self-Attention | Tokenized text | 100.00% | 1.0000 | ~2.10 ms | ~476.0 FPS |

### Per-Class Performance Breakdown (Final ResNet50 Classifier)
- **Crazing**: Precision: 100.0% | Recall: 100.0% | F1: 100.0%
- **Inclusion**: Precision: 100.0% | Recall: 100.0% | F1: 100.0%
- **Patches**: Precision: 100.0% | Recall: 100.0% | F1: 100.0%
- **Pitted Surface**: Precision: 100.0% | Recall: 100.0% | F1: 100.0%
- **Rolled-in Scale**: Precision: 100.0% | Recall: 100.0% | F1: 100.0%
- **Scratches**: Precision: 100.0% | Recall: 100.0% | F1: 100.0%

---

## 4. Detection Architecture & Single-Stage RetinaNet Fallback Justification

### Empirical Evaluation of RetinaNet (`keras-cv`)
As specified in the project requirements, a single-stage RetinaNet with a ResNet50 backbone was implemented via `keras-cv` (`models/retinanet_benchmark.py`) to evaluate multi-box object detection across all VOC bounding boxes per image:
1. **Computational Feasibility on CPU**:
   - Step latency measured for mini-batch size 2: **46.79 seconds per step**.
   - Projected training runtime for 1 single epoch (1,260 images / 630 steps): **491.2 minutes (~8.19 hours per epoch)**.
   - For a standard 10-epoch training schedule, total CPU compute time would exceed **81.8 hours**, severely violating the user-stipulated 30-minute training time budget.
2. **Framework Compatibility**:
   - Keras-CV 0.9.0 exhibits internal data-unpacking incompatibilities with Keras 3 (`unpack_input` expecting 2-element tuples rather than Keras 3's 3-element `(x, y, sample_weight)` format).
   - The Feature Pyramid Network (FPN) requires input dimensions strictly divisible by 32 (requiring padding from native $200 \times 200$ to $224 \times 224$ or $256 \times 256$, further compounding CPU FLOPs).

### Justified Fallback to Multi-Task Detection Head
In compliance with project directives, the platform fell back to a shared-backbone multi-task detection model (`models/detector.py`):
- **Backbone**: Frozen ResNet50 feature extractor.
- **Heads**: Simultaneous Softmax classification head + 4-coordinate bounding box regression head $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$ trained with Smooth L1 Loss.
- **Benchmark Results on Test Set**:
  - Overall Mean IoU: **0.3292**
  - Overall mAP@0.5: **24.07%**
  - Pitted Surface AP@0.5: **86.67%** (defect features crater-like localized contours well-captured by single-box localization)
  - Inference Latency: **21.22 ms** (47.1 FPS throughput on CPU, supporting real-time industrial line rates).

---

## 5. Architectural Decision: Freezing BatchNormalization in ResNet50

During Stage 2 fine-tuning of ResNet50 (`models/resnet50_classifier.py`), top convolutional blocks (`conv5_block1`, `conv5_block2`, `conv5_block3`) are unfrozen, but **all `BatchNormalization` layers remain strictly frozen (`layer.trainable = False`)**:

$$\hat{x} = \frac{x - \mu_{\text{running}}}{\sqrt{\sigma^2_{\text{running}} + \epsilon}}, \quad y = \gamma \hat{x} + \beta$$

### Theoretical and Practical Justification:
1. **Mini-Batch Statistic Distortion**: In industrial transfer learning on CPU/small batches (batch size 16 or 32), calculating mini-batch sample mean $\mu_B$ and variance $\sigma^2_B$ introduces high sampling variance. Updating running statistics on small domain batches causes representation collapse and destroys pre-trained ImageNet features.
2. **Gradient Variance Suppression**: Keeping BN layers in inference mode (`training=False`) treats them as fixed affine scalers. This stabilizes the backpropagation gradient flow through lower residual skip connections, preventing internal covariate shift during fine-tuning.

---

## 6. Industrial False-Reject Rate (FRR) & Manual Triage Analysis

Because the NEU-DET benchmark contains exclusively defective steel strip surfaces (no clean/defect-free class), standard binary False Reject Rate (classifying good surface as defect) cannot be computed directly from class labels. We define and report two rigorous industrial metrics:

### Metric Definitions:
1. **Missed Defect Rate ($\text{FRR}_{\text{miss}}(\theta)$)**: Fraction of defective test samples whose top predicted probability fails to exceed confidence threshold $\theta$:
   $$\text{FRR}_{\text{miss}}(\theta) = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\max_k P_{i,k} < \theta)$$
2. **Manual Triage Escalation Rate**: Fraction of images where model uncertainty warrants automatic escalation to a human QC inspector.

### Empirical Measurements on Test Set (270 Unseen Images):

| Confidence Threshold ($\theta$) | Flagged Uncertain Samples | Triage Escalation Rate | Defect Miss Rate | Industrial QC Implication |
| :---: | :---: | :---: | :---: | :--- |
| **0.50** | 0 / 270 | **0.00%** | **0.00%** | Fully automated inspection; zero human escalation |
| **0.60** | 0 / 270 | **0.00%** | **0.00%** | Fully automated inspection; zero human escalation |
| **0.70** | 0 / 270 | **0.00%** | **0.00%** | Fully automated inspection; zero human escalation |
| **0.80** | 0 / 270 | **0.00%** | **0.00%** | High assurance; all 270 test samples accepted automatically |
| **0.90** | 1 / 270 | **0.37%** | **0.37%** | Ultra-strict QA; 1 borderline crazing sample sent for human verification |

---

## 7. Explainability & Human-in-the-Loop Safeguards
1. **Grad-CAM Visual Heatmaps**: Every prediction generates a class-discriminative saliency heatmap indicating the exact spatial pixel groups that triggered the decision.
2. **Confidence Threshold Slider**: Operators can modulate the decision threshold (e.g., 60%). Any prediction below the threshold is flagged as **UNCERTAIN / OUT-OF-DISTRIBUTION**, routing the steel strip to manual human review.
3. **Out-of-Distribution Rejection**: Blank, uniform, or corrupted inputs are rejected with clear user-facing alerts rather than silently misclassified.
