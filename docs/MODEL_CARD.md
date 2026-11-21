# Model Card: Surface Defect Inspection Platform Models

## 1. Model Details
* **Developed By**: Student Engineering Team, CSAI3027P Deep Learning PBL Project 02.
* **Evaluation Date**: 23 November 2026.
* **Model Versions**:
  - Primary Classifier: `ResNet50_DefectClassifier` (Two-stage transfer learning with ImageNet weights; Stage 1 frozen backbone, Stage 2 fine-tuning top residual blocks with all `BatchNormalization` layers explicitly frozen).
  - Localization Model: `ResNet50_DefectDetector` (Documented single-box fallback: multi-task classification and bounding box regression head with Smooth L1 + Cross-Entropy Loss).
  - Benchmark Models: `InceptionV3_Benchmark` (Frozen feature extractor) and `From-Scratch MLP (NumPy)`.
  - Segmentation: `Lightweight_UNet` (Trained on weak Otsu-derived pseudo-masks at $128 \times 128$).
  - Generative & Language: `Conv-VAE` ($64 \times 64$) and `DefectReportTransformer` (4-head self-attention on synthetic report strings).
* **Framework**: TensorFlow 2.21, Keras 3.15, and pure NumPy.
* **License**: MIT Open Source for Academic Research.

---

## 2. Intended Use & Deployment Context
* **Intended Use**: Advisory decision support and visual inspection assistance for operators reviewing hot-rolled steel strip surface defect images. Provides predicted defect class, coarse single bounding box localization, and Grad-CAM visual explainability heatmaps.
* **Target Users**: Quality control engineers and manual review station operators.
* **Deployment Context**: Live interactive web platform (`http://localhost:8000`) designed for human-in-the-loop review.

---

## 3. Non-Intended Use
* **Fully Automated Release**: Must **NOT** be used as an autonomous, unmonitored gatekeeper for final product acceptance or release without certified human sign-off.
* **Domain Transfer Without Retraining**: Not intended or validated for materials outside hot-rolled steel strips (e.g., cold-rolled steel, aluminum, copper alloys, silicon wafers, or welded seams) or under radically different camera sensors and factory lighting.
* **Safety-Critical Certification**: Not intended for flight-critical aerospace, nuclear, or structural pressure-vessel component certification without accredited non-destructive testing (NDT) validation.
* **Sub-Millimeter Flaw Metrology**: Bounding boxes and pseudo-masks provide regional guidance and must not be used for precision dimensional metrology or thickness measurement.

---

## 4. Quantitative Performance Summary (Held-Out Test Set: 270 Images)

All metrics below are read directly from `outputs/metrics/*.json`:

| Architecture | Model Paradigm | Input Resolution | Test Accuracy | Test Macro F1 | Measured CPU Latency | Throughput (FPS) | Primary Evaluation Metric |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **From-Scratch MLP** | Pure NumPy Backprop | $32 \times 32$ | 56.67% | Not saved | Not saved | Not saved | Test Loss: 1.1545 |
| **InceptionV3 Benchmark** | Frozen Transfer Learning | $200 \times 200 \times 3$ | 97.78% | 0.9778 | 192.85 ms | 5.19 FPS | 264 / 270 correct |
| **ResNet50 (Fine-Tuned)** | Two-Stage (Frozen BN) | $200 \times 200 \times 3$ | **100.00%** | **1.0000** | 179.52 ms<br/>*(logs: 260–319 ms)* | **5.57 FPS** | **270 / 270 correct on seed-42 split** |
| **Multi-Task Detector** | ResNet50 Single-Box Head | $200 \times 200 \times 3$ | — | — | 200.07 ms | 5.00 FPS | **mAP@0.5: 24.07%**, Mean IoU: 0.3292 |
| **U-Net Segmentation** | Lightweight U-Net | $128 \times 128 \times 3$ | — | — | 27.36 ms | 36.54 FPS | **Mean IoU: 0.1292**, Dice: 0.1950 (weak labels) |
| **Report Transformer** | 4-Head Self-Attention | Text tokens | 100.00% | Not saved | Not saved | Not saved | Test Loss: 0.0003 (synthetic reports) |

### Multi-Seed Classifier Robustness (ResNet50)
Across 3 independent train/validation/test splits (`experiments/seed_robustness_results.json`):
* **Seed 42**: 100.00% (270 / 270)
* **Seed 101**: 95.93% (259 / 270, 11 misclassifications)
* **Seed 202**: 97.04% (262 / 270, 8 misclassifications)
* **Mean Accuracy**: **97.66% ± 2.10%** (std: 2.10 percentage points, standard error: 1.21 points)
* **95% Confidence Interval**: **[92.43%, 100.00%]** ($t_{\text{crit}} = 4.3027, n=3$)

### Per-Class Metrics Breakdown (ResNet50, Seed 42 Split)
* **Crazing**: Precision: 1.000 | Recall: 1.000 | F1: 1.000 (Support: 45)
* **Inclusion**: Precision: 1.000 | Recall: 1.000 | F1: 1.000 (Support: 45)
* **Patches**: Precision: 1.000 | Recall: 1.000 | F1: 1.000 (Support: 45)
* **Pitted Surface**: Precision: 1.000 | Recall: 1.000 | F1: 1.000 (Support: 45)
* **Rolled-in Scale**: Precision: 1.000 | Recall: 1.000 | F1: 1.000 (Support: 45)
* **Scratches**: Precision: 1.000 | Recall: 1.000 | F1: 1.000 (Support: 45)

---

## 5. Object Detection Architecture & Fallback Scope

* **Architecture**: ResNet50 backbone with a single-box multi-task prediction head (`models/detector.py`).
* **Documented Fallback**: This model serves as an explicitly documented fallback. Single-stage RetinaNet (`models/retinanet_benchmark.py`) was evaluated on CPU; measured step latency was 46.79 seconds/step (>8.1 hours per epoch on CPU), exceeding practical training feasibility. RetinaNet was not trained for final deployment.
* **Detector Performance**:
  * Overall **mAP@0.5 = 24.07%** (`0.2407`), **Mean IoU = 0.3292**
  * Per-class AP@0.5:
    - Crazing: 22.22%
    - Inclusion: 17.78%
    - Patches: 6.67%
    - Pitted Surface: 86.67%
    - Rolled-in Scale: 6.67%
    - Scratches: 4.44%
* **Known Detection Bottleneck**: The single-box head predicts exactly one box per image. However, 73.17% of NEU-DET images contain $\ge 2$ defect instances (mean 2.33 boxes/image, up to 9 instances), which caps detection mAP on multi-instance clusters.

---

## 6. Known Limitations

1. **Weak Supervision in U-Net**: Evaluated against weak Otsu pseudo-masks generated from bounding boxes; true human pixel masks do not exist in NEU-DET.
2. **Circular Transformer Evaluation**: The Report Transformer achieves 100.00% accuracy on synthetic inspection text generated deterministically from the same bounding box annotations as the labels.
3. **No Gain from Generative VAE Augmentation**: In `experiments/experiment_log.csv`, VAE data augmentation produced conflicting results across runs (+14.07 points from 57.0% to 71.1% on initial run; -16.67 points from 73.3% to 56.7% on clean leakage-free run), demonstrating no reliable benefit due to pixel-space reconstruction smoothing fine edge textures.
4. **Unmeasured True False-Reject Rate**: Because NEU-DET contains no clean/defect-free images, true false-reject rate cannot be measured.

---

## 7. Low-Confidence Flagging for Manual Review (Safeguards)

1. **Low-Confidence Flagging**: The web interface implements a confidence threshold slider (default 0.60, adjustable from 0.20 to 0.95). Any sample scoring $\max P < \tau$ triggers an advisory alert (*low-confidence flagging (confidence-threshold rule)*), routing the strip to an operator for visual verification.
2. **Heuristic Nature**: This safeguard is a confidence-threshold and low-contrast texture heuristic ($\sigma < 0.025$), **NOT a true out-of-distribution (OOD) detector or density estimator**.
3. **Explainability**: Grad-CAM visual heatmaps are generated alongside predictions to assist human review by displaying convolutional saliency patterns.

---

## 8. Ethical, Safety & Privacy Considerations

* **Data Privacy**: The dataset consists entirely of microscopic grayscale optical photographs of industrial steel strip surfaces. It contains **no personally identifiable information (PII)**, human subjects, biometric data, or sensitive proprietary metadata.
* **Worker Impact**: Designed strictly as an operator assistance tool to alleviate visual fatigue in repetitive manual quality inspection, not to eliminate human oversight.
* **Environmental & Compute Cost**: Training was completed within academic CPU resource budgets with negligible carbon impact.
