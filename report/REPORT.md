# Technical Project Report: Industrial Surface Defect Inspection Platform (CSAI3027P)

**Course**: CSAI3027P Deep Learning  
**Project Track**: Project 02: Industrial Surface Defect Inspection Platform (Manufacturing Quality Assurance)  
**Evaluation Date**: 23 November 2026  
**Dataset**: NEU-DET (Northeastern University Surface Defect Database)  
**Target Environment**: Live Web Inspection Station (`http://localhost:8000`)

---

## 1. Problem Formulation & Industrial Objectives

In high-speed hot-rolled steel strip manufacturing, visual surface defects (such as micro-cracks, non-metallic inclusions, and rolling scratches) degrade structural fatigue resistance and cause severe stamping failures during automotive fabrication. Industrial quality assurance systems face four primary technical challenges:
1. **Multi-Scale Defect Morphology**: Surface defects vary from microscopic craze cracking spanning sub-millimeter widths to continuous rolling scratches extending across entire coil widths.
2. **Real-Time Latency Budget**: Continuous strip lines travel at velocities $> 10 \text{ m/s}$, imposing strict latency constraints on inference systems ($< 50 \text{ ms}$ per visual inspection window).
3. **Imperfect & Weak Supervision**: Manual pixel-level semantic masks are cost-prohibitive in industrial settings; datasets provide only coarse bounding boxes.
4. **Safety & Out-of-Distribution Vulnerability**: Overconfident misclassification on unblemished steel produces costly false rejections (producer risk), while missed critical defects cause catastrophic field failures (consumer risk).

This project resolves these challenges by engineering an end-to-end industrial inspection platform featuring neural network fundamentals from scratch, state-of-the-art transfer learning classifiers, multi-task defect localization, explainable AI (Grad-CAM), and a live operator inspection dashboard.

---

## 2. Dataset Integrity Audit & Cryptographic Verification

The platform is evaluated on the official NEU-DET benchmark (1,800 images, 1,800 Pascal VOC XML annotations, 6 classes: `crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`).

### 2.1 Cryptographic Hash Audit & Duplicate Discovery
To evaluate data integrity, SHA-256 and MD5 cryptographic hashes were computed for every image in the dataset (`prepare_data.py`):
* **Total Image Files Audited**: 1,800
* **Unique Cryptographic Hashes**: 1,799
* **Duplicate Detection**: Discovered an exact byte-for-byte duplicate pair in the official benchmark: `patches_101.jpg` and `patches_105.jpg` (SHA-256: `4d2de82731b86cdbc7a66f2a9bfb01074bb4cb65e47bccf06b66470d53857071`).
* **Annotation Comparison**: Their corresponding XML files describe identical defect locations with slight 1–2 pixel boundary coordinate jitter resulting from duplicate human annotation passes.

### 2.2 Multi-Box Instance Distribution
An audit of all 4,189 Pascal VOC bounding boxes revealed that **73.17% of images contain 2 or more defect instances** (up to 9 boxes per image):
* 1 box: 483 images (26.83%)
* 2 boxes: 673 images (37.39%)
* 3 boxes: 390 images (21.67%)
* $\ge 4$ boxes: 254 images (14.11%)
* Mean instances per image: $2.33$ (Inclusions: $3.37$, Patches: $2.94$, Crazing: $2.30$).

### 2.3 Stratified Split & Zero-Leakage Guarantee
To eliminate data snooping and ensure unbiased evaluation:
1. The dataset was partitioned into a **70% Train (1,260 images), 15% Validation (270 images), and 15% Test (270 images)** split stratified at exactly 210 train, 45 val, and 45 test images per class (`SEED = 42`).
2. The duplicate pair (`patches_101.jpg` and `patches_105.jpg`) was deterministically grouped into the **Train partition**.
3. Set intersection across split hash sets confirmed complete isolation:
   $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$
   Programmatically verified in [hash_audit.json](file:///c:/Users/Acer/PBL%20PROJECT%20DL/data/splits/hash_audit.json).

---

## 3. Part A: Neural Network Fundamentals from Scratch (NumPy)

To demonstrate foundational theoretical mastery without framework abstractions, Part A implements neural network primitives in pure NumPy:

1. **Tensor Algebra & Broadcasting**:
   - Implemented tensor operations across ranks 0 through 4 (scalar to 4D mini-batch image tensors $[B, H, W, C]$).
   - Validated broadcasting rules in feature normalization, matrix multiplications, and Einstein summation (`np.einsum("bi,io->bo", X, W)`).
2. **Spectral SVD Analysis**:
   - Singular value spectrum decay showed that rank-20 reconstruction preserves **99.29% of total spectral energy** with $4.99\times$ compression and **25.37 dB** PSNR.
3. **Neuron Foundations**:
   - Proved linear inseparability of XOR for single McCulloch-Pitts neurons.
   - Trained logistic sigmoid neurons via binary cross-entropy gradient descent on defect feature spaces.
4. **Object-Oriented MLP with Analytical Backpropagation**:
   - Built modular components (`LinearLayer`, `ReLU`, `Dropout`, `BatchNorm1D`, `SoftmaxCrossEntropy`).
   - Implemented exact multivariate chain-rule analytical backpropagation.
   - Trained on downscaled $32 \times 32$ defect images as the baseline model, achieving **56.67% test accuracy** (against a 16.67% random baseline).
5. **Optimizer Benchmark**:
   - Compared Batch Gradient Descent (BGD), Stochastic Gradient Descent (SGD), Momentum ($\beta=0.9$), and Nesterov Accelerated Gradient (NAG).
   - NAG exhibited the fastest convergence (final loss $0.818$ vs BGD $1.675$), proving the theoretical benefit of lookahead gradient evaluation.
6. **High-Precision Gradient Verification**:
   - Centered finite difference checking ($\epsilon = 10^{-6}$) verified analytical gradients with relative errors $< 3 \times 10^{-10}$ in double precision.
7. **Unsupervised & Energy-Based Learning**:
   - Proved unconstrained Hebbian learning explodes exponentially ($\|\mathbf{w}\| \to \infty$).
   - Implemented Oja's rule, verifying asymptotic convergence to the maximal eigenvector of data covariance (1st Principal Component alignment $> 0.998$).
   - Trained a Restricted Boltzmann Machine (RBM) via 1-step Contrastive Divergence (CD-1) on binary defect patches.
8. **Automated Unit Testing**:
   - Automated pytest test suite (`tests/test_fundamentals.py`) passed all 6 test suites covering tensor broadcasting, MLP backprop shapes, optimizer updates, Oja's rule, SVD reconstruction, and numerical gradient verification in 2.24s.

---

## 4. Part B: Deep Transfer Learning & Localization Models (TensorFlow)

### 4.1 Primary Classifier: ResNet50 Two-Stage Transfer Learning
* **Stage 1 (Frozen Backbone)**: ImageNet-pretrained feature extractor frozen, training the custom dense head with Adam ($\eta = 10^{-3}$, 4 epochs).
* **Stage 2 (Fine-Tuning with Frozen BatchNormalization)**: Top convolutional blocks (`conv5_block*`) unfrozen with reduced learning rate ($\eta = 10^{-4}$). Crucially, **all `BatchNormalization` layers remained strictly frozen (`layer.trainable = False`)**:
  - *Theoretical Justification*: In small domain batches (16 or 32 samples), recalculating mini-batch sample mean and variance introduces high sampling variance, which corrupts the pre-trained ImageNet running statistics and causes catastrophic feature degradation. Keeping BN layers in inference mode stabilizes gradient flow through residual skip connections.
* **Test Performance (270 Unseen Holdout Images)**:
  - Test Accuracy: **100.00%**
  - Test Macro F1: **1.0000**
  - Per-Class Precision, Recall, and F1: **100.0%** across all 6 classes (`crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`).
  - CPU Inference Latency: **318.95 ms** per image (3.13 FPS).

### 4.2 Architecture Benchmark & Ablations
* **MLP Baseline (From-Scratch)**: 56.67% Test Accuracy | 0.5480 Macro F1 | ~0.4 ms Latency.
* **InceptionV3 Benchmark**: 97.78% Test Accuracy | 0.9778 Macro F1 | 192.80 ms Latency.
* **ResNet50 Ablations**:
  - Frozen Backbone Only: 100.00% Test Accuracy | 1.0000 Macro F1.
  - No Augmentation: 100.00% Test Accuracy | 1.0000 Macro F1.
  - With Class Weights: 100.00% Test Accuracy | 1.0000 Macro F1.
* **Selected Final Model**: ResNet50 Two-Stage Fine-Tuned with Frozen BatchNormalization.

### 4.3 Defect Detection: Single-Stage RetinaNet Benchmark & Fallback Justification
Following the project directives, a real single-stage RetinaNet (`keras-cv`) with a ResNet50 backbone was implemented and benchmarked on CPU (`models/retinanet_benchmark.py`):
1. **RetinaNet CPU Benchmark Results**:
   - Single mini-batch step latency (batch size 2): **46.79 seconds per step**.
   - Projected 1-epoch training time (1,260 images / 630 steps): **491.2 minutes (~8.19 hours per epoch)**.
   - A standard 10-epoch run would require over **81.8 hours** on CPU, directly violating the 30-minute training time limit.
   - Furthermore, Keras-CV 0.9.0 exhibited data-unpacking incompatibilities with Keras 3's 3-element tuple structure and required dimensions divisible by 32.
2. **Justified Fallback to Multi-Task Detection Head**:
   - To provide real-time inference within the hardware constraint, the platform fell back to the multi-task shared-backbone architecture (`models/detector.py`).
   - **Test Results**: Mean IoU: **0.3292**, overall mAP@0.5: **24.07%** (`pitted_surface` AP@0.5: **86.67%**), inference latency: **21.22 ms** (47.1 FPS on CPU).

### 4.4 Weak Pseudo-Mask Segmentation (U-Net)
Because NEU-DET lacks human ground-truth pixel masks, weak pseudo-labels were synthesized from bounding box fills combined with localized Otsu contrast thresholding:
* Lightweight U-Net trained at $128 \times 128$:
  - **Mean IoU**: **0.1292** (Against weak pseudo-masks)
  - **Mean Dice Score**: **0.1950** (Pixel-level F1 overlap)
  - **Inference Speed**: **27.40 ms** (36.5 FPS on CPU)
  - **Instance Analysis**: Connected components analysis (`scipy.ndimage`) successfully isolates discrete spatial defect clusters.
  - *Explicit Caveat*: Evaluated against synthesized pseudo-labels; human pixel-level ground truth does not exist in the official NEU-DET dataset.

### 4.5 Industrial False-Reject Rate (FRR) & Manual Triage Analysis
Because NEU-DET contains solely defective samples, false-reject behavior is defined via two empirical operational metrics:
1. **Missed Defect Rate ($\text{FRR}_{\text{miss}}$)**: Fraction of defective test samples whose top predicted probability fails to exceed confidence threshold $\theta$.
2. **Manual Triage Escalation Rate**: Fraction of images where model uncertainty warrants automatic escalation to a human QC inspector.

Across all 270 test samples:
* Threshold 0.50: **0.00%** triage rate (0/270 samples flagged)
* Threshold 0.60: **0.00%** triage rate (0/270 samples flagged)
* Threshold 0.70: **0.00%** triage rate (0/270 samples flagged)
* Threshold 0.80: **0.00%** triage rate (0/270 samples flagged)
* Threshold 0.90: **0.37%** triage rate (1/270 samples flagged for manual review)

---

## 5. Part C: Advanced Modules

1. **Synthetic Defect Report Transformer**:
   - Synthesized structured natural language inspection reports from Pascal VOC bounding box geometry (e.g., *"Severe irregular clustered crazing defect identified near central region spanning approximately 73 percent surface area"*).
   - Trained a 4-head Transformer Encoder in TensorFlow (Token & Positional Embedding, Multi-Head Attention, Pre-LN Feed-Forward), achieving **100.00% text classification accuracy** (Loss: 0.0003).
   - Generated 4-head self-attention heatmaps (`outputs/figures/transformer_attention_weights.png`) illustrating token focus on severity and morphology.
2. **Convolutional Variational Autoencoder (Conv-VAE)**:
   - Trained on $64 \times 64$ defect crops with latent dimension $d = 32$ using the reparameterization trick $z = \mu + \sigma \odot \epsilon$.
   - Final reconstruction loss: **2594.66**, KL divergence loss: **13.27**.
   - 2D PCA projection of the latent space explains **79.16%** of total latent variance with clear defect clustering.
   - Sampled novel synthetic defects from random Gaussian prior $z \sim \mathcal{N}(0, \mathbf{I})$.
   - **Synthetic Augmentation Test**: Evaluating classifier performance on 64x64 crops without synthetic data (57.04%) vs with 120 VAE synthetic samples (71.11%), showing a $+14.07\%$ delta on the small baseline model while noting mild blurring characteristics.

---

## 6. Part D: The Interactive Inspection Platform Web App

* **Backend**: High-throughput asynchronous FastAPI framework (`app/main.py`).
* **Frontend**: Responsive single-page industrial control dashboard (`app/static/index.html`, `style.css`, `app.js`).
* **Core Capabilities**:
  1. Live Image Upload with drag-and-drop & file picker.
  2. Real-time Defect Classification & 6-class probability distribution bar chart.
  3. Interactive Confidence Threshold Slider (range 20% to 95%).
  4. Out-of-Distribution / Uncertainty alert banner when max confidence falls below threshold.
  5. Multi-Modal Quad-View Visual Matrix: Original Surface, Bounding Box, Grad-CAM Saliency Heatmap, and Morphological Mask.
  6. Single-command launch via `python run_app.py`.
* **Automated Live Test Verification (`tests/test_api_live.py`)**:
  - Passed all 6/6 tests covering `/health`, `/samples`, real defect inference, uniform surface OOD rejection, unsupported extension rejection, and corrupt file interception.

---

## 7. Viva Defense & Technical Q&A Summary

1. **Why freeze BatchNormalization during fine-tuning?**  
   BatchNormalization calculates running mean and variance via exponential moving average. With small custom domain mini-batches on CPU, calculating sample batch statistics introduces high variance that destroys pre-trained ImageNet feature distributions. Freezing BN keeps running statistics fixed as a constant affine transformation.
2. **Why fall back from RetinaNet to the multi-task detection head?**  
   Empirical benchmarking demonstrated that a single step of Keras-CV RetinaNet took 46.79 seconds for batch size 2 on CPU, requiring over 8 hours per epoch (81.8 hours for 10 epochs), severely violating the 30-minute training budget. The multi-task shared head achieved 21.2 ms latency (47.1 FPS) with 24.07% mAP@0.5 (86.67% on pitted surface).
3. **How is the False-Reject Rate defined on a dataset without clean surfaces?**  
   FRR is defined as the Missed Defect Rate ($\max P < \theta$) plus the Operator Triage Escalation Rate. At an 80% threshold, the triage rate is 0.00%; at a 90% threshold, only 1 of 270 test samples is escalated for human inspection.
4. **Did you find any dataset anomalies in NEU-DET?**  
   A cryptographic hash audit revealed that `patches_101.jpg` and `patches_105.jpg` are 100% byte-for-byte identical images. Both duplicates were isolated strictly inside the Train split, guaranteeing zero cross-split leakage.
