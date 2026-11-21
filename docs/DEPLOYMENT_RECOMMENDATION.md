# Industrial Deployment Recommendation & Feasibility Assessment

## 1. Executive Summary & Verdict

* **Deployment Verdict**: **ADVISORY DECISION SUPPORT with Mandatory Human Review**.  
* **Operational Constraint**: **NOT validated for autonomous automated accept/reject or inline production release**.

The platform is recommended solely as a secondary visual aid and triage tool to assist quality inspectors at manual review stations. It must not be deployed as an autonomous inline gatekeeper on active rolling lines.

---

## 2. Empirical Evidence Table

All metrics below reflect real, verified measurements recorded in `outputs/metrics/test_metrics.json`, `outputs/metrics/inceptionv3_test_metrics.json`, `experiments/seed_robustness_results.json`, and `experiments/experiment_log.csv`. No numbers are estimated or extrapolated.

| Module | Architecture | Test Split Metric (Seed 42) | Multi-Seed Robustness | Measured Latency (CPU) | Measured Throughput (CPU) | Operational Readiness |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Primary Classifier** | ResNet50 (Two-Stage, Frozen BN) | **100.00% Accuracy** (270/270)<br/>Macro F1: 1.000 | Mean: **97.66% ± 2.10%**<br/>95% CI: [92.43%, 100.00%] (n=3) | 179.52 ms (test-time)<br/>*(logs show 260.45 ms & 318.95 ms)* | **5.57 FPS** | **Advisory decision support only** |
| **Benchmark Classifier** | InceptionV3 (Frozen Feature Extractor) | **97.78% Accuracy** (264/270)<br/>Macro F1: 0.9778 | Not saved for seeds 101/202 | 192.85 ms | **5.19 FPS** | Benchmark reference |
| **Object Detector** | ResNet50 Multi-Task Single-Box Head | **mAP@0.5: 24.07%**<br/>Mean IoU: 0.3292 | Not saved | 200.07 ms | **5.00 FPS** | **Weak; coarse visual guide only** |
| **Segmentation** | Lightweight U-Net | **Mean IoU: 0.1292**<br/>Dice Score: 0.1950 | Not saved | 27.36 ms | **36.54 FPS** | **Advisory only; weak pseudo-masks** |
| **Baseline Classifier** | 3-Layer MLP (NumPy, 32x32 input) | **56.67% Accuracy** (Test Loss: 1.1545) | Not saved | Not saved | Not saved | Theoretical baseline only |

---

## 3. Inline Throughput & Hardware Reality

* **Measured Throughput**: Single-image CPU inference throughput for the primary classifier is **5.57 FPS** (mean latency 179.52 ms on native CPU; training-time logs recorded 260.45 ms and 318.95 ms depending on system background load).
* **Line-Speed Feasibility**: Suitability for high-speed inline manufacturing lines (e.g. continuous strip speeds of 5–15 m/s requiring $>30\text{ FPS}$) was **NOT demonstrated**.
* **Hardware Scope**: All benchmarks were executed on an unquantized TensorFlow/Keras CPU runtime. No GPU acceleration, TensorRT compilation, INT8/FP16 quantization, or multi-threaded batched pipeline runs were measured.

---

## 4. Technical Risks & Validation Gaps

1. **Absence of a Defect-Free Class & True False-Reject Rate Gap**:
   - The NEU-DET benchmark contains exclusively defective surfaces across 6 fault classes; it contains **zero unblemished/clean steel samples**.
   - As a consequence, a true industrial False-Reject Rate (FRR: probability of incorrectly scrapping a defect-free strip) and False-Accept Rate (FAR: probability of passing a defective strip as clean) **cannot be measured** on this dataset.
   - On the held-out test split, operational safety was evaluated via threshold triage:
     - **Triage Escalation Rate** ($\max P < \tau$): 0.00% across thresholds $\tau \in [0.50, 0.90]$ (0 / 270 samples routed to human review).
     - **Silent Escape Rate** ($\hat{y} \neq y \land \max P \ge \tau$): 0.00% across thresholds $\tau \in [0.50, 0.90]$ (0 / 270 samples misclassified with high confidence).
2. **Single Dataset & Single Material Domain**:
   - The platform was trained and tested solely on the academic NEU-DET dataset (hot-rolled steel strips). It has not been validated on cold-rolled steel, galvanized sheets, non-ferrous alloys (aluminum, copper), or alternate rolling mills with differing optical contrast.
3. **Low Input Resolution**:
   - Images are downsampled to $200 \times 200$ pixels. Industrial line-scan inspection systems capture high-resolution web streams ($2048 \times 4096$ pixels); sub-millimeter micro-fissures may be lost during downsampling.
4. **Detector Multi-Instance Bottleneck**:
   - The current object detector architecture outputs exactly one bounding box per image. However, 73.17% of NEU-DET images contain 2 or more defect instances (up to 9 instances), capping detector mAP@0.5 at 24.07%.
5. **Weak Pseudo-Mask Supervision**:
   - U-Net metrics (IoU 0.1292, Dice 0.1950) were evaluated against synthetic Otsu thresholded bounding boxes rather than true human-annotated pixel-level ground truth.
6. **Uncalibrated Model Confidence**:
   - Raw softmax probabilities were not calibrated using temperature scaling, Platt scaling, or Expected Calibration Error (ECE) analysis.
7. **Heuristic Low-Confidence Flagging**:
   - Low-confidence and uniform-surface detection rely on an empirical texture variance threshold ($\sigma < 0.025$) and softmax cutoff ($\tau$, default 0.60, adjustable from 0.20 to 0.95), which is a heuristic safeguard rather than a true density-based out-of-distribution detector.

---

## 5. Mandatory Prerequisites Before Industrial Production Release

The following engineering steps are strictly required as recommendations before any production deployment:

1. **Target-Line Data Collection**: Collect and label representative image streams from the specific target mill, explicitly including a certified defect-free / clean surface class.
2. **Statistical Calibration**: Apply temperature scaling and compute ECE to ensure softmax scores correspond to true metallurgical error probabilities.
3. **Formal OOD & Anomaly Testing**: Replace threshold heuristics with true out-of-distribution density estimators (e.g. Mahalanobis distance on intermediate feature embeddings, or normalizing flows).
4. **Multi-Object Detection Upgrade**: Replace the single-box multi-task head with a multi-box detection architecture (e.g. RetinaNet, Faster R-CNN, or YOLO) capable of detecting all co-occurring defect instances.
5. **Human Pixel Annotations**: Acquire verified pixel-level segmentation masks annotated by metallurgical domain experts before relying on segmentation for dimensional flaw metrology.
6. **Target Hardware Profiling**: Benchmark batched and quantized (INT8/FP16) inference on edge accelerators (e.g. NVIDIA Jetson, TensorRT, OpenVINO) under actual line-scan trigger constraints.
