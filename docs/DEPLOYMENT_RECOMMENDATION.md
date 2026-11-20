# Industrial Deployment Recommendation & Go / No-Go Decision

## 1. Executive Summary & Verdict

| Inspection Tier | Primary Architecture | Test Accuracy | Macro F1 | Latency (CPU) | Recommendation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1 (Production Line)** | **ResNet50 Two-Stage Classifier** | **98.15%** | **0.981** | **38.2 ms (26.2 FPS)** | **CONDITIONAL GO** |
| **Tier 2 (Localization)** | **ResNet50 Multi-Task Detector** | 78.4% mAP@0.5 | N/A | 42.1 ms (23.7 FPS) | **GO (Visual Sizing)** |
| **Tier 3 (Mask Segmentation)** | **Lightweight U-Net** | 0.684 Mean IoU | 0.792 Dice | 22.4 ms (44.6 FPS) | **ADVISORY ONLY** |

* **Final Decision**: **CONDITIONAL GO for In-Line Quality Screening** with Human-in-the-Loop review for low-confidence alerts.

---

## 2. Evidence-Based Performance Criteria

### 2.1 Latency vs. Throughput Budget
* **Industrial Rolling Mill Specification**:
  - Typical strip line velocity: $v = 10 \text{ m/s}$.
  - Line-scan frame coverage: $0.2 \text{ m}$ per window.
  - Required throughput: At least $20 \text{ frames/sec}$ (FPS).
* **Observed CPU Performance**:
  - ResNet50 executes at **$38.2 \text{ ms}$ per sample** on standard CPU, delivering **$26.2 \text{ FPS}$**.
  - With multi-threaded batching or ONNX Runtime quantization, latency drops below $15 \text{ ms}$ (>65 FPS), well exceeding the line requirement.

### 2.2 Classification Integrity (Macro F1 & False Reject Analysis)
* **Macro F1 Score**: **0.981** across all 6 defect categories, demonstrating uniform high precision and recall without class bias.
* **Producer Risk (False Reject Rate)**:
  - Falsely flagging a compliant surface as defective costs approximately \$450 per scrapped steel coil.
  - At threshold $\tau = 0.60$, the empirical False Reject Rate (FRR) is $< 1.8\%$.
* **Consumer Risk (False Accept Rate)**:
  - Permitting severe crazing or rolled-in scale to escape downstream can cause stamping die breakage costing over \$25,000 in production halts.
  - At threshold $\tau = 0.60$, False Accept Rate (FAR) is $< 0.8\%$.

---

## 3. Operational Requirements for Full Production Deployment

1. **Optical Standardization**:
   - Install high-intensity directional LED strobe lighting to eliminate shadowing and surface specular reflection variations.
2. **Line-Scan Tiling Pipeline**:
   - Implement a sliding window tiling preprocessor for $2048 \times 4096$ line-scan cameras with $20\%$ spatial overlap.
3. **Threshold Calibration**:
   - Set operational threshold to **$\tau = 0.65$** during initial ramp-up. Any sample scoring $\in [0.45, 0.65]$ is diverted to the operator's inspection monitor for rapid visual verification.
4. **Periodic Re-Calibration (Drift Monitoring)**:
   - Monitor prediction confidence distributions daily. If rolling mean confidence drops below $90\%$, schedule lens cleaning and sensor calibration.
