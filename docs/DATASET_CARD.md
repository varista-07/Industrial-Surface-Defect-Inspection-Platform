# Dataset Card: NEU-DET Industrial Surface Defect Dataset

## 1. Dataset Summary
The **NEU-DET (Northeastern University Surface Defect Database - Detection subset)** is an industrial computer vision benchmark developed by the Key Laboratory of Integrated Automation of Process Industry (Ministry of Education), Northeastern University, Shenyang, China (Song & Yan, 2013). The dataset contains hot-rolled steel strip surface defects collected in real production lines to support automated defect inspection, localization, and classification.

---

## 2. Dataset Structure & Specifications

| Property | Value / Specification |
| :--- | :--- |
| **Total Images** | 1,800 surface images |
| **Total Pascal VOC XML Annotations** | 1,800 XML files |
| **Total Defect Instances (Bounding Boxes)** | 4,189 annotated bounding boxes |
| **Defect Classes** | 6 distinct surface defect classes (300 images per class) |
| **Native Image Resolution** | $200 \times 200$ pixels |
| **Color Space / Channels** | Grayscale (single channel), stored as 8-bit JPEG |
| **Annotation Format** | Pascal VOC XML (`<object>` with `<name>` and `<bndbox>` coordinates $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$) |
| **Storage Location** | `data/raw/NEU-DET/` (`IMAGES/` and `ANNOTATIONS/`) |

---

## 3. Defect Taxonomy & Class Distribution

The dataset exhibits exact balanced sampling at the image level (300 images per class), with natural morphology-dependent bounding box instance frequencies:

| Class Name | Image Count | Instance Count | Mean Instances/Image | Typical Morphology & Appearance |
| :--- | :--- | :--- | :--- | :--- |
| **Crazing** (`crazing`) | 300 | 689 | 2.30 | Web-like, fine, interconnected surface micro-cracks |
| **Inclusion** (`inclusion`) | 300 | 1,011 | 3.37 | Small, irregular non-metallic particle impurities |
| **Patches** (`patches`) | 300 | 881 | 2.94 | Localized surface deformation or flattened peelings |
| **Pitted Surface** (`pitted_surface`) | 300 | 432 | 1.44 | Crater-like depressions caused by localized oxidation/abrasion |
| **Rolled-in Scale** (`rolled-in_scale`) | 300 | 628 | 2.09 | Dark iron-oxide scale pressed into the strip during rolling |
| **Scratches** (`scratches`) | 300 | 548 | 1.83 | Linear, continuous mechanical abrasions spanning long distances |
| **Total** | **1,800** | **4,189** | **2.33** | Full Dataset |

---

## 4. Multi-Box Distribution Analysis

A key characteristic of industrial defect inspection is that a single surface patch frequently exhibits multiple defect instances (e.g., clusters of inclusions or multiple scratch tracks):

| Bounding Boxes per Image | Number of Images | Percentage of Dataset | Cumulative Percentage |
| :---: | :---: | :---: | :---: |
| **1 box** | 483 | 26.83% | 26.83% |
| **2 boxes** | 673 | 37.39% | 64.22% |
| **3 boxes** | 390 | 21.67% | 85.89% |
| **4 boxes** | 154 | 8.56% | 94.44% |
| **5 boxes** | 59 | 3.28% | 97.72% |
| **6 boxes** | 22 | 1.22% | 98.94% |
| **7 boxes** | 9 | 0.50% | 99.44% |
| **8 boxes** | 6 | 0.33% | 99.78% |
| **9 boxes** | 4 | 0.22% | 100.00% |

> [!NOTE]
> Over **73.17% of images** contain 2 or more bounding boxes. Consequently, single-box regression heads only capture the dominant defect; true multi-object detection (e.g. RetinaNet) is necessary to isolate every defect instance.

---

## 5. Cryptographic Hash Audit & Duplicate Image Detection

To evaluate data integrity, an exhaustive cryptographic audit was conducted across all 1,800 raw images using both SHA-256 and MD5 hashing algorithms (`prepare_data.py`):

| Audit Metric | Result |
| :--- | :--- |
| **Total Image Files Audited** | 1,800 |
| **Total Unique SHA-256 Hashes** | 1,799 |
| **Total Unique MD5 Hashes** | 1,799 |
| **Duplicate Image Files Detected** | 2 (`patches_101.jpg`, `patches_105.jpg`) |
| **Duplicate SHA-256 Hash** | `4d2de82731b86cdbc7a66f2a9bfb01074bb4cb65e47bccf06b66470d53857071` |
| **Annotation Difference** | 1–2 pixel bounding-box coordinate jitter between XML files |

### Analysis of Duplicate Pair
The files `patches_101.jpg` and `patches_105.jpg` are byte-for-byte identical images present in the original NEU-DET release. Their corresponding XML annotations describe the same 3 surface patch defects with slight coordinate differences of 1–2 pixels (likely from duplicate re-annotation passes by annotators).

---

## 6. Stratified Split & Zero Cross-Split Leakage Guarantee

To prevent optimistic bias and guarantee valid generalization estimates, the dataset is partitioned into a 70% Train, 15% Validation, and 15% Test split with deterministic random seed `SEED = 42`:

| Split Name | Image Count | Percentage | Per-Class Image Count | Unique Hashes | Split Role |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Train** | 1,260 | 70.0% | 210 | 1,259 | Model training and backpropagation |
| **Validation** | 270 | 15.0% | 45 | 270 | Hyperparameter tuning and model checkpointing |
| **Test** | 270 | 15.0% | 45 | 270 | Unbiased holdout benchmark (never seen during training) |
| **Total** | **1,800** | **100.0%** | **300** | **1,799** | — |

### Leakage Verification Protocol
To eliminate cross-split contamination caused by the duplicate image pair:
1. The duplicate pair (`patches_101.jpg` and `patches_105.jpg`) is deterministically placed **strictly within the Train split** (`train_cluster`).
2. Set-intersection verification between split hash sets confirms complete isolation:
   - $\text{Train} \cap \text{Validation} = \emptyset$ (0 collisions)
   - $\text{Train} \cap \text{Test} = \emptyset$ (0 collisions)
   - $\text{Validation} \cap \text{Test} = \emptyset$ (0 collisions)
3. Audit output is programmatically validated and saved in [hash_audit.json](data/splits/hash_audit.json).

---

## 7. Limitations & Industrial Deployment Considerations

1. **Resolution & Scale Invariance**: Native images are $200 \times 200$ pixels. High-speed continuous strip lines produce high-resolution line-scan streams ($2048 \times 4096$), necessitating a sliding-window patch extraction or tiling pipeline prior to inference.
2. **Illumination Variability**: Directional lighting is relatively uniform across NEU-DET. Real factory environments introduce variable specular reflection, oil films, and sensor noise requiring online contrast normalization.
3. **Absence of Clean Surfaces**: NEU-DET contains solely defective surfaces. In production, a two-stage filter using out-of-distribution detection (e.g., maximum softmax probability thresholding or reconstruction-error gating) must classify unblemished steel as "PASS" before defect localization.
