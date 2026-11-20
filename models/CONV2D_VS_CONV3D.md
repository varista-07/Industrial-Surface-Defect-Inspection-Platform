# Mathematical Foundations: 2D Convolution vs. 3D Convolution

## 1. Formulation & Dimensionality

In deep convolutional neural networks, the dimensionality of the convolution operation refers to the number of directions in which the convolutional kernel slides across the input tensor:

### 2D Convolution (`Conv2D`)
* **Input Tensor**: Typically a 4D tensor $\mathbf{X} \in \mathbb{R}^{B \times H \times W \times C_{\text{in}}}$, where $B$ is batch size, $H$ is height, $W$ is width, and $C_{\text{in}}$ is input channels (e.g. RGB or feature channels).
* **Kernel Weight Tensor**: $\mathbf{W} \in \mathbb{R}^{K_h \times K_w \times C_{\text{in}} \times C_{\text{out}}}$.
* **Sliding Action**: The kernel moves in **two spatial directions** (along height $H$ and width $W$). Over the channel dimension $C_{\text{in}}$, the kernel spans the *entire* depth and computes an inner product, summing across all channels:
  $$\mathbf{Y}_{b, i, j, k} = b_k + \sum_{c=1}^{C_{\text{in}}} \sum_{m=-M}^M \sum_{n=-N}^N \mathbf{W}_{m, n, c, k} \cdot \mathbf{X}_{b, i+m, j+n, c}$$
* **Output Tensor**: A 4D tensor $\mathbf{Y} \in \mathbb{R}^{B \times H' \times W' \times C_{\text{out}}}$.
* **Use Case**: Static image understanding, surface defect inspection, visual texture classification.

---

### 3D Convolution (`Conv3D`)
* **Input Tensor**: A 5D tensor $\mathbf{X} \in \mathbb{R}^{B \times D \times H \times W \times C_{\text{in}}}$, where $D$ represents a third physical or temporal dimension (e.g. depth in 3D MRI/CT scans, or time frames $T$ in high-speed video inspection).
* **Kernel Weight Tensor**: $\mathbf{W} \in \mathbb{R}^{K_d \times K_h \times K_w \times C_{\text{in}} \times C_{\text{out}}}$.
* **Sliding Action**: The kernel moves in **three distinct directions** (along depth/time $D$, height $H$, and width $W$). It preserves temporal or volumetric structure:
  $$\mathbf{Y}_{b, d, i, j, k} = b_k + \sum_{c=1}^{C_{\text{in}}} \sum_{p=-P}^P \sum_{m=-M}^M \sum_{n=-N}^N \mathbf{W}_{p, m, n, c, k} \cdot \mathbf{X}_{b, d+p, i+m, j+n, c}$$
* **Output Tensor**: A 5D tensor $\mathbf{Y} \in \mathbb{R}^{B \times D' \times H' \times W' \times C_{\text{out}}}$.
* **Use Case**: Volumetric defect scanning (ultrasonic NDT inspection, computed tomography of internal metal voids), and high-speed rolling mill video inspection where defect motion across successive frames indicates conveyor speed or defect propagation.

---

## 2. Comparison Summary

| Attribute | 2D Convolution (`Conv2D`) | 3D Convolution (`Conv3D`) |
| :--- | :--- | :--- |
| **Input Shape** | $(B, H, W, C_{\text{in}})$ | $(B, D, H, W, C_{\text{in}})$ |
| **Kernel Shape** | $(K_h, K_w, C_{\text{in}}, C_{\text{out}})$ | $(K_d, K_h, K_w, C_{\text{in}}, C_{\text{out}})$ |
| **Kernel Degrees of Freedom** | 2 spatial directions | 3 spatio-temporal/volumetric directions |
| **Channel Integration** | Collapses channel dimension via summation | Collapses channel dimension, preserves volumetric depth |
| **Computational Complexity** | $\mathcal{O}(H \cdot W \cdot K_h \cdot K_w \cdot C_{\text{in}} \cdot C_{\text{out}})$ | $\mathcal{O}(D \cdot H \cdot W \cdot K_d \cdot K_h \cdot K_w \cdot C_{\text{in}} \cdot C_{\text{out}})$ |
| **Surface Defect Relevance** | Optimal for static 2D planar images (e.g. NEU-DET steel strips) | Required for temporal video rolling mill monitoring or 3D tomographic internal void detection |
