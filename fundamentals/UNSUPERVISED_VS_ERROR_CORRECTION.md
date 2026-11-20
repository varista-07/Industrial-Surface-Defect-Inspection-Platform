# Theoretical Foundations: Unsupervised, Associative, and Energy-Based Learning vs. Error-Correction Learning

## 1. Paradigm Overview

In deep learning, neural architectures are trained under fundamentally different learning paradigms. The table below summarizes their core distinctions:

| Paradigm | Objective / Driving Force | Target Signal | Weight Update Formulation | Representative Models |
| :--- | :--- | :--- | :--- | :--- |
| **Error-Correction Learning** | Minimize supervised objective function $\mathcal{L}(\hat{y}, y)$ via gradient descent | Ground-truth labels $y$ | $\Delta \mathbf{w} = -\eta \nabla_\mathbf{w} \mathcal{L} = \eta (y - \hat{y}) \mathbf{x}$ | Perceptron, Backpropagation, MLP, ResNet |
| **Classical Hebbian Learning** | Strengthen synaptic coupling proportional to simultaneous pre- and post-synaptic activity | None (correlation-driven) | $\Delta \mathbf{w} = \eta \, y \, \mathbf{x}$ | Hebbian Associator (1949) |
| **Modified Hebbian (Oja's Rule)** | Maximize output variance under an intrinsic $L_2$ weight normalization constraint | None (PCA feature discovery) | $\Delta \mathbf{w} = \eta \, y \, (\mathbf{x} - y \mathbf{w})$ | Oja's Subspace Network (1982) |
| **Competitive Learning** | Vector quantization / cluster centroid discovery via winner-take-all (WTA) | None (distance minimization) | $\Delta \mathbf{w}_{i^*} = \eta (\mathbf{x} - \mathbf{w}_{i^*})$ | Kohonen SOM, Vector Quantization (VQ) |
| **Boltzmann / Energy-Based (RBM)** | Minimize energy of observed data configurations: $\min D_{KL}(P_{\text{data}} \parallel P_{\text{model}})$ | None (probabilistic density) | $\Delta W = \eta \left( \mathbb{E}_{\text{data}}[\mathbf{v}\mathbf{h}^T] - \mathbb{E}_{\text{model}}[\mathbf{v}\mathbf{h}^T] \right)$ | Restricted Boltzmann Machine, DBM |

---

## 2. In-Depth Comparative Analysis

### 2.1 Error-Correction Learning (Supervised Backpropagation)
* **Mechanics**: Computes explicit error residuals $e = y - \hat{y}$ between external targets and forward predictions. Gradients propagate backwards through the network via the calculus chain rule.
* **Strengths**: Highly sample-efficient for goal-directed tasks like defect classification and bounding box regression where labeled ground truth exists.
* **Limitations**: Relies heavily on human annotation, suffers from catastrophic forgetting, and does not model the underlying joint distribution $P(\mathbf{x})$.

### 2.2 Classical Hebbian Learning & Instability
* **Mechanics**: Formalized as "neurons that fire together, wire together." Weight update $\Delta \mathbf{w} = \eta y \mathbf{x} = \eta (\mathbf{w}^T \mathbf{x}) \mathbf{x}$.
* **Mathematical Instability**: Taking expectations over input distribution with covariance $C = \mathbb{E}[\mathbf{x}\mathbf{x}^T]$ yields $\mathbb{E}[\Delta \mathbf{w}] = \eta C \mathbf{w}$. Because covariance matrices are positive semi-definite, weights grow exponentially without bound ($\|\mathbf{w}\|_2 \to \infty$).

### 2.3 Modified Hebbian Learning (Oja's Rule)
* **Mechanics**: Erkki Oja (1982) introduced a local weight decay term derived from a first-order Taylor expansion of spherical normalization ($\mathbf{w}_{t+1} = \frac{\mathbf{w}_t + \eta y \mathbf{x}}{\|\mathbf{w}_t + \eta y \mathbf{x}\|_2}$):
  $$\Delta \mathbf{w} = \eta y (\mathbf{x} - y \mathbf{w})$$
* **Convergence**: The negative feedback term $-y^2 \mathbf{w}$ stabilizes $\|\mathbf{w}\|_2 \to 1$ and guarantees asymptotic convergence to the maximal eigenvector of the covariance matrix $C$. Hence, single-neuron Oja's rule computes the **First Principal Component (PCA)** in an entirely online, unsupervised manner.

### 2.4 Competitive Learning (Winner-Take-All)
* **Mechanics**: Neurons compete for the right to represent an input vector. Only the neuron whose weight vector is closest in Euclidean distance to the input adjusts its weights:
  $$i^* = \arg\min_i \|\mathbf{x} - \mathbf{w}_i\|_2, \quad \Delta \mathbf{w}_{i^*} = \eta (\mathbf{x} - \mathbf{w}_{i^*})$$
* **Role in Manufacturing Defect Analysis**: Automatically groups unlabelled surface defect patches into distinct morphological clusters (e.g. scratches vs pitted surface) without supervision.

### 2.5 Energy-Based Models: Restricted Boltzmann Machines (RBM)
* **Mechanics**: Defined over a bipartite graph of visible units $\mathbf{v} \in \{0, 1\}^D$ and hidden units $\mathbf{h} \in \{0, 1\}^K$ with energy function:
  $$E(\mathbf{v}, \mathbf{h}) = -\mathbf{v}^T W \mathbf{h} - \mathbf{a}^T \mathbf{v} - \mathbf{b}^T \mathbf{h}$$
* **Training via Contrastive Divergence (CD-1)**: Approximates the intractable gradient of the log-likelihood by balancing a "clamped" data-driven positive phase with a 1-step Gibbs-sampled "hallucinated" negative phase:
  $$\Delta W_{ij} = \eta \left( \langle v_i h_j \rangle_{\text{data}} - \langle v_i h_j \rangle_{\text{reconstruction}} \right)$$
* **Difference from Error-Correction**: RBMs do not minimize classification mismatch; rather, they carve the energy landscape such that valid steel surface defect textures correspond to low-energy valleys, while impossible or corrupted patterns correspond to high-energy peaks.
