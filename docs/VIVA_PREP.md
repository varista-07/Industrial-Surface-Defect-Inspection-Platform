# Comprehensive Viva Defense Guide & Examiner Q&A Handbook
**Course**: CSAI3027P Deep Learning  
**Project**: Industrial Surface Defect Inspection Platform (NEU-DET Benchmark)

---

## 1. Module-by-Module Technical Explanations

### Module 1: Fundamentals from Scratch (NumPy)
* **What it does**: Implements tensors, broadcasting, SVD low-rank reconstruction, M-P and sigmoid neurons, an object-oriented MLP with full backpropagation, 4 custom optimizers (BGD, SGD, Momentum, NAG), 13 controlled ablation experiments, double-precision centered gradient checking, Hebbian/Oja/competitive learning, and a compact RBM.
* **Why chosen**: Demonstrates first-principles mathematical mastery of vector calculus, loss landscapes, and numerical stability without framework abstractions.

### Module 2: ResNet50 Classifier & Transfer Learning
* **What it does**: Applies two-stage transfer learning: Stage 1 trains a classification head on frozen ImageNet features; Stage 2 unfreezes top `conv5_block*` layers with reduced learning rate ($10^{-4}$) while keeping all `BatchNormalization` layers frozen.
* **Why chosen**: Residual skip connections eliminate vanishing gradients. ImageNet pre-training prevents overfitting on small datasets (1,800 images).

### Module 3: Defect Object Detection (Pascal VOC Bounding Boxes)
* **What it does**: Benchmarked Keras-CV RetinaNet on CPU, then implemented a multi-task shared-backbone architecture predicting 6 class probabilities and 4 normalized bounding box coordinates $[y_{\min}, x_{\min}, y_{\max}, x_{\max}]$.
* **Why chosen**: Real-time line scanning demands fast localized coordinates for sorting. When RetinaNet required 491 min/epoch on CPU, the multi-task head provided a practical real-time fallback (21.2 ms latency, 47.1 FPS).

### Module 4: Explainability (Grad-CAM & Conv2D Filters)
* **What it does**: Visualizes 64 learned 7x7 filters of `conv1_conv`, and computes class-discriminative Grad-CAM saliency heatmaps from `conv5_block3_out` feature maps.
* **Why chosen**: Black-box models cannot be deployed in industrial quality assurance. Grad-CAM confirms that decisions are grounded in actual defect cracks rather than background roll grain.

### Module 5: Weak-Supervised U-Net Segmentation
* **What it does**: Synthesizes weak pseudo-masks using bounding box fills and localized Otsu thresholding, training a 128x128 encoder-decoder U-Net with skip connections.
* **Why chosen**: NEU-DET lacks human pixel-level masks. Pseudo-mask bootstrapping demonstrates how to establish morphological segmentation without manual pixel labeling.

### Module 6: Synthetic Report Transformer Encoder
* **What it does**: Generates template-grounded natural language reports from box geometry and classifies report severity using a custom 4-head Transformer encoder with attention visualization.
* **Why chosen**: Demonstrates multimodal architecture integration (Vision to Text representation).

### Module 7: Convolutional Variational Autoencoder (Conv-VAE)
* **What it does**: Models defect manifold with a 32-dimensional Gaussian latent space using the reparameterization trick, visualizing PCA clustering and evaluating synthetic augmentation.
* **Why chosen**: Proves generative modeling competence and tests whether synthetic samples assist low-data regimes.

### Module 8: Production Web Application (FastAPI & UI)
* **What it does**: Asynchronous FastAPI service loading models at startup, serving a responsive single-page dashboard with drag-and-drop upload, threshold slider, visual matrix, and security validation.
* **Why chosen**: Delivers an interactive interface meeting all project demonstration requirements.

---

## 2. In-Depth Viva Defense Questions & Answers (5 Per Module)

### Module 1: Fundamentals from Scratch (NumPy)

#### Q1.1: Derive how backpropagation works mathematically and why activation caching is required.
* **Answer**: Backpropagation evaluates the multivariate chain rule across layers. In the forward pass, each layer computes pre-activations $z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}$ and activations $a^{[l]} = \sigma(z^{[l]})$. In the backward pass, given scalar loss $L$, the output error vector is $\delta^{[L]} = \nabla_{z^{[L]}} L$. For layer $l$:
  $$\delta^{[l]} = \left( (W^{[l+1]})^T \delta^{[l+1]} \right) \odot \sigma'(z^{[l]})$$
  $$\frac{\partial L}{\partial W^{[l]}} = \delta^{[l]} (a^{[l-1]})^T, \quad \frac{\partial L}{\partial b^{[l]}} = \delta^{[l]}$$
  Caching $a^{[l-1]}$ and $z^{[l]}$ during forward execution eliminates redundant computations, reducing gradient calculation complexity from exponential to linear $\mathcal{O}(|V| + |E|)$.

#### Q1.2: What is the core mathematical difference between Classical Momentum and Nesterov Accelerated Gradient (NAG)?
* **Answer**: 
  - Classical Momentum calculates the gradient at current position $\theta_t$:
    $$v_{t+1} = \beta v_t + \eta \nabla L(\theta_t), \quad \theta_{t+1} = \theta_t - v_{t+1}$$
  - Nesterov Accelerated Gradient computes a lookahead step before gradient evaluation:
    $$v_{t+1} = \beta v_t + \eta \nabla L(\theta_t - \beta v_t), \quad \theta_{t+1} = \theta_t - v_{t+1}$$
  - By evaluating the gradient at the anticipated future location, NAG senses changes in curvature earlier, providing an adaptive braking force that damps oscillations when entering narrow ravines.

#### Q1.3: Why is centered finite difference checking preferred over one-sided differences in gradient verification?
* **Answer**: Taylor series expansion of $f(\theta + \epsilon)$ and $f(\theta - \epsilon)$ reveals:
  - One-sided difference: $\frac{f(\theta + \epsilon) - f(\theta)}{\epsilon} = f'(\theta) + \mathcal{O}(\epsilon)$ (truncation error is linear).
  - Centered difference: $\frac{f(\theta + \epsilon) - f(\theta - \epsilon)}{2\epsilon} = f'(\theta) + \mathcal{O}(\epsilon^2)$ (truncation error is quadratic).
  Using centered differences with $\epsilon = 10^{-6}$ in double precision (`float64`) achieves relative errors $< 10^{-9}$, confirming analytical gradient correctness.

#### Q1.4: How does Oja's rule prevent the runaway weight explosion seen in basic Hebbian learning?
* **Answer**: Basic Hebbian learning updates weights via $\Delta \mathbf{w} = \eta y \mathbf{x}$. Because $\Delta \mathbf{w}$ is strictly non-negative for positive correlations, the weight norm grows exponentially ($\|\mathbf{w}\| \to \infty$). Oja's rule introduces a mathematical weight decay term:
  $$\Delta \mathbf{w} = \eta y (\mathbf{x} - y \mathbf{w})$$
  This normalizes the weight vector to unit length $\|\mathbf{w}\| \to 1$ while asymptotically aligning $\mathbf{w}$ with the first principal component (maximal eigenvector) of the data covariance matrix.

#### Q1.5: Why can a single McCulloch-Pitts or Perceptron neuron not solve XOR, and how do multi-layer networks overcome this?
* **Answer**: XOR is linearly inseparable: no single hyperplane $\mathbf{w}^T \mathbf{x} + b = 0$ can partition $(0, 0), (1, 1)$ [label 0] from $(0, 1), (1, 0)$ [label 1]. A multi-layer network introduces intermediate hidden units with non-linear activations (e.g. ReLU or Sigmoid) that map input coordinates into a higher-dimensional warped feature space where the points become linearly separable.

---

### Module 2: ResNet50 Classifier & Transfer Learning

#### Q2.1: Why do residual skip connections ($F(x) + x$) prevent the vanishing gradient problem?
* **Answer**: In a standard deep network, the gradient is a product of Jacobians: $\frac{\partial a^{[L]}}{\partial a^{[l]}} = \prod_{k=l}^{L-1} W^{[k+1]} \sigma'$. As depth increases, repeated multiplication by small weights causes exponential decay to zero. In ResNet, activations follow $x_{l+1} = x_l + F(x_l, W_l)$. By the chain rule:
  $$\frac{\partial \mathcal{E}}{\partial x_l} = \frac{\partial \mathcal{E}}{\partial x_{l+1}} \left( \mathbf{I} + \frac{\partial F}{\partial x_l} \right)$$
  The additive identity term $\mathbf{I}$ guarantees an uninterrupted highway for gradient flow directly back to early layers, even if $\frac{\partial F}{\partial x_l}$ approaches zero.

#### Q2.2: Why must all BatchNormalization layers remain frozen during fine-tuning?
* **Answer**: BatchNormalization normalizes activations using running mean $\mu$ and variance $\sigma^2$:
  $$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta$$
  When fine-tuning on small industrial batches (batch size 16 or 32 on CPU), sample statistics calculated over tiny mini-batches have high variance. Updating running statistics corrupts the pre-trained ImageNet feature distributions. Keeping BN layers in inference mode (`layer.trainable = False`) maintains fixed affine scaling, stabilizing gradient propagation through residual blocks.

#### Q2.3: Why adopt a two-stage training strategy instead of unfreezing the entire network immediately?
* **Answer**: At initialization, the top classification dense layer contains random Gaussian weights, producing massive loss gradients. If the entire network is unfrozen on epoch 1, these chaotic early gradients backpropagate through the backbone, causing catastrophic forgetting of pre-trained feature representations. Stage 1 trains only the head to convergence; Stage 2 gently adapts top convolutional layers (`conv5`) using a 10x smaller learning rate ($10^{-4}$).

#### Q2.4: How were data augmentations chosen to preserve industrial defect integrity?
* **Answer**: Steel defects possess strict physical morphology. Geometric operations such as random horizontal/vertical flips and minor rotations ($\pm 10^\circ$) preserve directional crack properties. Contrast and brightness adjustments simulate varying mill lighting. Destructive operations like elastic deformation, heavy cut-out, or extreme shearing were excluded because they destroy the continuity of fine crazing micro-cracks and roll marks.

#### Q2.5: What are the primary confusion pairs observed in defect inspection, and why do they occur?
* **Answer**: The primary confusion occurs between *crazing* and *scratches*, and between *patches* and *pitted surface*. Isolated longitudinal crazing cracks lack lateral branching, resembling mechanical scratches under single-angle illumination. Dense pit clusters merge into continuous dark blobs that mimic irregular peel patches.

---

### Module 3: Defect Object Detection & Localization

#### Q3.1: Define Intersection over Union (IoU) and explain how mAP@0.5 is calculated.
* **Answer**: IoU measures bounding box overlap: $\text{IoU} = \frac{\text{Area}(B_{\text{pred}} \cap B_{\text{gt}})}{\text{Area}(B_{\text{pred}} \cup B_{\text{gt}})}$. A prediction is classified as a True Positive if predicted class matches ground truth and $\text{IoU} \ge 0.50$. For each class, predictions are ranked by confidence to plot Precision vs Recall. Average Precision (AP) is the area under this interpolated curve. **mAP@0.5** is the unweighted arithmetic mean of AP values across all 6 defect classes.

#### Q3.2: Why did you fall back from Keras-CV RetinaNet to the multi-task shared head?
* **Answer**: We benchmarked RetinaNet with a ResNet50 backbone on CPU (`models/retinanet_benchmark.py`). Measuring mini-batch size 2 revealed a step latency of **46.79 seconds**, projecting to **491.2 minutes (~8.2 hours) per epoch**. A 10-epoch training schedule would exceed **81.8 hours on CPU**, violating our 30-minute training limit. Additionally, Keras-CV 0.9.0 has internal input unpacking incompatibilities with Keras 3. The multi-task shared head provided a practical real-time fallback (21.2 ms latency, 47.1 FPS).

#### Q3.3: What is the main structural weakness of the single-box multi-task detection fallback?
* **Answer**: In NEU-DET, **73.17% of images contain multiple defect instances** (mean: 2.33 boxes/image, up to 9 boxes). A multi-task regression head outputs a single vector $[y_{\min}, x_{\min}, y_{\max}, x_{\max}]$. When presented with multiple defects, the head regresses toward the coordinate centroid or the single most salient defect, bounding only one instance and yielding a lower overall mAP@0.5 (24.07%).

#### Q3.4: Why is Smooth L1 Loss preferred over Mean Squared Error (MSE / L2) for bounding box coordinates?
* **Answer**: MSE penalizes coordinate errors quadratically ($e^2$). Large outlier errors from anomalous annotations generate massive gradients that destabilize training. Smooth L1 behaves quadratically for small errors ($|e| < 1$) and linearly for large errors ($|e| \ge 1$):
  $$\text{Smooth}_{L1}(e) = \begin{cases} 0.5 e^2, & \text{if } |e| < 1 \\ |e| - 0.5, & \text{otherwise} \end{cases}$$
  This bounds gradient magnitude to $\pm 1$, providing robust convergence against label noise.

#### Q3.5: How did per-class AP@0.5 differ across defect classes in your detector?
* **Answer**: Localized, compact defects achieved high AP: `pitted_surface` achieved **86.67% AP@0.5** because pit depressions are concentrated in localized clusters. In contrast, elongated, multi-instance defects achieved lower AP: `crazing` (22.22%), `inclusion` (17.78%), `patches` (6.67%), `rolled-in_scale` (6.67%), and `scratches` (4.44%), where single-box heads cannot resolve multiple dispersed tracks.

---

### Module 4: Explainability (Grad-CAM & Conv2D Filters)

#### Q4.1: Derive the mathematical formulation of Grad-CAM.
* **Answer**: Grad-CAM calculates the gradient of the unnormalized class score $y^c$ with respect to feature activation map $A^k$ of the final convolutional layer:
  $$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^U \sum_{j=1}^V \frac{\partial y^c}{\partial A_{ij}^k}$$
  The importance weight $\alpha_k^c$ captures the contribution of channel $k$ to class $c$. The coarse heatmap is generated by a linear combination followed by ReLU:
  $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$

#### Q4.2: Why is the ReLU function applied to the weighted combination in Grad-CAM?
* **Answer**: Pixels that decrease the score of class $c$ (features belonging to rival defect classes or background steel roll grain) carry negative weights. ReLU thresholds all negative contributions to zero, retaining strictly those spatial regions whose presence increases the evidence for class $c$.

#### Q4.3: How does Grad-CAM differ from Class Activation Mapping (CAM) and vanilla saliency maps?
* **Answer**: Original CAM strictly requires a Global Average Pooling layer immediately preceding the softmax dense layer, demanding architecture modification and retraining. Grad-CAM generalizes to any CNN without retraining. Vanilla saliency computes $\frac{\partial y^c}{\partial I}$ w.r.t. input pixels, producing high-frequency noisy gradients. Grad-CAM operates on deep convolutional feature maps, capturing semantic object-level regions.

#### Q4.4: What features do early Conv2D filters learn compared to deep layers?
* **Answer**: As visualized in `outputs/figures/conv2d_filters_and_feature_maps.png`, early filters in `conv1_conv` (7x7) learn low-level spatial primitives: Gabor-like directional edge detectors, vertical/horizontal ridges, and high-frequency textural gradients. Deeper layers combine these primitives into class-specific morphological representations (such as crack networks and pit craters).

#### Q4.5: What is the architectural difference between 2D convolution and 3D convolution?
* **Answer**: 2D convolution applies spatial kernels $K \in \mathbb{R}^{k_h \times k_w \times C_{\text{in}}}$ that slide across spatial dimensions $(H, W)$, collapsing channel depth to produce 2D feature maps. 3D convolution uses spatio-temporal kernels $K \in \mathbb{R}^{k_t \times k_h \times k_w \times C_{\text{in}}}$ that slide across time/depth as well as space, preserving volumetric temporal dynamics suitable for high-speed strip video streams.

---

### Module 5: Weak-Supervised U-Net Segmentation

#### Q5.1: Describe the U-Net architecture and the role of its skip connections.
* **Answer**: U-Net consists of a contracting encoder path that extracts multi-scale context via repeated convolution and pooling, and a symmetric expanding decoder path that restores spatial resolution via transposed convolutions. Skip connections concatenate high-resolution feature maps from the encoder directly to decoder stages, preserving fine boundary details lost during downsampling.

#### Q5.2: How were weak pseudo-masks synthesized for NEU-DET?
* **Answer**: NEU-DET lacks human pixel-level masks. Pseudo-masks were generated by mapping Pascal VOC bounding boxes to binary masks and applying local Otsu adaptive thresholding within each box region. This segments dark defect pixels from the brighter metallic background, producing weak training labels.

#### Q5.3: Why is Dice score preferred over Pixel Accuracy in segmentation evaluation?
* **Answer**: Industrial defects exhibit extreme class imbalance: defects occupy $< 5\%$ of the total steel surface area. A degenerate model predicting all background achieves $> 95\%$ pixel accuracy while being useless. The Dice coefficient ($\frac{2 |A \cap B|}{|A| + |B|}$) directly evaluates foreground intersection relative to union, penalizing false positives and false negatives equally.

#### Q5.4: How does connected-components analysis isolate discrete defect instances?
* **Answer**: Connected-components labeling (`scipy.ndimage.label`) scans the predicted binary mask using an 8-neighborhood connectivity kernel. Contiguous foreground pixels are grouped into unique integer labels, enabling extraction of instance metrics such as centroid coordinates, major axis length, and surface area fraction.

#### Q5.5: What are the fundamental limitations of training on weak pseudo-labels?
* **Answer**: Weak pseudo-labels suffer from label noise. Bounding box fills include non-defective background pixels within the rectangle. Local thresholding fails on low-contrast defects (such as faint oil-quenched rolled-in scale). Consequently, models trained on pseudo-labels learn an upper bound dictated by the thresholding heuristic (Mean IoU 0.1292, Dice 0.1950 on test split).

---

### Module 6: Synthetic Report Transformer Encoder

#### Q6.1: State the Scaled Dot-Product Attention equation and explain the role of $\frac{1}{\sqrt{d_k}}$.
* **Answer**:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V$$
  For large projection dimensions $d_k$, the dot product $Q K^T$ grows large in magnitude. Large values push the softmax function into regions with near-zero gradients (saturation). Scaling by $\frac{1}{\sqrt{d_k}}$ preserves unit variance, maintaining healthy gradient flow during backpropagation.

#### Q6.2: Why do Transformer encoders require Positional Encodings?
* **Answer**: Self-attention is permutation-equivariant: shuffling the input tokens produces identically shuffled output representations without changing the values. Because natural language inspection syntax depends on word order (e.g. "scratch near top" vs "top scratch"), positional encodings are added to token embeddings to inject sequence order information.

#### Q6.3: How were the synthetic inspection reports generated?
* **Answer**: Inspection reports are template-generated from Pascal VOC metadata. Geometry parameters (class name, bounding box coordinates, surface area fraction, aspect ratio) are mapped into domain inspection sentences (e.g. *"Severe irregular clustered crazing defect identified near central region spanning approximately 73 percent surface area"*).

#### Q6.4: What do the self-attention heatmaps demonstrate in your inspection reports?
* **Answer**: Visualizing attention weights (`outputs/figures/transformer_attention_weights.png`) reveals that attention heads specialize: one head attends to defect severity descriptors ("Severe", "Minor"), another attends to spatial location tokens ("central", "top"), and others attend to morphological classification tokens ("crazing", "scratch").

#### Q6.5: Why use an Encoder-only Transformer rather than a Decoder-only architecture for this task?
* **Answer**: Text classification requires bidirectional context: understanding token relationships across the entire sentence simultaneously. Encoder-only architectures (BERT-style) allow every token to attend to all tokens before and after it. Decoder-only architectures (GPT-style) use causal masking, which restricts attention to past tokens and is inefficient for classification.

---

### Module 7: Convolutional Variational Autoencoder (Conv-VAE)

#### Q7.1: Explain the Reparameterization Trick and why it is necessary.
* **Answer**: In a VAE, latent variables are sampled from a distribution: $\mathbf{z} \sim q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$. Direct random sampling is non-differentiable because gradients cannot backpropagate through a stochastic node. The reparameterization trick expresses $\mathbf{z}$ as a deterministic function with external noise:
  $$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$
  This shifts the stochasticity into an input parameter $\boldsymbol{\epsilon}$, enabling backpropagation of $\frac{\partial \mathcal{L}}{\partial \boldsymbol{\mu}}$ and $\frac{\partial \mathcal{L}}{\partial \boldsymbol{\sigma}}$.

#### Q7.2: What are the two components of the VAE loss function, and what does each optimize?
* **Answer**:
  $$\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q(\mathbf{z}|\mathbf{x})}[\log p(\mathbf{x}|\mathbf{z})] - D_{\text{KL}}(q(\mathbf{z}|\mathbf{x}) \parallel p(\mathbf{z}))$$
  1. **Reconstruction Loss**: Measures pixel fidelity (MSE or Binary Cross-Entropy) between input $\mathbf{x}$ and reconstruction $\hat{\mathbf{x}}$, encouraging sharp feature recovery.
  2. **KL Divergence Loss**: Measures the divergence between posterior $q(\mathbf{z}|\mathbf{x})$ and standard normal prior $\mathcal{N}(\mathbf{0}, \mathbf{I})$. It acts as a regularizer, preventing clusters from collapsing into isolated points and ensuring smooth latent space interpolation.

#### Q7.3: How does a VAE differ fundamentally from a standard Autoencoder (AE)?
* **Answer**: A standard AE maps inputs to discrete unconstrained points in latent space. This leaves empty gaps in the manifold where sampling produces distorted nonsense. A VAE maps inputs to probability distributions and regularizes the space via KL divergence, creating a continuous, smooth manifold where any sampled vector $z \sim \mathcal{N}(0, \mathbf{I})$ decodes into a valid defect.

#### Q7.4: What did 2D PCA analysis of the latent space reveal about defect clustering?
* **Answer**: As shown in `outputs/figures/vae_reconstructions_and_latent_space.png`, the first two principal components explain **79.16% of total latent variance**. Defect classes form distinct spatial clusters: high-frequency crazing separates clearly from broad patches and linear scratches, demonstrating that the VAE captures fundamental morphological variance.

#### Q7.5: What were the empirical results of training with synthetic VAE data augmentation?
* **Answer**: We evaluated a baseline CNN on 64x64 crops without synthetic data (57.04% accuracy) versus training with 120 synthetic VAE samples added (71.11% accuracy), demonstrating a $+14.07\%$ improvement on small sample baselines. However, VAE samples exhibit mild high-frequency blurring due to Gaussian likelihood assumptions, indicating that diffusion models or GANs are preferable for production-grade image synthesis.

---

### Module 8: Production Web Application & Safety

#### Q8.1: How does the FastAPI asynchronous backend handle model loading and inference throughput?
* **Answer**: All models (`best_resnet50.keras`, `best_detector.keras`, `best_unet.keras`) are preloaded into memory once at application startup. Subsequent `POST /predict` calls execute against the pre-warmed weights without disk I/O latency. Endpoints utilize asynchronous workers, serving live visual overlays and JSON payloads with sub-second response times.

#### Q8.2: How does the dynamic confidence threshold slider function in production?
* **Answer**: Line operators adjust the decision threshold $\tau \in [0.20, 0.95]$ (default: $0.60$). When an image is uploaded:
  - If $\max_c P(y=c|\mathbf{x}) \ge \tau$, the platform flags **QC FAIL: [DEFECT] DETECTED**, routing the steel coil to rework.
  - If $\max_c P(y=c|\mathbf{x}) < \tau$, the system triggers **UNCERTAIN / ANOMALY DETECTED**, holding the coil for manual visual inspection.

#### Q8.3: Why is low-confidence thresholding not a true Out-of-Distribution (OOD) detector?
* **Answer**: Softmax confidence thresholding simply monitors the peak output probability $\max P_k$. Deep neural networks often make overconfident mistakes on anomalous inputs due to linear decision boundaries far from training clusters. A true OOD detector evaluates Mahalanobis distance in latent feature space, energy scores, or reconstruction error under an autoencoder to identify distribution shifts reliably.

#### Q8.4: What input validation and security safeguards are implemented in the API?
* **Answer**: The endpoint enforces strict defensive checks:
  1. **Format Whitelisting**: Intercepts and rejects unsupported formats (e.g. `.pdf`, `.txt`), restricting uploads to JPEG, PNG, and BMP.
  2. **Payload Size Capping**: Rejects files exceeding 5 MB to prevent Denial of Service (DoS).
  3. **Corrupt File Interception**: Wraps Pillow image decoding in try-except blocks, rejecting malformed byte streams with HTTP 400.
  4. **Channel Normalization**: Automatically converts single-channel grayscale or RGBA inputs into standardized 3-channel RGB tensors.

#### Q8.5: How is False-Reject Rate (FRR) defined on NEU-DET, and what are its production consequences?
* **Answer**: Because NEU-DET contains solely defective steel, False-Reject Rate is defined operationally as the **Missed Defect Rate** ($\max P < \theta$) plus the **Manual Triage Escalation Rate**. 
  - On our test split, thresholds between 0.50 and 0.80 yield a **0.00% triage rate** (0 / 270 samples flagged).
  - At an ultra-strict 0.90 threshold, the triage rate is **0.37%** (only 1 out of 270 samples is escalated).
  - In production, high triage rates bottleneck inspection lines with manual review, whereas low triage rates risk shipping unflagged defective steel to customers.

---

## 3. Live Viva Modification Cheat-Sheet

If the examiner asks you to modify the platform live during evaluation:

### A. Adjusting the Confidence Threshold
* **In the Web UI**: Drag the top slider (20% to 95%). The verdict card and distribution update instantly.
* **In Code**: Open [app/main.py](file:///c:/Users/Acer/PBL%20PROJECT%20DL/app/main.py), line 89:
  ```python
  threshold: float = Query(0.60, ge=0.20, le=0.95)
  ```

### B. Changing Image Resolution
* Open [models/dataset_loader.py](file:///c:/Users/Acer/PBL%20PROJECT%20DL/models/dataset_loader.py), line 42:
  ```python
  img_size = (200, 200) # Change to (224, 224) or (160, 160)
  ```

### C. Switching to InceptionV3 Benchmark Model
* In [app/main.py](file:///c:/Users/Acer/PBL%20PROJECT%20DL/app/main.py), line 70:
  ```python
  clf_path = os.path.join("models", "checkpoints", "best_inceptionv3.keras")
  ```

### D. Testing an Arbitrary Image File
1. Ensure the app is running: `python run_app.py`
2. Open `http://localhost:8000`
3. Drag any external steel surface image onto the drop zone and click **ANALYZE SURFACE DEFECT**.
