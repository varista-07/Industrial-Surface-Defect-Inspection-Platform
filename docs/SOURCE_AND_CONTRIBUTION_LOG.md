# Source, Contribution, and Provenance Log

## 1. Primary Dataset
* **Dataset Name**: NEU-DET (Northeastern University Surface Defect Database)
* **Citation**:
  Song, K., & Yan, Y. (2013). *A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defect detection*. Applied Surface Science, 285, 858-864.
* **Origin**: Key Laboratory of Integrated Automation of Process Industry, Ministry of Education, Northeastern University, Shenyang, China.
* **License & Use**: Provided under academic and educational research terms for computer vision benchmarking.

---

## 2. Pretrained Models & Model Weights
* **ResNet50 Backbone**:
  - Paper: He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition*. Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR), 770-778.
  - Pretrained Weights: ImageNet-1k (via `tf.keras.applications.ResNet50(weights='imagenet')`).
* **InceptionV3 Backbone**:
  - Paper: Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). *Rethinking the Inception Architecture for Computer Vision*. CVPR 2016.
  - Pretrained Weights: ImageNet-1k (via `tf.keras.applications.InceptionV3(weights='imagenet')`).

---

## 3. Foundational Literature & Algorithmic Citations
1. **McCulloch-Pitts Neuron**:
   McCulloch, W. S., & Pitts, W. (1943). *A logical calculus of the ideas immanent in nervous activity*. The Bulletin of Mathematical Biophysics, 5(4), 115-133.
2. **Backpropagation & Multi-Layer Perceptrons**:
   Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). *Learning representations by back-propagating errors*. Nature, 323(6088), 533-536.
3. **Nesterov Accelerated Gradient (NAG)**:
   Nesterov, Y. (1983). *A method of solving a convex programming problem with convergence rate O(1/k^2)*. Soviet Mathematics Doklady, 27(2), 372-376.
4. **Oja's Subspace Learning Rule**:
   Oja, E. (1982). *Simplified neuron model as a principal component analyzer*. Journal of Mathematical Biology, 15(3), 267-273.
5. **Grad-CAM**:
   Selvaraju, R. R., Cogswell, M., Das, A., Vedaldi, A., Parikh, D., & Batra, D. (2017). *Grad-CAM: Visual explanations from deep networks via gradient-based localization*. IEEE International Conference on Computer Vision (ICCV), 618-626.
6. **U-Net Architecture**:
   Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional networks for biomedical image segmentation*. MICCAI 2015, 234-241.
7. **Variational Autoencoder (VAE)**:
   Kingma, D. P., & Welling, M. (2013). *Auto-encoding variational Bayes*. arXiv preprint arXiv:1312.6114.
8. **Transformer Encoder & Multi-Head Self-Attention**:
   Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention is all you need*. Advances in Neural Information Processing Systems (NeurIPS), 5998-6008.

---

## 4. Software Libraries & Tooling
* **Core Deep Learning**: TensorFlow 2.21.0, Keras 3.15.1
* **Scientific Computing**: NumPy 2.0.2, SciPy 1.18.1, Scikit-Learn 1.9.1, Pandas 3.1.1
* **Image Processing**: Pillow 13.0.0, Matplotlib 3.11.2
* **Web Services & API**: FastAPI 0.163.1, Starlette 1.10.0, Uvicorn 0.54.0, Pydantic 2.14.3
* **Testing & Quality Assurance**: PyTest 9.2.0, Python-Multipart 0.0.34

---

## 5. Record of AI Assistance (Antigravity AI Agent)
In accordance with academic integrity and ethical engineering guidelines, all AI agent collaboration is transparently recorded:
* **Assistance Scope**:
  - Drafting and executing pipeline scripts (`prepare_data.py`, `fundamentals/`, `models/`, `advanced/`, `app/`).
  - Mathematical derivation and double-precision numerical gradient verification of backpropagation equations.
  - Designing responsive industrial QC CSS/HTML templates for the FastAPI inspection dashboard.
  - Compiling experimental benchmark logs and metrics cards into standardized documentation.
* **Student Verification & Defense Ownership**:
  - The student has reviewed every line of source code, verified execution on local hardware, reproduced all figures and loss curves, and is fully prepared to defend mathematical formulations, hyperparameter trade-offs, and live code modifications in the academic viva.
