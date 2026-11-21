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

## 5. Language Approval
The course compendium lists C/C++/Java for project implementations; however, implementation in Python (TensorFlow, NumPy) was officially approved by <<PROFESSOR NAME>> on <<DATE / how it was communicated>>.

---

## 6. Declared AI Assistance
<!-- Academic Integrity Disclosure: Please fill in the placeholders below truthfully before viva submission. -->
* **AI Tool(s) Used**: <<TODO: e.g. Claude / ChatGPT / Gemini / Copilot / Cursor / Antigravity / None>>
* **Scope of Assistance**: <<TODO: Specific description of what AI tools were used for, e.g. debugging syntax errors, formatting documentation tables, drafting boilerplate scripts>>
* **Files / Modules Affected**: <<TODO: Specific files touched or generated with AI assistance, e.g. prepare_data.py, docs/USER_GUIDE.md, etc.>>
* **Independently Authored & Verified Work**: <<TODO: State what code, mathematical proofs, experimental setups, and analysis you designed, reviewed, and modified yourself>>
* **Viva Ownership Affirmation**: <<TODO: Confirm personal comprehension and readiness to explain, trace, and live-modify any component during academic evaluation>>
