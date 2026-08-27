# 🧠 Medical Image Segmentation: U-Net vs Vision Transformers

> **Author**: Mahnoor ([@mahnoor-2722](https://github.com/mahnoor-2722))  
> **Domain**: Computer Vision / Healthcare AI  
> **Dataset**: ISIC 2018 Skin Lesion Segmentation (2,594 dermatoscopic images)

---

## 📌 Project Overview
This project presents an end-to-end benchmark comparing traditional **Convolutional Neural Networks (CNNs)** with **Vision Transformers (ViTs)** for pixel-level medical image segmentation. 

### 🎯 Key Questions:
1. *Can Transformer-based architectures (SegFormer/UNETR) capture long-range contextual boundaries better than CNN skip-connections in skin lesions?*
2. *Does self-attention map visualization provide more clinically meaningful explanations compared to Grad-CAM?*

---

## 📊 Benchmark Results

| Model Architecture | Framework | Val Dice Score | Val IoU (Jaccard) | Parameters | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **U-Net (From Scratch)** | TensorFlow | **85.26%** | **74.53%** | 31.05M | ✅ Completed |
| **SegFormer (ViT)** | PyTorch | *In Progress* | *In Progress* | ~3.7M | ⏳ Next |

---

## 🏗️ Architecture 1: Custom U-Net (TensorFlow)
Built completely from scratch using an Encoder-Decoder structure with skip connections.

- **Encoder**: 4 Contracting blocks (Conv2D $\rightarrow$ BN $\rightarrow$ ReLU $\rightarrow$ MaxPool)
- **Bottleneck**: 1024-filter feature space at $16 \times 16$ resolution
- **Decoder**: 4 Expanding blocks (ConvTranspose $\rightarrow$ Skip Concat $\rightarrow$ Conv2D)
- **Loss Function**: Combined Binary Cross-Entropy + Dice Loss ($L_{BCE} + L_{Dice}$)

### 📈 Training Curves & Convergence
![Training Curves](results/training_curves.png)

### 🖼️ Test Set Segmentation Results
![Segmentation Results](results/segmentation_results.png)

---

## 🛠️ Tech Stack
- **Frameworks**: TensorFlow 2.x, PyTorch
- **Libraries**: OpenCV, Albumentations, Matplotlib, Scikit-learn
- **Hardware**: NVIDIA T4 GPU (Google Colab)

---

## 🚀 Next Steps
- [x] Phase 1: Build & train U-Net baseline from scratch in TensorFlow
- [ ] Phase 2: Implement SegFormer / UNETR Transformer model in PyTorch
- [ ] Phase 3: Evaluate attention maps vs CNN spatial feature maps
- [ ] Phase 4: Deploy interactive Streamlit web application