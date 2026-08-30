# 🧠 Medical Image Segmentation: U-Net vs Vision Transformers

> **Author**: Mahnoor ([@mahnoor-2722](https://github.com/mahnoor-2722))  
> **Domain**: Computer Vision · Healthcare AI · Medical Image Analysis  
> **Dataset**: ISIC 2018 Task 1 — Skin Lesion Segmentation (2,594 images)

---

## 📌 Project Overview

End-to-end benchmark comparing a **CNN (U-Net)** built from scratch in TensorFlow against a **Vision Transformer (SegFormer)** in PyTorch for pixel-level dermatoscopic lesion segmentation.

### 🎯 Research Questions
1. Can hierarchical Vision Transformers outperform CNN skip-connection architectures on skin lesion boundaries?
2. Can a ~3.7M-parameter Transformer match or beat a ~31M-parameter U-Net?

---

## 📊 Final Benchmark Results

| Model | Framework | Test/Val Dice | IoU (Jaccard) | Parameters | Status |
|:------|:---------:|:-------------:|:-------------:|:----------:|:------:|
| **U-Net (from scratch)** | TensorFlow | **85.26%** | **74.53%** | 31.05M | ✅ |
| **SegFormer MiT-B0** | PyTorch | **90.32%** | **82.50%** | ~3.7M | ✅ |
| **Improvement** | — | **+5.06%** | **+7.97%** | **8× smaller** | 🏆 |

**Winner: SegFormer** — higher overlap accuracy with ~88% fewer parameters.

---

## 🏗️ Architecture 1: Custom U-Net (TensorFlow)

- Encoder–decoder with skip connections  
- 4-level contracting / expanding path  
- Loss: BCE + Dice  
- Input: 256×256×3  

### Training Curves
![U-Net Training Curves](results/training_curves.png)

### U-Net Segmentation Samples
![U-Net Results](results/segmentation_results.png)

---

## 🏗️ Architecture 2: SegFormer (PyTorch)

- Backbone: NVIDIA MiT-B0 (Mix Transformer)  
- Hierarchical self-attention encoder + lightweight MLP decoder  
- Pretrained on ImageNet, fine-tuned on ISIC 2018  
- Loss: BCEWithLogits + Dice  
- Framework: PyTorch + Hugging Face `transformers`  

### SegFormer Segmentation Samples
![SegFormer Results](results/segformer_results.png)

---

## 🛠️ Tech Stack

| Component | Tools |
|-----------|--------|
| Frameworks | TensorFlow 2.x, PyTorch |
| Models | Custom U-Net, SegFormer (nvidia/mit-b0) |
| Data | ISIC 2018, Albumentations |
| Metrics | Dice Coefficient, IoU |
| Hardware | NVIDIA T4 (Google Colab) |
| Model Hosting | [Hugging Face — isic-segformer](https://huggingface.co/mahnoor-2722/isic-segformer) |

---

## 📂 Repository Structure

```text
├── notebooks/
│   ├── 01_baseline_unet_isic2018.ipynb
│   └── 02_segformer_pytorch_isic2018.ipynb
├── results/
│   ├── eda_samples.png
│   ├── training_curves.png
│   ├── segmentation_results.png
│   └── segformer_results.png
├── README.md
└── .gitignore