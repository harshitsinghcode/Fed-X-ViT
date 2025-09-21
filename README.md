# Fed-X-ViT: Federated Explainable Vision Transformers for Brain Tumor Diagnosis

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

> 🧠 A PyTorch framework for high-accuracy, private, and explainable brain tumor classification using a state-of-the-art hybrid CNN-ViT model.

---

## Table of Contents
-  [1. Overview](#-overview)
-  [2. Key Features](#-key-features)
-  [3. Architecture](#-architecture)
-  [4. Results & Performance](#-results--performance)
-  [5. Getting Started](#-getting-started)
-  [6. Next Steps](#️-next-steps-future-work)
-  [7. Contributors](#-contributors)

---


## 📖 Overview

This project tackles three critical challenges in clinical AI: **Accuracy**, **Privacy**, and **Trust**. Fed-X-ViT is a framework designed to diagnose brain tumors from MRI scans with state-of-the-art precision, while being architected for privacy-preserving training through **Federated Learning (FL)** and clinical trust through **Explainable AI (XAI)**.

This repository contains the code for the centralized training phase, which establishes a powerful baseline model that achieves over **99% test accuracy**.

---


## ✨ Key Features

- **State-of-the-Art Hybrid Architecture:** A novel hierarchical model combining a CNN (EfficientNetV2) for detailed feature extraction with a Vision Transformer (ViT) for global context analysis.
- **Exceptional Performance:** Achieves outstanding accuracy on complex multiclass brain tumor classification, validated on a held-out test set.
- **Framework Design:** Modular code structured to seamlessly integrate upcoming Federated Learning and Explainability modules.
- **Modern Training Practices:** Utilizes best-in-class techniques like the AdamW optimizer, CosineAnnealingLR scheduler, and TrivialAugmentWide for robust and efficient training.

---


## 🔬 Architecture

Our model mimics the workflow of an expert radiologist by using a two-stage analysis process:

1. **CNN Feature Extractor (EfficientNetV2):** Acts as a "detective," scanning the MRI for local, detailed patterns and textures indicative of anomalies.
2. **Vision Transformer (ViT):** Acts as a "strategist," receiving the rich feature map from the CNN. It uses self-attention to analyze the global context and relationships between all detected features to make a final, highly-informed classification.

---


## 🏆 Results & Performance

The centralized model was trained and evaluated on two public datasets, demonstrating its exceptional capabilities.

| Dataset Task | Dataset Used | Peak Metric | **Result** |
| :--- | :--- | :--- | :--- |
| **2-Class Classification** | Brain Tumor / Healthy | Validation Accuracy | **99.56%** |
| **4-Class Classification** | Glioma, Meningioma, Pituitary, No Tumor | Final Test Set Accuracy | **99.31%** |

Achieving over **99.3% accuracy** on the complex 4-class problem on a completely unseen test set validates the power of the hybrid architecture and establishes a state-of-the-art baseline.

---


## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA support

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/harshitsinghcode/FedXViT.git
    cd FedXViT
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```
3.  **Install PyTorch with CUDA support:**
    (Check your CUDA version with `nvidia-smi`)
    ```bash
    # Example for CUDA 12.1
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    ```
4.  **Install the remaining dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    

### Datasets and Pre-trained Models

The datasets used for this project and the final trained model weights are available for download.

-   [**Download Models & Datasets from Google Drive**](https://drive.google.com/drive/folders/16mnVK2X7_hBYtpJ6evgHGYcQ0ORyDtDm)

Place the `Brain Tumor MRI Dataset` folder inside the `dataset/` directory.

### Usage

The `train.py` script is the main entry point for training the centralized model.

```bash
# Train on the 4-class dataset for 30 epochs with a batch size of 16
python train.py --num_classes 4 --batch_size 16 --epochs 30
```
---


## 🛣️ Next Steps (Future Work)

This successful centralized model is the foundation for the next phases of the project:

1.  **Federated Learning (FL) Implementation:** Deploying the training process across multiple simulated clients to train the model without sharing private data.
2.  **Explainable AI (XAI) Integration:** Implementing Grad-CAM and Attention Rollout to generate heatmaps that visualize the model's decision-making process, making it transparent and trustworthy for clinicians.

---


## 👥 Contributors

1.   **Sinchan Shetty** (22BCE5238)
2.   **Riddhi BANDYOPADHYAY** (22BCE1068)
3.   **Harshit Kumar Singh** (22BLC1079)

### Guided By
   **Dr. Suganya R** (SCOPE - 52858)
