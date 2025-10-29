# 🧠 Fed-X-ViT: A Federated Transfer Learning & XAI Pipeline for Brain Tumor Diagnosis using Azure MLOps

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Azure](https://img.shields.io/badge/Azure_ML-SDK_v2-0078D4.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A federated simulation pipeline demonstrating how a pre-trained "expert" AI model can be securely fine-tuned on private, decentralized datasets (e.g., from multiple hospitals) to solve a new clinical task. All results, metrics, and models are tracked and versioned in the cloud using **Azure Machine Learning**.

---

## 📑 Table of Contents

1.  [**Project Overview**](#-1-project-overview)
2.  [**Key Features**](#-2-key-features)
3.  [**MLOps Architecture**](#-3-the-mlops-architecture)
4.  [**The Model: Fed-X-ViT**](#-4-the-model-fed-x-vit)
5.  [**Performance & Results**](#-5-performance--results)
6.  [**Explainable AI (XAI) Validation**](#-6-explainable-ai-xai-validation)
7.  [**How to Run the Simulation**](#-7-how-to-run-the-simulation)
8.  [**Contributors**](#-8-contributors)

---

## 📖 1. Project Overview

This project addresses three critical challenges in clinical AI: **Accuracy**, **Privacy**, and **Trust**.

* **Accuracy**: We use a state-of-the-art hybrid CNN-Vision Transformer model (**Fed-X-ViT**) to achieve greater than 99% accuracy in brain tumor classification.
* **Privacy**: We simulate a **Federated Learning (FL)** environment using the [Flower](https://flower.dev/) framework, where three "clients" (simulating 3 different hospitals) collaboratively train a global model on their own private data. No private data ever leaves the local client.
* **Trust**: We integrate **Explainable AI (XAI)** using Grad-CAM to visualize precisely *why* the model makes its predictions. To ensure our results are transparent and reproducible, we use **Azure Machine Learning** to log all metrics, parameters, and final model artifacts.

This repository demonstrates the full, end-to-end pipeline:
1.  **Phase 1**: Training a powerful 4-class "expert" model on a public dataset.
2.  **Phase 2**: Using **Federated Transfer Learning** to adapt this expert for a new 2-class (Tumor/Healthy) task, using data partitioned across 3 simulated clients.
3.  **Phase 3**: Running a final XAI analysis using Grad-CAM to validate that the final federated model's decisions are clinically relevant.

---

## ✨ 2. Key Features

* **State-of-the-Art Hybrid Model**: A novel hierarchical model combining a CNN ([EfficientNetV2](https://arxiv.org/abs/2104.00298)) for local feature extraction with a [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929) for global context analysis.
* **Federated Transfer Learning**: An advanced technique where a pre-trained "expert" model is surgically adapted for a new task. This preserves its core knowledge, allowing it to be fine-tuned rapidly and efficiently on small, private datasets.
* **Azure MLOps Integration**: The entire simulation is connected to an Azure Machine Learning Workspace. All metrics (e.g., `global_val_accuracy`, `global_val_loss`) are logged in real-time via [MLflow](https://mlflow.org/), and the final model is automatically versioned and stored.
* **Parallel GPU Simulation**: The 3-client simulation runs on a single local GPU (e.g., RTX 3060) using Flower's high-performance [Ray](https://www.ray.io/) backend for parallelization.
* **Explainable & Trustworthy**: Includes a `xai.py` script to run Grad-CAM analysis, generating heatmaps that prove the model's decisions are based on correct clinical features.

---

## 🔬 3. The MLOps Architecture

We simulate a "Hub-and-Spoke" model. The "Hub" is the central server and cloud infrastructure, and the "Spokes" are the private clients (hospitals).


| Component | Role & Technology |
| :--- | :--- |
| **Local Machine** | A single PC (e.g., with an RTX 3060 GPU) runs the entire simulation. |
| ↳ **Server Process** | The `server.py` script acts as the **Coordinator**. It loads the expert model, performs transfer learning, connects to Azure, and orchestrates the federated rounds using the **FedAvg** strategy. |
| ↳ **Client Processes** | The server spawns 3 client processes (`client.py`) using **Ray**. Each client is given its own sandboxed dataset and performs local training on the GPU. |
| **Azure Cloud** | The central, immutable "lab notebook" for the project. |
| ↳ **Azure ML Workspace** | Acts as the central hub for tracking experiments, models, and results. |
| ↳ **MLflow Tracking** | The API used by the server to log metrics and parameters to the Azure ML run history. |
| ↳ **Azure Blob Storage** | The secure storage where the final, globally aggregated model (`final_federated_model.pth`) and XAI heatmaps are automatically uploaded. |

---

## 🧠 4. The Model: Fed-X-ViT

Our model's architecture is inspired by the diagnostic workflow of an expert radiologist:

1.  **CNN (EfficientNetV2)**: Acts as the **"Eyes."** It performs a high-resolution scan of the MRI, extracting thousands of low-level features like textures, edges, and local patterns indicative of anomalies.
2.  **ViT (Vision Transformer)**: Acts as the **"Brain."** It receives the rich feature map from the CNN, segments it into patches, and uses its **self-attention mechanism** to analyze the global context and relationships between all features. This allows it to make a final, highly-informed classification.

### The "Brain Surgery": 4-Class to 2-Class Transfer Learning

This is the core of our Federated Transfer Learning pipeline. Instead of training a new model from scratch, we adapt our pre-trained expert.

1.  **Load Expert**: We start with our 99.31% accurate 4-class "expert" model, which has learned a rich representation of brain MRI features.
2.  **Create New Model**: We instantiate a new, blank model with the same architecture but a 2-class (Tumor/Healthy) classification head.
3.  **Transplant Weights**: We surgically transfer the weights from all the feature extraction layers (the entire CNN and ViT backbone) from the expert model to the new model.
4.  **Isolate Final Layer**: We discard the old 4-class "decision layer" and leave the new 2-class decision layer randomly initialized.

This new hybrid model, which is **99% pre-trained**, is sent to the clients. They only need to fine-tune the final layer on their small, private datasets, making the federated process incredibly data-efficient and fast.

---

## 🏆 5. Performance & Results

### Phase 1: Baseline "Expert" Model
First, the Fed-X-ViT model was trained centrally on a public 4-class dataset (Glioma, Meningioma, Pituitary, No Tumor) to establish a powerful baseline.

| Metric | Result |
| :--- | :--- |
| **Final Test Set Accuracy** | **99.31%** |

### Phase 2: Federated Transfer Learning Simulation
Next, we ran the 3-client federated simulation for 5 rounds on the new 2-class (Tumor/Healthy) dataset.

![WhatsApp Image 2025-10-25 at 20 12 11_0dc238ec](https://github.com/user-attachments/assets/a39d13c8-5b40-4acc-af43-9561f20ddd2c)

| Parameter | Details |
| :--- | :--- |
| **Clients** | 3 (Simulated hospitals) |
| **Federation Strategy** | Federated Averaging (FedAvg) |
| **Total Training Images** | 11,693 |
| **Total Validation Images** | 2,505 |
| **Total Testing Images** | 2,514 |
| **Global Rounds** | 5 |
| **Framework** | Flower + Ray |
| **GPU** | NVIDIA RTX 3060 |

![WhatsApp Image 2025-10-25 at 20 12 11_33dd8352](https://github.com/user-attachments/assets/8723198c-1776-486b-9807-0b6ec26faa5a)

![WhatsApp Image 2025-10-25 at 20 12 11_e0e2a97a](https://github.com/user-attachments/assets/33ce8c06-a2fa-494b-839e-498201a896c5)


#### Azure ML Dashboard:
The following metrics were logged in real-time to Azure ML during the simulation run. The model showed rapid convergence, starting at an impressive 99.32% and peaking at **99.72%**—demonstrating that federated fine-tuning can improve an already-expert model.

<img width="1280" height="671" alt="image" src="https://github.com/user-attachments/assets/348adb86-c3f8-418e-816c-f19481e22a6b" />

<img width="1280" height="609" alt="image" src="https://github.com/user-attachments/assets/72f36336-273d-480e-bad1-8371fe70528c" />

#### Round-by-Round Global Validation Performance:
| Round | Global Validation Accuracy | Global Validation Loss |
| :---: | :------------------------: | :--------------------: |
| 1 | 99.321% | 0.0201 |
| 2 | 99.481% | 0.0148 |
| 3 | 99.481% | 0.0133 |
| 4 | 99.600% | 0.0116 |
| 5 | **99.720%** | **0.0078** |

### Final Federated Model Evaluation
After 5 rounds, the server saved the final global model (`final_federated_model.pth`) and automatically tested it on a combined, unseen test set of 2,514 images.

| Metric | Final Result |
| :--- | :--- |
| **Final Combined Test Accuracy** | **99.60%** |
| **Final Combined Test Loss** | **0.0125** |

#### Detailed Classification Report:
This report shows the final federated model has near-perfect, well-balanced performance for both classes.

| Class | Precision | Recall | F1-Score | Support |
| :------ | :-------: | :----: | :------: | :-----: |
| Healthy | 0.99 | 0.99 | 0.99 | 666 |
| Tumor | 1.00 | 1.00 | 1.00 | 1848 |
| **macro avg** | **0.99** | **1.00** | **0.99** | **2514** |
| **weighted avg** | **1.00** | **1.00** | **1.00** | **2514** |

---

## 🔬 6. Explainable AI (XAI) Validation

To ensure our model is trustworthy, we must verify that it is looking at the correct regions of an MRI scan to make its predictions. We use **Gradient-weighted Class Activation Mapping (Grad-CAM)** for this purpose.

### Methodology for Hybrid Models
Grad-CAM is traditionally used on CNNs. Our hybrid model requires a more sophisticated approach:
1.  **The ViT is the "Brain"**: The Vision Transformer makes the final classification decision (e.g., "Tumor").
2.  **The CNN is the "Eyes"**: Grad-CAM is applied to the last convolutional layer of the EfficientNetV2 backbone.
3.  **We ask Grad-CAM**: "Show me the pixels in the final feature map that were most important for the ViT's final 'Tumor' decision."

The result is a heatmap where the ViT's high-level reasoning guides the Grad-CAM's pixel-level visualization. This shows us exactly which parts of the original image contributed most to the final classification.

### Results
The `xai.py` script was run after the simulation completed. It randomly selected **25 tumor images** from each client's private test set and generated Grad-CAM heatmaps.


<p align="center">
  <img src="https://github.com/user-attachments/assets/4e732e54-ada8-43a5-b595-1ad1673ef999" width="250" height="250" style="border-radius:10px; margin: 5px;"/>
  <img src="https://github.com/user-attachments/assets/5cbb7311-9af5-49ec-8ff7-11ff47c96257" width="250" height="250" style="border-radius:10px; margin: 5px;"/>
  <img src="https://github.com/user-attachments/assets/0a56db25-5ced-4552-80f0-e70821db8660" width="250" height="250" style="border-radius:10px; margin: 5px;"/>
</p>



As shown in the image above, the heatmaps confirm that the model correctly focuses its attention on the tumorous regions of the brain. This proves that the model's high accuracy is a result of learning genuine, clinically relevant features, not exploiting biases or artifacts in the data.

---

## 🚀 7. How to Run the Simulation

### 🧰 Prerequisites
* Python 3.10+
* An NVIDIA GPU with CUDA 12.1+ support (e.g., RTX 3060 or higher).
* An [Azure Account](https://azure.microsoft.com/en-us/free/) with an Azure Machine Learning Workspace created. (We used Visual Studio Enterprise Subscription, credits to MLSA pgm)

### ⚙️ Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/harshitsinghcode/Fed-X-ViT.git](https://github.com/harshitsinghcode/Fed-X-ViT.git)
    cd Fed-X-ViT
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # On Windows :(
    # source venv/bin/activate  # On macOS/Linux :)
    ```

3.  **Install all required libraries:**
    * First, install PyTorch matching your system's CUDA version. Check your version with `nvidia-smi`.
        ```bash
        # Example for CUDA 12.1
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ```
    * Then, install the remaining dependencies:
        ```bash
        pip install -r requirements.txt
        ```

###  Azure Configuration

1.  **Download Azure Config File**:
    * Go to your Azure ML Workspace in the Azure Portal.
    * On the *Overview* page, click **Download config.json**.
    * Place this `config.json` file in the root folder of the project (`D:\FedXViT\`).

2.  **Authenticate with Azure CLI**:
    * Run this command in your terminal. It will open a browser window for you to log in.
    ```bash
    az login
    ```

### Data Preparation

1.  **Download Data**: Download the pre-organized client datasets and pre-trained models from the provided link.
2.  **Run the Splitting Script**: Run the data splitting and counting script once to organize your source data into the final `SplitData` folder, perfectly structured for the simulation.
    ```bash
    python count.py
    ```

### 🏃 Run the Full Simulation

You only need to run a single command. This will start the server, connect to Azure, spawn the 3 clients, run 5 rounds of federated training, save the final model, and run the final evaluation on the test set.

```bash
python server.py
```

To save the complete console output to a log file for later review:

```
python server.py | tee logs.txt
```

### 👥 8. Contributors

1. Sinchan Shetty `22BCE5238`
2. Riddhi Bandyopadhyay `22BCE1068`
3. Harshit Kumar Singh `22BLC1079`

Guided By
Dr. Suganya R `SCOPE - 52858`
