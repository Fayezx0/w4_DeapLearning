# 🐎 Horse Breed Classification using MaxViT

> **A Deep Learning project leveraging State-of-the-Art Vision Transformers to solve fine-grained image classification challenges.**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Model](https://img.shields.io/badge/Model-MaxViT__T-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📌 Project Overview
Horse breed recognition is a classic example of **Fine-Grained Image Classification**. Unlike distinguishing a cat from a dog, distinguishing an *Akhal-Teke* from an *Arabian* horse requires a model that can perceive subtle details in muscle structure, coat texture, and body proportions.

This project implements a robust classification pipeline using **Transfer Learning**. We utilize **MaxViT (Multi-Axis Vision Transformer)**, a hybrid architecture that combines the local feature extraction capabilities of CNNs with the global context understanding of Transformers.

## 📂 The Dataset
The dataset was sourced from Kaggle and consists of images representing 7 distinct horse breeds. 

### Data Challenge & Solution
Unlike standard datasets where images are sorted into folders, this dataset required **custom parsing**. The class labels were embedded in the filenames (e.g., `01_005.png`). 

We implemented a **Custom Dataset Class** in PyTorch to:
1.  Parse filename prefixes to extract labels.
2.  Map numerical prefixes to breed names.
3.  Apply preprocessing pipelines on the fly.

### Classes
1.  **Akhal-Teke**
2.  **Appaloosa**
3.  **Orlov Trotter**
4.  **Vladimir Heavy Draft**
5.  **Percheron**
6.  **Arabian**
7.  **Friesian**

## 🧠 Model Architecture: Why MaxViT?
We selected **MaxViT-T (Tiny)** from the `torchvision` library. MaxViT is a hybrid architecture developed by Google Research that bridges the gap between Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs).

### Key Features:
* **Multi-Axis Attention:** Traditional Transformers are computationally expensive on high-resolution images. MaxViT solves this by decomposing attention into "Block Attention" (local) and "Grid Attention" (global).
* **Hybrid Design:** It uses convolutional layers in the early stages to capture low-level features (edges, textures of the horse coat) and Transformer layers in deeper stages to understand semantic structure (the shape of the horse).
* **Efficiency:** It achieves better accuracy than ResNet-50 and standard ViT models with lower computational cost.

## ⚙️ Methodology
We conducted a two-phase experiment to optimize performance:

### Phase 1: Feature Extraction (Frozen Model)
* **Objective:** Establish a baseline using the pre-trained knowledge of MaxViT (trained on ImageNet).
* **Technique:** All backbone layers were **frozen** (`requires_grad=False`). Only the final classification head was replaced and trained.
* **Result:** The model learned quickly but struggled with distinguishing visually similar breeds due to the generic nature of ImageNet features.

### Phase 2: Fine-Tuning (Unfrozen Model)
* **Objective:** Specialize the model specifically for equine features.
* **Technique:** We **unfroze** the entire network. A lower Learning Rate (`1e-4`) was used to gently update the weights without destroying the pre-trained patterns.
* **Result:** Significant improvement in accuracy and confidence scores. The model learned specific breed traits (e.g., the metallic sheen of the Akhal-Teke vs. the heavy build of the Percheron).

## 🛠️ Installation & Setup

### Prerequisites

```bash
python -m venv venv
.\venv\Scripts\activate
pip pip install -r requirements.txt