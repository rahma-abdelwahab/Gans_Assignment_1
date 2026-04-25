Here’s your cleaned, ready-to-paste **README.md** (properly formatted and polished). Just copy everything below 👇

````markdown
# DSAI 490 - Assignment 1: Medical Image Reconstruction with Autoencoders

## 📋 Project Overview

This project implements and compares two unsupervised deep learning models for **medical image reconstruction**:

- **Standard Autoencoder (AE)**
- **Variational Autoencoder (VAE)**

The models are trained separately on six anatomical regions from the **MedNIST** dataset:

- AbdomenCT  
- BreastMRI  
- CXR (Chest X-Ray)  
- ChestCT  
- Hand  
- HeadCT  

**Goal**: Reconstruct 64×64 grayscale medical images while comparing reconstruction quality (MSE) and training time between AE and VAE.

The project emphasizes clean modular code using `tf.data` pipelines, Keras models, and per-region training.

---

## 📁 Project Structure

```bash
Assignment 1/
├── notebook/                          
│   └── DSAI490_Assignment_1_Final_Notebook.ipynb
├── src/
│   ├── __init__.py
│   ├── ae_model.py                    
│   ├── vae_model.py                   
│   ├── config.py                      
│   ├── data_processing.py             
│   └── train.py                       
├── README.md
├── requirements.txt
└── (models/ - created during training)
````

---

## 🛠️ Setup & Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd Assignment-1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Dataset

The notebook expects the MedNIST dataset inside Google Drive at:

```
/content/drive/MyDrive/DSAI 490 : lab practice 7/archive.zip
```

It will be automatically extracted to:

```
/content/mnist_data/
```

Run on **Google Colab (GPU recommended)**.

---

## 🚀 How to Run

1. Open the notebook in **Google Colab**
2. Run all cells sequentially

The notebook will:

* Mount Google Drive & extract the dataset
* Build `tf.data` pipelines (90/10 train-validation split)
* Train **Autoencoder (AE)** and **Variational Autoencoder (VAE)** for each region
* Compute reconstruction MSE
* Generate comparison plots (AE vs VAE)

---

## ⚙️ Hyperparameters

Defined in `config.py` or notebook:

* Image size: **64×64 (grayscale)**
* Latent dimension: **32**
* Batch size: **128**
* Epochs: **30**
* Learning rate: **1e-3**
* Noise (optional): **std = 0.15**

---

## 📊 Results

* Final comparison uses **Mean Squared Error (MSE)** (lower is better)
* Bar chart compares AE vs VAE across all regions

Typical observations:

* **AE** → Lower reconstruction error (sharp outputs)
* **VAE** → Better latent structure & generalization

👉 *(Add your actual MSE values here after running the notebook)*

---

## 🧩 Key Features

* Modular design (`src/` separation)
* Efficient `tf.data` pipeline
* Independent training per anatomical region
* Reproducibility (fixed seeds)
* Optional denoising autoencoder support

