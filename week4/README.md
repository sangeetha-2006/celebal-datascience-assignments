# 📘 Week 4: CIFAR-10 Image Classification Project (ANN vs CNN)

## 🎯 Project Overview
This project demonstrates image classification using the **CIFAR-10 dataset** by implementing and comparing:

- Artificial Neural Network (ANN)
- Convolutional Neural Network (CNN)
- CNN with Data Augmentation

The goal is to understand why CNNs perform better than ANN for image-based tasks and how training strategies improve performance.

---

## 📦 Dataset
We use the CIFAR-10 dataset which contains:

- 60,000 color images (32×32×3)
- 10 classes:
  - airplane
  - automobile
  - bird
  - cat
  - deer
  - dog
  - frog
  - horse
  - ship
  - truck

Split:
- 50,000 training images
- 10,000 test images

---

## 🧠 Models Implemented

### 1️⃣ ANN (Artificial Neural Network)
- Input: Flattened image (3072 features)
- Dense layers with ReLU activation
- Dropout for regularization
- Softmax output layer

📊 Limitation:
- Ignores spatial structure of images

---

### 2️⃣ CNN (Convolutional Neural Network)
- Conv2D + MaxPooling layers
- Batch Normalization
- Dense + Dropout layers

📊 Advantage:
- Captures spatial features (edges, shapes, patterns)

---

### 3️⃣ CNN with Data Augmentation
- RandomFlip
- RandomRotation
- RandomZoom
- Improved generalization

---

## ⚙️ Training Strategy
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Batch Size: 64
- Epochs: 10–20
- Validation Split: 0.1
- EarlyStopping (for improved training stability)

---

## 📊 Results

| Model | Test Accuracy |
|------|--------------|
| ANN | ~42% |
| CNN | ~68% |
| Augmented CNN | ~71% |

---

## 📈 Key Observations
- ANN performs poorly on image data due to loss of spatial structure
- CNN significantly improves performance by learning visual features
- Data augmentation further improves generalization and reduces overfitting
- Best performance achieved using **CNN + augmentation**

---

## 🧪 Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Pandas
