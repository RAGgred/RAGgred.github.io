---
title: "CNN vs ANN: A Deep Dive into Image Classification using CIFAR-10"
author: RAG
date: 25-04-16 14:11
categories:
  - Machine Learning
  - Computer Vision
tags:
  - CNN
render_with_liquid: false
---

## 📌 Project Overview

In this project, I explore the effectiveness of **Convolutional Neural Networks (CNNs)** compared to traditional **Artificial Neural Networks (ANNs)** for image classification. Leveraging the **CIFAR-10** dataset, the task involves categorizing 32x32 color images across 10 distinct classes such as airplane, automobile, bird, cat, and more.

Full code found here. 


---

## 🎯 Objective

- Compare the performance of ANN and CNN architectures.
- Explore data preprocessing, augmentation, and sampling strategies.
- Build and optimize a CNN model using regularization, dropout, batch normalization, and early stopping.
- Evaluate model performance using stratified k-fold cross-validation and key metrics.

---

## 📊 Dataset Summary

- **Source**: CIFAR-10 (Krizhevsky, 2009)  
- **Images**: 60,000 color images (32x32x3)  
- **Classes**: 10  
- **Train/Test Split**: 50,000/10,000  
- **Challenge**: Class imbalance across training batches

---

## 🧪 Model Architecture Comparison

### ANN Structure:
```python
model = Sequential([
    Dense(3000, activation='relu', input_shape=(3072,)),
    Dense(1000, activation='relu'),
    Dense(10, activation='softmax')
])
```

### CNN Structure (Final Model):
```python
model = Sequential([
    Input(shape=(32, 32, 3)),

    Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
```

---

## 🧪 Training Techniques & Strategy

- **Stratified Sampling** to maintain class balance
- **Data Augmentation** using:
  - Rotation
  - Width/Height shift
  - Zoom
  - Horizontal Flip
- **Loss Function**: Categorical Cross Entropy
- **Optimizer**: Adam
- **Regularization**: L2 Kernel Regularization
- **Early Stopping** after 5 epochs without improvement
- **Evaluation**: Stratified K-Fold Cross Validation (5 folds)

---

## 📈 Performance Progression

| Model Version | Test Accuracy | Notes                        |
|---------------|---------------|-------------------------------|
| Test 14       | 54%           | No sampling or augmentation  |
| Model 19      | **85%**       | With full optimizations      |

### 📉 Accuracy & Loss Plot
![Training and Validation Accuracy Over Epochs](your-image-path/accuracy_plot.png)

### 🔍 Confusion Matrix (Final Model)
![Confusion Matrix](your-image-path/confusion_matrix.png)

---

## 🔄 Sample Predictions

| Image | Actual Class | Predicted Class |
|-------|--------------|-----------------|
| ![](your-image-path/img1.png) | Ship | Truck |
| ![](your-image-path/img2.png) | Cat | Bird |
| ![](your-image-path/img3.png) | Frog | Frog ✅ |

> Model correctly classified 8 out of 10 random images.

---

## 🧠 Key Learnings

- ✅ **CNNs vastly outperform ANNs** on image data due to spatial feature extraction.
- ✅ **Stratified k-fold CV** helps mitigate class imbalance.
- ✅ **Data augmentation** significantly improves generalization.
- ✅ **Regular logging and incremental tuning** are crucial for tracking improvements.

---

## 📚 References

- Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*  
- Kubat, M. (2021). *An Introduction to Machine Learning*  
- Burkov, A. (2019, 2020). *Machine Learning Engineering*  
- Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*  
- Brownlee, J. (2018, 2019). *Machine Learning Mastery*

---

## 🛠️ Next Steps

- Try **advanced activations** like Swish or GELU
- Benchmark against **ResNet/VGG** models
- Explore **automated hyperparameter tuning** (e.g., Optuna)
