# Face Mask Classification using ResNet-50 (Transfer Learning)

This project implements a **binary face mask classification system** using deep learning.
Faces are first extracted from images using ground-truth bounding boxes, then classified as **with mask** or **without mask** using a **ResNet-50** model pretrained on ImageNet.

---

## 📌 Project Overview

* **Task:** Binary image classification (with_mask vs without_mask)
* **Domain:** Computer Vision, Deep Learning
* **Model:** ResNet-50 (Transfer Learning)
* **Framework:** TensorFlow / Keras
* **Dataset:** Face Mask Detection Dataset (Kaggle)

---

## 🗂️ Dataset

* **Source:** Kaggle – Face Mask Detection Dataset
* **Annotations:** Pascal VOC (XML)
* **Original classes:**

  * `with_mask`
  * `without_mask`
  * `mask_weared_incorrect`

For this project, *incorrectly worn masks* are merged with `without_mask` to form a **binary classification problem**, reflecting real-world mask compliance.

---

## ⚙️ Methodology

### 1. Face Extraction

* Faces are cropped from original images using bounding boxes from XML annotations.
* Each face is saved into class-specific folders.

### 2. Data Preparation

* Image resizing to 224×224
* Train/validation split (80/20)
* Prefetching for performance

### 3. Model Architecture

* **Backbone:** ResNet-50 pretrained on ImageNet
* **Head:**

  * Global Average Pooling
  * Dropout (regularization)
  * Dense Softmax layer (2 classes)

The backbone is frozen to reduce overfitting and training time.

---

## 🧠 Model

* **Architecture:** ResNet-50 + custom classification head
* **Loss:** Sparse Categorical Crossentropy
* **Optimizer:** Adam
* **Metrics:** Accuracy

> ResNet-50 was selected for its residual connections, which facilitate the training of deep networks and robust feature extraction.

---

## 📊 Evaluation

The model is evaluated using:

* Accuracy (training & validation)
* Loss curves
* Confusion matrix
* Precision, Recall, F1-score
* ROC curve and AUC
* Error analysis (misclassified samples)

These metrics provide a comprehensive view of model performance and generalization.

---

## 📈 Results (example)

* Validation Accuracy: ~XX%
* F1-score: ~XX
* ROC-AUC: ~X.XX

*(Exact results depend on training configuration and random seed.)*

---

## 🧪 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the dataset

* Download the dataset from Kaggle
* Place it in the appropriate directory
* Run the face extraction script

### 3. Train the model

```bash
python train.py
```

---

## 📁 Project Structure

```
face-mask-classification-resnet50/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── CNN_Face_Mask_Detection.ipynb
├── src/
│   ├── make_dataset.py
│   ├── train.py
│   └── predict.py
├── models/
└── data/
```

*(Datasets and trained models are not included in the repository.)*

---

## 🚀 Future Improvements

* Fine-tuning upper layers of ResNet-50
* Data augmentation
* Real-time webcam inference
* Comparison with other architectures (MobileNet, EfficientNet)
* Deployment as a web or mobile application

