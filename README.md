# 🦴 Bone Cancer Detection using Deep Learning

## Overview

This project is an AI-powered Bone Cancer Detection System that analyzes bone X-ray images and classifies them as:

* Benign / Normal
* Malignant / Tumor

The system uses a ResNet50 deep learning model trained on bone X-ray images and provides visual explanations using Grad-CAM.

<img width="1280" height="583" alt="image" src="https://github.com/user-attachments/assets/d8e431fb-9a69-4b74-8b6f-5dd2e252c9d7" />


---

## Features

* Bone X-ray image classification
* Benign vs Malignant detection
* Confidence score generation
* Grad-CAM visualization
* FastAPI backend
* Interactive web frontend
* X-ray image validation

---

## Technology Stack

### Backend

* Python
* FastAPI
* PyTorch
* TorchVision
* OpenCV
* NumPy
* PIL

### Explainability

* Grad-CAM

### Frontend

* HTML
* CSS
* JavaScript

---

## Project Structure

bone-cancer-detection/

├── bone-cancer-ui/

│   ├── backend/

│   │   ├── main.py

│   │   ├── model_inference.py

│   │   ├── bone_cancer_model.pth

│   │   └── requirements.txt

│   └── frontend/

│       ├── index.html

│       ├── app.js

│       └── style.css

---

## Model Architecture

* ResNet50
* Transfer Learning
* Binary Classification
* Softmax Confidence Prediction

Classes:

1. Benign / Normal
2. Malignant / Tumor

---

## Explainable AI

The system uses Grad-CAM (Gradient-weighted Class Activation Mapping) to highlight image regions that influence the model's prediction.

This helps improve interpretability and trustworthiness of the AI model.

---

## Running the Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Start Backend

```bash
python -m uvicorn main:app --reload
```

### Access API Documentation

```text
http://127.0.0.1:8000/docs
```

### Launch Frontend

Open:

```text
frontend/index.html
```

in a browser.

---

## Future Improvements

* Cloud deployment
* Multi-class bone disease detection
* Mobile application support
* Enhanced dataset expansion
* Clinical integration workflow

---

## Author

Rabia Baiju

Computer Science Engineering Student

Artificial Intelligence & Healthcare Applications
