# AI Disease Detection System

An end-to-end machine learning system that predicts the risk of four major diseases — **Diabetes**, **Heart Disease**, **Chronic Kidney Disease**, and **Brain Tumor** — using trained ML models and a web-based interface.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Models](#models)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Notebooks](#notebooks)
- [Model Performance](#model-performance)
- [Technologies Used](#technologies-used)
- [License](#license)

---

## Overview

This project builds predictive models for four common diseases using tabular patient data and medical imaging:

| Disease | Data Type | Best Model | Approach |
|---------|-----------|------------|----------|
| **Diabetes** | Tabular (100K+ records) | Random Forest | Classification |
| **Heart Disease** | Tabular (302 records) | Random Forest | Classification |
| **Chronic Kidney Disease** | Tabular (20K+ records) | Random Forest | Classification |
| **Brain Tumor** | Medical Imaging (MRI) | CNN (Deep Learning) | Image Classification |

The trained models are served through a Streamlit web application where users can input patient parameters (age, BMI, blood pressure, lab values, etc.) and receive a risk prediction instantly.

---

## Features

- **Multi-disease prediction** — 4 disease models in a single application
- **Web-based interface** — Streamlit app with an intuitive, interactive UI
- **Pre-trained models** — Models are saved as `.pkl` files for fast inference
- **Brain tumor image upload** — Upload MRI scans for automated tumor detection
- **Data preprocessing pipelines** — Label encoding, scaling, and feature engineering
- **Exploratory Data Analysis (EDA)** — Comprehensive statistical analysis and visualizations
- **Model comparison** — Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost evaluated for each disease

---

## Project Structure

```
AI diseas Detection/
├── app/
│   └── app.py                  # Streamlit web application
├── dataset/
│   ├── diabetes_prediction_dataset.csv    # Raw diabetes data (100K rows)
│   ├── diabetes_cleaned.csv               # Cleaned diabetes dataset
│   ├── heart.csv                          # Raw heart disease data (UCI)
│   ├── heart_cleaned.csv                  # Cleaned heart dataset (302 rows)
│   ├── kidney_disease_dataset.csv         # Raw kidney disease data (20K rows)
│   ├── kidney_cleaned.csv                 # Cleaned kidney dataset
│   └── Brain_tumor/                       # MRI images for brain tumor detection
│       ├── Training/
│       │   ├── tumor/
│       │   └── notumor/
│       └── Testing/
│           ├── tumor/
│           └── notumor/
├── models/
│   ├── diabetes_rf.pkl          # Trained diabetes Random Forest model (59 MB)
│   ├── heart_rf.pkl             # Trained heart Random Forest model (0.7 MB)
│   ├── kidney_rf.pkl            # Trained kidney Random Forest model (36 MB)
│   ├── kidney_encoders.pkl      # Label encoders for kidney categorical features
│   ├── le_gender.pkl            # Label encoder for gender
│   ├── le_smoke.pkl             # Label encoder for smoking history
│   ├── scaler_heart.pkl         # Standard scaler for heart features
│   └── standard_scaler.pkl      # Standard scaler for diabetes features
├── notebooks/
│   ├── EDA.ipynb                # Exploratory Data Analysis for all 3 tabular datasets
│   └── ML_Model.ipynb           # Model training, evaluation, and hyperparameter tuning
├── .gitattributes               # Git LFS configuration for model files
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## Datasets

### Diabetes Prediction Dataset
- **Source**: Kaggle — Diabetes Prediction Dataset
- **Records**: 100,000 rows, 9 columns
- **Features**: gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level
- **Target**: diabetes (0 = No, 1 = Yes)
- **Preprocessing**: Removed duplicates (3,854), encoded categorical columns (gender, smoking_history)

### Heart Disease Dataset
- **Source**: UCI Machine Learning Repository — Heart Disease
- **Records**: 302 unique rows, 14 columns
- **Features**: age, sex, cp (chest pain type), trestbps (resting BP), chol (cholesterol), fbs, restecg, thalach (max heart rate), exang, oldpeak, slope, ca, thal
- **Target**: target (0 = no disease, 1 = disease present)
- **Preprocessing**: Removed duplicates (723 rows)

### Chronic Kidney Disease Dataset
- **Source**: Kaggle — Chronic Kidney Disease
- **Records**: 20,538 rows, 43 columns
- **Features**: 41 clinical/biochemical parameters (blood pressure, specific gravity, albumin, sugar, blood cells, glucose, hemoglobin, BMI, CRP, IL-6, etc.)
- **Target**: Target_Binary (0 = No Disease, 1 = Disease Risk)
- **Preprocessing**: Encoded 14 categorical columns (binary and multi-class)

### Brain Tumor Dataset
- **Type**: MRI image dataset
- **Classes**: `tumor` / `notumor`
- **Split**: Training and Testing directories with subfolders for each class
- **Preprocessing**: Images stored via Git LFS

---

## Models

Five classifiers were evaluated for each tabular disease: Logistic Regression, Decision Tree, Random Forest, SVM, and XGBoost. **Random Forest** was selected as the best model across all three diseases based on F1 score.

A **Convolutional Neural Network (CNN)** is used for brain tumor image classification (trained on MRI scans).

All models are serialized with `joblib` and loaded at runtime in the Streamlit app.

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**

```bash
git clone <repository-url>
cd "AI diseas Detection"
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Verify model files**

Ensure all model `.pkl` files are present in the `models/` directory. Git LFS was used to version these files — if cloning on a fresh machine, Git LFS must be installed:

```bash
# Install Git LFS
git lfs install
git lfs pull
```

### Running the Application

```bash
streamlit run app/app.py
```

This launches the web application at `http://localhost:8501` (or another available port).

---

## Usage

1. **Select a disease** from the sidebar (Diabetes, Heart Disease, Kidney Disease, or Brain Tumor)
2. **Enter patient parameters** — the form adapts to the selected disease
3. **Get prediction** — click "Predict" to see the result (Disease / No Disease)

For **Brain Tumor**, upload an MRI scan image instead of entering numeric parameters.

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `notebooks/EDA.ipynb` | Exploratory Data Analysis — data profiling, null checks, duplicate removal, distribution analysis, correlation heatmaps, and statistical summaries for all three tabular datasets |
| `notebooks/ML_Model.ipynb` | Model training pipeline — feature encoding, train/test split, standard scaling, 5-model comparison, ROC curve visualization, GridSearchCV hyperparameter tuning, and model serialization |

---

## Model Performance

### Diabetes (Random Forest — Best Model)
| Metric | Value |
|--------|-------|
| Accuracy | 97.0% |
| Precision | 94.2% |
| Recall | 68.9% |
| F1 Score | 79.6% |
| ROC-AUC | 96.4% |

### Heart Disease (Random Forest — Best Model)
| Metric | Value |
|--------|-------|
| Accuracy | 80.3% |
| Precision | 83.9% |
| Recall | 78.8% |
| F1 Score | 81.3% |
| ROC-AUC | 87.4% |

> **Note**: The kidney disease model showed lower performance metrics due to significant class imbalance (80% No Disease vs 20% Disease). Further work with resampling techniques (SMOTE) or alternative models is recommended to improve results.

---

## Technologies Used

| Category | Tools |
|----------|-------|
| **Language** | Python 3.x |
| **Web Framework** | Streamlit |
| **Data Processing** | pandas, NumPy |
| **Machine Learning** | scikit-learn, XGBoost |
| **Visualization** | matplotlib, seaborn |
| **Serialization** | joblib |
| **Version Control** | Git, Git LFS |

---

## License

This project is for educational and research purposes. Medical predictions should **never** be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for any health concerns.
