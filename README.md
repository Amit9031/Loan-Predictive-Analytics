# 🏦 Loan Prediction Analytics - Comprehensive ML Project

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange.svg)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.52.0-red.svg)](https://streamlit.io/)

A complete end-to-end machine learning solution for predicting loan approval status using **15+ different algorithms** including regression, classification, clustering, and deep learning models. Features an interactive Streamlit web application for real-time predictions and comprehensive model evaluation.

## 📋 Overview

This project demonstrates a comprehensive approach to loan prediction using multiple machine learning techniques. The system automatically trains, evaluates, and compares various models to identify the best performers, then provides an interactive web interface for making predictions and analyzing model performance.

**🎯 Key Highlights:**
- ✅ **15+ machine learning models** implemented (Regression, Classification, Clustering, Neural Networks)
- ✅ **Automatic model selection** - System identifies top 2 best-performing models
- ✅ **Comprehensive evaluation metrics** - Accuracy, Precision, Recall, F1, AUC, MAE, MSE, RMSE, R²
- ✅ **Interactive Streamlit web application** - Real-time predictions and visualizations
- ✅ **Production-ready code** - Proper data preprocessing, feature engineering, and model persistence
- ✅ **Advanced techniques** - PCA, Cross-validation, Ensemble methods, Deep Learning
- ✅ **Best Model Performance**: Logistic Regression (86.18% accuracy)

**Perfect for:**
- Learning various ML algorithms and their implementations
- Comparing model performance across different techniques
- Building production-ready loan prediction systems
- Understanding end-to-end ML pipeline development

## Features

### Regression Models
- Simple Linear Regression
- Multiple Linear Regression
- Polynomial Regression
- Logistic Regression

### Classification Models
- Naive Bayes
- Decision Trees
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Random Forest
- Bagging
- AdaBoost

### Clustering Models
- K-Means Clustering
- Hierarchical Clustering (Agglomerative)

### Neural Networks
- Multi-layer Perceptron (MLP)
- Convolutional Neural Network (CNN)
- Recurrent Neural Network (LSTM)

### Dimensionality Reduction
- Principal Component Analysis (PCA)

### Model Evaluation
- Accuracy, Precision, Recall, F1 Score
- Confusion Matrix
- ROC-AUC Score
- Logarithmic Loss
- MAE, MSE, RMSE, R² Score
- Cross-Validation (K-Fold, Leave-One-Out)

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Train Models
Train all models on the dataset:
```bash
python model_training.py
```

This will:
- Load and preprocess the data
- Train all models
- Evaluate model performance
- Save trained models
- Generate model comparison results

### Step 2: Run Web Application
Launch the Streamlit web interface:
```bash
streamlit run app.py
```

The web application provides:
- **Model Comparison**: View rankings and performance of all models
- **Make Prediction**: Interactive interface to predict loan approval
- **Model Performance**: Detailed metrics for each model
- **Dataset Info**: Dataset statistics and visualizations

## Project Structure

```
.
├── train_u6lujuX_CVtuZ9i (1).csv    # Dataset
├── model_training.py                  # Model training script
├── app.py                             # Streamlit web application
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
├── *.pkl                              # Saved models (generated after training)
├── *.h5                               # Saved neural networks (generated after training)
└── model_results.json                 # Model evaluation results
```

## Model Selection

The system automatically:
1. Trains all models
2. Evaluates performance using multiple metrics
3. Ranks models by accuracy
4. Selects the top 2 best-performing models
5. Uses these models for predictions in the web interface

## Dataset

The dataset contains loan application information with the following features:
- Gender
- Married
- Dependents
- Education
- Self_Employed
- ApplicantIncome
- CoapplicantIncome
- LoanAmount
- Loan_Amount_Term
- Credit_History
- Property_Area
- Loan_Status (Target Variable)

## Notes

- Models are saved after training for reuse
- The web application automatically loads the best models
- All preprocessing steps are handled automatically
- Missing values are imputed using appropriate strategies

## Requirements

- Python 3.8+
- See requirements.txt for package versions



