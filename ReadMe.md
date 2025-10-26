# Obesity Risk Prediction Project

## Project Overview
This project focuses on predicting obesity risk categories using various machine learning models. The goal is to classify individuals into different weight categories based on their physical and lifestyle characteristics.

## Dataset
The project uses multiple datasets:
- Training dataset (`data_train.csv`)
- Testing dataset (`data_test.csv`)
- Additional obesity dataset (`ObesityDataSet.csv`)

### Features
The dataset includes both numerical and categorical features:

#### Numerical Features:
- Age
- Height
- Weight
- FCVC (Frequency of consumption of vegetables)
- NCP (Number of main meals)
- CH2O (Consumption of water)
- FAF (Physical activity frequency)
- TUE (Time using technology devices)

#### Categorical Features:
Binary Features:
- Gender
- Family history with overweight
- FAVC (Frequent consumption of high caloric food)
- SMOKE (Smoking status)
- SCC (Calories consumption monitoring)

Target Variable:
- WeightCategory (Different levels of obesity risk)

## Methodology

### 1. Data Preprocessing
- Feature engineering including:
  - BMI calculation
  - Age-BMI interaction
  - FAF-CH2O interaction
  - FAVC-FCVC interaction
  - PCA on binary features
- Standard scaling for numerical features
- One-hot encoding for categorical features

### 2. Model Development
Multiple models were implemented and evaluated:

1. Base Models:
   - Decision Tree (CART)
   - K-Nearest Neighbors (KNN)
   - Gaussian Naive Bayes
   - AdaBoost Classifier

2. Advanced Model:
   - XGBoost (Final model)

### 3. Model Optimization
- Extensive hyperparameter tuning using:
  - GridSearchCV for base models
  - RandomizedSearchCV for XGBoost
- K-Fold cross-validation (K=5)
- Separate validation set for unbiased evaluation

### 4. Dataset Enhancement
- Combined original training data with additional obesity dataset
- Retrained the best model (XGBoost) on the expanded dataset

## Results
The project produced multiple submission files:
My best result in kaggle board is from ('letsee1.csv')

## Project Structure
```
.
├── AIT511_ML_Project.ipynb    # Main project notebook
├── ReadMe.md                  # Project documentation
├── data/                      # Data directory
│   ├── data_test.csv         # Test dataset
│   ├── data_train.csv        # Training dataset
│   └── ObesityDataSet.csv    # Additional obesity dataset
└── Results for Report/        # Model outputs
    ├── letsee1.csv
    ├── submission_xgboost_combined.csv
    └── submission_xgboost_tuned.csv
```

## Technical Implementation
The project is implemented in Python using libraries including:
- pandas & numpy for data manipulation
- scikit-learn for machine learning models
- xgboost for advanced gradient boosting
- matplotlib & seaborn for visualization
- ydata-profiling for data profiling

## Model Performance
The XGBoost model, after being trained on the combined dataset and properly tuned, showed the best performance among all tested models, demonstrating robust classification capabilities across different weight categories.
