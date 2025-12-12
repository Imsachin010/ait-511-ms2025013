# Smoker Status Prediction using Biosignals
## Binary Classification Project

### 📋 Project Overview
This project implements machine learning models to predict smoking status based on biosignal and demographic features. The dataset contains health indicators including blood test results, vital signs, and physical measurements for 38,984 individuals.

### 🎯 Objective
Develop and compare classification models to accurately identify smokers from non-smokers using health-related biosignals, with focus on maximizing both accuracy and F1-score for the positive (smoker) class.

### 📊 Dataset
- **Source:** [Kaggle - Smoker Status Prediction](https://www.kaggle.com/datasets/gauravduttakiit/smoker-status-prediction-using-biosignals)
- **Size:** 38,984 samples, 23 features
- **Target:** Binary (Smoker / Non-smoker)
- **Class Distribution:** 55.4% Smokers, 44.6% Non-smokers (after preprocessing)

### 🔧 Key Features
Original features include:
- **Demographics:** Age, height, weight
- **Vital Signs:** Systolic/diastolic blood pressure, heart rate
- **Blood Biomarkers:** Hemoglobin, cholesterol (HDL/LDL), triglycerides, Gtp, fasting blood sugar
- **Physical Measurements:** Waist circumference, eyesight, hearing

### 🛠️ Preprocessing & Feature Engineering

**Data Cleaning:**
- Removed 5,517 duplicate entries (14.2% of data)
- Final dataset: 33,467 samples
- No missing values detected

**Feature Engineering (254 total features):**
- Body Mass Index (BMI): weight(kg) / (height(m))²
- Height-to-Waist Ratio: height(cm) / waist(cm)
- Height-to-Age Ratio: height(cm) / age
- Systematic ratio generation between numerical features (231 engineered ratios)
- StandardScaler normalization applied to all features

**Data Split:**
- Training: 80% (26,774 samples)
- Testing: 20% (6,693 samples)
- Stratified sampling to maintain class distribution

### 🤖 Models Evaluated

| Model | Accuracy | Precision (Smoker) | Recall (Smoker) | F1-Score (Smoker) | Training Time |
|-------|----------|-------------------|-----------------|-------------------|---------------|
| **Tuned Logistic Regression** | **75.16%** | **0.70** | **0.62** | **0.65** | **2.9s** |
| Tuned SVM (RBF) | 74.59% | 0.69 | 0.59 | 0.65 | 23.4s |
| Basic Neural Network | 72.50% | 0.61 | 0.53 | 0.57 | 41.2s |

### 🔮 Future Improvements
- Implement dropout and early stopping for Neural Network
- Apply LASSO for automated feature selection from 254 features
- Test ensemble methods (Random Forest, XGBoost)
- Explore SHAP values for model interpretability

### 📚 Requirements
```
python >= 3.8
scikit-learn >= 1.0
pandas >= 1.3
numpy >= 1.21
optuna >= 3.0
tensorflow >= 2.8 (for Neural Network)
matplotlib >= 3.4
seaborn >= 0.11
```

# Forest Cover Type Prediction
## Multi-class Classification Project

### 📋 Project Overview
This project predicts the predominant forest cover type in 30×30 meter land parcels using cartographic and environmental features. The dataset contains geographical, topological, and soil composition data for wilderness areas in Roosevelt National Forest, Colorado.

### 🎯 Objective
Develop and compare classification models to accurately predict one of seven forest cover types based on 54 features including elevation, slope, distance measurements, wilderness area, and soil type indicators.

### 📊 Dataset
- **Source:** [Kaggle - Forest Cover Type Dataset](https://www.kaggle.com/datasets/uciml/forest-cover-type-dataset)
- **Size:** 581,012 samples, 54 features
- **Target:** 7-class (Cover Types 1-7)
- **Class Distribution:** Highly imbalanced (see table below)

| Cover Type | Description | Sample Count | Percentage |
|------------|-------------|--------------|------------|
| Type 2 | Lodgepole Pine | 226,640 | 39.0% |
| Type 1 | Spruce/Fir | 169,472 | 29.2% |
| Type 3 | Ponderosa Pine | 28,603 | 4.9% |
| Type 7 | Krummholz | 16,408 | 2.8% |
| Type 6 | Aspen | 13,894 | 2.4% |
| Type 5 | Douglas-fir | 7,594 | 1.3% |
| Type 4 | Cottonwood/Willow | 2,198 | 0.4% |

**Imbalance Ratio:** 103:1 (most frequent to least frequent)

### 🔧 Feature Categories

**Numerical Features (10):**
- Elevation (1859m - 3858m)
- Aspect (0° - 360°)
- Slope (0° - 66°)
- Horizontal_Distance_To_Hydrology (0m - 1397m)
- Vertical_Distance_To_Hydrology (-173m - 601m)
- Horizontal_Distance_To_Roadways (0m - 7117m)
- Hillshade_9am, Hillshade_Noon, Hillshade_3pm (0-255)
- Horizontal_Distance_To_Fire_Points (0m - 7173m)

**Categorical Features (44):**
- 4 Binary Wilderness Area indicators
- 40 Binary Soil Type indicators (highly sparse: 5 types with 0 occurrences, 18 types <1% frequency)

### 🛠️ Preprocessing Pipeline

**Data Preparation:**
- Train-test split: 80-20 (464,810 train / 116,202 test)
- Stratified sampling to maintain class proportions
- StandardScaler normalization (mean=0, std=1 for all features)

**Class Imbalance Mitigation:**
Applied SMOTE (Synthetic Minority Over-sampling Technique):
- **Pre-SMOTE:** 464,810 samples (imbalanced)
- **Post-SMOTE:** 1,269,184 samples (all classes balanced to 181,312 each)
- Configuration: `k_neighbors=5, sampling_strategy='auto'`
- Training set expansion: 2.7× original size

### 🤖 Models Evaluated

| Model | Test Accuracy | Macro F1-Score | Training Time | Status |
|-------|---------------|----------------|---------------|---------|
| K-Means Clustering | 31.10% | 0.16 | 47s | ❌ Failed |
| GMM Clustering | 19.86% | 0.19 | 112s | ❌ Failed |
| Logistic Regression | 56.40% | 0.48 | 12.0s | ⚠️ Limited |
| **Neural Network** | **87.47%** | **0.85** | **598s** | ✅ **Best** |

### 🏆 Best Model: Neural Network


### 👤 Author
**Sachin Mishra (MS2025013)**  
Department of Artificial Intelligence and Data Science  
IIIT-Bangalore

### 📝 License
Academic Project - AIT-511 Machine Learning

---
*Last Updated: December 12, 2025*

