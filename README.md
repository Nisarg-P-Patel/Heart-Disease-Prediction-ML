# Heart Disease Prediction Using Machine Learning

## Project Overview

This project aims to predict the likelihood of heart disease using machine learning algorithms. The dataset used for this analysis was obtained from Kaggle and includes various features related to heart health. Multiple machine learning models are tested and evaluated to determine the best-performing model for predicting heart disease.

## Dataset

Kaggle dataset link: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

The dataset consists of the following features:
1. **age**: Age in years
2. **sex**: Gender (1 = male; 0 = female)
3. **cp**: Chest pain type
   - 0: Typical angina
   - 1: Atypical angina
   - 2: Non-anginal pain
   - 3: Asymptomatic
4. **trestbps**: Resting blood pressure (in mm Hg)
5. **chol**: Serum cholesterol (in mg/dl)
6. **fbs**: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
7. **restecg**: Resting electrocardiographic results
   - 0: Normal
   - 1: ST-T Wave abnormality
   - 2: Left ventricular hypertrophy
8. **thalach**: Maximum heart rate achieved
9. **exang**: Exercise induced angina (1 = yes; 0 = no)
10. **oldpeak**: ST depression induced by exercise relative to rest
11. **slope**: Slope of the peak exercise ST segment
    - 0: Upsloping
    - 1: Flatsloping
    - 2: Downsloping
12. **ca**: Number of major vessels colored by fluoroscopy
13. **thal**: Thalium stress test result
    - 1,3: Normal
    - 6: Fixed defect
    - 7: Reversible defect
14. **target**: Presence of heart disease (1 = yes, 0 = no)

## Algorithms Used

1. **K-Nearest Neighbors (KNN) Classifier**
2. **Decision Tree Classifier**
3. **Gaussian Naive Bayes Classifier**
4. **Support Vector Machine (SVM) Classifier**
5. **Random Forest Classifier**

## Data Processing

- Converted categorical variables into dummy variables
- Scaled continuous features using StandardScaler
- Split data into training and testing sets

## Model Evaluation

The performance of each model was evaluated based on accuracy scores. The following metrics were calculated for each model:
- Training Accuracy
- Testing Accuracy
- Classification Report
- Confusion Matrix

## Results

The Random Forest Classifier outperformed other algorithms in terms of accuracy for this particular dataset.

![image](https://github.com/user-attachments/assets/dc7d8d6e-afaf-466d-af35-fbdff2338e4d)

## Installation and Setup

To run this notebook locally, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/Nisarg143/Heart-Disease-Prediction-ML.git
2. Install depenecies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn pydrive
