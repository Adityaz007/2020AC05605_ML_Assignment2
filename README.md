
## Problem Statement
The goal of this project is to build and evaluate multiple machine learning classification models to predict wine quality based on physicochemical properties. The dataset, sourced from the UCI Machine Learning Repository, contains measurements such as acidity, sugar content, pH, sulphates, alcohol percentage, and other chemical attributes of red and white wines.
## DatasetDescription
The Wine Quality dataset is sourced from the UCI Machine Learning Repository and contains physicochemical measurements of red and white wines. It has 4,898 instances and 12 input features such as acidity, sugar, pH, sulphates, and alcohol percentage. Each wine sample is rated for quality on a scale of 0–10 by expert tasters.
## Github Repository Link

## https://github.com/Adityaz007/2020AC05605_ML_Assignment2/tree/main

## Model Comparison Results

| Model               | Accuracy | Precision | Recall  | F1 Score | MCC     | AUC     |
|---------------------|----------|-----------|---------|----------|---------|---------|
| Logistic Regression | 0.821538 | 0.609091  | 0.261719| 0.366120 | 0.315116| 0.803610|
| Decision Tree       | 0.843077 | 0.595588  | 0.632812| 0.613636 | 0.515667| 0.763724|
| kNN                 | 0.831538 | 0.589372  | 0.476562| 0.526998 | 0.429469| 0.823792|
| Naive Bayes         | 0.750769 | 0.411917  | 0.621094| 0.495327 | 0.351334| 0.758250|
| Random Forest       | 0.892308 | 0.825843  | 0.574219| 0.677419 | 0.629915| 0.913767|

## Problem Statement
The goal of this project is to build and evaluate multiple machine learning classification models to predict wine quality based on physicochemical properties. The dataset, sourced from the UCI Machine Learning Repository, contains measurements such as acidity, sugar content, pH, sulphates, alcohol percentage, and other chemical attributes of red and white wines.
## DatasetDescription
The Wine Quality dataset is sourced from the UCI Machine Learning Repository and contains physicochemical measurements of red and white wines. It has 4,898 instances and 12 input features such as acidity, sugar, pH, sulphates, and alcohol percentage. Each wine sample is rated for quality on a scale of 0–10 by expert tasters.
## Github Repository Link

## https://github.com/Adityaz007/2020AC05605_ML_Assignment2/tree/main

## Model Comparison Results

| Model               | Accuracy | Precision | Recall  | F1 Score | MCC     | AUC     |
|---------------------|----------|-----------|---------|----------|---------|---------|
| Logistic Regression | 0.821538 | 0.609091  | 0.261719| 0.366120 | 0.315116| 0.803610|
| Decision Tree       | 0.843077 | 0.595588  | 0.632812| 0.613636 | 0.515667| 0.763724|
| kNN                 | 0.831538 | 0.589372  | 0.476562| 0.526998 | 0.429469| 0.823792|
| Naive Bayes         | 0.750769 | 0.411917  | 0.621094| 0.495327 | 0.351334| 0.758250|
| Random Forest       | 0.892308 | 0.825843  | 0.574219| 0.677419 | 0.629915| 0.913767|

## Model Performance Observations

| ML Model Name        | Observation about model performance |
|----------------------|--------------------------------------|
| Logistic Regression  | Achieved moderate accuracy (~82%) and decent AUC (~0.80). Recall was low, missing many positives. |
| Decision Tree        | Accuracy (~84%) with balanced precision and recall. Captured positives better but risked overfitting. AUC was weaker (~0.76). |
| kNN                  | Accuracy (~83%) and AUC (~0.82). Stable but recall modest; sensitive to scaling and neighborhood choice. |
| Naive Bayes          | Lowest accuracy (~75%). Recall higher (~62%) but precision poor, leading to many false alarms. |
| Random Forest        | Best performer: highest accuracy (~89%), strong precision (~82%), and best AUC (~0.91). Robust and generalizes well. |

**Overall Winner:** Random Forest is the most reliable model for this dataset, consistently outperforming others across metrics.
