# Week 2 - Tesla Deliveries Forecasting using Machine Learning

## Project Overview

This project develops an end-to-end Machine Learning pipeline for forecasting Tesla vehicle deliveries using historical production and delivery data from 2015–2025.

The workflow includes data preprocessing, exploratory data analysis (EDA), feature engineering, model training, hyperparameter tuning, forecasting, and time-series stationarity testing.

---

## Dataset

**File:** `tesla_deliveries_dataset_2015_2025.csv`

### Dataset Information

* Total Records: 2640
* Total Columns: 12

### Target Variable

* `Estimated_Deliveries`

---

## Objectives

1. Perform data cleaning and validation.
2. Conduct Exploratory Data Analysis (EDA).
3. Encode categorical variables.
4. Create lag and rolling mean features.
5. Train and evaluate Linear Regression.
6. Perform 5-Fold Cross Validation.
7. Optimize Random Forest using GridSearchCV.
8. Analyze feature importance.
9. Perform ADF stationarity testing.
10. Generate delivery forecasts.

---

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Statsmodels

---

## Exploratory Data Analysis

The following visualizations were created:

1. Deliveries by Model
2. Deliveries by Region
3. Correlation Heatmap
4. Production Units vs Estimated Deliveries
5. Delivery Trend Over Time

---

## Feature Engineering

### Label Encoding

The following categorical variables were encoded:

* Region
* Model
* Source_Type

### Engineered Features

#### Deliveries_Lag1

Previous period delivery value.

#### Rolling_Mean_3

Three-period rolling average of deliveries.

---

## Models Implemented

### Linear Regression

Evaluation Metrics:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

### Random Forest Regressor

Hyperparameter tuning performed using:

* n_estimators = [50, 100]
* max_depth = [5, 10, None]

GridSearchCV was used to select the best model.

---

## Validation

### 5-Fold Cross Validation

Cross-validation was performed to evaluate model stability and generalization performance.

Metrics Reported:

* Fold-wise R² Scores
* Mean R² Score
* Standard Deviation

---

## Time Series Analysis

### Augmented Dickey-Fuller (ADF) Test

The ADF test was applied to determine whether the delivery series is stationary.

Decision Rule:

* p-value < 0.05 → Stationary
* p-value ≥ 0.05 → Non-Stationary

---

## Author

**Sangeetha**

Celebal Data Science Program
