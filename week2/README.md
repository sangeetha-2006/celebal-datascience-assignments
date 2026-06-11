# Week 2 - Tesla Deliveries Forecasting using Machine Learning

## Project Overview

This project develops an end-to-end Machine Learning pipeline to forecast Tesla vehicle deliveries using historical Tesla production and delivery data from 2015 to 2025.

The workflow includes data cleaning, exploratory data analysis (EDA), feature engineering, predictive modeling, cross-validation, hyperparameter tuning, feature importance analysis, and time-series stationarity testing.

---

## Dataset Information

**Dataset:** Tesla Deliveries Dataset (2015–2025)

### Dataset Shape

* Rows: 2640
* Columns: 12

### Target Variable

* Estimated_Deliveries

---

## Objectives

* Perform data quality checks.
* Conduct exploratory data analysis.
* Create time-series features.
* Train and evaluate Linear Regression.
* Perform 5-Fold Cross Validation.
* Optimize Random Forest using GridSearchCV.
* Analyze feature importance.
* Perform ADF stationarity testing.
* Generate delivery forecasts.

---

## Data Preprocessing

### Data Quality Checks

* Missing values checked.
* Duplicate records checked.
* Dataset structure validated using:

  * shape
  * columns
  * info()
  * describe()

### Feature Engineering

Created two forecasting features:

#### Deliveries_Lag1

Previous period delivery value.

#### Rolling_Mean_3

Three-period rolling average of deliveries.

### Encoding

Label Encoding applied to:

* Region
* Model
* Source_Type

---

## Exploratory Data Analysis

The following visualizations were generated:

1. Deliveries by Model
2. Deliveries by Region
3. Correlation Heatmap
4. Production Units vs Estimated Deliveries Scatter Plot
5. Time-Series Delivery Trend

### Key Finding

The correlation analysis showed a very strong positive relationship between Production_Units and Estimated_Deliveries, making production volume one of the strongest predictors.

---

## Machine Learning Models

### Linear Regression

Performance:

* MAE: 310.33
* RMSE: 375.56
* R² Score: 0.9908

### Cross Validation

5-Fold Cross Validation Results:

* Fold 1: 0.9906
* Fold 2: 0.9905
* Fold 3: 0.9895
* Fold 4: 0.9905
* Fold 5: 0.9908

#### Mean R² Score

0.9904

#### Standard Deviation

0.0005

These results demonstrate excellent model stability and generalization.

---

## Random Forest Regressor

### Hyperparameter Tuning

GridSearchCV Best Parameters:

```python
{
    'max_depth': None,
    'n_estimators': 50
}
```

### Performance

* MAE: 303.86
* RMSE: 388.55
* R² Score: 0.9902

The Random Forest model achieved strong predictive performance and confirmed the effectiveness of the engineered features.

---

## Feature Importance Analysis

Feature importance analysis was performed using the optimized Random Forest model.

Most influential predictors included:

* Production_Units
* Deliveries_Lag1
* Rolling_Mean_3

These variables contributed significantly to delivery forecasting accuracy.

---

## Time Series Analysis

### Augmented Dickey-Fuller (ADF) Test

The ADF test was performed on the Estimated_Deliveries series to evaluate stationarity.

Decision Rule:

* p-value < 0.05 → Stationary
* p-value ≥ 0.05 → Non-Stationary

The test result was interpreted accordingly within the notebook.

---

## Forecasting

A forecast table was generated comparing:

* Actual Deliveries
* Predicted Deliveries
* Error Percentage

for the first 20 test observations.

---


## Author

**Lunavath Sangeetha**

Celebal Data Science Program
