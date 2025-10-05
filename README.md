# 🧠 AI-Powered Retail Forecasting

This project focuses on **forecasting store sales** using advanced **time series modeling** and **feature engineering** with Python, pandas, and machine learning tools.  
The goal is to predict future sales based on historical data, promotions, holidays, and calendar-based patterns.

---

## 📊 Project Overview

- **Objective:** Build a predictive model to forecast daily sales for multiple stores.  
- **Dataset:** Historical sales data enriched with store metadata, holiday flags, oil prices, and promotions.  
- **Techniques Used:**
  - Data cleaning and merging from multiple sources  
  - Feature engineering (lags, rolling stats, calendar features)
  - Time-aware validation (backtesting)
  - Regression models (e.g. LightGBM, XGBoost, or RandomForest)
  - Performance evaluation using RMSE and MAPE

---

## 🧩 Key Steps

### 1. Data Preparation
- Load datasets (`train.csv`, `test.csv`, `stores.csv`, `holidays.csv`, etc.)
- Handle missing values
- Merge external data such as oil price and holidays
- Generate target variable (`sales`)

### 2. Feature Engineering
- Calendar features: day of week, month, is_weekend, is_month_start/end  
- Lag features: `lag_1`, `lag_7`, `lag_28`  
- Rolling window statistics: mean, std, min, max over 7/14/28-day windows  
- External variables: `onpromotion`, `is_holiday`

### 3. Model Training
- Train a regression model with cross-validation (time-aware split)
- Tune hyperparameters for optimal forecasting accuracy
- Evaluate performance on validation folds

### 4. Prediction
- Predict future sales for unseen test periods
- Export results as submission-ready CSV file

---

## 🛠️ Tech Stack

| Category | Tools |
|-----------|-------|
| Language | Python 🐍 |
| Libraries | pandas, numpy, scikit-learn, lightgbm, matplotlib |
| Visualization | matplotlib, seaborn |
| Environment | Jupyter Notebook / `.py` script |

---

## 📈 Example Output

```python
# Display first few rows of processed features
print(df_feats2.head())
