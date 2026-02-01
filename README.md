# California Housing Price Prediction

This project builds a **robust machine learning pipeline** to predict median house prices in California using the California Housing Dataset.  
It includes **extensive exploratory data analysis (EDA), feature engineering, outlier handling, scaling, multiple regression models, and advanced ensemble methods** with hyperparameter tuning.

---

## Project Objectives

- Understand the structure and patterns in California housing data
- Perform in-depth exploratory data analysis and visualization
- Engineer meaningful features to improve model performance
- Compare baseline models with advanced ensemble models
- Select and save the best-performing model

---

## Dataset

- **Source:** California Housing Dataset (from _Hands-On Machine Learning_ by Aurélien Géron)
- **Target Variable:** `median_house_value`

### Original Features:

- `longitude`, `latitude`
- `housing_median_age`
- `total_rooms`, `total_bedrooms`
- `population`, `households`
- `median_income`
- `ocean_proximity`

---

## Exploratory Data Analysis (EDA)

### Steps Performed:

- Dataset inspection (`head`, `info`, `describe`)
- Univariate analysis using histograms
- Scatter plots for geospatial visualization
- Correlation analysis using scatter matrices
- Visualization of housing prices over geographic coordinates

### Key Insights:

- **Median income** is the strongest predictor of house prices
- Location (latitude & longitude) has a strong spatial influence
- Presence of extreme values in `median_house_value`

---

## Outlier Handling

- Applied **IQR-based filtering** on `median_house_value`
- Reduced the influence of extreme price values
- Improved model stability and generalization

---

## Feature Engineering

### Ratio-Based Features:

- `rooms_per_household`
- `bedrooms_per_room`
- `population_per_household`

### Nonlinear Transformations:

- Squared features (`income_sq`, `latitude_sq`, `longitude_sq`)
- Log transformations (`income_log`, `population_per_household_log`)
- Interaction features (`longitude × latitude`)

### Feature Selection:

- Removed redundant features (`total_rooms`, `total_bedrooms`, `population`)
- Dropped categorical column (`ocean_proximity`)
- Final feature set optimized using correlation analysis

---

## Data Preprocessing

- Train / validation / test split
- Standard scaling for input features
- Target variable scaling for regularized models
- Scalers saved using `joblib` for reproducibility

---

## Models Implemented

### Baseline Models:

- Linear Regression
- Lasso Regression (with GridSearchCV)
- Ridge Regression (with GridSearchCV)

### Tree & Ensemble Models:

- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor
- CatBoost Regressor

---

## Hyperparameter Tuning

- **GridSearchCV** for Linear, Lasso, Ridge, and Decision Tree
- **RandomizedSearchCV** for Random Forest, XGBoost, and CatBoost
- Extensive search space to optimize bias–variance tradeoff

---

## Model Evaluation Metrics

- **R2 Score**
- **Root Mean Squared Error (RMSE)**
- Train vs Test performance comparison
- Actual vs Predicted scatter plots

---

## Best Model

- **CatBoost Regressor**
- Achieved highest generalization performance
- Final model saved as: `catboost_california.cbm`
- Achieved R2 Score of 0.8517
- RMSE = $37028

---

## Tech Stack

- **Language:** Python
- **Libraries:**
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost
- CatBoost
- Joblib
- **Environment:** Jupyter Notebook

---

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/Krishnaarora18/california-housing.git
```
