# 🏠 Boston Housing Price Prediction  
**Mini Project 1: Regression Challenge**

## 📌 Project Overview/Description
This project builds and evaluates multiple regression models to predict median house prices in Boston suburbs using socioeconomic and housing-related features. The goal is to apply supervised learning techniques including data preprocessing, feature engineering, regularization, and model evaluation.

We compare baseline linear regression with regularized and engineered models to understand how different techniques impact predictive performance and generalization.

---

## 🎯 Problem Motivation
Accurately predicting housing prices is important for:
- Urban planning and policy analysis
- Real estate valuation
- Understanding socioeconomic factors that influence housing markets

This project aims to answer:

> *How well can we predict Boston housing prices using regression models, and which features contribute most to price variation?*

---

## 📊 Dataset Description
- **Dataset:** Boston Housing Dataset  
- **Source:** `fetch_openml("boston")` from `sklearn.datasets`  
- **Samples:** 506  
- **Features:** 13 numerical variables  
- **Target Variable** `MEDV` – Median value of owner-occupied homes (in $1000s)

### Feature Overview
| Feature | Description |
|-------|-------------|
| CRIM | Per capita crime rate |
| ZN | Proportion of residential land zoned |
| INDUS | Proportion of non-retail business acres |
| CHAS | Charles River dummy variable |
| NOX | Nitric oxide concentration |
| RM | Average number of rooms |
| AGE | Proportion of older homes |
| DIS | Distance to employment centers |
| RAD | Accessibility to highways |
| TAX | Property tax rate |
| PTRATIO | Pupil-teacher ratio |
| B | Proportion of Black residents |
| LSTAT | % lower status population |


## ⚙️ Setup Instructions
### 1. Clone the repository
```bash
git clone https://github.com/jackychen8173/AppliedAIProjects.git
cd mini-project-1 
```

### 2. Create Virtual environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install dependencies
``` bash
pip install -r requirements.txt
```

### 4. Run the notebooks
```bash
Launch Jupyter and run notebooks top to bottom:
jupyter notebook 
```
## 🔍 Methodology

Data Preprocessing

- Checked for missing values (none present)
- Performed an 80/20 train-test split
- Applied feature scaling using StandardScaler

Feature Engineering

- Polynomial features for selected predictors
- Log transformation for skewed variables
- Feature selection using correlation analysis

Models Trained

- Linear Regression (baseline)
- Ridge Regression
- Lasso Regression
- Polynomial Regression

## 📈 Results Summary

**Model Performance Comparison**

| Model                     | RMSE   | MAE   | R²     |
|----------------------------|--------|-------|--------|
| Polynomial (degree 2)      | 3.776  | 2.575 | 0.806  |
| Linear Regression          | 4.929  | 3.189 | 0.669  |
| Ridge (α = 0.1)            | 4.929  | 3.189 | 0.669  |
| Ridge (α = 1.0)            | 4.931  | 3.186 | 0.668  |
| Ridge (α = 10.0)           | 4.949  | 3.172 | 0.666  |
| Lasso (α = 0.1)            | 5.065  | 3.242 | 0.650  |
| Lasso (α = 1.0)            | 5.251  | 3.474 | 0.624  |
| Lasso (α = 10.0)           | 8.663  | 6.256 | -0.023 |

**Visualizations included:**

*   Predicted vs. actual values plot
    
*   Residual plot, showing errors centered around zero

## 📝Key Takeaways
----------------

*   Polynomial regression (degree 2) performed best (**R² = 0.81**) by capturing non-linear relationships
    
*   LSTAT (% lower-status population) and RM (average rooms) are the most influential features
    
*   Ridge regression helped stabilize coefficients but did not improve R² over linear regression
    
*   Lasso regression over-regularized at higher α values, removing important predictors
    
*   Scaling, feature engineering, and exploratory analysis were crucial for improving performance
    

**Potential Future Improvements**:

*   Hyperparameter tuning via cross-validation
    
*   ElasticNet regression to balance Ridge and Lasso benefits
    
*   Ensemble models (Random Forest, Gradient Boosting) for non-linear relationships
    


## 👥 Team Member Contributions

**Jacky Chen:** Data preprocessing, feature engineering, model implementation, results analysis, report writing