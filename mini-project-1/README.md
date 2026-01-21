# 🏠 Boston Housing Price Prediction  
**Mini Project 1: Regression Challenge**

## 📌 Project Overview
This project builds and evaluates multiple regression models to predict **median house prices** in Boston suburbs using socioeconomic and housing-related features. The goal is to apply supervised learning techniques including data preprocessing, feature engineering, regularization, and model evaluation.

We compare baseline linear regression with regularized and engineered models to understand how different techniques impact predictive performance and generalization.

---

## 🎯 Problem Statement & Motivation
Accurately predicting housing prices is important for:
- Urban planning and policy analysis
- Real estate valuation
- Understanding socioeconomic factors that influence housing markets

This project aims to answer:

> *How well can we predict Boston housing prices using regression models, and which features contribute most to price variation?*

**Target Variable**
- `MEDV` – Median value of owner-occupied homes (in $1000s)

---

## 📊 Dataset Description
- **Dataset:** Boston Housing Dataset  
- **Source:** `fetch_openml("boston")` from `sklearn.datasets`  
- **Samples:** 506  
- **Features:** 13 numerical variables  

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

---

## 🧪 Project Structure
```bash
mini-project-1/
├── README.md
├── requirements.txt
├── data/
│   └── README.md
├── notebooks/
│   ├── 01_exploration.ipynb
│   └── 02_modeling.ipynb
└── src/
```


---

## ⚙️ Setup Instructions
### 1. Clone the repository
```bash
git clone https://github.com/jackychen8173/AppliedAIProjects.git
cd mini-project-1

### 2. Create Virtual environment (Recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

### 3. Install dependencies
pip install -r requirements.txt

### 4. Run the notebooks
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


## 👥 Team Member Contributions