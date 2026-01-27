# ❤️ Heart Disease Prediction  
**Mini Project 2: Classification Challenge**

## 📌 Project Overview/Description  
This project builds and evaluates multiple classification models to predict whether a patient has heart disease based on clinical and demographic features. The goal is to apply supervised classification techniques including data preprocessing, model comparison, hyperparameter tuning, and evaluation using appropriate classification metrics.

We compare baseline and tree-based classifiers to understand how different techniques impact predictive performance and how to select a model based on both performance and practical considerations.

---

## 🎯 Problem Motivation  
Accurately predicting heart disease is important for:  
- Early medical diagnosis  
- Reducing missed high-risk patients  
- Supporting clinical decision-making  

This project aims to answer:

> *How well can we predict heart disease using classification models, and which model provides the best balance between recall and overall performance?*

---

## 📊 Dataset Description  
- **Dataset:** Heart Disease Dataset  
- **Source:** UCI Machine Learning Repository / Kaggle  
- **Samples:** 302 (after removing duplicates)  
- **Total Columns:** 14  
- **Features:** 13 input variables  
- **Target Variable:** `target` (0 = No disease, 1 = Disease)  

### Feature Overview  

| Feature | Description |
|-------|-------------|
| age | Age of patient |
| sex | Sex (0 = female, 1 = male) |
| cp | Chest pain type |
| trestbps | Resting blood pressure |
| chol | Serum cholesterol |
| fbs | Fasting blood sugar |
| restecg | Resting ECG results |
| thalach | Maximum heart rate |
| exang | Exercise induced angina |
| oldpeak | ST depression |
| slope | Slope of ST segment |
| ca | Number of major vessels |
| thal | Thalassemia type |

---

## ⚙️ Setup Instructions  

### 1. Clone the repository  
git clone https://github.com/jackychen8173/AppliedAIProjects.git  
cd mini-project-2  

### 2. Create Virtual environment (Recommended)  
python -m venv venv  
source venv/bin/activate   # Windows: venv\Scripts\activate  

### 3. Install dependencies  
pip install -r requirements.txt  

### 4. Run the notebooks  
Launch Jupyter and run notebooks top to bottom:  
jupyter notebook  

---

## 🔍 Methodology  

Data Preprocessing  

- Checked for missing values (none present)  
- Identified and removed duplicate observations (723 duplicates removed)  
- Final dataset size: 302 unique samples  
- Performed an 80/20 train-test split with stratification  
- Applied feature scaling using StandardScaler  

Exploratory Data Analysis  

- Examined target class distribution (roughly balanced)  
- Visualized feature distributions  
- Computed correlation heatmap  
- Verified no missing values after cleaning  

Models Trained  

- Logistic Regression (baseline)  
- Decision Tree  
- Random Forest  
- Tuned Random Forest (GridSearchCV)  
- Tuned Logistic Regression (GridSearchCV)  

---

## 🔧 Hyperparameter Tuning  

Hyperparameter tuning was performed using GridSearchCV with F1-score as the primary evaluation metric.

- **Random Forest tuned parameters:**  
  - `n_estimators`  
  - `max_depth`  
  - `min_samples_split`  

- **Logistic Regression tuned parameters:**  
  - Regularization strength `C`  
  - Penalty type  

5-fold cross-validation was used on the training set.

---

## 📈 Results Summary  

**Model Performance Comparison**

| Model | Precision | Recall | F1 | ROC-AUC |
|------|-----------|--------|----|---------|
| Logistic Regression | 0.800 | 0.848 | 0.824 | 0.871 |
| Decision Tree | 0.818 | 0.818 | 0.818 | 0.802 |
| Random Forest | 0.765 | 0.788 | 0.776 | 0.859 |
| Tuned Random Forest | 0.788 | 0.788 | 0.788 | 0.867 |
| Tuned Logistic Regression | 0.784 | 0.879 | 0.829 | 0.881 |

**Visualizations included:**  

* Confusion matrices for all models  
* ROC curves comparing all classifiers  
* Model comparison table  

---

## 📝 Key Takeaways  
----------------  

* Removing duplicate observations was critical to avoid data leakage and unrealistic performance  
* Logistic Regression achieved the best overall performance  
* Logistic Regression achieved the highest recall for detecting heart disease patients  
* Hyperparameter tuning improved Random Forest but did not outperform Logistic Regression  
* Simpler models generalized better on this small medical dataset  

**Potential Future Improvements**:  

* Try another classification algorithm
* Tune more model parameters  
* Analyze important features 
* Use a larger dataset 

---

## 👥 Team Member Contributions  

**Jacky Chen:**

**Henry Chen:** 
