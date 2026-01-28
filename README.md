# Bank Term Deposit Prediction  
*Kaggle Playground Series – Season 5, Episode 8*

## 📌 Overview
This project solves a binary classification problem from the Kaggle Playground
Series (Season 5, Episode 8). The goal is to predict whether a bank client will
subscribe to a term deposit based on demographic, financial, and
campaign-related features.

The solution focuses on building a robust, leakage-free machine learning
pipeline using XGBoost, with careful handling of class imbalance and
appropriate evaluation using ROC–AUC.

---

## 🎯 Problem Statement
The dataset is provided in two parts:
- **Training data** containing input features and the target variable `y`
- **Test data** containing only input features for final evaluation

The task is to train a model on the training dataset, generate probability
predictions for the test dataset, and submit them to Kaggle. Model performance
is evaluated using **ROC–AUC**, and scores are reported on both public and
private leaderboards.

---

## 📊 Dataset Description
Each row represents a client contacted during a bank marketing campaign.

### Target Variable
- **y**  
  - `1`: Client subscribed to the term deposit  
  - `0`: Client did not subscribe  
- The target variable is highly imbalanced.

### Feature Categories
- **Demographic**: age, job, marital status, education  
- **Financial**: balance, housing loan, personal loan, default status  
- **Campaign-related**: contact type, month, duration, previous campaign
  outcome  

Both numerical and categorical features are present.

---

## ⚙️ Approach

### Preprocessing
- Used `Pipeline` and `ColumnTransformer` for leakage-free preprocessing
- One-hot encoding for categorical variables
- No feature scaling or outlier removal, as XGBoost is robust to skewed
  distributions and outliers

### Evaluation Strategy
- **ROC–AUC** chosen due to class imbalance
- Accuracy alone avoided as it can be misleading
- Class imbalance handled using `scale_pos_weight`

---

## 🤖 Model

### Baseline Model
An XGBoost classifier was trained using manually selected hyperparameters:

```python
XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric='auc',
    random_state=42
)
```
## 🏆 Kaggle Scores

### Baseline Model
- **Public LB:** 0.95979  
- **Private LB:** 0.95871  

---

## 🔧 Hyperparameter Tuning

- Performed hyperparameter optimization using **RandomizedSearchCV**
- Used **Stratified K-Fold cross-validation** to preserve class distribution
- Best-performing model was saved and re-evaluated on the hold-out test set

### Kaggle Scores (Tuned Model)
- **Public LB:** 0.95965  
- **Private LB:** 0.95873  

Hyperparameter tuning improved model stability, while overall ROC–AUC remained
comparable, indicating a strong baseline configuration.

---

## 📈 Results

| Metric | Train | Test |
|------|------|------|
| ROC–AUC | ~0.96 | ~0.958 |
| Accuracy | ~0.92 | ~0.92 |
| Precision (Class 1) | ~0.65 | ~0.65 |
| Recall (Class 1) | ~0.75 | ~0.75 |

The small gap between training and test performance indicates good
generalization.

---

## 🔍 Feature Importance

Feature importance analysis from the trained XGBoost model shows that the most
influential predictors are:

- Contact strategy (contact type)
- Previous campaign outcome
- Call duration
- Client job and financial indicators

These features play a key role in determining client subscription behavior.

---

## 🧠 Key Learnings

- Proper handling of class imbalance is critical for real-world tabular data
- ROC–AUC is more reliable than accuracy for imbalanced classification problems
- Well-chosen baseline models can perform competitively with tuned models
- Tree-based models reduce the need for aggressive preprocessing

---

## 🚀 Future Improvements

- SHAP-based model interpretability
- Cost-sensitive threshold optimization
- Model ensembling for marginal performance gains

---

## 🛠️ Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- Matplotlib  

---

## 🏁 Kaggle Competition

**Binary Classification with a Bank Dataset**  
Playground Series – Season 5, Episode 8

---

## 📎 Author

**Himanshu Shekhar**
